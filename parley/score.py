"""Score one call: redact, deterministic checks, LLM judge, then evidence verification by code (PRD F5)."""
import re
import sys
from difflib import SequenceMatcher

from .judge import JudgeError, make_judge
from .redact import redact_call
from .rubric import Criterion, Rubric
from .transcribe import Call, Turn


def find_span(quote: str, text: str, threshold=0.9) -> str | None:
    """The stretch of `text` that best matches `quote` (word-level similarity >= threshold), in the transcript's own words.

    Showing this span instead of the model's quote keeps every displayed quote an exact transcript substring.
    """
    words = list(re.finditer(r"\w+", text))
    q = re.findall(r"\w+", quote.lower())
    if not q or not words:
        return None
    low = [w.group().lower() for w in words]
    best, at = 0.0, 0
    for i in range(max(1, len(low) - len(q) + 1)):
        ratio = SequenceMatcher(None, q, low[i:i + len(q)]).ratio()
        if ratio > best:
            best, at = ratio, i
    if best < threshold:
        return None
    return text[words[at].start():words[min(at + len(q), len(words)) - 1].end()]


def timing_metrics(turns: list[Turn]) -> dict:
    """Deterministic voice metrics from turn timestamps; no model involved."""
    turns = sorted(turns, key=lambda t: t.start)
    talk = {}
    for t in turns:
        talk[t.speaker] = talk.get(t.speaker, 0.0) + t.end - t.start
    longest, gap_turns, reach, reach_id = 0.0, [], 0.0, None
    for t in turns:
        if reach_id is not None and t.start - reach > longest:
            longest, gap_turns = t.start - reach, [reach_id, t.id]
        if t.end > reach:
            reach, reach_id = t.end, t.id
    overlaps = interruptions = 0
    latencies = []
    for a, b in zip(turns, turns[1:]):
        if a.speaker == b.speaker:
            continue
        if b.start < a.end:
            overlaps += 1
            interruptions += b.speaker == "agent"
        elif a.speaker == "customer" and b.speaker == "agent":
            latencies.append(b.start - a.end)
    r = lambda x: round(x, 2)
    return {"duration": r(max((t.end for t in turns), default=0.0)),
            "agent_talk_ratio": r(talk.get("agent", 0.0) / (sum(talk.values()) or 1)),
            "longest_silence": r(longest), "longest_silence_turns": gap_turns,
            "overlaps": overlaps, "agent_interruptions": interruptions,
            "agent_response_latency_avg": r(sum(latencies) / len(latencies)) if latencies else 0.0,
            "agent_response_latency_max": r(max(latencies, default=0.0))}


def _result(c: Criterion, verdict, evidence=(), rationale="", source="", note="", confidence=None):
    return {"criterion_id": c.id, "question": c.question, "severity": c.severity, "type": c.type, "verdict": verdict,
            "evidence": list(evidence), "rationale": rationale, "confidence": confidence,
            "source": source, "note": note}


def _ev(t: Turn, quote: str):
    return {"turn_id": t.id, "speaker": t.speaker, "start": t.start, "quote": quote}


def phrase_result(c: Criterion, call: Call):
    where = f"{c.speaker} turns" + (f" in the first {c.within_seconds:g}s" if c.within_seconds is not None else "")
    for t in call.turns:
        if (c.speaker != "any" and t.speaker != c.speaker) or (c.within_seconds is not None and t.start > c.within_seconds):
            continue
        for p in c.phrases:
            if span := find_span(p, t.text, threshold=0.85):
                return _result(c, "pass", [_ev(t, span)], f"Found {p!r} in {where}.", "phrase")
    return _result(c, "fail", [], f"None of the required phrases appear in {where}.", "phrase")


def timing_result(c: Criterion, metrics: dict, call: Call):
    v = metrics[c.metric]
    ok = (c.max is None or v <= c.max) and (c.min is None or v >= c.min)
    limit = " and ".join(s for s in (c.min is not None and f"at least {c.min:g}", c.max is not None and f"at most {c.max:g}") if s)
    by_id = {t.id: t for t in call.turns}
    ev = [_ev(by_id[i], by_id[i].text) for i in metrics["longest_silence_turns"]] if c.metric == "longest_silence" else []
    return _result(c, "pass" if ok else "fail", ev, f"{c.metric} = {v:g} (required {limit}).", "timing")


def verify(c: Criterion, j, call: Call):
    """Accept a model verdict only if every quote is found in the turn it cites (PRD §12.4)."""
    if j is None:
        return _result(c, "cannot_determine", source="llm", note="judge returned no verdict for this criterion")
    if j.verdict in ("na", "cannot_determine"):
        return _result(c, j.verdict, rationale=j.rationale, source="llm", confidence=j.confidence)
    by_id = {t.id: t for t in call.turns}
    evidence = []
    for e in j.evidence[:3]:
        t = by_id.get(e.turn_id)
        span = t and find_span(e.quote, t.text)
        if not span:
            return _result(c, "cannot_determine", rationale=j.rationale, source="llm", confidence=j.confidence,
                           note=f"evidence rejected: turn {e.turn_id} does not contain {e.quote!r}")
        evidence.append(_ev(t, span))
    if not evidence:
        return _result(c, "cannot_determine", rationale=j.rationale, source="llm", note="no evidence cited")
    return _result(c, j.verdict, evidence, j.rationale, "llm", confidence=j.confidence)


def _groups(criteria):
    """Critical criteria get their own judge call; the rest share one (PRD §12.4)."""
    rest = [c for c in criteria if c.severity != "critical"]
    return [[c] for c in criteria if c.severity == "critical"] + ([rest] if rest else [])


def score_call(call: Call, rubric: Rubric, judge=None) -> dict:
    """ponytail: whole transcript in one prompt; add turn-window chunking for calls beyond the model's context."""
    call, redactions = redact_call(call)  # nothing below sees unredacted text
    metrics = timing_metrics(call.turns)
    results, pending = {}, []
    for c in rubric.criteria:
        if c.applies_to not in ("both", call.agent_type):
            results[c.id] = _result(c, "na", rationale=f"Applies to {c.applies_to} agents only.", source="rubric")
        elif c.type == "phrase":
            results[c.id] = phrase_result(c, call)
        elif c.type == "timing":
            results[c.id] = timing_result(c, metrics, call)
        else:
            pending.append(c)
    usage = {"input_tokens": 0, "output_tokens": 0}
    if pending:
        judge = judge or make_judge(rubric.judge)
        for group in _groups(pending):
            try:
                judgments, u = judge(call, group)
                usage = {k: usage[k] + u.get(k, 0) for k in usage}
            except JudgeError as e:
                print(f"{call.id}: judge failed ({e}); marking {[c.id for c in group]} cannot_determine", file=sys.stderr)
                judgments = {}
            for c in group:
                results[c.id] = verify(c, judgments.get(c.id), call)
    return {"call_id": call.id, "agent_type": call.agent_type, "language": call.language,
            "rubric": rubric.name, "rubric_version": rubric.version, "judge_model": rubric.judge.model,
            "metrics": metrics, "redactions": redactions, "usage": usage,
            "results": [results[c.id] for c in rubric.criteria],
            "transcript": [t.model_dump() for t in call.turns]}
