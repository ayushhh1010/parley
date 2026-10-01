"""parley: score calls against a rubric, calibrate against human reviewers, compare transcription providers."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _files(paths, suffix):
    for p in map(Path, paths):
        yield from sorted(p.glob(f"*{suffix}")) if p.is_dir() else [p]


def _with_judge(rubric, preset):
    from .rubric import PRESETS
    if preset:
        rubric = rubric.model_copy(update={"judge": PRESETS[preset]})
    if "generativelanguage" in (rubric.judge.base_url or ""):
        print("note: on Gemini's free tier Google may use this content to improve its products; "
              "send synthetic calls only, never real recordings.", file=sys.stderr)
    return rubric


def cmd_check(a):
    from .rubric import load
    r = load(a.rubric)
    print(f"{r.name} v{r.version}: {len(r.criteria)} criteria OK (judge: {r.judge.provider}/{r.judge.model})")
    for w in r.warnings():
        print(f"warning: {w}")


def cmd_score(a):
    from .judge import make_judge
    from .rubric import load
    from .score import score_call
    from .transcribe import load_call
    rubric = _with_judge(load(a.rubric), a.judge)
    calls = [load_call(p) for p in _files(a.calls, ".json")]
    judge = make_judge(rubric.judge) if any(c.type == "llm" for c in rubric.criteria) else None
    fails = 0
    with open(a.out, "w", encoding="utf-8") as out, ThreadPoolExecutor(a.workers) as pool:
        for s in pool.map(lambda c: score_call(c, rubric, judge), calls):
            out.write(json.dumps(s, ensure_ascii=False) + "\n")
            by = lambda v: [r["criterion_id"] for r in s["results"] if r["verdict"] == v]
            crit = [r["criterion_id"] for r in s["results"] if r["verdict"] == "fail" and r["severity"] == "critical"]
            fails += bool(by("fail"))
            print(f"{s['call_id']}: {len(by('pass'))} pass, {len(by('fail'))} fail, {len(by('cannot_determine'))} cannot determine"
                  + (f"  CRITICAL: {', '.join(crit)}" if crit else "")
                  + (f"  redacted: {s['redactions']}" if s["redactions"] else ""))
    print(f"\n{len(calls)} calls scored, {fails} with at least one fail. Results: {a.out}")


def cmd_calibrate(a):
    from .calibrate import report
    text = report(a.scores, a.labels, min_labels=a.min_labels, min_fails=a.min_fails,
                  kappa_min=a.kappa, agree_min=a.agreement)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


def cmd_report(a):
    from .calibrate import compute
    from .report_html import render
    if a.labels:
        runs, stats, _, t = compute(a.scores, a.labels)
    else:
        runs, stats, t = [json.loads(l) for l in open(a.scores, encoding="utf-8-sig") if l.strip()], None, None
    Path(a.out).write_text(render(runs, stats, t), encoding="utf-8")
    print(f"Wrote {a.out} ({len(runs)} calls). Open it in a browser, or send it as an attachment.")


def cmd_transcribe(a):
    from .transcribe import Call, deepgram, whisper
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in _files(a.audio, ""):
        kw = {"stereo": not a.mono, "agent_channel": a.agent_channel}
        if a.provider == "deepgram":
            turns = deepgram(p, language=a.language or "multi", **kw)
        else:
            turns = whisper(p, language=a.language, base_url=a.base_url, model=a.model, api_key_env=a.api_key_env, **kw)
        call = Call(id=p.stem, agent_type=a.agent_type, language=a.language or "", turns=turns,
                    metadata={"source": str(p), "stt": a.provider})
        (out / f"{p.stem}.json").write_text(call.model_dump_json(indent=2), encoding="utf-8")
        print(f"{p.name}: {len(turns)} turns -> {out / (p.stem + '.json')}")


def cmd_wer(a):
    from .transcribe import number_accuracy, text_of, wer
    ref, hyp = Path(a.reference), Path(a.hypothesis)
    pairs = ([(r, h) for r in sorted(ref.iterdir()) for h in hyp.glob(r.stem + ".*")] if ref.is_dir() else [(ref, hyp)])
    if not pairs:
        sys.exit("no matching files (pairs are matched by file name without extension)")
    rows = [(r.stem, wer(text_of(r), text_of(h)), number_accuracy(text_of(r), text_of(h))) for r, h in pairs]
    for name, w, na in rows:
        print(f"{name}: WER {w:.1%}, numbers {'–' if na is None else f'{na:.0%}'}")
    nums = [na for _, _, na in rows if na is not None]
    print(f"\nmean WER {sum(w for _, w, _ in rows) / len(rows):.1%} over {len(rows)} files"
          + (f"; numbers correct {sum(nums) / len(nums):.0%}" if nums else ""))


def cmd_synth(a):
    from .rubric import load
    from .synth import synth
    rubric = _with_judge(load(a.rubric), a.judge)
    synth(rubric, a.n, a.out_dir, rubric.judge, seed=a.seed)
    print(f"\nWrote {a.n} calls and labels.csv to {a.out_dir}")


def main(argv=None):
    from .rubric import PRESETS
    p = argparse.ArgumentParser(prog="parley", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="validate a rubric file")
    s.add_argument("rubric")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("score", help="score call JSON files against a rubric")
    s.add_argument("rubric")
    s.add_argument("calls", nargs="+", help="call JSON files or directories")
    s.add_argument("-o", "--out", default="scores.jsonl")
    s.add_argument("-w", "--workers", type=int, default=4, help="calls scored in parallel (use 1 on free API tiers)")
    s.add_argument("--judge", choices=list(PRESETS), help="override the rubric's judge")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("calibrate", help="agreement with human labels, per criterion")
    s.add_argument("scores", help="scores.jsonl from `parley score`")
    s.add_argument("labels", help="CSV: call_id, criterion_id, reviewer, verdict")
    s.add_argument("-o", "--out", default="report.md")
    s.add_argument("--kappa", type=float, default=0.6)
    s.add_argument("--agreement", type=float, default=0.85)
    s.add_argument("--min-labels", type=int, default=20)
    s.add_argument("--min-fails", type=int, default=3)
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("report", help="shareable HTML report: verdicts next to the quotes behind them")
    s.add_argument("scores", help="scores.jsonl from `parley score`")
    s.add_argument("--labels", help="add the agreement table from a labels CSV")
    s.add_argument("-o", "--out", default="report.html")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("transcribe", help="audio -> call JSON")
    s.add_argument("audio", nargs="+", help="audio files or directories")
    s.add_argument("--provider", choices=["deepgram", "whisper"], default="whisper",
                   help="whisper = any OpenAI-compatible endpoint, Groq's free tier by default")
    s.add_argument("--mono", action="store_true", help="single-channel audio (default: stereo, one speaker per channel)")
    s.add_argument("--agent-channel", type=int, default=0, help="channel (or diarized speaker) that is the agent")
    s.add_argument("--agent-type", choices=["human", "ai"], default="human")
    s.add_argument("--language", help="language hint, e.g. hi, en; deepgram defaults to multi (code-switching)")
    s.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    s.add_argument("--model", default="whisper-large-v3-turbo")
    s.add_argument("--api-key-env", default="GROQ_API_KEY")
    s.add_argument("-o", "--out-dir", default="data/calls")
    s.set_defaults(fn=cmd_transcribe)

    s = sub.add_parser("wer", help="word error rate and number accuracy: reference vs hypothesis (files or dirs)")
    s.add_argument("reference")
    s.add_argument("hypothesis")
    s.set_defaults(fn=cmd_wer)

    s = sub.add_parser("synth", help="generate synthetic calls with planted failures and matching labels")
    s.add_argument("rubric")
    s.add_argument("-n", type=int, default=30)
    s.add_argument("-o", "--out-dir", default="data/synth")
    s.add_argument("--judge", choices=list(PRESETS), help="model that writes the calls (default: the rubric's judge)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
