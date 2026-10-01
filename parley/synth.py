"""Synthetic calls with planted failures, to build and test the pipeline before real calls exist.

Labels come from the planted failures (reviewer "synthetic"), so they are only as good as the
generator's obedience: use them to exercise the pipeline, never to claim accuracy.
"""
import csv
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .judge import JudgeError, make_parser
from .rubric import Judge, Rubric
from .score import timing_metrics, timing_result
from .transcribe import Call, Turn

SYSTEM = "You write realistic, fully fictional contact-center call transcripts for testing QA software."
PROMPT = """Write a {agent_type} agent phone call for this business: {business}
Language: {language}. Vary the customer's request and the outcome; this is call {i} of a batch.

Each situation below MUST come up in the call, and the agent must clearly get it WRONG:
{fails}

Each situation below MUST come up in the call, and the agent must clearly get it RIGHT:
{passes}

These situations must NOT come up at all:
{absent}

12-30 turns with realistic timestamps in seconds, natural pauses and the occasional slight overlap.
Include fictional personal details where natural (a phone number, a card or account number, a date of birth).
Never mention the checks."""


class SynthTurn(BaseModel):
    speaker: Literal["agent", "customer"]
    start: float
    end: float
    text: str


class SynthCall(BaseModel):
    turns: list[SynthTurn]


def synth(rubric: Rubric, n: int, out_dir, writer: Judge, seed=0, fail_rate=0.3):
    rng, parse = random.Random(seed), make_parser(writer)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        agent_type = rng.choice(["human", "ai"])
        language = rng.choice(["English", "Hinglish (Hindi-English code-switching, Latin script)"])
        checkable = [c for c in rubric.criteria if c.type != "timing" and c.applies_to in ("both", agent_type)]
        fails = {c.id for c in checkable if rng.random() < fail_rate}
        # Criteria that can be not-applicable get planted as absent half the time, so "na" is tested too.
        absent = {c.id for c in checkable if c.na_when and c.id not in fails and rng.random() < 0.5}
        passes = [c for c in checkable if c.id not in fails | absent]
        bullet = lambda c, how="": f"- {c.question}" + (f" ({how})" if how else "")
        prompt = PROMPT.format(
            agent_type=agent_type, business=rubric.description or rubric.name, language=language, i=i + 1,
            fails="\n".join(bullet(c, c.fail_when and f"fail like this: {c.fail_when}") for c in checkable if c.id in fails) or "- (none)",
            passes="\n".join(bullet(c) for c in passes) or "- (none)",
            absent="\n".join(f"- {c.na_when}" for c in checkable if c.id in absent) or "- (none)")
        try:
            gen, _ = parse(SYSTEM, prompt, SynthCall)
        except JudgeError as e:
            print(f"synth-{i + 1:03d}: skipped ({e})")
            continue
        call = Call(id=f"synth-{i + 1:03d}", agent_type=agent_type, language=language,
                    turns=[Turn(id=j, **t.model_dump()) for j, t in enumerate(gen.turns)])
        (out / f"{call.id}.json").write_text(call.model_dump_json(indent=2), encoding="utf-8")
        metrics = timing_metrics(call.turns)
        for c in rubric.criteria:
            if c.applies_to not in ("both", agent_type):
                verdict = "na"
            elif c.type == "timing":
                verdict = timing_result(c, metrics, call)["verdict"]
            else:
                verdict = "fail" if c.id in fails else "na" if c.id in absent else "pass"
            rows.append({"call_id": call.id, "criterion_id": c.id, "reviewer": "synthetic", "verdict": verdict})
        print(f"{call.id}: {len(call.turns)} turns, fail: {sorted(fails) or '-'}, absent: {sorted(absent) or '-'}")
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["call_id", "criterion_id", "reviewer", "verdict"])
        w.writeheader()
        w.writerows(rows)
