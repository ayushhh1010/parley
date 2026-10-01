"""Checks for the logic that must be exactly right: redaction, evidence verification, metrics, statistics."""
import csv
import json
from pathlib import Path

import pytest

from parley.calibrate import cohen_kappa, report, wilson
from parley.judge import Evidence, Judgment
from parley.redact import digit_runs, luhn, redact_call, redact_text, verhoeff
from parley.rubric import Criterion, Rubric, load
from parley.score import find_span, score_call, timing_metrics, verify
from parley.transcribe import Call, Turn, load_call, number_accuracy, wer

ROOT = Path(__file__).parent.parent


def turn(i, spk, a, b, text):
    return Turn(id=i, speaker=spk, start=a, end=b, text=text)


def aadhaar_like():
    base = "23456789012"
    return next(base + d for d in "0123456789" if verhoeff(base + d))


def test_checksums():
    assert luhn("4111111111111111") and not luhn("4111111111111112")
    assert verhoeff("2363") and not verhoeff("2364")


def test_redaction():
    assert redact_text("my card is 4111 1111 1111 1111")[0] == "my card is [CARD]"
    spoken = "four one one one one one one one one one one one one one one one"
    assert redact_text(spoken)[0] == "[CARD]"
    assert redact_text("call me on nine eight double seven six five four three two one")[0] == "call me on [PHONE]"
    assert redact_text(f"aadhaar {aadhaar_like()}")[0] == "aadhaar [AADHAAR]"
    assert redact_text("PAN is abcde1234f, email priya.s@example.com")[0] == "PAN is [PAN], email [EMAIL]"
    assert redact_text("you owe 45,000 by the 5th")[0] == "you owe 45,000 by the 5th"  # amounts stay readable
    call = Call(id="c", turns=[turn(0, "agent", 0, 2, "Please read the CVV on the back of your card"),
                               turn(1, "customer", 2, 4, "it's 7 3 9"),
                               turn(2, "customer", 4, 6, "and the card starts 4111 1111")])
    red, counts = redact_call(call)
    assert red.turns[1].text == "it's [CVV]" and red.turns[2].text == "and the card starts [CARD]"
    assert counts == {"CVV": 1, "CARD": 1}


def test_digit_runs_and_number_accuracy():
    assert [d for *_, d in digit_runs("pay 500 on the twenty")] == ["500"]
    assert number_accuracy("account 4 5 6 7", "account four five six seven") == 1.0


def test_timing_metrics():
    m = timing_metrics([turn(0, "agent", 0, 4, "hi"), turn(1, "customer", 5, 9, "hello"),
                        turn(2, "agent", 8.5, 12, "yes"), turn(3, "customer", 25, 26, "ok")])
    assert m["longest_silence"] == 13 and m["longest_silence_turns"] == [2, 3]
    assert m["overlaps"] == 1 and m["agent_interruptions"] == 1
    assert m["agent_talk_ratio"] == round(7.5 / 12.5, 2)


def test_find_span_returns_transcript_words():
    text = "Done. You're booked with Dr. Mehta on Thursday at 4 PM."
    assert find_span("booked with dr mehta on thursday", text) == "booked with Dr. Mehta on Thursday"
    assert find_span("booked with Dr Mehta on Thurs day", text) is None
    assert find_span("you are booked on Friday", text) is None


def test_verify_rejects_invented_evidence():
    c = Criterion(id="x", type="llm", question="q")
    call = Call(id="c", turns=[turn(0, "agent", 0, 1, "Your appointment is Thursday at 4 PM.")])
    ok = Judgment(criterion_id="x", verdict="pass", rationale="r", confidence=0.9,
                  evidence=[Evidence(turn_id=0, quote="appointment is Thursday at 4 PM")])
    assert verify(c, ok, call)["verdict"] == "pass"
    for bad in ([Evidence(turn_id=0, quote="appointment is Friday at 9 AM")],  # not said
                [Evidence(turn_id=7, quote="appointment is Thursday at 4 PM")],  # no such turn
                []):  # no receipt
        r = verify(c, ok.model_copy(update={"evidence": bad}), call)
        assert r["verdict"] == "cannot_determine" and r["evidence"] == []


def test_score_call_end_to_end():
    rubric = load(ROOT / "rubrics" / "appointment-booking.yaml")
    call = load_call(ROOT / "samples" / "clinic-booking.json")
    seen = []

    def fake_judge(call, criteria):
        seen.append([c.id for c in criteria])
        assert "[PHONE]" in call.turns[3].text  # the judge only ever sees redacted text
        return {c.id: Judgment(criterion_id=c.id, verdict="fail", rationale="r", confidence=1,
                               evidence=[Evidence(turn_id=8, quote="double the dose")]) for c in criteria}, {}

    s = score_call(call, rubric, fake_judge)
    by = {r["criterion_id"]: r for r in s["results"]}
    assert by["no_medical_advice"]["verdict"] == "fail"
    assert by["no_medical_advice"]["evidence"][0]["quote"] == "double the dose"
    assert by["longest_silence"]["verdict"] == "pass" and by["response_latency"]["verdict"] == "pass"
    assert s["redactions"] == {"PHONE": 1}
    critical = {c.id for c in rubric.criteria if c.type == "llm" and c.severity == "critical"}
    assert all(len(g) == 1 for g in seen if g[0] in critical)  # critical criteria judged alone


def test_statistics():
    assert cohen_kappa([("p", "p")] * 5 + [("f", "f")] * 5) == 1.0
    assert cohen_kappa([("p", "p")] * 10) is None  # no variance: kappa undefined, not 100%
    assert cohen_kappa([("p", "p"), ("p", "f"), ("f", "p"), ("f", "f")]) == 0.0
    lo, hi = wilson(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943


def test_calibration_report(tmp_path):
    scores, labels = tmp_path / "scores.jsonl", tmp_path / "labels.csv"
    runs = [{"call_id": f"c{i}", "rubric": "r", "rubric_version": 1, "judge_model": "m",
             "results": [{"criterion_id": "a", "severity": "major", "verdict": "fail" if i < 8 else "pass"},
                         {"criterion_id": "b", "severity": "major", "verdict": "pass"}]} for i in range(30)]
    scores.write_text("\n".join(json.dumps(r) for r in runs))
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["call_id", "criterion_id", "reviewer", "verdict"])
        for i in range(30):
            w.writerow([f"c{i}", "a", "divya", "fail" if i < 9 else "pass"])
            w.writerow([f"c{i}", "b", "divya", "pass"])
    text = report(scores, labels)
    assert "| a | major | 30 | 97%" in text and "Trusted" in text
    assert "only 0 human fails" in text  # b: always pass, so agreement means nothing yet
    assert "- **a**: c8" in text


def test_rubrics():
    for p in (ROOT / "rubrics").glob("*.yaml"):
        assert load(p).warnings() == [], p
    r = Rubric(name="x", version=1, criteria=[Criterion(id="calm", type="llm", question="Did the agent stay calm and show empathy?")])
    assert r.warnings()
    with pytest.raises(ValueError):
        Rubric(name="x", version=1, criteria=[Criterion(id="a", type="llm", question="q")] * 2)
    with pytest.raises(ValueError):
        Criterion(id="a", type="timing", question="q", metric="longest_silence")


def test_synth_labels_match_what_was_planted(tmp_path, monkeypatch):
    import parley.synth as synth_mod
    prompts = []

    def fake_parser(cfg):
        def parse(system, prompt, schema):
            prompts.append(prompt)
            return schema(turns=[{"speaker": "agent", "start": 0, "end": 2, "text": "Hello"}]), {}
        return parse

    monkeypatch.setattr(synth_mod, "make_parser", fake_parser)
    rubric = load(ROOT / "rubrics" / "appointment-booking.yaml")
    synth_mod.synth(rubric, 10, tmp_path, rubric.judge, seed=1)
    labels = {(r["call_id"], r["criterion_id"]): r["verdict"] for r in csv.DictReader(open(tmp_path / "labels.csv"))}
    assert {"pass", "fail", "na"} <= set(labels.values())
    for i, prompt in enumerate(prompts):
        absent = prompt.split("must NOT come up at all:")[1]
        for c in rubric.criteria:
            if c.na_when:  # labeled na exactly when the generator was told the situation must not come up
                assert (labels[(f"synth-{i + 1:03d}", c.id)] == "na") == (c.na_when in absent), (i, c.id)


def test_wer():
    assert wer("book it for thursday", "book it for thursday") == 0
    assert wer("book it for thursday", "book for tuesday") == 0.5
