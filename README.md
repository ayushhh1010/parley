# Parley

**Open-source QA for every call, human or AI. Every score has a receipt.**

Parley scores your calls (from AI voice agents or human agents) against your own checklist. Every verdict cites the exact words behind it, and code throws out any verdict whose quote isn't really in the transcript. It also measures how often it agrees with your own reviewers, criterion by criterion, so you know which checks you can trust.

- **Receipts, not vibes:** "Fail: gave medical advice" links to the exact sentence ("take two aspirin and lie down").
- **Runs free:** works with free Gemini and Groq API tiers, or local models. No account and no sign-up with us.
- **Your data stays yours:** card numbers, Aadhaar, PAN, phone numbers and emails are masked before any text reaches a model, including spoken digits like "nine eight double seven…".
- **Hinglish-ready:** tested on English and Hindi-English code-switched calls.

**See it:** [live demo report](https://ayushhh1010.github.io/parley/samples/demo-report.html). It's 25 test calls of a clinic-booking voice agent, with every verdict next to its quote.

## Try it in 5 minutes (free)

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ayushhh1010/parley && cd parley && uv sync
# free key from aistudio.google.com
export GEMINI_API_KEY=...
uv run parley score rubrics/appointment-booking.yaml samples/clinic-booking.json --judge gemini-lite
uv run parley report scores.jsonl
```

Open `report.html`. To score your own calls, convert them to the [call format](#call-format) or transcribe recordings with `parley transcribe`.

**Prefer clicking?** `uv run parley serve` opens a local web app: drop in call files (or use the sample calls), pick a checklist and an AI judge, and review each call with its verdicts and quotes. It runs on your machine only, with no sign-in.

## Setup

```bash
uv sync
```

API keys (set only the ones you use). **Everything can run on free tiers:**

| Variable | Used for | Cost |
|---|---|---|
| `GEMINI_API_KEY` | `--judge gemini` (Gemini 3.8 Flash), from aistudio.google.com | free tier, daily limits |
| `GROQ_API_KEY` | `--judge groq` (gpt-oss-120b) and Whisper transcription, from console.groq.com | free tier: ~1K requests and 200K tokens a day for the judge; 8 hours of audio a day |
| `DEEPGRAM_API_KEY` | Deepgram Nova-3 transcription, better on Hinglish and mono audio | signup credit |
| `ANTHROPIC_API_KEY` | `--judge claude` (default in the templates) | paid, per token |

**Free tiers may train on your data** (Gemini's does). Use them for synthetic calls only. Real recordings go to a local model, the customer's own API keys, or a paid tier whose terms rule out training.

## Workflow

```bash
# 1. No real calls yet? Generate synthetic ones with planted failures and matching labels
uv run parley synth rubrics/appointment-booking.yaml -n 30 --judge gemini

# 2. Score them (redaction runs first; nothing unredacted reaches a model)
uv run parley score rubrics/appointment-booking.yaml data/synth/ --judge gemini -w 1

# 3. Compare with the labels, per criterion
uv run parley calibrate scores.jsonl data/synth/labels.csv

# 4. Shareable HTML report: every verdict next to the quote behind it
uv run parley report scores.jsonl --labels data/synth/labels.csv
```

With real recordings:

```bash
uv run parley transcribe recordings/ --agent-type human          # stereo WAV, channel 0 = agent
uv run parley transcribe recordings/ --provider deepgram --mono   # mono: diarization
uv run parley wer references/ data/calls/                         # bake-off: WER + number accuracy
```

Your reviewers label calls in a CSV with columns `call_id, criterion_id, reviewer, verdict` (verdict is `pass`, `fail`, `na` or `cannot_determine`). Two reviewers on the same calls also gives human-to-human agreement.

## Rubrics

Rubrics are YAML files; `rubrics/` has three starter templates. Check one with `uv run parley check <file>`. Each criterion has one of three types:

- `phrase`: required wording, matched in code (`phrases`, `speaker`, `within_seconds`).
- `timing`: a metric from timestamps (`metric`, `max`/`min`): `longest_silence`, `agent_talk_ratio`, `agent_interruptions`, `overlaps`, `agent_response_latency_avg`, `agent_response_latency_max`, `duration`.
- `llm`: judged by a model (`question`, optional `pass_when`, `fail_when`, `na_when`, `examples`).

Every criterion also takes `severity` (`critical`, `major`, `minor`) and `applies_to` (`human`, `ai`, `both`).

`--judge claude|gemini|groq` overrides a rubric's judge. For anything else (Ollama, vLLM, a paid Gemini key), set the judge in the rubric:

```yaml
judge:
  provider: openai
  base_url: http://localhost:11434/v1   # any OpenAI-compatible endpoint
  model: <model name>
  api_key_env: MY_KEY_VARIABLE          # omit for local servers
```

Free tiers rate-limit hard, so score with `-w 1`; rate-limited requests are retried automatically. Different judges agree with your reviewers at different rates, so calibrate each one separately.

## How a call is scored

1. **Redact.** Cards (Luhn), CVVs in payment context, Aadhaar (Verhoeff), PAN, phones, emails and other long digit runs, including spoken digits ("nine eight double seven…") and card numbers split across turns.
2. **Deterministic checks.** Phrase and timing criteria, no model.
3. **Judge.** Remaining criteria. Critical criteria get their own call; the rest share one.
4. **Verify.** Every quote must match the turn it cites (≥0.9 word similarity). Otherwise the verdict becomes `cannot_determine`. The quote shown is always the transcript's own words.

Output is one JSON line per call in `scores.jsonl`: verdicts, evidence with timestamps, timing metrics, redaction counts, token usage and the redacted transcript.

## Call format

One JSON file per call. `speaker` is `agent` or `customer`; times are in seconds.

```json
{
  "id": "call-001",
  "agent_type": "ai",
  "language": "hi-en",
  "turns": [
    {"id": 0, "speaker": "agent", "start": 0.0, "end": 4.2, "text": "Hi, thanks for calling Sunrise Clinic. I'm the clinic's AI assistant."},
    {"id": 1, "speaker": "customer", "start": 4.8, "end": 7.1, "text": "Mujhe Dr. Mehta ke saath appointment chahiye."}
  ]
}
```

## Tests

```bash
uv run pytest
```

## Not built yet (MVP, PRD §9)

Web app and review workspace, Postgres job queue, webhooks (Vapi, Retell, Twilio), audio muting of redacted spans, Sarvam adapter, digest and dashboards, sign-in and roles.
