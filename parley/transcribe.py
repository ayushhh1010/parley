"""Call format, speech-to-text adapters and word error rate (PRD F2, Phase 0 bake-off)."""
import io
import mimetypes
import os
import re
import wave
from array import array
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel


class Turn(BaseModel):
    id: int
    speaker: Literal["agent", "customer", "unknown"]
    start: float
    end: float
    text: str


class Call(BaseModel):
    id: str
    agent_type: Literal["human", "ai"] = "human"
    language: str = ""
    turns: list[Turn]
    metadata: dict = {}


def load_call(path) -> Call:
    return Call.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _turns(segments) -> list[Turn]:
    """(start, end, speaker, text) tuples -> Turns ordered by start time."""
    segments = sorted(s for s in segments if s[3].strip())
    return [Turn(id=i, speaker=spk, start=round(a, 2), end=round(b, 2), text=t.strip())
            for i, (a, b, spk, t) in enumerate(segments)]


def deepgram(path, stereo=True, agent_channel=0, language="multi") -> list[Turn]:
    """Deepgram Nova-3. Stereo: one channel per speaker (exact). Mono: diarization.

    ponytail: on mono audio, diarized speaker `agent_channel` is assumed to be the agent;
    add a per-source speaker check when onboarding shows it guessing wrong.
    """
    params = {"model": "nova-3", "language": language, "smart_format": "true", "utterances": "true",
              "multichannel" if stereo else "diarize": "true"}
    r = httpx.post("https://api.deepgram.com/v1/listen", params=params, content=Path(path).read_bytes(),
                   headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
                            "Content-Type": mimetypes.guess_type(path)[0] or "audio/wav"},
                   timeout=600)
    r.raise_for_status()
    utts = r.json()["results"]["utterances"]
    who = lambda u: "agent" if (u["channel"] if stereo else u.get("speaker", 0)) == agent_channel else "customer"
    return _turns((u["start"], u["end"], who(u), u["transcript"]) for u in utts)


def whisper(path, stereo=True, agent_channel=0, language=None,
            base_url="https://api.groq.com/openai/v1", model="whisper-large-v3-turbo",
            api_key_env="GROQ_API_KEY") -> list[Turn]:
    """Any OpenAI-compatible transcription endpoint: Groq (free tier), OpenAI, or a local Whisper server.

    Whisper has no diarization, so mono audio yields speaker "unknown".
    """
    channels = split_stereo_wav(path) if stereo else [Path(path).read_bytes()]
    key = os.environ.get(api_key_env, "") if api_key_env else ""
    segments = []
    for ch, audio in enumerate(channels):
        data = {"model": model, "response_format": "verbose_json"}
        if language:
            data["language"] = language
        r = httpx.post(f"{base_url.rstrip('/')}/audio/transcriptions", data=data,
                       files={"file": (Path(path).with_suffix(".wav").name if stereo else Path(path).name, audio)},
                       headers={"Authorization": f"Bearer {key}"}, timeout=600)
        r.raise_for_status()
        spk = ("agent" if ch == agent_channel else "customer") if stereo else "unknown"
        segments += [(s["start"], s["end"], spk, s["text"]) for s in r.json()["segments"]]
    return _turns(segments)


def split_stereo_wav(path) -> list[bytes]:
    """Split a stereo WAV into two mono WAVs (stdlib only)."""
    with wave.open(str(path)) as w:
        if w.getnchannels() != 2:
            raise ValueError(f"{path}: expected a stereo WAV, got {w.getnchannels()} channel(s); pass --mono")
        width, rate, frames = w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())
    samples = array({1: "B", 2: "h", 4: "i"}[width], frames)
    out = []
    for ch in (0, 1):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as o:
            o.setnchannels(1), o.setsampwidth(width), o.setframerate(rate)
            o.writeframes(samples[ch::2].tobytes())
        out.append(buf.getvalue())
    return out


def _words(text):
    return re.findall(r"\w+", text.lower())


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance / reference length."""
    r, h = _words(ref), _words(hyp)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def number_accuracy(ref: str, hyp: str) -> float | None:
    """Share of the reference's numbers (digits or spoken) that appear in the hypothesis."""
    from .redact import digit_runs
    ref_nums = [d for _, _, d in digit_runs(ref)]
    hyp_nums = {d for _, _, d in digit_runs(hyp)}
    return sum(n in hyp_nums for n in ref_nums) / len(ref_nums) if ref_nums else None


def text_of(path) -> str:
    """Plain text of a .txt file or a call JSON."""
    p = Path(path)
    return " ".join(t.text for t in load_call(p).turns) if p.suffix == ".json" else p.read_text(encoding="utf-8")
