"""Rubric files: versioned YAML, validated on load (PRD F4)."""
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

METRICS = Literal["duration", "agent_talk_ratio", "longest_silence", "overlaps", "agent_interruptions",
                  "agent_response_latency_avg", "agent_response_latency_max"]
EMOTION = re.compile(r"\b(emotion\w*|feel\w*|mood|stress\w*|empath\w*|sentiment|frustrat\w*|angry|anger|"
                     r"happy|upset|tone of voice|enthusias\w*)\b", re.I)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Criterion(Strict):
    id: str
    type: Literal["llm", "phrase", "timing"]
    question: str
    severity: Literal["critical", "major", "minor"] = "major"
    applies_to: Literal["human", "ai", "both"] = "both"
    # llm
    pass_when: str = ""
    fail_when: str = ""
    na_when: str = ""
    examples: list[str] = []
    # phrase
    phrases: list[str] = []
    speaker: Literal["agent", "customer", "any"] = "agent"
    within_seconds: float | None = None
    # timing
    metric: METRICS | None = None
    max: float | None = None
    min: float | None = None

    @model_validator(mode="after")
    def _fields_for_type(self):
        if self.type == "phrase" and not self.phrases:
            raise ValueError(f"criterion {self.id!r}: phrase criteria need 'phrases'")
        if self.type == "timing" and (self.metric is None or (self.max is None and self.min is None)):
            raise ValueError(f"criterion {self.id!r}: timing criteria need 'metric' and 'max' or 'min'")
        return self


class Judge(Strict):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-opus-5-5"
    base_url: str | None = None  # openai: any OpenAI-compatible endpoint (Gemini, Groq, Ollama, vLLM)
    api_key_env: str | None = None

    @model_validator(mode="after")
    def _openai_needs_url(self):
        if self.provider == "openai" and not self.base_url:
            raise ValueError("judge: provider 'openai' needs 'base_url'")
        return self


# `--judge <name>` shortcuts. Gemini's free tier may use content to improve Google's products: synthetic data only.
PRESETS = {
    "claude": Judge(),
    "gemini": Judge(provider="openai", model="gemini-3.8-flash", api_key_env="GEMINI_API_KEY",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai"),
    "gemini-lite": Judge(provider="openai", model="gemini-3.1-flash-lite", api_key_env="GEMINI_API_KEY",
                         base_url="https://generativelanguage.googleapis.com/v1beta/openai"),
    "groq": Judge(provider="openai", model="openai/gpt-oss-120b", api_key_env="GROQ_API_KEY",
                  base_url="https://api.groq.com/openai/v1"),
    "qwen": Judge(provider="openai", model="qwen/qwen3.8-27b", api_key_env="GROQ_API_KEY",
                  base_url="https://api.groq.com/openai/v1"),
}


class Rubric(Strict):
    name: str
    version: int
    description: str = ""
    judge: Judge = Judge()
    criteria: list[Criterion]

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [c.id for c in self.criteria]
        if dupes := {i for i in ids if ids.count(i) > 1}:
            raise ValueError(f"duplicate criterion ids: {sorted(dupes)}")
        return self

    def warnings(self) -> list[str]:
        """Criteria that look like emotion inference, which the EU AI Act bans for employees (PRD §14.1)."""
        return [f"{c.id}: mentions emotions ({m.group()!r}); score an observable behavior instead"
                for c in self.criteria
                if (m := EMOTION.search(" ".join([c.question, c.pass_when, c.fail_when])))]


def load(path) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
