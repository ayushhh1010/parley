"""Local web app (`parley serve`): drop in calls, pick a rubric and judge, review verdicts with their quotes.

Single user, binds to 127.0.0.1, no sign-in. Runs are saved under data/runs/ so they survive restarts.
ponytail: one background thread per run, scoring calls one at a time (free API tiers rate-limit hard);
a Postgres job queue replaces this in the MVP.
"""
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from .judge import _hook as judge_hook
from .judge import make_judge
from .report_html import CSS, _call, render
from .rubric import PRESETS, Judge, load
from .score import score_call
from .transcribe import Call

PAGE = Path(__file__).with_name("app.html")


class NewRun(BaseModel):
    rubric: str
    judge: str
    calls: list[dict] = []
    use_samples: bool = False


def _ready(cfg: Judge) -> bool:
    """Whether the judge's API key is set (never returns the key itself)."""
    if cfg.provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return not cfg.api_key_env or bool(os.environ.get(cfg.api_key_env))


def _summary(s: dict) -> dict:
    if "error" in s:
        return {"call_id": s["call_id"], "error": s["error"]}
    results = s["results"]
    return {"call_id": s["call_id"], "agent_type": s["agent_type"], "language": s.get("language", ""),
            "counts": {v: sum(r["verdict"] == v for r in results) for v in ("fail", "cannot_determine", "pass", "na")},
            "critical": [r.get("question") or r["criterion_id"] for r in results
                         if r["verdict"] == "fail" and r["severity"] == "critical"]}


def create_app(rubrics_dir="rubrics", samples_dir="samples", runs_dir="data/runs", judge_factory=make_judge) -> FastAPI:
    rubrics_dir, samples_dir, runs_dir = Path(rubrics_dir), Path(samples_dir), Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}  # in-memory state of runs started or opened in this session
    app = FastAPI(title="Parley", docs_url="/api/docs")

    def get_run(run_id: str) -> dict:
        if run_id not in runs:
            meta_path = runs_dir / f"{run_id}.json"
            if not run_id.isalnum() or not meta_path.exists():
                raise HTTPException(404, "run not found")
            lines = (runs_dir / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
            runs[run_id] = {"meta": json.loads(meta_path.read_text(encoding="utf-8")),
                            "results": [json.loads(l) for l in lines if l.strip()], "finished": True}
        return runs[run_id]

    def work(run: dict, rubric, judge, calls: list[Call]):
        judge_hook.fn = lambda message: run.__setitem__("status", message)  # retry notices from this thread
        for i, call in enumerate(calls, 1):
            run["status"] = f"Scoring {call.id} ({i} of {len(calls)})"
            try:
                s = score_call(call, rubric, judge)
            except Exception as e:  # show any failure (bad key, network) on the call's row instead of killing the run
                s = {"call_id": call.id, "error": f"{type(e).__name__}: {e}"}
            run["results"].append(s)
            with open(runs_dir / f"{run['meta']['id']}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        run["status"], run["finished"] = "", True

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE.read_text(encoding="utf-8").replace("/*REPORT_CSS*/", CSS)

    @app.get("/api/setup")
    def setup():
        rubrics = []
        for p in sorted(rubrics_dir.glob("*.yaml")):
            try:
                r = load(p)
                rubrics.append({"file": p.name, "name": r.name, "version": r.version, "criteria": len(r.criteria),
                                "description": r.description})
            except (ValueError, OSError) as e:
                rubrics.append({"file": p.name, "error": str(e)[:300]})
        return {"rubrics": rubrics,
                "judges": [{"name": n, "model": cfg.model, "ready": _ready(cfg)} for n, cfg in PRESETS.items()],
                "samples": len(list(samples_dir.glob("*.json")))}

    @app.post("/api/runs")
    def start(req: NewRun):
        path = rubrics_dir / Path(req.rubric).name  # name only: no reading files outside rubrics/
        if not path.exists() or req.judge not in PRESETS:
            raise HTTPException(400, "Unknown checklist or judge.")
        rubric = load(path).model_copy(update={"judge": PRESETS[req.judge]})
        raw = req.calls + ([json.loads(p.read_text(encoding="utf-8")) for p in sorted(samples_dir.glob("*.json"))]
                           if req.use_samples else [])
        if not raw:
            raise HTTPException(400, "No calls yet: drop call files or use the sample calls.")
        calls = []
        for n, c in enumerate(raw, 1):
            try:
                calls.append(Call.model_validate(c))
            except ValidationError as e:
                err = e.errors()[0]
                name = c.get("id", f"#{n}") if isinstance(c, dict) else f"#{n}"
                raise HTTPException(400, f"Call {name} isn't in Parley's call format: '{'.'.join(map(str, err['loc']))}' "
                                         f"{err['msg'].lower()}. See \"Call format\" in the README.")
        run_id = uuid.uuid4().hex[:10]
        meta = {"id": run_id, "rubric": rubric.name, "rubric_version": rubric.version, "judge": req.judge,
                "model": rubric.judge.model, "created": datetime.now().isoformat(timespec="seconds"),
                "total": len(calls)}
        (runs_dir / f"{run_id}.json").write_text(json.dumps(meta), encoding="utf-8")
        (runs_dir / f"{run_id}.jsonl").write_text("", encoding="utf-8")
        runs[run_id] = {"meta": meta, "results": [], "finished": False}
        judge = judge_factory(rubric.judge) if any(c.type == "llm" for c in rubric.criteria) else None
        threading.Thread(target=work, args=(runs[run_id], rubric, judge, calls), daemon=True).start()
        return {"id": run_id}

    @app.get("/api/runs")
    def list_runs():
        out = []
        for p in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            meta = json.loads(p.read_text(encoding="utf-8"))
            done = len(runs[meta["id"]]["results"]) if meta["id"] in runs else sum(
                1 for l in open(p.with_suffix(".jsonl"), encoding="utf-8") if l.strip())
            out.append(meta | {"done": done, "stopped": meta["id"] not in runs and done < meta["total"]})
        return out

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str):
        run = get_run(run_id)
        return {"meta": run["meta"], "finished": run["finished"], "status": run.get("status", ""),
                "results": [_summary(s) for s in run["results"]]}

    @app.get("/api/runs/{run_id}/calls/{call_id}", response_class=HTMLResponse)
    def call_detail(run_id: str, call_id: str):
        s = next((s for s in get_run(run_id)["results"] if s["call_id"] == call_id and "error" not in s), None)
        if s is None:
            raise HTTPException(404, "call not found")
        return _call(s, True)

    @app.get("/runs/{run_id}/report", response_class=HTMLResponse)
    def full_report(run_id: str):
        return render([s for s in get_run(run_id)["results"] if "error" not in s])

    return app
