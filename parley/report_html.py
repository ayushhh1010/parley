"""Self-contained HTML report: every verdict next to the transcript words behind it (PRD S3 call detail, S5 calibration)."""
import re
from datetime import date
from html import escape

VERDICT = {"pass": ("✓", "Pass"), "fail": ("✕", "Fail"), "cannot_determine": ("?", "Cannot determine"),
           "na": ("–", "Not applicable")}
SEVERITY = {"critical": 0, "major": 1, "minor": 2}
REDACTED = re.compile(r"\[(CARD|CVV|AADHAAR|PAN|PHONE|EMAIL|NUMBER)\]")

CSS = """
:root{--bg:#f7f7f8;--panel:#fff;--line:#e3e4e8;--text:#1b1d22;--muted:#5d6270;--accent:#2848c9;
--pass:#1a7f45;--pass-bg:#e6f4ec;--fail:#c42b2b;--fail-bg:#fbeaea;--cd:#9a6200;--cd-bg:#fdf3e1;--na:#6b7080;--na-bg:#eef0f3;
--mark:#fff1b8;--target:#e8edff;--agent:#2848c9;--customer:#5d6270}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121317;--panel:#1a1c21;--line:#2c2f37;--text:#e8e9ec;
--muted:#9ba0ad;--accent:#8ea2ff;--pass:#5cc58a;--pass-bg:#16301f;--fail:#ff7b7b;--fail-bg:#3a1a1c;--cd:#f0b44c;--cd-bg:#35290f;
--na:#9ba0ad;--na-bg:#23262d;--mark:#4a3f10;--target:#232b4d;--agent:#8ea2ff;--customer:#9ba0ad}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;font-weight:600;margin:0 0 4px}h2{font-size:16px;font-weight:600;margin:32px 0 12px}
.sub,.muted{color:var(--muted)}.mono,.ts,.cid{font-family:"IBM Plex Mono",ui-monospace,monospace}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}
.scroll-x{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:500;color:var(--muted);white-space:nowrap}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.verdict{display:inline-flex;gap:4px;align-items:center;padding:1px 8px;border-radius:4px;font-size:12px;font-weight:500;white-space:nowrap}
.verdict.pass{color:var(--pass);background:var(--pass-bg)}.verdict.fail{color:var(--fail);background:var(--fail-bg)}
.verdict.cannot_determine{color:var(--cd);background:var(--cd-bg)}.verdict.na{color:var(--na);background:var(--na-bg)}
.sev{font-size:12px;color:var(--muted)}.sev.critical{color:var(--fail);font-weight:500}
details.call{margin:0 0 12px}
details.call>summary{cursor:pointer;padding:12px 16px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;list-style:none}
details.call>summary::-webkit-details-marker{display:none}
details.call>summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.call-id{font-weight:600}.counts{display:flex;gap:6px;flex-wrap:wrap}
.metrics{padding:0 16px 12px;font-size:12px}
.body{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);border-top:1px solid var(--line)}
.transcript{padding:12px 16px;max-height:70vh;overflow:auto;border-right:1px solid var(--line)}
.criteria{padding:12px 16px;max-height:70vh;overflow:auto}
.turn{display:grid;grid-template-columns:44px 72px minmax(0,1fr);gap:8px;padding:6px 8px;border-radius:6px;scroll-margin:12px}
.turn:target{background:var(--target)}
.turn p{margin:0;font-size:15px;line-height:24px}.ts{color:var(--muted);font-size:12px;line-height:24px}
.spk{font-size:12px;line-height:24px;font-weight:500;text-transform:capitalize}.agent .spk{color:var(--agent)}.customer .spk{color:var(--customer)}
mark{background:var(--mark);color:inherit;border-radius:2px;padding:0 1px}
.redacted{font-family:"IBM Plex Mono",monospace;font-size:12px;padding:0 4px;border:1px dashed var(--muted);border-radius:4px;color:var(--muted)}
.crit{padding:10px 0;border-bottom:1px solid var(--line)}.crit:last-child{border-bottom:0}
.crit-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.q{font-weight:500}
.why{margin:6px 0 0;color:var(--muted)}.note{margin:6px 0 0;font-size:12px;color:var(--cd)}
.chips{display:flex;flex-direction:column;gap:4px;margin-top:6px}
.chip{display:block;font-size:13px;color:var(--accent);text-decoration:none;padding:4px 8px;border:1px solid var(--line);border-radius:6px}
.chip:hover{border-color:var(--accent)}.chip .ts{margin-right:6px}
footer{margin-top:40px;font-size:12px;color:var(--muted)}
@media (max-width:800px){.body{grid-template-columns:1fr}.transcript{border-right:0;border-top:1px solid var(--line);max-height:none}
.criteria{order:-1;max-height:none}.turn{grid-template-columns:40px minmax(0,1fr)}.turn .spk{grid-column:2}.turn p{grid-column:1/-1}}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
"""


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _badge(verdict: str) -> str:
    icon, word = VERDICT[verdict]
    return f'<span class="verdict {verdict}"><span aria-hidden="true">{icon}</span>{word}</span>'


def _question(r) -> str:
    return r.get("question") or r["criterion_id"].replace("_", " ").capitalize()


def _turn(call_id: str, t: dict, quotes: list[str]) -> str:
    text, out, last = t["text"], [], 0
    # Quotes are exact transcript substrings (the verifier guarantees it), so a plain find is enough.
    for a, b in sorted({(i, i + len(q)) for q in quotes if (i := text.find(q)) >= 0}):
        if a >= last:
            out += [escape(text[last:a]), f"<mark>{escape(text[a:b])}</mark>"]
            last = b
    body = REDACTED.sub(r'<span class="redacted" title="redacted">\1</span>', "".join(out) + escape(text[last:]))
    return (f'<div class="turn {t["speaker"]}" id="{call_id}-t{t["id"]}"><span class="ts">{_mmss(t["start"])}</span>'
            f'<span class="spk">{t["speaker"]}</span><p>{body}</p></div>')


def _criterion(call_id: str, r: dict) -> str:
    chips = "".join(f'<a class="chip" href="#{call_id}-t{e["turn_id"]}"><span class="ts">{_mmss(e["start"])}</span>'
                    f'“{escape(e["quote"][:90])}{"…" if len(e["quote"]) > 90 else ""}”</a>' for e in r["evidence"])
    return (f'<div class="crit"><div class="crit-head">{_badge(r["verdict"])}<span class="sev {r["severity"]}">{r["severity"]}</span>'
            f'<span class="q">{escape(_question(r))}</span></div>'
            + (f'<p class="why">{escape(r["rationale"])}</p>' if r["rationale"] else "")
            + (f'<div class="chips">{chips}</div>' if chips else "")
            + (f'<p class="note">{escape(r["note"])}</p>' if r["note"] else "") + "</div>")


def _call(s: dict, open_: bool) -> str:
    cid = escape(s["call_id"])
    quotes = {}
    for r in s["results"]:
        if r["type"] != "timing":  # timing evidence is whole turns; highlighting them adds nothing
            for e in r["evidence"]:
                quotes.setdefault(e["turn_id"], []).append(e["quote"])
    results = sorted(s["results"], key=lambda r: (r["verdict"] != "fail", r["verdict"] != "cannot_determine",
                                                  SEVERITY.get(r["severity"], 3)))
    counts = {v: sum(r["verdict"] == v for r in s["results"]) for v in VERDICT}
    m = s["metrics"]
    return (f'<details class="call panel"{" open" if open_ else ""}><summary><span class="call-id mono">{cid}</span>'
            f'<span class="muted">{escape(s["agent_type"])} agent · {escape(s.get("language") or "language not set")} · {_mmss(m["duration"])}</span>'
            f'<span class="counts">' + "".join(f'{_badge(v)}&nbsp;{n}' for v, n in counts.items() if n) + "</span></summary>"
            f'<div class="metrics muted">Agent talk {m["agent_talk_ratio"]:.0%} · longest silence {m["longest_silence"]:.1f}s · '
            f'avg response {m["agent_response_latency_avg"]:.1f}s · interruptions {m["agent_interruptions"]}'
            + (f' · redacted {", ".join(f"{k} ×{n}" for k, n in s["redactions"].items())}' if s.get("redactions") else "")
            + '</div><div class="body"><div class="transcript" aria-label="Transcript">'
            + "".join(_turn(cid, t, quotes.get(t["id"], [])) for t in s["transcript"])
            + '</div><div class="criteria" aria-label="Verdicts">' + "".join(_criterion(cid, r) for r in results)
            + "</div></div></details>")


def _summary(runs) -> str:
    rows = []
    for cid in dict.fromkeys(r["criterion_id"] for s in runs for r in s["results"]):
        rs = [r for s in runs for r in s["results"] if r["criterion_id"] == cid]
        n = {v: sum(r["verdict"] == v for r in rs) for v in VERDICT}
        judged = n["pass"] + n["fail"]
        rows.append(f'<tr><td>{escape(_question(rs[0]))}</td><td><span class="sev {rs[0]["severity"]}">{rs[0]["severity"]}</span></td>'
                    f'<td class="num">{n["fail"]} of {judged}' + (f' ({n["fail"] / judged:.0%})' if judged else "") + "</td>"
                    f'<td class="num">{n["pass"]}</td><td class="num">{n["cannot_determine"]}</td><td class="num">{n["na"]}</td></tr>')
    return ('<div class="panel scroll-x"><table><thead><tr><th>Criterion</th><th>Severity</th><th>Failed</th><th>Passed</th>'
            '<th>Cannot determine</th><th>Not applicable</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def _calibration(stats, t) -> str:
    pct = lambda x: "–" if x is None else f"{x:.0%}"
    rows = "".join(
        f'<tr><td class="mono">{escape(s["criterion_id"])}</td><td class="num">{s["labels"]}</td>'
        f'<td class="num">{pct(s["agreement"])} <span class="muted">({s["ci"][0]:.0%}–{s["ci"][1]:.0%})</span></td>'
        f'<td class="num">{"–" if s["kappa"] is None else format(s["kappa"], ".2f")}</td><td class="num">{pct(s["fail_precision"])}</td>'
        f'<td class="num">{pct(s["fail_recall"])}</td><td>{escape(s["status"])}</td></tr>' for s in stats)
    return (f'<p class="muted">Trusted needs kappa ≥ {t["kappa_min"]}, agreement ≥ {t["agree_min"]:.0%}, ≥ {t["min_labels"]} labels '
            f'and ≥ {t["min_fails"]} human fails.</p><div class="panel scroll-x"><table><thead><tr><th>Criterion</th><th>Labels</th>'
            '<th>Agreement (95% CI)</th><th>Kappa</th><th>Fail precision</th><th>Fail recall</th><th>Status</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def render(runs: list[dict], stats=None, thresholds=None) -> str:
    first = runs[0] if runs else {"rubric": "?", "rubric_version": "?", "judge_model": "?"}
    synthetic = runs and all(s["call_id"].startswith("synth-") for s in runs)
    failing = sum(any(r["verdict"] == "fail" for r in s["results"]) for s in runs)
    runs = sorted(runs, key=lambda s: -sum(r["verdict"] == "fail" and r["severity"] == "critical" for r in s["results"]))
    title = f'{first["rubric"]} v{first["rubric_version"]}'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parley report: {escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>
<h1>Parley report: {escape(title)}</h1>
<p class="sub">{len(runs)} {"synthetic test " if synthetic else ""}calls · {failing} with at least one fail · judge {escape(first["judge_model"])} · {date.today():%d %b %Y}</p>
<p class="muted">Every verdict cites the exact words behind it. Code checks each quote against the transcript; a verdict whose quote cannot be found becomes "Cannot determine". Click a quote to jump to it. Personal data was masked before any model saw the text.</p>
<h2>Criteria across all calls</h2>{_summary(runs)}
{f"<h2>Agreement with reviewers</h2>{_calibration(stats, thresholds)}" if stats else ""}
<h2>Calls</h2><p class="muted">Calls with critical fails first.</p>
{"".join(_call(s, i == 0) for i, s in enumerate(runs))}
<footer>Generated by Parley, open-source conversation QA.</footer>
</main></body></html>"""
