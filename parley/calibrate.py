"""Agreement between Parley and human reviewers, per criterion (PRD F6).

labels.csv columns: call_id, criterion_id, reviewer, verdict (pass | fail | na | cannot_determine).
"""
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations

# PRD §12.5 defaults for marking a criterion Trusted.
THRESHOLDS = {"min_labels": 20, "min_fails": 3, "kappa_min": 0.6, "agree_min": 0.85}


def cohen_kappa(pairs) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def wilson(k: int, n: int, z=1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if not n:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def load(scores_path, labels_path):
    runs = [json.loads(line) for line in open(scores_path, encoding="utf-8-sig") if line.strip()]
    parley = {(s["call_id"], r["criterion_id"]): r["verdict"] for s in runs for r in s["results"]}
    severity = {r["criterion_id"]: r["severity"] for s in runs for r in s["results"]}
    human = defaultdict(dict)
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            human[(row["call_id"], row["criterion_id"])][row["reviewer"]] = row["verdict"].strip().lower()
    return runs, parley, severity, human


def criterion_stats(cid, parley, human, severity, min_labels, min_fails, kappa_min, agree_min):
    keys = [k for k in human if k[1] == cid]
    pairs = [(parley[k], v, k[0]) for k in keys if k in parley for v in human[k].values()]
    hh = [(a, b) for k in keys for a, b in combinations(human[k].values(), 2)]
    n, agree = len(pairs), sum(p == h for p, h, _ in pairs)
    tp = sum(p == h == "fail" for p, h, _ in pairs)
    flagged, actual = sum(p == "fail" for p, _, _ in pairs), sum(h == "fail" for _, h, _ in pairs)
    kappa = cohen_kappa([(p, h) for p, h, _ in pairs])
    precision = tp / flagged if flagged else None
    if n < min_labels:
        status = f"Assist: {n} labels, need {min_labels}"
    elif actual < min_fails:
        status = f"Assist: only {actual} human fails, need {min_fails}"
    elif (kappa is not None and kappa >= kappa_min and agree / n >= agree_min
          and (severity.get(cid) != "critical" or (precision or 0) >= 0.9)):
        status = "Trusted"
    else:
        status = "Assist"
    return {"criterion_id": cid, "severity": severity.get(cid, "?"), "labels": n,
            "agreement": agree / n if n else None, "ci": wilson(agree, n), "kappa": kappa,
            "human_agreement": sum(a == b for a, b in hh) / len(hh) if hh else None,
            "fail_precision": precision, "fail_recall": tp / actual if actual else None, "status": status,
            "disagreements": sorted({c for p, h, c in pairs if p != h})}


def reviewer_warnings(human) -> list[str]:
    """Flag reviewers who give the same verdict almost every time (PRD F6 edge case, §14.3 label poisoning)."""
    by_rev = defaultdict(Counter)
    for revs in human.values():
        for rev, v in revs.items():
            by_rev[rev][v] += 1
    out = []
    for rev, c in by_rev.items():
        total = sum(c.values())
        verdict, top = c.most_common(1)[0]
        if total >= 10 and top / total >= 0.95:
            out.append(f"{rev} labeled {top}/{total} as {verdict!r}; check they are reviewing, not rubber-stamping.")
    return out


def compute(scores_path, labels_path, **thresholds):
    """(runs, per-criterion stats, human labels, thresholds used)."""
    runs, parley, severity, human = load(scores_path, labels_path)
    t = THRESHOLDS | thresholds
    cids = list(dict.fromkeys(r["criterion_id"] for s in runs for r in s["results"]))
    return runs, [criterion_stats(c, parley, human, severity, **t) for c in cids], human, t


def report(scores_path, labels_path, **thresholds) -> str:
    runs, stats, human, t = compute(scores_path, labels_path, **thresholds)
    parley = {(s["call_id"], r["criterion_id"]) for s in runs for r in s["results"]}
    pct = lambda x: "–" if x is None else f"{x:.0%}"
    num = lambda x: "–" if x is None else f"{x:.2f}"
    labeled = [s for s in stats if s["labels"]]
    trusted = sum(s["status"] == "Trusted" for s in stats)
    lines = [f"# Calibration report: {runs[0]['rubric']} v{runs[0]['rubric_version']}" if runs else "# Calibration report", "",
             f"{sum(s['labels'] for s in stats)} labels from {len({r for v in human.values() for r in v})} reviewer(s) "
             f"on {len({k[0] for k in human if k in parley})} scored calls. Judge: {runs[0]['judge_model'] if runs else '?'}.", "",
             f"**Parley agrees with your reviewers at ≥{t['agree_min']:.0%} on "
             f"{sum((s['agreement'] or 0) >= t['agree_min'] for s in labeled)} of {len(labeled)} labeled criteria; "
             f"{trusted} can be Trusted.**", "",
             f"Trusted needs kappa ≥ {t['kappa_min']}, agreement ≥ {t['agree_min']:.0%}, ≥ {t['min_labels']} labels, "
             f"≥ {t['min_fails']} human fails, and fail precision ≥ 90% for critical criteria.", "",
             "| Criterion | Severity | Labels | Agreement (95% CI) | Kappa | Human–human | Fail precision | Fail recall | Suggested status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        lo, hi = s["ci"]
        lines.append(f"| {s['criterion_id']} | {s['severity']} | {s['labels']} | {pct(s['agreement'])} ({lo:.0%}–{hi:.0%}) | "
                     f"{num(s['kappa'])} | {pct(s['human_agreement'])} | {pct(s['fail_precision'])} | "
                     f"{pct(s['fail_recall'])} | {s['status']} |")
    split = [s for s in stats if s["disagreements"]]
    if split:
        lines += ["", "## Calls where people and Parley disagree", ""]
        lines += [f"- **{s['criterion_id']}**: {', '.join(s['disagreements'][:15])}"
                  + (f" (+{len(s['disagreements']) - 15} more)" if len(s["disagreements"]) > 15 else "") for s in split]
    if warns := reviewer_warnings(human):
        lines += ["", "## Reviewer warnings", ""] + [f"- {w}" for w in warns]
    return "\n".join(lines) + "\n"
