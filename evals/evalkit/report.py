"""Aggregation and reporting over results.jsonl files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

Key = Tuple[str, str]


def load_results(path: str | Path) -> Dict[Key, Dict]:
    """Latest result per (suite, id) — reruns supersede earlier rows."""
    out: Dict[Key, Dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out[(row["suite"], row["id"])] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def suite_stats(results: Dict[Key, Dict]) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    for (suite, _), row in sorted(results.items()):
        s = stats.setdefault(suite, {"n": 0, "passed": 0})
        s["n"] += 1
        s["passed"] += 1 if row["passed"] else 0
    for s in stats.values():
        s["rate"] = s["passed"] / s["n"] if s["n"] else 0.0
    return stats


def format_report(results: Dict[Key, Dict], title: str = "Evaluation report") -> str:
    stats = suite_stats(results)
    total_n = sum(s["n"] for s in stats.values())
    total_p = sum(s["passed"] for s in stats.values())
    lines = [f"# {title}", ""]
    lines.append("| Suite | Passed | Total | Rate |")
    lines.append("|---|---|---|---|")
    for suite, s in sorted(stats.items()):
        lines.append(f"| {suite} | {s['passed']} | {s['n']} | {s['rate']:.0%} |")
    lines.append(f"| **overall** | **{total_p}** | **{total_n}** | "
                 f"**{(total_p / total_n if total_n else 0):.0%}** |")

    failures = [(k, r) for k, r in sorted(results.items()) if not r["passed"]]
    if failures:
        lines.append("")
        lines.append("## Failures")
        for (suite, tid), row in failures:
            lines.append(f"- `{suite}/{tid}`: {row['details'][:200]}")
    return "\n".join(lines)


def format_comparison(a: Dict[Key, Dict], b: Dict[Key, Dict],
                      name_a: str = "A", name_b: str = "B") -> str:
    stats_a, stats_b = suite_stats(a), suite_stats(b)
    suites = sorted(set(stats_a) | set(stats_b))
    lines = [f"# Comparison: {name_a} vs {name_b}", ""]
    lines.append(f"| Suite | {name_a} | {name_b} | Δ |")
    lines.append("|---|---|---|---|")
    for suite in suites:
        ra = stats_a.get(suite, {}).get("rate")
        rb = stats_b.get(suite, {}).get("rate")
        cell_a = f"{ra:.0%}" if ra is not None else "—"
        cell_b = f"{rb:.0%}" if rb is not None else "—"
        delta = (f"{(rb - ra):+.0%}" if ra is not None and rb is not None else "—")
        lines.append(f"| {suite} | {cell_a} | {cell_b} | {delta} |")

    common = set(a) & set(b)
    fixed = sorted(k for k in common if not a[k]["passed"] and b[k]["passed"])
    regressed = sorted(k for k in common if a[k]["passed"] and not b[k]["passed"])
    if fixed:
        lines += ["", f"## Fixed in {name_b} ({len(fixed)})"]
        lines += [f"- `{s}/{t}`" for s, t in fixed]
    if regressed:
        lines += ["", f"## Regressed in {name_b} ({len(regressed)})"]
        lines += [f"- `{s}/{t}`: {b[(s, t)]['details'][:150]}" for s, t in regressed]
    return "\n".join(lines)
