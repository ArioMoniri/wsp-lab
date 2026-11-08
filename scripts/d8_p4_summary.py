#!/usr/bin/env python3
"""D8 step 6c: summary table of the recovered-set P4 suite (data/runs/p4/d3v2r-*) next to the retained d3v2 summary.

    d8_p4_summary.py --date 2026-11-06 [--out docs/reviews/P4/d3v2r-summary-<date>.md]

Reads the new metrics.json files (per run: seed_summary, seed-0 baselines, targets.n, paired n_clusters) and, for the
original runs, only committed documents: docs/reviews/P4/d3v2-summary-2026-11-05.md (printed table) and the six-digit source
values of docs/reviews/P4/paper-v2-audit-2026-11-05.md (seed means and ranges, rows, clusters). Writes the table and a
per-run comparison with the reason each run differs (the recovered set has fewer rows by construction, so every run differs).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P4 = Path("/home/ubuntu/wsp-data/runs/p4")
ORIG_SUMMARY = ROOT / "docs/reviews/P4/d3v2-summary-2026-11-05.md"
AUDIT = ROOT / "docs/reviews/P4/paper-v2-audit-2026-11-05.md"
TARGETS = ("mmlu_pro", "hellaswag")


def fmt(x, nd=2):
    return "nan" if x is None or math.isnan(x) else f"{x:.{nd}f}"


def cell(node):
    return f"{fmt(node['mean'])} [{fmt(node['range'][0])},{fmt(node['range'][1])}]"


def original_values():
    """run -> {(target, stat): value, 'rows': n, 'clusters': n} from the committed audit log (six digits)."""
    out = {}
    for line in AUDIT.read_text().splitlines():
        m = re.match(
            r"- (?:PASS|FAIL) `suite (d3v2-\S+) (rows|clusters)`: shown (\d+) vs source (\d+)", line
        )
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = int(m.group(4))
            continue
        m = re.match(
            r"- (?:PASS|FAIL) `suite (d3v2-\S+) (mmlu_pro|hellaswag) (mean|min|max)`: shown .* vs source (-?[\d.e-]+) ",
            line,
        )
        if m:
            out.setdefault(m.group(1), {})[(m.group(2), m.group(3))] = float(m.group(4))
    return out


def original_table():
    rows = {}
    for line in ORIG_SUMMARY.read_text().splitlines():
        if line.startswith("| m") and "---" not in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells
    return rows


PREFIX = "d3v2r"


def run_dirs():
    return sorted(p for p in P4.glob(f"{PREFIX}-*") if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--prefix",
        default="d3v2r",
        help="run-name prefix: d3v2r (recovered set) or d3v2f (full set after the chain replay)",
    )
    a = ap.parse_args()
    global PREFIX
    PREFIX = a.prefix
    out_path = a.out or ROOT / f"docs/reviews/P4/{PREFIX}-summary-{a.date}.md"
    orig_vals, orig_tab = original_values(), original_table()
    lines = [
        (
            f"# P4 suite, protocol v2, full set after the chain replay (runs data/runs/p4/d3v2f-*; {a.date}; 3 full seeds: mean [min,max]; baselines from seed 0)"
            if PREFIX == "d3v2f"
            else f"# P4 suite, protocol v2, recovered set (runs data/runs/p4/d3v2r-*; regenerated {a.date}; 3 full seeds: mean [min,max]; baselines from seed 0)"
        ),
        "",
        (
            "Full-set rerun after the chain replay (regeneration report §7): same code, protocol and run list as `d3v2-summary-2026-11-05.md`; "
            + "features of the 351 regenerated chain snapshots added to the recovered set; complete-ancestry rule. 342 matched rows in 278 ancestry "
            + "clusters against the original 345 / 278: the 3 missing rows depend on the 7 entries that stay lost. **This is the closest available "
            + "re-instantiation of the 5 Nov suite, not a bitwise reproduction**: lineages through a re-trained LoRA step carry re-instantiated bytes "
            + "(306 zoo entries), so values are expected to be close to, not equal to, the logged ones."
            if PREFIX == "d3v2f"
            else "Recovered-set rerun after the D8 data loss: same code, protocol and run list as `d3v2-summary-2026-11-05.md`, features re-extracted from the restored and aliased deltas, "
            + "and only the rows whose whole history (chain predecessors back to the root, task-vector sources) has features. **This is a different experiment from the 5 Nov suite, not a reproduction of it**: "
            + "48 matched chain rows are missing because their deltas are lost (see the regeneration report, step 6). Do not mix the two tables."
        ),
        "",
        "| model-split | mmlu_pro ρ | hellaswag ρ | KL ρ | wikitext ρ | StatNN mmlu/hs | descriptor-only mmlu/hs | delta-norm mmlu/hs | rows / clusters |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    compare = [
        "| run | rows orig → rerun | clusters orig → rerun | mmlu_pro ρ mean orig → rerun | hellaswag ρ mean orig → rerun | reason it differs |",
        "|---|---|---|---|---|---|",
    ]
    status = []
    for d in run_dirs():
        name = d.name.removeprefix(f"{PREFIX}-")
        mfile = d / "metrics.json"
        if not mfile.exists():
            lines.append(f"| {name} | no metrics | | | | | | | |")
            compare.append(
                f"| d3v2-{name} | — | — | — | — | no metrics.json in the recovered run (see log) |"
            )
            status.append((name, "no metrics"))
            continue
        m = json.loads(mfile.read_text())
        s0 = json.loads((d / "seed-0/metrics.json").read_text())
        ss = m["seed_summary"]
        cells = []
        for t in TARGETS:
            node = ss.get(t, {}).get("spearman")
            cells.append(cell(node) if node else "—")
        base = {}
        for b in ("statnn", "descriptor_only", "delta_norm"):
            vals = []
            for t in TARGETS:
                v = (
                    s0["baselines"]
                    .get(b, {})
                    .get("targets", {})
                    .get(t, {})
                    .get("spearman", {})
                    .get("value")
                )
                vals.append(fmt(v) if v is not None else "—")
            base[b] = " / ".join(vals)
        n = s0["targets"][TARGETS[0]]["n"]
        try:
            ncl = s0["paired_differences"]["statnn"][TARGETS[0]]["n_clusters"]
        except KeyError:
            ncl = "—"
        lines.append(
            f"| {name} | {cells[0]} | {cells[1]} | — | — | {base['statnn']} | {base['descriptor_only']} | {base['delta_norm']} | {n} / {ncl} |"
        )
        o = orig_vals.get(f"d3v2-{name}", {})
        ob = orig_tab.get(name)
        if PREFIX == "d3v2f":
            reasons = [
                "row set: 3 rows depend on the 7 entries that stay lost; lineages through a re-trained LoRA step are re-instantiated, not identical"
                if o.get("rows")
                else "no original rows in the audit log"
            ]
            if "holdout-chain" in name:
                reasons.append("the chain holdout's test rows fall from 98 to 95")
        else:
            reasons = [
                "row set: 48 chain rows with a lost ancestor delta removed from every fold"
                if o.get("rows")
                else "no original rows in the audit log"
            ]
            if "holdout-chain" in name:
                reasons.append("the chain holdout's test rows fall from 98 to 50")
        if name.startswith("m3"):
            reasons.append("successor pairs with both ends' features: 98 of 521, 25 evaluated")
        compare.append(
            "| d3v2-{n} | {r0} → {r1} | {c0} → {c1} | {a0} → {a1} | {b0} → {b1} | {why} |".format(
                n=name,
                r0=o.get("rows", "—"),
                r1=n,
                c0=o.get("clusters", "—"),
                c1=ncl,
                a0=fmt(o.get(("mmlu_pro", "mean")), 4)
                if ("mmlu_pro", "mean") in o
                else (ob[1].split(" ")[0] if ob else "—"),
                a1=fmt(ss["mmlu_pro"]["spearman"]["mean"], 4) if "mmlu_pro" in ss else "—",
                b0=fmt(o.get(("hellaswag", "mean")), 4)
                if ("hellaswag", "mean") in o
                else (ob[2].split(" ")[0] if ob else "—"),
                b1=fmt(ss["hellaswag"]["spearman"]["mean"], 4) if "hellaswag" in ss else "—",
                why="; ".join(reasons),
            )
        )
        status.append((name, "ok"))
    lines += [
        "",
        "KL and WikiText are not rate targets and are not trained (the original table shows — for them too).",
        "",
        "## Differences from the 5 Nov summary, row by row",
        "",
        "Original values are the six-digit source values of the committed audit log (`paper-v2-audit-2026-11-05.md`); where the log lacks a run, the two-digit printed value is used.",
        "",
        *compare,
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(out_path, status)


if __name__ == "__main__":
    sys.exit(main())
