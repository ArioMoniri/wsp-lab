#!/usr/bin/env python3
"""D8 step 5c: finish the G2 same-item bridge against the regenerated H0 fast-tier records.

The stock bridge (scripts/tf_v1_g2_bridge.py at 9922ea3) scores the 11 pilot records and then compares each with
``data/evals/qwen3.5-0.8b/<id>/records.jsonl``. After the incident that path holds the Lambda (H1) record of each control,
so the comparison is redirected to data/runs/d8/g2_fast_records/<id>/records.jsonl, which scripts/d8_controls_compare.py
stages from the regenerated H0 seed-0 records. The bridge module is imported unchanged; only its EVALS directory is
replaced. Writes tf_v1/bridge_g2/g2_report.json (same file name as the original) and prints the per-entry table next to
the one printed in docs/RESULTS.md ("TF-v1 G2 (round 3)").

    d8_g2_compare.py <path to the 9922ea3 worktree's scripts/tf_v1_g2_bridge.py>
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bridge_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("g2bridge", bridge_path)
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
g2.EVALS = ROOT / "data/runs/d8/g2_fast_records"  # <id>/records.jsonl
ids = ["base", *(ROOT / "data/runs/d3_shards/controls.txt").read_text().split()]
report = g2.build_report(ids)
printed = {}
text = (ROOT / "docs/RESULTS.md").read_text()
block = text[text.index("## TF-v1 G2 (round 3)") :]
for line in block.splitlines():
    m = re.match(r"\| (`?)([0-9a-f]{16}|base)\1 \|", line)
    if m:
        printed[m.group(2)] = [c.strip() for c in line.strip().strip("|").split("|")]


def fmt_e(x):
    return f"{x:.1e}"


rows = {}
mism = []
for eid in ids:
    m, h = report["per_entry"][eid]["mmlu_pro"], report["per_entry"][eid]["hellaswag"]
    got = [
        f"{m['accuracy_fast']:.2f}",
        f"{m['accuracy_tf_v1']:.2f}",
        f"{m['accuracy_abs_diff']:.2f}",
        f"{m['argmax_agreement']:.2f}",
        f"{fmt_e(m['gold_score_abs_diff_median'])} / {m['gold_score_abs_diff_p99']:.3f} / {m['gold_score_abs_diff_max']:.3f}",
        f"{h['accuracy_fast']:.2f} / {h['accuracy_tf_v1']:.2f}",
        f"{h['gold_score_abs_diff_max']:.4f}",
    ]
    want = printed[eid][1:]
    want = (
        [want[0], want[1], want[2], want[4], want[5], want[6], want[8]] if len(want) == 9 else want
    )
    rows[eid] = {"regenerated": got, "printed": want, "match": got == want}
    if got != want:
        mism.append(eid)
pooled = report["pooled"]
extra = {
    "mmlu_median_p99_argmax_worst": [
        fmt_e(pooled["mmlu_pro"]["gold_score_abs_diff_median_max"]),
        f"{pooled['mmlu_pro']['gold_score_abs_diff_p99_max']:.3f}",
        f"{pooled['mmlu_pro']['argmax_agreement_min']:.2f}",
    ],
    "mmlu_printed": ["2.2e-16", "0.066", "0.99"],
    "hellaswag_mean_correctness_agreement": round(
        pooled["hellaswag"]["correctness_agreement_mean"], 4
    ),
    "hellaswag_printed_mean_correctness_agreement": 0.9991,
    "hellaswag_worst_entry_correctness": round(pooled["hellaswag"]["correctness_agreement_min"], 2),
    "hellaswag_worst_argmax": round(pooled["hellaswag"]["argmax_agreement_min"], 2),
    "hellaswag_max_score_abs_diff": round(pooled["hellaswag"]["gold_score_abs_diff_max"], 4),
    "hellaswag_printed_max_score_abs_diff": 0.0111,
    "prereg_mmlu_criteria_pass": pooled["prereg_mmlu_criteria"]["pass"],
}
summary = {
    "rows": rows,
    "rows_mismatching": mism,
    "all_rows_match": not mism,
    "pooled": extra,
    "source_hash": report["source_hash"][:12],
    "dirty": report["dirty"],
}
(ROOT / "data/runs/d8/g2_compare.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
print(json.dumps(summary, indent=1, sort_keys=True))
