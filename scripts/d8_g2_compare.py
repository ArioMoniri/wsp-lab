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
print(
    json.dumps(
        {
            "pooled": report["pooled"],
            "source_hash": report["source_hash"][:12],
            "dirty": report["dirty"],
        },
        indent=1,
        sort_keys=True,
    )
)
any_key = next(iter(report["per_entry"][ids[0]]["mmlu_pro"]))
print(
    "per-entry keys:",
    sorted(report["per_entry"][ids[0]]["mmlu_pro"].keys()),
    sorted(report["per_entry"][ids[0]]["hellaswag"].keys()),
)
(ROOT / "data/runs/d8/g2_printed_rows.json").write_text(json.dumps(printed, indent=1))
