#!/usr/bin/env python3
"""D8 step 5b verification: the regenerated H0 seed-0 fast-tier records of the ten controls (and a base).

Usage: d8_controls_compare.py <data/runs/p2-batch/<run_id>>

Checks, against the surviving H0 record of each control (zoo/<id>/eval.json, run 20261104-202947-1dcb69a2 and
eval.local-run1.json, run 20261104-104702-5fac075c): every numeric field of eval.json bitwise equal. Base against the
regenerated 6f274a83 base (devices/H0/<run>): item records equal. Then stages data/runs/d8/g2_fast_records/<id>/records.jsonl
(symlinks) for the G2 bridge comparison.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
EV = ROOT / "data/evals/qwen3.5-0.8b"
CONTROLS = (ROOT / "data/runs/d3_shards/controls.txt").read_text().split()
BASE_6F = EV / "base/devices/H0/20261106-044003-37b3ae3d"
METRICS = [
    "gsm8k",
    "math500",
    "ifeval",
    "mmlu_pro",
    "hellaswag",
    "wikitext_ppl",
    "refusal_rate",
    "over_refusal",
    "kl_to_base",
    "loop_rate",
    "kl_first_token_mean",
    "kl_full_sequence_mean",
]


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    panel = Path(sys.argv[1])
    out = {"panel": str(panel), "controls": {}}
    stage = ROOT / "data/runs/d8/g2_fast_records"
    for i in CONTROLS + ["base"]:
        d = panel / i
        new = json.loads((d / "eval.json").read_text())
        recs = jl(d / "records.jsonl")
        if i == "base":
            ref_payloads = {
                "regenerated 6f274a83 base": json.loads((BASE_6F / "eval.json").read_text())
            }
            ref_recs = jl(BASE_6F / "records.jsonl")
            same_records = [{k: v for k, v in a.items() if k != "edit_id"} for a in recs] == [
                {k: v for k, v in a.items() if k != "edit_id"} for a in ref_recs
            ]
        else:
            ref_payloads = {
                "zoo eval.json (H0 retest 20261104-202947-1dcb69a2)": json.loads(
                    (ZOO / i / "eval.json").read_text()
                ),
                "eval.local-run1.json (20261104-104702-5fac075c)": json.loads(
                    (ZOO / i / "eval.local-run1.json").read_text()
                ),
            }
            same_records = None
        row = {
            "source_hash": new["source_hash"][:12],
            "dirty": new.get("dirty"),
            "batch_size": new.get("batch_size"),
            "status": new.get("eval_status"),
            "records": len(recs),
        }
        for name, ref in ref_payloads.items():
            row[name] = {m: new[m] == ref[m] for m in METRICS if m in new and m in ref}
            row[name]["all_metrics_bitwise_equal"] = all(row[name].values())
            row[name]["max_abs"] = max(
                abs(new[m] - ref[m]) for m in METRICS if m in new and m in ref
            )
        if same_records is not None:
            row["records_equal_to_regenerated_6f274a83_base"] = same_records
        out["controls"][i] = row
        (stage / i).mkdir(parents=True, exist_ok=True)
        link = stage / i / "records.jsonl"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((BASE_6F if i == "base" else d) / "records.jsonl")
    out["all_controls_equal_to_H0_retest"] = all(
        all(
            v["all_metrics_bitwise_equal"]
            for k, v in r.items()
            if isinstance(v, dict) and "all_metrics_bitwise_equal" in v
        )
        for r in out["controls"].values()
    )
    path = ROOT / "data/runs/d8/controls_compare.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
