#!/usr/bin/env python3
"""Per-entry minutes and completeness of the D3 TF-v1 scoring queue (reads eval.json only).

Appends one JSON line per newly completed entry to data/runs/h1a/d3_stratum_minutes.jsonl and prints
a summary (done / total, warm median wall minutes, projected remaining hours). With --check it exits
non-zero unless every scored id of the stratum list (and, with --null, the null-family list) has a
complete eval.json under the single expected `source_hash`.

Usage: tfv1_d3_minutes.py [--check] [--null] [--source-hash HASH]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / "data/evals/qwen3.5-0.8b/tf_v1"
IDS = ROOT / "data/runs/h1a/d3_ids"
LOG = ROOT / "data/runs/h1a/d3_stratum_minutes.jsonl"


def read_ids(kind):
    return Path(f"{IDS}.{kind}.txt").read_text().split()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--null", action="store_true", help="include the null-family list")
    parser.add_argument("--source-hash")
    args = parser.parse_args()
    ids = read_ids("stratum") + (read_ids("null") if args.null else [])
    logged = (
        {json.loads(x)["edit_id"] for x in LOG.read_text().splitlines()} if LOG.exists() else set()
    )
    base = json.loads((TF / "base/eval.json").read_text())
    source = args.source_hash or base["source_hash"]
    done, wall, bad, new = 0, [], [], []
    for i in ids:
        path = TF / i / "eval.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        if m["eval_status"] != "complete" or m["source_hash"] != source or m["dirty"]:
            bad.append(i)
            continue
        done += 1
        wall.append(m["wall_seconds"] / 60)
        if i not in logged:
            new.append(
                {
                    "edit_id": i,
                    "family": m["family"],
                    "wall_minutes": m["wall_seconds"] / 60,
                    "scoring_minutes": m["minutes"],
                    "stage_seconds": m["stage_seconds"],
                    "cache_mode": m["tf_cache"]["mode"],
                    "run_id": m["run_id"],
                    "source_hash": m["source_hash"],
                    "completed_mtime": path.stat().st_mtime,
                }
            )
    if new:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as stream:
            for row in sorted(new, key=lambda r: r["completed_mtime"]):
                stream.write(json.dumps(row) + "\n")
    median = float(np.median(wall)) if wall else float("nan")
    remaining = len(ids) - done
    print(
        json.dumps(
            {
                "source_hash": source[:12],
                "done": done,
                "total": len(ids),
                "remaining": remaining,
                "incomplete_or_wrong_source": bad,
                "median_wall_minutes": median,
                "projected_remaining_hours": remaining * median / 60 if wall else None,
            }
        )
    )
    if args.check and (remaining or bad):
        sys.exit(1)


if __name__ == "__main__":
    main()
