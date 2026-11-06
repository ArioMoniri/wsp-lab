#!/usr/bin/env python3
"""D8 step 6a (data layout): re-register the two Lambda base records under base/devices/<H>/<run_id>/.

The hub restore brings back ``base-lambda-<ip>/{eval,manifest,records,summary}`` as uploaded by the Lambda hosts (no device
id, no runtime fields). The original layout was produced by ``scripts/backfill_eval_metadata.py --apply``: the record gets its
device id from the private device map (default_map: lambda-209-20-158-168 -> H1, lambda-129-153-20-232 -> H2), the runtime
fields from configs/eval_runtime_backfill.json and item hashes from its records.jsonl, is archived as eval.json.orig,
rewritten in place and copied to base/devices/<H>/<run_id>/eval.json with records.jsonl and manifest.json linked.
This script performs exactly that branch of the backfill for the two Lambda base directories only; no restored zoo
eval.json is touched and no score changes.

    PYTHONPATH=<worktree>/src:<worktree>/scripts d8_backfill_lambda_bases.py [--apply]
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from backfill_eval_metadata import archive, backfill, default_map, runtime_metadata, write

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data/evals/qwen3.5-0.8b"

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
devices = default_map()
local = runtime_metadata(
    SimpleNamespace(
        model=SimpleNamespace(
            dtype=torch.bfloat16, config=SimpleNamespace(_attn_implementation="sdpa")
        )
    )
)
remote = json.loads((ROOT / "configs/eval_runtime_backfill.json").read_text())
now = datetime.now(UTC).isoformat()
for d in sorted(EV.glob("base-lambda-*")):
    path = d / "eval.json"
    old = json.loads(path.read_text())
    host = d.name.removeprefix("base-")
    new = backfill(path, old, devices, local, remote, now, host, {})
    dest = EV / "base/devices" / new["eval_device_id"] / new["run_id"]
    print(
        json.dumps(
            {
                "dir": d.name,
                "device": new["eval_device_id"],
                "run_id": new["run_id"],
                "metadata_backfill": new.get("metadata_backfill"),
                "missing": new.get("metadata_backfill_missing"),
                "dest": str(dest.relative_to(EV)),
                "applied": a.apply,
            }
        )
    )
    if a.apply:
        archive(path)
        write(path, new)
        write(dest / "eval.json.orig", old)
        write(dest / "eval.json", new)
        for name in ("records.jsonl", "manifest.json"):
            if (d / name).exists() and not (dest / name).exists():
                (dest / name).symlink_to((d / name).resolve())
