#!/usr/bin/env python3
"""D8 step 3b: add the runtime/item metadata to the regenerated H0 base records written by pre-metadata commits.

Same recipe as scripts/backfill_eval_metadata.py (the function `backfill`, applied to these records only; the zoo's
restored eval.json files are never touched). Scores are not changed; the original eval.json is kept as eval.json.orig.
Then prints, per source hash, whether the reference keys equal those of the restored H0 entry records of that hash.

    PYTHONPATH=<worktree>/src:<worktree>/scripts d8_backfill_bases.py [--apply]
"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from backfill_eval_metadata import backfill, default_map, runtime_metadata

from wsp.eval.repeats import REFERENCE_KEYS

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data/evals/qwen3.5-0.8b"
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"

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
now = datetime.now(UTC).isoformat()
restored = {}
for f in ZOO.glob("*/eval.json"):
    p = json.loads(f.read_text())
    if p.get("eval_device_id") == "H0":
        restored.setdefault(p["source_hash"][:8], p)
for d in sorted((EV / "base/devices/H0").glob("*")):
    path = d / "eval.json"
    old = json.loads(path.read_text())
    new = (
        backfill(path, old, devices, local, {}, now, host="local")
        if old.get("eval_device_id") is None
        else old
    )
    if a.apply and new != old:
        shutil.copy2(path, d / "eval.json.orig")
        path.write_text(json.dumps(new, indent=2) + "\n")
    ref = restored.get(new["source_hash"][:8])
    cmp = {k: new.get(k) == ref.get(k) for k in REFERENCE_KEYS} if ref else None
    print(
        json.dumps(
            {
                "run": d.name,
                "source": new["source_hash"][:8],
                "backfilled": new != old,
                "applied": a.apply and new != old,
                "metadata_backfill": new.get("metadata_backfill"),
                "keys_equal_to_restored_H0_entry": cmp and all(cmp.values()),
                "differing_keys": cmp and [k for k, v in cmp.items() if not v],
            }
        )
    )
