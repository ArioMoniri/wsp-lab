#!/usr/bin/env python3
"""Check the gates on a freshly written base record and move it under base/devices/H0/<run_id>/.

The batch runner of the historical commits writes the base to ``<evals>/base/``; the registered layout
(D2 addendum) is ``base/devices/<device>/<run_id>/``. Gates (Board A section 2): source_hash is the expected
one, ``dirty`` is false, status complete, nominal and actual batch 32.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--evals", type=Path, required=True)
p.add_argument("--commit", required=True)
p.add_argument("--source-hash-prefix", required=True)
p.add_argument("--device", default="H0")
a = p.parse_args()
base = a.evals / "base"
e = json.loads((base / "eval.json").read_text())
problems = []
if not e["source_hash"].startswith(a.source_hash_prefix):
    problems.append(f"source_hash {e['source_hash'][:12]} != {a.source_hash_prefix}")
if e.get("dirty"):
    problems.append("dirty")
if not e["git_sha"].startswith(a.commit):
    problems.append(f"git_sha {e['git_sha']}")
if e.get("eval_status") != "complete":
    problems.append(f"eval_status {e.get('eval_status')}")
if e.get("batch_size") != 32:
    problems.append(
        f"actual batch {e.get('batch_size')} (expected 32; an OOM retry makes the pair non-matching)"
    )
if problems:
    print("GATE FAILED:", problems, file=sys.stderr)
    sys.exit(2)
dest = base / "devices" / a.device / e["run_id"]
dest.mkdir(parents=True, exist_ok=False)
for f in sorted(base.iterdir()):
    if f.name != "devices":
        shutil.move(str(f), dest / f.name)
index = a.evals / "index.jsonl"
if index.exists():
    shutil.move(
        str(index), dest / "batch-index.jsonl"
    )  # derived; path fields point at base/ before registration
for lock in a.evals.glob("index.lock"):
    lock.unlink()
print(
    json.dumps(
        {
            "registered": str(dest),
            "run_id": e["run_id"],
            "source_hash": e["source_hash"][:12],
            "minutes": e.get("minutes"),
            "git_sha": e["git_sha"],
        }
    )
)
