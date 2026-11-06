#!/usr/bin/env python3
"""D8 step 2 verification: feature content root, and a ten-entry re-extraction in a fresh process / out dir.

No pre-incident v4 feature file survives anywhere (searched the host: only /tmp toy artefacts), so the planned comparison
with surviving copies has nothing to compare with. The ten-entry check instead re-extracts ten entries into a separate
output directory with a different worker count and compares every output file byte for byte with the main run
(resume identity, 1 torch thread per entry, is the same): determinism of the extractor, not equality with the lost files.

    d8_feature_check.py root                 # content root + manifest hash of data/features/qwen3.5-0.8b
    d8_feature_check.py recheck [--workers 2]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DATA = Path("/home/ubuntu/wsp-data")
FEAT = DATA / "features/qwen3.5-0.8b"
RECHECK = DATA / "runs/d8/feat_recheck"
STAGE = DATA / "runs/d8/staging_zoo"
ROOT = Path(__file__).resolve().parents[1]
FILES = ("features.json", "features.safetensors", "pooled.safetensors")
TEN = [
    "2f61a4086395b719",  # lora_sft
    "d7f933d3b4e5d25e",  # lora_sft
    "8cd3182b6e55dcbf",  # random_dir
    "5c766ee2fa8bf915",  # abliteration
    "023b384a7799a787",  # task_vector
    "0027efdf2eae7405",  # chain (step >= 3)
    "34ee195e55227908",  # chain predecessor
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def root():
    status = [
        json.loads(x) for x in (ROOT / "docs/prereg/artefact_status.jsonl").read_text().splitlines()
    ]
    h = hashlib.sha256()
    n = 0
    for d in sorted(p for p in FEAT.iterdir() if p.is_dir()):
        h.update(d.name.encode())
        for f in FILES[1:]:
            h.update(sha(d / f).encode())
        n += 1
    m = FEAT / "manifest.json"
    meta = json.loads(m.read_text())
    out = {
        "entries": n,
        "content_root_sha256": h.hexdigest(),
        "manifest_json_sha256": sha(m),
        "config_hash": meta["config_hash"],
        "identity_hash": meta["identity_hash"],
        "feature_version": meta["config"]["feature_version"],
        "n_entries_manifest": meta["n_entries"],
        "failed": meta["failed"],
        "no_artifacts": meta["last_run"]["no_artifacts"],
        "alias_entries_extracted": sum(
            1 for r in status if r["alias_of"] and (FEAT / r["id"] / "features.json").exists()
        ),
        "torch": meta["resume_identity"]["torch"],
        "numpy": meta["resume_identity"]["numpy"],
    }
    print(json.dumps(out, indent=1))
    (DATA / "runs/d8/feature_root.json").write_text(json.dumps(out, indent=1))


def recheck(workers):
    status = {
        json.loads(x)["id"]: json.loads(x)
        for x in (ROOT / "docs/prereg/artefact_status.jsonl").read_text().splitlines()
    }
    ids = list(TEN)
    ids.append(next(i for i, r in status.items() if r["alias_of"]))  # one byte-identical alias
    ids.append(
        next(i for i, r in status.items() if r["family"] == "heretic_ara" and r["delta_present"])
    )
    ids.append(
        next(
            i
            for i, r in status.items()
            if r["family"] == "chain" and r["delta_present"] and i not in ids
        )
    )
    if RECHECK.exists():
        sys.exit(f"{RECHECK} exists")
    cmd = [
        "/home/ubuntu/projects/wsp-lab/.venv/bin/python",
        "-m",
        "wsp.features.extract",
        "--zoo",
        str(STAGE),
        "--out",
        str(RECHECK),
        "--workers",
        str(workers),
        "--threads",
        "1",
        "--ids",
        ",".join(ids),
    ]
    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
    }
    subprocess.run(cmd, check=True, env=env)
    res = {}
    for i in ids:
        res[i] = {f: sha(FEAT / i / f) == sha(RECHECK / i / f) for f in FILES}
    out = {
        "ids": ids,
        "byte_identical": res,
        "all_identical": all(all(v.values()) for v in res.values()),
        "workers_main": 4,
        "workers_recheck": workers,
    }
    print(json.dumps(out, indent=1))
    (DATA / "runs/d8/feature_recheck.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["root", "recheck"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    root() if a.mode == "root" else recheck(a.workers)
