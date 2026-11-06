#!/usr/bin/env bash
# D8 step 2: re-extract v4 features for every manifest entry that has delta/lora bytes (stored or alias).
# A staging zoo (symlinks, manifest.jsonl untouched) lets the 63 byte-identical aliases be extracted under their own
# id and edit.json without writing anything into the canonical zoo.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
DATA=/home/ubuntu/wsp-data
Z=$DATA/zoo/qwen3.5-0.8b
ST=$DATA/runs/d8/staging_zoo
OUT=$DATA/features/qwen3.5-0.8b
[ ! -e "$OUT/manifest.json" ] || { echo "$OUT already has a manifest; refusing" >&2; exit 1; }
$PY - "$Z" "$ST" <<'PYE'
import json, sys, os
from pathlib import Path
Z, ST = Path(sys.argv[1]), Path(sys.argv[2])
ST.mkdir(parents=True, exist_ok=True)
for line in (Z / "artefact_status.jsonl").read_text().splitlines():
    r = json.loads(line)
    d = ST / r["id"]
    if r["delta_present"]:
        if not d.is_symlink(): d.symlink_to(Z / r["id"])
    elif r["alias_of"]:
        d.mkdir(exist_ok=True)
        for name, src in (("edit.json", Z / r["id"] / "edit.json"),):
            if not (d / name).is_symlink(): (d / name).symlink_to(src)
        t = Z / r["alias_of"]
        if (t / "delta.safetensors").exists():
            if not (d / "delta.safetensors").is_symlink(): (d / "delta.safetensors").symlink_to(t / "delta.safetensors")
        else:
            if not (d / "lora").is_symlink(): (d / "lora").symlink_to(t / "lora")
if not (ST / "manifest.jsonl").is_symlink():
    (ST / "manifest.jsonl").symlink_to(Z / "manifest.jsonl")
print("staging entries:", sum(1 for p in ST.iterdir() if p.name != "manifest.jsonl"))
PYE
cd "$CODE"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HF_HOME
git rev-parse HEAD
date -u +"start %FT%TZ"
$PY -m wsp.features.extract --zoo "$ST" --out "$OUT" --workers 4 --threads 1
date -u +"end %FT%TZ"
