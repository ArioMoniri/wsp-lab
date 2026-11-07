#!/usr/bin/env bash
# D8 step 7 pilot: replay fully-stored chains (verify mode) to measure reproduction of every step family.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
DATA=/home/ubuntu/wsp-data
OUT=$DATA/runs/d8/chain_replay/pilot
CHAINS=${CHAINS:-15f9b68343ea7718,4e0db6f0b0f455f0,1e03b2f23e8cba48}
cd "$CODE"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
git rev-parse HEAD; git status --porcelain | head -3
date -u +"start %FT%TZ"
$PY scripts/d8_chain_replay.py --zoo $DATA/zoo/qwen3.5-0.8b --sidecar $DATA/zoo/qwen3.5-0.8b/artefact_status.jsonl \
  --config configs/zoo_qwen3.5-0.8b.yaml --out "$OUT" --mode verify --chains "$CHAINS" \
  --report "$OUT/report.json"
date -u +"end %FT%TZ"
