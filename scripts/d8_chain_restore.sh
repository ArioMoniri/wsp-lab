#!/usr/bin/env bash
# D8 step 7: replay every chain with lost snapshots (restore mode) into a staging directory.
# Stored steps are verified on the way; lost steps are written to $OUT/<id>/delta.safetensors + regen.json.
# Nothing is written into the canonical zoo here; scripts/d8_chain_install.py moves verified entries in.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
DATA=/home/ubuntu/wsp-data
OUT=${OUT:-$DATA/runs/d8/chain_replay/restore}
SHARD=${SHARD:-}   # i/n to run one shard of the sorted chain list (several shards share $OUT; reports are per shard)
REPORT=$OUT/report${SHARD:+-$(echo "$SHARD" | tr / -)}.json
cd "$CODE"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
git rev-parse HEAD; git status --porcelain | head -3
date -u +"start %FT%TZ"
$PY scripts/d8_chain_replay.py --zoo $DATA/zoo/qwen3.5-0.8b --sidecar $DATA/zoo/qwen3.5-0.8b/artefact_status.jsonl \
  --config configs/zoo_qwen3.5-0.8b.yaml --out "$OUT" --mode restore ${CHAINS:+--chains "$CHAINS"} \
  --report "$REPORT" ${SHARD:+--shard "$SHARD"} "$@"
date -u +"end %FT%TZ"
