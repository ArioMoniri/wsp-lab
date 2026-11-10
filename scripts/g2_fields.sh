#!/usr/bin/env bash
# G2 (PREREG_H1_TF A5.8.1-A5.8.3; council S31 §3 G2): base fields, chart projectors, probe cache and the
# 20-predecessor audit on the exclusive GPU, from a clean worktree. Runs ONLY after Board A's A5 freeze:
# PANELS_SHA256 must be the panels.json sha256 that A5.1 lists at the freeze (scripts/g2_fields.py refuses
# any other panels.json). Launch through agent-tmux, e.g.
#   PANELS_SHA256=<A5.1 hash> agent-tmux g2 scripts/g2_fields.sh [--resume] [--stages ...]
# Extra arguments go to scripts/g2_fields.py (caps default to 2.0 GPU-h fields+cache, 1.0 GPU-h audit).
# Exit code is the driver's: 0 complete, 2 HARD_STOP, 3 GATE_FAILURE, 4 incomplete, 5 refused.
set -euo pipefail
: "${PANELS_SHA256:?set PANELS_SHA256 to the panels.json sha256 listed in PREREG_H1_TF A5.1 at the freeze}"
W="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
OUT=/home/ubuntu/projects/wsp-lab/data/runs/g2
cd "$W"
export PYTHONPATH="$W/src" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HF_HOME
[ -z "$(git status --porcelain)" ] || { echo "worktree not clean" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
mkdir -p "$OUT/logs"
LOG="$OUT/logs/g2_$(date -u +%Y%m%dT%H%M%SZ).log"
{
  date -u +"start %FT%TZ"
  echo "git $(git rev-parse HEAD)"
  echo "panels.json $(sha256sum docs/prereg/a5_manifests/panels.json | cut -d' ' -f1) expected $PANELS_SHA256"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee -a "$LOG"
set +e
$PY scripts/g2_fields.py --out "$OUT" --panels-sha256 "$PANELS_SHA256" "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
date -u +"end %FT%TZ rc=$rc" | tee -a "$LOG"
exit "$rc"
