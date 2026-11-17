#!/usr/bin/env bash
# Runs ON a Lambda CPU-rich instance (no API key there). RULING-wme-v2b simulation jobs (scripts/wme_v2b_sim.py),
# CPU only, one single-threaded job per worker. The venv is built once and reused by later calls on the same
# instance. Progress: out/<dir>/progress.json and logs; final state: STATUS (DONE:<tag> or FAILED:<step>).
#   nohup bash scripts/wme_v2b_remote.sh <sha> <tag> <cmd> <summary-cmd> <out-dir> [args...] &
set -uo pipefail
cd "$(dirname "$0")/.."
SHA="${1:-unknown}"; TAG="$2"; CMD="$3"; SUM="$4"; OUT="$5"; shift 5
mkdir -p logs out
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { echo "FAILED:$1" > STATUS; stamp "FAILED at $1"; exit 1; }
echo "RUNNING:$TAG" > STATUS
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
if [ ! -f .venv/READY ]; then
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
  uv venv -q --clear --python 3.12 .venv || fail venv
  uv pip install -q --python $PY "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu > logs/pip_torch.log 2>&1 || fail torch
  uv pip install -q --python $PY numpy scipy > logs/pip.log 2>&1 || fail pip
  uv pip freeze --python $PY > out/remote_freeze.txt
  touch .venv/READY
fi
P=$(( $(nproc) - 2 ))
stamp "$TAG: $CMD, $P workers, out $OUT, args $* (code $SHA)"
PYTHONPATH=src OMP_NUM_THREADS=1 $PY scripts/wme_v2b_sim.py "$CMD" --out "out/$OUT" --procs $P "$@" > "logs/$TAG.log" 2>&1 || fail "$TAG"
PYTHONPATH=src $PY scripts/wme_v2b_sim.py "$SUM" --out "out/$OUT" > "logs/$TAG-summary.log" 2>&1 || fail "$TAG-summary"
stamp "done $TAG"
echo "DONE:$TAG" > STATUS
