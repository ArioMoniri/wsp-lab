#!/usr/bin/env bash
# Runs ON a Lambda CPU-rich instance (no API key there). Full-refit validation of the selected simulation cell
# (scripts/wme_powersim_refit.py), CPU only, one single-threaded replicate per worker. Progress: out/progress and
# logs; final state: STATUS.   nohup bash scripts/wme_refit_remote.sh <sha> '<cell json>' <reps_null> <reps_alt> <gamma> &
set -uo pipefail
cd "$(dirname "$0")/.."
SHA="${1:-unknown}"; CELL="$2"; RN="$3"; RA="$4"; G="$5"
mkdir -p logs out
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { echo "FAILED:$1" > STATUS; stamp "FAILED at $1"; exit 1; }
echo RUNNING > STATUS
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --clear --python 3.12 .venv || fail venv
PY=.venv/bin/python
uv pip install -q --python $PY "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu > logs/pip_torch.log 2>&1 || fail torch
uv pip install -q --python $PY numpy scipy > logs/pip.log 2>&1 || fail pip
uv pip freeze --python $PY > out/remote_freeze.txt
P=$(( $(nproc) - 2 ))
stamp "refit: $P workers, cell $CELL, null $RN, alt $RA, gamma $G (code $SHA)"
PYTHONPATH=src OMP_NUM_THREADS=1 $PY scripts/wme_powersim_refit.py run --cell "$CELL" --reps-null $RN --reps-alt $RA \
  --gamma $G --procs $P --out out/refit > logs/refit.log 2>&1 || fail refit
PYTHONPATH=src $PY scripts/wme_powersim_refit.py summary --out out/refit > logs/summary.log 2>&1 || fail summary
stamp done
echo DONE > STATUS
