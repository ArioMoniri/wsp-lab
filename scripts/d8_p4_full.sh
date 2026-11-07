#!/usr/bin/env bash
# G0 fingerprint: protocol-v2 P4 suite on the full zoo after the chain replay (runs d3v2f-*), same run list and flags as
# scripts/d8_p4_suite.sh; input view from scripts/d8_p4_prepare.py (complete-ancestry rule, rebound e871923e base runs).
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
DATA=/home/ubuntu/wsp-data
VIEW=$DATA/runs/d8/p4_input_full
cd "$CODE"
export PYTHONPATH="$CODE/src"
git rev-parse HEAD
date -u +"start %FT%TZ"
$PY scripts/d8_p4_prepare.py --out "$VIEW" --rebind-e871923e
PREFIX=d3v2f P4_PARALLEL=${P4_PARALLEL:-3} scripts/d8_p4_suite.sh "$VIEW"
date -u +"end %FT%TZ"
