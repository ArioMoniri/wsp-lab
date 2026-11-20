#!/usr/bin/env bash
# Study M Gate 0 run (local GPU, niced, <=16 GB, <=3 GPU-h): scripts/wmm_gate0.sh <out-dir> [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."
WSP_VENV=wmm . scripts/env.sh
out="$1"; shift
mkdir -p "$out"
exec nice -n 10 python scripts/wmm_gate0.py --out "$out" "$@"
