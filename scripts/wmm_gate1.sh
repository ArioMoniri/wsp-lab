#!/usr/bin/env bash
# Study M Gate 1 (local GPU, niced, <=16 GB, <=12 GPU-h): scripts/wmm_gate1.sh <out-dir>
set -euo pipefail
cd "$(dirname "$0")/.."
WSP_VENV=wmm . scripts/env.sh
out="$1"; shift
mkdir -p "$out"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exec nice -n 10 python scripts/wmm_gate1.py --out "$out" "$@"
