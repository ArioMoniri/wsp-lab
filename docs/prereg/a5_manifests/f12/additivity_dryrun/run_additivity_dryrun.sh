#!/bin/bash
# F12-I dry run (two constructions into the scratch dir; CPU only; no outcome opened)
set -o pipefail
cd /home/ubuntu/projects/wsp-lab || exit 1
source .venv/bin/activate
export OMP_NUM_THREADS=2
echo "[$(date -u +%H:%M:%S)] start $(git rev-parse --short HEAD)"
python scripts/f12_additivity_prepare.py dry-run --threads 2 2>&1
echo "DRYRUN_EXIT=$?"
