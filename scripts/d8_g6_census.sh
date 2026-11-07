#!/usr/bin/env bash
# G6 chain census (SYNTHESIS 7 Nov §3; repriced 7 Nov: 635 stored chain entries lack a TF-v1 record, not 296) plus the two
# G1 repeats that the D8 audit could not run (the chain predecessors b7edbb9f8b339f52 and 86ea1fe0ec777cf8, now regenerated;
# their primaries were scored by scripts/d8_tf_yh.sh). Runs from the clean frozen-source worktree (dd2a609, a224779bf873)
# through the staged zoo of scripts/d8_tf_yh.py prepare (aliases materialised by symlink), device H0, exclusive GPU.
set -euo pipefail
W=/home/ubuntu/projects/wsp-lab/.worktrees/regen-dd2a609
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
D=/home/ubuntu/wsp-data
cd "$W"
export PYTHONPATH="$W/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
[ -z "$(git status --porcelain)" ] || { echo "worktree not clean" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
git rev-parse HEAD
date -u +"start %FT%TZ"
echo "=== G1 repeat (two regenerated predecessors) $(date -u +%FT%TZ)"
$PY -m wsp.eval.batch --zoo $D/runs/d8/stage_zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml --repeat g1b --ids b7edbb9f8b339f52 86ea1fe0ec777cf8  # repeat id g1 exists for the base from the 6 Nov audit
echo "=== G6 census $(date -u +%FT%TZ) n=$(wc -l < $D/runs/d8/g6_ids.txt)"
$PY -m wsp.eval.batch --zoo $D/runs/d8/stage_zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml --ids $(cat $D/runs/d8/g6_ids.txt)
date -u +"end %FT%TZ"
