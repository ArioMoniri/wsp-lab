#!/usr/bin/env bash
# D8 step 5d: TF-v1 scoring of the 31 entries behind the 28 recoverable (y, h) rows, from the clean dd2a609 worktree
# (source a224779bf873), through the staged zoo of scripts/d8_tf_yh.py prepare (aliases materialised by symlink).
set -euo pipefail
LOG=${1:?log of the controls regeneration to wait on}
W=/home/ubuntu/projects/wsp-lab/.worktrees/regen-dd2a609
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
until grep -q '^end 20' "$LOG"; do sleep 20; done
sleep 15
cd "$W"
export PYTHONPATH="$W/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
[ -z "$(git status --porcelain)" ] || { echo "worktree not clean" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
date -u +"start %FT%TZ"
$PY -m wsp.eval.batch --zoo /home/ubuntu/wsp-data/runs/d8/stage_zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml --ids $(cat /home/ubuntu/wsp-data/runs/d8/tf_yh_ids.txt)
date -u +"end %FT%TZ"
