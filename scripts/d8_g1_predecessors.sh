#!/usr/bin/env bash
# G1 for the two regenerated chain predecessors: their repeats (label g1b) were scored by scripts/d8_g6_census.sh; this scores
# the primaries once the census log shows its end marker (exclusive GPU), then runs the G1 report for label g1b.
set -euo pipefail
LOG=${1:?census log to wait on}
W=/home/ubuntu/projects/wsp-lab/.worktrees/regen-dd2a609
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
D=/home/ubuntu/wsp-data
until grep -q '^end 20' "$LOG"; do sleep 60; done
sleep 15
cd "$W"
export PYTHONPATH="$W/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
[ -z "$(git status --porcelain)" ] || { echo "worktree not clean" >&2; exit 1; }
date -u +"start %FT%TZ"
$PY -m wsp.eval.batch --zoo $D/runs/d8/stage_zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml --ids b7edbb9f8b339f52 86ea1fe0ec777cf8
cd /home/ubuntu/projects/wsp-lab
PYTHONPATH=/home/ubuntu/projects/wsp-lab/src TF_G1_LABEL=g1b $PY scripts/tf_v1_g1_report.py || true
date -u +"end %FT%TZ"
