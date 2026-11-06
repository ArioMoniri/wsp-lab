#!/usr/bin/env bash
# D8 step 5b: regenerate the H0 fast-tier item records (seed 0) of the ten D3 controls at source 6f274a83.
# Why: the surviving zoo/<id>/records.jsonl of the controls is the Lambda (H1) record (gsm8k .20 for 0a841d4082532e8b, the
# surviving H0 eval.json says .16), so the same-item G2 bridge has no H0 fast-tier records to compare against.
# Non-production run (external copy of configs/eval_fast.yaml: same bytes, same config_hash) so that nothing under
# zoo/<id> is written; output goes to data/runs/p2-batch/<run_id>/{base,<id>}/. Verified afterwards against the surviving
# H0 eval.json of each control (scalars) by scripts/d8_controls_compare.py.
set -euo pipefail
W=/home/ubuntu/projects/wsp-lab/.worktrees/regen-f63a08e
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
LOG=${1:?gpu-chain log to wait on}
until grep -q '^=== G2' "$LOG"; do sleep 15; done
while pgrep -f "[t]f_v1_g2_bridge" >/dev/null; do sleep 15; done
sleep 20
cd "$W"
export PYTHONPATH="$W/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
[ -z "$(git status --porcelain)" ] || { echo "worktree not clean" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
cp configs/eval_fast.yaml /tmp/d8_eval_fast_copy.yaml
$PY -c "import wsp; print('wsp from', wsp.__file__)"
date -u +"start %FT%TZ"
$PY -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --config /tmp/d8_eval_fast_copy.yaml --ids $(cat data/runs/d3_shards/controls.txt)
date -u +"end %FT%TZ"
