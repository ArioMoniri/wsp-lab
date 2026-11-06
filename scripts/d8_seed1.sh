#!/usr/bin/env bash
# D8 step 4: seed-1 repeat panel (10 controls + base, config eval_fast_seed1, --repeat seed1, batch 32, device H0)
# from the clean detached worktree at 0a8e708 (source_hash ec88cfd95d7d, the hash recorded by the original run,
# run p2-batch/20261105-050543-aad6c310, in ~/agent-logs/p4-after-chainpred2-20261105-041534.log).
set -euo pipefail
W=/home/ubuntu/projects/wsp-lab/.worktrees/regen-0a8e708
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
cd "$W"
export PYTHONPATH="$W/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
[ "$(git rev-parse --short HEAD)" = 0a8e708 ] && [ -z "$(git status --porcelain)" ] || { echo "worktree not clean" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
$PY -c "import wsp; print('wsp from', wsp.__file__)"
date -u +"start %FT%TZ"
$PY -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --config configs/eval_fast_seed1.yaml --repeat seed1 --ids $(cat data/runs/d3_shards/controls.txt)
date -u +"end %FT%TZ"
