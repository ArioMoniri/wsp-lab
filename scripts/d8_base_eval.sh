#!/usr/bin/env bash
# D8 step 3: re-evaluate the H0 base at one historical source tree and register it under base/devices/H0/<run>/.
# usage: d8_base_eval.sh <commit> <expected source_hash prefix>
# Runs from the clean detached worktree .worktrees/regen-<commit> (data -> /home/ubuntu/wsp-data), never from main.
set -euo pipefail
sha="$1"; want="$2"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
W=/home/ubuntu/projects/wsp-lab/.worktrees/regen-$sha
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
cd "$W"
export PYTHONPATH="$W/src" TOKENIZERS_PARALLELISM=false
unset HF_HOME
E=data/evals/qwen3.5-0.8b
[ "$(git rev-parse --short HEAD)" = "$sha" ] && [ -z "$(git status --porcelain)" ] || { echo "worktree $W not clean at $sha" >&2; exit 1; }
[ ! -e "$E/base/eval.json" ] && [ ! -e "$E/index.jsonl" ] || { echo "$E/base/eval.json or index.jsonl exists; refusing" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
$PY -c "import wsp; print('wsp from', wsp.__file__)"
date -u +"start %FT%TZ"
$PY -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --ids none
date -u +"end %FT%TZ"
$PY "$CODE/scripts/d8_register_base.py" --evals "$E" --commit "$sha" --source-hash-prefix "$want"
