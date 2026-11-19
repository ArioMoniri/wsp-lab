#!/usr/bin/env bash
# Runs ON the instance: the resident wme-c1 worker (restarted by the controller after a crash; the request inbox and
# the worker state survive). Never constructs a guard; reads only what the controller uploads plus the bundle.
cd "$(dirname "$0")/.."
[ -f CUDA_COMPAT ] && export LD_LIBRARY_PATH="$(cat CUDA_COMPAT):${LD_LIBRARY_PATH:-}"
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python -m wsp.wm.c1.serve --root w "$@" >> logs/worker.log 2>&1
