#!/usr/bin/env bash
# Runs ON the Lambda instance (no API key and no HF token there; the model is public). ENGINEERING ONLY.
# P-H100b (RULING-wme-v2b A9): the P-H100 platform stack and pinned Qwen/Qwen3.5-9B, then generation timing at larger
# batches on exposed wme-p2 construction prompts (scripts/wme_ph100.py genbatch). Final state: STATUS.
#   nohup bash scripts/wme_ph100b_remote.sh <code_sha> > logs/remote.log &
set -uo pipefail
cd "$(dirname "$0")/.."
SHA="${1:-unknown}"
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
mkdir -p logs out
t0=$(date +%s)
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { echo "FAILED:$1" > STATUS; stamp "FAILED at $1"; exit 1; }
echo RUNNING > STATUS
stamp "host: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
G=$(nvidia-smi -L | wc -l)
stamp "gpus: $G"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --clear --python 3.12 .venv || fail venv
PY=.venv/bin/python
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
# Lambda platform stack (fixed for every WM-E 9B Lambda run): torch 2.11.0+cu128 runs on the 570 and 580 driver
# branches alike (torch 2.15 ships only cu132/cu134 wheels, which need driver >= 580). Deviation from the AWS stack.
TORCH=2.11.0
IDX=https://download.pytorch.org/whl/cu128
stamp "driver $drv: installing torch $TORCH from $IDX"
uv pip install -q --python $PY "torch==$TORCH" --index-url "$IDX" > logs/pip_torch.log 2>&1 || fail torch
grep -v -E '^torch==|^torchvision==|^nvidia-|^triton==|^setuptools==|^cuda-|^nvidia' constraints.txt > constraints.nontorch.txt
echo "torch==$TORCH" >> constraints.nontorch.txt   # no later step may move torch off the platform pin
uv pip install -q --python $PY -c constraints.nontorch.txt transformers accelerate safetensors numpy \
  huggingface-hub hf-xet flash-linear-attention fla-core einops kernels jinja2 scipy > logs/pip.log 2>&1 || fail pip
# fla refuses the Hopper backward (chunk_bwd_dqkwg) under torch 2.11's Triton 3.6; tilelang is its supported path
uv pip install -q --python $PY -c constraints.nontorch.txt tilelang > logs/pip_tilelang.log 2>&1 || fail tilelang
stamp "venv ready ($(( $(date +%s) - t0 )) s)"
HF_HUB_OFFLINE=0 .venv/bin/hf download Qwen/Qwen3.5-9B --revision "$REV" > logs/download.log 2>&1 || fail download
stamp "model downloaded ($(( $(date +%s) - t0 )) s)"
echo "$(( $(date +%s) - t0 ))" > out/setup_seconds
uv pip freeze --python $PY > out/remote_freeze.txt 2>/dev/null
$PY -c "import torch,transformers;print('torch',torch.__version__,torch.version.cuda,'transformers',transformers.__version__)" > out/versions.txt
export PYTHONPATH=src
stamp "genbatch"
CUDA_VISIBLE_DEVICES=0 $PY scripts/wme_ph100.py genbatch --out out > logs/genbatch.log 2>&1 || fail genbatch
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > out/gpus.txt
stamp "done ($(( $(date +%s) - t0 )) s total)"
echo DONE > STATUS
