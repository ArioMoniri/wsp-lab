#!/usr/bin/env bash
# Runs ON the Lambda instance (no API key, no HF token there; the model is public). ENGINEERING ONLY.
# Bootstraps a Python 3.12 venv pinned to the AWS .venv freeze (constraints.txt), downloads
# Qwen/Qwen3.8-27B at the pinned revision, tries to build causal-conv1d for this GPU, then runs
# scripts/wme_27b_calib.py once. Progress: logs/*.log; final state: STATUS (DONE or FAILED:<step>).
#   nohup bash scripts/wme_27b_remote.sh <code_sha> > logs/remote.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
SHA="${1:-unknown}"
REV=37561816e66b5b9153b1b64c10968c18c1e6f4a0
mkdir -p logs out
t0=$(date +%s)
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { echo "FAILED:$1" > STATUS; stamp "FAILED at $1"; exit 1; }
echo RUNNING > STATUS
stamp "host: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
stamp "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo none)"

# 1. uv + venv
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --python 3.12 .venv || fail venv
PY=.venv/bin/python
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
IDX=https://download.pytorch.org/whl/cu132
[ "$drv" -lt 580 ] && { IDX=https://download.pytorch.org/whl/cu128; stamp "driver $drv < 580: torch from cu128 (deviation)"; }
stamp "installing torch 2.15.0 from $IDX"
uv pip install -q --python $PY "torch==2.15.0" --index-url "$IDX" > logs/pip_torch.log 2>&1 || fail torch
grep -v -E '^torch==|^torchvision==|^nvidia-|^triton==' constraints.txt > constraints.nontorch.txt
uv pip install -q --python $PY -c constraints.nontorch.txt transformers accelerate safetensors numpy \
  huggingface-hub hf-xet flash-linear-attention fla-core einops kernels jinja2 > logs/pip.log 2>&1 || fail pip
stamp "venv ready ($(( $(date +%s) - t0 )) s)"

# 2. model download and causal-conv1d build in parallel
( .venv/bin/hf download Qwen/Qwen3.8-27B --revision "$REV" > logs/download.log 2>&1 && echo ok > logs/download.ok ) &
dl=$!
( export TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=$(nproc) CAUSAL_CONV1D_FORCE_BUILD=TRUE
  uv pip install -q --python $PY setuptools wheel ninja packaging > logs/conv.log 2>&1
  timeout 900 uv pip install --python $PY --no-build-isolation "causal-conv1d==1.7.0" >> logs/conv.log 2>&1 \
    && echo ok > logs/conv.ok ) &
cv=$!
wait $dl; [ -f logs/download.ok ] || fail download
stamp "model downloaded ($(( $(date +%s) - t0 )) s)"
wait $cv; [ -f logs/conv.ok ] && stamp "causal-conv1d built" || stamp "causal-conv1d build failed: torch conv fallback (deviation)"
setup_s=$(( $(date +%s) - t0 ))
echo "$setup_s" > out/setup_seconds
$PY -c "import torch,transformers;print('torch',torch.__version__,torch.version.cuda,'transformers',transformers.__version__)"
$PY -m pip --version >/dev/null 2>&1 || true
uv pip freeze --python $PY > out/remote_freeze.txt 2>/dev/null

# 3. calibration (one process, one load)
stamp "calibration start"
WME_CODE_SHA="$SHA" $PY scripts/wme_27b_calib.py --out out --work /tmp/wme27-work > logs/calib.log 2>&1
rc=$?
stamp "calibration exit $rc ($(( $(date +%s) - t0 )) s total)"
[ $rc -eq 0 ] && echo DONE > STATUS || echo "FAILED:calib:$rc" > STATUS
