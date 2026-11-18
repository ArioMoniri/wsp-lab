#!/usr/bin/env bash
# Runs ON a Lambda instance for WM-E pilot 3 (no API key and no HF token there; the model is public). EXPLORATORY.
#   nohup bash scripts/wme_p3_remote.sh <code_sha> s|h > logs/remote.log &
# Stage s: the P-H100 platform stack (torch 2.11 cu128, transformers from the AWS freeze, fla + tilelang) runs
# scripts/wme_pilot3_s.py. Stage h: a separate venv with vllm==0.34.0 (its torch build; driver >= 580 required)
# runs scripts/wme_pilot3_h.py. Versions and the driver go to out/versions.txt. Progress: logs/, out/**; final
# state: STATUS (DONE, STOPPED_*, or FAILED:<step>). A background loop touches logs/alive while the job runs.
set -uo pipefail
cd "$(dirname "$0")/.."
SHA="${1:-unknown}"; STAGE="${2:?stage s|h}"
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
mkdir -p logs out
t0=$(date +%s)
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { echo "FAILED:$1" > STATUS; stamp "FAILED at $1"; exit 1; }
echo RUNNING > STATUS
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
stamp "host gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1); driver $drv; code $SHA"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --clear --python 3.12 .venv || fail venv
PY=.venv/bin/python
grep -v -E '^torch==|^torchvision==|^nvidia-|^triton==|^setuptools==|^cuda-|^nvidia|^vllm|^transformers==' constraints.txt > constraints.nontorch.txt
if [ "$STAGE" = s ]; then
  uv pip install -q --python $PY "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 > logs/pip_torch.log 2>&1 || fail torch
  echo "torch==2.11.0" >> constraints.nontorch.txt
  uv pip install -q --python $PY -c constraints.nontorch.txt "transformers==5.22.1" accelerate safetensors numpy scipy \
    huggingface-hub hf-xet einops jinja2 > logs/pip.log 2>&1 || fail pip
  # fused GatedDeltaNet kernels only speed up the R2 passes (jets use the reference kernels); optional on aarch64
  uv pip install -q --python $PY -c constraints.nontorch.txt flash-linear-attention fla-core kernels > logs/pip_fla.log 2>&1 \
    || stamp "fla not installed (reference kernels everywhere)"
  uv pip install -q --python $PY -c constraints.nontorch.txt tilelang > logs/pip_tilelang.log 2>&1 || stamp "tilelang not installed"
  $PY -c "import torch; torch.zeros(1).cuda(); print('cuda ok', torch.__version__, torch.version.cuda)" >> logs/pip.log 2>&1 || fail "cuda-runtime"
else
  uv pip install -q --python $PY "vllm==0.34.0" > logs/pip_vllm.log 2>&1 || fail vllm
  if [ "${drv%%.*}" -lt 580 ]; then
    # the cu130 build needs driver >= 580; H100 supports CUDA forward compatibility (cuda-compat user-mode driver)
    stamp "driver $drv < 580: installing cuda-compat-13-0"
    ( sudo apt-get update -qq && sudo apt-get install -y -qq cuda-compat-13-0 ) > logs/compat.log 2>&1 || {
      deb=$(curl -s https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ | grep -o 'cuda-compat-13-0_[^"]*_amd64.deb' | sort -V | tail -1)
      curl -sLO "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$deb" && sudo dpkg -i "$deb" >> logs/compat.log 2>&1; }
    CD=$(ls -d /usr/local/cuda-13*/compat 2>/dev/null | head -1)
    [ -n "$CD" ] || fail "cuda-compat"
    export LD_LIBRARY_PATH="$CD:${LD_LIBRARY_PATH:-}"
    echo "cuda-compat $CD" >> logs/compat.log
  fi
  $PY -c "import torch; torch.zeros(1).cuda(); print('cuda ok', torch.version.cuda)" >> logs/pip.log 2>&1 || fail "cuda-runtime"
  uv pip install -q --python $PY scipy hf-xet accelerate > logs/pip.log 2>&1 || fail pip
fi
stamp "venv ready ($(( $(date +%s) - t0 )) s)"
HF_HUB_OFFLINE=0 .venv/bin/hf download Qwen/Qwen3.5-9B --revision "$REV" > logs/download.log 2>&1 || fail download
stamp "model downloaded ($(( $(date +%s) - t0 )) s)"
echo "$(( $(date +%s) - t0 ))" > out/setup_seconds
uv pip freeze --python $PY > out/remote_freeze.txt 2>/dev/null
{ echo "driver $drv"; echo "arch $(uname -m)"; echo "LD_LIBRARY_PATH ${LD_LIBRARY_PATH:-}"; $PY -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"; \
  $PY -c "import vllm;print('vllm',vllm.__version__)" 2>/dev/null; } > out/versions.txt
export PYTHONPATH=src
SCRIPT=scripts/wme_pilot3_s.py; [ "$STAGE" = h ] && SCRIPT=scripts/wme_pilot3_h.py
$PY $SCRIPT --out out/$STAGE > logs/$STAGE.log 2>&1 &
pid=$!
bash scripts/wme_p3_watch.sh $pid $STAGE   # progress-based liveness; writes the final STATUS (amendment 2(d))
stamp "finished ($(( $(date +%s) - t0 )) s total)"
