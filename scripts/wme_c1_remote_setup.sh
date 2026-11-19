#!/usr/bin/env bash
# Runs ON the wme-c1 Lambda instance (no API key, no HF token: the model is public). Engineering setup only:
#   bash scripts/wme_c1_remote_setup.sh <code_sha>   (from the unpacked bundle root)
# Own uv venv from the pinned engine lock (scripts/wme_c1_remote_lock.txt: the stage-W2 freeze, vllm 0.34.0, torch
# 2.14.0+cu130, transformers 5.22.2), cuda-compat-13-0 when the driver is < 580 (as stage W/W2), the pinned model
# snapshot, then the versions record. Writes SETUP_OK or SETUP_FAILED:<step>. Progress lines go to progress.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."
SHA="${1:-unknown}"
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
mkdir -p logs w
t0=$(date +%s)
prog() { echo "{\"t\": $(date +%s), \"event\": \"setup:$1\"}" >> w/progress.jsonl; }
fail() { echo "SETUP_FAILED:$1" > SETUP_STATUS; prog "failed $1"; exit 1; }
echo RUNNING > SETUP_STATUS
prog start
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --clear --python 3.12 .venv || fail venv
PY=.venv/bin/python
uv pip install -q --python $PY -r scripts/wme_c1_remote_lock.txt > logs/pip.log 2>&1 || fail pip
prog "lock installed"
if [ "${drv%%.*}" -lt 580 ]; then
  ( sudo apt-get update -qq && sudo apt-get install -y -qq cuda-compat-13-0 ) > logs/compat.log 2>&1 || {
    REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(. /etc/os-release; echo "${VERSION_ID//./}")/x86_64"
    deb=$(curl -s "$REPO/" | grep -oE 'cuda-compat-13-0_[0-9][0-9A-Za-z.+~-]*_amd64\.deb' | sort -uV | tail -1)
    [ -n "$deb" ] && curl -sLO "$REPO/$deb" && sudo dpkg -i "$deb" >> logs/compat.log 2>&1; }
  CD=$(ls -d /usr/local/cuda-13*/compat 2>/dev/null | head -1)
  [ -n "$CD" ] || fail "cuda-compat"
  echo "$CD" > CUDA_COMPAT
fi
[ -f CUDA_COMPAT ] && export LD_LIBRARY_PATH="$(cat CUDA_COMPAT):${LD_LIBRARY_PATH:-}"
$PY -c "import torch; torch.zeros(1).cuda(); print('cuda ok', torch.version.cuda)" >> logs/pip.log 2>&1 || fail "cuda-runtime"
prog "cuda ok"
HF_HUB_OFFLINE=0 .venv/bin/hf download Qwen/Qwen3.5-9B --revision "$REV" > logs/download.log 2>&1 || fail download
prog "model downloaded"
uv pip freeze --python $PY > w/remote_freeze.txt 2>/dev/null
cmp -s <(sort w/remote_freeze.txt) <(sort scripts/wme_c1_remote_lock.txt) && echo "lock reproduced" > w/lock_check.txt || diff <(sort scripts/wme_c1_remote_lock.txt) <(sort w/remote_freeze.txt) > w/lock_check.txt
{ echo "driver $drv"; echo "code $SHA"; $PY -c "import torch,transformers,vllm;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'vllm',vllm.__version__)"; } > w/versions.txt
echo "$(( $(date +%s) - t0 ))" > w/setup_seconds
echo SETUP_OK > SETUP_STATUS
prog done
