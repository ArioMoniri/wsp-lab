#!/usr/bin/env bash
# Runs ON one stage-W2 Lambda instance (no API key, no HF token; the model is public). EXPLORATORY.
#   nohup bash scripts/wme_w2_remote.sh <code_sha> <slot a|b|c> <queue deadline, epoch s> > logs/remote.log &
# Copy of wme_w_remote.sh: own uv venv with vllm==0.34.0 (its torch build; driver >= 580 or cuda-compat), model
# download, then wme_stagew2.py train and eval for the slot. The evaluation queue stops at the deadline the AWS
# driver passes (instance launch + 3.10 h, RULING-wme-stagew2 §6). Every completed unit appends to
# out/w2/progress.jsonl (the only file the AWS heartbeat reads). The final progress line is written BEFORE the
# checksums, and SHA256SUMS is the last file written under out/w2 (fixes the stage-W ordering slip). Final state in
# STATUS (outside out/w2): DONE or FAILED:<step>.
set -uo pipefail
cd "$(dirname "$0")/.."
SHA="${1:-unknown}"; SLOT="${2:?slot}"; DEADLINE="${3:?deadline epoch}"
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
mkdir -p logs out/w2
t0=$(date +%s)
echo "$DEADLINE" > out/w2/deadline
echo "$SLOT" > out/w2/slot
prog() { echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"phase\": \"setup\", \"event\": \"$1\"}" >> out/w2/progress.jsonl; }
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { echo "FAILED:$1" > STATUS; stamp "FAILED at $1"; prog "failed $1"; exit 1; }
echo RUNNING > STATUS
prog start
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
stamp "host gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1); driver $drv; code $SHA; slot $SLOT"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --clear --python 3.12 .venv || fail venv
PY=.venv/bin/python
uv pip install -q --python $PY "vllm==0.34.0" > logs/pip_vllm.log 2>&1 || fail vllm
prog "vllm installed"
if [ "${drv%%.*}" -lt 580 ]; then
  stamp "driver $drv < 580: installing cuda-compat-13-0"
  ( sudo apt-get update -qq && sudo apt-get install -y -qq cuda-compat-13-0 ) > logs/compat.log 2>&1 || {
    deb=$(curl -s https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ | grep -o 'cuda-compat-13-0_[^"]*_amd64.deb' | sort -V | tail -1)
    curl -sLO "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$deb" && sudo dpkg -i "$deb" >> logs/compat.log 2>&1; }
  CD=$(ls -d /usr/local/cuda-13*/compat 2>/dev/null | head -1)
  [ -n "$CD" ] || fail "cuda-compat"
  export LD_LIBRARY_PATH="$CD:${LD_LIBRARY_PATH:-}"
fi
$PY -c "import torch; torch.zeros(1).cuda(); print('cuda ok', torch.version.cuda)" >> logs/pip.log 2>&1 || fail "cuda-runtime"
uv pip install -q --python $PY scipy hf-xet accelerate >> logs/pip.log 2>&1 || fail pip
prog "venv ready"
stamp "venv ready ($(( $(date +%s) - t0 )) s)"
HF_HUB_OFFLINE=0 .venv/bin/hf download Qwen/Qwen3.5-9B --revision "$REV" > logs/download.log 2>&1 || fail download
prog "model downloaded"
stamp "model downloaded ($(( $(date +%s) - t0 )) s)"
echo "$(( $(date +%s) - t0 ))" > out/w2/setup_seconds
uv pip freeze --python $PY > out/w2/remote_freeze.txt 2>/dev/null
{ echo "driver $drv"; echo "arch $(uname -m)"; echo "LD_LIBRARY_PATH ${LD_LIBRARY_PATH:-}"; \
  $PY -c "import torch,transformers,vllm;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'vllm',vllm.__version__)"; } > out/w2/versions.txt
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY scripts/wme_stagew2.py train --out out/w2 --slot "$SLOT" > logs/train.log 2>&1 || fail train
stamp "train done ($(( $(date +%s) - t0 )) s)"
$PY scripts/wme_stagew2.py eval --out out/w2 --slot "$SLOT" > logs/eval.log 2>&1 || fail eval
stamp "eval done ($(( $(date +%s) - t0 )) s)"
prog done
( cd out/w2 && find . -type f ! -name 'SHA256SUMS' ! -name '*.tmp' -print0 | sort -z | xargs -0 sha256sum > ../SHA256SUMS.w2 \
  && mv ../SHA256SUMS.w2 SHA256SUMS ) || fail checksums
echo DONE > STATUS
