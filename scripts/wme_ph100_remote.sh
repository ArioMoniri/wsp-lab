#!/usr/bin/env bash
# Runs ON the Lambda instance (no API key and no HF token there; the model is public). ENGINEERING ONLY.
# Bootstraps a Python 3.12 venv pinned to the AWS .venv freeze (constraints.txt), downloads Qwen/Qwen3.5-9B at the
# pinned revision, then runs P-H100 (scripts/wme_ph100.py): the k = 1 pass on GPU 0, the k = G pass on every GPU
# (G > 1 only), the timing extras on GPU 0 and the hash-only summary. Progress: logs/*.log and out/*/progress;
# final state: STATUS (DONE or FAILED:<step>).   nohup bash scripts/wme_ph100_remote.sh <code_sha> > logs/remote.log &
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
HF_HUB_OFFLINE=0 .venv/bin/hf download Qwen/Qwen3.5-9B --revision "$REV" > logs/download.log 2>&1 || fail download
stamp "model downloaded ($(( $(date +%s) - t0 )) s)"
echo "$(( $(date +%s) - t0 ))" > out/setup_seconds
uv pip freeze --python $PY > out/remote_freeze.txt 2>/dev/null
$PY -c "import torch,transformers;print('torch',torch.__version__,torch.version.cuda,'transformers',transformers.__version__)" > out/versions.txt
export PYTHONPATH=src
stamp "k=1 pass"
CUDA_VISIBLE_DEVICES=0 $PY scripts/wme_ph100.py worker --out out/k1 --worker g0 > logs/k1.log 2>&1 || fail k1
KK=""
if [ "$G" -gt 1 ]; then
  stamp "k=$G pass"
  pids=()
  for i in $(seq 0 $((G-1))); do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/wme_ph100.py worker --out out/kk --worker g$i > logs/kk_g$i.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail kk; done
  KK="--kk out/kk"
fi
stamp "extras"
CUDA_VISIBLE_DEVICES=0 $PY scripts/wme_ph100.py extras --out out --work /tmp/wme-ph100-work > logs/extras.log 2>&1 || stamp "extras failed (continuing to summary)"
stamp "summary"
$PY scripts/wme_ph100.py summarise --out out --k1 out/k1 $KK > logs/summary.log 2>&1 || fail summary
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > out/gpus.txt
stamp "done ($(( $(date +%s) - t0 )) s total)"
echo DONE > STATUS
