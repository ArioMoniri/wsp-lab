#!/usr/bin/env bash
# Runs ON a Lambda instance. Usage: lambda_job_ara.sh <n_entries> <trials>
# Builds the heretic_ara zoo family for Qwen3.5-0.8B, pushes entries to the private HF zoo dataset, terminates itself.
set -euo pipefail
N="${1:-12}"; TRIALS="${2:-10}"
export PATH=$HOME/.local/bin:$PATH
sudo apt-get update -qq >/dev/null && sudo apt-get install -y -qq git rsync tmux python3.12-venv python3.12-dev build-essential >/dev/null || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
mkdir -p ~/projects && cd ~/projects
[ -d wsp-lab ] || git clone -q https://github.com/ArioMoniri/wsp-lab.git
cd wsp-lab && git pull -q
[ -d .venv ] || uv venv --python 3.12 .venv -q
uv pip install --python .venv/bin/python -e ".[dev]" -q
# torch must match the host driver: pick the CUDA wheel index from nvidia-smi's reported CUDA version
drv=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
case "$drv" in 12.8*|12.9*) idx=cu128;; 12.6*|12.7*) idx=cu126;; 13.*) idx=cu132;; *) idx=cu128;; esac
.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || uv pip install --python .venv/bin/python --reinstall torch torchvision --index-url https://download.pytorch.org/whl/$idx -q
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
. scripts/env.sh   # reads ~/.config/wsp/env (HF_TOKEN, LAMBDA_API_KEY) written by the launcher
hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
mkdir -p data/zoo && hf download Aralario/wsp-lab-zoo --repo-type dataset --include "zoo/qwen3.5-0.8b/manifest.jsonl" "zoo/qwen3.5-0.8b/*/*.json" --local-dir data/.hf-pull-zoo >/dev/null 2>&1 && rsync -a data/.hf-pull-zoo/zoo/ data/zoo/ || true   # metadata only: deltas are public on HF now and not needed for a build
hf download Qwen/Qwen3.5-0.8B >/dev/null 2>&1
python - <<PY
import yaml
c=yaml.safe_load(open('configs/zoo_qwen3.5-0.8b.yaml'))
c['families']={'heretic_ara': {**c['families']['heretic_ara'], 'n': int("$N"), 'trials': int("$TRIALS")}}
c['out_dir']='data/zoo/qwen3.5-0.8b'
yaml.safe_dump(c, open('configs/zoo_qwen3.5-0.8b_ara_lambda.yaml','w'), sort_keys=False)
PY
echo "ARA build start $(date -u +%FT%TZ) n=$N trials=$TRIALS"
python -m wsp.zoo.build --config configs/zoo_qwen3.5-0.8b_ara_lambda.yaml --family heretic_ara --allow-missing-families 2>&1 | tail -40
echo "ARA build end $(date -u +%FT%TZ)"
scripts/hf_sync.sh push zoo 2>&1 | tail -1
echo JOB_DONE
# self-terminate: only the instance whose public IP is ours (several wsp-lab instances may run at once)
MYIP=$(curl -s --max-time 10 https://checkip.amazonaws.com || true)
ID=$(curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambda.ai/api/v1/instances | python3 -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin)['data'] if i.get('ip')=='$MYIP' and i['name'].startswith('wsp-lab')]" | head -1)
[ -n "$ID" ] && curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate -d "{\"instance_ids\":[\"$ID\"]}" >/dev/null && echo "terminated $ID"
