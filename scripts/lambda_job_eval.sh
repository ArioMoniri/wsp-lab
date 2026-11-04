#!/usr/bin/env bash
# Run ON a Lambda instance: evaluate one shard of zoo entries (fast tier, HF path) and upload the eval artefacts to the
# public HF zoo dataset. Self-terminates when done. Usage: bash ~/lambda_job_eval.sh ~/ids.txt
# Needs ~/.config/wsp/env (HF_TOKEN, LAMBDA_API_KEY) written by the launcher. Log: ~/eval.log (tee'd by the launcher).
set -euo pipefail
IDS="${1:?ids file}"
export PATH=$HOME/.local/bin:$PATH
sudo apt-get update -qq >/dev/null && sudo apt-get install -y -qq git rsync tmux python3.12-venv python3.12-dev build-essential >/dev/null || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
mkdir -p ~/projects && cd ~/projects
[ -d wsp-lab ] || git clone -q https://github.com/ArioMoniri/wsp-lab.git
cd wsp-lab && git pull -q
[ -d .venv ] || uv venv --python 3.12 .venv -q
uv pip install --python .venv/bin/python -e ".[dev]" -q
# torch wheel index must match the driver (Lambda images ship CUDA 12.8 drivers; cu132 wheels do not load)
drv=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
case "$drv" in 12.8*|12.9*) idx=cu128;; 12.6*|12.7*) idx=cu126;; 13.*) idx=cu132;; *) idx=cu128;; esac
.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || uv pip install --python .venv/bin/python --reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$idx" -q
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
. scripts/env.sh
hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
hf download Qwen/Qwen3.5-0.8B >/dev/null 2>&1
# shard inputs: manifest + the shard entries (edit.json, delta / lora) from the public zoo store
mkdir -p data/zoo
inc=(--include "zoo/qwen3.5-0.8b/manifest.jsonl")
while read -r i; do [ -n "$i" ] && inc+=("zoo/qwen3.5-0.8b/$i/*"); done < "$IDS"
hf download Aralario/wsp-lab-zoo --repo-type dataset "${inc[@]}" --local-dir data/.hf-pull-zoo >/dev/null 2>&1
rsync -a data/.hf-pull-zoo/zoo/ data/zoo/
rm -f data/zoo/qwen3.5-0.8b/*/eval.json   # every shard entry is evaluated here, never inherited from another host
HOST="lambda-$(hostname)"
echo "EVAL start $(date -u +%FT%TZ) host=$HOST ids=$(wc -l < "$IDS")"
set +e
python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --ids "$(paste -sd, "$IDS")" --config configs/eval_fast.yaml --resume 2>&1 | grep -E '^\{|entry=|Traceback|Error|error'
rc=${PIPESTATUS[0]}
set -e
echo "EVAL end $(date -u +%FT%TZ) rc=$rc done=$(ls data/zoo/qwen3.5-0.8b/*/eval.json 2>/dev/null | wc -l)"
# upload what exists (also on failure): per-entry eval artefacts + this host's base evaluation
for attempt in 1 2 3; do python - "$IDS" "$HOST" <<'PY' && break || sleep 60
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
host = sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
pats = [f"{i}/{f}" for i in ids for f in ("eval.json", "summary.json", "records.jsonl", "manifest.json")]
api.upload_folder(repo_id="Aralario/wsp-lab-zoo", repo_type="dataset", folder_path="data/zoo/qwen3.5-0.8b",
                  path_in_repo="zoo/qwen3.5-0.8b", allow_patterns=pats, commit_message=f"eval shard {host}")
base = Path("data/evals/qwen3.5-0.8b/base")
if base.exists():
    api.upload_folder(repo_id="Aralario/wsp-lab-zoo", repo_type="dataset", folder_path=str(base),
                      path_in_repo=f"evals/qwen3.5-0.8b/base-{host}", commit_message=f"base eval {host}")
print("UPLOAD_DONE")
PY
done
echo JOB_DONE
# self-terminate: only the instance whose public IP is ours (several wsp-lab instances may run at once)
MYIP=$(curl -s --max-time 10 https://checkip.amazonaws.com || true)
ID=$(curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambda.ai/api/v1/instances | python3 -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin)['data'] if i.get('ip')=='$MYIP' and i['name'].startswith('wsp-lab')]" | head -1)
[ -n "$ID" ] && curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate -d "{\"instance_ids\":[\"$ID\"]}" >/dev/null && echo "terminated $ID"
