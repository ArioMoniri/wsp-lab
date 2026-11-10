#!/usr/bin/env bash
# Runs ON a Lambda H100 instance: F5.3 4B scale diagnostic (diagnostic, not a gate). Launch only after Board A has frozen A5
# with amendment F5. Builds the 20 Gaussian edits of Qwen3.5-4B on the box with scripts/scale4b_build_edits.py, scores the
# base + 20 entries with TF-v1 restricted to its WikiText stage (configs/eval_tf_v1_wikitext.yaml; wsp.eval.tf_v1 runs
# only the stages listed under `benchmarks`), then the immutable same-device repeat `r1` of the base + the 5 A3-paired
# entries, uploads entries, records, log and a ledger line to the HF zoo dataset, and terminates itself.
# Hard wall-clock cap: CAP_H GPU-h (default 3.0) from LAUNCH_EPOCH; a detached watchdog terminates the instance at the cap
# whatever the job is doing.
#
# Env: WSP_REV (full SHA, required)  DEVICE_ID (opaque H<n>, not H0, allocated privately; required)
#      MODEL_REV (default: the checkpoint-verified Qwen3.5-4B commit)  LAUNCH_EPOCH (launcher's launch time, default now)
#      CAP_H (3.0)  KEEP_ON_FAILURE (0; 1 keeps a failed instance until the watchdog)  ~/.config/wsp/env: HF_TOKEN, LAMBDA_API_KEY
# Usage (from the launcher): ssh ubuntu@<ip> "WSP_REV=… DEVICE_ID=H<n> LAUNCH_EPOCH=… nohup bash ~/lambda_job_scale4b.sh &"
set -euo pipefail
WSP_REV="${WSP_REV:?set WSP_REV to the reviewed git revision (full SHA) to run}"
DEVICE_ID="${DEVICE_ID:?set DEVICE_ID to the opaque H<number> (not H0)}"
MODEL_REV="${MODEL_REV:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
LAUNCH_EPOCH="${LAUNCH_EPOCH:-$(date +%s)}"; CAP_H="${CAP_H:-3.0}"; KEEP_ON_FAILURE="${KEEP_ON_FAILURE:-0}"
UPLOAD_RESERVE_S=900   # scoring stops this long before the cap so results and the log still upload
[[ "$DEVICE_ID" =~ ^H[0-9]+$ && "$DEVICE_ID" != H0 ]] || { echo "DEVICE_ID must be H<number> and not H0"; exit 2; }
[[ "$WSP_REV" =~ ^[0-9a-f]{40}$ && "$MODEL_REV" =~ ^[0-9a-f]{40}$ ]] || { echo "WSP_REV and MODEL_REV must be full SHAs"; exit 2; }
LOG=~/scale4b.log; exec > >(tee -a "$LOG") 2>&1
ZOO=data/zoo/qwen3.5-4b-scale; EVALS=data/evals/qwen3.5-4b-scale/tf_v1; CFG=configs/eval_tf_v1_wikitext.yaml
TAG="scale4b-$DEVICE_ID-$(date -u +%Y%m%dT%H%MZ)"; HF_REPO=Aralario/wsp-lab-zoo
DEADLINE=$(python3 -c "print(int($LAUNCH_EPOCH + $CAP_H * 3600))")
echo "SCALE4B start $(date -u +%FT%TZ) launch=$(date -u -d "@$LAUNCH_EPOCH" +%FT%TZ) cap_h=$CAP_H deadline=$(date -u -d "@$DEADLINE" +%FT%TZ) device=$DEVICE_ID"

my_instance() {  # id, type, $/h of the wsp-lab instance whose public IP is ours (several may run at once)
  local ip; ip=$(curl -s --max-time 10 https://checkip.amazonaws.com || true)
  curl -s --max-time 20 -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambda.ai/api/v1/instances | python3 -c "
import sys, json
for i in json.load(sys.stdin)['data']:
    if i.get('ip') == '$ip' and i['name'].startswith('wsp-lab'):
        t = i['instance_type']; print(i['id'], t['name'], t['price_cents_per_hour'] / 100); break"
}
terminate_self() {
  local id; id=$(my_instance | awk '{print $1}')
  [ -n "$id" ] && curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json" \
    -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate -d "{\"instance_ids\":[\"$id\"]}" >/dev/null && echo "terminated $id"
}
ledger_line() {  # board/lambda_spend.md row; billed hours from LAUNCH_EPOCH to now (termination follows within a minute)
  local now id type price; now=$(date +%s); read -r id type price < <(my_instance || true)
  python3 - "$LAUNCH_EPOCH" "$now" "${id:-unknown}" "${type:-unknown}" "${price:-0}" "$1" <<'PY'
import sys, datetime as dt
s, e, iid, typ, price, job = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6]
f = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
h = (e - s) / 3600
print(f"| {f(s)} | {f(e)} | {iid} | {typ} | {price:.2f} | {h:.2f} | {h * price:.2f} | {job} |")
PY
}
upload() {  # entries (edit.json + delta), manifest, this host's TF-v1 records, the log and the ledger line; best effort
  python3 - "$TAG" "$DEVICE_ID" "$ZOO" "$EVALS" "$LOG" <<'PY' || echo "UPLOAD_FAILED"
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
tag, dev, zoo, evals, log = sys.argv[1:6]
api = HfApi(token=os.environ["HF_TOKEN"])
if Path(zoo, "manifest.jsonl").exists():
    api.upload_folder(repo_id="Aralario/wsp-lab-zoo", repo_type="dataset", folder_path=zoo, path_in_repo="zoo/qwen3.5-4b-scale",
                      allow_patterns=["manifest.jsonl", "*/edit.json", "*/delta.safetensors"], commit_message=f"{tag} entries")
if Path(evals).exists():
    api.upload_folder(repo_id="Aralario/wsp-lab-zoo", repo_type="dataset", folder_path=evals,
                      path_in_repo=f"evals/qwen3.5-4b-scale/lambda-{dev}", ignore_patterns=["*.lock", "*.tmp", "**/tf_cache/**"],
                      commit_message=f"{tag} TF-v1 WikiText records")
api.upload_file(path_or_fileobj=log, path_in_repo=f"logs/{tag}.log", repo_id="Aralario/wsp-lab-zoo", repo_type="dataset",
                commit_message=f"{tag} log")
print("UPLOAD_DONE")
PY
}

# Watchdog: detached, survives this shell; at the cap it uploads the log (60 s max) and terminates the instance.
setsid nohup bash -c "
  sleep \$(( $DEADLINE - \$(date +%s) )); echo \"SCALE4B cap reached \$(date -u +%FT%TZ)\" >> $LOG
  set -a; . ~/.config/wsp/env; set +a
  PY=\$HOME/projects/wsp-lab/.venv/bin/python; [ -x \$PY ] || PY=python3
  timeout 60 \$PY -c \"import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='$LOG', path_in_repo='logs/$TAG-cap.log', repo_id='$HF_REPO', repo_type='dataset')\" || true
  ip=\$(curl -s --max-time 10 https://checkip.amazonaws.com)
  id=\$(curl -s -H \"Authorization: Bearer \$LAMBDA_API_KEY\" https://cloud.lambda.ai/api/v1/instances | python3 -c \"import sys,json;[print(i['id']) for i in json.load(sys.stdin)['data'] if i.get('ip')=='\$ip' and i['name'].startswith('wsp-lab')]\" | head -1)
  [ -n \"\$id\" ] && curl -s -H \"Authorization: Bearer \$LAMBDA_API_KEY\" -H 'Content-Type: application/json' -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate -d \"{\\\"instance_ids\\\":[\\\"\$id\\\"]}\"
" >/dev/null 2>&1 < /dev/null &

finish() {
  local rc=$?; set +e
  echo "SCALE4B end $(date -u +%FT%TZ) rc=$rc"
  local job="F5.3 4B scale diagnostic: build 20 Gaussian edits + TF-v1 WikiText base+20 and repeat r1 (base+5 paired), rev ${WSP_REV:0:12}, rc=$rc"
  echo "LEDGER $(ledger_line "$job")"
  sleep 2   # let tee flush the log before it is uploaded
  [ -n "${HF_TOKEN:-}" ] && upload
  if [ "$rc" != 0 ] && [ "$KEEP_ON_FAILURE" = 1 ]; then echo "JOB_FAILED rc=$rc — instance kept until the cap watchdog"; exit "$rc"; fi
  [ "$rc" = 0 ] && echo JOB_DONE || echo "JOB_FAILED rc=$rc"
  [ -n "${LAMBDA_API_KEY:-}" ] && terminate_self
  exit "$rc"
}
trap finish EXIT

# ---- bootstrap (as lambda_job_ara.sh: pinned repo revision, driver-matched torch) ----
export PATH=$HOME/.local/bin:$PATH
sudo apt-get update -qq >/dev/null && sudo apt-get install -y -qq git rsync tmux python3.12-venv python3.12-dev build-essential >/dev/null || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
mkdir -p ~/projects && cd ~/projects
[ -d wsp-lab ] || git clone -q https://github.com/ArioMoniri/wsp-lab.git
cd wsp-lab && git fetch -q origin && git checkout -q --detach "$WSP_REV"
[ "$(git rev-parse HEAD)" = "$WSP_REV" ] || { echo "revision mismatch: $(git rev-parse HEAD) != $WSP_REV"; exit 1; }
[ -z "$(git status --porcelain -- src configs scripts)" ] || { echo "checkout not clean"; exit 1; }
echo "running wsp-lab at $(git rev-parse --short HEAD)"
[ -d .venv ] || uv venv --python 3.12 .venv -q
uv pip install --python .venv/bin/python -e ".[dev]" -q
drv=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
case "$drv" in 12.8*|12.9*) idx=cu128;; 12.6*|12.7*) idx=cu126;; 13.*) idx=cu132;; *) idx=cu128;; esac
.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || uv pip install --python .venv/bin/python --reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$idx" -q
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name())"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100 || { echo "not an H100 host"; exit 1; }
. scripts/env.sh   # HF_HOME, PYTHONPATH, secrets from ~/.config/wsp/env
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p ~/.config/wsp && echo "$DEVICE_ID" > ~/.config/wsp/device_id
hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true

# ---- pin and fetch the model ----
hf download Qwen/Qwen3.5-4B --revision "$MODEL_REV" >/dev/null
[ -d "$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/$MODEL_REV" ] || { echo "model snapshot $MODEL_REV missing"; exit 1; }
echo "model Qwen/Qwen3.5-4B pinned at $MODEL_REV"

# ---- build the 20 entries on the box (same script as everywhere; the dry run prints the plan first) ----
python scripts/scale4b_build_edits.py --dry-run --revision "$MODEL_REV" | python -c "import json,sys;h=json.load(sys.stdin)['header'];print('plan layers',h['layers'],'tensors',len(h['support']),'paired',h['paired_ids'])"
t0=$(date +%s)
python scripts/scale4b_build_edits.py --revision "$MODEL_REV" --out "$ZOO"
echo "BUILD seconds=$(( $(date +%s) - t0 )) entries=$(ls -d $ZOO/*/edit.json | wc -l)"
PAIRED=$(python -c "import json;print(','.join(json.loads(open('$ZOO/manifest.jsonl').readline())['paired_ids']))")

# ---- score: base + 20, then the immutable same-device repeat r1 of base + 5 paired ----
remaining() { echo $(( DEADLINE - UPLOAD_RESERVE_S - $(date +%s) )); }
t0=$(date +%s)
timeout "$(remaining)" python -m wsp.eval.batch --zoo "$ZOO" --config "$CFG" 2>&1 | { grep -E '^\{|entry=|TF wikitext|Traceback|Error|error' || true; } | cut -c1-400
echo "SCORE seconds=$(( $(date +%s) - t0 )) done=$(ls $EVALS/*/eval.json 2>/dev/null | wc -l)"
t0=$(date +%s)
timeout "$(remaining)" python -m wsp.eval.batch --zoo "$ZOO" --config "$CFG" --ids "$PAIRED" --repeat r1 2>&1 | { grep -E 'entry=|TF wikitext|Traceback|Error|error' || true; } | cut -c1-400
echo "REPEAT seconds=$(( $(date +%s) - t0 ))"

# ---- timing per entry (from each eval.json) ----
python - "$EVALS" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("**/eval.json")):
    m = json.loads(p.read_text())
    print(f"TIMING {p.parent.relative_to(sys.argv[1])} wall_s={m.get('wall_seconds', 0):.1f} "
          f"wikitext_s={(m.get('stage_seconds') or {}).get('wikitext_ppl', 0):.1f} log_ppl={m.get('wikitext', {}).get('log_ppl')}")
PY
n=$(ls "$EVALS"/*/eval.json 2>/dev/null | wc -l); [ "$n" -ge 21 ] || { echo "only $n of 21 evaluations"; exit 1; }
# finish() (EXIT trap) writes the ledger line, uploads, and terminates.
