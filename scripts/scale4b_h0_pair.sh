#!/usr/bin/env bash
# F5.3 4B scale diagnostic, local H0 half of the A3 device pairing. Run AFTER G2 and after the H100 job
# (scripts/lambda_job_scale4b.sh) has uploaded its entries and records; run it under agent-tmux, e.g.
#   SCALE4B_AFTER_G2=1 WSP_REV=<sha the H100 job ran> H100_ID=H<n> agent-tmux scale4b-h0 bash scripts/scale4b_h0_pair.sh
# Exclusive-GPU pattern of scripts/d8_tf_yh.sh: clean checkout at the same revision as the H100 job, no other compute
# process on the GPU. Steps: pull the manifest + the 5 paired entries (exact H100 bytes, sha256 checked against the
# manifest) and the H100 records from the HF zoo dataset; check that this host rebuilds the same 5 deltas (logged,
# descriptive); score base + the 5 paired entries with configs/eval_tf_v1_wikitext.yaml on H0, then the immutable
# repeat r1 of the same six; write the report with scripts/scale4b_report.py (the report is committed by hand).
set -euo pipefail
[ "${SCALE4B_AFTER_G2:-0}" = 1 ] || { echo "run only after G2 (set SCALE4B_AFTER_G2=1 to confirm)" >&2; exit 2; }
WSP_REV="${WSP_REV:?set WSP_REV to the full SHA the H100 job ran}"
H100_ID="${H100_ID:?set H100_ID to the H100 job DEVICE_ID (H<number>)}"
MODEL_REV="${MODEL_REV:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
W="$(cd "$(dirname "$0")/.." && pwd)"
cd "$W"
. scripts/env.sh
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
[ "$(git rev-parse HEAD)" = "$WSP_REV" ] || { echo "HEAD $(git rev-parse HEAD) is not $WSP_REV" >&2; exit 1; }
[ -z "$(git status --porcelain -- src configs scripts)" ] || { echo "worktree not clean" >&2; exit 1; }
[ "$(cat ~/.config/wsp/device_id)" = H0 ] || { echo "this half must run on H0" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
ZOO=data/zoo/qwen3.5-4b-scale; EVALS=data/evals/qwen3.5-4b-scale; CFG=configs/eval_tf_v1_wikitext.yaml
PULL=data/.hf-pull-scale4b
date -u +"start %FT%TZ"

# ---- inputs from the H100 job ----
hf download Aralario/wsp-lab-zoo --repo-type dataset --include "zoo/qwen3.5-4b-scale/manifest.jsonl" --local-dir "$PULL" >/dev/null
PAIRED=$(python -c "import json;print(' '.join(json.loads(open('$PULL/zoo/qwen3.5-4b-scale/manifest.jsonl').readline())['paired_ids']))")
# one download call per entry (a single call with many --include patterns can drop some; see lambda_job_eval.sh)
for i in $PAIRED; do hf download Aralario/wsp-lab-zoo --repo-type dataset --include "zoo/qwen3.5-4b-scale/$i/*" --local-dir "$PULL" >/dev/null; done
hf download Aralario/wsp-lab-zoo --repo-type dataset --include "evals/qwen3.5-4b-scale/lambda-$H100_ID/*" --local-dir "$PULL" >/dev/null
hf download Aralario/wsp-lab-zoo --repo-type dataset --include "logs/scale4b-$H100_ID-*.log" --local-dir "$PULL" >/dev/null || true
python - "$PULL/zoo/qwen3.5-4b-scale" $PAIRED <<'PY'
import hashlib, json, sys
from pathlib import Path
zoo, ids = Path(sys.argv[1]), sys.argv[2:]
rows = {r["id"]: r for r in map(json.loads, (zoo / "manifest.jsonl").read_text().splitlines()[1:])}
for i in ids:
    got = hashlib.sha256((zoo / i / "delta.safetensors").read_bytes()).hexdigest()
    assert got == rows[i]["delta_sha256"], f"{i}: downloaded delta does not match the manifest"
print("paired entries match the manifest:", " ".join(ids))
PY
mkdir -p "$ZOO" "$EVALS"
rsync -a "$PULL/zoo/qwen3.5-4b-scale/" "$ZOO/"
rsync -a "$PULL/evals/qwen3.5-4b-scale/lambda-$H100_ID/" "$EVALS/lambda-$H100_ID/"

# ---- model pin; rebuild check (descriptive: the scored bytes are the H100 ones checked above) ----
hf download Qwen/Qwen3.5-4B --revision "$MODEL_REV" >/dev/null
REV=$(python -c "import json;print(json.loads(open('$ZOO/manifest.jsonl').readline())['base_revision'])")
[ "$REV" = "$MODEL_REV" ] || { echo "manifest base revision $REV is not $MODEL_REV" >&2; exit 1; }
# shellcheck disable=SC2086
python scripts/scale4b_build_edits.py --revision "$MODEL_REV" --verify "$ZOO" --ids $PAIRED \
  && echo "H0 rebuild: identical deltas" || echo "H0 rebuild: deltas differ from the H100 build (descriptive; scoring uses the H100 bytes)"

# ---- score on H0: base + 5 paired, then the immutable repeat r1 of the same six ----
# shellcheck disable=SC2086
python -m wsp.eval.batch --zoo "$ZOO" --config "$CFG" --ids $PAIRED
# shellcheck disable=SC2086
python -m wsp.eval.batch --zoo "$ZOO" --config "$CFG" --ids $PAIRED --repeat r1
date -u +"end %FT%TZ"

# ---- report (written to docs/reviews/scale/; review and commit it by hand) ----
LOGF=$(ls -t "$PULL"/logs/scale4b-"$H100_ID"-*.log 2>/dev/null | grep -v -- '-cap.log$' | head -1 || true)
python scripts/scale4b_report.py --zoo "$ZOO" --h100 "$EVALS/lambda-$H100_ID" --h0 "$EVALS/tf_v1" ${LOGF:+--job-log "$LOGF"}
