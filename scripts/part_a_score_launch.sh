#!/bin/bash
# Part A scoring launch in two phases (PRESCORE_INVENTORY §10, LS-1 to LS-4). Run each phase inside agent-tmux, from the
# checkout whose scripts/ holds this file (the repository root is this script's parent directory).
#
#   Phase 1, proofs only. Reads the twelve supplementary originals' TF-v1 records, so run it only after the final freeze:
#     scripts/part_a_score_launch.sh <frozen manifest sha256>
#   Clean tree, `--verify-pins HEAD` = "all pins match", manifest sha256 = argument 1, strict re-hash of every file in the
#   manifest's byte inventory (delta_sha256 and variant_json_sha256; a missing file fails), then `score --proofs-only`,
#   which writes reference_proofs_f11.json beside the manifest and prints the repriced count. The script then STOPS.
#   Commit the proof record and have the repriced plan accepted before phase 2.
#
#   Phase 2, GPU scoring (exclusive GPU):
#     scripts/part_a_score_launch.sh <frozen manifest sha256> --phase score --max-minutes <stop-and-reprice threshold>
#   The same four checks again (so the proof record must be committed), refusal if the launch record already exists, a
#   dry-run gate (`score --dry-run --require-proofs --max-minutes T`: the proof record exists, matches the manifest, proves
#   or fails every original, and its price <= T), the GPU identity; only then the launch record and the timed scoring.
#
# Every refusal exits non-zero before any scoring command and before the launch record is written; nothing under
# data/evals is written by any refusal. Exit codes: 64 usage, 2 tree, 3 pins, 4 manifest hash, 5 re-hash, 6 proof step,
# 7 proof/price gate, 8 launch record exists, 9 no GPU identity; otherwise the scorer's exit code.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd) && cd "$ROOT" || exit 1
PY=.venv/bin/python
M=docs/prereg/a5_manifests/part_a/manifest.json
R=docs/prereg/a5_manifests/part_a/score_launch.json
STAGE=data/runs/a5/part_a/qwen3.5-0.8b
CFG=configs/eval_tf_v1.yaml
TG=docs/prereg/a5_manifests/text_groups.json

usage() {
    echo "usage: $0 <manifest sha256> [--phase proofs | --phase score --max-minutes <threshold>]" >&2
    echo "refusing: $1" >&2
    exit 64
}
refuse() { echo "$2; refusing" >&2; exit "$1"; }

EXPECT=${1:-}
[[ $EXPECT =~ ^[0-9a-f]{64}$ ]] || usage "argument 1 must be the frozen manifest sha256 (64 lowercase hex)"
shift
PHASE=proofs; MAXMIN=
while [ $# -gt 0 ]; do
    case "$1" in
        --phase) [ $# -ge 2 ] || usage "--phase needs a value"; PHASE=$2; shift 2 ;;
        --max-minutes) [ $# -ge 2 ] || usage "--max-minutes needs a value"; MAXMIN=$2; shift 2 ;;
        *) usage "unknown argument $1" ;;
    esac
done
case $PHASE in
    proofs) [ -z "$MAXMIN" ] || usage "--max-minutes belongs to --phase score" ;;
    score) [[ $MAXMIN =~ ^[0-9]+(\.[0-9]+)?$ ]] || usage "--phase score needs --max-minutes <minutes>" ;;
    *) usage "--phase must be proofs or score" ;;
esac

checks() {
    DIRTY=$(git status --porcelain | wc -l)
    [ "$DIRTY" = 0 ] || refuse 2 "TREE NOT CLEAN ($DIRTY files)"
    PINS=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $PY scripts/a5_build_manifests.py --verify-pins HEAD 2>&1)
    PINS_RC=$?
    echo "verify-pins HEAD ($(git rev-parse HEAD)): $PINS"
    if [ "$PINS_RC" != 0 ] || ! grep -qx "all pins match" <<<"$PINS"; then refuse 3 "PIN CHECK FAILED"; fi
    GOT=$(sha256sum "$M" 2>/dev/null | cut -d' ' -f1)
    [ "$GOT" = "$EXPECT" ] || refuse 4 "MANIFEST HASH ${GOT:-absent} != frozen $EXPECT"
    echo "manifest sha256 $GOT = frozen"
    PYTHONDONTWRITEBYTECODE=1 $PY - "$M" "$STAGE" <<'PY' || refuse 5 "STAGED HASH RECHECK FAILED"
import hashlib, json, os, sys
manifest, stage = sys.argv[1:]
m = json.load(open(manifest))
if m.get("staged_zoo") != stage:
    sys.exit(f"manifest staged_zoo {m.get('staged_zoo')!r} != {stage!r}")
want = []
for vid, sha in sorted(m["delta_sha256"].items()):
    for fn, exp in sorted((sha if isinstance(sha, dict) else {"delta.safetensors": sha}).items()):
        want.append((vid, fn, exp))
n_delta = len(want)
want += [(vid, "variant.json", exp) for vid, exp in sorted(m["variant_json_sha256"].items())]
if not n_delta or len(want) == n_delta:
    sys.exit("empty byte inventory")
missing = bad = nbytes = 0
for vid, fn, exp in want:
    p = os.path.join(stage, vid, fn)
    if not os.path.isfile(p):
        missing += 1; print("MISSING", vid, fn, flush=True); continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk); nbytes += len(chunk)
    if h.hexdigest() != exp:
        bad += 1; print("MISMATCH", vid, fn, flush=True)
print(f"rehashed {len(want)} files ({n_delta} delta/increment, {len(want) - n_delta} variant.json), "
      f"{missing} missing, {bad} mismatches, {nbytes} bytes")
sys.exit(1 if missing or bad else 0)
PY
}

SCORE=(-m wsp.eval.response_intervention score --stage "$STAGE" --manifest "$M" --config "$CFG" --text-groups "$TG")
checks

if [ "$PHASE" = proofs ]; then
    echo "PHASE 1 proofs only: start $(date -u +%Y-%m-%dT%H:%M:%SZ), HEAD $(git rev-parse HEAD)"
    PYTHONPATH=src $PY "${SCORE[@]}" --proofs-only || refuse 6 "PROOF STEP FAILED"
    echo "PHASE 1 end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "STOP: nothing scored. Commit docs/prereg/a5_manifests/part_a/reference_proofs_f11.json, have the repriced plan"
    echo "above accepted, then run: $0 $EXPECT --phase score --max-minutes <accepted threshold>"
    exit 0
fi

[ ! -e "$R" ] || refuse 8 "LAUNCH RECORD EXISTS ($R): one launch per record"
PYTHONPATH=src $PY "${SCORE[@]}" --require-proofs --max-minutes "$MAXMIN" --dry-run \
    || refuse 7 "PROOF/PRICE GATE FAILED (absent or incomplete proof record, or price above $MAXMIN min)"
GPU=$(nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader 2>/dev/null | head -1)
[ -n "$GPU" ] || refuse 9 "NO GPU IDENTITY (nvidia-smi)"
SHA=$(git rev-parse HEAD); START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CMD="env PYTHONPATH=src $PY ${SCORE[*]} --require-proofs --max-minutes $MAXMIN"
$PY - "$R" <<PY
import json, sys
gpu = """$GPU"""
json.dump({"schema": "a5-part-a-score-launch-v2", "start_utc": "$START", "execution_sha": "$SHA", "dirty_files": $DIRTY,
           "manifest_sha256": "$EXPECT", "pins": "all pins match", "max_minutes": float("$MAXMIN"),
           "gpu": gpu, "gpu_uuid": gpu.split(",")[-1].strip(), "command": """$CMD"""}, open(sys.argv[1], "w"), indent=1)
PY
/usr/bin/time -v env PYTHONPATH=src $PY "${SCORE[@]}" --require-proofs --max-minutes "$MAXMIN"
RC=$?; END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
$PY - "$R" "$END" "$RC" <<'PY'
import json, sys
p, end, rc = sys.argv[1:]
d = json.load(open(p)); d.update({"end_utc": end, "exit_code": int(rc)}); json.dump(d, open(p, "w"), indent=1)
PY
echo "SCORE_EXIT $RC"
exit "$RC"
