#!/bin/bash
# Part A scoring launch (exclusive GPU). Run ONLY after Board A accepts the re-stage report and (if yield < 40) the slot-masked
# calibration passes. Records UTC start/end, execution SHA, clean status, GPU identity, and re-hashes the staged bytes first.
cd /home/ubuntu/projects/wsp-lab || exit 1
R=docs/prereg/a5_manifests/part_a/score_launch.json
SHA=$(git rev-parse HEAD); DIRTY=$(git status --porcelain | wc -l); START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GPU=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
if [ "$DIRTY" != "0" ]; then echo "TREE NOT CLEAN ($DIRTY files); refusing"; exit 2; fi
PYTHONPATH=src .venv/bin/python - <<'PY' || { echo "STAGED HASH RECHECK FAILED"; exit 3; }
import json,hashlib,sys
m=json.load(open('docs/prereg/a5_manifests/part_a/manifest.json'))
stage=m['staged_zoo']; bad=0; n=0
for vid,sha in m['delta_sha256'].items():
    for fn in ('delta.safetensors','increment.safetensors'):
        p=f"{stage}/{vid}/{fn}"
        try: data=open(p,'rb').read()
        except FileNotFoundError: continue
        n+=1
        if isinstance(sha,dict): exp=sha.get(fn)
        else: exp=sha
        if exp and hashlib.sha256(data).hexdigest()!=exp: bad+=1; print('MISMATCH',vid,fn)
print(f"rehashed {n} files, {bad} mismatches"); sys.exit(1 if bad else 0)
PY
printf '{"schema":"a5-part-a-score-launch-v1","start_utc":"%s","execution_sha":"%s","dirty_files":%s,"gpu":"%s","command":"env PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention score --stage data/runs/a5/part_a/qwen3.5-0.8b --manifest docs/prereg/a5_manifests/part_a/manifest.json --config configs/eval_tf_v1.yaml --text-groups docs/prereg/a5_manifests/text_groups.json"}\n' "$START" "$SHA" "$DIRTY" "$GPU" > $R
/usr/bin/time -v env PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention score --stage data/runs/a5/part_a/qwen3.5-0.8b --manifest docs/prereg/a5_manifests/part_a/manifest.json --config configs/eval_tf_v1.yaml --text-groups docs/prereg/a5_manifests/text_groups.json
RC=$?; END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - "$R" "$END" "$RC" <<'PY'
import json,sys; p,end,rc=sys.argv[1:]; d=json.load(open(p)); d.update({"end_utc":end,"exit_code":int(rc)}); json.dump(d,open(p,'w'),indent=1)
PY
echo "SCORE_EXIT $RC"
