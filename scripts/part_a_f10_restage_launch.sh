#!/bin/bash
cd /home/ubuntu/projects/wsp-lab || exit 1
R=docs/prereg/a5_manifests/part_a/restage_launch.json
SHA=$(git rev-parse HEAD); DIRTY=$(git status --porcelain | wc -l)
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"schema":"a5-f10-restage-launch-v1","start_utc":"%s","execution_sha":"%s","dirty_files":%s,"cwd":"%s","command":"env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention prepare --rotation-only --audit-rule f10 --staged-manifest docs/prereg/a5_manifests/part_a/manifest.json --zoo data/zoo/qwen3.5-0.8b --selection docs/prereg/a5_manifests/selection.json --references docs/prereg/a5_manifests/references.json --out data/runs/a5/part_a --manifest-out docs/prereg/a5_manifests/part_a/manifest.json","nproc":%s,"load_at_start":"%s"}\n' "$START" "$SHA" "$DIRTY" "$PWD" "$(nproc)" "$(cut -d' ' -f1-3 /proc/loadavg)" > $R
if [ "$DIRTY" != "0" ]; then echo "TREE NOT CLEAN ($DIRTY files); refusing"; exit 2; fi
/usr/bin/time -v env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m wsp.eval.response_intervention prepare --rotation-only --audit-rule f10 --staged-manifest docs/prereg/a5_manifests/part_a/manifest.json --zoo data/zoo/qwen3.5-0.8b --selection docs/prereg/a5_manifests/selection.json --references docs/prereg/a5_manifests/references.json --out data/runs/a5/part_a --manifest-out docs/prereg/a5_manifests/part_a/manifest.json
RC=$?
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - "$R" "$END" "$RC" <<'PY'
import json,sys; p,end,rc=sys.argv[1:]; d=json.load(open(p)); d.update({"end_utc":end,"exit_code":int(rc)}); json.dump(d,open(p,'w'),indent=1)
PY
echo "RESTAGE_EXIT $RC"
