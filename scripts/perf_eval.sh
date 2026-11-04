#!/usr/bin/env bash
# Run through agent-tmux. Never borrows the zoo/post-zoo GPU reservation.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
until ! agent-tmux -l | grep -qE '^(p1-zoo-full|post-zoo):'; do sleep 120; done
echo "GPU_FREE $(date -u +%FT%TZ)"
mkdir -p data/runs/perf
id=$(python - <<'PY'
import json
for line in open('data/zoo/qwen3.5-0.8b/manifest.jsonl'):
    row = json.loads(line)
    if row.get('eligibility') == 'production' and row.get('family') == 'abliteration':
        print(row['id'])
        break
PY
)
test -n "$id"
echo "PERF_ENTRY=$id"
python -m pytest -m 'gpu and slow' tests/test_eval_perf.py -q > data/runs/perf/gpu-test.log 2>&1 || echo 'GPU_TEST_FAILED (see log)'
for mode in hf static compile vllm; do
    python -m wsp.eval.perf --entry "data/zoo/qwen3.5-0.8b/$id" --mode "$mode" --out "data/runs/perf/probe-$mode.json" > "data/runs/perf/probe-$mode.log" 2>&1 || echo "PROBE_FAILED=$mode"
done
python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --ids "$id" --out data/runs/perf/hf-b32 --skip-base > data/runs/perf/hf-b32.log 2>&1
python - <<'PY'
import json
from pathlib import Path
import yaml
root = Path('data/runs/perf')
baseline = json.loads((root / 'probe-hf.json').read_text())
valid = []
for name in ('static', 'compile'):
    path = root / f'probe-{name}.json'
    if path.exists():
        row = json.loads(path.read_text())
        if row['results'] == baseline['results']:
            valid.append(row)
cfg = yaml.safe_load(Path('configs/eval_fast.yaml').read_text())
if valid:
    best = min(valid, key=lambda row: row['seconds'])
    cfg['generation']['cache_implementation'] = 'static'
    cfg['generation']['choice_batch_size'] = 32
    cfg['generation']['compile'] = best['mode'] == 'compile'
(root / 'hf-best.yaml').write_text(yaml.safe_dump(cfg))
path = root / 'probe-vllm.json'
if path.exists():
    other = json.loads(path.read_text())
    matches = sum(a == b for a, b in zip(baseline['results'], other['results']))
    (root / 'agreement.json').write_text(json.dumps({'matches': matches, 'n': 32}, indent=2))
PY
python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --ids "$id" --config data/runs/perf/hf-best.yaml --out data/runs/perf/hf-best --skip-base > data/runs/perf/hf-best.log 2>&1
if test -f data/runs/perf/probe-vllm.json; then
    python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --ids "$id" --backend vllm --out data/runs/perf/vllm --skip-base > data/runs/perf/vllm.log 2>&1 || echo 'VLLM_BATCH_FAILED'
fi
python scripts/perf_report.py
echo "PERF_MEASUREMENTS_DONE $(date -u +%FT%TZ)"
