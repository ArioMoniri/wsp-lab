#!/usr/bin/env bash
# Runs after the P1 zoo build ends: sync zoo to HF, extract features for new entries, time one clean D2 entry.
set -uo pipefail; cd "$(dirname "$0")/.."; . scripts/env.sh
until ! agent-tmux -l 2>/dev/null | grep -q "^p1-zoo-full:"; do sleep 120; done
echo "zoo build ended $(date -u +%FT%TZ)"; grep -E "ZOO_BUILD_EXIT=" ~/agent-logs/p1-zoo-full-*.log | tail -1
python3 - <<'PY'
import json,collections
rows=[json.loads(l) for l in open('data/zoo/qwen3.5-0.8b/manifest.jsonl') if l.strip()]
print("manifest families:", dict(collections.Counter(r.get('family') for r in rows if 'id' in r)))
PY
scripts/hf_sync.sh push zoo 2>&1 | tail -1
until ! agent-tmux -l 2>/dev/null | grep -q "^p3-extract:"; do sleep 60; done
python -m wsp.features.extract --zoo data/zoo/qwen3.5-0.8b --out data/features/qwen3.5-0.8b --workers 6 2>&1 | tail -5
id=$(python3 -c "
import json
for l in open('data/zoo/qwen3.5-0.8b/manifest.jsonl'):
    r=json.loads(l)
    if r.get('family')=='abliteration' and r.get('eligibility')=='production': print(r['id']); break")
echo "D2 timing on $id"; start=$(date +%s)
python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --ids "$id" 2>&1 | tail -15
echo "D2_SECONDS=$(( $(date +%s) - start ))"; echo POST_ZOO_DONE
