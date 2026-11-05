#!/usr/bin/env bash
# TF-v1 D3 scoring queue (amendment A2.5): resumable, per-entry minutes logged.
#   phase A: the 229 D3 stratum entries (139 chain, 60 lora_sft, 30 task_vector) as scored (canonical)
#            ids, each chain predecessor included (they are D3 rows themselves);
#   phase B: the null-family D3 controls (abliteration, random_dir) that the section 3 fit trains on.
# Fresh confirmatory components (including the six spares) are NOT scored here.
# Resume by running this script again: completed entries are skipped (index.jsonl signature).
# Usage (as launched): agent-tmux tfv1-d3-stratum bash scripts/tfv1_d3_stratum.sh
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
python scripts/tfv1_scoring_plan.py --ids-file-prefix data/runs/h1a/d3_ids > data/runs/h1a/d3_plan_summary.json || exit 1
cat data/runs/h1a/d3_plan_summary.json
B="python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml"
stamp() { awk '{ print strftime("%FT%TZ"), $0; fflush() }'; }
( while sleep 300; do python scripts/tfv1_d3_minutes.py --null; done ) &
LOOP=$!
trap 'kill $LOOP 2>/dev/null' EXIT
run_phase() {
  for attempt in 1 2 3; do
    echo "=== $1 attempt $attempt $(date -u +%FT%TZ)"
    if $B --ids $(cat "data/runs/h1a/d3_ids.$1.txt") 2>&1 | stamp; then
      if python scripts/tfv1_d3_minutes.py --check $2; then return 0; fi
    fi
  done
  return 1
}
run_phase stratum "" || { echo "STRATUM FAILED"; exit 1; }
echo "=== STRATUM COMPLETE $(date -u +%FT%TZ)"
python scripts/tfv1_d3_minutes.py
run_phase null --null || { echo "NULL-FAMILY FAILED"; exit 1; }
echo "=== D3 COMPLETE $(date -u +%FT%TZ)"
python scripts/tfv1_d3_minutes.py --null
