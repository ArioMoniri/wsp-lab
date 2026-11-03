#!/usr/bin/env bash
# Launch only via agent-tmux p2-reproduce scripts/eval_p2.sh <run-prefix> <tmux-log>.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
prefix="${1:?run prefix required}"
log="${2:?tmux log path required}"
base="$HOME/projects/wsp-lab/data/runs/p2"
python -m wsp.eval.run --config configs/eval_cheap.yaml --model Qwen/Qwen3.5-0.8B --out "$base/$prefix-4096"
python -m wsp.eval.report --runs "$base/$prefix-4096" --log "$log"
python -m wsp.eval.run --config configs/eval_faithful.yaml --model Qwen/Qwen3.5-0.8B --out "$base/$prefix-32768"
python -m wsp.eval.report --runs "$base/$prefix-4096" "$base/$prefix-32768" --log "$log"
scripts/session.sh note "Completed P2 reproduction $prefix; report updated from records. Log $log"
WSP_PHASE=P2 scripts/board.sh done eval-engineer "src/wsp/eval configs/eval_subsets docs/RESULTS.md" "feat/p2-eval: both full reproduction budgets complete; per-item-derived scores, Wilson intervals and blog gaps in docs/RESULTS.md. Log $log. CPU and GPU determinism tests passed; complete second-run reproduction gate remains open."
git add docs/RESULTS.md board/P2.md "$(scripts/session.sh current)"
git commit -m "P2 record completed base reproduction and blog gaps"
