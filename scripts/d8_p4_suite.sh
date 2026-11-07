#!/usr/bin/env bash
# D8 step 6b: protocol-v2 P4 suite on the recovered set, runs labelled d3v2r-*; CPU, three full seeds, P4_PARALLEL runs in parallel (default 4, 2 torch threads each).
# Same run list as the 25 reported d3v2 runs (+ the m3-family attempt, which had no metrics): m1, m1_matched, m2, m2_matched,
# m5, m6 x {family, seed, trajectory}; m3 x {seed, trajectory, family}; m2 family holdouts x 5. The original launch script
# (data/runs/p4/run_suite.sh) was lost with data/; flags are the CLI defaults of configs/p4_predictors.json plus --seeds 0,1,2,
# --device cpu, --targets mmlu_pro hellaswag (the admissible teacher-forced / likelihood rate targets) and --n-boot 1000.
# Inputs: features view and evals from scripts/d8_p4_prepare.py (complete-ancestry rule).
set -uo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
DATA=/home/ubuntu/wsp-data
VIEW=${1:-$DATA/runs/d8/p4_input}
PREFIX=${PREFIX:-d3v2r}   # d3v2f for the full-set rerun after the chain regeneration
OUTROOT=$DATA/runs/p4
cd "$CODE"
export PYTHONPATH="$CODE/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTROOT"
jobs_file=$(mktemp)
for model in m1 m1_matched m2 m2_matched m5 m6; do for split in family seed trajectory; do echo "$model $split -"; done; done > "$jobs_file"
for split in seed trajectory family; do echo "m3 $split -"; done >> "$jobs_file"
for fam in abliteration chain lora_sft random_dir task_vector; do echo "m2 family $fam"; done >> "$jobs_file"
run_one() {
  model=$1; split=$2; hold=$3
  if [ "$hold" = "-" ]; then name="$PREFIX-$model-$split"; extra=(); else name="$PREFIX-$model-holdout-$hold"; extra=(--holdout "$hold"); fi
  out="$OUTROOT/$name"
  [ -e "$out" ] && { echo "SKIP $name exists"; return 0; }
  start=$(date +%s)
  $PY -m wsp.predictor.train --model "$model" --features "$VIEW/features" --evals "$VIEW/evals" \
    --zoo "$DATA/zoo/qwen3.5-0.8b" --split "$split" --seeds 0,1,2 --device cpu --n-boot 1000 \
    --targets mmlu_pro hellaswag "${extra[@]}" --out "$out" > "$DATA/runs/d8/p4-$name.log" 2>&1
  echo "DONE $name exit=$? seconds=$(( $(date +%s) - start ))"
}
export -f run_one; export PY DATA VIEW OUTROOT PREFIX
xargs -P "${P4_PARALLEL:-4}" -L 1 bash -c 'run_one $0 $1 $2' < "$jobs_file"
echo "P4-SUITE-FINISHED"
