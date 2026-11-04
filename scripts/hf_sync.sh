#!/usr/bin/env bash
# Sync large artefacts to the private Hugging Face store (user decision, 4 Nov 2026).
#   hf_sync.sh push zoo|evals|runs     upload data/<kind> → Aralario/wsp-lab-<kind> (dataset, private)
#   hf_sync.sh pull zoo|evals|runs     download back into data/<kind>
# Excludes: HF cache, build dirs, *.pt refusal caches are included (small), full model checkpoints never.
set -euo pipefail
cd "$(dirname "$0")/.."; . scripts/env.sh
kind="${2:?zoo|evals|runs}"; repo="Aralario/wsp-lab-${kind}"
case "${1:-}" in
  push) hf upload "$repo" "data/$kind" "$kind" --repo-type dataset --exclude "**/build/**" --exclude "**/*.lock" --commit-message "sync $kind $(date -u +%FT%TZ) $(git rev-parse --short HEAD)" ;;
  pull) hf download "$repo" --repo-type dataset --local-dir "data/.hf-pull-$kind" && rsync -a "data/.hf-pull-$kind/$kind/" "data/$kind/" ;;
  *) echo "usage: hf_sync.sh push|pull zoo|evals|runs"; exit 1 ;;
esac
