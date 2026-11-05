#!/usr/bin/env bash
# TF-v1 round 3 (PREREG A2.4): archive round 2, re-pilot, gate, G1 (primary + repeat), G2.
# Stops at the first failing step (set -e); the round-3 gate check exits non-zero if the warm median
# exceeds 1.40 min or any unchanged path is not bitwise-identical to run 20261105-144427-fb58ffe6.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
EV=data/evals/qwen3.5-0.8b
if [ -d "$EV/tf_v1" ] && [ ! -d "$EV/tf_v1_round2_archive" ]; then mv "$EV/tf_v1" "$EV/tf_v1_round2_archive"; fi
B="python -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml"
G1="2f61a4086395b719 b5d0ac5d76c3cad6 d7f933d3b4e5d25e c3ae3e8c8edca6b8 0027efdf2eae7405 00f8111cbad8218f 057dcfbedea3251f 023b384a7799a787 02a1b5e21971290f 5c766ee2fa8bf915 b7edbb9f8b339f52 34ee195e55227908 86ea1fe0ec777cf8"
echo "=== pilot $(date -u +%FT%TZ)"
$B --ids $(cat data/runs/d3_shards/controls.txt)
python scripts/tf_v1_round3_check.py
echo "=== G1 primary $(date -u +%FT%TZ)"
$B --ids $G1
echo "=== G1 repeat $(date -u +%FT%TZ)"
$B --repeat g1 --ids $G1
python scripts/tf_v1_g1_report.py
echo "=== G2 $(date -u +%FT%TZ)"
python scripts/tf_v1_g2_bridge.py
echo "=== done $(date -u +%FT%TZ)"
