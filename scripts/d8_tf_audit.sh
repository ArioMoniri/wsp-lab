#!/usr/bin/env bash
# D8 step 5: TF-v1 fingerprint audit + G1 + G2 at the frozen source a224779bf873.
# Batch evaluations run from the clean detached worktree at dd2a609 (Board A's named commit). The report/bridge scripts run
# from regen-9922ea3 (the round-3 driver commit: src/wsp byte-identical, source_hash a224779bf873), because the G2 bridge
# there scores HellaSwag through the batched path as in the original round 3.
set -euo pipefail
W=/home/ubuntu/projects/wsp-lab/.worktrees/regen-dd2a609
R=/home/ubuntu/projects/wsp-lab/.worktrees/regen-9922ea3
PY=/home/ubuntu/projects/wsp-lab/.venv/bin/python
EV=/home/ubuntu/wsp-data/evals/qwen3.5-0.8b
G1="2f61a4086395b719 b5d0ac5d76c3cad6 d7f933d3b4e5d25e c3ae3e8c8edca6b8 0027efdf2eae7405 00f8111cbad8218f 057dcfbedea3251f 023b384a7799a787 02a1b5e21971290f 5c766ee2fa8bf915 34ee195e55227908"  # 13 minus the two lost chain predecessors b7edbb9f8b339f52, 86ea1fe0ec777cf8
unset HF_HOME
export TOKENIZERS_PARALLELISM=false
for t in "$W" "$R"; do [ -z "$(git -C $t status --porcelain)" ] || { echo "$t not clean" >&2; exit 1; }; done
[ ! -e "$EV/tf_v1" ] && [ ! -e "$EV/base/tf_cache" ] || { echo "tf_v1 or tf_cache exists; refusing" >&2; exit 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU not exclusive" >&2; exit 1; } || true
B="$PY -m wsp.eval.batch --zoo data/zoo/qwen3.5-0.8b --config configs/eval_tf_v1.yaml"
cd "$W"; export PYTHONPATH="$W/src"
$PY -c "import wsp; print('wsp from', wsp.__file__)"
echo "=== pilot (base + 10 controls; KL cache rebuild) $(date -u +%FT%TZ)"
$B --ids $(cat data/runs/d3_shards/controls.txt)
echo "=== G1 primary $(date -u +%FT%TZ)"
$B --ids $G1
echo "=== G1 repeat $(date -u +%FT%TZ)"
$B --repeat g1 --ids $G1
cd "$R"; export PYTHONPATH="$W/src"
echo "=== G1 report $(date -u +%FT%TZ)"
$PY scripts/tf_v1_g1_report.py
echo "=== G2 $(date -u +%FT%TZ)"
$PY scripts/tf_v1_g2_bridge.py
echo "=== TF-AUDIT-DONE $(date -u +%FT%TZ)"
