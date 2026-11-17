#!/usr/bin/env bash
# AWS side: launch one WM-E pilot-3 stage on Lambda through the dead-man terminator (EXPLORATORY).
#   [LABEL=s2] scripts/wme_p3_launch.sh s|h <t_max_hours> <max_usd>   (LABEL names the instance/state; default = stage)
# Starts, each under agent-tmux: the watcher (own state dir ~/.local/state/wsp-wme-p3-<stage>), the capacity poller
# (guarded launch of wsp-wme-p3-<stage>-20261117) and the driver (bundle push, remote start, heartbeat loop).
# Each stage has its own ssh config file outside the repo, so no other track's config is touched.
set -euo pipefail
ST="${1:?s|h}"; TMAX="${2:?hours}"; USD="${3:?usd}"; LB="${LABEL:-$ST}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$DIR"
STATE="$HOME/.local/state/wsp-wme-p3-$LB"; mkdir -p "$STATE"
export WME_SSH_CONFIG="$HOME/.config/wsp/wme_p3_${LB}_ssh_config"
SHA=$(git rev-parse --short HEAD)
BUNDLE="/tmp/wme-p3-$LB-$SHA.tgz"
scripts/wme_p3_bundle.sh "$BUNDLE" > /dev/null
POL=(--dir "$STATE" --t-max-hours "$TMAX" --max-usd "$USD" --max-price 5.5 --idle-limit-s 1200)
PY="$DIR/.venv/bin/python"
agent-tmux "wme-p3-$LB-dm" "$PY" scripts/lambda_deadman.py "${POL[@]}" watch
agent-tmux "wme-p3-$LB-poll" "$PY" scripts/wme_lambda_poll.py --dir "$STATE" --name "wsp-wme-p3-$LB-20261117" \
  --types "${TYPES:-gpu_1x_h100_sxm5,gpu_1x_h100_pcie}" --fallback-after-min "${FALLBACK_MIN:-0}" \
  --t-max-hours "$TMAX" --max-usd "$USD" \
  --max-price 5.5 --idle-limit-s 1200
agent-tmux "wme-p3-$LB-drive" env WME_SSH_CONFIG="$WME_SSH_CONFIG" bash scripts/wme_lambda_drive.sh "$BUNDLE" "$SHA" \
  "wmep3$LB" "$STATE" wmep3 scripts/wme_p3_remote.sh "$ST"
echo "launched stage $ST as $LB (code $SHA); state $STATE"
