#!/usr/bin/env bash
# AWS side: launch stage W on one Lambda gpu_1x_h100_sxm5 through the dead-man terminator (RULING-wme-stagew §1).
#   scripts/wme_w_launch.sh [suffix]      (suffix -k for a relaunch: wsp-wme-w-main-20261118-k)
# Under agent-tmux: the watcher (state ~/.local/state/wsp-wme-w; t_max 4.7 cumulative instance-h, max $30, idle
# 1,800 s), the capacity poller (SXM5 only, guarded launch) and the driver (bundle push, remote start, heartbeat).
set -euo pipefail
SUF="${1:-}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$DIR"
STATE="$HOME/.local/state/wsp-wme-w"; mkdir -p "$STATE"
export WME_SSH_CONFIG="$HOME/.config/wsp/wme_w_ssh_config"
export WME_LOCAL="$DIR/data/wme/stagew/run-20261118/remote"
SHA=$(git rev-parse --short HEAD)
BUNDLE="/tmp/wme-w-$SHA.tgz"
scripts/wme_w_bundle.sh "$BUNDLE" > /dev/null
POL=(--dir "$STATE" --t-max-hours 4.7 --max-usd 30 --max-price 5.5 --idle-limit-s 1800)
PY="$DIR/.venv/bin/python"
pgrep -f "lambda_deadman.py --dir $STATE .* watch" >/dev/null || agent-tmux "wme-w-dm" "$PY" scripts/lambda_deadman.py "${POL[@]}" watch
agent-tmux "wme-w-poll$SUF" "$PY" scripts/wme_lambda_poll.py --dir "$STATE" --name "wsp-wme-w-main-20261118$SUF" \
  --types gpu_1x_h100_sxm5 --fallback-after-min 1e9 --t-max-hours 4.7 --max-usd 30 --max-price 5.5 --idle-limit-s 1800
agent-tmux "wme-w-drive$SUF" env WME_SSH_CONFIG="$WME_SSH_CONFIG" WME_LOCAL="$WME_LOCAL" bash scripts/wme_w_drive.sh \
  "$BUNDLE" "$SHA" "wmew" "$STATE" wmew scripts/wme_w_remote.sh
echo "launched stage W (code $SHA); state $STATE"
