#!/usr/bin/env bash
# AWS side: launch one stage-W2 slot (a, b or c) on a Lambda gpu_1x_h100_sxm5 through its own dead-man terminator
# (RULING-wme-stagew2 §6).   scripts/wme_w2_launch.sh <slot> <bundle.tgz> <code_sha> [suffix]   (suffix -k: relaunch)
# Under agent-tmux: the slot's watcher (state ~/.local/state/wsp-wme-w2-<slot>; cumulative t_max 3.33 instance-h,
# max $19.90, idle 1,800 s; three slots <= 9.99 h and <= $59.70), the capacity poller (SXM5 only, guarded launch,
# name wsp-wme-w2-<slot>-20261118[suffix]) and the driver (bundle push, remote start with the queue deadline
# launch + 3.10 h, progress-driven heartbeat).
set -euo pipefail
SLOT="${1:?slot}"; BUNDLE="${2:?bundle}"; SHA="${3:?sha}"; SUF="${4:-}"
case "$SLOT" in a|b|c) ;; *) echo "slot must be a, b or c" >&2; exit 2 ;; esac
DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$DIR"
STATE="$HOME/.local/state/wsp-wme-w2-$SLOT"; mkdir -p "$STATE"
export WME_SSH_CONFIG="$HOME/.config/wsp/wme_w2_${SLOT}_ssh_config"
export WME_LOCAL="$DIR/data/wme/stagew2/run-20261118/$SLOT"
POL=(--dir "$STATE" --t-max-hours "${W2_T_MAX:-3.33}" --max-usd "${W2_MAX_USD:-19.9}" --max-price 5.5 --idle-limit-s 1800)
PY="$DIR/.venv/bin/python"
pgrep -f "lambda_deadman.py --dir $STATE .* watch" >/dev/null || agent-tmux "wme-w2-$SLOT-dm" "$PY" scripts/lambda_deadman.py "${POL[@]}" watch
agent-tmux "wme-w2-$SLOT-poll$SUF" "$PY" scripts/wme_lambda_poll.py --dir "$STATE" --name "wsp-wme-w2-$SLOT-20261118$SUF" \
  --types gpu_1x_h100_sxm5 --fallback-after-min 1e9 --t-max-hours "${W2_T_MAX:-3.33}" --max-usd "${W2_MAX_USD:-19.9}" \
  --max-price 5.5 --idle-limit-s 1800
agent-tmux "wme-w2-$SLOT-drive$SUF" env WME_SSH_CONFIG="$WME_SSH_CONFIG" WME_LOCAL="$WME_LOCAL" W2_T_MAX="${W2_T_MAX:-3.33}" \
  bash scripts/wme_w2_drive.sh \
  "$BUNDLE" "$SHA" "wmew2$SLOT" "$STATE" wmew2 "$SLOT"
echo "launched stage-W2 slot $SLOT (code $SHA); state $STATE"
