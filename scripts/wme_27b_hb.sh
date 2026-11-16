#!/usr/bin/env bash
# Progress-driven heartbeat for the Lambda dead-man terminator (runs on the AWS server under agent-tmux).
# Every 60 s: if the instance's logs changed in the last 10 min, reset the terminator's idle clock.
# When the remote job stops making progress (finished, hung or unreachable), heartbeats stop and the
# terminator's idle limit terminates the instance, even if no agent is watching.
#   agent-tmux wsp-wme-hb scripts/wme_27b_hb.sh <ssh-alias>      # stop: touch <state>/HB_STOP
set -uo pipefail
ALIAS="${1:?ssh alias}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$HOME/.local/state/wsp-wme-deadman"
CFG="$HOME/.config/wsp/wme_ssh_config"
while [ ! -f "$STATE/HB_STOP" ]; do
  if ssh -F "$CFG" -o ConnectTimeout=10 -o BatchMode=yes "$ALIAS" \
      'test -n "$(find ~/wme27/logs -type f -mmin -10 2>/dev/null | head -1)"' 2>/dev/null; then
    "$DIR/.venv/bin/python" "$DIR/scripts/lambda_deadman.py" heartbeat && echo "[$(date -u +%H:%M:%S)] heartbeat"
  else
    echo "[$(date -u +%H:%M:%S)] no remote progress"
  fi
  sleep 60
done
echo "HB_STOP seen; exiting"
