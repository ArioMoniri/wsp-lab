#!/usr/bin/env bash
# Progress-driven heartbeat for a WM-E Lambda job (AWS server, under agent-tmux). Every 60 s: if any file under the
# remote job's logs/ or out/**/progress changed in the last 10 min, reset the dead-man terminator's idle clock.
# When the remote job stops making progress (finished, hung or unreachable), heartbeats stop and the terminator's
# idle limit ends the instance even if no agent is watching. Stop: touch <state>/HB_STOP.
#   agent-tmux <name> scripts/wme_lambda_hb.sh <ssh-alias> <state-dir> <remote-dir>
set -uo pipefail
ALIAS="${1:?ssh alias}"; STATE="${2:?state dir}"; RDIR="${3:?remote dir}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$HOME/.config/wsp/wme_ssh_config"
while [ ! -f "$STATE/HB_STOP" ]; do
  if ssh -F "$CFG" -o ConnectTimeout=10 -o BatchMode=yes "$ALIAS" \
      "test -n \"\$(find ~/$RDIR/logs ~/$RDIR/out -type f -mmin -10 2>/dev/null | head -1)\"" 2>/dev/null; then
    "$DIR/.venv/bin/python" "$DIR/scripts/lambda_deadman.py" --dir "$STATE" heartbeat && echo "[$(date -u +%H:%M:%S)] heartbeat"
  else
    echo "[$(date -u +%H:%M:%S)] no remote progress"
  fi
  sleep 60
done
echo "HB_STOP seen; exiting"
