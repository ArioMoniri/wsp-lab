#!/usr/bin/env bash
# wme-c1 heartbeat (AWS side, under agent-tmux): every 60 s reset the dead-man idle clock ONLY if the worker's
# progress.jsonl grew within the last 30 min (§II.13 item 3); a hung worker therefore lets the terminator end the
# instance after the 1,800 s idle limit. Stops when the dead-man has no live instance.
#   wme_c1_hb.sh <ssh-alias> <deadman-state> <remote worker dir>
set -uo pipefail
ALIAS="${1:?alias}"; STATE="${2:?state}"; RW="${3:?remote dir}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
DM=("$DIR/.venv/bin/python" "$DIR/scripts/lambda_deadman.py" --dir "$STATE")
CFG="$HOME/.config/wsp/wme_c1_ssh_config"
while true; do
  live=$("${DM[@]}" status | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["live"]))')
  [ "$live" = 0 ] && { echo "[$(date -u +%H:%M:%S)] no live instance; heartbeat loop ends"; exit 0; }
  if ssh -F "$CFG" -o ConnectTimeout=10 -o BatchMode=yes "$ALIAS" "test -n \"\$(find $RW/progress.jsonl -mmin -30 2>/dev/null)\"" 2>/dev/null; then
    "${DM[@]}" heartbeat && echo "[$(date -u +%H:%M:%S)] heartbeat (progress)"
  else
    echo "[$(date -u +%H:%M:%S)] no worker progress in 30 min (or unreachable): no heartbeat"
  fi
  sleep 60
done
