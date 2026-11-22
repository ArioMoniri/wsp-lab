#!/usr/bin/env bash
# wme-c1 heartbeat (AWS side, under agent-tmux): every 60 s reset the dead-man idle clock ONLY if the worker's
# progress.jsonl grew within the last 30 min (§II.13 item 3); a hung worker therefore lets the terminator end the
# instance after the 1,800 s idle limit. Stops when the dead-man has no live instance.
#   wme_c1_hb.sh <ssh-alias> <deadman-state> <remote worker dir>
# RULING-wme-c1-bank-halt addendum 5: it fails loudly (exit 3) when its Python is missing and (exit 4) when the
# dead-man status cannot be read, instead of looping silently without heartbeats; each progress heartbeat is
# recorded in <deadman-state>/heartbeats.jsonl (the controller requires two before any request). WME_HB_PYTHON
# overrides the interpreter (tests only).
set -uo pipefail
ALIAS="${1:?alias}"; STATE="${2:?state}"; RW="${3:?remote dir}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${WME_HB_PYTHON:-$DIR/.venv/bin/python}"
DM=("$PY" "$DIR/scripts/lambda_deadman.py" --dir "$STATE")
CFG="$HOME/.config/wsp/wme_c1_ssh_config"
fatal() { local rc=$1; shift; echo "[$(date -u +%H:%M:%S)] FATAL: $*" >&2; echo "[$(date -u +%H:%M:%S)] FATAL: $*"; exit "$rc"; }
while true; do
  [ -x "$PY" ] || fatal 3 "heartbeat interpreter $PY is missing (create the worktree's .venv link)"
  out=$("${DM[@]}" status) || fatal 4 "lambda_deadman.py status failed (rc $?)"
  live=$(printf '%s' "$out" | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["live"]))') \
    || fatal 4 "lambda_deadman.py status output is not readable"
  [ "$live" = 0 ] && { echo "[$(date -u +%H:%M:%S)] no live instance; heartbeat loop ends"; exit 0; }
  if ssh -F "$CFG" -o ConnectTimeout=10 -o BatchMode=yes "$ALIAS" "test -n \"\$(find $RW/progress.jsonl -mmin -30 2>/dev/null)\"" 2>/dev/null; then
    if "${DM[@]}" heartbeat; then
      printf '{"t": %s, "utc": "%s", "alias": "%s", "source": "progress"}\n' "$(date +%s.%N)" "$(date -u +%FT%TZ)" "$ALIAS" >> "$STATE/heartbeats.jsonl"
      echo "[$(date -u +%H:%M:%S)] heartbeat (progress)"
    else
      fatal 4 "lambda_deadman.py heartbeat failed"
    fi
  else
    echo "[$(date -u +%H:%M:%S)] no worker progress in 30 min (or unreachable): no heartbeat"
  fi
  sleep 60
done
