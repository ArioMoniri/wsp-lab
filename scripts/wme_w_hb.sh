#!/usr/bin/env bash
# Stage-W heartbeat, sync and finish (AWS server, under agent-tmux). Every 60 s: reset the dead-man idle clock ONLY if
# the remote out/w/progress.jsonl grew within the last 30 min (a hung job therefore stops the heartbeat and the
# terminator's idle limit ends the instance). Every 10 min: pull out/w and logs. On STATUS DONE or FAILED*: final
# pull, sha256 verification (DONE), request-terminate, wait for the API-confirmed termination, write HB_STOP.
#   wme_w_hb.sh <ssh-alias> <state-dir> <remote-dir> <local-dir>
set -uo pipefail
ALIAS="${1:?alias}"; STATE="${2:?state}"; RDIR="${3:?remote dir}"; LOCAL="${4:?local dir}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${WME_SSH_CONFIG:?ssh config}"
DM=("$DIR/.venv/bin/python" "$DIR/scripts/lambda_deadman.py" --dir "$STATE")
SSH=(ssh -F "$CFG" -o ConnectTimeout=10 -o BatchMode=yes "$ALIAS")
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
pull() { mkdir -p "$LOCAL"; rsync -az -e "ssh -F $CFG -o BatchMode=yes" "$ALIAS:$RDIR/out/w/" "$LOCAL/w/" && \
         rsync -az -e "ssh -F $CFG -o BatchMode=yes" "$ALIAS:$RDIR/logs/" "$LOCAL/logs/" && \
         rsync -az -e "ssh -F $CFG -o BatchMode=yes" "$ALIAS:$RDIR/STATUS" "$LOCAL/STATUS"; }
n=0
while [ ! -f "$STATE/HB_STOP" ]; do
  n=$((n + 1))
  if "${SSH[@]}" "test -n \"\$(find ~/$RDIR/out/w/progress.jsonl -mmin -30 2>/dev/null)\"" 2>/dev/null; then
    "${DM[@]}" heartbeat && say "heartbeat (progress)"
  else
    say "no remote progress in 30 min (or unreachable): no heartbeat"
  fi
  st=$("${SSH[@]}" "cat ~/$RDIR/STATUS" 2>/dev/null || echo "?")
  if [ $((n % 10)) -eq 0 ]; then pull && say "pulled"; fi
  case "$st" in
    DONE|FAILED*)
      say "remote status $st: final pull"
      for _ in 1 2 3; do pull && break; sleep 20; done
      if [ "$st" = DONE ]; then
        ( cd "$LOCAL/w" && sha256sum -c --quiet SHA256SUMS ) && say "sha256 verified" || say "SHA256 MISMATCH"
      fi
      id=$("${DM[@]}" status | python3 -c 'import sys,json; l=json.load(sys.stdin)["live"]; print(l[0] if l else "")')
      [ -n "$id" ] && "${DM[@]}" request-terminate "$id" --reason "stage W $st" && say "terminate requested"
      for _ in $(seq 1 60); do
        live=$("${DM[@]}" status | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["live"]))')
        [ "$live" = 0 ] && { say "termination confirmed by the API"; break; }
        sleep 30
      done
      touch "$STATE/HB_STOP"
      ;;
  esac
  sleep 60
done
say "HB_STOP seen; exiting"
