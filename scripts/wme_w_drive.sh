#!/usr/bin/env bash
# Post-launch driver for the WM-E stage-W Lambda job (copy of wme_lambda_drive.sh; starts wme_w_hb.sh) (AWS server, under agent-tmux). Waits for the terminator's live instance
# to be active, writes the ssh config outside the repo (the IP is never printed), pushes the bundle, starts the
# remote job with nohup, then starts the progress-driven heartbeat loop. Heartbeats are sent while waiting for boot.
#   scripts/wme_lambda_drive.sh <bundle.tgz> <code_sha> <alias> <state-dir> <remote-dir> <remote-script> [args...]
set -uo pipefail
BUNDLE="${1:?bundle}"; SHA="${2:?sha}"; ALIAS="${3:?alias}"; STATE="${4:?state}"; RDIR="${5:?remote dir}"; RS="${6:?remote script}"
shift 6
DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$DIR"
DM=(".venv/bin/python" "scripts/lambda_deadman.py" --dir "$STATE")
CFG="${WME_SSH_CONFIG:-$HOME/.config/wsp/wme_ssh_config}"
SSH=(ssh -F "$CFG" -o ConnectTimeout=10 -o BatchMode=yes "$ALIAS")
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
id=""
for _ in $(seq 1 2400); do
  id=$("${DM[@]}" status | python3 -c 'import sys,json; l=json.load(sys.stdin)["live"]; print(l[0] if l else "")')
  [ -n "$id" ] && break; sleep 15
done
[ -n "$id" ] || { say "no live instance"; exit 1; }
say "live instance registered"
for _ in $(seq 1 80); do
  "${DM[@]}" heartbeat
  out=$("${DM[@]}" ssh-config "$id" "$ALIAS" 2>&1) && echo "$out" | grep -q '"status": "active"' && break
  st=$(echo "$out" | python3 -c 'import sys,json
try: print(json.loads(sys.stdin.read().strip().splitlines()[-1]).get("status"))
except Exception: print("?")')
  say "waiting for active: $st"
  case "$st" in terminat*) say "instance terminated by the provider during boot"; exit 3 ;; esac
  sleep 15
done
for _ in $(seq 1 40); do
  "${DM[@]}" heartbeat
  "${SSH[@]}" true 2>/dev/null && break
  sleep 15
done
"${SSH[@]}" true || { say "ssh never came up"; exit 1; }
say "ssh up"
"${DM[@]}" heartbeat
scp -F "$CFG" -o BatchMode=yes -q "$BUNDLE" "$ALIAS:$RDIR-bundle.tgz" || { say "scp failed"; exit 1; }
"${SSH[@]}" "tar -xzf $RDIR-bundle.tgz && mkdir -p $RDIR/logs $RDIR/out && cd $RDIR && (nohup bash $RS $SHA $* > logs/remote.log 2>&1 < /dev/null &) && sleep 2 && cat STATUS" \
  || { say "remote start failed"; exit 1; }
say "remote job started"
"${DM[@]}" heartbeat
agent-tmux "$(basename "$STATE")-hb" env WME_SSH_CONFIG="$CFG" bash "$DIR/scripts/wme_w_hb.sh" "$ALIAS" "$STATE" "$RDIR" "$WME_LOCAL"
say "heartbeat loop started; driver done"
