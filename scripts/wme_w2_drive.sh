#!/usr/bin/env bash
# Post-launch driver for one stage-W2 slot (copy of wme_w_drive.sh) (AWS server, under agent-tmux). Waits for the
# slot terminator's live instance to be active, writes the ssh config outside the repo (the IP is never printed),
# pushes the bundle, starts the remote job with nohup and the queue deadline (launch + 3.10 h on a first launch, from
# the terminator's launch record; a relaunch gets the slot's remaining hours - 0.23 h), then starts the progress-driven heartbeat loop (wme_w2_hb.sh). Heartbeats are sent while waiting.
#   scripts/wme_w2_drive.sh <bundle.tgz> <code_sha> <alias> <state-dir> <remote-dir> <slot>
set -uo pipefail
BUNDLE="${1:?bundle}"; SHA="${2:?sha}"; ALIAS="${3:?alias}"; STATE="${4:?state}"; RDIR="${5:?remote dir}"; SLOT="${6:?slot}"
RS=scripts/wme_w2_remote.sh
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
# queue deadline = launch + (slot cap - hours of earlier instances of this slot - 0.23 h); 3.10 h on a first launch
DEADLINE=$("${DM[@]}" status | W2_T_MAX="${W2_T_MAX:-3.33}" python3 -c 'import sys,json,os
s=json.load(sys.stdin); i=s["live"][0]; ins=s["instances"]
prev=sum((r["confirmed_terminated_at"]-r["launched_at"])/3600 for r in ins.values() if r.get("confirmed_terminated_at"))
print(int(ins[i]["launched_at"] + (float(os.environ["W2_T_MAX"]) - prev - 0.23) * 3600))')
say "queue deadline set (slot cap minus earlier hours minus 0.23 h after launch)"
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
"${SSH[@]}" "tar -xzf $RDIR-bundle.tgz && mkdir -p $RDIR/logs $RDIR/out && cd $RDIR && (nohup bash $RS $SHA $SLOT $DEADLINE > logs/remote.log 2>&1 < /dev/null &) && sleep 2 && cat STATUS" \
  || { say "remote start failed"; exit 1; }
say "remote job started"
"${DM[@]}" heartbeat
agent-tmux "$(basename "$STATE")-hb" env WME_SSH_CONFIG="$CFG" bash "$DIR/scripts/wme_w2_hb.sh" "$ALIAS" "$STATE" "$RDIR" "$WME_LOCAL"
say "heartbeat loop started; driver done"
