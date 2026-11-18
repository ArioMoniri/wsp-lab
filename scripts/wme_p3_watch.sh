#!/usr/bin/env bash
# Runs ON the Lambda instance next to a pilot-3 job (amendment 2(d)): while the job's process lives, refresh
# logs/alive only on progress (the stage ledger changed in the last 10 minutes, or the GPU is busy > 5 %), so a hung
# job stops the AWS-side heartbeat and the dead-man idle limit ends the instance. Writes the final STATUS.
#   bash scripts/wme_p3_watch.sh <pid> <stage s|h>
PID="${1:?pid}"; ST="${2:?stage}"
cd "$(dirname "$0")/.."
while kill -0 "$PID" 2>/dev/null; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ -n "$(find out/$ST/ledger.jsonl -mmin -10 2>/dev/null)" ] || [ "${util:-0}" -gt 5 ]; then touch logs/alive; fi
  sleep 60
done
wait "$PID" 2>/dev/null; rc=$?
st=$(cat out/$ST/STATUS 2>/dev/null || echo "FAILED:run")
[ "$st" = RUNNING ] && st="FAILED:run"
echo "[$(date -u +%H:%M:%S)] process $PID ended; status=$st" >> logs/remote.log
echo "$st" > STATUS
