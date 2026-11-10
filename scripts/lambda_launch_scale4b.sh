#!/usr/bin/env bash
# Launcher (runs HERE): starts one Lambda H100 PCIe, waits for it, copies the env file and the F5.3 job, starts the job with
# the pinned repo revision and a private device id, and leaves. The job self-terminates at its cap (scripts/lambda_job_scale4b.sh).
# Allowed only after Board A's freeze of A5 with F5 (accepted 10 Nov 2026, docs/reviews/council/board-a-freeze-reaudit-2026-11-10.md).
set -euo pipefail
set -a; source ~/.config/wsp/env; set +a
TYPE=${TYPE:-gpu_1x_h100_pcie}; REGION=${REGION:-us-west-3}; KEY=${KEY:-wsp-lab-aws}
WSP_REV=${WSP_REV:?full main sha}; DEVICE_ID=${DEVICE_ID:-H3}
API=https://cloud.lambda.ai/api/v1; H=(-H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json")
date -u +"start %FT%TZ"
out=$(MAX_USD_PER_H=${MAX_USD_PER_H:-3.5} scripts/lambda.sh launch "$TYPE" "$REGION" "$KEY"); echo "$out"
ID=$(echo "$out" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["instance_ids"][0])')
echo "instance $ID"
for i in $(seq 1 60); do
  sleep 20
  J=$(curl -s "${H[@]}" "$API/instances/$ID")
  ST=$(echo "$J" | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(d["status"], d.get("ip") or "")')
  echo "$(date -u +%T) $ST"; set -- $ST
  [ "$1" = "active" ] && [ -n "${2:-}" ] && IP=$2 && break
done
: "${IP:?instance did not become active}"
for i in $(seq 1 30); do ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes ubuntu@"$IP" true 2>/dev/null && break; sleep 10; done
ssh -o StrictHostKeyChecking=accept-new ubuntu@"$IP" "mkdir -p ~/.config/wsp && chmod 700 ~/.config/wsp"
scp -q -o StrictHostKeyChecking=accept-new ~/.config/wsp/env ubuntu@"$IP":~/.config/wsp/env
ssh ubuntu@"$IP" "chmod 600 ~/.config/wsp/env"
scp -q scripts/lambda_job_scale4b.sh ubuntu@"$IP":~/lambda_job_scale4b.sh
LAUNCH_EPOCH=$(date -u +%s)
ssh ubuntu@"$IP" "WSP_REV=$WSP_REV DEVICE_ID=$DEVICE_ID LAUNCH_EPOCH=$LAUNCH_EPOCH nohup bash ~/lambda_job_scale4b.sh > ~/scale4b.log 2>&1 &"
echo "job started on $ID ($IP) rev $WSP_REV device $DEVICE_ID epoch $LAUNCH_EPOCH"
echo "| $(date -u +%FT%TZ) | (running) | $ID | $TYPE ($REGION) | 3.29 | ≤2.0 cap | ≤6.58 | F5.3 4B scale diagnostic (WikiText, 20 Gaussian edits + FP4 pairing); self-terminates |" >> board/lambda_spend.md
date -u +"end %FT%TZ"
