#!/usr/bin/env bash
# Lambda Cloud helper. Needs LAMBDA_API_KEY. lambda.sh types|ls|launch <type> <region> <ssh-key-name>|terminate <id>
set -euo pipefail
: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY in ~/.config/wsp/env}"
API=https://cloud.lambda.ai/api/v1; H=(-H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json")
case "${1:-}" in
  types) curl -s "${H[@]}" "$API/instance-types" | python3 -c 'import sys,json
d=json.load(sys.stdin)["data"]
for k,v in sorted(d.items(), key=lambda kv: kv[1]["instance_type"]["price_cents_per_hour"]):
    t=v["instance_type"]; r=[x["name"] for x in v["regions_with_capacity_available"]]
    print("%-28s $%6.2f/h %sxGPU regions=%s" % (k, t["price_cents_per_hour"]/100, t["specs"]["gpus"], r))' ;;
  ls) curl -s "${H[@]}" "$API/instances" | python3 -c 'import sys,json; [print(i["id"],i["name"],i["instance_type"]["name"],i["status"],i["region"]["name"]) for i in json.load(sys.stdin)["data"]]' ;;
  launch) : "${MAX_USD_PER_H:?set MAX_USD_PER_H as a hard price cap}"
     price=$(curl -s "${H[@]}" "$API/instance-types" | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['$2']['instance_type']['price_cents_per_hour']/100)")
     python3 -c "import sys; sys.exit(0 if $price <= $MAX_USD_PER_H else 1)" || { echo "price $price > cap $MAX_USD_PER_H"; exit 1; }
     curl -s "${H[@]}" -X POST "$API/instance-operations/launch" -d "{\"region_name\":\"$3\",\"instance_type_name\":\"$2\",\"ssh_key_names\":[\"$4\"],\"quantity\":1,\"name\":\"wsp-lab\"}" ;;
  terminate) curl -s "${H[@]}" "$API/instances/$2" | grep -q "\"name\":\"wsp-lab" || { echo "refusing: instance $2 is not a wsp-lab instance"; exit 1; }; curl -s "${H[@]}" -X POST "$API/instance-operations/terminate" -d "{\"instance_ids\":[\"$2\"]}" ;;
  *) echo "usage: lambda.sh types|ls|launch <type> <region> <sshkey>|terminate <id>"; exit 1 ;;
esac
