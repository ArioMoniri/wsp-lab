#!/usr/bin/env python3
"""Capacity poller for WM-E Lambda jobs; launches through the dead-man terminator's guarded path only.

    wme_lambda_poll.py --dir <state> --name wsp-wme-... --types gpu_2x_h100_sxm5,gpu_1x_h100_sxm5 \
        [--fallback-after-min 30] [--late-types gpu_4x_h100_sxm5 --late-after-min 90] \
        --t-max-hours 3 --max-usd 30 --max-price 10

Types are tried in order; ``--fallback-after-min`` admits every type after the first, ``--late-types`` are admitted
only after ``--late-after-min``. Read-only API calls until a launch; ``Deadman.launch`` enforces the prefix, the
price, hour and dollar caps and one live instance. The key is read from the env file and never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsp.ops import lambda_deadman as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--types", required=True)
    ap.add_argument("--fallback-after-min", type=float, default=30)
    ap.add_argument("--late-types", default="")
    ap.add_argument("--late-after-min", type=float, default=1e9)
    ap.add_argument("--poll-s", type=float, default=30)
    ap.add_argument("--max-min", type=float, default=600)
    ap.add_argument("--t-max-hours", type=float, required=True)
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--max-price", type=float, required=True)
    ap.add_argument("--idle-limit-s", type=float, default=900)
    ap.add_argument("--ssh-key", default="wsp-lab-aws")
    a = ap.parse_args()
    client = D.LambdaClient(D.read_key(Path.home() / ".config/wsp/env"))
    pol = D.Policy(
        idle_limit_s=a.idle_limit_s,
        t_max_hours=a.t_max_hours,
        max_usd=a.max_usd,
        max_price_usd_h=a.max_price,
    )
    dm = D.Deadman(client, Path(a.dir), policy=pol)
    types = [t for t in a.types.split(",") if t]
    late = [t for t in a.late_types.split(",") if t]
    t0 = time.time()
    stamp = lambda: time.strftime("%H:%M:%S")
    while time.time() - t0 < a.max_min * 60:
        el = (time.time() - t0) / 60
        wanted = types[:1] + (types[1:] if el > a.fallback_after_min else [])
        wanted += late if el > a.late_after_min else []
        try:
            avail = client.instance_types()
        except (D.ApiError, OSError) as e:
            print(f"[{stamp()}] api error {client.redact(str(e))}", flush=True)
            time.sleep(a.poll_s)
            continue
        for itype in wanted:
            regions = [
                r["name"] for r in avail.get(itype, {}).get("regions_with_capacity_available", [])
            ]
            if not regions:
                continue
            region = "us-southeast-1" if "us-southeast-1" in regions else regions[0]
            print(f"[{stamp()}] CAPACITY {itype} {regions}; launching in {region}", flush=True)
            try:
                ids = dm.launch(itype, region, a.ssh_key, a.name)
            except (D.Refused, D.ApiError, OSError) as e:
                print(f"[{stamp()}] LAUNCH FAILED {client.redact(str(e))}", flush=True)
                time.sleep(15)
                continue
            print(f"[{stamp()}] LAUNCHED {json.dumps(ids)} {itype} {region}", flush=True)
            return 0
        print(f"[{stamp()}] no capacity for {wanted}", flush=True)
        time.sleep(a.poll_s)
    print(f"[{stamp()}] GAVE UP after {a.max_min} min", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
