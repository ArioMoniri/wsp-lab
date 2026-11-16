#!/usr/bin/env python3
"""Polls Lambda capacity and launches the calibration instance through the dead-man terminator's guarded path.

Preference: 1x H100 SXM5 (any region, us-southeast-1 first). After ``--fallback-after-min`` minutes, 1x H100 PCIe
(also 80 GB; a reported deviation). Never anything below 80 GB. Gives up after ``--max-min`` minutes. Read-only
API calls until a launch; the launch itself is ``Deadman.launch`` (prefix, price, hours, one-live guards).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsp.ops import lambda_deadman as D

STATE_DIR = Path.home() / ".local/state/wsp-wme-deadman"
PRIMARY, FALLBACK = "gpu_1x_h100_sxm5", "gpu_1x_h100_pcie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--poll-s", type=float, default=30)
    ap.add_argument("--fallback-after-min", type=float, default=45)
    ap.add_argument("--max-min", type=float, default=240)
    a = ap.parse_args()
    client = D.LambdaClient(D.read_key(Path.home() / ".config/wsp/env"))
    dm = D.Deadman(
        client, STATE_DIR, policy=D.Policy(t_max_hours=3.0, max_usd=20.0, max_price_usd_h=5.5)
    )
    t0 = time.time()
    while time.time() - t0 < a.max_min * 60:
        try:
            types = client.instance_types()
        except (D.ApiError, OSError) as e:
            print(f"[{time.strftime('%H:%M:%S')}] api error {client.redact(str(e))}", flush=True)
            time.sleep(a.poll_s)
            continue
        wanted = [PRIMARY] + ([FALLBACK] if time.time() - t0 > a.fallback_after_min * 60 else [])
        for itype in wanted:
            regions = [
                r["name"] for r in types.get(itype, {}).get("regions_with_capacity_available", [])
            ]
            if regions:
                region = "us-southeast-1" if "us-southeast-1" in regions else regions[0]
                print(
                    f"[{time.strftime('%H:%M:%S')}] CAPACITY {itype} {regions}; launching in {region}",
                    flush=True,
                )
                try:
                    ids = dm.launch(itype, region, "wsp-lab-aws", a.name)
                except (D.Refused, D.ApiError, OSError) as e:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] LAUNCH FAILED {client.redact(str(e))}",
                        flush=True,
                    )
                    time.sleep(15)
                    continue
                print(
                    f"[{time.strftime('%H:%M:%S')}] LAUNCHED {json.dumps(ids)} {itype} {region}",
                    flush=True,
                )
                return 0
        print(f"[{time.strftime('%H:%M:%S')}] no capacity for {wanted}", flush=True)
        time.sleep(a.poll_s)
    print(f"[{time.strftime('%H:%M:%S')}] GAVE UP after {a.max_min} min", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
