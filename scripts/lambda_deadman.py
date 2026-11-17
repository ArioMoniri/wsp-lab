#!/usr/bin/env python3
"""CLI for the Lambda dead-man terminator (``wsp.ops.lambda_deadman``). Runs on the AWS server only.

    lambda_deadman.py watch                       # run under agent-tmux; polls every 30 s
    lambda_deadman.py launch --name wsp-wme-...   # guarded launch (prefix, price, capacity, hours, one live)
    lambda_deadman.py heartbeat                   # the driver is alive; resets the idle clock
    lambda_deadman.py request-terminate <id>      # the watcher terminates it on its next tick
    lambda_deadman.py status                      # hours, cost, live ids, watcher liveness (no IPs)
    lambda_deadman.py ssh-config <id> <alias>     # writes an ssh config outside the repo (IP never printed)

The key is read from ``LAMBDA_API_KEY`` or ``~/.config/wsp/env`` and is never printed. State and logs live in
``~/.local/state/wsp-wme-deadman/`` (outside the repository).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsp.ops import lambda_deadman as D

STATE_DIR = Path.home() / ".local/state/wsp-wme-deadman"
ENV_FILE = Path.home() / ".config/wsp/env"
SSH_CONFIG = Path(os.environ.get("WME_SSH_CONFIG", Path.home() / ".config/wsp/wme_ssh_config"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(STATE_DIR))
    ap.add_argument("--idle-limit-s", type=float, default=900)
    ap.add_argument("--t-max-hours", type=float, default=3.0)
    ap.add_argument("--max-usd", type=float, default=20.0)
    ap.add_argument("--max-price", type=float, default=5.5)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("watch").add_argument("--poll-s", type=float, default=30)
    la = sub.add_parser("launch")
    la.add_argument("--name", required=True)
    la.add_argument("--type", default="gpu_1x_h100_sxm5")
    la.add_argument("--region", default="us-southeast-1")
    la.add_argument("--ssh-key", default="wsp-lab-aws")
    sub.add_parser("heartbeat")
    rt = sub.add_parser("request-terminate")
    rt.add_argument("id")
    rt.add_argument("--reason", default="calibration finished")
    sub.add_parser("status")
    sc = sub.add_parser("ssh-config")
    sc.add_argument("id")
    sc.add_argument("alias")
    a = ap.parse_args()

    policy = D.Policy(
        idle_limit_s=a.idle_limit_s,
        t_max_hours=a.t_max_hours,
        max_usd=a.max_usd,
        max_price_usd_h=a.max_price,
    )
    client = D.LambdaClient(D.read_key(ENV_FILE))
    dm = D.Deadman(client, Path(a.dir), policy=policy)
    if a.cmd == "watch":
        dm.run(poll_s=a.poll_s)
    elif a.cmd == "launch":
        ids = dm.launch(a.type, a.region, a.ssh_key, a.name)
        print(json.dumps({"ids": ids}))
    elif a.cmd == "heartbeat":
        dm.heartbeat()
    elif a.cmd == "request-terminate":
        dm.request_terminate(a.id, a.reason)
    elif a.cmd == "status":
        st = dm.state()
        print(
            json.dumps(
                {
                    "live": dm.live(st),
                    "instances": {
                        i: {
                            k: r.get(k)
                            for k in (
                                "name",
                                "type",
                                "region",
                                "launched_at",
                                "confirmed_terminated_at",
                                "terminate_reasons",
                            )
                        }
                        for i, r in st["instances"].items()
                    },
                    "instance_hours": round(dm.instance_hours(st), 4),
                    "cost_usd": round(dm.cost_usd(st), 3),
                    "watcher_last_tick_age_s": round(time.time() - st.get("last_tick", 0), 1),
                    "alert": dm.alert_path.exists(),
                },
                indent=1,
            )
        )
    elif a.cmd == "ssh-config":
        if a.id not in dm.state()["instances"]:
            raise D.Refused(f"{a.id} is not ours")
        info = client.get_instance(a.id)
        ip = (info or {}).get("ip")
        if not ip:
            print(json.dumps({"status": (info or {}).get("status"), "ip": None}))
            return 1
        SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        block = (
            f"Host {a.alias}\n  HostName {ip}\n  User ubuntu\n  IdentityFile ~/.ssh/id_ed25519\n"
            "  StrictHostKeyChecking accept-new\n  UserKnownHostsFile ~/.config/wsp/wme_known_hosts\n"
            "  ServerAliveInterval 30\n  ServerAliveCountMax 6\n"
        )
        # upsert: keep other aliases' blocks (several wsp-wme-* instances may be live, each in its own state dir)
        old = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else ""
        SSH_CONFIG.write_text(D.upsert_host_block(old, a.alias, block))
        os.chmod(SSH_CONFIG, 0o600)
        print(json.dumps({"status": info.get("status"), "ssh_config": str(SSH_CONFIG)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
