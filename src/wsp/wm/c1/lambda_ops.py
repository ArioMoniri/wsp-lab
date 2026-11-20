"""Controller-side Lambda operations for wme-c1 (engineering; PREREG_WME §II.13 items 1-3, §II.28.2-§II.28.3).

* One dead-man state directory for the whole 9B study (``~/.local/state/wsp-wme-c1``): every study instance is
  launched through ``wsp.ops.lambda_deadman.Deadman.launch`` (prefix ``wsp-wme-``, our names ``wsp-wme-c1-*``, one
  live instance), terminated only by registered id and confirmed through the API.
* **Allowance (chair decision 2).** Before a launch the policy is tightened so that the dead-man's cumulative dollar
  limit leaves exactly the remaining combined budget: ``max_usd = spent_usd(deadman) + remaining_gpu_h x
  price_per_gpu_h`` (the guard's ``remaining_usd()`` converted from the 4.89 x 1.10 ledger convention to the
  provider price, so the dead-man's GPU-hour limit equals the cap's remaining GPU-hours). ``t_max_hours`` is the
  priced remaining phases x 1.25 (§II.13 item 2), never above the cap. Policies only tighten.
* **Authoritative time.** ``authoritative(iid)`` returns (launched_at, terminated_at) from the dead-man ledger
  (``confirmed_terminated_at``) for ``ProductionGuard.reconcile_instance``.
* Instance ids enter the guard as ``lambda:<id>`` (unique physical identity, §II.28.3 item 5); the worker's GPU UUID
  and machine-id digests are recorded by the ``hello`` request.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from wsp.ops import lambda_deadman as D
from wsp.wm import runguard as RG

STATE = Path.home() / ".local/state/wsp-wme-c1"
ENV_FILE = Path.home() / ".config/wsp/env"
SSH_CONFIG = Path.home() / ".config/wsp/wme_c1_ssh_config"
TYPE = "gpu_1x_h100_sxm5"
TYPE2 = "gpu_2x_h100_sxm5"
PRICE = {TYPE: 4.89, TYPE2: 9.58}
GPUS = {TYPE: 1, TYPE2: 2}
NAME_PREFIX = "wsp-wme-c1-"


def deadman(policy: D.Policy | None = None, state=STATE) -> D.Deadman:
    client = D.LambdaClient(D.read_key(ENV_FILE))
    return D.Deadman(client, Path(state), policy=policy or D.Policy())


def allowance(ledger_remaining_gpu_h: float, dm_spent_usd: float, itype: str = TYPE) -> float:
    """The dead-man max_usd that leaves exactly the remaining combined GPU-hours at the provider price."""
    per_gpu_h = PRICE[itype] / GPUS[itype]
    return float(dm_spent_usd + max(0.0, ledger_remaining_gpu_h) * per_gpu_h)


def deadman_gpu_h(state: dict, now: float | None = None) -> float:
    """GPU-hours of every dead-man-registered instance (launch to confirmed termination, or to ``now`` if live)."""
    now = time.time() if now is None else float(now)
    tot = 0.0
    for rec in (state.get("instances") or {}).values():
        la = rec.get("launched_at")
        if la is None:
            continue
        te = rec.get("confirmed_terminated_at") or now
        tot += max(0.0, float(te) - float(la)) * GPUS.get(rec.get("type"), 1) / 3600.0
    return tot


def tighten(
    state: Path,
    max_usd: float,
    t_max_hours: float,
    idle_limit_s: float = 1800.0,
    cap_generation: int | None = None,
):
    """Write the policy override (the dead-man reloads it every tick; its CLI base policy bounds every override).

    ``max_usd`` and the idle limit only tighten, except that ``max_usd`` is re-based once per newly logged cap
    generation (PREREG_WME §II.31: a cap raise is recorded in the study log before the units it funds, and the
    dead-man allowance is the remaining cap). ``t_max_hours`` (cumulative instance-hours) is set per launch.

    ``policy.json`` holds dead-man policy keys only (the dead-man refuses any other key, Phase L part 1 incident,
    20 Nov 2026); the cap-generation marker of the last re-base lives in ``policy_cap.json``. A legacy marker in
    ``policy.json`` is moved there."""
    p = Path(state) / "policy.json"
    pc = Path(state) / "policy_cap.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    legacy = cur.pop("cap_generation", None)
    seen = json.loads(pc.read_text()).get("cap_generation") if pc.exists() else legacy
    seen = -1 if seen is None else int(seen)
    new = {
        "max_usd": float(max_usd),
        "t_max_hours": float(t_max_hours),
        "idle_limit_s": float(idle_limit_s),
    }
    rebase = cap_generation is not None and int(cap_generation) > seen
    for k, v in list(new.items()):
        if k == "t_max_hours" or (k == "max_usd" and rebase):
            continue
        if k in cur and v > cur[k]:
            new[k] = cur[k]
    gen = max(int(cap_generation), seen) if cap_generation is not None else seen
    if gen >= 0:
        pc.write_text(json.dumps({"cap_generation": gen}))
    p.write_text(json.dumps(new, indent=1))
    return dict(new, cap_generation=gen if gen >= 0 else None)


def authoritative(state: Path, iid: str):
    st = json.loads((Path(state) / "state.json").read_text())
    rec = st["instances"].get(iid)
    if rec is None:
        raise KeyError(f"{iid} is not a dead-man-registered instance")
    return rec.get("launched_at"), rec.get("confirmed_terminated_at")


def guard_instance(iid: str) -> str:
    return f"lambda:{iid}"


def reconcile(guard: RG.ProductionGuard, state: Path, iid: str, gpus: int):
    la, te = authoritative(state, iid)
    if la is None or te is None:
        raise RuntimeError(f"{iid}: no confirmed termination in the dead-man ledger yet")
    guard.reconcile_instance(
        la, te, gpus, "dead-man ledger (API-confirmed termination)", instance=guard_instance(iid)
    )
    return la, te


def ssh(alias, cmd, cfg=SSH_CONFIG, timeout=120, check=True):
    p = subprocess.run(
        ["ssh", "-F", str(cfg), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", alias, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"ssh {cmd[:40]!r}: exit {p.returncode}: {p.stderr[-300:]}")
    return p.stdout


def wait_live(dm: D.Deadman, timeout_s=36000, poll_s=15, heartbeat=True):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = dm.state()
        live = dm.live(st)
        if live:
            return live[0]
        if heartbeat:
            dm.heartbeat()
        time.sleep(poll_s)
    raise TimeoutError("no live instance")


def wait_active_ssh(dm: D.Deadman, iid: str, alias: str, timeout_s=2700, poll_s=15):
    t0 = time.time()
    SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    while time.time() - t0 < timeout_s:
        dm.heartbeat()
        info = dm.client.get_instance(iid) or {}
        if info.get("status") == "active" and info.get("ip"):
            block = (
                f"Host {alias}\n  HostName {info['ip']}\n  User ubuntu\n  IdentityFile ~/.ssh/id_ed25519\n"
                "  StrictHostKeyChecking accept-new\n  UserKnownHostsFile ~/.config/wsp/wme_c1_known_hosts\n"
                "  ServerAliveInterval 30\n  ServerAliveCountMax 6\n"
            )
            old = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else ""
            SSH_CONFIG.write_text(D.upsert_host_block(old, alias, block))
            os.chmod(SSH_CONFIG, 0o600)
            try:
                ssh(alias, "true", timeout=30)
                return True
            except Exception:  # noqa: BLE001, S110 - sshd not up yet
                pass
        if str(info.get("status", "")).startswith("terminat"):
            raise RuntimeError(f"{iid} terminated by the provider during boot")
        time.sleep(poll_s)
    raise TimeoutError("ssh never came up")


def terminate_and_confirm(dm: D.Deadman, iid: str, reason: str, timeout_s=1800, poll_s=20):
    dm.request_terminate(iid, reason)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = dm.state()
        rec = st["instances"].get(iid, {})
        if rec.get("confirmed_terminated_at"):
            return rec["confirmed_terminated_at"]
        time.sleep(poll_s)
    raise TimeoutError(f"{iid}: termination not confirmed")
