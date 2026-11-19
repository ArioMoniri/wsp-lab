"""Controller core (engineering; PREREG_WME §II.13, §II.22.6, §II.24-§II.28).

* :class:`Remote` issues one request and waits for exactly its result (``protocol.verify_result``). Transient
  transport faults are retried inside the same attempt (the request stays in the worker's inbox, so a restarted
  worker reprocesses it); a worker silent for longer than ``dead_after_s`` or a lost instance ends the attempt.
* :func:`technical` maps failures to the registered technical causes; anything unrecognised (including a result
  that does not answer its request) is ``None``, so ``run_units`` halts (still charged).
* :func:`run_bracketed` runs units through ``harness.run_units`` with the mandatory ``ProductionGuard`` in chunks,
  each followed by a canary (start canary first), so every result is bracketed; it then resolves only eligible
  results (``guard.resolve``), never a pending or ineligible one.
* :func:`composite_canary`: the guard holds one ``canary_reference``; it is the sha256 of the committed per-engine
  reference map, and every canary point runs every engine (``wm.c1.canary``).

Everything here is outcome-blind: no function reads a result's value to decide what runs next, except the registered
data-flow (a later step consumes earlier eligible results as inputs).
"""

from __future__ import annotations

import time
from pathlib import Path

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import protocol as P


class InstanceLost(RuntimeError):
    """The provider or the dead-man ledger reports the instance gone (terminated or unreachable for good)."""


class WorkerDead(RuntimeError):
    """The worker process stopped touching ``alive`` for longer than the tolerance."""


def technical(exc: BaseException) -> str | None:
    if isinstance(exc, P.WorkerError):
        return exc.cause if exc.cause in RG.TECHNICAL_CAUSES else None
    if isinstance(exc, (P.TransportError, WorkerDead)):
        return "infrastructure"
    if isinstance(exc, InstanceLost):
        return "preemption_or_capacity"
    return None


class Remote:
    def __init__(
        self,
        transport,
        local_root,
        poll_s: float = 5.0,
        dead_after_s: float = 1800.0,
        transport_retry_s: float = 600.0,
        lost=None,
        sleep=time.sleep,
        log=None,
        revive=None,
    ):
        self.revive = (
            revive  # restarts a dead worker process (same instance); the inbox request survives
        )
        self.t = transport
        self.local = Path(local_root)
        self.local.mkdir(parents=True, exist_ok=True)
        self.poll_s, self.dead_after_s, self.retry_s = poll_s, dead_after_s, transport_retry_s
        self.lost = lost or (lambda: False)
        self.sleep = sleep
        self.log = log or (lambda *a, **k: None)
        self.journal = self.local / "dispatch.jsonl"

    def _journal(self, rec):
        import json

        with open(self.journal, "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")

    def _with_retry(self, fn, what):
        t0 = time.time()
        delay = 5.0
        while True:
            try:
                return fn()
            except P.TransportError as exc:
                if self.lost():
                    raise InstanceLost(f"{what}: instance lost") from exc
                if time.time() - t0 > self.retry_s:
                    raise
                self.log(f"transport fault during {what}: {exc}; retrying")
                self.sleep(delay)
                delay = min(delay * 2, 60.0)

    def call(
        self, kind, epoch, instance, unit_key, label, args, timeout_s=None
    ) -> tuple[dict, Path]:
        req = P.make_request(kind, epoch, instance, unit_key, label, args)
        self._journal(
            {"event": "put", "id": req["id"], "kind": kind, "label": label, "epoch": epoch}
        )
        self._with_retry(lambda: self.t.put(req), "put")
        t0 = time.time()
        revived = False
        while True:
            if self._with_retry(lambda: self.t.ready(req["id"]), "poll"):
                break
            if self.lost():
                raise InstanceLost(f"{label}: instance lost while waiting")
            age = self._with_retry(self.t.alive_age, "alive")
            if age is not None and age > self.dead_after_s:
                if self.revive is not None and not revived:
                    revived = True
                    self._journal({"event": "revive", "id": req["id"], "alive_age": age})
                    self.log(
                        f"{label}: worker silent for {age:.0f} s; restarting the worker process"
                    )
                    self.revive()
                    t_rev = time.time()
                    while time.time() - t_rev < self.dead_after_s:
                        a2 = self._with_retry(self.t.alive_age, "alive")
                        if a2 is not None and a2 < self.dead_after_s:
                            break
                        self.sleep(self.poll_s)
                    continue
                raise WorkerDead(f"{label}: worker silent for {age:.0f} s")
            if timeout_s is not None and time.time() - t0 > timeout_s:
                self._with_retry(lambda: self.t.cancel(req["id"]), "cancel")
                raise P.TransportError(f"{label}: no result within {timeout_s} s")
            self.sleep(self.poll_s)
        dest = self.local / "results" / req["id"]
        self._with_retry(lambda: self.t.fetch(req["id"], dest), "fetch")
        res = P.verify_result(dest, req)
        self._journal({"event": "result", "id": req["id"], "status": res["status"]})
        P.raise_if_error(res)
        return res["payload"], dest


def composite_canary(per_engine: dict) -> str:
    """The single guard canary value: sha256 of the canonical per-engine map (``wm.c1.canary.ENGINES`` keys)."""
    if not per_engine or any(not isinstance(v, str) or len(v) != 64 for v in per_engine.values()):
        raise ValueError("every engine canary must be a sha256 hex digest")
    return P.digest({"schema": "wme-c1-canary-v1", "engines": dict(sorted(per_engine.items()))})


def run_canary(guard, remote: Remote, args=None) -> dict:
    """Run every engine's canary on the worker and record the composite with the guard. Returns the record."""
    payload, _ = remote.call(
        "canary", guard.epoch, guard.instance, "canary", "canary", dict(args or {})
    )
    comp = composite_canary(payload["engines"])
    ok = guard.canary(comp)
    return {"engines": payload["engines"], "composite": comp, "pass": bool(ok), "detail": payload}


def resolve_all(guard, store, units, theta0, panel) -> dict:
    out = {}
    for recipe, label in units:
        out[label] = guard.resolve(store, H.unit_key(recipe, theta0, panel))
    return out


def run_bracketed(
    guard,
    store,
    units,
    fn,
    theta0: str,
    panel: str,
    canary,
    chunk: int = 8,
    max_chunk_s: float = 3600.0,
    clock=time.time,
) -> dict:
    """Start canary, then chunks of at most ``chunk`` units (or ``max_chunk_s`` of wall time), each closed by a
    canary. A failed canary halts (``CanaryMismatch``) after it is recorded: results since the previous canary are
    then retained-ineligible by the guard. Returns {label: eligible payload | fallback | None}."""
    units = list(units)
    rec = canary()
    if not rec["pass"]:
        raise RG.CanaryMismatch(f"start canary {rec['composite'][:12]} != reference")
    i = 0
    while i < len(units):
        t0 = clock()
        j = i
        while j < len(units) and j - i < chunk:
            H.run_units(store, [units[j]], fn, theta0, panel, guard=guard)
            j += 1
            if clock() - t0 > max_chunk_s:
                break
        rec = canary()
        if not rec["pass"]:
            raise RG.CanaryMismatch(f"closing canary {rec['composite'][:12]} != reference")
        i = j
    return resolve_all(guard, store, units, theta0, panel)
