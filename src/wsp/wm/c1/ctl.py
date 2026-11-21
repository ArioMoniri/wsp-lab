"""Controller core (engineering; PREREG_WME §II.13, §II.22.6, §II.24-§II.28).

* :class:`Remote` issues one request and waits for exactly its result (``protocol.verify_result``). Transient
  transport faults are retried inside the same attempt (the request stays in the worker's inbox, so a restarted
  worker reprocesses it); a worker silent for longer than ``dead_after_s`` or a lost instance ends the attempt.
* :func:`technical` maps failures to the registered technical causes; anything unrecognised (including a result
  that does not answer its request) is ``None``, so ``run_units`` halts (still charged).
* :func:`run_bracketed` runs units through ``harness.run_units`` with the mandatory ``ProductionGuard`` in chunks,
  each followed by a canary (start canary first), so every result is bracketed; it then resolves only eligible
  results (``guard.resolve``), never a pending or ineligible one.
* Fail fast (chair item FF, after the run4 resume halt): a request the worker refuses or aborts without a result
  (``rejected/``: stale epoch, fenced while running, malformed) is detected at the next poll and is a
  ``request_rejected`` run-health incident naming the reason; a request without a result or a rejection by its
  response deadline (``wsp.wm.c1.deadlines``) is withdrawn and is a ``request_deadline`` incident. Both are
  ``IncidentAbort``: an in-flight unit attempt is ``attempt_aborted`` (charged, never counted). The incident is written
  through the guard bound by :func:`take_over` (every owner takeover: fence + bind); without a guard (the parity
  instance, the first-launch checks) it is journalled and raised all the same.
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
from wsp.wm.c1 import deadlines as DL
from wsp.wm.c1 import protocol as P


class InstanceLost(RuntimeError):
    """The provider or the dead-man ledger reports the instance gone (terminated or unreachable for good)."""


class WorkerDead(RuntimeError):
    """The worker process stopped touching ``alive`` for longer than the tolerance."""


class RequestRejected(RG.IncidentAbort):
    """The worker refused or aborted the request without a result (its reason is in the message and the incident)."""

    incident_kind = "request_rejected"


class RequestDeadline(RG.IncidentAbort):
    """No result and no rejection within the request's response deadline; the request was withdrawn."""

    incident_kind = "request_deadline"


def take_over(transport, remote, guard):
    """An owner takeover: the worker fence moves to the guard's epoch (older requests are refused or aborted) and the
    guard is bound to the remote, so every request failure is recorded with the current owner."""
    transport.set_fence(guard.epoch, guard.instance)
    bind = getattr(remote, "bind", None)
    if bind is not None:
        bind(guard)


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
        watch=None,
        restart=None,
        withdraw_grace_s: float = 120.0,
        clock=time.time,
    ):
        self.clock = clock
        self.guard = (
            None  # bound by take_over / Study.run: the owner that records request incidents
        )
        self.restart = restart  # kills and restarts the worker process (a withdrawn request that does not stop)
        self.withdraw_grace_s = withdraw_grace_s
        self.watch = (
            watch  # called on every poll while a request is in flight (run-health rules, §II.31)
        )
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

    def bind(self, guard):
        self.guard = guard

    def _incident(self, exc_type, req: dict, detail: dict):
        """Record a request incident (bound guard, else the dispatch journal only) and return the exception."""
        d = {
            "id": req["id"],
            "kind": req["kind"],
            "label": req["label"],
            "epoch": req["epoch"],
            **detail,
        }
        self._journal({"event": "incident", "incident": exc_type.incident_kind, **d})
        if self.guard is not None:
            try:
                self.guard.incident(exc_type.incident_kind, d)
            except Exception as exc:  # noqa: BLE001 - a stale owner cannot write; the raise below still halts
                self._journal(
                    {"event": "incident_unrecorded", "id": req["id"], "error": str(exc)[:300]}
                )
        self.log(f"INCIDENT {exc_type.incident_kind}: {d}")
        return exc_type(f"{exc_type.incident_kind} {req['label']} ({req['kind']}): {detail}")

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

    def withdraw(self, req: dict) -> str:
        """Really stop an in-flight request (PREREG_WME §II.31.8 R6): withdraw it from the inbox, fence it (worker
        fence epoch above the request's, so the handler's next ``ctx.check`` raises ``Fenced``), wait up to
        ``withdraw_grace_s`` for the worker to drop it, else kill and restart the worker process. The controller
        resumes only under a new owner epoch (a new guard). Returns "fenced", "restarted" or "unconfirmed"."""
        self._with_retry(lambda: self.t.cancel(req["id"]), "cancel")
        self._with_retry(lambda: self.t.set_fence(int(req["epoch"]) + 1, req["instance"]), "fence")
        self._journal(
            {"event": "health_abort", "id": req["id"], "fence_epoch": int(req["epoch"]) + 1}
        )
        hb_of = getattr(self.t, "heartbeat", None)
        out = "unconfirmed"
        if hb_of is not None:
            t0 = self.clock()
            while self.clock() - t0 <= self.withdraw_grace_s:
                hb = self._with_retry(hb_of, "heartbeat") or {}
                if (hb.get("inflight") or {}).get("id") != req["id"]:
                    out = "fenced"
                    break
                self.sleep(self.poll_s)
        if out != "fenced" and self.restart is not None:
            self.restart()
            out = "restarted"
        self._journal({"event": "withdrawn", "id": req["id"], "how": out})
        return out

    def call(
        self, kind, epoch, instance, unit_key, label, args, timeout_s=None
    ) -> tuple[dict, Path]:
        req = P.make_request(kind, epoch, instance, unit_key, label, args)
        self._journal(
            {"event": "put", "id": req["id"], "kind": kind, "label": label, "epoch": epoch}
        )
        self._with_retry(lambda: self.t.put(req), "put")
        t0 = self.clock()
        deadline = (
            float(timeout_s) if timeout_s is not None else DL.DEADLINE_S.get(kind, DL.DEFAULT_S)
        )
        status_of = getattr(self.t, "status", None)
        revived = False
        while True:
            if status_of is not None:
                st = self._with_retry(lambda: status_of(req["id"]), "poll")
            else:
                st = (
                    "ready"
                    if self._with_retry(lambda: self.t.ready(req["id"]), "poll")
                    else "pending"
                )
            if st == "ready":
                break
            if st == "rejected":
                why = self._with_retry(lambda: self.t.rejection(req["id"]), "rejection") or {}
                detail = {"reason": why.get("reason", "unknown")}
                detail.update({k: v for k, v in why.items() if k not in ("id", "t", "reason")})
                raise self._incident(RequestRejected, req, detail)
            if self.clock() - t0 > deadline:
                # the incident first (a withdrawal may wait or restart the worker), then withdraw (R6), then halt
                exc = self._incident(
                    RequestDeadline, req, {"deadline_s": deadline, "waited_s": self.clock() - t0}
                )
                self.withdraw(req)
                raise exc
            if self.lost():
                raise InstanceLost(f"{label}: instance lost while waiting")
            if self.watch is not None:
                try:
                    self.watch()
                except Exception:
                    self.withdraw(req)
                    raise
            age = self._with_retry(self.t.alive_age, "alive")
            if age is not None and age > self.dead_after_s:
                if self.revive is not None and not revived:
                    revived = True
                    self._journal({"event": "revive", "id": req["id"], "alive_age": age})
                    self.log(
                        f"{label}: worker silent for {age:.0f} s; restarting the worker process"
                    )
                    self.revive()
                    t_rev = self.clock()
                    while self.clock() - t_rev < self.dead_after_s:
                        a2 = self._with_retry(self.t.alive_age, "alive")
                        if a2 is not None and a2 < self.dead_after_s:
                            break
                        self.sleep(self.poll_s)
                    continue
                raise WorkerDead(f"{label}: worker silent for {age:.0f} s")
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
    try:
        payload, _ = remote.call(
            "canary", guard.epoch, guard.instance, "canary", "canary", dict(args or {})
        )
        comp = composite_canary(payload["engines"])
    except RG.IncidentAbort:
        raise  # a run-health abort while the canary was in flight: already recorded by the monitor
    except Exception as exc:
        # RULING-wme-c1-bank-halt K2: a worker or child error inside a canary is a run-health incident, written to
        # the study log before the run halts (the guard then refuses every unit until it is resolved)
        guard.incident(
            "worker_error",
            {"where": "canary", "error": type(exc).__name__, "message": str(exc)[:1000]},
        )
        raise
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
    on_chunk=None,
    before_begin=None,
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
            H.run_units(
                store, [units[j]], fn, theta0, panel, guard=guard, before_begin=before_begin
            )
            j += 1
            if clock() - t0 > max_chunk_s:
                break
        rec = canary()
        if on_chunk is not None:  # the chunk-boundary health summary (§II.31), before any halt
            on_chunk(rec, j - i)
        if not rec["pass"]:
            raise RG.CanaryMismatch(f"closing canary {rec['composite'][:12]} != reference")
        i = j
    return resolve_all(guard, store, units, theta0, panel)
