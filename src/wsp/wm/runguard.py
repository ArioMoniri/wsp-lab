"""Validity, repair and stopping contract of the 9B confirmatory runs, enforced in the production path
(PREREG_WME §II.13, §II.21.1, §II.22.6, §II.24.2; Board A v2 required change 6 and confirmation blocker B).

``harness.run_units`` takes a **mandatory** guard. Confirmatory phases use :class:`ProductionGuard`; engineering
smokes and toy workers must pass an explicit :class:`EngineeringGuard` (results are plain, never eligible evidence).

**ProductionGuard** keeps one append-only, fsync'd event log (``events.jsonl`` in its state directory). Every
counter is rebuilt by replaying it, so a new process or a new instance resumes with the same state:

* **Launch validity.** Construction requires a :class:`LaunchAttestation` (inert identity and output bitwise, 1-vs-2-GPU
  parity bitwise, the canary reference hash, the engine-configuration hash, evidence references). Before any unit
  the instance's latest canary must equal the reference, otherwise :class:`CanaryMismatch` halts the run.
* **Attempts and relaunches.** At most :data:`UNIT_ATTEMPTS` = 3 attempts per unit **in total, across processes,
  instances and relaunches** (an attempt whose result is later made ineligible by a failed canary counts); then the
  unit's registered fallback is recorded. At most :data:`PHASE_RELAUNCHES` = 1 relaunch per phase.
* **Failures.** A failed attempt's measured seconds are charged to the ledger. A failure is retried only if the
  caller's classifier maps it to one of :data:`TECHNICAL_CAUSES`; anything else halts (re-raised, still charged).
* **Results and eligibility.** Each successful attempt is stored under its own key (``<unit>.a<n>``) with its event
  sequence number and instance. A result is **eligible** only if the instance's last canary before it and its first
  canary after it both match the reference. ``resolve`` (used by ``run_units`` for skip/resume) returns the **first
  eligible attempt in sequence order**, whatever its value; attempts before a failed canary stay stored as
  *retained, ineligible*; pending attempts (no closing canary yet) are not complete. Units with no eligible attempt
  are re-run on resume while attempts remain.
* **Absolute cap.** The charged total is the larger of (i) the measured GPU-seconds of every attempt, successful or
  failed, and (ii) the authoritative instance time (open-to-close, open instances to now, times GPUs per instance;
  setup, download, idle and canaries included). Before every attempt, charged + the priced unit estimate must stay
  <= :data:`CAP_GPU_H` and <= :data:`CAP_USD`, else :class:`CapExceeded` (engineering abort, no override). A phase above
  1.25x its price raises :class:`PhasePause`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

CAP_GPU_H = 90.0
CAP_USD = 1000.0
USD_PER_GPU_H = 4.89 * 1.10
PHASE_PAUSE = 1.25
UNIT_ATTEMPTS = 3
PHASE_RELAUNCHES = 1
TECHNICAL_CAUSES = (
    "infrastructure",
    "out_of_memory",
    "driver_or_cuda",
    "theta_restore_mismatch",
    "canary_mismatch",
    "preemption_or_capacity",
    "code_defect_crash",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CapExceeded(RuntimeError):
    pass


class PhasePause(RuntimeError):
    pass


class RetryExhausted(RuntimeError):
    pass


class CanaryMismatch(RuntimeError):
    pass


class NotAttested(ValueError):
    pass


def _append(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------ persistent building blocks


class CombinedLedger:
    """Append-only ledger of measured GPU seconds per phase (successful and failed attempts) and of authoritative
    instance time; one per 9B study."""

    def __init__(self, path, priced_gpu_h: dict, clock=time.time):
        self.path = Path(path)
        self.priced = dict(priced_gpu_h)
        self.clock = clock
        self.rows = _read(self.path)

    def add(self, phase: str, gpu_seconds: float, note: str = "", status: str = "ok"):
        if phase not in self.priced:
            raise KeyError(f"unpriced phase {phase}")
        r = {
            "kind": "unit",
            "phase": phase,
            "gpu_s": float(gpu_seconds),
            "note": note,
            "status": status,
        }
        self.rows.append(r)
        _append(self.path, r)

    def instance_open(self, instance: str, gpus: int, t: float | None = None):
        r = {
            "kind": "instance_open",
            "instance": instance,
            "gpus": int(gpus),
            "t": self.clock() if t is None else t,
        }
        self.rows.append(r)
        _append(self.path, r)

    def instance_close(self, instance: str, t: float | None = None):
        r = {"kind": "instance_close", "instance": instance, "t": self.clock() if t is None else t}
        self.rows.append(r)
        _append(self.path, r)

    def unit_gpu_h(self, phase: str | None = None) -> float:
        return (
            sum(
                r["gpu_s"]
                for r in self.rows
                if r.get("kind", "unit") == "unit" and phase in (None, r["phase"])
            )
            / 3600
        )

    def instance_gpu_h(self, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        opened, total = {}, 0.0
        for r in self.rows:
            if r.get("kind") == "instance_open":
                opened[r["instance"]] = (r["t"], r["gpus"])
            elif r.get("kind") == "instance_close" and r["instance"] in opened:
                t0, g = opened.pop(r["instance"])
                total += (r["t"] - t0) * g
        total += sum((now - t0) * g for t0, g in opened.values())
        return total / 3600

    def gpu_h(self, phase: str | None = None) -> float:
        """Phase totals are measured unit time; the study total is the larger of unit and instance time."""
        if phase is not None:
            return self.unit_gpu_h(phase)
        return max(self.unit_gpu_h(), self.instance_gpu_h())

    def usd(self) -> float:
        return self.gpu_h() * USD_PER_GPU_H

    def remaining_usd(self) -> float:
        """For the per-instance dead-man ``max_usd``: never more than what the combined cap leaves."""
        return max(0.0, min(CAP_USD - self.usd(), (CAP_GPU_H - self.gpu_h()) * USD_PER_GPU_H))

    def check_before(self, phase: str, est_gpu_h: float):
        tot = self.gpu_h() + est_gpu_h
        if tot > CAP_GPU_H or tot * USD_PER_GPU_H > CAP_USD:
            raise CapExceeded(
                f"combined cap: {self.gpu_h():.3f} + {est_gpu_h:.3f} GPU-h-eq > {CAP_GPU_H} or $ > {CAP_USD}"
            )
        if self.unit_gpu_h(phase) + est_gpu_h > PHASE_PAUSE * self.priced[phase]:
            raise PhasePause(
                f"{phase}: above {PHASE_PAUSE} x its priced {self.priced[phase]} GPU-h"
            )


class RetryPolicy:
    """Attempt and relaunch counters; persisted to ``path`` (JSONL) when given, so a restart resumes the counts."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.attempts: dict = {}
        self.relaunches: dict = {}
        self.repairs: list = []
        for r in _read(self.path) if self.path else []:
            if r["kind"] == "attempt":
                self.attempts[r["unit"]] = r["n"]
            elif r["kind"] == "relaunch":
                self.relaunches[r["phase"]] = r["n"]
            elif r["kind"] == "repair":
                self.repairs.append(r)

    def _log(self, rec):
        if self.path:
            _append(self.path, rec)

    def attempt(self, unit: str, cause: str | None = None) -> int:
        if cause is not None and cause not in TECHNICAL_CAUSES:
            raise ValueError(f"retry cause {cause!r} is not a registered technical cause")
        n = self.attempts.get(unit, 0) + 1
        if n > UNIT_ATTEMPTS:
            raise RetryExhausted(
                f"{unit}: {UNIT_ATTEMPTS} attempts used; take its registered fallback"
            )
        self.attempts[unit] = n
        self._log({"kind": "attempt", "unit": unit, "n": n, "cause": cause})
        return n

    def relaunch(self, phase: str, cause: str):
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"relaunch cause {cause!r} is not a registered technical cause")
        n = self.relaunches.get(phase, 0) + 1
        if n > PHASE_RELAUNCHES:
            raise RetryExhausted(f"{phase}: the one registered relaunch is used; engineering abort")
        self.relaunches[phase] = n
        self._log({"kind": "relaunch", "phase": phase, "n": n, "cause": cause})
        return n

    def repair(self, cause: str, description: str, commit: str, outcome_blind: bool) -> dict:
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"repair cause {cause!r} is not a registered technical cause")
        if outcome_blind is not True:
            raise ValueError("repairs must be decided without reading any outcome")
        if not description or len(commit) < 7:
            raise ValueError("a repair needs a description and the code commit")
        r = {
            "kind": "repair",
            "cause": cause,
            "description": description,
            "commit": commit,
            "outcome_blind": True,
        }
        self.repairs.append(r)
        self._log(r)
        return r


class CanaryBook:
    """Canary events (instance, order, sha256) against one reference; persisted to ``path`` when given. The order is a
    time or an event sequence number."""

    def __init__(self, reference: str, path=None):
        self.reference = reference
        self.path = Path(path) if path else None
        self.events: dict = {}
        for r in _read(self.path) if self.path else []:
            self.events.setdefault(r["instance"], []).append((float(r["t"]), r["sha"]))

    def canary(self, instance: str, t: float, sha: str):
        self.events.setdefault(instance, []).append((float(t), sha))
        if self.path:
            _append(self.path, {"instance": instance, "t": float(t), "sha": sha})

    def last(self, instance: str):
        ev = sorted(self.events.get(instance, []))
        return ev[-1][1] if ev else None

    def status(self, instance: str, t: float) -> str:
        ev = sorted(self.events.get(instance, []))
        before = [s for u, s in ev if u <= t]
        after = [s for u, s in ev if u > t]
        if not before or before[-1] != self.reference:
            return "retained_ineligible"
        if not after:
            return "pending"
        return "eligible" if after[0] == self.reference else "retained_ineligible"

    def eligible(self, instance: str, t: float) -> bool:
        return self.status(instance, t) == "eligible"

    def eligible_result(self, attempts):
        """``attempts`` [(instance, order, result)] of one unit: the first eligible attempt in order, or None."""
        for inst, t, res in sorted(attempts, key=lambda a: a[1]):
            if self.eligible(inst, t):
                return res
        return None


# ------------------------------------------------------------------ guards for harness.run_units


class EngineeringGuard:
    """Explicit opt-out for engineering smokes and toy workers: plain UnitStore semantics, no cap, no eligibility.
    Results produced under it are never confirmatory evidence (``confirmatory = False``)."""

    confirmatory = False

    def __init__(self, purpose: str):
        if not purpose:
            raise ValueError("state the engineering purpose")
        self.purpose = purpose
        self.clock = time.time

    def resolve(self, store, key):
        return store.read(key) if store.done(key) else None

    def begin(self, key, label) -> int:
        return 1

    def failed(self, key, label, seconds, exc) -> None:
        raise exc

    def succeeded(self, store, key, label, n, payload, seconds, theta0):
        store.write(key, payload, theta_after=theta0)
        return payload

    def fallback(self, key, label):  # pragma: no cover - never reached (begin never raises)
        raise RetryExhausted(key)


@dataclass
class LaunchAttestation:
    """Launch validity preconditions (§II.21.1, §II.22.6), each evidenced before the first confirmatory unit."""

    inert_identity_bitwise: bool
    gpu_parity_bitwise: bool
    canary_reference: str
    engine_config_sha256: str
    evidence: list = field(default_factory=list)

    def check(self):
        if self.inert_identity_bitwise is not True or self.gpu_parity_bitwise is not True:
            raise NotAttested(
                "inert identity and 1-vs-2-GPU parity must be attested bitwise before launch"
            )
        if not _HEX64.match(self.canary_reference or "") or not _HEX64.match(
            self.engine_config_sha256 or ""
        ):
            raise NotAttested(
                "the canary reference and engine-config hash must be sha256 hex digests"
            )
        if not self.evidence:
            raise NotAttested("cite the evidence (report paths and hashes) for each attestation")


class ProductionGuard:
    """The mandatory confirmatory guard (see the module docstring)."""

    confirmatory = True

    def __init__(
        self,
        state_dir,
        phase: str,
        priced_gpu_h: dict,
        attestation: LaunchAttestation,
        instance: str,
        est_unit_gpu_h: float,
        technical=None,
        fallback_value=None,
        clock=time.time,
    ):
        attestation.check()
        self.state = Path(state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.phase, self.instance, self.est = phase, instance, float(est_unit_gpu_h)
        self.attestation = attestation
        self.technical = technical or (lambda exc: None)
        self.fallback_value = fallback_value
        self.clock = clock
        self.ledger = CombinedLedger(self.state / "ledger.jsonl", priced_gpu_h, clock=clock)
        self.retry = RetryPolicy(self.state / "retry.jsonl")
        self.canaries = CanaryBook(attestation.canary_reference, self.state / "canary.jsonl")
        self.events_path = self.state / "events.jsonl"
        self.events = _read(self.events_path)
        if not any(e["type"] == "attestation" for e in self.events):
            self._event(
                "attestation",
                **{k: getattr(attestation, k) for k in attestation.__dataclass_fields__},
            )

    # -------------------------------------------------- event log and sequence numbers
    def _seq(self) -> int:
        return max((e["seq"] for e in self.events), default=0) + 1

    def _event(self, type_, **kw) -> dict:
        e = {
            "seq": self._seq(),
            "type": type_,
            "instance": self.instance,
            "phase": self.phase,
            **kw,
        }
        self.events.append(e)
        _append(self.events_path, e)
        return e

    # -------------------------------------------------- instances and canaries
    def open_instance(self, gpus: int = 1):
        self.ledger.instance_open(self.instance, gpus)
        self._event("instance_open", gpus=gpus)

    def close_instance(self):
        self.ledger.instance_close(self.instance)
        self._event("instance_close")

    def canary(self, sha: str) -> bool:
        e = self._event("canary", sha=sha)
        self.canaries.canary(self.instance, e["seq"], sha)
        return sha == self.attestation.canary_reference

    def relaunch(self, cause: str, new_instance: str):
        self.retry.relaunch(self.phase, cause)
        self._event("relaunch", cause=cause, new_instance=new_instance)
        self.instance = new_instance

    # -------------------------------------------------- unit protocol used by harness.run_units
    def _attempts(self, key):
        return [e for e in self.events if e["type"] == "attempt_ok" and e["unit"] == key]

    def unit_status(self, key) -> dict:
        att = self._attempts(key)
        st = [(a["attempt_key"], self.canaries.status(a["instance"], a["seq"])) for a in att]
        fb = any(e["type"] == "fallback" and e["unit"] == key for e in self.events)
        return {"attempts": st, "fallback": fb}

    def resolve(self, store, key):
        for a in sorted(self._attempts(key), key=lambda a: a["seq"]):
            if self.canaries.eligible(a["instance"], a["seq"]):
                return store.read(a["attempt_key"])
        if any(e["type"] == "fallback" and e["unit"] == key for e in self.events):
            return {"fallback": True, "value": self.fallback_value}
        return None

    def begin(self, key, label) -> int:
        if self.canaries.last(self.instance) != self.attestation.canary_reference:
            raise CanaryMismatch(
                f"{self.instance}: no passing canary since the last unit; halt for repair"
            )
        if any(
            e["type"] == "attempt_ok"
            and e["unit"] == key
            and self.canaries.status(e["instance"], e["seq"]) == "pending"
            for e in self.events
        ):
            raise CanaryMismatch(
                f"{label}: an attempt is pending its closing canary; close it before re-running"
            )
        self.ledger.check_before(self.phase, self.est)
        prev = [e for e in self.events if e["type"] == "attempt_fail" and e["unit"] == key]
        n = self.retry.attempt(key, prev[-1]["cause"] if prev else None)
        self._event("attempt_start", unit=key, label=label, n=n)
        return n

    def failed(self, key, label, seconds, exc):
        self.ledger.add(self.phase, seconds, note=f"{label} failed", status="failed")
        cause = self.technical(exc)
        self._event(
            "attempt_fail",
            unit=key,
            label=label,
            seconds=seconds,
            cause=cause,
            error=type(exc).__name__,
        )
        if cause not in TECHNICAL_CAUSES:
            raise exc

    def succeeded(self, store, key, label, n, payload, seconds, theta0):
        akey = f"{key}.a{n}"
        store.write(akey, payload, theta_after=theta0)
        self.ledger.add(self.phase, seconds, note=str(label))
        self._event("attempt_ok", unit=key, label=label, n=n, attempt_key=akey, seconds=seconds)
        return {"status": "pending_canary", "attempt_key": akey}

    def fallback(self, key, label):
        self._event("fallback", unit=key, label=label)
        return {"fallback": True, "value": self.fallback_value}

    def report(self) -> dict:
        units = sorted({e["unit"] for e in self.events if "unit" in e})
        return {
            "units": {u: self.unit_status(u) for u in units},
            "charged_gpu_h": self.ledger.gpu_h(),
            "unit_gpu_h": self.ledger.unit_gpu_h(),
            "instance_gpu_h": self.ledger.instance_gpu_h(),
            "relaunches": dict(self.retry.relaunches),
        }
