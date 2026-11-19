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
* **Instance lifecycle (§II.25).** No unit runs unless the guard's instance is registered and active
  (:class:`InstanceNotOpen`). A duplicate opening is idempotent and keeps the earliest opening; a closed id cannot be
  reopened; authoritative records (provider API, dead-man ledger: ``reconcile_instance``) override local times.
* **Contract binding (§II.25).** The digest of the full attestation is persisted. A guard rebuilt with a different
  contract refuses to run or return results (:class:`ContractMismatch`) unless an explicit, persisted, technical
  :class:`ContractTransition` is recorded; results and canaries count only under the contract they were made in.
* **Lost instances (§II.25).** ``declare_lost`` (technical, outcome-blind, persisted) closes a preempted instance at
  its authoritative termination (charged), makes its unbracketed attempts retained-ineligible without any
  fabricated canary, keeps their attempt counts, and lets the remaining attempts or the fallback proceed elsewhere.
* **Registered fallback (§II.25).** The fallback payload and its sha256 are persisted at registration and with every
  fallback; a guard rebuilt with a different fallback is refused (:class:`FallbackMismatch`).
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


class InstanceNotOpen(RuntimeError):
    pass


class ContractMismatch(RuntimeError):
    pass


class FallbackMismatch(RuntimeError):
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

    def instance_state(self, instance: str) -> str:
        """``unknown``, ``active`` or ``closed`` (a local close or an authoritative termination)."""
        st = "unknown"
        for r in self.rows:
            if r.get("instance") != instance:
                continue
            if r.get("kind") == "instance_open" and st == "unknown":
                st = "active"
            elif r.get("kind") == "instance_close" or (
                r.get("kind") == "instance_authoritative" and r.get("terminated_at") is not None
            ):
                st = "closed"
            elif r.get("kind") == "instance_authoritative" and st == "unknown":
                st = "active"
        return st

    def instance_open(self, instance: str, gpus: int, t: float | None = None) -> bool:
        """Register an instance opening. Idempotent for an active instance (the earliest opening is kept; a restart
        never resets time); a closed instance id cannot be reopened (a replacement needs a new id)."""
        st = self.instance_state(instance)
        if st == "active":
            return False
        if st == "closed":
            raise ValueError(f"{instance} is closed; a replacement instance needs a new id")
        r = {
            "kind": "instance_open",
            "instance": instance,
            "gpus": int(gpus),
            "t": self.clock() if t is None else t,
        }
        self.rows.append(r)
        _append(self.path, r)
        return True

    def instance_close(self, instance: str, t: float | None = None):
        if self.instance_state(instance) != "active":
            raise ValueError(f"{instance} is not active")
        r = {"kind": "instance_close", "instance": instance, "t": self.clock() if t is None else t}
        self.rows.append(r)
        _append(self.path, r)

    def reconcile_instance(
        self, instance: str, launched_at: float, terminated_at, gpus: int, source: str
    ):
        """An authoritative instance record (provider API or dead-man terminator ledger). It overrides the local
        opening and closing for that instance; ``terminated_at`` None means still running (counted to now). The
        authoritative launch can only be earlier than or equal to a local opening."""
        if not source:
            raise ValueError("name the authoritative source")
        local = [
            r["t"]
            for r in self.rows
            if r.get("kind") == "instance_open" and r["instance"] == instance
        ]
        if local and launched_at > min(local):
            raise ValueError("an authoritative launch cannot be later than the recorded opening")
        if terminated_at is not None and terminated_at < launched_at:
            raise ValueError("termination before launch")
        r = {
            "kind": "instance_authoritative",
            "instance": instance,
            "launched_at": float(launched_at),
            "terminated_at": None if terminated_at is None else float(terminated_at),
            "gpus": int(gpus),
            "source": source,
        }
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
        """Per instance: the latest authoritative record if any, else the earliest local opening to the local close
        (or to now while active); times GPUs."""
        now = self.clock() if now is None else now
        auth, opened, closed = {}, {}, {}
        for r in self.rows:
            k = r.get("kind")
            if k == "instance_authoritative":
                auth[r["instance"]] = r
            elif k == "instance_open":
                opened.setdefault(r["instance"], (r["t"], r["gpus"]))
            elif k == "instance_close":
                closed.setdefault(r["instance"], r["t"])
        total = 0.0
        for inst in set(auth) | set(opened):
            if inst in auth:
                a = auth[inst]
                end = now if a["terminated_at"] is None else a["terminated_at"]
                total += (end - a["launched_at"]) * a["gpus"]
            else:
                t0, g = opened[inst]
                total += (closed.get(inst, now) - t0) * g
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


def _digest(obj) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def contract_digest(att: LaunchAttestation) -> str:
    """The registered run contract: every attestation field (evidence order-free)."""
    d = {k: getattr(att, k) for k in att.__dataclass_fields__}
    d["evidence"] = sorted(d["evidence"])
    return _digest(d)


@dataclass
class ContractTransition:
    """An explicit, persisted, auditable technical repair that changes the run contract (e.g. a driver update that
    changes the engine-config digest). Results made under the earlier contract never become eligible under the new
    one; the new contract needs its own canaries."""

    cause: str
    description: str
    commit: str
    outcome_blind: bool

    def check(self):
        if self.cause not in TECHNICAL_CAUSES:
            raise ValueError(f"transition cause {self.cause!r} is not a registered technical cause")
        if self.outcome_blind is not True:
            raise ValueError("a contract transition must be decided without reading any outcome")
        if not self.description or len(self.commit) < 7:
            raise ValueError("a contract transition needs a description and the code commit")


class ProductionGuard:
    """The mandatory confirmatory guard (see the module docstring and §II.25)."""

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
        transition: ContractTransition | None = None,
    ):
        attestation.check()
        self.state = Path(state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.phase, self.instance, self.est = phase, instance, float(est_unit_gpu_h)
        self.attestation = attestation
        self.technical = technical or (lambda exc: None)
        self.clock = clock
        self.ledger = CombinedLedger(self.state / "ledger.jsonl", priced_gpu_h, clock=clock)
        self.retry = RetryPolicy(self.state / "retry.jsonl")
        self.events_path = self.state / "events.jsonl"
        self.events = _read(self.events_path)
        # ---- the run contract (gap 2)
        new = contract_digest(attestation)
        contracts = [e for e in self.events if e["type"] in ("attestation", "contract_transition")]
        self.contract = new
        if not contracts:
            self._event("attestation", digest=new, **self._att_fields(attestation))
        elif contracts[-1]["digest"] != new:
            if transition is None:
                raise ContractMismatch(
                    f"persisted contract {contracts[-1]['digest'][:12]} != {new[:12]}: refuse to run or return "
                    "results without an explicit ContractTransition"
                )
            transition.check()
            self._event(
                "contract_transition",
                digest=new,
                previous=contracts[-1]["digest"],
                cause=transition.cause,
                description=transition.description,
                commit=transition.commit,
                outcome_blind=True,
                **self._att_fields(attestation),
            )
        elif transition is not None:
            raise ValueError("a ContractTransition was given but the contract is unchanged")
        self.references = {
            e["digest"]: e["canary_reference"]
            for e in self.events
            if e["type"] in ("attestation", "contract_transition")
        }
        # ---- the registered fallback (gap 4)
        try:
            fb_digest = _digest(fallback_value)
        except TypeError as exc:
            raise ValueError("the registered fallback must be JSON-serialisable") from exc
        reg = [e for e in self.events if e["type"] == "fallback_registered" and e["phase"] == phase]
        if not reg:
            self._event("fallback_registered", payload=fallback_value, payload_sha256=fb_digest)
        elif reg[0]["payload_sha256"] != fb_digest:
            raise FallbackMismatch(
                f"registered fallback {reg[0]['payload_sha256'][:12]} != {fb_digest[:12]}; refuse"
            )
        self.fallback_payload = reg[0]["payload"] if reg else fallback_value
        self.fallback_sha256 = fb_digest

    @staticmethod
    def _att_fields(att):
        d = {k: getattr(att, k) for k in att.__dataclass_fields__}
        d["evidence"] = sorted(d["evidence"])
        return d

    # -------------------------------------------------- event log and sequence numbers
    def _seq(self) -> int:
        return max((e["seq"] for e in self.events), default=0) + 1

    def _event(self, type_, **kw) -> dict:
        e = {
            "seq": self._seq(),
            "type": type_,
            "instance": self.instance,
            "phase": self.phase,
            "contract": getattr(self, "contract", None),
            **kw,
        }
        self.events.append(e)
        _append(self.events_path, e)
        return e

    # -------------------------------------------------- instances and canaries (gaps 1 and 3)
    def _lost(self, instance) -> bool:
        return any(
            e["type"] == "instance_lost" and e["lost_instance"] == instance for e in self.events
        )

    def open_instance(self, gpus: int = 1):
        if self._lost(self.instance):
            raise InstanceNotOpen(f"{self.instance} was declared lost; relaunch on a new instance")
        new = self.ledger.instance_open(self.instance, gpus)
        self._event("instance_open" if new else "instance_resume", gpus=gpus)

    def close_instance(self):
        self.ledger.instance_close(self.instance)
        self._event("instance_close")

    def reconcile_instance(self, launched_at, terminated_at, gpus, source, instance=None):
        inst = instance or self.instance
        self.ledger.reconcile_instance(inst, launched_at, terminated_at, gpus, source)
        self._event("instance_reconciled", target=inst, source=source, terminated_at=terminated_at)

    def declare_lost(
        self,
        instance: str,
        cause: str,
        description: str,
        commit: str,
        outcome_blind: bool,
        terminated_at=None,
        source: str = "operator: provider termination record",
    ):
        """Technical disposition for an instance lost before its closing canary (preemption, crash): its
        unbracketed attempts become retained-ineligible (no canary is fabricated), its attempt counts stay, and
        its instance time is charged to ``terminated_at`` (now if unknown)."""
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"loss cause {cause!r} is not a registered technical cause")
        if outcome_blind is not True:
            raise ValueError("a loss disposition must be decided without reading any outcome")
        if not description or len(commit) < 7:
            raise ValueError("a loss disposition needs a description and the code commit")
        if self.ledger.instance_state(instance) == "active":
            opened = [
                r["t"]
                for r in self.ledger.rows
                if r.get("kind") == "instance_open" and r["instance"] == instance
            ]
            g = [
                r["gpus"]
                for r in self.ledger.rows
                if r.get("kind") == "instance_open" and r["instance"] == instance
            ]
            end = self.clock() if terminated_at is None else terminated_at
            self.ledger.reconcile_instance(instance, min(opened), end, g[0], source)
        self._event(
            "instance_lost",
            lost_instance=instance,
            cause=cause,
            description=description,
            commit=commit,
            outcome_blind=True,
            terminated_at=terminated_at,
        )

    def canary(self, sha: str) -> bool:
        self._event("canary", sha=sha)
        return sha == self.attestation.canary_reference

    def relaunch(self, cause: str, new_instance: str):
        self.retry.relaunch(self.phase, cause)
        self._event("relaunch", cause=cause, new_instance=new_instance)
        self.instance = new_instance

    # -------------------------------------------------- eligibility (per contract)
    def attempt_status(self, a) -> str:
        if a.get("contract") != self.contract:
            return "superseded_contract"
        ref = self.references[a["contract"]]
        cans = [
            e
            for e in self.events
            if e["type"] == "canary"
            and e["instance"] == a["instance"]
            and e.get("contract") == a["contract"]
        ]
        before = [e for e in cans if e["seq"] < a["seq"]]
        after = [e for e in cans if e["seq"] > a["seq"]]
        if not before or before[-1]["sha"] != ref:
            return "retained_ineligible"
        if not after:
            return "retained_ineligible" if self._lost(a["instance"]) else "pending"
        return "eligible" if after[0]["sha"] == ref else "retained_ineligible"

    def _attempts(self, key):
        return [e for e in self.events if e["type"] == "attempt_ok" and e["unit"] == key]

    def unit_status(self, key) -> dict:
        st = [(a["attempt_key"], self.attempt_status(a)) for a in self._attempts(key)]
        fb = any(e["type"] == "fallback" and e["unit"] == key for e in self.events)
        return {"attempts": st, "fallback": fb}

    # -------------------------------------------------- unit protocol used by harness.run_units
    def resolve(self, store, key):
        for a in sorted(self._attempts(key), key=lambda a: a["seq"]):
            if self.attempt_status(a) == "eligible":
                return store.read(a["attempt_key"])
        fbs = [e for e in self.events if e["type"] == "fallback" and e["unit"] == key]
        if fbs:
            fb = fbs[0]
            if (
                fb["payload_sha256"] != self.fallback_sha256
                or _digest(fb["payload"]) != fb["payload_sha256"]
            ):
                raise FallbackMismatch(f"{key}: persisted fallback does not verify")
            return {
                "fallback": True,
                "value": fb["payload"],
                "payload_sha256": fb["payload_sha256"],
            }
        return None

    def begin(self, key, label) -> int:
        if self.ledger.instance_state(self.instance) != "active" or self._lost(self.instance):
            raise InstanceNotOpen(
                f"{self.instance}: no validated open instance; open_instance() first"
            )
        cans = [
            e
            for e in self.events
            if e["type"] == "canary"
            and e["instance"] == self.instance
            and e.get("contract") == self.contract
        ]
        if not cans or cans[-1]["sha"] != self.attestation.canary_reference:
            raise CanaryMismatch(
                f"{self.instance}: no passing canary since the last unit; halt for repair"
            )
        if any(self.attempt_status(a) == "pending" for a in self._attempts(key)):
            raise CanaryMismatch(
                f"{label}: an attempt is pending its closing canary; close it (or declare its instance lost)"
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
        self._event(
            "fallback",
            unit=key,
            label=label,
            payload=self.fallback_payload,
            payload_sha256=self.fallback_sha256,
        )
        return {
            "fallback": True,
            "value": self.fallback_payload,
            "payload_sha256": self.fallback_sha256,
        }

    def report(self) -> dict:
        units = sorted({e["unit"] for e in self.events if "unit" in e})
        return {
            "contract": self.contract,
            "units": {u: self.unit_status(u) for u in units},
            "charged_gpu_h": self.ledger.gpu_h(),
            "unit_gpu_h": self.ledger.unit_gpu_h(),
            "instance_gpu_h": self.ledger.instance_gpu_h(),
            "relaunches": dict(self.retry.relaunches),
            "lost_instances": sorted(
                {e["lost_instance"] for e in self.events if e["type"] == "instance_lost"}
            ),
        }
