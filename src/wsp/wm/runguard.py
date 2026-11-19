"""Validity, repair and stopping contract of the 9B confirmatory runs (PREREG_WME §II.13, §II.21.1, §II.22.6;
Board A v2 required change 6). CPU logic; every confirmatory driver passes a :class:`UnitGuard` to
``harness.run_units`` and keeps one :class:`CombinedLedger` for the whole 9B study.

* **Absolute combined cap** (:class:`CombinedLedger`). Bank + confirmatory + secondaries <= :data:`CAP_GPU_H`
  (90 GPU-h-eq) and <= :data:`CAP_USD` ($1,000 at GPU-h x 4.89 x 1.10). Before any unit, if the recorded total plus
  the unit's priced estimate would exceed either cap, :class:`CapExceeded` is raised: the study stops ("engineering
  abort", never a chair override). Separately, a phase whose measured total would exceed 1.25x its priced value
  raises :class:`PhasePause` (the chair may resume from ledger seconds only, within the absolute cap).
* **Attempts vs relaunches** (:class:`RetryPolicy`). An *attempt* is one try of one unit (an E2 episode-arm, a bank
  state, a generation block); a unit gets at most :data:`UNIT_ATTEMPTS` = 3 attempts **in total, counted across
  relaunches**; after the third failure it takes its registered fallback (an E2 arm-episode -> the base winner,
  Part I §6.5.5) and is reported. A *relaunch* is a new instance for a phase after an infrastructure failure; at
  most :data:`PHASE_RELAUNCHES` = 1 per phase, resuming from content-addressed units. These are separate counters.
* **Repairs** (:meth:`RetryPolicy.repair`). Technical and outcome-blind only: the cause must be one of
  :data:`TECHNICAL_CAUSES`; the record carries a description, the code commit and an attestation that no outcome was
  read to decide it. No scientific redesign, no re-selection of winners, no change of margins, N, seeds or methods.
* **Canary eligibility** (:class:`CanaryBook`). Each instance runs the canary at start and end (and after any
  restart). A unit is *eligible* only if the last canary before it and the first canary after it on its instance
  both match the reference hash. Units scored before a failed canary are *retained, ineligible*; they are re-run on
  a canary-valid instance and **the first canary-valid attempt in time order is the eligible result**, whatever its
  value (:meth:`CanaryBook.eligible_result`) - no choice between runs.
"""

from __future__ import annotations

import json
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


class CapExceeded(RuntimeError):
    pass


class PhasePause(RuntimeError):
    pass


class RetryExhausted(RuntimeError):
    pass


class CombinedLedger:
    """Append-only JSONL ledger of measured GPU seconds per phase (one per 9B study)."""

    def __init__(self, path, priced_gpu_h: dict):
        self.path = Path(path)
        self.priced = dict(priced_gpu_h)
        self.rows = []
        if self.path.exists():
            self.rows = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def add(self, phase: str, gpu_seconds: float, note: str = ""):
        if phase not in self.priced:
            raise KeyError(f"unpriced phase {phase}")
        r = {"phase": phase, "gpu_s": float(gpu_seconds), "note": note}
        self.rows.append(r)
        with open(self.path, "a") as f:
            f.write(json.dumps(r) + "\n")

    def gpu_h(self, phase: str | None = None) -> float:
        return sum(r["gpu_s"] for r in self.rows if phase in (None, r["phase"])) / 3600

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
        if self.gpu_h(phase) + est_gpu_h > PHASE_PAUSE * self.priced[phase]:
            raise PhasePause(
                f"{phase}: above {PHASE_PAUSE} x its priced {self.priced[phase]} GPU-h"
            )


class UnitGuard:
    """The ``guard`` of ``harness.run_units``: cap checks before a unit, measured seconds after it."""

    def __init__(self, ledger: CombinedLedger, phase: str, est_unit_gpu_h: float, gpus: int = 1):
        self.ledger, self.phase, self.est, self.gpus = ledger, phase, float(est_unit_gpu_h), gpus

    def before_unit(self, label):
        self.ledger.check_before(self.phase, self.est)

    def after_unit(self, label, wall_seconds: float):
        self.ledger.add(self.phase, wall_seconds * self.gpus, note=str(label))


class RetryPolicy:
    def __init__(self):
        self.attempts: dict = {}
        self.relaunches: dict = {}
        self.repairs: list = []

    def attempt(self, unit: str, cause: str | None = None) -> int:
        """Record one attempt of ``unit`` (``cause`` of the previous failure if this is a retry)."""
        if cause is not None and cause not in TECHNICAL_CAUSES:
            raise ValueError(f"retry cause {cause!r} is not a registered technical cause")
        n = self.attempts.get(unit, 0) + 1
        if n > UNIT_ATTEMPTS:
            raise RetryExhausted(
                f"{unit}: {UNIT_ATTEMPTS} attempts used; take its registered fallback"
            )
        self.attempts[unit] = n
        return n

    def relaunch(self, phase: str, cause: str):
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"relaunch cause {cause!r} is not a registered technical cause")
        n = self.relaunches.get(phase, 0) + 1
        if n > PHASE_RELAUNCHES:
            raise RetryExhausted(f"{phase}: the one registered relaunch is used; engineering abort")
        self.relaunches[phase] = n
        return n

    def repair(self, cause: str, description: str, commit: str, outcome_blind: bool) -> dict:
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"repair cause {cause!r} is not a registered technical cause")
        if outcome_blind is not True:
            raise ValueError("repairs must be decided without reading any outcome")
        if not description or len(commit) < 7:
            raise ValueError("a repair needs a description and the code commit")
        r = {"cause": cause, "description": description, "commit": commit, "outcome_blind": True}
        self.repairs.append(r)
        return r


class CanaryBook:
    """Canary events [(instance, time, sha256)] against one reference hash, and unit eligibility."""

    def __init__(self, reference: str):
        self.reference = reference
        self.events: dict = {}

    def canary(self, instance: str, t: float, sha: str):
        self.events.setdefault(instance, []).append((float(t), sha))

    def eligible(self, instance: str, t: float) -> bool:
        ev = sorted(self.events.get(instance, []))
        before = [s for u, s in ev if u <= t]
        after = [s for u, s in ev if u > t]
        return bool(
            before and after and before[-1] == self.reference and after[0] == self.reference
        )

    def status(self, instance: str, t: float) -> str:
        return "eligible" if self.eligible(instance, t) else "retained_ineligible"

    def eligible_result(self, attempts):
        """``attempts`` [(instance, time, result)] of one unit: the first canary-valid attempt in time order, or
        None. Never chooses by value."""
        for inst, t, res in sorted(attempts, key=lambda a: a[1]):
            if self.eligible(inst, t):
                return res
        return None
