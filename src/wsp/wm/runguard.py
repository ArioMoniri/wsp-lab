"""Validity, repair and stopping contract of the 9B confirmatory runs, enforced in the production path
(PREREG_WME §II.13, §II.21.1, §II.22.6, §II.24.2, §II.25, §II.26, §II.27; Board A v2 change 6 and blocker B).

``harness.run_units`` takes a **mandatory** guard. Confirmatory phases use :class:`ProductionGuard`; engineering
smokes and toy workers must pass an explicit :class:`EngineeringGuard` (results are plain, never eligible evidence).

**One authoritative study log (§II.27).** A study state directory holds exactly one append-only, fsync'd log
(``study.jsonl``, :class:`StudyLog`). Every record carries a global, gap-free sequence number and the sha256 of the
previous record (a hash chain); ``study.jsonl.head`` holds the sequence and hash of the last appended record.
Everything the guard knows (the combined ledger and clock, attempt and relaunch counters, canaries, contract
generations, instance lifecycle, loss dispositions, fallbacks and the owner epoch) is a pure function of that log
(:class:`StudyState`, applied one record at a time). There are no other state files.

**Every public operation is serialised against the shared log.** It takes an exclusive OS lock (``fcntl.flock`` on
``study.jsonl.lock``), reads and verifies the log tail appended since the object's cached sequence, re-validates
against that current state, and only then appends, all under the same lock. Validation and mutation therefore use
one snapshot, a stale object automatically sees the current state, and sequence numbers stay globally unique.

**Owner epoch.** Constructing a :class:`ProductionGuard` makes it the production owner: it appends an
``owner_epoch`` record with the next epoch. Every operation of an object whose epoch is no longer the current one is
refused (:class:`StaleGuard`) before anything is read out or written. A contract transition always constructs a new
owner, so an older object can never act under a superseded contract generation.

**Torn or truncated logs fail closed.** A partial last line, an invalid record, a broken hash chain, a log shorter
than its head record or a changed already-read record raises :class:`LogCorrupt` on every operation. The only
automatic reconciliation is :func:`repair_torn_tail` for a partial last line (a crash mid-append): it quarantines the
torn bytes, cuts the log back to its last complete verified record and appends an audited ``log_repaired`` record,
which also ends the current ownership. Any other damage needs an audited manual restoration.

The registered rules enforced on that state (unchanged in substance from §II.22.6-§II.26):

* **Launch validity.** Construction requires a :class:`LaunchAttestation`. Before any unit the instance's latest
  canary under the current contract generation must equal the reference, otherwise :class:`CanaryMismatch`.
* **Attempts and relaunches.** At most :data:`UNIT_ATTEMPTS` = 3 attempts per unit in total, counted from the log
  across processes, objects, instances and relaunches; then the registered fallback. At most
  :data:`PHASE_RELAUNCHES` = 1 relaunch per phase.
* **Failures.** A failed attempt's measured seconds are charged. A failure is retried only if the classifier maps it
  to one of :data:`TECHNICAL_CAUSES`; anything else halts (re-raised, still charged).
* **Results and eligibility.** Each successful attempt is stored under its own key (``<unit>.a<n>``, n global per
  unit) with the sha256 of its payload. It is eligible only if the instance's last canary before the attempt started
  and its first canary after the result, both under the same instance, contract and generation and before the
  instance's closure, equal that generation's reference. ``resolve`` returns the first eligible attempt in log order
  (its payload digest verified), else the registered fallback, else nothing.
* **Instance lifecycle.** Units and canaries need a registered, active, not-lost instance (:class:`InstanceNotOpen`).
  Closing, loss or authoritative termination irreversibly disposes of attempts lacking a closing canary.
* **Clock.** Every reading goes through :meth:`CombinedLedger.tick`; a backward reading persists ``clock_halt`` and
  raises :class:`ClockInconsistent` until :meth:`CombinedLedger.clear_clock_halt` (authoritative termination of every
  instance active at the halt, and a recovered clock). Each instance is charged max(local, authoritative); no
  clamping.
* **Contract generations.** Every attestation or :class:`ContractTransition` is a new generation; attempts and
  canaries bind to (digest, generation); supersession is irreversible.
* **Absolute cap.** Charged total = max(measured unit GPU time, instance time); before every attempt, charged +
  the priced unit estimate must stay <= :data:`CAP_GPU_H` and <= :data:`CAP_USD` (:class:`CapExceeded`); a phase above
  1.25x its price raises :class:`PhasePause`.
* **Cap generations (PREREG_WME §II.31).** The cap in force is the last persisted ``cap_generation`` record (else the
  registered :data:`CAP_GPU_H`). A cap generation only raises the cap, is dated, names its decider and source (a raise
  after §II.31 is the chair's and must have been announced to the user), never lets the cap exceed :data:`CAP_USD` at
  :data:`USD_PER_GPU_H`, binds the contract generation it was recorded under, and is recorded before the units it
  funds. It is not a contract transition: no result or canary is superseded.
* **Run-health incidents (§II.31 run-health duties).** An ``incident`` record halts issuing: :meth:`ProductionGuard.begin`
  raises :class:`HealthHalt` (before any attempt is recorded) until an ``incidents_resolved`` record names the
  diagnosis and the chair report.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
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
GENESIS = "0" * 64
LOCK_TIMEOUT_S = 60.0
LOG_NAME = "study.jsonl"


class CapExceeded(RuntimeError):
    pass


class IncidentAbort(RuntimeError):
    """A run-health incident that aborts an in-flight attempt without the unit having failed (order, GPU idle,
    telemetry, progress; PREREG_WME §II.31.8 R4). The attempt is recorded ``attempt_aborted``: charged, never counted
    toward the attempt limit, never a failure cause."""

    incident_kind = "incident"


class RepairRequired(RuntimeError):
    """The unit's last failure had no registered technical cause: it is retried only after a ``repair`` record
    (registered cause, commit, outcome-blind attestation; §II.31.8 R3)."""


class HealthHalt(RuntimeError):
    """An open run-health incident: no unit is issued until it is diagnosed and reported (incidents_resolved)."""


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


class ClockInconsistent(RuntimeError):
    pass


class ContractMismatch(RuntimeError):
    pass


class FallbackMismatch(RuntimeError):
    pass


class StaleGuard(RuntimeError):
    """An operation by an object whose owner epoch (or contract generation) is no longer current."""


class LogCorrupt(RuntimeError):
    """The study log is torn, truncated or fails its hash chain: fail closed until reconciled."""


class StudyLocked(RuntimeError):
    """The study lock could not be acquired within :data:`LOCK_TIMEOUT_S`."""


class UnitResolved(RuntimeError):
    """An attempt was requested for a unit that already has an eligible result or its fallback."""


class ResultMismatch(RuntimeError):
    """A stored attempt payload does not match the digest recorded in the study log."""


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _digest(obj) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def record_hash(rec: dict) -> str:
    """sha256 of the canonical JSON of a log record without its own ``hash`` field."""
    return _digest({k: v for k, v in rec.items() if k != "hash"})


# ------------------------------------------------------------------ the derived state


class StudyState:
    """Everything known about a study, derived only by replaying its log, one record at a time."""

    def __init__(self):
        self.n = 0
        self.epoch = 0
        self.contracts: list = []  # attestation / contract_transition records; index = generation
        self.fallbacks: dict = {}  # phase -> fallback_registered record
        self.max_tick = float("-inf")
        self.halted = False
        self.halt_active: list = []
        self.authoritative_since_halt: set = set()
        self.opened: dict = {}  # instance -> [t, gpus] of its first opening
        self.closed: dict = {}  # instance -> t of its first local close
        self.auth: dict = {}  # instance -> max authoritative GPU-seconds
        self.closure: dict = {}  # instance -> seq of its first close / authoritative / loss record
        self.lost: dict = {}  # instance -> seq of its loss disposition
        self.unit_s: dict = {}  # phase -> measured GPU-seconds (successful and failed attempts)
        self.attempts: dict = {}  # unit -> attempts started
        self.inflight: dict = {}  # unit -> attempt_start record awaiting its outcome
        self.last_cause: dict = {}  # unit -> cause of its last failure
        self.relaunches: dict = {}  # phase -> relaunches used
        self.repairs: list = []
        self.canaries: dict = {}  # (instance, generation, contract) -> [[seq, sha]]
        self.ok: dict = {}  # unit -> [attempt_ok records] in log order
        self.disposed: set = set()  # attempt_ok seqs disposed irreversibly
        self.fallback_units: dict = {}  # unit -> its first fallback record
        self.caps: list = []  # cap_generation records, in log order
        self.aborted: dict = {}  # unit -> attempts aborted by a run-health incident (not counted, R4)
        self.last_fail_seq: dict = {}  # unit -> seq of its last attempt_fail
        self.incidents: list = []  # open run-health incidents

    @property
    def generation(self):
        return len(self.contracts) - 1 if self.contracts else None

    @property
    def contract(self):
        return self.contracts[-1]["digest"] if self.contracts else None

    def apply(self, r: dict):
        self.n = r["seq"]
        t = r["type"]
        if "t" in r:
            self.max_tick = max(self.max_tick, float(r["t"]))
        if t == "owner_epoch":
            self.epoch = r["epoch"]
        elif t == "log_repaired":
            self.epoch += 1  # a repair ends the current ownership
        elif t in ("attestation", "contract_transition"):
            self.contracts.append(r)
        elif t == "fallback_registered":
            self.fallbacks.setdefault(r["phase"], r)
        elif t == "clock_halt":
            self.halted, self.halt_active = True, list(r["active"])
            self.authoritative_since_halt = set()
        elif t == "clock_resolved":
            self.halted, self.halt_active = False, []
        elif t == "unit_time":
            self.unit_s[r["phase"]] = self.unit_s.get(r["phase"], 0.0) + float(r["gpu_s"])
        elif t == "instance_open":
            self.opened.setdefault(r["instance"], [float(r["t"]), int(r["gpus"])])
        elif t == "instance_close":
            self.closed.setdefault(r["instance"], float(r["t"]))
            self.closure.setdefault(r["instance"], r["seq"])
        elif t == "instance_authoritative":
            d = (float(r["terminated_at"]) - float(r["launched_at"])) * int(r["gpus"])
            self.auth[r["instance"]] = max(self.auth.get(r["instance"], 0.0), d)
            self.closure.setdefault(r["instance"], r["seq"])
            if self.halted:
                self.authoritative_since_halt.add(r["instance"])
        elif t == "instance_lost":
            self.lost.setdefault(r["instance"], r["seq"])
            self.closure.setdefault(r["instance"], r["seq"])
        elif t == "canary":
            key = (r["instance"], r["generation"], r["contract"])
            self.canaries.setdefault(key, []).append([r["seq"], r["sha"]])
        elif t == "attempt_start":
            self.attempts[r["unit"]] = self.attempts.get(r["unit"], 0) + 1
            self.inflight[r["unit"]] = r
        elif t == "attempt_fail":
            self.inflight.pop(r["unit"], None)
            self.last_cause[r["unit"]] = r["cause"]
            self.last_fail_seq[r["unit"]] = r["seq"]
        elif t == "attempt_aborted":
            self.inflight.pop(r["unit"], None)
            self.aborted[r["unit"]] = self.aborted.get(r["unit"], 0) + 1
        elif t == "attempt_ok":
            self.inflight.pop(r["unit"], None)
            self.ok.setdefault(r["unit"], []).append(r)
        elif t == "attempt_disposed":
            self.disposed.add(r["attempt_seq"])
        elif t == "fallback":
            self.fallback_units.setdefault(r["unit"], r)
        elif t == "relaunch":
            self.relaunches[r["phase"]] = self.relaunches.get(r["phase"], 0) + 1
        elif t == "repair":
            self.repairs.append(r)
        elif t == "cap_generation":
            self.caps.append(r)
        elif t == "incident":
            self.incidents.append(r)
        elif t == "incidents_resolved":
            self.incidents = []

    @property
    def cap_gpu_h(self) -> float:
        return float(self.caps[-1]["cap_gpu_h"]) if self.caps else CAP_GPU_H

    # -------------------------------------------------- queries
    def instance_state(self, instance: str) -> str:
        if instance in self.closed or instance in self.auth:
            return "closed"
        return "active" if instance in self.opened else "unknown"

    def active_instances(self) -> list:
        return sorted(i for i in self.opened if self.instance_state(i) == "active")

    def charges(self, now: float) -> dict:
        """{instance: charged GPU-hours} at ``now``: max(local open-to-close, authoritative); never negative."""
        out = {}
        for inst in set(self.opened) | set(self.auth):
            local = 0.0
            if inst in self.opened:
                t0, g = self.opened[inst]
                local = (self.closed.get(inst, now) - t0) * g
            if local < 0:
                raise ClockInconsistent(f"{inst}: negative local duration")
            out[inst] = max(local, self.auth.get(inst, 0.0)) / 3600
        return out

    def status(self, a: dict) -> str:
        """Eligibility of an ``attempt_ok`` record under the current contract generation."""
        if a["generation"] != self.generation or a["contract"] != self.contract:
            return "superseded_contract"
        if a["seq"] in self.disposed:
            return "retained_ineligible"
        ref = self.contracts[a["generation"]]["canary_reference"]
        close = self.closure.get(a["instance"])
        cans = [
            c
            for c in self.canaries.get((a["instance"], a["generation"], a["contract"]), [])
            if close is None or c[0] < close
        ]
        before = [sha for q, sha in cans if q < a["start_seq"]]
        after = [sha for q, sha in cans if q > a["seq"]]
        if not before or before[-1] != ref:
            return "retained_ineligible"
        if not after:
            return "retained_ineligible" if close is not None else "pending"
        return "eligible" if after[0] == ref else "retained_ineligible"

    def snapshot(self) -> dict:
        d = {}
        for k, v in vars(self).items():
            if isinstance(v, set):
                v = sorted(v)
            elif isinstance(v, dict):
                v = {str(kk): vv for kk, vv in v.items()}
            d[k] = v
        return json.loads(json.dumps(d, sort_keys=True, default=str))


# ------------------------------------------------------------------ the single authoritative log


class StudyLog:
    """The one append-only, hash-chained, fsync'd study log (``path``; in memory when ``path`` is None).

    Use :meth:`locked` around every read-validate-append: it takes the exclusive OS lock, then reads and verifies the
    tail appended since this object's cached sequence. :meth:`append` is only allowed under the lock."""

    def __init__(self, path=None, *, load: bool = True, lock_timeout: float = LOCK_TIMEOUT_S):
        self.path = Path(path) if path is not None else None
        self.lock_timeout = lock_timeout
        self._reset()
        self._depth = 0
        self._fd = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if load:
                with self.locked():
                    pass

    def _reset(self):
        self.records: list = []
        self.state = StudyState()
        self._offset = 0
        self._last_line = b""

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @property
    def head_path(self) -> Path:
        return self.path.with_name(self.path.name + ".head")

    @property
    def tail(self) -> str:
        return self.records[-1]["hash"] if self.records else GENESIS

    # -------------------------------------------------- locking
    def _acquire(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    os.close(fd)
                    raise StudyLocked(f"{self.lock_path}: lock not acquired") from None
                time.sleep(0.005)
        self._fd = fd

    def _release(self):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    @contextmanager
    def locked(self):
        """Exclusive lock + verified re-read of the tail; re-entrant within this object. Yields the state."""
        if self.path is not None and self._depth == 0:
            self._acquire()
            try:
                self._refresh()
            except BaseException:
                self._release()
                raise
        self._depth += 1
        try:
            yield self.state
        finally:
            self._depth -= 1
            if self.path is not None and self._depth == 0:
                self._release()

    # -------------------------------------------------- reading and verification
    def _read_head(self) -> dict:
        if not self.head_path.exists():
            return {"seq": 0, "hash": GENESIS}
        try:
            h = json.loads(self.head_path.read_text())
            return {"seq": int(h["seq"]), "hash": str(h["hash"])}
        except (ValueError, KeyError, TypeError) as exc:
            raise LogCorrupt(f"{self.head_path}: unreadable head record") from exc

    def _ingest(self, line: bytes):
        n = len(self.records) + 1
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise LogCorrupt(f"record {n} is not valid JSON") from exc
        if (
            not isinstance(rec, dict)
            or rec.get("seq") != n
            or rec.get("prev") != self.tail
            or rec.get("hash") != record_hash(rec)
        ):
            raise LogCorrupt(f"record {n}: sequence or hash chain broken")
        self.records.append(rec)
        self.state.apply(rec)
        self._offset += len(line)
        self._last_line = line

    def _refresh(self):
        head = self._read_head()
        if not self.path.exists():
            if self._offset or head["seq"]:
                raise LogCorrupt(f"{self.path} is missing")
            return
        with open(self.path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < self._offset:
                raise LogCorrupt(f"{self.path} is shorter than already read (truncated)")
            if self._last_line:
                f.seek(self._offset - len(self._last_line))
                if f.read(len(self._last_line)) != self._last_line:
                    raise LogCorrupt(f"{self.path}: an already-read record changed")
            f.seek(self._offset)
            data = f.read()
        if data and not data.endswith(b"\n"):
            raise LogCorrupt(
                f"{self.path}: a partial (torn) last record; reconcile with repair_torn_tail()"
            )
        for line in data.splitlines(keepends=True):
            self._ingest(line)
        if head["seq"] > len(self.records):
            raise LogCorrupt(f"{self.path} ends before its head record {head['seq']} (truncated)")
        if head["seq"] and self.records[head["seq"] - 1]["hash"] != head["hash"]:
            raise LogCorrupt(f"{self.path}: head record does not match the log")

    # -------------------------------------------------- appending
    def append(self, type_: str, **fields) -> dict:
        if self.path is not None and self._depth == 0:
            raise RuntimeError("append outside the study lock")
        rec = {**fields, "type": type_, "seq": len(self.records) + 1, "prev": self.tail}
        rec["hash"] = record_hash(rec)
        line = (_canon(rec) + "\n").encode()
        rec = json.loads(line)
        if self.path is not None:
            with open(self.path, "ab") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            tmp = self.head_path.with_name(self.head_path.name + ".tmp")
            with open(tmp, "w") as f:
                f.write(json.dumps({"seq": rec["seq"], "hash": rec["hash"]}))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.head_path)
            self._offset += len(line)
            self._last_line = line
        self.records.append(rec)
        self.state.apply(rec)
        return rec

    # -------------------------------------------------- the one automatic reconciliation
    def repair_torn_tail(self, source: str, description: str) -> dict:
        """Audited repair of a partial last line (a crash mid-append): quarantine the torn bytes to
        ``<log>.torn-<seq>``, cut the log back to its last complete record (which must verify in full) and append
        ``log_repaired``, which ends the current ownership. Any other damage stays fail-closed."""
        if not source or not description:
            raise ValueError("name the source and describe the repair")
        self._acquire()
        self._depth += 1
        try:
            data = self.path.read_bytes() if self.path.exists() else b""
            cut = data.rfind(b"\n") + 1
            torn = data[cut:]
            if not torn:
                raise ValueError("no torn last record to repair")
            self._reset()
            for line in data[:cut].splitlines(keepends=True):
                self._ingest(line)
            head = self._read_head()
            if head["seq"] > len(self.records) or (
                head["seq"] and self.records[head["seq"] - 1]["hash"] != head["hash"]
            ):
                raise LogCorrupt("the complete records do not reach the head; manual restoration")
            q = self.path.with_name(f"{self.path.name}.torn-{len(self.records) + 1}")
            with open(q, "wb") as f:
                f.write(torn)
                f.flush()
                os.fsync(f.fileno())
            with open(self.path, "r+b") as f:
                f.truncate(cut)
                f.flush()
                os.fsync(f.fileno())
            return self.append(
                "log_repaired",
                source=source,
                description=description,
                torn_bytes=len(torn),
                torn_sha256=hashlib.sha256(torn).hexdigest(),
                quarantine=q.name,
            )
        finally:
            self._depth -= 1
            self._release()


def repair_torn_tail(state_dir, source: str, description: str) -> dict:
    """Reconcile a study state directory whose log ends in a partial record (see :meth:`StudyLog.repair_torn_tail`)."""
    return StudyLog(Path(state_dir) / LOG_NAME, load=False).repair_torn_tail(source, description)


def read_study(state_dir) -> StudyState:
    """A verified, read-only replay of a study state directory (takes the lock; writes nothing)."""
    return StudyLog(Path(state_dir) / LOG_NAME).state


# ------------------------------------------------------------------ views over the log


class _View:
    def __init__(self, log=None, fence=None, ctx=None):
        self.log = log if isinstance(log, StudyLog) else StudyLog(log)
        self._fence = fence
        self._ctx = ctx or dict

    @contextmanager
    def _op(self):
        """A mutating or result-returning operation: lock, re-read, fence, then act on the yielded state."""
        with self.log.locked() as s:
            if self._fence is not None:
                self._fence()
            yield s

    def _read(self):
        return self.log.locked()

    def _append(self, type_, **fields):
        return self.log.append(type_, **{**self._ctx(), **fields})


class CombinedLedger(_View):
    """The combined ledger and clock of one 9B study: a view of the study log (``log``: a :class:`StudyLog` or a path).

    **Clock discipline (fail closed, §II.26).** Every clock reading goes through :meth:`tick`, which persists each
    new maximum. A reading below the recorded maximum persists a ``clock_halt`` and raises :class:`ClockInconsistent`;
    while halted every timed operation raises. Local instance times are ticks, so a local duration is never negative
    and never offsets another instance. An authoritative record (complete: launch and termination) can only add time:
    each instance is charged max(local duration, authoritative duration). :meth:`clear_clock_halt` requires a clock at
    or above the recorded maximum and an authoritative termination for every instance active when the clock halted.
    Nothing is clamped."""

    def __init__(self, log, priced_gpu_h: dict, clock=time.time, fence=None, ctx=None):
        super().__init__(log, fence, ctx)
        self.priced = dict(priced_gpu_h)
        self.clock = clock
        self.r8 = False  # §II.31.8 R8 applies under a §II.31 cap regime (ProductionGuard(cap=...)); else registered

    @property
    def rows(self) -> list:
        with self._read():
            return list(self.log.records)

    # -------------------------------------------------- clock
    def max_tick(self) -> float:
        with self._read() as s:
            return s.max_tick

    def halted(self) -> bool:
        with self._read() as s:
            return s.halted

    def tick(self) -> float:
        """A trusted clock reading: persisted if it is a new maximum; a backward reading halts (persisted)."""
        with self._op() as s:
            if s.halted:
                raise ClockInconsistent(
                    "the clock is halted; reconcile authoritatively and clear_clock_halt()"
                )
            now = float(self.clock())
            if now < s.max_tick:
                mt = s.max_tick
                self._append("clock_halt", observed=now, max_tick=mt, active=s.active_instances())
                raise ClockInconsistent(
                    f"clock reading {now} is earlier than the recorded {mt}; halted"
                )
            if now > s.max_tick:
                self._append("tick", t=now)
            return now

    def _trusted_now(self) -> float:
        """The last trusted time while halted (never a reading from the inconsistent clock)."""
        with self._op() as s:
            return s.max_tick if s.halted else self.tick()

    def clear_clock_halt(self, source: str, description: str):
        with self._op() as s:
            if not s.halted:
                raise ValueError("the clock is not halted")
            if not source or not description:
                raise ValueError("name the authoritative source and describe the clock repair")
            need = set(s.halt_active) - s.authoritative_since_halt
            if need:
                raise ClockInconsistent(f"authoritative termination missing for {sorted(need)}")
            if float(self.clock()) < s.max_tick:
                raise ClockInconsistent("the clock is still behind the recorded maximum")
            self._append("clock_resolved", source=source, description=description)
            self.tick()

    # -------------------------------------------------- units
    def add(self, phase: str, gpu_seconds: float, note: str = "", status: str = "ok"):
        with self._op():
            if phase not in self.priced:
                raise KeyError(f"unpriced phase {phase}")
            if gpu_seconds < 0:
                raise ClockInconsistent("a negative measured duration")
            self._append(
                "unit_time", phase=phase, gpu_s=float(gpu_seconds), note=note, status=status
            )

    # -------------------------------------------------- instances
    def instance_state(self, instance: str) -> str:
        """``unknown``, ``active`` or ``closed`` (a local close or an authoritative termination; final)."""
        with self._read() as s:
            return s.instance_state(instance)

    def active_instances(self) -> list:
        with self._read() as s:
            return s.active_instances()

    def instance_open(self, instance: str, gpus: int) -> bool:
        """Idempotent for an active instance (the earliest opening is kept); a closed id is never reopened."""
        with self._op() as s:
            st = s.instance_state(instance)
            if st == "closed":
                raise ValueError(f"{instance} is closed; a replacement instance needs a new id")
            t = self.tick()
            if st == "active":
                self._append("instance_resume", instance=instance)
                return False
            self._append("instance_open", instance=instance, gpus=int(gpus), t=t)
            return True

    def instance_close(self, instance: str):
        with self._op() as s:
            if s.instance_state(instance) != "active":
                raise ValueError(f"{instance} is not active")
            self._append("instance_close", instance=instance, t=self.tick())

    @staticmethod
    def check_authoritative(launched_at, terminated_at, source):
        if not source:
            raise ValueError("name the authoritative source")
        if terminated_at is None or launched_at is None:
            raise ValueError("an authoritative record needs both launch and termination")
        if float(terminated_at) < float(launched_at):
            raise ClockInconsistent("authoritative termination before launch")

    def reconcile_instance(
        self, instance, launched_at, terminated_at, gpus, source, close_at: float | None = None
    ):
        """A complete authoritative record (provider API or dead-man terminator ledger). It closes the instance
        (locally at the last trusted time) and can only add time: the instance is charged max(local, authoritative).
        Allowed while the clock is halted (it is the way out of a halt)."""
        self.check_authoritative(launched_at, terminated_at, source)
        with self._op() as s:
            if s.instance_state(instance) == "active":
                t = self._trusted_now() if close_at is None else float(close_at)
                self._append("instance_close", instance=instance, t=t)
            self._append(
                "instance_authoritative",
                instance=instance,
                launched_at=float(launched_at),
                terminated_at=float(terminated_at),
                gpus=int(gpus),
                source=source,
            )

    def instance_charges(self, now: float | None = None) -> dict:
        """{instance: charged GPU-hours}; every term is non-negative by construction."""
        if now is None:
            now = self.tick()
        with self._read() as s:
            if float(now) < s.max_tick:
                raise ClockInconsistent("a query time earlier than the recorded maximum")
            return s.charges(float(now))

    def unit_gpu_h(self, phase: str | None = None) -> float:
        with self._read() as s:
            if phase is not None:
                return s.unit_s.get(phase, 0.0) / 3600
            return sum(s.unit_s.values()) / 3600

    def instance_gpu_h(self, now: float | None = None) -> float:
        return sum(self.instance_charges(now).values())

    def gpu_h(self, phase: str | None = None) -> float:
        """Phase totals are measured unit time; the study total is the larger of unit and instance time."""
        if phase is not None:
            return self.unit_gpu_h(phase)
        return max(self.unit_gpu_h(), self.instance_gpu_h())

    def usd(self) -> float:
        return self.gpu_h() * USD_PER_GPU_H

    def cap_gpu_h(self) -> float:
        """The cap in force: the last persisted cap generation, else the registered :data:`CAP_GPU_H`."""
        with self._read() as s:
            return s.cap_gpu_h

    def cap_generations(self) -> list:
        with self._read() as s:
            return [dict(r) for r in s.caps]

    def remaining_gpu_h(self) -> float:
        """GPU-h-eq the cap in force still allows (the dead-man allowance's GPU-hours)."""
        return max(0.0, min(self.cap_gpu_h() - self.gpu_h(), CAP_USD / USD_PER_GPU_H - self.gpu_h()))

    def remaining_usd(self) -> float:
        """For the per-instance dead-man ``max_usd``: never more than what the combined cap leaves."""
        return max(0.0, min(CAP_USD - self.usd(), (self.cap_gpu_h() - self.gpu_h()) * USD_PER_GPU_H))

    def record_cap(self, cap: CapGeneration, contract=None, generation=None) -> dict | None:
        """Persist ``cap`` as a new cap generation if it is above the cap in force (idempotent if equal; a lower cap
        raises). Returns the record, or None if nothing was written."""
        cap.check()
        with self._op() as s:
            cur = s.cap_gpu_h
            if cap.cap_gpu_h < cur:
                raise ValueError(f"a cap generation never lowers the cap ({cap.cap_gpu_h} < {cur})")
            if cap.cap_gpu_h == cur and (s.caps or cap.cap_gpu_h == CAP_GPU_H):
                return None
            return self._append(
                "cap_generation",
                index=len(s.caps),
                previous_cap_gpu_h=cur,
                contract=contract,
                contract_generation=generation,
                **cap.fields(),
            )

    def priced_remainder(self) -> float:
        """GPU-h-eq of the registered design not yet measured: sum over priced phases of max(0, price - measured)."""
        return sum(max(0.0, float(v) - self.unit_gpu_h(p)) for p, v in self.priced.items())

    def check_before(self, phase: str, est_gpu_h: float, contract=None, generation=None):
        with self._op():
            tot = self.gpu_h() + est_gpu_h
            cap = self.cap_gpu_h()
            if self.r8 and tot > cap and tot * USD_PER_GPU_H <= CAP_USD:
                # §II.31.8 R8: the mechanical raise, from ledger seconds only (never a discretionary decision)
                basis = {
                    "charged_gpu_h": self.gpu_h(),
                    "priced_remainder_gpu_h": self.priced_remainder(),
                    "previous_cap_gpu_h": cap,
                }
                new = r8_cap(basis)
                if new is not None and new > cap:
                    now = self._trusted_now()
                    self.record_cap(
                        CapGeneration(
                            cap_gpu_h=new,
                            date=time.strftime("%Y-%m-%d", time.gmtime(now)),
                            decided_by="chair",
                            source="PREREG_WME §II.31.8 R8 (mechanical cap raise)",
                            announced_to_user=False,  # a notification owed to the user, not a gate
                            rule="R8",
                            basis=basis,
                            outcome_blind=True,
                        ),
                        contract=contract,
                        generation=generation,
                    )
                    cap = self.cap_gpu_h()
            if tot > cap or tot * USD_PER_GPU_H > CAP_USD:
                raise CapExceeded(
                    f"combined cap: {tot - est_gpu_h:.3f} + {est_gpu_h:.3f} GPU-h-eq > {cap} or $ > {CAP_USD}"
                )
            if self.unit_gpu_h(phase) + est_gpu_h > PHASE_PAUSE * self.priced[phase]:
                raise PhasePause(
                    f"{phase}: above {PHASE_PAUSE} x its priced {self.priced[phase]} GPU-h"
                )


class RetryPolicy(_View):
    """Attempt and relaunch counters and repair records: a view of the study log (in memory when ``log`` is None)."""

    @property
    def attempts(self) -> dict:
        with self._read() as s:
            return dict(s.attempts)

    @property
    def relaunches(self) -> dict:
        with self._read() as s:
            return dict(s.relaunches)

    @property
    def repairs(self) -> list:
        with self._read() as s:
            return list(s.repairs)

    def attempt(self, unit: str, cause: str | None = None, **fields) -> int:
        if cause is not None and cause not in TECHNICAL_CAUSES:
            raise ValueError(f"retry cause {cause!r} is not a registered technical cause")
        with self._op() as s:
            n = s.attempts.get(unit, 0) + 1  # attempt numbers stay unique (attempt keys)
            counted = s.attempts.get(unit, 0) - s.aborted.get(unit, 0)  # incident aborts never count (R4)
            if counted + 1 > UNIT_ATTEMPTS:
                raise RetryExhausted(
                    f"{unit}: {UNIT_ATTEMPTS} attempts used; take its registered fallback"
                )
            self._append("attempt_start", unit=unit, n=n, cause=cause, **fields)
            return n

    def relaunch(self, phase: str, cause: str, **fields) -> int:
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"relaunch cause {cause!r} is not a registered technical cause")
        with self._op() as s:
            n = s.relaunches.get(phase, 0) + 1
            if n > PHASE_RELAUNCHES:
                raise RetryExhausted(
                    f"{phase}: the one registered relaunch is used; engineering abort"
                )
            self._append("relaunch", phase=phase, n=n, cause=cause, **fields)
            return n

    def repair(self, cause: str, description: str, commit: str, outcome_blind: bool) -> dict:
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"repair cause {cause!r} is not a registered technical cause")
        if outcome_blind is not True:
            raise ValueError("repairs must be decided without reading any outcome")
        if not description or len(commit) < 7:
            raise ValueError("a repair needs a description and the code commit")
        with self._op():
            return self._append(
                "repair", cause=cause, description=description, commit=commit, outcome_blind=True
            )


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

    def validate(self):
        return None

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
class CapGeneration:
    """One dated cap decision (PREREG_WME §II.31, §II.31.8 R8): ``decided_by`` "user" (the §II.31 decision, rule
    "user") or "chair" (a later raise, only by the mechanical rule "R8": its ledger-only ``basis`` must reproduce the
    cap, and ``outcome_blind`` attests that no outcome was read). Announcing a raise to the user is a notification,
    not a gate."""

    cap_gpu_h: float
    date: str
    decided_by: str
    source: str
    announced_to_user: bool
    rule: str = "user"
    basis: dict | None = None
    outcome_blind: bool = True

    def check(self):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", self.date or ""):
            raise ValueError("a cap generation needs an ISO date")
        if self.decided_by not in ("user", "chair"):
            raise ValueError("a cap generation is decided by the user or the chair")
        if not self.source:
            raise ValueError("a cap generation names its source (amendment / ruling)")
        if self.outcome_blind is not True:
            raise ValueError("a cap generation is decided without reading any outcome")
        if self.decided_by == "user" and self.rule != "user":
            raise ValueError("the user's cap decision carries rule 'user'")
        if self.decided_by == "chair":
            if self.rule != "R8" or not isinstance(self.basis, dict):
                raise ValueError("a later raise is only the mechanical R8 raise, with its ledger basis")
            want = r8_cap(self.basis)
            if want is None or abs(float(self.cap_gpu_h) - want) > 1e-9:
                raise ValueError(f"cap {self.cap_gpu_h} is not the R8 value {want} for basis {self.basis}")
        if not (CAP_GPU_H <= float(self.cap_gpu_h)) or float(self.cap_gpu_h) * USD_PER_GPU_H > CAP_USD:
            raise ValueError(
                f"cap {self.cap_gpu_h} GPU-h-eq: below the registered {CAP_GPU_H} or beyond ${CAP_USD} "
                f"at {USD_PER_GPU_H:.3f}/GPU-h"
            )

    def fields(self) -> dict:
        return {
            "cap_gpu_h": float(self.cap_gpu_h),
            "date": self.date,
            "decided_by": self.decided_by,
            "source": self.source,
            "announced_to_user": bool(self.announced_to_user),
            "rule": self.rule,
            "basis": None if self.basis is None else dict(self.basis),
            "outcome_blind": bool(self.outcome_blind),
        }


def r8_cap(basis: dict) -> float | None:
    """§II.31.8 R8 (Board A, accepted by the chair): when the cap would stop the run and combined spend plus the
    priced remainder of the registered design is within $1,000 (ledger convention), the cap is raised by 1.25 x the
    priced remainder, or to the $1,000 ceiling, whichever is lower. None: no raise (the run stops at the cap)."""
    charged = float(basis["charged_gpu_h"])
    rem = float(basis["priced_remainder_gpu_h"])
    prev = float(basis["previous_cap_gpu_h"])
    if (charged + rem) * USD_PER_GPU_H > CAP_USD:
        return None
    return min(prev + PHASE_PAUSE * rem, CAP_USD / USD_PER_GPU_H)


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


def _att_fields(att: LaunchAttestation) -> dict:
    d = {k: getattr(att, k) for k in att.__dataclass_fields__}
    d["evidence"] = sorted(d["evidence"])
    return d


def contract_digest(att: LaunchAttestation) -> str:
    """The registered run contract: every attestation field (evidence order-free)."""
    return _digest(_att_fields(att))


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
    """The mandatory confirmatory guard and production owner of one study state directory (module docstring)."""

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
        cap: CapGeneration | None = None,
    ):
        attestation.check()
        if cap is not None:
            cap.check()
        self._cap_regime = cap is not None
        self.state = Path(state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        legacy = [
            n for n in ("events.jsonl", "ledger.jsonl", "retry.jsonl") if (self.state / n).exists()
        ]
        if legacy:
            raise ValueError(f"{self.state}: pre-§II.27 state files {legacy}; one study log only")
        self.phase, self.instance, self.est = phase, instance, float(est_unit_gpu_h)
        self.attestation = attestation
        self.technical = technical or (lambda exc: None)
        self.clock = clock
        self.epoch = self.contract = self.generation = None
        self.log = StudyLog(self.state / LOG_NAME)
        ctx = lambda: {"epoch": self.epoch, "phase": self.phase}
        self.ledger = CombinedLedger(
            self.log, priced_gpu_h, clock=clock, fence=self._fence, ctx=ctx
        )
        self.retry = RetryPolicy(self.log, fence=self._fence, ctx=ctx)
        self.ledger.r8 = self._cap_regime
        try:
            fb_digest = _digest(fallback_value)
        except TypeError as exc:
            raise ValueError("the registered fallback must be JSON-serialisable") from exc
        new = contract_digest(attestation)
        with self.log.locked() as s:
            # validate everything first; a refused construction writes nothing
            if s.contracts and s.contract != new:
                if transition is None:
                    raise ContractMismatch(
                        f"persisted contract {s.contract[:12]} != {new[:12]}: refuse to run or return "
                        "results without an explicit ContractTransition"
                    )
                transition.check()
            elif transition is not None:
                raise ValueError("a ContractTransition was given but there is no other contract")
            reg = s.fallbacks.get(phase)
            if reg is not None and reg["payload_sha256"] != fb_digest:
                raise FallbackMismatch(
                    f"registered fallback {reg['payload_sha256'][:12]} != {fb_digest[:12]}; refuse"
                )
            # take ownership, then record the contract and the fallback under it
            self.epoch = s.epoch + 1
            self.log.append("owner_epoch", epoch=self.epoch, phase=phase, instance=instance)
            if not s.contracts:
                self._append("attestation", digest=new, generation=0, **_att_fields(attestation))
            elif s.contract != new:
                self._append(
                    "contract_transition",
                    digest=new,
                    generation=len(s.contracts),
                    previous=s.contract,
                    previous_generation=s.generation,
                    cause=transition.cause,
                    description=transition.description,
                    commit=transition.commit,
                    outcome_blind=True,
                    **_att_fields(attestation),
                )
            if reg is None:
                self._append(
                    "fallback_registered", payload=fallback_value, payload_sha256=fb_digest
                )
            self.contract, self.generation = s.contract, s.generation
            if cap is not None:
                self.ledger.record_cap(cap, contract=s.contract, generation=s.generation)
            self.fallback_payload = s.fallbacks[phase]["payload"]
            self.fallback_sha256 = fb_digest

    # -------------------------------------------------- serialisation and fencing
    def _fence(self):
        s = self.log.state
        if self.epoch is None:
            return
        if s.epoch != self.epoch:
            raise StaleGuard(
                f"owner epoch {self.epoch} is superseded by {s.epoch}; construct a new guard"
            )
        if s.generation != self.generation or s.contract != self.contract:
            raise StaleGuard("the contract generation changed; construct a new guard")

    @contextmanager
    def _op(self):
        with self.log.locked() as s:
            self._fence()
            yield s

    def _append(self, type_, **fields):
        return self.log.append(type_, **{"epoch": self.epoch, "phase": self.phase, **fields})

    @property
    def events(self) -> list:
        """The current study log (re-read and verified)."""
        with self.log.locked():
            return list(self.log.records)

    # -------------------------------------------------- lifecycle
    def _require_active(self, s):
        self.ledger.tick()  # fail closed on an inconsistent clock
        if s.instance_state(self.instance) != "active" or self.instance in s.lost:
            raise InstanceNotOpen(f"{self.instance}: no validated open instance")

    def validate(self):
        """Owner epoch, log integrity, contract generation, clock and lifecycle; ``run_units`` calls it first."""
        with self._op() as s:
            self._require_active(s)

    def open_instance(self, gpus: int = 1):
        with self._op() as s:
            if self.instance in s.lost:
                raise InstanceNotOpen(
                    f"{self.instance} was declared lost; relaunch on a new instance"
                )
            self.ledger.instance_open(self.instance, gpus)

    def _dispose_unbracketed(self, s, instance, reason):
        """Irreversibly mark attempts on ``instance`` lacking a closing canary as retained-ineligible."""
        pend = [
            a
            for oks in s.ok.values()
            for a in oks
            if a["instance"] == instance and s.status(a) == "pending"
        ]
        for a in sorted(pend, key=lambda a: a["seq"]):
            self._append(
                "attempt_disposed",
                instance=instance,
                unit=a["unit"],
                attempt_seq=a["seq"],
                reason=reason,
            )

    def close_instance(self):
        with self._op() as s:
            self._require_active(s)
            self._dispose_unbracketed(s, self.instance, "closed without a closing canary")
            self.ledger.instance_close(self.instance)

    def reconcile_instance(self, launched_at, terminated_at, gpus, source, instance=None):
        inst = instance or self.instance
        CombinedLedger.check_authoritative(launched_at, terminated_at, source)
        with self._op() as s:
            close_at = self.ledger._trusted_now() if s.instance_state(inst) == "active" else None
            self._dispose_unbracketed(s, inst, f"authoritatively terminated ({source})")
            self.ledger.reconcile_instance(
                inst, launched_at, terminated_at, gpus, source, close_at=close_at
            )

    def declare_lost(
        self,
        instance: str,
        cause: str,
        description: str,
        commit: str,
        outcome_blind: bool,
        launched_at=None,
        terminated_at=None,
        source: str = "operator: provider termination record",
    ):
        """Technical disposition for an instance lost before its closing canary (preemption, crash). Its unbracketed
        attempts become retained-ineligible irreversibly (no canary is fabricated, and no canary can be recorded for
        it afterwards); attempt counts stay; its time is charged (max of local and the termination record)."""
        if cause not in TECHNICAL_CAUSES:
            raise ValueError(f"loss cause {cause!r} is not a registered technical cause")
        if outcome_blind is not True:
            raise ValueError("a loss disposition must be decided without reading any outcome")
        if not description or len(commit) < 7:
            raise ValueError("a loss disposition needs a description and the code commit")
        with self._op() as s:
            if instance in s.lost:
                raise ValueError(f"{instance} is already declared lost")
            active = s.instance_state(instance) == "active"
            if active:
                now = self.ledger._trusted_now()
                t0 = s.opened[instance][0] if launched_at is None else launched_at
                end = now if terminated_at is None else terminated_at
                CombinedLedger.check_authoritative(t0, end, source)
            self._dispose_unbracketed(s, instance, f"lost: {cause}")
            if active:
                self.ledger.reconcile_instance(
                    instance, t0, end, s.opened[instance][1], source, close_at=now
                )
            self._append(
                "instance_lost",
                instance=instance,
                cause=cause,
                description=description,
                commit=commit,
                outcome_blind=True,
                terminated_at=terminated_at,
            )

    def canary(self, sha: str) -> bool:
        with self._op() as s:
            self._require_active(s)  # never for a closed, lost or unregistered instance
            self._append(
                "canary",
                instance=self.instance,
                generation=self.generation,
                contract=self.contract,
                sha=sha,
            )
            return sha == s.contracts[self.generation]["canary_reference"]

    # -------------------------------------------------- run health (§II.31)
    def incident(self, kind: str, detail: dict) -> dict:
        """Persist a run-health incident; issuing halts until :meth:`resolve_incidents`."""
        if not kind:
            raise ValueError("name the incident kind")
        with self._op():
            return self._append("incident", kind=kind, detail=detail, instance=self.instance)

    def open_incidents(self) -> list:
        with self.log.locked() as s:
            return [dict(r) for r in s.incidents]

    def resolve_incidents(self, diagnosis: str, report: str) -> dict:
        """Close every open incident after diagnosis and the report to the chair (both named)."""
        if not diagnosis or not report:
            raise ValueError("name the diagnosis and the chair report")
        with self._op() as s:
            if not s.incidents:
                raise ValueError("no open incident")
            return self._append(
                "incidents_resolved",
                diagnosis=diagnosis,
                report=report,
                incidents=[r["seq"] for r in s.incidents],
            )

    def relaunch(self, cause: str, new_instance: str):
        with self._op():
            self.ledger.tick()
            self.retry.relaunch(
                self.phase, cause, instance=self.instance, new_instance=new_instance
            )
            self.instance = new_instance

    # -------------------------------------------------- eligibility (current state only)
    def attempt_status(self, a) -> str:
        with self.log.locked() as s:
            return s.status(a)

    def unit_status(self, key) -> dict:
        with self.log.locked() as s:
            return {
                "attempts": [(a["attempt_key"], s.status(a)) for a in s.ok.get(key, [])],
                "fallback": key in s.fallback_units,
            }

    # -------------------------------------------------- unit protocol used by harness.run_units
    def _resolved(self, s, store, key):
        for a in s.ok.get(key, []):
            if s.status(a) == "eligible":
                payload = store.read(a["attempt_key"])
                if _digest(payload) != a["payload_sha256"]:
                    raise ResultMismatch(f"{a['attempt_key']}: stored payload != logged digest")
                return payload
        fb = s.fallback_units.get(key)
        if fb is not None:
            reg = s.fallbacks[fb["phase"]]
            if (
                fb["payload_sha256"] != reg["payload_sha256"]
                or _digest(fb["payload"]) != fb["payload_sha256"]
            ):
                raise FallbackMismatch(f"{key}: persisted fallback does not verify")
            return {
                "fallback": True,
                "value": fb["payload"],
                "payload_sha256": fb["payload_sha256"],
            }
        return None

    def resolve(self, store, key):
        with self._op() as s:
            if s.halted:
                raise ClockInconsistent(
                    "the clock is halted; no result is returned until it is cleared"
                )
            return self._resolved(s, store, key)

    def begin(self, key, label) -> int:
        with self._op() as s:
            if s.incidents:
                k = [i["kind"] for i in s.incidents]
                raise HealthHalt(f"{label}: open run-health incidents {k}; diagnose and report first")
            self._require_active(s)
            cans = s.canaries.get((self.instance, self.generation, self.contract), [])
            if not cans or cans[-1][1] != s.contracts[self.generation]["canary_reference"]:
                raise CanaryMismatch(
                    f"{self.instance}: no passing canary since the last unit; halt for repair"
                )
            if key in s.fallback_units or any(s.status(a) == "eligible" for a in s.ok.get(key, [])):
                raise UnitResolved(f"{label}: already has an eligible result or its fallback")
            if any(s.status(a) == "pending" for a in s.ok.get(key, [])):
                raise CanaryMismatch(
                    f"{label}: an attempt is pending its closing canary; close it (or declare its instance lost)"
                )
            fs = s.last_fail_seq.get(key)
            if (
                fs is not None
                and s.last_cause.get(key) not in TECHNICAL_CAUSES
                and not any(rp["seq"] > fs for rp in s.repairs)
            ):
                raise RepairRequired(
                    f"{label}: its last failure had no registered technical cause; retry only after a repair record"
                )
            self.ledger.check_before(
                self.phase, self.est, contract=self.contract, generation=self.generation
            )
            return self.retry.attempt(
                key,
                s.last_cause.get(key),
                label=label,
                instance=self.instance,
                generation=self.generation,
                contract=self.contract,
            )

    def _inflight(self, s, key, n=None):
        st = s.inflight.get(key)
        if st is None or st["epoch"] != self.epoch or (n is not None and st["n"] != n):
            raise StaleGuard(f"{key}: no in-flight attempt of this owner")
        return st

    def failed(self, key, label, seconds, exc):
        if isinstance(exc, IncidentAbort):  # R4: charged, not a failure, never counted toward the limit
            with self._op() as s:
                st = self._inflight(s, key)
                self.ledger.add(self.phase, seconds, note=f"{label} aborted", status="aborted")
                self._append(
                    "attempt_aborted",
                    unit=key,
                    label=label,
                    n=st["n"],
                    start_seq=st["seq"],
                    seconds=seconds,
                    incident=getattr(exc, "incident_kind", "incident"),
                )
            raise exc
        with self._op() as s:
            st = self._inflight(s, key)
            self.ledger.add(self.phase, seconds, note=f"{label} failed", status="failed")
            cause = self.technical(exc)
            self._append(
                "attempt_fail",
                unit=key,
                label=label,
                n=st["n"],
                start_seq=st["seq"],
                seconds=seconds,
                cause=cause,
                error=type(exc).__name__,
            )
        if cause not in TECHNICAL_CAUSES:
            raise exc

    def succeeded(self, store, key, label, n, payload, seconds, theta0):
        with self._op() as s:
            st = self._inflight(s, key, n)
            self.ledger.add(self.phase, seconds, note=str(label))  # charged even if refused below
            self._require_active(s)  # the instance is still active and the clock consistent
            akey = f"{key}.a{n}"
            store.write(akey, payload, theta_after=theta0)
            self._append(
                "attempt_ok",
                unit=key,
                label=label,
                n=n,
                attempt_key=akey,
                start_seq=st["seq"],
                instance=st["instance"],
                generation=st["generation"],
                contract=st["contract"],
                payload_sha256=_digest(payload),
                seconds=seconds,
            )
        return {"status": "pending_canary", "attempt_key": akey}

    def fallback(self, key, label):
        with self._op() as s:
            fb = s.fallback_units.get(key)
            if fb is None:
                if s.attempts.get(key, 0) < UNIT_ATTEMPTS:
                    raise ValueError(f"{label}: attempts remain; no fallback yet")
                if any(s.status(a) == "eligible" for a in s.ok.get(key, [])):
                    raise UnitResolved(f"{label}: has an eligible result")
                reg = s.fallbacks[self.phase]
                fb = self._append(
                    "fallback",
                    unit=key,
                    label=label,
                    payload=reg["payload"],
                    payload_sha256=reg["payload_sha256"],
                )
            return {
                "fallback": True,
                "value": fb["payload"],
                "payload_sha256": fb["payload_sha256"],
            }

    def report(self) -> dict:
        with self._op() as s:
            units = sorted(set(s.ok) | set(s.attempts) | set(s.fallback_units))
            return {
                "epoch": s.epoch,
                "contract": s.contract,
                "generation": s.generation,
                "units": {u: self.unit_status(u) for u in units},
                "charged_gpu_h": self.ledger.gpu_h(),
                "unit_gpu_h": self.ledger.unit_gpu_h(),
                "instance_gpu_h": self.ledger.instance_gpu_h(),
                "relaunches": dict(s.relaunches),
                "lost_instances": sorted(s.lost),
            }
