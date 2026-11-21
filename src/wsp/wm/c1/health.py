"""Run-health duties (PREREG_WME §II.31; chair ruling ``RULING-wme-c1-phasel.md`` addendum). Engineering only: no
rule here reads an outcome.

Worker side
    :func:`gpu_sample` reads GPU utilisation and memory (NVML through ``pynvml`` when importable, else
    ``nvidia-smi``). :class:`Sampler` is a daemon thread in the worker that samples every ``period_s`` and writes the
    heartbeat file :data:`HEARTBEAT` (``t``, ``gpu`` [{index, util_pct, mem_used_mib, mem_total_mib}], ``low_since``
    = start of the current run of samples with max utilisation < :data:`UTIL_LOW_PCT`, and ``inflight`` = the request
    being handled, {id, kind, since}).

Controller side
    :class:`HealthMonitor` turns observations into incidents, each persisted through
    ``ProductionGuard.incident`` (the guard then refuses to issue any unit, ``runguard.HealthHalt``, until the
    incident is diagnosed and reported to the chair, ``resolve_incidents``):

    * ``gpu_idle``: a unit of a GPU kind (:data:`KIND_CLASS`) in flight and GPU utilisation below 20 % for more than
      15 min (worker clock: ``t - max(low_since, inflight.since) > 900 s``); a heartbeat without GPU telemetry for
      that long is ``gpu_telemetry_missing``;
    * ``no_progress``: a unit of a CPU kind (E2 planner search, fits; addendum 2, answer 3) with no progress
      (``Ctx.check`` / ``Ctx.progress``) for more than 15 min;
    * ``worker_error``: an error result, a traceback in a result payload, or a traceback / error line in the
      worker log;
    * ``canary_mismatch``: a canary that does not equal the reference;
    * ``order``: a unit kind that is not registered, or whose §II.18 stage is not the current one or the next
      (going back, or skipping a stage that was not declared skipped).

    :meth:`HealthMonitor.chunk_summary` writes the health summary file at every chunk boundary.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

HEARTBEAT = "heartbeat.json"
POLL_MIN_S = 60.0  # controller heartbeat reads at most once a minute while a unit is in flight
UTIL_LOW_PCT = 20.0
IDLE_LIMIT_S = 15 * 60.0
SAMPLE_PERIOD_S = 30.0

# §II.18 "after the freeze": the registered stage order and the unit kinds of each stage
STAGES = (
    ("bank_field", ("field_group", "field_vocab", "bank_base")),
    ("bank_states", ("bank_side", "bank_comp", "bank_inert")),
    ("fits", ("library", "dev_base", "fit")),
    ("e2", ("e2_arm",)),
    ("sbon", ("sbon_arm",)),
    ("sealed", ("sealed_eval",)),
    ("e1_gen", ("e1_gen",)),
    ("ste_calibrate", ("ste_calibrate",)),
    ("ste_train", ("ste_train",)),
    ("ste_generate", ("ste_generate",)),
)
STAGE_OF = {k: i for i, (_s, kinds) in enumerate(STAGES) for k in kinds}
NO_PROGRESS_LIMIT_S = 15 * 60.0
# RULING-wme-c1-phasel addendum 2, answer 3: the GPU-idle rule applies only to kinds that use the GPU; CPU kinds (the
# E2 planner search, the fits; the analysis runs on the controller, no worker unit) are checked by a progress
# heartbeat instead. Every handler kind of the worker registry is classified (tests enforce it).
GPU_KINDS = (
    "canary",
    "field_group",
    "field_vocab",
    "bank_base",
    "bank_side",
    "bank_comp",
    "bank_inert",
    "library",
    "dev_base",
    "dev_eval",
    "sbon_arm",
    "sealed_eval",
    "e1_gen",
    "ste_calibrate",
    "ste_train",
    "ste_generate",
    "inert_identity",
    "parity",
    "gen_repeat",
    "timing",
)
# non-unit requests on the GPU: the K3 field repair (RULING-wme-c1-bank-halt)
GPU_KINDS = (*GPU_KINDS, "field_repair")
CPU_KINDS = ("fit", "e2_arm", "theta_of", "describe", "edit_bundles", "rstats", "hello")
KIND_CLASS = {**dict.fromkeys(GPU_KINDS, "gpu"), **dict.fromkeys(CPU_KINDS, "cpu")}
TRACE_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*(?:[\w.]+(?:Error|Exception)|CUDA error|RuntimeError)\b[:\s]",
    re.MULTILINE,
)


from wsp.wm import runguard as RG


class PostCompletionError(RuntimeError):
    """A traceback found after the unit's ``fn`` returned a payload (in the payload or the worker log). Not an
    abort: the completed payload is discarded as a counted ``attempt_fail`` with no registered cause (halt, a repair
    record, a counted retry), so a nondeterministic unit (the LLM-reader arms) never gets an uncounted redraw
    (PREREG_WME §II.31.9, Board A N1)."""


class HealthIncident(RG.IncidentAbort):
    """A non-failure incident (order, GPU idle, telemetry, progress). Raised before ``guard.begin`` (order) or as an
    attempt abort while the unit is in flight: recorded ``attempt_aborted``, never counted toward the attempt limit
    (§II.31.8 R4). Worker errors are not wrapped: they keep their registered cause (R3). Tracebacks found after
    completion are :class:`PostCompletionError` (counted; §II.31.9 N1)."""

    def __init__(self, message: str, kind: str = "incident"):
        super().__init__(message)
        self.incident_kind = kind


# ------------------------------------------------------------------ worker side


def _nvml():
    import pynvml

    pynvml.nvmlInit()
    out = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        out.append(
            {
                "index": i,
                "util_pct": float(u.gpu),
                "mem_used_mib": float(m.used) / 2**20,
                "mem_total_mib": float(m.total) / 2**20,
            }
        )
    return out


def parse_smi(text: str) -> list:
    out = []
    for line in text.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        i, u, mu, mt = parts
        out.append(
            {
                "index": int(i),
                "util_pct": float(u),
                "mem_used_mib": float(mu),
                "mem_total_mib": float(mt),
            }
        )
    return out


def _smi():
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return parse_smi(p.stdout)


def gpu_sample() -> list | None:
    """[{index, util_pct, mem_used_mib, mem_total_mib}] or None when no telemetry is available."""
    for fn in (_nvml, _smi):
        try:
            r = fn()
            if r:
                return r
        except Exception:  # noqa: BLE001, S112 - the next source, else None (reported as missing telemetry)
            continue
    return None


class Sampler:
    """Worker heartbeat thread: GPU utilisation and memory plus the in-flight request, every ``period_s``."""

    def __init__(self, root, period_s: float = SAMPLE_PERIOD_S, sample=gpu_sample, clock=time.time):
        self.root = Path(root)
        self.period_s, self.sample, self.clock = period_s, sample, clock
        self.inflight = None
        self.low_since = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._write_lock = (
            threading.Lock()
        )  # one heartbeat writer at a time (sampler thread vs worker thread)

    def set_inflight(self, req: dict | None):
        with self._lock:
            now = float(self.clock())
            self.inflight = (
                None
                if req is None
                else {"id": req["id"], "kind": req["kind"], "since": now, "last_progress": now}
            )
        try:
            self.beat()
        except Exception:  # noqa: BLE001, S110 - a heartbeat write never stops the worker loop; the sampler re-beats
            pass

    def note_progress(self):
        """A handler sub-step completed (``Ctx.check`` / ``Ctx.progress``): the CPU-kind progress heartbeat."""
        with self._lock:
            if self.inflight is not None:
                self.inflight["last_progress"] = float(self.clock())

    def beat(self) -> dict:
        t = float(self.clock())
        gpu = self.sample()
        with self._lock:
            if gpu:
                if max(g["util_pct"] for g in gpu) < UTIL_LOW_PCT:
                    self.low_since = t if self.low_since is None else self.low_since
                else:
                    self.low_since = None
            inf = None if self.inflight is None else dict(self.inflight)
            hb = {"t": t, "gpu": gpu, "low_since": self.low_since, "inflight": inf}
        with (
            self._write_lock
        ):  # the sampler thread and the worker thread both beat: never share a temp file
            tmp = self.root / f".{HEARTBEAT}.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(hb))
            os.replace(tmp, self.root / HEARTBEAT)
        return hb

    def _run(self):
        while not self._stop.is_set():
            try:
                self.beat()
            except Exception:  # noqa: BLE001, S110 - a missed beat shows as a stale heartbeat on the controller
                pass
            self._stop.wait(self.period_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="wme-c1-health", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# ------------------------------------------------------------------ controller side


def idle_check(
    hb: dict | None,
    classes: dict | None = None,
    idle_limit_s: float = IDLE_LIMIT_S,
    progress_limit_s: float = NO_PROGRESS_LIMIT_S,
) -> tuple[str, dict] | None:
    """The idle rules on one heartbeat (worker clock). Returns (kind, detail) or None.

    * GPU kinds: utilisation < 20 % for more than 15 min (``gpu_idle``), or no GPU telemetry for that long
      (``gpu_telemetry_missing``); progress does not excuse an idle GPU.
    * CPU kinds: more than 15 min without progress (``no_progress``), counted from the later of the unit's start and
      its last progress; GPU utilisation is not checked.
    A kind missing from ``classes`` is treated as a GPU kind (the stricter rule)."""
    if not hb or not hb.get("inflight"):
        return None
    inf = hb["inflight"]
    t, since = float(hb["t"]), float(inf["since"])
    cls = (KIND_CLASS if classes is None else classes).get(inf.get("kind"), "gpu")
    if cls == "cpu":
        last = max(since, float(inf.get("last_progress") or since))
        if t - last > progress_limit_s:
            return "no_progress", {"inflight": inf, "stalled_s": t - last, "class": "cpu"}
        return None
    if not hb.get("gpu"):
        if t - since > idle_limit_s:
            return "gpu_telemetry_missing", {"inflight": inf, "for_s": t - since}
        return None
    if hb.get("low_since") is None:
        return None
    low_for = t - max(float(hb["low_since"]), since)
    if low_for > idle_limit_s:
        return "gpu_idle", {
            "inflight": inf,
            "low_for_s": low_for,
            "gpu": hb["gpu"],
            "threshold_pct": UTIL_LOW_PCT,
            "class": "gpu",
        }
    return None


def log_errors(text: str) -> list:
    """Lines of new worker-log text that are a traceback or an error line."""
    return [m.group(0).strip() for m in TRACE_RE.finditer(text or "")]


def payload_traceback(payload) -> bool:
    return "Traceback (most recent call last)" in json.dumps(payload, default=str)


class HealthMonitor:
    """Controller-side incident rules and chunk-boundary summaries for one study (state in ``<runs>/health``)."""

    def __init__(
        self,
        runs_dir,
        transport=None,
        clock=time.time,
        log=None,
        order=None,
        classes=None,
        limits=None,
    ):
        # fixture classes / limits / poll interval only in the CPU drill; production uses the registered values
        self.classes = dict(KIND_CLASS if classes is None else classes)
        lim = dict(limits or {})
        self.idle_limit_s = float(lim.get("idle_s", IDLE_LIMIT_S))
        self.progress_limit_s = float(lim.get("progress_s", NO_PROGRESS_LIMIT_S))
        self.poll_min_s = float(lim.get("poll_s", POLL_MIN_S))
        self.stages = tuple(order or STAGES)  # a fixture order only in the CPU drill
        self.stage_of = {k: i for i, (_s, kinds) in enumerate(self.stages) for k in kinds}
        self.dir = Path(runs_dir) / "health"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t = transport
        self.clock = clock
        self.log = log or (lambda *a, **k: None)
        self.state_path = self.dir / "state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"stage": -1, "skipped": [], "log_offset": 0, "chunks": 0}
        )
        self.last_hb = None
        self._last_poll = float("-inf")
        self.unit = (
            None  # the controller-side unit in flight: {"label", "kind", "since", "last_progress"}
        )
        self._last_progress_write = float("-inf")

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.state_path)

    @contextlib.contextmanager
    def _state(self):
        """Reload the shared state under a file lock, yield it, save it (R6: several monitor objects of one study,
        e.g. ``Study.run``'s and a controller's skip declaration, never lose each other's updates)."""
        import fcntl

        with open(self.dir / "state.lock", "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                if self.state_path.exists():
                    self.state = json.loads(self.state_path.read_text())
                yield self.state
                self._save()
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _incident(self, guard, kind: str, detail: dict):
        rec = guard.incident(kind, detail)
        with open(self.dir / "incidents.jsonl", "a") as f:
            f.write(
                json.dumps({"t": self.clock(), "kind": kind, "detail": detail}, default=str) + "\n"
            )
        self.log(f"RUN-HEALTH INCIDENT {kind}: {json.dumps(detail, default=str)[:400]}")
        return rec

    # -------------------------------------------------- order (§II.18)
    def skip_stage(self, stage: str, reason: str):
        """Declare a registered stage skipped by the registered data flow (e.g. E1 not designated)."""
        names = [s for s, _k in self.stages]
        if stage not in names:
            raise KeyError(stage)
        with self._state() as st:
            st["skipped"] = sorted(set(st["skipped"]) | {stage})
            st.setdefault("skip_reasons", {})[stage] = reason

    def check_order(self, guard, recipe: dict, label=None):
        """The §II.18 order rule. Called BEFORE ``guard.begin`` (``run_units(before_begin=)``): an order incident
        never records an attempt (R4)."""
        kind = recipe.get("kind")
        names = [s for s, _k in self.stages]
        with self._state() as st:
            cur = int(st["stage"])
            i = self.stage_of.get(kind)
            between = (
                []
                if i is None
                else [names[j] for j in range(cur + 1, i) if names[j] not in st["skipped"]]
            )
            ok = i is not None and i >= cur and not between
            if ok and i != cur:
                st["stage"] = i
        if i is None:
            self._incident(guard, "order", {"kind": kind, "reason": "unregistered unit kind"})
            raise HealthIncident(f"unit kind {kind!r} is not in the registered order", "order")
        if not ok:
            detail = {
                "kind": kind,
                "stage": names[i],
                "current": names[cur] if cur >= 0 else None,
                "skipped_without_declaration": between,
            }
            self._incident(guard, "order", detail)
            raise HealthIncident(
                f"unit kind {kind!r} out of the registered §II.18 order: {detail}", "order"
            )

    # -------------------------------------------------- results and logs
    def check_result_error(self, guard, recipe: dict, exc: BaseException):
        self._incident(
            guard,
            "worker_error",
            {"kind": recipe.get("kind"), "label": recipe.get("label"), "error": type(exc).__name__,
             "message": str(exc)[:500]},
        )  # fmt: skip

    def check_payload(self, guard, recipe: dict, payload):
        if payload_traceback(payload):
            self._incident(
                guard,
                "worker_error",
                {"kind": recipe.get("kind"), "reason": "traceback in payload"},
            )
            raise PostCompletionError(
                "traceback in a worker result payload (completed payload discarded)"
            )

    def check_log(self, guard):
        if self.t is None or not hasattr(self.t, "read_log"):
            return
        with self._state() as st:
            text, off = self.t.read_log(int(st["log_offset"]))
            st["log_offset"] = int(off)
        errs = log_errors(text)
        if errs:
            self._incident(guard, "worker_error", {"reason": "worker log", "lines": errs[:20]})
            raise PostCompletionError(f"traceback or error in the worker log: {errs[:3]}")

    # -------------------------------------------------- heartbeat (polled while a unit is in flight)
    def poll(self, guard):
        if self.t is None or not hasattr(self.t, "heartbeat"):
            return
        now = float(self.clock())
        if now - self._last_poll < self.poll_min_s:
            return
        self._last_poll = now
        hb = self.t.heartbeat()
        self.last_hb = hb
        r = idle_check(hb, self.classes, self.idle_limit_s, self.progress_limit_s)
        if r is not None:
            self._incident(guard, r[0], r[1])
            raise HealthIncident(f"{r[0]}: {json.dumps(r[1], default=str)[:300]}", r[0])

    # -------------------------------------------------- controller-side progress (LLM-reader waits, R7)
    def note_progress(self, event: str, **kw):
        """Progress of the controller-side unit in flight (an LLM-reader completion, a worker sub-request). The
        progress file is written at most every 30 s; the in-memory clock always moves."""
        now = float(self.clock())
        if self.unit is not None:
            self.unit["last_progress"] = now
        if now - self._last_progress_write >= 30.0:
            self._last_progress_write = now
            with open(self.dir / "controller_progress.jsonl", "a") as f:
                f.write(json.dumps({"t": now, "event": event, **kw}, default=str) + "\n")

    def check_controller_stall(self, guard):
        """A controller-side unit (CPU class: the LLM-reader arms) without progress for more than 15 min."""
        u = self.unit
        if u is None:
            return
        stalled = float(self.clock()) - max(u["since"], u["last_progress"])
        if stalled > self.progress_limit_s:
            d = {"unit": dict(u), "stalled_s": stalled, "class": "cpu", "side": "controller"}
            self._incident(guard, "no_progress", d)
            raise HealthIncident(f"no_progress (controller side): {u['label']}", "no_progress")

    # -------------------------------------------------- canaries
    def check_canary(self, guard, rec: dict):
        if not rec.get("pass"):
            self._incident(
                guard,
                "canary_mismatch",
                {"composite": rec.get("composite"), "engines": rec.get("engines")},
            )

    # -------------------------------------------------- wrappers used by Study.run
    def wrap_fn(self, guard, fn):
        """Around one attempt (after ``guard.begin``). A worker error is recorded as an incident and re-raised
        UNWRAPPED, so ``attempt_fail`` carries its registered technical cause or None exactly as without the monitor
        (R3); the next ``begin`` then refuses (HealthHalt) until the incident is resolved. Incidents while the
        unit is in flight (idle, telemetry, progress) raise :class:`HealthIncident` (attempt aborted, not counted);
        a traceback found after completion (payload or log) raises :class:`PostCompletionError`: a counted
        ``attempt_fail`` with no registered cause (§II.31.9, N1)."""

        def run(recipe):
            now = float(self.clock())
            self.unit = {
                "label": recipe.get("label"),
                "kind": recipe.get("kind"),
                "since": now,
                "last_progress": now,
            }
            try:
                try:
                    payload = fn(recipe)
                except RG.IncidentAbort:
                    raise
                except Exception as exc:
                    self.check_result_error(guard, recipe, exc)
                    raise
                self.check_payload(guard, recipe, payload)
                self.check_log(guard)
                return payload
            finally:
                self.unit = None

        return run

    def before_begin(self, guard):
        return lambda recipe, label: self.check_order(guard, recipe, label)

    def wrap_canary(self, guard, canary):
        def run():
            rec = canary()
            self.check_canary(guard, rec)
            return rec

        return run

    def chunk_summary(self, guard, rec: dict | None = None, units_done: int | None = None) -> Path:
        """The health summary file at a chunk boundary (``<runs>/health/summary.jsonl`` and ``latest.json``)."""
        led = guard.ledger
        with self._state() as st:
            st["chunks"] = int(st["chunks"]) + 1
        hb = self.last_hb
        if self.t is not None and hasattr(self.t, "heartbeat"):
            try:
                hb = self.t.heartbeat()
            except Exception as exc:  # noqa: BLE001 - recorded in the summary
                hb = {"error": f"{type(exc).__name__}: {exc}"}
        s = {
            "t": self.clock(),
            "chunk": self.state["chunks"],
            "phase": guard.phase,
            "instance": guard.instance,
            "stage": self.stages[self.state["stage"]][0] if self.state["stage"] >= 0 else None,
            "units_done_in_call": units_done,
            "canary_pass": None if rec is None else bool(rec.get("pass")),
            "gpu": (hb or {}).get("gpu"),
            "heartbeat_t": (hb or {}).get("t"),
            "open_incidents": [i["kind"] for i in guard.open_incidents()],
            "charged_gpu_h": led.gpu_h(),
            "cap_gpu_h": led.cap_gpu_h(),
            "remaining_gpu_h": led.remaining_gpu_h(),
        }
        with open(self.dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(s, default=str) + "\n")
        (self.dir / "latest.json").write_text(json.dumps(s, indent=1, default=str))
        return self.dir / "latest.json"
