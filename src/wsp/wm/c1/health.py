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
CPU_KINDS = ("fit", "e2_arm", "theta_of", "describe", "edit_bundles", "rstats", "hello")
KIND_CLASS = {**dict.fromkeys(GPU_KINDS, "gpu"), **dict.fromkeys(CPU_KINDS, "cpu")}
TRACE_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*(?:[\w.]+(?:Error|Exception)|CUDA error|RuntimeError)\b[:\s]",
    re.MULTILINE,
)


class HealthIncident(RuntimeError):
    """Raised to the issuing loop on an incident (not a technical cause: ``run_units`` halts, charged)."""


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
        except Exception:  # noqa: BLE001 - the next source, else None (reported as missing telemetry)
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

    def set_inflight(self, req: dict | None):
        with self._lock:
            now = float(self.clock())
            self.inflight = (
                None
                if req is None
                else {"id": req["id"], "kind": req["kind"], "since": now, "last_progress": now}
            )
        self.beat()

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
        tmp = self.root / f".{HEARTBEAT}.tmp"
        tmp.write_text(json.dumps(hb))
        os.replace(tmp, self.root / HEARTBEAT)
        return hb

    def _run(self):
        while not self._stop.is_set():
            try:
                self.beat()
            except Exception:  # noqa: BLE001 - a missed beat shows as a stale heartbeat on the controller
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

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.state_path)

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
        self.state["skipped"] = sorted(set(self.state["skipped"]) | {stage})
        self.state.setdefault("skip_reasons", {})[stage] = reason
        self._save()

    def check_order(self, guard, recipe: dict):
        kind = recipe.get("kind")
        cur = int(self.state["stage"])
        if kind not in self.stage_of:
            self._incident(guard, "order", {"kind": kind, "reason": "unregistered unit kind"})
            raise HealthIncident(f"unit kind {kind!r} is not in the registered order")
        i = self.stage_of[kind]
        names = [s for s, _k in self.stages]
        between = [names[j] for j in range(cur + 1, i) if names[j] not in self.state["skipped"]]
        if i < cur or between:
            detail = {
                "kind": kind,
                "stage": names[i],
                "current": names[cur] if cur >= 0 else None,
                "skipped_without_declaration": between,
            }
            self._incident(guard, "order", detail)
            raise HealthIncident(f"unit kind {kind!r} out of the registered §II.18 order: {detail}")
        if i != cur:
            self.state["stage"] = i
            self._save()

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
            raise HealthIncident("traceback in a worker result payload")

    def check_log(self, guard):
        if self.t is None or not hasattr(self.t, "read_log"):
            return
        text, off = self.t.read_log(int(self.state["log_offset"]))
        self.state["log_offset"] = int(off)
        self._save()
        errs = log_errors(text)
        if errs:
            self._incident(guard, "worker_error", {"reason": "worker log", "lines": errs[:20]})
            raise HealthIncident(f"traceback or error in the worker log: {errs[:3]}")

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
            raise HealthIncident(f"{r[0]}: {json.dumps(r[1], default=str)[:300]}")

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
        def run(recipe):
            self.check_order(guard, recipe)
            try:
                payload = fn(recipe)
            except HealthIncident:
                raise
            except Exception as exc:
                self.check_result_error(guard, recipe, exc)
                raise HealthIncident(f"worker error: {type(exc).__name__}: {exc}") from exc
            self.check_payload(guard, recipe, payload)
            self.check_log(guard)
            return payload

        return run

    def wrap_canary(self, guard, canary):
        def run():
            rec = canary()
            self.check_canary(guard, rec)
            return rec

        return run

    def chunk_summary(self, guard, rec: dict | None = None, units_done: int | None = None) -> Path:
        """The health summary file at a chunk boundary (``<runs>/health/summary.jsonl`` and ``latest.json``)."""
        led = guard.ledger
        self.state["chunks"] = int(self.state["chunks"]) + 1
        self._save()
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
