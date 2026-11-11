#!/usr/bin/env python3
"""F12d auditable CPU budget stops for the F12 simulation programme (Board A F12c review condition 7; chair ruling on
F12c, 11 Nov 2026 18:55 UTC: all-in ceiling 190 CPU-h).

An append-only JSON-lines ledger records every launch, checkpoint, end and stop of every job of the programme: the
noise check, the timing pilot, stage 1, the coverage target, stage 2, assembly, verification and anything else
(``other``), including failed and aborted attempts. Each event carries the attempt's cumulative process-tree CPU
(parent plus live and reaped workers, ``process_tree_cpu_s``), the responsible operator and the checkpoint frequency.

The barrier (``Meter.launch`` and every ``Meter.tick`` checkpoint) computes

    spent (all attempts of all jobs in the ledger, including the running one)
  + projected remaining (planned items not yet completed; the running job's remaining units at the larger of its
    observed and planned per-unit rate; overhead less the overhead already spent)

and raises ``BudgetStop`` before work continues if that total exceeds the ceiling. A stop never reduces a replicate
count, a draw or a qualification step and never yields a decision from a partial run: the run ends without a report
and waits for a dated prospective repricing (F12a.7(b), F12c.7(b), the chair ruling).

Checkpoints: every ``CHECKPOINT_EVERY_UNITS`` completed replicates and whenever ``CHECKPOINT_MAX_SECONDS`` have passed
since the last checkpoint, whichever comes first. With at most two workers, work in flight at a stop is at most two
replicates (about 0.13 CPU-h at the measured synthetic rates).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path

LEDGER_SCHEMA = "f12d-budget-ledger-v1"
CEILING_CPU_H = 190.0
CHECKPOINT_EVERY_UNITS = 10
CHECKPOINT_MAX_SECONDS = 900.0
PLAN_KINDS = ("noise_check", "pilot", "stage1", "target", "stage2", "overhead")
JOB_KINDS = (
    "noise_check",
    "pilot",
    "stage1",
    "target",
    "stage2",
    "assembly",
    "verification",
    "other",
)
OVERHEAD_KINDS = ("assembly", "verification", "other")
STOP_TEXT = (
    "no count or draw is reduced and no decision is taken from the partial run; the run stops for a dated "
    "prospective repricing"
)


class BudgetStop(RuntimeError):
    pass


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------- CPU metering


def _proc_cpu_s(pid) -> float:
    """utime + stime of one live process from /proc (Linux), 0 if it has gone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return 0.0
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _children(pid) -> list:
    kids = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            ppid = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == pid:
            kids.append(int(p.name))
    return kids


def process_tree_cpu_s() -> float:
    """CPU seconds of this process, its live descendants (workers) and its reaped children."""
    t = os.times()
    total = t.user + t.system + t.children_user + t.children_system
    stack, seen = _children(os.getpid()), set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _proc_cpu_s(pid)
        stack.extend(_children(pid))
    return total


# ------------------------------------------------------------------------------- ledger


class Ledger:
    """Append-only JSON-lines ledger of budget events (fsynced per event)."""

    def __init__(self, path):
        self.path = Path(path)

    def events(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def append(self, event: dict) -> dict:
        event = {"schema": LEDGER_SCHEMA, **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return event

    def attempts(self) -> dict:
        """(job, attempt) -> {"kind", "cpu_s" (latest), "status" (latest)}."""
        out = {}
        for e in self.events():
            key = (e["job"], e["attempt"])
            cur = out.setdefault(
                key, {"kind": e["kind"], "cpu_s": 0.0, "status": "running", "done_units": 0}
            )
            cur["cpu_s"] = max(cur["cpu_s"], float(e.get("cpu_s") or 0.0))
            cur["done_units"] = int(e.get("done_units") or 0)
            if e["event"] in ("end", "stop"):
                cur["status"] = e["status"]
        return out

    def spent_cpu_h(self) -> float:
        return sum(a["cpu_s"] for a in self.attempts().values()) / 3600

    def next_attempt(self, job) -> int:
        return 1 + max((a for (j, a) in self.attempts() if j == job), default=0)


def _plan_item(kind):
    return "overhead" if kind in OVERHEAD_KINDS else kind


def projection(ledger, plan, *, current=None) -> dict:
    """Spent + projected remaining CPU-h.

    ``current``: {"key", "kind", "planned_units", "done_units", "cpu_s"} of the running attempt. When the plan
    gives ``units_by_kind`` (the whole item's planned replicates, e.g. stage 2 = 2,200 run in parts), an item's
    remaining units are its planned units less the units of completed attempts and the running attempt's done
    units, priced at the larger of the planned and the observed per-unit rate (observed over completed and
    running attempts; failed attempts are spent but never count as done). Without ``units_by_kind`` one
    completed attempt completes its item."""
    items = plan["items_cpu_h"]
    units = plan.get("units_by_kind") or {}
    att = ledger.attempts()
    if current is not None:
        att[current["key"]] = {
            "kind": current["kind"],
            "cpu_s": current["cpu_s"],
            "status": "running",
            "done_units": current["done_units"],
        }
    spent = sum(a["cpu_s"] for a in att.values()) / 3600
    done_kinds = {_plan_item(a["kind"]) for a in att.values() if a["status"] == "completed"}
    overhead_spent = sum(a["cpu_s"] for a in att.values() if a["kind"] in OVERHEAD_KINDS) / 3600
    remaining = {}
    for item in PLAN_KINDS:
        planned = float(items.get(item, 0.0))
        cur = current is not None and current["kind"] == item
        if item == "overhead":
            remaining[item] = max(planned - overhead_spent, 0.0)
        elif units.get(item):
            total = int(units[item])
            good = [a for a in att.values() if a["kind"] == item and a["status"] in ("completed", "running")]
            done = sum(int(a.get("done_units") or 0) for a in good)
            used = sum(a["cpu_s"] for a in good)
            rate = max(planned * 3600 / total, used / done if done else 0.0)
            remaining[item] = max(total - done, 0) * rate / 3600
        elif cur:
            left = current["planned_units"] - current["done_units"]
            n = current["planned_units"]
            rate_plan = planned * 3600 / n if n else 0.0
            rate_obs = current["cpu_s"] / current["done_units"] if current["done_units"] else 0.0
            remaining[item] = left * max(rate_plan, rate_obs) / 3600
        elif item in done_kinds:
            remaining[item] = 0.0
        else:
            remaining[item] = planned
    total = spent + sum(remaining.values())
    return {
        "spent_cpu_h": spent,
        "remaining_cpu_h": remaining,
        "projected_total_cpu_h": total,
    }


class Meter:
    """Launch / checkpoint barrier for one job attempt."""

    def __init__(self, ledger, plan, *, job, kind, planned_units, operator,
                 ceiling_cpu_h=CEILING_CPU_H, every=CHECKPOINT_EVERY_UNITS,
                 max_seconds=CHECKPOINT_MAX_SECONDS, cpu=None):  # fmt: skip
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown budget kind {kind!r}")
        if not operator:
            raise ValueError("a responsible operator must be named")
        self.ledger, self.plan, self.job, self.kind = ledger, plan, job, kind
        self.planned_units, self.operator = int(planned_units), operator
        self.ceiling, self.every, self.max_seconds = float(ceiling_cpu_h), int(every), max_seconds
        self._cpu = cpu or process_tree_cpu_s
        self._cpu0 = None
        self.attempt = None
        self._last = None
        self._done = 0

    def cpu_s(self) -> float:
        return self._cpu() - self._cpu0

    def _event(self, event, status, done, cpu_s, proj):
        return self.ledger.append({
            "event": event, "status": status, "job": self.job, "kind": self.kind, "attempt": self.attempt,
            "operator": self.operator, "utc": utc_now(), "cpu_s": cpu_s, "done_units": done,
            "planned_units": self.planned_units, "checkpoint_every_units": self.every,
            "checkpoint_max_seconds": self.max_seconds, "ceiling_cpu_h": self.ceiling,
            "plan_basis": self.plan.get("basis"), **proj,
        })  # fmt: skip

    def _check(self, event, done):
        cpu = self.cpu_s()
        cur = {"key": (self.job, self.attempt), "kind": _plan_item(self.kind),
               "planned_units": self.planned_units, "done_units": done, "cpu_s": cpu}  # fmt: skip
        if self.kind in OVERHEAD_KINDS:
            cur = None
        proj = projection(self.ledger, self.plan, current=cur)
        if cur is None:
            proj["spent_cpu_h"] += cpu / 3600
            proj["projected_total_cpu_h"] += cpu / 3600
        if proj["projected_total_cpu_h"] > self.ceiling:
            self._event("stop", "stopped_budget", done, cpu, proj)
            raise BudgetStop(
                f"{self.job} ({self.kind}): spent plus projected remaining "
                f"{proj['projected_total_cpu_h']:.2f} CPU-h exceeds the {self.ceiling:g} CPU-h ceiling at "
                f"{done}/{self.planned_units} units; {STOP_TEXT}"
            )
        return self._event(event, "running", done, cpu, proj)

    def launch(self) -> dict:
        self._cpu0 = self._cpu()
        self.attempt = self.ledger.next_attempt(self.job)
        self._last = time.monotonic()
        return self._check("launch", 0)

    def tick(self, done) -> dict | None:
        self._done = done
        now = time.monotonic()
        if done % self.every == 0 or now - self._last >= self.max_seconds:
            self._last = now
            return self._check("checkpoint", done)
        return None

    def end(self, status, cpu_s=None, done=None) -> dict:
        """Close the attempt. A completed attempt did all its planned units; any other status keeps only the
        units checkpointed so far (and its CPU is spent either way)."""
        cpu = self.cpu_s() if cpu_s is None else float(cpu_s)
        if done is None:
            done = self.planned_units if status == "completed" else self._done
        proj = projection(self.ledger, self.plan)
        proj["spent_cpu_h"] += cpu / 3600
        return self._event("end", status, done, cpu, proj)


def plan_from_resource_plan(obj) -> dict:
    """The pre-pilot plan: the chair-accepted synthetic all-in items (f12c_resource_plan.json)."""
    return {
        "basis": "pre-pilot synthetic projection (f12c_resource_plan.json items_by_kind_cpu_h)",
        "items_cpu_h": dict(obj["items_by_kind_cpu_h"]),
    }


def plan_from_pilot_acceptance(obj) -> dict:
    """After the pilot: the chair-signed all-in projection (pilot_acceptance.projection.items_by_kind_cpu_h)."""
    return {
        "basis": "chair-accepted measured-pilot projection (pilot_acceptance)",
        "items_cpu_h": dict(obj["projection"]["items_by_kind_cpu_h"]),
    }


def load_plan(path) -> dict:
    obj = json.loads(Path(path).read_text())
    if "projection" in obj and obj.get("chair_accepted") is True:
        return plan_from_pilot_acceptance(obj)
    if "items_by_kind_cpu_h" in obj:
        return plan_from_resource_plan(obj)
    raise ValueError(f"{path}: neither a resource plan nor a pilot acceptance")
