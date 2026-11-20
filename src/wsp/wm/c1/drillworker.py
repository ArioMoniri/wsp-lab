"""Fixture worker for the CPU deployment drill (§II.28.3 precondition 6): the real ``wsp.wm.c1.worker.Worker`` loop
with fixture handlers (no model). ``python -m wsp.wm.c1.drillworker --root <dir>``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from wsp.wm.c1 import health as HL
from wsp.wm.c1 import worker as W

ENGINES = {"tf": "1" * 64, "hf_gen": "2" * 64, "vllm": "3" * 64}


def square(req, ctx):
    a = req["args"]
    if a.get("fail_once"):  # scenario H: one injected worker error (marker in the worker root)
        mark = ctx.worker.root / f"failed_{a['x']}"
        if not mark.exists():
            mark.write_text("1")
            if a["fail_once"] == "technical":  # classified "infrastructure" (a registered technical cause)
                raise OSError("drill: injected infrastructure error")
            raise RuntimeError("drill: injected worker error")  # no registered cause
    if a.get("hang_once"):  # scenario H (R6): a hung op that never reaches ctx.check
        mark = ctx.worker.root / f"hung_{a['x']}"
        if not mark.exists():
            mark.write_text("1")
            time.sleep(float(a["hang_once"]))
    for _ in range(int(a.get("steps", 1))):
        time.sleep(float(a.get("sleep", 0.0)))
        ctx.check()
    out = ctx.scratch / "out.json"
    out.write_text(json.dumps({"x": a["x"]}))
    return {"sq": a["x"] * a["x"]}, {"out.json": out}


def plan(req, ctx):
    """A CPU-kind fixture unit (the planner-search class): progress by ``ctx.check`` per step; ``stall_once`` sleeps
    once without any progress (scenario H)."""
    a = req["args"]
    if a.get("stall_once"):
        mark = ctx.worker.root / f"stalled_{a['x']}"
        if not mark.exists():
            mark.write_text("1")
            time.sleep(float(a["stall_once"]))
    for _ in range(int(a.get("steps", 1))):
        time.sleep(float(a.get("sleep", 0.0)))
        ctx.check()
    return {"plan": a["x"] + 1000}, {}


def fixture_gpu(root):
    """Scripted GPU telemetry for the drill: utilisation from ``<root>/gpu_util`` (default 90 %)."""
    p = Path(root) / "gpu_util"

    def sample():
        u = float(p.read_text()) if p.exists() else 90.0
        return [{"index": 0, "util_pct": u, "mem_used_mib": 1000.0, "mem_total_mib": 81559.0}]

    return sample


def canary(req, ctx):
    return {"engines": dict(ENGINES)}, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    sampler = HL.Sampler(a.root, period_s=0.1, sample=fixture_gpu(a.root))
    W.Worker(
        a.root, {"square": square, "plan": plan, "canary": canary}, poll_s=0.05, sampler=sampler
    ).serve()


if __name__ == "__main__":
    main()
