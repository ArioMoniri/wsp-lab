"""Fixture worker for the CPU deployment drill (§II.28.3 precondition 6): the real ``wsp.wm.c1.worker.Worker`` loop
with fixture handlers (no model). ``python -m wsp.wm.c1.drillworker --root <dir>``."""

from __future__ import annotations

import argparse
import json
import time

from wsp.wm.c1 import worker as W

ENGINES = {"tf": "1" * 64, "hf_gen": "2" * 64, "vllm": "3" * 64}


def square(req, ctx):
    a = req["args"]
    for _ in range(int(a.get("steps", 1))):
        time.sleep(float(a.get("sleep", 0.0)))
        ctx.check()
    out = ctx.scratch / "out.json"
    out.write_text(json.dumps({"x": a["x"]}))
    return {"sq": a["x"] * a["x"]}, {"out.json": out}


def canary(req, ctx):
    return {"engines": dict(ENGINES)}, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    W.Worker(a.root, {"square": square, "canary": canary}, poll_s=0.05).serve()


if __name__ == "__main__":
    main()
