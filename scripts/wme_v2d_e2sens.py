#!/usr/bin/env python3
"""Board A v2 required change 4: outcome-blind sensitivity of the H-E2 sign-flip test (CPU; run via agent-tmux).

    wme_v2d_e2sens.py --out OUT.json [--reps 4000]

Inputs: the v2 simulation inputs (docs/reviews/wme/wme-powersim-v2-2026-11-17.json); the decision rule and its
consequences are those of ``wsp.wm.powersim_e2sens`` (frozen before this run)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import powersim_e2sens as ES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    inp = json.loads((ROOT / "docs/reviews/wme/wme-powersim-v2-2026-11-17.json").read_text())[
        "inputs"
    ]
    out = {
        "label": "EXPLORATORY SYNTHETIC SIMULATION: H-E2 sign-flip sensitivity (Board A v2 required change 4)",
        "design": ES.DESIGN,
        "rule": ES.__doc__,
        "results": {},
    }
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        jobs = {n: pool.apply_async(ES.run, (n, a.reps, inp)) for n in ES.SCENARIOS}
        for name, j in jobs.items():
            out["results"][name] = j.get()
            out["seconds"] = time.time() - t0
            Path(a.out).write_text(json.dumps(out, indent=1, default=float))
            print(
                name, {k: out["results"][name][k]["rate"] for k in ("sf", "rb", "iu")}, flush=True
            )
    out["decision"] = ES.decide(out["results"])
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["decision"]))


if __name__ == "__main__":
    main()
