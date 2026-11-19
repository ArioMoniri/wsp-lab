#!/usr/bin/env python3
"""Run the registered adversarial property test of the runtime contract at full size (PREREG_WME §II.26):
``tests/test_wme_runguard_property.run_sequences`` over N seeded sequences in parallel chunks; writes a report with
the operation counts, the coverage counters and every failure (seed and assertion). CPU only.

    wme_v2h_runguard_property.py --out OUT.json [--n 5000] [--procs 5]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
SEED0 = 7_000_000


def _chunk(args):
    lo, hi = args
    import test_wme_runguard_property as T

    from wsp.wm import runguard as RG

    RG.CAP_GPU_H, RG.CAP_USD = T.CAP, 1e9
    ops, cov, fails = {}, {}, []
    for s in range(lo, hi):
        try:
            o, c = T.run_sequences(1, seed0=SEED0 + s)
        except Exception as e:  # noqa: BLE001 - any failure is reported with its seed
            fails.append(
                {
                    "seed": SEED0 + s,
                    "error": f"{type(e).__name__}: {e}"[:400],
                    "tb": traceback.format_exc()[-800:],
                }
            )
            continue
        for k, v in o.items():
            ops[k] = ops.get(k, 0) + v
        for k, v in c.items():
            cov[k] = cov.get(k, 0) + v
    return ops, cov, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    import test_wme_runguard_property as T

    t0 = time.time()
    step = (a.n + a.procs * 4 - 1) // (a.procs * 4)
    chunks = [(lo, min(lo + step, a.n)) for lo in range(0, a.n, step)]
    ops, cov, fails = {}, {}, []
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for o, c, f in pool.imap_unordered(_chunk, chunks):
            for k, v in o.items():
                ops[k] = ops.get(k, 0) + v
            for k, v in c.items():
                cov[k] = cov.get(k, 0) + v
            fails += f
    out = {
        "label": "adversarial property test of the runtime contract (PREREG_WME §II.26)",
        "test_module": "tests/test_wme_runguard_property.py",
        "sequences": a.n,
        "steps": T.STEPS,
        "seed0": SEED0,
        "cap_gpu_h_in_test": T.CAP,
        "ops": dict(sorted(ops.items())),
        "coverage": dict(sorted(cov.items())),
        "failures": sorted(fails, key=lambda f: f["seed"]),
        "passed": not fails,
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("sequences", "passed", "coverage", "seconds")}, indent=1))
    print("failures", len(fails))


if __name__ == "__main__":
    main()
