#!/usr/bin/env python3
"""SYNTHESIS-r9 C6 planner gate: synthetic real regret at B = 32, sep-CMA-ES vs CEM (random: descriptive). CPU only.

Kill rule: sep-CMA-ES is used only if its mean regret is lower than CEM's; otherwise CEM. The result is frozen before
any real search (Board A condition 10). Output: docs/reviews/wme/wme-planner-synthetic-<date>.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import planner as PL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument(
        "--out", default=str(ROOT / "docs/reviews/wme/wme-planner-synthetic-2026-11-17.json")
    )
    a = ap.parse_args()
    t = time.time()
    res = PL.synthetic_regret(
        n_worlds=a.n_worlds, B=32, methods=("sep_cma", "cem", "random"), procs=a.procs
    )
    res["wall_s"] = round(time.time() - t, 1)
    res["cpu_h_estimate"] = round(res["wall_s"] * a.procs / 3600, 3)
    res["peak_rss_mb_parent"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res["label"] = "synthetic planner gate (r9 C6); no model, no data"
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                "decision": res["decision"],
                "mean_regret": {m: v["mean_regret"] for m, v in res["methods"].items()},
                "wall_s": res["wall_s"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
