#!/usr/bin/env python3
"""PREREG_WME §8 outcome-blind power simulation (Board A condition 11). CPU only.

price  CPU pilot (default 20 replicates per priced cell, 2,000 bootstrap draws) -> CPU-h for the full grid at
       1,000 replicates, plus the GPU feasibility table (bank with/without per-state oracle grounding;
       confirmatory primary and all arms). Writes docs/reviews/wme/wme-powersim-pricing-<date>.json.
run    the full simulation. Refuses unless --board-a-approval names the approving review (not yet given).
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

from wsp.wm import bank as B
from wsp.wm import exposure as X
from wsp.wm import powersim as PS

REPORT = ROOT / "docs/reviews/wme/wme-pilot2-2026-11-16.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["price", "run"])
    ap.add_argument("--pilot-reps", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=PS.N_BOOT)
    ap.add_argument("--board-a-approval", default=None)
    ap.add_argument(
        "--out", default=str(ROOT / "docs/reviews/wme/wme-powersim-pricing-2026-11-17.json")
    )
    a = ap.parse_args()
    if a.mode == "run":
        PS.run_full(a.board_a_approval)
        return
    t0 = time.time()
    inp = PS.pilot2_inputs(REPORT)
    rec = B.bank_recipe()
    sizes = PS.component_test_sizes(rec)
    pricing = PS.price(inp, sizes, pilot_reps=a.pilot_reps, n_boot=a.n_boot)
    fold = PS.measure_nested_fold_seconds()
    ht_cs = pricing["claims"]["ht"]["cell_x_scenario"]
    nested = {
        "fold_fit_seconds": fold,
        "full_ht_grid": PS.price_nested(fold["per_fold"], cell_scenarios=ht_cs),
        "chosen_cell_null_and_alt": PS.price_nested(fold["per_fold"], cell_scenarios=2),
        "note": "single-threaded synthetic-scale fits; the real bank has more observation dimensions",
    }
    val_items = sum(v["validation"] for v in X.DEFAULT_SIZES.values())
    dev_items = sum(v["dev"] for v in X.DEFAULT_SIZES.values())
    gpu = {}
    for e in PS.GRID["n_episodes"]:
        for Bq in PS.GRID["B"]:
            c = PS.Cell(n_episodes=e, B=Bq)
            for g in (False, True):
                gpu[f"episodes={e},B={Bq},grounding={g}"] = PS.gpu_cost(
                    inp, rec["n_states_max"], val_items, dev_items, c, g
                )
    out = {
        "label": "power-simulation PRICING only (CPU pilot); the full simulation has not been run",
        "inputs": inp,
        "component_test_sizes": sizes,
        "requirements": PS.REQUIREMENTS,
        "grid": PS.GRID,
        "scenarios": PS.SCENARIOS,
        "defaults": PS.DEFAULTS,
        "pricing": pricing,
        "nested_pipeline_pricing": nested,
        "gpu_feasibility": gpu,
        "panel_items": {"validation": val_items, "dev": dev_items},
        "bank_states_max": rec["n_states_max"],
        "wall_s": round(time.time() - t0, 1),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(
        json.dumps(
            {"cpu_h_total": pricing["cpu_h_total"], "nested": nested, "wall_s": out["wall_s"]},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
