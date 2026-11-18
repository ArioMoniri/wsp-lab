#!/usr/bin/env python3
"""RULING-wme-prereg-v2-issues item 4: S-BoN power at the measured-basis parity budget (CPU only, synthetic).

    wme_v2c_sbon.py --out OUT.json [--reps 1000]

Identical to ``wme_w_analyse.py e2sim`` (stage-W inputs ``powersim_w.e2_inputs``, the same seeds 1000 + i and
2000, 10,000 flips) except that the WM's bank charge is the bank re-priced without the per-state oracle gradients that
RULING-wme-post-pilot3 P1 removed (``powersim_v2b.price`` with ``oracle_s = 0``, the same code and unit times), so
B_par = B + (bank_P1 * 3600 + field) / n_episodes / t_bundle. In the confirmatory run B_par is fixed from ledger
seconds (PREREG v2 §II.5); this is its planning value. S-BoN sits in the 10-test H-E2 secondary Holm family, so the
power is reported at alpha 0.0025 (Holm's first step), 0.0125 (stage W's reference level) and 0.025 (last step).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import powersim_v2b as PB
from wsp.wm import powersim_w as PW
from wsp.wm import prereg_v2 as P


def p1_bank_gpu_h() -> float:
    v2b = json.loads((ROOT / "docs/reviews/wme/wme-powersim-v2b-2026-11-17.json").read_text())
    man = json.loads((ROOT / "docs/prereg/wme_manifest.json").read_text())
    unit = dict(man["cost.unit_times"]["unit"], oracle_s=0.0)
    g = v2b["generation"]
    return PB.price(v2b["selected"], unit, g["greedy_s_per_answer"], g["sampled_s_per_answer"])[
        "bank_gpu_h"
    ]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    inp = PW.e2_inputs()
    bank = p1_bank_gpu_h()
    b_par = P.parity_budget(inp["B"], bank, inp["field_s"], inp["n_episodes"], inp["t_bundle_s"])
    inp = dict(inp, bank_gpu_h_v2b=inp["bank_gpu_h"], bank_gpu_h=bank, budget_bon=b_par)
    rho = inp["rho_hat"]
    runs = {
        "measured_parity": {"rho": rho, "budget_bon": b_par},
        "null_exchangeable": {"rho": 0.0, "budget_bon": inp["B"], "exchangeable": True},
        "rho_half_parity": {"rho": 0.5 * rho, "budget_bon": b_par},
        "rho_sqrt_parity": {"rho": float(np.sqrt(rho)), "budget_bon": b_par},
        "planning_no_charge": {"rho": rho, "budget_bon": inp["B"]},
        "perfect_prior_parity": {"rho": 1.0, "budget_bon": b_par},
    }
    out = {
        "label": "S-BoN power at the P1 (measured-basis) parity budget; RULING-wme-prereg-v2-issues item 4",
        "inputs": {k: v for k, v in inp.items() if k != "pools"},
        "results": {},
    }
    for i, (name, kw) in enumerate(runs.items()):
        out["results"][name] = PW.e2_power(
            inp, reps=a.reps, seed=1000 + i, alpha=0.0125, n_flip=10_000, **kw
        )
        Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))
    for alpha, seed in ((0.025, 2000), (0.0025, 2001)):
        out["results"][f"measured_parity_alpha_{alpha}"] = PW.e2_power(
            inp, rho=rho, budget_bon=b_par, reps=a.reps, seed=seed, alpha=alpha, n_flip=10_000
        )
        Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))
    out["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps({k: v["rate"] for k, v in out["results"].items()}))


if __name__ == "__main__":
    main()
