"""Cheap grid, pricing and selection under RULING-wme-v2b (A7-A10). Extends ``wm.powersim_v2``.

* **H-T (A7).** The cheap error-level model of v2 with the scenario set replaced:
  - ``alt``: C_T = the achieved C_T of the registered A7 alternative, measured by full-refit replicates on
    calibration seeds at the calibrated magnitude (a derived quantity of the registered alternative, not a tuning
    target; :data:`HT_SELECT` takes it as ``ct_alt``);
  - ``null`` and ``null_heavy`` (WM-CP exchangeable with the selected static member, normal and t3 tails);
  - ``null_quadratic`` (A7's paired null: in a quadratic world both models sit on the exact field, so their
    errors are label noise plus a small shared model error, :data:`QUAD_SCALE` of the static scale).
  The full refit (``wm.powersim_refit_v2b``) remains the §4(b) gate.
* **E1 (A10).** Powered at 0.5x the generated (pilot-template) effect, ``alt_gen_half`` (gen 0.025); the full
  effect ``alt`` (gen 0.05) is a sensitivity. Nulls unchanged.
* **E2.** Unchanged from v2 (the v2 rows are reused).
* **Pricing (A9).** ``powersim_v1.price`` with the P-H100b generation seconds per answer, greedy and sampled
  separately, at the registered generation batch; caps: bank <= 35, confirmatory <= 90 GPU-h-eq, Lambda 9B
  <= $1,000 at :data:`USD_PER_GPU_H` x (1 + :data:`USD_OVERHEAD`).
* **Selection (A8).** Claim-set priority {HT, E1, E2} > {HT, E1} > {E1, E2} > {E1}; the cheapest powered design
  within the highest feasible rank (ties: more held-out components, then more test items).
"""

from __future__ import annotations

from wsp.wm import powersim_v1 as PV
from wsp.wm import powersim_v2 as P2

GEN_GRID = (500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000)
QUAD_SCALE = 0.25
CAPS = {"bank_gpu_h": 35.0, "confirmatory_gpu_h": 90.0, "lambda_usd_9b": 1000.0}
USD_PER_GPU_H = 4.89  # Lambda gpu_1x_h100_sxm5 list price, 17 Nov 2026 (GPU-h-eq are H100 hours)
USD_OVERHEAD = 0.10  # instance setup, model download and idle margin
PRIORITY = (("ht", "e1", "e2"), ("ht", "e1"), ("e1", "e2"), ("e1",))
E1_POWER = "alt_gen_half"
E1_SELECT = (E1_POWER,) + tuple(s for s in PV.E1_SELECT if s != "alt")
E1_SENSITIVITY = ("alt",)


def ht_scenarios(ct_alt: float) -> dict:
    return {
        "null": {"ct": 0.0, "null": True},
        "null_heavy": {"ct": 0.0, "tails": "t3", "null": True},
        "null_quadratic": {"ct": 0.0, "null": True, "scale_mult": QUAD_SCALE},
        "alt": {"ct": float(ct_alt)},
    }


HT_SELECT = ("null", "null_heavy", "null_quadratic", "alt")


def run_cell(claim, cell, scenario, reps, inp, n_flip=P2.N_FLIP_GRID, ct_alt=None):
    if claim == "ht":
        saved = dict(P2.HT_SCENARIOS)
        P2.HT_SCENARIOS.update(ht_scenarios(ct_alt))
        try:
            r = P2.run_cell("ht", cell, scenario, reps, inp, n_flip=n_flip)
        finally:
            P2.HT_SCENARIOS.clear()
            P2.HT_SCENARIOS.update(saved)
        r["ct_alt"] = ct_alt
        return r
    return P2.run_cell(claim, cell, scenario, reps, inp, n_flip=n_flip)


def passes(res) -> bool:
    return P2.passes(res)


def price(design: dict, unit: dict, gen_greedy_s: float, gen_sampled_s: float) -> dict:
    """``powersim_v1.price`` with sycophancy generation re-priced: 2 models x 2 templates x N greedy answers and
    4 seeds x 2 x 2 x N sampled answers, at the P-H100b seconds per answer."""
    u = dict(unit, gen_s=dict(unit["gen_s"], sycophancy_real=0.0))
    p = PV.price(design, u)
    n = 2 * design["n_gen_syc"]
    gen_h = (2 * n * gen_greedy_s + 4 * 2 * n * gen_sampled_s) / 3600
    conf = p["confirmatory_gpu_h"] + gen_h
    total = p["bank_gpu_h"] + conf
    usd = total * USD_PER_GPU_H * (1 + USD_OVERHEAD)
    return dict(
        p,
        confirmatory_gpu_h=conf,
        total_gpu_h=total,
        generation_gpu_h=gen_h,
        usd_9b=usd,
        within_caps=bool(
            p["bank_gpu_h"] <= CAPS["bank_gpu_h"]
            and conf <= CAPS["confirmatory_gpu_h"]
            and usd <= CAPS["lambda_usd_9b"]
        ),
    )


def select(candidates):
    """``candidates``: iterable of (design, price, passed {claim: bool}, m, n_test). Returns (rank index, claim set,
    design, price) for the highest-priority claim set with a powered design within caps, cheapest within it."""
    best = {}
    for d, pr, passed, m, nt in candidates:
        if not pr["within_caps"]:
            continue
        for i, cs in enumerate(PRIORITY):
            if all(passed.get(c) for c in cs):
                key = (pr["total_gpu_h"], -m, -nt)
                if i not in best or key < best[i][0]:
                    best[i] = (key, d, pr)
                break  # a design counts for its highest rank only
    if not best:
        return None
    i = min(best)
    return i, PRIORITY[i], best[i][1], best[i][2]
