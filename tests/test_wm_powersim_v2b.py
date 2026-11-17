"""RULING-wme-v2b A8-A10: selection priority, caps and generation pricing, scenarios."""

import json
from pathlib import Path

import numpy as np

from wsp.wm import powersim_v1 as PV
from wsp.wm import powersim_v2 as P2
from wsp.wm import powersim_v2b as PB

INP = Path("/home/ubuntu/projects/wsp-lab/data/wme/powersim_v1/run-20261117/inputs.json")


def _pr(total, ok=True):
    return {"total_gpu_h": total, "within_caps": ok}


def test_priority_beats_cost_and_cost_breaks_ties_within_rank():
    c = [
        ("a", _pr(80), {"ht": True, "e1": True, "e2": False}, 60, 96),
        ("b", _pr(40), {"ht": False, "e1": True, "e2": True}, 60, 96),
        ("c", _pr(70), {"ht": True, "e1": True, "e2": False}, 60, 96),
        ("d", _pr(10, ok=False), {"ht": True, "e1": True, "e2": True}, 60, 96),
        ("e", _pr(20), {"ht": False, "e1": True, "e2": False}, 60, 96),
    ]
    i, cs, d, _p = PB.select(c)
    assert (i, cs, d) == (
        1,
        ("ht", "e1"),
        "c",
    )  # {HT,E1} outranks the cheaper {E1,E2}; d is over the caps
    i, cs, d, _p = PB.select([x for x in c if x[0] in ("b", "e")])
    assert (cs, d) == (("e1", "e2"), "b")
    assert PB.select([("x", _pr(5), {"ht": True, "e1": False, "e2": True}, 1, 1)]) is None


def test_generation_is_repriced_and_caps_are_the_v2b_caps():
    if not INP.exists():
        return
    unit = json.loads(INP.read_text())["unit"]
    d = {
        "r_f": 16,
        "n_blk": 6,
        "n_cal": 12,
        "n_bank": 48,
        "n_inert": 12,
        "n_rep": 2,
        "B": 16,
        "n_dev": 64,
        "n_test": 96,
        "n_gen_syc": 500,
        "n_rep_sec": 0,
        "n_gen_gsm": 200,
        "n_gen_math": 200,
        "n_ep_sec": "all",
        "shard_contrast": 16,
        "shard_lora": 64,
        "e2_shard": 64,
        "pool_size": 64,
    }
    old = PV.price(d, unit)
    same = PB.price(d, unit, unit["gen_s"]["sycophancy_real"], unit["gen_s"]["sycophancy_real"])
    np.testing.assert_allclose(same["confirmatory_gpu_h"], old["confirmatory_gpu_h"], rtol=1e-12)
    fast = PB.price(d, unit, 0.5, 0.6)
    assert fast["generation_gpu_h"] == (2 * 1000 * 0.5 + 8 * 1000 * 0.6) / 3600
    assert fast["usd_9b"] == fast["total_gpu_h"] * PB.USD_PER_GPU_H * (1 + PB.USD_OVERHEAD)
    assert PB.CAPS == {"bank_gpu_h": 35.0, "confirmatory_gpu_h": 90.0, "lambda_usd_9b": 1000.0}


def test_scenarios_and_e1_power_at_half_effect():
    s = PB.ht_scenarios(0.07)
    assert (
        s["alt"]["ct"] == 0.07
        and s["null_quadratic"]["null"]
        and s["null_quadratic"]["scale_mult"] < 1
    )
    assert PV.E1_SCENARIOS["alt_gen_half"]["gen"] == 0.5 * PV.E1_SCENARIOS["alt"]["gen"]
    assert PB.E1_SELECT[0] == "alt_gen_half" and "alt" not in PB.E1_SELECT
    if not INP.exists():
        return
    inp = json.loads(INP.read_text())
    inp["se_ratio"] = np.asarray(inp["se_ratio"])
    before = dict(P2.HT_SCENARIOS)
    r = PB.run_cell(
        "ht",
        {"r_f": 8, "n_blk": 6, "n_cal": 12, "n_bank": 48},
        "null_quadratic",
        20,
        inp,
        n_flip=199,
        ct_alt=0.07,
    )
    assert r["null"] and r["reps"] == 20 and r["ct_alt"] == 0.07
    assert P2.HT_SCENARIOS == before  # the v2 scenario table is restored
