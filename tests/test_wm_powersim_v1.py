"""v1 power simulator (PREREG_WME v1 §8; RULING-wme-resources §4). CPU, small replicate counts."""

import math

import numpy as np
import pytest

from wsp.wm import confirm_stats as CS
from wsp.wm import powersim_v1 as PV

INP = {
    "se_ratio": np.ones(6),
    "syc_se_64": 0.024,
    "sd_cand": 0.81,
    "sd_cand_mad": 0.117,
    "guard_se_64": {
        g: {"pp": 0.003, "native": 0.01}
        for g in (
            "wikitext_nll",
            "gsm8k_gold_lp",
            "mathcomp_gold_lp",
            "mmlupro_gold_lp",
            "syc_neutral",
            "refusal_margin_harmful",
            "refusal_margin_harmless",
        )
    },
    "guard_point_sd_item": 0.02,
}


def test_grid_matches_prereg_v1():
    assert PV.GRID["r_f"] == (8, 12, 16, 20) and PV.GRID["B"] == (16, 24, 32, 48)
    assert PV.GRID["n_test"] == (96, 128, 192, 256) and PV.GRID["n_inert"] == (12, 24, 48)
    assert PV.ALPHA == 0.025 and PV.REQ["size_max"] == 0.035 and PV.REQ["coverage_min"] == 0.965


def test_graph_blocks_and_singletons():
    g = PV.graph(8, 6)
    assert g["n_components"] == 6 + (48 - 24)
    blocks = [c for c in g["components"] if c["stratum"] == "block"]
    assert len(blocks) == 6 and all(len(set(c["families"])) == 4 for c in blocks)
    assert PV.graph(8, 24) is None  # 96 block roots > 48 roots
    assert PV.graph(8, 6) == PV.graph(8, 6)


def test_vectorised_bootstrap_t_matches_reference_shape():
    rng = np.random.default_rng(0)
    m = 40
    S = rng.gamma(4, 1, m)
    A = S * (0.85 + 0.1 * rng.standard_normal(m))
    r = PV.ct_bootstrap_t(A, S, ["a"] * m, n_boot=1999, seed=1)
    ref = CS.ct_lower_bound(A, S, list(range(m)), n_boot=1999, seed=1, alpha=0.025)
    assert r["ct"] == pytest.approx(ref["ct"], abs=1e-12)
    assert abs(r["lb"] - ref["lb"]) < 0.02  # different resample streams, same estimator
    assert r["lb"] < r["ct"] < r["ub"]


def test_tango_interval():
    # no discordant difference -> lower limit below 0; strong positive -> above 0
    assert PV.tango_lower(10, 10, 200) < 0
    assert PV.tango_lower(40, 5, 200) > 0
    # monotone in b
    assert PV.tango_lower(30, 5, 200) < PV.tango_lower(40, 5, 200)
    # agrees with a Wald interval roughly in the interior
    b, c, n = 40, 10, 300
    d = (b - c) / n
    wald = d - 1.96 * math.sqrt((b + c) / n**2 - d**2 / n)
    assert abs(PV.tango_lower(b, c, n) - wald) < 0.02


def test_ht_null_and_alt_behave():
    cell = {"r_f": 12, "n_blk": 12, "n_cal": 12, "n_bank": 64, "n_inert": 12}
    null = PV.run_cell("ht", cell, "null_ignores", 60, INP, n_boot=499)
    assert null["rate"] == 0.0 and null["c2s_pass"] == 0.0  # void without step sensitivity
    alt = PV.run_cell("ht", cell, "alt", 60, INP, n_boot=499)
    nob = PV.run_cell("ht", cell, "null_nobetter", 60, INP, n_boot=499)
    assert alt["rate"] > nob["rate"]
    assert alt["m"] == PV.graph(12, 12)["n_components"] - 12


def test_sparse_outer_scenario_reports_strata():
    cell = {"r_f": 12, "n_blk": 12, "n_cal": 12, "n_bank": 64, "n_inert": 12}
    r = PV.run_cell("ht", cell, "sparse_outer", 30, INP, n_boot=299)
    for k in ("outer_reportable", "inner_reportable", "rule_damaging"):
        assert 0.0 <= r[k] <= 1.0
    assert r["inner_reportable"] >= r["outer_reportable"]


def test_e1_nulls_fail_and_alt_can_pass():
    cell = {"n_rep": 4, "B": 32, "n_dev": 128, "n_test": 192, "n_gen_syc": 250}
    z = PV.run_cell("e1", cell, "null_zero", 80, INP)
    assert z["rate"] <= 0.05
    nt = PV.run_cell("e1", cell, "null_no_transfer", 80, INP)
    assert nt["rate"] <= 0.06
    a = PV.run_cell("e1", cell, "alt", 80, INP)
    assert a["rate"] > nt["rate"]


def test_e2_null_size_and_alt_power_direction():
    cell = {"n_rep": 4, "B": 32, "n_dev": 128, "n_test": 128}
    n = PV.run_cell("e2", cell, "null", 60, INP, n_boot=299)
    a = PV.run_cell("e2", cell, "alt", 60, INP, n_boot=299)
    assert n["rate"] <= 0.1 and a["rate"] > n["rate"]


def test_requirements_rule():
    base = {
        "claim": "e1",
        "reps": 1000,
        "null": True,
        "rate": 0.03,
        "power_lb": 0.0,
        "scenario": "null_zero",
    }
    assert PV.passes(base)
    assert not PV.passes(dict(base, rate=0.04))
    assert not PV.passes(dict(base, reps=999))
    alt = dict(base, null=False, rate=0.84, power_lb=0.82, scenario="alt")
    assert PV.passes(alt) and not PV.passes(dict(alt, power_lb=0.79))
    ht = {
        "claim": "ht",
        "reps": 1000,
        "null": False,
        "rate": 0.9,
        "power_lb": 0.88,
        "scenario": "alt",
        "coverage": 0.96,
    }
    assert not PV.passes(ht)


def test_c2i_and_c14_rates():
    r = PV.c2i_rates(12, reps=4000)
    assert r["false_fail"] <= 0.05 and r["detect_2f"] > 0.9
    c = PV.c14_rates(12, reps=4000)
    assert c["false_switch"] < c["switch_under_15pp"]


def test_price_and_pool_demand_monotone():
    unit = {
        "per_item_s": {o: 0.02 for o in PV.OBJ6 + ("mathcomp_gold_lp",)},
        "overhead_s": 0.2,
        "oracle_s": 40,
        "field_s": 3000,
        "dirs_s_per_root": 5,
        "lora_s_per_root": 120,
        "gen_s": {"sycophancy_real": 0.3, "gsm8k_gold_lp": 1.0, "mathcomp_gold_lp": 2.0},
        "library_s": 900,
        "lora_sft_s": 600,
    }
    d = {
        "r_f": 8,
        "n_blk": 6,
        "n_cal": 12,
        "n_bank": 48,
        "n_dev": 64,
        "n_test": 96,
        "n_gen_syc": 100,
        "n_gen_gsm": 200,
        "n_gen_math": 200,
        "shard_contrast": 16,
        "shard_lora": 64,
        "e2_shard": 64,
        "n_rep": 2,
        "n_ep_sec": "half",
        "n_rep_sec": 0,
        "B": 16,
        "n_inert": 12,
        "pool_size": 64,
    }
    p1 = PV.price(d, unit)
    p2 = PV.price(dict(d, r_f=16, B=32), unit)
    assert (
        p2["bank_gpu_h"] > p1["bank_gpu_h"] and p2["confirmatory_gpu_h"] > p1["confirmatory_gpu_h"]
    )
    dem = PV.pool_demand(d)
    assert dem["sycophancy_real"] == 24 + 48 + 64 + 96 + 100 + 64
