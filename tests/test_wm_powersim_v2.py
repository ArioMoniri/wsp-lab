"""Cheap v2 simulator (RULING-wme-v2 A2-A4, A6). Small replicate counts; behaviour only."""

import numpy as np

from wsp.wm import powersim_v2 as P2

INP = {
    "se_ratio": np.ones(6),
    "syc_se_64": 0.02,
    "sd_cand": 0.81,
    "sd_cand_mad": 0.117,
    "guard_point_sd_item": 0.01,
    "guard_se_64": {
        g: {"pp": 0.001, "native": 0.007}
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
}
CELL = {"r_f": 12, "n_blk": 12, "n_cal": 12, "n_bank": 48, "n_inert": 12}


def test_ht_signflip_null_and_alt():
    n = P2.run_cell("ht", CELL, "null", 120, INP, n_flip=499)
    a = P2.run_cell("ht", CELL, "alt", 120, INP, n_flip=499)
    assert n["rate"] <= 0.07 and n["coverage"] >= 0.9 and a["rate"] > 0.5
    assert n["truth_mean_diff"] == 0.0 and a["mean_ct"] > 0


def test_e2_signflip_size_controlled_with_few_episodes():
    c = {"n_rep": 2, "B": 32, "n_dev": 96, "n_test": 128}
    n = P2.run_cell("e2", c, "null", 300, INP, n_flip=499)
    assert n["rate"] <= 0.06  # v1's percentile bootstrap gave ~0.14 here


def test_grid_extends_generation_n():
    assert P2.GRID["n_gen_syc"][-1] == 2000 and 250 in P2.GRID["n_gen_syc"]
