"""wm.powersim: the outcome-blind §8 simulator (Board A condition 11). Small replicate counts only; the full
>= 1,000-replicate run is gated on Board A approval and never runs in tests."""

from pathlib import Path

import pytest

from wsp.wm import bank as B
from wsp.wm import powersim as PS

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/reviews/wme/wme-pilot2-2026-11-16.json"


@pytest.fixture(scope="module")
def inp():
    return PS.pilot2_inputs(REPORT)


@pytest.fixture(scope="module")
def sizes():
    return PS.component_test_sizes(B.bank_recipe())


def test_inputs_are_pilot2_design_quantities(inp):
    se = inp["paired_se_validation_n64"]
    assert se["sycophancy_real"] == pytest.approx(0.0668, abs=1e-3) and set(se) == set(PS.OBJ6)
    assert inp["between_candidate_sd_robust"] == pytest.approx(0.1167, abs=1e-3)
    assert inp["between_candidate_sd_raw"] > 5 * inp["between_candidate_sd_robust"]
    assert inp["seconds"]["oracle_grad_per_state"] == pytest.approx(44.17, abs=0.1)


def test_component_sizes_come_from_the_bank_graph(sizes):
    assert len(sizes) == B.bank_recipe()["components"]["source"]["n_components"]
    assert sum(sizes) > 0 and sizes == sorted(sizes, reverse=True)


def test_alt_calibration_hits_the_registered_ct():
    p = dict(PS.DEFAULTS)
    w = PS.calibrate_wm_scale(0.15, p, n=200_000)
    w0 = PS.calibrate_wm_scale(0.0, p, n=200_000)
    assert 0 < w < w0


def test_ht_cell_reports_size_power_and_both_tails(inp, sizes):
    c = PS.Cell(n_components=44)
    null = PS.run_cell("ht", c, "null", 60, inp, sizes, n_boot=200)
    alt = PS.run_cell("ht", c, "alt", 60, inp, sizes, n_boot=200)
    for r in (null, alt):
        assert {
            "rejection_rate",
            "rejection_ci95",
            "lower_tail_miss",
            "upper_tail_miss",
            "power_lb_one_sided",
        } <= set(r)
        lo, hi = r["rejection_ci95"]
        assert lo <= r["rejection_rate"] <= hi
    assert alt["rejection_rate"] >= null["rejection_rate"]
    assert not PS.evaluate_requirements(null)["pass"]  # 60 < 1,000 replicates can never pass


def test_e1_nulls_and_alternative(inp, sizes):
    c = PS.Cell(n_test=160, n_gen=160, B=32)
    z = PS.run_cell("e1", c, "null_zero", 200, inp, sizes)
    ng = PS.run_cell("e1", c, "null_no_generation", 200, inp, sizes)
    nd = PS.run_cell("e1", c, "null_neutral_destroyed", 200, inp, sizes)
    alt = PS.run_cell("e1", c, "alt", 200, inp, sizes)
    assert (
        z["rejection_rate"] <= 0.05
        and ng["rejection_rate"] <= 0.07
        and nd["rejection_rate"] <= 0.07
    )
    assert alt["rejection_rate"] > max(z["rejection_rate"], ng["rejection_rate"])


def test_e2_family_clusters_are_not_estimable_and_holm_fwer_is_reported(inp, sizes):
    c = PS.Cell(n_episodes=12, n_test=160, B=32)
    fam = PS.run_cell("e2", c, "null_family_clusters", 30, inp, sizes, n_boot=200)
    assert fam["rejection_rate"] == 0 and fam["n_estimable"] == 0
    null = PS.run_cell("e2", c, "null", 200, inp, sizes, n_boot=300)
    assert "fwer_secondary" in null and 0 <= null["fwer_secondary"] <= 1
    alt = PS.run_cell("e2", c, "alt", 200, inp, sizes, n_boot=300)
    assert alt["rejection_rate"] > null["rejection_rate"]


def test_grid_covers_every_open_n():
    assert set(PS.GRID) == {"n_components", "n_episodes", "n_test", "n_gen", "B"}
    assert all(c.n_gen <= c.n_test for c in PS.grid_cells("e1"))
    assert len(PS.grid_cells("e2")) == len(PS.GRID["n_episodes"]) * len(PS.GRID["n_test"]) * len(
        PS.GRID["B"]
    )


def test_pricing_extrapolates_cpu_hours(inp, sizes):
    pr = PS.price(inp, sizes, pilot_reps=3, n_boot=100, cells_per_claim=1)
    assert set(pr["claims"]) == {"ht", "e1", "e2"} and pr["cpu_h_total"] > 0
    for v in pr["claims"].values():
        assert v["cpu_h_full"] == pytest.approx(
            v["seconds_per_replicate"] * 1000 * v["cell_x_scenario"] / 3600
        )


def test_gpu_cost_reprices_grounding_and_flags_infeasibility(inp):
    rec = B.bank_recipe()
    c = PS.Cell()
    no = PS.gpu_cost(inp, rec["n_states_max"], 520, 672, c, grounding=False)
    yes = PS.gpu_cost(inp, rec["n_states_max"], 520, 672, c, grounding=True)
    assert yes["oracle_grounding_gpu_h"] == pytest.approx(
        rec["n_states_max"] * 44.17 / 3600, rel=1e-2
    )
    assert yes["bank_total_gpu_h"] > no["bank_total_gpu_h"] and not yes["fits_bank"]
    assert isinstance(no["fits_confirmatory_primary"], bool)


def test_full_run_is_gated():
    with pytest.raises(PS.NotApproved):
        PS.run_full(None)


def test_nested_pricing_arithmetic():
    r = PS.price_nested(30.0, folds=5, reps=1000, cell_scenarios=2)
    assert r["seconds_per_replicate"] == 150.0 and r["cpu_h"] == pytest.approx(
        2 * 150 * 1000 / 3600
    )
