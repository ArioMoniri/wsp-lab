"""wm.powersim_w: the H-T2 and E2 best-of-N power simulations of stage W (RULING-wme-stagew §7-§8). Synthetic
worlds only; small replicate counts."""

import math

import numpy as np
import pytest

from wsp.wm import powersim_w as PW
from wsp.wm import stagew as W


@pytest.fixture(scope="module")
def cal():
    w = W.synthetic_world(seed=11)
    return PW.calibrate(w)


def test_calibration_has_truth_trajectories_and_path_variance(cal):
    assert set(cal["traj"]) == {"r1", "r2", "r3", "r4"}
    assert cal["traj"]["r1"]["S"].shape == (8, len(W.FEATURES))
    for p in W.PANELS:
        tf, tc = cal["tau"][p]
        assert tf >= 0 and tc >= 0 and math.isfinite(tf) and math.isfinite(tc)


def test_simulated_worlds_have_the_requested_paths_and_kinds(cal):
    rng = np.random.default_rng(0)
    for kind in ("alt", "n1", "n2", "n3"):
        w = PW.sim_world(cal, n_paths=16, rng=rng, kind=kind)
        assert len(w["paths"]) == 18  # 16 paths -> 6 whole runs of 3
        assert {p["type"] for p in w["paths"]} == {"S", "L", "M"}
        for p in w["paths"]:
            assert len(p["keys"]) == 8 and all(k in w["members"] for k in p["keys"])
    assert PW.runs_for(40) == 14 and PW.runs_for(24) == 8


def test_no_edit_world_curves_ignore_features(cal):
    rng = np.random.default_rng(1)
    z_a = PW.truth_logits(cal, "r1", "S", 1.0, kind="n3")
    z_b = PW.truth_logits(cal, "r3", "S", 3.0, kind="n3")
    for p in W.PANELS:
        assert np.allclose(z_a[p][0], z_b[p][0]) and np.allclose(z_a[p][1], z_b[p][1])
    z_c = PW.truth_logits(cal, "r1", "S", 3.0, kind="alt")
    assert not np.allclose(z_a[W.PANELS[0]][0], z_c[W.PANELS[0]][0])
    del rng


def test_one_replicate_returns_setting_decisions(cal):
    r = PW.replicate(cal, n_paths=16, kind="alt", seed=3, lcpfn=None, n_flip=2000)
    assert set(r["reject"]) == {0, 2, 4} and all(isinstance(v, bool) for v in r["reject"].values())
    assert set(r["p_iu"]) == {0, 2, 4}


def test_empirical_bootstrap_power():
    rng = np.random.default_rng(2)
    E = {
        0: {
            "wm": rng.random(12),
            "persistence": rng.random(12) + 1.0,
            "pooled": rng.random(12) + 1.0,
        },
        2: {"wm": rng.random(12), "persistence": rng.random(12) + 0.0},
    }
    bl = {0: ["persistence", "pooled"], 2: ["persistence"]}
    res = PW.empirical_power(E, bl, n_paths=24, reps=200, seed=0)
    assert res[0] > 0.9 and res[2] < 0.5


def test_clopper_pearson_futility_bound():
    assert PW.cp_upper(0, 400) < 0.01
    assert PW.cp_upper(300, 400) > 0.75
    assert PW.futile({0: 250, 2: 100, 4: 0}, 400) is True
    assert PW.futile({0: 330, 2: 100, 4: 0}, 400) is False


# ------------------------------------------------------------------ E2 best-of-N


def test_e2_inputs_follow_the_plan():
    inp = PW.e2_inputs()
    assert inp["n_episodes"] == 24 and inp["B"] == 48 and inp["n_dev"] == 64 and inp["n_test"] == 96
    assert inp["budget_bon"] > inp["B"]
    assert math.isclose(
        inp["budget_bon"],
        inp["B"]
        + (inp["bank_gpu_h"] * 3600 + inp["field_s"]) / inp["n_episodes"] / inp["t_bundle_s"],
    )
    assert 0 < inp["rho_hat"] < 1
    assert set(inp["pools"]) == {"steer", "proj_remove", "lora_tv"}


def test_e2_successive_halving_spends_its_budget():
    c16, c64 = PW.sh_costs(14.58, 1.03)
    assert 0 < c16 < c64 <= 1.0
    inc = PW.promote_cost(14.58, 1.03)
    assert inc > c64 - c16
    k = PW.screened(48, 14.58, 1.03)
    assert k * c16 + math.ceil(k / 4) * inc <= 48 + 1e-9
    assert (k + 1) * c16 + math.ceil((k + 1) / 4) * inc > 48


def test_e2_null_is_exchangeable_and_a_perfect_prior_helps():
    inp = PW.e2_inputs()
    null = PW.e2_power(inp, rho=0.0, budget_bon=inp["B"], reps=150, seed=1, exchangeable=True)
    assert null["rate"] <= 0.08
    strong = PW.e2_power(inp, rho=1.0, budget_bon=inp["B"], reps=150, seed=2)
    assert strong["mean_d"] > 0
