"""wm.wmm.powersim: samplers reproduce their inputs, power behaves, the registered claims are what is simulated."""

import numpy as np
import pytest

from wsp.wm.wmm import powersim as S


def test_paired_binary_sampler_matches_d_and_m():
    rng = np.random.default_rng(0)
    e, b = S.paired_binary_sample(200_000, 0.2, -0.08, 0.25, rng)
    assert np.mean(e != b) == pytest.approx(0.2, abs=0.005)
    assert np.mean(e - b) == pytest.approx(-0.08, abs=0.005)
    assert b.mean() == pytest.approx(0.25, abs=0.005)
    with pytest.raises(ValueError):
        S.paired_binary_sample(10, 0.4, 0.05, 0.03, rng)  # p10 > base rate


def _guards(n=2000, brier_m=-0.02):
    return {
        "harm_compliance": {"n": n, "d": 0.05, "m": 0.0, "base_rate": 0.4},
        "benign_over_refusal": {"n": n, "d": 0.02, "m": 0.0, "base_rate": 0.02},
        "general_accuracy": {"n": n, "d": 0.1, "m": 0.0, "base_rate": 0.75},
        "brier": {"n": n, "m": brier_m, "sd": 0.1},
    }


def test_msyc_power_increases_with_effect_and_fails_with_unpowered_brier():
    base = {
        "adoption": {"n": 1500, "d": 0.2, "m": -0.08, "base_rate": 0.25, "trunc": 0.03},
        "neutral": {"n": 3500, "d": 0.1, "m": 0.0, "base_rate": 0.65},
        "guards": _guards(),
        "alpha_f": 0.0125,
    }
    hi = S.msyc_power(base, reps=60, seed=1)
    assert hi["primary"] > 0.9 and hi["pass"] > 0.6
    lo = S.msyc_power({**base, "adoption": {**base["adoption"], "m": -0.01}}, reps=60, seed=1)
    assert lo["primary"] < hi["primary"]
    g = _guards()
    g["brier"] = {"n": 3500, "m": 0.0, "sd": 0.44}
    nob = S.msyc_power({**base, "guards": g}, reps=60, seed=1)
    assert nob["brier"] < 0.4 and nob["pass"] < hi["pass"]


def test_mte_sampler_and_power():
    rng = np.random.default_rng(0)
    n0 = 2000
    triples = (
        (rng.random(n0) < 0.64).astype(int),
        rng.integers(300, 3000, n0),
        np.zeros(n0, int),
    )
    spec = {
        "n": 3000,
        "mu": -1.0,
        "sd": 0.5,
        "q": 0.5,
        "delta_full": 0.0,
        "guards": _guards(),
        "alpha_f": 0.0125,
    }
    d = S.mte_sample(spec, triples, rng)
    assert np.median(d["edit"][1]) < np.median(d["base"][1]) * 0.5
    pw = S.mte_power(spec, triples, reps=30, seed=2)
    assert pw["tokens"] == 1.0 and pw["low_budget"] > 0.9
    zero = S.mte_power(
        {**spec, "mu": 0.0, "sd": 1e-6}, triples, reps=30, seed=2
    )  # no length change
    assert zero["low_budget"] < 0.2


def test_ma_power_monotone_in_feature_information():
    lo = S.ma_power(spread_se=2.0, r_feature=0.0, reps=12, seed=0)
    hi = S.ma_power(spread_se=2.0, r_feature=0.9, reps=12, seed=0)
    assert hi["power"] >= lo["power"]
    assert lo["unit_se"] == pytest.approx(np.sqrt(0.18 / 200))


def test_wilson_upper():
    assert S.wilson_upper(13, 100) == pytest.approx(0.2098, abs=1e-3)
