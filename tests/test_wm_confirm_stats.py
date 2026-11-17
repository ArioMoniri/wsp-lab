"""wm.confirm_stats: C_T ratio with component-cluster bootstrap-t, paired bounds, Bonferroni guards, Holm, binomial."""

import numpy as np
import pytest

from wsp.wm import confirm_stats as CS


def test_ct_is_one_minus_ratio_of_absolute_error_sums():
    assert CS.ct([1, -1], [2, 2]) == pytest.approx(0.5)


def test_ct_lower_bound_is_below_estimate_and_not_estimable_with_few_clusters():
    rng = np.random.default_rng(0)
    n = 400
    clusters = np.repeat(np.arange(40), 10)
    e_st = np.abs(rng.normal(1, 0.3, n))
    e_wm = 0.8 * e_st + np.abs(rng.normal(0, 0.05, n))
    out = CS.ct_lower_bound(e_wm, e_st, clusters, n_boot=500)
    assert out["estimable"] and out["lb"] < out["ct"] and out["lb"] > 0
    few = CS.ct_lower_bound(e_wm[:30], e_st[:30], clusters[:30])
    assert not few["estimable"] and np.isnan(few["lb"])
    zero = CS.ct_lower_bound(e_wm, np.zeros(n), clusters)
    assert not zero["estimable"]


def test_component_weights_give_each_component_unit_mass():
    w = CS.component_weights(["a", "a", "b", "c", "c", "c"])
    np.testing.assert_allclose(w, [0.5, 0.5, 1, 1 / 3, 1 / 3, 1 / 3])
    out = CS.ct_lower_bound(
        np.ones(6) * 0.5, np.ones(6), ["a", "a", "b", "c", "c", "c"], weights="component"
    )
    assert out["weights"] == "component" and not out["estimable"]  # 3 clusters < 5


def test_bootstrap_t_has_near_nominal_one_sided_coverage_under_a_simple_model():
    rng = np.random.default_rng(1)
    miss = 0
    reps = 150
    for r in range(reps):
        clusters = np.repeat(np.arange(30), 4)
        e_st = np.abs(rng.normal(1, 0.5, 120))
        e_wm = np.abs(e_st * 0.9 + rng.normal(0, 0.2, 120))
        true_ct = 1 - 0.9  # approximately; coverage is checked loosely
        lb = CS.ct_lower_bound(e_wm, e_st, clusters, n_boot=300, seed=r)["lb"]
        miss += lb > true_ct + 0.03
    assert miss / reps < 0.15


def test_paired_lb_holm_bonferroni_and_binomial():
    d = np.r_[np.ones(50) * 0.2, np.ones(50) * 0.1]
    out = CS.paired_lb(d + np.random.default_rng(0).normal(0, 0.01, 100))
    assert out["lb"] < out["mean"] < out["ub"]
    assert CS.bonferroni_z(1) == pytest.approx(1.959964, abs=1e-5)
    assert CS.bonferroni_z(6) > CS.bonferroni_z(1)
    g = CS.guard_pass({"w": (0.0, 0.001, "ucb", 0.01), "m": (-0.02, 0.001, "lcb", -0.01)})
    assert g["breaches"] == ["m"] and not g["pass"]
    h = CS.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h == {"a": True, "c": False, "b": False}
    lo, hi = CS.clopper_pearson(50, 1000)
    assert lo < 0.05 < hi and CS.one_sided_lower(1000, 1000) > 0.99
