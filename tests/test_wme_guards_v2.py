"""Registered S-TE generated-guard test: alpha 0.025 in code and the sparse-outcome rule (Board A v2 change 5)."""

import numpy as np
import pytest

from wsp.wm import guards_v2 as GV
from wsp.wm import p3gen as G
from wsp.wm import prereg_v2 as P


def test_alpha_is_bound_to_0025():
    assert GV.GUARD_ALPHA == 0.025
    rng = np.random.default_rng(0)
    base = (rng.random(523) < 0.5).astype(float)
    cand = np.where(rng.random(523) < 0.1, 1 - base, base)
    r = GV.guard_test(base, cand, 0.04, base, cand, 0.04)
    assert r["alpha"] == 0.025 and r["over_refusal_method"] == "bootstrap_t"
    d = cand - base
    assert r["over_refusal_ucb"] == G.bootstrap_t_ucb(d, n_boot=GV.N_BOOT, alpha=0.025, seed=0)
    # the pilot-3 function threads its level through (default 0.05 kept for reproduction)
    lo = G.guard_test(base, cand, 0.04, base, cand, 0.04, alpha=0.025)["over_refusal_ucb"]
    hi = G.guard_test(base, cand, 0.04, base, cand, 0.04)["over_refusal_ucb"]
    assert lo > hi


def test_degenerate_samples_use_the_conservative_rule():
    z = np.zeros(200)
    r = GV.ucb(z)
    assert r["method"] == "sparse_cp" and r["ucb"] > 0  # not the point mean 0
    assert r["ucb"] == pytest.approx(1 - 0.025 ** (1 / 200))
    one = np.zeros(200)
    one[:3] = 1
    assert GV.ucb(one)["method"] == "sparse_cp"
    with pytest.raises(ValueError):
        GV.ucb(np.array([0.5, 0.0]))


@pytest.mark.parametrize(
    "n,delta,p_minus",
    [(200, 0.0234375, 0.0), (200, 0.0234375, 0.005), (523, 0.0390625, 0.0), (200, 0.01, 0.0)],
)
def test_sparse_boundary_nulls_pass_at_most_alpha(n, delta, p_minus):
    """True mean increase exactly at the margin with sparse discordance: the guard must pass (wrongly) with
    probability <= alpha (+ Monte Carlo tolerance). The bootstrap-t alone fails this (e.g. 0.15 at n 200)."""
    rng = np.random.default_rng(n + int(1e4 * delta) + int(1e5 * p_minus))
    p_plus = delta + p_minus
    reps, passed = 400, 0
    for s in range(reps):
        u = rng.random(n)
        d = np.where(u < p_plus, 1.0, np.where(u < p_plus + p_minus, -1.0, 0.0))
        passed += GV.ucb(d, n_boot=500, seed=s)["ucb"] <= delta
    assert passed / reps <= 0.025 + 3 * np.sqrt(0.025 * 0.975 / reps)


def test_te_sizing_reproduces_the_registered_n():
    for v in P.TE_SIZING.values():
        assert P.te_n(v["delta"], v["discordant"], v["n_probe"]) == v["n"]
    assert P.wilson_upper(33, 256) == pytest.approx(0.167275, abs=1e-6)
