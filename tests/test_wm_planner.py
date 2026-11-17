"""wm.planner: the shared R0 planner (SYNTHESIS-r9 C6 / pro-engineer W3; Board A condition 10).

* sep-CMA-ES (diagonal covariance, Ros & Hansen 2008) and CEM inner optimisers with identical budgets and bounds;
* R0 beam over action identities (<= 4 actions) with the inner optimiser over doses at each node;
* Thompson batches (one posterior draw per slot), LCB guardrails, support penalty;
* penalty mode matched across arms with a symmetric ranks-only fallback;
* the outer loop charges exactly B bundles and closes the arm; base is always available;
* the synthetic regret harness at B = 32 (the full comparison runs in scripts/wme_planner_synthetic.py).
"""

import numpy as np
import pytest

from wsp.wm import planner as PL
from wsp.wm.plan import BudgetLedger


def _ellipsoid(x):
    x = np.atleast_2d(x)
    return -np.sum((10.0 ** np.linspace(0, 2, x.shape[1])) * (x - 0.3) ** 2, axis=1)


@pytest.mark.parametrize("opt", [PL.sep_cma_es, PL.cem])
def test_inner_optimisers_improve_respect_bounds_and_are_deterministic(opt):
    lo, hi = -np.ones(5), np.ones(5)
    x0 = np.zeros(5)
    best, val, n = opt(_ellipsoid, x0, lo, hi, budget=600, seed=1)
    assert n <= 600 and (best >= lo).all() and (best <= hi).all()
    assert val > _ellipsoid(x0)[0] * 0.1  # at least a tenfold improvement from the start
    b2, _, _ = opt(_ellipsoid, x0, lo, hi, budget=600, seed=1)
    np.testing.assert_array_equal(best, b2)


def test_sep_cma_parameters_follow_ros_hansen():
    p = PL.sep_cma_params(10)
    assert p["lam"] == 4 + int(3 * np.log(10)) and p["mu"] == p["lam"] // 2
    np.testing.assert_allclose(p["weights"].sum(), 1.0)
    assert p["c1"] > 0 and p["cmu"] > 0 and p["c1"] + p["cmu"] <= 1


def test_penalty_mode_is_symmetric():
    assert PL.penalty_mode({"wm": True, "statnn": True}) == "lcb"
    assert PL.penalty_mode({"wm": True, "statnn": False}) == "ranks"
    assert PL.penalty_mode({"wm": False, "statnn": True}) == "ranks"


@pytest.fixture()
def world():
    return PL.SyntheticPlanningWorld(seed=3)


def test_beam_proposals_are_bounded_distinct_and_r0(world):
    pr = world.surrogate(seed=0)
    planner = PL.Planner(
        world.library_size,
        world.dose_bounds,
        inner="sep_cma",
        cfg=PL.PlannerConfig(inner_budget=60),
    )
    cands = planner.propose(pr, q=4, seed=0, mode="lcb", support=world.support_ref)
    assert 1 <= len(cands) <= 4 and len({PL.candidate_key(c) for c in cands}) == len(cands)
    for c in cands:
        assert len(c) <= 4 and len({a for a, _ in c}) == len(c)
        assert all(world.dose_bounds[0] <= d <= world.dose_bounds[1] for _, d in c)
    assert world.real_calls == 0  # planning never runs the edit


def test_episode_spends_exactly_b_and_returns_feasible_winner_or_base(world):
    led = BudgetLedger(32, ["wm", "statnn"])
    cfg = PL.PlannerConfig(inner_budget=40, batch=4)
    out = PL.run_episode(
        world, world.surrogate(seed=1), "wm", led, inner="sep_cma", cfg=cfg, seed=0, mode="lcb"
    )
    assert led.report()["units"]["wm"] == 32 and "wm" in led.report()["closed"]
    assert world.real_calls == 32
    if out["winner"] is not None:
        obs = out["observed"][PL.candidate_key(out["winner"])]
        assert obs["lcb"] > 0 and obs["guard_ok"]
    assert out["incumbents"].keys() >= {1, 4, 8, 16, 32}


def test_arms_share_planner_code_only_the_predictor_differs(world):
    a = PL.Planner(
        world.library_size, world.dose_bounds, inner="cem", cfg=PL.PlannerConfig(inner_budget=30)
    )
    p1, p2 = world.surrogate(seed=0), world.surrogate(seed=0)
    assert [PL.candidate_key(c) for c in a.propose(p1, 2, 0, "lcb", world.support_ref)] == [
        PL.candidate_key(c) for c in a.propose(p2, 2, 0, "lcb", world.support_ref)
    ]


def test_ranks_only_mode_ignores_sigma(world):
    pr = world.surrogate(seed=0)
    planner = PL.Planner(
        world.library_size, world.dose_bounds, inner="cem", cfg=PL.PlannerConfig(inner_budget=30)
    )
    s_r = planner.score(pr, [[(0, 1.0)]], draw=None, mode="ranks", support=None)
    pr.sigma_scale = 100.0
    s_r2 = planner.score(pr, [[(0, 1.0)]], draw=None, mode="ranks", support=None)
    np.testing.assert_allclose(s_r, s_r2)


def test_synthetic_regret_harness_runs_small():
    res = PL.synthetic_regret(
        n_worlds=2, B=8, methods=("sep_cma", "cem"), cfg=PL.PlannerConfig(inner_budget=20, batch=4)
    )
    assert set(res["methods"]) == {"sep_cma", "cem"}
    for m in res["methods"].values():
        assert len(m["regret"]) == 2 and all(r >= -1e-9 for r in m["regret"])
    assert res["decision"]["use"] in ("sep_cma", "cem")
