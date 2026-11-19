"""CPU integration check of the one executable E2 configuration (wsp.wm.e2run; Board A v2 required change 2)."""

import dataclasses
import inspect

import numpy as np
import pytest
import torch

from wsp.wm import adapter9b as AD
from wsp.wm import e2run as E
from wsp.wm import model as M
from wsp.wm import synthetic as S
from wsp.wm import wmcp as CP

CFG = dataclasses.replace(E.E2_WM_CONFIG, epochs=3)  # only the epoch count is reduced for CPU speed


@pytest.fixture(scope="module")
def setup():
    world = S.SynthWorld(kind="saturating", seed=0)
    spec = CP.spec_synthetic(world)
    rng = np.random.default_rng(1)
    dirs = [u / np.linalg.norm(u) for u in rng.standard_normal((world.J, world.J))]

    def realise(steps):
        """Realised cumulative coefficients: each step adds its direction, then an order-dependent map is applied
        (a non-commuting composition, like bf16 realisation of overlapping edits)."""
        c = np.zeros(world.J)
        for a, dose in steps:
            c = c + dose * world.step * dirs[int(a[1:])]
            c = c * (1.0 + 0.3 * np.tanh(c[0] / world.step))
        return c

    def r_of(steps):
        return CP.r_synthetic(world, realise(steps))

    trajs, comps = [], []
    for j in range(6):
        for s in (1.0, -1.0):
            for steps in ([(f"d{j}", 0.5 * s)], [(f"d{j}", 0.5 * s), (f"d{(j + 1) % 6}", 0.5)]):
                cs = [realise(steps[:k]) for k in range(len(steps) + 1)]
                y = np.stack([world.outcomes(torch.as_tensor(c)).numpy() for c in cs])
                obs = np.stack([world.observe(torch.as_tensor(c)) for c in cs])
                trajs.append(E.trajectory(spec, r_of, steps, y - y[0], obs))
                comps.append(f"c{j}")
    bo = world.base_observation()
    return world, spec, r_of, trajs, comps, bo


def test_frozen_config_is_the_production_default():
    assert E.E2_WM_CONFIG.slope_transport is False
    sig = inspect.signature(E.fit_planning_predictors)
    assert sig.parameters["cfg"].default is E.E2_WM_CONFIG
    assert sig.parameters["n_members"].default == E.N_MEMBERS == 5
    eff = E.effective_config()
    assert eff["slope_transport"] is False and eff["wm_config"]["slope_transport"] is False
    assert "V0RArm" in eff["wm_arm"] and "MatchedStatic" in eff["static_arm"]
    assert "R6 not applied" in eff["shrinkage"]


def test_components_are_mandatory(setup):
    _w, spec, _r, trajs, comps, bo = setup
    with pytest.raises(ValueError):
        E.fit_planning_predictors(trajs, None, spec, bo, cfg=CFG)
    with pytest.raises(ValueError):
        E.fit_planning_predictors(trajs, comps[:-1], spec, bo, cfg=CFG)
    with pytest.raises(ValueError):
        E.fit_planning_predictors(
            trajs, comps, spec, bo, cfg=dataclasses.replace(CFG, slope_transport=True)
        )


@pytest.fixture(scope="module")
def fitted(setup):
    _w, spec, _r, trajs, comps, bo = setup
    return E.fit_planning_predictors(
        trajs, comps, spec, bo, calibration={"c5"}, cfg=CFG, n_members=2
    )


def test_production_fit_uses_frozen_choices_and_removes_A(setup, fitted):
    _world, _spec, _r, _t, _c, bo = setup
    wm, st = fitted["wm"], fitted["static"]
    assert isinstance(wm, M.WMEnsemble) and isinstance(st, M.MatchedStatic)
    assert wm.cfg == CFG and st.cfg == dataclasses.replace(CFG, static_state=True)
    assert all(
        getattr(wm.cfg, f.name) == getattr(E.E2_WM_CONFIG, f.name)
        for f in dataclasses.fields(CFG)
        if f.name != "epochs"
    )
    # component ids reached the fit: component bootstraps and a component-level selection split
    assert wm.member_samples is not None and st.member_samples is not None
    assert wm.val_index == st.val_index
    assert "c5" not in fitted["components"]
    # A_t targets removed
    lay = fitted["base_obs"].payload["layout"]
    assert "A" not in lay and "A" in bo.payload["layout"]
    a0, a1 = bo.payload["layout"]["A"]
    assert len(fitted["base_obs"].payload["obs"]) == len(bo.payload["obs"]) - (a1 - a0)
    assert "slope_target" not in wm.ctx and "slope_target" not in st.ctx
    # the same shrinkage rule and grid for both arms
    assert wm.shrink in M.SHRINK_GRID and st.shrink in M.SHRINK_GRID


def test_both_arms_receive_identical_realised_information(setup, fitted):
    world, spec, r_of, *_ = setup
    acts = [f"d{j}" for j in range(world.J)]
    doses = {a: [-1.0, -0.5, 0.5, 1.0] for a in acts}
    arms = E.make_arms(
        fitted,
        spec,
        r_of,
        acts,
        doses,
        primary=0,
        orient=1.0,
        q_hats={"wm": 1.0, "static": 1.3},
        guards=[(1, 1.0, 0.5)],
        support=None,
        mode="lcb",
    )
    assert all(isinstance(a, AD.V0RArm) for a in arms.values())
    X = np.array([[0.5, 1.0, 0.5], [1.0, 0.5, -0.5]])
    for order in ([0, 1, 0], [1, 0, 0]):  # overlapping (action 0 twice) and non-commuting orders
        tw, ts = arms["wm"]._trajs(order, X), arms["static"]._trajs(order, X)
        for a, b, row in zip(tw, ts, X, strict=True):
            for k in ("x", "e", "obs"):
                np.testing.assert_array_equal(a[k], b[k])
            steps = arms["wm"].steps(order, row)
            ref = E.trajectory(spec, r_of, steps, obs=fitted["base_obs"].payload["obs"])
            np.testing.assert_array_equal(a["x"], ref["x"])
        np.testing.assert_array_equal(arms["wm"].embed(order, X), arms["static"].embed(order, X))
    # non-commuting: the two orders of the same steps reach different realised edits
    s1 = arms["wm"].steps([0, 1], X[0, :2])
    s2 = [s1[1], s1[0]]
    assert not np.allclose(r_of(s1), r_of(s2))


def test_static_prediction_depends_on_the_realised_final_edit_only(setup, fitted):
    _w, spec, r_of, *_ = setup
    bo = fitted["base_obs"]
    a = r_of([("d0", 0.5)])
    b = r_of([("d1", 0.5)])
    c = r_of([("d0", 0.5), ("d1", 0.5)])
    t1 = CP.v0_trajectory_r(spec, [np.zeros_like(a), a, c], obs=bo.payload["obs"])
    t2 = CP.v0_trajectory_r(spec, [np.zeros_like(a), b, c], obs=bo.payload["obs"])
    p1, p2 = fitted["static"].predict(bo, t1)[0], fitted["static"].predict(bo, t2)[0]
    np.testing.assert_allclose(p1[2], p2[2], rtol=1e-6, atol=1e-8)
