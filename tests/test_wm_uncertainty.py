"""wm.uncertainty + wm.model sigma (SYNTHESIS-r9 C1 / pro-engineer W4; PREREG_WME §6 "Uncertainty prerequisites").

1. sigma floor: sigma = gate(E_t) sqrt(sigma_F^2 + (lambda sigma_ens)^2), so sigma > 0 off the zero edit even at
   shrink lambda = 0, with sigma_F the field-only residual RMS on the selection split;
2. StaticPredictor returns (mean, sigma) with the same treatment;
3. ensembles are resampled by source-closure component (whole components in or out);
4. split-conformal scaling fitted on h <= 2 only, coverage checked at h = 3-4, kill when the miss exceeds 10 pp.
"""

import numpy as np
import pytest

from wsp.wm import model as M
from wsp.wm import synthetic as S
from wsp.wm import uncertainty as U

CFG = M.WMConfig(epochs=40, hidden=16, latent=8)


def _comps(trajs, size=4):
    # synthetic closure components: consecutive blocks of `size` trajectories share a source
    return [f"c{i // size}" for i in range(len(trajs))]


@pytest.fixture(scope="module")
def fitted():
    world = S.SynthWorld(kind="saturating", seed=0)
    train = world.trajectories(n=96, horizon=2, seed=1)
    comps = _comps(train)
    wm = M.WMEnsemble(CFG, world.field(), world.base_observation(), n_seeds=3).fit(
        train, components=comps
    )
    st = M.StaticPredictor(CFG, world.field(), n_seeds=3).fit(train, components=comps)
    return world, train, comps, wm, st


# ------------------------------------------------------------------ component bootstrap


def test_component_bootstrap_keeps_whole_components_and_is_deterministic():
    comps = ["a", "a", "b", "c", "c", "c", "d"]
    boots = U.component_bootstrap(comps, n=5, seed=3)
    assert boots == U.component_bootstrap(comps, n=5, seed=3)
    assert len(boots) == 5 and len({tuple(b) for b in boots}) > 1
    for b in boots:
        counts = {}
        for i in b:
            counts.setdefault(comps[i], []).append(i)
        for c, idx in counts.items():
            members = [i for i, x in enumerate(comps) if x == c]
            # each drawn component contributes all its rows, once per draw
            assert len(idx) % len(members) == 0
            assert sorted(set(idx)) == members
        assert sum(1 for _ in b) >= 1


def test_component_selection_split_never_splits_a_component():
    comps = [f"c{i // 3}" for i in range(30)]
    fit, val = U.component_selection_split(comps, frac=0.2, seed=1)
    assert set(fit).isdisjoint(val) and sorted(fit + val) == list(range(30))
    assert {comps[i] for i in fit}.isdisjoint({comps[i] for i in val})
    assert len({comps[i] for i in val}) == 2  # 20% of 10 components


def test_members_train_on_component_bootstraps(fitted):
    _, _, comps, wm, st = fitted
    assert len(wm.member_samples) == 3 and len(st.member_samples) == 3
    val_comps = {comps[i] for i in wm.val_index}
    for sample in wm.member_samples + st.member_samples:
        drawn = {comps[i] for i in sample}
        assert drawn.isdisjoint(val_comps)  # selection components never train a member
    assert len({tuple(s) for s in wm.member_samples}) == 3


# ------------------------------------------------------------------ sigma floor


def test_sigma_floor_is_field_residual_rms_on_selection_split(fitted):
    world, train, _, wm, _ = fitted
    val = [train[i] for i in wm.val_index]
    r = np.concatenate([t["y"][1:] - S.raw_field(world, t["x"][1:, : world.J]) for t in val])
    np.testing.assert_allclose(wm.sigma_floor, np.sqrt((r**2).mean(0)), rtol=1e-9)
    assert (wm.sigma_floor > 0).all()


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_sigma_combines_floor_and_spread_and_is_positive_at_zero_shrink(fitted, lam):
    world, _, _, wm, _ = fitted
    traj = world.trajectories(n=1, horizon=3, seed=11)[0]
    base = world.base_observation()
    old = wm.shrink
    try:
        wm.shrink = lam
        _, sigma = wm.predict(base, traj)
        _, spread = wm._raw_predict(np.asarray(base.payload["obs"]), traj)
    finally:
        wm.shrink = old
    expect = np.sqrt(wm.sigma_floor**2 + (lam * spread) ** 2)
    np.testing.assert_allclose(sigma[1:], expect[1:], rtol=1e-6)
    assert (sigma[1:] > 0).all() and np.array_equal(sigma[0], np.zeros_like(sigma[0]))


def test_static_predictor_returns_sigma_with_the_same_treatment(fitted):
    world, _, _, _, st = fitted
    traj = world.trajectories(n=1, horizon=4, seed=12)[0]
    mean, sigma = st.predict(world.base_observation(), traj)
    assert mean.shape == sigma.shape == traj["y"].shape
    assert (sigma[1:] > 0).all() and np.array_equal(sigma[0], np.zeros_like(sigma[0]))
    _, spread = st._raw_predict(traj)
    np.testing.assert_allclose(
        sigma[1:], np.sqrt(st.sigma_floor**2 + (st.shrink * spread[1:]) ** 2), rtol=1e-6
    )


def test_zero_edit_keeps_exactly_zero_sigma_for_both_arms(fitted):
    world, _, _, wm, st = fitted
    zero = world.trajectory_from_actions([(0, 0.0), (5, 0.0), (2, 0.0)])
    for m in (wm, st):
        mean, sigma = m.predict(world.base_observation(), zero)
        assert np.array_equal(mean, np.zeros_like(mean))
        assert np.array_equal(sigma, np.zeros_like(sigma))


def test_fit_without_components_still_works_and_floors_sigma():
    world = S.SynthWorld(kind="quadratic", seed=1)
    train = world.trajectories(n=40, horizon=2, seed=2)
    st = M.StaticPredictor(CFG, world.field(), n_seeds=2).fit(train)
    _, sigma = st.predict(world.base_observation(), world.trajectories(1, 3, seed=3)[0])
    assert np.isfinite(sigma).all() and (sigma[1:] >= 0).all()


# ------------------------------------------------------------------ conformal


def test_conformal_quantile_is_finite_sample_corrected():
    s = np.arange(1, 11, dtype=float)  # scores 1..10
    sc = U.ConformalScaler(alpha=0.2).fit_scores(s)
    # ceil((n+1)(1-alpha)) = ceil(8.8) = 9 -> 9th smallest score
    assert sc.q == pytest.approx(9.0)


def test_conformal_calibrated_coverage_on_gaussian_errors():
    rng = np.random.default_rng(0)
    sig = rng.uniform(0.5, 2.0, size=4000)
    err = (
        rng.normal(size=4000) * sig * 1.7
    )  # sigma is miscalibrated by 1.7x; conformal fixes the scale
    sc = U.ConformalScaler(alpha=0.2).fit(np.abs(err[:2000]), sig[:2000])
    cov = sc.coverage(np.abs(err[2000:]), sig[2000:])
    assert abs(cov - 0.8) < 0.03


def test_coverage_check_kills_beyond_ten_pp_and_sets_ranks_only():
    ok = U.coverage_decision(0.74, target=0.8)
    assert ok["pass"] and not ok["ranks_only"] and ok["miss_pp"] == pytest.approx(6.0)
    bad = U.coverage_decision(0.69, target=0.8)
    assert not bad["pass"] and bad["ranks_only"]
    over = U.coverage_decision(0.95, target=0.8)
    assert not over["pass"]  # "within 10 pp" is two-sided (PREREG §6)


def test_conformal_fit_refuses_long_horizons_and_check_uses_only_three_four(fitted):
    world, _, _, wm, _ = fitted
    base = world.base_observation()
    with pytest.raises(ValueError):
        U.calibrate(wm, base, world.trajectories(n=4, horizon=3, seed=20))
    sc = U.calibrate(wm, base, world.trajectories(n=24, horizon=2, seed=21))
    assert sc.n == 24 * 2 * world.K and sc.q > 0
    out = U.check_coverage(wm, sc, base, world.trajectories(n=12, horizon=4, seed=22))
    assert out["horizons"] == [3, 4] and out["n"] == 12 * 2 * world.K
    assert 0.0 <= out["coverage"] <= 1.0 and set(out) >= {"pass", "ranks_only", "miss_pp"}


def test_calibrated_predictor_widens_sigma_by_q_and_reports_lcb(fitted):
    world, _, _, wm, _ = fitted
    base = world.base_observation()
    sc = U.calibrate(wm, base, world.trajectories(n=24, horizon=2, seed=21))
    cp = U.CalibratedPredictor(wm, sc, ranks_only=False)
    traj = world.trajectories(n=1, horizon=3, seed=23)[0]
    m0, s0 = wm.predict(base, traj)
    m1, s1 = cp.predict(base, traj)
    np.testing.assert_allclose(m0, m1)
    np.testing.assert_allclose(s1, sc.q * s0)
    np.testing.assert_allclose(cp.lcb(base, traj, orient=1.0), m0 - sc.q * s0)
    ro = U.CalibratedPredictor(wm, sc, ranks_only=True)
    with pytest.raises(U.RanksOnlyError):
        ro.lcb(base, traj, orient=1.0)
