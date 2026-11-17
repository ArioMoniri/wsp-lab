"""wm.model: WM-E v0 (D3) on a synthetic target with RMSNorm and saturating units (D8).

Trained on horizons <= 2 and tested on horizons 3-4 against the same-input static cumulative-edit predictor:
it must beat the static predictor on non-quadratic dynamics, show NO gain on quadratic dynamics or on shuffled
outcomes, and be exactly zero at the zero edit.
"""

import numpy as np
import pytest

from wsp.wm import access as X
from wsp.wm import model as M
from wsp.wm import synthetic as S

CFG = M.WMConfig(epochs=250, hidden=32, latent=16)


def _fit(kind, shuffle=False, seed=0):
    world = S.SynthWorld(kind=kind, seed=seed)
    train = world.trajectories(n=320, horizon=2, seed=seed + 1)
    test = world.trajectories(n=120, horizon=4, seed=seed + 2)
    if shuffle:
        train = S.shuffle_outcomes(train, seed=seed + 3, world=world)
        test = S.shuffle_outcomes(test, seed=seed + 4, world=world)
    wm = M.WMEnsemble(CFG, world.field(), world.base_observation(), n_seeds=5).fit(train)
    st = M.StaticPredictor(CFG, world.field()).fit(train)
    return world, test, wm, st


@pytest.fixture(scope="module")
def saturating():
    return _fit("saturating")


@pytest.fixture(scope="module")
def quadratic():
    return _fit("quadratic")


@pytest.fixture(scope="module")
def shuffled():
    return _fit("saturating", shuffle=True)


def _h34(world, test, model):
    err = M.horizon_errors(model, world.base_observation(), test)
    return float(np.mean([err[3], err[4]])), err


def test_beats_static_at_h3_plus_on_nonquadratic_dynamics(saturating):
    world, test, wm, st = saturating
    e_wm, by_h_wm = _h34(world, test, wm)
    e_st, by_h_st = _h34(world, test, st)
    e_field, _ = _h34(world, test, M.FieldOnly(world.field()))
    assert e_st < e_field  # the static residual is a real competitor, not a strawman
    assert e_wm < 0.8 * e_st, (by_h_wm, by_h_st)


def test_no_gain_on_quadratic_dynamics(quadratic):
    world, test, wm, st = quadratic
    e_wm, _ = _h34(world, test, wm)
    e_st, _ = _h34(world, test, st)
    scale = M.outcome_scale(test, horizons=(3, 4))
    assert e_st - e_wm <= 0.05 * scale, (e_wm, e_st, scale)
    e_field, _ = _h34(world, test, M.FieldOnly(world.field()))
    assert (
        e_field <= 1e-6 * scale
    )  # inside the quadratic regime the shared field nest is already exact


def test_no_gain_on_shuffled_outcomes(shuffled):
    """Null of the contrast: everything beyond the shared R0 field is permuted (with random signs)."""
    world, test, wm, st = shuffled
    e_wm, _ = _h34(world, test, wm)
    e_st, _ = _h34(world, test, st)
    scale = M.outcome_scale(test, horizons=(3, 4))
    assert e_st - e_wm <= 0.05 * scale, (e_wm, e_st, scale)


def test_zero_edit_predicts_exactly_zero_with_zero_sigma(saturating):
    world, _, wm, st = saturating
    zero = world.trajectory_from_actions([(0, 0.0), (5, 0.0), (2, 0.0), (1, 0.0)])
    mean, sigma = wm.predict(world.base_observation(), zero)
    assert np.array_equal(mean, np.zeros_like(mean)) and np.array_equal(sigma, np.zeros_like(sigma))
    st_mean, st_sigma = st.predict(world.base_observation(), zero)  # r9 C1.2: (mean, sigma)
    assert np.array_equal(st_mean, np.zeros_like(mean)) and np.array_equal(
        st_sigma, np.zeros_like(mean)
    )


def test_ensemble_has_five_distinct_members_and_positive_sigma_off_zero(saturating):
    world, test, wm, _ = saturating
    assert len(wm.members) == 5 and len({m.seed for m in wm.members}) == 5
    _, sigma = wm.predict(world.base_observation(), test[0])
    assert (sigma[1:] > 0).all()


def test_bookkeeping_is_input_never_predicted(saturating):
    world, test, wm, _ = saturating
    assert wm.loss_terms == ("outcome", "observation")
    traj = test[0]
    np.testing.assert_array_equal(traj["x"], S.bookkeeping(world, traj["actions"]))
    for m in wm.members:
        assert not any("bookkeeping" in n for n, _ in m.net.named_parameters())


def test_training_refuses_horizons_above_two():
    world = S.SynthWorld(kind="quadratic", seed=0)
    long = world.trajectories(n=4, horizon=3, seed=1)
    with pytest.raises(ValueError):
        M.WMEnsemble(CFG, world.field(), world.base_observation(), n_seeds=1).fit(long)


def test_predict_refuses_non_theta0_artefacts(saturating):
    world, test, wm, _ = saturating
    edited = X.Artefact("tap", "f" * 64, X.Tier.R2, world.base_observation().payload)
    with pytest.raises(X.AccessError):
        wm.predict(edited, test[0])


def test_shrinkage_is_selected_on_training_horizons_with_one_grid(saturating):
    world, _, wm, st = saturating
    assert wm.shrink in M.SHRINK_GRID and st.shrink in M.SHRINK_GRID
    tr = world.trajectories(n=20, horizon=2, seed=9)
    fit, val = M.selection_split(tr)
    assert len(val) == 4 and not {id(t) for t in fit} & {id(t) for t in val}
    lam = M.select_shrink(lambda t: t["y"] * 0 + 1.0, lambda t: t["y"] * 0, val)
    assert lam in M.SHRINK_GRID


def test_plain_permutation_shuffle_keeps_base_observation():
    world = S.SynthWorld(kind="saturating", seed=0)
    tr = world.trajectories(n=6, horizon=2, seed=1)
    sh = S.shuffle_outcomes(tr, seed=2)
    for a, b in zip(tr, sh, strict=True):
        np.testing.assert_array_equal(a["obs"][0], b["obs"][0])
        np.testing.assert_array_equal(a["x"], b["x"])


def test_scheduled_sampling_anneals_teacher_forcing():
    p = [M.teacher_forcing_prob(e, 100, p_min=0.0) for e in (0, 50, 99)]
    assert p[0] == 1.0 and p[0] > p[1] > p[2] >= 0.0


def test_multi_horizon_loss_uses_every_horizon():
    world = S.SynthWorld(kind="saturating", seed=0)
    tr = world.trajectories(n=8, horizon=2, seed=1)
    m = M.WMEnsemble(M.WMConfig(epochs=1), world.field(), world.base_observation(), n_seeds=1).fit(
        tr
    )
    terms = m.members[0].loss_by_horizon(M.batch(tr), teacher_p=0.0)
    assert sorted(terms) == [1, 2]
