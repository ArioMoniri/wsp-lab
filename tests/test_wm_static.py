"""Board A conditions 3-4: h <= 2 discipline for every learned stage; a static comparator matched to WM-E.

* ``MatchedStatic``: the WM-E network with the recurrence removed. The state at t is a function of the base
  observation and the cumulative oriented edit E_t only, with identical parameter count, normalisation, labels,
  auxiliary grounding targets, shrinkage grid, seeds and component bootstrap. Its prediction at T is a function of
  the realised final edit only (invariant over consistent decompositions).
* auxiliary-target ablation (``obs_weight=0``) for either arm;
* ``AnchorShift``: static prediction shifted by the residual at the nearest evaluated anchor (r9 C7 planning
  comparator; same observations as WM, no state);
* every learned stage records its maximum training horizon (<= 2); the optional F3u refit refuses h > 2.
"""

import dataclasses

import numpy as np
import pytest
import torch

from wsp.wm import controls as C
from wsp.wm import model as M
from wsp.wm import synthetic as S

CFG = M.WMConfig(epochs=30, hidden=16, latent=8)


@pytest.fixture(scope="module")
def pair():
    world = S.SynthWorld(kind="saturating", seed=0)
    train = world.trajectories(n=60, horizon=2, seed=1)
    comps = [i // 3 for i in range(60)]
    wm = M.WMEnsemble(CFG, world.field(), world.base_observation(), n_seeds=2).fit(train, comps)
    ms = M.MatchedStatic(CFG, world.field(), world.base_observation(), n_seeds=2).fit(train, comps)
    return world, train, wm, ms


def _nparams(m):
    return sum(p.numel() for p in m.members[0].net.parameters())


def test_matched_static_has_identical_capacity_and_training_contract(pair):
    _, _, wm, ms = pair
    assert _nparams(wm) == _nparams(ms)
    assert ms.cfg.static_state and not wm.cfg.static_state
    assert dataclasses.replace(ms.cfg, static_state=False) == wm.cfg
    assert ms.shrink in M.SHRINK_GRID and len(ms.members) == len(wm.members)
    assert [m.seed for m in ms.members] == [m.seed for m in wm.members]
    assert ms.member_samples == wm.member_samples  # the same component bootstraps
    assert ms.val_index == wm.val_index


def test_matched_static_is_a_function_of_the_realised_final_edit(pair):
    world, _, _, ms = pair
    retoken = lambda a: (S.action_tokens(world, a), S.bookkeeping(world, a))

    def realise(actions):
        c = np.zeros(world.J)
        for j, d in actions:
            c[j] += d * world.step
        return c

    base = world.base_observation()
    for t in world.trajectories(n=6, horizon=3, seed=4):
        p = ms.predict(base, t)[0][3]
        for a in C.consistent_decompositions(t, retoken, realise):
            np.testing.assert_allclose(ms.predict(base, a)[0][a["horizon"]], p, rtol=0, atol=1e-6)


def test_matched_static_zero_edit_is_exactly_zero(pair):
    world, _, _, ms = pair
    zero = world.trajectory_from_actions([(0, 0.0), (3, 0.0)])
    mean, sigma = ms.predict(world.base_observation(), zero)
    assert np.array_equal(mean, np.zeros_like(mean)) and np.array_equal(sigma, np.zeros_like(sigma))


def test_auxiliary_ablation_removes_grounding_gradients():
    world = S.SynthWorld(kind="saturating", seed=0)
    train = world.trajectories(n=12, horizon=2, seed=1)
    cfg = M.aux_ablation(dataclasses.replace(CFG, epochs=1))
    assert cfg.obs_weight == 0.0
    for cls in (M.WMEnsemble, M.MatchedStatic):
        m = cls(cfg, world.field(), world.base_observation(), n_seeds=1).fit(train)
        mem = m.members[0]
        b = M.batch([t for t in train if t["horizon"] == 2])
        mem.net.zero_grad()
        sum(mem.loss_by_horizon(b).values()).backward()
        g = sum(p.grad.abs().sum() for p in mem.net.dec.parameters() if p.grad is not None)
        assert float(g) == 0.0


def test_stage_audit_records_horizons_at_most_two(pair):
    _, _, wm, ms = pair
    for m in (wm, ms):
        assert set(m.stage_audit) >= {"members", "normalisation", "shrinkage", "sigma_floor"}
        assert max(m.stage_audit.values()) <= 2


def test_f3u_refit_refuses_outer_or_long_horizon_states():
    world = S.SynthWorld(kind="saturating", seed=0)
    f = M.F3u(world.field().payload)
    ok = world.trajectories(n=20, horizon=2, seed=1)
    M.refit_f3u(f, ok)
    assert np.isfinite(f.beta).all()
    with pytest.raises(ValueError):
        M.refit_f3u(f, world.trajectories(n=4, horizon=3, seed=2))


def test_anchor_shift_corrects_by_nearest_evaluated_residual(pair):
    world, _, _, ms = pair
    base = world.base_observation()
    anchors = world.trajectories(n=4, horizon=2, seed=7)
    ash = M.AnchorShift(ms, base, anchors)
    a = anchors[0]
    # at an anchor's own endpoint, the shifted prediction equals the observed outcome
    np.testing.assert_allclose(ash.predict(base, a)[0][2], a["y"][2], atol=1e-9)
    far = world.trajectories(n=1, horizon=3, seed=8)[0]
    m0, s0 = ms.predict(base, far)
    m1, s1 = ash.predict(base, far)
    assert m1.shape == m0.shape and np.array_equal(s1, s0)
    zero = world.trajectory_from_actions([(1, 0.0)])
    assert np.array_equal(ash.predict(base, zero)[0], np.zeros((2, world.K)))
    assert torch.is_tensor(torch.as_tensor(ash.length_scale))
