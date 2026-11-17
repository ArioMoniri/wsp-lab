"""WM-CP: the canonical-path world model on the shared representation R(E) (RULING-wme-v2 A1, A5)."""

import numpy as np
import pytest
import torch

from wsp.wm import powersim_refit as R
from wsp.wm import synthetic as S
from wsp.wm import wmcp as CP

CFG = CP.CPConfig(epochs=60, hidden=32, latent=16)


def _states(world, rng, n_roots=10, max_h=2):
    """Ladder and two-step composition states with h <= max_h (labels from the world, no noise)."""
    out = []
    dirs = [u / np.linalg.norm(u) for u in rng.standard_normal((n_roots, world.J))]
    for i, u in enumerate(dirs):
        for s in (1.0, -1.0):
            t = R.traj_vec(world, [(u, 0.25 * s), (u, 0.25 * s)])
            for h in range(1, max_h + 1):
                out.append(CP.state_from_traj(world, t, h, comp=i))
    for i in range(0, n_roots - 1, 2):
        t = R.traj_vec(world, [(dirs[i], 1.0), (dirs[i + 1], 1.0)])
        for h in (1, 2):
            out.append(CP.state_from_traj(world, t, h, comp=i))
    return out, dirs


@pytest.fixture(scope="module")
def sat():
    w = S.SynthWorld(kind="saturating", seed=4, n_dirs=6, n_obj=3)
    st, dirs = _states(w, np.random.default_rng(0))
    m = CP.CanonicalPathWM(
        CFG, CP.spec_synthetic(w), w.base_observation().payload["obs"], n_seeds=3
    )
    m.fit(st, [s["comp"] for s in st])
    return w, m, dirs


def test_representation_scales_exactly_along_the_path():
    w = S.SynthWorld(kind="saturating", seed=1, n_dirs=5, n_obj=3)
    spec = CP.spec_synthetic(w)
    E = np.random.default_rng(0).standard_normal(5) * 0.3
    r = CP.r_synthetic(w, E)
    for t in (0.25, 0.5, 1.0):
        np.testing.assert_allclose(CP.r_at(r, spec, t), CP.r_synthetic(w, t * E), atol=1e-12)
    np.testing.assert_allclose(CP.field(r, spec), S.raw_field(w, E), atol=1e-12)


def test_k_rule_frozen_and_monotone(sat):
    _, m, _ = sat
    assert m.k_rule["s0"] > 0 and m.k_rule["k_max"] == CP.K_MAX
    ks = [m.k_of(np.array([x])) for x in (0.0, 0.5 * m.k_rule["s0"], 3.2 * m.k_rule["s0"], 1e9)]
    assert ks == [0, 1, 4, CP.K_MAX]


def test_exact_zero_edit_predicts_exactly_zero(sat):
    _w, m, _ = sat
    mean, sigma = m.predict(np.zeros(m.spec.dim))
    assert np.all(mean == 0.0) and np.all(sigma == 0.0)


def test_alternative_decompositions_identical(sat):
    w, m, dirs = sat
    u = dirs[0]
    a = R.traj_vec(w, [(u, 0.25), (u, 0.25), (u, 0.5)])  # ladder chain to dose 1
    b = R.traj_vec(w, [(u, 1.0)])  # one step
    c = R.traj_vec(w, [(u, 0.75), (u, 0.25)])
    preds = [m.predict(CP.state_from_traj(w, t, t["horizon"])["r"])[0] for t in (a, b, c)]
    for p in preds[1:]:
        np.testing.assert_allclose(p, preds[0], rtol=0, atol=1e-12)


def test_round_trip_predicts_exactly_zero(sat):
    w, m, dirs = sat
    t = R.traj_vec(w, [(dirs[1], 1.0), (dirs[1], -1.0)])
    st = CP.state_from_traj(w, t, 2)
    assert np.all(st["r"] == 0.0)
    assert np.all(m.predict(st["r"])[0] == 0.0)


def test_equals_static_on_quadratic_dynamics():
    w = S.SynthWorld(kind="quadratic", seed=2, n_dirs=5, n_obj=3)
    st, dirs = _states(w, np.random.default_rng(1), n_roots=8)
    comps = [s["comp"] for s in st]
    bo = w.base_observation().payload["obs"]
    spec = CP.spec_synthetic(w)
    cp = CP.CanonicalPathWM(CFG, spec, bo, n_seeds=2).fit(st, comps)
    s2 = CP.StaticR(CFG, spec, bo, n_seeds=2, aux=False).fit(st, comps)
    s3 = CP.StaticR(CFG, spec, bo, n_seeds=2, aux=True).fit(st, comps)
    assert cp.shrink == 0.0 and s2.shrink == 0.0 and s3.shrink == 0.0
    big = R.traj_vec(w, [(dirs[0], 1.0), (dirs[1], 2.0), (dirs[2], -1.0)])
    r = CP.state_from_traj(w, big, 3)["r"]
    f = CP.field(r, spec)
    for m in (cp, s2, s3):
        np.testing.assert_array_equal(m.predict(r)[0], f)


def test_matched_capacity_and_member_bootstrap(sat):
    _w, m, _ = sat
    s3 = CP.StaticR(CFG, m.spec, m.base_obs, n_seeds=3, aux=True)
    n_cp, n_s3 = CP.param_count(m), CP.param_count_untrained(s3, m)
    assert abs(n_cp - n_s3) / n_cp <= 0.10
    assert len(m.members) == 3 and m.member_samples is not None


def test_learns_beyond_the_field_on_saturating_dynamics(sat):
    w, m, dirs = sat
    rng = np.random.default_rng(9)
    errs_cp, errs_f = [], []
    for u in dirs[:6]:
        t = R.traj_vec(w, [(u, 0.25), (u, 0.25), (u, 0.5), (u, 1.0)])
        for h in (3, 4):
            st = CP.state_from_traj(w, t, h)
            errs_cp.append(np.abs(m.predict(st["r"])[0] - st["y"]).mean())
            errs_f.append(np.abs(CP.field(st["r"], m.spec) - st["y"]).mean())
    assert np.isfinite(errs_cp).all() and np.mean(errs_cp) <= 1.5 * np.mean(errs_f)
    assert rng is not None and torch.get_num_threads() >= 1


def test_r_on_9b_from_contractions_flags_unsupported_q():
    objs = ("wikitext_nll", "sycophancy_real")
    spec = CP.spec_9b(objs, chart_columns=("c0",), norm_groups=("writers",))
    con = {
        "wikitext_nll": {"t1": {"signed": 0.5, "q": 0.1}, "t2": {"signed": -0.2, "q": 0.05}},
        "sycophancy_real": {"t1": {"signed": 1.0, "q": None}, "t2": {"signed": 0.0, "q": 0.2}},
    }
    r, flags = CP.r_from_contractions(
        spec, objs, con, chart=[0.3], group_norms=[0.7], total_norm=0.7
    )
    np.testing.assert_allclose(r, [0.3, 1.0, 0.15, 0.2, 0.3, 0.7, 0.7])
    assert flags == ["sycophancy_real"]
    np.testing.assert_allclose(CP.field(r, spec), [0.45, 1.2])
    np.testing.assert_allclose(CP.r_at(r, spec, 0.5), [0.15, 0.5, 0.0375, 0.05, 0.15, 0.35, 0.35])
    with pytest.raises(ValueError):
        CP.r_from_contractions(spec, objs, con, chart=[], group_norms=[0.7], total_norm=0.7)


def test_v0_descriptive_arm_reads_R(sat):
    from wsp.wm import model as M

    w, m, dirs = sat
    spec = m.spec
    t = R.traj_vec(w, [(dirs[0], 0.25), (dirs[0], 0.25)])
    tr = CP.v0_trajectory(w, spec, [t["x"][h, : w.J] for h in range(3)], t["y"], t["obs"])
    f = CP.v0_field(spec)
    np.testing.assert_allclose(
        M.F3u(f.payload)(tr["x"][:, : spec.dim]),
        CP.field(np.stack([CP.r_synthetic(w, t["x"][h, : w.J]) for h in range(3)]), spec),
        atol=1e-12,
    )
    trajs = []
    for u in dirs[:6]:
        for s in (1.0, -1.0):
            tt = R.traj_vec(w, [(u, 0.25 * s), (u, 0.25 * s)])
            trajs.append(
                CP.v0_trajectory(w, spec, [tt["x"][h, : w.J] for h in range(3)], tt["y"], tt["obs"])
            )
    v0 = M.WMEnsemble(M.WMConfig(epochs=10, slope_transport=False), f, CP.v0_base_obs(w), n_seeds=2)
    v0.fit(trajs, [i // 2 for i in range(len(trajs))])
    zero = CP.v0_trajectory(
        w, spec, [np.zeros(w.J), np.zeros(w.J)], np.zeros((2, w.K)), t["obs"][:2]
    )
    assert np.all(v0.predict(CP.v0_base_obs(w), zero)[0] == 0.0)
