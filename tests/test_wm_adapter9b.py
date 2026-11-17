"""9B planner adapter: WM-E / StatNN -> the shared CEM planner (PREREG_WME v1 §6.5.3-6.5.5; RULING §5)."""

import numpy as np
import pytest

from wsp.wm import adapter9b as AD
from wsp.wm import e2 as E
from wsp.wm import model as M
from wsp.wm import plan as PL
from wsp.wm import planner as PN
from wsp.wm import synthetic as S


@pytest.fixture(scope="module")
def world():
    return S.SynthWorld(kind="saturating", seed=0)


@pytest.fixture(scope="module")
def fitted(world):
    tr = world.trajectories(n=40, horizon=2, seed=1) + world.trajectories(n=40, horizon=1, seed=2)
    comps = [i // 4 for i in range(len(tr))]
    cfg = M.WMConfig(epochs=15)
    wm = M.WMEnsemble(cfg, world.field(), world.base_observation(), n_seeds=3).fit(tr, comps)
    st = M.StaticPredictor(cfg, world.field(), n_seeds=3).fit(tr, comps)
    return wm, st, tr


def _dict(world):
    Ed = world.E.reshape(world.J, -1).numpy()
    return AD.Dictionary([f"d{j}" for j in range(world.J)], Ed @ Ed.T, unit=[world.step] * world.J)


def test_bookkeeping_is_identical_to_the_bank_path(world):
    d = _dict(world)
    acts = [(2, 0.5), (0, -1.0), (2, 0.25)]
    t = AD.trajectory(
        d, [(f"d{j}", dose) for j, dose in acts], world.base_observation().payload["obs"], K=world.K
    )
    ref = world.trajectory_from_actions(acts)
    np.testing.assert_allclose(t["x"], ref["x"], atol=1e-9)
    np.testing.assert_allclose(t["e"], ref["e"], atol=1e-12)
    assert t["horizon"] == 3 and t["obs"].shape == ref["obs"].shape


def test_predict_batch_matches_single_predict(world, fitted):
    wm, st, tr = fitted
    bo = world.base_observation()
    for model in (wm, st):
        out = AD.predict_batch(model, bo, tr[:5])
        for i, t in enumerate(tr[:5]):
            m, s = model.predict(bo, t)
            np.testing.assert_allclose(out["mean"][i], m, atol=1e-5)
            np.testing.assert_allclose(out["sigma"][i], s, atol=1e-5)
        assert out["members"].shape[0] == 3
        np.testing.assert_allclose(out["members"].mean(0), out["mean"], atol=1e-5)


def test_snap_to_registered_doses():
    av = [-0.0625, -0.03125, 0.03125, 0.0625]
    assert (
        AD.snap(0.9, av) == 0.0625
        and AD.snap(-0.04, av) == -0.03125
        and AD.snap(0.0, av) == 0.03125
    )
    assert AD.snap(-1.0, [0.25, 0.5, 1.0, 2.0]) == 0.25


def test_mahalanobis_support_penalty_is_arm_independent(world, fitted):
    _, _, tr = fitted
    R = np.stack([AD.features(t)[-1] for t in tr])
    sup = AD.MahalanobisSupport(R)
    assert np.all(sup.penalty(R) >= 0)
    assert np.mean(sup.penalty(R) > 0) <= 0.06  # excess over the training 95th percentile
    far = R.mean(0, keepdims=True) + 50 * R.std(0, keepdims=True)
    assert sup.penalty(far)[0] > 1.0


def _arm(world, model, mode="lcb", q=1.5, psi=0.5, support=None):
    d = _dict(world)
    acts = [f"d{j}" for j in range(world.J)]
    doses = {a: [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0] for a in acts}
    return AD.PredictorArm(
        model,
        world.base_observation(),
        d,
        acts,
        doses,
        primary=0,
        orient=1.0,
        q_hat=q,
        guards=[(1, 1.0, 0.5)],
        support=support,
        psi=psi,
        mode=mode,
    )


def test_arm_interface_for_planner(world, fitted):
    wm, _st, _ = fitted
    arm = _arm(world, wm)
    rng = np.random.default_rng(0)
    draw = arm.draw(rng)
    assert 0 <= draw < 3
    out = arm.predict([0, 3], np.array([[0.5, -0.3], [1.0, 1.0]]), draw)
    assert out["value"].shape == (2,) and out["sigma"].shape == (2,)
    assert np.all(out["sigma"] > 0)
    ((gm, _gs, lim),) = out["guards"]
    assert gm.shape == (2,) and lim == 0.5
    mean = arm.predict([0, 3], np.array([[0.5, -0.3]]), None)
    # the same snapped candidate gives the same prediction
    again = arm.predict([0, 3], np.array([[0.6, -0.29]]), None)
    np.testing.assert_allclose(mean["value"], again["value"])
    ranks = _arm(world, wm, mode="ranks").predict([0, 3], np.array([[0.5, -0.3]]), None)
    assert np.all(ranks["sigma"] == 0)


def test_no_online_refit(world, fitted):
    wm, _, _ = fitted
    arm = _arm(world, wm)
    before = arm.predict([1], np.array([[0.5]]), None)["value"].copy()
    arm.update(((1, 0.5),), {"gain": 99.0})
    np.testing.assert_array_equal(arm.predict([1], np.array([[0.5]]), None)["value"], before)
    assert arm.n_updates_ignored == 1


def test_arms_differ_only_in_predictor(world, fitted):
    wm, st, tr = fitted
    sup = AD.MahalanobisSupport(np.stack([AD.features(t)[-1] for t in tr]))
    a, b = _arm(world, wm, support=sup), _arm(world, st, support=sup)
    X = np.array([[0.5, 1.0]])
    pa = a.penalty([0, 1], X)
    np.testing.assert_array_equal(pa, b.penalty([0, 1], X))
    assert a.q_hat == b.q_hat and a.psi == b.psi


def test_planner_unchanged_without_seed_nodes(world, fitted):
    wm, _, _ = fitted
    arm = _arm(world, wm)
    p = PN.Planner(
        world.J, (-1.0, 1.0), "cem", PN.PlannerConfig(max_actions=2, beam_width=2, inner_budget=12)
    )
    a = p.beam(arm, None, "lcb", None, seed=3)
    b = p.beam(arm, None, "lcb", None, seed=3, seed_nodes=None)
    assert [PN.candidate_key(c) for _, c in a] == [PN.candidate_key(c) for _, c in b]
    seeded = p.beam(arm, None, "lcb", None, seed=3, seed_nodes=[((0,), np.array([0.5]))])
    assert seeded  # seeding by incumbents is accepted


class _FakeWorld:
    """A dev world over the synthetic model: gain = the world's true primary change + noise; one guard."""

    def __init__(self, world, ep):
        self.w, self.ep = world, ep
        self.calls = []

    def evaluate(self, cand_ids):
        acts = [(int(p.split(":")[1][1:]) % self.w.J, 0.5) for p in cand_ids]
        t = self.w.trajectory_from_actions(acts) if acts else None
        y = t["y"][-1] if t is not None else np.zeros(self.w.K)
        self.calls.append(tuple(cand_ids))
        return {
            "gain": float(y[0]),
            "gain_se": 0.01,
            "guards_pass": bool(abs(y[1]) < 5),
            "rel_norm": 0.1 * len(acts),
            "theta_hash": "|".join(cand_ids),
            "seconds": 0.0,
        }


def test_run_episode_v1_schedule_budget_and_winner(world, fitted):
    _wm, _, _ = fitted
    ep = E.episodes("sycophancy_real", n_rep=1)[0]
    acts, doses = AD.episode_actions(ep, E.library_index("sycophancy_real"))
    assert len(acts) == len({E.library_index()[p]["action"] for p in ep["subsample"]})
    J = len(acts)
    rng = np.random.default_rng(0)
    gram = np.eye(J)
    d = AD.Dictionary(acts, gram, unit=[1.0] * J)

    class _M:  # a tiny deterministic predictor over the episode's actions
        n_members = 2

        def predict_batch(self, trajs):
            E_ = np.stack([t["x"][:, :J] for t in trajs])
            v = E_ @ rng.standard_normal((J, 3)) * 0 + E_.sum(-1, keepdims=True) * np.array(
                [1.0, 0.1, 0.0]
            )
            return {"mean": v, "sigma": np.full_like(v, 0.1), "members": np.stack([v, v + 0.01])}

    arm = AD.PredictorArm(
        _M(),
        None,
        d,
        acts,
        doses,
        primary=0,
        orient=1.0,
        q_hat=1.0,
        guards=[(1, 1.0, 10.0)],
        support=None,
        psi=0.0,
        mode="lcb",
    )
    led = PL.BudgetLedger(8, ["wm"])
    fw = _FakeWorld(world, ep)
    res = AD.run_episode_v1(
        fw,
        arm,
        led,
        "wm",
        ep,
        B=8,
        cfg=PN.PlannerConfig(max_actions=2, beam_width=2, inner_budget=8),
    )
    assert led.report()["units"]["wm"] == 8
    assert [len(r) for r in res["rounds"]] == [2, 2, 2, 2]
    for c in fw.calls:
        E.validate_candidate(c, ep)
    assert res["winner"]["candidate"] == () or res["winner"]["candidate"] in fw.calls
    led.verify_parity()
