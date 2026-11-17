"""wm.arms.direct: WM greedy one-step (isolates planning), best-of-N steering (all layers, direct evaluation) and the
LoRA-SFT arm (construction/e2 only, <= min(B, 32) checkpoint evaluations, training charged). CPU, synthetic."""

import numpy as np
import pytest

from wsp.wm import access as AC
from wsp.wm.arms import direct as D
from wsp.wm.arms import spec as S
from wsp.wm.plan import BudgetLedger
from wsp.wm.planner import Planner, PlannerConfig

BASE = "b" * 64
SUB = list(range(8))
W = np.array([0.05, -0.02, 0.03, 0.01, -0.04, 0.02, 0.0, 0.04])


class FakePred:
    """Planner predictor protocol: value = Σ w[a] dose, sigma 0.01, one guard 0.001 per action."""

    n_members = 1

    def embed(self, actions, doses):
        E = np.zeros((len(np.atleast_2d(doses)), len(SUB)))
        for j, a in enumerate(actions):
            E[:, a] = np.atleast_2d(doses)[:, j]
        return E

    def draw(self, rng):
        return None

    def predict(self, actions, doses, draw):
        d = np.atleast_2d(doses)
        v = d @ W[list(actions)]
        return {
            "value": v,
            "sigma": np.full(len(d), 0.01),
            "guards": [(0.001 * d.sum(1), np.zeros(len(d)), 0.01)],
        }


def _world(c):
    c = tuple(c)
    return {
        "gain": float(W[list(c)].sum()),
        "se": 0.005,
        "guards": [(0.0, 0.001, 0.01)],
        "rel_norm": 1e-3 * len(c),
    }


def _ctx(arm, budget, resources, evaluate=_world, theta_of=repr, reg=None):
    led = BudgetLedger(budget, [arm])
    ctx = S.ArmContext(
        S.ARM_SPECS[arm],
        led,
        "e0",
        evaluate,
        theta_of=theta_of,
        resources=resources,
        stores=reg or S.StoreRegistry(),
        guard=AC.R0Guard(BASE),
    )
    return ctx, led


# ------------------------------------------------------------------ WM greedy one-step


def test_greedy_round_one_takes_top_singletons_by_shared_scorer():
    ctx, led = _ctx(
        "wm_greedy",
        8,
        {
            "episode_library": SUB,
            "wm_predictor": FakePred(),
            "field": None,
            "support_penalty": None,
        },
    )
    out = D.run_greedy_episode(ctx, B=8, cfg=PlannerConfig(support_lambda=0.0))
    r0 = out["rounds"][0]["evaluated"]
    assert r0 == [(0,), (7,)]  # w = 0.05, 0.04
    # round 2 extends the best feasible incumbent (0,) by one primitive (or takes the next singleton)
    r1 = out["rounds"][1]["evaluated"]
    assert all(len(c) <= 2 for c in r1) and any(c[:1] == (0,) and len(c) == 2 for c in r1)
    assert led.verify_parity()["units"]["wm_greedy"] == 8.0
    for rnd in out["rounds"]:
        for c in rnd["evaluated"]:
            assert len(c) <= 4 and len(set(c)) == len(c)


def test_greedy_scores_equal_the_planner_scorer():
    cfg = PlannerConfig(support_lambda=0.0)
    cands = [(0,), (3, 5), (1, 2, 4)]
    ours = D.greedy_scores(FakePred(), cands, "lcb", None, cfg)
    pl = Planner(len(SUB), (-1.0, 1.0), "cem", cfg).score(
        FakePred(), [[(a, 1.0) for a in c] for c in cands], None, "lcb", None
    )
    np.testing.assert_allclose(ours, pl)
    ranks = D.greedy_scores(FakePred(), cands, "ranks", None, cfg)
    assert not np.allclose(ranks, ours)  # sigma dropped symmetrically in ranks mode


def test_greedy_requires_wm_predictor_access():
    ctx, _ = _ctx("wsr_reader", 4, {"episode_library": SUB, "wm_predictor": FakePred()})
    with pytest.raises(S.AccessDenied):
        D.run_greedy_episode(ctx, B=4)


# ------------------------------------------------------------------ best-of-N steering


def test_steering_space_covers_all_layers():
    sp = D.steering_space(32)
    assert len(sp) == 32 * len(D.STEER_CONTRASTS) * len(D.STEER_SHIFTS) == 512
    assert {a[1] for a in sp} == set(range(32))
    assert all(a[0] == "steer" for a in sp)


def test_bon_steering_evaluates_B_distinct_single_actions():
    seen = []

    def ev(c):
        seen.append(c)
        return {
            "gain": 0.01 if c[2] == "syc" else -0.01,
            "se": 0.004,
            "guards": [(0.0, 0.001, 0.01)],
            "rel_norm": 1e-3,
        }

    ctx, led = _ctx("bon_steering", 16, {"steer_contrasts_all_layers": "dirs"}, evaluate=ev)
    out = D.run_bon_steering(ctx, B=16, seed=5, n_layers=32)
    assert len(seen) == 16 == len(set(seen))
    assert led.verify_parity()["units"]["bon_steering"] == 16.0
    assert out["winner"]["candidate"][2] == "syc"
    ctx2, _ = _ctx("bon_steering", 16, {"steer_contrasts_all_layers": "dirs"}, evaluate=ev)
    assert D.run_bon_steering(ctx2, B=16, seed=5, n_layers=32)["evaluated"] == out["evaluated"]


# ------------------------------------------------------------------ LoRA SFT


def test_checkpoint_schedule():
    assert D.checkpoint_steps(100, 4) == [25, 50, 75, 100]
    assert D.checkpoint_steps(10, 32) == list(range(1, 11))  # never more checkpoints than steps
    assert D.n_checkpoints(48) == 32 and D.n_checkpoints(16) == 16
    with pytest.raises(ValueError):
        D.checkpoint_steps(0, 4)


def test_shard_guard():
    items = [{"hash": "h1"}, {"hash": "h2"}]
    D.check_shard(items, shard_hashes={"h1", "h2", "h3"}, forbidden_hashes={"d1"})
    with pytest.raises(D.SplitError):
        D.check_shard(items + [{"hash": "x"}], {"h1", "h2"}, set())
    with pytest.raises(D.SplitError):
        D.check_shard(
            items, {"h1", "h2"}, forbidden_hashes={"h2"}
        )  # a dev/test hash inside the shard


def _sft_resources(items):
    return {
        "construction_e2": {
            "items": items,
            "shard_hashes": {i["hash"] for i in items},
            "forbidden_hashes": {"dev1"},
        }
    }


def test_lora_sft_arm_charges_checkpoints_and_training():
    items = [{"hash": f"h{i}"} for i in range(40)]
    calls = {}

    def train_fn(its, steps_at, seed):
        calls["steps_at"] = steps_at
        calls["n"] = len(its)
        return {
            "checkpoints": {s: ("lora", seed, s) for s in steps_at},
            "gpu_seconds": 123.0,
            "total_steps": steps_at[-1],
        }

    def ev(c):
        return {
            "gain": 0.002 * c[2],
            "se": 0.01,
            "guards": [(0.0, 0.001, 0.01)],
            "rel_norm": 1e-3 * c[2],
            "seconds": 2.0,
        }

    ctx, led = _ctx("lora_sft", 16, _sft_resources(items), evaluate=ev)
    out = D.run_lora_sft(ctx, train_fn, B=16, seed=11, total_steps=80)
    assert calls["steps_at"] == D.checkpoint_steps(80, 16) and calls["n"] == 40
    rep = led.verify_parity()
    assert rep["by_kind"]["lora_sft"] == {"checkpoint": 16.0}
    assert out["costs"]["train_gpu_s"] == 123.0 and out["costs"]["eval_gpu_s"] == 32.0
    assert out["winner"]["candidate"] == ("lora", 11, 80)  # largest LCB


def test_lora_sft_refuses_more_than_32_checkpoints_and_non_shard_items():
    items = [{"hash": "h0"}]
    ctx, _ = _ctx("lora_sft", 48, _sft_resources(items))
    with pytest.raises(ValueError):
        D.run_lora_sft(ctx, lambda *a: None, B=48, seed=1, total_steps=100, n_ckpt=33)
    bad = {
        "construction_e2": {
            "items": [{"hash": "dev1"}],
            "shard_hashes": {"dev1"},
            "forbidden_hashes": {"dev1"},
        }
    }
    ctx2, _ = _ctx("lora_sft", 8, bad)
    with pytest.raises(D.SplitError):
        D.run_lora_sft(ctx2, lambda *a: None, B=8, seed=1, total_steps=10)


def test_train_with_checkpoints_snapshots_factors_and_keeps_base_weights():
    import torch

    from tests.test_wm_lora import _gsm, _rolemap
    from tests.test_wm_panel import ToyTok, toy_model
    from wsp.wm import lora as LO
    from wsp.wm import panel as PN

    model = toy_model()
    items = [_gsm(i) for i in range(6)]
    man = {
        "objectives": {"gsm8k_gold_lp": {"construction": {"hashes": [i["hash"] for i in items]}}}
    }
    rows = LO.sft_rows("gsm8k_gold_lp", items, ToyTok(), PN.PanelConfig(), man)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ad = LO.LoRA(model, LO.writer_targets(_rolemap(model)), rank=8, alpha=16, seed=3)
    out = D.train_with_checkpoints(
        model, ad, rows, 0, steps_at=[2, 5, 9], lr=5e-3, batch_rows=3, seed=3, device="cpu"
    )
    assert out["steps"] == 9 and sorted(out["checkpoints"]) == [2, 5, 9]
    b2 = next(iter(out["checkpoints"][2].values()))[0]
    b9 = next(iter(out["checkpoints"][9].values()))[0]
    assert not torch.equal(b2, b9) and b2.abs().sum() > 0  # B starts at 0 and moves
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n
    acts = LO.lowrank_actions(out["checkpoints"][9], 1.0)
    assert len(acts) == len(ad.names)
