"""wm.arms.pool: §6.5.3 pool mode shared by the reader arms and random; R0 candidate descriptions; the failure rule.

Synthetic worlds only (no model, no outcome)."""

import pytest

from wsp.wm import access as AC
from wsp.wm.arms import pool as P
from wsp.wm.arms import spec as S
from wsp.wm.plan import BudgetLedger

BASE = "b" * 64
SUB = list(range(10, 22))  # an episode subsample of 12 primitive ids


def _world():
    eff = {p: (0.02 if p % 3 == 0 else -0.01) for p in SUB}

    def evaluate(c):
        g = sum(eff[p] for p in c)
        return {
            "gain": g,
            "se": 0.005,
            "guards": [(0.001 * len(c), 0.001, 0.01)],
            "rel_norm": 1e-3 * len(c),
            "seconds": 2.0,
        }

    return evaluate


def _meta():
    fam = ("steer", "proj_remove")
    return {
        p: {
            "family": fam[p % 2],
            "layer": 20 + 2 * (p % 6),
            "contrast": ("syc", "gsm", "mathcomp", "trivia")[p % 4],
            "param": 0.0625 if p % 2 == 0 else -0.125,
            "rel_norm": 1e-3,
        }
        for p in SUB
    }


def _features(c):
    return {
        "sycophancy_real": {"a": 0.01 * len(c), "q": 1e-4},
        "wikitext_nll": {"a": -0.001, "q": None},
    }


def _ctx(arm, led, reg, episode="e0"):
    return S.ArmContext(
        S.ARM_SPECS[arm],
        led,
        episode,
        _world(),
        theta_of=lambda c: "t" + ".".join(map(str, c)),
        resources={
            "episode_library": SUB,
            "episode_pool": None,
            "r0_candidate_description": AC.Artefact("library_meta", BASE, AC.Tier.R0, _meta()),
        },
        stores=reg,
        guard=AC.R0Guard(BASE),
    )


def test_pool_is_seeded_well_formed_and_shared_in_round_one():
    a = P.draw_pool(SUB, 16, seed=7, incumbents=[])
    b = P.draw_pool(SUB, 16, seed=7, incumbents=[])
    assert a == b and len(a) == 16 and len(set(a)) == 16
    for c in a:
        assert 1 <= len(c) <= P.MAX_LEN and len(set(c)) == len(c) and set(c) <= set(SUB)
    assert P.draw_pool(SUB, 16, seed=8, incumbents=[]) != a


def test_pool_mutates_own_incumbents_and_excludes_evaluated():
    inc = [(10, 13)]
    pool = P.draw_pool(SUB, 16, seed=7, incumbents=inc, exclude={(10, 13)})
    assert (10, 13) not in pool
    muts = [c for c in pool if P.is_mutation(c, inc[0])]
    assert len(muts) >= 1
    assert len(muts) <= round(16 * P.MUTATION_FRACTION) + 1


def test_round_seed_is_shared_across_arms_and_distinct_per_round():
    assert P.round_seed(3, 0) == P.round_seed(3, 0)
    assert P.round_seed(3, 0) != P.round_seed(3, 1) != P.round_seed(4, 1)


def test_description_is_r0_and_marks_unsupported():
    v = P.describe((10, 13), "c00", _meta(), _features)
    assert v["id"] == "c00" and len(v["steps"]) == 2
    assert v["steps"][0]["family"] == "proj_remove" or v["steps"][0]["family"] == "steer"
    assert v["field"]["wikitext_nll"]["q"] is None
    txt = P.render_view(v)
    assert "unsupported" in txt and "c00" in txt
    assert "gain" not in txt  # no outcome in an R0 description


def test_random_arm_spends_exactly_B_with_parity():
    led, reg = BudgetLedger(8, ["random"]), S.StoreRegistry()
    out = P.run_pool_episode(
        P.RandomRanker(), _ctx("random", led, reg), B=8, pool_size=12, seed=1, features=_features
    )
    rep = led.verify_parity()
    assert rep["units"]["random"] == 8.0 and rep["forfeited"]["random"] == 0.0
    assert len(out["rounds"]) == 4 and all(r["n_evaluated"] == 2 for r in out["rounds"])
    assert out["winner"] is None or out["winner"]["guard_ok"]


class _Failing:
    name = "failing"

    def __init__(self, fail_round):
        self.fail_round = fail_round

    def rank(self, views, history, b, round_index):
        if round_index == self.fail_round:
            return P.RankResult(None, "truncated", [])
        return P.RankResult([v["id"] for v in views], None, [])


def test_failed_round_forfeits_its_bundles_without_fallback():
    led, reg = BudgetLedger(8, ["wsr_reader"]), S.StoreRegistry()
    out = P.run_pool_episode(
        _Failing(1), _ctx("wsr_reader", led, reg), B=8, pool_size=12, seed=1, features=_features
    )
    rep = led.verify_parity()
    assert rep["units"]["wsr_reader"] == 6.0 and rep["forfeited"]["wsr_reader"] == 2.0
    assert out["rounds"][1]["failure"] == "truncated" and out["rounds"][1]["n_evaluated"] == 0
    assert out["failures"] == 1


class _Bad:
    name = "bad"

    def rank(self, views, history, b, round_index):
        return P.RankResult(["c99", "c00"], None, [])  # unknown id


def test_invalid_ranking_is_a_failure():
    led, reg = BudgetLedger(4, ["wsr_reader"]), S.StoreRegistry()
    out = P.run_pool_episode(
        _Bad(), _ctx("wsr_reader", led, reg), B=4, pool_size=8, seed=1, features=_features
    )
    assert all(r["failure"] == "invalid_ranking" for r in out["rounds"])
    assert led.verify_parity()["forfeited"]["wsr_reader"] == 4.0


def test_history_given_to_ranker_is_own_only():
    seen = []

    class Spy:
        name = "spy"

        def rank(self, views, history, b, round_index):
            seen.append([h["candidate"] for h in history])
            return P.RankResult([v["id"] for v in views], None, [])

    reg = S.StoreRegistry()
    led = BudgetLedger(8, ["random", "scripted_reader"])
    P.run_pool_episode(
        P.RandomRanker(), _ctx("random", led, reg), B=8, pool_size=12, seed=1, features=_features
    )
    P.run_pool_episode(
        Spy(), _ctx("scripted_reader", led, reg), B=8, pool_size=12, seed=1, features=_features
    )
    assert seen[0] == []  # nothing from the random arm's store
    assert len(seen[1]) == 2


def test_B_must_split_into_four_rounds():
    led, reg = BudgetLedger(6, ["random"]), S.StoreRegistry()
    with pytest.raises(ValueError):
        P.run_pool_episode(
            P.RandomRanker(), _ctx("random", led, reg), B=6, pool_size=8, seed=1, features=_features
        )


def test_pool_arms_require_pool_mode_specs():
    led, reg = BudgetLedger(8, ["bon_steering"]), S.StoreRegistry()
    ctx = S.ArmContext(
        S.ARM_SPECS["bon_steering"],
        led,
        "e0",
        _world(),
        theta_of=str,
        resources={},
        stores=reg,
        guard=AC.R0Guard(BASE),
    )
    with pytest.raises(S.AccessDenied):
        P.run_pool_episode(P.RandomRanker(), ctx, B=8, pool_size=8, seed=1, features=_features)
