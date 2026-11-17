"""wm.arms.spec: the registered E2 arms, their declared information access and exposure (Board A condition 10;
PREREG_WME §6.8.3, §7), the per-(arm, episode) store, and the §6.5.5 winner rule. CPU only, no outcome."""

import pytest

from wsp.wm import access as AC
from wsp.wm.arms import spec as S
from wsp.wm.plan import BUNDLE_DEFINITION, BudgetExceeded, BudgetLedger

BASE = "b" * 64


def test_registry_covers_primary_and_nine_secondary_arms():
    prim = [a for a, s in S.ARM_SPECS.items() if s.role == "primary"]
    sec = [a for a, s in S.ARM_SPECS.items() if s.role == "secondary"]
    assert sorted(prim) == ["statnn", "wm_e"]
    assert len(sec) == 9  # the Holm family of 9 (§6.8.3)
    groups = {a: S.ARM_SPECS[a].group for a in sec}
    assert {a for a, g in groups.items() if g == "R"} == {
        "random",
        "field_ranked",
        "scripted_reader",
        "wsr_reader",
        "fugu_team",
        "wm_greedy",
    }
    assert {a for a, g in groups.items() if g == "D"} == {"direct_bon_two_stage"}
    assert {a for a, g in groups.items() if g == "C"} == {"bon_steering", "lora_sft"}


def test_every_arm_declares_access_exposure_and_shares_bundle_and_field():
    for name, s in S.ARM_SPECS.items():
        assert s.access, name
        assert set(s.access) <= set(S.ACCESS), name
        assert not set(s.access) & set(S.FORBIDDEN), name
        assert s.exposure in S.EXPOSURE, name
        assert "dev_evaluation" in s.access  # every arm is charged real evaluations on one ledger
        assert s.bundle_definition == BUNDLE_DEFINITION
        if "r0_candidate_description" in s.access or "field" in s.access:
            assert s.field_spec == S.SHARED_FIELD_SPEC


def test_condition10_disclosures():
    A = S.ARM_SPECS
    # readers are zero-shot: no bank, no predictor, no construction data
    for r in ("scripted_reader", "wsr_reader", "fugu_team"):
        assert A[r].exposure == "zero_shot"
        assert not {"bank", "wm_predictor", "statnn_predictor", "construction_e2"} & set(
            A[r].access
        )
        assert {"r0_candidate_description", "own_dev_results", "episode_library"} <= set(
            A[r].access
        )
    assert A["wsr_reader"].llm and A["fugu_team"].llm and not A["scripted_reader"].llm
    assert A["wsr_reader"].token_cap == A["fugu_team"].token_cap == "[SIM: reader_token_cap]"
    assert A["lora_sft"].exposure == "construction_e2_sft"
    assert "construction_e2" in A["lora_sft"].access and "train_gpu_s" in A["lora_sft"].charged
    assert A["bon_steering"].exposure == "none" and A["direct_bon_two_stage"].exposure == "none"
    assert A["wm_greedy"].exposure == "bank_trained" and "wm_predictor" in A["wm_greedy"].access
    assert A["wm_e"].uses_bank and A["statnn"].uses_bank and A["wm_greedy"].uses_bank
    assert not A["wsr_reader"].uses_bank
    # group R shares the episode subsample; group C has its own action space
    for a, s in A.items():
        if s.group in ("R", "D") or s.role == "primary":
            assert s.action_space == "episode_subsample", a
    assert A["bon_steering"].action_space == "steer_all_layers"
    assert A["lora_sft"].action_space == "lora_sft_construction_e2"


def test_access_table_is_serialisable():
    t = S.access_table()
    assert set(t) == set(S.ARM_SPECS)
    assert t["fugu_team"]["roles"] == {
        "thinker": ["episode_pool", "own_dev_results"],
        "worker": ["episode_pool", "thinker_output", "verifier_critique"],
        "verifier": ["episode_pool", "own_dev_results", "worker_ranking"],
    }


def _ctx(arm="wsr_reader", budget=4, episode="e0", reg=None, resources=None, calls=None):
    led = BudgetLedger(budget, [arm])
    calls = calls if calls is not None else []

    def evaluate(c):
        calls.append(c)
        return {"gain": 0.1, "se": 0.01, "guards": [(0.0, 0.001, 0.01)], "rel_norm": 1e-3}

    return (
        S.ArmContext(
            S.ARM_SPECS[arm],
            led,
            episode,
            evaluate,
            theta_of=lambda c: "t" + "-".join(map(str, c)),
            resources=resources or {},
            stores=reg or S.StoreRegistry(),
            guard=AC.R0Guard(BASE),
        ),
        led,
        calls,
    )


def test_context_refuses_undeclared_and_forbidden_resources():
    ctx, _, _ = _ctx(resources={"bank": object(), "episode_library": [1, 2]})
    assert ctx.get("episode_library") == [1, 2]
    with pytest.raises(S.AccessDenied):
        ctx.get("bank")  # wsr_reader is zero-shot
    with pytest.raises(S.AccessDenied):
        ctx.get("other_arm_results")
    with pytest.raises(S.AccessDenied):
        ctx.get("sealed_test")


def test_context_resources_must_be_r0():
    bad = AC.Artefact("tap", "c" * 64, AC.Tier.R2, None)
    ctx, _, _ = _ctx(resources={"r0_candidate_description": {"x": bad}})
    with pytest.raises(AC.AccessError):
        ctx.get("r0_candidate_description")
    good = AC.Artefact("desc", BASE, AC.Tier.R0, None)
    ctx2, _, _ = _ctx(resources={"r0_candidate_description": {"x": good}})
    assert ctx2.get("r0_candidate_description")["x"] is good


def test_evaluate_charges_shared_ledger_and_caches_repeats():
    ctx, led, calls = _ctx(budget=3)
    r1 = ctx.evaluate((1, 2))
    r2 = ctx.evaluate((1, 2))  # same theta: cached, 0 units, no second real evaluation
    assert calls == [(1, 2)]
    assert r1 is r2
    assert led.report()["units"]["wsr_reader"] == 1.0
    assert led.report()["by_kind"]["wsr_reader"] == {"evaluated": 1.0}
    ctx.evaluate((3,))
    ctx.evaluate((4,))
    with pytest.raises(BudgetExceeded):
        ctx.evaluate((5,))


def test_store_is_per_arm_and_episode():
    reg = S.StoreRegistry()
    a, _, _ = _ctx("wsr_reader", reg=reg, episode="e0")
    b, _, _ = _ctx("fugu_team", reg=reg, episode="e0")
    a.evaluate((1,))
    assert len(a.history()) == 1 and b.history() == []
    with pytest.raises(S.AccessDenied):
        reg.read("wsr_reader", "e0", requester="fugu_team")
    assert len(reg.read("wsr_reader", "e0", requester="wsr_reader")) == 1
    c, _, _ = _ctx("wsr_reader", reg=reg, episode="e1")
    assert c.history() == []


def test_history_requires_own_results_access():
    spec = S.ARM_SPECS["bon_steering"]
    assert (
        "own_dev_results" in spec.access
    )  # every arm sees its own results under the identical rule


def _rec(i, gain, se, guard_m=0.0, rel=1e-3):
    return {
        "candidate": (i,),
        "gain": gain,
        "se": se,
        "guards": [(guard_m, 0.001, 0.01)],
        "rel_norm": rel,
        "order": i,
    }


def test_winner_rule_lcb_ties_and_base_fallback():
    recs = [
        _rec(0, 0.10, 0.04),  # lcb 0.0216
        _rec(1, 0.10, 0.02, rel=2e-3),  # lcb 0.0608
        _rec(2, 0.10, 0.02, rel=1e-3),  # same lcb, smaller norm -> wins
        _rec(3, 0.50, 0.01, guard_m=0.05),  # guard breach
    ]
    assert S.select_winner(recs)["candidate"] == (2,)
    tie = [_rec(4, 0.1, 0.02), _rec(5, 0.1, 0.02)]
    assert S.select_winner(tie)["candidate"] == (4,)  # earlier evaluation
    assert S.select_winner([_rec(6, 0.01, 0.02)]) is None  # lcb <= 0: base
    assert S.select_winner([]) is None


def test_arm_seed_rule():
    s = S.arm_seed("lora_sft", 1234)
    assert s == S.arm_seed("lora_sft", 1234) and 0 <= s < 2**32
    assert s != S.arm_seed("bon_steering", 1234) != S.arm_seed("lora_sft", 1235)
    with pytest.raises(KeyError):
        S.arm_seed("nope", 1)
