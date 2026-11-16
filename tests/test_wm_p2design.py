"""wm.p2design: pilot-2 ladders, stop rule, compositions, negative-control exclusion and gates (RULING-wme-pilot1)."""

import math

import numpy as np
import pytest

from wsp.wm import p2design as G

# ------------------------------------------------------------------ ladders and stop rule


def test_ladder_multipliers_and_nominals_are_the_declared_ones():
    assert G.LADDER == (0.25, 0.5, 1.0, 2.0, 4.0)
    assert G.NOMINAL == {"scale": 1.5, "steer": 0.125, "proj_remove": 0.5}
    assert G.DNLL_STOP == 1.0


def test_proj_remove_alpha_is_capped_at_one_and_a_capped_duplicate_is_not_visited():
    assert G.effective_param("proj_remove", 2.0) == (1.0, False)
    assert G.effective_param("proj_remove", 4.0) == (1.0, True)
    assert G.effective_param("proj_remove", -4.0) == (-2.0, False)  # the cap is alpha <= 1 only
    pos, neg = G.ladder_sides("proj_remove")
    assert pos == [0.25, 0.5, 1.0, 2.0]
    assert neg == [-0.25, -0.5, -1.0, -2.0, -4.0]
    for fam in ("scale", "steer"):
        pos, neg = G.ladder_sides(fam)
        assert pos == list(G.LADDER) and neg == [-d for d in G.LADDER]


def test_scale_and_steer_parameters_follow_the_nominals():
    assert G.effective_param("scale", 2.0) == (1.5, False)  # factor; W(s) = [1 + s(f - 1)] W
    assert G.effective_param("steer", -4.0) == (-0.5, False)  # signed residual-norm fraction


def test_walk_stops_after_the_first_state_above_one_nat_and_keeps_it_as_labelled_endpoint():
    dnll = {0.25: 0.01, 0.5: 0.2, 1.0: 1.3, 2.0: 5.0, 4.0: 9.0}
    calls = []

    def ev(d):
        calls.append(d)
        return dnll[d]

    visited = G.walk_ladder([0.25, 0.5, 1.0, 2.0, 4.0], ev)
    assert calls == [0.25, 0.5, 1.0]
    assert [v["dose"] for v in visited] == [0.25, 0.5, 1.0]
    assert visited[-1]["endpoint"] is True and visited[-1]["label"] == "dnll>1.0"
    assert not any(v["endpoint"] for v in visited[:-1])


def test_walk_at_exactly_one_nat_continues():
    visited = G.walk_ladder([0.25, 0.5], lambda d: 1.0)
    assert len(visited) == 2 and not any(v["endpoint"] for v in visited)


# ------------------------------------------------------------------ roots and compositions

FAMILIES = {
    "scale_a": "scale",
    "scale_b": "scale",
    "steer_a": "steer",
    "steer_b": "steer",
    "projrm_a": "proj_remove",
    "projrm_b": "proj_remove",
    "projrm_random": "control",
}


def test_compositions_counts_reverses_families_and_no_control():
    c3, c4 = G.make_compositions(FAMILIES, seed=G.seed_for("compositions"))
    assert len(c3) == 16 and len(c4) == 8
    for seqs, k in ((c3, 3), (c4, 4)):
        assert len(set(seqs)) == len(seqs)
        for s in seqs:
            assert len(s) == k and len(set(s)) == k
            assert tuple(reversed(s)) in seqs
            fams = {FAMILIES[r] for r in s}
            assert fams == {"scale", "steer", "proj_remove"}
    assert G.make_compositions(FAMILIES, seed=G.seed_for("compositions")) == (c3, c4)


def test_composition_states_are_prefixes_from_h2():
    states = G.composition_states([("a", "b", "c"), ("c", "b", "a")])
    ids = [s["id"] for s in states]
    assert ids == ["comp:a>b", "comp:a>b>c", "comp:c>b", "comp:c>b>a"]
    assert states[1]["h"] == 3 and states[1]["steps"] == [("a", 1.0), ("b", 1.0), ("c", 1.0)]


# ------------------------------------------------------------------ steps and control exclusion


def _visited():
    return {
        "scale_a": {"pos": [0.25, 0.5, 1.0], "neg": [-0.25]},
        "projrm_random": {"pos": [0.25, 0.5], "neg": [-0.25]},
    }


def test_steps_chain_visited_ladder_doses_from_base_and_compositions_from_their_prefix():
    steps = G.steps_list(
        _visited(), ["comp:scale_a>steer_a", "comp:scale_a>steer_a>projrm_a"], FAMILIES
    )
    pairs = [(s["from"], s["to"]) for s in steps]
    assert ("base", "ladder:scale_a:+0.25") in pairs
    assert ("ladder:scale_a:+0.5", "ladder:scale_a:+1") in pairs
    assert ("base", "ladder:scale_a:-0.25") in pairs
    assert ("ladder:scale_a:+1", "comp:scale_a>steer_a") in pairs
    assert ("comp:scale_a>steer_a", "comp:scale_a>steer_a>projrm_a") in pairs
    ctl = [s for s in steps if s["control"]]
    assert {s["to"] for s in ctl} == {
        "ladder:projrm_random:+0.25",
        "ladder:projrm_random:+0.5",
        "ladder:projrm_random:-0.25",
    }


def test_negative_control_steps_are_excluded_from_the_resolvability_count():
    rows = []
    for i in range(4):
        rows.append({"control": False, "objective": "o", "mean": 1.0, "se": 0.1})  # resolvable
    for i in range(4):
        rows.append({"control": False, "objective": "o", "mean": 0.0, "se": 0.1})  # not
    for i in range(10):
        rows.append({"control": True, "objective": "o", "mean": 0.0, "se": 0.1})  # control, not
    r = G.resolvability(rows)
    assert r["noncontrol"] == 0.5 and r["n_noncontrol"] == 8
    assert r["control"] == 0.0 and r["n_control"] == 10
    assert math.isclose(r["pooled_with_control_descriptive"], 4 / 18)


def test_resolvable_is_abs_mean_at_least_two_se():
    assert G.is_resolvable(0.2, 0.1) and G.is_resolvable(-0.2, 0.1)
    assert not G.is_resolvable(0.19, 0.1)


# ------------------------------------------------------------------ Go-T2 static endpoint


def test_static_endpoint_is_pooled_median_over_h3plus_states_and_six_objectives_with_refit_f3u():
    objs = ("o1", "o2")
    ladder = []
    rng = np.random.default_rng(0)
    for k in range(6):
        a, q = rng.normal(), abs(rng.normal())
        for o in objs:
            ladder.append({"objective": o, "a": a, "q": q, "observed": 2 * a + 3 * q})
    comps = []
    for h, err in ((2, 100.0), (3, 1.0), (3, 5.0), (4, 3.0)):
        for o in objs:
            comps.append(
                {"objective": o, "h": h, "a": 1.0, "q": 1.0, "observed": 5.0 + err, "se": 1.0}
            )
    res = G.static_endpoint(ladder, comps, objs)
    assert np.allclose(res["coef"]["o1"], [2.0, 3.0])
    # h >= 3 errors are 1, 5, 3 for each objective; pooled median 3; h = 2 ignored
    assert math.isclose(res["median_err_over_se_h3plus"], 3.0)
    assert res["n_pairs"] == 6
    assert G.go_t2(0.5, 3.0)["pass"] and not G.go_t2(0.49, 3.0)["pass"]
    assert not G.go_t2(0.6, 2.0)["pass"]


# ------------------------------------------------------------------ Go-R2


def test_go_r2_uses_trapezoid_against_static_step_on_steps_h2_plus():
    rows = [
        {"h": 1, "observed": 0.0, "se": 1.0, "static": 10.0, "trap": 0.0},  # ignored (h = 1)
        {"h": 2, "observed": 1.0, "se": 1.0, "static": 2.0, "trap": 1.5},
        {"h": 3, "observed": 1.0, "se": 0.5, "static": 0.0, "trap": 0.5},
    ]
    r = G.go_r2(rows)
    # static errors/SE: 1, 2 -> 1.5; trapezoid: 0.5, 1 -> 0.75; cut 0.5
    assert math.isclose(r["cut"], 0.5) and r["pass"] and r["n"] == 2
    rows[1]["trap"] = 1.95
    rows[2]["trap"] = 0.05
    assert not G.go_r2(rows)["pass"]


# ------------------------------------------------------------------ Go-E2


def _cand(gain_mean, gain_se, guards):
    return {"objective_stats": {"mean": gain_mean, "se": gain_se}, "guards": guards}


def test_guardrails_are_wikitext_ucb_and_minus_one_pp_on_the_others():
    g = G.guard_breaches(
        {
            "wikitext_nll": {"mean": 0.005, "se": 0.002},  # UCB 0.0089 ok
            "gsm8k_gold_lp": {"mean": -0.005, "se": 0.002},  # pp LCB -0.0089 ok
            "refusal_margin_harmful": {"mean": -0.01, "se": 0.001},  # pp LCB -0.012 breach
            "refusal_margin_harmless": {"mean": 0.009, "se": 0.001},  # pp UCB 0.011 breach
            "sycophancy_real": {"mean": 0.0, "se": 0.001},
        },
        objective="mathcomp_gold_lp",
    )
    assert set(g) == {"refusal_margin_harmful", "refusal_margin_harmless"}
    assert G.guard_breaches({"wikitext_nll": {"mean": 0.009, "se": 0.001}}, "gsm8k_gold_lp") == [
        "wikitext_nll"
    ]


def test_the_objective_itself_and_the_descriptive_proxy_are_not_guardrails():
    stats = {
        "gsm8k_gold_lp": {"mean": -1.0, "se": 0.001},
        "sycophancy_gap": {"mean": 5.0, "se": 0.001},
    }
    assert G.guard_breaches(stats, "gsm8k_gold_lp") == []
    with pytest.raises(KeyError):
        G.guard_breaches({"unknown": {"mean": 0, "se": 1}}, "gsm8k_gold_lp")


def test_gain_is_oriented_and_qualifies_on_lcb_above_zero_without_breach():
    assert G.gain_lcb("sycophancy_real", -0.1, 0.02) == pytest.approx(0.1 - 1.96 * 0.02)
    assert G.gain_lcb("gsm8k_gold_lp", 0.1, 0.02) == pytest.approx(0.1 - 1.96 * 0.02)
    assert G.qualifies("gsm8k_gold_lp", 0.1, 0.02, [])
    assert not G.qualifies("gsm8k_gold_lp", 0.1, 0.02, ["wikitext_nll"])
    assert not G.qualifies("gsm8k_gold_lp", 0.03, 0.02, [])


def test_go_e2_needs_two_of_32_and_selects_most_qualifying_with_tie_order_iii_ii_i():
    counts = {"sycophancy_real": 3, "gsm8k_gold_lp": 3, "mathcomp_gold_lp": 1}
    r = G.go_e2(counts, n_candidates={k: 32 for k in counts})
    assert (
        r["pass"]["sycophancy_real"]
        and r["pass"]["gsm8k_gold_lp"]
        and not r["pass"]["mathcomp_gold_lp"]
    )
    assert r["primary"] == "gsm8k_gold_lp"  # tie 3-3 -> (ii) beats (i)
    counts["mathcomp_gold_lp"] = 3
    assert G.go_e2(counts, {k: 32 for k in counts})["primary"] == "mathcomp_gold_lp"
    none = G.go_e2({k: 1 for k in counts}, {k: 32 for k in counts})
    assert none["primary"] is None and not any(none["pass"].values())
    with pytest.raises(ValueError):
        G.go_e2(counts, {k: 31 for k in counts})


def test_e2_candidates_are_three_lora_scales_plus_29_steering_sampled_without_replacement():
    prims = [
        (L, d, s)
        for L in (20, 24, 28, 31)
        for d in ("x", "y")
        for s in (-0.0625, -0.03125, 0.03125, 0.0625)
    ]
    c = G.e2_candidates("gsm8k_gold_lp", prims, seed=G.seed_for("e2:gsm8k_gold_lp"))
    assert len(c) == 32
    lora = [x for x in c if x["kind"] == "lora"]
    steer = [x for x in c if x["kind"] == "steer"]
    assert [x["lam"] for x in lora] == [0.25, 0.5, 1.0] and all(
        x["objective"] == "gsm8k_gold_lp" for x in lora
    )
    assert len(steer) == 29 and len({tuple(x["primitive"]) for x in steer}) == 29
    for x in steer:
        L, _d, s = x["primitive"]
        assert L >= 20 and abs(s) <= 0.0625
    with pytest.raises(ValueError):
        G.e2_candidates("gsm8k_gold_lp", prims[:10], seed=1)
    with pytest.raises(ValueError):
        G.e2_candidates("gsm8k_gold_lp", [(12, "x", 0.03)] * 40, seed=1)


# ------------------------------------------------------------------ consequences (ruling section 4)


def test_consequence_rows():
    assert G.consequence(True, True, None)["row"] == "T2 passes and E2 passes"
    assert G.consequence(False, True, None)["row"] == "T2 fails, E2 passes"
    assert G.consequence(True, False, False)["row"] == "E2 fails on both 9B and 4B"
    assert G.consequence(False, False, False)["row"] == "E2 fails on both 9B and 4B"
    assert (
        G.consequence(True, False, True)["row"] is None
    )  # not covered by the table: back to the chair
    with pytest.raises(ValueError):
        G.consequence(True, False, None)  # 9B failed: the 4B E2 must have been run
