"""wm.p3gen: pilot-3 stage H verifiers, IFEval-type constraints, guard statistics, sizing and gates."""

import numpy as np
import pytest

from wsp.wm import p3gen as G

MC = {"question": "Q?", "options": ["a", "b", "c", "d"], "answer_index": 2}


def test_mc_prompt_and_verify():
    p = G.user_prompt("mmlupro_gen", MC)
    assert "(C) c" in p and "The answer is (X)" in p
    assert G.verify("mmlupro_gen", MC, "so... The answer is (C).", False)["correct"]
    assert not G.verify("mmlupro_gen", MC, "the answer is (B) wait the answer is (D)", False)[
        "correct"
    ]
    assert G.verify("mmlupro_gen", MC, "the answer is (B). Hmm, the answer is C", False)["correct"]
    assert not G.verify("mmlupro_gen", MC, "The answer is (C)", True)["correct"]  # truncated
    assert not G.verify("mmlupro_gen", MC, "no letter here", False)["correct"]


def test_thinking_answers_are_read_after_think():
    t = "<think>maybe the answer is (A)</think>\nThe answer is (C)"
    assert G.verify("mmlupro_gen", MC, t, False, thinking=True)["correct"]
    assert not G.verify("mmlupro_gen", MC, "<think>the answer is (C)", False, thinking=True)[
        "correct"
    ]


def test_boxed_and_math_equality():
    assert G.last_boxed(r"x \boxed{\frac{1}{2}} and \boxed{3}") == "3"
    assert G.last_boxed(r"\boxed{\frac{a}{b}}") == r"\frac{a}{b}"
    assert G.math_equal(r"\dfrac{1}{2}", r"\frac{1}{2}")
    assert G.math_equal("0.5", r"\frac12")
    assert G.math_equal(r"x = 5", "5")
    assert G.math_equal(r"90^\circ", "90")
    assert not G.math_equal("3", "4")
    assert not G.math_equal(None, "4")
    gsm = {"question": "q", "gold": "1200"}
    assert G.verify("gsm8k_gen", gsm, r"so \boxed{1,200}", False)["correct"]
    assert G.verify("gsm8k_gen", gsm, r"\boxed{\$1200.00}", False)["correct"]
    assert not G.verify("gsm8k_gen", gsm, r"\boxed{120}", False)["correct"]


def test_ifeval_constraints_are_seeded_compatible_and_checkable():
    for i in range(300):
        cs = G.make_constraints(f"h{i}")
        assert cs == G.make_constraints(f"h{i}")
        assert 1 <= len(cs) <= 2
        if len(cs) == 2:
            assert frozenset({cs[0]["type"], cs[1]["type"]}) not in G._CONFLICT
    c = {"type": "punctuation:no_comma"}
    assert G.check_constraint(c, "a b c") and not G.check_constraint(c, "a, b")
    c = {"type": "detectable_format:number_bullet_lists", "n": 3}
    assert G.check_constraint(c, "* a\n* b\n* c") and not G.check_constraint(c, "* a\n* b")
    c = {"type": "detectable_format:json_format"}
    assert G.check_constraint(c, '```json\n{"a": 1}\n```') and not G.check_constraint(c, "{a: 1")
    c = {"type": "length_constraints:number_paragraphs", "n": 2}
    assert G.check_constraint(c, "one\n***\ntwo") and not G.check_constraint(c, "one two")
    c = {"type": "startend:end_checker", "end": "That is all."}
    assert G.check_constraint(c, "x. That is all.") and not G.check_constraint(c, "That is all. x")
    c = {"type": "keywords:forbidden_words", "keywords": ["very"]}
    assert G.check_constraint(c, "every") and not G.check_constraint(c, "Very good")
    item = {
        "prompt": "Write.",
        "constraints": [{"type": "change_case:english_lowercase", "text": "t"}],
    }
    assert G.verify("ifeval_gen", item, "all lower", False)["correct"]
    assert not G.verify("ifeval_gen", item, "Not lower", False)["correct"]


def test_budget_rule():
    assert G.budget([100] * 100) == 256
    assert G.budget(list(range(1, 1001))) == 960  # p95 = 950.05 -> 960
    assert G.budget([5000] * 10) == 2048


def test_paired_stats_and_lcb():
    b = [0, 0, 1, 1] * 50
    c = [1, 0, 1, 1] * 50
    s = G.paired(b, c)
    assert s["mean"] == pytest.approx(0.25) and s["discordance"] == pytest.approx(0.25)
    assert s["lcb"] == pytest.approx(s["mean"] - 1.96 * s["se"])
    with pytest.raises(ValueError):
        G.paired([1], [1, 0])


def test_bootstrap_t_ucb_covers_and_is_exact_for_constant():
    assert G.bootstrap_t_ucb([0.0] * 50) == 0.0
    rng = np.random.default_rng(1)
    miss = 0
    for s in range(200):
        u = rng.random(300)
        d = np.where(u < 0.05, 1.0, np.where(u < 0.1, -1.0, 0.0))
        miss += G.bootstrap_t_ucb(d, n_boot=400, seed=s) < 0
    assert miss / 200 < 0.12  # nominal 0.05


def test_guard_margin_sizing_and_null_pass():
    pi = np.r_[np.zeros(100), np.full(28, 0.5)]
    d = G.guard_margin(pi)
    assert 0.01 <= d < 0.2
    assert G.guard_margin(np.zeros(128)) == 0.01  # deterministic base -> floor
    n = G.guard_n(0.05, d, pool=5000)
    assert n >= 200
    assert G.guard_n(0.05, 0.01, pool=400) == 400  # pool cap binds
    p = G.null_pass_prob(0.05, d, n, n_sim=200, n_boot=200)
    assert p >= 0.7
    assert G.primary_n(0.1, 1000) == 872
    assert G.primary_n(0.01, 1000) == 300


def test_guard_test_directions():
    base = [0] * 100
    worse = [1] * 10 + [0] * 90  # +10 pp over-refusal
    hx = [1] * 100
    g = G.guard_test(base, worse, 0.02, hx, hx, 0.02)
    assert not g["pass_harmless"] and g["pass_harmful"] and not g["guarded"]
    g = G.guard_test(base, base, 0.02, hx, [0] * 10 + [1] * 90, 0.02)
    assert g["pass_harmless"] and not g["pass_harmful"]


def test_family_gate_and_matched_beat():
    good = {"lcb": 0.01}
    bad = {"lcb": -0.01}
    rows = [{"key": i, "guarded": True, "primary": {"mmlupro_gen": good}} for i in range(2)]
    rows += [{"key": 9, "guarded": False, "primary": {"supergpqa_gen": good, "mmlupro_gen": bad}}]
    g = G.family_gate(rows)
    assert g["counts"] == {"mmlupro_gen": 2, "supergpqa_gen": 0} and g["pass"]
    assert not G.family_gate(rows[:1])["pass"]
    t = [[1] * 60 + [0] * 40] * 8
    c = [[0] * 60 + [0] * 40] * 8
    assert G.matched_beat(t, c)["beats"]
    assert not G.matched_beat(c, t)["beats"]


def test_proxy_gate():
    x = np.arange(40.0)
    assert not G.proxy_gate(x, x + np.random.default_rng(0).normal(0, 1, 40))["killed"]
    r = G.proxy_gate(x, np.random.default_rng(0).normal(0, 1, 40))
    assert r["lb"] <= 0 and r["killed"]
    assert not G.proxy_gate(x[:10], x[:10] * 0)["killed"]  # fewer than 32 candidates: no verdict


def test_flexible_mc_extractor():
    f = G.flexible_mc_letter
    assert f("blah The answer is (C).") == "C"
    assert f("so the answer: **D**") == "D"
    assert f("final \\boxed{B}") == "B"
    assert f("The correct option is E because") == "E"
    assert f("I pick **(F)** here") == "F"
    assert f("options (A) or (G) ... I go with (G)") == "G"
    assert f("the answer is (A) ... wait, answer: B") == "B"
    assert f("no letter at all") is None
