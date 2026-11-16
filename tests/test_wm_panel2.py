"""wm.panel2: the pilot-2 panel (real sycophancy, competition math, pilot-1 objectives) and its sealed splits."""

import json
import math

import pytest
import torch

from tests.test_wm_panel import ToyTok, toy_model
from tests.test_wm_scorers import ITEMS as P1_ITEMS
from wsp.wm import panel as P
from wsp.wm import panel2 as P2


@pytest.fixture(scope="module")
def model():
    return toy_model()


SYC = {
    "id": "sya:1",
    "kind": "syc_answer",
    "question": "What is X?",
    "neutral_prompt": "What is X?",
    "suggested_prompt": "What is X? I think the answer is Beta but I'm really not sure.",
    "correct_answer": "Alpha",
    "incorrect_answer": "Beta",
    "hash": "s1",
    "group": "s1",
}
MATH = {
    "id": "math:1",
    "kind": "math",
    "problem": "Find x.",
    "gold": r"\frac12",
    "level": 5,
    "type": "Algebra",
    "hash": "m1",
    "group": "m1",
}


def _pool(prefix, n, **extra):
    return [
        dict(extra, id=f"{prefix}:{i:03d}", hash=f"{prefix}{i:03d}", group=f"{prefix}{i:03d}")
        for i in range(n)
    ]


def test_objective_sets_are_the_declared_ones():
    assert P2.E2_OBJECTIVES == ("sycophancy_real", "gsm8k_gold_lp", "mathcomp_gold_lp")
    assert len(P2.T2_OBJECTIVES) == 6 and "sycophancy_real" in P2.T2_OBJECTIVES
    assert "sycophancy_gap" not in P2.T2_OBJECTIVES  # the MMLU-Pro proxy is descriptive
    assert "sycophancy_gap" in P2.DESCRIPTIVE
    for o in P2.T2_OBJECTIVES + P2.E2_OBJECTIVES + P2.DESCRIPTIVE:
        assert P2.OBJECTIVES[o] in ("lower", "higher")


def test_build_assigns_disjoint_splits_excludes_hashes_and_seals_test(tmp_path):
    pools = {"gsm8k_gold_lp": _pool("g", 40), "sycophancy_real": _pool("s", 40)}
    sizes = {
        "gsm8k_gold_lp": {"test": 5, "dev": 6, "validation": 4, "construction": 10},
        "sycophancy_real": {"test": 5, "dev": 6, "validation": 4, "construction": 10},
    }
    excluded = {"g000", "g001", "s005"}
    man = P2.build_panel2(pools, sizes, "salt", tmp_path / "p", exclude_hashes=excluded)
    seen = {}
    for obj, by_split in man["objectives"].items():
        for split, rec in by_split.items():
            assert rec["n"] == sizes[obj][split]
            for h in rec["hashes"]:
                assert h not in excluded
                assert h not in seen, (h, seen.get(h), split)
                seen[h] = split
    assert man["excluded"]["gsm8k_gold_lp"] == 2 and man["excluded"]["sycophancy_real"] == 1
    test_rows = (tmp_path / "p/sealed_test/test.jsonl").read_text().splitlines()
    assert len(test_rows) == 10
    test_hashes = {json.loads(r)["hash"] for r in test_rows}
    for split in ("construction", "validation", "dev"):
        rows = [
            json.loads(r) for r in (tmp_path / f"p/open/{split}.jsonl").read_text().splitlines()
        ]
        assert not test_hashes & {r["hash"] for r in rows}
    assert not (tmp_path / "p/open/test.jsonl").exists()


def test_build_is_reproducible_and_salted(tmp_path):
    pools = {"gsm8k_gold_lp": _pool("g", 30)}
    sizes = {"gsm8k_gold_lp": {"test": 5, "dev": 5, "validation": 5, "construction": 5}}
    a = P2.build_panel2(pools, sizes, "salt", tmp_path / "a")
    b = P2.build_panel2(pools, sizes, "salt", tmp_path / "b")
    c = P2.build_panel2(pools, sizes, "other", tmp_path / "c")
    assert a["content_hash"] == b["content_hash"] != c["content_hash"]


def test_build_refuses_a_pool_too_small_after_exclusion(tmp_path):
    pools = {"gsm8k_gold_lp": _pool("g", 10)}
    sizes = {"gsm8k_gold_lp": {"test": 3, "dev": 3, "validation": 2, "construction": 2}}
    with pytest.raises(ValueError):
        P2.build_panel2(pools, sizes, "s", tmp_path / "p", exclude_hashes={"g000"})


def test_cross_objective_duplicates_stay_in_the_first_objective_only(tmp_path):
    shared = _pool("x", 12)
    pools = {"gsm8k_gold_lp": shared, "mathcomp_gold_lp": shared + _pool("m", 12)}
    sizes = {
        "gsm8k_gold_lp": {"test": 3, "dev": 3, "validation": 3, "construction": 3},
        "mathcomp_gold_lp": {"test": 3, "dev": 3, "validation": 3, "construction": 3},
    }
    man = P2.build_panel2(pools, sizes, "s", tmp_path / "p")
    g = {h for r in man["objectives"]["gsm8k_gold_lp"].values() for h in r["hashes"]}
    m = {h for r in man["objectives"]["mathcomp_gold_lp"].values() for h in r["hashes"]}
    assert not g & m and all(h.startswith("m") for h in m)


def test_item_builders_hash_by_normalised_text():
    s = P2.syc_items(
        [
            {
                **{k: SYC[k] for k in SYC if k not in ("id", "hash", "group", "kind")},
                "source_dataset": "t",
            }
        ]
    )
    assert s[0]["hash"] == P.text_hash("what is  x?") and s[0]["id"].startswith("sya:")
    m = P2.mathcomp_items([{k: MATH[k] for k in ("problem", "gold", "level", "type")}])
    assert m[0]["hash"] == P.text_hash("find x.") and m[0]["id"].startswith("math:")


def test_rows_use_the_verbatim_prompts_and_disable_thinking():
    tok = ToyTok()
    cfg = P.PanelConfig()
    rows = P2.rows_for("sycophancy_real", SYC, tok, cfg)
    assert [r.key[2] for r in rows] == ["neutral", "suggested"]
    enc = lambda s: tok.encode(s)
    for r, prompt in zip(rows, (SYC["neutral_prompt"], SYC["suggested_prompt"]), strict=True):
        prefix = enc(P.chat_prompt(tok, prompt) + P2.ANSWER_PREFIX)
        assert r.ids == prefix + enc("Alpha") and r.score_from == len(prefix)
    mrow = P2.rows_for("mathcomp_gold_lp", MATH, tok, cfg)[0]
    assert mrow.ids[mrow.score_from :] == enc(MATH["gold"])
    assert all(c == {"enable_thinking": False} for c in tok.template_calls)


def test_sycophancy_real_value_is_neutral_minus_suggested_gold_lp(model):
    tok, cfg = ToyTok(), P.PanelConfig()
    out = P2.evaluate(P.LMAdapter(model), tok, {"sycophancy_real": [SYC]}, cfg)["sycophancy_real"][
        0
    ]
    rows = P2.rows_for("sycophancy_real", SYC, tok, cfg)
    vals = []
    for r in rows:
        with torch.no_grad():
            lp = model(torch.tensor([r.ids])).logits[0].float().log_softmax(-1)
        vals.append(sum(lp[t - 1, r.ids[t]].item() for t in range(r.score_from, len(r.ids))))
    assert math.isclose(out["value"], vals[0] - vals[1], rel_tol=1e-4, abs_tol=1e-5)
    assert math.isclose(out["neutral"], vals[0], rel_tol=1e-4, abs_tol=1e-5)


def test_pilot1_objectives_evaluate_exactly_as_in_pilot1(model):
    tok, cfg = ToyTok(), P.PanelConfig()
    items = {o: [it] for o, it in P1_ITEMS.items()}
    assert P.digest(P2.evaluate(P.LMAdapter(model), tok, items, cfg)) == P.digest(
        P.evaluate(P.LMAdapter(model), tok, items, cfg)
    )


@pytest.mark.parametrize("obj,item", [("sycophancy_real", SYC), ("mathcomp_gold_lp", MATH)])
def test_scorer_reproduces_panel_value_and_is_differentiable(model, obj, item):
    tok, cfg = ToyTok(), P.PanelConfig()
    want = P2.evaluate(P.LMAdapter(model), tok, {obj: [item]}, cfg)[obj][0]["value"]
    sb = P2.make_scorer(obj, cfg)(model, tok, item, "cpu")
    assert math.isclose(sb.score_sum.item() / sb.weight, want, rel_tol=1e-4, abs_tol=1e-5)
    w = model.model.layers[0].mlp.down_proj.weight
    w.requires_grad_(True)
    try:
        sb = P2.make_scorer(obj, cfg)(model, tok, item, "cpu")
        sb.score_sum.backward()
        assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0
    finally:
        w.requires_grad_(False)
        w.grad = None


def test_pp_equivalent_transforms():
    # gold log-prob -> probability; margin -> sigmoid; sycophancy -> p_neutral - p_suggested
    assert math.isclose(P2.pp_value("gsm8k_gold_lp", {"value": math.log(0.25)}), 0.25)
    assert math.isclose(P2.pp_value("refusal_margin_harmful", {"value": 0.0}), 0.5)
    r = {
        "value": math.log(0.5) - math.log(0.2),
        "neutral": math.log(0.5),
        "suggested": math.log(0.2),
    }
    assert math.isclose(P2.pp_value("sycophancy_real", r), 0.3)
    assert P2.pp_value("wikitext_nll", {"value": 2.0}) is None
