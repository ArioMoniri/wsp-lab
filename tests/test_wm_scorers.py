"""wm.scorers: differentiable field scorers agree with the lean panel's per-item values (same definitions)."""

import pytest
import torch

from tests.test_wm_panel import ToyTok, toy_model
from wsp.features import response_field as rf
from wsp.wm import panel as P
from wsp.wm import scorers as Sc


@pytest.fixture(scope="module")
def model():
    return toy_model()


ITEMS = {
    "wikitext_nll": {
        "id": "w",
        "kind": "wikitext",
        "text": "the cat sat on the mat " * 3,
        "hash": "a",
        "group": "a",
    },
    "gsm8k_gold_lp": {
        "id": "g",
        "kind": "gsm8k",
        "question": "1+2?",
        "gold": "3",
        "hash": "b",
        "group": "b",
    },
    "mmlupro_gold_lp": {
        "id": "m",
        "kind": "mmlu_pro",
        "question": "Q?",
        "options": ["x", "y", "z"],
        "answer_index": 2,
        "hash": "c",
        "group": "c",
    },
    "sycophancy_gap": {
        "id": "s",
        "kind": "sycophancy",
        "question": "Q2?",
        "options": ["x", "y", "z"],
        "answer_index": 0,
        "wrong_index": 1,
        "hash": "d",
        "group": "d",
    },
    "refusal_margin_harmful": {
        "id": "r",
        "kind": "harmful",
        "prompt": "do bad",
        "hash": "e",
        "group": "e",
    },
    "refusal_margin_harmless": {
        "id": "h",
        "kind": "harmless",
        "prompt": "do good",
        "hash": "f",
        "group": "f",
    },
}
CFG = P.PanelConfig(
    window_tokens=32, refusal_phrases={"refusal": ["No", "Sorry"], "compliance": ["Yes", "Sure"]}
)


@pytest.mark.parametrize("obj", list(ITEMS))
def test_scorer_value_matches_panel(model, obj):
    tok = ToyTok()
    item = ITEMS[obj]
    want = P.evaluate(P.LMAdapter(model), tok, {obj: [item]}, CFG)[obj][0]["value"]
    sc = Sc.make_scorer(obj, CFG)
    with torch.no_grad():
        b = sc(model, tok, item, "cpu")
    assert isinstance(b, rf.ScoreBatch)
    value = b.score_sum.item() / b.weight
    if obj == "wikitext_nll":
        assert value == pytest.approx(want, rel=1e-4)
    else:
        assert value == pytest.approx(want, rel=1e-4, abs=1e-5)
    assert b.valid_mask.shape == b.logits.shape[:-1]
    assert b.valid_mask.any()


def test_scorer_is_differentiable(model):
    tok = ToyTok()
    sc = Sc.make_scorer("gsm8k_gold_lp", CFG)
    w = model.model.layers[0].mlp.down_proj.weight
    w.requires_grad_(True)
    try:
        b = sc(model, tok, ITEMS["gsm8k_gold_lp"], "cpu")
        b.score_sum.backward()
        assert w.grad is not None and w.grad.abs().sum() > 0
    finally:
        w.grad = None


def test_objective_specs_carry_panel_identity():
    p = Sc.field_panel("construction", [ITEMS["gsm8k_gold_lp"]], scored_groups=("zz",))
    specs = Sc.objective_specs(["gsm8k_gold_lp"], {"construction": p}, CFG)
    assert specs[0].name == "gsm8k_gold_lp" and specs[0].sign == "higher"
    assert specs[0].text_group_hash == p.text_group_hash
    with pytest.raises(ValueError):
        Sc.field_panel("construction", [ITEMS["gsm8k_gold_lp"]], scored_groups=("b",)).validate()
