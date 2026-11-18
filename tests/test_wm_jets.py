"""wm.jets: S2 nested forward-mode jets and S1 forward-over-reverse HVP at θ₀ vs autograd on a tiny Qwen3.5.

The reference GatedDeltaNet kernels compute in fp32 internally, so agreement is to ~1e-6 relative even in fp64.
"""

import pytest
import torch
from test_wm_panel import ToyTok, toy_model

from wsp.wm import jets as J
from wsp.wm import panel as P
from wsp.wm import panel2 as P2

ITEMS = {
    "wikitext_nll": [{"id": "w1", "text": "Some article text about things " * 3}],
    "gsm8k_gold_lp": [
        {"id": "g1", "question": "What is 2+3?", "gold": "5"},
        {"id": "g2", "question": "1+1?", "gold": "2"},
    ],
    "mmlupro_gold_lp": [
        {"id": "m1", "question": "Q?", "options": ["a", "b", "c"], "answer_index": 1}
    ],
    "sycophancy_gap": [
        {
            "id": "s1",
            "question": "Q2?",
            "options": ["x", "y", "z"],
            "answer_index": 0,
            "wrong_index": 2,
        }
    ],
    "refusal_margin_harmful": [
        {"id": "h1", "prompt": "do bad"},
        {"id": "h2", "prompt": "do worse"},
    ],
}
CFG = P.PanelConfig(window_tokens=24)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    m = toy_model().double()
    J.eager_attention(m)
    for p in m.parameters():
        p.requires_grad_(False)
    names = [
        n
        for n, _ in m.named_parameters()
        if "layers.1." in n and ("down_proj" in n or "out_proj" in n)
    ]
    params = dict(m.named_parameters())
    p0 = {n: params[n] for n in names}
    g = torch.Generator().manual_seed(1)
    D = {n: torch.randn(v.shape, generator=g, dtype=v.dtype) * 0.05 for n, v in p0.items()}
    E = {n: torch.randn(v.shape, generator=g, dtype=v.dtype) * 0.05 for n, v in p0.items()}
    return m, ToyTok(), p0, D, E


@pytest.mark.parametrize("obj", sorted(ITEMS))
def test_item_values_match_the_panel_evaluator(setup, obj):
    m, tok, *_ = setup
    b = J.ItemBatch(obj, ITEMS[obj], tok, CFG)
    got = b.values(m)
    ref = P2.evaluate(P.LMAdapter(m), tok, {obj: ITEMS[obj]}, CFG)[obj]
    assert torch.allclose(got, torch.tensor([r["value"] for r in ref], dtype=got.dtype), atol=1e-4)


def _autograd(m, b, p0, D, E):
    p = {n: v.clone().requires_grad_(True) for n, v in p0.items()}
    f = b.values(m, p)
    out = []
    for i in range(f.numel()):
        g = torch.autograd.grad(f[i], list(p.values()), create_graph=True, retain_graph=True)
        gd = sum((gi * D[n]).sum() for gi, n in zip(g, p, strict=True))
        h = torch.autograd.grad(gd, list(p.values()), retain_graph=True)
        out.append(
            (
                float(gd),
                float(sum((hi * D[n]).sum() for hi, n in zip(h, p, strict=True))),
                float(sum((hi * E[n]).sum() for hi, n in zip(h, p, strict=True))),
            )
        )
    return out


@pytest.mark.parametrize("obj", ["refusal_margin_harmful", "sycophancy_gap", "wikitext_nll"])
def test_jets_match_autograd_double_backward(setup, obj):
    m, tok, p0, D, E = setup
    b = J.ItemBatch(obj, ITEMS[obj], tok, CFG)
    with torch.enable_grad():
        j = J.jets(m, b, p0, D, E)
        ref = _autograd(m, b, p0, D, E)
    for i, (g, dd, ed) in enumerate(ref):
        assert j["J1"][i].item() == pytest.approx(g, rel=1e-4, abs=1e-8)
        assert j["J2"][i].item() == pytest.approx(dd, rel=1e-4, abs=1e-8)
        assert j["X"][i].item() == pytest.approx(ed, rel=1e-4, abs=1e-8)


def test_zero_directions_give_exact_zeros(setup):
    m, tok, p0, D, _ = setup
    b = J.ItemBatch("gsm8k_gold_lp", ITEMS["gsm8k_gold_lp"], tok, CFG)
    Z = {n: torch.zeros_like(v) for n, v in D.items()}
    with torch.enable_grad():
        j = J.jets(m, b, p0, Z, Z)
        k = J.jets(m, b, p0, D, None)
    assert torch.equal(j["J1"], torch.zeros_like(j["J1"])) and torch.equal(
        j["J2"], torch.zeros_like(j["J2"])
    )
    assert torch.equal(j["X"], torch.zeros_like(j["X"])) and torch.equal(
        k["X"], torch.zeros_like(k["X"])
    )


def test_s1_hvp_agrees_with_s2_jets_within_1e3(setup):
    """The slope seat's kill check: S2 vs S1 mismatch above 1e-3 relative on a CPU toy kills S2."""
    m, tok, p0, D, E = setup
    batches = [J.ItemBatch(o, ITEMS[o], tok, CFG) for o in ("refusal_margin_harmful",)]
    with torch.enable_grad():
        t = J.hvp_terms(m, batches, p0, D, E)
        j = J.jets(m, batches[0], p0, D, E)
    s2 = float(J.s2_prediction(j).mean())
    s1 = J.s1_prediction(t)
    assert abs(s1 - s2) <= 1e-3 * max(1e-12, abs(s2))
    assert t["DHD"] == pytest.approx(float(j["J2"].mean()), rel=1e-4)
    assert t["EHD"] == pytest.approx(float(j["X"].mean()), rel=1e-4)


def test_values_are_unchanged_by_functional_call_at_theta0(setup):
    m, tok, p0, *_ = setup
    b = J.ItemBatch("mmlupro_gold_lp", ITEMS["mmlupro_gold_lp"], tok, CFG)
    assert torch.equal(b.values(m), b.values(m, dict(p0)))
