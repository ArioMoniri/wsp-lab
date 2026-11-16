"""wm.lora: rank-8 LoRA SFT on construction items and its scaled task vector (RULING-wme-pilot1, E2 action space)."""

import pytest
import torch

from tests.test_wm_panel import ToyTok, toy_model
from wsp.wm import actions as A
from wsp.wm import lora as L
from wsp.wm import panel as P
from wsp.wm import panel2 as P2


def _rolemap(model):
    cfg = model.config.to_dict()
    return A.RoleMap.from_config(cfg, prefix="model.layers.")


def _gsm(i):
    return {
        "id": f"g{i}",
        "kind": "gsm8k",
        "question": f"{i}+{i}?",
        "gold": str(2 * i),
        "hash": f"h{i}",
        "group": f"h{i}",
    }


def test_targets_are_the_audited_writers_of_every_layer():
    model = toy_model()
    rm = _rolemap(model)
    t = L.writer_targets(rm)
    assert len(t) == 2 * rm.n_layers
    assert all(n.endswith(("mlp.down_proj.weight", "o_proj.weight", "out_proj.weight")) for n in t)


def test_task_vector_reproduces_the_adapter_function_and_lambda_scales_it():
    model = toy_model()
    rm = _rolemap(model)
    names = L.writer_targets(rm)
    ad = L.LoRA(model, names, rank=8, alpha=16, seed=1)
    with torch.no_grad():
        for p in ad.parameters():  # nonzero B so the adapter does something
            p.normal_(0, 0.05)
    x = torch.tensor([[3, 4, 5, 6, 7, 8]])
    with ad.active():
        y_adapter = model(x).logits
    y_base = model(x).logits
    assert not torch.allclose(y_adapter, y_base)
    tv = ad.task_vector()
    assert set(tv) == set(names) and all(
        tv[n].shape == dict(model.named_parameters())[n].shape for n in names
    )
    params = dict(model.named_parameters())
    saved = {n: params[n].detach().clone() for n in names}
    with torch.no_grad():
        for n in names:
            params[n].add_(tv[n])
    y_merged = model(x).logits
    with torch.no_grad():
        for n in names:
            params[n].copy_(saved[n])
    assert torch.allclose(y_merged, y_adapter, atol=1e-4)
    acts = L.lowrank_actions(ad.factors(), lam=0.5)
    assert {a.tensor for a in acts} == set(names)
    for a in acts:
        assert a.family == "lora_task_vector" and a.dose == 0.5
        d = a.intended_delta(params[a.tensor].detach())
        assert torch.allclose(d, 0.5 * tv[a.tensor], atol=1e-6)
    assert L.lowrank_actions(ad.factors(), lam=0.25)[0].key() != acts[0].key()


def test_lean_apply_drops_realised_increments_and_restores_bitwise():
    model = toy_model().to(torch.bfloat16)
    rm = _rolemap(model)
    names = L.writer_targets(rm)
    ad = L.LoRA(model, names, rank=8, alpha=16, seed=1)
    with torch.no_grad():
        for p in ad.parameters():
            p.normal_(0, 0.05)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    sess = A.EditSession(dict(model.named_parameters()), base_hash="b", rolemap=rm)
    L.apply_lean(sess, L.lowrank_actions(ad.factors(), lam=1.0))
    assert sess.theta_hash() != "b"
    assert all(r.realised.numel() == 0 for r in sess.records)
    sess.restore()
    assert sess.theta_hash() == "b"
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n])


def test_sft_rows_train_the_panel_target_on_construction_items_only():
    tok = ToyTok()
    cfg = P.PanelConfig()
    items = [_gsm(i) for i in range(4)]
    manifest = {
        "objectives": {
            "gsm8k_gold_lp": {
                "construction": {"hashes": ["h0", "h1", "h2", "h3"]},
                "dev": {"hashes": ["d"]},
            }
        }
    }
    rows = L.sft_rows("gsm8k_gold_lp", items, tok, cfg, manifest)
    want = P2.rows_for("gsm8k_gold_lp", items[0], tok, cfg)[0]
    assert rows[0].ids == want.ids and rows[0].score_from == want.score_from
    bad = items + [dict(_gsm(9), hash="d")]
    with pytest.raises(L.SplitError):
        L.sft_rows("gsm8k_gold_lp", bad, tok, cfg, manifest)


def test_sycophancy_sft_uses_the_suggested_prompt_row():
    tok = ToyTok()
    it = {
        "id": "s",
        "kind": "syc_answer",
        "question": "Q?",
        "neutral_prompt": "Q?",
        "suggested_prompt": "Q? I think the answer is B but I'm really not sure.",
        "correct_answer": "A",
        "incorrect_answer": "B",
        "hash": "hs",
        "group": "hs",
    }
    man = {"objectives": {"sycophancy_real": {"construction": {"hashes": ["hs"]}}}}
    rows = L.sft_rows("sycophancy_real", [it], tok, P.PanelConfig(), man)
    assert len(rows) == 1 and rows[0].key[2] == "suggested"


def test_training_lowers_target_loss_and_leaves_base_weights_bitwise_unchanged():
    model = toy_model()
    rm = _rolemap(model)
    tok = ToyTok()
    items = [_gsm(i) for i in range(6)]
    man = {
        "objectives": {"gsm8k_gold_lp": {"construction": {"hashes": [i["hash"] for i in items]}}}
    }
    rows = L.sft_rows("gsm8k_gold_lp", items, tok, P.PanelConfig(), man)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ad = L.LoRA(model, L.writer_targets(rm), rank=8, alpha=16, seed=3)
    log = L.train(
        model,
        ad,
        rows,
        pad_id=0,
        lr=5e-3,
        epochs=6,
        batch_rows=3,
        seed=3,
        device="cpu",
        max_seconds=60,
    )
    assert log["steps"] == 12 and log["loss"][-1] < log["loss"][0]
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n
    assert not any(p.requires_grad for p in model.parameters())


def test_training_respects_the_wall_clock_cap():
    model = toy_model()
    rm = _rolemap(model)
    tok = ToyTok()
    items = [_gsm(i) for i in range(6)]
    man = {
        "objectives": {"gsm8k_gold_lp": {"construction": {"hashes": [i["hash"] for i in items]}}}
    }
    rows = L.sft_rows("gsm8k_gold_lp", items, tok, P.PanelConfig(), man)
    ad = L.LoRA(model, L.writer_targets(rm), rank=8, alpha=16, seed=3)
    log = L.train(
        model,
        ad,
        rows,
        pad_id=0,
        lr=1e-3,
        epochs=1000,
        batch_rows=3,
        seed=3,
        device="cpu",
        max_seconds=0.0,
    )
    assert log["steps"] <= 1 and log["stopped_by_cap"]
