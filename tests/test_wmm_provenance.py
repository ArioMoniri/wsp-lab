"""SYNTHESIS-r14 R2 (a)-(c): provenance record, weight-sketch canary, batch-1 behavioural canary (wm/wmm)."""

import json

import pytest
import torch

from wsp.wm import actions as A
from wsp.wm.wmm import canary as K
from wsp.wm.wmm import provenance as PV


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16, generator=g).bfloat16(),
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 24, generator=g).bfloat16(),
        "model.embed_tokens.weight": torch.randn(50, 16, generator=g).bfloat16(),
    }


# ------------------------------------------------------------------ (a) provenance record


def test_bytes_hash_is_recomputed_from_bytes_and_matches_state_hash():
    p = _params()
    h = PV.bytes_sha256(p)
    assert h == A.state_hash(p)
    q = {k: v.clone() for k, v in p.items()}
    q["model.embed_tokens.weight"][0, 0] += 1.0
    assert PV.bytes_sha256(q) != h


def test_edit_record_contents():
    before = PV.tensor_hashes(_params())
    p = _params()
    p["model.layers.0.mlp.down_proj.weight"][1, 2] += 0.5
    after = PV.tensor_hashes(p)
    rec = PV.edit_fields({"family": "syc_lora", "lambda": 1.0, "seed": 3}, before, after)
    assert rec["base_sha256"] == PV.combine(before)
    assert rec["edit_sha256"] == PV.combine(after) != rec["base_sha256"]
    assert list(rec["changed_tensors"]) == ["model.layers.0.mlp.down_proj.weight"]
    assert rec["recipe_sha256"] == PV.digest(rec["recipe"])
    none = PV.edit_fields({"family": "null"}, before, before)
    assert none["changed_tensors"] == {} and none["edit_sha256"] == none["base_sha256"]


def test_log_is_hash_chained_and_detects_tampering(tmp_path):
    path = tmp_path / "prov.jsonl"
    log = PV.ProvenanceLog(path)
    before = PV.tensor_hashes(_params())
    r1 = log.append_edit({"family": "tv", "lambda": 0.9}, before, before, job="j1")
    r2 = log.append("canary", job="j1", sketch_sha256="ab" * 32)
    assert r1["seq"] == 1 and r2["prev"] == r1["hash"] and r1["prev"] == PV.GENESIS
    again = PV.ProvenanceLog(path)
    assert [r["hash"] for r in again.records] == [r1["hash"], r2["hash"]]
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["recipe"]["lambda"] = 1.1  # tamper with a field
    path.write_text(json.dumps(rec) + "\n" + lines[1] + "\n")
    with pytest.raises(PV.LogCorrupt):
        PV.ProvenanceLog(path)


def test_log_detects_truncation(tmp_path):
    path = tmp_path / "prov.jsonl"
    log = PV.ProvenanceLog(path)
    log.append("canary", job="j")
    log.append("canary", job="j")
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n")
    with pytest.raises(PV.LogCorrupt):
        PV.ProvenanceLog(path)


# ------------------------------------------------------------------ (b) weight-sketch canary


def test_sketch_is_deterministic_and_exact_when_unchanged():
    p = _params()
    s1 = K.weight_sketch(p, seed=7)
    s2 = K.weight_sketch({k: v.clone() for k, v in p.items()}, seed=7)
    assert all(s1[n].shape == (K.N_PROJ,) for n in s1)
    cmp = K.compare_sketches(s1, s2)
    assert cmp["pass"] and cmp["changed"] == []
    assert (
        K.weight_sketch(p, seed=8)["model.embed_tokens.weight"].tolist()
        != s1["model.embed_tokens.weight"].tolist()
    )


def test_sketch_estimates_the_size_of_a_change():
    p = _params()
    s1 = K.weight_sketch(p, seed=1)
    q = {k: v.clone() for k, v in p.items()}
    delta = torch.zeros(50, 16)
    delta[3] = 0.5
    q["model.embed_tokens.weight"] = (q["model.embed_tokens.weight"].float() + delta).bfloat16()
    true = float(
        (q["model.embed_tokens.weight"].float() - p["model.embed_tokens.weight"].float()).norm()
    )
    cmp = K.compare_sketches(s1, K.weight_sketch(q, seed=1))
    assert not cmp["pass"] and cmp["changed"] == ["model.embed_tokens.weight"]
    est = cmp["estimated_norm"]["model.embed_tokens.weight"]
    assert 0.5 * true < est < 1.6 * true  # JL estimate with 64 projections


def test_sketch_digest_is_stable():
    p = _params()
    assert K.sketch_sha256(K.weight_sketch(p, seed=1)) == K.sketch_sha256(
        K.weight_sketch(p, seed=1)
    )


# ------------------------------------------------------------------ (c) behavioural canary at batch 1


def _tiny():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=4,
        max_position_embeddings=128,
        query_pre_attn_scalar=16,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    return Gemma3ForCausalLM(cfg).eval()


def test_behavioural_canary_batch1_fixed_order_and_sensitive():
    m = _tiny()
    prompts = [[2, 5, 9, 11], [2, 7], [2, 13, 14, 15, 16, 3]]
    c1 = K.behavioural_canary(m, prompts, max_new_tokens=6, stop_ids=[1])
    c2 = K.behavioural_canary(m, prompts, max_new_tokens=6, stop_ids=[1])
    assert c1["sha256"] == c2["sha256"] and c1["batch_size"] == 1 and len(c1["per_prompt"]) == 3
    rev = K.behavioural_canary(m, prompts[::-1], max_new_tokens=6, stop_ids=[1])
    assert (
        rev["per_prompt"] == c1["per_prompt"][::-1]
    )  # batch 1: an item's output does not depend on its neighbours
    with torch.no_grad():
        m.model.layers[0].mlp.down_proj.weight.add_(0.5)
    c3 = K.behavioural_canary(m, prompts, max_new_tokens=6, stop_ids=[1])
    assert c3["sha256"] != c1["sha256"]
