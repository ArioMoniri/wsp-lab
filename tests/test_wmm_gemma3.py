"""Study M Gate 0 port (SYNTHESIS-r12 Gate 0): Gemma 3 / MedGemma text-only paths on tiny random configs (CPU).

The four silent bugs named by the synthesis: tied embeddings, the vision tower slipping past the ``.visual.`` guard,
``<|im_end|>`` instead of ``<end_of_turn>``, and the ignored ``enable_thinking``. Qwen behaviour must not change.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from wsp.wm import actions as A
from wsp.wm import chat as CH
from wsp.wm import controls as C
from wsp.wm import gemma3 as G3
from wsp.wm import hf
from wsp.wm import lora as LR

TYPES6 = ["sliding_attention"] * 5 + ["full_attention"]


def _text_cfg(**kw):
    from transformers import Gemma3TextConfig

    base = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "sliding_window": 4,
        "layer_types": TYPES6,
        "max_position_embeddings": 128,
        "query_pre_attn_scalar": 16,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }
    base.update(kw)
    return Gemma3TextConfig(**base)


def _tiny(seed=0, **kw):
    from transformers import Gemma3ForCausalLM

    torch.manual_seed(seed)
    m = Gemma3ForCausalLM(_text_cfg(**kw)).to(torch.bfloat16).eval()
    with torch.no_grad():  # random (non-zero) norm weights so the (1 + w) gains are exercised
        for n, p in m.named_parameters():
            if "norm" in n:
                p.normal_(0, 0.1)
    return m


def _mm_config(text_cfg) -> dict:
    """A multimodal-style config.json as shipped by google/medgemma-4b-it (no top-level tie key)."""
    t = text_cfg.to_dict()
    t["model_type"] = "gemma3_text"
    t.pop("tie_word_embeddings", None)
    return {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": t,
        "vision_config": {"model_type": "siglip_vision_model", "hidden_size": 8},
        "eos_token_id": [1, 106],
    }


def _write_checkpoint(model, path, prefix="language_model.model.", shards=2, extra=None):
    """Save ``model`` as a multimodal-style sharded checkpoint: text keys under ``prefix``, plus vision/projector
    tensors that a text-only loader must never read."""
    path.mkdir(parents=True, exist_ok=True)
    sd = {
        prefix + k[len("model.") :]: v.contiguous().clone()
        for k, v in model.state_dict().items()
        if k != "lm_head.weight"
    }
    sd["vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"] = torch.ones(8, 8).bfloat16()
    sd["multi_modal_projector.mm_input_projection_weight"] = torch.ones(8, 32).bfloat16()
    sd.update(extra or {})
    keys = sorted(sd)
    wm = {}
    for i in range(shards):
        part = {k: sd[k] for k in keys[i::shards]}
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        save_file(part, str(path / name))
        wm.update(dict.fromkeys(part, name))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (path / "config.json").write_text(json.dumps(_mm_config(model.config)))
    return path


# ------------------------------------------------------------------ role map and frozen tensors


def test_rolemap_accepts_sliding_attention_and_names_gemma_writers():
    cfg = _mm_config(_text_cfg())
    rm = A.RoleMap.from_config(cfg)
    assert rm.layer_types == tuple(TYPES6)
    assert rm.name(0, "o") == "model.layers.0.self_attn.o_proj.weight"
    assert rm.name(5, "o") == "model.layers.5.self_attn.o_proj.weight"
    for role in ("q", "k", "v"):
        assert rm.name(0, role) == f"model.layers.0.self_attn.{role}_proj.weight"
    with pytest.raises(KeyError):
        rm.name(0, "lin_qkv")
    assert len(rm.writers()) == 12


def test_tie_default_comes_from_the_model_type_not_false():
    """Bug 1: the MedGemma config has no tie key; Gemma 3 ties lm_head to embed_tokens."""
    cfg = _mm_config(_text_cfg())
    assert "tie_word_embeddings" not in cfg and "tie_word_embeddings" not in cfg["text_config"]
    assert A.RoleMap.from_config(cfg).tied is True
    assert A.RoleMap.from_config(cfg["text_config"]).tied is True
    explicit = {**cfg, "tie_word_embeddings": False}
    assert A.RoleMap.from_config(explicit).tied is False
    # Qwen behaviour unchanged: absent key -> untied
    qwen = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "vocab_size": 50,
            "layer_types": ["linear_attention", "full_attention"],
        },
    }
    assert A.RoleMap.from_config(qwen).tied is False


def test_vision_tower_and_projector_are_frozen_and_rejected_by_the_text_only_guard():
    """Bug 2: Gemma 3 names its vision tower ``vision_tower``, which the ``.visual.`` check let through."""
    for n in (
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "multi_modal_projector.mm_input_projection_weight",
        "model.multi_modal_projector.mm_soft_emb_norm.weight",
        "model.visual.blocks.0.attn.proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ):
        assert A.is_frozen(n), n
        with pytest.raises(RuntimeError):
            hf.assert_text_only([n, "model.layers.0.mlp.down_proj.weight"])
    hf.assert_text_only(["model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"])
    assert not A.is_frozen("model.layers.0.self_attn.o_proj.weight")


# ------------------------------------------------------------------ chat template and end token


class _Tok:
    def __init__(self, vocab, template, eos=1, unk=3):
        self.vocab, self.chat_template, self.eos_token_id, self.unk_token_id = (
            vocab,
            template,
            eos,
            unk,
        )
        self.calls = []

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def apply_chat_template(self, msgs, **kw):
        self.calls.append(kw)
        return "<prompt>" + msgs[0]["content"]


GEMMA_TEMPLATE = "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}...<end_of_turn>{% endfor %}"
QWEN_TEMPLATE = (
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>{% endif %}"
)


def test_end_token_is_end_of_turn_for_gemma_and_im_end_for_qwen():
    """Bug 3: ``<|im_end|>`` maps to <unk> on Gemma, so ``<end_of_turn>`` was never a stop token."""
    gem = _Tok({"<end_of_turn>": 106, "<eos>": 1}, GEMMA_TEMPLATE)
    qwen = _Tok({"<|im_end|>": 248046}, QWEN_TEMPLATE, eos=248044)
    assert CH.end_token(gem) == "<end_of_turn>" and CH.end_token_id(gem) == 106
    assert CH.end_token(qwen) == "<|im_end|>" and CH.end_token_id(qwen) == 248046
    assert CH.stop_ids(gem) == {1, 106}
    assert CH.stop_ids(qwen) == {248044, 248046}
    with pytest.raises(KeyError):
        CH.end_token(_Tok({}, ""))


def test_chat_prompt_never_silently_ignores_enable_thinking():
    """Bug 4: Gemma has no thinking switch; passing it was silently ignored. Qwen keeps the explicit False."""
    gem = _Tok({"<end_of_turn>": 106}, GEMMA_TEMPLATE)
    qwen = _Tok({"<|im_end|>": 7}, QWEN_TEMPLATE)
    CH.chat_prompt(gem, "hi")
    assert "enable_thinking" not in gem.calls[-1]
    with pytest.raises(ValueError):
        CH.chat_prompt(gem, "hi", thinking=True)
    CH.chat_prompt(qwen, "hi")
    assert qwen.calls[-1]["enable_thinking"] is False
    CH.chat_prompt(qwen, "hi", thinking=True)
    assert qwen.calls[-1]["enable_thinking"] is True


def test_panel_p3gen_and_genaudit_route_through_the_chat_helpers():
    from wsp.wm import genaudit as GA
    from wsp.wm import p3gen as PG
    from wsp.wm import panel as P

    gem = _Tok({"<end_of_turn>": 106}, GEMMA_TEMPLATE)
    P.chat_prompt(gem, "x")
    PG.chat(gem, "x")
    assert all("enable_thinking" not in c for c in gem.calls)
    assert GA.stop_ids(gem) == {1, 106}


# ------------------------------------------------------------------ text-only loading


def test_text_key_mapping_drops_vision_and_handles_both_prefixes():
    assert G3.text_key("language_model.model.layers.3.mlp.down_proj.weight") == (
        "model.layers.3.mlp.down_proj.weight"
    )
    assert G3.text_key("model.language_model.layers.3.mlp.down_proj.weight") == (
        "model.layers.3.mlp.down_proj.weight"
    )
    assert G3.text_key("language_model.model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert G3.text_key("model.norm.weight") == "model.norm.weight"
    assert G3.text_key("vision_tower.vision_model.post_layernorm.weight") is None
    assert G3.text_key("multi_modal_projector.mm_input_projection_weight") is None
    assert G3.text_key("language_model.lm_head.weight") == "lm_head.weight"


@pytest.mark.parametrize("prefix", ["language_model.model.", "model.language_model."])
def test_text_only_load_matches_the_source_model_and_ties_the_head(tmp_path, prefix):
    src = _tiny(seed=1)
    path = _write_checkpoint(src, tmp_path / "ck", prefix=prefix)
    m = G3.load_text_from_path(path, device="cpu")
    names = [n for n, _ in m.named_parameters()]
    assert not any(A.is_frozen(n) for n in names)
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    ids = torch.tensor([[2, 5, 9, 11, 13, 7, 21, 30, 4]])
    with torch.no_grad():
        a = src(ids).logits
        b = m(ids).logits
    assert torch.equal(a, b)
    rm = A.RoleMap.from_config(json.loads((path / "config.json").read_text()))
    sd = dict(m.named_parameters())
    for name in rm.writers().values():
        assert name in sd


def test_text_only_load_fails_on_missing_or_unknown_text_tensors(tmp_path):
    src = _tiny(seed=2)
    path = _write_checkpoint(
        src, tmp_path / "ck", extra={"language_model.model.bogus.weight": torch.ones(2).bfloat16()}
    )
    with pytest.raises(KeyError):
        G3.load_text_from_path(path, device="cpu")
    path2 = tmp_path / "ck2"
    _write_checkpoint(src, path2)
    idx = json.loads((path2 / "model.safetensors.index.json").read_text())
    idx["weight_map"].pop("language_model.model.norm.weight")
    (path2 / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(KeyError):
        G3.load_text_from_path(path2, device="cpu")


def test_architecture_signature_detects_shape_and_config_mismatch(tmp_path):
    a = _write_checkpoint(_tiny(seed=1), tmp_path / "a")
    b = _write_checkpoint(_tiny(seed=2), tmp_path / "b")
    c = _write_checkpoint(_tiny(seed=3, intermediate_size=40), tmp_path / "c")
    rep = G3.compare_architectures({"a": a, "b": b})
    assert rep["same"] and rep["n_text_tensors"] == len(G3.text_shapes(a))
    rep2 = G3.compare_architectures({"a": a, "c": c})
    assert not rep2["same"] and rep2["shape_mismatches"]


# ------------------------------------------------------------------ weight-space lines (task vector, interpolation)


def test_line_state_endpoints_are_bitwise_and_restore_is_exact(tmp_path):
    base_m, other_m = _tiny(seed=1), _tiny(seed=2)
    pa = _write_checkpoint(base_m, tmp_path / "base")
    pb = _write_checkpoint(other_m, tmp_path / "other")
    m = G3.load_text_from_path(pa, device="cpu")
    base_hash = G3.text_state_hash(m)
    line = G3.WeightLine(G3.WeightSource(pa), G3.WeightSource(pb), chunk_elems=37)
    names = line.names()
    assert "model.embed_tokens.weight" in names and not any(A.is_frozen(n) for n in names)
    assert "lm_head.weight" not in names  # tied: written through embed_tokens
    rec = line.apply(m, 0.0)
    assert rec["changed_tensors"] == 0 and G3.text_state_hash(m) == base_hash
    line.apply(m, 1.0)  # the far endpoint reproduces the other model bitwise
    other_sd = dict(other_m.named_parameters())
    for n, p in m.named_parameters():
        assert A.same_bytes(p.detach(), other_sd[n].detach()), n
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    rec = line.apply(m, 0.5)
    assert 0 < rec["changed_tensors"] <= len(names)
    line.apply(m, 0.0)
    assert G3.text_state_hash(m) == base_hash


def test_line_state_matches_the_dense_formula_and_is_chunk_invariant(tmp_path):
    pa = _write_checkpoint(_tiny(seed=1), tmp_path / "a")
    pb = _write_checkpoint(_tiny(seed=2), tmp_path / "b")
    m1 = G3.load_text_from_path(pa, device="cpu")
    m2 = G3.load_text_from_path(pa, device="cpu")
    G3.WeightLine(G3.WeightSource(pa), G3.WeightSource(pb), chunk_elems=13).apply(m1, -0.5)
    G3.WeightLine(G3.WeightSource(pa), G3.WeightSource(pb), chunk_elems=10**9).apply(m2, -0.5)
    sa, sb = G3.WeightSource(pa), G3.WeightSource(pb)
    for n, p in m1.named_parameters():
        assert A.same_bytes(p.detach(), dict(m2.named_parameters())[n].detach())
        wa, wb = sa.get(n).double(), sb.get(n).double()
        expect = torch.where(wb != wa, (1.5 * wa - 0.5 * wb).to(torch.bfloat16), sa.get(n))
        assert A.same_bytes(p.detach(), expect), n


def test_line_far_endpoint_is_bitwise_even_with_wide_exponent_gaps(tmp_path):
    """The additive fp32 form loses b + (o - b) = o when |o| and |b| differ by > 2^24 (5.7k entries on the real models)."""
    a_m, b_m = _tiny(seed=1), _tiny(seed=2)
    with torch.no_grad():
        w = a_m.model.layers[0].mlp.down_proj.weight
        w[0, :4] = torch.tensor([1e-30, 3e-12, 1e10, -7e-20]).bfloat16()
        v = b_m.model.layers[0].mlp.down_proj.weight
        v[0, :4] = torch.tensor([5.0, -2e8, 1e-25, 0.75]).bfloat16()
    pa = _write_checkpoint(a_m, tmp_path / "a")
    pb = _write_checkpoint(b_m, tmp_path / "b")
    m = G3.load_text_from_path(pa, device="cpu")
    rec = G3.WeightLine(G3.WeightSource(pa), G3.WeightSource(pb)).apply(m, 1.0)
    assert rec["far_endpoint_inexact_entries"] == 0
    for n, p in m.named_parameters():
        assert A.same_bytes(p.detach(), dict(b_m.named_parameters())[n].detach()), n


def test_task_vector_lambda_maps_to_the_line_coordinate():
    """τ = θ_target − θ_anchor; θ(λ) = θ_anchor + λτ = θ_target + (1 − λ)(θ_anchor − θ_target)."""
    assert G3.tv_t(1.0) == 0.0
    assert G3.tv_t(0.5) == 0.5 and G3.tv_t(1.5) == -0.5 and G3.tv_t(0.0) == 1.0


# ------------------------------------------------------------------ controls on Gemma 3


def _logits(m, ids):
    with torch.no_grad():
        return m(ids).logits.float()


def test_compensated_and_qk_norm_controls_cover_sliding_layers_and_are_exact():
    m = _tiny(seed=4)
    rm = A.RoleMap.from_config(_mm_config(m.config))
    ids = torch.tensor([[2, 5, 9, 11, 13, 7, 21, 30, 4]])
    ref = _logits(m, ids)
    params = dict(m.named_parameters())
    sess = A.EditSession(params, "base", rm)
    for bundle in C.compensated_pairs(rm, mlp_layers=(0, 5), attention_layers=(0, 5)):
        sess.apply_bundle(bundle)
    for act in C.post_norm_cancelled_scales(rm, layers=(0, 1, 5)):
        sess.apply(act)
    out = _logits(m, ids)
    assert torch.equal(out, ref)
    sess.restore()
    assert torch.equal(_logits(m, ids), ref)


def test_uniform_writer_rescale_is_cancelled_by_gemma_post_norms():
    """Engineer item 11: o_proj / down_proj feed post-norms, so a uniform writer rescale is (almost) inert. A
    directional (rank-one) writer edit of the same relative norm is not."""
    m = _tiny(
        seed=5, rms_norm_eps=1e-12
    )  # tiny random writers output ~1e-4, where a 1e-6 eps would dominate
    rm = A.RoleMap.from_config(_mm_config(m.config))
    ids = torch.tensor([[2, 5, 9, 11, 13, 7, 21, 30, 4]])
    ref = _logits(m, ids)
    sess = A.EditSession(dict(m.named_parameters()), "base", rm)
    for L in range(rm.n_layers):
        for r in ("o", "down"):
            sess.apply(A.scale(rm.name(L, r), factor=2.0, dose=1.0))
    d_scale = (_logits(m, ids) - ref).abs().max().item()
    sess.restore()
    g = torch.Generator().manual_seed(0)
    for L in range(rm.n_layers):
        name = rm.name(L, "o")
        w = dict(m.named_parameters())[name]
        u, v = torch.randn(w.shape[0], generator=g), torch.randn(w.shape[1], generator=g)
        alpha = float(w.detach().float().norm()) / float(u.norm() * v.norm())
        sess.apply(A.rank_one(name, u, v, alpha=alpha, dose=1.0))
    d_dir = (_logits(m, ids) - ref).abs().max().item()
    sess.restore()
    assert d_scale < 0.05 * d_dir


# ------------------------------------------------------------------ LoRA on Gemma 3


def test_lora_targets_and_training_run_on_a_tied_gemma_text_model():
    m = _tiny(seed=6).float()
    rm = A.RoleMap.from_config(_mm_config(m.config))
    targets = LR.writer_targets(rm)
    assert len(targets) == 12 and all(
        t.endswith(("o_proj.weight", "down_proj.weight")) for t in targets
    )

    class Row:
        def __init__(self, ids, s):
            self.ids, self.score_from = ids, s

    rows = [Row([2, 5, 9, 11, 13, 106 % 64], 3), Row([2, 7, 8, 21, 30, 4, 1], 4)] * 4
    ad = LR.LoRA(m, targets, rank=2, alpha=4, seed=0)
    with torch.no_grad():
        for k in ad.B:
            ad.B[k].normal_(0, 0.01)
    log = LR.train(m, ad, rows, pad_id=0, lr=1e-2, epochs=3, batch_rows=4, device="cpu", warmup=1)
    assert log["steps"] == 6 and log["loss"][-1] < log["loss"][0]
    for p in m.parameters():
        assert not p.requires_grad
    # the dense task-vector actions reproduce the hooked adapter (fp32 model, so no bf16 rounding)
    ids = torch.tensor([[2, 5, 9, 11, 13, 7]])
    with torch.no_grad(), ad.active():
        hooked = m(ids).logits
    ref_sd = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for n, (b, a, s) in ad.factors().items():
            dict(m.named_parameters())[n].add_(s * (b @ a))
        dense = m(ids).logits
        for n, p in m.named_parameters():
            p.copy_(ref_sd[n])
    assert torch.allclose(hooked, dense, atol=1e-4)


def test_sparse_config_is_resolved_through_gemma3_defaults(tmp_path):
    """google/gemma-3-4b-it ships a text_config with only 6 keys; sizes and layer types come from the class."""
    path = _write_checkpoint(_tiny(seed=1), tmp_path / "ck")
    cfg = json.loads((path / "config.json").read_text())
    full = cfg["text_config"]
    cfg["text_config"] = {
        k: full[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "model_type")
    }
    cfg["text_config"]["vocab_size"] = full[
        "vocab_size"
    ]  # the tiny vocab differs from the class default
    (path / "config.json").write_text(json.dumps(cfg))
    res = G3.resolved_config(path)
    rm = A.RoleMap.from_config(res)
    assert rm.tied and rm.n_layers == 6 and len(rm.layer_types) == 6
    assert res["text_config"]["head_dim"] == 256 and res["text_config"]["sliding_window"] == 4096
    sig = G3.architecture_signature(path)
    assert sig["text_config"]["num_key_value_heads"] == res["text_config"]["num_key_value_heads"]
