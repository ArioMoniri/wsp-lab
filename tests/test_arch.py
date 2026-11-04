"""Architecture registry and per-model zoo wiring (CPU, no weights).

Parameter names/shapes come from tests/fixtures/arch/*.json, written by scripts/arch_inspect.py from
each model's config on the meta device (cross-checked against the checkpoint's safetensors names).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from wsp.zoo import arch as A

FIXTURES = Path(__file__).parent / "fixtures" / "arch"
SLUGS = ["qwen3.5-0.8b", "qwen3.5-4b", "lfm3-1.7b", "gemma-3-270m"]

# Hand-written from the architectures (docs/ARCHITECTURES.md): per layer type, the module paths
# whose output is added to the residual stream (before any sandwich post-norm).
EXPECTED_WRITERS = {
    "qwen3.5-0.8b": {
        "attn_out": {"linear_attn.out_proj", "self_attn.o_proj"},
        "mlp_out": {"mlp.down_proj"},
    },
    "qwen3.5-4b": {
        "attn_out": {"linear_attn.out_proj", "self_attn.o_proj"},
        "mlp_out": {"mlp.down_proj"},
    },
    "lfm3-1.7b": {
        "attn_out": {"conv.out_proj", "self_attn.o_proj"},
        "mlp_out": {"feed_forward.w2"},
    },
    "gemma-3-270m": {"attn_out": {"self_attn.o_proj"}, "mlp_out": {"mlp.down_proj"}},
}


def fixture(slug: str) -> dict:
    return json.loads((FIXTURES / f"{slug}.json").read_text())


@pytest.fixture(params=SLUGS)
def model(request):
    fx = fixture(request.param)
    return request.param, fx


def info_for(fx: dict) -> A.ArchInfo:
    return A.arch_info_from_dict(fx["text_config"])


def test_fixtures_exist_and_match_checkpoint_names():
    for slug in SLUGS:
        fx = fixture(slug)
        assert fx["params"], slug
        ckpt = fx["checkpoint_names"]
        if ckpt is None:  # gated: Gemma has no checkpoint listing (see docs/ARCHITECTURES.md)
            assert slug == "gemma-3-270m"
            continue
        norm = {k.replace("model.language_model.", "model.", 1) for k in ckpt}
        lm = {k for k in fx["params"] if k != "lm_head.weight"}
        assert lm <= norm, sorted(lm - norm)[:5]


def test_detects_architecture_hidden_size_and_layers_from_config(model):
    slug, fx = model
    info = info_for(fx)
    assert info.hidden_size == fx["text_config"]["hidden_size"]
    assert info.num_layers == fx["text_config"]["num_hidden_layers"]
    expected = {"qwen3.5": "qwen3_5", "lfm3-1": "lfm3", "gemma-3": "gemma3"}
    assert info.arch.name == next(v for k, v in expected.items() if slug.startswith(k))
    # The embedding matrix width is the residual dimension.
    assert fx["params"]["model.embed_tokens.weight"][1] == info.hidden_size


def test_every_layer_has_exactly_one_writer_per_component_writing_the_residual(model):
    slug, fx = model
    info = info_for(fx)
    index = A.index_state_keys(fx["params"])
    seen = {"attn_out": set(), "mlp_out": set()}
    for layer in range(info.num_layers):
        for component in ("attn_out", "mlp_out"):
            keys = A.writer_keys(info.arch.writers[component], layer, index)
            assert len(keys) == 1, (slug, layer, component, keys)
            assert fx["params"][keys[0]][0] == info.hidden_size, keys[0]  # rows = residual dim
            seen[component].add(A.split_key(keys[0])[1])
    assert seen == EXPECTED_WRITERS[slug]


def test_post_writer_norms_exist_exactly_for_sandwich_norm_models(model):
    slug, fx = model
    info = info_for(fx)
    index = A.index_state_keys(fx["params"])
    sandwich = slug in ("lfm3-1.7b", "gemma-3-270m")
    assert bool(info.post_norms) == sandwich
    for path in info.post_norms.values():
        for layer in range(info.num_layers):
            key = A.layer_key(index, layer, path)
            assert fx["params"][key] == [info.hidden_size], key
    if slug == "gemma-3-270m":
        assert info.norm_gain == "one_plus_weight"
    if slug == "lfm3-1.7b":
        assert info.norm_gain == "weight"


def test_resolve_spec_legacy_names_regexes_and_unknowns():
    q = A.ARCHS["qwen3_5"]
    # Legacy 0.8B config names keep their meaning, and cover the linear-attention writer.
    assert A.resolve_spec("self_attn.o_proj", q) == q.writers["attn_out"]
    assert A.resolve_spec("mlp.down_proj", q) == q.writers["mlp_out"]
    assert A.resolve_spec("attn_out", q) == q.writers["attn_out"]
    # Anything else is a regex over the in-layer module path.
    assert A.resolve_spec(r"mlp\.(gate|up)_proj", q) == (r"mlp\.(gate|up)_proj",)
    lfm = A.ARCHS["lfm3"]
    assert A.resolve_spec("mlp.down_proj", lfm) == (r"feed_forward\.w2",)
    with pytest.raises(ValueError, match="invalid"):
        A.resolve_spec("(", q)


def test_legacy_qwen_selection_equals_old_hard_coded_names():
    fx = fixture("qwen3.5-0.8b")
    index = A.index_state_keys(fx["params"])
    q = A.ARCHS["qwen3_5"]
    for layer in range(24):
        old = []
        for t in ("mlp.down_proj", "self_attn.o_proj"):
            key = f"model.layers.{layer}.{t}.weight"
            if key not in fx["params"] and t == "self_attn.o_proj":
                key = f"model.layers.{layer}.linear_attn.out_proj.weight"
            old.append(key)
        new = [
            k
            for spec in ("mlp.down_proj", "self_attn.o_proj")
            for k in A.writer_keys(A.resolve_spec(spec, q), layer, index)
        ]
        assert new == old


def test_spec_matching_nothing_names_model_and_layer():
    fx = fixture("lfm3-1.7b")
    index = A.index_state_keys(fx["params"])
    with pytest.raises(KeyError, match=r"nope.*layer 3.*lfm3"):
        A.select_keys(["nope"], 3, index, A.ARCHS["lfm3"])


def test_validate_config_layers_against_model_depth():
    info = info_for(fixture("gemma-3-270m"))  # 18 layers
    cfg = {
        "families": {
            "random_dir": {"layers": [6, 17]},
            "abliteration": {
                "direction_index": {"min": -1, "max": 17},
                "max_weight_position": {"min": 0, "max": 18},
            },
        }
    }
    A.validate_family_config(cfg, info)
    cfg["families"]["random_dir"]["layers"] = [6, 18]
    with pytest.raises(ValueError, match="layer 18"):
        A.validate_family_config(cfg, info)
    cfg["families"]["random_dir"]["layers"] = [6]
    cfg["families"]["abliteration"]["direction_index"]["max"] = 18
    with pytest.raises(ValueError, match="direction_index"):
        A.validate_family_config(cfg, info)


def test_declared_architecture_must_agree_with_model_config():
    info = info_for(fixture("qwen3.5-4b"))
    A.check_declared({"hidden_size": 2560, "num_layers": 32}, info)
    with pytest.raises(ValueError, match="hidden_size"):
        A.check_declared({"hidden_size": 1024, "num_layers": 32}, info)
    with pytest.raises(ValueError, match="num_layers"):
        A.check_declared({"hidden_size": 2560, "num_layers": 24}, info)


def test_validate_directions_shape():
    info = info_for(fixture("gemma-3-270m"))
    A.validate_directions(torch.zeros(19, 640), info)
    with pytest.raises(ValueError, match="640"):
        A.validate_directions(torch.zeros(19, 1024), info)
    with pytest.raises(ValueError, match="19"):
        A.validate_directions(torch.zeros(25, 640), info)


def _peft_hits(names, targets):
    """PEFT target_modules semantics: a module matches when its path ends with a target name."""
    return {n for n in names if any(n == t or n.endswith("." + t) for t in targets)}


def test_lora_targets_hit_every_layer_mlp_and_never_the_lfm_attention_gate():
    default = ["q", "k", "v", "o", "up", "down"]
    for slug in SLUGS:
        fx = fixture(slug)
        info = info_for(fx)
        modules = {k[: -len(".weight")] for k, s in fx["params"].items() if len(s) >= 2}
        hits = _peft_hits(modules, A.lora_target_modules(default, info.arch))
        for layer in range(info.num_layers):
            in_layer = {m for m in hits if f".layers.{layer}." in m}
            assert any(m.endswith(("down_proj", "w2")) for m in in_layer), (slug, layer)
            assert any(m.endswith(("up_proj", "w3")) for m in in_layer), (slug, layer)
        assert not any(m.endswith("self_attn.gate_proj") for m in hits), slug
        # 'gate' means the MLP gate; on LFM3 it must be w1, not the attention head gate.
        gate = _peft_hits(modules, A.lora_target_modules(["gate"], info.arch))
        assert gate and all(m.endswith(("mlp.gate_proj", "feed_forward.w1")) for m in gate)
    # Qwen mapping is unchanged from the 0.8B zoo (existing LoRA entries stay reproducible).
    q = A.ARCHS["qwen3_5"]
    assert A.lora_target_modules(["q", "k", "v", "o", "up", "down"], q) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
    ]
    # Unknown names pass through untouched (explicit module names stay possible).
    assert A.lora_target_modules(["in_proj_qkv"], q) == ["in_proj_qkv"]


def test_remote_code_is_opt_in_by_allowlist_or_config():
    assert A.remote_code_kwargs("LiquidAI/LFM3-1.7B") == {"trust_remote_code": True}
    assert A.remote_code_kwargs("Qwen/Qwen3.5-4B") == {}
    assert A.remote_code_kwargs("someone/else", {"trust_remote_code": True}) == {
        "trust_remote_code": True
    }
    assert A.remote_code_kwargs("someone/else", {}) == {}


def test_index_state_keys_handles_both_prefixes():
    keys = [
        "model.language_model.layers.2.mlp.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.embed_tokens.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "model.visual.blocks.3.attn.proj.weight",
    ]
    index = A.index_state_keys(keys)
    assert set(index) == {0, 2}  # mtp / visual blocks are not language-model layers
    assert A.split_key(keys[0]) == (2, "mlp.down_proj")
    assert A.split_key("model.norm.weight") is None


def test_regex_in_config_is_fullmatched_against_module_path():
    fx = fixture("qwen3.5-4b")
    index = A.index_state_keys(fx["params"])
    q = A.ARCHS["qwen3_5"]
    keys = A.select_keys([r"mlp\.(gate|up)_proj"], 5, index, q)
    assert keys == ["model.layers.5.mlp.gate_proj.weight", "model.layers.5.mlp.up_proj.weight"]
    # fullmatch, not search: a partial pattern selects nothing.
    with pytest.raises(KeyError):
        A.select_keys([r"gate"], 5, index, q)


def test_post_norm_gain_vector():
    w = torch.tensor([0.5, -0.25])
    assert torch.equal(A.effective_gain(w, "weight"), w)
    assert torch.equal(A.effective_gain(w, "one_plus_weight"), torch.tensor([1.5, 0.75]))
    with pytest.raises(ValueError):
        A.effective_gain(w, "bogus")


def test_arch_ids_cover_registry_model_types():
    for slug in SLUGS:
        mt = fixture(slug)["text_config"]["model_type"]
        assert A.detect_arch(mt).name != "dense", mt
    assert re.fullmatch(r"\w+", A.detect_arch("llama").name)  # generic fallback exists


def test_remote_code_kwargs_reach_refusal_extraction_loader(monkeypatch):
    import transformers

    import wsp.edits.refusal as R

    seen = {}

    class Fake:
        def eval(self):
            return self

    def fake(model, **kw):
        seen.update(kw)
        return Fake()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake)
    monkeypatch.setattr(R, "residual_mean", lambda *a, **k: torch.ones(3, 4))
    key = {"model_commit": "c", "dtype": "float32"}
    prompts = {"good": ["a"], "bad": ["b"]}
    R._compute({"base_model": "LiquidAI/LFM3-1.7B"}, None, key, prompts)
    assert seen["trust_remote_code"] is True
    seen.clear()
    R._compute({"base_model": "Qwen/Qwen3.5-4B"}, None, key, prompts)
    assert "trust_remote_code" not in seen
