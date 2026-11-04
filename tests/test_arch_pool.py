"""Pooling audit: every tensor of every zoo model lands in a per-type pool (CPU, names only)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from wsp.features.pool import TYPES, classify, pool_features, pooled_dims

FIXTURES = Path(__file__).parent / "fixtures" / "arch"
SLUGS = ["qwen3.5-0.8b", "qwen3.5-4b", "lfm3-1.7b", "gemma-3-270m"]


def names_of(slug: str) -> dict[str, list[str]]:
    fx = json.loads((FIXTURES / f"{slug}.json").read_text())
    out = {"zoo namespace (AutoModelForCausalLM state dict)": sorted(fx["params"])}
    if fx["checkpoint_names"]:
        out["checkpoint (safetensors names)"] = fx["checkpoint_names"]
    return out


@pytest.mark.parametrize("slug", SLUGS)
def test_every_parameter_of_every_model_has_a_typed_per_type_pool(slug):
    n_layers = json.loads((FIXTURES / f"{slug}.json").read_text())["text_config"][
        "num_hidden_layers"
    ]
    for space, names in names_of(slug).items():
        unknown = [n for n in names if classify(n)[0] == "other"]
        assert not unknown, (slug, space, unknown[:5])
        for name in names:
            kind, layer = classify(name)
            rec = {"name": name, "stats": torch.full((18,), 7.0), "svd": torch.full((8, 129), 7.0)}
            stats, svd = pool_features([rec], n_layers=n_layers, buckets=4)
            t = TYPES.index(kind)
            assert torch.all(svd[8 * t : 8 * (t + 1)] == 7), name  # per-type spectrum pool
            typed = stats[18:].reshape(len(TYPES), 4, 18)[t]
            assert torch.all(typed.sum(0) == 7), name  # exactly one bucket of that type
            assert torch.all(stats[:18] == 7), name  # and the global mean
            if kind == "aux":
                assert layer is None, name  # mtp.layers.0 must not pollute layer bucket 0


def counts(slug: str, space_prefix: str = "zoo") -> Counter:
    names = next(v for k, v in names_of(slug).items() if k.startswith(space_prefix))
    return Counter(classify(n)[0] for n in names)


def test_lfm3_type_counts_match_the_architecture():
    c = counts("lfm3-1.7b")
    # 24 layers: 11 attention (full + sliding), 13 short-conv.
    assert c["q"] == c["k"] == c["v"] == c["o"] - 13 == 11
    assert c["attn_gate"] == 11  # self_attn.gate_proj is NOT the MLP gate
    assert c["gate"] == c["up"] == c["down"] == 24  # feed_forward.w1 / w3 / w2
    assert c["lin_qkv"] == 13  # conv.in_proj
    assert c["lin_state"] == 13  # conv.conv
    # operator/ffn/post_operator/post_ffn per layer + q_norm/k_norm on attention + embedding_norm
    assert c["norm"] == 24 * 4 + 11 * 2 + 1
    assert c["embed"] == 2 and c["aux"] == 0 and c["other"] == 0


def test_gemma3_type_counts_match_the_architecture():
    c = counts("gemma-3-270m")
    assert c["q"] == c["k"] == c["v"] == c["o"] == 18
    assert c["gate"] == c["up"] == c["down"] == 18
    # input / post_attention / pre_feedforward / post_feedforward + q_norm / k_norm per layer + final
    assert c["norm"] == 18 * 6 + 1
    assert c["embed"] == 2 and c["other"] == 0 and c["attn_gate"] == 0


def test_qwen35_4b_checkpoint_assigns_visual_and_mtp_to_aux():
    ck = counts("qwen3.5-4b", "checkpoint")
    zoo = counts("qwen3.5-4b", "zoo")
    assert zoo["aux"] == 0 and zoo["other"] == 0
    assert ck["aux"] > 0 and ck["other"] == 0
    assert zoo["lin_qkv"] == zoo["lin_z"] == zoo["lin_b"] == zoo["lin_a"] == 24
    assert zoo["q"] == zoo["k"] == zoo["v"] == 8 and zoo["o"] == 8 + 24
    assert zoo["gate"] == zoo["up"] == zoo["down"] == 32


def test_pooled_layout_is_versioned():
    stats, spectrum = pooled_dims(4)
    assert (stats, spectrum) == (18 * (1 + len(TYPES) * 4), 8 * len(TYPES))
    assert TYPES[-1] == "other"  # catch-all stays last
    assert {"attn_gate", "aux"} <= set(TYPES)
    from wsp.features.extract import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["feature_version"] == 4


def test_lfm3_attention_gate_does_not_pollute_the_mlp_gate_pool():
    recs = [
        {"name": "model.layers.2.self_attn.gate_proj.weight", "stats": torch.full((18,), 9.0)}
        | {"svd": torch.full((8, 129), 9.0)},
        {"name": "model.layers.2.feed_forward.w1.weight", "stats": torch.full((18,), 1.0)}
        | {"svd": torch.full((8, 129), 1.0)},
    ]
    _stats, svd = pool_features(recs, n_layers=24)
    spec = svd.reshape(len(TYPES), 8)
    assert torch.all(spec[TYPES.index("gate")] == 1)
    assert torch.all(spec[TYPES.index("attn_gate")] == 9)
