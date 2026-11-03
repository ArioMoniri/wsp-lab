"""Unit tests for edit kernels (CPU)."""

import pytest
import torch

from wsp.edits.kernels import heretic_abliteration_kernel, layer_weight_profile
from wsp.zoo import random_unit_direction


def test_layer_weight_profile_peaks_and_tapers():
    layers = list(range(6))
    profile = layer_weight_profile(
        layers=layers,
        max_weight=1.0,
        max_weight_position=3,
        min_weight=0.1,
        min_weight_distance=2,
    )
    assert profile[3] == 1.0
    assert profile[2] == pytest.approx(0.55, rel=1e-3)
    assert profile[4] == pytest.approx(0.55, rel=1e-3)
    assert profile[1] == pytest.approx(0.1)
    assert 0 not in profile
    assert profile[5] == pytest.approx(0.1)


def test_heretic_kernel_contract_on_toy_weights(toy_weights):
    """Ablating direction d with weight w should leave d^T W' = (1-w) d^T W on rows."""
    state = {k: v.clone() for k, v in toy_weights.items()}
    layers = [0, 1]
    layer_tensors = {
        0: ["layers.0.attn.o_proj", "layers.0.mlp.down_proj"],
        1: ["layers.1.attn.o_proj", "layers.1.mlp.down_proj"],
    }
    refusal_dirs = {
        layer: torch.stack([random_unit_direction(32, seed=layer + 7)]) for layer in layers
    }
    state_new, touched = heretic_abliteration_kernel(
        state,
        refusal_dirs,
        layer_tensors,
        direction_index=0,
        max_weight=0.8,
        max_weight_position=0,
        min_weight=0.0,
        min_weight_distance=2,
    )
    assert set(touched) == {
        "layers.0.attn.o_proj",
        "layers.0.mlp.down_proj",
        "layers.1.attn.o_proj",
        "layers.1.mlp.down_proj",
    }
    profile = layer_weight_profile(layers, 0.8, 0, 0.0, 2)
    for layer in layers:
        d = refusal_dirs[layer][0]
        w = profile[layer]
        for name in layer_tensors[layer]:
            W_old = toy_weights[name]
            W_new = state_new[name]
            assert torch.allclose(d @ W_new, (1 - w) * (d @ W_old), atol=1e-5)
            assert not torch.equal(W_old, W_new)
