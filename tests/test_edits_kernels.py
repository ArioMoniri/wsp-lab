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
    refusal_dirs = torch.stack([random_unit_direction(32, seed=k + 7) for k in range(3)])
    state_new, touched = heretic_abliteration_kernel(
        state,
        refusal_dirs,
        layer_tensors,
        direction_index=None,  # per-layer directions: row layer+1
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
        d = refusal_dirs[layer + 1]
        w = profile[layer]
        for name in layer_tensors[layer]:
            W_old = toy_weights[name]
            W_new = state_new[name]
            assert torch.allclose(d @ W_new, (1 - w) * (d @ W_old), atol=1e-5)
            assert not torch.equal(W_old, W_new)


def _heretic_delta(directions, layers, weights, direction_index, dist_params, n_layers):
    """Run Heretic 2.0's own Abliteration.modify_model on fake LoRA-wrapped modules."""
    from types import SimpleNamespace

    from heretic.modifiers.abliteration import (
        Abliteration,
        Parameters,
        RowNormalization,
        Settings,
        WeightDistribution,
    )

    class FakeLinear:
        def __init__(self, W):
            self.weight = W
            self.base_layer = SimpleNamespace(weight=W)
            self.lora_A = {"default": SimpleNamespace(weight=torch.zeros(1, W.shape[1]))}
            self.lora_B = {"default": SimpleNamespace(weight=torch.zeros(W.shape[0], 1))}

    modules = {layer: FakeLinear(weights[layer]) for layer in range(n_layers)}

    class FakeModel:
        def get_layers(self):
            return list(range(n_layers))

        def get_layer_modules(self, i):
            return {"mlp.down_proj": [modules[i]]}

    mod = Abliteration.__new__(Abliteration)
    mod.settings = Settings(row_normalization=RowNormalization.NONE, orthogonalize_direction=True)
    mod.residual_directions = directions
    params = Parameters(
        direction_index=direction_index,
        weight_distributions={"mlp.down_proj": WeightDistribution(**dist_params)},
    )
    mod.modify_model(SimpleNamespace(get_model=lambda: FakeModel()), params)
    return {
        layer: m.lora_B["default"].weight @ m.lora_A["default"].weight
        for layer, m in modules.items()
    }


@pytest.mark.parametrize("direction_index", [-1.0, 0.0, 2.75, 5.5, None])
def test_kernel_matches_installed_heretic_modify_model(direction_index):
    """Parity with heretic.modifiers.abliteration.Abliteration.modify_model on a toy layer count.

    Covers: finite support (layers beyond min_weight_distance untouched even with min_weight > 0),
    the boundary layer at exactly min_weight_distance, fractional positions, embedding-offset
    direction interpolation and the per-layer (None) direction mode.
    """
    n_layers, d_model, d_in = 8, 16, 12
    g = torch.Generator().manual_seed(3)
    directions = torch.nn.functional.normalize(
        torch.randn(n_layers + 1, d_model, generator=g), dim=1
    )
    weights = {i: torch.randn(d_model, d_in, generator=g) for i in range(n_layers)}
    dist = {
        "max_weight": 0.9,
        "max_weight_position": 4.5,
        "min_weight": 0.2,
        "min_weight_distance": 2.5,
    }
    expected = _heretic_delta(directions, None, weights, direction_index, dist, n_layers)

    state = {f"l{i}.w": w.clone() for i, w in weights.items()}
    layer_tensors = {i: [f"l{i}.w"] for i in range(n_layers)}
    new_state, touched = heretic_abliteration_kernel(
        state, directions, layer_tensors, direction_index, **dist
    )
    for i in range(n_layers):
        ours = new_state[f"l{i}.w"] - weights[i]
        assert torch.allclose(ours, expected[i], atol=1e-5), f"layer {i}"
    # Layers 0, 1, 7 are outside radius 2.5 around 4.5; layers 2..7 distance <= 2.5 except 7(2.5 ok)
    inside = {i for i in range(n_layers) if abs(i - 4.5) <= 2.5}
    assert {int(k[1:].split(".")[0]) for k in touched} == inside
    assert 0 not in inside and 1 not in inside


def test_profile_boundary_is_inclusive_and_outside_is_absent():
    profile = layer_weight_profile(list(range(10)), 1.0, 5, 0.3, 2)
    assert set(profile) == {3, 4, 5, 6, 7}
    assert profile[3] == pytest.approx(0.3) and profile[7] == pytest.approx(0.3)


def test_zero_weight_layers_are_skipped():
    dirs = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    state = {"a": torch.randn(4, 4), "b": torch.randn(4, 4)}
    _, touched = heretic_abliteration_kernel(state, dirs, {0: ["a"], 1: ["b"]}, 0.0, 1.0, 0, 0.0, 1)
    assert touched == ["a"]  # layer 1 sits at the radius with min_weight 0 -> weight 0 -> skipped


def test_fold_post_norm_makes_the_written_vector_orthogonal_after_the_post_norm():
    """Sandwich norm: the residual receives g * rmsnorm(W x), so ablating d in W's output space leaves
    a component along d unless the per-channel gain g is folded into the direction (d' ~ g * d)."""
    g_ = torch.Generator().manual_seed(11)
    d_model, d_in = 8, 6
    W = torch.randn(d_model, d_in, generator=g_)
    x = torch.randn(d_in, 5, generator=g_)
    gain = torch.rand(d_model, generator=g_) + 0.5
    d = torch.nn.functional.normalize(torch.randn(d_model, generator=g_), dim=0)
    dirs = torch.stack([d, d])

    def written(W_):
        y = W_ @ x
        z = gain[:, None] * y / y.pow(2).mean(0, keepdim=True).sqrt()
        return d @ z  # component of what is added to the residual along d

    kw = {
        "direction_index": 0.0,
        "max_weight": 1.0,
        "max_weight_position": 0,
        "min_weight": 0.0,
        "min_weight_distance": 1,
    }
    plain, _ = heretic_abliteration_kernel({"w": W}, dirs, {0: ["w"]}, **kw)
    folded, touched = heretic_abliteration_kernel(
        {"w": W}, dirs, {0: ["w"]}, gains={"w": gain}, **kw
    )
    assert touched == ["w"]
    assert written(folded["w"]).abs().max() < 1e-5
    assert (
        written(plain["w"]).abs().max() > 1e-2
    )  # without folding the gain leaks the direction back
    assert written(W).abs().max() > 1e-2


def test_fold_direction_is_unit_and_identity_for_unit_gain():
    from wsp.edits.kernels import fold_direction

    d = torch.nn.functional.normalize(torch.randn(8), dim=0)
    assert torch.allclose(fold_direction(d, torch.ones(8)), d, atol=1e-6)
    f = fold_direction(d, torch.rand(8) + 0.5)
    assert f.norm().item() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="zero"):
        fold_direction(d, torch.zeros(8))
