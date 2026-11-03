import torch

from wsp.zoo import ablate_direction, random_unit_direction


def test_ablation_removes_direction_from_rows(toy_weights):
    W = toy_weights["layers.0.mlp.down_proj"]
    d = random_unit_direction(W.shape[0], seed=3)
    W2 = ablate_direction(W, d, weight=1.0)
    assert torch.allclose(d @ W2, torch.zeros(W.shape[1]), atol=1e-5)
    assert not torch.allclose(W, W2)


def test_partial_weight_scales_linearly(toy_weights):
    W = toy_weights["layers.1.attn.o_proj"]
    d = random_unit_direction(W.shape[0], seed=1)
    half = ablate_direction(W, d, 0.5)
    assert torch.allclose(d @ half, 0.5 * (d @ W), atol=1e-5)


def test_direction_is_seeded():
    assert torch.equal(random_unit_direction(10, 5), random_unit_direction(10, 5))
    assert not torch.equal(random_unit_direction(10, 5), random_unit_direction(10, 6))
