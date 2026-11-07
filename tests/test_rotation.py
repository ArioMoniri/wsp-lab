import torch

from wsp.features.rotation import (
    apply_and_round,
    audit,
    block_svd,
    changed_block,
    exclusion_reason,
    lora_factors_reproduce,
    realised_increment,
    rotate_block,
    scaled_variant,
)


def test_rotation_preserves_spectrum_and_changes_orientation():
    d = torch.zeros(6, 7)
    d[1:5, 2:6] = torch.tensor(
        [[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 2.0]]
    )
    rows, cols = changed_block(d)
    r = rotate_block(d, rows, cols, 42)
    assert torch.allclose(torch.linalg.svdvals(d), torch.linalg.svdvals(r), atol=1e-5)
    assert torch.allclose(d.norm(), r.norm(), atol=1e-5)
    assert torch.linalg.matrix_rank(d) == torch.linalg.matrix_rank(r)
    assert torch.count_nonzero(r[:1]) == 0 and torch.count_nonzero(r[:, :2]) == 0
    w = torch.arange(42.0).reshape(6, 7)
    assert not torch.allclose((d**3).sum(), (r**3).sum())
    assert not torch.allclose(
        torch.nn.functional.cosine_similarity(d.flatten(), w.flatten(), dim=0),
        torch.nn.functional.cosine_similarity(r.flatten(), w.flatten(), dim=0),
    )
    assert audit(d, r, rows, cols)["passes"]
    assert not audit(d, r * 1.2, rows, cols)["passes"]
    assert block_svd(d, rows, cols)[-1] == 4


def test_shared_generator_reproducible_sequence():
    d = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    rows, cols = changed_block(d)
    g = torch.Generator().manual_seed(4)
    first = rotate_block(d, rows, cols, 0, g)
    second = rotate_block(d, rows, cols, 0, g)
    g = torch.Generator().manual_seed(4)
    assert torch.equal(first, rotate_block(d, rows, cols, 999, g))
    assert torch.equal(second, rotate_block(d, rows, cols, 999, g))
    assert torch.equal(first, second)


def test_lora_rounding_and_scaling():
    a = torch.tensor([[1.003, 2.007]])
    b = torch.tensor([[1.0], [2.0]])
    d = b @ a
    assert lora_factors_reproduce(d, a, b, 1.0)
    assert not lora_factors_reproduce(d.to(torch.bfloat16).float(), a, b, 1.0)
    w = torch.ones(2, 2, dtype=torch.bfloat16)
    new, eff = apply_and_round(w, d)
    assert torch.equal(eff, realised_increment(w, new))
    assert all(torch.equal(x, y) for x, y in zip(scaled_variant(w, d, 1), (new, eff)))


def test_exclusions():
    assert exclusion_reason(torch.zeros(2), None, None) == "non_2d"
    d = torch.zeros(2, 2)
    rows, cols = changed_block(d)
    assert exclusion_reason(d, rows, cols) == "zero_block"
