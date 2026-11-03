import torch

from wsp.features import (
    DELTA_STAT_NAMES,
    canonical_lora_tokens,
    delta_stats,
    truncated_delta_tokens,
)


def test_delta_stats_shape_and_zero_delta():
    W = torch.randn(16, 24)
    s = delta_stats(W, W)
    assert s.shape == (len(DELTA_STAT_NAMES),)
    assert s[0] == 0 and abs(s[2] - 1) < 1e-5  # fro=0, cosine=1


def test_delta_stats_detects_rank_one_edit():
    W = torch.randn(16, 24)
    d = torch.randn(16)
    d /= d.norm()
    W2 = W - torch.outer(d, d @ W)
    s = dict(zip(DELTA_STAT_NAMES, delta_stats(W, W2).tolist()))
    assert s["sv1"] > 0 and s["sv2"] < 1e-4 and s["stable_rank"] < 1.01


def test_lora_tokens_invariant_to_gl_r_reparametrisation():
    torch.manual_seed(0)
    B = torch.randn(40, 4)
    A = torch.randn(4, 30)
    G = torch.randn(4, 4) + 3 * torch.eye(4)
    t1 = canonical_lora_tokens(B, A)
    t2 = canonical_lora_tokens(B @ G, torch.linalg.inv(G) @ A)
    assert t1.shape == (4, 1 + 2 * 64)
    assert torch.allclose(t1, t2, atol=1e-3), (t1 - t2).abs().max()


def test_lora_tokens_match_truncated_delta_tokens():
    torch.manual_seed(1)
    B = torch.randn(40, 3)
    A = torch.randn(3, 30)
    t_lora = canonical_lora_tokens(B, A)
    t_full = truncated_delta_tokens(B @ A, rank=3)
    assert torch.allclose(t_lora[:, 0], t_full[:, 0], atol=1e-3)  # singular values agree
    assert torch.allclose(t_lora.abs(), t_full.abs(), atol=1e-2)


def test_truncated_tokens_pad_to_rank():
    t = truncated_delta_tokens(torch.randn(5, 7), rank=8)
    assert t.shape == (8, 129) and torch.all(t[5:] == 0)
