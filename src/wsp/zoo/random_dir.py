"""Control edit family: project a unit direction out of chosen weight matrices (classic abliteration kernel)."""

from __future__ import annotations

import torch


def random_unit_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(dim, generator=g)
    return d / d.norm()


@torch.no_grad()
def ablate_direction(W: torch.Tensor, d: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """W' = W - weight * (d dᵀ) W  for W: (d_model, in) writing into the residual stream along rows.
    Afterwards dᵀ W' ≈ (1-weight) dᵀ W. Returns a new tensor (dtype preserved)."""
    d = d.to(W.dtype) / d.norm()
    proj = torch.outer(d, d @ W.float().to(W.dtype))
    return W - weight * proj
