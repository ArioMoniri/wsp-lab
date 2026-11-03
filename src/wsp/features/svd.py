"""φ_svd: W2T-style canonical tokens for low-rank updates, generalised to full deltas by truncation.

A LoRA update ΔW = B A (B: out×r, A: r×in) is invariant to (B G, G⁻¹ A). W2T canonicalises by QR then SVD so
every factorisation maps to the same (U, Σ, Vᵀ). We produce rank-wise tokens [σ_k, proj(u_k), proj(v_k)].
"""

from __future__ import annotations

import torch


def _proj(dim_in: int, dim_out: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + dim_in * 7919)
    return torch.randn(dim_in, dim_out, generator=g) / dim_out**0.5


def _fix_sign(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # resolve (u,v) ~ (-u,-v): make the largest-|.| entry of each u_k positive
    idx = u.abs().argmax(dim=0)
    sgn = torch.sign(u[idx, torch.arange(u.shape[1])])
    sgn[sgn == 0] = 1
    return u * sgn, v * sgn[None, :]


@torch.no_grad()
def canonical_lora_tokens(
    B: torch.Tensor, A: torch.Tensor, proj_dim: int = 64, seed: int = 0
) -> torch.Tensor:
    """(r, 1 + 2*proj_dim) tokens, identical for all factorisations of the same B@A."""
    B = B.float()
    A = A.float()
    qb, rb = torch.linalg.qr(B)  # B = qb rb
    qa, ra = torch.linalg.qr(A.T)  # A^T = qa ra  -> A = ra^T qa^T
    core = rb @ ra.T  # r×r
    uc, s, vhc = torch.linalg.svd(core)
    U = qb @ uc  # out×r
    V = qa @ vhc.T  # in×r
    U, V = _fix_sign(U, V)
    pu = _proj(U.shape[0], proj_dim, seed)
    pv = _proj(V.shape[0], proj_dim, seed + 1)
    return torch.cat([s[:, None], U.T @ pu, V.T @ pv], dim=1)


@torch.no_grad()
def truncated_delta_tokens(
    delta: torch.Tensor, rank: int = 8, proj_dim: int = 64, seed: int = 0
) -> torch.Tensor:
    """Same token format for an arbitrary delta via truncated SVD (rank r)."""
    d = delta.float()
    if d.ndim != 2:
        d = d.reshape(d.shape[0], -1)
    r = min(rank, min(d.shape))
    U, s, Vh = torch.linalg.svd(d, full_matrices=False)
    U, V = _fix_sign(U[:, :r], Vh[:r].T)
    pu = _proj(U.shape[0], proj_dim, seed)
    pv = _proj(V.shape[0], proj_dim, seed + 1)
    tok = torch.cat([s[:r, None], U.T @ pu, V.T @ pv], dim=1)
    if r < rank:
        tok = torch.nn.functional.pad(tok, (0, 0, 0, rank - r))
    return tok
