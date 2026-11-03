"""φ_stats: cheap per-tensor statistics of a weight delta (StatNN-style + spectral)."""

from __future__ import annotations

import torch

DELTA_STAT_NAMES = [
    "fro",
    "rel_fro",
    "cos_prev_new",
    "mean",
    "std",
    "abs_max",
    "frac_pos",
    "frac_nonzero",
    "sv1",
    "sv2",
    "sv3",
    "sv4",
    "sv5",
    "sv6",
    "sv7",
    "sv8",
    "spectral_entropy",
    "stable_rank",
]


def _top_svals(m: torch.Tensor, k: int = 8) -> torch.Tensor:
    if m.ndim != 2:
        m = m.reshape(m.shape[0], -1)
    q = min(k, min(m.shape))
    try:
        s = torch.linalg.svdvals(m.float())[:q]
    except RuntimeError:  # pragma: no cover
        s = torch.zeros(q)
    return torch.nn.functional.pad(s, (0, k - q))


@torch.no_grad()
def delta_stats(prev: torch.Tensor, new: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Return a vector of len(DELTA_STAT_NAMES) describing new - prev."""
    prev = prev.float()
    new = new.float()
    d = new - prev
    fro = torch.linalg.norm(d)
    rel = fro / (torch.linalg.norm(prev) + 1e-12)
    cos = torch.nn.functional.cosine_similarity(prev.flatten(), new.flatten(), dim=0)
    s = _top_svals(d, k)
    p = s / (s.sum() + 1e-12)
    ent = -(p * (p + 1e-12).log()).sum()
    stable_rank = (s**2).sum() / (s[0] ** 2 + 1e-12)
    base = torch.stack(
        [
            fro,
            rel,
            cos,
            d.mean(),
            d.std(),
            d.abs().max(),
            (d > 0).float().mean(),
            (d != 0).float().mean(),
        ]
    )
    return torch.cat([base, s, ent.unsqueeze(0), stable_rank.unsqueeze(0)])
