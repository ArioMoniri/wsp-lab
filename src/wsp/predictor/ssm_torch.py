"""Selective state-space (Mamba-style) predictor over edit trajectories, pure torch fallback.

Input: sequence of per-edit feature vectors x_t (pooled φ_svd ⊕ φ_stats ⊕ edit-descriptor embedding).
Output: predicted performance vector y_t ∈ [0,1]^K after each edit. Uses mamba_ssm.Mamba2 if importable.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .scan import loop_scan, selective_scan

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on CUDA build
    from mamba_ssm import Mamba2 as _Mamba2  # type: ignore

    _BACKEND = "mamba_ssm"
except Exception:  # noqa: BLE001
    _Mamba2 = None
    _BACKEND = "torch"


def ssm_backend() -> str:
    return "mamba_ssm" if _Mamba2 is not None and torch.cuda.is_available() else "loop"


class SelectiveSSMBlock(nn.Module):
    """Minimal selective scan (S6): h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t ; y_t = C_t h_t + D x_t."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, backend: str = "loop"):
        if backend not in {"loop", "chunked"}:
            raise ValueError(f"Unknown scan backend: {backend}")
        super().__init__()
        self.backend = backend
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, d_model)
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = torch.nn.functional.silu(u)
        bcd = self.x_proj(u)
        Bm, Cm, dt = bcd.split([self.A_log.shape[1], self.A_log.shape[1], 1], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt))  # (B,T,d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        scan = loop_scan if self.backend == "loop" else selective_scan
        y = scan(u, dt, A, Bm, Cm, self.D) * torch.nn.functional.silu(z)
        return res + self.out_proj(y)


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba2 block; choose the largest divisor up to 64 per head."""

    def __init__(self, d_model: int, d_state: int, expand: int = 2):
        super().__init__()
        inner = d_model * expand
        headdim = next(h for h in range(min(64, inner), 0, -1) if inner % h == 0)
        self.norm = nn.LayerNorm(d_model)
        self.mixer = _Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_targets: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        backend: str | None = None,
    ):
        super().__init__()
        self.backend = backend or ssm_backend()
        if self.backend == "torch":
            self.backend = "loop"
        if self.backend not in {"loop", "chunked", "mamba_ssm"}:
            raise ValueError(f"Unknown backend: {self.backend}")
        if self.backend == "mamba_ssm" and _Mamba2 is None:
            raise ImportError("mamba_ssm is unavailable")
        self.inp = nn.Linear(d_in, d_model)
        if self.backend == "mamba_ssm" and _Mamba2 is not None:
            self.blocks = nn.ModuleList(
                [MambaBlock(d_model, d_state) for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SelectiveSSMBlock(d_model, d_state, backend=self.backend) for _ in range(n_layers)]
            )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_targets))
        log.info("TrajectoryPredictor backend=%s", self.backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,d_in) -> (B,T,K) in [0,1]
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return torch.sigmoid(self.head(h))
