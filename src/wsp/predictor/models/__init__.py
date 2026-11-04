"""P4 model factory. Hierarchical attention bounds memory for full production tensor sets."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wsp.features.pool import TYPES
from wsp.predictor.ssm_torch import TrajectoryPredictor


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_rank(z):
    """Entropy effective rank of centered latents; zero for constant representations."""
    z = z.detach().reshape(-1, z.shape[-1]).float()
    s = torch.linalg.svdvals(z - z.mean(0))
    if s.sum() < 1e-8:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def variance_loss(z):
    return F.relu(1 - (z.var(0, unbiased=False) + 1e-4).sqrt()).mean()


def transformer(d, layers=2, dropout=0, extra_ff=0):
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d,
            4,
            dim_feedforward=4 * d + extra_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        ),
        layers,
        enable_nested_tensor=False,
    )


class TensorEncoder(nn.Module):
    def __init__(self, token_dim, n_layers, d, attention_layers=2, dropout=0):
        super().__init__()
        self.proj = nn.Linear(token_dim, d)
        self.types = nn.Embedding(len(TYPES), d)
        self.layers = nn.Embedding(n_layers, d)
        self.attention = transformer(d, attention_layers, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        x = self.proj(b["tokens"]) + self.types(b["types"]) + self.layers(b["layers"])
        shape = x.shape
        mask = b["token_mask"].reshape(-1, shape[2])
        # Only real edits enter the encoder: all-padding attention would yield NaNs.
        valid = b["mask"].reshape(-1)
        h = self.attention(
            x.reshape(-1, shape[2], shape[3])[valid], src_key_padding_mask=~mask[valid]
        )
        pooled = (h * mask[valid, :, None]).sum(1) / mask[valid].sum(1, keepdim=True)
        z = x.new_zeros(shape[0] * shape[1], shape[3])
        z[valid] = self.norm(pooled)
        return z.reshape(shape[0], shape[1], shape[3])


class TokenTransformer(nn.Module):
    def __init__(self, ds, k, d, trajectory=False, matched=False):
        super().__init__()
        self.encoder = TensorEncoder(
            ds.token_dim,
            ds.n_layers,
            d,
            1 if trajectory else 2,
            0.2 if trajectory or matched else 0,
        )
        if matched:
            # +4 feedforward units = 8d+4 parameters, vs M5 positions = 8d.
            layer = self.encoder.attention.layers[-1]
            layer.linear1 = nn.Linear(d, 4 * d + 4)
            layer.linear2 = nn.Linear(4 * d + 4, d)
        self.descriptor_proj = nn.Linear(32, d)
        self.source_dim = ds.stats_dim
        self.source_proj = nn.Linear(ds.stats_dim, d, bias=False)
        self.trajectory = trajectory
        if trajectory:
            self.positions = nn.Embedding(8, d)
            self.edits = transformer(d, 1, 0.2)
        self.head = nn.Linear(d, k)
        self.backend = "torch_attention"

    def forward(self, b):
        z = (
            self.encoder(b)
            + self.descriptor_proj(b["descriptor"])
            + self.source_proj(b["pooled"][..., -self.source_dim :])
        )
        if self.trajectory:
            t = z.shape[1]
            z = z + self.positions(torch.arange(t, device=z.device))
            causal = torch.ones(t, t, device=z.device, dtype=torch.bool).triu(1)
            z = self.edits(z, mask=causal, src_key_padding_mask=~b["mask"])
        return self.head(z)


class MambaTrajectory(nn.Module):
    def __init__(self, ds, k, d, device):
        super().__init__()
        # Backend selection must follow the requested device, not mere GPU availability.
        self.model = TrajectoryPredictor(
            ds.pooled_dim + 32,
            k,
            d_model=d,
            backend="mamba_ssm" if str(device).startswith("cuda") else "loop",
        )
        self.backend = self.model.backend

    def forward(self, b):
        return self.model(torch.cat([b["pooled"], b["descriptor"]], -1), bounded=False)


class ResidualPred(nn.Module):
    """Identity linear predictor plus latent attention with trainable diagonal bias β=10."""

    def __init__(self, dz=32):
        super().__init__()
        self.delta = nn.Linear(dz, dz, bias=False)
        nn.init.zeros_(self.delta.weight)
        self.beta = nn.Parameter(torch.tensor(10.0))

    @property
    def weight(self):
        return (
            torch.eye(self.delta.in_features, device=self.delta.weight.device) + self.delta.weight
        )

    def forward(self, z):
        d = z.shape[-1]
        logits = z.unsqueeze(-1) * z.unsqueeze(-2) / d**0.5
        logits = logits + self.beta * torch.eye(d, device=z.device)
        return (logits.softmax(-1) @ F.linear(z, self.weight).unsqueeze(-1)).squeeze(-1)


class JEPA(nn.Module):
    """M3 adjacent cumulative-delta latent prediction; M6 complementary tabular views.

    Stage one never uses labels. Stage two explicitly freezes encoder and predictor.
    Available features are φ(θ_t−θ₀), not absolute φ(θ_t); this is a documented proxy.
    """

    def __init__(self, ds, k, d, tabular=False):
        super().__init__()
        self.tabular = tabular
        dz = 32 if tabular else 16
        self.encoder = (
            nn.Sequential(nn.Linear(ds.stats_dim, 128), nn.GELU(), nn.Linear(128, dz))
            if tabular
            else TensorEncoder(ds.token_dim, ds.n_layers, d)
        )
        self.latent = nn.Identity() if tabular else nn.Linear(d, dz)
        self.predictor = (
            ResidualPred(dz)
            if tabular
            else nn.Sequential(nn.Linear(dz + 32, 128), nn.GELU(), nn.Linear(128, dz))
        )
        self.head = (
            nn.Sequential(nn.Linear(dz, 64), nn.GELU(), nn.Linear(64, k))
            if tabular
            else nn.Linear(dz, k)
        )
        self.backend = "torch_jepa"
        self.frozen = False

    def encode(self, b):
        return self.encoder(b["stats"]) if self.tabular else self.latent(self.encoder(b))

    def pretrain_loss(self, b):
        z = self.encode(b)
        valid_z = z[b["mask"]]
        if self.tabular:
            # Seeded torch RNG produces fresh complementary column masks each minibatch.
            columns = torch.rand(b["stats"].shape[-1], device=z.device) < 0.5
            context = self.encoder(b["stats"] * columns)[b["mask"]]
            target = self.encoder(b["stats"] * ~columns)[b["mask"]]
            alignment = F.mse_loss(self.predictor(context), target.detach())
            reg = variance_loss(context) + variance_loss(target)
        else:
            pairs = b["mask"][:, 1:] & b["mask"][:, :-1]
            if not pairs.any():
                raise ValueError("M3 pretraining needs at least one adjacent parent-child pair")
            pred = self.predictor(torch.cat([z[:, :-1], b["descriptor"][:, 1:]], -1))
            alignment = F.mse_loss(pred[pairs], z[:, 1:][pairs].detach())
            reg = variance_loss(valid_z)
        return alignment + reg, valid_z

    def freeze_encoder(self):
        self.frozen = True
        for module in (self.encoder, self.latent, self.predictor):
            module.requires_grad_(False)
        self.encoder.eval()

    def forward(self, b):
        if self.frozen:
            with torch.no_grad():
                z = self.encode(b)
        else:
            z = self.encode(b)
        return self.head(z)


def build_model(name, ds, *, seed=0, d_model=256, device="cpu"):
    seed_all(seed)
    k = len(ds.names)
    if name in {"m1", "m1_matched", "m5"}:
        model = TokenTransformer(
            ds, k, d_model, trajectory=name == "m5", matched=name == "m1_matched"
        )
    elif name == "m2":
        model = MambaTrajectory(ds, k, d_model, device)
    elif name in {"m3", "m6"}:
        model = JEPA(ds, k, d_model, tabular=name == "m6")
    else:
        raise ValueError(f"unknown model {name}")
    return model.to(device)
