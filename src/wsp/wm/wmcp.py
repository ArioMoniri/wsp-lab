"""WM-CP: the canonical-path world model on the shared representation R(E) (RULING-wme-v2 A1; A5 "R(E) on 9B").

**Representation R(E)** (:class:`RSpec`). One vector per realised cumulative edit E (relative to theta0), columns
with a declared degree in E: per objective k the field features a_k = <g0, E> (degree 1) and
q_k = 1/2 E^T H0 E (degree 2); degree-1 columns (dictionary coordinates in synthetic worlds; on 9B the chart columns
of the realised delta) and the cumulative norm (degree 1). Hence R(tE) = R(E) * t^degree exactly
(:func:`r_at`), and the field nest is F(E) = a + q (:func:`field`). On 9B, :func:`r_from_contractions` builds R from
``wm.collect.contract`` of the realised cumulative deltas (unsupported q -> 0 and flagged), the response-chart
columns and the realised norms, so WM-CP and every static member read the same R(E_T).

**WM-CP** (:class:`CanonicalPathWM`). The learned residual step operator of D3 integrated along a canonical path:
K(E_T) equal sub-steps of the realised cumulative oriented delta, E_k = (k / K) E_T:

    h_0 = enc(o_0);  h_k = h_{k-1} + f(h_{k-1}, R(E_{k-1}), R(E_k) - R(E_{k-1}))
    y_hat(E_T) = F(E_T) + lambda * y_scale * sum_k rho(h_{k-1}, h_k, R(E_k) - R(E_{k-1}))

with an auxiliary observation decoder dec(h_K) (grounding on the training states' tap observations).
**K rule (frozen; :data:`K_MAX`)**: K = 0 for an exactly zero R; otherwise
K = min(K_MAX, max(1, ceil(norm(E_T) / s0))), s0 = the median cumulative norm of the training states with h = 1
(training components only; recorded in ``k_rule``). The prediction is a function of R(E_T) only, so an exact zero
edit gives exactly 0 (masked sums), alternative decompositions give the identical prediction, and a round trip
(realised E_T = 0) gives exactly 0. C2-S and C2-D therefore hold by construction; C2-I stays a gate (never-emitted
rows have E_T != 0). Shrinkage lambda is selected on a component-level selection split from {0, .25, .5, .75, 1}
(ties to the smaller), so on quadratic dynamics, where the residual target is 0, lambda = 0 and WM-CP equals the
static field map exactly. sigma~ = 1[E != 0] sqrt(sigma_F^2 + (lambda s_ens)^2) as C1.1.

**Static members on R** (:class:`StaticR`): S2 = F(E_T) + lambda y_scale [MLP(R) - MLP(0)] (outcome-only); S3 = the same
with the auxiliary observation head (StatNN-aux; same targets as WM-CP), width sized to WM-CP's parameter count
(+-10 %). S1 is the least-squares F3u on (a, q).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from torch import nn

from wsp.wm import model as M

K_MAX = 16
SHRINK_GRID = M.SHRINK_GRID


@dataclasses.dataclass(frozen=True)
class CPConfig:
    hidden: int = 64
    latent: int = 32
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    obs_weight: float = 1.0
    k_max: int = K_MAX  # K rule: K = min(k_max, max(1, ceil(norm / s0)))
    s0_scale: float = 1.0  # s0 = s0_scale * median training h = 1 norm


@dataclasses.dataclass(frozen=True)
class RSpec:
    names: tuple
    degree: tuple
    n_obj: int  # the first 2 * n_obj columns are a_1..a_K, q_1..q_K
    norm_index: int

    @property
    def dim(self):
        return len(self.names)


def r_at(r, spec: RSpec, t: float) -> np.ndarray:
    return np.asarray(r, float) * float(t) ** np.asarray(spec.degree, float)


def field(r, spec: RSpec) -> np.ndarray:
    r = np.asarray(r, float)
    K = spec.n_obj
    return r[..., :K] + r[..., K : 2 * K]


# ------------------------------------------------------------------ representation builders


def spec_synthetic(world) -> RSpec:
    K, J = world.K, world.J
    names = (
        tuple(f"a{k}" for k in range(K))
        + tuple(f"q{k}" for k in range(K))
        + tuple(f"e{j}" for j in range(J))
        + ("norm",)
    )
    deg = (1,) * K + (2,) * K + (1,) * J + (1,)
    return RSpec(names, deg, K, len(names) - 1)


def r_synthetic(world, E) -> np.ndarray:
    E = np.asarray(E, float)
    g, H = world._g0.numpy(), world._H0.numpy()
    a = g @ E
    q = 0.5 * np.einsum("j,kji,i->k", E, H, E)
    Ed = world.E.reshape(world.J, -1).numpy()
    return np.concatenate([a, q, E, [np.linalg.norm(Ed.T @ E)]])


def state_from_traj(world, t, h, comp=None) -> dict:
    E = t["x"][h, : world.J]
    return {
        "r": r_synthetic(world, E),
        "y": np.asarray(t["y"][h], float),
        "obs": np.asarray(t["obs"][h], float),
        "h": int(h),
        "comp": comp,
    }


def spec_9b(objectives, chart_columns, norm_groups) -> RSpec:
    K = len(objectives)
    names = (
        tuple(f"a:{o}" for o in objectives)
        + tuple(f"q:{o}" for o in objectives)
        + tuple(f"chart:{c}" for c in chart_columns)
        + tuple(f"norm:{g}" for g in norm_groups)
        + ("norm",)
    )
    deg = (1,) * K + (2,) * K + (1,) * (len(chart_columns) + len(norm_groups) + 1)
    return RSpec(names, deg, K, len(names) - 1)


def r_from_contractions(
    spec: RSpec,
    objectives,
    contraction: dict,
    chart=None,
    group_norms=None,
    total_norm: float = 0.0,
) -> tuple[np.ndarray, list]:
    """R(E_T) on 9B from ``wm.collect.contract(field, realised cumulative deltas)``. Unsupported q (None) enters as
    0 and is returned in the flag list (never silently)."""
    a, q, flags = [], [], []
    for o in objectives:
        rows = contraction[o]
        a.append(sum(v["signed"] for v in rows.values()))
        qs = [v["q"] for v in rows.values()]
        if any(x is None for x in qs):
            flags.append(o)
        q.append(sum(x for x in qs if x is not None))
    chart = list(chart or [])
    gn = list(group_norms or [])
    r = np.array(a + q + chart + gn + [total_norm], float)
    if len(r) != spec.dim:
        raise ValueError(f"R has {len(r)} columns, spec {spec.dim}")
    return r, flags


# ------------------------------------------------------------------ nets


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class _Norm:
    def __init__(self, arr):
        arr = np.asarray(arr, float).reshape(-1, np.asarray(arr).shape[-1])
        self.mu, self.sd = arr.mean(0), arr.std(0) + 1e-6

    def __call__(self, a, center=True):
        return ((a - self.mu) if center else a) / self.sd


class _CPNet(nn.Module):
    def __init__(self, d_r, d_obs, K, cfg):
        super().__init__()
        L, Hd = cfg.latent, cfg.hidden
        self.enc = _mlp(d_obs, Hd, L)
        self.step = _mlp(L + 2 * d_r, Hd, L)
        self.read = _mlp(2 * L + d_r, Hd, K)
        self.dec = _mlp(L, Hd, d_obs)


class _StaticNet(nn.Module):
    def __init__(self, d_r, d_obs, K, width, aux):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_r, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.out = nn.Linear(width, K)
        self.dec = nn.Linear(width, d_obs) if aux else None


def _count(net):
    return sum(p.numel() for p in net.parameters())


# ------------------------------------------------------------------ shared fitting scaffold


class _Base:
    def __init__(self, cfg: CPConfig, spec: RSpec, base_obs, n_seeds=5):
        self.cfg, self.spec, self.n_seeds = cfg, spec, int(n_seeds)
        self.base_obs = np.asarray(base_obs, float)
        self.shrink = 1.0
        self.member_samples = None

    def _prep(self, states):
        R = np.stack([s["r"] for s in states])
        Y = np.stack([s["y"] for s in states])
        O = np.stack([s["obs"] for s in states])
        return R, Y, O

    def _setup(self, fit_states):
        R, Y, O = self._prep(fit_states)
        self.rnorm = _Norm(np.vstack([R, np.zeros((1, R.shape[1]))]))
        self.onorm = _Norm(np.vstack([O, self.base_obs[None]]))
        self.y_scale = (Y - field(R, self.spec)).std(0) + 1e-9

    def fit(self, states, components=None):
        if any(s["h"] > 2 for s in states):
            raise ValueError("training states must have h <= 2")
        fit, val, samples, _ = M.split_and_bootstrap(list(states), components, self.n_seeds)
        self._setup(fit)
        self._fit_members(fit, samples)
        if components is not None:
            pos = {id(s): i for i, s in enumerate(states)}
            self.member_samples = [[pos[id(fit[i])] for i in s] for s in samples]
        self.shrink = self._select_shrink(val)
        ref = val or fit
        Rv, Yv, _ = self._prep(ref)
        self.sigma_floor = np.sqrt(((Yv - field(Rv, self.spec)) ** 2).mean(0))
        return self

    def _select_shrink(self, val):
        if not val:
            return 1.0
        R, Y, _ = self._prep(val)
        f = field(R, self.spec)
        res = self._raw(R).mean(0)
        errs = {lam: float(np.abs(f + lam * res - Y).mean()) for lam in SHRINK_GRID}
        return _tie_to_field(errs, getattr(self, "shrink_tie_rel", 0.0))

    def predict_batch(self, R):
        R = np.atleast_2d(np.asarray(R, float))
        f = field(R, self.spec)
        raw = self._raw(R)  # [M, N, K]
        gate = np.any(R != 0, axis=1, keepdims=True).astype(float)
        mean = f + self.shrink * raw.mean(0)
        spread = raw.std(0)
        sigma = gate * np.sqrt(self.sigma_floor**2 + (self.shrink * spread) ** 2)
        return {"mean": mean, "sigma": sigma, "members": f[None] + self.shrink * raw}

    def predict(self, r):
        out = self.predict_batch(np.asarray(r, float)[None])
        return out["mean"][0], out["sigma"][0]


# ------------------------------------------------------------------ WM-CP


class CanonicalPathWM(_Base):
    def _setup(self, fit_states):
        super()._setup(fit_states)
        R = np.stack([s["r"] for s in fit_states])
        n = R[:, self.spec.norm_index]
        h1 = [
            s["r"][self.spec.norm_index]
            for s in fit_states
            if s["h"] == 1 and s["r"][self.spec.norm_index] > 0
        ]
        s0 = self.cfg.s0_scale * (float(np.median(h1)) if h1 else float(np.median(n[n > 0])))
        self.k_rule = {
            "rule": "K = 0 if R == 0 else min(k_max, max(1, ceil(norm(E_T) / s0)))",
            "s0": s0,
            "k_max": self.cfg.k_max,
            "s0_scale": self.cfg.s0_scale,
            "s0_source": "s0_scale x median cumulative norm of training h = 1 states",
        }
        steps = []
        for r in R:
            k = self.k_of(np.array([r[self.spec.norm_index]]))
            if k:
                steps.append(r_at(r, self.spec, 1.0 / k))
        self.dnorm = _Norm(np.vstack(steps + [np.zeros(R.shape[1])]))

    def k_of(self, norm) -> int:
        x = float(np.asarray(norm).ravel()[0])
        if x <= 0:
            return 0
        return int(min(self.k_rule["k_max"], max(1, math.ceil(x / self.k_rule["s0"] - 1e-12))))

    def _path(self, R):
        """[N, K_MAX + 1, d] sub-step representations and the [N, K_MAX] step mask."""
        N, d = R.shape
        KM = self.k_rule["k_max"]
        P = np.zeros((N, KM + 1, d))
        mask = np.zeros((N, KM))
        deg = np.asarray(self.spec.degree, float)
        for i, r in enumerate(R):
            k = 0 if not np.any(r != 0) else self.k_of(np.array([r[self.spec.norm_index]]))
            for j in range(1, k + 1):
                P[i, j] = r * (j / k) ** deg
            P[i, k + 1 :] = P[i, k]
            mask[i, :k] = 1.0
        return P, mask

    def _tensors(self, R):
        P, mask = self._path(R)
        cur = torch.as_tensor(self.rnorm(P[:, :-1]), dtype=torch.float32)
        dlt = torch.as_tensor(self.dnorm(P[:, 1:] - P[:, :-1], center=False), dtype=torch.float32)
        return cur, dlt, torch.as_tensor(mask, dtype=torch.float32)

    def _rollout(self, net, cur, dlt, mask):
        N = cur.shape[0]
        o0 = torch.as_tensor(self.onorm(self.base_obs)[None], dtype=torch.float32).expand(N, -1)
        h = net.enc(o0)
        res = torch.zeros(N, self.spec.n_obj)
        for k in range(mask.shape[1]):
            m = mask[:, k : k + 1]
            if not bool(m.any()):
                break
            hn = h + m * net.step(torch.cat([h, cur[:, k], dlt[:, k]], -1))
            res = res + m * net.read(torch.cat([h, hn, dlt[:, k]], -1))
            h = hn
        return res, net.dec(h)

    def _fit_members(self, fit, samples):
        R, Y, O = self._prep(fit)
        cur, dlt, mask = self._tensors(R)
        T = torch.as_tensor((Y - field(R, self.spec)) / self.y_scale, dtype=torch.float32)
        OT = torch.as_tensor(self.onorm(O), dtype=torch.float32)
        self.members = []
        for s, idx in enumerate(samples):
            idx = torch.as_tensor(idx)
            torch.manual_seed(1000 + s)
            net = _CPNet(R.shape[1], O.shape[1], self.spec.n_obj, self.cfg).float()
            opt = torch.optim.AdamW(
                net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
            for _ in range(self.cfg.epochs):
                res, dec = self._rollout(net, cur[idx], dlt[idx], mask[idx])
                loss = ((res - T[idx]) ** 2).mean() + self.cfg.obs_weight * (
                    (dec - OT[idx]) ** 2
                ).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.members.append(net)

    @torch.no_grad()
    def _raw(self, R):
        cur, dlt, mask = self._tensors(np.atleast_2d(R))
        return (
            np.stack([self._rollout(n, cur, dlt, mask)[0].numpy() for n in self.members])
            * self.y_scale
        )


# ------------------------------------------------------------------ static members on R


class StaticR(_Base):
    def __init__(self, cfg, spec, base_obs, n_seeds=5, aux=False, width=None):
        super().__init__(cfg, spec, base_obs, n_seeds)
        self.aux, self.width = bool(aux), width or cfg.hidden

    def size_to(self, target: int, d_obs: int) -> int:
        best = None
        for w in range(4, 1025):
            c = _count(_StaticNet(self.spec.dim, d_obs, self.spec.n_obj, w, self.aux))
            if best is None or abs(c - target) < abs(best[1] - target):
                best = (w, c)
        self.width = best[0]
        return best[1]

    def _fit_members(self, fit, samples):
        R, Y, O = self._prep(fit)
        X = torch.as_tensor(self.rnorm(R), dtype=torch.float32)
        X0 = torch.as_tensor(self.rnorm(np.zeros((1, R.shape[1]))), dtype=torch.float32)
        T = torch.as_tensor((Y - field(R, self.spec)) / self.y_scale, dtype=torch.float32)
        OT = torch.as_tensor(self.onorm(O), dtype=torch.float32)
        self.members = []
        for s, idx in enumerate(samples):
            idx = torch.as_tensor(idx)
            torch.manual_seed(1000 + s)
            net = _StaticNet(R.shape[1], O.shape[1], self.spec.n_obj, self.width, self.aux).float()
            opt = torch.optim.AdamW(
                net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
            for _ in range(self.cfg.epochs):
                b = net.body(X[idx])
                loss = ((net.out(b) - net.out(net.body(X0)) - T[idx]) ** 2).mean()
                if self.aux:
                    loss = loss + self.cfg.obs_weight * ((net.dec(b) - OT[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.members.append(net)

    @torch.no_grad()
    def _raw(self, R):
        R = np.atleast_2d(R)
        X = torch.as_tensor(self.rnorm(R), dtype=torch.float32)
        X0 = torch.as_tensor(self.rnorm(np.zeros((1, R.shape[1]))), dtype=torch.float32).expand_as(
            X
        )
        zero = torch.as_tensor(~np.any(R != 0, axis=1, keepdims=True))
        out = [
            torch.where(zero, 0.0, n.out(n.body(X)) - n.out(n.body(X0))).numpy()
            for n in self.members
        ]
        return np.stack(out) * self.y_scale


class F3uR:
    """S1: per-objective least squares y = c1 a + c2 q on training h <= 2 states (zero at the zero edit)."""

    def __init__(self, spec: RSpec):
        self.spec = spec
        self.beta = np.ones((spec.n_obj, 2))

    def fit(self, states, components=None):
        if any(s["h"] > 2 for s in states):
            raise ValueError("training states must have h <= 2")
        R = np.stack([s["r"] for s in states])
        Y = np.stack([s["y"] for s in states])
        K = self.spec.n_obj
        for k in range(K):
            X = np.stack([R[:, k], R[:, K + k]], 1)
            self.beta[k] = np.linalg.lstsq(X, Y[:, k], rcond=None)[0]
        return self

    def predict(self, r):
        r = np.asarray(r, float)
        K = self.spec.n_obj
        return r[:K] * self.beta[:, 0] + r[K : 2 * K] * self.beta[:, 1], None


def _tie_to_field(errs: dict, tie_rel: float = 0.0) -> float:
    """The smallest shrinkage whose loss is within ``tie_rel`` (relative) of the best: ties go toward the exact field.
    ``tie_rel = 0`` is the original rule (exact ties to the smaller shrinkage)."""
    best = min(errs.values())
    return min(lam for lam, e in errs.items() if e <= best * (1 + tie_rel) + 1e-15)


class ShrunkF3uR(F3uR):
    """S1 under SYNTHESIS-r11 R6 (flag 1): the F3u refit shrunk toward the raw field F = a + q,
    prediction F + lam (F3u - F), with lam on :data:`SHRINK_GRID` chosen on the same component-level selection split,
    with the same loss (mean raw |error|) and the same tie rule as WM-CP; the refit uses the fit part only, as WM-CP's
    members do."""

    def __init__(self, spec: RSpec, tie_rel: float = 0.0):
        super().__init__(spec)
        self.shrink, self.shrink_tie_rel = 1.0, float(tie_rel)

    def fit(self, states, components=None):
        if any(s["h"] > 2 for s in states):
            raise ValueError("training states must have h <= 2")
        states = list(states)
        if components is None:
            super().fit(states)
            self.shrink = 1.0
            return self
        fit, val, _samples, _ = M.split_and_bootstrap(states, components, 1)
        super().fit(fit)
        if val:
            R = np.stack([s["r"] for s in val])
            Y = np.stack([s["y"] for s in val])
            f = field(R, self.spec)
            res = np.stack([super(ShrunkF3uR, self).predict(r)[0] for r in R]) - f
            errs = {lam: float(np.abs(f + lam * res - Y).mean()) for lam in SHRINK_GRID}
            self.shrink = _tie_to_field(errs, self.shrink_tie_rel)
        return self

    def predict(self, r):
        r = np.asarray(r, float)
        f = field(r[None], self.spec)[0]
        return f + self.shrink * (super().predict(r)[0] - f), None


def param_count(model) -> int:
    return _count(model.members[0])


def param_count_untrained(static: StaticR, ref: CanonicalPathWM) -> int:
    return static.size_to(param_count(ref), len(ref.base_obs))


# ------------------------------------------------------------------ WM-E v0 on R (descriptive arm; RULING-wme-v2 A1)


def v0_field(spec: RSpec, theta_hash: str | None = None):
    """A field artefact for ``wm.model`` whose nest on R-coordinates is exactly F = a + q: g = [I | I | 0], H = 0.
    ``theta_hash`` is the base hash of the weights R was computed at (default: the synthetic base)."""
    from wsp.wm.access import Artefact, Tier
    from wsp.wm.synthetic import BASE_HASH as _SYN

    BASE_HASH = theta_hash or _SYN

    K, d = spec.n_obj, spec.dim
    g = np.zeros((K, d))
    g[:, :K] = np.eye(K)
    g[:, K : 2 * K] = np.eye(K)
    return Artefact("field", BASE_HASH, Tier.R0, {"g": g, "H": np.zeros((K, d, d))})


def v0_base_obs(world):
    return world.base_observation()


def v0_trajectory(world, spec: RSpec, Es, y, obs) -> dict:
    """WM-E v0 trajectory whose coordinates are R(E_t) (the history-dependent descriptive arm), from synthetic
    cumulative edits E_t (or R rows); see :func:`v0_trajectory_r`."""
    Rs = np.stack([r_synthetic(world, E) if len(E) == world.J else E for E in Es])
    if Rs.shape[1] != spec.dim:
        Rs = np.stack([np.concatenate([r[:-1], [0.0], r[-1:]]) for r in Rs])
    return v0_trajectory_r(spec, Rs, y, obs)


def v0_trajectory_r(spec: RSpec, Rs, y=None, obs=None) -> dict:
    """WM-E v0 (``wm.model``) trajectory on the shared representation, from R rows R(E_0 = 0), ..., R(E_T) (on 9B
    from :func:`r_from_contractions` of each realised cumulative prefix): cumulative R_t, step-local R_t - R_{t-1},
    position, signed overlap cos(dR_t, R_{t-1}) and the cumulative norm column. ``y`` defaults to zeros (unknown,
    planner use); a 1-D ``obs`` is broadcast over the horizon."""
    Rs = np.asarray(Rs, float)
    if Rs.ndim != 2 or Rs.shape[1] != spec.dim:
        raise ValueError(f"R rows must be (T + 1) x {spec.dim}")
    T, d = len(Rs) - 1, spec.dim
    x = np.zeros((T + 1, 2 * d + 3))
    for t in range(1, T + 1):
        D = Rs[t] - Rs[t - 1]
        den = np.linalg.norm(Rs[t - 1]) * np.linalg.norm(D)
        x[t, :d], x[t, d : 2 * d], x[t, -3] = Rs[t], D, t
        x[t, -2] = float(Rs[t - 1] @ D / den) if den > 0 else 0.0
        x[t, -1] = Rs[t, spec.norm_index]
    y = np.zeros((T + 1, spec.n_obj)) if y is None else np.asarray(y)[: T + 1]
    obs = np.zeros((T + 1, 1)) if obs is None else np.asarray(obs, float)
    obs = np.broadcast_to(obs, (T + 1, obs.shape[-1])).copy() if obs.ndim == 1 else obs[: T + 1]
    return {"horizon": T, "e": np.diff(Rs, axis=0), "x": x, "y": y, "obs": obs}
