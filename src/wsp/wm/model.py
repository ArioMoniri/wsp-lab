"""WM-E v0 world model (SYNTHESIS-r8 D3), with the same-input static baseline and the field-only predictor.

Per objective k and horizon t (default ``slope_transport=True``):

    h_0     = enc(o_0)                                                          (base observation: R0)
    h_{t}   = h_{t-1} + gate(e_t) f(h_{t-1}, xl_t, e_t),  gate(e) = 1[e != 0]   (local step operator)
    A_hat_t = A0 + H0 E_t + a_s [C(h_t) - C(h_0)]                               (grounded slope; field-nested)
    y_hat_t = sum_{s<=t} { 1/2 (A_hat_{s-1} + A_hat_s) . e_s + gate(e_s) rho(h_{s-1}, h_s, xl_s, e_s) }

* **Field nest.** With the learned parts at zero (C = const, rho = 0) the trapezoid sum of the base-field slopes
  ``A0 + H0 E`` equals the raw field ``F(E) = <g0, E> + 1/2 E^T H0 E`` exactly, so the model is nested on the field
  and is exact for quadratic dynamics. Both arms use this raw (unit-coefficient) nest; r7's least-squares F3u
  coefficients are not refitted here, because an outcome-fitted nest differs between arms under the null.
  ``slope_transport=False`` gives the D3 readout form ``F(E_t) + s [g(h_t, xl_t) - g(h_0, 0)]`` (ablation).
* **Zero at the zero edit.** Every learned term is gated by ``1[e != 0]`` or differenced against h_0, so a zero
  action sequence predicts exactly 0 with exactly 0 ensemble spread.
* **Exact bookkeeping** x_t (``synthetic.bookkeeping`` / ``actions.step_tokens``) is an input, never a target. The
  step operator and the per-step correction receive its step-local part ``xl_t`` (realised D_t and the signed overlap
  with E_{t-1}); cumulative E_t enters through the field nest. No head predicts bookkeeping.
* **Grounding.** ``dec(h_t)`` predicts the observation vector at theta_t (tap sketches, per-item losses, action
  derivatives A_t) and ``C(h_t)`` the slope departure ``A_t - A0 - H0 E_t``; the targets are external and fixed.
* **Training.** Whole-trajectory unrolls on horizons <= ``max_train_horizon`` (2), loss at every horizon plus the
  observation and slope losses; scheduled sampling replaces the rolled state by ``enc(o_t)`` with probability p
  annealed from 1 to ``p_min``; AdamW. 20% of the training trajectories (horizons <= 2) select a shrinkage
  ``lambda`` of the learned part over the nest from one grid shared by every arm.
* **Uncertainty (SYNTHESIS-r9 C1).** A 5-member ensemble; ``fit(trajs, components=...)`` resamples members by
  source-closure component and splits selection by component (``wm.uncertainty``). sigma is
  ``gate(E_t) sqrt(sigma_F^2 + (lambda sigma_ens)^2)``: the field-residual floor sigma_F keeps sigma > 0 off the
  zero edit even at lambda = 0 (r9 W4 code finding 1). The static arm gets the identical treatment.

The static baseline receives the same cumulative edit: the same raw-field nest plus an MLP on the cumulative-E
bookkeeping, the same outcome labels and the same selection rule, without intermediate observations.
``predict`` is R0-guarded (``wm.access``): it reads the base observation and the action recipes only.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import nn

from wsp.wm import uncertainty as U
from wsp.wm.access import R0Guard, r0_only


@dataclasses.dataclass(frozen=True)
class WMConfig:
    hidden: int = 64
    latent: int = 32
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    obs_weight: float = 1.0
    p_min: float = 0.0
    max_train_horizon: int = 2
    anneal_frac: float = 0.6
    slope_transport: bool = True
    slope_rule: str = "trapezoid"  # or "left": A_hat(h_{t-1}) e_t + 1/2 e_t^T H0 e_t
    static_state: bool = (
        False  # MatchedStatic: the state is a function of (o_0, E_t) only (Board A cond. 4)
    )


def aux_ablation(cfg: WMConfig) -> WMConfig:
    """The auxiliary-target ablation (Board A condition 4): no observation or slope grounding loss."""
    return dataclasses.replace(cfg, obs_weight=0.0)


def teacher_forcing_prob(
    epoch: int, epochs: int, p_min: float = 0.0, anneal_frac: float = 0.6
) -> float:
    span = max(1, int(anneal_frac * epochs))
    return float(max(p_min, 1.0 - epoch / span)) if epoch < span else float(p_min)


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


# ------------------------------------------------------------------ F3u


class F3u:
    """Calibrated field a/q on the cumulative edit; zero at the zero edit."""

    def __init__(self, field_payload):
        self.g = np.asarray(field_payload["g"], dtype=np.float64)  # K x J
        self.H = np.asarray(field_payload["H"], dtype=np.float64)  # K x J x J
        self.beta = np.ones((self.g.shape[0], 2))

    def features(self, E):
        E = np.asarray(E, dtype=np.float64)
        a = E @ self.g.T
        q = 0.5 * np.einsum("...j,kji,...i->...k", E, self.H, E)
        return a, q

    def fit(self, E, y):
        a, q = self.features(E)
        for k in range(self.g.shape[0]):
            X = np.stack([a[:, k], q[:, k]], 1)
            self.beta[k] = np.linalg.lstsq(X, y[:, k], rcond=None)[0]
        return self

    def __call__(self, E):
        a, q = self.features(E)
        return a * self.beta[:, 0] + q * self.beta[:, 1]


# ------------------------------------------------------------------ data


def _split_x(x, J):
    """Step-local bookkeeping: realised D_t coordinates and the signed overlap cos(D_t, E_{t-1})."""
    return np.concatenate([x[..., J : 2 * J], x[..., -2:-1]], -1)


def batch(trajs):
    hs = {t["horizon"] for t in trajs}
    if len(hs) != 1:
        raise ValueError("a batch holds one horizon")
    J = trajs[0]["e"].shape[1]
    st = lambda k: np.stack([t[k] for t in trajs])
    x = st("x")
    return {
        "E": x[..., :J],
        "xl": _split_x(x, J),
        "xs": np.concatenate([x[..., :J], x[..., -1:]], -1),
        "e": st("e"),
        "y": st("y"),
        "obs": st("obs"),
        "T": hs.pop(),
    }


class _Norm:
    def __init__(self, arr):
        flat = arr.reshape(-1, arr.shape[-1])
        self.mu = flat.mean(0)
        self.sd = flat.std(0) + 1e-6

    def __call__(self, a):
        return (a - self.mu) / self.sd


SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def selection_split(trajs, frac=0.2, seed=20261116):
    """Deterministic split of training trajectories into fit and selection (validation) parts."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajs))
    n_val = max(1, round(frac * len(trajs))) if len(trajs) > 4 else 0
    val = set(idx[:n_val].tolist())
    return [t for i, t in enumerate(trajs) if i not in val], [
        t for i, t in enumerate(trajs) if i in val
    ]


def select_shrink(predict_fn, field_fn, val):
    """Shrinkage lambda of the learned part over the raw-field nest, y = F(E) + lambda (y_model - F(E)), chosen
    on the selection set (horizons <= 2 only) from the same grid for every arm; ties go to the smaller lambda."""
    if not val:
        return 1.0
    preds = [(predict_fn(t), field_fn(t), t) for t in val]
    errs = {
        lam: float(
            np.mean(
                [np.abs(f[1:] + lam * (p[1:] - f[1:]) - t["y"][1:]).mean() for p, f, t in preds]
            )
        )
        for lam in SHRINK_GRID
    }
    return min(errs, key=lambda lam: (errs[lam], lam))


def split_and_bootstrap(trajs, components, n_members, seed=20261117):
    """(fit trajectories, selection trajectories, per-member samples of fit-trajectory indices, val indices).

    Without ``components`` (pilot behaviour): the row-level selection split and every member on all fit rows.
    With ``components``: a component-level selection split and per-member component bootstraps (r9 C1.3)."""
    if components is None:
        fit, val = selection_split(trajs)
        ids = {id(t): i for i, t in enumerate(trajs)}
        return fit, val, [list(range(len(fit)))] * n_members, [ids[id(t)] for t in val]
    if len(components) != len(trajs):
        raise ValueError("one component id per trajectory")
    fit_i, val_i = U.component_selection_split(components)
    fit = [trajs[i] for i in fit_i]
    samples = U.component_bootstrap([components[i] for i in fit_i], n_members, seed)
    return fit, [trajs[i] for i in val_i], samples, val_i


def _by_horizon(trajs):
    by_h = {}
    for t in trajs:
        by_h.setdefault(t["horizon"], []).append(t)
    return [batch(v) for _, v in sorted(by_h.items())]


# ------------------------------------------------------------------ world model member


class WMNet(nn.Module):
    def __init__(self, d_obs, d_xl, d_e, K, d_slope, cfg: WMConfig):
        super().__init__()
        L, H = cfg.latent, cfg.hidden
        self.enc = _mlp(d_obs, H, L)
        self.step = _mlp(L + d_xl + d_e, H, L)
        self.dec = _mlp(L, H, d_obs)
        # slope transport: a per-step local correction rho(h_{t-1}, h_t, xl_t, e_t); otherwise a state readout
        self.readout = _mlp(2 * L + d_xl + d_e, H, K) if d_slope else _mlp(L + d_xl, H, K)
        self.slope = _mlp(L, H, d_slope) if d_slope else None


class Member:
    """One ensemble member. Units: outcomes in raw units; heads work in standardised units."""

    def __init__(self, cfg, seed, ctx=None):
        self.cfg, self.seed, self.ctx = cfg, seed, ctx
        torch.manual_seed(seed)
        self.net = WMNet(*ctx["dims"], cfg).float() if ctx else None

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    @staticmethod
    def _gate(e):
        return (e != 0).any(-1, keepdim=True).to(e.dtype)

    def _prep(self, b):
        """Tensors of one batch under the shared normalisation context (cached on the batch dict)."""
        if "_prep" in b:
            return b["_prep"]
        ctx, f = self.ctx, self.ctx["f3u"]
        live = np.abs(b["xl"]).sum(-1, keepdims=True) > 0
        p = {
            "obs": self._t(ctx["obs_norm"](b["obs"])),
            "xl": self._t(ctx["xl_norm"](b["xl"]) * live),
            "e_raw": self._t(b["e"]),
            "y": self._t(b["y"]),
            "field_slopes": self._t(f.g.T + np.einsum("kji,nti->ntjk", f.H, b["E"])),  # A0 + H0 E_t
            "f3u": self._t(f(b["E"])),
            "y_scale": self._t(ctx["y_scale"]),
            "step_scale": self._t(ctx["step_scale"]),
        }
        if self.cfg.static_state:
            p["xs"] = self._t(ctx["xs_norm"](b["xs"]) * live)
            p["E_e"] = self._t(b["E"]) / ctx["e_scale"]
        p["e"] = p["e_raw"] / ctx["e_scale"]
        if self.cfg.slope_transport:
            p["a_scale"] = self._t(ctx["a_scale"])
            p["slope_target"] = [None] + [
                self._t(ctx["slope_target"](b, t)) for t in range(1, b["T"] + 1)
            ]
        b["_prep"] = p
        return p

    def _static_rollout(self, b, p):
        """MatchedStatic: h_t = h_0 + gate(E_t) step(h_0, xs_t, E_t); y_t = F(E_t) + gate readout(h_0, h_t, xs_t,
        E_t); the same heads, widths and grounding targets as WM-E, no recurrence and no path sum."""
        ctx, net, cfg = self.ctx, self.net, self.cfg
        obs, xs, E = p["obs"], p["xs"], p["E_e"]
        N, T = obs.shape[0], b["T"]
        K = len(ctx["y_scale"])
        h0 = net.enc(obs[:, 0])
        c0 = net.slope(h0) if cfg.slope_transport else None
        g0 = (
            None
            if cfg.slope_transport
            else net.readout(torch.cat([h0, torch.zeros_like(xs[:, 0])], -1))
        )
        ys, dec, cor = [torch.zeros(N, K, dtype=h0.dtype)], [net.dec(h0)], [None]
        for t in range(1, T + 1):
            gate = self._gate(E[:, t])
            h = h0 + gate * net.step(torch.cat([h0, xs[:, t], E[:, t]], -1))
            if cfg.slope_transport:
                r = gate * net.readout(torch.cat([h0, h, xs[:, t], E[:, t]], -1)) * p["y_scale"]
                cor.append(net.slope(h) - c0)
            else:
                r = (net.readout(torch.cat([h, xs[:, t]], -1)) - g0) * p["y_scale"]
            ys.append(p["f3u"][:, t] + r)
            dec.append(net.dec(h))
        return torch.stack(ys, 1), torch.stack(dec, 1), cor, obs

    def rollout(self, b, teacher_p=0.0, rng=None):
        """Returns predicted cumulative change [N,T+1,K], decoded obs, slope corrections and normalised obs."""
        ctx, net, cfg = self.ctx, self.net, self.cfg
        p = self._prep(b)
        if cfg.static_state:
            return self._static_rollout(b, p)
        obs, xl, e_raw, e = p["obs"], p["xl"], p["e_raw"], p["e"]
        N, T = obs.shape[0], b["T"]
        h = net.enc(obs[:, 0])
        K = len(ctx["y_scale"])
        if cfg.slope_transport:
            g0 = None
            c0 = net.slope(h)
        else:
            g0 = net.readout(torch.cat([h, torch.zeros_like(xl[:, 0])], -1))
            c0 = None
        ys, dec, cor = [torch.zeros(N, K, dtype=h.dtype)], [net.dec(h)], [None]
        cum = torch.zeros(N, K, dtype=h.dtype)
        prev_slope = p["field_slopes"][:, 0]
        for t in range(T):
            gate = self._gate(e[:, t])
            h_prev = h
            h = h + gate * net.step(torch.cat([h, xl[:, t + 1], e[:, t]], -1))
            if teacher_p > 0:
                m = torch.as_tensor(rng.random(N) < teacher_p)[:, None].to(h.dtype)
                h = m * net.enc(obs[:, t + 1]) + (1 - m) * h
            if cfg.slope_transport:
                # per-step local correction rho(h_{t-1}, h_t, xl_t, e_t), gated to zero for a zero action
                r = (
                    gate
                    * net.readout(torch.cat([h_prev, h, xl[:, t + 1], e[:, t]], -1))
                    * p["step_scale"]
                )
                c = net.slope(h) - c0
                slope = p["field_slopes"][:, t + 1] + c.view(N, *ctx["slope_shape"]) * p["a_scale"]
                if cfg.slope_rule == "left":
                    H = self._t(ctx["f3u"].H)
                    curv = 0.5 * torch.einsum("nj,kji,ni->nk", e_raw[:, t], H, e_raw[:, t])
                    cum = cum + torch.einsum("njk,nj->nk", prev_slope, e_raw[:, t]) + curv
                else:
                    cum = cum + 0.5 * torch.einsum("njk,nj->nk", prev_slope + slope, e_raw[:, t])
                prev_slope = slope
                cum = cum + r
                ys.append(cum)
                cor.append(c)
            else:
                r = (net.readout(torch.cat([h, xl[:, t + 1]], -1)) - g0) * p["y_scale"]
                ys.append(p["f3u"][:, t + 1] + r)
            dec.append(net.dec(h))
        return torch.stack(ys, 1), torch.stack(dec, 1), cor, obs

    def loss_by_horizon(self, b, teacher_p=0.0, rng=None):
        rng = rng or np.random.default_rng(self.seed)
        ys, dec, cor, obs = self.rollout(b, teacher_p, rng)
        p = self._prep(b)
        out = {}
        for t in range(1, b["T"] + 1):
            term = (((ys[:, t] - p["y"][:, t]) / p["y_scale"]) ** 2).mean()
            term = term + self.cfg.obs_weight * ((dec[:, t] - obs[:, t]) ** 2).mean()
            if self.cfg.slope_transport:
                term = term + self.cfg.obs_weight * ((cor[t] - p["slope_target"][t]) ** 2).mean()
            out[t] = term
        return out

    def fit(self, batches):
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        rng = np.random.default_rng(self.seed + 7)
        o0 = self._t(self.ctx["obs_norm"](batches[0]["obs"][:1, 0]))
        for epoch in range(self.cfg.epochs):
            p = teacher_forcing_prob(epoch, self.cfg.epochs, self.cfg.p_min, self.cfg.anneal_frac)
            loss = self.cfg.obs_weight * ((self.net.dec(self.net.enc(o0)) - o0) ** 2).mean()
            for b in batches:
                loss = loss + sum(self.loss_by_horizon(b, p, rng).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, b):
        return self.rollout(b)[0].numpy()


class WMEnsemble:
    loss_terms = ("outcome", "observation")

    def __init__(self, cfg: WMConfig, field, base_obs, n_seeds=5):
        self.cfg = cfg
        self.guard = R0Guard(field.theta_hash)
        self.guard.check(field, base_obs)
        self.f3u = F3u(field.payload)
        self.layout = base_obs.payload.get("layout", {})
        self.base_vec = np.asarray(base_obs.payload["obs"])
        if cfg.slope_transport and "A" not in self.layout:
            raise ValueError("slope transport needs the A_t block in the observation layout")
        self.members = [Member(cfg, s) for s in range(n_seeds)]

    def _check(self, trajs):
        if any(t["horizon"] > self.cfg.max_train_horizon for t in trajs):
            raise ValueError(
                f"training trajectories must have horizon <= {self.cfg.max_train_horizon}"
            )

    def fit(self, trajs, components=None):
        self._check(trajs)
        self._all_trajs = list(trajs)
        trajs, val, samples, self.val_index = split_and_bootstrap(
            trajs, components, len(self.members)
        )
        batches = _by_horizon(trajs)
        cat = lambda k, s=slice(1, None): np.concatenate(
            [b[k][:, s].reshape(-1, b[k].shape[-1]) for b in batches]
        )
        E, Y = cat("E"), cat("y")
        self.f3u.beta[:] = 1.0  # raw field: the nest is exact for quadratic dynamics
        y_scale = (Y - self.f3u(E)).std(0) + 1e-9
        steps = np.concatenate(
            [np.diff(b["y"], axis=1).reshape(-1, b["y"].shape[-1]) for b in batches]
        )
        step_E = np.concatenate(
            [np.diff(b["E"], axis=1).reshape(-1, b["E"].shape[-1]) for b in batches]
        )
        allobs = cat("obs", slice(None))
        allxl = cat("xl")
        J, K = self.f3u.g.shape[1], self.f3u.g.shape[0]
        ctx = {
            "f3u": self.f3u,
            "obs_norm": _Norm(allobs),
            "xl_norm": _Norm(allxl),
            "xs_norm": _Norm(cat("xs")),
            "e_scale": float(np.abs(cat("e", slice(None))).max()),
            "y_scale": y_scale,
            # scale of the per-step correction: per-step change not explained by the base field on the step
            "step_scale": (steps - self.f3u(step_E)).std(0) + 1e-9,
            "slope_shape": (J, K),
        }
        d_slope = 0
        if self.cfg.slope_transport:
            a0, a1 = self.layout["A"]

            def raw_target(b, t):
                A = b["obs"][:, t, a0:a1].reshape(-1, J, K)
                return (
                    A - (self.f3u.g.T + np.einsum("kji,ni->njk", self.f3u.H, b["E"][:, t]))
                ).reshape(len(A), -1)

            tg = np.concatenate([raw_target(b, t) for b in batches for t in range(1, b["T"] + 1)])
            a_scale = tg.std() + 1e-9
            ctx["a_scale"] = a_scale
            ctx["slope_target"] = lambda b, t: raw_target(b, t) / a_scale
            d_slope = J * K
        ctx["dims"] = (allobs.shape[-1], allxl.shape[-1], batches[0]["e"].shape[-1], K, d_slope)
        self.ctx = ctx
        # member indices refer to the fit trajectories; with components they are component bootstraps
        self.members = [
            Member(self.cfg, m.seed, ctx).fit(_by_horizon([trajs[i] for i in s]))
            for m, s in zip(self.members, samples, strict=True)
        ]
        _set_member_samples(self, trajs, samples, components)
        self.shrink = select_shrink(
            lambda t: self._raw_predict(self.base_vec, t)[0], self._field, val
        )
        self.sigma_floor = U.field_floor(self._field, val or trajs)
        self.stage_audit = _stage_audit(trajs, val, samples)
        return self

    def _raw_predict(self, base_obs_vec, traj):
        b = batch([traj])
        b = dict(b, obs=np.broadcast_to(base_obs_vec, b["obs"].shape).copy())
        preds = np.stack([m.predict(b)[0] for m in self.members])
        return preds.mean(0), preds.std(0)

    @r0_only
    def predict(self, base_obs, traj):
        # inference reads the base observation only (R0); intermediate observations are never used
        mean, spread = self._raw_predict(np.asarray(base_obs.payload["obs"]), traj)
        f = self._field(traj)
        gate = U.edit_gate(batch([traj])["E"][0])
        return f + self.shrink * (mean - f), U.combine(self.sigma_floor, self.shrink, spread, gate)

    def _field(self, traj):
        return self.f3u(batch([traj])["E"][0])


# ------------------------------------------------------------------ static and field-only baselines


class _StaticNet(nn.Module):
    def __init__(self, d, K, cfg):
        super().__init__()
        self.f = _mlp(d, cfg.hidden, K)


class MatchedStatic(WMEnsemble):
    """The static comparator matched to WM-E (Board A condition 4): identical network, parameter count, base
    features, cumulative oriented representation, labels, auxiliary grounding targets, normalisation, shrinkage
    grid, seeds and component bootstrap; the state depends on (o_0, E_t) only, so the prediction at T is a function
    of the realised final edit, independent of its decomposition."""

    def __init__(self, cfg: WMConfig, field, base_obs, n_seeds=5):
        super().__init__(dataclasses.replace(cfg, static_state=True), field, base_obs, n_seeds)


def _stage_audit(fit, val, samples):
    """Maximum training horizon seen by every learned stage (Board A condition 3: all must be <= 2)."""
    mh = lambda ts: max((t["horizon"] for t in ts), default=0)
    return {
        "normalisation": mh(fit),
        "members": max(mh([fit[i] for i in s]) for s in samples),
        "shrinkage": mh(val),
        "sigma_floor": mh(val or fit),
        "f3u": 0,  # raw field nest, unit coefficients: no outcome refit
    }


def refit_f3u(f3u: F3u, trajs, max_h: int = 2) -> F3u:
    """Least-squares F3u refit on training-component states of horizon <= ``max_h`` only (never outer roots)."""
    if any(t["horizon"] > max_h for t in trajs):
        raise ValueError(f"F3u refits use horizons <= {max_h} only")
    bs = [batch([t]) for t in trajs]
    E = np.concatenate([b["E"][0, 1:] for b in bs])
    Y = np.concatenate([b["y"][0, 1:] for b in bs])
    return f3u.fit(E, Y)


class AnchorShift:
    """Anchor-shift static planning comparator (r9 C7; Board A condition 4): the static prediction plus the
    residual at the nearest evaluated anchor state (cumulative-E distance d), weighted exp(-d^2 / 2 l^2), l = the
    median pairwise anchor distance. Uses the same real observations a WM planning filter would; no state."""

    def __init__(self, static, base_obs, anchors):
        self.static, self.guard = static, static.guard
        self.anchors = []
        for a in anchors:
            T = a["horizon"]
            E = batch([a])["E"][0][T]
            r = np.asarray(a["y"][T]) - static.predict(base_obs, a)[0][T]
            self.anchors.append((E, r))
        Es = np.stack([e for e, _ in self.anchors]) if self.anchors else np.zeros((0, 1))
        d = np.sqrt(((Es[:, None] - Es[None]) ** 2).sum(-1))
        pos = d[np.triu_indices(len(Es), 1)]
        self.length_scale = float(np.median(pos)) if len(pos) else 1.0

    @r0_only
    def predict(self, base_obs, traj):
        mean, sigma = self.static.predict(base_obs, traj)
        mean = mean.copy()
        E = batch([traj])["E"][0]
        for t in range(1, len(E)):
            if not self.anchors or not np.any(E[t] != 0):
                continue
            d = [float(np.linalg.norm(E[t] - e)) for e, _ in self.anchors]
            i = int(np.argmin(d))
            mean[t] = (
                mean[t] + np.exp(-(d[i] ** 2) / (2 * self.length_scale**2)) * self.anchors[i][1]
            )
        return mean, sigma


def _set_member_samples(model, fit_trajs, samples, components):
    """Record each member's training sample as indices into the trajectories passed to ``fit``."""
    model.member_samples = None
    if components is not None:
        pos = {id(t): i for i, t in enumerate(model._all_trajs)}
        model.member_samples = [[pos[id(fit_trajs[i])] for i in s] for s in samples]


class StaticPredictor:
    """Same-input static predictor: F3u(E_T) + MLP(cumulative-E bookkeeping) - MLP(0); 5-seed ensemble mean.

    ``predict`` returns (mean, sigma) with the WM-E sigma treatment (field floor, component bootstrap; r9 C1.2)."""

    def __init__(self, cfg: WMConfig, field, n_seeds=5):
        self.cfg = cfg
        self.guard = R0Guard(field.theta_hash)
        self.f3u = F3u(field.payload)
        self.n_seeds = n_seeds

    def fit(self, trajs, components=None):
        if any(t["horizon"] > self.cfg.max_train_horizon for t in trajs):
            raise ValueError("training trajectories must have horizon <= max_train_horizon")
        self._all_trajs = list(trajs)
        trajs, val, samples, self.val_index = split_and_bootstrap(trajs, components, self.n_seeds)
        bs = [batch([t]) for t in trajs]
        rows = lambda key, idx: np.concatenate([bs[i][key][0, 1:] for i in idx])
        E = rows("E", range(len(bs)))
        XS = rows("xs", range(len(bs)))
        Y = rows("y", range(len(bs)))
        self.f3u.beta[:] = (
            1.0  # the same raw-field nest as the world model (no outcome-fitted coefficients)
        )
        resid = Y - self.f3u(E)
        self.y_scale = resid.std(0) + 1e-9
        self.norm = _Norm(XS)
        self.nets = []
        X0 = torch.as_tensor(self.norm(np.zeros((1, XS.shape[1]))), dtype=torch.float32)
        for s, sample in enumerate(samples):
            XSs, Es, Ys = rows("xs", sample), rows("E", sample), rows("y", sample)
            X = torch.as_tensor(self.norm(XSs), dtype=torch.float32)
            T = torch.as_tensor((Ys - self.f3u(Es)) / self.y_scale, dtype=torch.float32)
            torch.manual_seed(1000 + s)
            net = _StaticNet(XS.shape[1], Y.shape[1], self.cfg).float()
            opt = torch.optim.AdamW(
                net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
            for _ in range(self.cfg.epochs):
                loss = ((net.f(X) - net.f(X0) - T) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.nets.append(net)
        _set_member_samples(self, trajs, samples, components)
        self.shrink = select_shrink(lambda t: self._raw_predict(t)[0], self._field, val)
        self.sigma_floor = U.field_floor(self._field, val or trajs)
        self.stage_audit = _stage_audit(trajs, val, samples)
        return self

    @r0_only
    def predict(self, base_obs, traj):
        f = self._field(traj)
        mean, spread = self._raw_predict(traj)
        gate = U.edit_gate(batch([traj])["E"][0])
        return f + self.shrink * (mean - f), U.combine(self.sigma_floor, self.shrink, spread, gate)

    def _field(self, traj):
        return self.f3u(batch([traj])["E"][0])

    def _raw_predict(self, traj):
        b = batch([traj])
        E, XS = b["E"][0], b["xs"][0]
        out = self.f3u(E)
        with torch.no_grad():
            X = torch.as_tensor(self.norm(XS), dtype=torch.float32)
            X0 = torch.as_tensor(
                self.norm(np.zeros((1, XS.shape[1]))), dtype=torch.float32
            ).expand_as(X)
            at_zero = (X == X0).all(-1, keepdim=True)  # x(D) = x(0): s(x) - s(x(0)) is exactly 0
            rs = np.stack([torch.where(at_zero, 0.0, n.f(X) - n.f(X0)).numpy() for n in self.nets])
        return out + rs.mean(0) * self.y_scale, rs.std(0) * self.y_scale


class FieldOnly:
    """The base field alone (raw a + q, unit coefficients): the trust-region predictor."""

    def __init__(self, field):
        self.guard = R0Guard(field.theta_hash)
        self.f3u = F3u(field.payload)

    @r0_only
    def predict(self, base_obs, traj):
        return self.f3u(batch([traj])["E"][0])


def _point(pred):
    return pred[0] if isinstance(pred, tuple) else pred


def horizon_errors(model, base_obs, trajs) -> dict[int, float]:
    """Mean absolute error of the cumulative outcome change by horizon (over trajectories and objectives)."""
    errs: dict[int, list] = {}
    for t in trajs:
        p = _point(model.predict(base_obs, t))
        for h in range(1, t["horizon"] + 1):
            errs.setdefault(h, []).append(np.abs(p[h] - t["y"][h]).mean())
    return {h: float(np.mean(v)) for h, v in sorted(errs.items())}


def outcome_scale(trajs, horizons=(3, 4)) -> float:
    return float(
        np.mean([np.abs(t["y"][h]).mean() for t in trajs for h in horizons if h <= t["horizon"]])
    )
