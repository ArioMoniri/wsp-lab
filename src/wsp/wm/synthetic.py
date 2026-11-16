"""Synthetic validation target for WM-E v0 (SYNTHESIS-r8 D8): a small network with RMSNorm and saturating units.

The network maps an input u (d_in) through ``z = W1 u`` (m hidden units), ``n = RMSNorm(z) * g``, ``a = tanh(beta n)``
and logits ``W2 a`` (C classes). Outcomes are K objectives, each the mean cross-entropy on its own fixed item group.
Edits move W1 along a dictionary of J unit-Frobenius rank-one directions E_j; a step is (direction j, dose alpha).

* ``kind="saturating"``: outcomes are the network's actual losses (non-quadratic: normalisation and saturation).
* ``kind="quadratic"``: outcomes are replaced by the exact second-order Taylor expansion of those losses at theta0,
  so the second-order field is exact and no model can gain over the static field.

Observations at theta_t (the grounding targets of D3): residual sketches (post-norm activations on fixed probe inputs
projected to k dims), per-item losses (a fixed subset of outcome items) and action derivatives
``A_t[j, k] = <grad y_k(theta_t), E_j>``. The field at theta0 (R0) is the gradient and exact Hessian of y_k in the
W1 coordinates of the edit subspace.
"""

from __future__ import annotations

import numpy as np
import torch

from wsp.wm.access import Artefact, Tier

BASE_HASH = "5" * 64  # synthetic theta0 identity (a fixed tag; there is no checkpoint)
LADDER = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)


class SynthWorld:
    def __init__(
        self,
        kind="saturating",
        seed=0,
        d_in=6,
        m=4,
        n_classes=4,
        n_dirs=24,
        n_obj=3,
        items_per_obj=48,
        n_probe=8,
        k_sketch=3,
        beta=2.5,
        step=0.4,
        input_rank=2,
    ):
        if kind not in ("saturating", "quadratic"):
            raise ValueError(kind)
        self.kind, self.seed = kind, seed
        g = torch.Generator().manual_seed(1000 + seed)
        self.d_in, self.m, self.C, self.J, self.K = d_in, m, n_classes, n_dirs, n_obj
        self.beta, self.step = beta, step
        self.W1 = torch.randn(m, d_in, generator=g, dtype=torch.float64) / np.sqrt(d_in)
        self.gain = 1.0 + 0.1 * torch.randn(m, generator=g, dtype=torch.float64)
        self.W2 = torch.randn(n_classes, m, generator=g, dtype=torch.float64)
        dirs = []
        for _ in range(n_dirs):
            p = torch.randn(m, generator=g, dtype=torch.float64)
            q = torch.randn(d_in, generator=g, dtype=torch.float64)
            e = torch.outer(p, q)
            dirs.append(e / e.norm())
        self.E = torch.stack(dirs)  # J x m x d_in
        r = input_rank or d_in
        basis = torch.linalg.qr(torch.randn(d_in, r, generator=g, dtype=torch.float64))[
            0
        ].T * np.sqrt(d_in / r)
        self.U = torch.randn(n_obj, items_per_obj, r, generator=g, dtype=torch.float64) @ basis
        with torch.no_grad():
            logits0 = self._logits(self.W1, self.U)
        # labels: a perturbed argmax so base losses are moderate and edits can move them either way
        noise = torch.randn(logits0.shape, generator=g, dtype=torch.float64)
        self.labels = (logits0 + 1.5 * noise).argmax(-1)
        self.probes = torch.randn(n_probe, r, generator=g, dtype=torch.float64) @ basis
        self.P = torch.randn(m, k_sketch, generator=g, dtype=torch.float64) / np.sqrt(k_sketch)
        self.obs_items = torch.arange(0, items_per_obj, 6)
        self._coef0 = torch.zeros(n_dirs, dtype=torch.float64)
        self._g0, self._H0 = self._grad_hess(self._coef0)
        self._y0 = self._true_losses(self._coef0)

    # --------------------------------------------------------------- network
    def _act(self, W1, U):
        z = U @ W1.T
        n = z / torch.sqrt((z**2).mean(-1, keepdim=True) + 1e-6) * self.gain
        return torch.tanh(self.beta * n)

    def _logits(self, W1, U):
        return self._act(W1, U) @ self.W2.T

    def _W1(self, coef):
        return self.W1 + torch.einsum("j,jmd->md", coef, self.E)

    def _item_losses(self, coef):
        logits = self._logits(self._W1(coef), self.U)  # K x N x C
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.C), self.labels.reshape(-1), reduction="none"
        ).view(self.K, -1)

    def _true_losses(self, coef):
        return self._item_losses(coef).mean(-1)

    def _grad_hess(self, coef):
        f = lambda c: self._true_losses(c)
        jac = torch.autograd.functional.jacobian(f, coef)  # K x J
        hess = torch.stack(
            [torch.autograd.functional.hessian(lambda c, k=k: f(c)[k], coef) for k in range(self.K)]
        )
        return jac.detach(), hess.detach()

    def outcomes(self, coef):
        coef = torch.as_tensor(coef, dtype=torch.float64)
        if self.kind == "quadratic":
            return (
                self._y0 + self._g0 @ coef + 0.5 * torch.einsum("j,kji,i->k", coef, self._H0, coef)
            )
        return self._true_losses(coef).detach()

    def slopes(self, coef):
        coef = torch.as_tensor(coef, dtype=torch.float64)
        if self.kind == "quadratic":
            return (self._g0 + torch.einsum("kji,i->kj", self._H0, coef)).T  # J x K
        return torch.autograd.functional.jacobian(self._true_losses, coef).detach().T

    def observe(self, coef):
        """Grounding targets at theta(coef): sketches, per-item losses, action derivatives (flattened)."""
        coef = torch.as_tensor(coef, dtype=torch.float64)
        with torch.no_grad():
            sk = (self._act(self._W1(coef), self.probes) @ self.P).reshape(-1)
            if self.kind == "quadratic":
                # observations consistent with the quadratic world: per-item losses from its own expansion
                losses = self.outcomes(coef).repeat_interleave(len(self.obs_items))
            else:
                losses = self._item_losses(coef)[:, self.obs_items].reshape(-1)
        a = self.slopes(coef).reshape(-1)
        return torch.cat([sk, losses, a]).numpy()

    # --------------------------------------------------------------- R0 artefacts
    def field(self) -> Artefact:
        return Artefact("field", BASE_HASH, Tier.R0, {"g": self._g0.numpy(), "H": self._H0.numpy()})

    def base_observation(self) -> Artefact:
        n_sk = len(self.probes) * self.P.shape[1]
        n_loss = self.K * len(self.obs_items)
        layout = {
            "sketch": (0, n_sk),
            "losses": (n_sk, n_sk + n_loss),
            "A": (n_sk + n_loss, n_sk + n_loss + self.J * self.K),
        }
        return Artefact(
            "tap", BASE_HASH, Tier.R0, {"obs": self.observe(self._coef0), "layout": layout}
        )

    # --------------------------------------------------------------- trajectories
    def trajectory_from_actions(self, actions):
        coef = torch.zeros(self.J, dtype=torch.float64)
        coefs, ys, obs = [coef.clone()], [self.outcomes(coef).numpy()], [self.observe(coef)]
        for j, dose in actions:
            coef = coef.clone()
            coef[j] += dose * self.step
            coefs.append(coef)
            ys.append(self.outcomes(coef).numpy())
            obs.append(self.observe(coef))
        y = np.stack(ys)
        return {
            "actions": [(int(j), float(d)) for j, d in actions],
            "coef": np.stack([c.numpy() for c in coefs]),
            "e": action_tokens(self, actions),
            "x": bookkeeping(self, actions),
            "y": y - y[0],  # outcome change relative to the base
            "obs": np.stack(obs),
            "horizon": len(actions),
        }

    def trajectories(self, n, horizon, seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            js = rng.choice(self.J, size=horizon, replace=False)
            doses = rng.choice(LADDER, size=horizon)
            out.append(self.trajectory_from_actions(list(zip(js, doses, strict=True))))
        return out


def action_tokens(world: SynthWorld, actions) -> np.ndarray:
    """e_t: one-hot direction times dose (the action recipe; CPU arithmetic only)."""
    e = np.zeros((len(actions), world.J))
    for t, (j, d) in enumerate(actions):
        e[t, j] = d * world.step
    return e


def bookkeeping(world: SynthWorld, actions) -> np.ndarray:
    """Exact bookkeeping x_t (t = 0..T): cumulative E_t coordinates, realised D_t, step position, signed overlaps
    cos(D_t, E_{t-1}) and the cumulative norm. Computed by algebra on the recipes, never predicted."""
    T = len(actions)
    x = np.zeros((T + 1, 2 * world.J + 3))
    E = np.zeros(world.J)
    Ed = world.E.reshape(world.J, -1).numpy()
    for t, (j, d) in enumerate(actions, start=1):
        D = np.zeros(world.J)
        D[j] = d * world.step
        prevE = Ed.T @ E
        Dm = Ed.T @ D
        denom = np.linalg.norm(prevE) * np.linalg.norm(Dm)
        overlap = float(prevE @ Dm / denom) if denom > 0 else 0.0
        E = E + D
        x[t, : world.J] = E
        x[t, world.J : 2 * world.J] = D
        x[t, -3] = t
        x[t, -2] = overlap
        x[t, -1] = np.linalg.norm(Ed.T @ E)
    return x


def raw_field(world: SynthWorld, E) -> np.ndarray:
    """The shared R0 nest: a + q = <g0, E> + 1/2 E^T H0 E on the cumulative edit (per objective)."""
    g, H = world._g0.numpy(), world._H0.numpy()
    E = np.asarray(E)
    return E @ g.T + 0.5 * np.einsum("...j,kji,...i->...k", E, H, E)


def shuffle_outcomes(trajs, seed, world: SynthWorld | None = None, sign_flip: bool = True):
    """Null control for the world-model-vs-static contrast: break every edit -> outcome link beyond the shared R0
    field. Outcome residuals over the raw field, r_t = y_t - F(E_t), and the observations o_t (t >= 1) are permuted
    across trajectories of equal horizon and (``sign_flip``) multiplied by a random sign per trajectory, so they have
    zero mean at every horizon; the base observation o_0 and the actions/bookkeeping are unchanged. With
    ``world=None`` the plain permutation of y and o_t is returned (a diagnostic: its marginal structure is learnable
    and extrapolates arbitrarily, so it is not a clean null)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trajs))
    signs = (
        rng.choice([-1.0, 1.0], size=len(trajs))
        if (sign_flip and world is not None)
        else np.ones(len(trajs))
    )
    out = []
    for t, p, sgn in zip(trajs, perm, signs, strict=True):
        src = trajs[p]
        if src["horizon"] != t["horizon"]:
            raise ValueError("shuffle within one horizon")
        obs = src["obs"].copy()
        obs[0] = t["obs"][0]
        if world is None:
            y = src["y"].copy()
        else:
            J = world.J
            f_t, f_src = raw_field(world, t["x"][:, :J]), raw_field(world, src["x"][:, :J])
            y = f_t + sgn * (src["y"] - f_src)
            obs[1:] = obs[0] + sgn * (obs[1:] - obs[0])
        out.append(dict(t, y=y, obs=obs))
    return out
