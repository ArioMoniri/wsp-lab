"""Calibrated uncertainty for WM-E and the static arms (SYNTHESIS-r9 C1, pro-engineer W4; PREREG_WME §6).

* **sigma floor.** ``sigma_t = gate(E_t) * sqrt(sigma_F^2 + (lambda sigma_ens)^2)``: sigma_F is the per-objective
  RMS of the field-only residual ``y_t - F(E_t)`` on the selection split (horizons <= 2), lambda the selected
  shrinkage and sigma_ens the across-member spread. ``gate(E_t) = 1[E_t != 0]`` keeps the zero edit exactly
  zero with exactly zero sigma; elsewhere sigma > 0 even when lambda = 0.
* **Component bootstrap.** Ensemble members are fitted on resamples of source-closure components (with
  replacement; a drawn component brings all its trajectories), never rows. The selection split is by component.
* **Split-conformal scaling.** Scores ``|y - yhat| / sigma`` on calibration trajectories of horizon <= 2 only; the
  finite-sample quantile ``q`` at level ``ceil((n+1)(1-alpha))/n`` multiplies sigma. Coverage is then checked at
  h = 3-4; the kill is a miss beyond 10 pp of the 80% target (two-sided: "within 10 pp"), after which E2 uses
  ranks only and no LCB (:class:`CalibratedPredictor` raises :class:`RanksOnlyError`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

TARGET_COVERAGE = 0.80
COVERAGE_TOL_PP = 10.0
CAL_MAX_HORIZON = 2
CHECK_HORIZONS = (3, 4)


class RanksOnlyError(RuntimeError):
    """Coverage failed: the planner may use predicted ranks only, never an LCB."""


# ------------------------------------------------------------------ components


def _groups(components: Sequence) -> dict:
    g: dict = {}
    for i, c in enumerate(components):
        g.setdefault(c, []).append(i)
    return g


def component_bootstrap(components: Sequence, n: int, seed: int) -> list[list[int]]:
    """``n`` bootstrap samples of row indices, resampling whole components with replacement."""
    g = _groups(components)
    keys = sorted(g, key=str)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        draw = rng.integers(0, len(keys), size=len(keys))
        out.append([i for d in sorted(draw.tolist()) for i in g[keys[d]]])
    return out


def component_selection_split(components: Sequence, frac=0.2, seed=20261116):
    """(fit indices, selection indices); a component is wholly in one part."""
    g = _groups(components)
    keys = sorted(g, key=str)
    rng = np.random.default_rng(seed)
    order = [keys[i] for i in rng.permutation(len(keys))]
    n_val = max(1, round(frac * len(keys))) if len(keys) > 4 else 0
    val_keys = set(order[:n_val])
    fit = [i for i, c in enumerate(components) if c not in val_keys]
    val = [i for i, c in enumerate(components) if c in val_keys]
    return fit, val


# ------------------------------------------------------------------ floor


def edit_gate(E: np.ndarray) -> np.ndarray:
    """[T+1, 1] indicator that the cumulative edit E_t is nonzero."""
    return (np.abs(np.asarray(E)) > 0).any(-1, keepdims=True).astype(float)


def field_floor(field_fn, trajs, max_h: int = CAL_MAX_HORIZON) -> np.ndarray:
    """Per-objective RMS of the field-only residual y_t - F(E_t) over t = 1..min(h, max_h)."""
    rs = []
    for t in trajs:
        f = field_fn(t)
        hmax = min(t["horizon"], max_h)
        rs.append(np.asarray(t["y"])[1 : hmax + 1] - f[1 : hmax + 1])
    r = np.concatenate(rs)
    return np.sqrt((r**2).mean(0))


def combine(sigma_floor, shrink, spread, gate) -> np.ndarray:
    return gate * np.sqrt(np.asarray(sigma_floor) ** 2 + (shrink * np.asarray(spread)) ** 2)


# ------------------------------------------------------------------ conformal


class ConformalScaler:
    def __init__(self, alpha: float = 1.0 - TARGET_COVERAGE):
        self.alpha = float(alpha)
        self.q = float("nan")
        self.n = 0

    def fit_scores(self, scores) -> ConformalScaler:
        s = np.sort(np.asarray(scores, dtype=float).ravel())
        s = s[np.isfinite(s)]
        if len(s) == 0:
            raise ValueError("no calibration scores")
        self.n = len(s)
        k = min(self.n, math.ceil((self.n + 1) * (1.0 - self.alpha)))
        self.q = float(s[k - 1])
        return self

    def fit(self, abs_err, sigma) -> ConformalScaler:
        abs_err, sigma = np.asarray(abs_err, float).ravel(), np.asarray(sigma, float).ravel()
        keep = sigma > 0
        return self.fit_scores(abs_err[keep] / sigma[keep])

    def coverage(self, abs_err, sigma) -> float:
        abs_err, sigma = np.asarray(abs_err, float).ravel(), np.asarray(sigma, float).ravel()
        keep = sigma > 0
        return float(np.mean(abs_err[keep] <= self.q * sigma[keep]))


def _pairs(model, base_obs, trajs, horizons):
    errs, sigs = [], []
    for t in trajs:
        mean, sigma = model.predict(base_obs, t)
        for h in horizons:
            if h <= t["horizon"]:
                errs.append(np.abs(mean[h] - t["y"][h]))
                sigs.append(sigma[h])
    return np.concatenate(errs) if errs else np.zeros(0), np.concatenate(
        sigs
    ) if sigs else np.zeros(0)


def calibrate(model, base_obs, cal_trajs, alpha: float = 1.0 - TARGET_COVERAGE) -> ConformalScaler:
    """Fit the conformal scale on calibration trajectories of horizon <= 2 only."""
    if any(t["horizon"] > CAL_MAX_HORIZON for t in cal_trajs):
        raise ValueError(f"conformal calibration uses horizons <= {CAL_MAX_HORIZON} only")
    e, s = _pairs(model, base_obs, cal_trajs, range(1, CAL_MAX_HORIZON + 1))
    return ConformalScaler(alpha).fit(e, s)


def coverage_decision(coverage: float, target=TARGET_COVERAGE, tol_pp=COVERAGE_TOL_PP) -> dict:
    miss = 100.0 * abs(coverage - target)
    ok = miss <= tol_pp
    return {
        "coverage": float(coverage),
        "target": target,
        "miss_pp": miss,
        "pass": ok,
        "ranks_only": not ok,
    }


def check_coverage(model, scaler, base_obs, test_trajs, horizons=CHECK_HORIZONS) -> dict:
    e, s = _pairs(model, base_obs, test_trajs, horizons)
    keep = s > 0
    out = coverage_decision(scaler.coverage(e, s))
    out.update(horizons=list(horizons), n=int(keep.sum()), q=scaler.q)
    return out


class CalibratedPredictor:
    """A predictor with conformally scaled sigma; ``lcb`` refuses when coverage failed (ranks only)."""

    def __init__(self, model, scaler: ConformalScaler, ranks_only: bool):
        self.model, self.scaler, self.ranks_only = model, scaler, bool(ranks_only)

    def predict(self, base_obs, traj):
        mean, sigma = self.model.predict(base_obs, traj)
        return mean, self.scaler.q * sigma

    def lcb(self, base_obs, traj, orient):
        if self.ranks_only:
            raise RanksOnlyError("conformal coverage failed at h = 3-4: ranks only, no LCB")
        mean, sigma = self.predict(base_obs, traj)
        return np.asarray(orient) * mean - sigma
