"""Confirmatory statistics for WM-E (PREREG_WME §5-§8; Board A conditions 3, 8, 9, 11). Pure numpy, CPU.

* :func:`ct` / :func:`ct_lower_bound`: C_T = 1 - sum w|e_WM| / sum w|e_static| (a ratio of cluster totals) with a
  component-cluster **bootstrap-t** one-sided lower bound. Studentisation: the linearised (delta-method) SE of the
  ratio R = A/B from cluster totals, recomputed inside every resample; the bound is 1 - (R - q_alpha(t*) se). Not
  estimable (``lb = nan``, ``estimable = False``) when fewer than ``MIN_CLUSTERS`` clusters or a zero static total.
  Row weights implement the registered weighting: ``"pooled"`` (every state x objective pair equal) or
  ``"component"`` (each component's pairs share weight 1).
* :func:`paired_lb`: paired item-cluster bootstrap lower bound of a mean (percentile; two-sided 95% => 2.5th pct).
* :func:`guard_pass`: simultaneous Bonferroni guardrails on the pp-equivalent scale (z at alpha/(2m)).
* :func:`holm`: Holm step-down adjusted decisions; :func:`clopper_pearson`: exact binomial interval (Monte Carlo
  uncertainty of size, coverage and power).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

MIN_CLUSTERS = 5
N_BOOT = 2000
ALPHA_ONE_SIDED = 0.05


def _cluster_totals(a, b, clusters, w=None):
    a, b = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    keys, inv = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)
    A = np.bincount(inv, weights=w * a, minlength=len(keys))
    B = np.bincount(inv, weights=w * b, minlength=len(keys))
    return A, B


def component_weights(clusters) -> np.ndarray:
    _keys, inv, cnt = np.unique(
        np.asarray([str(c) for c in clusters]), return_inverse=True, return_counts=True
    )
    return 1.0 / cnt[inv]


def ct(e_wm, e_static, w=None) -> float:
    a, b = np.abs(np.asarray(e_wm, float)), np.abs(np.asarray(e_static, float))
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    return float(1.0 - (w * a).sum() / (w * b).sum())


def _ratio_se(A, B):
    n = len(A)
    R = A.sum() / B.sum()
    resid = A - R * B
    return R, math.sqrt(n / max(1, n - 1) * (resid**2).sum()) / B.sum()


def ct_lower_bound(
    e_wm, e_static, clusters, weights="pooled", n_boot=N_BOOT, seed=20261117, alpha=ALPHA_ONE_SIDED
) -> dict:
    w = component_weights(clusters) if weights == "component" else None
    A, B = _cluster_totals(e_wm, e_static, clusters, w)
    n = len(A)
    out = {"n_clusters": n, "weights": weights, "alpha": alpha, "n_boot": n_boot}
    if n < MIN_CLUSTERS or B.sum() <= 0:
        return dict(out, ct=float("nan"), lb=float("nan"), se=float("nan"), estimable=False)
    R, se = _ratio_se(A, B)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))
    tstar = []
    for d in draw:
        Ab, Bb = A[d], B[d]
        if Bb.sum() <= 0:
            continue
        Rb, seb = _ratio_se(Ab, Bb)
        if seb > 0:
            tstar.append((Rb - R) / seb)
    q = float(np.quantile(tstar, alpha)) if tstar else float("nan")
    r_upper = R - q * se
    return dict(out, ct=1.0 - R, lb=1.0 - r_upper, se=se, estimable=True)


def paired_lb(
    diffs, clusters=None, level=0.95, two_sided=True, n_boot=N_BOOT, seed=20261117
) -> dict:
    d = np.asarray(diffs, float)
    clusters = np.arange(len(d)) if clusters is None else np.asarray([str(c) for c in clusters])
    keys, inv = np.unique(clusters, return_inverse=True)
    S = np.bincount(inv, weights=d, minlength=len(keys))
    N = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = S[draw].sum(1) / N[draw].sum(1)
    tail = (1 - level) / 2 if two_sided else 1 - level
    return {
        "mean": float(d.mean()),
        "lb": float(np.quantile(means, tail)),
        "ub": float(np.quantile(means, 1 - tail)),
    }


def bonferroni_z(m: int, alpha=0.05, two_sided=True) -> float:
    return float(stats.norm.ppf(1 - alpha / ((2 if two_sided else 1) * max(1, m))))


def guard_pass(guards: dict, alpha=0.05) -> dict:
    """``guards[name] = (mean, se, kind, limit)``, kind "ucb" (mean + z se <= limit) or "lcb" (>= limit)."""
    z = bonferroni_z(len(guards), alpha)
    breaches = []
    for g, (m, se, kind, lim) in guards.items():
        if kind == "ucb" and m + z * se > lim:
            breaches.append(g)
        if kind == "lcb" and m - z * se < lim:
            breaches.append(g)
    return {"z": z, "breaches": breaches, "pass": not breaches}


def holm(pvals: dict, alpha=0.05) -> dict:
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out, stop = {}, False
    for i, k in enumerate(order):
        ok = (not stop) and pvals[k] <= alpha / (m - i)
        stop = stop or not ok
        out[k] = ok
    return out


def clopper_pearson(k: int, n: int, level=0.95) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def one_sided_lower(k: int, n: int, level=0.95) -> float:
    """One-sided exact lower bound of a binomial proportion (the "power lower bound" of §8)."""
    return 0.0 if k == 0 else float(stats.beta.ppf(1 - level, k, n - k + 1))
