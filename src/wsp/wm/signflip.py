"""Sign-flip randomisation tests (RULING-wme-v2 A2-A3; replaces the H-T ratio bootstrap-t and the E2 two-way
percentile bootstrap). Exact under the null that the unit-level paired differences are symmetric about zero.

* :func:`test`: one-sided test of mean(d) <= 0 against > 0 for unit values d with weights w (sum 1): statistic
  T = sum w d; R seeded sign-flip vectors s (``n_flip`` = 10,000 registered; ``None`` = exact enumeration);
  p = (1 + #{T*_j >= T}) / (R + 1) with the observed pattern counted once. The one-sided lower confidence bound
  is obtained **by inverting the test** with the same flips: shifting d by delta changes T*_j - T by
  delta (1 - sum w s_j), so each flip has a threshold c_j and p(delta) is a step function; the bound is the
  smallest delta with p(delta) > alpha (closed form, no bisection).
* :func:`ht_test` (H-T): d_c = mean over a held-out component's (state, objective) pairs of |e_static| - |e_WM|
  (SE-scaled absolute errors), equal weight per component. C_T = 1 - sum|e_WM| / sum|e_static| is reported with its
  lower bound from inverting the same sign-flip test on u_c(c) = (1 - c) mean|e_static|_c - mean|e_WM|_c.
* :func:`e2_test` (H-E2): family-stratified: T = mean over the three families of the family-mean paired difference
  (WM-guided minus StatNN-guided guarded test score); each episode's sign is flipped independently.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

ALPHA = 0.025
N_FLIP = 10_000
SEED = 20261117
N_FAMILIES = 3


def flips(n: int, n_flip: int | None = N_FLIP, seed: int = SEED) -> np.ndarray:
    if n_flip is None:
        return np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=(n_flip, n))


def _thresholds(d, w, S):
    """Per flip j: the shift delta at and above which T*_j(delta) >= T(delta) holds."""
    a = (w * d).sum() - (S * (w * d)).sum(1)
    b = 1.0 - (S * w).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(
            b > 1e-15, a / np.where(b > 1e-15, b, 1.0), np.where(a <= 1e-15, -np.inf, np.inf)
        )
    return c


def test(
    d, weights=None, n_flip: int | None = N_FLIP, seed: int = SEED, alpha: float = ALPHA
) -> dict:
    d = np.asarray(d, float)
    n = len(d)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float) / np.sum(weights)
    S = flips(n, n_flip, seed)
    T = float((w * d).sum())
    Ts = S @ (w * d)
    tol = 1e-12 * max(1.0, abs(T))
    if n_flip is None:
        p = float(np.mean(Ts >= T - tol))
        c = np.sort(_thresholds(d, w, S))
        k = math.floor(alpha * len(S)) + 1  # p(delta) = #{c_j <= delta} / 2^n > alpha
    else:
        p = float((1 + np.sum(Ts >= T - tol)) / (n_flip + 1))
        c = np.sort(_thresholds(d, w, S))
        k = math.floor(alpha * (n_flip + 1) - 1) + 1  # (1 + #{c_j <= delta}) / (R + 1) > alpha
    lb = float(c[k - 1]) if 0 < k <= len(c) else float("-inf")
    return {
        "stat": T,
        "p": p,
        "lb": lb,
        "reject": p <= alpha,
        "n": n,
        "n_flip": len(S),
        "alpha": alpha,
    }


def ht_test(
    e_wm,
    e_static,
    components,
    n_flip: int | None = N_FLIP,
    seed: int = SEED,
    alpha: float = ALPHA,
    ct_grid: int = 60,
) -> dict:
    a = np.abs(np.asarray(e_wm, float))
    s = np.abs(np.asarray(e_static, float))
    keys, inv = np.unique(np.asarray([str(c) for c in components]), return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    am = np.bincount(inv, weights=a) / cnt
    sm = np.bincount(inv, weights=s) / cnt
    r = test(sm - am, n_flip=n_flip, seed=seed, alpha=alpha)
    ct = float(1.0 - a.sum() / s.sum()) if s.sum() > 0 else float("nan")
    # C_T bound by inverting the same test on u_c(c) = (1 - c) sm_c - am_c (p increases with c)
    S = flips(len(keys), n_flip, seed)
    w = np.full(len(keys), 1.0 / len(keys))

    def p_of(cv):
        u = (1 - cv) * sm - am
        T = (w * u).sum()
        Ts = S @ (w * u)
        if n_flip is None:
            return float(np.mean(Ts >= T - 1e-12))
        return float((1 + np.sum(Ts >= T - 1e-12 * max(1, abs(T)))) / (n_flip + 1))

    lo, hi = -10.0, 1.0
    if p_of(lo) > alpha:
        ct_lb = float("-inf")
    else:
        for _ in range(ct_grid):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if p_of(mid) <= alpha else (lo, mid)
        ct_lb = hi
    return dict(r, n_components=len(keys), ct=ct, ct_lb=ct_lb, mean_diff=r["stat"])


def e2_test(
    d, families, n_flip: int | None = N_FLIP, seed: int = SEED, alpha: float = ALPHA
) -> dict:
    d = np.asarray(d, float)
    fam = np.asarray(families)
    fs = np.unique(fam)
    if len(fs) != N_FAMILIES:
        raise ValueError(f"the E2 test needs episodes from all {N_FAMILIES} families")
    w = np.zeros(len(d))
    for f in fs:
        m = fam == f
        w[m] = 1.0 / (N_FAMILIES * m.sum())
    return dict(test(d, weights=w, n_flip=n_flip, seed=seed, alpha=alpha), n_episodes=len(d))
