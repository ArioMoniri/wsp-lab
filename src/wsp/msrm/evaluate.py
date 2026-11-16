"""Contrasts, component-bootstrap intervals, descriptive calibration and the MSRM-1 v0 development kill criteria.

C(x : ref) = 1 − Σ|e_x| / Σ|e_ref| over the pooled out-of-fold errors (ratio of sums; SYNTHESIS-r7 "Report").
Intervals resample whole components with replacement (PCG64 seed 20261107, 4,000 draws by default; percentile
2.5 / 97.5, numpy linear quantiles); a draw is valid when Σ|e_ref| / n > 1e-12. These are development diagnostics on
exposed outcomes, not registered intervals.
"""

from __future__ import annotations

import numpy as np

BOOT_SEED = 20261107
DRAWS = 4000
H_STAR = (
    0.0049397  # nats/token, F8a.2 / F12a floor; a screening reference for descriptive labels only
)
DENOM_TOL = 1e-12


def contrast(y, pred, ref) -> float:
    y, pred, ref = map(np.asarray, (y, pred, ref))
    a_ref = float(np.sum(np.abs(y - ref)))
    return 1.0 - float(np.sum(np.abs(y - pred))) / a_ref if a_ref > 0 else float("nan")


def _comp_sums(err, comps, uniq):
    idx = {c: i for i, c in enumerate(uniq)}
    s = np.zeros(len(uniq))
    np.add.at(s, [idx[c] for c in comps], err)
    return s


def component_bootstrap(y, preds: dict, comps, ref: str, draws=DRAWS, seed=BOOT_SEED, n_rows=None):
    """{name: {C, ci95, draws, valid_draws}} for every prediction against ``preds[ref]``."""
    y = np.asarray(y, dtype=np.float64)
    comps = np.asarray(comps)
    uniq = sorted(set(comps.tolist()))
    k = len(uniq)
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros((draws, k))
    pick = rng.integers(0, k, size=(draws, k))
    for d in range(draws):
        counts[d] = np.bincount(pick[d], minlength=k)
    rows_per = _comp_sums(np.ones(len(y)), comps, uniq)
    n_draw = counts @ rows_per
    s_ref = _comp_sums(np.abs(y - preds[ref]), comps, uniq)
    a_ref = counts @ s_ref
    valid = (a_ref / np.maximum(n_draw, 1)) > DENOM_TOL
    out = {}
    for name, p in preds.items():
        if name == ref:
            continue
        s = _comp_sums(np.abs(y - p), comps, uniq)
        cs = 1.0 - (counts @ s)[valid] / a_ref[valid]
        lo, hi = np.quantile(cs, [0.025, 0.975]) if cs.size else (np.nan, np.nan)
        out[name] = {
            "C": contrast(y, p, preds[ref]),
            "ci95": (float(lo), float(hi)),
            "draws": int(draws),
            "valid_draws": int(valid.sum()),
        }
    return out


def mz_calibration(y, pred, comps, draws=DRAWS, seed=BOOT_SEED):
    """Mincer–Zarnowitz slope/intercept of y on the prediction, with a component-bootstrap slope interval."""
    y, pred, comps = np.asarray(y), np.asarray(pred), np.asarray(comps)

    def fit(w):
        X = np.column_stack([np.ones(len(pred)), pred]) * np.sqrt(w)[:, None]
        return np.linalg.lstsq(X, y * np.sqrt(w), rcond=None)[0]

    b = fit(np.ones(len(y)))
    uniq = sorted(set(comps.tolist()))
    idx = np.array([uniq.index(c) for c in comps])
    rng = np.random.Generator(np.random.PCG64(seed))
    slopes = []
    for _ in range(draws):
        cnt = np.bincount(rng.integers(0, len(uniq), size=len(uniq)), minlength=len(uniq))
        w = cnt[idx].astype(float)
        if np.ptp(pred[w > 0]) > 0:
            slopes.append(fit(w)[1])
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return {"intercept": float(b[0]), "slope": float(b[1]), "slope_ci95": (float(lo), float(hi))}


def calibration_failure(cal) -> bool:
    """Pinned development rule: the slope interval lies entirely outside [0.5, 2.0]."""
    lo, hi = cal["slope_ci95"]
    return hi < 0.5 or lo > 2.0


def three_way(v, h=H_STAR):
    """+1 / -1 beyond ±h*, 0 = unresolved (not equivalence)."""
    v = np.asarray(v)
    return np.where(v > h, 1, np.where(v < -h, -1, 0))


def label_table(y, pred, h=H_STAR):
    ty, tp = three_way(y, h), three_way(pred, h)
    tab = {f"{a}|{b}": int(np.sum((ty == a) & (tp == b))) for a in (-1, 0, 1) for b in (-1, 0, 1)}
    resolved = (ty != 0) & (tp != 0)
    return {
        "counts_truth|pred": tab,
        "sign_agreement_when_both_resolved": float(np.mean(ty[resolved] == tp[resolved]))
        if resolved.any()
        else None,
        "n_both_resolved": int(resolved.sum()),
        "reading": "labels at h* = 0.0049397 nats/token are descriptive; 0 means unresolved, not equivalence",
    }


MEANING = "development decision, not a scientific null (SYNTHESIS-r7 Decision 1)"


def kill_criteria(c: dict) -> dict:
    """k1, k2, k2b on the pooled contrasts. ``c[key] = {"C": point, "ci95": (lo, hi)}`` for keys
    ``msrm_vs_f3u``, ``msrm_vs_desc``, ``msrm_vs_scripted``. Margin rules for k2/k2b are the implementer's pinned
    reading of "no margin": the point contrast is ≤ 0 or its 95% component-bootstrap lower bound is ≤ 0."""

    def no_margin(key):
        lo = c[key]["ci95"][0]
        return bool(c[key]["C"] <= 0 or lo <= 0)

    f3 = c["msrm_vs_f3u"]
    return {
        "k1": {
            "rule": "C(MSRM-1 : F3u) < 0.05, or its 95% upper bound < 0.10",
            "fires": bool(f3["C"] < 0.05 or f3["ci95"][1] < 0.10),
            "meaning": MEANING,
        },
        "k2": {
            "rule": "no margin over description-only: C(MSRM-1 : desc) <= 0 or its 95% lower bound <= 0",
            "fires": no_margin("msrm_vs_desc"),
            "meaning": MEANING + "; firing reads as 'the gain is a prior'",
        },
        "k2b": {
            "rule": "no margin over the scripted reader: C(MSRM-1 : scripted) <= 0 or its 95% lower bound <= 0",
            "fires": no_margin("msrm_vs_scripted"),
            "meaning": MEANING + "; firing reads as 'the field adds nothing beyond statistics'",
        },
    }
