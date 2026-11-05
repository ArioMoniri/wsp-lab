#!/usr/bin/env python3
"""TF-v1 re-qualification on the D3 stratum and the section 5 gate (PREREG_H1_TF amendments A2.1, A2.3).

Run by the orchestrator once `agent-tmux tfv1-d3-stratum` has finished (it launches no scoring):

    python scripts/tfv1_requalify.py [--panels 4000] [--draws 4000] [--workers 4] [--date YYYY-MM-DD]

Part A, A2.1 on the 229 D3 stratum entries (139 chain, 60 lora_sft, 30 task_vector) scored under one
`source_hash`: item half-widths h_i from the common joint item draw (4,000 draws, seed 20261105),
(i) panel-mean component-bootstrap 97.5th percentile <= delta/2, (ii) max h_i <= delta, (iii) the
stratum-weighted tripwire P <= 0.20 (97.5th percentile of the family-stratified component
bootstrap), (iv) the disclosure list, the repeat bounds from the round-3 G1, and the signal-reliability
lower bound with the nuisance regression refit inside every bootstrap draw (incremental effective
norm). MMLU-Pro gold log-prob (delta .10) is the endpoint; WikiText log-PPL (delta log 1.05) is
evaluated identically.

Part B, A2.3 gate on the D3 out-of-fold losses: the frozen section 3 fit (nuisance OLS with
intercept, family dummies and log1p(incremental effective norm) on training components; StatNN =
LightGBM seeds 0,1,2 averaged before the loss; isotonic on the raw incremental norm; five
source-closed inner folds hashed with `20261105:inner:`; learners trained on inner-OOF residuals),
then power / size / coverage simulations with the registered primary interval (cluster bootstrap-t on
ln R, R = sum A / sum B, 4,000 draws, linearised cluster-robust se), the BCa fallback and the
percentile interval reported alongside, for the conditional (component) and the joint (component x
item) procedures. Scenarios: empirical, item-noise inflated, cluster-merged. A pass is declared only at
the registered scale (4,000 panels x 4,000 draws, both procedures); smaller runs are labelled
INDICATIVE. Writes docs/reviews/P5/tf-v1-requalification-<date>.md and
data/runs/h1a/requalification_report.json.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
SEED, DRAWS = 20261105, 4000
STRATUM = ("lora_sft", "task_vector", "chain")
ENDPOINTS = {  # name: (delta, repeat limit, record benchmark, value field, item block size)
    "mmlu_gold_log_prob": (0.10, 1e-4, "mmlu_pro", "gold_log_prob", 1),
    "wikitext_log_ppl": (math.log(1.05), 1e-4, "wikitext_ppl", "mean_nll", 2),
}
TRIPWIRE, RELIABILITY_FLOOR = 0.20, 0.90
DENOM_TOL = 1e-12  # nats per row (PREREG section 4)
THETA_TARGETS = (0.15, 0.20, 0.25)


# ------------------------------------------------------------------ part A: qualification


def draw_indices(n_items, block, rng, draws=DRAWS):
    """One joint item/block draw shared by every edit and reference (draws x n_items)."""
    n_blocks = n_items // block
    picks = rng.integers(n_blocks, size=(draws, n_blocks))
    return (picks[:, :, None] * block + np.arange(block)).reshape(draws, -1)


def item_halfwidths(diffs, block, seed=SEED, draws=DRAWS):
    """95% half-widths (97.5th - 2.5th percentile)/2 and se of each entry's paired mean difference.

    diffs: (entries x items), the edit-minus-reference item values; one common draw for all entries."""
    index = draw_indices(diffs.shape[1], block, np.random.default_rng(seed), draws)
    half, se = np.empty(len(diffs)), np.empty(len(diffs))
    for i, row in enumerate(diffs):
        boot = row[index].mean(1)
        half[i] = (np.quantile(boot, 0.975) - np.quantile(boot, 0.025)) / 2
        se[i] = boot.std(ddof=1)
    return half, se


def component_picks(codes, draws, seed):
    """Draws x n_components resampling table over the distinct component codes."""
    uniq = sorted(set(codes))
    return uniq, np.random.default_rng(seed).integers(len(uniq), size=(draws, len(uniq)))


def component_bootstrap(stat, codes, draws=DRAWS, seed=SEED):
    codes = np.asarray(codes)
    uniq, pick = component_picks(codes, draws, seed)
    members = {c: np.flatnonzero(codes == c) for c in uniq}
    out = np.full(draws, np.nan)
    for b, row in enumerate(pick):
        idx = np.concatenate([members[uniq[k]] for k in row])
        out[b] = stat(idx)
    return out


def stratum_weights(manifest):
    """w_f: family share of stratum rows of the sealed (confirmatory) partition."""
    rows = Counter()
    for c in manifest["components"]:
        if c["status"] == "confirmatory":
            for fam, n in c["family_counts"].items():
                if fam in STRATUM:
                    rows[fam] += n
    total = sum(rows.values())
    return {f: rows[f] / total for f in STRATUM}, dict(rows)


def tripwire_P(exceed, families, weights):
    """P = sum_f w_f p_f with p_f the exceedance share among family-f panel entries."""
    out = 0.0
    for f in STRATUM:
        sel = np.asarray(families) == f
        if sel.any():
            out += weights[f] * float(np.mean(exceed[sel]))
    return out


def tripwire_upper(exceed, families, codes, weights, draws=DRAWS, seed=SEED):
    """97.5th percentile of P under the family-stratified component bootstrap."""
    exceed, families, codes = np.asarray(exceed), np.asarray(families), np.asarray(codes)
    rng = np.random.default_rng(seed + 2)
    draws_p = np.zeros(draws)
    for f in STRATUM:
        sel = np.flatnonzero(families == f)
        if not len(sel):
            continue
        uniq = sorted(set(codes[sel]))
        members = {c: sel[codes[sel] == c] for c in uniq}
        sizes = np.array([len(members[c]) for c in uniq])
        hits = np.array([exceed[members[c]].sum() for c in uniq])
        pick = rng.integers(len(uniq), size=(draws, len(uniq)))
        draws_p += weights[f] * hits[pick].sum(1) / sizes[pick].sum(1)
    return float(np.quantile(draws_p, 0.975))


def design(families, log_norm, levels=None):
    """Intercept, log1p(norm), treatment-coded family dummies (sorted levels, first = reference)."""
    levels = sorted(set(families)) if levels is None else levels
    cols = [np.ones(len(families)), np.asarray(log_norm, float)]
    cols += [np.array([f == lv for f in families], float) for lv in levels[1:]]
    return np.column_stack(cols)


def ols_residuals(y, x):
    beta = np.linalg.pinv(x) @ y  # Moore-Penrose, float64
    return y - x @ beta, int(np.linalg.matrix_rank(x))


def reliability_refit(y, families, log_norm, error_var, codes, draws=DRAWS, seed=SEED):
    """Reliability = signal / (signal + error) on the nuisance-adjusted residual, nuisance REFIT in
    every component-bootstrap draw (A2.1: not in-sample on the point estimate only)."""
    y, error_var = np.asarray(y, float), np.asarray(error_var, float)
    families = np.asarray(families)
    resid, rank = ols_residuals(y, design(list(families), log_norm))
    total = float(np.var(resid, ddof=1))
    signal = total - float(error_var.mean())
    point = signal / total if total > 0 else float("nan")

    def draw(idx):
        if len(idx) < 4:
            return np.nan
        r, _ = ols_residuals(y[idx], design(list(families[idx]), np.asarray(log_norm)[idx]))
        tot = float(np.var(r, ddof=1))
        return (tot - float(error_var[idx].mean())) / tot if tot > 0 else np.nan

    boots = component_bootstrap(draw, codes, draws, seed + 3)
    lower = float(np.nanquantile(boots, 0.025))
    return {
        "design_rank": rank,
        "residual_variance": total,
        "mean_error_variance": float(error_var.mean()),
        "signal_variance_untruncated": signal,
        "reliability": point,
        "reliability_ci95": [lower, float(np.nanquantile(boots, 0.975))],
        "valid_draws": int(np.isfinite(boots).sum()),
        "lower_bound_threshold": RELIABILITY_FLOOR,
        "pass_lower_bound": bool(lower >= RELIABILITY_FLOOR and signal > 0),
    }


def repeat_bounds(primary, repeat, codes, delta, limit, draws=DRAWS, seed=SEED):
    """primary/repeat: (entries x items) paired differences from the primary and repeat passes."""
    diffs = np.abs(np.asarray(repeat) - np.asarray(primary))
    flat = diffs.ravel()
    codes = np.asarray(codes)
    uniq, pick = component_picks(codes, draws, seed + 1)
    members = {c: np.flatnonzero(codes == c) for c in uniq}
    p95 = np.array(
        [np.quantile(diffs[np.concatenate([members[uniq[k]] for k in row])], 0.95) for row in pick]
    )
    upper = float(np.quantile(p95, 0.975))
    return {
        "entries": int(len(diffs)),
        "max_abs_discrepancy": float(flat.max()),
        "p95_abs_discrepancy": float(np.quantile(flat, 0.95)),
        "p95_upper97_5_component_bootstrap": upper,
        "bitwise_identical_items": bool((flat == 0).all()),
        "limit_max": limit,
        "limit_p95_upper": delta / 2,
        "pass_max": bool(flat.max() <= limit),
        "pass_p95_upper": bool(upper <= delta / 2),
        "repeat_error_variance": float(
            np.mean(np.asarray(repeat).mean(1) - np.asarray(primary).mean(1)) ** 2 / 2
        ),
    }


def a2_item_condition(diffs, families, codes, weights, delta, block, draws=DRAWS, seed=SEED):
    """A2.1 (i)-(iv) on the panel of paired item differences (entries x items)."""
    half, se = item_halfwidths(diffs, block, seed, draws)
    y = diffs.mean(1)
    exceed = half > delta / 2
    mean_half = component_bootstrap(lambda idx: half[idx].mean(), codes, draws, seed)
    upper_mean = float(np.nanquantile(mean_half, 0.975))
    p_point = tripwire_P(exceed, families, weights)
    p_upper = tripwire_upper(exceed, families, codes, weights, draws, seed)
    result = {
        "delta": delta,
        "n_entries": len(y),
        "n_items": int(diffs.shape[1]),
        "i_panel_mean": {
            "mean_halfwidth": float(half.mean()),
            "upper97_5": upper_mean,
            "limit_delta_over_2": delta / 2,
            "pass": bool(upper_mean <= delta / 2),
        },
        "ii_ceiling": {
            "max_halfwidth": float(half.max()),
            "limit_delta": delta,
            "pass": bool(half.max() <= delta),
        },
        "iii_tripwire": {
            "P_point": p_point,
            "P_upper97_5": p_upper,
            "limit": TRIPWIRE,
            "p_by_family": {
                f: float(np.mean(exceed[np.asarray(families) == f]))
                for f in STRATUM
                if (np.asarray(families) == f).any()
            },
            "weights": weights,
            "pass": bool(p_upper <= TRIPWIRE),
        },
        "iv_disclosure_over_half_delta": [
            {
                "index": int(i),
                "family": families[i],
                "y": float(y[i]),
                "halfwidth": float(half[i]),
                "halfwidth_over_abs_y": float(half[i] / abs(y[i])) if y[i] else math.inf,
            }
            for i in np.flatnonzero(exceed)
        ],
        "a1_3_per_entry_every_entry_le_half_delta": bool((~exceed).all()),
        "n_entries_over_half_delta": int(exceed.sum()),
    }
    result["pass"] = bool(
        result["i_panel_mean"]["pass"]
        and result["ii_ceiling"]["pass"]
        and result["iii_tripwire"]["pass"]
    )
    return result, y, half, se


# --------------------------------------------------------------- part B: intervals


def lnR_and_cluster(a_rows, b_rows, comp_of_row, n_comp):
    """Cluster totals of the row losses and the row counts per cluster."""
    ca = np.bincount(comp_of_row, weights=a_rows, minlength=n_comp)
    cb = np.bincount(comp_of_row, weights=b_rows, minlength=n_comp)
    cn = np.bincount(comp_of_row, minlength=n_comp).astype(float)
    return ca, cb, cn


def linearised_se(ca, cb, mult=None):
    """se^2 = n/(n-1) sum_c M_c u_c^2, u_c = A_c/sum A - B_c/sum B (M = replicate multiplicities)."""
    n = ca.shape[-1]
    mult = np.ones(n) if mult is None else mult
    sa, sb = (mult * ca).sum(-1), (mult * cb).sum(-1)
    u = ca / sa[..., None] - cb / sb[..., None]
    return np.sqrt(n / (n - 1) * (mult * u**2).sum(-1))


def theta_interval(ln_lo, ln_hi):
    return 1 - math.exp(ln_hi), 1 - math.exp(ln_lo)


def bca_levels(ln_star, ln_hat, jackknife):
    from scipy.stats import norm

    frac = np.clip(np.mean(ln_star < ln_hat), 1e-6, 1 - 1e-6)
    z0 = norm.ppf(frac)
    d = jackknife.mean() - jackknife
    denom = 6 * (np.sum(d**2) ** 1.5)
    acc = float(np.sum(d**3) / denom) if denom > 0 else 0.0
    out = []
    for alpha in (0.025, 0.975):
        z = norm.ppf(alpha)
        adj = norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        out.append(float(np.quantile(ln_star, np.clip(adj, 0, 1))))
    return out


def cluster_intervals(ca, cb, cn, ln_star, se_star, jackknife_fn=None):
    """Bootstrap-t (primary), BCa and percentile intervals for ln R, plus denominator failures.

    ln_star, se_star: replicate ln R* and se* (NaN = denominator failure)."""
    ln_hat = math.log(ca.sum() / cb.sum())
    se = float(linearised_se(ca, cb))
    ok = np.isfinite(ln_star) & np.isfinite(se_star) & (se_star > 0)
    t = (ln_star[ok] - ln_hat) / se_star[ok]
    q_hi, q_lo = np.quantile(t, 0.975), np.quantile(t, 0.025)
    boot_t = (ln_hat - q_hi * se, ln_hat - q_lo * se)
    pct = (float(np.quantile(ln_star[ok], 0.025)), float(np.quantile(ln_star[ok], 0.975)))
    jack = jackknife_fn() if jackknife_fn else np.array([ln_hat])
    bca = tuple(bca_levels(ln_star[ok], ln_hat, jack))
    return {
        "ln_hat": ln_hat,
        "se": se,
        "bootstrap_t": boot_t,
        "bca": bca,
        "percentile": pct,
        "n_valid": int(ok.sum()),
        "n_failed": int((~ok).sum()),
    }


def conditional_replicates(ca, cb, cn, rng, draws):
    """Component-only bootstrap: multinomial component multiplicities, cluster totals fixed."""
    n = len(ca)
    mult = rng.multinomial(n, np.full(n, 1 / n), size=draws).astype(float)
    sa, sb, rows = mult @ ca, mult @ cb, mult @ cn
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_star = np.log(sa / sb)
        se_star = linearised_se(ca[None, :], cb[None, :], mult)
    bad = (sb / np.maximum(rows, 1) <= DENOM_TOL) | (sa <= 0) | (rows == 0)
    ln_star[bad] = np.nan
    return ln_star, se_star


def jackknife_cluster(ca, cb):
    sa, sb = ca.sum(), cb.sum()
    return np.log((sa - ca) / (sb - cb))


def decide(interval, theta_hat, key):
    lo, hi = interval[key]
    t_lo, t_hi = theta_interval(lo, hi)
    return {
        "theta_interval": (t_lo, t_hi),
        "reject_null": bool(t_lo > 0),
        "half_width": (t_hi - t_lo) / 2,
        "decision_pass_uncorrected": bool(
            theta_hat >= 0.15 and t_lo > 0 and (t_hi - t_lo) / 2 <= theta_hat / 2
        ),
    }


# --------------------------------------------------------------------- joint procedure


def joint_replicates(pop, rows, comp_of_row, n_comp, w_obs, kappa, rng, draws, chunk=100):
    """Two-way (component x item) bootstrap of ln R with the two-way studentiser.

    pop: dict(D=(rows x J) paired item differences, qa, qb = predictions-plus-nuisance centres);
    rows: population row index of each panel row; w_obs: observed item multiplicities (J,).
    Loss A_i = kappa |ybar_i - qa_i|, B_i = |ybar_i - qb_i| with ybar the item-weighted mean.
    se*^2 = V_comp + V_item - V_both, with V_both the cell (component, item) intersection variance."""
    D = pop["D"][rows]
    J = D.shape[1]
    order = np.argsort(comp_of_row, kind="stable")
    D, qa, qb, comp = D[order], pop["qa"][rows][order], pop["qb"][rows][order], comp_of_row[order]
    starts = np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]])
    n = len(starts)
    sizes = np.diff(np.r_[starts, len(comp)])
    comp_idx = np.repeat(np.arange(n), sizes)
    ln_star, se_star = np.full(draws, np.nan), np.full(draws, np.nan)
    for lo in range(0, draws, chunk):
        b = min(chunk, draws - lo)
        mult_c = rng.multinomial(n, np.full(n, 1 / n), size=b).astype(float)
        mult_j = rng.multinomial(J, np.full(J, 1 / J), size=b).astype(float)
        ybar = mult_j @ D.T / J  # b x rows
        a, bb = ybar - qa, ybar - qb
        abs_a, abs_b = np.abs(a), np.abs(bb)
        row_mult = mult_c[:, comp_idx]
        sa0, sb0 = (row_mult * abs_a).sum(1), (row_mult * abs_b).sum(1)
        rows_n = (row_mult).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_star[lo : lo + b] = np.log(kappa * sa0 / sb0)
            ca = np.add.reduceat(abs_a, starts, axis=1)
            cb = np.add.reduceat(abs_b, starts, axis=1)
            u = ca / sa0[:, None] - cb / sb0[:, None]
            v_comp = n / (n - 1) * (mult_c * u**2).sum(1)
            g = np.sign(a) / sa0[:, None] - np.sign(bb) / sb0[:, None]  # b x rows
            lam = (row_mult * g) @ D  # b x J
            p = mult_j / J
            lam_bar = (p * lam).sum(1)
            v_item = (p * (lam - lam_bar[:, None]) ** 2).sum(1) / (J - 1)
            h = np.sign(a) * qa / sa0[:, None] - np.sign(bb) * qb / sb0[:, None]
            h_c = np.add.reduceat(h, starts, axis=1)  # b x C
            v_both = np.zeros(b)
            for r in range(b):
                xi = np.add.reduceat(g[r][:, None] * D, starts, axis=0) - h_c[r][:, None]
                v_both[r] = (mult_c[r] * ((xi**2) @ mult_j[r])).sum() / J**2
            nn = n * J
            v_both *= nn / (nn - 1)
            var = v_comp + v_item - v_both
            se_star[lo : lo + b] = np.sqrt(np.where(var > 0, var, np.nan))
        bad = (sb0 / np.maximum(rows_n, 1) <= DENOM_TOL) | (sa0 <= 0)
        ln_star[lo : lo + b][bad] = np.nan
    return ln_star, se_star


def joint_point(pop, rows, comp_of_row, n_comp, w_obs, kappa):
    """Observed-panel cluster totals and the two-way se at the observed item weights."""
    D = pop["D"][rows]
    J = D.shape[1]
    ybar = (w_obs @ D.T) / w_obs.sum()
    a, b = ybar - pop["qa"][rows], ybar - pop["qb"][rows]
    ca = np.bincount(comp_of_row, weights=kappa * np.abs(a), minlength=n_comp)
    cb = np.bincount(comp_of_row, weights=np.abs(b), minlength=n_comp)
    cn = np.bincount(comp_of_row, minlength=n_comp).astype(float)
    return ca, cb, cn


# ----------------------------------------------------------------------- simulation


def sealed_structure(manifest):
    """(rows per sealed component, class) for the confirmatory components: metadata only."""
    sizes, classes = [], []
    for c in manifest["components"]:
        if c["status"] == "confirmatory":
            sizes.append(len(c["stratum_rows"]))
            fam = c["family_counts"]
            classes.append("mixed" if fam.get("lora_sft") or fam.get("task_vector") else "chain")
    return np.array(sizes), classes


def donor_pools(pop_codes, pop_families):
    pools = defaultdict(lambda: defaultdict(list))
    for r, (code, fam) in enumerate(zip(pop_codes, pop_families, strict=True)):
        pools[code][fam].append(r)
    out = {"mixed": [], "chain": []}
    for code, fams in pools.items():
        rows = sorted(r for rs in fams.values() for r in rs)
        kind = "mixed" if (fams.get("lora_sft") or fams.get("task_vector")) else "chain"
        out[kind].append(rows)
    return out


def make_panel(rng, sizes, classes, pools, merge=False):
    """Row indices (into the D3 population) and component index per row of one simulated panel.

    Each sealed component draws a donor D3 component of its class (with replacement) and takes the
    donor's rows, cycled/truncated to the sealed row count: empirical within-component structure."""
    rows, comp = [], []
    sizes, classes = list(sizes), list(classes)
    for k, (n_rows, cls) in enumerate(zip(sizes, classes, strict=True)):
        pool = pools[cls] or pools["mixed" if cls == "chain" else "chain"]
        donor = pool[rng.integers(len(pool))]
        take = [donor[j % len(donor)] for j in range(n_rows)]
        rows += take
        comp += [k // 2 if merge else k] * n_rows
    return np.array(rows), np.array(comp)


def calibrate_kappa(pop, theta, weights=None):
    """kappa with sum(kappa A)/sum(B) = 1 - theta over the D3 population at the full item set."""
    a = np.abs(pop["D"].mean(1) - pop["qa"])
    b = np.abs(pop["D"].mean(1) - pop["qb"])
    return (1 - theta) * b.sum() / a.sum()


def inflate_items(pop, factor):
    """Item-noise sensitivity: scale each row's item deviations about its mean by `factor`."""
    mean = pop["D"].mean(1, keepdims=True)
    return {**pop, "D": mean + factor * (pop["D"] - mean)}


def one_panel(args):
    """One simulated panel: conditional and joint intervals; returns the per-panel outcome."""
    (pop, sizes, classes, pools, kappa, merge, seed, draws, world) = args
    rng = np.random.default_rng(seed)
    rows, comp = make_panel(rng, sizes, classes, pools, merge)
    n_comp = int(comp.max()) + 1
    J = pop["D"].shape[1]
    if world == "conditional":  # items fixed: the conditional claim
        w = np.ones(J)
        ca, cb, cn = joint_point(pop, rows, comp, n_comp, w, kappa)
        ln_star, se_star = conditional_replicates(ca, cb, cn, rng, draws)
        iv = cluster_intervals(ca, cb, cn, ln_star, se_star, lambda: jackknife_cluster(ca, cb))
    else:  # fresh sealed items per panel: the joint claim
        w = rng.multinomial(J, np.full(J, 1 / J)).astype(float)
        ca, cb, cn = joint_point(pop, rows, comp, n_comp, w, kappa)
        # joint se at the observed sample uses the same two-way formula with unit multiplicities.
        ln_star, se_star = joint_replicates(pop, rows, comp, n_comp, w, kappa, rng, draws)
        iv = cluster_intervals(ca, cb, cn, ln_star, se_star, lambda: jackknife_cluster(ca, cb))
        iv["se"] = float(joint_observed_se(pop, rows, comp, n_comp, w, kappa))
        t = (ln_star - iv["ln_hat"]) / se_star
        ok = np.isfinite(t)
        qh, ql = np.quantile(t[ok], 0.975), np.quantile(t[ok], 0.025)
        iv["bootstrap_t"] = (iv["ln_hat"] - qh * iv["se"], iv["ln_hat"] - ql * iv["se"])
    theta_hat = 1 - math.exp(iv["ln_hat"])
    out = {"theta_hat": theta_hat, "n_failed": iv["n_failed"], "n_draws": draws}
    for key in ("bootstrap_t", "bca", "percentile"):
        out[key] = decide(iv, theta_hat, key)
    out["ln_hat"] = iv["ln_hat"]
    return out


def joint_observed_se(pop, rows, comp_of_row, n_comp, w_obs, kappa):
    """Two-way studentiser at the observed sample (unit component multiplicities, item weights w)."""
    return joint_replicates_once(pop, rows, comp_of_row, w_obs, kappa)[1]


def joint_replicates_once(pop, rows, comp_of_row, w_obs, kappa):
    D = pop["D"][rows]
    J = D.shape[1]
    order = np.argsort(comp_of_row, kind="stable")
    D, qa, qb, comp = D[order], pop["qa"][rows][order], pop["qb"][rows][order], comp_of_row[order]
    starts = np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]])
    n = len(starts)
    mult_j = w_obs
    ybar = D @ mult_j / J
    a, bb = ybar - qa, ybar - qb
    sa0, sb0 = np.abs(a).sum(), np.abs(bb).sum()
    ca, cb = np.add.reduceat(np.abs(a), starts), np.add.reduceat(np.abs(bb), starts)
    u = ca / sa0 - cb / sb0
    v_comp = n / (n - 1) * (u**2).sum()
    g = np.sign(a) / sa0 - np.sign(bb) / sb0
    lam = g @ D
    p = mult_j / J
    v_item = (p * (lam - (p * lam).sum()) ** 2).sum() / (J - 1)
    h = np.sign(a) * qa / sa0 - np.sign(bb) * qb / sb0
    h_c = np.add.reduceat(h, starts)
    xi = np.add.reduceat(g[:, None] * D, starts, axis=0) - h_c[:, None]
    nn = n * J
    v_both = nn / (nn - 1) * ((xi**2) @ mult_j).sum() / J**2
    var = v_comp + v_item - v_both
    return math.log(kappa * sa0 / sb0), math.sqrt(max(var, 1e-300))


def wilson_lower(k, n):
    """Lower 95% Wilson bound of a binomial proportion (as in the planning power script)."""
    if n == 0:
        return float("nan")
    z, p = 1.959964, k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n)


def summarise(outcomes, procedure_key, theta_true):
    rej = np.mean([o[procedure_key]["reject_null"] for o in outcomes])
    cover = np.mean(
        [
            o[procedure_key]["theta_interval"][0]
            <= theta_true
            <= o[procedure_key]["theta_interval"][1]
            for o in outcomes
        ]
    )
    passed = np.mean([o[procedure_key]["decision_pass_uncorrected"] for o in outcomes])
    k = int(round(rej * len(outcomes)))
    fails = sum(o["n_failed"] for o in outcomes)
    total = sum(o["n_draws"] for o in outcomes)
    return {
        "panels": len(outcomes),
        "rejection": float(rej),
        "rejection_wilson_lower95": wilson_lower(k, len(outcomes)),
        "coverage95": float(cover),
        "mean_theta_hat": float(np.mean([o["theta_hat"] for o in outcomes])),
        "mean_half_width": float(np.mean([o[procedure_key]["half_width"] for o in outcomes])),
        "p_decision_pass": float(passed),
        "denominator_failure_fraction": fails / total if total else float("nan"),
    }


def population_theta(pop, sizes, classes, pools, kappa, merge, seed, n_panels=1000):
    """True theta of the DGP: 1 - E[sum A]/E[sum B] over simulated panels at the full item set."""
    rng = np.random.default_rng(seed)
    a = np.abs(pop["D"].mean(1) - pop["qa"]) * kappa
    b = np.abs(pop["D"].mean(1) - pop["qb"])
    sa = sb = 0.0
    for _ in range(n_panels):
        rows, _ = make_panel(rng, sizes, classes, pools, merge)
        sa += a[rows].sum()
        sb += b[rows].sum()
    return 1 - sa / sb


def run_scenarios(pop, manifest, panels, joint_panels, draws, workers, scenarios, log):
    sizes, classes = sealed_structure(manifest)
    pools = donor_pools(pop["codes"], pop["families"])
    results = {}
    for name, (item_factor, merge) in scenarios.items():
        sp = inflate_items(pop, item_factor) if item_factor != 1 else pop
        results[name] = {}
        for theta in (0.0, *THETA_TARGETS):
            kappa = calibrate_kappa(sp, theta)
            true = population_theta(sp, sizes, classes, pools, kappa, merge, SEED + 99)
            entry = {"kappa": kappa, "theta_true_population": true}
            for world, n_panels in (("conditional", panels), ("joint", joint_panels)):
                t0 = time.monotonic()
                jobs = [
                    (sp, sizes, classes, pools, kappa, merge, SEED + 1000 * k + 7, draws, world)
                    for k in range(n_panels)
                ]
                if workers > 1:
                    with ProcessPoolExecutor(workers) as ex:
                        outcomes = list(ex.map(one_panel, jobs, chunksize=4))
                else:
                    outcomes = [one_panel(j) for j in jobs]
                entry[world] = {
                    key: summarise(outcomes, key, true)
                    for key in ("bootstrap_t", "bca", "percentile")
                }
                entry[world]["seconds"] = time.monotonic() - t0
                log(
                    f"{name} theta={theta} {world}: {n_panels} panels in {time.monotonic() - t0:.0f}s"
                )
            results[name][f"theta_{theta}"] = entry
    return results


def gate_verdict(results, panels, joint_panels, draws, qualification_pass):
    """(a)-(e) of A2.3 for the primary interval, falling back to BCa on simulated data only."""
    registered = panels >= 4000 and joint_panels >= 4000 and draws >= 4000

    def checks(method):
        out = {}
        for scen, by in results.items():
            null, eff = by["theta_0.0"], by["theta_0.15"]
            for world in ("conditional", "joint"):
                out[(scen, world, "null_size_le_0.035")] = null[world][method]["rejection"] <= 0.035
                out[(scen, world, "coverage_ge_0.93")] = eff[world][method]["coverage95"] >= 0.93
                out[(scen, world, "denominator_failures_lt_0.1pct")] = (
                    null[world][method]["denominator_failure_fraction"] < 0.001
                    and eff[world][method]["denominator_failure_fraction"] < 0.001
                )
        # power is evaluated in the empirical scenario only
        for world in ("conditional", "joint"):
            emp = results["empirical"]["theta_0.15"][world][method]
            out[("empirical", world, "power_lower95_ge_0.80")] = (
                emp["rejection_wilson_lower95"] >= 0.80
            )
        return out

    verdict = {"registered_scale": registered, "methods": {}}
    for method in ("bootstrap_t", "bca"):
        c = checks(method)
        verdict["methods"][method] = {
            "checks": {"/".join(k): bool(v) for k, v in c.items()},
            "all_simulation_checks_pass": bool(all(c.values())),
        }
    chosen = next(
        (m for m in ("bootstrap_t", "bca") if verdict["methods"][m]["all_simulation_checks_pass"]),
        None,
    )
    verdict["method_chosen"] = chosen
    verdict["qualification_pass"] = bool(qualification_pass)
    verdict["confirmatory"] = bool(chosen and qualification_pass and registered)
    verdict["status"] = (
        "CONFIRMATORY (gate passed)"
        if verdict["confirmatory"]
        else "EXPLORATORY (power-limited)"
        if registered
        else "INDICATIVE ONLY (below registered simulation scale; status stays exploratory)"
    )
    return verdict


# ------------------------------------------------------------------ real-data loading


def load_module(name):
    path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_values(directory, benchmark, field):
    rows = [json.loads(x) for x in (Path(directory) / "records.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["benchmark"] == benchmark]
    return [r["item_id"] for r in rows], np.array([r["extra"][field] for r in rows])


def effective_increment_norms(rows, ids, zoo, features_dir, cache_path):
    """||Delta_eff||_F measured against the actual predecessor (chain: parent state; others: base).

    Non-chain entries and chain step 0: the effective-bf16 norm from the weight-only feature table
    (relative to base). Chain steps with a parent: ||bf16(W + D_i) - bf16(W + D_parent)||_F from the
    stored deltas and the base weights (outcome-free). Cached in cache_path."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = []
    out = {}
    for i in ids:
        row = rows[i]
        if row["family"] == "chain" and row.get("parent_id") in rows:
            if i in cache:
                out[i] = cache[i]
            else:
                todo.append(i)
        else:
            meta = json.loads((features_dir / i / "features.json").read_text())
            out[i] = float(meta["norms"]["fro_effective"])
    if todo:
        import torch
        from safetensors import safe_open

        from wsp.eval.zoo_compat import load_entry_delta

        header = json.loads((zoo / "manifest.jsonl").read_text().splitlines()[0])
        from huggingface_hub import snapshot_download

        snap = Path(
            snapshot_download(
                header["base_model"], revision=header["base_revision"], local_files_only=True
            )
        )
        base = {}
        for f in sorted(snap.glob("*.safetensors")):
            with safe_open(f, "pt") as stream:
                for k in stream.keys():  # noqa: SIM118
                    base[k] = None  # filled lazily below
        keys = list(base)

        def tensor(name):
            for f in sorted(snap.glob("*.safetensors")):
                with safe_open(f, "pt") as stream:
                    if name in stream.keys():
                        return stream.get_tensor(name)
            raise KeyError(name)

        def merged(delta, name):
            w = tensor(name).float()
            d = delta.get(name)
            return (
                (w + d.float()).to(torch.bfloat16) if d is not None else w.to(torch.bfloat16)
            ).float()

        for n, i in enumerate(todo):
            di = load_entry_delta(zoo / i, keys)
            dp = load_entry_delta(zoo / rows[i]["parent_id"], keys)
            total = 0.0
            for name in set(di) | set(dp):
                total += float((merged(di, name) - merged(dp, name)).pow(2).sum())
            out[i] = cache[i] = math.sqrt(total)
            if n % 20 == 0:
                cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))
    return out


# ----------------------------------------------------------------------- D3 fit (OOF)


def inner_folds(component_ids, k=5):
    ordered = sorted(
        set(component_ids),
        key=lambda c: (hashlib.sha256(f"{SEED}:inner:{c}".encode()).hexdigest(), c),
    )
    return {c: n % k for n, c in enumerate(ordered)}


def fit_oof(features, y, families, inc_norm, codes, seeds=(0, 1, 2)):
    """Frozen section 3 pipeline on development rows; returns OOF nuisance, residual, StatNN, isotonic."""
    from sklearn.isotonic import IsotonicRegression

    from wsp.predictor.baselines import _make_model

    y = np.asarray(y, float)
    fold_of = inner_folds(codes)
    fold = np.array([fold_of[c] for c in codes])
    log_norm = np.log1p(np.asarray(inc_norm, float))
    n = len(y)
    m = np.zeros(n)
    for k in sorted(set(fold)):
        train, test = fold != k, fold == k
        levels = sorted(set(np.asarray(families)[train]))
        xtr = design(list(np.asarray(families)[train]), log_norm[train], levels)
        beta = np.linalg.pinv(xtr) @ y[train]
        fam_test = list(np.asarray(families)[test])
        unseen = set(fam_test) - set(levels)
        if unseen:
            raise ValueError(f"unseen family level in fold {k}: {sorted(unseen)} (fails closed)")
        m[test] = design(fam_test, log_norm[test], levels) @ beta
    resid = y - m
    p_nn, p_iso = np.zeros(n), np.zeros(n)
    for k in sorted(set(fold)):
        train, test = fold != k, fold == k
        preds = []
        for s in seeds:
            preds.append(
                _make_model("statnn", s).fit(features[train], resid[train]).predict(features[test])
            )
        p_nn[test] = np.mean(preds, axis=0)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(np.asarray(inc_norm, float)[train], resid[train])
        p_iso[test] = iso.predict(np.asarray(inc_norm, float)[test])
    return {"fold": fold, "m": m, "resid": resid, "p_nn": p_nn, "p_iso": p_iso}


# --------------------------------------------------------------------------- report


def fmt(x, n=4):
    return f"{x:.{n}g}" if isinstance(x, float) else str(x)


def yes(flag):
    return "PASS" if flag else "FAIL"


def render(report):
    q, g = report["qualification"], report.get("gate")
    lines = [
        f"# TF-v1 re-qualification on the D3 stratum and the section 5 gate ({report['date']})\n",
        f"Generated by `scripts/tfv1_requalify.py` (PREREG_H1_TF amendments A2.1 and A2.3). Source `source_hash` `{report['source_hash'][:12]}`, partition seal `{report['partition_sha256'][:12]}`, "
        f"{report['n_stratum']} D3 stratum entries ({report['family_counts']}), device H0 only. Development panel; nothing here is a fresh outcome.\n",
        "## A. MMLU-Pro qualification under A2.1\n",
    ]
    for name, res in q.items():
        lines += [
            f"### {name} (delta {fmt(res['delta'])}, {res['n_entries']} entries, {res['n_items']} items)\n",
            "| Condition | Value | Limit | Result |",
            "|---|---:|---:|---|",
            f"| (i) panel-mean half-width, upper 97.5% | {fmt(res['i_panel_mean']['upper97_5'])} | {fmt(res['i_panel_mean']['limit_delta_over_2'])} | {yes(res['i_panel_mean']['pass'])} |",
            f"| (ii) max half-width | {fmt(res['ii_ceiling']['max_halfwidth'])} | {fmt(res['ii_ceiling']['limit_delta'])} | {yes(res['ii_ceiling']['pass'])} |",
            f"| (iii) tripwire P, upper 97.5% (point {fmt(res['iii_tripwire']['P_point'])}) | {fmt(res['iii_tripwire']['P_upper97_5'])} | {res['iii_tripwire']['limit']} | {yes(res['iii_tripwire']['pass'])} |",
            f"| repeat max discrepancy | {fmt(res['repeat']['max_abs_discrepancy'])} | {res['repeat']['limit_max']} | {yes(res['repeat']['pass_max'])} |",
            f"| repeat p95 upper bound | {fmt(res['repeat']['p95_upper97_5_component_bootstrap'])} | {fmt(res['repeat']['limit_p95_upper'])} | {yes(res['repeat']['pass_p95_upper'])} |",
            f"| reliability lower 95% (refit per draw; point {fmt(res['reliability']['reliability'])}) | {fmt(res['reliability']['reliability_ci95'][0])} | {RELIABILITY_FLOOR} | {yes(res['reliability']['pass_lower_bound'])} |",
            f"\n**{name}: {'QUALIFIED' if res['qualified'] else 'NOT QUALIFIED'}** (A2.1 conditions and repeat and reliability bounds). "
            f"A1.3 per-entry reading (every h_i <= delta/2): {yes(res['a1_3_per_entry_every_entry_le_half_delta'])}, {res['n_entries_over_half_delta']} entries over delta/2.\n",
        ]
        disclosed = res["iv_disclosure_over_half_delta"]
        if disclosed:
            lines += [
                "Entries with h_i > delta/2 (A2.1 iv):\n",
                "| Entry | Family | y | h | h/\\|y\\| |",
                "|---|---|---:|---:|---:|",
            ]
            lines += [
                f"| `{d['entry']}` | {d['family']} | {fmt(d['y'])} | {fmt(d['halfwidth'])} | {fmt(d['halfwidth_over_abs_y'])} |"
                for d in disclosed
            ]
            lines.append("")
    if g:
        v = g["verdict"]
        lines += [
            "## B. Section 5 gate on the D3 out-of-fold losses (A2.3)\n",
            f"**Status: {v['status']}.** Simulation scale: {g['panels']} conditional panels, {g['joint_panels']} joint panels, {g['draws']} bootstrap draws (registered: 4,000 / 4,000 / 4,000). Interval chosen on simulated data: `{v['method_chosen']}`.\n",
            f"D3 fit: {g['fit']['n_rows']} development rows ({g['fit']['n_stratum']} stratum + {g['fit']['n_null']} null-family controls), five inner folds, StatNN seeds 0,1,2 averaged before the loss, isotonic on incremental norm. "
            f"Empirical OOF row-summed ratio on the stratum rows: theta_hat = {fmt(g['fit']['theta_hat_stratum'])}; n_eff = {fmt(g['fit']['n_eff'])}; top-five component share of sum B = {fmt(g['fit']['top5_share'])}.\n",
            "| Scenario | theta (true) | procedure | method | rejection | power lower 95% | coverage | mean half-width | P(decision pass) | denom. failures |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for scen, by in g["results"].items():
            for key, entry in by.items():
                for world in ("conditional", "joint"):
                    for method in ("bootstrap_t", "bca", "percentile"):
                        s = entry[world][method]
                        lines.append(
                            f"| {scen} | {fmt(entry['theta_true_population'], 3)} | {world} | {method} | {fmt(s['rejection'])} | {fmt(s['rejection_wilson_lower95'])} | {fmt(s['coverage95'])} | {fmt(s['mean_half_width'])} | {fmt(s['p_decision_pass'])} | {fmt(s['denominator_failure_fraction'])} |"
                        )
        lines += ["", "Gate checks (A2.3 a-c, simulated data only):\n"]
        for method, res in v["methods"].items():
            lines.append(
                f"- `{method}`: all simulation checks {yes(res['all_simulation_checks_pass'])}"
            )
            lines += [f"  - {k}: {yes(ok)}" for k, ok in res["checks"].items()]
        lines += [
            "",
            f"(d) qualification (A2.1) passes: {yes(v['qualification_pass'])}.",
            "(e) P(decision pass) at theta = 0.15, 0.20, 0.25 is in the table above (rows labelled by the population theta).",
            "",
            "Limitation: the D3 fit uses the 229 stratum rows plus the 160 null-family controls, a small training set for the StatNN-versus-isotonic comparison. Residual learners are trained on inner-OOF residuals, so the OOF nuisance of a training row has seen held-out rows (the registered nested-fold design, not a leak of the held-out label into its own prediction).",
        ]
    return "\n".join(lines) + "\n"


def observed_intervals(pop, rows, comp, kappa=1.0, draws=DRAWS, seed=SEED):
    """Conditional (items fixed) and joint bootstrap-t intervals of theta on one fixed panel."""
    rng = np.random.default_rng(seed)
    n_comp = int(comp.max()) + 1
    J = pop["D"].shape[1]
    ones = np.ones(J)
    ca, cb, cn = joint_point(pop, rows, comp, n_comp, ones, kappa)
    ln_c, se_c = conditional_replicates(ca, cb, cn, rng, draws)
    cond = cluster_intervals(ca, cb, cn, ln_c, se_c, lambda: jackknife_cluster(ca, cb))
    ln_j, se_j = joint_replicates(pop, rows, comp, n_comp, ones, kappa, rng, draws)
    joint = cluster_intervals(ca, cb, cn, ln_j, se_j, lambda: jackknife_cluster(ca, cb))
    ln_hat, se_obs = joint_replicates_once(pop, rows, comp, ones, kappa)
    t = (ln_j - ln_hat) / se_j
    t = t[np.isfinite(t)]
    joint["se"] = se_obs
    joint["bootstrap_t"] = (
        ln_hat - np.quantile(t, 0.975) * se_obs,
        ln_hat - np.quantile(t, 0.025) * se_obs,
    )
    out = {"theta_hat": 1 - math.exp(cond["ln_hat"])}
    for name, iv in (("conditional", cond), ("joint", joint)):
        for method in ("bootstrap_t", "bca", "percentile"):
            out[f"{name}_{method}"] = list(theta_interval(*iv[method]))
    return out


def assemble(args, log):
    plan = load_module("tfv1_scoring_plan")
    manifest = json.loads(args.manifest.read_text())
    rows = plan.load_zoo_rows()
    amap = plan.canonical_map(manifest["alias_ledger"])
    evaluated = plan.evaluation_membership(rows)
    sealed = manifest["snapshot"]["evaluation_membership_sha256"]
    if plan.membership_sha(evaluated) != sealed:
        raise SystemExit("D3 evaluation membership differs from the sealed snapshot")
    p = plan.plan(manifest, rows, evaluated)
    component_of = {m: c["component_id"] for c in manifest["components"] for m in c["members"]}
    tf = args.tf_dir
    base_meta = json.loads((tf / "base/eval.json").read_text())
    source_hash = base_meta["source_hash"]

    def meta(directory):
        m = json.loads((Path(directory) / "eval.json").read_text())
        if m["eval_status"] != "complete" or m["source_hash"] != source_hash or m["dirty"]:
            raise SystemExit(f"{directory}: incomplete, dirty or a different source_hash")
        return m

    scored = lambda i: plan.scored_id(i, amap)
    ref_of = lambda i: plan.reference_of(i, rows, amap)
    data = {}

    def values(edit_id, endpoint, label=None):
        key = (edit_id, endpoint, label)
        if key not in data:
            directory = tf / edit_id if label is None else tf / edit_id / "repeats" / label
            meta(directory)
            _, bench, field = ENDPOINTS[endpoint][2], ENDPOINTS[endpoint][2], ENDPOINTS[endpoint][3]
            ids, v = read_values(directory, bench, field)
            data[key] = (ids, v)
        return data[key]

    def paired(i, endpoint, label=None):
        ids_e, ve = values(scored(i), endpoint, label)
        ref = ref_of(i)
        ids_r, vr = values("base" if ref == "base" else ref, endpoint, label)
        if ids_e != ids_r:
            raise SystemExit(f"{i}: item ids differ between the edit and its reference")
        return ve - vr

    stratum = [i for i in p["d3_stratum_entries"]]
    missing = [i for i in stratum if not (tf / scored(i) / "eval.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} stratum entries are not scored yet, e.g. {missing[:3]}")
    families = [rows[i]["family"] for i in stratum]
    codes = [component_of[i] for i in stratum]
    log(f"{len(stratum)} stratum entries, {len(set(codes))} components, source {source_hash[:12]}")
    inc = effective_increment_norms(
        rows,
        sorted(set(stratum) | set(p["d3_null_entries"])),
        ZOO,
        args.features,
        ROOT / "data/runs/h1a/incremental_norms.json",
    )
    weights, _ = stratum_weights(manifest)
    qual_panel = json.loads((ROOT / "docs/prereg/qualification_panel.json").read_text())
    g1_entries = qual_panel["entries"]
    g1_ref = dict(
        zip(qual_panel["composition"]["chain"], qual_panel["chain_predecessors"], strict=True)
    )
    report = {
        "date": args.date,
        "source_hash": source_hash,
        "partition_sha256": manifest["partition_sha256"],
        "n_stratum": len(stratum),
        "family_counts": dict(Counter(families)),
        "stratum_weights": weights,
        "qualification": {},
    }
    for endpoint, (delta, limit, _, _, block) in ENDPOINTS.items():
        diffs = np.array([paired(i, endpoint) for i in stratum])
        res, y, half, se = a2_item_condition(diffs, families, codes, weights, delta, block)
        for d in res["iv_disclosure_over_half_delta"]:
            d["entry"] = stratum[d["index"]]
        # repeat bounds from the round-3 G1 (primary vs repeat of the section 6 composition)
        prim, rep, rep_codes = [], [], []
        for i in g1_entries:
            ref = g1_ref.get(i, "base")
            pe, re_ = values(i, endpoint)[1], values(i, endpoint, "g1")[1]
            pr = values(ref, endpoint)[1]
            rr = values(ref, endpoint, "g1")[1]
            prim.append(pe - pr)
            rep.append(re_ - rr)
            rep_codes.append(component_of[i])
        res["repeat"] = repeat_bounds(np.array(prim), np.array(rep), rep_codes, delta, limit)
        err = se**2 + res["repeat"]["repeat_error_variance"]
        res["reliability"] = reliability_refit(
            y, families, np.log1p([inc[i] for i in stratum]), err, codes
        )
        res["qualified"] = bool(
            res["pass"]
            and res["repeat"]["pass_max"]
            and res["repeat"]["pass_p95_upper"]
            and res["reliability"]["pass_lower_bound"]
        )
        report["qualification"][endpoint] = res
        report.setdefault("_arrays", {})[endpoint] = (diffs, y, half)
        log(f"{endpoint}: qualified={res['qualified']}")
    return report, dict(
        plan=p,
        manifest=manifest,
        rows=rows,
        stratum=stratum,
        families=families,
        codes=codes,
        inc=inc,
        paired=paired,
        component_of=component_of,
        scored=scored,
        tf=tf,
    )


def d3_fit(ctx, args, log):
    """Frozen section 3 fit on all complete D3 rows; returns the population of stratum rows."""
    from wsp.predictor.baselines import load_feature_table

    rows, stratum, tf, scored = ctx["rows"], ctx["stratum"], ctx["tf"], ctx["scored"]
    null = [i for i in ctx["plan"]["d3_null_entries"] if (tf / scored(i) / "eval.json").exists()]
    entries = stratum + null
    feats = {r["id"]: r["statnn"] for r in load_feature_table(args.features)}
    missing = [i for i in entries if i not in feats]
    if missing:
        raise SystemExit(f"{len(missing)} D3 entries without weight-only features")
    X = np.stack([feats[i] for i in entries])
    y = np.array([ctx["paired"](i, "mmlu_gold_log_prob").mean() for i in entries])
    fams = [rows[i]["family"] for i in entries]
    codes = [ctx["component_of"][i] for i in entries]
    inc = [ctx["inc"][i] for i in entries]
    fit = fit_oof(X, y, fams, inc, codes)
    n = len(stratum)
    D = np.array([ctx["paired"](i, "mmlu_gold_log_prob") for i in stratum])
    qa = fit["m"][:n] + fit["p_nn"][:n]
    qb = fit["m"][:n] + fit["p_iso"][:n]
    a = np.abs(fit["resid"][:n] - fit["p_nn"][:n])
    b = np.abs(fit["resid"][:n] - fit["p_iso"][:n])
    cb = np.array([b[np.array(ctx["codes"]) == c].sum() for c in sorted(set(ctx["codes"]))])
    pop = {"D": D, "qa": qa, "qb": qb, "codes": ctx["codes"], "families": ctx["families"]}
    info = {
        "n_rows": len(entries),
        "n_stratum": n,
        "n_null": len(null),
        "null_family_complete": len(null) == len(ctx["plan"]["d3_null_entries"]),
        "theta_hat_stratum": float(1 - a.sum() / b.sum()),
        "n_eff": float(cb.sum() ** 2 / (cb**2).sum()),
        "top5_share": float(np.sort(cb)[-5:].sum() / cb.sum()),
    }
    log(f"D3 fit: theta_hat(stratum) = {info['theta_hat_stratum']:.4f}")
    return pop, info, fit


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--panels", type=int, default=4000, help="conditional simulation panels")
    parser.add_argument(
        "--joint-panels", type=int, default=None, help="joint panels (default: --panels)"
    )
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--skip-gate", action="store_true", help="Part A only")
    parser.add_argument("--tf-dir", type=Path, default=ROOT / "data/evals/qwen3.5-0.8b/tf_v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/prereg/panel_manifest.json")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/qwen3.5-0.8b")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    joint_panels = args.joint_panels or args.panels
    report, ctx = assemble(args, log)
    report.pop("_arrays")
    report["panels"], report["joint_panels"], report["draws"] = (
        args.panels,
        joint_panels,
        args.draws,
    )
    if not args.skip_gate:
        pop, info, _ = d3_fit(ctx, args, log)
        over = set()
        for res in report["qualification"].values():
            over |= {d["entry"] for d in res["iv_disclosure_over_half_delta"]}
        keep = np.array([i not in over for i in ctx["stratum"]])
        comp_idx = {c: k for k, c in enumerate(sorted(set(ctx["codes"])))}
        row_comp = np.array([comp_idx[c] for c in ctx["codes"]])
        sens = {}
        for label, sel in (
            ("all_rows", np.ones(len(keep), bool)),
            ("without_over_half_delta", keep),
        ):
            rows_sel = np.flatnonzero(sel)
            _, comp_sel = np.unique(row_comp[rows_sel], return_inverse=True)
            sens[label] = observed_intervals(pop, rows_sel, comp_sel, 1.0, args.draws)
        scenarios = {
            "empirical": (1.0, False),
            "item_noise_x1.5": (1.5, False),
            "components_merged_in_pairs": (1.0, True),
        }
        results = run_scenarios(
            pop,
            ctx["manifest"],
            args.panels,
            joint_panels,
            args.draws,
            args.workers,
            scenarios,
            log,
        )
        qual_ok = report["qualification"]["mmlu_gold_log_prob"]["qualified"]
        info["leave_out_sensitivity"] = sens
        report["gate"] = {
            "panels": args.panels,
            "joint_panels": joint_panels,
            "draws": args.draws,
            "fit": info,
            "results": results,
            "verdict": gate_verdict(results, args.panels, joint_panels, args.draws, qual_ok),
        }
    out_json = ROOT / "data/runs/h1a/requalification_report.json"
    out_json.write_text(
        json.dumps(
            report,
            indent=1,
            sort_keys=True,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )
    doc = ROOT / f"docs/reviews/P5/tf-v1-requalification-{args.date}.md"
    doc.write_text(render(report))
    log(f"wrote {doc} and {out_json}")


if __name__ == "__main__":
    main()
