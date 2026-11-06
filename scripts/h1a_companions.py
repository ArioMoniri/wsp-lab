#!/usr/bin/env python3
"""Descriptive companions to the TF-v1 D3 gate (Board A 6 Nov 2026 section 5 step 1; PREREG A4.3).

Run (CPU only, about 10-15 min for MMLU-Pro; launches no scoring):

    python scripts/h1a_companions.py --endpoint mmlu_gold_log_prob \
        --out-doc docs/reviews/P5/h1a-companions-2026-11-06.md \
        --out-json data/runs/h1a/companions_mmlu.json

Everything here is descriptive, post hoc and on development data; nothing replaces the registered
primary theta of `tfv1_requalify.py` (A2.3). Reported, each labelled by n rows / n components:

* the registered theta (row-summed ratio on the stratum rows, OOF), for reference;
* outcome-free strata: family, incremental-norm tertile (weight-only) and their cross table;
* the flexible nuisance (PREREG section 3: training-only cubic spline in log1p(incremental norm),
  knots at the training-row quartiles, family interactions) next to the registered one, with all
  six intervals of the gate (conditional and joint x bootstrap-t, BCa, percentile) for the flexible
  fit and the equal-component-weight ratio;
* item-split replicates (labels from one half of the endpoint's resampling blocks, scored on the
  other half): parity split plus random splits;
* the 1/h_i^2-weighted and h_i <= delta/2-restricted variants of theta (Board A: sensitivities only,
  never decisions; h is a function of the outcome and the weights are outcome-derived).

`compute_companions` and `render_companions` are reused by `h1w_gate.py` (A4.3, WikiText).
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import tfv1_requalify as rq  # noqa: E402  (sibling script module)

SPLIT_SEED = 20261106
N_RANDOM_SPLITS = 20


# ------------------------------------------------------------------ losses and variants


def theta_of(a, b, sel=None, w=None):
    """Row-summed theta = 1 - sum(w A) / sum(w B) over the selected rows (w = 1 by default)."""
    sel = np.ones(len(a), bool) if sel is None else sel
    w = np.ones(len(a)) if w is None else w
    den = float((w[sel] * b[sel]).sum())
    return float(1 - (w[sel] * a[sel]).sum() / den) if den > 0 else float("nan")


def n_components(codes, sel):
    return len(set(np.asarray(codes)[sel]))


def cluster_percentile(a, b, codes, sel=None, w=None, draws=rq.DRAWS, seed=rq.SEED):
    """Component-bootstrap percentile interval of `theta_of` (descriptive; the rows of each resampled
    component are carried together and the selection is applied inside the resample)."""
    sel = np.ones(len(a), bool) if sel is None else sel
    codes = np.asarray(codes)
    uniq = sorted(set(codes))
    members = {c: np.flatnonzero(codes == c) for c in uniq}
    pick = np.random.default_rng(seed).integers(len(uniq), size=(draws, len(uniq)))
    vals = np.full(draws, np.nan)
    for k, row in enumerate(pick):
        idx = np.concatenate([members[uniq[j]] for j in row])
        idx = idx[sel[idx]]
        if len(idx):
            vals[k] = _theta_idx(a, b, idx, w)
    ok = np.isfinite(vals)
    if not ok.any():
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals[ok], 0.025)), float(np.quantile(vals[ok], 0.975))]


def _theta_idx(a, b, idx, w):
    w = np.ones(len(a)) if w is None else w
    den = float((w[idx] * b[idx]).sum())
    return float(1 - (w[idx] * a[idx]).sum() / den) if den > 0 else float("nan")


def equal_component_theta(a, b, codes):
    """Equal-component-weight ratio: each row weighs 1/n_c (a component contributes one unit of loss
    mass regardless of its row count), and the mean of the per-component ratios for comparison."""
    codes = np.asarray(codes)
    size = {c: int((codes == c).sum()) for c in set(codes)}
    w = np.array([1.0 / size[c] for c in codes])
    per = [
        1 - a[codes == c].sum() / b[codes == c].sum()
        for c in sorted(set(codes))
        if b[codes == c].sum() > 0
    ]
    return theta_of(a, b, w=w), float(np.mean(per)), w


def effective_n(weights):
    """Kish effective sample size of non-negative weights."""
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / (w**2).sum())


def b_concentration(b, codes, w=None):
    """n_eff = (sum_c B_c)^2 / sum_c B_c^2 and the top-five-component share of sum B (gate diagnostics)."""
    w = np.ones(len(b)) if w is None else w
    codes = np.asarray(codes)
    cb = np.array([(w * b)[codes == c].sum() for c in sorted(set(codes))])
    return float(cb.sum() ** 2 / (cb**2).sum()), float(np.sort(cb)[-5:].sum() / cb.sum())


# ------------------------------------------------------------------------- strata


def tertile_labels(inc):
    """Incremental-norm tertile of each stratum row (weight-only; cut points from the stratum rows)."""
    cuts = np.quantile(inc, [1 / 3, 2 / 3])
    return np.digitize(inc, cuts), [float(c) for c in cuts]


def stratum_cell(a, b, codes, sel, total_b, draws):
    n = int(sel.sum())
    if n == 0:
        return {"n_rows": 0, "n_components": 0}
    return {
        "n_rows": n,
        "n_components": n_components(codes, sel),
        "share_of_sum_B": float(b[sel].sum() / total_b),
        "theta": theta_of(a, b, sel),
        "cluster_percentile95": cluster_percentile(a, b, codes, sel, draws=draws),
    }


def strata_tables(a, b, codes, fam, inc, draws):
    """Outcome-free strata: family, incremental-norm tertile, family x tertile (and within-family
    terciles, because the norm distributions of the families barely overlap)."""
    fam, codes = np.asarray(fam), np.asarray(codes)
    tert, cuts = tertile_labels(inc)
    total_b = float(b.sum())
    out = {"tertile_cuts_incremental_norm": cuts, "family": {}, "tertile": {}, "cross": {}}
    out["median_incremental_norm_by_family"] = {
        f: float(np.median(inc[fam == f])) for f in rq.STRATUM if (fam == f).any()
    }
    for f in rq.STRATUM:
        out["family"][f] = stratum_cell(a, b, codes, fam == f, total_b, draws)
    for t in range(3):
        out["tertile"][f"T{t + 1}"] = stratum_cell(a, b, codes, tert == t, total_b, draws)
        for f in rq.STRATUM:
            out["cross"][f"{f}|T{t + 1}"] = stratum_cell(
                a, b, codes, (fam == f) & (tert == t), total_b, draws
            )
    within = {}
    for f in rq.STRATUM:
        sel = fam == f
        if sel.sum() < 6:
            continue
        cut = np.quantile(inc[sel], [1 / 3, 2 / 3])
        lab = np.digitize(inc, cut)
        for t in range(3):
            within[f"{f}|within-T{t + 1}"] = stratum_cell(
                a, b, codes, sel & (lab == t), total_b, draws
            )
    out["within_family_tertile"] = within
    out["pre_declared"] = {  # Board A: high-norm / LoRA stratum (looked at after the fact on D3)
        "lora_sft_and_high_norm_tertile": stratum_cell(
            a, b, codes, (fam == "lora_sft") & (tert == 2), total_b, draws
        ),
        "high_norm_tertile_all_families": stratum_cell(a, b, codes, tert == 2, total_b, draws),
        "lora_sft_all": stratum_cell(a, b, codes, fam == "lora_sft", total_b, draws),
    }
    return out


# ---------------------------------------------------------------- flexible nuisance


def spline_basis(x, knots):
    """Truncated-power cubic spline in x: 1, x, x^2, x^3, (x - k)_+^3 for each knot (unbounded tails)."""
    cols = [np.ones_like(x), x, x**2, x**3]
    cols += [np.clip(x - k, 0, None) ** 3 for k in knots]
    return np.column_stack(cols)


def natural_spline_basis(x, knots):
    """Restricted (natural) cubic spline of Harrell: 1, x and k - 2 nonlinear terms, linear beyond the
    outer knots, so held-out norms outside the training range do not blow up."""
    t = np.asarray(knots, float)
    k = len(t)
    cube = lambda u: np.clip(u, 0, None) ** 3
    cols = [np.ones_like(x), x]
    for j in range(k - 2):
        cols.append(
            cube(x - t[j])
            - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        )
    return np.column_stack(cols)


NUISANCE_VARIANTS = (
    "family_x_natural_spline",
    "shared_natural_spline_family_slope",
    "family_x_truncated_cubic",
)


def flexible_nuisance(families, log_norm, y, variant="family_x_natural_spline"):
    """Nuisance(train, test) -> held-out predictions; every choice uses training rows only.

    Knots are the training-row quartiles of log1p(incremental norm) (PREREG section 3). Variants:
    family_x_natural_spline: for each family level in training, indicator x [1, x, N1(x)] with the
      restricted cubic spline on the three quartile knots (the stable reading of 'cubic spline with
      knots at the quartiles, family interactions' and the primary flexible nuisance);
    shared_natural_spline_family_slope: one shared natural spline + family intercepts and
      family x linear slopes (less flexible);
    family_x_truncated_cubic: indicator x [1, x, x^2, x^3, (x-k)+^3] (unbounded tails; reported only
      because it was the first reading tried and it is numerically unstable, see the report).
    OLS by the Moore-Penrose solution in float64; an unseen family level fails closed. The returned
    function records, in `.extrapolated`, how many held-out rows lie outside the training norm range."""
    families = np.asarray(families)
    log_norm = np.asarray(log_norm, float)

    def design(rows, levels, knots):
        x = log_norm[rows]
        ind = {lv: (families[rows] == lv).astype(float) for lv in levels}
        if variant == "family_x_truncated_cubic":
            basis = spline_basis(x, knots)
            return np.column_stack([ind[lv][:, None] * basis for lv in levels])
        basis = natural_spline_basis(x, knots)
        if variant == "family_x_natural_spline":
            return np.column_stack([ind[lv][:, None] * basis for lv in levels])
        return np.column_stack(
            [ind[lv] for lv in levels] + [ind[lv] * x for lv in levels[1:]] + [basis[:, 2:]]
        )

    def nuisance(train, test):
        tr, te = np.flatnonzero(train), np.flatnonzero(test)
        levels = sorted(set(families[tr]))
        unseen = set(families[te]) - set(levels)
        if unseen:
            raise ValueError(f"unseen family level: {sorted(unseen)} (fails closed)")
        knots = np.quantile(log_norm[tr], [0.25, 0.5, 0.75])
        beta = np.linalg.pinv(design(tr, levels, knots)) @ y[tr]
        nuisance.extrapolated += int(
            ((log_norm[te] < log_norm[tr].min()) | (log_norm[te] > log_norm[tr].max())).sum()
        )
        return design(te, levels, knots) @ beta

    nuisance.extrapolated = 0
    return nuisance


# ------------------------------------------------------------------- item splits


def item_split_theta(X, D_all, fams, incs, codes, n_stratum, ia, ib):
    """theta (stratum rows, OOF) for training labels from one half and test labels from either half:
    A->A, B->B, A->B, B->A. Item halves are index sets over the endpoint's resampling blocks."""
    ya, yb = D_all[:, ia].mean(1), D_all[:, ib].mean(1)
    out = {}
    for name, (train, test) in {
        "A->A": (ya, ya),
        "B->B": (yb, yb),
        "A->B": (ya, yb),
        "B->A": (yb, ya),
    }.items():
        f = rq.fit_oof(X, train, fams, incs, codes)
        m, pn, pi = f["m"][:n_stratum], f["p_nn"][:n_stratum], f["p_iso"][:n_stratum]
        r = test[:n_stratum] - m
        out[name] = float(1 - np.abs(r - pn).sum() / np.abs(r - pi).sum())
    return out


# --------------------------------------------------------------------- main routine


def compute_companions(
    ctx, args, endpoint, half, log, draws=rq.DRAWS, random_splits=N_RANDOM_SPLITS
):
    """All companions for one endpoint on the D3 stratum rows (descriptive).

    half: the A2.1 item half-widths h_i of the stratum rows (from `tfv1_requalify.assemble`)."""
    from wsp.predictor.baselines import load_feature_table

    delta, _, _, _, block = rq.ENDPOINTS[endpoint]
    pop, info, fit = rq.d3_fit(ctx, args, log, endpoint)
    n = info["n_stratum"]
    stratum = ctx["stratum"]
    fam, codes = np.asarray(ctx["families"]), np.asarray(ctx["codes"])
    inc = np.array([ctx["inc"][i] for i in stratum])
    r = fit["resid"][:n]
    a, b = np.abs(r - fit["p_nn"][:n]), np.abs(r - fit["p_iso"][:n])
    comp_idx = {c: k for k, c in enumerate(sorted(set(codes)))}
    row_comp = np.array([comp_idx[c] for c in codes])
    out = {
        "endpoint": endpoint,
        "delta": delta,
        "n_rows": n,
        "n_components": len(comp_idx),
        "draws": draws,
        "n_null_controls": info["n_null"],
        "n_development_rows": info["n_rows"],
        "family_counts": {f: int((fam == f).sum()) for f in rq.STRATUM},
        "registered": {
            "theta": theta_of(a, b),
            "n_eff": info["n_eff"],
            "top5_share": info["top5_share"],
            "cluster_percentile95": cluster_percentile(a, b, codes, draws=draws),
        },
    }
    # (1) equal-component weights
    eq, eq_mean, w_eq = equal_component_theta(a, b, codes)
    out["equal_component_weight"] = {
        "theta_row_weight_1_over_nc": eq,
        "theta_mean_of_component_ratios": eq_mean,
        "cluster_percentile95_row_weight_1_over_nc": cluster_percentile(
            a, b, codes, w=w_eq, draws=draws
        ),
        "n_rows": n,
        "n_components": len(comp_idx),
    }
    # (2) outcome-free strata
    log("strata")
    out["strata"] = strata_tables(a, b, codes, fam, inc, draws)
    # (3) flexible nuisance: the same pipeline with the nuisance replaced
    feats = {x["id"]: x["statnn"] for x in load_feature_table(args.features)}
    entries = stratum + [
        i
        for i in ctx["plan"]["d3_null_entries"]
        if (ctx["tf"] / ctx["scored"](i) / "eval.json").exists()
    ]
    X = np.stack([feats[i] for i in entries])
    fams_all = [ctx["rows"][i]["family"] for i in entries]
    codes_all = [ctx["component_of"][i] for i in entries]
    incs_all = [ctx["inc"][i] for i in entries]
    y_all = np.array([ctx["paired"](i, endpoint).mean() for i in entries])
    log_norm = np.log1p(np.asarray(incs_all, float))
    out["flexible_nuisance"] = {}
    for variant in NUISANCE_VARIANTS:
        log(f"flexible nuisance {variant}")
        nuis = flexible_nuisance(fams_all, log_norm, y_all, variant)
        f = rq.fit_oof(X, y_all, fams_all, incs_all, codes_all, nuisance=nuis)
        rf = f["resid"][:n]
        af, bf = np.abs(rf - f["p_nn"][:n]), np.abs(rf - f["p_iso"][:n])
        entry = {
            "theta": theta_of(af, bf),
            "n_rows": n,
            "n_components": len(comp_idx),
            "n_eff": b_concentration(bf, codes)[0],
            "top5_share": b_concentration(bf, codes)[1],
            "cluster_percentile95": cluster_percentile(af, bf, codes, draws=draws),
            "held_out_rows_outside_training_norm_range": nuis.extrapolated,
            "nuisance_oof_r2_vs_registered": {
                "registered_resid_var": float(np.var(r)),
                "flexible_resid_var": float(np.var(rf)),
            },
            "mean_abs_residual": {
                "registered": float(np.abs(r).mean()),
                "flexible": float(np.abs(rf).mean()),
            },
            "equal_component_weight_theta": equal_component_theta(af, bf, codes)[0],
        }
        # the six gate intervals on the flexible fit: pop with the flexible predictions
        pop_flex = {
            **pop,
            "qa": f["m"][:n] + f["p_nn"][:n],
            "qb": f["m"][:n] + f["p_iso"][:n],
        }
        entry["six_intervals"] = rq.observed_intervals(pop_flex, np.arange(n), row_comp, 1.0, draws)
        entry["strata_by_family"] = {
            fm: {
                "n_rows": int((fam == fm).sum()),
                "n_components": n_components(codes, fam == fm),
                "theta": theta_of(af, bf, fam == fm),
            }
            for fm in rq.STRATUM
        }
        tert, _ = tertile_labels(inc)
        entry["strata_by_tertile"] = {
            f"T{t + 1}": {
                "n_rows": int((tert == t).sum()),
                "n_components": n_components(codes, tert == t),
                "theta": theta_of(af, bf, tert == t),
            }
            for t in range(3)
        }
        out["flexible_nuisance"][variant] = entry
    # (4) item-split replicates: halves of the endpoint's resampling blocks
    log("item splits")
    D_all = rq.block_means(np.array([ctx["paired"](i, endpoint) for i in entries]), block)
    n_items = D_all.shape[1]
    idx = np.arange(n_items)
    parity = item_split_theta(
        X, D_all, fams_all, incs_all, codes_all, n, idx % 2 == 0, idx % 2 == 1
    )
    rng = np.random.default_rng(SPLIT_SEED)
    rand = []
    for k in range(random_splits):
        perm = rng.permutation(n_items)
        ia, ib = np.zeros(n_items, bool), np.zeros(n_items, bool)
        ia[perm[: n_items // 2]], ib[perm[n_items // 2 :]] = True, True
        rand.append(item_split_theta(X, D_all, fams_all, incs_all, codes_all, n, ia, ib))
        log(f"random item split {k + 1}/{random_splits}")
    out["item_split"] = {
        "n_rows": n,
        "n_components": len(comp_idx),
        "n_blocks": int(n_items),
        "block_windows_or_items": block,
        "n_blocks_per_half": int(n_items // 2),
        "parity": parity,
        "random_splits": {
            "n_splits": random_splits,
            "seed": SPLIT_SEED,
            "by_combination": {
                name: {
                    "mean": float(np.mean([s[name] for s in rand])),
                    "q05": float(np.quantile([s[name] for s in rand], 0.05)),
                    "q95": float(np.quantile([s[name] for s in rand], 0.95)),
                    "min": float(np.min([s[name] for s in rand])),
                    "max": float(np.max([s[name] for s in rand])),
                    "n_positive": int(sum(s[name] > 0 for s in rand)),
                }
                for name in ("A->A", "B->B", "A->B", "B->A")
            },
        },
    }
    # (5) h-weighted and h-restricted variants (descriptive only)
    half = np.asarray(half, float)
    over = half > delta / 2
    out["h_variants"] = h_variants(a, b, codes, half, over, delta, draws)
    out["h_variants"]["refit_without_over_half_delta"] = refit_without(
        ctx, args, endpoint, X, entries, over, n, draws, log
    )
    out["half_width_summary"] = {
        "n_rows": n,
        "n_over_half_delta": int(over.sum()),
        "max_h": float(half.max()),
        "median_h": float(np.median(half)),
    }
    state = dict(
        pop=pop,
        fit=fit,
        a=a,
        b=b,
        fam=fam,
        codes=codes,
        inc=inc,
        row_comp=row_comp,
        info=info,
        half=half,
    )
    return out, state


def h_variants(a, b, codes, half, over, delta, draws):
    """theta with 1/h_i^2 weights and restricted to h_i <= delta/2. The fits (nuisance, StatNN,
    isotonic) are those of the registered all-row OOF run, as in the A2.1 (iv) leave-out; only the
    scored sums change. Weights are normalised to mean 1 (the ratio is scale-free)."""
    w = 1.0 / half**2
    w = w / w.mean()
    keep = ~over
    n_eff_b, top5_b = b_concentration(b, codes, w)
    out = {
        "all_rows_unweighted": {
            "theta": theta_of(a, b),
            "n_rows": len(a),
            "n_components": n_components(codes, np.ones(len(a), bool)),
        },
        "inverse_h_squared_weighted": {
            "theta": theta_of(a, b, w=w),
            "n_rows": len(a),
            "n_components": n_components(codes, np.ones(len(a), bool)),
            "kish_n_eff_of_weights": effective_n(w),
            "n_eff_component_weighted_B": n_eff_b,
            "top5_component_share_of_weighted_B": top5_b,
            "weight_ratio_max_over_min": float(w.max() / w.min()),
            "share_of_weight_on_rows_over_half_delta": float(w[over].sum() / w.sum()),
            "cluster_percentile95": cluster_percentile(a, b, codes, w=w, draws=draws),
        },
        "restricted_h_le_half_delta": {
            "theta": theta_of(a, b, keep),
            "n_rows": int(keep.sum()),
            "n_components": n_components(codes, keep),
            "cluster_percentile95": cluster_percentile(a, b, codes, keep, draws=draws),
        },
        "only_h_gt_half_delta": {
            "theta": theta_of(a, b, over),
            "n_rows": int(over.sum()),
            "n_components": n_components(codes, over) if over.any() else 0,
            "cluster_percentile95": cluster_percentile(a, b, codes, over, draws=draws)
            if over.any()
            else None,
        },
    }
    if keep.any():
        w_keep = np.where(keep, w, 0.0)
        out["inverse_h_squared_weighted_restricted"] = {
            "theta": theta_of(a, b, keep, w_keep),
            "n_rows": int(keep.sum()),
            "n_components": n_components(codes, keep),
            "kish_n_eff_of_weights": effective_n(w[keep]),
        }
    return out


def refit_without(ctx, args, endpoint, X, entries, over, n, draws, log):
    """Variant: the h_i > delta/2 rows are dropped from training and scoring and everything is refit
    (the null-family controls stay in training). Differs from the A2.1 (iv) leave-out, which scores
    on the restricted rows but fits on all rows (stated in the report)."""
    if not over.any():
        return {"note": "no row has h_i > delta/2: identical to the all-row fit", "n_rows": n}
    drop = set(np.flatnonzero(over))
    keep_idx = [k for k in range(len(entries)) if k not in drop]
    sub = [entries[k] for k in keep_idx]
    fams = [ctx["rows"][i]["family"] for i in sub]
    codes = [ctx["component_of"][i] for i in sub]
    incs = [ctx["inc"][i] for i in sub]
    y = np.array([ctx["paired"](i, endpoint).mean() for i in sub])
    f = rq.fit_oof(X[keep_idx], y, fams, incs, codes)
    n_keep = n - len(drop)
    r = f["resid"][:n_keep]
    a, b = np.abs(r - f["p_nn"][:n_keep]), np.abs(r - f["p_iso"][:n_keep])
    stratum_codes = np.asarray(codes[:n_keep])
    log("refit without h > delta/2 done")
    return {
        "theta": theta_of(a, b),
        "n_rows": n_keep,
        "n_components": n_components(stratum_codes, np.ones(n_keep, bool)),
        "n_training_rows": len(sub),
        "cluster_percentile95": cluster_percentile(a, b, stratum_codes, draws=draws),
    }


# ------------------------------------------------------------------------ rendering


def g(x, n=4):
    return "n/a" if x is None else f"{x:.{n}g}" if isinstance(x, float) else str(x)


def ci(v):
    return "n/a" if v is None else f"[{g(v[0])}, {g(v[1])}]"


def cell_row(label, c):
    if not c.get("n_rows"):
        return f"| {label} | 0 | 0 | | | |"
    return (
        f"| {label} | {c['n_rows']} | {c['n_components']} | {g(c['share_of_sum_B'], 3)} | {g(c['theta'])} "
        f"| {ci(c['cluster_percentile95'])} |"
    )


def render_companions(c, level="###"):
    """Markdown lines for one endpoint's companions; every number carries its n."""
    n, nc = c["n_rows"], c["n_components"]
    reg = c["registered"]
    lines = [
        f"{level} Registered estimate (reference)\n",
        f"theta_hat = {g(reg['theta'])} on n = {n} stratum rows in {nc} components "
        f"(families {c['family_counts']}; fit on {c['n_development_rows']} development rows incl. {c['n_null_controls']} null-family controls); "
        f"n_eff = {g(reg['n_eff'])}, top-five component share of sum B = {g(reg['top5_share'])}; "
        f"component-bootstrap percentile 95% {ci(reg['cluster_percentile95'])} (descriptive, {c['draws']:,} draws).\n",
    ]
    st = c["strata"]
    lines += [
        f"{level} Outcome-free strata (family and weight-only incremental-norm tertile)\n",
        f"Tertile cut points of the incremental effective norm (stratum rows): {', '.join(g(x) for x in st['tertile_cuts_incremental_norm'])}; "
        f"median incremental norm by family: {', '.join(f'{k} {g(v)}' for k, v in st['median_incremental_norm_by_family'].items())}. "
        "Intervals are component-bootstrap percentile, rows filtered inside the resample (descriptive; the cells were chosen after the fact on D3 and cannot be counted as pre-registered).\n",
        "| Stratum | n rows | n components | share of sum B | theta | percentile 95% |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, key in (
        ("family", "family"),
        ("tertile", "tertile"),
        ("family x tertile", "cross"),
        ("within-family tercile", "within_family_tertile"),
        ("pre-declared candidates", "pre_declared"),
    ):
        lines += [cell_row(f"{label}: {k}", v) for k, v in st[key].items()]
    lines.append("")
    eq = c["equal_component_weight"]
    lines += [
        f"{level} Equal-component weights\n",
        f"theta with row weight 1/n_c (each component carries one unit of loss mass) = {g(eq['theta_row_weight_1_over_nc'])} "
        f"{ci(eq['cluster_percentile95_row_weight_1_over_nc'])} (n = {eq['n_rows']} rows, {eq['n_components']} components); "
        f"mean of the {eq['n_components']} per-component ratios = {g(eq['theta_mean_of_component_ratios'])} (dominated by components with near-zero B; not a stable summary). Registered row-summed theta_hat = {g(reg['theta'])} (same n).\n",
        f"{level} Flexible nuisance (PREREG section 3: training-only cubic spline in log1p(incremental norm), knots at training-row quartiles, family interactions)\n",
        "| Nuisance | theta_hat | n rows | n components | n_eff | top-5 share | percentile 95% | mean abs residual (registered / flexible) | held-out development rows outside training norm range |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|",
        f"| registered (family dummies + log1p norm) | {g(reg['theta'])} | {n} | {nc} | {g(reg['n_eff'])} | {g(reg['top5_share'])} | {ci(reg['cluster_percentile95'])} | | |",
    ]
    for variant, e in c["flexible_nuisance"].items():
        lines.append(
            f"| {variant} | {g(e['theta'])} | {e['n_rows']} | {e['n_components']} | {g(e['n_eff'])} | {g(e['top5_share'])} | "
            f"{ci(e['cluster_percentile95'])} | {g(e['mean_abs_residual']['registered'])} / {g(e['mean_abs_residual']['flexible'])} | {e['held_out_rows_outside_training_norm_range']} |"
        )
    lines.append("")
    lines += [
        "Six intervals of the gate (conditional and joint; bootstrap-t, BCa, percentile) for the flexible fit (n rows and components as above):\n",
        "| Nuisance | theta_hat | cond. bootstrap-t | joint bootstrap-t | cond. BCa | joint BCa | cond. percentile | joint percentile |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for variant, e in c["flexible_nuisance"].items():
        iv = e["six_intervals"]
        lines.append(
            f"| {variant} | {g(iv['theta_hat'])} | {ci(iv['conditional_bootstrap_t'])} | {ci(iv['joint_bootstrap_t'])} | "
            f"{ci(iv['conditional_bca'])} | {ci(iv['joint_bca'])} | {ci(iv['conditional_percentile'])} | {ci(iv['joint_percentile'])} |"
        )
    lines += [
        "",
        "Flexible-nuisance theta by outcome-free stratum (n rows / n components in brackets):\n",
    ]
    for variant, e in c["flexible_nuisance"].items():
        cells = [
            f"{k} {g(v['theta'])} [{v['n_rows']}/{v['n_components']}]"
            for k, v in {**e["strata_by_family"], **e["strata_by_tertile"]}.items()
        ]
        lines.append(f"- {variant}: " + "; ".join(cells))
    lines.append("")
    sp = c["item_split"]
    lines += [
        f"{level} Item-split replicates (labels from one half of the {sp['n_blocks']} resampling blocks, scored on the other; n = {sp['n_rows']} rows, {sp['n_components']} components, {sp['n_blocks_per_half']} blocks per half)\n",
        "| Split | A->A | B->B | A->B | B->A |",
        "|---|---:|---:|---:|---:|",
        f"| parity (index parity) | {g(sp['parity']['A->A'])} | {g(sp['parity']['B->B'])} | {g(sp['parity']['A->B'])} | {g(sp['parity']['B->A'])} |",
    ]
    rs = sp["random_splits"]
    lines.append(
        f"| {rs['n_splits']} random splits, mean [5%, 95%], positive/{rs['n_splits']} | "
        + " | ".join(
            f"{g(rs['by_combination'][k]['mean'])} [{g(rs['by_combination'][k]['q05'])}, {g(rs['by_combination'][k]['q95'])}], {rs['by_combination'][k]['n_positive']}"
            for k in ("A->A", "B->B", "A->B", "B->A")
        )
        + " |"
    )
    lines.append("")
    hv = c["h_variants"]
    hs = c["half_width_summary"]
    lines += [
        f"{level} 1/h_i^2-weighted and h_i <= delta/2-restricted variants (descriptive sensitivities; never decisions)\n",
        f"n = {hs['n_rows']} stratum rows; {hs['n_over_half_delta']} have h_i > delta/2 = {g(c['delta'] / 2)}; max h = {g(hs['max_h'])}. "
        "The fits are the registered all-row OOF fits; only the scored sums change (A2.1 (iv) convention), except the refit row.\n",
        "| Variant | theta | n rows | n components | percentile 95% | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    w = hv["inverse_h_squared_weighted"]
    lines += [
        f"| all rows, unweighted (registered) | {g(hv['all_rows_unweighted']['theta'])} | {hv['all_rows_unweighted']['n_rows']} | {hv['all_rows_unweighted']['n_components']} | {ci(reg['cluster_percentile95'])} | |",
        f"| weights 1/h_i^2 (mean-normalised) | {g(w['theta'])} | {w['n_rows']} | {w['n_components']} | {ci(w['cluster_percentile95'])} | Kish n_eff of weights {g(w['kish_n_eff_of_weights'])}; component n_eff of weighted B {g(w['n_eff_component_weighted_B'])}; top-5 share {g(w['top5_component_share_of_weighted_B'])}; max/min weight {g(w['weight_ratio_max_over_min'])}; {g(w['share_of_weight_on_rows_over_half_delta'], 3)} of weight on rows over delta/2 |",
    ]
    r_ = hv["restricted_h_le_half_delta"]
    lines.append(
        f"| restricted to h_i <= delta/2 (scored; fits on all rows) | {g(r_['theta'])} | {r_['n_rows']} | {r_['n_components']} | {ci(r_['cluster_percentile95'])} | A2.1 (iv) leave-out |"
    )
    o_ = hv["only_h_gt_half_delta"]
    lines.append(
        f"| only h_i > delta/2 | {g(o_['theta'])} | {o_['n_rows']} | {o_['n_components']} | {ci(o_['cluster_percentile95'])} | selected on a function of the outcome: description only |"
    )
    if "inverse_h_squared_weighted_restricted" in hv:
        q = hv["inverse_h_squared_weighted_restricted"]
        lines.append(
            f"| 1/h_i^2 weights within h_i <= delta/2 | {g(q['theta'])} | {q['n_rows']} | {q['n_components']} | | Kish n_eff of weights {g(q['kish_n_eff_of_weights'])} |"
        )
    rf = hv["refit_without_over_half_delta"]
    if "theta" in rf:
        lines.append(
            f"| refit without the h_i > delta/2 rows (training and scoring) | {g(rf['theta'])} | {rf['n_rows']} | {rf['n_components']} | {ci(rf['cluster_percentile95'])} | {rf['n_training_rows']} training rows |"
        )
    else:
        lines.append(
            f"| refit without the h_i > delta/2 rows | {g(reg['theta'])} | {rf['n_rows']} | {nc} | | {rf['note']} |"
        )
    lines.append("")
    return lines


# ----------------------------------------------------------------------------- CLI


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--endpoint", default="mmlu_gold_log_prob", choices=sorted(rq.ENDPOINTS))
    parser.add_argument("--draws", type=int, default=rq.DRAWS)
    parser.add_argument("--random-splits", type=int, default=N_RANDOM_SPLITS)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--out-doc", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--tf-dir", type=Path, default=ROOT / "data/evals/qwen3.5-0.8b/tf_v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/prereg/panel_manifest.json")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/qwen3.5-0.8b")
    args = parser.parse_args()
    args.restrict_ids = None
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    report, ctx = rq.assemble(args, log)
    half = report["_arrays"][args.endpoint][2]
    comp, _ = compute_companions(
        ctx, args, args.endpoint, half, log, args.draws, args.random_splits
    )
    comp["date"] = args.date
    comp["source_hash"] = report["source_hash"]
    out_json = args.out_json or ROOT / f"data/runs/h1a/companions_{args.endpoint}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            comp,
            indent=1,
            sort_keys=True,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )
    if args.out_doc:
        args.out_doc.write_text("\n".join(render_companions(comp)) + "\n")
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
