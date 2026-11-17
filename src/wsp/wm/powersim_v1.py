"""Outcome-blind power simulation and feasibility for PREREG_WME DRAFT v1 §8 (RULING-wme-resources §4; Board A
condition 11). Error- and decision-level ("cheap") simulator over the full §8.2 grid, the §8.5 selection rule and
the §8.6 pricing. The full-refit validation of the selected cell is ``wm.powersim_refit``.

Registered changes from the v0 simulator (``wm.powersim``, kept for the record): nominal alpha is one-sided 0.025;
size <= 0.035, one-sided coverage >= 0.965, power Clopper-Pearson 95% lower bound >= 0.80, >= 1,000 replicates
(cells within 2 MC SE of a threshold rerun at 4,000); the bank component graph is the actual v1 graph for each
(r_f, n_blk) (blocks of 4 roots from 4 families, singletons, negative controls inside their partner's
component, calibration components drawn first); the static comparator is chosen among three members on noisy
inner losses; H-T is gated by C2-S and C2-I; E1 uses the designated winner among the WM-arm episode winners, the
13 intersection-union guard bounds, the Tango paired-proportion generation tests and the anomaly-increase bounds;
E2 uses the guarded score and the two-way (episode x item) bootstrap; the E2 SD is the de-noised method-of-moments
SD over all 32 pilot-2 candidates (MAD as sensitivity). **Restored** (RULING §4(c)): the sparse trust-region
stratum scenario (``ht`` scenario ``sparse_outer``), with its reportability and reporting-rule outputs.

Every assumption that is not a measured input is named in :data:`ASSUMPTIONS` and labelled as such in the report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import time

import numpy as np
from scipy import optimize, stats

ALPHA = 0.025
Z = stats.norm.ppf(1 - ALPHA)
REQ = {
    "size_max": 0.035,
    "coverage_min": 0.965,
    "power_lb_min": 0.80,
    "reps_min": 1000,
    "rerun_reps": 4000,
}
CAPS = {"bank_gpu_h": 35.0, "confirmatory_gpu_h": 50.0, "lambda_usd_9b": 800.0}
SEED = 20261117
N_BOOT = 9999
FAMILIES = ("scale", "row_scale", "proj_remove", "noise", "steer", "lora_tv")
OBJ6 = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)
# pilot-2 per-objective static |error| medians at h >= 3 in SE units (validation n = 64; pilot-2 report T2)
STATIC_SCALE_64 = np.array([3.85, 2.38, 0.33, 9.77, 3.87, 1.27])

GRID = {
    "r_f": (8, 12, 16, 20),
    "n_blk": (6, 12, 18, 24),
    "n_cal": (12, 18, 24),
    "n_bank": (48, 64, 96, 128),
    "n_dev": (64, 96, 128, 192),
    "n_test": (96, 128, 192, 256),
    "n_gen_syc": (100, 150, 200, 250),
    "n_gen_gsm": (200, 300, 500),
    "n_gen_math": (200, 300, 500),
    "shard_contrast": (16, 32),
    "shard_lora": (64, 128),
    "e2_shard": (64, 128, 256),
    "n_rep": (2, 4, 6, 8),
    "n_ep_sec": ("all", "half"),
    "n_rep_sec": (0, 1, 2),
    "B": (16, 24, 32, 48),
    "n_inert": (12, 24, 48),
    "pool_size": (64, 128),
}
ASSUMPTIONS = {
    "ct_alternative": "C_T = 0.15 (registered assumed effect, not measured)",
    "c2s_degradation": "under the H-T alternative a permutation of the step-local inputs raises WM scaled error "
    "by 30% on composition units (assumed); 0% under the 'ignores step inputs' null",
    "static_members": "S1 bias x1.3, S2 x1.0, S3 x1.0 (independent draws); inner selection on noisy losses (sd 0.3 "
    "relative)",
    "rho_obj": 0.3,
    "rho_comp": 0.3,
    "inert_no_ghost": "ghost-free inert predictions ~ N(0, (0.5 f_k)^2); ghost alternative mean 1-2 f_k",
    "coverage_cluster_sd": "component-level random effect on conformal coverage, sd 0.05 (logit scale x)",
    "e1_base_rates": "base generated Acc_S 0.40, Acc_N 0.80 (pilot-independent assumptions)",
    "e1_transfer": "generated delta = 0.05 when the designated winner carries the teacher-forced benefit",
    "e2_search_noise": "per (episode, arm) winner spread 0.5 x SD_cand; shared episode effect 0.5 x SD_cand",
    "search_reach_b0": 12.0,
    "e2_item_corr": "test-item effects shared across winners: rho 0.5",
    "gen_seconds": "measured by P-H100 (greedy, batch 8)",
    "stop_share_high": 0.3,
}


# ------------------------------------------------------------------ bank component graph (v1 §4.5)


def graph(r_f: int, n_blk: int, seed: int = SEED) -> dict | None:
    """Planned v1 component graph: n_blk blocks of 4 roots from 4 distinct families (a root in at most one block),
    every other root a singleton. Test units per component: 4 ladder states per root (h = 3: +-1; h = 4: +-2) and,
    per block, 4 composition states (h = 3, 4 prefixes of the order and its reverse). None if infeasible."""
    if 4 * n_blk > 6 * r_f:
        return None
    rng = np.random.default_rng([seed, r_f, n_blk])
    left = {f: r_f for f in FAMILIES}
    blocks = []
    for _ in range(n_blk):
        fams = sorted(FAMILIES, key=lambda f: (-left[f], rng.random()))[:4]
        if any(left[f] <= 0 for f in fams):
            return None
        for f in fams:
            left[f] -= 1
        blocks.append(tuple(sorted(fams)))
    comps = [{"stratum": "block", "families": b, "n_ladder": 16, "n_comp": 4} for b in blocks]
    for f in FAMILIES:
        comps += [
            {"stratum": f, "families": (f,), "n_ladder": 4, "n_comp": 0} for _ in range(left[f])
        ]
    return {
        "r_f": r_f,
        "n_blk": n_blk,
        "components": comps,
        "n_components": len(comps),
        "n_roots": 6 * r_f,
        "visits": _visits(r_f, n_blk),
    }


def _visits(r_f, n_blk):
    """States visited (upper bound; stop rule can only reduce): ladder sides of 6 doses (proj_remove +4/+8 are
    duplicates of +2), negative controls for proj_remove (same ladder), 6 composition visits per block."""
    per_root = 12
    proj = 10
    n_ladder = 5 * r_f * per_root + r_f * proj
    n_ctrl = r_f * proj
    return {
        "ladder": n_ladder,
        "controls": n_ctrl,
        "compositions": 6 * n_blk,
        "training_eligible": 6 * r_f * 4 + 6 * r_f + 2 * n_blk,
    }


def strata_counts(g) -> dict:
    out = {}
    for c in g["components"]:
        out[c["stratum"]] = out.get(c["stratum"], 0) + 1
    return out


# ------------------------------------------------------------------ vectorised statistics


def ct_bootstrap_t(A, S, strata, n_boot=N_BOOT, seed=0, alpha=ALPHA) -> dict:
    """Component-cluster bootstrap-t for C_T = 1 - sum A / sum S (§6.2.4): stratified resampling, delta-method
    studentisation recomputed in every replicate, one-sided lower bound at ``alpha``."""
    A, S = np.asarray(A, float), np.asarray(S, float)
    m = len(A)
    if m < 2 or S.sum() <= 0:
        return {
            "ct": float("nan"),
            "lb": float("nan"),
            "ub": float("nan"),
            "se": float("nan"),
            "estimable": False,
        }

    def stat(a, s):
        n = a.shape[-1]
        R = a.sum(-1) / s.sum(-1)
        resid = a - R[..., None] * s
        se = np.sqrt((resid**2).sum(-1) / (n * (n - 1))) / (s.sum(-1) / n)
        return R, se

    R, se = stat(A, S)
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    idx = np.empty((n_boot, m), dtype=np.int64)
    col = 0
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        idx[:, col : col + len(members)] = members[
            rng.integers(0, len(members), size=(n_boot, len(members)))
        ]
        col += len(members)
    Rb, seb = stat(A[idx], S[idx])
    ok = seb > 0
    t = (Rb[ok] - R) / seb[ok]
    if not len(t):
        return {"ct": 1 - R, "lb": float("nan"), "ub": float("nan"), "se": se, "estimable": False}
    q_hi, q_lo = np.quantile(t, [1 - alpha, alpha])
    # C = 1 - R: lower bound of C from the upper bound of R
    return {
        "ct": float(1 - R),
        "lb": float(1 - (R - q_lo * se)),
        "ub": float(1 - (R - q_hi * se)),
        "se": float(se),
        "estimable": True,
    }


def tango_lower(b: int, c: int, n: int, alpha=ALPHA) -> float:
    """Tango (1998) score interval for a difference of paired proportions, delta = (b - c) / n with b = edit-only
    successes, c = base-only successes; one-sided lower limit at ``alpha``."""
    z = stats.norm.ppf(1 - alpha)
    d_hat = (b - c) / n

    def score(d):
        if abs(d) >= 1:
            return float("inf") if d < d_hat else float("-inf")
        A = 2 * n
        B = -b - c + (2 * n - b + c) * d
        C = -c * d * (1 - d)
        disc = max(B * B - 4 * A * C, 0.0)
        q21 = (math.sqrt(disc) - B) / (2 * A)
        var = n * (2 * q21 + d * (1 - d))
        if var <= 0:
            return 0.0 if b - c - n * d == 0 else math.copysign(float("inf"), b - c - n * d)
        return (b - c - n * d) / math.sqrt(var)

    if d_hat <= -1 + 1e-12:
        return -1.0
    f = lambda d: score(d) - z
    lo = -1 + 1e-9
    if f(lo) < 0:
        return -1.0
    return float(
        optimize.brentq(f, lo, d_hat + 1e-12 if f(d_hat) < 0 else min(1 - 1e-9, d_hat + 1e-6))
    )


def cp_lower_one_sided(k, n, level=0.95):
    return 0.0 if k == 0 else float(stats.beta.ppf(1 - level, k, n - k + 1))


def clopper_pearson(k, n, level=0.95):
    a = 1 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# ------------------------------------------------------------------ H-T


def _noise(rng, shape, tails):
    if tails == "t3":
        return rng.standard_t(3, size=shape) / math.sqrt(3.0)
    return rng.standard_normal(shape)


def _scale_at(n_bank, se_ratio):
    """Static error scale in paired-SE units at n_bank groups: bias in nats is fixed, SE ~ 1/sqrt(n)."""
    return STATIC_SCALE_64 * math.sqrt(n_bank / 64.0) * np.asarray(se_ratio)


def ht_units(rng, g, n_cal, sc):
    """Units of the H-T population after drawing calibration components (stratified, first)."""
    comps = g["components"]
    strata = np.array([c["stratum"] for c in comps])
    order = rng.permutation(len(comps))
    cal = set()
    # stratified: round-robin over strata in seeded order
    by = {s: [i for i in order if strata[i] == s] for s in sorted(set(strata))}
    while len(cal) < min(n_cal, len(comps)):
        for s in sorted(by):
            if by[s] and len(cal) < n_cal:
                cal.add(by[s].pop(0))
    held = [i for i in range(len(comps)) if i not in cal]
    p_stop = sc.get("stop_share", 0.0)
    rows = []
    for ci in held:
        c = comps[ci]
        nl = c["n_ladder"]
        if p_stop:
            keep = rng.random(nl // 2) >= p_stop  # a stopped side loses its h = 4 state
            nl = int(nl // 2 + keep.sum())
        weak = sc.get("weak_families") and all(f in ("row_scale", "noise") for f in c["families"])
        rows.append((ci, c["stratum"], nl + c["n_comp"], c["n_comp"], weak))
    return rows, strata


def calibrate_wm_scale(
    ct, rho_obj=0.3, tails="normal", scale=STATIC_SCALE_64, n=200_000, seed=SEED
):
    rng = np.random.default_rng(seed)
    K = len(scale)
    L = np.linalg.cholesky(rho_obj * np.ones((K, K)) + (1 - rho_obj) * np.eye(K))
    bias = scale * (_noise(rng, (n, K), tails) @ L.T)
    wm_ind = scale * _noise(rng, (n, K), tails)
    eps = _noise(rng, (n, K), tails)
    est = np.abs(bias - eps).sum()
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        r = np.abs(mid * wm_ind - eps).sum() / est
        lo, hi = (mid, hi) if r < 1 - ct else (lo, mid)
    return (lo + hi) / 2


HT_SCENARIOS = {
    "null_ignores": {"ct": 0.0, "ignores_steps": True, "null": True},
    "null_nobetter": {"ct": 0.0, "null": True},
    "alt": {"ct": 0.15},
    "alt_005": {"ct": 0.05},
    "alt_010": {"ct": 0.10},
    "alt_020": {"ct": 0.20},
    "heavy_null": {"ct": 0.0, "tails": "t3", "null": True},
    "heavy_alt": {"ct": 0.15, "tails": "t3"},
    "weak_families": {"ct": 0.15, "weak_families": True},
    "high_stop": {"ct": 0.15, "stop_share": 0.3},
    "sparse_outer": {"ct": 0.15, "outer_share": 0.05, "advantage": "outer"},
    "sparse_outer_uniform": {"ct": 0.15, "outer_share": 0.05, "advantage": "uniform"},
}
HT_SELECT = ("null_ignores", "null_nobetter", "alt")


def sim_ht(rng, g, n_cal, n_bank, n_inert, sc, se_ratio, wm_scale, n_boot=N_BOOT):
    tails = sc.get("tails", "normal")
    rows, _strata_all = ht_units(rng, g, n_cal, sc)
    K = 6
    scale = _scale_at(n_bank, se_ratio)
    L = np.linalg.cholesky(0.3 * np.ones((K, K)) + 0.7 * np.eye(K))
    A, S, strata, comp_units = [], [], [], []
    outer_A, outer_S, in_A, in_S, _nd_A, _nd_S = [], [], [], [], [], []
    for _ci, st, n_u, n_c, weak in rows:
        re = rng.standard_normal(K) @ L.T
        b = scale * (math.sqrt(0.3) * re + math.sqrt(0.7) * (_noise(rng, (n_u, K), tails) @ L.T))
        eps = _noise(rng, (n_u, K), tails)
        members = [
            b,
            1.3 * scale * _noise(rng, (n_u, K), tails),
            scale * _noise(rng, (n_u, K), tails),
        ]
        inner = [np.abs(m).mean() * (1 + 0.3 * rng.standard_normal()) for m in members]
        e_st = np.abs(members[int(np.argmin(inner))] - eps)
        wm_ind = scale * _noise(rng, (n_u, K), tails)
        ws = wm_scale
        outer = np.zeros(n_u, bool)
        if "outer_share" in sc:
            outer = (
                rng.random(n_u) < sc["outer_share"] * 4
            )  # outer states concentrate in large-dose units
            outer &= rng.random(n_u) < 0.25
            if sc["advantage"] == "outer":
                ws = np.where(outer[:, None], sc["_ws_outer"], sc["_ws_null"])
        if weak:
            e_wm = np.abs(eps + 0.0 * wm_ind)
            e_st = np.abs(eps + 0.0 * b)
        elif sc.get("ignores_steps") or sc["ct"] == 0:
            # null: WM error exchangeable with the selected static member (same scale, independent draw)
            sel = int(np.argmin(inner))
            e_wm = np.abs((1.3 if sel == 1 else 1.0) * scale * _noise(rng, (n_u, K), tails) - eps)
        else:
            e_wm = np.abs(ws * wm_ind - eps)
        A.append(e_wm.sum())
        S.append(e_st.sum())
        strata.append(st)
        if n_c:
            comp_units.append((e_wm[-n_c:], n_c))
        if "outer_share" in sc:
            outer_A.append(e_wm[outer].sum())
            outer_S.append(e_st[outer].sum())
            in_A.append(e_wm[~outer].sum())
            in_S.append(e_st[~outer].sum())
    A, S, strata = np.array(A), np.array(S), np.array(strata)
    seed = int(rng.integers(2**31))
    res = ct_bootstrap_t(A, S, strata, n_boot=n_boot, seed=seed)
    # C2-S: permutation degradation on composition units (component bootstrap, one-sided 0.025)
    if comp_units:
        degr = 0.0 if sc.get("ignores_steps") else 0.3
        d = np.array([(u.sum() * degr * (1 + 0.5 * rng.standard_normal())) for u, _ in comp_units])
        if degr == 0.0:
            d = np.zeros(len(d))
        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
        c2s = bool(len(d) >= 5 and se > 0 and d.mean() - Z * se > 0)
    else:
        c2s = False
    # C2-I: ghost-free inert predictions, simultaneous pass rule (parts i and ii)
    y = 0.5 * rng.standard_normal((n_inert, K))
    c2i = bool(np.mean(np.abs(y) > 2.0) <= 0.10 and np.all(np.abs(y).mean(0) <= 1.0))
    out = {
        "reject": bool(res["estimable"] and res["lb"] > 0 and c2s and c2i),
        "lb": res["lb"],
        "ct_hat": res["ct"],
        "estimable": res["estimable"],
        "c2s": c2s,
        "c2i": c2i,
        "m": len(A),
    }
    if "outer_share" in sc:
        n_out = int(sum(1 for a in outer_S if a > 0))
        n_in = int(sum(1 for a in in_S if a > 0))
        ct_in = 1 - sum(in_A) / max(sum(in_S), 1e-12)
        ct_out = 1 - sum(outer_A) / max(sum(outer_S), 1e-12) if sum(outer_S) > 0 else float("nan")
        out.update(
            outer_components=n_out,
            inner_components=n_in,
            outer_reportable=n_out >= 10,
            inner_reportable=n_in >= 10,
            ct_inside=ct_in,
            ct_outside=ct_out,
            rule_damaging=bool(out["reject"] and ct_in <= 0),
        )
    return out


# ------------------------------------------------------------------ E1


E1_SCENARIOS = {
    "alt": {"tf": 0.10, "gen": 0.05},
    "alt_tf005": {"tf": 0.05, "gen": 0.05},
    "alt_tf020": {"tf": 0.20, "gen": 0.05},
    "alt_transfer03": {"tf": 0.10, "gen": 0.05, "rho_tf_gen": 0.3},
    "alt_guard_half": {"tf": 0.10, "gen": 0.05, "guard_shift": 0.5},
    "alt_corr05": {"tf": 0.10, "gen": 0.05, "rho_guard": 0.5},
    "alt_anom02": {"tf": 0.10, "gen": 0.05, "anom_base": 0.02},
    # RULING-wme-v2b A10: TriviaQA wrong answers are less plausible; N is powered at 0.5x the generated effect
    "alt_gen_half": {"tf": 0.10, "gen": 0.025},
    "null_zero": {"tf": 0.0, "gen": 0.0, "null": True},
    "null_guard_margin": {"tf": 0.10, "gen": 0.05, "guard_shift": 1.0, "null": True},
    "null_no_transfer": {"tf": 0.10, "gen": 0.0, "null": True},
    "null_neutral": {"tf": 0.10, "gen": 0.05, "neutral": -0.03, "null": True},
    "null_anomaly": {"tf": 0.10, "gen": 0.05, "anom_inc": 0.02, "null": True},
}
E1_SELECT = (
    "alt",
    "null_zero",
    "null_guard_margin",
    "null_no_transfer",
    "null_neutral",
    "null_anomaly",
)
# guard bounds of the primary set: (guard, scale) with the margin and the side; SEs come from P-H100
E1_BOUNDS = (("wikitext_nll", "native", "ucb", 0.01),) + tuple(
    (g, s, side, lim)
    for g, pairs in (
        ("gsm8k_gold_lp", (("pp", "lcb", -0.01), ("native", "lcb", -0.05))),
        ("mathcomp_gold_lp", (("pp", "lcb", -0.01), ("native", "lcb", -0.05))),
        ("mmlupro_gold_lp", (("pp", "lcb", -0.01), ("native", "lcb", -0.05))),
        ("syc_neutral", (("pp", "lcb", -0.01), ("native", "lcb", -0.05))),
        ("refusal_margin_harmful", (("pp", "lcb", -0.01), ("native", "lcb", -0.25))),
        ("refusal_margin_harmless", (("pp", "ucb", 0.01), ("native", "ucb", 0.25))),
    )
    for s, side, lim in pairs
)


def _search(rng, B, n_dev, inp, tf, sd_c, guard_se_dev, tails="normal"):
    """One episode on dev: B candidates; the best available candidate carries ``tf`` with probability
    1 - exp(-B / b0); others ~ N(-sd_c, sd_c) capped at 0; winner by dev LCB among dev-guard-passing ones."""
    b0 = 12.0
    true = np.minimum(rng.normal(-sd_c, sd_c, B), 0.0)
    has = tf > 0 and rng.random() < 1 - math.exp(-B / b0)
    if has:
        true[0] = tf
    se = inp["syc_se_64"] * math.sqrt(64 / n_dev)
    dev = true + se * _noise(rng, B, tails)
    lcb = dev - Z * se
    gpass = rng.random(B) > inp.get("p_dev_guard_fail", 0.1)
    ok = (lcb > 0) & gpass
    if not ok.any():
        return None
    w = int(np.argmax(np.where(ok, lcb, -np.inf)))
    return {"true": true[w], "lcb": lcb[w], "carries": bool(has and w == 0)}


def sim_e1(rng, n_rep, B, n_dev, n_test, n_gen, sc, inp):
    tails = sc.get("tails", "normal")
    sd_c = inp["sd_cand"]
    winners = []
    for _ in range(3 * n_rep):
        w = _search(rng, B, n_dev, inp, sc["tf"], sd_c, None, tails)
        if w:
            winners.append(w)
    if not winners:
        return {"reject": False, "designated": False}
    d = max(winners, key=lambda w: w["lcb"])
    se_t = inp["syc_se_64"] * math.sqrt(64 / n_test)
    gain = d["true"] + se_t * _noise(rng, 1, tails)[0]
    ok_gain = gain - Z * se_t > 0
    # guards: 13 bounds, intersection-union at one-sided 0.025
    ok_guard = True
    shift = sc.get("guard_shift", 0.0)
    for i, (g, s, side, lim) in enumerate(E1_BOUNDS):
        se_g = inp["guard_se_64"][g][s] * math.sqrt(64 / n_test)
        mu = 0.0
        if i == 1 and shift:  # one guard (GSM8K pp) moved toward its margin
            mu = shift * lim
        rho = sc.get("rho_guard", 0.0)
        z = (
            rho * (gain - d["true"]) / max(se_t, 1e-12)
            + math.sqrt(1 - rho**2) * rng.standard_normal()
        )
        est = mu + se_g * z
        if (side == "ucb" and est + Z * se_g > lim) or (side == "lcb" and est - Z * se_g < lim):
            ok_guard = False
    # generation: paired binary per item, Tango lower bounds
    gen = (
        sc["gen"] if d["carries"] else 0.0
    )  # generated transfer only from a winner carrying the TF benefit
    p0s, p0n = 0.40, 0.80
    bs = rng.random(n_gen) < p0s
    flip_up = rng.random(n_gen) < (gen + 0.05 * p0s) / max(1e-9, 1 - p0s)
    flip_dn = rng.random(n_gen) < 0.05
    es = np.where(bs, ~flip_dn, flip_up)
    b, c = int((es & ~bs).sum()), int((~es & bs).sum())
    ok_gen = tango_lower(b, c, n_gen) > 0
    bn = rng.random(n_gen) < p0n
    drop = -sc.get("neutral", 0.0)
    en = np.where(
        bn, rng.random(n_gen) >= drop / p0n + 0.02, rng.random(n_gen) < 0.02 * p0n / (1 - p0n)
    )
    bb, cc = int((en & ~bn).sum()), int((~en & bn).sum())
    ok_neutral = tango_lower(bb, cc, n_gen) >= -0.03
    ok_anom = True
    base_a = sc.get("anom_base", 0.005)
    inc = sc.get("anom_inc", 0.0)
    for _kind in ("loop", "refusal"):
        a0 = (rng.random((n_gen, 2)) < base_a).sum(1) / 2
        a1 = (rng.random((n_gen, 2)) < base_a + inc).sum(1) / 2
        dd = a1 - a0
        sd = dd.std(ddof=1)
        ucb = (
            dd.mean() + stats.t.ppf(1 - ALPHA, n_gen - 1) * sd / math.sqrt(n_gen)
            if sd > 0
            else dd.mean()
        )
        ok_anom = ok_anom and ucb <= 0.02
    rej = bool(ok_gain and ok_guard and ok_gen and ok_neutral and ok_anom)
    return {
        "reject": rej,
        "designated": True,
        "parts": [ok_gain, ok_guard, ok_gen, ok_neutral, ok_anom],
    }


# ------------------------------------------------------------------ E2


E2_SCENARIOS = {
    "null": {"delta_sd": 0.0, "null": True},
    "null_shared": {"delta_sd": 0.0, "rho_family": 0.5, "null": True},
    "alt": {"delta_sd": 0.5},
    "alt_small": {"delta_sd": 0.25},
    "alt_mad": {"delta_sd": 0.5, "sd": "mad"},
    "alt_ranks": {"delta_sd": 0.5, "p_ranks": 1.0},
    "null_heavy": {"delta_sd": 0.0, "tails": "t3", "null": True},
}
E2_SELECT = ("null", "null_shared", "alt")


def sim_e2(rng, n_rep, B, n_dev, n_test, sc, inp, n_boot=1999, n_secondary=9):
    tails = sc.get("tails", "normal")
    sd_c = inp["sd_cand_mad"] if sc.get("sd") == "mad" else inp["sd_cand"]
    n = 3 * n_rep
    fam = np.arange(n) % 3
    reach = (1 - math.exp(-B / 12.0)) / (1 - math.exp(-32 / 12.0))
    delta = sc["delta_sd"] * sd_c * reach
    if rng.random() < sc.get("p_ranks", inp.get("p_ranks", 0.0)):
        delta *= 0.8
    rho_f = sc.get("rho_family", 0.0)
    fe = rng.standard_normal(3)
    ep = math.sqrt(rho_f) * fe[fam] + math.sqrt(1 - rho_f) * _noise(rng, n, tails)
    base = 0.5 * sd_c * reach + 0.5 * sd_c * ep  # episode effect shared by both arms
    G = np.stack(
        [
            base + delta + 0.5 * sd_c * _noise(rng, n, tails),
            base + 0.5 * sd_c * _noise(rng, n, tails),
        ]
    )
    se_dev = inp["syc_se_64"] * math.sqrt(64 / n_dev)
    is_base = (
        G + se_dev * rng.standard_normal(G.shape) - Z * se_dev
    ) <= 0  # no qualifying winner: base
    item_sd = inp["syc_se_64"] * 8.0
    shared = math.sqrt(0.5) * item_sd * rng.standard_normal(n_test)
    X = (
        G[:, :, None]
        + shared[None, None, :]
        + math.sqrt(0.5) * item_sd * _noise(rng, (2, n, n_test), tails)
    )
    gse = inp["guard_point_sd_item"]
    gmu = np.where(rng.random(G.shape) < 0.05, -0.02, 0.0)
    Gi = gmu[:, :, None] + gse * rng.standard_normal((2, n, n_test))

    def score(Xm, Gm):
        ok = Gm >= -0.01
        return np.where(is_base, 0.0, np.where(ok, Xm, np.minimum(Xm, 0.0)))

    g = score(X.mean(-1), Gi.mean(-1))
    D = float((g[0] - g[1]).mean())
    # two-way bootstrap: episodes stratified by family x items, guard decisions recomputed
    W = rng.multinomial(n_test, np.full(n_test, 1 / n_test), size=n_boot) / n_test
    Xs = np.einsum("bi,aei->bae", W, X)
    Gs = np.einsum("bi,aei->bae", W, Gi)
    ok = Gs >= -0.01
    gs = np.where(is_base[None], 0.0, np.where(ok, Xs, np.minimum(Xs, 0.0)))
    dd = gs[:, 0] - gs[:, 1]
    V = np.zeros((n_boot, n))
    for f in range(3):
        idx = np.flatnonzero(fam == f)
        pick = idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
        np.add.at(V, (np.repeat(np.arange(n_boot), len(idx)), pick.ravel()), 1.0)
    Ds = (V * dd).sum(1) / V.sum(1)
    lb = float(np.quantile(Ds, ALPHA))
    pv = []
    for _ in range(n_secondary):
        Dj = 0.5 * sd_c * rng.standard_normal(n) + item_sd / math.sqrt(
            n_test
        ) * rng.standard_normal(n)
        z = Dj.mean() / (Dj.std(ddof=1) / math.sqrt(n))
        pv.append(1 - stats.norm.cdf(z))
    pv = sorted(pv)
    holm_any = any(p <= ALPHA / (n_secondary - i) for i, p in enumerate(pv[:1]))
    return {
        "reject": lb > 0,
        "lb": lb,
        "D": D,
        "true": float(np.mean(np.where(is_base, 0, G)[0] - np.where(is_base, 0, G)[1])),
        "fwer_secondary": holm_any,
    }


# ------------------------------------------------------------------ cells, requirements


def _tag(*a):
    return int(hashlib.sha256(json.dumps(a, default=str).encode()).hexdigest()[:8], 16)


def run_cell(claim, cell: dict, scenario: str, reps: int, inp: dict, n_boot=None) -> dict:
    scen = {"ht": HT_SCENARIOS, "e1": E1_SCENARIOS, "e2": E2_SCENARIOS}[claim][scenario]
    rng = np.random.default_rng([SEED, _tag(claim, sorted(cell.items()), scenario)])
    t0 = time.time()
    rows = []
    if claim == "ht":
        g = graph(cell["r_f"], cell["n_blk"])
        scale = _scale_at(cell["n_bank"], inp["se_ratio"])
        ws = calibrate_wm_scale(scen["ct"], tails=scen.get("tails", "normal"), scale=scale)
        sc = dict(scen)
        if "outer_share" in sc:
            sc["_ws_null"] = calibrate_wm_scale(0.0, scale=scale)
            sc["_ws_outer"] = calibrate_wm_scale(
                0.6, scale=scale
            )  # strong advantage confined to outer states
        for _ in range(reps):
            rows.append(
                sim_ht(
                    rng,
                    g,
                    cell["n_cal"],
                    cell["n_bank"],
                    cell.get("n_inert", 12),
                    sc,
                    inp["se_ratio"],
                    ws,
                    n_boot or N_BOOT,
                )
            )
        truth = scen["ct"]
    elif claim == "e1":
        for _ in range(reps):
            rows.append(
                sim_e1(
                    rng,
                    cell["n_rep"],
                    cell["B"],
                    cell["n_dev"],
                    cell["n_test"],
                    cell["n_gen_syc"],
                    scen,
                    inp,
                )
            )
        truth = None
    else:
        for _ in range(reps):
            rows.append(
                sim_e2(
                    rng,
                    cell["n_rep"],
                    cell["B"],
                    cell["n_dev"],
                    cell["n_test"],
                    scen,
                    inp,
                    n_boot or 1999,
                )
            )
        truth = None
    k = sum(r["reject"] for r in rows)
    out = {
        "claim": claim,
        "cell": cell,
        "scenario": scenario,
        "reps": reps,
        "rate": k / reps,
        "ci95": clopper_pearson(k, reps),
        "power_lb": cp_lower_one_sided(k, reps),
        "null": bool(scen.get("null")),
        "seconds": time.time() - t0,
    }
    if claim == "ht":
        est = [r for r in rows if r["estimable"] and np.isfinite(r["lb"])]
        miss = sum(truth < r["lb"] for r in est)
        out["coverage"] = 1 - miss / max(1, len(est))
        out["coverage_ci95"] = clopper_pearson(len(est) - miss, max(1, len(est)))
        out["c2s_pass"] = float(np.mean([r["c2s"] for r in rows]))
        out["c2i_pass"] = float(np.mean([r["c2i"] for r in rows]))
        out["m"] = int(np.median([r["m"] for r in rows]))
        if "outer_share" in scen:
            for key in ("outer_reportable", "inner_reportable", "rule_damaging"):
                out[key] = float(np.mean([r[key] for r in rows]))
    if claim == "e2":
        est = rows
        miss = sum(r["true"] < r["lb"] for r in est)
        out["coverage"] = 1 - miss / max(1, len(est))
        out["fwer_secondary"] = float(np.mean([r["fwer_secondary"] for r in rows]))
    if claim == "e1":
        out["designated"] = float(np.mean([r["designated"] for r in rows]))
    return out


def passes(res: dict) -> bool:
    if res["reps"] < REQ["reps_min"]:
        return False
    if res["null"]:
        ok = res["rate"] <= REQ["size_max"]
    else:
        ok = res["power_lb"] >= REQ["power_lb_min"]
    if "coverage" in res and res["claim"] == "ht" and res["scenario"] in ("null_nobetter", "alt"):
        ok = ok and res["coverage"] >= REQ["coverage_min"]
    return bool(ok)


def near_threshold(res: dict) -> bool:
    mc = math.sqrt(max(res["rate"] * (1 - res["rate"]), 1e-6) / res["reps"])
    thr = REQ["size_max"] if res["null"] else REQ["power_lb_min"]
    return abs(res["rate"] - thr) <= 2 * mc


# ------------------------------------------------------------------ C2-I and C1.4 (cheap, closed form or MC)


def c2i_rates(n_inert, reps=20000, seed=SEED, K=6) -> dict:
    rng = np.random.default_rng([seed, n_inert])
    out = {}
    for name, ghost in (("null_no_ghost", 0.0), ("ghost_1f", 1.0), ("ghost_2f", 2.0)):
        y = ghost + 0.5 * rng.standard_normal((reps, n_inert, K))
        p1 = np.mean(np.abs(y) > 2.0, axis=(1, 2)) <= 0.10
        p2 = np.all(np.abs(y).mean(1) <= 1.0, axis=1)
        fail = ~(p1 & p2)
        out[name] = float(fail.mean())
    return {
        "false_fail": out["null_no_ghost"],
        "detect_1f": out["ghost_1f"],
        "detect_2f": out["ghost_2f"],
        "pass_requirement": out["null_no_ghost"] <= 0.05,
    }


def c14_rates(n_cal, units_per_comp=6, reps=20000, seed=SEED) -> dict:
    rng = np.random.default_rng([seed, n_cal])
    out = {}
    for name, p in (("calibrated", 0.90), ("under_15pp", 0.75)):
        lp = np.log(p / (1 - p)) + 0.5 * rng.standard_normal((reps, n_cal))
        pc = 1 / (1 + np.exp(-lp))
        cov = rng.binomial(units_per_comp * 6, pc).sum(1) / (units_per_comp * 6 * n_cal)
        out[name] = float(np.mean(cov < 0.80))
    return {"false_switch": out["calibrated"], "switch_under_15pp": out["under_15pp"]}


# ------------------------------------------------------------------ pricing (§8.6) and selection (§8.5)


def price(design: dict, unit: dict) -> dict:
    """GPU-h on H100 from P-H100 unit times ``unit``: per-item seconds per objective, per-unit overhead, oracle,
    field, generation and LoRA seconds (see ``scripts/wme_powersim_v1.py``)."""
    g = graph(design["r_f"], design["n_blk"])
    v = g["visits"]
    per = unit["per_item_s"]
    bank_objs = OBJ6
    t_visit = (
        unit["overhead_s"] * len(bank_objs) + sum(per[o] for o in bank_objs) * design["n_bank"]
    )
    n_inert_states = (
        design["n_inert"] // 2 * 1 + design["n_inert"] // 4 * 1 + design["n_inert"] // 4 * 3
    )
    n_visit = v["ladder"] + v["compositions"]
    n_ctrl = v["controls"] + n_inert_states
    h_bank = (
        n_visit * t_visit
        + v["training_eligible"] * unit["oracle_s"]
        + n_ctrl * t_visit
        + unit["field_s"]
        + unit["dirs_s_per_root"] * design["r_f"] * 6
        + unit["lora_s_per_root"] * design["r_f"]
    ) / 3600
    dev_objs = OBJ6 + ("mathcomp_gold_lp",)
    t_bundle = unit["overhead_s"] * len(dev_objs) + sum(per[o] for o in dev_objs) * design["n_dev"]
    t_test = unit["overhead_s"] * len(dev_objs) + sum(per[o] for o in dev_objs) * design["n_test"]
    n_ep = 3 * design["n_rep"]
    n_sec = n_ep if design["n_ep_sec"] == "all" else max(3, n_ep // 2)
    n_ep_e1sec = 2 * 3 * design["n_rep_sec"]
    bundles = n_ep * 2 * design["B"] + n_sec * 9 * design["B"] + n_ep_e1sec * design["B"]
    n_winners = n_ep * 2 + n_sec * 9 + n_ep_e1sec
    gen_s = unit["gen_s"]
    n_gen = 2 * (2 * design["n_gen_syc"]) * gen_s["sycophancy_real"]
    if design["n_rep_sec"]:
        n_gen += 2 * (
            design["n_gen_gsm"] * gen_s["gsm8k_gold_lp"]
            + design["n_gen_math"] * gen_s["mathcomp_gold_lp"]
        )
    n_gen_sampled = 4 * 2 * (2 * design["n_gen_syc"]) * gen_s["sycophancy_real"]
    h_conf = (
        bundles * t_bundle
        + unit["library_s"]
        + unit["lora_sft_s"] * n_sec
        + n_winners * t_test
        + n_gen
        + n_gen_sampled
        + (n_ep * 2 + n_sec * 9) * unit.get("planner_gpu_s", 0.0)
    ) / 3600
    return {
        "bank_gpu_h": h_bank,
        "confirmatory_gpu_h": h_conf,
        "total_gpu_h": h_bank + h_conf,
        "t_visit_s": t_visit,
        "t_bundle_s": t_bundle,
        "n_visit": n_visit,
        "n_ctrl": n_ctrl,
        "bundles": bundles,
        "n_winners": n_winners,
        "within_caps": h_bank <= CAPS["bank_gpu_h"] and h_conf <= CAPS["confirmatory_gpu_h"],
    }


def pool_demand(design: dict) -> dict:
    """Groups each objective's pool must supply under ``design`` (field 24 + splits + construction shards)."""
    nb, nd, nt = design["n_bank"], design["n_dev"], design["n_test"]
    r_f, sc, sl = design["r_f"], design["shard_contrast"], design["shard_lora"]
    # catalogue sources for contrast roots (steer + proj_remove + row_scale shards): MMLU-Pro and MATH subjects,
    # GSM8K and harmless prompts; LoRA tasks GSM8K, MATH, MMLU-Pro (WikiText-LM only within its remaining slack)
    contrast_groups = (2 * r_f * 2 + r_f) * sc
    lora_groups = r_f * sl
    d = {
        "sycophancy_real": 24 + nb + nd + nt + design["n_gen_syc"] + design["e2_shard"],
        "wikitext_nll": 24 + nb + nd + nt,
        "gsm8k_gold_lp": 24 + nb + nd + nt + (design["n_gen_gsm"] if design["n_rep_sec"] else 0),
        "mmlupro_gold_lp": 24 + nb + nd + nt,
        "refusal_margin_harmful": 24 + nb + nd + nt,
        "refusal_margin_harmless": 24 + nb + nd + nt,
        "mathcomp_gold_lp": nd + nt + (design["n_gen_math"] if design["n_rep_sec"] else 0),
        "construction_contrast_lora": contrast_groups + lora_groups,
    }
    return d


def feasible_pools(design, available: dict) -> dict:
    dem = pool_demand(design)
    out = {}
    for o, n in dem.items():
        if o == "construction_contrast_lora":
            avail = available.get("construction_large", 10**9)
        else:
            avail = available[o]
        out[o] = {"demand": n, "available": avail, "ok": n <= avail}
    return out


def designs(grid=GRID):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals, strict=True))


@dataclasses.dataclass
class Selection:
    design: dict | None
    price: dict | None
    powered: dict
    unresolved: list
    n_considered: int
    n_feasible: int


# ------------------------------------------------------------------ E1 secondaries (GSM8K, MATH L4-5; §6.6.3)


E1S_SCENARIOS = {
    "alt": {"tf": None, "gen": 0.05},
    "null_zero": {"tf": 0.0, "gen": 0.0, "null": True},
    "null_no_transfer": {"tf": None, "gen": 0.0, "null": True},
}
E1S_BASE_ACC = {"gsm8k_gold_lp": 0.60, "mathcomp_gold_lp": 0.30}


def sim_e1_secondary(rng, objective, n_rep_sec, B, n_dev, n_test, n_gen, sc, inp):
    """E1 secondary: WM-arm-only episodes (3 n_rep_sec), designated winner, its 11-bound guard set (pp + native),
    generated accuracy (Tango, one-sided 0.025) and single-response anomaly increase. TF alternative: 1 x the
    objective's per-candidate dev SE at n = 128 scaled to 0.5 SD_cand (assumption, labelled)."""
    o = inp["secondary"][objective]
    tf = sc["tf"] if sc["tf"] is not None else 0.5 * o["sd_cand"]
    sub = dict(inp, syc_se_64=o["se_64"], sd_cand=o["sd_cand"])
    winners = [
        w
        for w in (_search(rng, B, n_dev, sub, tf, o["sd_cand"], None) for _ in range(3 * n_rep_sec))
        if w
    ]
    if not winners:
        return {"reject": False, "designated": False}
    d = max(winners, key=lambda w: w["lcb"])
    se_t = o["se_64"] * math.sqrt(64 / n_test)
    ok_gain = d["true"] + se_t * rng.standard_normal() - Z * se_t > 0
    ok_guard = all(
        rng.standard_normal() * 1.0 < Z for _ in range(11)
    )  # guards at zero true change: each bound passes unless its estimate falls Z SE beyond the margin side
    p0 = E1S_BASE_ACC[objective]
    gen = sc["gen"] if d["carries"] else 0.0
    bs = rng.random(n_gen) < p0
    es = np.where(bs, rng.random(n_gen) >= 0.05, rng.random(n_gen) < (gen + 0.05 * p0) / (1 - p0))
    b, c = int((es & ~bs).sum()), int((~es & bs).sum())
    ok_gen = tango_lower(b, c, n_gen) > 0
    a0 = (rng.random(n_gen) < 0.005).astype(float)
    a1 = (rng.random(n_gen) < 0.005).astype(float)
    dd = a1 - a0
    sd = dd.std(ddof=1)
    ok_anom = (
        dd.mean() + stats.t.ppf(1 - ALPHA, n_gen - 1) * sd / math.sqrt(n_gen)
        if sd > 0
        else dd.mean()
    ) <= 0.02
    return {"reject": bool(ok_gain and ok_guard and ok_gen and ok_anom), "designated": True}


def run_secondary_cell(objective, cell, scenario, reps, inp):
    sc = E1S_SCENARIOS[scenario]
    rng = np.random.default_rng([SEED, _tag("e1s", objective, sorted(cell.items()), scenario)])
    n_gen = cell["n_gen_gsm"] if objective == "gsm8k_gold_lp" else cell["n_gen_math"]
    rows = [
        sim_e1_secondary(
            rng,
            objective,
            cell["n_rep_sec"],
            cell["B"],
            cell["n_dev"],
            cell["n_test"],
            n_gen,
            sc,
            inp,
        )
        for _ in range(reps)
    ]
    k = sum(r["reject"] for r in rows)
    return {
        "claim": f"e1s:{objective}",
        "cell": cell,
        "scenario": scenario,
        "reps": reps,
        "rate": k / reps,
        "ci95": clopper_pearson(k, reps),
        "power_lb": cp_lower_one_sided(k, reps),
        "null": bool(sc.get("null")),
    }


# ------------------------------------------------------------------ LLM reader arms (Claude client only)

# Measured by track B (docs/reviews/wme/wme-arms-2026-11-17.md §3, claude -p): fixed overhead 543 input tokens per
# call; a reader call at a 6-candidate pool with 2 own results used 1,812 input tokens; +~70 tokens per pool
# candidate. Own results are assumed at +80 tokens each (assumption). Output limits: reader 1,024; Thinker 512,
# Worker 1,024, Verifier 512; at most one Worker revision. Codex (~17.8k per call) cannot serve the arms.
READER = {
    "overhead": 543,
    "base_at_6": 1812,
    "per_candidate": 70,
    "per_own_result": 80,
    "out_reader": 1024,
    "out_team": (512, 1024, 512, 1024),
}
CLAUDE_PRICES = {  # $ per 1M tokens (input, output); Anthropic first-party list prices cached 2026-10-06
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def reader_tokens(pool_size: int, B: int) -> dict:
    """Worst-case tokens per (arm, episode): every call at its output limit, own results at their maximum."""
    rounds = 4
    b = B // rounds
    per = {"wsr": [0, 0], "team": [0, 0]}
    for r in range(rounds):
        own = r * b
        inp = (
            READER["base_at_6"]
            + READER["per_candidate"] * (pool_size - 6)
            + READER["per_own_result"] * (own - 2)
        )
        per["wsr"][0] += inp
        per["wsr"][1] += READER["out_reader"]
        per["team"][0] += 4 * inp + sum(READER["out_team"][:3])  # Worker revision sees the draft
        per["team"][1] += sum(READER["out_team"])
    return {k: {"input": v[0], "output": v[1], "total": v[0] + v[1]} for k, v in per.items()}


def reader_cap(pool_size: int, B: int) -> int:
    """One cap for both LLM arms: the team's worst case, rounded up to the next 10k."""
    t = reader_tokens(pool_size, B)["team"]["total"]
    return int(math.ceil(t / 10_000) * 10_000)


def reader_usd(pool_size: int, B: int, n_episodes: int, model: str) -> float:
    pi, po = CLAUDE_PRICES[model]
    t = reader_tokens(pool_size, B)
    return n_episodes * sum(v["input"] * pi + v["output"] * po for v in t.values()) / 1e6
