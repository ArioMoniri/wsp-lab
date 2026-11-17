"""Outcome-blind power simulator for PREREG_WME §8 (Board A condition 11). BUILT AND PRICED; THE FULL RUN IS GATED.

``scripts/wme_powersim.py price`` runs a small CPU pilot to price the grid; ``run`` refuses unless
``--board-a-approval`` names the approving review. Inputs are pilot-2 *design* quantities only (paired SEs, the
E2 between-candidate spread under a frozen robust estimator) and the bank's planned closure graph; no
confirmatory outcome exists or is read.

**Inputs** (:func:`pilot2_inputs`). Paired SE per objective: the median step SE of the pilot-2 R2 rows (validation
items, n = 64), scaled by sqrt(64 / n) for other item counts. Guard SEs on the pp scale: medians over the 32
pilot-2 sycophancy candidates (dev, n = 128). Between-candidate SD of the dev gain: **1.4826 x MAD over all 32
candidates** (robust; not the 0.82 raw SD, which three LoRA candidates dominate, and not a favourable subset).
Component sizes: the bank recipe's source-policy graph (h = 3-4 state counts per component).

**H-T** (:func:`sim_ht`). Per component (sizes resampled from the bank graph), states x six objectives; labels
y_obs = y + eps (paired-SE units; Gaussian or t5 heavy tails; objectives correlated rho_obj, component random
effect rho_comp). Two static candidates (StatNN, refitted F3u) with biases of different scale; the registered
comparator is the one with the smaller inner-fold error (selection on independent inner noise). Null: WM error
exchangeable with the *better* static (so a wrong inner selection can inflate size, as in the real pipeline).
Alternative: WM bias shrunk so the population C_T equals ``ct`` (0.15 as written; 0.075 smaller). Statistic: C_T
with the component-cluster bootstrap-t one-sided 95% LB (``wm.confirm_stats``); reject when LB > 0 **and** the
fixed-endpoint control passes (LB on alternates, path noise) **and** the inert check passes (Bernoulli).

**H-E1** (:func:`sim_e1`). B candidates evaluated on dev (selection by dev gain LCB with dev guard pass; winner's
curse included), then the winner scored once on n_test sealed items: primary gain (z = 1.96 two-sided LB > 0),
six guardrails simultaneously (Bonferroni), generation (paired non-sycophancy LB > 0 on n_gen questions), neutral
preservation (LB >= -0.02) and anomalies (incidence <= 0.02, paired-increase UCB <= 0.02). Nulls: zero benefit;
boundary-unsafe (best candidate's guard exactly at its margin); no generation transfer; boundary anomaly (2%);
neutral destruction. Alternatives: benefit delta with generation transfer dp.

**H-E2** (:func:`sim_e2`). Episodes x (WM-guided, StatNN-guided) winners; difference D_e = Delta + episode noise
(sd = between-candidate SD x ``search_noise``) + paired test noise; shared-source correlation within family;
coverage failure (prob ``p_ranks``) switches both arms to ranks (effect x ``rank_factor``; symmetric). Inference:
cluster bootstrap LB (2.5th percentile) > 0, clustered by independent episode source (``"episode"``) or by family
(``"family"``, three clusters: flagged not estimable). Holm family over nine null secondary arms (FWER reported).

**Outputs** per (claim, cell, scenario): rejection rate with Clopper-Pearson 95% interval (size under nulls; power
and its one-sided 95% lower bound under alternatives), lower- and upper-tail miss rates of the interval for the
true contrast (H-T, E2) with Clopper-Pearson intervals, and Monte Carlo replicate counts. Requirements (§8, as a
simulation tolerance for nominal 5% tests): size <= 0.07, tail coverage error <= 0.07, power LB >= 0.80, >= 1,000
replicates per cell.

**GPU feasibility** (:func:`gpu_cost`): bank states x seconds per state (pilot-2 T2 rate scaled by validation
items; the pilot-1 formula as a lower variant), plus per-state oracle gradients for retained slope grounding
(44.2 s/state, pilot-2 R2), plus confirmatory evaluation (bundles x dev-panel seconds, test scoring, generation),
against 12 GPU-h bank + 10 GPU-h confirmatory; infeasible cells are "unresolved — not powered".
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from wsp.wm import confirm_stats as CS

OBJ6 = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)
GUARDS = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mathcomp_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
)
REQUIREMENTS = {
    "size_max": 0.07,
    "coverage_error_max": 0.07,
    "power_lb_min": 0.80,
    "replicates_min": 1000,
}
BUDGET_GPU_H = {"bank": 12.0, "confirmatory": 10.0}
N_BOOT = 2000
SEED = 20261117


# ------------------------------------------------------------------ inputs


def pilot2_inputs(report_path) -> dict:
    d = json.loads(Path(report_path).read_text())
    se = {}
    for r in d["R2"]["rows"]:
        se.setdefault(r["objective"], []).append(r["se"])
    rows = d["E2"]["9b"]["objectives"]["sycophancy_real"]["rows"]
    gains = np.array([r["gain"] for r in rows])
    gse = {}
    for r in rows:
        for k, v in r["guards"].items():
            gse.setdefault(k, []).append(v["se"])
    led = {r["phase"]: r["wall_seconds"] for r in d["ledger"]["rows"]}
    return {
        "paired_se_validation_n64": {k: float(np.median(v)) for k, v in se.items()},
        "dev_gain_se_n128": float(np.median([r["se"] for r in rows])),
        "between_candidate_sd_robust": float(1.4826 * np.median(np.abs(gains - np.median(gains)))),
        "between_candidate_sd_raw": float(gains.std(ddof=1)),
        "guard_se_pp_dev_n128": {k: float(np.median(v)) for k, v in gse.items()},
        "static_median_h3plus_se": d["T2"]["descriptive"]["per_objective_median_h3plus"],
        "seconds": {
            "t2_per_state": led["t2"] / d["T2"]["n_states"],
            "oracle_grad_per_state": d["R2"]["mean_grad_seconds"],
            "dev_eval_per_bundle": led["e2-9b-score"] / d["E2"]["9b"]["unique_evaluations"],
        },
        "source": "docs/reviews/wme/" + Path(report_path).name,
    }


def component_test_sizes(recipe, policy="source") -> list:
    """Number of h = 3-4 trajectory states per closure component of the planned bank."""
    from wsp.wm import bank as B

    comp = recipe["components"][policy]["assignment"]
    sizes = {c: 0 for c in set(comp.values())}
    for t in B.trajectories(recipe):
        rid = t["root"] if t["kind"] == "ladder" else t["roots"][0]
        sizes[comp[rid]] += sum(1 for s in t["states"] if s["h"] in B.TEST_H)
    return sorted(sizes.values(), reverse=True)


# ------------------------------------------------------------------ cells and scenarios


@dataclasses.dataclass(frozen=True)
class Cell:
    n_components: int = 44
    n_episodes: int = 12
    n_test: int = 160
    n_gen: int = 160
    B: int = 32


GRID = {
    "n_components": (20, 30, 44, 60),
    "n_episodes": (6, 9, 12, 18, 24),
    "n_test": (96, 128, 160),
    "n_gen": (96, 128, 160),
    "B": (16, 32, 48),
}
SCENARIOS = {
    "ht": {
        "null": {"ct": 0.0},
        "null_heavy": {"ct": 0.0, "tails": "t5"},
        "alt": {"ct": 0.15},
        "alt_small": {"ct": 0.075},
        "alt_heavy": {"ct": 0.15, "tails": "t5"},
    },
    "e1": {
        "null_zero": {"delta": 0.0, "dp": 0.0},
        "null_guard_boundary": {"delta": 0.10, "dp": 0.10, "guard_at_margin": True},
        "null_no_generation": {"delta": 0.10, "dp": 0.0},
        "null_anomaly_boundary": {"delta": 0.10, "dp": 0.10, "anomaly": 0.02},
        "null_neutral_destroyed": {"delta": 0.10, "dp": 0.10, "neutral_drop": 0.05},
        "alt": {"delta": 0.10, "dp": 0.10},
        "alt_small": {"delta": 0.05, "dp": 0.05},
        "alt_heavy": {"delta": 0.10, "dp": 0.10, "tails": "t5"},
    },
    "e2": {
        "null": {"delta_sd": 0.0},
        "null_family_clusters": {"delta_sd": 0.0, "clusters": "family"},
        "null_shared_source": {"delta_sd": 0.0, "rho_family": 0.5},
        "alt": {"delta_sd": 0.5},
        "alt_small": {"delta_sd": 0.25},
        "alt_ranks": {"delta_sd": 0.5, "p_ranks": 1.0},
    },
}
DEFAULTS = {
    "rho_obj": 0.3,
    "rho_comp": 0.3,
    "tails": "normal",
    "inert_pass": 0.95,
    "path_noise": 0.05,
    "selection_noise": 0.3,
    "search_noise": 0.5,
    "rank_factor": 0.8,
    "p_ranks": 0.0,
    "rho_family": 0.0,
    "clusters": "episode",
    "base_rate": 0.4,
    "anomaly": 0.005,
    "n_secondary": 9,
    "b0": 12.0,
}


def _noise(rng, shape, tails):
    if tails == "t5":
        return rng.standard_t(5, size=shape) / math.sqrt(5 / 3)
    return rng.standard_normal(shape)


# ------------------------------------------------------------------ H-T


def _ht_draw(rng, sizes, p, wm_scale, n_obj=6):
    """Per (state, objective): |WM error|, |selected static error| in paired-SE units, cluster ids, the true
    population ratio inputs."""
    K = n_obj
    comp = np.repeat(np.arange(len(sizes)), sizes)
    n = len(comp)
    L = np.linalg.cholesky(p["rho_obj"] * np.ones((K, K)) + (1 - p["rho_obj"]) * np.eye(K))
    re = _noise(rng, (len(sizes), K), "normal") @ L.T
    scale = np.array([3.85, 2.38, 0.33, 9.77, 3.87, 1.27])[
        :K
    ]  # pilot-2 per-objective static medians (SE units)
    bias = scale * (
        math.sqrt(p["rho_comp"]) * re[comp]
        + math.sqrt(1 - p["rho_comp"]) * (_noise(rng, (n, K), p["tails"]) @ L.T)
    )
    b_alt = 1.3 * scale * _noise(rng, (n, K), p["tails"])  # the weaker static candidate
    wm_ind = scale * _noise(rng, (n, K), p["tails"])
    b_wm = wm_scale * wm_ind
    eps = _noise(rng, (n, K), p["tails"])
    e_true_static = np.abs(bias - eps)
    e_alt_static = np.abs(b_alt - eps)
    # inner-fold selection on independent noise: the better candidate is chosen with error
    inner = (
        np.abs(bias).mean() - np.abs(b_alt).mean() + p["selection_noise"] * rng.standard_normal()
    )
    e_static = e_true_static if inner < 0 else e_alt_static
    e_wm = np.abs(b_wm - eps)
    return e_wm.ravel(), e_static.ravel(), np.repeat(comp, K)


_CAL: dict = {}


def calibrate_wm_scale(ct: float, p: dict, n=400_000, seed=SEED) -> float:
    """WM bias scale so that the population C_T against the better static equals ``ct`` (bisection; cached)."""
    key = (ct, p["rho_obj"], p["tails"], n, seed)
    if key not in _CAL:
        _CAL[key] = _calibrate(ct, p, n, seed)
    return _CAL[key]


def _calibrate(ct, p, n, seed):
    rng = np.random.default_rng(seed)
    K = 6
    scale = np.array([3.85, 2.38, 0.33, 9.77, 3.87, 1.27])
    L = np.linalg.cholesky(p["rho_obj"] * np.ones((K, K)) + (1 - p["rho_obj"]) * np.eye(K))
    bias = scale * (_noise(rng, (n, K), p["tails"]) @ L.T)
    wm_ind = scale * _noise(rng, (n, K), p["tails"])
    eps = _noise(rng, (n, K), p["tails"])
    est = np.abs(bias - eps).sum()
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        r = np.abs(mid * wm_ind - eps).sum() / est
        lo, hi = (mid, hi) if r < 1 - ct else (lo, mid)
    return (lo + hi) / 2


def sim_ht(rng, cell: Cell, sc: dict, sizes_pool, wm_scale, n_boot=N_BOOT) -> dict:
    p = {**DEFAULTS, **sc}
    sizes = rng.choice(sizes_pool, size=cell.n_components, replace=True)
    sizes = np.maximum(1, sizes)
    e_wm, e_st, cl = _ht_draw(rng, sizes, p, wm_scale)
    seed = int(rng.integers(2**31))
    lb = CS.ct_lower_bound(e_wm, e_st, cl, n_boot=n_boot, seed=seed)
    ub = CS.ct_lower_bound(
        e_st, e_wm, cl, n_boot=n_boot, seed=seed + 1
    )  # LB of the reversed ratio -> UB
    ct_ub = 1 - 1 / (1 - ub["lb"]) if ub["estimable"] and ub["lb"] < 1 else float("nan")
    alt_noise = 1 + p["path_noise"] * rng.standard_normal(len(e_wm))
    lb_alt = CS.ct_lower_bound(
        e_wm * np.abs(alt_noise), e_st, cl, n_boot=max(200, n_boot // 4), seed=seed + 2
    )
    inert = rng.random() < p["inert_pass"]
    rej = bool(lb["estimable"] and lb["lb"] > 0 and lb_alt["lb"] > 0 and inert)
    return {
        "reject": rej,
        "lb": lb["lb"],
        "ub": ct_ub,
        "ct_hat": lb["ct"],
        "estimable": lb["estimable"],
    }


# ------------------------------------------------------------------ H-E1


def sim_e1(rng, cell: Cell, sc: dict, inp: dict) -> dict:
    p = {**DEFAULTS, **sc}
    sd_c = inp["between_candidate_sd_robust"]
    item_sd = inp["dev_gain_se_n128"] * math.sqrt(128)
    # candidate pool: B evaluated candidates; the best true benefit grows with B (search quality, b0)
    reach = 1 - math.exp(-cell.B / p["b0"])
    true = rng.normal(-sd_c, sd_c, cell.B)
    best = int(np.argmax(true))
    true[best] = (
        p["delta"] * reach / (1 - math.exp(-32 / p["b0"]))
        if p["delta"] > 0
        else min(0.0, true[best])
    )
    if p["delta"] <= 0:
        true = np.minimum(true, 0.0)
    dev_se = item_sd / math.sqrt(128)
    dev = true + dev_se * _noise(rng, cell.B, p["tails"])
    w = int(np.argmax(dev - 1.96 * dev_se))
    # sealed test
    se_t = item_sd / math.sqrt(cell.n_test)
    gain = true[w] + se_t * _noise(rng, 1, p["tails"])[0]
    gain_ok = gain - 1.96 * se_t > 0
    gse = inp["guard_se_pp_dev_n128"]
    limits = {
        g: (0.01 if g in ("wikitext_nll", "refusal_margin_harmless") else -0.01) for g in GUARDS
    }
    kinds = {
        g: ("ucb" if g in ("wikitext_nll", "refusal_margin_harmless") else "lcb") for g in GUARDS
    }
    guards = {}
    for i, g in enumerate(GUARDS):
        s = max(gse.get(g, 0.001), 1e-4) * math.sqrt(128 / cell.n_test)
        mu = 0.0
        if p.get("guard_at_margin") and i == 0 and w == best:
            mu = limits[g]
        guards[g] = (mu + s * _noise(rng, 1, p["tails"])[0], s, kinds[g], limits[g])
    guard_ok = CS.guard_pass(guards)["pass"]
    n = cell.n_gen
    p0 = p["base_rate"]
    dp = p["dp"] if w == best else 0.0
    base_c = rng.random(n) < p0
    edit_c = np.where(
        base_c, rng.random(n) > 0.05, rng.random(n) < dp / max(1e-9, 1 - p0) + 0.05 * p0 / (1 - p0)
    )
    d = edit_c.astype(float) - base_c.astype(float)
    g_lb = d.mean() - 1.96 * d.std(ddof=1) / math.sqrt(n) if d.std() > 0 else d.mean()
    neutral_base = rng.random(n) < 0.8
    drop = p.get("neutral_drop", 0.0)
    neutral_edit = np.where(neutral_base, rng.random(n) > drop / 0.8, rng.random(n) < 0.0)
    dn = neutral_edit.astype(float) - neutral_base.astype(float)
    n_lb = dn.mean() - 1.96 * dn.std(ddof=1) / math.sqrt(n) if dn.std() > 0 else dn.mean()
    a_rate = p["anomaly"] if w == best else DEFAULTS["anomaly"]
    ae = rng.random(2 * n) < a_rate
    ab = rng.random(2 * n) < DEFAULTS["anomaly"]
    inc = ae.astype(float) - ab.astype(float)
    inc_ucb = (
        inc.mean() + 1.645 * inc.std(ddof=1) / math.sqrt(2 * n) if inc.std() > 0 else inc.mean()
    )
    anom_ok = ae.mean() <= 0.02 and inc_ucb <= 0.02
    rej = bool(gain_ok and guard_ok and g_lb > 0 and n_lb >= -0.02 and anom_ok)
    return {
        "reject": rej,
        "winner_is_best": w == best,
        "parts": [gain_ok, guard_ok, g_lb > 0, n_lb >= -0.02, anom_ok],
    }


# ------------------------------------------------------------------ H-E2


def sim_e2(rng, cell: Cell, sc: dict, inp: dict, n_boot=N_BOOT) -> dict:
    p = {**DEFAULTS, **sc}
    sd_c = inp["between_candidate_sd_robust"]
    n = cell.n_episodes
    fam = np.arange(n) % 3
    reach = (1 - math.exp(-cell.B / p["b0"])) / (1 - math.exp(-32 / p["b0"]))
    delta = p["delta_sd"] * sd_c * reach
    if rng.random() < p["p_ranks"]:
        delta *= p["rank_factor"]
    fam_eff = rng.standard_normal(3)
    ep = math.sqrt(p["rho_family"]) * fam_eff[fam] + math.sqrt(1 - p["rho_family"]) * _noise(
        rng, n, p["tails"]
    )
    item_sd = inp["dev_gain_se_n128"] * math.sqrt(128)
    meas = item_sd * math.sqrt(2 * (1 - 0.5)) / math.sqrt(cell.n_test)
    D = delta + p["search_noise"] * sd_c * ep + meas * _noise(rng, n, p["tails"])
    clusters = fam if p["clusters"] == "family" else np.arange(n)
    k = len(set(clusters.tolist()))
    if k < CS.MIN_CLUSTERS:
        return {
            "reject": False,
            "estimable": False,
            "lb": float("nan"),
            "ub": float("nan"),
            "true": delta,
            "fwer_secondary": False,
        }
    b = CS.paired_lb(D, clusters, n_boot=n_boot, seed=int(rng.integers(2**31)))
    sec = {}
    for j in range(p["n_secondary"]):
        Dj = p["search_noise"] * sd_c * rng.standard_normal(n) + meas * rng.standard_normal(n)
        z = Dj.mean() / (Dj.std(ddof=1) / math.sqrt(n))
        sec[j] = float(2 * (1 - _phi(abs(z))))
    holm = CS.holm(sec)
    return {
        "reject": bool(b["lb"] > 0),
        "estimable": True,
        "lb": b["lb"],
        "ub": b["ub"],
        "true": delta,
        "fwer_secondary": any(holm.values()),
    }


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ------------------------------------------------------------------ cells


def run_cell(
    claim, cell: Cell, scenario: str, reps: int, inp: dict, sizes_pool, seed=SEED, n_boot=N_BOOT
) -> dict:
    sc = SCENARIOS[claim][scenario]
    tag = json.dumps([claim, scenario, dataclasses.astuple(cell)])
    rng = np.random.default_rng([seed, 0, int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)])
    rows = []
    if claim == "ht":
        ws = calibrate_wm_scale(sc["ct"], {**DEFAULTS, **sc})
    t = time.time()
    if claim == "ht":
        rows = [sim_ht(rng, cell, sc, sizes_pool, ws, n_boot) for _ in range(reps)]
        truth = sc["ct"]
    elif claim == "e1":
        rows = [sim_e1(rng, cell, sc, inp) for _ in range(reps)]
        truth = None
    else:
        rows = [sim_e2(rng, cell, sc, inp, n_boot) for _ in range(reps)]
        truth = None
    k = sum(r["reject"] for r in rows)
    out = {
        "claim": claim,
        "scenario": scenario,
        "cell": dataclasses.asdict(cell),
        "reps": reps,
        "rejection_rate": k / reps,
        "rejection_ci95": CS.clopper_pearson(k, reps),
        "power_lb_one_sided": CS.one_sided_lower(k, reps),
        "seconds": time.time() - t,
        "null": scenario.startswith("null"),
    }
    if claim in ("ht", "e2"):
        est = [r for r in rows if r.get("estimable", True) and np.isfinite(r["lb"])]
        tr = (lambda r: truth) if claim == "ht" else (lambda r: r["true"])
        lo = sum(tr(r) < r["lb"] for r in est)
        hi = sum(np.isfinite(r["ub"]) and tr(r) > r["ub"] for r in est)
        m = max(1, len(est))
        out["n_estimable"] = len(est)
        out["lower_tail_miss"] = lo / m
        out["lower_tail_ci95"] = CS.clopper_pearson(lo, m)
        out["upper_tail_miss"] = hi / m
        out["upper_tail_ci95"] = CS.clopper_pearson(hi, m)
    if claim == "e2":
        out["fwer_secondary"] = float(np.mean([r["fwer_secondary"] for r in rows]))
    return out


def evaluate_requirements(res: dict) -> dict:
    """Pass/fail of one cell against §8 (nominal 5% tests; tolerances as simulation tolerances)."""
    ok = res["reps"] >= REQUIREMENTS["replicates_min"]
    if res["null"]:
        ok = ok and res["rejection_rate"] <= REQUIREMENTS["size_max"]
    else:
        ok = ok and res["power_lb_one_sided"] >= REQUIREMENTS["power_lb_min"]
    for tail, nominal in (("lower_tail_miss", 0.05), ("upper_tail_miss", 0.025)):
        if tail in res:
            ok = ok and abs(res[tail] - nominal) <= REQUIREMENTS["coverage_error_max"]
    return {"pass": bool(ok)}


def grid_cells(claim) -> list:
    if claim == "ht":
        return [Cell(n_components=c) for c in GRID["n_components"]]
    if claim == "e1":
        return [
            Cell(n_test=t, n_gen=g, B=b)
            for t, g, b in itertools.product(GRID["n_test"], GRID["n_gen"], GRID["B"])
            if g <= t
        ]
    return [
        Cell(n_episodes=e, n_test=t, B=b)
        for e, t, b in itertools.product(GRID["n_episodes"], GRID["n_test"], GRID["B"])
    ]


def price(
    inp,
    sizes_pool,
    pilot_reps=20,
    n_boot=N_BOOT,
    reps_full=REQUIREMENTS["replicates_min"],
    cells_per_claim=2,
    seed=SEED,
) -> dict:
    """CPU pilot: time ``pilot_reps`` replicates on a few cells per claim and scenario; extrapolate to the full grid
    at ``reps_full`` replicates. Returns per-replicate seconds and CPU-h."""
    out = {"pilot_reps": pilot_reps, "reps_full": reps_full, "n_boot": n_boot, "claims": {}}
    total = 0.0
    for claim in ("ht", "e1", "e2"):
        cells = grid_cells(claim)
        pick = [cells[0], cells[-1]][:cells_per_claim]
        per_rep = []
        for c in pick:
            for s in SCENARIOS[claim]:
                r = run_cell(claim, c, s, pilot_reps, inp, sizes_pool, seed, n_boot)
                per_rep.append(r["seconds"] / pilot_reps)
        sec = float(np.mean(per_rep))
        n_cells = len(cells) * len(SCENARIOS[claim])
        cpu_h = sec * reps_full * n_cells / 3600
        total += cpu_h
        out["claims"][claim] = {
            "grid_cells": len(cells),
            "scenarios": len(SCENARIOS[claim]),
            "cell_x_scenario": n_cells,
            "seconds_per_replicate": sec,
            "cpu_h_full": cpu_h,
        }
    out["cpu_h_total"] = total
    return out


# ------------------------------------------------------------------ GPU feasibility


def gpu_cost(
    inp: dict,
    n_states: int,
    val_items: int,
    dev_items: int,
    cell: Cell,
    grounding: bool,
    n_arms_full: int = 11,
    pilot2_val_items: int = 448,
    pilot2_dev_items: int = 704,
) -> dict:
    s = inp["seconds"]
    t_state = s["t2_per_state"] * val_items / pilot2_val_items
    bank = n_states * t_state / 3600
    bank_p1 = (
        n_states * (7.8 * 3600 / 960) * val_items / pilot2_val_items / 3600
    )  # pilot-1 formula (lower variant)
    oracle = n_states * s["oracle_grad_per_state"] / 3600 if grounding else 0.0
    t_bundle = s["dev_eval_per_bundle"] * dev_items / pilot2_dev_items
    e2_primary = cell.n_episodes * 2 * cell.B * t_bundle / 3600
    e2_all = cell.n_episodes * n_arms_full * cell.B * t_bundle / 3600
    gen = (
        2 * cell.n_gen * (1 + 2 * cell.n_episodes) * 1.5 / 3600
    )  # ~1.5 s per greedy answer (assumption)
    test = (1 + 2 * cell.n_episodes) * t_bundle / 3600
    conf_primary = e2_primary + gen + test
    return {
        "bank_gpu_h": bank,
        "bank_gpu_h_pilot1_formula": bank_p1,
        "oracle_grounding_gpu_h": oracle,
        "bank_total_gpu_h": bank + oracle,
        "confirmatory_primary_gpu_h": conf_primary,
        "confirmatory_all_arms_gpu_h": e2_all + gen + test,
        "fits_bank": bank + oracle <= BUDGET_GPU_H["bank"],
        "fits_confirmatory_primary": conf_primary <= BUDGET_GPU_H["confirmatory"],
        "fits_confirmatory_all_arms": e2_all + gen + test <= BUDGET_GPU_H["confirmatory"],
    }


def measure_nested_fold_seconds(seed=SEED) -> dict:
    """Seconds to fit WM-E, MatchedStatic and StatNN once (one outer fold) at synthetic bank scale (default
    WMConfig, 5 members, ~430 trajectories of h <= 2); prices the fully nested-pipeline simulation variant."""
    from wsp.wm import model as M
    from wsp.wm import synthetic as S

    w = S.SynthWorld(kind="saturating", seed=0)
    tr = w.trajectories(n=216, horizon=2, seed=1) + w.trajectories(n=216, horizon=1, seed=2)
    comps = [i // 4 for i in range(len(tr))]
    out = {}
    for name, mk in (
        ("wm", lambda: M.WMEnsemble(M.WMConfig(), w.field(), w.base_observation(), n_seeds=5)),
        (
            "matched_static",
            lambda: M.MatchedStatic(M.WMConfig(), w.field(), w.base_observation(), n_seeds=5),
        ),
        ("statnn", lambda: M.StaticPredictor(M.WMConfig(), w.field(), n_seeds=5)),
    ):
        t = time.time()
        mk().fit(tr, comps)
        out[name] = time.time() - t
    out["per_fold"] = sum(out.values())
    return out


def price_nested(
    fold_seconds: float, folds=5, reps=REQUIREMENTS["replicates_min"], cell_scenarios=1
) -> dict:
    per_rep = folds * fold_seconds
    return {
        "seconds_per_replicate": per_rep,
        "cpu_h_per_cell_scenario": per_rep * reps / 3600,
        "cpu_h": per_rep * reps * cell_scenarios / 3600,
        "cell_scenarios": cell_scenarios,
    }


class NotApproved(PermissionError):
    pass


def run_full(approval: str | None, *a, **k):
    """The full >= 1,000-replicate simulation runs only after Board A's review approves the simulator."""
    if not approval:
        raise NotApproved(
            "the full power simulation runs only after Board A approval (pass --board-a-approval)"
        )
    raise NotImplementedError(
        "full run intentionally not wired in this build; see scripts/wme_powersim.py"
    )
