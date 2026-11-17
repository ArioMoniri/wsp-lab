"""Cheap (error- and decision-level) simulator under RULING-wme-v2 A1-A4. Reuses ``wm.powersim_v1`` for the bank
graph, error model, E1 search model, pricing and requirements; the alternatives are **not re-tuned**.

* **H-T (A1, A2).** WM-CP is path-consistent, so C2-S and C2-D hold by construction and are not gates; C2-I stays a
  gate (ghost-free pass rate as v1). Inference: the component-level sign-flip test (``wm.signflip.ht_test``,
  10,000 flips, one-sided 0.025) on component means of |e_static| - |e_WM|; coverage is that of the inverted lower
  bound for the population mean difference. Nulls: WM-CP exchangeable with the selected static member, with normal
  and t3 tails. Alternative: C_T = 0.15 as in v1.
* **H-E2 (A3).** Episode paired differences as v1 (same model of search, shared test items and guarded scores);
  inference by the family-stratified sign-flip test (``wm.signflip.e2_test``).
* **H-E1 (A4).** v1's E1 model with the generation N extended over the pinned TriviaQA pool (:data:`GEN_GRID`).
"""

from __future__ import annotations

import math

import numpy as np

from wsp.wm import powersim_v1 as PV
from wsp.wm import signflip as SF

GEN_GRID = (100, 150, 200, 250, 500, 1000, 2000)
GRID = dict(PV.GRID, n_gen_syc=GEN_GRID)
N_FLIP_GRID = 2000  # cheap-grid flips; the selected cells are re-run at the registered 10,000
HT_SCENARIOS = {
    "null": {"ct": 0.0, "null": True},
    "null_heavy": {"ct": 0.0, "tails": "t3", "null": True},
    "alt": {"ct": 0.15},
    "alt_005": {"ct": 0.05},
    "alt_010": {"ct": 0.10},
    "alt_020": {"ct": 0.20},
    "heavy_alt": {"ct": 0.15, "tails": "t3"},
    "weak_families": {"ct": 0.15, "weak_families": True},
    "high_stop": {"ct": 0.15, "stop_share": 0.3},
}
HT_SELECT = ("null", "null_heavy", "alt")
E2_SELECT = ("null", "null_shared", "alt")


def sim_ht(rng, g, n_cal, n_bank, n_inert, sc, se_ratio, wm_scale, n_flip=N_FLIP_GRID):
    tails = sc.get("tails", "normal")
    rows, _ = PV.ht_units(rng, g, n_cal, sc)
    K = 6
    if len(rows) < 2:  # no (or one) held-out component: H-T is not estimable and never rejects
        return {
            "reject": False,
            "lb": float("nan"),
            "stat": float("nan"),
            "ct_hat": float("nan"),
            "c2i": True,
            "m": len(rows),
            "estimable": False,
        }
    scale = PV._scale_at(n_bank, se_ratio) * sc.get("scale_mult", 1.0)
    L = np.linalg.cholesky(0.3 * np.ones((K, K)) + 0.7 * np.eye(K))
    e_w, e_s, comp = [], [], []
    for ci, (_c, _st, n_u, _n_c, weak) in enumerate(rows):
        re = rng.standard_normal(K) @ L.T
        b = scale * (math.sqrt(0.3) * re + math.sqrt(0.7) * (PV._noise(rng, (n_u, K), tails) @ L.T))
        eps = PV._noise(rng, (n_u, K), tails)
        members = [
            b,
            1.3 * scale * PV._noise(rng, (n_u, K), tails),
            scale * PV._noise(rng, (n_u, K), tails),
        ]
        inner = [np.abs(m).mean() * (1 + 0.3 * rng.standard_normal()) for m in members]
        sel = int(np.argmin(inner))
        e_st = np.abs(members[sel] - eps)
        if weak:
            e_wm, e_st = np.abs(eps), np.abs(eps)
        elif sc["ct"] == 0:
            e_wm = np.abs(
                (1.3 if sel == 1 else 1.0) * scale * PV._noise(rng, (n_u, K), tails) - eps
            )
        else:
            e_wm = np.abs(wm_scale * scale * PV._noise(rng, (n_u, K), tails) - eps)
        e_w.append(e_wm.ravel())
        e_s.append(e_st.ravel())
        comp.append(np.full(e_wm.size, ci))
    e_w, e_s, comp = np.concatenate(e_w), np.concatenate(e_s), np.concatenate(comp)
    r = SF.ht_test(e_w, e_s, comp, n_flip=n_flip, seed=int(rng.integers(2**31)), ct_grid=0)
    y = 0.5 * rng.standard_normal((n_inert, K))
    c2i = bool(np.mean(np.abs(y) > 2.0) <= 0.10 and np.all(np.abs(y).mean(0) <= 1.0))
    return {
        "reject": bool(r["reject"] and c2i),
        "lb": r["lb"],
        "stat": r["stat"],
        "ct_hat": r["ct"],
        "c2i": c2i,
        "m": r["n_components"],
        "estimable": True,
    }


def _ht_truth(sc, g, n_cal, n_bank, se_ratio, wm_scale, seed=1):
    """Population mean of the component-mean difference under the scenario (large Monte Carlo), so coverage of
    the inverted bound can be scored against the truth (0 under nulls)."""
    if sc["ct"] == 0:
        return 0.0
    rng = np.random.default_rng([seed, 99])
    vals = []
    for _ in range(40):
        rows, _ = PV.ht_units(rng, g, n_cal, sc)
        r = sim_ht(rng, g, n_cal, n_bank, 12, sc, se_ratio, wm_scale, n_flip=10)
        vals.append(r["stat"])
        del rows
    return float(np.mean(vals))


def sim_e2(rng, n_rep, B, n_dev, n_test, sc, inp, n_flip=N_FLIP_GRID):
    """v1's E2 data model (search, shared test items, guarded scores), inferred by the family-stratified sign-flip
    test on the episode paired differences."""
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
    ep = math.sqrt(rho_f) * fe[fam] + math.sqrt(1 - rho_f) * PV._noise(rng, n, tails)
    base = 0.5 * sd_c * reach + 0.5 * sd_c * ep
    G = np.stack(
        [
            base + delta + 0.5 * sd_c * PV._noise(rng, n, tails),
            base + 0.5 * sd_c * PV._noise(rng, n, tails),
        ]
    )
    se_dev = inp["syc_se_64"] * math.sqrt(64 / n_dev)
    is_base = (G + se_dev * rng.standard_normal(G.shape) - PV.Z * se_dev) <= 0
    item_sd = inp["syc_se_64"] * 8.0
    shared = math.sqrt(0.5) * item_sd * rng.standard_normal(n_test)
    X = (
        G[:, :, None]
        + shared[None, None, :]
        + math.sqrt(0.5) * item_sd * PV._noise(rng, (2, n, n_test), tails)
    )
    gmu = np.where(rng.random(G.shape) < 0.05, -0.02, 0.0)
    Gi = gmu[:, :, None] + inp["guard_point_sd_item"] * rng.standard_normal((2, n, n_test))
    xm, gm = X.mean(-1), Gi.mean(-1)
    g = np.where(is_base, 0.0, np.where(gm >= -0.01, xm, np.minimum(xm, 0.0)))
    d = g[0] - g[1]
    r = SF.e2_test(d, fam, n_flip=n_flip, seed=int(rng.integers(2**31)))
    truth = float(np.mean(np.where(is_base, 0, G)[0] - np.where(is_base, 0, G)[1]))
    return {"reject": r["reject"], "lb": r["lb"], "true": truth}


def run_cell(claim, cell, scenario, reps, inp, n_flip=N_FLIP_GRID):
    import time

    rng = np.random.default_rng([PV.SEED, 2, PV._tag(claim, sorted(cell.items()), scenario)])
    t0 = time.time()
    if claim == "ht":
        sc = HT_SCENARIOS[scenario]
        g = PV.graph(cell["r_f"], cell["n_blk"])
        scale = PV._scale_at(cell["n_bank"], inp["se_ratio"])
        ws = PV.calibrate_wm_scale(sc["ct"], tails=sc.get("tails", "normal"), scale=scale)
        rows = [
            sim_ht(
                rng,
                g,
                cell["n_cal"],
                cell["n_bank"],
                cell.get("n_inert", 12),
                sc,
                inp["se_ratio"],
                ws,
                n_flip,
            )
            for _ in range(reps)
        ]
        truth = _ht_truth(sc, g, cell["n_cal"], cell["n_bank"], inp["se_ratio"], ws)
        est = [r for r in rows if r["estimable"]]
        cov = float(np.mean([truth >= r["lb"] for r in est])) if est else float("nan")
        extra = {
            "coverage": cov,
            "truth_mean_diff": truth,
            "mean_ct": float(np.mean([r["ct_hat"] for r in rows])),
            "c2i_pass": float(np.mean([r["c2i"] for r in rows])),
            "m": int(np.median([r["m"] for r in rows])),
        }
        null = bool(sc.get("null"))
    elif claim == "e2":
        sc = PV.E2_SCENARIOS[scenario]
        rows = [
            sim_e2(rng, cell["n_rep"], cell["B"], cell["n_dev"], cell["n_test"], sc, inp, n_flip)
            for _ in range(reps)
        ]
        extra = {"coverage": float(np.mean([r["true"] >= r["lb"] for r in rows]))}
        null = bool(sc.get("null"))
    else:
        r = PV.run_cell(claim, cell, scenario, reps, inp)
        r["v2"] = True
        return r
    k = sum(r["reject"] for r in rows)
    return {
        "claim": claim,
        "cell": cell,
        "scenario": scenario,
        "reps": reps,
        "rate": k / reps,
        "ci95": PV.clopper_pearson(k, reps),
        "power_lb": PV.cp_lower_one_sided(k, reps),
        "null": null,
        "seconds": time.time() - t0,
        "n_flip": n_flip,
        **extra,
    }


def passes(res) -> bool:
    if res["reps"] < PV.REQ["reps_min"]:
        return False
    ok = (
        res["rate"] <= PV.REQ["size_max"]
        if res["null"]
        else res["power_lb"] >= PV.REQ["power_lb_min"]
    )
    if res["claim"] == "ht" and res["null"] and "coverage" in res:
        ok = ok and res["coverage"] >= PV.REQ["coverage_min"]
    return bool(ok)
