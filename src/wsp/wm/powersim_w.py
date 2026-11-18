"""Stage-W power simulations (docs/council/2026-11-18/RULING-wme-stagew.md §7-§8). EXPLORATORY; CPU only.

* **H-T2** (:func:`calibrate`, :func:`sim_world`, :func:`replicate`). The alternative world is calibrated on the
  complete stage-W paths: truth = the WM fitted on all of them, evaluated on a stage-W run's feature trajectories
  (S, L, M) scaled by a run factor LogNormal(0, 0.3), plus a path random effect on each logit whose variance tau^2 is
  the method-of-moments excess of the cross-fitted path offsets over their sampling variance; counts are binomial
  with the stage-W base cells. Simulated runs carry 3 paths; each path has its own members (no shared point).
  Nulls: N1 (features shuffled across paths) and N2 (curves permuted within type) applied to an alternative world;
  N3 (each type's mean truth curve plus the path effect, independent of the features). Every replicate refits the
  full ``wm.stagew.analyse`` pipeline (4 folds by run, 3 inner folds).
* **Empirical** (:func:`empirical_power`): P paths resampled from the observed stage-W path scores, the same IU
  sign-flip and Holm rule.
* **E2** (:func:`e2_inputs`, :func:`e2_power`): WM-prior vs no-prior two-rung successive halving over a 512-member
  sycophancy library per episode, at total-compute parity (the WM's bank and R1 field compute converted into extra
  full evaluations for best-of-N), family-stratified sign-flip at alpha 0.0125.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from wsp.wm import stagew as W

ROOT = Path(__file__).resolve().parents[3]
P_GRID = (16, 24, 32, 40)
REPS = 400
POWER_MIN = 0.8
NULL_MAX = 0.035
RUN_FACTOR_SD = 0.3
KINDS = ("alt", "n1", "n2", "n3")


# ------------------------------------------------------------------ H-T2


def _traj(world):
    out = {}
    for p in world["paths"]:
        if p["type"] in "SLM":
            out.setdefault(p["runs"][0], {})[p["type"]] = np.stack(
                [world["members"][k]["phi"] for k in p["keys"]]
            )
    return {r: v for r, v in out.items() if set(v) == set("SLM")}


def _info(wm, members, offsets):
    """Sampling information of the two offsets per benchmark at the fitted values."""
    X = np.stack([m["phi"] for m in members])
    z = wm.logits(X, offsets)
    out = {}
    for p in W.PANELS:
        n = np.stack([m["counts"][p][:, 0] for m in members])
        pf, pc = W._sig(z[p][0]), W._sig(z[p][1])
        out[p] = (float((n * pf * (1 - pf)).sum()), float((n * pf * pc * (1 - pc)).sum()))
    return out


def calibrate(world) -> dict:
    """Truth WM on all complete S/L/M paths, run trajectories and the path random-effect variances."""
    runs = W._runs(world)
    allm = W._train_members(world, runs) + [world["base"]]
    truth = W.WM().fit(allm, world["cells"])
    deltas = {p: ([], []) for p in W.PANELS}
    sampv = {p: ([], []) for p in W.PANELS}
    for r in runs:
        wm = W.WM().fit(
            W._train_members(world, [q for q in runs if q != r]) + [world["base"]], world["cells"]
        )
        wm.offset_penalty = 1e-6
        for p in world["paths"]:
            if p["type"] not in "SLM" or p["runs"][0] != r:
                continue
            ms = [world["members"][k] for k in p["keys"]]
            off = wm.fit_offsets(ms)
            inf = _info(wm, ms, off)
            for pn in W.PANELS:
                for i in range(2):
                    deltas[pn][i].append(off[pn][i])
                    sampv[pn][i].append(1.0 / max(inf[pn][i], 1e-9))
    tau = {}
    for pn in W.PANELS:
        tau[pn] = tuple(
            float(math.sqrt(max(0.0, np.var(deltas[pn][i], ddof=1) - np.mean(sampv[pn][i]))))
            for i in range(2)
        )
    return {
        "truth": truth,
        "traj": _traj(world),
        "tau": tau,
        "cells": world["cells"],
        "N": world["N"],
        "offsets_raw": {pn: [list(map(float, v)) for v in deltas[pn]] for pn in W.PANELS},
    }


def truth_logits(cal, run, ptype, factor, kind="alt") -> dict:
    tr = cal["truth"]
    if kind == "n3":
        zs = [tr.logits(cal["traj"][r][ptype]) for r in sorted(cal["traj"])]
        return {
            p: (np.mean([z[p][0] for z in zs], 0), np.mean([z[p][1] for z in zs], 0))
            for p in W.PANELS
        }
    return tr.logits(factor * cal["traj"][run][ptype])


def runs_for(n_paths: int) -> int:
    return math.ceil(n_paths / 3)


def sim_world(cal, n_paths, rng, kind="alt") -> dict:
    real = sorted(cal["traj"])
    cells = cal["cells"]
    members, paths = {}, []
    for i in range(runs_for(n_paths)):
        run = f"s{i:02d}"
        src = real[int(rng.integers(len(real)))]
        f = float(np.exp(rng.normal(0.0, RUN_FACTOR_SD)))
        for t in "SLM":
            phi = f * cal["traj"][src][t]
            z = truth_logits(cal, src, t, f, "n3" if kind == "n3" else "alt")
            u = {
                p: (rng.normal(0, cal["tau"][p][0]), rng.normal(0, cal["tau"][p][1]))
                for p in W.PANELS
            }
            keys = [f"{run}|{t}|{j}" for j in range(1, W.N_POINTS + 1)]
            for j, k in enumerate(keys):
                counts = {
                    p: W.generate_counts(
                        z[p][0][j] + u[p][0], z[p][1][j] + u[p][1], cells[p]["n"], rng
                    )
                    for p in W.PANELS
                }
                members[k] = {"phi": phi[j], "counts": counts, "y": W.y_of(counts)}
            paths.append(
                {
                    "id": f"{t}:{run}",
                    "type": t,
                    "runs": (run,),
                    "keys": keys,
                    "x": list(range(1, 9)),
                }
            )
    z0 = cal["truth"].logits(np.zeros((1, len(W.FEATURES))))
    bc = {p: W.generate_counts(z0[p][0][0], z0[p][1][0], cells[p]["n"], rng) for p in W.PANELS}
    w = {
        "cells": cells,
        "members": members,
        "base": {"phi": np.zeros(len(W.FEATURES)), "counts": bc, "y": W.y_of(bc)},
        "paths": paths,
        "N": cal["N"],
    }
    seed = int(rng.integers(2**31))
    if kind == "n1":
        return W.null_shuffle_features(w, seed=seed)
    if kind == "n2":
        return W.null_permute_paths(w, seed=seed)
    return w


def replicate(cal, n_paths, kind, seed, lcpfn=None, n_flip=10_000) -> dict:
    rng = np.random.default_rng([seed, n_paths, KINDS.index(kind)])
    w = sim_world(cal, n_paths, rng, kind)
    res = W.analyse(w, n_flip=n_flip, lcpfn=lcpfn, inner=True, n_folds=4, n_inner=3, seed=seed)
    return {
        "reject": {k: bool(v["reject"]) for k, v in res["settings"].items()},
        "p_iu": {k: float(v["p_iu"]) for k, v in res["settings"].items()},
        "C": {k: v["C"] for k, v in res["settings"].items()},
        "n_paths": len(res["paths"]),
    }


def _p_fast(d, S):
    T = d.mean()
    Ts = S @ d / len(d)
    return (1 + np.sum(Ts >= T - 1e-12 * max(1.0, abs(T)))) / (len(S) + 1)


def empirical_power(E, baselines, n_paths, reps=2000, seed=0, n_flip=10_000, alpha=W.ALPHA) -> dict:
    """E {setting: {arm: per-path scores}}; resample n_paths paths; IU sign-flip within settings, Holm across."""
    rng = np.random.default_rng(seed)
    S = np.random.default_rng(seed + 1).choice(np.array([-1.0, 1.0]), size=(n_flip, n_paths))
    P0 = len(next(iter(E.values()))["wm"])
    hits = {k: 0 for k in E}
    for _ in range(reps):
        ix = rng.integers(0, P0, n_paths)
        pv = {}
        for k, Ek in E.items():
            wm = np.asarray(Ek["wm"])[ix]
            pv[k] = {b: float(_p_fast(np.asarray(Ek[b])[ix] - wm, S)) for b in baselines[k]}
        dec = W.decide(pv, alpha)
        for k in hits:
            hits[k] += dec[k]["reject"]
    return {k: v / reps for k, v in hits.items()}


def cp_upper(k, n, level=0.95):
    return 1.0 if k >= n else float(stats.beta.ppf(level, k + 1, n - k))


def cp_lower(k, n, level=0.95):
    return 0.0 if k <= 0 else float(stats.beta.ppf(1 - level, k, n - k + 1))


def futile(counts: dict, n: int) -> bool:
    """True when every setting's one-sided 95 % upper bound of the power is below 0.8."""
    return all(cp_upper(c, n) < POWER_MIN for c in counts.values())


# ------------------------------------------------------------------ E2: WM vs direct multi-fidelity best-of-N


def _family(name):
    return {"lora": "lora_tv", "steer": "steer", "projrm": "proj_remove", "proj": "proj_remove"}[
        name.split(":")[0]
    ]


def e2_inputs() -> dict:
    """Planning inputs from committed reports only (pilot-2 sycophancy candidates and R2 steps, v2 inputs, v2b price,
    the frozen manifest's unit times)."""
    p2 = json.loads((ROOT / "docs/reviews/wme/wme-pilot2-2026-11-16.json").read_text())
    v2 = json.loads((ROOT / "docs/reviews/wme/wme-powersim-v2-2026-11-17.json").read_text())[
        "inputs"
    ]
    v2b = json.loads((ROOT / "docs/reviews/wme/wme-powersim-v2b-2026-11-17.json").read_text())
    man = json.loads((ROOT / "docs/prereg/wme_manifest.json").read_text())
    rows = p2["E2"]["9b"]["objectives"]["sycophancy_real"]["rows"]
    pools = {"steer": [], "proj_remove": [], "lora_tv": []}
    for r in rows:
        pools[_family(r["candidate"])].append((float(r["gain"]), bool(r["breaches"])))
    stand_in = {}
    if not pools[
        "proj_remove"
    ]:  # pilot 2 scored no proj_remove candidate on sycophancy: steer stands in
        pools["proj_remove"] = list(pools["steer"])
        stand_in["proj_remove"] = "steer"
    r2 = [r for r in p2["R2"]["rows"] if r["objective"] == "sycophancy_real"]
    rho = float(stats.spearmanr([r["static"] for r in r2], [r["observed"] for r in r2]).statistic)
    sel = v2b["selected"]
    n_ep = 3 * int(sel["n_rep"])
    unit = man["cost.unit_times"]["unit"]
    t_b = float(v2b["price"]["t_bundle_s"])
    bank = float(v2b["price"]["bank_gpu_h"])
    field = float(unit["field_s"])
    B = int(sel["B"])
    return {
        "pools": pools,
        "stand_in": stand_in,
        "rho_hat": rho,
        "rho_n_steps": len(r2),
        "syc_se_64": float(v2["syc_se_64"]),
        "jitter_sd": 0.5 * float(v2["sd_cand_mad"]),
        "n_episodes": n_ep,
        "B": B,
        "n_dev": int(sel["n_dev"]),
        "n_test": int(sel["n_test"]),
        "t_bundle_s": t_b,
        "overhead_s": float(unit["overhead_s"]),
        "bank_gpu_h": bank,
        "field_s": field,
        "budget_bon": B + (bank * 3600 + field) / n_ep / t_b,
        "library_size": 512,
        "p_dev_guard_error": 0.1,
    }


def sh_costs(t_bundle, overhead, n_dev=64, n_screen=16):
    """Cost (in full-evaluation units) of a 16-item screen and of a full 64-item evaluation."""
    var = t_bundle - overhead
    return (n_screen / n_dev * var + overhead) / t_bundle, 1.0


def promote_cost(t_bundle, overhead):
    """A promotion scores the remaining 48 items and re-applies the edit (one more overhead)."""
    c16, c64 = sh_costs(t_bundle, overhead)
    return c64 - c16 + overhead / t_bundle


def screened(budget, t_bundle, overhead):
    """Largest K with K screens plus ceil(K/4) promotions within budget (full-evaluation units)."""
    c16, _ = sh_costs(t_bundle, overhead)
    inc = promote_cost(t_bundle, overhead)
    k = 0
    while (k + 1) * c16 + math.ceil((k + 1) / 4) * inc <= budget + 1e-12:
        k += 1
    return k


def _arm(rng, g, breach, order, budget, inp):
    """Two-rung successive halving over ``order``; returns the winner's index or None (the base)."""
    se = inp["syc_se_64"]
    k = min(len(order), screened(budget, inp["t_bundle_s"], inp["overhead_s"]))
    if k == 0:
        return None
    cand = np.asarray(order[:k])
    d16 = g[cand] + se * math.sqrt(64 / 16) * rng.standard_normal(k)
    top = cand[np.argsort(-d16)[: math.ceil(k / 4)]]
    d16t = d16[np.argsort(-d16)[: math.ceil(k / 4)]]
    d48 = g[top] + se * math.sqrt(64 / 48) * rng.standard_normal(len(top))
    d64 = (16 * d16t + 48 * d48) / 64
    lcb = d64 - 1.959963984540054 * se
    gpass = np.where(rng.random(len(top)) < inp["p_dev_guard_error"], breach[top], ~breach[top])
    ok = (lcb > 0) & gpass
    if not ok.any():
        return None
    return int(top[np.argmax(np.where(ok, lcb, -np.inf))])


def e2_episode(rng, fam, inp, rho, budget_wm, budget_bon, exchangeable=False):
    pool = inp["pools"][fam]
    M = inp["library_size"]
    ix = rng.integers(0, len(pool), M)
    g = np.array([pool[i][0] for i in ix]) + inp["jitter_sd"] * rng.standard_normal(M)
    breach = np.array([pool[i][1] for i in ix])
    zg = (g - g.mean()) / (g.std() + 1e-12)
    prior = rho * zg + math.sqrt(max(0.0, 1 - rho**2)) * rng.standard_normal(M)
    order_wm = rng.permutation(M) if exchangeable else np.argsort(-prior)
    order_bon = rng.permutation(M)
    w_wm = _arm(rng, g, breach, order_wm, budget_wm, inp)
    w_bon = _arm(rng, g, breach, order_bon, budget_bon, inp)
    se_t = inp["syc_se_64"] * math.sqrt(64 / inp["n_test"])
    shared = math.sqrt(0.5) * se_t * rng.standard_normal()

    def score(w):
        if w is None:
            return 0.0
        x = g[w] + shared + math.sqrt(0.5) * se_t * rng.standard_normal()
        return min(x, 0.0) if breach[w] else x

    return score(w_wm) - score(w_bon), (0.0 if w_wm is None else g[w_wm]) - (
        0.0 if w_bon is None else g[w_bon]
    )


def e2_power(
    inp, rho, budget_bon, reps=1000, seed=0, alpha=0.0125, exchangeable=False, n_flip=2000
) -> dict:
    from wsp.wm import signflip as SF

    rng = np.random.default_rng(seed)
    fams = np.array(["steer", "proj_remove", "lora_tv"] * (inp["n_episodes"] // 3))
    k, ds, truths = 0, [], []
    for _ in range(reps):
        out = [e2_episode(rng, f, inp, rho, inp["B"], budget_bon, exchangeable) for f in fams]
        d = np.array([o[0] for o in out])
        r = SF.e2_test(d, fams, n_flip=n_flip, seed=int(rng.integers(2**31)), alpha=alpha)
        k += r["p"] <= alpha
        ds.append(d.mean())
        truths.append(np.mean([o[1] for o in out]))
    return {
        "rate": k / reps,
        "reps": reps,
        "power_lb": cp_lower(k, reps),
        "ci95_upper": cp_upper(k, reps),
        "mean_d": float(np.mean(ds)),
        "mean_true_gain_diff": float(np.mean(truths)),
        "rho": rho,
        "budget_wm": inp["B"],
        "budget_bon": budget_bon,
        "alpha": alpha,
    }
