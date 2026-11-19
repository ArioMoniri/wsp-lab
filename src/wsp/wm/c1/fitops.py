"""E2 planning predictors from the bank (PREREG_WME §6.1, §6.4, §II.4 delta 1, §II.22.2): trajectory assembly,
fits, conformal scaling, the C1.4 coverage switch, the shared support object and the inert-control predictions.

* **Trajectories.** Every state's R row (``wm.c1.field.RCache``), label y = per-objective mean paired change
  against the base on the ``bank`` panel (equal item weights; one item per group), and observation o = the
  per-objective mean tap sketches followed by the per-item native values (amendment G11; layout ``sketch``,
  ``losses``). Ladder side histories are the chain of visited doses from the base; a composition's history is its
  first root's +1 ladder state (composition h = 1, the same theta-state) then its prefixes. Negative and inert
  controls are never trained on; calibration components never train (``e2run``).
* **Training set.** Histories with h <= 2 of every non-calibration component (both arms, ``e2run.
  fit_planning_predictors``: component bootstraps, component selection split, shrinkage, 5 members).
* **Conformal q_hat (C1.4).** Scores |y - yhat| / sigma~ on inner out-of-fold predictions at h <= 2: K_in = 4
  component folds (seed ``bank.folds.inner_seed``; seeded permutation within stratum, round-robin dealing); for each
  fold both predictors are refitted on the other folds; q_hat = the ceil((n + 1) 0.9)-th order statistic of the
  pooled scores (every objective and horizon).
* **Coverage check.** Final fits (all non-calibration components) with their q_hat, on calibration-component units at
  h in {3, 4}, pooled over objectives: coverage < 0.80 for either arm -> both arms in ranks-only mode.
"""

from __future__ import annotations

import math

import numpy as np

from wsp.wm import model as M
from wsp.wm.c1.schema import OBJ6

K_IN = 4
TARGET = 0.80
QUANT = 0.9


def obs_vector(state: dict) -> np.ndarray:
    sk = np.concatenate([np.asarray(state["sketch"][o], float) for o in OBJ6])
    vals = np.concatenate([np.asarray(state["values"][o], float) for o in OBJ6])
    return np.concatenate([sk, vals])


def obs_layout(base: dict) -> dict:
    n_sk = sum(len(base["sketch"][o]) for o in OBJ6)
    n_v = sum(len(base["values"][o]) for o in OBJ6)
    return {"sketch": (0, n_sk), "losses": (n_sk, n_sk + n_v)}


def label(state: dict, base: dict) -> np.ndarray:
    return np.array(
        [
            float(np.mean(np.asarray(state["values"][o]) - np.asarray(base["values"][o])))
            for o in OBJ6
        ]
    )


def se_row(state: dict, base: dict) -> np.ndarray:
    out = []
    for o in OBJ6:
        d = np.asarray(state["values"][o]) - np.asarray(base["values"][o])
        out.append(float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0)
    return np.array(out)


def histories(bank: dict) -> list:
    """[{"component", "kind", "states": [state ids h = 1..T], "R": [R_0..R_T], "y": [...], "obs": [...]}] for every
    prefix of every ladder side and composition order (controls and inert excluded)."""
    from wsp.wm.c1.field import spec_9b

    d = spec_9b().dim
    base = bank["base"]
    by = {s["state_id"]: s for s in bank["states"]}
    o0 = obs_vector(base)
    out = []
    sides: dict = {}
    for s in bank["states"]:
        if s["kind"] == "ladder":
            sides.setdefault((s["roots"][0], s["side"]), []).append(s)
    plus1 = {}
    for (root, side), ss in sides.items():
        ss.sort(key=lambda s: s["h"])
        if side == "pos":
            for s in ss:
                if abs(s["dose"] - 1.0) < 1e-12:
                    plus1[root] = s
        for T in range(1, len(ss) + 1):
            chain = ss[:T]
            out.append(_hist(chain[0]["component"], "ladder", chain, d, base, o0))
    for s in bank["states"]:
        if s["kind"] == "composition":
            order = s["order_id"]
            sides.setdefault(("comp", order), []).append(s)
    for (tag, order), ss in sides.items():
        if tag != "comp":
            continue
        ss.sort(key=lambda s: s["h"])
        first = plus1.get(ss[0]["prefix"][0])
        if first is None:
            continue  # the first root's +1 state is unlabelled (its side stopped earlier): no h = 1 history
        chain = [first] + ss
        for T in range(1, len(chain) + 1):
            out.append(_hist(ss[0]["component"], "composition", chain[:T], d, base, o0))
    _ = by
    return out


def _hist(component, kind, chain, d, base, o0):
    R = [np.zeros(d)] + [np.asarray(s["R"], float) for s in chain]
    y = [np.zeros(len(OBJ6))] + [label(s, base) for s in chain]
    obs = [o0] + [obs_vector(s) for s in chain]
    return {
        "component": component,
        "kind": kind,
        "states": [s["state_id"] for s in chain],
        "h": len(chain),
        "R": np.stack(R),
        "y": np.stack(y),
        "obs": np.stack(obs),
    }


def to_traj(h: dict, spec):
    from wsp.wm import wmcp as CP

    return CP.v0_trajectory_r(spec, h["R"], h["y"], h["obs"])


def inner_folds(components, strata: dict, seed: int, k: int = K_IN) -> dict:
    """{component: fold} by seeded permutation within stratum and round-robin dealing (Part I §4.6)."""
    rng = np.random.default_rng(int(seed))
    comps = sorted(set(components), key=str)
    fold, j = {}, 0
    for st in sorted({strata[c] for c in comps}):
        members = [c for c in comps if strata[c] == st]
        for i in rng.permutation(len(members)):
            fold[members[i]] = j % k
            j += 1
    return fold


def predict(model, base_obs, trajs):
    """(mean, sigma~) at each trajectory's final horizon, grouped by horizon (``adapter9b.predict_batch``)."""
    from wsp.wm import adapter9b as AD

    mu = np.zeros((len(trajs), len(OBJ6)))
    sg = np.zeros_like(mu)
    by: dict = {}
    for i, t in enumerate(trajs):
        by.setdefault(t["horizon"], []).append(i)
    for T, idx in by.items():
        out = AD.predict_batch(model, base_obs, [trajs[i] for i in idx])
        mu[idx] = out["mean"][:, T]
        sg[idx] = out["sigma"][:, T]
    return mu, sg


def conformal_q(scores) -> float:
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * QUANT)
    return float(s[min(k, n) - 1]) if n else float("nan")


def fit_all(
    bank: dict, calibration, strata: dict, inner_seed: int, base_obs, spec, fit_fn=None, log=None
) -> dict:
    """The full E2 fit: q_hat by inner folds, final fits, coverage, support and mode."""
    from wsp.wm import e2run as ER

    fit_fn = fit_fn or (
        lambda trajs, comps: ER.fit_planning_predictors(trajs, comps, spec, base_obs)
    )
    cal = set(calibration)
    H = histories(bank)
    train = [h for h in H if h["h"] <= 2 and h["component"] not in cal]
    covset = [h for h in H if h["h"] in (3, 4) and h["component"] in cal]
    folds = inner_folds([h["component"] for h in train], strata, inner_seed)
    scores = {"wm": [], "static": []}
    for f in range(K_IN):
        tr = [h for h in train if folds[h["component"]] != f]
        te = [h for h in train if folds[h["component"]] == f]
        if not te:
            continue
        fitted = fit_fn([to_traj(h, spec) for h in tr], [h["component"] for h in tr])
        tt = [to_traj(h, spec) for h in te]
        y = np.stack([h["y"][-1] for h in te])
        for arm in ("wm", "static"):
            mu, sg = predict(fitted[arm], fitted["base_obs"], tt)
            ok = sg > 0
            scores[arm] += list((np.abs(y - mu)[ok] / sg[ok]).ravel())
        if log:
            log(f"inner fold {f} done")
    q = {arm: conformal_q(scores[arm]) for arm in scores}
    final = fit_fn([to_traj(h, spec) for h in train], [h["component"] for h in train])
    cov = {}
    if covset:
        tt = [to_traj(h, spec) for h in covset]
        y = np.stack([h["y"][-1] for h in covset])
        for arm in ("wm", "static"):
            mu, sg = predict(final[arm], final["base_obs"], tt)
            inside = np.abs(y - mu) <= q[arm] * sg
            c = float(inside.mean())
            cov[arm] = {"coverage": c, "n": int(inside.size), "pass": bool(c >= TARGET)}
    mode = "lcb" if cov and all(v["pass"] for v in cov.values()) else "ranks"
    if not covset:
        mode = "ranks"
    from wsp.wm import adapter9b as AD

    support = AD.MahalanobisSupport(np.stack([h["R"][-1] for h in train]))
    base = bank["base"]
    by = {s["state_id"]: s for s in bank["states"]}
    se = np.stack([se_row(by[sid], base) for h in train for sid in h["states"][-1:]])
    f_k = np.quantile(se, 0.10, axis=0)
    return {
        "fitted": final,
        "q_hat": q,
        "coverage": cov,
        "mode": mode,
        "support": support,
        "f_k": f_k.tolist(),
        "n_train": len(train),
        "n_cov_units": len(covset),
        "n_scores": {k: len(v) for k, v in scores.items()},
        "train_components": sorted({h["component"] for h in train}, key=str),
    }


def inert_predictions(fit: dict, bank: dict, spec) -> dict:
    """Planning predictions on the registered inert controls (D-C2): the history of each construction's visited
    states, prediction and sigma~ (q_hat-scaled) at the endpoint."""
    out = {}
    o0 = obs_vector(bank["base"])
    for arm in ("wm", "static"):
        rows = []
        for ic in bank.get("inert", []):
            R = [np.zeros(spec.dim)] + [np.asarray(s["R"], float) for s in ic["states"]]
            from wsp.wm import wmcp as CP

            t = CP.v0_trajectory_r(spec, np.stack(R), obs=o0)
            mu, sg = predict(fit["fitted"][arm], fit["fitted"]["base_obs"], [t])
            rows.append(
                {
                    "spec": ic["spec"],
                    "pred": mu[0].tolist(),
                    "sigma": (fit["q_hat"][arm] * sg[0]).tolist(),
                }
            )
        out[arm] = rows
    return out


def base_artefact(bank: dict, theta0: str):
    from wsp.wm.access import Artefact, Tier

    base = bank["base"]
    return Artefact("tap", theta0, Tier.R0, {"obs": obs_vector(base), "layout": obs_layout(base)})


def _unused():
    return M
