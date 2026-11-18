"""Stage W2 (docs/council/2026-11-18/RULING-wme-stagew2.md; RULING-wme-post-stagew Q4). EXPLORATORY; CPU only.

Pure numpy/scipy (Python 3.10 compatible, so the comparators run in the LC-PFN venv).

* **Design** (:data:`RUNS`, :data:`SLOTS`, :func:`split_teacher`, :func:`ckpt_steps`, :func:`make_paths`,
  :func:`queue`): 15 fresh-seed H1-type runs (3 disjoint teacher shards x 5 learning rates), each with a training-step
  path S, a dose path L and a mask-growth path M on the stage-W grid thinned to j in {1, 2, 3, 4, 6, 8} (6 points;
  the three paths share the point (final step, lambda 1, all layers)); 45 paths, 16 unique members per run. Runs are
  dealt to three instance slots of five; each slot's queue is run-major.
* **Nested world model** (:func:`fit`, :func:`predict`, :func:`analyse`): prediction = the pooled same-path-type
  curve c_t(j) (the stage-W pooled baseline, fitted on the training runs) + s * g(m), where the edit-informed residual
  g(m) = ridge(alpha 1) map from the standardised type-centred features phi(m) - mean_t,j(phi) to the endpoint
  residuals y - c_t(j) of the training paths; s in {0, .25, .5, .75, 1} is chosen per setting by inner
  leave-one-run-out folds **inside the training runs only** (ties within 0.5 % to the smaller s, i.e. toward the
  pooled curve). At k = 2 the combined curve is offset by its mean residual on the k given points (exactly as the
  pooled baseline). s = 0 is the pooled curve itself, so the model is nested.
* **Primary contrast** (:func:`cluster_test`, :func:`holm`): d_p = E_pooled,p - E_nested,p (E = stage-W path score,
  mean |error|/SE over the points after k and the 4 endpoints); leave-one-run-out; one-sided sign-flip over **runs**
  (a run's paths share a point and a training run; exact enumeration up to 16 runs), statistic = mean over paths;
  lower bound by test inversion; Holm over the settings k = 0 and k = 2 at family alpha 0.025.
* **Nulls** (:func:`null_world`, :func:`calibrate_n3`, :func:`n3_world`, :func:`null_sizes`): N1 shuffled edit
  features and N2 path-label permutation exactly as stage W (``wm.stagew.null_shuffle_features`` /
  ``null_permute_paths``); N3 a no-edit-information world: each member's counts drawn from its type's pooled
  per-cell rates at its point plus a path random effect (method-of-moments tau) and binomial noise, features kept.
  Size = share of null replicates in which the whole pass rule holds.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import optimize

from wsp.wm import stagew as W

NAMESPACE = "wme-w2"
SALT = "wme-w2-panel-20261118"
PANELS = W.PANELS
N_DEV = 500
BUDGET = W.BUDGET
N_FEATURE = 32  # R1 feature items per benchmark
GRID = (
    1,
    2,
    3,
    4,
    6,
    8,
)  # stage-W point indices kept (fractions j/8 of the steps; lambda = j/4; j fine groups)
LAMS = tuple(j / 4 for j in GRID)
SHARDS = ("T1", "T2", "T3")
LRS = (5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4)
N_RUNS = 15
RUN_ORDER = [f"r{i + 1:02d}" for i in range(N_RUNS)]
RUNS = {r: (SHARDS[i % 3], LRS[i % 5]) for i, r in enumerate(RUN_ORDER)}
SLOTS = {"a": RUN_ORDER[0:5], "b": RUN_ORDER[5:10], "c": RUN_ORDER[10:15]}
KS = (0, 2)
MIN_PATHS = 40
ALPHA = 0.025
NULL_MAX = 0.035
SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SHRINK_TIE = 0.005
RIDGE_ALPHA = 1.0
EXACT_MAX_RUNS = 16
N_FLIP = 10_000
NULL_REPS = 1000
COMPARATORS = ("persistence", "pow2", "lc_ensemble", "lcpfn", "statnn", "static", "wm")


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}:{label}".encode()).hexdigest()[:8], 16)


def _rank(label: str, *parts) -> str:
    return hashlib.sha256(
        ":".join((NAMESPACE, label) + tuple(map(str, parts))).encode()
    ).hexdigest()


# ------------------------------------------------------------------ design


def split_teacher(rows, n_feature: int = N_FEATURE) -> dict:
    """Eligible rows (correct, non-empty answer, <= BUDGET tokens) -> {'feature', 'T1', 'T2', 'T3'}: ``n_feature``
    per benchmark held out (seed ``wme-w2:feature-items``), the rest dealt round-robin into three shards by
    ``wme-w2:shards``."""
    ok = [r for r in rows if r["correct"] and r["answer"] and r["n_tokens"] <= BUDGET]
    if len({(r["panel"], r["id"]) for r in ok}) != len(ok):
        raise ValueError(
            "duplicate (panel, id) among eligible teacher rows: a feature item could be trained on"
        )
    out = {"feature": [], **{s: [] for s in SHARDS}}
    for p in PANELS:
        rs = sorted(
            (r for r in ok if r["panel"] == p), key=lambda r: _rank("feature-items", p, r["id"])
        )
        out["feature"] += rs[:n_feature]
        rest = sorted(rs[n_feature:], key=lambda r: _rank("shards", p, r["id"]))
        for i, r in enumerate(rest):
            out[SHARDS[i % 3]].append(r)
    return out


def ckpt_steps(n_steps: int) -> list:
    s = [round(n_steps * j / 8) for j in GRID]
    if n_steps < 8 or s != sorted(set(s)) or s[0] < 1:
        raise ValueError(f"checkpoints not strictly increasing for {n_steps} steps: {s}")
    return s


def make_paths(final_steps: dict) -> list:
    """S/L/M paths per run (in RUN_ORDER); each {'id', 'type', 'runs', 'keys', 'x'} with x = GRID (stage-W units)."""
    out = []
    x = list(GRID)
    for r in [q for q in RUN_ORDER if q in final_steps] + sorted(set(final_steps) - set(RUN_ORDER)):
        T = final_steps[r]
        out.append(
            {
                "id": f"S:{r}",
                "type": "S",
                "runs": (r,),
                "keys": [W.key(W.member(r, t)) for t in ckpt_steps(T)],
                "x": x,
            }
        )
        out.append(
            {
                "id": f"L:{r}",
                "type": "L",
                "runs": (r,),
                "keys": [W.key(W.member(r, T, lam=lm)) for lm in LAMS],
                "x": x,
            }
        )
        out.append(
            {
                "id": f"M:{r}",
                "type": "M",
                "runs": (r,),
                "keys": [W.key(W.member(r, T, groups=j)) for j in GRID],
                "x": x,
            }
        )
    return out


def queue(paths, runs) -> list:
    """Unique member keys of ``runs``, run-major (S, L, M within a run): a deadline cuts at most one run."""
    seen, out = set(), []
    for r in runs:
        for t in "SLM":
            for p in paths:
                if p["type"] != t or p["runs"][0] != r:
                    continue
                for k in p["keys"]:
                    if k not in seen:
                        seen.add(k)
                        out.append(k)
    return out


def thin_world(world) -> dict:
    """A stage-W world restricted to the W2 grid (the non-evidential code check on stage-W data)."""
    w = dict(world)
    w["paths"] = []
    for p in world["paths"]:
        if p["type"] not in "SLM":
            continue
        q = dict(p)
        q["keys"] = [p["keys"][j - 1] for j in GRID]
        q["x"] = list(GRID)
        w["paths"].append(q)
    keep = {k for p in w["paths"] for k in p["keys"]}
    w["members"] = {k: v for k, v in world["members"].items() if k in keep}
    return w


# ------------------------------------------------------------------ arrays


def arrays(world) -> dict:
    """Complete S/L/M paths -> Y [P, J, 4], PHI [P, J, F], SE, types, runs (path order kept)."""
    ps = [p for p in world["paths"] if p["type"] in "SLM"]
    Y = np.stack([[world["members"][k]["y"] for k in p["keys"]] for p in ps]).astype(float)
    PHI = np.stack([[world["members"][k]["phi"] for k in p["keys"]] for p in ps]).astype(float)
    runs = np.array([p["runs"][0] for p in ps])
    return {
        "Y": Y,
        "PHI": PHI,
        "SE": W.se_of(Y, world["N"]),
        "types": np.array([p["type"] for p in ps]),
        "runs": runs,
        "run_ids": list(dict.fromkeys(runs.tolist())),
        "ids": [p["id"] for p in ps],
        "y0": np.asarray(world["base"]["y"], float),
        "N": world["N"],
    }


# ------------------------------------------------------------------ the nested model


def fit(A, train_runs) -> dict:
    """Pooled same-type curves, type-centred feature means and the residual ridge on the training runs."""
    tr = np.isin(A["runs"], list(train_runs))
    curve, phibar = {}, {}
    Xs, Rs = [], []
    for t in "SLM":
        ix = tr & (A["types"] == t)
        if not ix.any():
            curve[t] = np.repeat(A["y0"][None, :], A["Y"].shape[1], 0)
            phibar[t] = np.zeros(A["PHI"].shape[1:])
            continue
        curve[t] = A["Y"][ix].mean(0)
        phibar[t] = A["PHI"][ix].mean(0)
        Xs.append((A["PHI"][ix] - phibar[t]).reshape(-1, A["PHI"].shape[2]))
        Rs.append((A["Y"][ix] - curve[t]).reshape(-1, 4))
    X = np.concatenate(Xs) if Xs else np.zeros((0, A["PHI"].shape[2]))
    R = np.concatenate(Rs) if Rs else np.zeros((0, 4))
    s = np.sqrt((X**2).mean(0)) if len(X) else np.ones(X.shape[1])
    scale = np.where(s > 1e-12, s, 1.0)
    Xn = X / scale
    Wr = np.linalg.solve(Xn.T @ Xn + RIDGE_ALPHA * np.eye(Xn.shape[1]), Xn.T @ R)
    return {"curve": curve, "phibar": phibar, "scale": scale, "W": Wr}


def residual(fitted, A, i) -> np.ndarray:
    t = A["types"][i]
    return ((A["PHI"][i] - fitted["phibar"][t]) / fitted["scale"]) @ fitted["W"]


def predict(fitted, A, i, s: float, k: int) -> np.ndarray:
    """[J, 4]: c_t + s g, offset at k >= 1 by the mean residual of the combined curve on the first k points."""
    c = fitted["curve"][A["types"][i]] + s * residual(fitted, A, i)
    if k > 0:
        c = c + (A["Y"][i, :k] - c[:k]).mean(0)
    return c


def score(pred, A, i, k: int) -> float:
    return float(np.mean(np.abs(pred[k:] - A["Y"][i, k:]) / A["SE"][i, k:]))


def _scores_grid(fitted, A, idx, ks, grid):
    """{k: [len(grid), len(idx)]} path scores for every s."""
    out = {k: np.zeros((len(grid), len(idx))) for k in ks}
    for n, i in enumerate(idx):
        g = residual(fitted, A, i)
        c0 = fitted["curve"][A["types"][i]]
        for a, s in enumerate(grid):
            c = c0 + s * g
            for k in ks:
                cc = c + (A["Y"][i, :k] - c[:k]).mean(0) if k > 0 else c
                out[k][a, n] = float(np.mean(np.abs(cc[k:] - A["Y"][i, k:]) / A["SE"][i, k:]))
    return out


def choose_shrink(losses, grid=SHRINK_GRID) -> float:
    losses = np.asarray(losses, float)
    best = losses.min()
    return float(min(s for s, v in zip(grid, losses) if v <= best * (1 + SHRINK_TIE) + 1e-15))


def inner_shrink(A, train_runs, ks=KS, grid=SHRINK_GRID) -> dict:
    """s per setting from leave-one-run-out folds inside ``train_runs`` (mean path score)."""
    train_runs = list(train_runs)
    if len(train_runs) < 2 or len(grid) == 1:
        return {k: float(grid[0]) if len(grid) == 1 else 0.0 for k in ks}
    tot = {k: np.zeros(len(grid)) for k in ks}
    n = 0
    for q in train_runs:
        f = fit(A, [r for r in train_runs if r != q])
        idx = np.flatnonzero(A["runs"] == q)
        sc = _scores_grid(f, A, idx, ks, grid)
        for k in ks:
            tot[k] += sc[k].sum(1)
        n += len(idx)
    return {k: choose_shrink(tot[k] / max(n, 1), grid) for k in ks}


def cluster_test(d, runs, alpha: float = ALPHA, n_flip=N_FLIP, seed: int = 0) -> dict:
    """One-sided sign-flip over runs (whole runs flip); T = mean of d over paths (run weight = its path count);
    lower bound by inverting the same test (``wm.signflip.test``)."""
    from wsp.wm import signflip as SF

    d = np.asarray(d, float)
    runs = np.asarray(runs)
    us = list(dict.fromkeys(runs.tolist()))
    sums = np.array([d[runs == u].sum() for u in us])
    cnt = np.array([(runs == u).sum() for u in us], float)
    D = sums / cnt
    nf = None if len(us) <= EXACT_MAX_RUNS else n_flip
    r = SF.test(D, weights=cnt, n_flip=nf, seed=seed, alpha=alpha)
    r["run_means"] = D.tolist()
    r["runs"] = us
    return r


def holm(p: dict, alpha: float = ALPHA) -> dict:
    """{setting: p} -> per setting the Holm level used and the decision (family alpha, one-sided)."""
    order = sorted(p, key=lambda k: p[k])
    m = len(order)
    out = {}
    go = True
    for i, k in enumerate(order):
        a = alpha / (m - i) if go else alpha / m
        rej = go and p[k] <= a
        out[k] = {"alpha": a if go else alpha / m, "reject": bool(rej)}
        go = go and rej
    return out


def analyse(
    A, ks=KS, shrink_grid=SHRINK_GRID, inner: bool = True, n_flip=N_FLIP, seed: int = 0
) -> dict:
    """Leave-one-run-out nested WM vs the pooled curve; per path scores, run-cluster tests, Holm, the pass rule."""
    runs = A["run_ids"]
    P = len(A["ids"])
    E = {k: {"nested": np.zeros(P), "pooled": np.zeros(P)} for k in ks}
    shr = {}
    for r in runs:
        tr = [q for q in runs if q != r]
        s = (
            inner_shrink(A, tr, ks, shrink_grid)
            if inner
            else {k: float(shrink_grid[-1]) for k in ks}
        )
        shr[r] = s
        f = fit(A, tr)
        for i in np.flatnonzero(A["runs"] == r):
            for k in ks:
                E[k]["nested"][i] = score(predict(f, A, i, s[k], k), A, i, k)
                E[k]["pooled"][i] = score(predict(f, A, i, 0.0, k), A, i, k)
    out = {"paths": list(A["ids"]), "runs": list(A["runs"]), "shrink": shr, "settings": {}}
    pv = {}
    for k in ks:
        d = E[k]["pooled"] - E[k]["nested"]
        t = cluster_test(d, A["runs"], alpha=ALPHA, n_flip=n_flip, seed=seed)
        pv[k] = t["p"]
        out["settings"][k] = {
            "E": {a: v.tolist() for a, v in E[k].items()},
            "mean_E": {a: float(v.mean()) for a, v in E[k].items()},
            "d_mean": float(d.mean()),
            "C": float(1 - E[k]["nested"].mean() / E[k]["pooled"].mean()),
            "p": t["p"],
            "run_means": t["run_means"],
            "shrink_mean": float(np.mean([shr[r][k] for r in runs])),
            "by_type": {
                ty: float(d[A["types"] == ty].mean()) for ty in "SLM" if (A["types"] == ty).any()
            },
        }
    dec = holm(pv)
    any_pass = False
    for k in ks:
        st = out["settings"][k]
        lb = cluster_test(
            np.asarray(st["E"]["pooled"]) - np.asarray(st["E"]["nested"]),
            A["runs"],
            alpha=dec[k]["alpha"],
            n_flip=n_flip,
            seed=seed,
        )["lb"]
        st.update(holm_alpha=dec[k]["alpha"], reject=dec[k]["reject"], lb_holm=lb)
        st["pass"] = bool(dec[k]["reject"] and lb > 0)
        any_pass = any_pass or st["pass"]
    out["n_paths"] = P
    out["n_runs"] = len(runs)
    out["enough_paths"] = P >= MIN_PATHS
    out["pass_primary"] = bool(any_pass)
    return out


def decision(A, **kw) -> dict:
    """Only what a null replicate needs: per setting reject/pass, and the whole rule."""
    r = analyse(A, **kw)
    return {
        "reject": {k: r["settings"][k]["reject"] for k in r["settings"]},
        "pass": r["pass_primary"],
        "d_mean": {k: r["settings"][k]["d_mean"] for k in r["settings"]},
    }


# ------------------------------------------------------------------ nulls


def null_world(world, kind: str, seed: int = 0) -> dict:
    """N1 / N2 exactly as stage W (``wm.stagew``)."""
    if kind == "n1":
        return W.null_shuffle_features(world, seed=seed)
    if kind == "n2":
        return W.null_permute_paths(world, seed=seed)
    raise KeyError(kind)


def _owners(world):
    """Each member key -> (path type, point index, path id) of the first S/L/M path that contains it."""
    own = {}
    for p in world["paths"]:
        if p["type"] not in "SLM":
            continue
        for j, k in enumerate(p["keys"]):
            own.setdefault(k, (p["type"], j, p["id"]))
    return own


def _logit(q):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def _offset_mle(z, n, k):
    """1-D MLE of a common logit offset for counts k ~ Bin(n, sig(z + d)); returns (d, information)."""
    n, k, z = (np.asarray(v, float).ravel() for v in (n, k, z))
    if n.sum() <= 0:
        return 0.0, 0.0

    def f(d):
        zd = z + d[0]
        nll = -(k * W._logsig(zd) + (n - k) * W._logsig(-zd)).sum()
        return nll + 0.5e-6 * d[0] ** 2, np.array([(n * W._sig(zd) - k).sum() + 1e-6 * d[0]])

    d = float(optimize.minimize(f, np.zeros(1), jac=True, method="L-BFGS-B").x[0])
    pz = W._sig(z + d)
    return d, float((n * pz * (1 - pz)).sum())


def calibrate_n3(world) -> dict:
    """Per type, point and benchmark the pooled per-cell finish and correct|finish logits over all paths of the type;
    tau^2 = var(path offsets) - mean(sampling variance), per benchmark, method of moments."""
    own = _owners(world)
    ps = [p for p in world["paths"] if p["type"] in "SLM"]
    J = len(ps[0]["keys"])
    q = {}
    for t in "SLM":
        q[t] = []
        tp = [p for p in ps if p["type"] == t]
        for j in range(J):
            d = {}
            for pn in PANELS:
                cnt = np.sum([world["members"][p["keys"][j]]["counts"][pn] for p in tp], 0)
                n, nf, nc = cnt[:, 0], cnt[:, 1], cnt[:, 2]
                qf = (nf + 0.5) / (n + 1)
                qc = (nc + 0.5) / (nf + 1)
                d[pn] = (_logit(qf), _logit(qc))
            q[t].append(d)
    offs = {pn: ([], []) for pn in PANELS}
    sv = {pn: ([], []) for pn in PANELS}
    for p in ps:
        for pn in PANELS:
            zf = np.stack([q[p["type"]][j][pn][0] for j in range(J)])
            zc = np.stack([q[p["type"]][j][pn][1] for j in range(J)])
            cnt = np.stack([world["members"][k]["counts"][pn] for k in p["keys"]])
            for i, (z, n, k) in enumerate(
                ((zf, cnt[..., 0], cnt[..., 1]), (zc, cnt[..., 1], cnt[..., 2]))
            ):
                d, info = _offset_mle(z, n, k)
                offs[pn][i].append(d)
                sv[pn][i].append(1.0 / max(info, 1e-9))
    tau = {
        pn: tuple(
            float(math.sqrt(max(0.0, np.var(offs[pn][i], ddof=1) - np.mean(sv[pn][i]))))
            for i in range(2)
        )
        for pn in PANELS
    }
    return {"q": q, "tau": tau, "owners": own, "J": J}


def n3_world(world, cal, rng) -> dict:
    """No-edit-information world: every member's counts from its owner path's type curve at its point + the owner
    path's random effect + binomial noise; features and the base unchanged."""
    w = dict(world)
    w["members"] = {k: dict(v) for k, v in world["members"].items()}
    eff = {}
    for p in world["paths"]:
        if p["type"] in "SLM":
            eff[p["id"]] = {
                pn: (rng.normal(0, cal["tau"][pn][0]), rng.normal(0, cal["tau"][pn][1]))
                for pn in PANELS
            }
    cells = world["cells"]
    for k, (t, j, pid) in cal["owners"].items():
        counts = {}
        for pn in PANELS:
            zf, zc = cal["q"][t][j][pn]
            counts[pn] = W.generate_counts(
                zf + eff[pid][pn][0], zc + eff[pid][pn][1], cells[pn]["n"], rng
            )
        w["members"][k]["counts"] = counts
        w["members"][k]["y"] = W.y_of(counts)
    return w


def null_decisions(world, kind: str, idx, seed: int = 0, cal=None, n_flip=N_FLIP) -> list:
    """Pass-rule decisions of the null replicates ``idx`` (seeded by (seed, kind, index); order-free, so replicates
    can be split across processes)."""
    rows = []
    for s in idx:
        if kind == "n3":
            w = n3_world(world, cal, np.random.default_rng([seed, 3, s]))
        else:
            w = null_world(world, kind, seed=seed * 100_000 + s)
        rows.append(dict(decision(arrays(w), n_flip=n_flip, seed=s), rep=int(s)))
    return rows


def summarize_null(rows) -> dict:
    n_pass = int(sum(r["pass"] for r in rows))
    reps = len(rows)
    return {
        "reps": reps,
        "size": n_pass / reps,
        "n_pass": n_pass,
        "size_cp_upper95": cp_upper(n_pass, reps),
        "reject_rate": {
            k: float(np.mean([r["reject"][k] for r in rows])) for k in rows[0]["reject"]
        },
        "mean_d": {k: float(np.mean([r["d_mean"][k] for r in rows])) for k in rows[0]["d_mean"]},
    }


def null_sizes(
    world, kinds=("n1", "n2", "n3"), reps: int = NULL_REPS, seed: int = 0, n_flip=N_FLIP
) -> dict:
    """Share of null replicates in which the whole pass rule holds (and per-setting rejection rates)."""
    out = {}
    cal = calibrate_n3(world) if "n3" in kinds else None
    for kind in kinds:
        out[kind] = summarize_null(null_decisions(world, kind, range(reps), seed, cal, n_flip))
        if kind == "n3":
            out[kind]["tau"] = cal["tau"]
    return out


def comparator_contrasts(nested: dict, comp: dict, A) -> dict:
    """Secondary comparators (descriptive): ``nested`` = :func:`analyse` output, ``comp`` = ``wm.stagew.analyse``
    output on the same world; per setting and comparator mean E, C, mean d (comparator - nested) overall and per
    type, and the run-cluster p and lower bound at alpha 0.025 (unadjusted)."""
    out = {}
    pos = {p: i for i, p in enumerate(nested["paths"])}
    for k, st in nested["settings"].items():
        cs = comp["settings"][k]
        en = np.asarray(st["E"]["nested"])
        res = {}
        for arm, v in cs["E"].items():
            arm_name = "wm_unnested" if arm == "wm" else arm
            e = np.zeros(len(en))
            for pid, val in zip(comp["paths"], v):
                e[pos[pid]] = val
            d = e - en
            t = cluster_test(d, A["runs"], alpha=ALPHA)
            res[arm_name] = {
                "mean_E": float(e.mean()),
                "C": float(1 - en.mean() / e.mean()) if e.mean() > 0 else float("nan"),
                "d_mean": float(d.mean()),
                "by_type": {
                    ty: float(d[A["types"] == ty].mean())
                    for ty in "SLM"
                    if (A["types"] == ty).any()
                },
                "p": t["p"],
                "lb": t["lb"],
            }
        out[k] = res
    return out


R_GRID = (15, 18, 21, 24, 27, 30)
HT3_ALPHA = 0.0125
HT3_SHIFT = 0.5
HT3_REPS = 2000
POWER_MIN = 0.8


def ht3_power(
    d, runs, R: int, reps: int = HT3_REPS, seed: int = 0, shift: float = HT3_SHIFT
) -> float:
    """Run-cluster bootstrap power: W2's per-path improvements shifted to ``shift`` x the observed mean, R runs drawn
    with replacement (a run's paths as a unit), the run-cluster sign-flip at alpha 0.0125."""
    d = np.asarray(d, float)
    runs = np.asarray(runs)
    d = d - (1 - shift) * d.mean()
    us = list(dict.fromkeys(runs.tolist()))
    groups = [d[runs == u] for u in us]
    rng = np.random.default_rng([seed, R])
    hit = 0
    for b in range(reps):
        pick = rng.integers(len(groups), size=R)
        dd = np.concatenate([groups[i] for i in pick])
        rr = np.concatenate([np.full(len(groups[i]), n) for n, i in enumerate(pick)])
        hit += cluster_test(dd, rr, alpha=HT3_ALPHA, seed=b)["p"] <= HT3_ALPHA
    return hit / reps


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    from scipy import stats

    return 1.0 if k >= n else float(stats.beta.ppf(conf, k + 1, n - k))


# ------------------------------------------------------------------ fresh panel wme-w2

# every exposure a W2 panel must exclude (RULING-wme-post-stagew Q4: incl. wme-p3* and stage W)
REQUIRED_EXPOSURES = (
    "wme-v0-p1",
    "wme-p2",
    "calib27-20261116",
    "wme-p2-smoke",
    "wme-p3s",
    "wme-p3h",
    "wme-w",
)


def panel_select(pools, exposed, registered, n: int = N_DEV) -> dict:
    """As ``wm.stagew.panel_select`` with the W2 salt; refuses unless every required exposure is registered."""
    from wsp.wm import p3panel as PP

    missing = [x for x in REQUIRED_EXPOSURES if x not in set(registered)]
    if missing:
        raise RuntimeError(f"exposures not registered: {missing}")
    fams = {"mc": {"pools": pools}}
    sizes = {p: {"dev": n} for p in PANELS}
    res = PP.select(fams, PANELS, sizes, SALT, set(exposed), {p: ("dev",) for p in PANELS})
    res["audit"] = PP.audit(res["splits"], set(exposed))
    if not res["audit"]["clean"]:
        raise RuntimeError(f"panel audit failed: {res['audit']}")
    return res


# ------------------------------------------------------------------ loading the W2 slots


def load_runs(dirs: dict) -> dict:
    """{slot: run dir} (train_meta.json, features.npz, evals.jsonl) -> one analysis world. The base (cells, y0) is
    the first slot's (a, b, c order) base; only paths whose 6 points were all evaluated are kept."""
    import json
    from pathlib import Path

    sq, j1, final_steps, recs, bases, slot_of = {}, {}, {}, {}, {}, {}
    n_layers = None
    for slot in sorted(dirs):
        d = Path(dirs[slot])
        tm = json.loads((d / "train_meta.json").read_text())
        if n_layers is None:
            n_layers = int(tm["n_layers"])
            W.configure_layers(n_layers)
        elif int(tm["n_layers"]) != n_layers:
            raise ValueError("slots disagree on the layer count")
        z = np.load(d / "features.npz")
        for i, s in enumerate(z["ckpts"]):
            c = W._ck(str(s))
            sq[c] = z["sq"][i]
            j1[c] = z["j1"][:, :, i, :]
        for r, s in tm["final_steps"].items():
            final_steps[r] = int(s)
            slot_of[r] = slot
        for line in (d / "evals.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["tag"] == "member":
                recs[r["key"]] = r
            else:
                bases[(slot, r["tag"])] = r
    base_slot = next(s for s in sorted(dirs) if (s, "base") in bases)
    base = bases[(base_slot, "base")]
    cells = {p: W.item_cells(base["panels"][p]) for p in PANELS}
    N = len(base["panels"][PANELS[0]])
    F = len(W.FEATURES)

    def mk(rec, phi):
        counts = {p: W.cell_counts(rec["panels"][p], cells[p]) for p in PANELS}
        return {"phi": phi, "counts": counts, "y": W.y_of(counts)}

    members = {k: mk(r, W.member_features(r["spec"], sq, {}, j1)) for k, r in recs.items()}
    paths = make_paths(final_steps)
    complete = [p for p in paths if all(k in members for k in p["keys"])]
    world = {
        "cells": cells,
        "members": members,
        "base": mk(base, np.zeros(F)),
        "paths": complete,
        "N": N,
        "incomplete": [p["id"] for p in paths if p not in complete],
        "n_layers": n_layers,
        "base_slot": base_slot,
        "slot_of_run": slot_of,
        "slot_bases": {
            s: mk(r, np.zeros(F))["y"].tolist() for (s, t), r in bases.items() if t == "base"
        },
    }
    if (base_slot, "base_rerun") in bases:
        rr = bases[(base_slot, "base_rerun")]
        world["base_rerun"] = mk(rr, np.zeros(F))
        world["base_rerun_discordance"] = {
            p: float(
                np.mean(
                    [
                        a["correct"] != b["correct"]
                        for a, b in zip(base["panels"][p], rr["panels"][p])
                    ]
                )
            )
            for p in PANELS
        }
    return world
