#!/usr/bin/env python3
"""Exploratory post-hoc model search on the exposed 9B stage-W2 data (SYNTHESIS-r13 addendum item 3). POST HOC.

CPU only. The pre-declaration (candidates, grids, tests, the selection rule) is section 0 of
docs/reviews/wme/wme-stagew2-modelsearch-2026-11-20.md, committed before this script was run on the W2 data.

* Metric: the W2 path score E (``wsp.wm.stagew2.score``), mean |pred - y| / SE(y) over the points after k and the 4
  endpoints; k in {0, 2}; outer leave-one-run-out (LORO); at k > 0 the k-point offset of W2 (weight omega).
* Candidates: C1 pooled, C2 pooled + dose (per (type, point, endpoint) OLS on log ||dW||, the mandatory baseline),
  C3a ridge / C3b GP on per-layer J1 (+ dose), C4 per-endpoint heads, C5a tiny GRU, C5b selective diagonal SSM.
  Every tuned candidate's first grid entry is C2 itself; hyperparameters by nested LORO inside the training runs.
* Tests: per candidate and k, d = E_C2 - E_c, one-sided run-cluster sign-flip (``W2.cluster_test``), Holm over the
  10 (candidate, k) hypotheses at 0.025; nulls N1-N3 and fresh-data power for the frozen contrast.

    wme_w2_model_search.py --w2 data/wme/stagew2/run-20261118 --json docs/reviews/wme/wme-stagew2-modelsearch-2026-11-20.json

Reads only exposed W2 artefacts; writes only the JSON given by --json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from wsp.wm import stagew as W
from wsp.wm import stagew2 as W2

KS = (0, 2)
ALPHA = 0.025
TIE = 0.005
NULL_MAX = 0.035
TESTED = ("ridge", "gp", "per_endpoint", "gru", "ssm")
ALL = ("pooled", "dose") + TESTED
LABEL = {
    "pooled": "C1 pooled",
    "dose": "C2 pooled + dose",
    "ridge": "C3a C2 + ridge(J1G)",
    "gp": "C3b C2 + GP(J1G, D)",
    "per_endpoint": "C4 per-endpoint heads",
    "gru": "C5a C2 + GRU",
    "ssm": "C5b C2 + diag. SSM",
}
INF = float("inf")
RIDGE_A = (INF, 1000.0, 100.0, 10.0, 1.0)
GP_ELL = (2.0, 1.0, 0.5)
GP_RHO = (0.1, 0.3, 1.0, 3.0)
PE_BASES = ("dose", "pooled", "logit_dose", "ridge1000", "ridge100", "ridge10")
SEQ_H = (4, 8)
SEQ_L2 = (0.1, 0.01, 0.001)
SEQ_SEEDS = 3
SEQ_STEPS = 300
SEQ_LR = 0.02
N_IN = 13
R_GRID = (15, 20, 30, 45, 60)
POWER_ALPHA = 0.0125


def _ridge_grid(k):
    return list(RIDGE_A)


def _gp_grid(k):
    return [(None, 0.0)] + [(ell, rho) for ell in GP_ELL for rho in GP_RHO]


def _pe_grid(k):
    om = (1.0,) if k == 0 else (1.0, 0.5, 0.0)
    return [(b, w) for b in PE_BASES for w in om]


def _seq_grid(k):
    return ["off"] + [(h, l2) for h in SEQ_H for l2 in SEQ_L2]


GRIDS = {
    "ridge": _ridge_grid,
    "gp": _gp_grid,
    "per_endpoint": _pe_grid,
    "gru": _seq_grid,
    "ssm": _seq_grid,
}


# ------------------------------------------------------------------ features and loading


def dose_of(phi) -> np.ndarray:
    """log ||dW|| from the 4 R0 group norms (W.member_features columns 0-3)."""
    phi = np.asarray(phi, float)
    return np.log(np.sqrt((phi[..., :4] ** 2).sum(-1)) + 1e-12)


def j1_groups(spec, j1m) -> np.ndarray:
    """Per-layer J1 of a member, summed within the 8 fine layer groups: [3 objectives x 8 groups] flattened; j1m[(run,
    step)] = [3, n_layers] item-mean projections at lambda 1. Groups beyond the mask are zero."""
    v = np.asarray(j1m[(spec["run"], spec["step"])], float) * spec["lam"]
    L = v.shape[1]
    g = v.reshape(3, 8, L // 8).sum(-1)
    g[:, spec["groups"] :] = 0.0
    return g.ravel()


def load(run_dir) -> tuple:
    run_dir = Path(run_dir)
    dirs = {s: run_dir / s / "w2" for s in "abc" if (run_dir / s / "w2" / "evals.jsonl").exists()}
    world = W2.load_runs(dirs)
    j1m = {}
    for d in dirs.values():
        z = np.load(d / "features.npz")
        for i, s in enumerate(z["ckpts"]):
            j1m[W._ck(str(s))] = z["j1"][:, :, i, :].mean(1)
    A = W2.arrays(world)
    attach(A, world, j1m)
    return A, world, j1m


def attach(A, world, j1m) -> dict:
    ps = [p for p in world["paths"] if p["type"] in "SLM"]
    A["D"] = dose_of(A["PHI"])
    A["R1S"] = A["PHI"][..., 4:7].copy()
    A["J1G"] = np.stack([[j1_groups(W.unkey(k), j1m) for k in p["keys"]] for p in ps])
    return A


# ------------------------------------------------------------------ metric and baselines


def offset(pred, Y, k, omega) -> np.ndarray:
    if k == 0 or omega == 0:
        return pred.copy()
    return pred + omega * (Y[:, :k] - pred[:, :k]).mean(1, keepdims=True)


def endpoint_scores(pred, A, k) -> np.ndarray:
    """[P, 4]: per path and endpoint the mean |error|/SE over the points after k (E = mean over endpoints)."""
    return (np.abs(pred[:, k:] - A["Y"][:, k:]) / A["SE"][:, k:]).mean(1)


def _logit(y, n):
    q = np.clip(y, 0.5 / n, 1 - 0.5 / n)
    return np.log(q / (1 - q))


def base_predict(A, trm, kind) -> np.ndarray:
    """[P, J, 4] predictions for every path from the training paths ``trm`` (bool mask)."""
    Y, types = A["Y"], A["types"]
    out = np.zeros_like(Y)
    for t in "SLM":
        it = types == t
        tt = trm & it
        if kind == "pooled":
            out[it] = Y[tt].mean(0)
            continue
        for j in range(Y.shape[1]):
            X = np.c_[np.ones(len(Y)), A["D"][:, j]]
            tgt = Y[tt, j] if kind == "dose" else _logit(Y[tt, j], A["N"])
            b = np.linalg.lstsq(X[tt], tgt, rcond=None)[0]
            v = X[it] @ b
            out[it, j] = v if kind == "dose" else 1 / (1 + np.exp(-v))
    return out


def crossfit(A, train_runs, kind) -> np.ndarray:
    """Base predictions for the training paths, each from the other training runs (LORO); NaN elsewhere."""
    out = np.full_like(A["Y"], np.nan)
    for q in train_runs:
        trm = np.isin(A["runs"], [r for r in train_runs if r != q])
        ix = A["runs"] == q
        out[ix] = base_predict(A, trm, kind)[ix]
    return out


def std_within(X, types, trm) -> np.ndarray:
    """Standardise [P, J, F] within (type, point) with the training paths' statistics; constant columns -> 0."""
    Z = np.zeros_like(X, dtype=float)
    for t in "SLM":
        it = types == t
        mu = X[trm & it].mean(0)
        sd = X[trm & it].std(0)
        ok = sd > 1e-12
        Z[it] = np.where(ok, (X[it] - mu) / np.where(ok, sd, 1.0), 0.0)
    return Z


# ------------------------------------------------------------------ residual learners


def _ridge_all(Xtr, R, Xte, alphas) -> dict:
    """{alpha: Xte @ w_alpha}, w_alpha = (X'X + alpha I)^-1 X'R, by one eigendecomposition."""
    s, V = np.linalg.eigh(Xtr.T @ Xtr)
    XtR = V.T @ (Xtr.T @ R)
    out = {}
    for a in alphas:
        out[a] = (
            np.zeros((len(Xte), R.shape[1])) if a == INF else Xte @ (V @ (XtR / (s + a)[:, None]))
        )
    return out


def _gp_all(Xtr, R, Xte, grid) -> dict:
    F = Xtr.shape[1]
    out = {}

    def sq(a, b):
        return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) / F

    dtt, dst = sq(Xtr, Xtr), sq(Xte, Xtr)
    for ell in {e for e, _ in grid if e is not None}:
        K, Ks = np.exp(-dtt / (2 * ell**2)), np.exp(-dst / (2 * ell**2))
        s, V = np.linalg.eigh(K)
        VtR = V.T @ R
        for e, rho in grid:
            if e == ell:
                out[(e, rho)] = Ks @ (V @ (VtR / (np.maximum(s, 0) + 1.0 / rho)[:, None]))
    for e, rho in grid:
        if rho == 0.0:
            out[(e, rho)] = np.zeros((len(Xte), R.shape[1]))
    return out


def residual_grid(A, train_runs, learner, grid) -> dict:
    """{hp: [P, J, 4]} residual corrections on top of C2 from a per-(type, endpoint) learner fitted to the training
    paths' cross-fitted C2 residuals, shared across points."""
    trm = np.isin(A["runs"], list(train_runs))
    cf = crossfit(A, train_runs, "dose")
    if learner == "ridge":
        Z = std_within(A["J1G"], A["types"], trm)
    else:
        Z = std_within(np.concatenate([A["J1G"], A["D"][..., None]], -1), A["types"], trm)
    _, J_, F = Z.shape
    out = {h: np.zeros_like(A["Y"]) for h in grid}
    for t in "SLM":
        it = A["types"] == t
        tt = trm & it
        Xtr = Z[tt].reshape(-1, F)
        R = (A["Y"][tt] - cf[tt]).reshape(-1, 4)
        Xte = Z[it].reshape(-1, F)
        res = _ridge_all(Xtr, R, Xte, grid) if learner == "ridge" else _gp_all(Xtr, R, Xte, grid)
        for h in grid:
            out[h][it] = res[h].reshape(it.sum(), J_, 4)
    return out


# ------------------------------------------------------------------ sequence models (torch, CPU)


def _torch():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch

    torch.set_num_threads(1)
    return torch


def make_net(arch, h, seed=0):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(N_IN, h, batch_first=True)
            self.out = nn.Linear(h, 4)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, u):
            return self.out(self.rnn(u)[0])

    class SSMNet(nn.Module):
        """Selective diagonal SSM with the Mamba output gate: h_t = a_t * h_{t-1} + dt_t * (B u_t),
        a_t = exp(-dt_t exp(A)), dt_t = softplus(W u_t + b); y_t = C (h_t * silu(Z u_t + z)) + D u_t + c."""

        def __init__(self):
            super().__init__()
            self.B = nn.Linear(N_IN, h, bias=False)
            self.dt = nn.Linear(N_IN, h)
            self.A = nn.Parameter(torch.zeros(h))
            self.Z = nn.Linear(N_IN, h)
            self.C = nn.Linear(h, 4, bias=False)
            self.Du = nn.Linear(N_IN, 4)
            for m in (self.C, self.Du):
                nn.init.zeros_(m.weight)
            nn.init.zeros_(self.Du.bias)

        def forward(self, u):
            dt = nn.functional.softplus(self.dt(u))
            a = torch.exp(-dt * torch.exp(self.A))
            bu = dt * self.B(u)
            hs, hh = [], torch.zeros(u.shape[0], self.A.shape[0])
            for t in range(u.shape[1]):
                hh = a[:, t] * hh + bu[:, t]
                hs.append(hh)
            gate = nn.functional.silu(self.Z(u))
            return self.C(torch.stack(hs, 1) * gate) + self.Du(u)

    return (GRUNet if arch == "gru" else SSMNet)()


def n_params(arch, h) -> int:
    return sum(p.numel() for p in make_net(arch, h).parameters())


def seq_inputs(A, base_raw, k, trm) -> np.ndarray:
    """[P, J, 13]: type one-hot, x/8, D and R1 summaries (standardised within (type, point)), the raw C2 residual in
    SE units at the given points (0 after k), the given-point flag."""
    P_, J_ = A["Y"].shape[:2]
    oh = np.stack([(A["types"] == t).astype(float) for t in "SLM"], -1)[:, None, :].repeat(J_, 1)
    x = np.broadcast_to((np.asarray(W2.GRID, float) / 8)[None, :, None], (P_, J_, 1))
    Z = std_within(np.concatenate([A["D"][..., None], A["R1S"]], -1), A["types"], trm)
    obs = np.zeros((P_, J_, 4))
    flag = np.zeros((P_, J_, 1))
    if k:
        obs[:, :k] = (A["Y"][:, :k] - base_raw[:, :k]) / A["SE"][:, :k]
        flag[:, :k] = 1.0
    return np.concatenate([oh, x, Z, obs, flag], -1)


def seq_grid(A, train_runs, k, arch, seeds=SEQ_SEEDS, steps=SEQ_STEPS, grid=None) -> dict:
    """{hp: [P, J, 4]} predictions: C2 (offset omega 1) + SE x the network's correction; 'off' = C2."""
    torch = _torch()
    grid = grid or _seq_grid(k)
    trm = np.isin(A["runs"], list(train_runs))
    base = base_predict(A, trm, "dose")
    cf = crossfit(A, train_runs, "dose")
    base[trm] = cf[trm]  # training paths see their cross-fitted baseline
    base_off = offset(base, A["Y"], k, 1.0)
    U = torch.tensor(seq_inputs(A, base, k, trm), dtype=torch.float32)
    T = torch.tensor((A["Y"] - base_off) / A["SE"], dtype=torch.float32)
    mask = torch.zeros(T.shape[:2])
    mask[:, k:] = 1.0
    tr = torch.tensor(trm)
    out = {}
    for hp in grid:
        if hp == "off":
            out[hp] = base_off
            continue
        h, l2 = hp
        corr = np.zeros_like(A["Y"])
        for s in range(seeds):
            net = make_net(arch, h, seed=1000 * s + h)
            opt = torch.optim.Adam(net.parameters(), lr=SEQ_LR)
            for _ in range(steps):
                opt.zero_grad()
                e = torch.nn.functional.huber_loss(net(U[tr]), T[tr], reduction="none", delta=1.0)
                loss = (e.mean(-1) * mask[tr]).sum() / mask[tr].sum()
                loss = loss + l2 * sum((p**2).sum() for p in net.parameters())
                loss.backward()
                opt.step()
            with torch.no_grad():
                corr += net(U).numpy() / seeds
        out[hp] = base_off + corr * A["SE"]
    return out


# ------------------------------------------------------------------ candidate predictions


def grid_preds(A, train_runs, k, cand, **kw) -> dict:
    """{hp: [P, J, 4]} predictions (valid for paths outside ``train_runs``) for every grid entry of ``cand``."""
    trm = np.isin(A["runs"], list(train_runs))
    if cand in ("pooled", "dose"):
        return {None: offset(base_predict(A, trm, cand), A["Y"], k, 1.0)}
    if cand in ("gru", "ssm"):
        return seq_grid(A, train_runs, k, cand, **kw)
    base = base_predict(A, trm, "dose")
    if cand == "ridge":
        g = residual_grid(A, train_runs, "ridge", RIDGE_A)
        return {a: offset(base + g[a], A["Y"], k, 1.0) for a in RIDGE_A}
    if cand == "gp":
        grid = _gp_grid(k)
        g = residual_grid(A, train_runs, "gp", grid)
        return {h: offset(base + g[h], A["Y"], k, 1.0) for h in grid}
    if cand == "per_endpoint":
        g = residual_grid(A, train_runs, "ridge", (1000.0, 100.0, 10.0))
        bases = {
            "dose": base,
            "pooled": base_predict(A, trm, "pooled"),
            "logit_dose": base_predict(A, trm, "logit_dose"),
            "ridge1000": base + g[1000.0],
            "ridge100": base + g[100.0],
            "ridge10": base + g[10.0],
        }
        return {(b, w): offset(bases[b], A["Y"], k, w) for b, w in _pe_grid(k)}
    raise KeyError(cand)


def choose(losses, grid):
    """The first grid entry within TIE (relative) of the minimum."""
    losses = np.asarray(losses, float)
    best = losses.min()
    return next(h for h, v in zip(grid, losses) if v <= best * (1 + TIE) + 1e-15)


def _fold(r, A, cand, ks, kw):
    """One outer fold: (run, {k: (chosen hp, [n_test, 4] endpoint scores, [n_test, J, 4] predictions)})."""
    runs = A["run_ids"]
    tr = [q for q in runs if q != r]
    te = A["runs"] == r
    res = {}
    for k in ks:
        if cand in ("pooled", "dose"):
            pred = grid_preds(A, tr, k, cand)[None]
            res[k] = (None, endpoint_scores(pred, A, k)[te], pred[te])
            continue
        grid = GRIDS[cand](k)
        tot = np.zeros((len(grid), 4))
        n = 0
        for q in tr:
            g = grid_preds(A, [x for x in tr if x != q], k, cand, **kw)
            iq = A["runs"] == q
            for a, h in enumerate(grid):
                tot[a] += endpoint_scores(g[h], A, k)[iq].sum(0)
            n += iq.sum()
        loss = tot / n
        if cand == "per_endpoint":
            hp = tuple(choose(loss[:, e], grid) for e in range(4))
            g = grid_preds(A, tr, k, cand, **kw)
            pred = np.stack([g[hp[e]][..., e] for e in range(4)], -1)
        else:
            hp = choose(loss.mean(1), grid)
            pred = grid_preds(A, tr, k, cand, **kw)[hp]
        res[k] = (hp, endpoint_scores(pred, A, k)[te], pred[te])
    return r, res


def run_candidate(A, cand, ks=KS, workers=1, **kw) -> dict:
    """Outer LORO: per k the per-path E [P], per-endpoint scores [P, 4] and the chosen hyperparameters per run."""
    runs = A["run_ids"]
    f = partial(_fold, A=A, cand=cand, ks=ks, kw=kw)
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            folds = dict(pool.map(f, runs))
    else:
        folds = dict(map(f, runs))
    P_ = len(A["ids"])
    out = {"E": {}, "S": {}, "chosen": {}, "pred": {}}
    for k in ks:
        S = np.zeros((P_, 4))
        pred = np.zeros_like(A["Y"])
        for r in runs:
            te = A["runs"] == r
            S[te] = folds[r][k][1]
            pred[te] = folds[r][k][2]
        out["S"][k], out["E"][k], out["pred"][k] = S, S.mean(1), pred
        out["chosen"][k] = {r: folds[r][k][0] for r in runs}
    return out


# ------------------------------------------------------------------ tests, selection, nulls, power


def contrast(res, ref, A, alpha=ALPHA) -> dict:
    """Per k: d = E_ref - E_res (positive favours ``res``), run-cluster sign-flip p and LB at ``alpha``."""
    out = {}
    for k in res["E"]:
        d = ref["E"][k] - res["E"][k]
        t = W2.cluster_test(d, A["runs"], alpha=alpha)
        out[k] = {
            "mean_E": float(res["E"][k].mean()),
            "mean_E_ref": float(ref["E"][k].mean()),
            "d_mean": float(d.mean()),
            "p": float(t["p"]),
            "lb": float(t["lb"]),
            "d": d.tolist(),
            "by_type": {ty: float(d[A["types"] == ty].mean()) for ty in "SLM"},
        }
    return out


def holm_family(p: dict, alpha=ALPHA) -> dict:
    return W2.holm(p, alpha)


def select_frozen(rows, beats):
    """The candidate that beats C2 with the largest mean d over k (None if none beats C2)."""
    ok = [c for c in rows if beats.get(c)]
    if not ok:
        return None
    return max(ok, key=lambda c: np.mean([rows[c][k]["d_mean"] for k in rows[c]]))


def _derange(n, rng):
    while True:
        p = rng.permutation(n)
        if n < 2 or np.all(p != np.arange(n)):
            return p


NULL_FEATURES = {"n1": ("J1G", "R1S"), "n1_dose": ("D",)}


def null_arrays(A, kind, seed=0, world=None, cal=None) -> dict:
    """N1 (``n1``: non-dose features; ``n1_dose``: the dose trajectory) deranged within type; N2 outcome curves
    deranged within type; N3 the W2 no-edit-information world (features kept)."""
    B = dict(A)
    rng = np.random.default_rng([seed, {"n1": 1, "n1_dose": 11, "n2": 2, "n3": 3}[kind]])
    if kind == "n3":
        w = W2.n3_world(world, cal, rng)
        C = W2.arrays(w)
        B["Y"], B["SE"] = C["Y"], C["SE"]
        return B
    cols = NULL_FEATURES.get(kind, ("Y",))
    for c in cols:
        B[c] = A[c].copy()
    for t in "SLM":
        ix = np.flatnonzero(A["types"] == t)
        pi = ix[_derange(len(ix), rng)]
        for c in cols:
            B[c][ix] = A[c][pi]
    if kind == "n2":
        B["SE"] = W.se_of(B["Y"], A["N"])
    return B


def pass_rule(res, ref, A) -> dict:
    """The W2 pass rule for one contrast: Holm over k at 0.025, LB at the Holm level > 0 in some k."""
    c = contrast(res, ref, A)
    dec = W2.holm({k: c[k]["p"] for k in c})
    ok = {}
    for k in c:
        lb = W2.cluster_test(np.asarray(c[k]["d"]), A["runs"], alpha=dec[k]["alpha"])["lb"]
        ok[k] = bool(dec[k]["reject"] and lb > 0)
    return {
        "reject": {k: dec[k]["reject"] for k in c},
        "pass": any(ok.values()),
        "d_mean": {k: c[k]["d_mean"] for k in c},
    }


def _null_rep(s, A, kind, cand, ref, world, cal, kw):
    B = null_arrays(A, kind, seed=s, world=world, cal=cal)
    r = pass_rule(run_candidate(B, cand, **kw), run_candidate(B, ref), B)
    r["rep"] = int(s)
    return r


def null_sizes(A, world, cand, ref, kinds, reps, workers=1, **kw) -> dict:
    cal = W2.calibrate_n3(world) if "n3" in kinds else None
    out = {}
    for kind in kinds:
        f = partial(_null_rep, A=A, kind=kind, cand=cand, ref=ref, world=world, cal=cal, kw=kw)
        if workers > 1:
            import multiprocessing as mp

            with mp.get_context("fork").Pool(workers) as pool:
                rows = pool.map(f, range(reps))
        else:
            rows = list(map(f, range(reps)))
        out[kind] = W2.summarize_null(rows)
    return out


def power_table(d, runs, reps, R_grid=R_GRID) -> dict:
    return {
        f"{shift:g}x": {R: W2.ht3_power(d, runs, R, reps=reps, shift=shift) for R in R_grid}
        for shift in (1.0, 0.5)
    }


def _js(x):
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, np.ndarray):
        return _js(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--w2", default="data/wme/stagew2/run-20261118")
    ap.add_argument("--json", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cands", default=",".join(ALL))
    ap.add_argument("--null-reps", type=int, default=1000)
    ap.add_argument("--null-reps-seq", type=int, default=100)
    ap.add_argument("--power-reps", type=int, default=2000)
    ap.add_argument("--seq-steps", type=int, default=SEQ_STEPS)
    ap.add_argument("--seq-seeds", type=int, default=SEQ_SEEDS)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    A, world, _ = load(a.w2)
    log = lambda *m: print(f"[{time.time() - t0:7.1f}s]", *m, flush=True)
    log("paths", len(A["ids"]), "runs", len(A["run_ids"]))
    cands = [c for c in a.cands.split(",") if c]
    seqkw = {"steps": a.seq_steps, "seeds": a.seq_seeds}
    res, secs = {}, {}
    for c in cands:
        t1 = time.time()
        res[c] = run_candidate(A, c, workers=a.workers, **(seqkw if c in ("gru", "ssm") else {}))
        secs[c] = time.time() - t1
        log(c, {k: round(float(res[c]["E"][k].mean()), 4) for k in KS}, f"{secs[c]:.0f}s")
    rows = {c: contrast(res[c], res["dose"], A) for c in cands if c != "dose"}
    tested = [c for c in TESTED if c in rows]
    dec = holm_family({(c, k): rows[c][k]["p"] for c in tested for k in KS})
    beats = {}
    for c in tested:
        for k in KS:
            dd = dec[(c, k)]
            lb = W2.cluster_test(np.asarray(rows[c][k]["d"]), A["runs"], alpha=dd["alpha"])["lb"]
            rows[c][k].update(
                holm_alpha=dd["alpha"],
                reject=dd["reject"],
                lb_holm=float(lb),
                beats=bool(dd["reject"] and lb > 0),
            )
        beats[c] = any(rows[c][k]["beats"] for k in KS)
    frozen = select_frozen({c: rows[c] for c in tested}, beats)
    best_point = (
        max(tested, key=lambda c: np.mean([rows[c][k]["d_mean"] for k in KS])) if tested else None
    )
    # the frozen contrast: candidate vs C2, or C2 vs C1 when nothing beats C2
    fz_cand, fz_ref = (frozen, "dose") if frozen else ("dose", "pooled")
    fz = contrast(res[fz_cand], res[fz_ref], A)
    log("frozen", fz_cand, "vs", fz_ref, "best point", best_point)
    seq = fz_cand in ("gru", "ssm")
    nkinds = ("n1", "n2", "n3") if fz_cand != "dose" else ("n1_dose", "n2", "n3")
    nulls = null_sizes(
        A,
        world,
        fz_cand,
        fz_ref,
        nkinds,
        a.null_reps_seq if seq else a.null_reps,
        workers=a.workers,
        **(seqkw if seq else {}),
    )
    log("nulls", {k: v["size"] for k, v in nulls.items()})
    power = {k: power_table(np.asarray(fz[k]["d"]), A["runs"], a.power_reps) for k in KS}
    power_best = None
    if best_point and best_point != fz_cand:
        power_best = {
            k: power_table(np.asarray(rows[best_point][k]["d"]), A["runs"], a.power_reps)
            for k in KS
        }
    log("power done")
    out = {
        "label": "POST HOC / EXPLORATORY",
        "source": "SYNTHESIS-r13 addendum item 3; pre-declaration = section 0 of the markdown review",
        "data": str(a.w2),
        "n_paths": len(A["ids"]),
        "n_runs": len(A["run_ids"]),
        "settings": list(KS),
        "candidates": {
            c: {
                "label": LABEL[c],
                "mean_E": {k: float(res[c]["E"][k].mean()) for k in KS},
                "E_by_endpoint": {k: res[c]["S"][k].mean(0).tolist() for k in KS},
                "E_by_type": {
                    k: {t: float(res[c]["E"][k][A["types"] == t].mean()) for t in "SLM"} for k in KS
                },
                "chosen": {k: {r: repr(h) for r, h in res[c]["chosen"][k].items()} for k in KS},
                "seconds": secs[c],
                "E_paths": {k: res[c]["E"][k].tolist() for k in KS},
            }
            for c in cands
        },
        "vs_dose": rows,
        "holm_family": {f"{c}@k{k}": v for (c, k), v in dec.items()},
        "beats_dose": beats,
        "frozen": frozen,
        "verdict": f"{LABEL[frozen]} beats pooled + dose" if frozen else "none beats pooled + dose",
        "best_point_estimate": best_point,
        "frozen_contrast": {"cand": fz_cand, "ref": fz_ref, "settings": fz},
        "nulls": nulls,
        "null_max": NULL_MAX,
        "power": {"alpha": POWER_ALPHA, "contrast": f"{fz_cand} vs {fz_ref}", "table": power},
        "power_best_point": {"contrast": f"{best_point} vs dose", "table": power_best}
        if power_best
        else None,
        "params": {f"{arch}_h{h}": n_params(arch, h) for arch in ("gru", "ssm") for h in SEQ_H},
        "paths": A["ids"],
        "runs": A["runs"].tolist(),
        "seconds_total": time.time() - t0,
    }
    Path(a.json).write_text(json.dumps(_js(out), indent=1) + "\n")
    log("wrote", a.json)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
