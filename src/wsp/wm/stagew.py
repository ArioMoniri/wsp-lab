"""Stage W (docs/council/2026-11-18/RULING-wme-stagew.md; SYNTHESIS-r11 R4-R6). EXPLORATORY; CPU only.

Pure numpy/scipy (scikit-learn for StatNN; ``lcpfn`` optional), importable on Python 3.10 so the analysis can run in
the LC-PFN venv.

* **Design** (:func:`split_teacher`, :func:`ckpt_steps`, :func:`member`, :func:`make_paths`, :func:`queue`): four
  fresh-seed H1-type runs; per run a training-step path S, a dose path L and a mask-growth path M of 8 points that
  share the point (final step, lambda 1, all layers); merge paths X between two runs' final checkpoints (zero-shot,
  descriptive). The evaluation queue is S paths, then L, then M, then X.
* **Features** (:func:`member_features`): R0 = Frobenius norm of the realised delta per coarse layer group (4);
  R1 = mean over feature items of <grad f_o(theta0), dW(m)> for the stop, target and margin objectives (3). Both are
  exact functions of per-layer quantities of the saved checkpoints (squared norms, cross inner products, gradient
  projections), so every dose, mask and merge member is computed without touching the model.
* **State** (:class:`WM`, R5): per benchmark and base item cell c = (base truncated, base correct, length bin),
  logit P(finish) = a_c + phi.(beta + eta l_c + zeta T_c) + delta_f and logit P(correct | finish) =
  b_c + phi.(kappa + nu l_c) + delta_c; accuracy = sum_c n_c P(f) P(c|f) / N, truncation = 1 - sum_c n_c P(f) / N.
  Grouped-binomial likelihood, L2 penalty 1.0 on every phi coefficient; path offsets delta from the first k points.
* **Baselines** (R4): persistence, 2-parameter power law, a parametric learning-curve ensemble, LC-PFN, the pooled
  cross-path curve with offset, StatNN (MLP) and the static map (ridge); learned arms use the k given points as a
  per-endpoint offset, the WM through delta. Zero-shot symmetric shrinkage toward the base (R6 analogue).
* **Statistic** (:func:`analyse`, :func:`decide`): leave-one-run-out cross-fitting; path score = mean |error|/SE over
  the points after k and the 4 endpoints; one-sided sign-flip over paths per baseline; intersection-union within a
  setting; Holm over the settings k in {0, 2, 4}.
* **Nulls** (:func:`null_shuffle_features`, :func:`null_permute_paths`) and the synthetic test world
  (:func:`synthetic_world`, :func:`generate_counts`).
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy import optimize

NAMESPACE = "wme-w"
SALT = "wme-w-panel-20261118"
PANELS = ("mmlupro_gen", "supergpqa_gen")
N_DEV = 500
BUDGET = 2048
N_LAYERS = 32
N_POINTS = 8
FINE_GROUP = 4  # layers per fine group (8 groups)
COARSE_GROUP = 8  # layers per coarse group (4 groups)
RUNS = {"r1": ("A", 1e-4), "r2": ("B", 1e-4), "r3": ("A", 2e-4), "r4": ("B", 2e-4)}
RUN_ORDER = ("r1", "r2", "r3", "r4")
LAMS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
MERGES = (("r1", "r2"), ("r3", "r4"))
MERGE_W = tuple(1.0 - j / 7.0 for j in range(8))
N_FEATURE = 32  # feature items per benchmark
OBJECTIVES = ("stop", "target", "margin")
FEATURES = ("norm_g1", "norm_g2", "norm_g3", "norm_g4", "j1_stop", "j1_target", "j1_margin")
ENDPOINTS = tuple(f"{e}:{p}" for e in ("acc", "trunc") for p in PANELS)
KS = (0, 2, 4)
N_LBINS = 6
N_CELLS = 1 + 2 * N_LBINS
PENALTY = 1.0
INTERCEPT_PENALTY = 1e-4  # numerical only (keeps intercepts of cells with no variation finite)
OFFSET_PENALTY = 1.0
SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SHRINK_TIE = 0.005
RIDGE_ALPHA = 1.0
MLP = {"hidden": 32, "alpha": 1e-2, "seeds": (0, 1, 2, 3, 4), "max_iter": 2000}
ALPHA = 0.025
LCPFN_X_STEP = 12.5
SETTING_BASELINES = {
    0: ("persistence", "pooled", "statnn", "static"),
    2: ("persistence", "pow2", "lc_ensemble", "lcpfn", "pooled", "statnn", "static"),
    4: ("persistence", "pow2", "lc_ensemble", "lcpfn", "pooled", "statnn", "static"),
}
BASE = {"run": None, "step": 0, "lam": 0.0, "groups": 0}


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}:{label}".encode()).hexdigest()[:8], 16)


def _rank(label: str, *parts) -> str:
    return hashlib.sha256(
        ":".join((NAMESPACE, label) + tuple(map(str, parts))).encode()
    ).hexdigest()


# ------------------------------------------------------------------ design


def split_teacher(rows, n_feature: int = N_FEATURE) -> dict:
    """Eligible rows (correct, non-empty answer, <= BUDGET tokens) -> {'feature', 'A', 'B'}: ``n_feature`` per
    benchmark held out (seed ``wme-w:feature-items``), the rest alternated into shards A/B by ``wme-w:shards``."""
    ok = [r for r in rows if r["correct"] and r["answer"] and r["n_tokens"] <= BUDGET]
    out = {"feature": [], "A": [], "B": []}
    for p in PANELS:
        rs = sorted(
            (r for r in ok if r["panel"] == p), key=lambda r: _rank("feature-items", p, r["id"])
        )
        out["feature"] += rs[:n_feature]
        rest = sorted(rs[n_feature:], key=lambda r: _rank("shards", p, r["id"]))
        for i, r in enumerate(rest):
            out["AB"[i % 2]].append(r)
    return out


def ckpt_steps(n_steps: int) -> list:
    if n_steps < N_POINTS:
        raise ValueError(f"a run needs >= {N_POINTS} steps, got {n_steps}")
    s = [max(1, round(n_steps * j / N_POINTS)) for j in range(1, N_POINTS + 1)]
    if s != sorted(set(s)):
        raise ValueError(f"checkpoints not strictly increasing: {s}")
    return s


def member(run, step, lam: float = 1.0, groups: int = 8, other=None, w: float = 1.0) -> dict:
    """Canonical member spec; a merge with w = 1 or w = 0 is the plain parent member."""
    lam = round(float(lam), 6)
    if other is not None:
        w = round(float(w), 6)
        if w == 1.0:
            other = None
        elif w == 0.0:
            run, step, other = other[0], other[1], None
    m = {"run": run, "step": int(step), "lam": lam, "groups": int(groups)}
    if other is not None:
        m["merge"] = {"run": other[0], "step": int(other[1]), "w": w}
    return m


def key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def unkey(k: str) -> dict:
    return json.loads(k)


def make_paths(final_steps: dict) -> list:
    """S/L/M paths per run (in RUN_ORDER) and the X merge paths; each {'id', 'type', 'runs', 'keys', 'x'}."""
    out = []
    runs = [r for r in RUN_ORDER if r in final_steps]
    x = list(range(1, N_POINTS + 1))
    for r in runs:
        T = final_steps[r]
        out.append(
            {
                "id": f"S:{r}",
                "type": "S",
                "runs": (r,),
                "keys": [key(member(r, t)) for t in ckpt_steps(T)],
                "x": x,
            }
        )
        out.append(
            {
                "id": f"L:{r}",
                "type": "L",
                "runs": (r,),
                "keys": [key(member(r, T, lam=lm)) for lm in LAMS],
                "x": x,
            }
        )
        out.append(
            {
                "id": f"M:{r}",
                "type": "M",
                "runs": (r,),
                "keys": [key(member(r, T, groups=j)) for j in range(1, 9)],
                "x": x,
            }
        )
    for a, b in MERGES:
        if a in final_steps and b in final_steps:
            ks = [key(member(a, final_steps[a], other=(b, final_steps[b]), w=w)) for w in MERGE_W]
            out.append({"id": f"X:{a}-{b}", "type": "X", "runs": (a, b), "keys": ks, "x": x})
    return out


def queue(paths) -> list:
    """Unique member keys in the frozen priority order: S paths, L, M, then X."""
    seen, out = set(), []
    for t in "SLMX":
        for p in paths:
            if p["type"] != t:
                continue
            for k in p["keys"]:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
    return out


# ------------------------------------------------------------------ features


def _terms(spec):
    lam = spec["lam"]
    m = spec.get("merge")
    if m is None:
        return [((spec["run"], spec["step"]), lam)]
    return [
        ((spec["run"], spec["step"]), lam * m["w"]),
        ((m["run"], m["step"]), lam * (1 - m["w"])),
    ]


def _inner(a, b, sq, cross):
    if a == b:
        return np.asarray(sq[a], float)
    k = (a[0], a[1], b[0], b[1])
    if k in cross:
        return np.asarray(cross[k], float)
    return np.asarray(cross[(b[0], b[1], a[0], a[1])], float)


def member_features(spec, sq, cross, j1) -> np.ndarray:
    """sq[(run, step)]: per-layer squared Frobenius norm of dW at lambda 1 (both writers); cross[(r, t, r', t')]:
    per-layer inner products; j1[(run, step)]: [3 objectives, items, layer] gradient projections at lambda 1."""
    if spec.get("run") is None or spec["lam"] == 0 or spec["groups"] == 0:
        return np.zeros(len(FEATURES))
    mask = np.zeros(N_LAYERS, bool)
    mask[: FINE_GROUP * spec["groups"]] = True
    terms = _terms(spec)
    sqm = np.zeros(N_LAYERS)
    for a, ca in terms:
        for b, cb in terms:
            sqm += ca * cb * _inner(a, b, sq, cross)
    sqm = np.where(mask, np.maximum(sqm, 0.0), 0.0)
    norms = [math.sqrt(sqm[g * COARSE_GROUP : (g + 1) * COARSE_GROUP].sum()) for g in range(4)]
    jj = sum(c * np.asarray(j1[a], float) for a, c in terms)
    j = (jj * mask).sum(-1).mean(-1)
    return np.array(norms + list(j))


# ------------------------------------------------------------------ item cells


def item_cells(base_records, n_bins: int = N_LBINS, budget: int = BUDGET) -> dict:
    """Cells from the base generation on the dev items: 0 = base truncated; 1 + C * n_bins + bin otherwise (bins are
    quantiles of the base length among untruncated items)."""
    lens = np.array([r["len"] for r in base_records if not r["trunc"]], float)
    edges = np.quantile(lens, np.arange(1, n_bins) / n_bins) if len(lens) else np.zeros(n_bins - 1)
    cell = []
    for r in base_records:
        if r["trunc"]:
            cell.append(0)
        else:
            b = int(np.searchsorted(edges, r["len"], side="right"))
            cell.append(1 + int(bool(r["correct"])) * n_bins + b)
    nc = 1 + 2 * n_bins
    n = np.bincount(cell, minlength=nc).astype(float)
    lsum = np.bincount(
        cell, weights=[min(r["len"], budget) / budget for r in base_records], minlength=nc
    )
    ell = np.where(n > 0, lsum / np.maximum(n, 1), 0.5)
    ell[0] = 1.0
    T = np.zeros(nc)
    T[0] = 1.0
    C = np.array([0.0] + [0.0] * n_bins + [1.0] * n_bins)
    return {"cell": cell, "n": n, "l": ell, "T": T, "C": C, "edges": edges.tolist()}


def cell_counts(records, cells) -> np.ndarray:
    """[cell, (n, n_finish, n_correct)]; a truncated response is never correct."""
    out = np.zeros((len(cells["n"]), 3))
    for c, r in zip(cells["cell"], records):
        out[c, 0] += 1
        if not r["trunc"]:
            out[c, 1] += 1
            out[c, 2] += bool(r["correct"])
    return out


def endpoints_from_counts(cnt) -> tuple:
    n = cnt[:, 0].sum()
    return float(cnt[:, 2].sum() / n), float(1.0 - cnt[:, 1].sum() / n)


def y_of(counts: dict) -> np.ndarray:
    acc = [endpoints_from_counts(counts[p])[0] for p in PANELS]
    tr = [endpoints_from_counts(counts[p])[1] for p in PANELS]
    return np.array(acc + tr)


def se_of(y, n) -> np.ndarray:
    y = np.asarray(y, float)
    return np.sqrt(np.maximum(y * (1 - y), 1.0 / n) / n)


# ------------------------------------------------------------------ the R5 state model


def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _logsig(z):
    return -np.logaddexp(0.0, -z)


class WM:
    """The per-benchmark finish x correct|finish state on standardised member features (zero at the base)."""

    def __init__(self, penalty: float = PENALTY, offset_penalty: float = OFFSET_PENALTY):
        self.penalty, self.offset_penalty = penalty, offset_penalty

    def _std(self, phi):
        return np.asarray(phi, float) / self.scale

    def fit(self, members, cells):
        X = np.stack([m["phi"] for m in members])
        s = np.sqrt((X**2).mean(0))
        self.scale = np.where(s > 0, s, 1.0)
        self.cells = cells
        Xs = self._std(X)
        F = Xs.shape[1]
        self.params = {}
        for p in PANELS:
            cnt = np.stack([m["counts"][p] for m in members])
            n, nf, nc = cnt[..., 0], cnt[..., 1], cnt[..., 2]
            ell, T = cells[p]["l"], cells[p]["T"]
            C = len(ell)
            pf = self._fit_one(Xs, n, nf, [ell, T], C, F)
            pc = self._fit_one(Xs, nf, nc, [ell], C, F)
            self.params[p] = (pf, pc)
        return self

    def _fit_one(self, Xs, n, k, mods, C, F):
        nm = len(mods)
        pen = np.concatenate([np.full(C, INTERCEPT_PENALTY), np.full(F * (1 + nm), self.penalty)])

        def f(theta):
            a = theta[:C]
            B = theta[C:].reshape(1 + nm, F)
            G = Xs @ B.T  # [M, 1 + nm]
            z = (
                a[None, :]
                + G[:, :1]
                + sum(G[:, i + 1 : i + 2] * mods[i][None, :] for i in range(nm))
            )
            nll = -(k * _logsig(z) + (n - k) * _logsig(-z)).sum()
            r = n * _sig(z) - k
            ga = r.sum(0)
            gB = [Xs.T @ r.sum(1)] + [Xs.T @ (r @ mods[i]) for i in range(nm)]
            g = np.concatenate([ga] + gB) + pen * theta
            return nll + 0.5 * (pen * theta**2).sum(), g

        res = optimize.minimize(f, np.zeros(C + F * (1 + nm)), jac=True, method="L-BFGS-B")
        return res.x

    def _z(self, Xs, p, delta=(0.0, 0.0)):
        (pf, pc), cells = self.params[p], self.cells[p]
        C, F = len(cells["l"]), Xs.shape[1]
        Bf = pf[C:].reshape(3, F)
        G = Xs @ Bf.T
        zf = (
            pf[None, :C]
            + G[:, :1]
            + G[:, 1:2] * cells["l"][None, :]
            + G[:, 2:3] * cells["T"][None, :]
        )
        Bc = pc[C:].reshape(2, F)
        H = Xs @ Bc.T
        zc = pc[None, :C] + H[:, :1] + H[:, 1:2] * cells["l"][None, :]
        return zf + delta[0], zc + delta[1]

    def logits(self, phi, offsets=None):
        Xs = self._std(np.atleast_2d(phi))
        return {p: self._z(Xs, p, (offsets or {}).get(p, (0.0, 0.0))) for p in PANELS}

    def predict(self, phi, offsets=None) -> np.ndarray:
        """[members, ENDPOINTS] (accuracy per benchmark, then truncation per benchmark)."""
        z = self.logits(phi, offsets)
        acc, tr = [], []
        for p in PANELS:
            n = self.cells[p]["n"]
            pf, pc = _sig(z[p][0]), _sig(z[p][1])
            acc.append((pf * pc * n).sum(1) / n.sum())
            tr.append(1.0 - (pf * n).sum(1) / n.sum())
        return np.stack(acc + tr, 1)

    def fit_offsets(self, members) -> dict:
        """Two path offsets per benchmark (finish, correct) on the given points, all else fixed."""
        if not members:
            return {p: (0.0, 0.0) for p in PANELS}
        X = np.stack([m["phi"] for m in members])
        z = self.logits(X)
        out = {}
        for p in PANELS:
            cnt = np.stack([m["counts"][p] for m in members])
            n, nf, nc = cnt[..., 0], cnt[..., 1], cnt[..., 2]
            ds = []
            for zz, nn, kk in ((z[p][0], n, nf), (z[p][1], nf, nc)):

                def f(d, zz=zz, nn=nn, kk=kk):
                    zd = zz + d[0]
                    nll = -(kk * _logsig(zd) + (nn - kk) * _logsig(-zd)).sum()
                    g = (nn * _sig(zd) - kk).sum() + self.offset_penalty * d[0]
                    return nll + 0.5 * self.offset_penalty * d[0] ** 2, np.array([g])

                ds.append(
                    float(optimize.minimize(f, np.zeros(1), jac=True, method="L-BFGS-B").x[0])
                )
            out[p] = tuple(ds)
        return out


def shift_member(m, cells, df=0.0, dc=0.0) -> dict:
    """A member with expected counts after shifting every cell's empirical logits (tests)."""
    out = {"phi": m["phi"], "counts": {}}
    for p in PANELS:
        c = m["counts"][p]
        n = c[:, 0]
        pf = np.clip((c[:, 1] + 0.5) / (n + 1), 1e-4, 1 - 1e-4)
        pc = np.clip((c[:, 2] + 0.5) / (c[:, 1] + 1), 1e-4, 1 - 1e-4)
        pf2 = _sig(np.log(pf / (1 - pf)) + df)
        pc2 = _sig(np.log(pc / (1 - pc)) + dc)
        out["counts"][p] = np.stack([n, n * pf2, n * pf2 * pc2], 1)
    out["y"] = y_of(out["counts"])
    return out


# ------------------------------------------------------------------ learned baselines


class Static:
    """Ridge (no intercept; zero at the base) from standardised features to the endpoint changes."""

    def fit(self, X, D):
        s = np.sqrt((X**2).mean(0))
        self.scale = np.where(s > 0, s, 1.0)
        Xs = X / self.scale
        self.W = np.linalg.solve(Xs.T @ Xs + RIDGE_ALPHA * np.eye(Xs.shape[1]), Xs.T @ D)
        return self

    def predict(self, X):
        return (np.atleast_2d(X) / self.scale) @ self.W


class StatNN:
    """MLP on the same standardised features (no structure), averaged over seeds."""

    def fit(self, X, D):
        from sklearn.neural_network import MLPRegressor

        s = np.sqrt((X**2).mean(0))
        self.scale = np.where(s > 0, s, 1.0)
        Xs = X / self.scale
        self.models = []
        for sd in MLP["seeds"]:
            m = MLPRegressor(
                hidden_layer_sizes=(MLP["hidden"],),
                activation="tanh",
                alpha=MLP["alpha"],
                solver="lbfgs",
                max_iter=MLP["max_iter"],
                random_state=sd,
            )
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xs, D)
            self.models.append(m)
        return self

    def predict(self, X):
        Xs = np.atleast_2d(X) / self.scale
        return np.mean([m.predict(Xs).reshape(len(Xs), -1) for m in self.models], 0)


# ------------------------------------------------------------------ per-path curve baselines


def _fam(name, y0):
    if name == "lin":
        return 1, lambda x, t: y0 + t[0] * x, ([-1], [1])
    if name == "log":
        return 1, lambda x, t: y0 + t[0] * np.log1p(x), ([-1], [1])
    if name == "pow2":
        return 2, lambda x, t: y0 + t[0] * x ** t[1], ([-2, 0.01], [2, 3])
    if name == "exp2":
        return 2, lambda x, t: y0 + t[0] * (1 - np.exp(-t[1] * x)), ([-2, 0.01], [2, 5])
    if name == "pow3":
        return 3, lambda x, t: t[0] - t[1] * x ** (-t[2]), ([-2, -2, 0.01], [3, 3, 3])
    if name == "exp3":
        return 3, lambda x, t: t[0] - t[1] * np.exp(-t[2] * x), ([-2, -2, 0.01], [3, 3, 5])
    raise KeyError(name)


def _fit_family(name, y0, xk, yk):
    npar, fn, (lo, hi) = _fam(name, y0)
    if len(xk) < npar:
        return None
    best = None
    for start in ([0.1, 0.5, 0.5][:npar], [0.5, 1.0, 1.0][:npar], [-0.1, 0.5, 0.5][:npar]):
        x0 = np.clip(np.array(start, float), np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            r = optimize.least_squares(lambda t: fn(xk, t) - yk, x0, bounds=(lo, hi))
        except (ValueError, FloatingPointError):
            continue
        if r.success and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        return None
    return lambda x, t=best.x: fn(np.asarray(x, float), t)


ENSEMBLE = {2: ("lin", "log", "pow2", "exp2"), 4: ("lin", "log", "pow2", "exp2", "pow3", "exp3")}


def curve_predict(name, y0, xk, yk, xp) -> np.ndarray:
    """Per-path curve baselines on (x, y) with the base anchor (0, y0); predictions clipped to [0, 1]."""
    xk, yk, xp = (np.asarray(v, float) for v in (xk, yk, xp))
    last = yk[-1] if len(yk) else y0
    if name == "persistence" or len(xk) < 2:
        return np.full(len(xp), float(last))
    if name == "pow2":
        g = _fit_family("pow2", y0, xk, yk)
        out = g(xp) if g is not None else np.full(len(xp), last)
        return np.clip(out, 0, 1)
    if name == "lc_ensemble":
        fams = ENSEMBLE[4] if len(xk) >= 4 else ENSEMBLE[2]
        preds = []
        for f in fams:
            g = _fit_family(f, y0, xk, yk)
            if g is None:
                continue
            v = g(xp)
            if np.all(np.isfinite(v)) and np.all(np.abs(v) < 5):
                preds.append(np.clip(v, 0, 1))
        return np.mean(preds, 0) if preds else np.full(len(xp), float(last))
    raise KeyError(name)


def pooled_curve(train_changes) -> np.ndarray:
    return np.mean(np.asarray(train_changes, float), 0)


def offset_predict(curve, y, k: int) -> np.ndarray:
    curve = np.asarray(curve, float)
    if k == 0:
        return curve
    return curve + float(np.mean(np.asarray(y, float)[:k] - curve[:k]))


class LCPFN:
    """LC-PFN (lcpfn 0.1.3) posterior mean, batched over curves; the base is the anchor at x = 0 and point j sits at
    x = 12.5 j. Values are in [0, 1]; truncation is fed as 1 - rate (an improving curve)."""

    def __init__(self, model_name="EMSIZE512_NLAYERS12_NBUCKETS1000"):
        import lcpfn
        import torch

        self.torch = torch
        self.m = lcpfn.LCPFN(model_name=model_name)
        self.name = model_name

    def predict(self, xk, Yk, xp) -> np.ndarray:
        """xk [n] (anchor included), Yk [n, B], xp [m] -> [m, B]."""
        t = self.torch
        n, B = Yk.shape
        x = np.concatenate([xk, xp]).astype(np.float32) * LCPFN_X_STEP
        xx = t.tensor(x)[:, None, None].expand(len(x), B, 1).contiguous()
        yy = t.tensor(np.clip(Yk, 0, 1).astype(np.float32))
        with t.no_grad():
            logits = self.m.model((xx, yy), single_eval_pos=n)
            mean = self.m.model.criterion.mean(logits)
        return np.asarray(mean.reshape(len(xp), B).numpy(), float)


# ------------------------------------------------------------------ analysis


def choose_shrink(losses: dict) -> float:
    best = min(losses.values())
    return min(s for s, v in losses.items() if v <= best * (1 + SHRINK_TIE) + 1e-15)


def _p_signflip(d, n_flip, seed):
    from wsp.wm import signflip as SF

    d = np.asarray(d, float)
    if len(d) < 2:
        return 1.0
    return float(
        SF.test(d, n_flip=None if n_flip is None or len(d) <= 16 else n_flip, seed=seed)["p"]
    )


def decide(p: dict, alpha: float = ALPHA) -> dict:
    """{setting: {baseline: p}} -> per setting the intersection-union p (max over baselines) and the Holm decision
    over settings."""
    piu = {k: max(v.values()) for k, v in p.items()}
    order = sorted(piu, key=lambda k: piu[k])
    m = len(order)
    out = {k: {"p_iu": piu[k], "reject": False} for k in piu}
    for i, k in enumerate(order):
        if piu[k] <= alpha / (m - i):
            out[k]["reject"] = True
        else:
            break
    return out


def _runs(world):
    return sorted({p["runs"][0] for p in world["paths"] if p["type"] in "SLM"})


def _train_members(world, runs):
    keys = []
    for p in world["paths"]:
        if p["type"] in "SLM" and p["runs"][0] in runs:
            keys += p["keys"]
    return [world["members"][k] for k in dict.fromkeys(keys)]


def _fit_arms(world, runs):
    ms = _train_members(world, runs) + [world["base"]]
    X = np.stack([m["phi"] for m in ms])
    D = np.stack([m["y"] for m in ms]) - world["base"]["y"][None, :]
    return {
        "wm": WM().fit(ms, world["cells"]),
        "statnn": StatNN().fit(X, D),
        "static": Static().fit(X, D),
    }


def _arm_predict(arms, name, X, y0):
    if name == "wm":
        return arms["wm"].predict(X)
    return y0[None, :] + arms[name].predict(X)


def _score(pred, y, n, k):
    se = se_of(y, n)
    return float(np.mean(np.abs(pred[k:] - y[k:]) / se[k:]))


def fold_groups(runs, n_folds=None) -> list:
    """Leave-one-run-out (``n_folds`` None) or runs dealt round-robin into ``n_folds`` folds."""
    runs = list(runs)
    if n_folds is None or n_folds >= len(runs):
        return [[r] for r in runs]
    return [runs[i::n_folds] for i in range(n_folds)]


def _shrinks(world, runs, n_inner=None):
    """Zero-shot s per learned arm from inner folds on the training runs (same loss, same grid, every arm)."""
    y0 = world["base"]["y"]
    loss = {a: {s: 0.0 for s in SHRINK_GRID} for a in ("wm", "statnn", "static")}
    if len(runs) < 2:
        return {a: 1.0 for a in loss}
    for Q in fold_groups(runs, n_inner):
        arms = _fit_arms(world, [r for r in runs if r not in Q])
        for p in world["paths"]:
            if p["type"] not in "SLM" or p["runs"][0] not in Q:
                continue
            X = np.stack([world["members"][k]["phi"] for k in p["keys"]])
            y = np.stack([world["members"][k]["y"] for k in p["keys"]])
            for a, la in loss.items():
                pr = _arm_predict(arms, a, X, y0)
                for s in SHRINK_GRID:
                    la[s] += _score(y0[None, :] + s * (pr - y0[None, :]), y, world["N"], 0)
    return {a: choose_shrink(v) for a, v in loss.items()}


def _curve_preds(name, y0, x, y, k, lcpfn):
    """[8, endpoints]; points <= k are copied from y (unused)."""
    out = y.copy()
    if name == "lcpfn":
        if lcpfn is None:
            return None
        xk = np.concatenate([[0.0], x[:k]])
        Y = np.concatenate([y0[None, :], y[:k]], 0)
        flip = np.array([0, 0, 1, 1], bool)
        Yt = np.where(flip[None, :], 1 - Y, Y)
        P = lcpfn.predict(xk, Yt, x[k:])
        out[k:] = np.where(flip[None, :], 1 - P, P)
        return out
    for e in range(y.shape[1]):
        out[k:, e] = curve_predict(name, y0[e], x[:k], y[:k, e], x[k:])
    return out


def _one_thread(fn):
    """BLAS thread pools pinned to one thread (small problems; oversubscription on a shared host is ~20x slower)."""

    def wrap(*a, **k):
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            return fn(*a, **k)

    wrap.__name__, wrap.__doc__ = fn.__name__, fn.__doc__
    return wrap


@_one_thread
def analyse(
    world, n_flip=10_000, lcpfn=None, inner=True, ks=KS, seed=0, n_folds=None, n_inner=None
) -> dict:
    """Leave-one-run-out H-T2 analysis on S/L/M paths; per setting per arm per path scores, contrasts, tests."""
    N, y0 = world["N"], world["base"]["y"]
    runs = _runs(world)
    E = {k: {} for k in ks}
    pid = []
    ptype = []
    zs = {"pred": [], "true": [], "statnn": []}
    shr = {}
    for F in fold_groups(runs, n_folds):
        tr = [q for q in runs if q not in F]
        arms = _fit_arms(world, tr)
        s = _shrinks(world, tr, n_inner) if inner else {a: 1.0 for a in ("wm", "statnn", "static")}
        shr["+".join(F)] = s
        pooled = {}
        for t in "SLM":
            ch = [
                np.stack([world["members"][kk]["y"] for kk in p["keys"]]) - y0[None, :]
                for p in world["paths"]
                if p["type"] == t and p["runs"][0] in tr
            ]
            pooled[t] = (
                y0[None, :] + pooled_curve(ch) if ch else np.repeat(y0[None, :], N_POINTS, 0)
            )
        for p in world["paths"]:
            if p["type"] not in "SLM" or p["runs"][0] not in F:
                continue
            pid.append(p["id"])
            ptype.append(p["type"])
            ms = [world["members"][kk] for kk in p["keys"]]
            X = np.stack([m["phi"] for m in ms])
            y = np.stack([m["y"] for m in ms])
            x = np.asarray(p["x"], float)
            raw = {a: _arm_predict(arms, a, X, y0) for a in ("wm", "statnn", "static")}
            zs["pred"] += list(raw["wm"][:, :2].mean(1) - y0[:2].mean())
            zs["true"] += list(y[:, :2].mean(1) - y0[:2].mean())
            zs["statnn"] += list(raw["statnn"][:, :2].mean(1) - y0[:2].mean())
            for k in ks:
                preds = {}
                if k == 0:
                    for a, ra in raw.items():
                        preds[a] = y0[None, :] + s[a] * (ra - y0[None, :])
                    preds["persistence"] = np.repeat(y0[None, :], N_POINTS, 0)
                    preds["pooled"] = pooled[p["type"]]
                else:
                    off = arms["wm"].fit_offsets(ms[:k])
                    preds["wm"] = arms["wm"].predict(X, off)
                    for a in ("statnn", "static"):
                        preds[a] = np.stack(
                            [offset_predict(raw[a][:, e], y[:, e], k) for e in range(4)], 1
                        )
                    preds["pooled"] = np.stack(
                        [offset_predict(pooled[p["type"]][:, e], y[:, e], k) for e in range(4)], 1
                    )
                    for c in ("persistence", "pow2", "lc_ensemble", "lcpfn"):
                        v = _curve_preds(c, y0, x, y, k, lcpfn)
                        if v is not None:
                            preds[c] = v
                for a, v in preds.items():
                    E[k].setdefault(a, []).append(_score(v, y, N, k))
    out = {"paths": pid, "types": ptype, "shrink": shr, "settings": {}, "runs": runs}
    pvals = {}
    for k in ks:
        Ek = {a: np.array(v) for a, v in E[k].items()}
        bl = [b for b in SETTING_BASELINES[k] if b in Ek]
        st = {
            "E": {a: v.tolist() for a, v in Ek.items()},
            "baselines": bl,
            "C": {},
            "d_mean": {},
            "p": {},
        }
        st["mean_E"] = {a: float(v.mean()) for a, v in Ek.items()}
        for b in bl:
            d = Ek[b] - Ek["wm"]
            st["C"][b] = float(1 - Ek["wm"].mean() / Ek[b].mean())
            st["d_mean"][b] = float(d.mean())
            st["p"][b] = _p_signflip(d, n_flip, seed)
            st.setdefault("by_type", {})
            for t in "SLM":
                ix = [i for i, tt in enumerate(ptype) if tt == t]
                if ix:
                    st["by_type"].setdefault(t, {})[b] = float(np.mean(d[ix]))
        pvals[k] = st["p"]
        out["settings"][k] = st
    dec = decide(pvals)
    for k in ks:
        out["settings"][k].update(dec[k])
    from scipy.stats import spearmanr

    tru, pre, snn = (np.array(zs[k]) for k in ("true", "pred", "statnn"))
    out["gate"] = {
        "what": "zero-shot (unshrunk) predicted vs measured change in mean accuracy over both benchmarks, every "
        "held-out S/L/M point (world-model seat's stage-W gate; descriptive)",
        "spearman_wm_vs_measured_dacc": float(spearmanr(pre, tru).statistic),
        "spearman_lb": _spearman_lb(pre, tru, pid_groups=_groups(pid, world)),
        "wm_mean_abs_err_dacc": float(np.mean(np.abs(pre - tru))),
        "statnn_mean_abs_err_dacc": float(np.mean(np.abs(snn - tru))),
    }
    return out


def _groups(pid, world):
    g = []
    for i in pid:
        p = next(q for q in world["paths"] if q["id"] == i)
        g += [p["runs"][0]] * len(p["keys"])
    return g


def _spearman_lb(x, y, pid_groups, n_boot=2000, seed=0):
    """Run-cluster bootstrap 2.5 % bound of Spearman (points of a run move together)."""
    from scipy.stats import spearmanr

    g = np.asarray(pid_groups)
    us = sorted(set(g))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(us, len(us))
        ix = np.concatenate([np.flatnonzero(g == u) for u in pick])
        if len(set(x[ix])) < 2 or len(set(y[ix])) < 2:
            continue
        vals.append(spearmanr(x[ix], y[ix]).statistic)
    return float(np.quantile(vals, 0.025)) if vals else float("nan")


@_one_thread
def merges_zero_shot(world) -> dict:
    """Merge members predicted zero-shot with both parent runs held out (descriptive)."""
    out = {}
    N, y0 = world["N"], world["base"]["y"]
    runs = _runs(world)
    for p in world["paths"]:
        if p["type"] != "X" or not all(k in world["members"] for k in p["keys"]):
            continue
        tr = [r for r in runs if r not in p["runs"]]
        if not tr:
            continue
        arms = _fit_arms(world, tr)
        s = _shrinks(world, tr)
        ms = [world["members"][k] for k in p["keys"]]
        X = np.stack([m["phi"] for m in ms])
        y = np.stack([m["y"] for m in ms])
        res = {"persistence": _score(np.repeat(y0[None, :], len(ms), 0), y, N, 0)}
        for a in ("wm", "statnn", "static"):
            pr = _arm_predict(arms, a, X, y0)
            res[a] = _score(y0[None, :] + s[a] * (pr - y0[None, :]), y, N, 0)
            res[a + "_unshrunk"] = _score(pr, y, N, 0)
        out[p["id"]] = res
    return out


# ------------------------------------------------------------------ nulls


def _copy_world(world):
    w = dict(world)
    w["members"] = {k: dict(v) for k, v in world["members"].items()}
    w["paths"] = [dict(p) for p in world["paths"]]
    return w


def _derangement(n, rng):
    if n < 2:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def null_shuffle_features(world, seed=0) -> dict:
    """N1: feature trajectories permuted across S/L/M paths (point j of path p takes point j of path pi(p))."""
    rng = np.random.default_rng([seed, 1])
    w = _copy_world(world)
    ps = [p for p in world["paths"] if p["type"] in "SLM"]
    pi = _derangement(len(ps), rng)
    done = set()
    for i, p in enumerate(ps):
        src = ps[pi[i]]
        for k, ks in zip(p["keys"], src["keys"]):
            if k in done:
                continue
            done.add(k)
            w["members"][k]["phi"] = world["members"][ks]["phi"]
    return w


def null_permute_paths(world, seed=0) -> dict:
    """N2: outcome curves reassigned among paths of the same type (features stay)."""
    rng = np.random.default_rng([seed, 2])
    w = _copy_world(world)
    done = set()
    for t in "SLM":
        ps = [p for p in world["paths"] if p["type"] == t]
        pi = _derangement(len(ps), rng)
        for i, p in enumerate(ps):
            src = ps[pi[i]]
            for k, ks in zip(p["keys"], src["keys"]):
                if k in done:
                    continue
                done.add(k)
                w["members"][k]["counts"] = world["members"][ks]["counts"]
                w["members"][k]["y"] = world["members"][ks]["y"]
    return w


# ------------------------------------------------------------------ generation (synthetic world, simulation)


def generate_counts(zf, zc, n, rng) -> np.ndarray:
    """Binomial counts per cell from finish/correct logits."""
    nf = rng.binomial(n.astype(int), _sig(zf))
    nc = rng.binomial(nf, _sig(zc))
    return np.stack([n, nf, nc], 1).astype(float)


def default_cells(n_items=N_DEV, frac_trunc=0.25, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i, p in enumerate(PANELS):
        recs = []
        for _ in range(n_items):
            t = rng.random() < frac_trunc + 0.1 * i
            recs.append(
                {
                    "trunc": t,
                    "correct": (not t) and rng.random() < 0.7 - 0.3 * i,
                    "len": 2048 if t else int(rng.integers(80, 2000)),
                }
            )
        out[p] = item_cells(recs)
    return out


def synthetic_world(seed=0, n_runs=4, edit_effect=True, path_sd=0.15, cells=None) -> dict:
    """A stage-W-shaped world: features grow along S (sqrt step), L (dose) and M (layers), with a run factor; the
    truth is the WM's own form (edit_effect) or the pooled curve of each path type without feature dependence."""
    rng = np.random.default_rng(seed)
    cells = cells or default_cells(seed=seed)
    runs = list(RUN_ORDER[:n_runs])
    steps = {r: 96 for r in runs}
    paths = [p for p in make_paths(steps) if p["type"] in "SLM"]
    v = np.array([1.0, 0.8, 0.6, 0.4, 1.0, 0.6, 0.3])
    truth = WM()
    truth.scale = np.ones(len(FEATURES))
    truth.cells = cells
    truth.params = {}
    F = len(FEATURES)
    for p in PANELS:
        C = len(cells[p]["n"])
        a = np.where(cells[p]["T"] > 0, -2.5, 3.0)
        beta = np.zeros(F)
        beta[4] = 0.6
        zeta = np.zeros(F)
        zeta[4] = 1.2
        eta = np.zeros(F)
        b = np.where(cells[p]["C"] > 0, 2.0, -1.5)
        b[0] = -0.5
        kappa = np.zeros(F)
        kappa[:4] = -0.08
        nu = np.zeros(F)
        nu[5] = 0.3
        truth.params[p] = (np.concatenate([a, beta, eta, zeta]), np.concatenate([b, kappa, nu]))
        del C
    phis = {}
    fac = {r: float(np.exp(rng.normal(0, 0.4))) for r in runs}
    for pth in paths:
        r = pth["runs"][0]
        for j, k in enumerate(pth["keys"], start=1):
            if pth["type"] == "S":
                ph = fac[r] * v * math.sqrt(j / 8)
            elif pth["type"] == "L":
                ph = fac[r] * v * LAMS[j - 1]
            else:
                mask = np.array([min(1.0, max(0.0, (j - 2 * g) / 2)) for g in range(4)])
                ph = fac[r] * np.concatenate([v[:4] * np.sqrt(mask), v[4:] * j / 8])
            phis.setdefault(k, ph)
    keys = list(dict.fromkeys(k for p in paths for k in p["keys"]))
    X = np.stack([phis[k] for k in keys])
    z = truth.logits(X)
    z0 = truth.logits(np.zeros((1, F)))
    if not edit_effect:
        # each path's truth is its type's mean curve (no feature dependence)
        idx = {k: i for i, k in enumerate(keys)}
        zt = {p: (np.zeros_like(z[p][0]), np.zeros_like(z[p][1])) for p in PANELS}
        for t in "SLM":
            ps = [p for p in paths if p["type"] == t]
            for p in PANELS:
                mf = np.mean([[z[p][0][idx[k]] for k in q["keys"]] for q in ps], 0)
                mc = np.mean([[z[p][1][idx[k]] for k in q["keys"]] for q in ps], 0)
                for q in ps:
                    for j, k in enumerate(q["keys"]):
                        zt[p][0][idx[k]] = mf[j]
                        zt[p][1][idx[k]] = mc[j]
        z = zt
    members = {}
    eff = {pth["id"]: {p: rng.normal(0, path_sd, 2) for p in PANELS} for pth in paths}
    for i, k in enumerate(keys):
        pth = next(q for q in paths if k in q["keys"])
        counts = {
            p: generate_counts(
                z[p][0][i] + eff[pth["id"]][p][0],
                z[p][1][i] + eff[pth["id"]][p][1],
                cells[p]["n"],
                rng,
            )
            for p in PANELS
        }
        members[k] = {"phi": phis[k], "counts": counts, "y": y_of(counts)}
    bc = {p: generate_counts(z0[p][0][0], z0[p][1][0], cells[p]["n"], rng) for p in PANELS}
    base = {"phi": np.zeros(F), "counts": bc, "y": y_of(bc)}
    return {
        "cells": cells,
        "members": members,
        "base": base,
        "paths": paths,
        "N": N_DEV,
        "truth": truth,
    }


# ------------------------------------------------------------------ fresh panel wme-w


def _mc_row(r, prefix):
    from wsp.wm.panel import text_hash

    g = r["question"] + "||" + "||".join(r["options"])
    h = text_hash(g)
    return {
        "kind": prefix,
        "question": r["question"],
        "options": list(r["options"]),
        "answer_index": int(r["answer_index"]),
        "hash": h,
        "id": f"{prefix}:{h[:16]}",
        "gtext": g,
    }


def panel_pools(mmlu_pro_rows, supergpqa_rows) -> dict:
    """Pool items exactly as the pilot-3 builder forms them (ids 'mmlug:' / 'sgpqa:', group text = question + options)."""
    return {
        "mmlupro_gen": [_mc_row(r, "mmlug") for r in mmlu_pro_rows],
        "supergpqa_gen": [_mc_row(r, "sgpqa") for r in supergpqa_rows],
    }


def panel_select(pools, exposed, n: int = N_DEV) -> dict:
    """One joint near-duplicate family over both benchmarks; every group with an exposed hash dropped; one item per
    group; dev = the first n groups in salted rank order; asserted disjoint and exposure-free."""
    from wsp.wm import p3panel as PP

    fams = {"mc": {"pools": pools}}
    sizes = {p: {"dev": n} for p in PANELS}
    res = PP.select(fams, PANELS, sizes, SALT, set(exposed), {p: ("dev",) for p in PANELS})
    res["audit"] = PP.audit(res["splits"], set(exposed))
    if not res["audit"]["clean"]:
        raise RuntimeError(f"panel audit failed: {res['audit']}")
    return res


# ------------------------------------------------------------------ loading a stage-W run


def configure_layers(n_layers: int) -> None:
    """Layer groups follow the model's depth (32 layers on the 9B: fine 4, coarse 8)."""
    global N_LAYERS, FINE_GROUP, COARSE_GROUP
    if n_layers % 8:
        raise ValueError(n_layers)
    N_LAYERS, FINE_GROUP, COARSE_GROUP = n_layers, n_layers // 8, n_layers // 4


def _ck(s):
    r, t = s.split(":")
    return r, int(t)


def load_run(d) -> dict:
    """A stage-W run directory (train_meta.json, features.npz, evals.jsonl) -> the analysis world. Only paths whose
    8 points were all evaluated are kept; the others are listed in ``incomplete``."""
    from pathlib import Path

    d = Path(d)
    tm = json.loads((d / "train_meta.json").read_text())
    configure_layers(int(tm["n_layers"]))
    z = np.load(d / "features.npz")
    cks = [_ck(str(s)) for s in z["ckpts"]]
    sq = {c: z["sq"][i] for i, c in enumerate(cks)}
    j1 = {c: z["j1"][:, :, i, :] for i, c in enumerate(cks)}
    cross = {}
    for i, s in enumerate(z["cross_pairs"]):
        a, b = str(s).split("|")
        cross[_ck(a) + _ck(b)] = z["cross"][i]
    recs = {}
    for line in (d / "evals.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs[r["key"]] = r
    base = recs["base"]
    cells = {p: item_cells(base["panels"][p]) for p in PANELS}
    N = len(base["panels"][PANELS[0]])

    def mk(rec, phi):
        counts = {p: cell_counts(rec["panels"][p], cells[p]) for p in PANELS}
        return {"phi": phi, "counts": counts, "y": y_of(counts)}

    members = {
        k: mk(r, member_features(r["spec"], sq, cross, j1))
        for k, r in recs.items()
        if r["tag"] == "member"
    }
    paths = make_paths({r: int(s) for r, s in tm["final_steps"].items()})
    complete = [p for p in paths if all(k in members for k in p["keys"])]
    world = {
        "cells": cells,
        "members": members,
        "base": mk(base, np.zeros(len(FEATURES))),
        "paths": complete,
        "N": N,
        "incomplete": [p["id"] for p in paths if p not in complete],
        "n_layers": int(tm["n_layers"]),
    }
    if "base_rerun" in recs:
        world["base_rerun"] = mk(recs["base_rerun"], np.zeros(len(FEATURES)))
        world["base_rerun_discordance"] = {
            p: float(
                np.mean(
                    [
                        a["correct"] != b["correct"]
                        for a, b in zip(base["panels"][p], recs["base_rerun"]["panels"][p])
                    ]
                )
            )
            for p in PANELS
        }
    return world
