"""MSRM-1 v0 arms and their nested fitting (SYNTHESIS-r7 Decision 1; exploratory development).

ŷ(D) = base(a, q) + h(z(x(D))) − h(z(x(0))) with

* base ``f3u``: β₀ + β₁Σa + β₂Σq by **unpenalised** least squares on the training rows; ``const``: the training mean
  (intercept-only least squares); ``zero_a``: ŷ = a with nothing fitted;
* z the train-only input transform (signed asinh by the training median |x| of each column's nonzero entries, then
  standardisation; a zero-SD column is centred with scale 1, F12a.2(h));
* h one head class: ``mlp`` (d → 8 GELU → 1, zero-initialised output, target standardised by the training residual SD
  (F12b.1), L1, AdamW lr 1e-3 / wd 1e-2 / batch 16, ≤ 300 epochs, patience 30, refit epochs
  max(1, ⌊(e₍₂₎ + e₍₃₎)/2⌋), seeds 0/1/2 averaged; F5.1 / F5 pins / F12.3), ``ridge`` (on the anchored design
  z − z(0) without intercept, λ from {1e-2 … 1e3} by inner L1, larger λ on ties), ``tree`` (LightGBM with the F12
  settings and an L1 objective, plus one anchor row (z(0), 0) weighted by the training-row count; trees from
  {100, 300, 600} by inner L1 averaged over seeds, smaller count on ties); ``statnn`` is the F12.3 augmented StatNN comparator (LightGBM, F12
  settings, L2, on the wide [x | v4 | 186 × 43 token] matrix, ŷ = F3u + tree, no zero-edit subtraction).

Nesting (F12a.2(f)): for each of the four inner folds (components of the training rows ordered by
sha256(``20261110:lol-inner:`` + component), round-robin) the base and the transform are refitted on the inner-training
rows and supply the targets of both inner-training and inner-validation rows; hyperparameters are chosen from the
inner-validation L1 losses; then base, transform and head are refitted on the whole training split.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch

INNER_DOMAIN = "20261110:lol-inner:"
N_INNER = 4
F12_LGBM = (
    ("learning_rate", 0.05),
    ("num_leaves", 8),
    ("min_child_samples", 5),
    ("subsample", 0.8),
    ("subsample_freq", 1),
    ("colsample_bytree", 0.5),
    ("reg_lambda", 1.0),
    ("n_jobs", 2),
)
HEAD_CLASSES = ("mlp", "ridge", "tree")


@dataclass(frozen=True)
class FitConfig:
    seeds: tuple = (0, 1, 2)
    max_epochs: int = 300
    patience: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch: int = 16
    hidden: int = 8
    tree_grid: tuple = (100, 300, 600)
    ridge_grid: tuple = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    lgbm: tuple = F12_LGBM
    tree_objective: str = "l1"

    def as_dict(self):
        d = dataclasses.asdict(self)
        d["lgbm"] = dict(self.lgbm)
        return d


@dataclass
class Dataset:
    row_ids: np.ndarray
    comps: np.ndarray
    y: np.ndarray
    a: np.ndarray
    q: np.ndarray
    inputs: dict = field(default_factory=dict)  # name -> (X [n, d], x0 [d])


@dataclass(frozen=True)
class ArmSpec:
    name: str
    base: str = "f3u"  # f3u | const | zero_a
    inputs: str | None = None
    head: str | None = None  # mlp | ridge | tree | statnn


# ------------------------------------------------------------------------------------------------ base


def f3u_fit(a, q, y) -> np.ndarray:
    """Unpenalised least squares of y on [1, a, q] (numpy lstsq; minimum-norm if rank-deficient)."""
    X = np.column_stack([np.ones(len(a)), a, q])
    return np.linalg.lstsq(X, np.asarray(y, dtype=np.float64), rcond=None)[0]


def f3u_predict(beta, a, q) -> np.ndarray:
    return beta[0] + beta[1] * np.asarray(a) + beta[2] * np.asarray(q)


def _base(kind, data, train):
    """(predict(rows) function, fitted parameters)."""
    if kind == "f3u":
        beta = f3u_fit(data.a[train], data.q[train], data.y[train])
        return (lambda rows: f3u_predict(beta, data.a[rows], data.q[rows])), beta.tolist()
    if kind == "const":
        m = float(np.mean(data.y[train]))
        return (lambda rows: np.full(len(rows), m)), [m]
    if kind == "zero_a":
        return (lambda rows: np.asarray(data.a[rows], dtype=np.float64)), []
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------- transform


class InputTransform:
    """Train-only signed-asinh scaling then standardisation (fitted on the rows passed to ``fit`` only)."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        s = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            nz = np.abs(X[:, j][X[:, j] != 0])
            if nz.size:
                s[j] = float(np.median(nz))
        self.scale = s
        U = np.arcsinh(X / s)
        mu, sd = U.mean(0), U.std(0)
        zero = ~(sd > 1e-12)
        self.mu, self.sd = mu, np.where(zero, 1.0, sd)
        return self

    def transform(self, X):
        return (np.arcsinh(np.asarray(X, dtype=np.float64) / self.scale) - self.mu) / self.sd


# ---------------------------------------------------------------------------------------- inner folds


def _sha_key(domain, key):
    return hashlib.sha256(f"{domain}{key}".encode()).hexdigest()


def inner_fold_of_rows(comps) -> np.ndarray:
    """F5 pins: components ordered by sha256(``20261110:lol-inner:`` + component), round-robin to four folds."""
    order = sorted(set(comps), key=lambda c: _sha_key(INNER_DOMAIN, c))
    f = {c: i % N_INNER for i, c in enumerate(order)}
    return np.array([f[c] for c in comps])


def refit_epochs(best):
    e = sorted(best)
    return max(1, (e[1] + e[2]) // 2)


# ---------------------------------------------------------------------------------------------- heads


def mlp_param_count(d: int, hidden: int = 8) -> int:
    return d * hidden + hidden + hidden + 1


class _MLP(torch.nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.lin1 = torch.nn.Linear(d, hidden)
        self.out = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, z):
        return self.out(torch.nn.functional.gelu(self.lin1(z))).squeeze(-1)


def _train_mlp(Z, r, z0, seed, cfg, *, Zv=None, rv=None, epochs=None):
    """Train h on (h(z) − h(z0)) ≈ r / sd. Early stopping on (Zv, rv) returns the best epoch; else trains
    ``epochs`` epochs and returns the model. Targets are standardised by the training-residual SD."""
    sd = float(np.std(r))
    if not math.isfinite(sd) or sd <= 1e-12:
        raise ValueError(f"residual target SD {sd} <= 1e-12")
    torch.manual_seed(seed)
    model = _MLP(Z.shape[1], cfg.hidden).double()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    Zt, rt = torch.as_tensor(Z), torch.as_tensor(r / sd)
    z0t = torch.as_tensor(z0)[None, :]
    if Zv is not None:
        Zvt, rvt = torch.as_tensor(Zv), torch.as_tensor(rv / sd)
    limit = cfg.max_epochs if epochs is None else epochs
    best, best_ep, since = math.inf, 0, 0
    for ep in range(1, limit + 1):
        perm = torch.randperm(len(Zt), generator=gen)
        for s in range(0, len(Zt), cfg.batch):
            idx = perm[s : s + cfg.batch]
            pred = model(Zt[idx]) - model(z0t)
            loss = (pred - rt[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Zv is not None:
            with torch.no_grad():
                vl = float((model(Zvt) - model(z0t) - rvt).abs().mean())
            if vl < best:
                best, best_ep, since = vl, ep, 0
            else:
                since += 1
                if since >= cfg.patience:
                    break
    if Zv is not None:
        return best_ep
    return model, sd


def _mlp_predict(model, sd, Z, z0):
    """Row by row, so a row equal to z0 goes through exactly the computation of z0 (correction exactly 0)."""
    with torch.no_grad():
        h0 = model(torch.as_tensor(z0)[None, :])
        return np.array(
            [float((model(torch.as_tensor(z)[None, :]) - h0)[0]) * sd for z in np.asarray(Z)]
        )


def _ridge_fit(Z, r, lam, z0):
    """Ridge on the anchored design (Z − z0) with no intercept: the fitted function is exactly the evaluated
    h(z) − h(z0) (residual targets have training mean 0 under an intercept-bearing base)."""
    Za = np.asarray(Z) - z0[None, :]
    return np.linalg.solve(Za.T @ Za + lam * np.eye(Za.shape[1]), Za.T @ r)


def _ridge_predict(beta, Z, z0):
    return (np.asarray(Z) - z0[None, :]) @ beta


def _lgbm(seed, n_trees, cfg, objective):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=n_trees,
        random_state=seed,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
        objective=objective,
        **dict(cfg.lgbm),
    )


def _tree_fit(seed, n_trees, cfg, objective, Z, r, z0, anchor=True):
    """LightGBM on the training rows plus, for the residual head, one anchor row (z0, 0) weighted by the number of
    training rows, so the fitted trees already carry h(z0) ≈ 0 before the exact subtraction."""
    m = _lgbm(seed, n_trees, cfg, objective)
    if not anchor:
        return m.fit(Z, r)
    Za = np.vstack([Z, z0[None, :]])
    ra = np.concatenate([r, [0.0]])
    w = np.concatenate([np.ones(len(r)), [float(len(r))]])
    return m.fit(Za, ra, sample_weight=w)


def _tree_predict(model, Z, z0, n_trees, subtract_zero=True):
    p = model.predict(np.asarray(Z), num_iteration=n_trees)
    if subtract_zero:
        p = p - model.predict(np.asarray(z0)[None, :], num_iteration=n_trees)[0]
    return p


# ------------------------------------------------------------------------------------- one nested fit


def fit_predict_arm(arm: ArmSpec, data: Dataset, train, test, cfg: FitConfig, head=None):
    """Held-out predictions of ``arm`` on ``test`` from ``train`` only (seeds averaged) and a fit record."""
    train, test = np.asarray(train), np.asarray(test)
    head = head or arm.head
    base_pred, params = _base(arm.base, data, train)
    info = {"base": arm.base, "base_params": params, "head": head}
    if head is None:
        return base_pred(test), info
    X, x0 = data.inputs[arm.inputs]
    X, x0 = np.asarray(X, dtype=np.float64), np.asarray(x0, dtype=np.float64)
    inner = inner_fold_of_rows(data.comps[train])
    y = data.y
    statnn = head == "statnn"
    objective = "l2" if statnn else cfg.tree_objective

    # --- inner selection
    folds = []
    for j in range(N_INNER):
        itr, iva = train[inner != j], train[inner == j]
        bp, _ = _base(arm.base, data, itr)
        rtr, rva = y[itr] - bp(itr), y[iva] - bp(iva)
        if statnn:
            Ztr, Zva, z0 = X[itr], X[iva], x0
        else:
            T = InputTransform().fit(X[itr])
            Ztr, Zva, z0 = T.transform(X[itr]), T.transform(X[iva]), T.transform(x0[None, :])[0]
        folds.append((Ztr, rtr, Zva, rva, z0))
    if head == "mlp":
        best = {s: [] for s in cfg.seeds}
        for Ztr, rtr, Zva, rva, z0 in folds:
            for s in cfg.seeds:
                best[s].append(_train_mlp(Ztr, rtr, z0, s, cfg, Zv=Zva, rv=rva))
        chosen = {s: refit_epochs(best[s]) for s in cfg.seeds}
        info["inner_best_epochs"] = best
    elif head == "ridge":
        losses = np.zeros(len(cfg.ridge_grid))
        for Ztr, rtr, Zva, rva, z0 in folds:
            for k, lam in enumerate(cfg.ridge_grid):
                beta = _ridge_fit(Ztr, rtr, lam, z0)
                losses[k] += np.abs(_ridge_predict(beta, Zva, z0) - rva).mean()
        losses /= N_INNER
        k = int(np.flatnonzero(losses == losses.min())[-1])  # larger lambda on ties
        chosen = cfg.ridge_grid[k]
        info["inner_l1"] = losses.tolist()
    elif head in ("tree", "statnn"):
        grid = sorted(cfg.tree_grid)
        losses = np.zeros(len(grid))
        for Ztr, rtr, Zva, rva, z0 in folds:
            preds = np.zeros((len(grid), len(rva)))
            for s in cfg.seeds:
                m = _tree_fit(s, max(grid), cfg, objective, Ztr, rtr, z0, anchor=not statnn)
                for g, nt in enumerate(grid):
                    preds[g] += _tree_predict(m, Zva, z0, nt, not statnn) / len(cfg.seeds)
            losses += np.abs(preds - rva[None, :]).mean(1)
        losses /= N_INNER
        chosen = grid[int(np.flatnonzero(losses == losses.min())[0])]  # smaller count on ties
        info["inner_l1"] = losses.tolist()
    else:
        raise ValueError(head)
    info["chosen"] = chosen

    # --- refit on the whole training split
    r = y[train] - base_pred(train)
    if statnn:
        Ztr, Zte, z0 = X[train], X[test], x0
    else:
        T = InputTransform().fit(X[train])
        Ztr, Zte, z0 = T.transform(X[train]), T.transform(X[test]), T.transform(x0[None, :])[0]
    corr = np.zeros(len(test))
    at_zero = 0.0
    for s in cfg.seeds:
        if head == "mlp":
            model, sd = _train_mlp(Ztr, r, z0, s, cfg, epochs=chosen[s])
            corr += _mlp_predict(model, sd, Zte, z0)
            at_zero += float(_mlp_predict(model, sd, z0[None, :], z0)[0])
        elif head == "ridge":
            beta = _ridge_fit(Ztr, r, chosen, z0)
            corr += _ridge_predict(beta, Zte, z0)
            at_zero += float(_ridge_predict(beta, z0[None, :], z0)[0])
            break  # deterministic: one fit (seed averaging is the identity)
        else:
            m = _tree_fit(s, chosen, cfg, objective, Ztr, r, z0, anchor=not statnn)
            corr += _tree_predict(m, Zte, z0, chosen, not statnn)
            if not statnn:
                at_zero += float(_tree_predict(m, z0[None, :], z0, chosen)[0])
    n_fits = 1 if head == "ridge" else len(cfg.seeds)
    info["correction_at_zero"] = at_zero / n_fits if not statnn else None
    return base_pred(test) + corr / n_fits, info


# ------------------------------------------------------------------------------------------ runners


def eligible_rows(data: Dataset) -> np.ndarray:
    """A row is eligible for every arm iff its outcome, a, q and every input row are finite."""
    ok = np.isfinite(data.y) & np.isfinite(data.a) & np.isfinite(data.q)
    for X, _ in data.inputs.values():
        ok &= np.all(np.isfinite(np.asarray(X, dtype=np.float64)), axis=1)
    return ok


def run_cv(arms, data: Dataset, fold_of_row, cfg: FitConfig, heads=None, record=None):
    """Out-of-fold predictions of every arm on identical eligible rows and identical outer folds (NaN elsewhere)."""
    fold_of_row = np.asarray(fold_of_row)
    elig = eligible_rows(data)
    out = {a.name: np.full(len(data.y), np.nan) for a in arms}
    for f in sorted(set(fold_of_row[elig].tolist())):
        test = np.flatnonzero(elig & (fold_of_row == f))
        train = np.flatnonzero(elig & (fold_of_row != f))
        if set(data.comps[test]) & set(data.comps[train]):
            raise ValueError("outer folds are not component-disjoint")
        for arm in arms:
            head = (heads or {}).get(arm.name)
            p, info = fit_predict_arm(arm, data, train, test, cfg, head=head)
            out[arm.name][test] = p
            if record is not None:
                record.setdefault(arm.name, {})[int(f)] = info
    return out


def select_head_class(data: Dataset, rows, inputs, cfg, candidates=HEAD_CLASSES, base="f3u"):
    """The development head class, chosen once by grouped CV over ``rows`` (the four lol-inner component folds as
    held-out folds, each candidate's own hyperparameters nested inside): lowest pooled mean |error|; ties keep the
    earlier candidate."""
    rows = np.asarray(rows)
    sub = subset(data, rows)
    fold = inner_fold_of_rows(sub.comps)
    table = {}
    for c in candidates:
        oof = run_cv([ArmSpec(c, base=base, inputs=inputs, head=c)], sub, fold, cfg)[c]
        table[c] = float(np.nanmean(np.abs(sub.y - oof)))
    best = min(table.values())
    return next(c for c in candidates if table[c] == best), table


def subset(data: Dataset, rows) -> Dataset:
    rows = np.asarray(rows)
    return Dataset(
        row_ids=data.row_ids[rows],
        comps=data.comps[rows],
        y=data.y[rows],
        a=data.a[rows],
        q=data.q[rows],
        inputs={k: (np.asarray(X)[rows], x0) for k, (X, x0) in data.inputs.items()},
    )


def shuffle_outcomes(data: Dataset, seed: int) -> Dataset:
    """The shuffled-outcome control: outcomes permuted across rows (numpy PCG64 ``seed``; never the identity)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data.y))
    while np.array_equal(perm, np.arange(len(perm))):
        perm = rng.permutation(len(data.y))
    return dataclasses.replace(data, y=data.y[perm].copy())
