"""P3 baselines on pooled features, evaluated with group-held-out folds.

    python -m wsp.predictor.baselines --features data/features/<zoo> --evals data/evals/<zoo> \
        --split family|seed|trajectory --out data/runs/p3/<name> [--zoo data/zoo/<zoo>]

Baselines (one model per target, out-of-fold predictions pooled over folds):
  delta_norm_ridge  ridge on [‖Δ_eff‖_F, ‖Δ_eff‖_F / ‖W_changed‖_F]          (2 features)
  statnn            LightGBM on the pooled φ_stats vector                      (pooled stats, 1098 features)
  mlp_svd           MLP on the pooled φ_svd singular spectrum                  (pooled spectrum, 120 features)

Metrics per target: MAE, RMSE, Spearman ρ, each with a seeded 95 % percentile-bootstrap CI over edits.
Targets come from ``<evals>/<id>/eval.json`` (flat ``{name: value}`` or under ``"metrics"``); entries
without one are skipped and counted. ``--delta-from-base`` turns absolute scores into Δy using
``<evals>/base/eval.json`` and fails if a target has no base value.
Trajectory splits group by the full ancestry chain read from the zoo's ``manifest.jsonl`` (``--zoo``,
default the zoo recorded in the features manifest), not only from edits that have targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file
from scipy.stats import rankdata

from wsp.predictor.splits import make_splits

log = logging.getLogger("wsp.predictor.baselines")

MODELS = ("ridge", "statnn", "mlp")
# baseline name -> (model, feature-block)
BASELINES = {
    "delta_norm_ridge": ("ridge", "delta_norm"),
    "statnn": ("statnn", "statnn"),
    "mlp_svd": ("mlp", "mlp_svd"),
}


def _make_model(model: str, seed: int, params: Mapping[str, Any] | None = None):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(params or {})
    if model == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13), **params))
    if model == "statnn":
        from lightgbm import LGBMRegressor

        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 8,
            "min_child_samples": 5,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.5,
            "reg_lambda": 1.0,
        }
        return LGBMRegressor(
            **{**defaults, **params},
            random_state=seed,
            n_jobs=2,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
    if model == "mlp":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.neural_network import MLPRegressor

        defaults = {"hidden_layer_sizes": (64, 64), "alpha": 1e-3, "max_iter": 1000}
        mlp = MLPRegressor(**{**defaults, **params}, random_state=seed)
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), mlp), transformer=StandardScaler()
        )
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    if np.ptp(y) == 0 or np.ptp(p) == 0:
        return math.nan
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


EDIT_CI_METHOD = "seeded percentile bootstrap over edits (not independent trajectories)"


def metrics_with_ci(y: np.ndarray, pred: np.ndarray, seed: int = 0, n_boot: int = 1000) -> dict:
    """MAE / RMSE / Spearman with 95 % percentile bootstrap CIs (resampling edits, seeded)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    n = len(y)
    err = pred - y
    point = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "spearman": _spearman(y, pred),
    }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = {
        "mae": np.abs(err)[idx].mean(1),
        "rmse": np.sqrt((err[idx] ** 2).mean(1)),
        "spearman": np.array([_spearman(y[i], pred[i]) for i in idx]),
    }
    out: dict[str, Any] = {"n": n}
    for k, v in point.items():
        finite = boots[k][np.isfinite(boots[k])]
        ci = (
            [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
            if len(finite) >= max(10, n_boot // 10)
            else [math.nan, math.nan]
        )
        out[k] = {"value": v, "ci95": ci, "ci_method": EDIT_CI_METHOD}
    return out


def fit_eval(
    X: np.ndarray,
    Y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    model: str = "ridge",
    *,
    target_names: Sequence[str] | None = None,
    seed: int = 0,
    n_boot: int = 1000,
    model_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Out-of-fold evaluation of ``model`` for every column of Y (NaN = target missing for that edit)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    names = list(target_names) if target_names is not None else [f"y{j}" for j in range(Y.shape[1])]
    if len(names) != Y.shape[1]:
        raise ValueError("target_names must match the columns of Y")
    result: dict[str, Any] = {
        "model": model,
        "n": int(X.shape[0]),
        "n_folds": len(folds),
        "n_features": int(X.shape[1]),
        "targets": {},
        "skipped_folds": [],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn ConvergenceWarning on tiny folds
        for j, name in enumerate(names):
            pred = np.full(len(X), np.nan)
            for fold, (train, test) in enumerate(folds):
                train = np.asarray(train)[np.isfinite(Y[np.asarray(train), j])]
                if len(train) < 2:
                    reason = "fewer than two finite training labels"
                    result["skipped_folds"].append(
                        {"fold": fold, "target": name, "skipped_reason": reason}
                    )
                    log.warning("Skipping baseline fold %d target %s: %s", fold, name, reason)
                    continue
                est = _make_model(model, seed, model_params).fit(X[train], Y[train, j])
                pred[test] = est.predict(X[test])
            ok = np.isfinite(Y[:, j]) & np.isfinite(pred)
            if ok.sum() < 2:
                result["targets"][name] = {"n": int(ok.sum())}
                continue
            result["targets"][name] = metrics_with_ci(Y[ok, j], pred[ok], seed, n_boot)
    return result


def _read_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    for key in ("metrics", "summary"):
        if isinstance(data.get(key), dict):
            data = data[key]
            break
    return {
        k: float(v)
        for k, v in data.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }


def load_targets(
    evals_dir: str | Path, ids: Sequence[str]
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """``({id: {target: value}}, sorted target names)``; entries without a usable eval.json are skipped."""
    evals_dir = Path(evals_dir)
    targets: dict[str, dict[str, float]] = {}
    for i in ids:
        f = evals_dir / i / "eval.json"
        if f.exists():
            metrics = _read_metrics(f)
            if metrics:
                targets[i] = metrics
    log.info(
        "targets: %d of %d entries have eval.json (%d skipped)",
        len(targets),
        len(ids),
        len(ids) - len(targets),
    )
    return targets, sorted({k for t in targets.values() for k in t})


def load_zoo_ancestry(zoo: str | Path) -> dict[str, str | None]:
    """``{id: parent_id}`` for every entry of the zoo manifest, whether or not it is evaluated."""
    path = Path(zoo) / "manifest.jsonl"
    if not path.exists():
        raise ValueError(f"trajectory split needs the zoo manifest ({path} not found); pass --zoo")
    parents: dict[str, str | None] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.warning("%s:%d is not valid JSON; ignored for ancestry", path, n)
            continue
        if not isinstance(row, dict) or "id" not in row:
            continue  # header rows
        parent = row.get("parent_id")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"{path}:{n} has invalid parent_id {parent!r}")
        if parents.get(row["id"], parent) != parent:
            raise ValueError(f"{path}:{n}: conflicting parent_id for {row['id']}")
        parents[row["id"]] = parent
    return parents


def load_feature_table(features_dir: str | Path) -> list[dict[str, Any]]:
    """One row per extracted entry: metadata plus the three baseline feature blocks."""
    features_dir = Path(features_dir)
    rows = []
    for meta_path in sorted(features_dir.glob("*/features.json")):
        meta = json.loads(meta_path.read_text())
        pooled = load_file(meta_path.parent / "pooled.safetensors")
        norms = meta["norms"]
        rows.append(
            {
                "id": meta["edit_id"],
                "family": meta["family"],
                "seed": meta["seed"],
                "parent_id": meta["parent_id"],
                "delta_norm": np.array([norms["fro_effective"], norms["rel_fro_effective"]]),
                "statnn": pooled["stats"].numpy(),
                "mlp_svd": pooled["svd"].numpy(),
            }
        )
    return rows


def _fmt(m: Mapping[str, Any] | None, digits: int = 3) -> str:
    if not m:
        return "n/a"
    lo, hi = m["ci95"]
    return f"{m['value']:.{digits}g} [{lo:.{digits}g}, {hi:.{digits}g}]"


def results_markdown(res: Mapping[str, Any], run_id: str) -> str:
    lines = [
        f"## P3 baselines — split by {res['split']} (run `{run_id}`)",
        "",
        (
            f"Edits with targets: {res['n_edits']} of {res['n_entries']} extracted; "
            f"{res['n_splits']}-fold group split; bootstrap 95 % CI over edits "
            f"({res['n_boot']} resamples, seed {res['seed']})."
        ),
        "",
        "| model | target | n | MAE [95 % CI] | RMSE [95 % CI] | Spearman ρ [95 % CI] |",
        "|---|---|---:|---|---|---|",
    ]
    for model, r in res["models"].items():
        for target, m in r["targets"].items():
            lines.append(
                f"| {model} | {target} | {m['n']} | {_fmt(m.get('mae'))} | "
                f"{_fmt(m.get('rmse'))} | {_fmt(m.get('spearman'))} |"
            )
    return "\n".join(lines) + "\n"


def run_baselines(
    features_dir: str | Path,
    evals_dir: str | Path,
    split: str,
    out: str | Path,
    *,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
    delta_from_base: bool = False,
    models: Sequence[str] | None = None,
    zoo: str | Path | None = None,
) -> dict[str, Any]:
    features_dir, evals_dir, out = Path(features_dir), Path(evals_dir), Path(out)
    table = load_feature_table(features_dir)
    if not table:
        raise ValueError(f"no extracted entries (features.json) under {features_dir}")
    fm = features_dir / "manifest.json"
    feat_manifest = json.loads(fm.read_text()) if fm.exists() else {}
    ancestry = None
    if split == "trajectory":
        zoo = zoo or feat_manifest.get("zoo")
        if not zoo:
            raise ValueError("trajectory split needs the zoo manifest; pass --zoo")
        ancestry = load_zoo_ancestry(zoo)  # full chains, independent of which entries have evals
    targets, names = load_targets(evals_dir, [r["id"] for r in table])
    if not targets:
        raise ValueError(f"no eval.json found under {evals_dir} for any extracted entry")
    if delta_from_base:
        base_file = evals_dir / "base" / "eval.json"
        if not base_file.exists():
            raise ValueError(f"--delta-from-base needs {base_file}")
        base = _read_metrics(base_file)
        missing = sorted({k for t in targets.values() for k in t} - set(base))
        if missing:
            raise ValueError(f"--delta-from-base: {base_file} has no value for targets {missing}")
        targets = {i: {k: v - base[k] for k, v in t.items()} for i, t in targets.items()}
    rows = [r for r in table if r["id"] in targets]
    Y = np.array([[targets[r["id"]].get(n, np.nan) for n in names] for r in rows])
    if split != "trajectory" and any(r[split] is None for r in rows):
        raise ValueError(f"some entries have no {split!r}")
    folds_n = n_splits
    while True:  # fewer groups than folds (e.g. 5 families, 6 splits) -> shrink and say so
        try:
            folds = make_splits(rows, split, n_splits=folds_n, ancestry=ancestry)
            break
        except ValueError as e:
            if "number of groups" not in str(e) or folds_n <= 2:
                raise ValueError(f"cannot make a {split} split of {len(rows)} edits: {e}") from e
            folds_n -= 1
    if folds_n < n_splits:
        log.warning(
            "too few groups for split=%s; using %d folds instead of %d", split, folds_n, n_splits
        )
    results: dict[str, Any] = {}
    dims: dict[str, int] = {}
    for key, (model, block) in BASELINES.items():
        if models is not None and key not in models:
            continue
        X = np.stack([r[block] for r in rows])
        dims[block] = int(X.shape[1])
        results[key] = fit_eval(X, Y, folds, model, target_names=names, seed=seed, n_boot=n_boot)
        log.info("%s done", key)
    res = {
        "split": split,
        "n_edits": len(rows),
        "n_entries": len(table),
        "n_splits": folds_n,
        "n_boot": n_boot,
        "seed": seed,
        "delta_from_base": delta_from_base,
        "target_names": names,
        "feature_dims": dims,
        "families": sorted({r["family"] for r in rows}),
        "folds": [sorted(rows[int(i)]["id"] for i in test) for _train, test in folds],
        "models": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    (out / "results.md").write_text(results_markdown(res, out.name))
    from wsp.features.extract import _git_sha
    from wsp.schema import write_manifest

    cfg = {k: res[k] for k in ("split", "n_splits", "n_boot", "seed", "delta_from_base")}
    write_manifest(
        out / "manifest.json",
        run_id=out.name,
        seed=seed,
        config=cfg,
        config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        git_sha=_git_sha(),
        features=str(features_dir),
        features_config_hash=feat_manifest.get("config_hash"),
        features_git_sha=feat_manifest.get("git_sha"),
        evals=str(evals_dir),
        base_commit=feat_manifest.get("base_commit"),
    )
    return res


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--features", required=True)
    p.add_argument("--evals", required=True)
    p.add_argument("--split", required=True, choices=["family", "seed", "trajectory"])
    p.add_argument("--out", required=True)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delta-from-base", action="store_true")
    p.add_argument("--zoo", help="zoo dir whose manifest.jsonl gives ancestry (trajectory split)")
    p.add_argument("--models", help=f"comma-separated subset of {sorted(BASELINES)}")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_baselines(
            args.features,
            args.evals,
            args.split,
            args.out,
            n_splits=args.n_splits,
            n_boot=args.n_boot,
            seed=args.seed,
            delta_from_base=args.delta_from_base,
            models=args.models.split(",") if args.models else None,
            zoo=args.zoo,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(Path(args.out) / "results.md")


if __name__ == "__main__":
    main()
