"""Fold-local controls and conditional paired ancestry-cluster uncertainty for P4."""

import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wsp.predictor.baselines import _make_model, _spearman, metrics_with_ci
from wsp.predictor.models import effective_rank
from wsp.predictor.splits import trajectory_groups


def collapse_check(latents, ablation):
    rank = float(latents) if np.isscalar(latents) else effective_rank(latents)
    standard = float(ablation) if np.isscalar(ablation) else effective_rank(ablation)
    return {
        "erank": rank,
        "frozen_predictor_erank": standard,
        "n_representations": None
        if np.isscalar(latents)
        else int(latents.reshape(-1, latents.shape[-1]).shape[0]),
        "threshold": 2,
        "required_rank_ratio": 2,
        "collapsed": rank < 2,
        "acceptance_gate": rank >= 2 and rank >= 2 * standard,
    }


PAIRED_CI_METHOD = (
    "paired percentile bootstrap resampling independent full-ancestry clusters with replacement, "
    "retaining all edits and paired predictions in each sampled cluster; conditional on fitted models"
)


def paired_cluster_difference(y, model, baseline, clusters, seed, n_boot):
    y, model, baseline, clusters = map(np.asarray, (y, model, baseline, clusters))
    groups = np.unique(clusters)
    members = [np.flatnonzero(clusters == g) for g in groups]
    rng = np.random.default_rng(seed)

    def difference(idx):
        a, b, obs = model[idx], baseline[idx], y[idx]
        return [
            np.abs(a - obs).mean() - np.abs(b - obs).mean(),
            np.sqrt(np.mean((a - obs) ** 2)) - np.sqrt(np.mean((b - obs) ** 2)),
            _spearman(obs, a) - _spearman(obs, b),
        ]

    point = difference(np.arange(len(y)))
    draws = np.array(
        [
            difference(
                np.concatenate([members[i] for i in rng.integers(len(groups), size=len(groups))])
            )
            for _ in range(n_boot)
        ]
    )
    out = {"n": len(y), "n_clusters": len(groups)}
    for j, name in enumerate(("mae", "rmse", "spearman")):
        finite = draws[:, j][np.isfinite(draws[:, j])]
        out[name] = {
            "value": float(point[j]),
            "ci_method": PAIRED_CI_METHOD,
            "ci95": np.percentile(finite, [2.5, 97.5]).tolist()
            if len(finite) >= 2 and len(groups) >= 2
            else [float("nan")] * 2,
        }
    return out


def audit_run(ds, predictions, folds, records, out, seed, n_boot, scaling):
    from wsp.predictor.train import Normalizer

    y = ds.targets.numpy()
    clusters = np.asarray(trajectory_groups(ds.rows, ds.ancestry))
    families = np.asarray([r["family"] for r in ds.rows])
    stats = np.stack([ds.entry(r["id"])["stats"].numpy() for r in ds.rows])
    descriptors = np.stack([ds.entry(r["id"])["descriptor"].numpy() for r in ds.rows])
    norms = np.asarray([ds.metadata[r["id"]]["norms"]["fro_effective"] for r in ds.rows])
    devices = np.asarray([r["eval_device_id"] for r in ds.rows])
    controls = {
        k: np.full_like(predictions, np.nan)
        for k in ("statnn", "fold_mean", "delta_norm", "descriptor_only", "device_only")
    }
    support, per_family = {}, {}
    for f, (train, test) in enumerate(folds):
        norm = Normalizer(ds, train, scaling)
        support[str(f)] = {}
        for key, (_, _, constant, low, high) in norm.state.items():
            values = [ds.entry(ds.rows[i]["id"])[key] for i in test]
            import torch

            x = norm.log(torch.cat(values) if key == "tokens" else torch.stack(values))
            support[str(f)][key] = {
                "fraction_outside_training_range": float(((x < low) | (x > high)).float().mean()),
                "constant_training_coordinates": int(constant.sum()),
            }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(y.shape[1]):
                fit = np.asarray(train)[np.isfinite(y[train, j])]
                if not len(fit):
                    continue
                controls["fold_mean"][test, j] = y[fit, j].mean()
                if len(fit) >= 2:
                    est = _make_model("statnn", seed + f).fit(stats[fit], y[fit, j])
                    controls["statnn"][test, j] = est.predict(stats[test])
                    # Direction chosen from training labels only, clipped beyond training support.
                    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
                        norms[fit], y[fit, j]
                    )
                    controls["delta_norm"][test, j] = iso.predict(norms[test])
                    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                        descriptors[fit], y[fit, j]
                    )
                    controls["descriptor_only"][test, j] = reg.predict(descriptors[test])
                else:
                    for key in ("statnn", "delta_norm", "descriptor_only"):
                        controls[key][test, j] = y[fit, j].mean()
                # Categorical training-device means; unseen devices use training-fold mean.
                for i in test:
                    matched = fit[devices[fit] == devices[i]]
                    controls["device_only"][i, j] = y[matched if len(matched) else fit, j].mean()
        frame = pd.read_parquet(out / f"fold-{f}.parquet")
        row_index = {ds.rows[i]["id"]: i for i in test}
        target_index = {name: j for j, name in enumerate(ds.names)}
        frame["family"] = [ds.rows[row_index[i]]["family"] for i in frame.id]
        frame["ancestry_cluster"] = [clusters[row_index[i]] for i in frame.id]
        for key, values in controls.items():
            frame[key] = [
                float(values[row_index[i], target_index[t]])
                for i, t in zip(frame.id, frame.target, strict=True)
            ]
        frame.to_parquet(out / f"fold-{f}.parquet", index=False)
        per_family[str(f)] = {}
        for family in sorted(set(families[test])):
            subset = np.asarray(test)[families[test] == family]
            per_family[str(f)][family] = {
                label: score_targets(y, pred, subset, ds.names, seed, n_boot)
                for label, pred in {"model": predictions, **controls}.items()
            }
    baselines, paired, centered, centered_pooled = {}, {}, {}, {}
    for label, pred in {"model": predictions, **controls}.items():
        if label != "model":
            baselines[label] = {
                "targets": score_targets(y, pred, np.arange(len(y)), ds.names, seed, n_boot)
            }
            paired[label] = {}
        centered[label] = {}
        centered_pooled[label] = {}
        for j, name in enumerate(ds.names):
            ok = np.isfinite(y[:, j]) & np.isfinite(pred[:, j]) & np.isfinite(predictions[:, j])
            if not ok.any():
                continue
            within, pooled = family_centered_correlations(
                y[:, j],
                pred[:, j],
                families,
                [np.asarray(test)[ok[test]] for _, test in folds],
            )
            centered[label][name] = within
            centered_pooled[label][name] = pooled
            if label != "model":
                paired[label][name] = paired_cluster_difference(
                    y[ok, j], predictions[ok, j], pred[ok, j], clusters[ok], seed, n_boot
                )
    return {
        "baselines": baselines,
        "paired_differences": paired,
        "per_family": per_family,
        "family_centered": centered,
        "family_centered_pooled": centered_pooled,
        "out_of_support": support,
    }


def family_centered_correlations(y, pred, families, folds):
    """Centre within family × test fold; average finite fold correlations by fold n."""

    def score(indices):
        obs, p, fam = y[indices].copy(), pred[indices].copy(), families[indices]
        for family in set(fam):
            mask = fam == family
            obs[mask] -= obs[mask].mean()
            p[mask] -= p[mask].mean()
        return {"n": len(indices), "spearman": _spearman(obs, p)}

    fold_scores = [score(indices) for indices in folds if len(indices)]
    finite = [r for r in fold_scores if np.isfinite(r["spearman"])]
    within = {
        "n": sum(r["n"] for r in fold_scores),
        "spearman": float(
            np.average([r["spearman"] for r in finite], weights=[r["n"] for r in finite])
        )
        if finite
        else float("nan"),
        "folds": fold_scores,
        "estimand": "family centring within outer test folds; fold-n weighted correlation",
        "diagnostic_only": True,
    }
    pooled = score(np.concatenate(folds))
    pooled.update(estimand="family centring after pooling outer test folds", diagnostic_only=True)
    return within, pooled


def score_targets(y, pred, indices, names, seed, n_boot):
    result = {}
    for j, name in enumerate(names):
        ok = indices[np.isfinite(y[indices, j]) & np.isfinite(pred[indices, j])]
        result[name] = metrics_with_ci(y[ok, j], pred[ok, j], seed, n_boot) if len(ok) else {"n": 0}
    return result


def seed_summary(results, seeds):
    summary = {}
    for target in results[0]["targets"]:
        summary[target] = {}
        for metric in ("mae", "rmse", "spearman"):
            values = np.asarray(
                [r["targets"][target].get(metric, {}).get("value", np.nan) for r in results]
            )
            values = values[np.isfinite(values)]
            summary[target][metric] = {
                "mean": float(values.mean()) if len(values) else float("nan"),
                "range": [float(values.min()), float(values.max())]
                if len(values)
                else [float("nan")] * 2,
                "n_seeds": len(values),
            }
    return {
        "seeds": seeds,
        "seed_summary": summary,
        "seed_range_definition": "minimum and maximum across complete full experiment seeds; not a confidence interval",
        "m6_all_seeds_collapse_gate": all(r["m6_collapse_gate"] for r in results)
        if results[0]["m6_collapse_gate"] is not None
        else None,
    }
