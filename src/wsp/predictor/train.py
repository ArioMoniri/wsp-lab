"""CPU/CUDA P4 training with nested group validation and genuine targets only.

Default targets are per-step y_t - y_predecessor. --target level fits absolute rates.
PPL/KL are excluded. --selftest writes explicitly synthetic artifacts outside results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from wsp.predictor.audit import PAIRED_CI_METHOD, audit_run, collapse_check, seed_summary
from wsp.predictor.baselines import metrics_with_ci
from wsp.predictor.data import FeatureDataset, collate_trajectories
from wsp.predictor.models import build_model, effective_rank, seed_all

log = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/p4_predictors.json"


class Normalizer:
    """Signed log1p then training-only standardization, persisted for inference."""

    keys = ("tokens", "pooled", "stats")

    def __init__(self, ds, indices, scaling="standard"):
        self.scaling = scaling
        ids = sorted({i for idx in indices for i in ds.chain(ds.rows[idx]["id"])})
        self.ids = ids
        self.state = {}
        for key in self.keys:
            values = [ds.entry(i)[key] for i in ids]
            x = torch.cat(values) if key == "tokens" else torch.stack(values)
            x = self.log(x)
            if not torch.isfinite(x).all():
                raise ValueError(f"nonfinite {key} features")
            constant = x.amax(0) == x.amin(0)
            center = x.median(0).values if scaling == "robust" else x.mean(0)
            scale = (
                torch.quantile(x, 0.75, dim=0) - torch.quantile(x, 0.25, dim=0)
                if scaling == "robust"
                else x.std(0, unbiased=False)
            )
            self.state[key] = (center, scale.clamp_min(1e-4), constant, x.amin(0), x.amax(0))

    @staticmethod
    def log(x):
        return x.sign() * x.abs().log1p()

    def __call__(self, batch, device):
        b = {k: v.to(device) for k, v in batch.items()}
        for key, (mean, std, constant, _, _) in self.state.items():
            b[key] = ((self.log(b[key]) - mean.to(device)) / std.to(device)).clamp(-20, 20)
            b[key][..., constant.to(device)] = 0
        return b


def loader(ds, indices, batch_size, seed, shuffle=False):
    return DataLoader(
        Subset(ds, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_trajectories,
        generator=torch.Generator().manual_seed(seed),
    )


def last_predictions(model, batch):
    p = model(batch)
    return p[torch.arange(len(p), device=p.device), batch["mask"].sum(1) - 1]


def masked_loss(pred, target):
    valid = torch.isfinite(target)
    return (pred[valid] - target[valid]).square().mean()


def predict(model, ds, indices, normalizer, device, batch_size):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader(ds, indices, batch_size, 0):
            b = normalizer(batch, device)
            output.append(last_predictions(model, b).cpu())
    return torch.cat(output).numpy()


def run_training(
    features,
    evals,
    split,
    out,
    *,
    model="m1",
    epochs=100,
    pretrain_epochs=30,
    patience=10,
    batch_size=16,
    lr=1e-3,
    seed=0,
    n_splits=5,
    n_boot=1000,
    d_model=256,
    device="cpu",
    target="delta",
    synthetic=False,
    targets=None,
    zoo=None,
    holdout=None,
    holdout_variant="purge",
    features_variant="cumulative",
    scaling="standard",
):
    if min(epochs, patience, batch_size, n_boot) < 1 or pretrain_epochs < 1:
        raise ValueError(
            "epochs, pretrain epochs, patience, batch size and bootstrap count must be positive"
        )
    seed_all(seed)
    ds = FeatureDataset(
        features, evals, target=target, targets=targets, zoo=zoo, features_variant=features_variant
    )
    folds = ds.holdout(holdout, holdout_variant) if holdout else ds.folds(split, n_splits)
    # Plan all validation sets before writing anything. A two-group dataset cannot
    # honestly support both a held-out group and an inner group for early stopping.
    plans = []
    for i, (train, test) in enumerate(folds):
        inner = ds.folds(split, min(3, n_splits), train, scope=f"inner:{i}")
        if not inner:
            log.warning("Skipping outer fold %d: no inner folds survive ancestry filtering", i)
            continue
        plans.append((train, inner, test))
    if not plans:
        raise ValueError("no folds survive ancestry filtering")
    config = {
        "features_variant": features_variant,
        "scaling": scaling,
        "selection_protocol": "all surviving inner folds; median best epoch (rounded up); fresh outer-train refit for fixed epoch budget",
        "holdout": holdout,
        "holdout_variant": holdout_variant,
        "model": model,
        "features": str(features),
        "evals": str(evals),
        "split": split,
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_splits": len(folds),
        "n_boot": n_boot,
        "d_model": d_model,
        "device": device,
        "target": target,
        "synthetic": synthetic,
        "targets": ds.names,
        "zoo": str(zoo) if zoo else ds.manifest.get("zoo"),
    }
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "zoo-manifest.snapshot.jsonl").write_bytes(ds.manifest_snapshot)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    planned = []
    ids_for = lambda indices: [ds.rows[i]["id"] for i in indices]
    for train, inner, test in plans:
        planned.append(
            {
                "outer_train_ids": ids_for(train),
                "outer_test_ids": ids_for(test),
                "inner": [
                    {
                        "train_ids": ids_for(a),
                        "validation_ids": ids_for(b),
                        "train_target_variance": np.nanvar(ds.targets[a].numpy(), axis=0).tolist(),
                        "validation_target_variance": np.nanvar(
                            ds.targets[b].numpy(), axis=0
                        ).tolist(),
                    }
                    for a, b in inner
                ],
            }
        )
    (out / "split-plan.json").write_text(
        json.dumps(
            {
                "protocol": config["selection_protocol"],
                "folds": planned,
                "split_audit": ds.split_audit,
            },
            indent=2,
        )
    )
    predictions = np.full(ds.targets.shape, np.nan)
    fold_records, backends, baseline_folds = [], [], []

    def fit_candidate(train, val, budget, fold_seed):
        seed_all(fold_seed)
        net = build_model(model, ds, seed=fold_seed, d_model=d_model, device=device)
        normalizer = Normalizer(ds, train, scaling)
        pretrain_history = []
        collapse = None
        ablation = None
        if model == "m6":
            # Same initial encoder; frozen standard linear predictor ablation.
            ablation = copy.deepcopy(net)
            dz = ablation.predictor.delta.in_features
            ablation.predictor = torch.nn.Linear(dz, dz).to(device)
            ablation.predictor.requires_grad_(False)
            ablation_opt = torch.optim.AdamW(
                [p for p in ablation.parameters() if p.requires_grad], lr=lr
            )

        if model in {"m3", "m6"}:
            opt = torch.optim.AdamW(net.parameters(), lr=lr)
            for epoch in range(pretrain_epochs):
                seed_all(fold_seed + epoch)
                net.train()
                losses, latents = [], []
                for batch in loader(ds, train, batch_size, fold_seed + epoch, True):
                    b = normalizer(batch, device)
                    if model == "m3" and not (b["mask"][:, 1:] & b["mask"][:, :-1]).any():
                        continue
                    opt.zero_grad()
                    loss, z = net.pretrain_loss(b)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite JEPA loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                    opt.step()
                    losses.append(loss.item())
                    latents.append(z.detach().cpu())
                if not losses:
                    raise ValueError(
                        "M3 training fold has no parent-child pairs; use trajectory splits or more chains"
                    )
                pretrain_history.append(
                    {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        "erank": effective_rank(torch.cat(latents)),
                    }
                )
            if ablation is not None:
                # Train the ablation with identical batches/epochs and RNG streams.
                for epoch in range(pretrain_epochs):
                    seed_all(fold_seed + epoch)
                    for batch in loader(ds, train, batch_size, fold_seed + epoch, True):
                        b = normalizer(batch, device)
                        ablation_opt.zero_grad()
                        loss, _ = ablation.pretrain_loss(b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(ablation.parameters(), 1)
                        ablation_opt.step()

                def rank_of(module, train=train, normalizer=normalizer):
                    module.eval()
                    with torch.no_grad():
                        zs = []
                        for batch in loader(ds, train, batch_size, 0):
                            b = normalizer(batch, device)
                            zs.append(module.encode(b)[b["mask"]])
                    return torch.cat(zs)

                collapse = collapse_check(rank_of(net), rank_of(ablation))
                log.info("M6 collapse inputs: %s", collapse)
            net.freeze_encoder()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
        best, stale, best_state, history = float("inf"), 0, None, []
        for epoch in range(budget):
            net.train()
            losses = []
            for batch in loader(ds, train, batch_size, fold_seed + epoch, True):
                b = normalizer(batch, device)
                opt.zero_grad()
                loss = masked_loss(last_predictions(net, b), b["target"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite supervised loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                opt.step()
                losses.append(loss.item())
            val_loss = (
                float(
                    np.nanmean(
                        (
                            predict(net, ds, val, normalizer, device, batch_size)
                            - ds.targets[val].numpy()
                        )
                        ** 2
                    )
                )
                if val is not None
                else float(np.mean(losses))
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": val_loss if val is not None else None,
                }
            )
            if val_loss < best:
                best, stale = val_loss, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                stale += 1
            if val is not None and stale >= patience:
                break
        if best_state is None:
            raise ValueError("validation fold has no finite targets")
        if val is not None:
            net.load_state_dict(best_state)

        return net, normalizer, history, pretrain_history, collapse, best

    for fold, (train, inner, test) in enumerate(plans):
        selected = []
        for inner_index, (inner_train, inner_val) in enumerate(inner):
            _, inner_norm, h, _, _, _ = fit_candidate(
                inner_train, inner_val, epochs, seed + fold * 100 + inner_index
            )
            best_epoch = min(h, key=lambda r: r["validation_loss"])["epoch"] + 1
            selected.append(
                {
                    "train_ids": [ds.rows[i]["id"] for i in inner_train],
                    "validation_ids": [ds.rows[i]["id"] for i in inner_val],
                    "normalization_context_ids": inner_norm.ids,
                    "best_epoch": best_epoch,
                    "history": h,
                    "target_variance": {
                        "train": np.nanvar(ds.targets[inner_train].numpy(), axis=0).tolist(),
                        "validation": np.nanvar(ds.targets[inner_val].numpy(), axis=0).tolist(),
                    },
                }
            )
        refit_epochs = int(np.ceil(np.median([r["best_epoch"] for r in selected])))
        fold_seed = seed + fold
        net, normalizer, history, pretrain_history, collapse, best = fit_candidate(
            train, None, refit_epochs, fold_seed
        )
        val = []
        backends.append(net.backend)
        predictions[test] = predict(net, ds, test, normalizer, device, batch_size)
        frame = []
        for i in test:
            for j, name in enumerate(ds.names):
                frame.append(
                    {
                        "id": ds.rows[i]["id"],
                        "fold": fold,
                        "target": name,
                        "observed": float(ds.targets[i, j]),
                        "predicted": float(predictions[i, j]),
                        "trajectory_length": len(ds.chain(ds.rows[i]["id"])),
                    }
                )
        pd.DataFrame(frame).to_parquet(out / f"fold-{fold}.parquet", index=False)
        torch.save(
            {"model": net.state_dict(), "normalization": normalizer.state, "config": config},
            out / f"fold-{fold}.pt",
        )
        ids = lambda indices: [ds.rows[i]["id"] for i in indices]
        fold_records.append(
            {
                "fold": fold,
                "inner_selection": selected,
                "refit_ids": ids(train),
                "refit_epochs": refit_epochs,
                "target_variance": {"train": np.nanvar(ds.targets[train].numpy(), axis=0).tolist()},
                "seed": fold_seed,
                "train_ids": ids(train),
                "validation_ids": ids(val),
                "test_ids": ids(test),
                "test_families": sorted({ds.rows[i]["family"] for i in test}),
                "normalization_context_ids": normalizer.ids,
                "pretraining": pretrain_history,
                "collapse_check": collapse,
                "history": history,
                "parameter_count": sum(p.numel() for p in net.parameters()),
            }
        )
        baseline_folds.append((train, test))
        log.info("fold %d backend=%s refit training MSE=%g", fold, net.backend, best)
    scores = {}
    long_scores = {}
    long_mask = np.array([len(ds.chain(r["id"])) >= 4 for r in ds.rows])
    for j, name in enumerate(ds.names):
        ok = np.isfinite(ds.targets[:, j].numpy()) & np.isfinite(predictions[:, j])
        scores[name] = metrics_with_ci(ds.targets[ok, j].numpy(), predictions[ok, j], seed, n_boot)
        long_ok = ok & long_mask
        if long_ok.any():
            long_scores[name] = metrics_with_ci(
                ds.targets[long_ok, j].numpy(), predictions[long_ok, j], seed, n_boot
            )
    audit = audit_run(ds, predictions, baseline_folds, fold_records, out, seed, n_boot, scaling)
    baseline = audit["baselines"]["statnn"]
    m6_gate = None
    if model == "m6" and holdout == "lora_sft":
        transfer_pass = False
        for record, (_, test) in zip(fold_records, baseline_folds, strict=True):
            fold_scores = {}
            for j, name in enumerate(ds.names):
                ok = np.isfinite(ds.targets[test, j].numpy()) & np.isfinite(predictions[test, j])
                fold_scores[name] = metrics_with_ci(
                    ds.targets[test, j].numpy()[ok], predictions[test, j][ok], seed, n_boot
                )
            record["transfer_metrics"] = fold_scores
            count = sum(
                m.get("spearman", {}).get("value", float("nan")) >= 0.5
                for m in fold_scores.values()
            )
            record["h3_benchmarks_passing"] = count
            transfer_pass |= count >= 6
        m6_gate = {
            "holdout_family": holdout,
            "h3_transfer_gate": transfer_pass,
            "collapse_gate": all(r["collapse_check"]["acceptance_gate"] for r in fold_records),
            "collapsed_fold_seeds": sum(r["collapse_check"]["collapsed"] for r in fold_records),
        }
        m6_gate["accepted"] = transfer_pass and m6_gate["collapse_gate"]
    stratified = {}
    for semantics in ("sequential", "lora_vs_base", "unknown"):
        mask = np.array(
            [
                ds.relations[r["id"]].get("step_semantics", None) == semantics
                if semantics != "unknown"
                else (r["family"] == "chain" and not ds.relations[r["id"]].get("step_semantics"))
                for r in ds.rows
            ]
        )
        stratified[semantics] = {}
        for j, name in enumerate(ds.names):
            ok = mask & np.isfinite(predictions[:, j]) & np.isfinite(ds.targets[:, j].numpy())
            stratified[semantics][name] = (
                metrics_with_ci(ds.targets[ok, j].numpy(), predictions[ok, j], seed, n_boot)
                if ok.any()
                else {"n": 0}
            )
    results = {
        **audit,
        "config": config,
        "targets": scores,
        "h2_by_step_semantics": stratified,
        "length_ge_4": {
            "n_edits": int((long_mask & np.isfinite(predictions).any(axis=1)).sum()),
            "targets": long_scores,
        },
        "statnn": baseline,
        "folds": fold_records,
        "m6_acceptance": m6_gate,
        "split_audit": ds.split_audit,
        "skipped_folds": [r for r in ds.split_audit if r.get("skipped_reason")],
        "missing_predecessor_targets": ds.missing_predecessor_targets,
        "missing_matched_reference": ds.missing_matched_reference,
        "matched_reference_targets": ds.matched_reference_targets,
        "targets_with_inferred_runtime": ds.targets_with_inferred_runtime,
        "m6_collapse_gate": (
            all(r["collapse_check"]["acceptance_gate"] for r in fold_records)
            if model == "m6"
            else None
        ),
        "marginal_ci_method": "seeded percentile bootstrap resampling edits, conditional on fitted models; descriptive marginal intervals only",
        "ci_method": "seeded percentile bootstrap over edits (not independent trajectories)",
        "paired_ci_method": PAIRED_CI_METHOD,
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    lines = [
        f"# P4 {model} — run `{out.name}`",
        "",
        f"Split: {split}; labeled edits: {len(ds)}.",
        "Marginal target intervals resample edits; paired differences resample full-ancestry clusters.",
        "",
        "| model | target | n | MAE [95% CI] | RMSE [95% CI] | Spearman [95% CI] |",
        "|---|---|---:|---|---|---|",
    ]
    if synthetic:
        lines.insert(1, "SYNTHETIC SELFTEST — no hypothesis evidence.")
    for label, metrics in [
        (model, scores),
        ("statnn", baseline["targets"]),
        *[(f"H2 {key}", value) for key, value in stratified.items()],
    ]:
        for name, m in metrics.items():
            cells = [
                f"{m[k]['value']:.5g} {m[k]['ci95']}" if k in m else "unavailable"
                for k in ("mae", "rmse", "spearman")
            ]
            lines.append(f"| {label} | {name} | {m['n']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Target: {target}; delta = measured entry rate minus measured predecessor rate.",
        "H2 measures history beyond the latest cumulative-from-base snapshot.",
        "JEPA state inputs are cumulative effective-delta features, not absolute weight features.",
        "M5 uses one tensor-attention layer and one causal edit-attention layer; dropout 0.2.",
        "m1_matched widens one M1 feedforward layer; capacity differs from M5 by four parameters.",
        f"M6 acceptance: {m6_gate}.",
    ]
    lines += [
        "",
        "| fold | family | target | predictor | n | MAE | Spearman |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for f, families in audit["per_family"].items():
        for family, predictors in families.items():
            for label, targets_ in predictors.items():
                for name, m in targets_.items():
                    lines.append(
                        f"| {f} | {family} | {name} | {label} | {m['n']} | {m.get('mae', {}).get('value', float('nan')):.5g} | {m.get('spearman', {}).get('value', float('nan')):.5g} |"
                    )
    (out / "results.md").write_text("\n".join(lines) + "\n")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=DEFAULT_CONFIG_PATH.parents[1]
    ).strip()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    input_hash = hashlib.sha256()
    for file in sorted(Path(features).glob("*/features.json")):
        input_hash.update(file.read_bytes())
    source_hash = hashlib.sha256()
    for file in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(file.read_bytes())
    manifest = {
        "zoo_manifest_snapshot": ds.snapshot_info,
        "git_sha": sha,
        "synthetic": synthetic,
        "split_audit": ds.split_audit,
        "skipped_folds": [r for r in ds.split_audit if r.get("skipped_reason")],
        "missing_predecessor_targets": ds.missing_predecessor_targets,
        "missing_matched_reference": ds.missing_matched_reference,
        "matched_reference_targets": ds.matched_reference_targets,
        "targets_with_inferred_runtime": ds.targets_with_inferred_runtime,
        "relations": {
            i: {k: m.get(k) for k in ("parent_id", "relation", "step_semantics")}
            for i, m in ds.relations.items()
        },
        "source_hash": source_hash.hexdigest(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src", "tests", "configs"], text=True
            ).strip()
        ),
        "default_config": str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1])),
        "default_config_hash": hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest(),
        "base_scores": ds.base_scores,
        "normalization": f"training-context-only signed-log1p, {scaling}, zero constant coordinates, clamp [-20,20]",
        "config": config,
        "config_hash": config_hash,
        "seed": seed,
        "fold_seeds": [seed + i for i in range(len(folds))],
        "backend": backends,
        "feature_metadata_hash": input_hash.hexdigest(),
        "features_manifest": ds.manifest,
        "target_hash": hashlib.sha256(ds.targets.numpy().tobytes()).hexdigest(),
        "model_revisions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mamba_ssm": importlib.metadata.version("mamba_ssm")
            if "mamba_ssm" in backends
            else None,
        },
        "state_semantics": "cumulative_effective_delta_from_base",
        "parameter_counts": [r["parameter_count"] for r in fold_records],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", choices=["m1", "m1_matched", "m2", "m2_matched", "m3", "m5", "m6"], required=True
    )
    for name in ("features", "evals", "out"):
        p.add_argument(f"--{name}")
    p.add_argument("--split", choices=["family", "seed", "trajectory"], default="trajectory")
    for name, default in [
        ("epochs", 100),
        ("pretrain-epochs", 30),
        ("patience", 10),
        ("batch-size", 16),
        ("seed", 0),
        ("n-splits", 5),
        ("n-boot", 1000),
        ("d-model", 256),
    ]:
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--seeds", help="comma-separated full experiment seeds, e.g. 0,1,2")
    p.add_argument(
        "--features-variant", choices=["cumulative", "incremental"], default="cumulative"
    )
    p.add_argument("--scaling", choices=["standard", "robust"], default="standard")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--target", choices=["delta", "level"], default="delta")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--targets", nargs="+")
    p.add_argument("--zoo")
    p.add_argument("--holdout")
    p.add_argument("--holdout-variant", choices=["purge", "chain-free"], default="purge")
    p.set_defaults(**json.loads(DEFAULT_CONFIG_PATH.read_text()))
    args = vars(p.parse_args(argv))
    selftest = args.pop("selftest")
    seeds = args.pop("seeds")
    if selftest:
        from wsp.predictor.selftest import prepare_selftest

        root = prepare_selftest()
        args.update(
            features=root / "features",
            evals=root / "evals",
            out=root / "run",
            synthetic=True,
            split="trajectory",
            epochs=2,
            pretrain_epochs=2,
            n_splits=3,
            n_boot=10,
            d_model=32,
            batch_size=4,
        )
        log.warning("SYNTHETIC SELFTEST: artifacts under %s", root)
    elif not all(args.get(k) for k in ("features", "evals", "out")):
        p.error("--features, --evals and --out are required unless --selftest")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    logging.basicConfig(level=logging.INFO)
    try:
        if seeds:
            seed_values = [int(v) for v in seeds.split(",")]
            if len(set(seed_values)) != len(seed_values):
                raise ValueError("seeds must be unique")
            root = Path(args["out"])
            if root.exists() and any(root.iterdir()):
                raise ValueError(f"output directory is not empty: {root}")
            results = []
            for value in seed_values:
                results.append(
                    run_training(**(args | {"seed": value, "out": root / f"seed-{value}"}))
                )
            summary = seed_summary(results, seed_values)
            (root / "metrics.json").write_text(json.dumps(summary, indent=2))
            lines = [
                "# Full-seed summary",
                "",
                "Mean and min/max range across complete experiment seeds (not confidence intervals).",
                "",
                "| target | metric | mean | range |",
                "|---|---|---:|---|",
            ]
            for name, metrics in summary["seed_summary"].items():
                for metric, values in metrics.items():
                    lines.append(
                        f"| {name} | {metric} | {values['mean']:.6g} | {values['range']} |"
                    )
            (root / "results.md").write_text("\n".join(lines) + "\n")
            first_manifest = json.loads(
                (root / f"seed-{seed_values[0]}" / "manifest.json").read_text()
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "seed": seed_values,
                        "seeds": seed_values,
                        "config_hash": hashlib.sha256(
                            json.dumps(
                                args | {"out": str(root), "seeds": seed_values},
                                sort_keys=True,
                                default=str,
                            ).encode()
                        ).hexdigest(),
                        "git_sha": first_manifest["git_sha"],
                        "model_revisions": first_manifest["model_revisions"],
                        "runs": [f"seed-{v}/manifest.json" for v in seed_values],
                    },
                    indent=2,
                )
            )
        else:
            run_training(**args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
