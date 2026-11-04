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

from wsp.predictor.baselines import fit_eval, metrics_with_ci
from wsp.predictor.data import FeatureDataset, collate_trajectories
from wsp.predictor.models import build_model, effective_rank, seed_all

log = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/p4_predictors.json"


class Normalizer:
    """Signed log1p then training-only standardization, persisted for inference."""

    keys = ("tokens", "pooled", "stats")

    def __init__(self, ds, indices):
        ids = sorted({i for idx in indices for i in ds.chain(ds.rows[idx]["id"])})
        self.ids = ids
        self.state = {}
        for key in self.keys:
            values = [ds.entry(i)[key] for i in ids]
            x = torch.cat(values) if key == "tokens" else torch.stack(values)
            x = self.log(x)
            if not torch.isfinite(x).all():
                raise ValueError(f"nonfinite {key} features")
            self.state[key] = (x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4))

    @staticmethod
    def log(x):
        return x.sign() * x.abs().log1p()

    def __call__(self, batch, device):
        b = {k: v.to(device) for k, v in batch.items()}
        for key, (mean, std) in self.state.items():
            b[key] = ((self.log(b[key]) - mean.to(device)) / std.to(device)).clamp(-20, 20)
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
):
    if min(epochs, patience, batch_size, n_boot) < 1 or pretrain_epochs < 1:
        raise ValueError(
            "epochs, pretrain epochs, patience, batch size and bootstrap count must be positive"
        )
    seed_all(seed)
    ds = FeatureDataset(features, evals, target=target, targets=targets, zoo=zoo)
    folds = ds.holdout(holdout, holdout_variant) if holdout else ds.folds(split, n_splits)
    # Plan all validation sets before writing anything. A two-group dataset cannot
    # honestly support both a held-out group and an inner group for early stopping.
    plans = []
    for i, (train, test) in enumerate(folds):
        inner = ds.folds(split, min(3, n_splits), train, scope=f"inner:{i}")
        if not inner:
            log.warning("Skipping outer fold %d: no inner folds survive ancestry filtering", i)
            continue
        plans.append((inner[0], test))
    if not plans:
        raise ValueError("no folds survive ancestry filtering")
    config = {
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
    predictions = np.full(ds.targets.shape, np.nan)
    fold_records, backends, baseline_folds = [], [], []
    for fold, ((train, val), test) in enumerate(plans):
        fold_seed = seed + fold
        seed_all(fold_seed)
        net = build_model(model, ds, seed=fold_seed, d_model=d_model, device=device)
        normalizer = Normalizer(ds, train)
        backends.append(net.backend)
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
                    return effective_rank(torch.cat(zs))

                rank, standard_rank = rank_of(net), rank_of(ablation)
                collapse = {
                    "erank": rank,
                    "frozen_predictor_erank": standard_rank,
                    "collapsed": rank < 2,
                    "acceptance_gate": rank >= 2 and rank >= 2 * standard_rank,
                }
            net.freeze_encoder()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
        best, stale, best_state, history = float("inf"), 0, None, []
        for epoch in range(epochs):
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
            pred = predict(net, ds, val, normalizer, device, batch_size)
            val_loss = float(np.nanmean((pred - ds.targets[val].numpy()) ** 2))
            history.append(
                {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss}
            )
            if val_loss < best:
                best, stale = val_loss, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise ValueError("validation fold has no finite targets")
        net.load_state_dict(best_state)
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
        log.info("fold %d backend=%s validation MSE=%g", fold, net.backend, best)
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
    # Same fitting rows and test groups as the neural head, including inner validation exclusion.
    x_stats = np.stack([ds.entry(r["id"])["stats"].numpy() for r in ds.rows])
    baseline = fit_eval(
        x_stats,
        ds.targets.numpy(),
        baseline_folds,
        model="statnn",
        target_names=ds.names,
        seed=seed,
        n_boot=n_boot,
    )
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
        "ci_method": "seeded percentile bootstrap over edits (not independent trajectories)",
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    lines = [
        f"# P4 {model} — run `{out.name}`",
        "",
        f"Split: {split}; labeled edits: {len(ds)}.",
        "95% percentile bootstrap over edits; chain correlations are not corrected.",
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
        "normalization": "training-context-only signed-log1p, standardize, clamp [-20,20]",
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
    p.add_argument("--model", choices=["m1", "m1_matched", "m2", "m3", "m5", "m6"], required=True)
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
        run_training(**args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
