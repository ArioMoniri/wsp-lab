"""CPU/CUDA P4 training with nested group validation and genuine targets only.

Sigmoid heads predict absolute rates; --delta-from-base subtracts the known base
from predictions and labels for both training and reporting. PPL/KL are excluded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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


def last_predictions(model, batch, base):
    p = model(batch)
    return p[torch.arange(len(p), device=p.device), batch["mask"].sum(1) - 1] - base


def masked_loss(pred, target):
    valid = torch.isfinite(target)
    return (pred[valid] - target[valid]).square().mean()


def predict(model, ds, indices, normalizer, device, batch_size):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader(ds, indices, batch_size, 0):
            b = normalizer(batch, device)
            output.append(last_predictions(model, b, ds.base.to(device)).cpu())
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
    delta_from_base=False,
    targets=None,
    zoo=None,
):
    if min(epochs, patience, batch_size, n_boot) < 1 or pretrain_epochs < 1:
        raise ValueError(
            "epochs, pretrain epochs, patience, batch size and bootstrap count must be positive"
        )
    seed_all(seed)
    ds = FeatureDataset(features, evals, delta_from_base=delta_from_base, targets=targets, zoo=zoo)
    folds = ds.folds(split, n_splits)
    # Plan all validation sets before writing anything. A two-group dataset cannot
    # honestly support both a held-out group and an inner group for early stopping.
    plans = [(ds.folds(split, min(3, n_splits), train)[0], test) for train, test in folds]
    config = {
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
        "delta_from_base": delta_from_base,
        "targets": ds.names,
        "zoo": str(zoo) if zoo else ds.manifest.get("zoo"),
    }
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    predictions = np.full(ds.targets.shape, np.nan)
    fold_records, backends, baseline_folds = [], [], []
    for fold, ((train, val), test) in enumerate(plans):
        fold_seed = seed + fold
        seed_all(fold_seed)
        net = build_model(model, ds, seed=fold_seed, d_model=d_model, device=device)
        normalizer = Normalizer(ds, train)
        backends.append(net.backend)
        pretrain_history = []
        if model in {"m3", "m6"}:
            opt = torch.optim.AdamW(net.parameters(), lr=lr)
            for epoch in range(pretrain_epochs):
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
            net.freeze_encoder()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
        best, stale, best_state, history = float("inf"), 0, None, []
        for epoch in range(epochs):
            net.train()
            losses = []
            for batch in loader(ds, train, batch_size, fold_seed + epoch, True):
                b = normalizer(batch, device)
                opt.zero_grad()
                loss = masked_loss(last_predictions(net, b, ds.base.to(device)), b["target"])
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
                "normalization_context_ids": normalizer.ids,
                "pretraining": pretrain_history,
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
    results = {
        "config": config,
        "targets": scores,
        "length_ge_4": {"n_edits": int(long_mask.sum()), "targets": long_scores},
        "statnn": baseline,
        "folds": fold_records,
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
    for label, metrics in [(model, scores), ("statnn", baseline["targets"])]:
        for name, m in metrics.items():
            cells = [
                f"{m[k]['value']:.5g} {m[k]['ci95']}" if k in m else "unavailable"
                for k in ("mae", "rmse", "spearman")
            ]
            lines.append(f"| {label} | {name} | {m['n']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Heads predict absolute bounded rates; delta mode subtracts measured base rates.",
        "JEPA state inputs are cumulative effective-delta features, not absolute weight features.",
        "M5 uses tensor attention followed by causal edit attention.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    input_hash = hashlib.sha256()
    for file in sorted(Path(features).glob("*/features.json")):
        input_hash.update(file.read_bytes())
    source_hash = hashlib.sha256()
    for file in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(file.read_bytes())
    manifest = {
        "git_sha": sha,
        "source_hash": source_hash.hexdigest(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src", "tests", "configs"], text=True
            ).strip()
        ),
        "default_config": str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1])),
        "default_config_hash": hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest(),
        "base_scores": ds.base.tolist(),
        "normalization": "training-context-only signed-log1p, standardize, clamp [-20,20]",
        "config": config,
        "config_hash": config_hash,
        "seed": seed,
        "fold_seeds": [seed + i for i in range(len(folds))],
        "backend": backends,
        "feature_metadata_hash": input_hash.hexdigest(),
        "features_manifest": ds.manifest,
        "target_hash": hashlib.sha256(ds.targets.numpy().tobytes()).hexdigest(),
        "model_revisions": {"torch": torch.__version__},
        "state_semantics": "cumulative_effective_delta_from_base",
        "parameter_counts": [r["parameter_count"] for r in fold_records],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["m1", "m2", "m3", "m5", "m6"], required=True)
    for name in ("features", "evals", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--split", choices=["family", "seed", "trajectory"], required=True)
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
    p.add_argument("--delta-from-base", action="store_true")
    p.add_argument("--targets", nargs="+")
    p.add_argument("--zoo")
    p.set_defaults(**json.loads(DEFAULT_CONFIG_PATH.read_text()))
    args = vars(p.parse_args(argv))
    torch.set_num_threads(min(2, torch.get_num_threads()))
    logging.basicConfig(level=logging.INFO)
    try:
        run_training(**args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
