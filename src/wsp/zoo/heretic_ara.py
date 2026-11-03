"""Heretic ARA (Arbitrary-Rank Ablation) edit family module."""

from __future__ import annotations

import gc
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import optuna
import torch
from heretic.config import InitOnlySettings as Settings
from heretic.evaluator import Evaluator
from heretic.model import Model
from heretic.modifiers import ara
from heretic.plugin import Context
from heretic.system import empty_cache
from optuna.samplers import TPESampler

from wsp.schema import EditDescriptor, write_manifest


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def select_best_trial(
    trials: list[dict[str, Any]],
    kl_divergence_target: float | None = None,
) -> dict[str, Any]:
    """Select the best trial from study records.

    If kl_divergence_target is given, chooses the trial with lowest refusal rate
    among those satisfying kl_divergence <= kl_divergence_target. If none qualify,
    chooses the one closest to kl_divergence_target.
    Otherwise, chooses the trial minimizing (refusals, kl_divergence).
    """
    if not trials:
        raise ValueError("Cannot select from an empty trial list.")

    if kl_divergence_target is not None:
        eligible = [t for t in trials if t.get("kl_divergence", 0.0) <= kl_divergence_target]
        if eligible:
            return min(eligible, key=lambda t: t.get("refusals", float("inf")))
        return min(
            trials,
            key=lambda t: (
                abs(t.get("kl_divergence", 0.0) - kl_divergence_target),
                t.get("refusals", float("inf")),
            ),
        )

    return min(
        trials,
        key=lambda t: (t.get("refusals", float("inf")), t.get("kl_divergence", float("inf"))),
    )


def compute_adapter_stats(peft_model: torch.nn.Module) -> tuple[list[str], float]:
    """Inspect PEFT model adapters to find touched tensors and compute Frobenius norm."""
    touched_tensors: list[str] = []
    fro_sq = 0.0

    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            scaling = getattr(module, "scaling", {}).get("default", 1.0)
            delta_W = (B @ A) * scaling
            norm_sq = float((delta_W.float() ** 2).sum().item())
            if norm_sq > 0:
                touched_tensors.append(f"{name}.weight")
                fro_sq += norm_sq

    return sorted(touched_tensors), math.sqrt(fro_sq)


def generate(
    base_model: str,
    params: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> EditDescriptor:
    """Run Heretic ARA modifier programmatically and save the PEFT LoRA adapter.

    Args:
        base_model: Hugging Face model ID or local model path.
        params: Hyperparameters (preserve_good_behavior_weight, trials, kl_divergence_target, etc.).
        seed: Random seed.
        out_dir: Directory where edit.json, lora/, and study.json will be written.

    Returns:
        EditDescriptor describing the generated edit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    n_trials = int(params.get("trials", 40))
    if n_trials < 1:
        raise ValueError("trials must be positive")
    lora_rank = int(params.get("lora_rank", 50))
    p_weight = params.get("preserve_good_behavior_weight")
    kl_target = params.get("kl_divergence_target")

    settings_values = {
        "model": base_model,
        "seed": seed,
        "model_commit": params.get("base_revision"),
        "batch_size": int(params.get("batch_size", 1)),
        "dtypes": ["bfloat16"],
        "device_map": params.get("device_map", "auto"),
        "max_response_length": int(params.get("max_response_length", 100)),
    }
    settings_values.update(params.get("heretic_settings", {}))
    settings = Settings(**settings_values)

    model = Model(settings)
    evaluator = Evaluator(settings, model)

    ara_settings = ara.Settings(lora_rank=lora_rank, **params.get("ara_settings", {}))
    modifier = ara.ARA(heretic_settings=settings, settings=ara_settings)
    ctx = Context(settings=settings, model=model)
    modifier.init(ctx)

    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> tuple[float, ...]:
        parameters = modifier.suggest_parameters(ctx, trial)
        if p_weight is not None:
            parameters.preserve_good_behavior_weight = float(p_weight)

        modifier.reset_model(ctx)
        modifier.modify_model(ctx, parameters)

        scores = evaluator.get_scores()
        objective_values = evaluator.get_objective_values(scores)

        refusals = next(
            (
                s.value
                for name, s in scores
                if "refusal" in name.lower() or "keyword" in name.lower()
            ),
            scores[0][1].value if scores else 0.0,
        )
        kl_div = next(
            (s.value for name, s in scores if "kl" in name.lower()),
            scores[1][1].value if len(scores) > 1 else 0.0,
        )

        trial_records.append(
            {
                "trial": trial.number + 1,
                "parameters": parameters.to_dict(),
                "scores": {name: score.value for name, score in scores},
                "refusals": float(refusals),
                "kl_divergence": float(kl_div),
            }
        )
        return objective_values

    sampler = TPESampler(
        n_startup_trials=min(10, n_trials),
        n_ei_candidates=128,
        multivariate=True,
        seed=seed,
    )
    study = optuna.create_study(
        sampler=sampler,
        directions=evaluator.get_objective_directions(),
        study_name="heretic_ara",
    )

    study.optimize(objective, n_trials=n_trials)

    best_trial = select_best_trial(trial_records, kl_divergence_target=kl_target)
    best_params = ara.Parameters(**best_trial["parameters"])
    modifier.reset_model(ctx)
    modifier.modify_model(ctx, best_params)

    lora_dir = out_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(lora_dir))

    study_file = out_dir / "study.json"
    study_data = {
        "trials": trial_records,
        "best_trial": best_trial,
        "seed": seed,
        "params": params,
    }
    study_file.write_text(json.dumps(study_data, indent=2))

    touched_tensors, delta_fro_norm = compute_adapter_stats(model.model)

    desc_params = dict(params)
    desc_params["delta_format"] = "lora"
    desc = EditDescriptor(
        family="heretic_ara",
        params=desc_params,
        seed=seed,
        base_model=base_model,
        base_revision=params.get("base_revision", "main"),
        git_sha=get_git_sha(),
        touched_tensors=touched_tensors,
        delta_fro_norm=delta_fro_norm,
    )
    (out_dir / "edit.json").write_text(desc.to_json())

    write_manifest(
        out_dir / "manifest.json",
        seed=seed,
        config_hash=desc.id,
        git_sha=desc.git_sha,
        model_revisions={base_model: desc.base_revision},
        params=desc_params,
    )
    # Break the objective closure and release all live tensors before emptying CUDA cache.
    ctx = modifier = evaluator = model = None
    gc.collect()
    empty_cache()

    return desc
