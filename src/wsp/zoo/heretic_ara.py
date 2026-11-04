"""Heretic ARA (Arbitrary-Rank Ablation) edit family module."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
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
from safetensors.torch import save_file

from wsp.schema import EditDescriptor, write_manifest
from wsp.zoo.apply import load_edited_model
from wsp.zoo.common import get_git_sha, git_provenance, resolve_base_revision


def select_best_trial(
    trials: list[dict[str, Any]],
    kl_divergence_target: float | None = None,
) -> dict[str, Any]:
    """Select the best trial from study records.

    If kl_divergence_target is given, chooses the trial with lowest refusal rate
    among those satisfying kl_divergence <= kl_divergence_target. If none qualify,
    raises rather than exporting an infeasible edit.
    Otherwise, chooses the trial minimizing (refusals, kl_divergence).
    """
    if not trials:
        raise ValueError("Cannot select from an empty trial list.")

    for trial in trials:
        if not all(math.isfinite(trial[key]) for key in ("refusals", "kl_divergence")):
            raise ValueError("Nonfinite trial metrics")
    if kl_divergence_target is not None:
        eligible = [t for t in trials if t["kl_divergence"] <= kl_divergence_target]
        if not eligible:
            raise ValueError("No trial satisfies KL divergence target")
        return min(eligible, key=lambda t: t["refusals"])

    return min(
        trials,
        key=lambda t: (t.get("refusals", float("inf")), t.get("kl_divergence", float("inf"))),
    )


class FixedWeightTrial:
    """Keep a configured weight out of the sampler's search dimensions."""

    def __init__(self, trial, weight):
        self.trial, self.weight = trial, weight

    def __getattr__(self, name):
        return getattr(self.trial, name)

    def suggest_float(self, name, low, high, **kwargs):
        if name == "preserve_good_behavior_weight" and self.weight is not None:
            return self.trial.suggest_float(name, float(self.weight), float(self.weight))
        return self.trial.suggest_float(name, low, high, **kwargs)


def extract_trial_metrics(evaluator, scores):
    """Match Heretic plugin instances to their actual output labels."""
    from heretic.scorers.keyword_rate import KeywordRate
    from heretic.scorers.kl_divergence import KLDivergence

    values = dict(scores)
    metrics = []
    for plugin in (KeywordRate, KLDivergence):
        names = [e.name for e in evaluator._scorer_entries if isinstance(e.scorer, plugin)]
        if len(names) != 1 or names[0] not in values:
            raise ValueError(f"Missing or ambiguous {plugin.__name__} output: {names}")
        value = float(values[names[0]].value)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite {names[0]}")
        metrics.append(value)
    return tuple(metrics)


def merged_delta(base, merged):
    """Effective deployed delta: subtract in fp32 after rounding merged weights to bf16.

    Keep fp32 differences so applying them in fp32 recovers each bf16 weight exactly.
    Unchanged tensors (including buffers) are omitted.
    """
    if base.keys() != merged.keys():
        raise ValueError("Merged/base state keys differ")
    delta = {}
    for key, weight in merged.items():
        original = base[key].cpu()
        deployed = weight.detach().cpu()
        if deployed.is_floating_point():
            deployed = deployed.to(torch.bfloat16)
        diff = deployed.float() - original.float()
        if not torch.isfinite(diff).all():
            raise ValueError(f"Nonfinite merged delta: {key}")
        if torch.count_nonzero(diff):
            delta[key] = diff.contiguous()
    return delta


def generate(
    base_model: str,
    params: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> EditDescriptor:
    """Run ARA, evaluate merged weights, and gate dense export on reloaded KL.

    Args:
        base_model: Hugging Face model ID or local model path.
        params: Hyperparameters (preserve_good_behavior_weight, trials, kl_divergence_target, etc.).
        seed: Random seed.
        out_dir: Directory where edit.json, delta.safetensors, and study.json will be written.

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
        "model_commit": resolve_base_revision(base_model, params.get("base_revision")),
        "batch_size": int(params.get("batch_size", 1)),
        "dtypes": ["bfloat16"],
        "device_map": params.get("device_map", "auto"),
        "max_response_length": int(params.get("max_response_length", 100)),
    }
    settings_values.update(params.get("heretic_settings", {}))
    settings_values["model_commit"] = resolve_base_revision(
        base_model, settings_values["model_commit"]
    )
    settings = Settings(**settings_values)

    ctx = modifier = evaluator = model = None
    try:
        model = Model(settings)
        # Snapshot before PEFT wraps/mutates the base, on CPU to spare shared VRAM.
        base_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        evaluator = Evaluator(settings, model)

        ara_settings = ara.Settings(**{"lora_rank": lora_rank, **params.get("ara_settings", {})})
        modifier = ara.ARA(heretic_settings=settings, settings=ara_settings)
        ctx = Context(settings=settings, model=model)
        modifier.init(ctx)

        trial_records: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            parameters = modifier.suggest_parameters(ctx, FixedWeightTrial(trial, p_weight))

            modifier.reset_model(ctx)
            modifier.modify_model(ctx, parameters)

            scores = evaluator.get_scores()
            objective_values = evaluator.get_objective_values(scores)

            refusals, kl_div = extract_trial_metrics(evaluator, scores)

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
            n_startup_trials=30,
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

        # Reapply selected parameters once through Heretic, never rerun the objective.
        # Score the deployed merged weights themselves, not stale search-trial metrics.
        model.model = model.get_merged_model()
        model.model.to(dtype=torch.bfloat16)
        verified_refusals, verified_kl = extract_trial_metrics(evaluator, evaluator.get_scores())
        delta = merged_delta(base_state, model.model.state_dict())
        touched_tensors = sorted(delta)
        delta_fro_norm = math.sqrt(sum(t.square().sum().item() for t in delta.values()))

        study_file = out_dir / "study.json"
        study_data = {
            "trials": trial_records,
            "best_trial": best_trial,
            "verified_refusals": verified_refusals,
            "verified_kl": verified_kl,
            "kl_constraint_met": kl_target is None or verified_kl <= kl_target,
            "verification_tolerance": 0.01,
            "verification_status": "pending",
            "seed": seed,
            "params": params,
        }
        study_file.write_text(json.dumps(study_data, indent=2))

        desc_params = dict(params)
        desc_params["delta_format"] = "dense"
        desc_params.update(git_provenance())
        desc = EditDescriptor(
            family="heretic_ara",
            params=desc_params,
            seed=seed,
            base_model=base_model,
            base_revision=settings_values["model_commit"],
            git_sha=get_git_sha(),
            touched_tensors=touched_tensors,
            delta_fro_norm=delta_fro_norm,
        )
        # Stage the descriptor required by the public loader; publish only after verification.
        with TemporaryDirectory(prefix=".verify-", dir=out_dir) as staging:
            stage = Path(staging)
            save_file(delta, stage / "delta.safetensors")
            (stage / "edit.json").write_text(desc.to_json())
            try:
                reloaded_state = load_edited_model(base_state, stage)
                model.model.load_state_dict(reloaded_state, strict=True)
                del reloaded_state
                reloaded_refusals, reloaded_kl = extract_trial_metrics(
                    evaluator, evaluator.get_scores()
                )
                study_data.update(
                    reloaded_kl=reloaded_kl,
                    reloaded_refusals=reloaded_refusals,
                    kl_reload_difference=abs(reloaded_kl - verified_kl),
                )
                if abs(reloaded_kl - verified_kl) > 0.01:
                    raise ValueError("Saved ARA verification failed: KL difference exceeds 0.01")
                if not study_data["kl_constraint_met"]:
                    raise ValueError("Saved ARA verification failed: verified KL exceeds target")
            except Exception as exc:
                study_data.update(verification_status="failed", verification_error=str(exc))
                study_file.write_text(json.dumps(study_data, indent=2))
                raise
            study_data["verification_status"] = "passed"
            study_file.write_text(json.dumps(study_data, indent=2))
            (stage / "delta.safetensors").replace(out_dir / "delta.safetensors")
            (stage / "edit.json").replace(out_dir / "edit.json")

        write_manifest(
            out_dir / "manifest.json",
            seed=seed,
            config_hash=desc.id,
            git_sha=desc.git_sha,
            model_revisions={base_model: desc.base_revision},
            params=desc_params,
        )
    finally:
        ctx = modifier = evaluator = model = None
        gc.collect()
        empty_cache()

    return desc
