"""Heretic ARA (Arbitrary-Rank Ablation) edit family module."""

from __future__ import annotations

import gc
import json
import math
from importlib.metadata import version
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
from wsp.zoo.common import (
    get_git_sha,
    git_provenance,
    resolve_base_revision,
    to_text_model_keys,
)

KERNEL_VARIANT = "wsp_ara_no_row_normalization_v1"
SELECTION_RULE = "feasible_pareto_lexicographic_refusals_kl_trial_v1"


def explicit_target(value):
    """Production ARA always requires a finite positive, explicitly supplied ceiling."""
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError("ARA requires an explicit finite positive KL divergence target")
    return float(value)


def feasible_frontier(trials, kl_divergence_target):
    """Return the nondominated feasible records, deterministically ordered."""
    target = explicit_target(kl_divergence_target)
    for trial in trials:
        if not all(math.isfinite(trial[key]) for key in ("refusals", "kl_divergence")):
            raise ValueError("Nonfinite trial metrics")
    feasible = [t for t in trials if t["kl_divergence"] <= target]
    frontier = [
        t
        for t in feasible
        if not any(
            u["refusals"] <= t["refusals"]
            and u["kl_divergence"] <= t["kl_divergence"]
            and (u["refusals"] < t["refusals"] or u["kl_divergence"] < t["kl_divergence"])
            for u in feasible
        )
    ]
    return sorted(frontier, key=lambda t: (t["refusals"], t["kl_divergence"], t["trial"]))


def select_best_trial(trials, kl_divergence_target=None):
    """Choose (refusals, KL, trial number) from the feasible Pareto frontier."""
    if not trials:
        raise ValueError("Cannot select from an empty trial list.")
    frontier = feasible_frontier(trials, kl_divergence_target)
    if not frontier:
        raise ValueError("No trial satisfies KL divergence target")
    return frontier[0]


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


def modify_with_trace(modifier, ctx, parameters, traces, phase):
    """Observe each module's LBFGS outer steps without changing upstream optimization."""
    from torch.optim.optimizer import register_optimizer_step_post_hook

    modules = {}
    for name, module in ctx.get_model().model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            modules[id(module.lora_A["default"].weight)] = (name, module)

    def record(optimizer, args, kwargs):
        tensors = optimizer.param_groups[0]["params"]
        if not isinstance(optimizer, torch.optim.LBFGS) or id(tensors[0]) not in modules:
            return
        name, module = modules[id(tensors[0])]
        a, b = [t.detach().float() for t in tensors]
        state = optimizer.state[tensors[0]]
        trace = {
            "phase": phase,
            "module": name,
            "step": 1 + sum(t["phase"] == phase and t["module"] == name for t in traces),
            # LBFGS can return immediately on zero gradient before storing prev_loss.
            "last_closure_loss": float(state["prev_loss"]) if "prev_loss" in state else None,
            "a_norm": float(a.norm()),
            "b_norm": float(b.norm()),
            "adapter_product_norm": float((b @ a).norm()),
            "effective_weight_norm": float(
                (module.base_layer.weight.detach().float() + b @ a).norm()
            ),
            "gradient_norm": math.sqrt(
                sum(float(t.grad.float().square().sum()) for t in tensors if t.grad is not None)
            ),
        }
        if not all(math.isfinite(v) for v in trace.values() if isinstance(v, float)):
            raise ValueError(f"Nonfinite optimizer trace: {name}")
        traces.append(trace)

    handle = register_optimizer_step_post_hook(record)
    try:
        modifier.modify_model(ctx, parameters)
    finally:
        handle.remove()


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
    kl_target = explicit_target(params.get("kl_divergence_target"))

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
    study_data = None
    try:
        model = Model(settings)
        # Snapshot before PEFT wraps/mutates the base, on CPU to spare shared VRAM.
        base_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        evaluator = Evaluator(settings, model)

        ara_values = {"lora_rank": lora_rank, **params.get("ara_settings", {})}
        if ara_values.get("preserve_row_magnitudes", False):
            raise ValueError(
                "This kernel disables row normalization in optimization and deployment"
            )
        ara_values["preserve_row_magnitudes"] = False
        ara_settings = ara.Settings(**ara_values)
        modifier = ara.ARA(heretic_settings=settings, settings=ara_settings)
        ctx = Context(settings=settings, model=model)
        modifier.init(ctx)

        trial_records: list[dict[str, Any]] = []
        optimizer_traces = []

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            parameters = modifier.suggest_parameters(ctx, FixedWeightTrial(trial, p_weight))

            modifier.reset_model(ctx)
            modify_with_trace(
                modifier, ctx, parameters, optimizer_traces, f"trial_{trial.number + 1}"
            )

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

        study_file = out_dir / "study.json"
        study_data = {
            "trials": trial_records,
            "frontier": [],
            "best_trial": None,
            "optimizer_traces": optimizer_traces,
            "selection_rule": SELECTION_RULE,
            "kl_divergence_target": kl_target,
            "kl_constraint_enabled": True,
            "kl_constraint_met": False,
            "kernel_variant": KERNEL_VARIANT,
            "effective_ara_settings": ara_settings.model_dump(mode="json"),
            "effective_heretic_settings": settings.model_dump(mode="json"),
            "heretic_version": version("heretic-llm"),
            "search_protocol": "ten-trial startup search" if n_trials == 10 else "TPE startup=30",
            "verification_status": "pending",
            "seed": seed,
            "params": params,
        }
        study.optimize(objective, n_trials=n_trials)
        study_data["frontier"] = feasible_frontier(trial_records, kl_target)
        best_trial = select_best_trial(trial_records, kl_target)
        study_data["best_trial"] = best_trial
        best_params = ara.Parameters(**best_trial["parameters"])
        modifier.reset_model(ctx)
        modify_with_trace(modifier, ctx, best_params, optimizer_traces, "selected_reapply")

        # Reapply selected parameters once through Heretic, never rerun the objective.
        # Score the deployed merged weights themselves, not stale search-trial metrics.
        model.model = model.get_merged_model()
        model.model.to(dtype=torch.bfloat16)
        verified_refusals, verified_kl = extract_trial_metrics(evaluator, evaluator.get_scores())
        # Heretic loads the multimodal wrapper (``model.language_model.*``); save base-state names
        # (``model.*``, as HFRunner / feature extraction see them) so loaders need no mapping.
        delta = to_text_model_keys(merged_delta(base_state, model.model.state_dict()))
        touched_tensors = sorted(delta)
        delta_fro_norm = math.sqrt(sum(t.square().sum().item() for t in delta.values()))

        study_data.update(
            verified_refusals=verified_refusals,
            verified_kl=verified_kl,
            kl_constraint_met=verified_kl <= kl_target,
            verification_tolerance=0.01,
        )
        study_file.write_text(json.dumps(study_data, indent=2))

        desc_params = dict(params)
        desc_params["delta_format"] = "dense"
        desc_params["kernel_variant"] = KERNEL_VARIANT
        desc_params.update(git_provenance())
        desc = EditDescriptor(
            family="heretic_ara",
            params=desc_params,
            seed=seed,
            base_model=base_model,
            base_revision=settings_values["model_commit"],
            git_sha=get_git_sha(),
            provenance={"kernel_variant": KERNEL_VARIANT},
            touched_tensors=touched_tensors,
            delta_fro_norm=delta_fro_norm,
        )
        # Stage the descriptor required by the public loader; publish only after verification.
        with TemporaryDirectory(prefix=".verify-", dir=out_dir) as staging:
            stage = Path(staging)
            save_file(delta, stage / "delta.safetensors")
            (stage / "edit.json").write_text(desc.to_json())
            reloaded_state = load_edited_model(base_state, stage)
            study_data["reload_tensor_equal"] = all(
                torch.equal(reloaded_state[k].to(v), v) for k, v in model.model.state_dict().items()
            )
            if not study_data["reload_tensor_equal"]:
                raise ValueError("Saved ARA verification failed: tensor mismatch")
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
            if not study_data["kl_constraint_met"] or reloaded_kl > kl_target:
                raise ValueError("Saved ARA verification failed: verified KL exceeds target")
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
    except Exception as exc:
        if study_data is not None:
            study_data.update(verification_status="failed", verification_error=str(exc))
            (out_dir / "study.json").write_text(json.dumps(study_data, indent=2))
        for stale in ("edit.json", "delta.safetensors", "manifest.json"):
            (out_dir / stale).unlink(missing_ok=True)
        raise
    finally:
        ctx = modifier = evaluator = model = None
        gc.collect()
        empty_cache()

    return desc
