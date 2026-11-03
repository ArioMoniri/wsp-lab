"""Tests for Heretic ARA edit family."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from wsp.schema import EditDescriptor
from wsp.zoo.heretic_ara import generate, select_best_trial


def test_select_best_trial_without_target():
    trials = [
        {"trial": 1, "refusals": 0.8, "kl_divergence": 0.05},
        {"trial": 2, "refusals": 0.2, "kl_divergence": 0.30},
        {"trial": 3, "refusals": 0.4, "kl_divergence": 0.15},
    ]
    # Default selection: lowest refusals
    best = select_best_trial(trials)
    assert best["trial"] == 2


def test_select_best_trial_with_kl_target():
    trials = [
        {"trial": 1, "refusals": 0.8, "kl_divergence": 0.05},
        {"trial": 2, "refusals": 0.1, "kl_divergence": 0.35},
        {"trial": 3, "refusals": 0.3, "kl_divergence": 0.12},
    ]
    # With KL target <= 0.20, trial 3 is the lowest refusal satisfying the constraint
    best = select_best_trial(trials, kl_divergence_target=0.20)
    assert best["trial"] == 3

    # If no trial satisfies KL target, pick the one closest to target
    best_fallback = select_best_trial(trials, kl_divergence_target=0.01)
    assert best_fallback["trial"] == 1


def test_generate_heretic_ara_toy(tmp_path: Path, fake_heretic):
    out_dir = tmp_path / "heretic_ara_toy"
    params = {
        "preserve_good_behavior_weight": 1.0,
        "trials": 2,
        "lora_rank": 4,
        "ara_settings": {"n_optimization_steps": 1, "max_iter": 2},
        "kl_divergence_target": 0.25,
    }
    desc = generate(base_model="toy", params=params, seed=42, out_dir=out_dir)

    assert isinstance(desc, EditDescriptor)
    assert desc.family == "heretic_ara"
    assert desc.seed == 42
    assert desc.base_model == "toy"
    assert desc.params["delta_format"] == "lora"
    assert desc.params["preserve_good_behavior_weight"] == 1.0
    assert len(desc.touched_tensors) > 0
    assert desc.delta_fro_norm is not None and desc.delta_fro_norm >= 0

    # Check written artifacts
    edit_file = out_dir / "edit.json"
    assert edit_file.exists()
    roundtrip = EditDescriptor.from_json(edit_file.read_text())
    assert roundtrip.id == desc.id
    assert roundtrip.family == "heretic_ara"
    assert roundtrip.params["delta_format"] == "lora"

    study_file = out_dir / "study.json"
    assert study_file.exists()
    study = json.loads(study_file.read_text())
    assert "trials" in study
    assert len(study["trials"]) == 2
    for t in study["trials"]:
        assert "refusals" in t
        assert "kl_divergence" in t
        assert "parameters" in t
    assert "best_trial" in study

    lora_dir = out_dir / "lora"
    assert lora_dir.is_dir()
    assert (lora_dir / "adapter_config.json").exists()
    assert (lora_dir / "adapter_model.safetensors").exists()


@pytest.mark.gpu
@pytest.mark.slow
def test_generate_heretic_ara_smoke_gpu(tmp_path: Path, cuda):
    out_dir = Path(__file__).parents[1] / "data/runs/p1-generator-smoke/heretic_ara"
    import yaml

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/p1_generator_smoke.yaml").read_text()
    )
    params = config["heretic_ara"]
    desc = generate(
        base_model="Qwen/Qwen3.5-0.8B",
        params=params,
        seed=123,
        out_dir=out_dir,
    )
    assert desc.family == "heretic_ara"
    assert desc.params["delta_format"] == "lora"
    assert (out_dir / "edit.json").exists()
    assert (out_dir / "study.json").exists()
    study = json.loads((out_dir / "study.json").read_text())
    assert len(study["trials"]) == 2
    assert (out_dir / "lora" / "adapter_config.json").exists()


@pytest.fixture
def fake_heretic(monkeypatch, toy_model):
    """Replace model loading and evaluation, retaining installed ARA and PEFT code."""
    import torch
    from heretic.scorer import Score
    from peft import LoraConfig, get_peft_model

    from wsp.zoo import heretic_ara

    class Model:
        def __init__(self, settings):
            self.model = copy.deepcopy(toy_model)

        def get_layers(self):
            return self.model.layers

        def get_layer_modules(self, i):
            layer = self.get_layers()[i]
            return {"attn": [layer["attn"]["o_proj"]], "mlp": [layer["mlp"]["down_proj"]]}

        def get_module_io_batched(self, prompts):
            g = torch.Generator().manual_seed(1 if prompts[0] == "good" else 2)
            return [
                {
                    key: [
                        (x := torch.randn(4, m.in_features, generator=g), m(x).detach())
                        for m in modules
                    ]
                    for key, modules in self.get_layer_modules(i).items()
                }
                for i in range(2)
            ]

        def apply_lora(self, rank):
            self.model = get_peft_model(
                self.model,
                LoraConfig(r=rank, lora_alpha=rank, target_modules=["o_proj", "down_proj"]),
            )

        def reset_model(self):
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if "lora_" in name:
                        p.zero_()
            return True

    class Evaluator:
        def __init__(self, settings, model):
            self.model = model

        def get_scores(self):
            _, norm = heretic_ara.compute_adapter_stats(self.model.model)
            return [
                ("Refusals", Score(1 / (1 + norm), "", "")),
                ("KL divergence", Score(norm, "", "")),
            ]

        def get_objective_values(self, scores):
            return tuple(s.value for _, s in scores)

        def get_objective_directions(self):
            return ["minimize", "minimize"]

    monkeypatch.setattr(heretic_ara, "Model", Model)
    monkeypatch.setattr(heretic_ara, "Evaluator", Evaluator)
    monkeypatch.setattr(
        heretic_ara.Context,
        "load_prompts",
        lambda self, spec: ["good" if "harmless" in spec.dataset else "bad"] * 4,
    )


def test_ara_seed_reproduces_saved_adapter(tmp_path, fake_heretic):
    import torch
    from safetensors.torch import load_file

    params = {
        "trials": 2,
        "lora_rank": 2,
        "preserve_good_behavior_weight": 2.0,
        "ara_settings": {"n_optimization_steps": 1, "max_iter": 2},
    }
    a = generate("toy", params, 7, tmp_path / "a")
    b = generate("toy", params, 7, tmp_path / "b")
    assert a.id == b.id
    assert a.delta_fro_norm > 0
    wa = load_file(tmp_path / "a/lora/adapter_model.safetensors")
    wb = load_file(tmp_path / "b/lora/adapter_model.safetensors")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    study = json.loads((tmp_path / "a/study.json").read_text())
    assert all(t["parameters"]["preserve_good_behavior_weight"] == 2 for t in study["trials"])
    assert study["best_trial"] == select_best_trial(study["trials"])
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    assert manifest["seed"] == 7 and manifest["git_sha"] == a.git_sha


def test_select_best_trial_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        select_best_trial([])
