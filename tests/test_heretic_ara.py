"""Tests for Heretic ARA edit family."""

from __future__ import annotations

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


def test_generate_heretic_ara_toy(tmp_path: Path):
    out_dir = tmp_path / "heretic_ara_toy"
    params = {
        "preserve_good_behavior_weight": 1.0,
        "trials": 2,
        "lora_rank": 4,
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
    out_dir = tmp_path / "heretic_ara_gpu"
    params = {
        "preserve_good_behavior_weight": 1.0,
        "trials": 2,
        "lora_rank": 8,
    }
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
