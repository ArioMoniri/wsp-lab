"""Tests for LoRA SFT edit family."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wsp.schema import EditDescriptor
from wsp.zoo.lora_sft import generate, map_targets


def test_map_targets():
    raw = ["q", "k", "v", "o", "up", "down"]
    mapped = map_targets(raw)
    assert mapped == ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

    # Passthrough for already mapped names
    assert map_targets(["q_proj", "custom"]) == ["q_proj", "custom"]


def test_generate_lora_sft_toy(tmp_path: Path):
    out_dir = tmp_path / "lora_sft_toy"
    params = {
        "rank": 4,
        "lr": 1e-3,
        "steps": 3,
        "targets": ["o", "up", "down"],
    }
    desc = generate(base_model="toy", params=params, seed=7, out_dir=out_dir)

    assert isinstance(desc, EditDescriptor)
    assert desc.family == "lora_sft"
    assert desc.seed == 7
    assert desc.base_model == "toy"
    assert desc.params["delta_format"] == "lora"
    assert desc.params["rank"] == 4
    assert len(desc.touched_tensors) > 0
    assert desc.delta_fro_norm is not None and desc.delta_fro_norm > 0

    # Check written artifacts
    edit_file = out_dir / "edit.json"
    assert edit_file.exists()
    roundtrip = EditDescriptor.from_json(edit_file.read_text())
    assert roundtrip.id == desc.id
    assert roundtrip.family == "lora_sft"
    assert roundtrip.params["delta_format"] == "lora"

    train_file = out_dir / "train.json"
    assert train_file.exists()
    train_data = json.loads(train_file.read_text())
    assert "loss_curve" in train_data
    assert len(train_data["loss_curve"]) == 3
    for entry in train_data["loss_curve"]:
        assert "step" in entry
        assert "loss" in entry
        assert isinstance(entry["loss"], float)

    lora_dir = out_dir / "lora"
    assert lora_dir.is_dir()
    assert (lora_dir / "adapter_config.json").exists()
    assert (lora_dir / "adapter_model.safetensors").exists()


@pytest.mark.gpu
@pytest.mark.slow
def test_generate_lora_sft_smoke_gpu(tmp_path: Path, cuda):
    out_dir = tmp_path / "lora_sft_gpu"
    params = {
        "rank": 4,
        "lr": 2e-4,
        "steps": 5,
        "targets": ["q", "v"],
    }
    desc = generate(
        base_model="Qwen/Qwen3.5-0.8B",
        params=params,
        seed=42,
        out_dir=out_dir,
    )
    assert desc.family == "lora_sft"
    assert desc.params["delta_format"] == "lora"
    assert (out_dir / "edit.json").exists()
    assert (out_dir / "train.json").exists()
    train_data = json.loads((out_dir / "train.json").read_text())
    assert len(train_data["loss_curve"]) == 5
    assert (out_dir / "lora" / "adapter_config.json").exists()
