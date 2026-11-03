"""Tests for LoRA SFT edit family."""

from __future__ import annotations

import copy
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


def test_generate_lora_sft_toy(tmp_path: Path, fake_sft):
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
    out_dir = Path(__file__).parents[1] / "data/runs/p1-generator-smoke/lora_sft"
    import yaml

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/p1_generator_smoke.yaml").read_text()
    )
    params = config["lora_sft"]
    desc = generate(
        base_model="Qwen/Qwen3.5-0.8B",
        params=params,
        seed=42,
        out_dir=out_dir,
    )
    import math

    from huggingface_hub import hf_hub_download

    from wsp.zoo.common import get_git_sha, resolve_base_revision

    assert desc.git_sha == get_git_sha()
    assert not desc.params["git_dirty"]
    assert desc.base_revision == resolve_base_revision(desc.base_model)
    index = json.loads(
        Path(
            hf_hub_download(
                desc.base_model, "model.safetensors.index.json", revision=desc.base_revision
            )
        ).read_text()
    )
    assert set(desc.touched_tensors) <= set(index["weight_map"])
    assert desc.touched_tensors and math.isfinite(desc.delta_fro_norm) and desc.delta_fro_norm > 0
    assert json.loads((out_dir / "manifest.json").read_text())["git_sha"] == get_git_sha()
    assert desc.family == "lora_sft"
    assert desc.params["delta_format"] == "lora"
    assert (out_dir / "edit.json").exists()
    assert (out_dir / "train.json").exists()
    train_data = json.loads((out_dir / "train.json").read_text())
    assert len(train_data["loss_curve"]) == 5
    assert all(math.isfinite(x["loss"]) for x in train_data["loss_curve"])
    assert (out_dir / "lora" / "adapter_config.json").exists()


@pytest.fixture
def fake_sft(monkeypatch, toy_model):
    import torch
    from transformers import BatchEncoding

    from wsp.zoo import lora_sft

    monkeypatch.setattr(lora_sft, "resolve_base_revision", lambda *a: "a" * 40)

    class Tokenizer:
        pad_token = None
        eos_token = "eos"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, texts, **kwargs):
            ids = torch.tensor([[ord(c) % 31 + 1 for c in text[:8]] for text in texts])
            mask = torch.ones_like(ids)
            mask[:, -1] = 0
            return BatchEncoding({"input_ids": ids, "attention_mask": mask})

    loaded_models = []

    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = copy.deepcopy(toy_model)
            loaded_models.append(model)
            return model

    monkeypatch.setattr(lora_sft, "AutoTokenizer", Tokenizer)
    monkeypatch.setattr(lora_sft, "get_model_class", lambda _: Loader)
    monkeypatch.setattr(lora_sft.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        lora_sft, "load_dataset", lambda *a, **kw: [{"question": "1+1?", "answer": "2"}]
    )
    return loaded_models


def test_sft_seed_reproduces_saved_adapter_and_freezes_base(tmp_path, fake_sft, toy_model):
    import torch
    from safetensors.torch import load_file

    before = {k: v.clone() for k, v in toy_model.state_dict().items()}
    params = {"rank": 2, "steps": 2, "targets": ["o", "down"], "lr": 0.001}
    a = generate("toy", params, 8, tmp_path / "a")
    b = generate("toy", params, 8, tmp_path / "b")
    assert a.id == b.id and a.delta_fro_norm > 0
    wa = load_file(tmp_path / "a/lora/adapter_model.safetensors")
    wb = load_file(tmp_path / "b/lora/adapter_model.safetensors")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    for model in fake_sft:
        for name, value in model.state_dict().items():
            if "lora_" not in name:
                original_name = name.replace(".base_layer", "")
                assert torch.equal(value, before[original_name])
        assert all(
            not p.requires_grad for name, p in model.named_parameters() if "lora_" not in name
        )
    assert all("o_proj" in name or "down_proj" in name for name in a.touched_tensors)
    assert json.loads((tmp_path / "a/manifest.json").read_text())["seed"] == 8


@pytest.mark.parametrize("params", [{"steps": 0}, {"rank": 0}, {"batch_size": 0}])
def test_sft_invalid_parameters_fail_before_loading(params, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        generate("unused", params, 0, tmp_path)
