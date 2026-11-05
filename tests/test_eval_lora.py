"""Regression coverage for evaluating zoo LoRA entries."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from wsp.eval.run import apply_zoo_entry


def test_lora_eval_matches_manual_delta(tmp_path, toy_model):
    from peft import LoraConfig, get_peft_model

    original = copy.deepcopy(toy_model)
    manual = copy.deepcopy(toy_model)
    adapter = get_peft_model(toy_model, LoraConfig(r=2, lora_alpha=4, target_modules=["o_proj"]))
    for name, param in adapter.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            torch.nn.init.constant_(param, 0.1)
    lora = tmp_path / "lora"
    adapter.save_pretrained(lora)
    tensors = load_file(lora / "adapter_model.safetensors")
    save_file(tensors, lora / "adapter_model.safetensors")
    for i in range(2):
        prefix = f"base_model.model.layers.{i}.attn.o_proj"
        a, b = tensors[prefix + ".lora_A.weight"], tensors[prefix + ".lora_B.weight"]
        with torch.no_grad():
            manual.layers[i]["attn"]["o_proj"].weight.add_(b @ a * 2)
    ids = torch.tensor([[1, 2, 3, 4]])
    base_loss = original(ids, labels=ids).loss
    apply_zoo_entry(original, tmp_path)
    edited_loss = original(ids, labels=ids).loss
    assert not torch.allclose(edited_loss, base_loss)
    torch.testing.assert_close(edited_loss, manual(ids, labels=ids).loss)

    bad = {
        k.replace("layers.0.attn.o_proj", "layers.99.attn.o_proj"): v for k, v in tensors.items()
    }
    save_file(bad, lora / "adapter_model.safetensors")
    with pytest.raises(KeyError, match="no matching tensor"):
        apply_zoo_entry(original, tmp_path)


@pytest.mark.gpu
def test_real_lora_smoke_changes_output(cuda):
    from wsp.eval.run import HFRunner

    default = Path("/home/ubuntu/projects/wsp-lab/data/zoo/qwen3.5-0.8b/04ce1ebc4749f1c8")
    configured = os.environ.get("WSP_REAL_LORA_ENTRY")
    entry = Path(configured) if configured is not None else default
    if configured is not None:
        assert entry.is_dir(), f"WSP_REAL_LORA_ENTRY does not exist: {entry}"
    elif not entry.is_dir():
        pytest.skip("default real LoRA entry unavailable")
    descriptor = json.loads((entry / "edit.json").read_text())
    runner = HFRunner(
        descriptor["base_model"], {}, revision=descriptor.get("base_revision", "main")
    )
    ids = runner.tokenizer("The answer is", return_tensors="pt").to(runner.model.device)
    from wsp.eval.zoo_compat import resolve_state_key

    tensors = load_file(entry / "lora/adapter_model.safetensors")
    config = json.loads((entry / "lora/adapter_config.json").read_text())
    state = runner.model.state_dict()
    originals = {}
    with torch.inference_mode():
        base = runner.model(**ids).logits.float().cpu()
        for name, a in tensors.items():
            if ".lora_A." not in name:
                continue
            module = name.split(".lora_A.")[0].removeprefix("base_model.model.")
            key = resolve_state_key(module + ".weight", state.keys())
            assert key is not None
            b = tensors[name.replace(".lora_A.", ".lora_B.")]
            alpha = next(
                (v for k, v in (config.get("alpha_pattern") or {}).items() if module.endswith(k)),
                config["lora_alpha"],
            )
            scale = alpha / (a.shape[0] ** 0.5 if config.get("use_rslora") else a.shape[0])
            originals[key] = state[key].clone()
            delta = (b.float() @ a.float()).to(state[key].device) * scale
            state[key].copy_((state[key].float() + delta).to(state[key].dtype))
        manual = runner.model(**ids).logits.float().cpu()
        for key, value in originals.items():
            state[key].copy_(value)
        stats = apply_zoo_entry(runner.model, entry)
        edited = runner.model(**ids).logits.float().cpu()
    assert len(originals) == sum(".lora_A." in name for name in tensors)
    assert len(stats) == len(originals)
    assert sum(not torch.equal(state[key], value) for key, value in originals.items()) == len(
        originals
    )
    assert not torch.equal(base, edited)
    torch.testing.assert_close(edited, manual)


def test_hfrunner_cpu_lora(tmp_path, monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    from wsp.eval.run import HFRunner

    model_dir = tmp_path / "model"
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=2, n_head=2)
    )
    model.save_pretrained(model_dir)
    adapter = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=6, target_modules=["c_attn"], fan_in_fan_out=True)
    )
    for name, param in adapter.named_parameters():
        if "lora_" in name:
            torch.nn.init.constant_(param, 0.1)
    entry = tmp_path / "entry"
    adapter.save_pretrained(entry / "lora")
    config_path = entry / "lora" / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["fan_in_fan_out"] = False
    config_path.write_text(json.dumps(config))
    # GPT-2 Conv1D is transposed; use a square projection to exercise a normal weight layout.
    tensors_path = entry / "lora" / "adapter_model.safetensors"
    tensors = load_file(tensors_path)
    for key in list(tensors):
        if ".lora_B." in key:
            tensors[key] = tensors[key][:16]
        new_key = key.replace("c_attn", "c_proj")
        tensors[new_key] = tensors.pop(key)
    save_file(tensors, tensors_path)
    (entry / "edit.json").write_text(
        json.dumps({"id": "tiny", "delta_format": "lora", "base_model": str(model_dir)})
    )

    class Tokenizer:
        pad_token_id = 0

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    base = HFRunner(str(model_dir), {})
    edited = HFRunner(str(model_dir), {}, entry)
    key = "transformer.h.0.attn.c_proj.weight"
    expected = (base.model.state_dict()[key].float() + 0.06).bfloat16()
    assert torch.equal(edited.model.state_dict()[key], expected)
    assert edited.edit_id == "tiny"
    assert edited.entry_hashes == {
        str(path.relative_to(entry)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (config_path, tensors_path)
    }
    assert edited.rounding_stats[key]["delta_norm"] > 0
    from wsp.eval import run

    cfg = tmp_path / "config.yaml"
    cfg.write_text("generation:\n  seed: 1\nbenchmarks: {}\n")
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--config",
            str(cfg),
            "--model",
            str(model_dir),
            "--zoo-entry",
            str(entry),
            "--out",
            str(out),
        ],
    )
    run.main()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["edit_id"] == "tiny"
    assert manifest["delta_format"] == "lora"
    assert manifest["entry_hashes"] == edited.entry_hashes
    assert manifest["rounding_stats"] == edited.rounding_stats


def test_rounding_stats_sub_ulp():
    from wsp.eval.run import apply_delta

    model = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    stats = apply_delta(model, {"weight": torch.tensor([[0.001, 0.125]])})["weight"]
    assert stats["fraction_nonzero_lost"] == 0.5
    assert stats["max_abs_lost"] == pytest.approx(0.001)
    assert stats["effective_delta_norm"] == pytest.approx(0.125)


def test_adapter_validation_and_dispatch(tmp_path):
    from wsp.eval.zoo_compat import load_entry_delta, lora_dense_delta, resolve_state_key

    lora = tmp_path / "lora"
    lora.mkdir()
    config = {"lora_alpha": 2, "bias": "none"}
    config_path = lora / "adapter_config.json"
    config_path.write_text(json.dumps(config))
    weights = lora / "adapter_model.safetensors"
    save_file({"x.lora_B.weight": torch.ones(2, 1)}, weights)
    with pytest.raises(ValueError, match="unpaired"):
        lora_dense_delta(lora)
    config_path.write_text(json.dumps(config | {"fan_in_fan_out": True}))
    with pytest.raises(NotImplementedError):
        lora_dense_delta(lora)
    config_path.write_text(json.dumps(config | {"bias": "all"}))
    with pytest.raises(NotImplementedError):
        lora_dense_delta(lora)
    (tmp_path / "edit.json").write_text(json.dumps({"delta_format": "dense"}))
    with pytest.raises(FileNotFoundError):
        load_entry_delta(tmp_path)
    assert resolve_state_key("model.layers.0.foo.weight", ["other.layers.0.foo.weight"]) is None


def test_asymmetric_lora_orientation(tmp_path):
    from wsp.eval.zoo_compat import lora_dense_delta

    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 6}))
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0], [8.0, 9.0]])
    save_file({"x.lora_A.weight": a, "x.lora_B.weight": b}, tmp_path / "adapter_model.safetensors")
    torch.testing.assert_close(lora_dense_delta(tmp_path)["x.weight"], (b @ a) * 3)


def test_rejected_adapter_contents(tmp_path):
    from wsp.eval.zoo_compat import load_entry_delta, lora_dense_delta

    lora = tmp_path / "lora"
    lora.mkdir()
    config = lora / "adapter_config.json"
    config.write_text(json.dumps({"lora_alpha": 2, "modules_to_save": ["head"]}))
    with pytest.raises(NotImplementedError, match="modules_to_save"):
        lora_dense_delta(lora)
    config.write_text(json.dumps({"lora_alpha": 2}))
    for name in ("lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"):
        save_file({f"x.{name}.weight": torch.ones(1)}, lora / "adapter_model.safetensors")
        with pytest.raises(NotImplementedError, match=name):
            lora_dense_delta(lora)
    descriptor = tmp_path / "edit.json"
    descriptor.write_text(json.dumps({"delta_format": "unknown"}))
    with pytest.raises(ValueError, match="unsupported delta_format"):
        load_entry_delta(tmp_path)
    descriptor.write_text(json.dumps({"delta_format": "lora"}))
    save_file({"weight": torch.ones(1)}, tmp_path / "delta.safetensors")
    with pytest.raises(ValueError, match="stray delta.safetensors"):
        load_entry_delta(tmp_path)


def test_eval_apply_path_maps_wrapped_dense_delta_names(tmp_path):
    """The evaluation loader (zoo_compat + apply_zoo_entry) accepts Heretic's wrapped names."""
    from wsp.eval.zoo_compat import load_entry_delta

    model = torch.nn.Sequential()
    model.add_module("layers", torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False)]))
    with torch.no_grad():
        model.layers[0].weight.zero_()
    # state_dict key here is "layers.0.weight"; emulate the Qwen layout with a wrapper module
    outer = torch.nn.Module()
    outer.model = model
    key = "model.layers.0.weight"
    assert key in outer.state_dict()
    (tmp_path / "edit.json").write_text(json.dumps({"delta_format": "dense"}))
    save_file(
        {"model.language_model.layers.0.weight": torch.full((2, 3), 0.5)},
        tmp_path / "delta.safetensors",
    )
    delta = load_entry_delta(tmp_path, outer.state_dict())
    assert list(delta) == [key] and delta.key_mapping == "language_model_prefix_v1"
    apply_zoo_entry(outer, tmp_path)
    assert torch.equal(outer.model.layers[0].weight, torch.full((2, 3), 0.5))
    with pytest.raises(KeyError, match="not in the base state"):
        load_entry_delta(tmp_path, ["x.weight"])
