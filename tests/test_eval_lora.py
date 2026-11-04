"""Regression coverage for evaluating zoo LoRA entries."""

import copy
import json
import os
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
    with torch.inference_mode():
        base = runner.model(**ids).logits.float().cpu()
        apply_zoo_entry(runner.model, entry)
        edited = runner.model(**ids).logits.float().cpu()
    assert not torch.equal(base, edited)


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
    assert edited.entry_hashes and edited.rounding_stats


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
