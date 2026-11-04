"""Regression coverage for evaluating zoo LoRA entries."""

import copy
import json
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
    tensors = {
        k.replace("base_model.model.layers.", "base_model.model.model.language_model.layers."): v
        for k, v in tensors.items()
    }
    save_file(tensors, lora / "adapter_model.safetensors")
    for i in range(2):
        prefix = f"base_model.model.model.language_model.layers.{i}.attn.o_proj"
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

    root = Path(__file__).resolve().parents[1] / "data" / "zoo"
    entries = [
        p.parent.parent
        for p in root.rglob("adapter_model.safetensors")
        if "lora_sft" in str(p) and (p.parent.parent / "edit.json").exists()
    ]
    if not entries:
        pytest.skip("no lora_sft smoke adapter under data/zoo")
    entry = entries[0]
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
