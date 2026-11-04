"""Toy zoo for P3 tests: tiny bf16 base state plus dense, LoRA and chain entries on disk."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from wsp.schema import EditDescriptor

N_LAYERS = 4
D, FF = 16, 32
BASE_MODEL = "toy/model"
ATTN = ("q_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def toy_base_state(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    state = {"model.embed_tokens.weight": torch.randn(40, D, generator=g).bfloat16()}
    for i in range(N_LAYERS):
        for n in ATTN:
            state[f"model.layers.{i}.self_attn.{n}.weight"] = (
                torch.randn(D, D, generator=g) * 0.05
            ).bfloat16()
        shapes = {"gate_proj": (FF, D), "up_proj": (FF, D), "down_proj": (D, FF)}
        for n in MLP:
            state[f"model.layers.{i}.mlp.{n}.weight"] = (
                torch.randn(*shapes[n], generator=g) * 0.05
            ).bfloat16()
    return state


def _descriptor(family, seed, parent_id=None, **params) -> EditDescriptor:
    return EditDescriptor(
        family=family,
        params={"k": seed, **params},
        seed=seed,
        base_model=BASE_MODEL,
        parent_id=parent_id,
    )


def write_dense(zoo: Path, base, family, seed, scale, layers=(0, 2), parent_id=None) -> str:
    g = torch.Generator().manual_seed(1000 + seed)
    delta = {}
    for i in layers:
        for n in ("mlp.down_proj", "self_attn.o_proj"):
            key = f"model.layers.{i}.{n}.weight"
            delta[key] = (torch.randn(base[key].shape, generator=g) * scale).bfloat16()
    desc = _descriptor(family, seed, parent_id)
    desc.touched_tensors = sorted(delta)
    desc.delta_fro_norm = float(sum((t.float() ** 2).sum() for t in delta.values()) ** 0.5)
    d = zoo / desc.id
    d.mkdir(parents=True)
    (d / "edit.json").write_text(desc.to_json())
    save_file(delta, d / "delta.safetensors")
    return desc.id


def write_lora(zoo: Path, base, seed, scale, rank=4, alpha=8, refactor=None) -> str:
    g = torch.Generator().manual_seed(2000 + seed)
    tensors = {}
    layers = (1, 3)
    for i in layers:
        key = f"model.layers.{i}.mlp.up_proj.weight"
        out_dim, in_dim = base[key].shape
        B = torch.randn(out_dim, rank, generator=g) * scale
        A = torch.randn(rank, in_dim, generator=g)
        if refactor is not None:
            G = refactor
            B, A = B @ G, torch.linalg.inv(G) @ A
        prefix = f"base_model.model.model.language_model.layers.{i}.mlp.up_proj"
        tensors[f"{prefix}.lora_A.weight"] = A.contiguous()
        tensors[f"{prefix}.lora_B.weight"] = B.contiguous()
    desc = _descriptor("lora_sft", seed, rank=rank)
    desc.params["delta_format"] = "lora"
    d = zoo / desc.id
    (d / "lora").mkdir(parents=True)
    (d / "edit.json").write_text(desc.to_json())
    save_file(tensors, d / "lora" / "adapter_model.safetensors")
    cfg = {"lora_alpha": alpha, "r": rank, "peft_type": "LORA", "use_rslora": False}
    (d / "lora" / "adapter_config.json").write_text(json.dumps(cfg))
    return desc.id


def build_toy_zoo(zoo: Path, base) -> list[str]:
    """12 entries: 3 random_dir, 3 abliteration, 3 lora_sft, 3-step chain; manifest.jsonl written."""
    zoo.mkdir(parents=True, exist_ok=True)
    ids = []
    for s in range(3):
        ids.append(write_dense(zoo, base, "random_dir", s, 0.004 * (s + 1)))
    for s in range(3):
        ids.append(write_dense(zoo, base, "abliteration", s, 0.01 * (s + 1), layers=(1, 2, 3)))
    for s in range(3):
        ids.append(write_lora(zoo, base, s, 0.05 * (s + 1)))
    parent = None
    for s in range(3):
        parent = write_dense(zoo, base, "chain", s, 0.006 * (s + 1), parent_id=parent)
        ids.append(parent)
    rows = []
    for i in ids:
        e = json.loads((zoo / i / "edit.json").read_text())
        rows.append(
            {"id": i, "family": e["family"], "seed": e["seed"], "parent_id": e["parent_id"]}
        )
    (zoo / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ids


def tiny_qwen35_shapes():
    """Every parameter of a tiny Qwen3.5 hybrid (GatedDeltaNet + full attention) model."""
    import transformers
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    assert transformers.__version__
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}
