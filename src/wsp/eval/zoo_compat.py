"""Temporary copy of zoo apply helpers; replace with wsp.zoo.apply after the zoo merge."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file


def resolve_state_key(key: str, available: Iterable[str]) -> str | None:
    """Map a PEFT-derived weight name onto a state-dict key.

    Handles the ``model.language_model.`` prefix that multimodal checkpoints (e.g. Qwen3.5 loaded
    through AutoModelForImageTextToText, as Heretic does) add relative to the text-only
    CausalLM state dict, falling back to a unique suffix match.
    """
    available = set(available)
    candidates = [
        key,
        key.replace("model.language_model.", "model.", 1),
        key.replace("model.", "model.language_model.", 1),
    ]
    for c in candidates:
        if c in available:
            return c
    suffix = "." + key.split("layers.", 1)[-1] if "layers." in key else None
    if suffix:
        hits = [
            a
            for a in available
            if a.endswith(suffix)
            and a.startswith(("model.layers.", "model.language_model.layers."))
        ]
        if len(hits) == 1:
            return hits[0]
    return None


def lora_dense_delta(
    lora_dir: str | Path, state_keys: Iterable[str] | None = None
) -> dict[str, torch.Tensor]:
    """Dense per-tensor delta (B @ A * scaling, float32) of a PEFT LoRA adapter directory.

    Keys are state-dict weight names; if ``state_keys`` is given, adapter modules are mapped onto
    them with :func:`resolve_state_key` and unmappable modules raise (never silently dropped).
    """
    lora_dir = Path(lora_dir)
    cfg = json.loads((lora_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        raise NotImplementedError("DoRA adapters are not supported")
    if (
        cfg.get("fan_in_fan_out")
        or cfg.get("bias", "none") != "none"
        or cfg.get("modules_to_save")
        or cfg.get("lora_bias")
    ):
        raise NotImplementedError("unsupported LoRA orientation, bias or modules_to_save")
    tensors = load_file(lora_dir / "adapter_model.safetensors")
    for name in tensors:
        if ".lora_A." in name:
            partner = name.replace(".lora_A.", ".lora_B.")
        elif ".lora_B." in name:
            partner = name.replace(".lora_B.", ".lora_A.")
        else:
            raise NotImplementedError(f"unsupported adapter tensor: {name}")
        if partner not in tensors:
            raise ValueError(f"unpaired adapter tensor: {name}")
    state_keys = list(state_keys) if state_keys is not None else None
    alpha_pattern = cfg.get("alpha_pattern") or {}
    out: dict[str, torch.Tensor] = {}
    for name, A in tensors.items():
        if ".lora_A." not in name:
            continue
        module = name.split(".lora_A.")[0].removeprefix("base_model.model.")
        B = tensors[name.replace(".lora_A.", ".lora_B.")]
        rank = A.shape[0]
        alpha = next(
            (a for pat, a in alpha_pattern.items() if module.endswith(pat)), cfg["lora_alpha"]
        )
        scaling = alpha / (rank**0.5 if cfg.get("use_rslora") else rank)
        key = f"{module}.weight"
        if state_keys is not None:
            resolved = resolve_state_key(key, state_keys)
            if resolved is None:
                raise KeyError(f"LoRA module {key} has no matching tensor in the base state")
            key = resolved
        out[key] = (B.float() @ A.float()) * scaling
    if not out:
        raise ValueError(f"no LoRA weights found in {lora_dir}")
    return out


def load_entry_delta(
    entry_dir: str | Path, state_keys: Iterable[str] | None = None
) -> dict[str, torch.Tensor]:
    """Delta tensors of a zoo entry: dense ``delta.safetensors`` or LoRA ``lora/`` adapter."""
    d = Path(entry_dir)
    descriptor = d / "edit.json"
    fmt = json.loads(descriptor.read_text()).get("delta_format") if descriptor.exists() else None
    if fmt is None:
        warnings.warn("missing delta_format; inferring from files", UserWarning, stacklevel=2)
        fmt = "dense" if (d / "delta.safetensors").exists() else "lora"
    if fmt == "dense":
        return load_file(d / "delta.safetensors")
    if fmt == "lora":
        return lora_dense_delta(d / "lora", state_keys)
    raise ValueError(f"unsupported delta_format: {fmt}")
