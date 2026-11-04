"""Load an edited model from a base model plus a zoo delta."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from wsp.schema import EditDescriptor

LORA_ADAPTER_FILE = "lora/adapter_model.safetensors"


def entry_has_artifacts(entry_dir: str | Path) -> bool:
    """True if the zoo entry has an edit.json plus a dense delta or a LoRA adapter."""
    d = Path(entry_dir)
    return (d / "edit.json").exists() and (
        (d / "delta.safetensors").exists() or (d / LORA_ADAPTER_FILE).exists()
    )


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
        hits = [a for a in available if a.endswith(suffix) and "visual" not in a]
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
    tensors = load_file(lora_dir / "adapter_model.safetensors")
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
    if (d / "delta.safetensors").exists():
        return load_file(d / "delta.safetensors")
    return lora_dense_delta(d / "lora", state_keys)


def load_edited_model(base: Any, zoo_entry_dir: str | Path) -> dict[str, torch.Tensor]:
    """Reconstruct an edited model state dict = base + delta.

    Args:
        base: a state dict dict, a path to a safetensors/bin file, or a model object with
            ``.state_dict()``.
        zoo_entry_dir: directory containing ``edit.json`` and either ``delta.safetensors`` or a
            PEFT LoRA adapter under ``lora/`` (dense delta = B @ A * alpha / r).

    Returns:
        A state dictionary with the delta added to the base tensors. The result retains the
        base tensor's dtype/device for tensors present in both base and delta.
    """
    entry_dir = Path(zoo_entry_dir)
    # Validate the descriptor parses; touched_tensors/delta_fro_norm are checked implicitly.
    EditDescriptor.from_json((entry_dir / "edit.json").read_text())

    if isinstance(base, (str, Path)):
        base_state = load_file(base)
    elif isinstance(base, dict):
        base_state = base
    elif hasattr(base, "state_dict"):
        base_state = base.state_dict()
    else:
        raise TypeError(f"unsupported base type: {type(base)}")
    delta = load_entry_delta(entry_dir, base_state.keys())

    state = dict(base_state)
    for key, d in delta.items():
        if key in state:
            target = state[key]
            d_on_device = d.to(target.device)
            state[key] = (target.to(d_on_device.dtype) + d_on_device).to(target.dtype)
        else:
            state[key] = d
    return state
