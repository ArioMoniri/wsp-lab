"""Load an edited model from a base model plus a zoo delta."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from wsp.schema import EditDescriptor


def load_edited_model(base: Any, zoo_entry_dir: str | Path) -> dict[str, torch.Tensor]:
    """Reconstruct an edited model state dict = base + delta.

    Args:
        base: a state dict dict, a path to a safetensors/bin file, or a model object with
            ``.state_dict()``.
        zoo_entry_dir: directory containing ``edit.json`` and ``delta.safetensors``.

    Returns:
        A state dictionary with the delta added to the base tensors. The result retains the
        base tensor's dtype/device for tensors present in both base and delta.
    """
    entry_dir = Path(zoo_entry_dir)
    # Validate the descriptor parses; touched_tensors/delta_fro_norm are checked implicitly.
    EditDescriptor.from_json((entry_dir / "edit.json").read_text())
    delta = load_file(entry_dir / "delta.safetensors")

    if isinstance(base, (str, Path)):
        base_state = load_file(base)
    elif isinstance(base, dict):
        base_state = base
    elif hasattr(base, "state_dict"):
        base_state = base.state_dict()
    else:
        raise TypeError(f"unsupported base type: {type(base)}")

    state = dict(base_state)
    for key, d in delta.items():
        if key in state:
            target = state[key]
            d_on_device = d.to(target.device)
            state[key] = (target.to(d_on_device.dtype) + d_on_device).to(target.dtype)
        else:
            state[key] = d
    return state
