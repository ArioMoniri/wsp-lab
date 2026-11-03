"""Edit kernels: per-layer weight modifications used by the zoo builder."""

from __future__ import annotations

import torch

from wsp.edits.refusal import interpolate_direction
from wsp.zoo.random_dir import ablate_direction


def layer_weight_profile(
    layers: list[int],
    max_weight: float,
    max_weight_position: int,
    min_weight: float,
    min_weight_distance: int,
) -> dict[int, float]:
    """Linear taper from max_weight at max_weight_position to min_weight beyond min_weight_distance."""
    profile: dict[int, float] = {}
    for layer in layers:
        dist = abs(layer - max_weight_position)
        if dist >= min_weight_distance:
            profile[layer] = min_weight
        else:
            profile[layer] = max_weight - (dist / min_weight_distance) * (max_weight - min_weight)
    return profile


@torch.no_grad()
def heretic_abliteration_kernel(
    state_dict: dict[str, torch.Tensor],
    refusal_dirs: torch.Tensor | dict[int, torch.Tensor],
    layer_tensors: dict[int, list[str]],
    direction_index: float,
    max_weight: float,
    max_weight_position: int,
    min_weight: float,
    min_weight_distance: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Apply a Heretic-style per-layer abliteration kernel.

    Args:
        state_dict: full model state dictionary.
        refusal_dirs: residual directions including the embedding direction.
        layer_tensors: mapping layer_index -> list of state-dict keys to edit in that layer.
        direction_index: fractional Heretic layer index, with -1 for embeddings.
        max_weight: peak ablation weight.
        max_weight_position: layer at which the weight peaks.
        min_weight: floor weight applied far from the peak.
        min_weight_distance: number of layers over which to taper from max to min.

    Returns:
        (new_state_dict, touched_tensor_names).
    """
    new_state = dict(state_dict)
    layers = sorted(layer_tensors.keys())
    profile = layer_weight_profile(
        layers=layers,
        max_weight=max_weight,
        max_weight_position=max_weight_position,
        min_weight=min_weight,
        min_weight_distance=min_weight_distance,
    )
    touched: list[str] = []
    for layer in layers:
        if isinstance(refusal_dirs, dict):
            dirs = refusal_dirs[layer]
            d = dirs[int(direction_index) % dirs.shape[0]]
        else:
            d = interpolate_direction(refusal_dirs, direction_index)
        weight = profile[layer]
        for key in layer_tensors[layer]:
            if key not in new_state:
                raise KeyError(f"tensor {key} not found in state_dict")
            new_state[key] = ablate_direction(new_state[key], d, weight=weight)
            touched.append(key)
    return new_state, touched
