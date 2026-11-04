"""Edit kernels: per-layer weight modifications used by the zoo builder.

Variant identity: this reproduces Heretic 2.0's *classic abliteration modifier* weight kernel and
direction interpolation (``heretic/modifiers/abliteration.py::Abliteration.modify_model``) with
``row_normalization="none"`` and ``orthogonalize_direction=True`` (projected abliteration), i.e.
``W' = W - weight * v (v^T W)`` with no row normalisation. Heretic's default is ``row_normalization
="full"``; that default is NOT implemented here (see ``KERNEL_VARIANT``).
"""

from __future__ import annotations

import math

import torch

from wsp.edits.refusal import interpolate_direction
from wsp.zoo.random_dir import ablate_direction

KERNEL_VARIANT = "heretic2-classic-projected-rownorm-none"


def layer_weight_profile(
    layers: list[int],
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
) -> dict[int, float]:
    """Heretic finite-support kernel.

    Layers farther than ``min_weight_distance`` from ``max_weight_position`` are absent from the
    result (Heretic skips them); the rest interpolate linearly from ``max_weight`` to ``min_weight``.
    ``min_weight`` is the absolute floor (Heretic's sampler expresses it as a fraction of
    ``max_weight`` and multiplies before building the ``WeightDistribution``).
    """
    profile: dict[int, float] = {}
    for layer in layers:
        distance = abs(layer - max_weight_position)
        if distance > min_weight_distance:
            continue
        profile[layer] = max_weight + (distance / min_weight_distance) * (min_weight - max_weight)
    return profile


def kernel_direction(
    refusal_dirs: torch.Tensor, direction_index: float | None, layer: int
) -> torch.Tensor:
    """Heretic direction choice: global interpolated direction, or per-layer (index None)."""
    if direction_index is None:
        return refusal_dirs[layer + 1]  # row 0 is the embedding residual
    return interpolate_direction(refusal_dirs, direction_index)


@torch.no_grad()
def heretic_abliteration_kernel(
    state_dict: dict[str, torch.Tensor],
    refusal_dirs: torch.Tensor,
    layer_tensors: dict[int, list[str]],
    direction_index: float | None,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Apply a Heretic-style per-layer abliteration kernel.

    Args:
        state_dict: full model state dictionary.
        refusal_dirs: (n_layers + 1, d_model) residual directions; row 0 is the embedding.
        layer_tensors: mapping layer_index -> list of state-dict keys to edit in that layer.
        direction_index: fractional Heretic layer index (-1 is embeddings) or None for per-layer.
        max_weight: peak ablation weight.
        max_weight_position: layer at which the weight peaks.
        min_weight: floor weight at ``min_weight_distance``.
        min_weight_distance: support radius; layers beyond it are not edited.

    Returns:
        (new_state_dict, touched_tensor_names).
    """
    new_state = dict(state_dict)
    layers = sorted(layer_tensors.keys())
    profile = layer_weight_profile(
        layers, max_weight, max_weight_position, min_weight, min_weight_distance
    )
    touched: list[str] = []
    for layer in layers:
        # Heretic: outside the radius -> skip; weight == 0 -> skip.
        if layer not in profile or profile[layer] == 0 or math.isnan(profile[layer]):
            continue
        d = kernel_direction(refusal_dirs, direction_index, layer)
        for key in layer_tensors[layer]:
            if key not in new_state:
                raise KeyError(f"tensor {key} not found in state_dict")
            new_state[key] = ablate_direction(new_state[key], d, weight=profile[layer])
            touched.append(key)
    return new_state, touched
