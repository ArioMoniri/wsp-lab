"""Mean pooling into 522 stats and 56 singular-spectrum values by default.

Stats order is global, then q/k/v/o/gate/up/down, each in layer-bucket order.
Spectrum order is q/k/v/o/gate/up/down. Empty groups are zero.
Linear-attention ``out_proj`` (Qwen3.5 GatedDeltaNet layers) pools with ``o_proj``.
"""

import re
from collections.abc import Sequence
from typing import Any

import torch

TYPES = ("q", "k", "v", "o", "gate", "up", "down")


def pool_features(
    records: Sequence[dict[str, Any]], n_layers: int, buckets: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool per-tensor features; n_layers is the base model's total layer count."""
    if n_layers <= 0 or buckets <= 0:
        raise ValueError("n_layers and buckets must be positive")
    stats = [[] for _ in range(1 + len(TYPES) * buckets)]
    spectra = [[] for _ in TYPES]
    for record in records:
        name = record["name"]
        stat = record["stats"].float()
        stats[0].append(stat)
        match = re.search(r"\.layers\.(\d+)\.", name)
        kind = re.search(r"\.(q|k|v|o|gate|up|down|out)_proj\.", name)
        if not match or not kind:
            continue
        layer = int(match.group(1))
        if layer >= n_layers:
            raise ValueError(f"layer {layer} outside model with {n_layers} layers")
        index = TYPES.index("o" if kind.group(1) == "out" else kind.group(1))
        bucket = min(layer * buckets // n_layers, buckets - 1)
        stats[1 + index * buckets + bucket].append(stat)
        spectra[index].append(record["svd"][:, 0].float())

    def means(groups: list[list[torch.Tensor]], size: int) -> torch.Tensor:
        return torch.cat([torch.stack(g).mean(0) if g else torch.zeros(size) for g in groups])

    return means(stats, 18), means(spectra, 8)
