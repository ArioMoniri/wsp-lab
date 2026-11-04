"""Mean pooling of per-tensor features into fixed-width vectors (``feature_version`` 4).

Every changed tensor lands in exactly one per-type pool; nothing is global-only except the global
stats group itself. Types (``TYPES``), in pool order:

  q k v o gate up down   full-attention and MLP projections. The output projection of any other token
                         mixer pools as ``o`` (Qwen3.5 ``linear_attn.out_proj``, LFM3 ``conv.out_proj``).
                         LFM3 MLP ``feed_forward.w1/w3/w2`` pool as ``gate/up/down`` (SiLU gate, up, down).
  lin_qkv                fused input projection of a linear mixer: Qwen3.5 GatedDeltaNet
                         ``linear_attn.in_proj_qkv``, LFM3 short-conv ``conv.in_proj`` (B, C, x gates)
  lin_z                  ``linear_attn.in_proj_z`` (output gate)
  lin_b, lin_a           ``linear_attn.in_proj_b`` / ``in_proj_a`` (delta-rule beta / decay inputs)
  lin_state              mixer state parameters: ``A_log``, ``dt_bias``, ``conv1d``, ``linear_attn.norm``,
                         LFM3 depthwise ``conv.conv``
  attn_gate              LFM3 per-head attention output gate ``self_attn.gate_proj`` (NOT the MLP gate)
  norm                   all layer norms (incl. sandwich post-norms, q_norm / k_norm, final norm,
                         LFM3 ``embedding_norm`` which is its final norm)
  embed                  ``embed_tokens`` and ``lm_head``
  aux                    towers outside the language stack: ``model.visual.*`` and ``mtp.*`` (never
                         assigned a layer bucket, even though ``mtp.layers.0`` looks like layer 0)
  other                  any tensor name no rule recognises (catch-all, so no change is lost)

Stats layout: the global mean, then each type in ``TYPES`` order as ``buckets`` layer-depth buckets
(18 values each). Spectrum layout: the mean top-8 singular values per type in ``TYPES`` order. Tensors
outside ``.layers.<i>.`` use bucket 0. Empty groups are zero.

Version history: 3 added hybrid/norm/embed types; 4 added ``attn_gate`` and ``aux`` (inserted before
``other``) and the LFM3 / Gemma-3 name rules. A v3 vector is NOT a prefix of a v4 vector.
"""

import re
from collections.abc import Sequence
from typing import Any

import torch

TYPES = (
    "q",
    "k",
    "v",
    "o",
    "gate",
    "up",
    "down",
    "lin_qkv",
    "lin_z",
    "lin_b",
    "lin_a",
    "lin_state",
    "attn_gate",
    "norm",
    "embed",
    "aux",
    "other",
)
N_STATS, N_SPECTRUM = 18, 8

# (pattern, type); first match wins. Patterns are searched against the full tensor name.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p), t)
    for p, t in (
        (r"(^|\.)(visual|mtp)\.", "aux"),
        (r"\.self_attn\.gate_proj\.", "attn_gate"),
        (r"\.linear_attn\.in_proj_qkv\.", "lin_qkv"),
        (r"\.conv\.in_proj\.", "lin_qkv"),
        (r"\.linear_attn\.in_proj_z\.", "lin_z"),
        (r"\.linear_attn\.in_proj_b\.", "lin_b"),
        (r"\.linear_attn\.in_proj_a\.", "lin_a"),
        (r"\.(linear_attn|conv)\.out_proj\.", "o"),
        (r"\.linear_attn\.(A_log|dt_bias|conv1d\.|norm\.)", "lin_state"),
        (r"\.conv\.conv\.", "lin_state"),
        (r"\.feed_forward\.w1\.", "gate"),
        (r"\.feed_forward\.w3\.", "up"),
        (r"\.feed_forward\.w2\.", "down"),
        (r"\.(q|k|v|o|gate|up|down)_proj\.", ""),  # type taken from the captured letter below
        (r"(^|\.)(embed_tokens|lm_head)\.", "embed"),
        (r"(^|\.)\w*norm\.", "norm"),
    )
)
_PROJ = re.compile(r"\.(q|k|v|o|gate|up|down)_proj\.")


def classify(name: str) -> tuple[str, int | None]:
    """``(type, layer index or None)`` of a tensor name; ``other`` when no rule matches."""
    kind = "other"
    for pattern, t in _RULES:
        if pattern.search(name):
            kind = t or _PROJ.search(name).group(1)  # type: ignore[union-attr]
            break
    layer = None if kind == "aux" else re.search(r"\.layers\.(\d+)\.", name)
    return kind, int(layer.group(1)) if layer else None


def pooled_dims(buckets: int = 4) -> tuple[int, int]:
    """Widths of the pooled stats and spectrum vectors."""
    return N_STATS * (1 + len(TYPES) * buckets), N_SPECTRUM * len(TYPES)


def pool_features(
    records: Sequence[dict[str, Any]], n_layers: int, buckets: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool per-tensor features; n_layers is the base model's total layer count."""
    if n_layers <= 0 or buckets <= 0:
        raise ValueError("n_layers and buckets must be positive")
    stats = [[] for _ in range(1 + len(TYPES) * buckets)]
    spectra = [[] for _ in TYPES]
    for record in records:
        stat = record["stats"].float()
        stats[0].append(stat)
        kind, layer = classify(record["name"])
        if layer is not None and layer >= n_layers:
            raise ValueError(f"layer {layer} outside model with {n_layers} layers")
        index = TYPES.index(kind)
        bucket = 0 if layer is None else min(layer * buckets // n_layers, buckets - 1)
        stats[1 + index * buckets + bucket].append(stat)
        spectra[index].append(record["svd"][:, 0].float())

    def means(groups: list[list[torch.Tensor]], size: int) -> torch.Tensor:
        return torch.cat([torch.stack(g).mean(0) if g else torch.zeros(size) for g in groups])

    return means(stats, N_STATS), means(spectra, N_SPECTRUM)
