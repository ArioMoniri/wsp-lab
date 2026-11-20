"""Gemma 3 inert controls (study M), without editing ``wm/controls.py``.

* :func:`compensated_pairs`: power-of-two compensated scales, (up×2, down×½) per MLP layer and (v×2, o×½) per attention
  layer, full or sliding. Exact through operations linear in the scaled input (Gemma 3 has no v-norm). This is the
  Gemma 3 inert spec (bitwise in Gate 0).
* :func:`k_norm_scales`: k_proj×2 on attention layers. Descriptive only: it is not exact on Gemma 3 and is removed from
  the inert spec (RULING-wmm-gate0 M3).
"""

from __future__ import annotations

from wsp.wm import actions as A
from wsp.wm.controls import NotInertError
from wsp.wm.wmm.roles import ATTENTION_TYPES


def compensated_pairs(rolemap, mlp_layers=(), attention_layers=()) -> list:
    out = [
        [
            A.scale(rolemap.name(L, "up"), factor=2.0, dose=1.0),
            A.scale(rolemap.name(L, "down"), factor=0.5, dose=1.0),
        ]
        for L in mlp_layers
    ]
    for L in attention_layers:
        if rolemap.layer_types[L] not in ATTENTION_TYPES:
            raise NotInertError(f"layer {L}: not a softmax-attention layer")
        out.append(
            [
                A.scale(rolemap.name(L, "v"), factor=2.0, dose=1.0),
                A.scale(rolemap.name(L, "o"), factor=0.5, dose=1.0),
            ]
        )
    return out


def k_norm_scales(rolemap, layers, factor=2.0) -> list:
    out = [
        A.scale(rolemap.name(L, "k"), factor=factor, dose=1.0)
        for L in layers
        if rolemap.layer_types[L] in ATTENTION_TYPES
    ]
    if not out:
        raise NotInertError("no attention layer among the requested layers")
    return out
