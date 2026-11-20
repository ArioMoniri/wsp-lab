"""Gemma 3 role map, frozen-tensor rule and text-only guard (study M), without editing ``wm/actions.py``/``wm/hf.py``.

* :class:`GemmaRoleMap` is an :class:`wsp.wm.actions.RoleMap` (so ``EditSession``, ``lora.writer_targets`` and
  ``step_tokens`` accept it) that knows ``sliding_attention`` layers (q/k/v/o like full attention) and defaults
  ``tie_word_embeddings`` to True when the config omits it (MedGemma's config does; Gemma 3 ties lm_head).
* :func:`is_frozen` adds the Gemma 3 vision tower and projector to the Qwen ``visual``/``mtp`` rule;
  :func:`assert_text_only` raises if any such tensor was loaded.
"""

from __future__ import annotations

import dataclasses
import re

from wsp.wm import actions as A

ATTENTION_TYPES = ("full_attention", "sliding_attention")
GEMMA_TYPES = ("gemma3", "gemma3_text")
FROZEN_PATTERNS = A.FROZEN_PATTERNS + (
    re.compile(r"(^|\.)vision_tower\."),
    re.compile(r"(^|\.)multi_modal_projector\."),
)


def is_frozen(name: str) -> bool:
    return any(p.search(name) for p in FROZEN_PATTERNS)


def assert_text_only(names) -> None:
    bad = [n for n in names if is_frozen(n)]
    if bad:
        raise RuntimeError(
            f"vision/MTP tensors were loaded (e.g. {bad[0]}); the target must be text-only"
        )


@dataclasses.dataclass(frozen=True)
class GemmaRoleMap(A.RoleMap):
    @classmethod
    def from_config(cls, config: dict, prefix: str = "model.layers.") -> GemmaRoleMap:
        text = config.get("text_config", config)
        mt = config.get("model_type") or text.get("model_type")
        if mt not in GEMMA_TYPES:
            raise ValueError(f"not a Gemma 3 config (model_type {mt!r})")
        n = int(text["num_hidden_layers"])
        types = tuple(text.get("layer_types") or ["full_attention"] * n)
        if len(types) != n:
            raise ValueError(f"layer_types has {len(types)} entries for {n} layers")
        unknown = set(types) - set(ATTENTION_TYPES)
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        tied = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", True)))
        return cls(
            n,
            int(text["hidden_size"]),
            int(text["intermediate_size"]),
            int(text["vocab_size"]),
            types,
            tied,
            prefix,
        )

    def name(self, layer: int, role: str) -> str:
        if not 0 <= layer < self.n_layers:
            raise KeyError(f"layer {layer} outside 0..{self.n_layers - 1}")
        if role in ("o", "q", "k", "v"):
            path = f"self_attn.{role}_proj"
        elif role in ("down", "up", "gate"):
            path = f"mlp.{role}_proj"
        else:
            raise KeyError(f"no role {role!r} in layer {layer} ({self.layer_types[layer]})")
        return f"{self.prefix}{layer}.{path}.weight"

    def layer_of(self, name: str) -> int | None:
        m = re.search(r"\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m and not is_frozen(name) else None
