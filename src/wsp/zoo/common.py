"""Shared checkpoint provenance and PEFT adapter inspection for zoo families."""

import math
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def resolve_base_revision(base_model: str, revision: str | None = None) -> str:
    """Resolve a Hub ref to the actual cached snapshot used for loading."""
    snapshot = Path(
        snapshot_download(
            repo_id=base_model, revision=revision or "main", allow_patterns=["config.json"]
        )
    )
    if snapshot.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        raise ValueError(f"Expected a resolved HF cache snapshot, got {snapshot}")
    return snapshot.name


def metadata_revision(base_model: str, revision: str | None = None) -> str:
    """Commit hash of a Hub ref from repo metadata alone (works for gated repos without file access)."""
    from huggingface_hub import HfApi

    return HfApi().model_info(base_model, revision=revision or "main").sha


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def git_provenance() -> dict:
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    return {
        "git_dirty": dirty,
        "git_tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
    }


def compute_adapter_stats(peft_model: torch.nn.Module) -> tuple[list[str], float]:
    """Inspect PEFT model adapters to find touched tensors and compute Frobenius norm."""
    touched_tensors: list[str] = []
    fro_sq = 0.0

    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            scaling = getattr(module, "scaling", {}).get("default", 1.0)
            delta_W = (B @ A) * scaling
            norm_sq = float((delta_W.float() ** 2).sum().item())
            if norm_sq > 0:
                touched_tensors.append(f"{name.removeprefix('base_model.model.')}.weight")
                fro_sq += norm_sq

    return sorted(touched_tensors), math.sqrt(fro_sq)


# --- delta tensor-name normalisation (D6 addendum 5) -----------------------------------------
#
# Multimodal Qwen3.5 checkpoints (what Heretic loads) name the language stack
# ``model.language_model.<...>``; the text-only CausalLM state dict (HFRunner, feature extraction)
# names the same tensors ``model.<...>``. The first heretic_ara exports kept the wrapped names.
# Stored deltas are immutable (D1), so names are reconciled at load time with this one helper.

DELTA_KEY_MAPPING = "language_model_prefix_v1"
_WRAPPED_PREFIX = "model.language_model."
_TEXT_PREFIX = "model."


class MappedDelta(dict):
    """A delta ``{base-state name: tensor}`` that remembers which key mapping produced it."""

    key_mapping: str | None = None


def _alternate_name(name: str) -> str | None:
    """The same tensor under the other naming (strip or add the ``language_model.`` segment)."""
    if name.startswith(_WRAPPED_PREFIX):
        return _TEXT_PREFIX + name[len(_WRAPPED_PREFIX) :]
    if name.startswith(_TEXT_PREFIX):
        return _WRAPPED_PREFIX + name[len(_TEXT_PREFIX) :]
    return None


def map_delta_keys(
    delta: Mapping[str, torch.Tensor], state: Iterable[str] | Mapping[str, torch.Tensor]
) -> MappedDelta:
    """Rename a delta's tensors onto the base state's keys, deterministically and strictly.

    An exact match wins; otherwise the other ``language_model.`` naming is tried. Any tensor that
    still has no base key, two saved names landing on one base key, or (when ``state`` is a
    mapping of tensors) a shape difference raises, so nothing is silently dropped or misapplied.
    ``key_mapping`` is :data:`DELTA_KEY_MAPPING` iff at least one tensor was renamed.
    """
    keys = state.keys() if isinstance(state, Mapping) else set(state)
    shaped = isinstance(state, Mapping)
    out, source, unmatched, renamed = MappedDelta(), {}, [], False
    for name, tensor in delta.items():
        target = name if name in keys else _alternate_name(name)
        if target is None or target not in keys:
            unmatched.append(name)
            continue
        if target in out:
            raise ValueError(
                f"delta tensors {source[target]} and {name} map onto the same base tensor {target}"
            )
        if shaped and tuple(tensor.shape) != tuple(state[target].shape):
            raise ValueError(
                f"delta {name} has shape {tuple(tensor.shape)}, base {target} has shape "
                f"{tuple(state[target].shape)}"
            )
        renamed |= target != name
        out[target], source[target] = tensor, name
    if unmatched:
        raise KeyError(
            f"{len(unmatched)} delta tensor(s) not in the base state "
            f"(first: {unmatched[0]}; unmatched: {sorted(unmatched)[:5]})"
        )
    out.key_mapping = DELTA_KEY_MAPPING if renamed else None
    return out


def to_text_model_keys(delta: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exporter side: drop the multimodal ``language_model.`` segment so names are base-state names."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in delta.items():
        new = name
        if name.startswith(_WRAPPED_PREFIX):
            new = _TEXT_PREFIX + name[len(_WRAPPED_PREFIX) :]
        if new in out:
            raise ValueError(f"exported tensor names collide after normalisation: {new}")
        out[new] = tensor
    return out
