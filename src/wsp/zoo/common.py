"""Shared checkpoint provenance and PEFT adapter inspection for zoo families."""

import math
import re
import subprocess
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
