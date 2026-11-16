"""Hugging Face helpers for the WM-E pilot: frozen target loading and CPU reference kernels for toy tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

TARGETS = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.8-27B": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",  # C9 calibration (Lambda)
}


def use_reference_kernels():
    """Pure-torch Qwen3.5 GatedDeltaNet/conv kernels (CPU toy models; as ``scripts/g2_fields.py``)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        function = getattr(qwen, name, None)
        if function is None:
            continue
        if isinstance(function, torch.nn.Module):
            function = inspect.getclosurevars(function.forward).nonlocals.get("func")
            if function is None:
                raise ValueError("unknown reference kernel wrapper")
        setattr(qwen, name, inspect.unwrap(function))


def snapshot_dir(repo: str, revision: str) -> Path:
    """The pinned snapshot directory in the local HF cache; requires the config, tokenizer and every indexed shard
    (repository metadata files such as README/LICENSE are not needed and may be absent)."""
    from huggingface_hub import constants

    path = (
        Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    )
    need = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    missing = [f for f in need if not (path / f).exists()]
    if not missing:
        index = json.loads((path / "model.safetensors.index.json").read_text())
        missing = [f for f in sorted(set(index["weight_map"].values())) if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{repo}@{revision}: missing {missing} in {path}")
    return path


def base_hash_from_snapshot(path: Path, repo: str, revision: str) -> str:
    """theta0 identity: repo, revision and the LFS sha256 (blob name) of every weight shard, text-only, bf16."""
    import hashlib

    shards = {}
    for f in sorted(path.glob("*.safetensors")):
        blob = Path(f).resolve().name
        if len(blob) != 64:
            raise ValueError(f"{f.name}: blob name is not a sha256")
        shards[f.name] = blob
    payload = {
        "repo": repo,
        "revision": revision,
        "shards": shards,
        "load": "text-only CausalLM bf16",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_target(repo: str, device="cuda"):
    """Text-only CausalLM in bf16, loaded shard-by-shard directly to ``device``; vision and MTP are not loaded."""
    import transformers

    revision = TARGETS[repo]
    path = snapshot_dir(repo, revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map=device, local_files_only=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    names = [n for n, _ in model.named_parameters()]
    if any(".visual." in n or n.startswith("mtp.") for n in names):
        raise RuntimeError("vision/MTP tensors were loaded; the target must be text-only")
    cfg = json.loads((path / "config.json").read_text())
    return model, tok, cfg, base_hash_from_snapshot(path, repo, revision), path
