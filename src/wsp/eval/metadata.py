"""Public runtime identity without infrastructure identifiers."""

import hashlib
import json
import re
from pathlib import Path

import torch
import transformers


def runtime_metadata(runner):
    path = Path.home() / ".config/wsp/device_id"
    if not path.exists():
        # No default: a second machine must never be mistaken for H0 (G3 device identity).
        raise FileNotFoundError(f"{path} is required: it holds this machine's opaque H<number>")
    device_id = path.read_text().strip()
    if not re.fullmatch(r"H\d+", device_id):
        raise ValueError("device_id must be an opaque H<number> identifier")
    model = runner.model
    return {
        "eval_device_id": device_id,
        "gpu_class": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "engine": "hf",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "dtype": str(getattr(model, "dtype", None)),
        "attention_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
    }


def item_hashes(records):
    ids = [(r.benchmark, r.item_id) for r in records]

    def digest(value):
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()

    return {"item_order_hash": digest(ids), "item_id_hash": digest(sorted(ids))}
