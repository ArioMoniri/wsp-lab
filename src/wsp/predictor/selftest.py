"""Explicitly synthetic tiny on-disk zoo, never a source of scientific results."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


def prepare_selftest():
    root = Path(tempfile.mkdtemp(prefix="wsp-p4-synthetic-"))
    generator = torch.Generator().manual_seed(42)
    rows = []
    for trajectory in range(6):
        parent = None
        for step in range(4):
            i = f"toy-{trajectory}-{step}"
            params = {"inner_family": "random_dir", "step_semantics": "sequential"}
            row = {
                "id": i,
                "family": "chain",
                "seed": trajectory,
                "parent_id": parent,
                "params": params,
            }
            rows.append(row)
            dest = root / "features" / i
            dest.mkdir(parents=True)
            meta = dict(
                edit_id=i,
                n_layers=1,
                tensors=[{"name": "model.layers.0.mlp.down_proj.weight", "row": 0}],
                **row,
            )
            (dest / "features.json").write_text(json.dumps(meta))
            save_file(
                {"svd": torch.randn(1, 2, 4, generator=generator)}, dest / "features.safetensors"
            )
            save_file(
                {
                    "stats": torch.randn(12, generator=generator),
                    "svd": torch.randn(8, generator=generator),
                },
                dest / "pooled.safetensors",
            )
            dest = root / "evals" / i
            dest.mkdir(parents=True)
            (dest / "eval.json").write_text(
                json.dumps(
                    {"synthetic": True, "metrics": {"acc": 0.5 + 0.02 * step + 0.01 * trajectory}}
                )
            )
            parent = i
    (root / "zoo").mkdir()
    (root / "zoo" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "features" / "manifest.json").write_text(
        json.dumps({"synthetic": True, "zoo": str(root / "zoo")})
    )
    (root / "evals" / "base").mkdir()
    (root / "evals" / "base" / "eval.json").write_text(
        '{"synthetic": true, "metrics": {"acc": 0.5}}'
    )
    return root
