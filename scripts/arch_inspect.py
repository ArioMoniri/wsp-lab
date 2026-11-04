"""Write architecture fixtures (config + parameter names/shapes, no weights) for the zoo models.

    python scripts/arch_inspect.py [--out tests/fixtures/arch]

For each model: the text config, the in-memory zoo namespace (``AutoModelForCausalLM`` state-dict
names and shapes, built on the meta device) and, where a local checkpoint exists, the names in the
checkpoint's safetensors header/index. CPU only; no weights are read or written.

Gemma-3-270M-it is gated and our token has no access, so its config comes from the ungated mirror
``unsloth/gemma-3-270m-it`` (same safetensors byte size and parameter count as the official repo,
checked via Hub metadata). The fixture records that source; re-run once access is granted.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM

MODELS = {
    "qwen3.5-0.8b": {"repo": "Qwen/Qwen3.5-0.8B"},
    "qwen3.5-4b": {"repo": "Qwen/Qwen3.5-4B"},
    "lfm3-1.7b": {"repo": "LiquidAI/LFM3-1.7B", "remote": True},
    "gemma-3-270m": {
        "repo": "google/gemma-3-270m-it",
        "mirror": "unsloth/gemma-3-270m-it",
        "mirror_revision": "23cf460f6bb16954176b3ddcc8d4f250501458a9",
    },
}


def safetensors_header_names(path: Path) -> dict[str, list[int]]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: v["shape"] for k, v in header.items()}


def verify_shapes(snapshot: Path, params: dict[str, list[int]]) -> bool | None:
    """True if every zoo-namespace shape equals the shape in the local safetensors headers.

    None when the snapshot has no local weight files (shapes then come from the config only).
    """
    real: dict[str, list[int]] = {}
    for f in snapshot.glob("*.safetensors"):
        if f.exists():  # broken symlink = file not downloaded
            real |= safetensors_header_names(f.resolve())
    if not real:
        return None
    real = {k.replace("model.language_model.", "model.", 1): v for k, v in real.items()}
    bad = [k for k, v in params.items() if k != "lm_head.weight" and real.get(k) != v]
    if bad:
        raise SystemExit(f"shape/name mismatch against checkpoint headers: {bad[:5]}")
    return True


def checkpoint_names(snapshot: Path) -> list[str] | None:
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        return sorted(json.loads(index.read_text())["weight_map"])
    names: set[str] = set()
    for f in snapshot.glob("*.safetensors"):
        names |= set(safetensors_header_names(f.resolve()))
    return sorted(names) or None


def inspect(slug: str, spec: dict) -> dict:
    repo = spec["repo"]
    try:
        snapshot = Path(snapshot_download(repo, allow_patterns=["config.json"]))
        source = repo
        revision = snapshot.name
    except Exception:  # noqa: BLE001  (gated / not cached: fall back to the mirror config)
        snapshot = Path(
            snapshot_download(
                spec["mirror"], revision=spec["mirror_revision"], allow_patterns=["config.json"]
            )
        )
        source = f"{spec['mirror']} (ungated mirror; {repo} is gated and not accessible)"
        revision = spec["mirror_revision"]
    remote = bool(spec.get("remote"))
    cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=remote)
    text_cfg = cfg.get_text_config()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(text_cfg, trust_remote_code=remote)
    params = {k: list(v.shape) for k, v in model.state_dict().items()}
    return {
        "slug": slug,
        "repo": repo,
        "config_source": source,
        "config_revision": revision,
        "model_class": type(model).__name__,
        "text_config": {
            k: v
            for k, v in text_cfg.to_dict().items()
            if k
            in {
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "layer_types",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "tie_word_embeddings",
                "use_sandwich_norm",
                "linear_num_value_heads",
                "linear_num_key_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "conv_L_cache",
                "linear_conv_kernel_dim",
                "sliding_window",
                "attn_output_gate",
            }
        },
        "params": params,
        "shapes_verified_against_checkpoint_headers": verify_shapes(snapshot, params),
        "n_params": sum(
            torch.Size(s).numel()
            for k, s in params.items()
            if k != "lm_head.weight" or not text_cfg.tie_word_embeddings
        ),
        "checkpoint_names": checkpoint_names(snapshot) if source == repo else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/arch"))
    parser.add_argument("--only", nargs="*", default=list(MODELS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for slug in args.only:
        data = inspect(slug, MODELS[slug])
        (args.out / f"{slug}.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        print(slug, data["model_class"], len(data["params"]), "params tensors", data["n_params"])


if __name__ == "__main__":
    main()
