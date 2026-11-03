"""Build the edit zoo: sample edit descriptors, apply edits, write deltas + manifests.

CLI:
    python -m wsp.zoo.build --config configs/zoo_qwen3.5-0.8b.yaml [--family F] [--n N] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from wsp.edits.kernels import heretic_abliteration_kernel
from wsp.edits.refusal import extract_directions
from wsp.schema import EditDescriptor
from wsp.zoo.random_dir import ablate_direction, random_unit_direction


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def config_hash(path: Path) -> str:
    """SHA-256 of the raw config file contents (stable across parses)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def sample_param(rng: random.Random, spec: dict[str, Any]) -> Any:
    """Sample a single parameter from a YAML range spec.

    Supported specs:
        {min: x, max: y}       -> uniform float in [x, y]
        {choices: [a, b, c]}   -> uniform choice
        {log_min: x, log_max: y} -> uniform float in log space
    """
    if isinstance(spec, list):
        return rng.choice(spec)
    if "choices" in spec:
        return rng.choice(spec["choices"])
    if "log_min" in spec and "log_max" in spec:
        log_min = math.log(spec["log_min"])
        log_max = math.log(spec["log_max"])
        return math.exp(rng.uniform(log_min, log_max))
    if "min" in spec and "max" in spec:
        return rng.uniform(spec["min"], spec["max"])
    raise ValueError(f"unsupported parameter spec: {spec}")


def _model_key(layer: int, tensor_name: str) -> str:
    """Default transformers Llama/Qwen-style key format."""
    return f"model.layers.{layer}.{tensor_name}.weight"


def _delta_fro_norm(delta: dict[str, torch.Tensor]) -> float:
    total = sum((t.float() ** 2).sum().item() for t in delta.values())
    return math.sqrt(total)


def _write_entry(
    zoo_dir: Path,
    desc: EditDescriptor,
    delta: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> None:
    entry_dir = zoo_dir / desc.id
    entry_dir.mkdir(parents=True, exist_ok=True)
    desc.touched_tensors = sorted(delta.keys())
    desc.delta_fro_norm = _delta_fro_norm(delta)
    (entry_dir / "edit.json").write_text(desc.to_json())
    save_file(
        {k: v.to(dtype).contiguous() for k, v in delta.items()}, entry_dir / "delta.safetensors"
    )


def _append_manifest(manifest_path: Path, line: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def _descriptor(
    family: str,
    params: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    parent_id: str | None,
    git_sha: str,
    touched_tensors: list[str] | None = None,
    delta_fro_norm: float | None = None,
) -> EditDescriptor:
    return EditDescriptor(
        family=family,
        params=params,
        seed=seed,
        base_model=config["base_model"],
        base_revision=config.get("base_revision", "main"),
        parent_id=parent_id,
        git_sha=git_sha,
        touched_tensors=touched_tensors or [],
        delta_fro_norm=delta_fro_norm,
    )


def sample_random_dir_params(rng: random.Random, family_cfg: dict[str, Any]) -> dict[str, Any]:
    n_layers = rng.randint(1, len(family_cfg["layers"]))
    return {
        "layers": sorted(rng.sample(family_cfg["layers"], k=n_layers)),
        "tensors": list(family_cfg["tensors"]),
        "n_directions": int(sample_param(rng, family_cfg["n_directions"])),
        "weight": float(sample_param(rng, family_cfg["weight"])),
    }


def build_random_dir(
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    dry_run: bool = False,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    rng = random.Random(seed)
    params = sample_random_dir_params(rng, family_cfg)
    if dry_run:
        return [(_descriptor("random_dir", params, seed, config, None, "", [], None), {}, [])]

    new_state = dict(base_state)
    touched: list[str] = []
    for layer in params["layers"]:
        for tensor_name in params["tensors"]:
            key = _model_key(layer, tensor_name)
            if key not in new_state and tensor_name == "self_attn.o_proj":
                key = _model_key(layer, "linear_attn.out_proj")
            if key not in new_state:
                raise KeyError(f"tensor {key} not found in base model")
            W = new_state[key]
            d_model = W.shape[0]
            for di in range(params["n_directions"]):
                d = random_unit_direction(d_model, seed=seed + layer * 1000 + di * 100)
                W = ablate_direction(W, d, weight=params["weight"])
            new_state[key] = W
            touched.append(key)
    return [
        (
            _descriptor("random_dir", params, seed, config, None, "", touched, None),
            new_state,
            touched,
        )
    ]


def sample_abliteration_params(rng: random.Random, family_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "direction_index": float(sample_param(rng, family_cfg["direction_index"])),
        "max_weight": float(sample_param(rng, family_cfg["max_weight"])),
        "max_weight_position": float(sample_param(rng, family_cfg["max_weight_position"])),
        "min_weight": float(sample_param(rng, family_cfg["min_weight"])),
        "min_weight_distance": float(sample_param(rng, family_cfg["min_weight_distance"])),
        "components": sample_param(rng, family_cfg["components"]),
    }


def build_abliteration(
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    zoo_dir: Path,
    dry_run: bool = False,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    rng = random.Random(seed)
    params = sample_abliteration_params(rng, family_cfg)
    if dry_run:
        return [(_descriptor("abliteration", params, seed, config, None, "", [], None), {}, [])]

    refusal_dirs = extract_directions(config, zoo_dir)
    layers = family_cfg.get("layers", list(range(len(refusal_dirs) - 1)))
    components = params["components"]
    tensor_names: list[str] = []
    if components in ("attn", "both"):
        tensor_names.append("self_attn.o_proj")
    if components in ("mlp", "both"):
        tensor_names.append("mlp.down_proj")
    layer_tensors = {layer: [_model_key(layer, tn) for tn in tensor_names] for layer in layers}

    for layer, keys in layer_tensors.items():
        layer_tensors[layer] = [
            key if key in base_state else key.replace("self_attn.o_proj", "linear_attn.out_proj")
            for key in keys
        ]

    new_state, touched = heretic_abliteration_kernel(
        base_state,
        refusal_dirs,
        layer_tensors,
        direction_index=params["direction_index"],
        max_weight=params["max_weight"],
        max_weight_position=params["max_weight_position"],
        min_weight=params["min_weight"],
        min_weight_distance=params["min_weight_distance"],
    )
    return [
        (
            _descriptor("abliteration", params, seed, config, None, "", touched, None),
            new_state,
            touched,
        )
    ]


def build_heretic_ara(
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    dry_run: bool = False,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    rng = random.Random(seed)
    params = {
        "preserve_good_behavior_weight": sample_param(
            rng, family_cfg["preserve_good_behavior_weight"]
        ),
        "trials": int(family_cfg.get("trials", 40)),
    }
    if dry_run:
        return [(_descriptor("heretic_ara", params, seed, config, None, "", [], None), {}, [])]
    try:
        import wsp.zoo.heretic_ara as ha
    except ImportError as e:
        raise NotImplementedError("heretic_ara edit family is not yet implemented") from e
    return ha.build_heretic_ara(config, family_cfg, seed, base_state)


def build_lora_sft(
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    dry_run: bool = False,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    rng = random.Random(seed)
    params = {
        "dataset": sample_param(rng, family_cfg["datasets"]),
        "rank": int(sample_param(rng, family_cfg["rank"])),
        "lr": float(sample_param(rng, family_cfg["lr"])),
        "steps": int(sample_param(rng, family_cfg["steps"])),
    }
    if dry_run:
        return [(_descriptor("lora_sft", params, seed, config, None, "", [], None), {}, [])]
    try:
        import wsp.zoo.lora_sft as lora
    except ImportError as e:
        raise NotImplementedError("lora_sft edit family is not yet implemented") from e
    return lora.build_lora_sft(config, family_cfg, seed, base_state)


def build_task_vector(
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    zoo_dir: Path,
    manifest: list[dict[str, Any]],
    dry_run: bool = False,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    rng = random.Random(seed)
    params = {"scale": float(sample_param(rng, family_cfg["scale"]))}
    candidates = [m for m in manifest if m.get("family") != "chain"]
    if not candidates:
        return []
    parent = rng.choice(candidates)
    parent_id = parent["id"]
    if dry_run:
        return [(_descriptor("task_vector", params, seed, config, parent_id, "", [], None), {}, [])]

    parent_delta = load_file(zoo_dir / parent_id / "delta.safetensors")
    new_state = dict(base_state)
    touched: list[str] = []
    for key, dv in parent_delta.items():
        if key in new_state:
            new_state[key] = new_state[key] + params["scale"] * dv
        else:
            new_state[key] = params["scale"] * dv
        touched.append(key)
    return [
        (
            _descriptor("task_vector", params, seed, config, parent_id, "", touched, None),
            new_state,
            touched,
        )
    ]


def _build_single_family(
    family: str,
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    zoo_dir: Path,
    manifest: list[dict[str, Any]],
    dry_run: bool,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    if family == "random_dir":
        return build_random_dir(config, family_cfg, seed, base_state, dry_run=dry_run)
    if family == "abliteration":
        return build_abliteration(config, family_cfg, seed, base_state, zoo_dir, dry_run=dry_run)
    if family == "heretic_ara":
        return build_heretic_ara(config, family_cfg, seed, base_state, dry_run=dry_run)
    if family == "lora_sft":
        return build_lora_sft(config, family_cfg, seed, base_state, dry_run=dry_run)
    if family == "task_vector":
        return build_task_vector(
            config, family_cfg, seed, base_state, zoo_dir, manifest, dry_run=dry_run
        )
    raise ValueError(f"unknown single-edit family: {family}")


def build_chain(
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    zoo_dir: Path,
    manifest: list[dict[str, Any]],
    dry_run: bool = False,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    rng = random.Random(seed)
    length = int(sample_param(rng, family_cfg["length"]))
    single_families = ["random_dir", "abliteration", "heretic_ara", "lora_sft"]
    available = [
        f
        for f in single_families
        if f in config.get("families", {})
        and (
            f not in ("heretic_ara", "lora_sft")
            or importlib.util.find_spec(f"wsp.zoo.{f}") is not None
        )
    ]
    if not available:
        raise ValueError("no single-edit families configured; chain cannot be built")

    if dry_run:
        entries: list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]] = []
        parent_id: str | None = None
        for step in range(length):
            family = rng.choice(available)
            inner_cfg = config["families"][family]
            step_seed = seed + step + 1
            inner_entries = _build_single_family(
                family, config, inner_cfg, step_seed, {}, zoo_dir, manifest, dry_run=True
            )
            inner_desc = inner_entries[0][0]
            params = {
                "length": length,
                "step": step,
                "inner_family": family,
                "inner_params": inner_desc.params,
                "inner_seed": step_seed,
            }
            desc = _descriptor("chain", params, seed, config, parent_id, "", [], None)
            entries.append((desc, {}, []))
            parent_id = desc.id
        return entries

    cumulative = dict(base_state)
    entries = []
    all_touched: set[str] = set()
    parent_id = None
    for step in range(length):
        family = rng.choice(available)
        inner_cfg = config["families"][family]
        step_seed = seed + step + 1
        inner_entries = _build_single_family(
            family, config, inner_cfg, step_seed, cumulative, zoo_dir, manifest, dry_run=False
        )
        inner_desc, cumulative, step_touched = inner_entries[0]
        all_touched.update(step_touched)
        params = {
            "length": length,
            "step": step,
            "inner_family": family,
            "inner_params": inner_desc.params,
            "inner_seed": step_seed,
        }
        desc = _descriptor("chain", params, seed, config, parent_id, "", [], None)
        entries.append((desc, cumulative, sorted(all_touched)))
        parent_id = desc.id
    return entries


def _load_base_state(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Load the base model in the requested dtype and return its CPU state dict."""
    import transformers

    dtype_str = config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=config.get("base_revision", "main"),
        torch_dtype=dtype,
        device_map="cpu",
    )
    state = {k: v.detach().cpu().to(dtype) for k, v in model.state_dict().items()}
    del model
    return state


def build_zoo(
    config_path: str | Path,
    family: str | None = None,
    n: int | None = None,
    dry_run: bool = False,
    git_sha: str | None = None,
) -> list[dict[str, Any]]:
    """Build (or dry-run) the edit zoo described by ``config_path``.

    Args:
        config_path: path to the zoo YAML config.
        family: if given, build only this family.
        n: if given, override the family's ``n`` count.
        dry_run: if True, sample descriptors and write the manifest but do not load
            the model or write per-entry deltas.
        git_sha: override the git SHA recorded in the manifest.

    Returns:
        The manifest as a list of line dicts.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    cfg_hash = config_hash(config_path)
    git_sha = git_sha or get_git_sha()
    zoo_dir = Path(config["out_dir"])
    zoo_dir.mkdir(parents=True, exist_ok=True)

    known_families = {
        "random_dir",
        "abliteration",
        "heretic_ara",
        "lora_sft",
        "task_vector",
        "chain",
    }
    configured = set(config.get("families", {}))
    unknown = configured - known_families
    if unknown:
        raise ValueError(f"unknown edit family: {sorted(unknown)}")
    if family and family not in known_families:
        raise ValueError(f"unknown edit family: {family}")

    base_state: dict[str, torch.Tensor] = {}
    if not dry_run:
        base_state = _load_base_state(config)

    manifest_path = zoo_dir / ("manifest.dryrun.jsonl" if dry_run else "manifest.jsonl")
    manifest: list[dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = [json.loads(line) for line in f if line.strip()]

    dtype_str = config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    family_order = ["random_dir", "abliteration", "heretic_ara", "lora_sft", "task_vector", "chain"]
    families = [family] if family else [f for f in family_order if f in config.get("families", {})]

    for fam in families:
        family_cfg = config["families"][fam]
        count = n if n is not None else int(family_cfg.get("n", 0))
        for i in range(count):
            seed = int(config["seed"]) + i
            if fam == "chain":
                entries = build_chain(
                    config, family_cfg, seed, base_state, zoo_dir, manifest, dry_run=dry_run
                )
            else:
                entries = _build_single_family(
                    fam, config, family_cfg, seed, base_state, zoo_dir, manifest, dry_run=dry_run
                )
            for desc, new_state, touched in entries:
                desc.git_sha = git_sha
                if dry_run:
                    line = {
                        "id": desc.id,
                        "family": desc.family,
                        "parent_id": desc.parent_id,
                        "seed": desc.seed,
                        "config_hash": cfg_hash,
                        "git_sha": git_sha,
                    }
                    if not any(m.get("id") == line["id"] for m in manifest):
                        manifest.append(line)
                        _append_manifest(manifest_path, line)
                else:
                    delta = {
                        k: (new_state[k] - base_state[k]).to(dtype)
                        for k in touched
                        if k in base_state and not torch.equal(new_state[k], base_state[k])
                    }
                    _write_entry(zoo_dir, desc, delta, dtype)
                    line = {
                        "id": desc.id,
                        "family": desc.family,
                        "parent_id": desc.parent_id,
                        "seed": desc.seed,
                        "config_hash": cfg_hash,
                        "git_sha": git_sha,
                    }
                    if not any(m.get("id") == line["id"] for m in manifest):
                        manifest.append(line)
                        _append_manifest(manifest_path, line)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WSP edit zoo.")
    parser.add_argument("--config", required=True, type=Path, help="zoo YAML config")
    parser.add_argument("--family", default=None, help="build only this family")
    parser.add_argument("--n", type=int, default=None, help="override number of edits")
    parser.add_argument("--dry-run", action="store_true", help="write manifest only, no GPU")
    args = parser.parse_args()
    build_zoo(args.config, family=args.family, n=args.n, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
