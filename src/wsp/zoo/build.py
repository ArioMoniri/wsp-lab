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

from wsp.edits.kernels import KERNEL_VARIANT, heretic_abliteration_kernel
from wsp.edits.refusal import get_refusal_directions
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


FAMILY_ORDER = ["random_dir", "abliteration", "heretic_ara", "lora_sft", "task_vector", "chain"]
SINGLE_FAMILIES = ["random_dir", "abliteration", "heretic_ara", "lora_sft"]
OPTIONAL_MODULES = ("heretic_ara", "lora_sft")


class MissingFamiliesError(ValueError):
    """Configured edit families cannot be built in this environment."""


def _family_implemented(family: str) -> bool:
    return (
        family not in OPTIONAL_MODULES or importlib.util.find_spec(f"wsp.zoo.{family}") is not None
    )


def family_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Which configured families can be built here; recorded in every manifest/edit.json."""
    requested = [f for f in FAMILY_ORDER if f in config.get("families", {})]
    available = [f for f in requested if _family_implemented(f)]
    skipped = [f for f in requested if f not in available]
    return {
        "families_requested": requested,
        "families_available": available,
        "skipped_families": skipped,
        "skipped_reasons": {f: f"module wsp.zoo.{f} is not implemented" for f in skipped},
    }


def _has_artifacts(zoo_dir: Path, entry_id: str) -> bool:
    d = zoo_dir / entry_id
    return (d / "edit.json").exists() and (d / "delta.safetensors").exists()


def _load_manifest(
    path: Path, zoo_dir: Path, require_artifacts: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (entry rows, header rows): deduplicated by id, artifact-less rows dropped if asked.

    Rewrites the file if cleaning changed it (older builds appended dry-run/duplicate rows).
    """
    if not path.exists():
        return [], []
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    headers = [r for r in raw if r.get("type") == "header"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in raw:
        if r.get("type") == "header" or r["id"] in seen:
            continue
        if require_artifacts and not _has_artifacts(zoo_dir, r["id"]):
            continue
        seen.add(r["id"])
        rows.append(r)
    if len(rows) + len(headers) != len(raw):
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in [*headers, *rows]))
    return rows, headers


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

    refusal = get_refusal_directions(config, zoo_dir, config.get("_config_hash"))
    refusal_dirs = refusal.directions
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
    desc = _descriptor("abliteration", params, seed, config, None, "", touched, None)
    desc.provenance.update(
        {
            "kernel_variant": KERNEL_VARIANT,
            "refusal_fingerprint": refusal.fingerprint,
            "refusal_manifest": refusal.manifest_path.name,
        }
    )
    return [(desc, new_state, touched)]


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
    candidates = sorted(
        (
            m
            for m in manifest
            if m.get("family") in SINGLE_FAMILIES and (dry_run or _has_artifacts(zoo_dir, m["id"]))
        ),
        key=lambda m: m["id"],
    )
    if not candidates:
        return []
    parent = rng.choice(candidates)
    parent_id = parent["id"]
    pool_digest = hashlib.sha256(",".join(m["id"] for m in candidates).encode()).hexdigest()
    if dry_run:
        desc = _descriptor("task_vector", params, seed, config, parent_id, "", [], None)
        desc.provenance["parent_candidates_sha256"] = pool_digest
        return [(desc, {}, [])]

    parent_delta = load_file(zoo_dir / parent_id / "delta.safetensors")
    new_state = dict(base_state)
    touched: list[str] = []
    for key, dv in parent_delta.items():
        if key in new_state:
            new_state[key] = new_state[key] + params["scale"] * dv
        else:
            new_state[key] = params["scale"] * dv
        touched.append(key)
    desc = _descriptor("task_vector", params, seed, config, parent_id, "", touched, None)
    desc.provenance["parent_candidates_sha256"] = pool_digest
    return [(desc, new_state, touched)]


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
    available = [f for f in SINGLE_FAMILIES if f in family_policy(config)["families_available"]]
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
        desc.provenance.update(inner_desc.provenance)
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
    allow_missing_families: bool = False,
) -> list[dict[str, Any]]:
    """Build (or dry-run) the edit zoo described by ``config_path``.

    Args:
        config_path: path to the zoo YAML config.
        family: if given, build only this family.
        n: if given, override the family's ``n`` count.
        dry_run: if True, sample descriptors and write the manifest but do not load
            the model or write per-entry deltas.
        git_sha: override the git SHA recorded in the manifest.
        allow_missing_families: record and skip configured families that are not implemented
            (default: fail).

    Returns:
        The manifest as a list of line dicts.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    cfg_hash = config_hash(config_path)
    git_sha = git_sha or get_git_sha()
    zoo_dir = Path(config["out_dir"])
    zoo_dir.mkdir(parents=True, exist_ok=True)

    configured = set(config.get("families", {}))
    unknown = configured - set(FAMILY_ORDER)
    if unknown:
        raise ValueError(f"unknown edit family: {sorted(unknown)}")
    if family and family not in FAMILY_ORDER:
        raise ValueError(f"unknown edit family: {family}")

    # Family policy: never silently drop configured families (H2 chain distribution depends on it).
    policy = family_policy(config)
    if policy["skipped_families"] and not allow_missing_families:
        raise MissingFamiliesError(
            f"configured families not buildable here: {policy['skipped_reasons']}; "
            "implement them or pass --allow-missing-families to record and skip them"
        )
    if family and family in policy["skipped_families"]:
        raise MissingFamiliesError(f"requested family {family!r} is not buildable here")
    config["_config_hash"] = cfg_hash
    provenance = {**policy, "allow_missing_families": bool(allow_missing_families)}

    base_state: dict[str, torch.Tensor] = {}
    if not dry_run:
        base_state = _load_base_state(config)

    # Dry runs plan into a separate file and never touch the production manifest.
    manifest_path = zoo_dir / ("manifest.dryrun.jsonl" if dry_run else "manifest.jsonl")
    manifest, headers = _load_manifest(manifest_path, zoo_dir, require_artifacts=not dry_run)
    header = {
        "type": "header",
        "dry_run": dry_run,
        "config_hash": cfg_hash,
        "git_sha": git_sha,
        "base_model": config["base_model"],
        "base_revision": config.get("base_revision", "main"),
        "seed": int(config["seed"]),
        "families_selected": [family] if family else policy["families_available"],
        **provenance,
    }
    if header not in headers:
        _append_manifest(manifest_path, header)

    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    families = [family] if family else policy["families_available"]

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
                desc.provenance.update(provenance)
                if not dry_run:
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
                if "refusal_fingerprint" in desc.provenance:
                    line["refusal_fingerprint"] = desc.provenance["refusal_fingerprint"]
                if not any(m["id"] == line["id"] for m in manifest):  # idempotent registration
                    manifest.append(line)
                    _append_manifest(manifest_path, line)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WSP edit zoo.")
    parser.add_argument("--config", required=True, type=Path, help="zoo YAML config")
    parser.add_argument("--family", default=None, help="build only this family")
    parser.add_argument("--n", type=int, default=None, help="override number of edits")
    parser.add_argument("--dry-run", action="store_true", help="plan into manifest.dryrun.jsonl")
    parser.add_argument(
        "--allow-missing-families",
        action="store_true",
        help="skip (and record) configured families that are not implemented instead of failing",
    )
    args = parser.parse_args()
    build_zoo(
        args.config,
        family=args.family,
        n=args.n,
        dry_run=args.dry_run,
        allow_missing_families=args.allow_missing_families,
    )


if __name__ == "__main__":
    main()
