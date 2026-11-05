"""Build the edit zoo: sample edit descriptors, apply edits, write deltas + manifests.

CLI:
    python -m wsp.zoo.build --config configs/zoo_qwen3.5-0.8b.yaml [--family F] [--n N] [--dry-run]
        [--override families.heretic_ara.trials=2 ...]

``heretic_ara`` and ``lora_sft`` entries are produced by those modules' ``generate`` and stored
as PEFT adapters (``lora/`` instead of ``delta.safetensors``; ``wsp.zoo.apply`` densifies them).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

from wsp.edits.kernels import FOLD_SUFFIX, KERNEL_VARIANT, heretic_abliteration_kernel
from wsp.edits.refusal import get_refusal_directions
from wsp.schema import EditDescriptor
from wsp.zoo import arch as _arch
from wsp.zoo.apply import entry_has_artifacts, load_entry_delta
from wsp.zoo.common import metadata_revision, resolve_base_revision
from wsp.zoo.random_dir import ablate_direction, random_unit_direction


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def config_hash(path: Path, overrides: list[str] | None = None) -> str:
    """SHA-256 of the raw config file contents plus any CLI overrides (stable across parses)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    if overrides:
        h.update(json.dumps(list(overrides)).encode())
    return h.hexdigest()


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides in place (value parsed as YAML).

    A leading family name is shorthand for ``families.<family>`` (``heretic_ara.trials=2``).
    Intermediate keys must exist, so typos fail loudly; the leaf may be new.
    """
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        path = key.split(".")
        if path[0] in FAMILY_ORDER:
            path = ["families", *path]
        node = config
        for part in path[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"override {item!r}: no config key {part!r}")
            node = node[part]
        if not isinstance(node, dict):
            raise TypeError(f"override {item!r}: {'.'.join(path[:-1])} is not a mapping")
        node[path[-1]] = yaml.safe_load(raw)
    return config


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
        [a, b, c] -> uniform choice;  a bare scalar -> that value
    """
    if isinstance(spec, list):
        return rng.choice(spec)
    if not isinstance(spec, dict):  # fixed value (e.g. from a --override)
        return spec
    if "choices" in spec:
        return rng.choice(spec["choices"])
    if "log_min" in spec and "log_max" in spec:
        log_min = math.log(spec["log_min"])
        log_max = math.log(spec["log_max"])
        return math.exp(rng.uniform(log_min, log_max))
    if "min" in spec and "max" in spec:
        return rng.uniform(spec["min"], spec["max"])
    raise ValueError(f"unsupported parameter spec: {spec}")


FAMILY_ORDER = [
    "llm_edit",
    "random_dir",
    "abliteration",
    "heretic_ara",
    "lora_sft",
    "task_vector",
    "chain",
]
SINGLE_FAMILIES = ["random_dir", "abliteration", "heretic_ara", "lora_sft"]
OPTIONAL_MODULES = ("heretic_ara", "lora_sft")
LORA_FAMILIES = ("heretic_ara", "lora_sft")  # generated via wsp.zoo.<family>.generate -> lora/
# Optional family-config keys forwarded verbatim to ``generate`` when present.
GENERATE_PASSTHROUGH = {
    "heretic_ara": (
        "kl_divergence_target",
        "lora_rank",
        "batch_size",
        "max_response_length",
        "ara_settings",
        "heretic_settings",
    ),
    "lora_sft": ("targets", "batch_size", "max_length"),
}
STAGING_DIR = ".staging"
# Entries below these sampled-parameter budgets are wiring smokes, not research treatments.
# Overridable per family under ``eligibility.min_params`` in the zoo config.
DEFAULT_MIN_PARAMS: dict[str, dict[str, float]] = {
    "heretic_ara": {"trials": 20},
    "lora_sft": {"steps": 50},
}
LORA_VS_BASE = "lora_vs_base"  # LoRA optimised against the base checkpoint, added onto the chain
SEQUENTIAL = "sequential"  # edit computed on / applied to the accumulated state


class MissingFamiliesError(ValueError):
    """Configured edit families cannot be built in this environment."""


class EntryIdentityMismatch(RuntimeError):
    """An existing entry/manifest row has the planned id but a different ``reuse_key``."""


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
    return entry_has_artifacts(zoo_dir / entry_id)


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


def _is_access_error(e: Exception) -> bool:
    """Gated / unauthorised Hub access (as opposed to a network or cache fault)."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if isinstance(e, GatedRepoError):
        return True
    response = getattr(e, "response", None)
    return isinstance(e, HfHubHTTPError) and getattr(response, "status_code", None) in (401, 403)


def _architecture_record(info: _arch.ArchInfo | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return {
        "family": info.arch.name,
        "hidden_size": info.hidden_size,
        "num_layers": info.num_layers,
        "verified": info.verified,
    }


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


def _override_scope(item: str) -> tuple[str | None, str]:
    """(family or None for a global key, leaf key) of a ``key.path=value`` override."""
    path = item.partition("=")[0].split(".")
    if path[0] in FAMILY_ORDER:
        path = ["families", *path]
    if path[0] == "families" and len(path) >= 3:
        return path[1], path[-1]
    return None, path[-1]


def applicable_overrides(families: set[str], overrides: list[str] | None) -> list[str]:
    """Overrides that can change an entry of ``families``: global ones and family-scoped ones.

    ``<family>.n`` is excluded: the count of entries does not change any single entry.
    """
    out = []
    for item in overrides or []:
        fam, leaf = _override_scope(item)
        if fam is None or (fam in families and leaf != "n"):
            out.append(item)
    return sorted(out)


def classify_eligibility(
    family: str, params: dict[str, Any], config: dict[str, Any]
) -> tuple[str, list[str]]:
    """``("production"|"smoke", reasons)``: smoke = reduced budget or any applicable override."""
    reasons: list[str] = []
    configured = (config.get("eligibility") or {}).get("min_params", {})
    mins = {**DEFAULT_MIN_PARAMS.get(family, {}), **configured.get(family, {})}
    for key, lo in sorted(mins.items()):
        if key in params and params[key] < lo:
            reasons.append(f"{key}={params[key]} below production minimum {lo}")
    applied = applicable_overrides({family}, config.get("_overrides"))
    if applied:
        reasons.append(f"built with config overrides {applied}")
    return ("smoke" if reasons else "production"), reasons


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
    eligibility, reasons = classify_eligibility(family, params, config)
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
        eligibility=eligibility,
        eligibility_reasons=reasons,
    )


def _stored_format(desc: EditDescriptor) -> str:
    if desc.family == "chain":
        return "dense_cumulative"  # a cumulative snapshot relative to base, not a per-step delta
    fmt = desc.provenance.get("delta_format") or desc.params.get("delta_format")
    return "lora" if fmt == "lora" else "dense"


def _step_identity(desc: EditDescriptor, config: dict[str, Any]) -> dict[str, Any]:
    """Everything outside the id that changes what this family's artefact contains."""
    prov = desc.provenance
    return {
        "family": desc.family,
        "base_model": desc.base_model,
        "base_revision": desc.base_revision,
        "dtype": config.get("dtype", "bfloat16"),
        "stored_format": _stored_format(desc),
        "overrides": applicable_overrides({desc.family}, config.get("_overrides")),
        "kernel_variant": prov.get("kernel_variant"),
        "refusal_fingerprint": prov.get("refusal_fingerprint"),
        "step_semantics": prov.get("step_semantics"),
        "parent_reuse_key": prov.get("parent_reuse_key"),
    }


def finalize_identity(desc: EditDescriptor, config: dict[str, Any]) -> str:
    """Record ``reuse_identity``/``reuse_key`` in the descriptor's provenance; return the key."""
    steps = desc.provenance.get("step_identities")
    identity = _step_identity(desc, config)
    if steps:  # chain: the whole prefix of steps, each with its own extraction/generation identity
        identity["chain"] = steps
    desc.provenance["reuse_identity"] = identity
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    desc.provenance["reuse_key"] = key
    return key


def check_reusable(zoo_dir: Path, manifest: list[dict[str, Any]], desc: EditDescriptor) -> bool:
    """True if an identical entry exists; raise if one exists with the same id but another key.

    An entry id (family, sampled params, seed, pinned base, parent) does not cover refusal
    extraction, dtype, kernel variant or config overrides; ``reuse_key`` does. A mismatch is never
    resolved silently (neither reused nor overwritten): move the old entry directory aside and
    rebuild (artefact-less manifest rows are dropped automatically).
    """
    want = desc.provenance["reuse_key"]
    for row in manifest:
        if row["id"] == desc.id and row.get("reuse_key") != want:
            raise EntryIdentityMismatch(
                f"manifest row {desc.id} has reuse_key {row.get('reuse_key')!r}, planned {want!r}; "
                "same sampled id built with different extraction/generation settings"
            )
    entry = zoo_dir / desc.id
    if not entry_has_artifacts(entry):
        return False
    have = json.loads((entry / "edit.json").read_text()).get("provenance", {}).get("reuse_key")
    if have != want:
        raise EntryIdentityMismatch(
            f"existing entry {desc.id} has reuse_key {have!r} (none = legacy/unprovenanced), "
            f"planned {want!r}: refusing to reuse or overwrite it; move {entry} aside to rebuild"
        )
    return True


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
    info = config.get("_arch")
    arch = info.arch if info else _arch.ARCHS["qwen3_5"]  # no info: legacy Qwen3.5 naming
    index = _arch.index_state_keys(base_state)
    for layer in params["layers"]:
        for key in _arch.select_keys(params["tensors"], layer, index, arch):
            W = new_state[key]
            d_model = W.shape[0]
            if info is not None and d_model != info.hidden_size:
                raise ValueError(
                    f"{key} has {d_model} rows but the residual stream is {info.hidden_size}: "
                    "random_dir only edits tensors that write into the residual stream"
                )
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
    info = config.get("_arch")
    if info is not None:
        _arch.validate_directions(refusal_dirs, info)
    arch = info.arch if info else _arch.ARCHS["qwen3_5"]  # no info: legacy Qwen3.5 naming
    layers = family_cfg.get("layers", list(range(len(refusal_dirs) - 1)))
    components = params["components"]
    specs: list[str] = []
    if components in ("attn", "both"):
        specs.append("attn_out")
    if components in ("mlp", "both"):
        specs.append("mlp_out")
    index = _arch.index_state_keys(base_state)
    layer_tensors = {layer: _arch.select_keys(specs, layer, index, arch) for layer in layers}

    gains = None
    if family_cfg.get("fold_post_norm"):
        if info is None or not info.post_norms:
            raise ValueError(
                "abliteration.fold_post_norm is set but this model has no post-writer norm "
                "(sandwich norm); remove the flag or declare architecture.post_norms"
            )
        gains = {}
        for keys in layer_tensors.values():
            for key in keys:
                component = _arch.component_of(key, arch)
                norm_key = _arch.post_norm_key(key, info.post_norms[component])
                gains[key] = _arch.effective_gain(base_state[norm_key], info.norm_gain)

    new_state, touched = heretic_abliteration_kernel(
        base_state,
        refusal_dirs,
        layer_tensors,
        direction_index=params["direction_index"],
        max_weight=params["max_weight"],
        max_weight_position=params["max_weight_position"],
        min_weight=params["min_weight"],
        min_weight_distance=params["min_weight_distance"],
        gains=gains,
    )
    desc = _descriptor("abliteration", params, seed, config, None, "", touched, None)
    desc.provenance.update(
        {
            "kernel_variant": KERNEL_VARIANT + (FOLD_SUFFIX if gains else ""),
            "refusal_fingerprint": refusal.fingerprint,
            "refusal_manifest": refusal.manifest_path.name,
        }
    )
    return [(desc, new_state, touched)]


def sample_heretic_ara_params(rng: random.Random, family_cfg: dict[str, Any]) -> dict[str, Any]:
    params = {
        "preserve_good_behavior_weight": sample_param(
            rng, family_cfg["preserve_good_behavior_weight"]
        ),
        "trials": int(sample_param(rng, family_cfg.get("trials", 40))),
    }
    return _with_lora_extras("heretic_ara", params, family_cfg)


def sample_lora_sft_params(rng: random.Random, family_cfg: dict[str, Any]) -> dict[str, Any]:
    params = {
        "dataset": sample_param(rng, family_cfg["datasets"]),
        "rank": int(sample_param(rng, family_cfg["rank"])),
        "lr": float(sample_param(rng, family_cfg["lr"])),
        "steps": int(sample_param(rng, family_cfg["steps"])),
    }
    return _with_lora_extras("lora_sft", params, family_cfg)


def _with_lora_extras(family: str, params: dict[str, Any], family_cfg: dict[str, Any]):
    for key in GENERATE_PASSTHROUGH[family]:
        if key in family_cfg:
            params[key] = family_cfg[key]
    params["delta_format"] = "dense" if family == "heretic_ara" else "lora"
    if family == "heretic_ara":
        params["kernel_variant"] = "wsp_ara_no_row_normalization_v1"
    return params


_LORA_SAMPLERS = {"heretic_ara": sample_heretic_ara_params, "lora_sft": sample_lora_sft_params}


def _run_generate(
    family: str, config: dict[str, Any], params: dict[str, Any], seed: int, out_dir: Path
) -> EditDescriptor:
    """Call ``wsp.zoo.<family>.generate`` (imported lazily; it pulls in heretic/peft/datasets)."""
    try:
        module = importlib.import_module(f"wsp.zoo.{family}")
    except ImportError as e:
        raise RuntimeError(f"edit family {family!r} cannot be imported: {e}") from e
    gen_params = {k: v for k, v in params.items() if k != "delta_format"}
    gen_params["base_revision"] = config.get("base_revision", "main")
    return module.generate(config["base_model"], gen_params, seed, out_dir)


def _lora_descriptor(
    family: str,
    config: dict[str, Any],
    params: dict[str, Any],
    seed: int,
    generated: EditDescriptor,
    parent_id: str | None = None,
) -> EditDescriptor:
    """Builder-level descriptor (id = planned id) carrying what ``generate`` measured."""
    if generated.base_revision != config["base_revision"]:
        raise EntryIdentityMismatch(
            f"{family} generate() resolved base {generated.base_revision!r} but the build pinned "
            f"{config['base_revision']!r}"
        )
    desc = _descriptor(
        family,
        params,
        seed,
        config,
        parent_id,
        generated.git_sha,
        generated.touched_tensors,
        generated.delta_fro_norm,
    )
    desc.provenance.update(generated.provenance)
    # the stored format is what the generator actually wrote (heretic_ara exports a dense
    # evaluated delta; lora_sft writes a PEFT adapter), never the planning label
    desc.provenance.update(
        {
            "delta_format": generated.params.get("delta_format") or "lora",
            "generator_edit_id": generated.id,
            "resolved_base_revision": generated.base_revision,
            "generator_git_dirty": generated.params.get("git_dirty"),
            "generator_git_tree": generated.params.get("git_tree"),
        }
    )
    return desc


def build_lora_entry(
    family: str,
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    zoo_dir: Path,
    git_sha: str,
    extra_provenance: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> EditDescriptor:
    """Generate one LoRA-format entry directly into ``zoo_dir/<id>/`` (edit.json + lora/ + logs).

    Generation happens in a staging dir and is moved into place only on success; an entry that
    already has artifacts is reused, since heretic_ara/lora_sft runs are expensive.
    """
    params = _LORA_SAMPLERS[family](random.Random(seed), family_cfg)
    planned = _descriptor(family, params, seed, config, None, git_sha)
    planned.provenance.update(extra_provenance)
    finalize_identity(planned, config)
    final_dir = zoo_dir / planned.id
    if check_reusable(zoo_dir, manifest, planned):
        return EditDescriptor.from_json((final_dir / "edit.json").read_text())

    stage = zoo_dir / STAGING_DIR / planned.id
    shutil.rmtree(stage, ignore_errors=True)
    stage.parent.mkdir(parents=True, exist_ok=True)
    generated = _run_generate(family, config, params, seed, stage)
    desc = _lora_descriptor(family, config, params, seed, generated)
    desc.git_sha = git_sha
    desc.provenance.update(extra_provenance)
    finalize_identity(desc, config)
    if (stage / "edit.json").exists():
        (stage / "edit.json").rename(stage / "generator_edit.json")
    (stage / "edit.json").write_text(desc.to_json())
    shutil.rmtree(final_dir, ignore_errors=True)
    shutil.move(str(stage), str(final_dir))
    return desc


def _build_lora_in_memory(
    family: str,
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    zoo_dir: Path,
    dry_run: bool,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    """Chain-step variant: generate, densify the adapter, add it onto ``base_state``.

    ``generate`` always trains/optimises against the *base checkpoint*, so inside a chain the
    step's edit is the base-relative delta added to the accumulated state (task-arithmetic style),
    not an edit optimised against the accumulated model.
    """
    params = _LORA_SAMPLERS[family](random.Random(seed), family_cfg)
    if dry_run:
        return [(_descriptor(family, params, seed, config, None, "", [], None), {}, [])]
    staging = zoo_dir / STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as tmp:
        generated = _run_generate(family, config, params, seed, Path(tmp))
        delta = load_entry_delta(Path(tmp), base_state.keys())
    new_state = dict(base_state)
    for key, dv in delta.items():
        new_state[key] = (new_state[key].float() + dv).to(new_state[key].dtype)
    desc = _lora_descriptor(family, config, params, seed, generated)
    return [(desc, new_state, sorted(delta))]


def parent_exclusion_reason(
    row: dict[str, Any], config: dict[str, Any], include_smoke: bool, dry_run: bool
) -> str | None:
    """Why a manifest row may not be a task-vector parent (None = eligible).

    Legacy rows (no eligibility/reuse_key, i.e. built before the policy) and rows from another
    resolved base commit are never eligible; smoke rows only with ``--include-smoke``.
    """
    if "eligibility" not in row or (not dry_run and "reuse_key" not in row):
        return "legacy_unprovenanced"
    if row.get("base_revision") != config.get("base_revision"):
        return "incompatible_base"
    eligibility = row["eligibility"]
    if eligibility == "production":
        # Re-classify against the *current* thresholds: a row stored as production under looser
        # thresholds must not enter default sampling once the config tightens (Board A final review).
        recomputed, _ = classify_eligibility(
            row.get("family", ""), row.get("params") or {}, {**config, "_overrides": None}
        )
        if recomputed == "smoke":
            eligibility = "smoke"
    if eligibility == "production":
        return None
    if eligibility == "smoke":
        return None if include_smoke else "smoke_under_current_thresholds"
    return "unclassified"


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
    include_smoke = bool(config.get("_include_smoke"))
    candidates, excluded = [], {}
    for m in sorted(manifest, key=lambda m: m["id"]):
        if m.get("family") not in SINGLE_FAMILIES:
            continue
        if not dry_run and not _has_artifacts(zoo_dir, m["id"]):
            continue
        reason = parent_exclusion_reason(m, config, include_smoke, dry_run)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            candidates.append(m)
    if not candidates:
        return []
    parent = rng.choice(candidates)
    parent_id = parent["id"]
    pool_digest = hashlib.sha256(",".join(m["id"] for m in candidates).encode()).hexdigest()

    touched: list[str] = []
    new_state: dict[str, torch.Tensor] = {}
    if not dry_run:
        parent_delta = load_entry_delta(zoo_dir / parent_id, base_state.keys())
        new_state = dict(base_state)
        for key, dv in parent_delta.items():
            if key in new_state:
                cur = new_state[key]
                new_state[key] = (cur.float() + params["scale"] * dv.float()).to(cur.dtype)
            else:
                new_state[key] = params["scale"] * dv
            touched.append(key)
    desc = _descriptor("task_vector", params, seed, config, parent_id, "", touched, None)
    desc.provenance.update(
        {
            "parent_candidates_sha256": pool_digest,
            "parent_pool_excluded": excluded,
            "include_smoke": include_smoke,
            "parent_reuse_key": parent.get("reuse_key"),
            "parent_eligibility": parent["eligibility"],
        }
    )
    if parent["eligibility"] != "production":
        desc.eligibility = "smoke"
        desc.eligibility_reasons.append(f"parent {parent_id} is {parent['eligibility']}")
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
    if family == "llm_edit":
        from wsp.editor.zoo import build

        return build(config, family_cfg, seed, base_state, dry_run=dry_run)
    if family == "random_dir":
        return build_random_dir(config, family_cfg, seed, base_state, dry_run=dry_run)
    if family == "abliteration":
        return build_abliteration(config, family_cfg, seed, base_state, zoo_dir, dry_run=dry_run)
    if family in LORA_FAMILIES:
        return _build_lora_in_memory(
            family, config, family_cfg, seed, base_state, zoo_dir, dry_run=dry_run
        )
    if family == "task_vector":
        return build_task_vector(
            config, family_cfg, seed, base_state, zoo_dir, manifest, dry_run=dry_run
        )
    raise ValueError(f"unknown single-edit family: {family}")


def chain_step_semantics(family: str) -> str:
    """How a chain step's edit relates to the accumulated state.

    ``generate()`` of the LoRA families always optimises against the *base checkpoint*, so inside a
    chain the step is a base-relative delta added to the accumulated state (task arithmetic),
    ``lora_vs_base``. random_dir/abliteration edit the accumulated weights directly: ``sequential``.
    The sequential LoRA variant (optimising against the accumulated state) needs generate() to
    start from a given state dict and is not implemented; see docs/ZOO_IDENTITY.md.
    """
    return LORA_VS_BASE if family in LORA_FAMILIES else SEQUENTIAL


def build_chain(
    config: dict[str, Any],
    family_cfg: dict[str, Any],
    seed: int,
    base_state: dict[str, torch.Tensor],
    zoo_dir: Path,
    manifest: list[dict[str, Any]],
    dry_run: bool = False,
) -> list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]]:
    """Chain entries are cumulative snapshots: entry k stores (state after step k) - base.

    H2 consumers must take consecutive differences of snapshots for the per-step delta, never
    apply a snapshot onto its parent. Each entry records the step's ``step_semantics`` (in params,
    hence in the id), the semantics of every step so far, and stored vs source delta format.
    """
    rng = random.Random(seed)
    length = int(sample_param(rng, family_cfg["length"]))
    available = [f for f in SINGLE_FAMILIES if f in family_policy(config)["families_available"]]
    if not available:
        raise ValueError("no single-edit families configured; chain cannot be built")

    cumulative = {} if dry_run else dict(base_state)
    entries: list[tuple[EditDescriptor, dict[str, torch.Tensor], list[str]]] = []
    all_touched: set[str] = set()
    parent_id: str | None = None
    semantics: list[str] = []
    identities: list[dict[str, Any]] = []
    smoke_reasons: list[str] = []
    for step in range(length):
        family = rng.choice(available)
        step_seed = seed + step + 1
        inner_entries = _build_single_family(
            family,
            config,
            config["families"][family],
            step_seed,
            cumulative,
            zoo_dir,
            manifest,
            dry_run=dry_run,
        )
        inner_desc, cumulative, step_touched = inner_entries[0]
        all_touched.update(step_touched)
        step_sem = chain_step_semantics(family)
        semantics.append(step_sem)
        params = {
            "length": length,
            "step": step,
            "inner_family": family,
            "inner_params": inner_desc.params,
            "inner_seed": step_seed,
            "step_semantics": step_sem,
        }
        desc = _descriptor("chain", params, seed, config, parent_id, "", [], None)
        source_format = "lora" if inner_desc.params.get("delta_format") == "lora" else "dense"
        inherited = {
            k: v
            for k, v in inner_desc.provenance.items()
            if k not in ("delta_format", "reuse_identity", "reuse_key")
        }
        desc.provenance.update(inherited)
        desc.provenance.update(
            {
                "step_semantics": step_sem,
                "chain_step_semantics": list(semantics),
                "source_delta_format": source_format,
                "stored_delta_format": "dense_cumulative",
                "cumulative_snapshot": True,
            }
        )
        identities.append(_step_identity(inner_desc, config))
        desc.provenance["step_identities"] = list(identities)
        if inner_desc.eligibility != "production":
            smoke_reasons.append(
                f"step {step} ({family}): {'; '.join(inner_desc.eligibility_reasons)}"
            )
        if smoke_reasons:
            desc.eligibility = "smoke"
            desc.eligibility_reasons = [*desc.eligibility_reasons, *smoke_reasons]
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
        **_arch.remote_code_kwargs(config["base_model"], config),
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
    overrides: list[str] | None = None,
    include_smoke: bool = False,
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
        overrides: ``key.path=value`` config overrides (e.g. ``heretic_ara.trials=2``); they are
            folded into the config hash and recorded in the header and every edit's provenance.
            Entries built with an applicable override are ``eligibility: smoke``.
        include_smoke: let ``eligibility: smoke`` entries (reduced budget / overrides) be sampled
            as task-vector parents (legacy and other-base entries are never eligible); the children
            are then smoke too.

    Returns:
        The manifest as a list of line dicts.
    """
    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    cfg_hash = config_hash(config_path, overrides)
    # Pin the base *before* any id is built: ids, refusal cache, generators and the CPU base copy
    # all use the resolved commit, so upstream drift can never alias an old entry.
    requested_revision = config.get("base_revision", "main")
    revision_source = "snapshot"
    try:
        config["base_revision"] = resolve_base_revision(config["base_model"], requested_revision)
    except Exception as e:
        if not (dry_run and _is_access_error(e)):
            raise
        # Gated repo without approval: a dry run only plans ids, so the Hub's commit metadata (readable
        # without file access) pins the revision; a real build still fails above.
        config["base_revision"] = metadata_revision(config["base_model"], requested_revision)
        revision_source = "hub metadata (repo files not accessible: gated)"
    config["_overrides"] = list(overrides or [])
    config["_include_smoke"] = bool(include_smoke)
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
    info = _arch.load_arch_info(config)
    if info is not None:
        _arch.validate_family_config(config, info)
    config["_arch"] = info
    provenance = {
        **policy,
        "allow_missing_families": bool(allow_missing_families),
        "requested_base_revision": requested_revision,
        "include_smoke": bool(include_smoke),
        "base_revision_source": revision_source,
        "architecture": _architecture_record(info),
    }
    if overrides:
        provenance["config_overrides"] = list(overrides)
    families = [family] if family else policy["families_available"]

    base_state: dict[str, torch.Tensor] = {}
    # LoRA families load their own model in generate(); skip the CPU copy if only they are built.
    if not dry_run and any(f not in LORA_FAMILIES for f in families):
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
        "base_revision": config["base_revision"],
        "seed": int(config["seed"]),
        "families_selected": [family] if family else policy["families_available"],
        **provenance,
    }
    if header not in headers:
        _append_manifest(manifest_path, header)

    dtype = getattr(torch, config.get("dtype", "bfloat16"))

    for fam in families:
        family_cfg = config["families"][fam]
        count = n if n is not None else int(family_cfg.get("n", 0))
        for i in range(count):
            seed = int(config["seed"]) + i
            if fam in LORA_FAMILIES and not dry_run:
                desc = build_lora_entry(
                    fam, config, family_cfg, seed, zoo_dir, git_sha, provenance, manifest
                )
                entries = [(desc, None, [])]
            elif fam == "chain":
                entries = build_chain(
                    config, family_cfg, seed, base_state, zoo_dir, manifest, dry_run=dry_run
                )
            else:
                entries = _build_single_family(
                    fam, config, family_cfg, seed, base_state, zoo_dir, manifest, dry_run=dry_run
                )
            for desc, new_state, touched in entries:
                if new_state is not None:  # LoRA entries are finalised in build_lora_entry
                    desc.git_sha = git_sha
                    desc.provenance.update(provenance)
                if not dry_run and new_state is not None:
                    finalize_identity(desc, config)
                    check_reusable(zoo_dir, manifest, desc)  # raises on a same-id/other-key entry
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
                    "base_revision": desc.base_revision,
                    "eligibility": desc.eligibility,
                    "stored_delta_format": _stored_format(desc),
                }
                if "reuse_key" in desc.provenance:
                    line["reuse_key"] = desc.provenance["reuse_key"]
                if desc.params.get("delta_format") == "lora" or "delta_format" in desc.provenance:
                    line["delta_format"] = desc.provenance.get("delta_format") or "lora"
                for key in ("step_semantics", "chain_step_semantics"):
                    if key in desc.provenance:
                        line[key] = desc.provenance[key]
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
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. heretic_ara.trials=2 or lora_sft.steps=5 (repeatable)",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="allow smoke-eligibility entries (reduced budget / overrides) as task-vector parents",
    )
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
        overrides=args.override,
        include_smoke=args.include_smoke,
    )


if __name__ == "__main__":
    main()
