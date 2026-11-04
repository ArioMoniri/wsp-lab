"""Per-architecture tensor naming for the zoo builder (writers, post-norms, LoRA targets).

A *writer* is a weight matrix whose output is added to the residual stream: its rows live in the
residual space, so a residual-space direction can be projected out of it (``W' = W - w d dᵀ W``).
Per architecture family (``Arch``) we record, as regexes over the in-layer module path (the part
after ``layers.<i>.``, without ``.weight``):

* ``writers['attn_out']`` the token-mixing output projection(s): full attention ``o_proj`` and, in
  hybrids, the linear-attention / short-conv output projection of that layer;
* ``writers['mlp_out']`` the MLP down projection;
* ``post_norms`` the norm applied to a writer's output *before* the residual add (sandwich norm:
  LFM3 ``post_operator_norm`` / ``post_ffn_norm``, Gemma-3 ``post_attention_layernorm`` /
  ``post_feedforward_layernorm``). Its per-channel gain breaks exact orthogonality of an edit made
  in the pre-norm space, see ``wsp.edits.kernels`` (``fold_post_norm``);
* ``lora_targets`` the meaning of the shorthand LoRA targets (``q k v o gate up down``) as PEFT
  module names (suffix match), so each shorthand means the same *role* in every model.

A zoo config may override any of these under ``architecture:`` (see ``configs/zoo_*.yaml``), and
every entry of ``tensors:`` in ``random_dir`` is either a component / legacy name or a regex.

Key layout: the in-memory zoo namespace is ``AutoModelForCausalLM`` state-dict names, i.e.
``model.layers.<i>.<path>.weight``. Multimodal checkpoints (Qwen3.5 on disk) use
``model.language_model.layers.<i>...``; both are indexed here. ``mtp.*`` and ``model.visual.*``
are not language-model layers and are never indexed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

_LAYER_KEY = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)\.weight$")
_LAYER_PREFIX = re.compile(r"^(model\.(?:language_model\.)?layers\.\d+)\.")

LEGACY_LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "q": ("q_proj",),
    "k": ("k_proj",),
    "v": ("v_proj",),
    "o": ("o_proj",),
    "gate": ("gate_proj",),
    "up": ("up_proj",),
    "down": ("down_proj",),
}
# Legacy ``tensors:`` names of the 0.8B config: they name roles, not literal modules.
LEGACY_ALIASES = {"self_attn.o_proj": "attn_out", "mlp.down_proj": "mlp_out"}

# Models whose custom modelling code we allow to run (pinned snapshot under data/hf). Others need
# ``trust_remote_code: true`` in the zoo config.
REMOTE_CODE_ALLOWLIST = frozenset({"LiquidAI/LFM3-1.7B"})


@dataclass(frozen=True)
class Arch:
    name: str
    model_types: tuple[str, ...]
    writers: Mapping[str, tuple[str, ...]]
    post_norms: Mapping[str, str] = field(default_factory=dict)
    post_norms_flag: str | None = None  # text-config key that must be truthy for post_norms
    norm_gain: str = "weight"  # "weight" (T5/LFM RMSNorm) or "one_plus_weight" (Gemma)
    lora_targets: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: LEGACY_LORA_TARGETS)


ARCHS: dict[str, Arch] = {
    "qwen3_5": Arch(
        name="qwen3_5",
        model_types=("qwen3_5", "qwen3_5_text"),
        writers={
            "attn_out": (r"self_attn\.o_proj", r"linear_attn\.out_proj"),
            "mlp_out": (r"mlp\.down_proj",),
        },
    ),
    "lfm3": Arch(
        name="lfm3",
        model_types=("lfm3",),
        writers={
            "attn_out": (r"self_attn\.o_proj", r"conv\.out_proj"),
            "mlp_out": (r"feed_forward\.w2",),
        },
        post_norms={"attn_out": "post_operator_norm", "mlp_out": "post_ffn_norm"},
        post_norms_flag="use_sandwich_norm",
        # w1 = SiLU gate, w3 = up, w2 = down. ``self_attn.gate_proj`` is a per-head output gate,
        # not the MLP gate, so the bare ``gate_proj`` name must never be used here.
        lora_targets={
            **LEGACY_LORA_TARGETS,
            "gate": ("w1",),
            "up": ("w3",),
            "down": ("w2",),
        },
    ),
    "gemma3": Arch(
        name="gemma3",
        model_types=("gemma3_text", "gemma3"),
        writers={"attn_out": (r"self_attn\.o_proj",), "mlp_out": (r"mlp\.down_proj",)},
        post_norms={
            "attn_out": "post_attention_layernorm",
            "mlp_out": "post_feedforward_layernorm",
        },
        norm_gain="one_plus_weight",
    ),
    "dense": Arch(  # generic Llama/Qwen2/Qwen3/Mistral-style decoder
        name="dense",
        model_types=(),
        writers={"attn_out": (r"self_attn\.o_proj",), "mlp_out": (r"mlp\.down_proj",)},
    ),
}
COMPONENTS = ("attn_out", "mlp_out")
ABLITERATION_COMPONENTS = {"attn": "attn_out", "mlp": "mlp_out"}


def detect_arch(model_type: str | None, family: str | None = None) -> Arch:
    """Registry entry for a text-config ``model_type`` (or an explicit ``architecture.family``)."""
    if family:
        if family not in ARCHS:
            raise ValueError(f"unknown architecture family {family!r}; known: {sorted(ARCHS)}")
        return ARCHS[family]
    for arch in ARCHS.values():
        if model_type in arch.model_types:
            return arch
    return ARCHS["dense"]


@dataclass(frozen=True)
class ArchInfo:
    arch: Arch
    hidden_size: int
    num_layers: int
    model_type: str | None
    layer_types: tuple[str, ...] | None
    post_norms: Mapping[str, str]  # empty unless the model really applies post-writer norms
    norm_gain: str
    verified: bool  # True: read from the model's own config; False: declared in the zoo config


def arch_info_from_dict(
    text_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    verified: bool = True,
) -> ArchInfo:
    """Build ``ArchInfo`` from a text-config dict (``hidden_size``, ``num_hidden_layers``, ...).

    ``overrides`` is the zoo config's ``architecture:`` block (``family``, ``writers``,
    ``post_norms``, ``norm_gain``, ``lora_targets``).
    """
    overrides = dict(overrides or {})
    arch = detect_arch(text_config.get("model_type"), overrides.get("family"))
    changes: dict[str, Any] = {}
    if "writers" in overrides:
        writers = {**arch.writers, **{k: tuple(v) for k, v in overrides["writers"].items()}}
        changes["writers"] = writers
    if "lora_targets" in overrides:
        changes["lora_targets"] = {
            **arch.lora_targets,
            **{k: tuple(v) for k, v in overrides["lora_targets"].items()},
        }
    if "norm_gain" in overrides:
        changes["norm_gain"] = overrides["norm_gain"]
    if "post_norms" in overrides:
        changes["post_norms"] = dict(overrides["post_norms"])
        changes["post_norms_flag"] = None
    if changes:
        arch = Arch(**{**arch.__dict__, **changes})
    if "layer_types" in text_config and text_config["layer_types"] is not None:
        layer_types = tuple(text_config["layer_types"])
    else:
        layer_types = None
    flag = arch.post_norms_flag
    active = bool(arch.post_norms) and (flag is None or bool(text_config.get(flag)))
    return ArchInfo(
        arch=arch,
        hidden_size=int(text_config["hidden_size"]),
        num_layers=int(text_config["num_hidden_layers"]),
        model_type=text_config.get("model_type"),
        layer_types=layer_types,
        post_norms=dict(arch.post_norms) if active else {},
        norm_gain=arch.norm_gain,
        verified=verified,
    )


def remote_code_kwargs(base_model: str, config: Mapping[str, Any] | None = None) -> dict[str, bool]:
    """``from_pretrained`` kwargs for models with custom modelling code (explicit opt-in only)."""
    if base_model in REMOTE_CODE_ALLOWLIST or (config or {}).get("trust_remote_code"):
        return {"trust_remote_code": True}
    return {}


def read_text_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Text config (as a dict) of ``config['base_model']`` at its pinned revision; raises if unreadable."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        config["base_model"],
        revision=config.get("base_revision", "main"),
        **remote_code_kwargs(config["base_model"], config),
    )
    return cfg.get_text_config().to_dict()


def load_arch_info(config: Mapping[str, Any]) -> ArchInfo | None:
    """Architecture of ``config['base_model']`` at its pinned revision.

    Reads the model's own config. If the repo is not readable (gated, offline) the zoo config's
    declared ``architecture.hidden_size`` / ``num_layers`` are used and the result is marked
    ``verified=False`` (recorded in the manifest header). A declared value that disagrees with a
    readable config is an error. With neither a readable config nor a declaration there is nothing to
    validate against: returns None (the builder then falls back to Qwen3.5 naming and unchecked layers).
    """
    declared = dict(config.get("architecture") or {})
    try:
        text = read_text_config(config)
    except Exception:  # noqa: BLE001  (gated / offline / unknown revision)
        if not {"hidden_size", "num_layers"} <= set(declared):
            return None
        text = {
            "model_type": declared.get("model_type"),
            "hidden_size": declared["hidden_size"],
            "num_hidden_layers": declared["num_layers"],
            "use_sandwich_norm": declared.get("use_sandwich_norm", False),
        }
        return arch_info_from_dict(text, declared, verified=False)
    info = arch_info_from_dict(text, declared, verified=True)
    check_declared(declared, info)
    return info


def check_declared(declared: Mapping[str, Any], info: ArchInfo) -> None:
    for key, actual in (("hidden_size", info.hidden_size), ("num_layers", info.num_layers)):
        if key in declared and int(declared[key]) != actual:
            raise ValueError(
                f"architecture.{key}={declared[key]} in the zoo config but the model has {actual}"
            )


# ---- key indexing and selection ------------------------------------------------------------------


def split_key(key: str) -> tuple[int, str] | None:
    """``(layer, in-layer module path)`` of a language-model layer weight, else None."""
    m = _LAYER_KEY.match(key)
    return (int(m.group(1)), m.group(2)) if m else None


def index_state_keys(keys: Iterable[str]) -> dict[int, dict[str, str]]:
    """``{layer: {module path: state key}}`` for every ``layers.<i>.<path>.weight``."""
    index: dict[int, dict[str, str]] = {}
    for key in keys:
        parsed = split_key(key)
        if parsed:
            index.setdefault(parsed[0], {})[parsed[1]] = key
    return index


def layer_key(index: Mapping[int, Mapping[str, str]], layer: int, path: str) -> str:
    """State key of module ``path`` (plus ``.weight``) in ``layer``; KeyError if absent."""
    try:
        return index[layer][path]
    except KeyError:
        raise KeyError(f"layer {layer} has no tensor {path}.weight") from None


def resolve_spec(spec: str, arch: Arch) -> tuple[str, ...]:
    """Regexes for one ``tensors:`` entry: a component, a legacy alias, or a regex itself."""
    if spec in COMPONENTS:
        return tuple(arch.writers[spec])
    if spec in LEGACY_ALIASES:
        return tuple(arch.writers[LEGACY_ALIASES[spec]])
    try:
        re.compile(spec)
    except re.error as e:
        raise ValueError(f"invalid tensor regex {spec!r}: {e}") from e
    return (spec,)


def writer_keys(
    patterns: Sequence[str], layer: int, index: Mapping[int, Mapping[str, str]]
) -> list[str]:
    """State keys in ``layer`` whose module path fullmatches any of ``patterns`` (path order)."""
    compiled = [re.compile(p) for p in patterns]
    return [
        key
        for path, key in sorted(index.get(layer, {}).items())
        if any(c.fullmatch(path) for c in compiled)
    ]


def select_keys(
    specs: Sequence[str], layer: int, index: Mapping[int, Mapping[str, str]], arch: Arch
) -> list[str]:
    """Keys selected by every spec in ``layer``, in spec order; each spec must match something."""
    keys: list[str] = []
    for spec in specs:
        found = writer_keys(resolve_spec(spec, arch), layer, index)
        if not found:
            raise KeyError(
                f"tensor spec {spec!r} matches nothing in layer {layer} of a {arch.name} model"
            )
        keys.extend(k for k in found if k not in keys)
    return keys


def post_norm_key(key: str, post_norm_path: str) -> str:
    """State key of the post-writer norm in the same layer as writer ``key``."""
    m = _LAYER_PREFIX.match(key)
    if not m:
        raise KeyError(f"{key} is not a layer tensor")
    return f"{m.group(1)}.{post_norm_path}.weight"


def component_of(key: str, arch: Arch) -> str | None:
    """Writer component (attn_out / mlp_out) of a layer tensor key, or None."""
    parsed = split_key(key)
    if parsed is None:
        return None
    for component in COMPONENTS:
        if any(re.fullmatch(p, parsed[1]) for p in arch.writers[component]):
            return component
    return None


def effective_gain(weight: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-channel multiplier a post-writer RMSNorm applies, in float32."""
    w = weight.float()
    if kind == "weight":
        return w
    if kind == "one_plus_weight":
        return 1.0 + w
    raise ValueError(f"unknown norm_gain {kind!r}")


# ---- validation ----------------------------------------------------------------------------------


def validate_family_config(config: Mapping[str, Any], info: ArchInfo) -> None:
    """Layer indices in the config must exist in this model (a 24-layer config on 18 layers fails)."""
    fams = config.get("families", {})
    n = info.num_layers
    for fam in ("random_dir", "abliteration"):
        for layer in fams.get(fam, {}).get("layers") or []:
            if not 0 <= int(layer) < n:
                raise ValueError(f"{fam}.layers: layer {layer} outside {n}-layer model")
    abl = fams.get("abliteration")
    if abl:
        di = abl.get("direction_index")
        if isinstance(di, dict):
            lo, hi = di.get("min", di.get("log_min", -1)), di.get("max", di.get("log_max", n - 1))
            if lo < -1 or hi > n - 1:
                raise ValueError(
                    f"abliteration.direction_index {lo}..{hi} outside -1..{n - 1} for {n} layers"
                )
        pos = abl.get("max_weight_position")
        if isinstance(pos, dict) and not (0 <= pos.get("min", 0) and pos.get("max", n) <= n):
            raise ValueError(f"abliteration.max_weight_position {pos} outside 0..{n}")


def validate_directions(directions: torch.Tensor, info: ArchInfo) -> None:
    """Refusal directions are ``(num_layers + 1, hidden_size)``: embedding row + every layer."""
    rows, dim = directions.shape
    if dim != info.hidden_size:
        raise ValueError(
            f"refusal directions have width {dim}, model residual is {info.hidden_size}"
        )
    if rows != info.num_layers + 1:
        raise ValueError(f"refusal directions have {rows} rows, expected {info.num_layers + 1}")


def lora_target_modules(targets: Sequence[str], arch: Arch) -> list[str]:
    """Expand shorthand LoRA targets (``q``, ``up``...) to PEFT module names for ``arch``."""
    out: list[str] = []
    for t in targets:
        for name in arch.lora_targets.get(t, (t,)):
            if name not in out:
                out.append(name)
    return out
