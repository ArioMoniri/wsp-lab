"""Study M (SYNTHESIS-r12 + 19 Nov addendum): Gemma 3 / MedGemma text-only loading and weight-space lines.

* **Targets.** ``google/medgemma-1.5-4b-it`` is the study M target (text use only). ``google/medgemma-4b-it`` is the
  MedGemma 1 ↔ 1.5 interpolation endpoint and ``google/gemma-3-4b-it`` the task-vector anchor. Revisions are pinned.
* **Text-only load.** The checkpoints are ``Gemma3ForConditionalGeneration`` (SigLIP ``vision_tower`` +
  ``multi_modal_projector`` + ``language_model``). We build a ``Gemma3ForCausalLM`` from ``text_config`` and copy only
  language-model tensors into it (:func:`text_key`), so vision weights are never read. Every text parameter must be
  covered exactly once; an unknown language-model tensor raises. ``lm_head`` is tied to ``embed_tokens``.
* **Weight-space line.** ``θ(t) = bf16((1 − t) · f64(θ_B) + t · f64(θ_O))`` per language-model tensor, computed
  from the two snapshots on disk (never from the live state), in row chunks so the 262k × 2560 embedding needs no
  dense copies. ``t = 0`` rewrites θ_B bytes exactly (restore); ``t = 1`` reproduces θ_O bitwise. The additive form
  ``b + t(o − b)`` is not used: in fp32 it loses ``o`` when |o| and |b| differ by more than 2^24 (5.7k entries on
  the real models in the smoke). Entries where θ_O = θ_B keep θ_B's bytes.
* **Task vector.** τ = θ_target − θ_anchor over language-model weights; θ(λ) = θ_anchor + λτ, i.e. the line from the
  target (B) to the anchor (O) at ``t = 1 − λ`` (:func:`tv_t`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from wsp.wm import actions as A
from wsp.wm.wmm import roles as R

TARGETS = {
    "google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",  # study M target
    "google/medgemma-4b-it": "290cda5eeccbee130f987c4ad74a59ae6f196408",  # interpolation endpoint
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",  # task-vector anchor
}
TARGET = "google/medgemma-1.5-4b-it"
TEXT_PREFIXES = (
    ("language_model.model.", "model."),  # checkpoints as shipped (transformers 4.x serialisation)
    ("model.language_model.", "model."),  # transformers 5 serialisation
    ("language_model.lm_head.", "lm_head."),
    ("model.", "model."),  # a text-only Gemma3ForCausalLM checkpoint
    ("lm_head.", "lm_head."),
)
NON_TEXT = (
    "vision_tower.",
    "multi_modal_projector.",
    "model.vision_tower.",
    "model.multi_modal_projector.",
)


def text_key(ckpt_key: str) -> str | None:
    """Checkpoint key -> ``Gemma3ForCausalLM`` parameter name; None for vision/projector tensors."""
    if ckpt_key.startswith(NON_TEXT) or R.is_frozen(ckpt_key):
        return None
    for src, dst in TEXT_PREFIXES:
        if ckpt_key.startswith(src):
            return dst + ckpt_key[len(src) :]
    raise KeyError(f"unrecognised checkpoint tensor {ckpt_key}")


class WeightSource:
    """Per-tensor reads of language-model weights from a (sharded) safetensors snapshot, by text name."""

    def __init__(self, path):
        from safetensors import safe_open

        self.path = Path(path)
        idx = self.path / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            with safe_open(str(self.path / "model.safetensors"), "pt") as f:
                wm = dict.fromkeys(f.keys(), "model.safetensors")
        self.map: dict[str, tuple[str, str]] = {}
        for k, shard in wm.items():
            t = text_key(k)
            if t is None:
                continue
            if t in self.map:
                raise KeyError(f"{t}: two checkpoint tensors map to it")
            self.map[t] = (k, shard)
        self._open: dict[str, object] = {}

    def _f(self, shard):
        from safetensors import safe_open

        if shard not in self._open:
            self._open[shard] = safe_open(str(self.path / shard), "pt")
        return self._open[shard]

    def names(self) -> list[str]:
        return sorted(self.map)

    def get(self, name: str) -> torch.Tensor:
        k, shard = self.map[name]
        return self._f(shard).get_tensor(k)

    def get_rows(self, name: str, r0: int, r1: int) -> torch.Tensor:
        k, shard = self.map[name]
        return self._f(shard).get_slice(k)[r0:r1]

    def shape(self, name: str) -> tuple[int, ...]:
        k, shard = self.map[name]
        return tuple(self._f(shard).get_slice(k).get_shape())


def text_shapes(path) -> dict[str, list[int]]:
    s = WeightSource(path)
    return {n: list(s.shape(n)) for n in s.names()}


def text_config(path):
    """``Gemma3TextConfig`` with the class defaults filled in (gemma-3-4b-it ships only 6 text keys) and the tie
    flag from :class:`actions.RoleMap` (absent key -> tied for Gemma 3)."""
    from transformers import Gemma3TextConfig

    cfg = json.loads((Path(path) / "config.json").read_text())
    t = dict(cfg.get("text_config", cfg))
    t.pop("model_type", None)
    tc = Gemma3TextConfig(**t)
    tc.tie_word_embeddings = R.GemmaRoleMap.from_config(
        {
            **cfg,
            "text_config": {
                **t,
                "vocab_size": tc.vocab_size,
                "num_hidden_layers": tc.num_hidden_layers,
                "hidden_size": tc.hidden_size,
                "intermediate_size": tc.intermediate_size,
                "model_type": "gemma3_text",
            },
        }
    ).tied
    return tc


def resolved_config(path) -> dict:
    """The snapshot's config with ``text_config`` replaced by its fully resolved form (for RoleMap and signatures)."""
    cfg = json.loads((Path(path) / "config.json").read_text())
    tc = text_config(path)
    t = tc.to_dict()
    t["model_type"] = "gemma3_text"
    t["layer_types"] = list(tc.layer_types)
    out = {k: v for k, v in cfg.items() if k != "vision_config"}
    out["text_config"] = t
    out["tie_word_embeddings"] = bool(tc.tie_word_embeddings)
    return out


ARCH_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "layer_types",
    "sliding_window",
    "rope_theta",
    "rope_local_base_freq",
    "rope_scaling",
    "query_pre_attn_scalar",
    "rms_norm_eps",
    "hidden_activation",
    "final_logit_softcapping",
    "attn_logit_softcapping",
)


def architecture_signature(path) -> dict:
    cfg = resolved_config(path)
    t = cfg["text_config"]
    shapes = text_shapes(path)
    return {
        "architectures": cfg.get("architectures"),
        "text_config": {k: t.get(k) for k in ARCH_KEYS},
        "tied": R.GemmaRoleMap.from_config(cfg).tied,
        "n_text_tensors": len(shapes),
        "n_text_params": int(sum(int(torch.tensor(s).prod()) for s in shapes.values())),
        "shapes_sha256": hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
    }


def compare_architectures(paths: dict) -> dict:
    """Same base architecture: identical text config keys (:data:`ARCH_KEYS`), tie flag and tensor name/shape set."""
    sigs = {k: architecture_signature(p) for k, p in paths.items()}
    shapes = {k: text_shapes(p) for k, p in paths.items()}
    ref = next(iter(paths))
    cfg_diff, shape_mm = {}, []
    for k in paths:
        for key in ARCH_KEYS:
            if sigs[k]["text_config"][key] != sigs[ref]["text_config"][key]:
                cfg_diff.setdefault(k, {})[key] = [
                    sigs[ref]["text_config"][key],
                    sigs[k]["text_config"][key],
                ]
        if sigs[k]["tied"] != sigs[ref]["tied"]:
            cfg_diff.setdefault(k, {})["tied"] = [sigs[ref]["tied"], sigs[k]["tied"]]
        names = set(shapes[k]) | set(shapes[ref])
        shape_mm += [
            [k, n, shapes[ref].get(n), shapes[k].get(n)]
            for n in sorted(names)
            if shapes[ref].get(n) != shapes[k].get(n)
        ]
    return {
        "same": not cfg_diff and not shape_mm,
        "reference": ref,
        "signatures": sigs,
        "config_differences": cfg_diff,
        "shape_mismatches": shape_mm[:20],
        "n_shape_mismatches": len(shape_mm),
        "n_text_tensors": sigs[ref]["n_text_tensors"],
    }


# ------------------------------------------------------------------ loading


def build_text_model(config, device="cpu", dtype=torch.bfloat16):
    from transformers import Gemma3ForCausalLM

    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Gemma3ForCausalLM(config)
    finally:
        torch.set_default_dtype(old)
    return model.to(dtype)


def load_text_from_path(path, device="cpu"):
    """``Gemma3ForCausalLM`` with exactly the checkpoint's language-model tensors; vision is never read."""
    path = Path(path)
    cfg = text_config(path)
    model = build_text_model(cfg, device=device)
    src = WeightSource(path)
    params = dict(model.named_parameters())  # tied lm_head is deduplicated (embed_tokens only)
    if cfg.tie_word_embeddings:
        model.tie_weights()
        params = dict(model.named_parameters())
    unknown = sorted(
        set(src.names()) - set(params) - ({"lm_head.weight"} if cfg.tie_word_embeddings else set())
    )
    missing = sorted(set(params) - set(src.names()))
    if unknown or missing:
        raise KeyError(f"text tensors: unknown {unknown[:5]}, missing {missing[:5]}")
    with torch.no_grad():
        for n, p in params.items():
            t = src.get(n)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{n}: checkpoint {tuple(t.shape)} vs model {tuple(p.shape)}")
            p.copy_(t.to(p.dtype))
    if (
        cfg.tie_word_embeddings
        and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    ):
        raise RuntimeError("lm_head is not tied to embed_tokens")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    R.assert_text_only(n for n, _ in model.named_parameters())
    return model


def snapshot(repo: str) -> Path:
    from huggingface_hub import constants

    path = (
        Path(constants.HF_HUB_CACHE)
        / f"models--{repo.replace('/', '--')}"
        / "snapshots"
        / TARGETS[repo]
    )
    if not (path / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"{repo}@{TARGETS[repo]} not in {path}")
    return path


def base_hash(path, repo: str) -> str:
    p = Path(path)
    shards = {f.name: f.resolve().name for f in sorted(p.glob("*.safetensors"))}
    payload = {
        "repo": repo,
        "revision": TARGETS[repo],
        "shards": shards,
        "load": "text-only Gemma3ForCausalLM bf16",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_text(repo: str, device="cuda"):
    import transformers

    path = snapshot(repo)
    model = load_text_from_path(path, device=device)
    tok = transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    cfg = resolved_config(path)
    return model, tok, cfg, base_hash(path, repo), path


def text_state_hash(model) -> str:
    return A.state_hash({n: p.detach() for n, p in model.named_parameters()})


# ------------------------------------------------------------------ weight-space lines


def tv_t(lam: float) -> float:
    """Line coordinate from the target toward the anchor for task-vector scale λ (θ_anchor + λτ)."""
    return float(1.0 - lam) + 0.0


class WeightLine:
    """θ(t) between snapshot B (t = 0) and snapshot O (t = 1) over the shared language-model tensors."""

    def __init__(self, base: WeightSource, other: WeightSource, chunk_elems: int = 1 << 24):
        nb, no = set(base.names()), set(other.names())
        if nb != no:
            raise KeyError(f"tensor sets differ: {sorted(nb ^ no)[:5]}")
        self.base, self.other, self.chunk = base, other, int(chunk_elems)

    def names(self) -> list[str]:
        return [n for n in self.base.names() if n != "lm_head.weight"]

    @torch.no_grad()
    def apply(self, model, t: float) -> dict:
        """Write θ(t) into ``model`` in place. Returns ``changed_tensors`` (tensors whose bytes differ from the state
        before the call), the intended and realised displacement norms from θ_B, and at t = 1 the entries that
        do not reproduce θ_O exactly."""
        params = dict(model.named_parameters())
        t = float(t)
        changed, inten2, real2, inexact_far = 0, 0.0, 0.0, 0
        for n in self.names():
            p = params[n]
            if self.base.shape(n) != tuple(p.shape) or self.other.shape(n) != tuple(p.shape):
                raise ValueError(f"{n}: shape mismatch")
            rows = p.shape[0]
            per = max(1, self.chunk // max(1, p[0].numel()))
            any_change = False
            for r0 in range(0, rows, per):
                r1 = min(rows, r0 + per)
                b = self.base.get_rows(n, r0, r1).to(p.device)
                if t == 0.0:
                    new = b
                else:
                    o = self.other.get_rows(n, r0, r1).to(p.device)
                    bd, od = b.double(), o.double()
                    delta = t * (od - bd)
                    lerp = ((1.0 - t) * bd + t * od).to(torch.bfloat16)
                    new = torch.where(od != bd, lerp, b)
                    inten2 += float(delta.pow(2).sum())
                    del bd, od
                    real2 += float((new.float() - b.float()).double().pow(2).sum())
                    if t == 1.0:
                        inexact_far += int((new != o).sum())
                    del o, delta
                if not any_change and not A.same_bytes(p[r0:r1].detach(), new):
                    any_change = True
                p[r0:r1].copy_(new)
                del b, new
            changed += int(any_change)
        return {
            "t": t,
            "changed_tensors": changed,
            "intended_norm": inten2**0.5,
            "realised_norm": real2**0.5,
            "far_endpoint_inexact_entries": inexact_far if t == 1.0 else None,
        }
