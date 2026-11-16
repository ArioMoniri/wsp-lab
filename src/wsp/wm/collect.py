"""Grouped, streamed, support-only gradient and K-FAC collector (representation's memory contract, SYNTHESIS-r8 D8).

Numerically the same estimator as ``wsp.features.response_field.collect_field`` (same scorer adapters, panels,
seeded model-sampled Fisher targets, fp32 sufficient statistics, the same per-item two-pass order), restructured so
a 9B target fits on one GPU:

* objectives and layer groups are processed one at a time; only the current group's parameters require gradients
  (later frozen layers still propagate derivatives); the Fisher RNG is reseeded per (objective, group) so every group
  replays exactly the reference draws;
* gradients and A/G factors accumulate in fp32 on the model's device and are written to disk per (objective, group),
  ``<objective>.g<i>.pt``; nothing for other groups is retained;
* support-only: only the declared modules' parameters are collected; every requested parameter gets an explicit
  coverage row, and unsupported tensors are marked with a reason (``not_linear`` for fused/recurrent/conv/norm
  operators, ``vocabulary_output`` for vocabulary-sized Linears) instead of being given q = 0;
* a vocabulary-sized covariance is never formed: vocabulary Linears get gradients only, never Fisher hooks.

Equivalence with ``response_field`` is tested on CPU toy models (``tests/test_wm_collect.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from wsp.features import response_field as rf

SCHEMA = "wme-grouped-field-v1"


class UnsupportedTensor(ValueError):
    pass


def _hash(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def _is_vocab_linear(name: str, mod: nn.Linear, vocab: int | None) -> bool:
    if name.split(".")[-1] in ("lm_head", "embed_tokens"):
        return True
    return bool(vocab) and (mod.out_features == vocab or mod.in_features == vocab)


def _rss_peak_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _group_pass(model, tokenizer, obj, panel, group_params, hooked, device, fisher):
    """One objective x one group: per item, a gradient pass then (optionally) a sampled-Fisher pass."""
    grads: dict[str, torch.Tensor] = {}
    sums: dict[str, dict] = {}
    handles = []
    active = {"fisher": False, "mask": None}

    def forward(name, module, inputs, output):
        if not active["fisher"]:
            return
        if not output.requires_grad:
            raise rf.MissingHookError(f"{name}: output does not require gradients")
        x = inputs[0].detach().float()
        if module.bias is not None:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)

        def backward(dy):
            mask = active["mask"]
            if mask is None or tuple(mask.shape) != tuple(x.shape[:-1]):
                raise rf.MissingHookError(f"{name}: valid-position mask/layout mismatch")
            xv = x[mask].reshape(-1, x.shape[-1])
            gv = dy.detach().float()[mask].reshape(-1, dy.shape[-1])
            if len(xv) == 0:
                return
            a, g = xv.T @ xv, gv.T @ gv
            if name not in sums:
                sums[name] = {"A": a, "G": g, "count": len(xv)}
            else:
                sums[name]["A"] += a
                sums[name]["G"] += g
                sums[name]["count"] += len(xv)

        output.register_hook(backward)

    if fisher:
        for name, module in hooked.items():
            handles.append(
                module.register_forward_hook(lambda m, x, y, name=name: forward(name, m, x, y))
            )
    generator = torch.Generator(device="cpu").manual_seed(rf.SEED)
    total, item_hashes = 0.0, []
    try:
        for item in panel.items():
            item_hashes.append(rf._hash(rf._hash_item(item)))
            model.zero_grad(set_to_none=True)
            batch = obj.scorer(model, tokenizer, item, device)
            if (
                not math.isfinite(batch.weight)
                or batch.weight <= 0
                or not torch.isfinite(batch.score_sum)
            ):
                raise ValueError("nonfinite score or nonpositive objective weight")
            batch.score_sum.backward()
            total += batch.weight
            for pname, p in group_params.items():
                if p.grad is not None:
                    g = p.grad.detach().float()
                    if pname not in grads:
                        grads[pname] = g.clone()
                    else:
                        grads[pname] += g
            del batch
            model.zero_grad(set_to_none=True)
            if fisher:
                active["fisher"] = True
                batch = obj.scorer(model, tokenizer, item, device)
                mask = batch.valid_mask.to(device=device, dtype=torch.bool)
                active["mask"] = mask
                logits = batch.logits.float()
                if logits.dim() == mask.dim() + 1 and tuple(logits.shape[:-1]) == tuple(mask.shape):
                    logits = logits[mask]
                elif not (logits.dim() == 2 and logits.shape[0] == int(mask.sum())):
                    raise ValueError(
                        "logits must be [*mask.shape, vocab] or gathered [n_valid, vocab]"
                    )
                if not len(logits):
                    raise ValueError("empty valid-token mask")
                targets = torch.multinomial(
                    logits.detach().softmax(-1).cpu(), 1, generator=generator
                )
                F.cross_entropy(logits, targets[:, 0].to(logits.device), reduction="sum").backward()
                del batch, logits, targets
                active["fisher"] = False
                model.zero_grad(set_to_none=True)
        if not total:
            raise ValueError("empty objective panel")
        for name in hooked if fisher else ():
            if name not in sums:
                raise rf.MissingHookError(f"{name}: supported module without valid hooks")
        for pair in sums.values():
            pair["A"] /= pair["count"]
            pair["G"] /= pair["count"]
        for name, g in grads.items():
            g.div_(total)
            if not torch.isfinite(g).all():
                raise ValueError(f"nonfinite gradient: {name}")
        return grads, sums, rf._hash(item_hashes)
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)


class GroupedField:
    """Manifest + per-(objective, group) tensor files; tensors are loaded on demand and cached on CPU."""

    def __init__(self, root: Path, identity: dict):
        self.root = Path(root)
        self.identity = identity
        self.coverage = identity["coverage"]
        self._cache: dict[str, dict] = {}
        self._kept: dict[tuple, tuple] = {}

    @classmethod
    def load(cls, root) -> GroupedField:
        root = Path(root)
        man = json.loads((root / "manifest.json").read_text())
        if man["identity"]["schema"] != SCHEMA:
            raise ValueError("grouped field schema mismatch")
        field = cls(root, man["identity"])
        for f, h in man["file_sha256"].items():
            if hashlib.sha256((root / f).read_bytes()).hexdigest() != h:
                raise ValueError(f"{f}: content hash mismatch")
        return field

    def _file(self, objective, group):
        key = f"{objective}.g{group}.pt"
        if key not in self._cache:
            self._cache[key] = torch.load(self.root / key, map_location="cpu", weights_only=True)
        return self._cache[key]

    def drop_cache(self):
        self._cache.clear()

    def gradient(self, objective, param):
        row = self.coverage[objective].get(param)
        if row is None or not row["gradient"]:
            raise UnsupportedTensor(f"{param}: no gradient collected for {objective}")
        return self._file(objective, row["group"])["gradients"][param]

    def factors(self, objective, module):
        for row in self.coverage[objective].values():
            if row.get("module") == module and row["factors"]:
                return self._file(objective, row["group"])["factors"].get(module)
        return None

    def kept_factors(self, objective, module):
        """Spectrally audited, damped, 99%-energy factors (``response_field.spectral_factor``), cached."""
        key = (objective, module)
        if key not in self._kept:
            f = self.factors(objective, module)
            self._kept[key] = (rf.spectral_factor(f["A"])[0], rf.spectral_factor(f["G"])[0])
        return self._kept[key]


def collect_grouped(
    model,
    tokenizer,
    objectives,
    panels,
    groups,
    device,
    out_dir,
    fisher: bool = True,
    vocab_size: int | None = None,
    theta_hash: str | None = None,
    log=None,
):
    """Collect support-only gradients and K-FAC factors one (objective, group) at a time; returns a GroupedField."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    objectives = list(objectives)
    if not objectives or len({o.name for o in objectives}) != len(objectives):
        raise ValueError("nonempty unique objective names required")
    groups = [list(g) for g in groups]
    flat = [m for g in groups for m in g]
    if len(set(flat)) != len(flat):
        raise ValueError("a module belongs to at most one group")
    if vocab_size is None:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    plan = []  # per group: (params, hooked modules, coverage rows)
    for gi, group in enumerate(groups):
        gparams, hooked, rows = {}, {}, {}
        for mname in group:
            if mname not in modules:
                raise KeyError(f"unknown module {mname}")
            mod = modules[mname]
            if isinstance(mod, nn.Linear):
                reason = "vocabulary_output" if _is_vocab_linear(mname, mod, vocab_size) else None
            else:
                reason = "not_linear"
            if reason is None:
                hooked[mname] = mod
            for pn, p in mod.named_parameters(recurse=False):
                full = f"{mname}.{pn}"
                gparams[full] = p
                rows[full] = {
                    "gradient": True,
                    "factors": reason is None and fisher,
                    "module": mname if reason is None else None,
                    "kind": pn,
                    "reason": reason if reason else (None if fisher else "fisher_not_collected"),
                    "group": gi,
                    "shape": list(p.shape),
                }
        plan.append((gparams, hooked, rows))
    if theta_hash is None:
        from wsp.wm.actions import state_hash

        theta_hash = state_hash({n: p for n, p in params.items()})
    identity = {
        "schema": SCHEMA,
        "seed": rf.SEED,
        "precision": "fp32 sufficient statistics; bf16/fp32 model as loaded",
        "fisher": "model-sampled per-valid-token NLL sum; one target/token" if fisher else None,
        "theta_hash": theta_hash,
        "groups": groups,
        "objectives": [o.identity() for o in objectives],
        "coverage": {},
        "panels": {},
        "timing": {},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    flags = {n: p.requires_grad for n, p in params.items()}
    training = {m: m.training for m in model.modules()}
    model.eval()
    file_hashes = {}
    try:
        for obj in objectives:
            panel = panels[obj.panel_id]
            panel.validate()
            if panel.panel_id != obj.panel_id or panel.text_group_hash != obj.text_group_hash:
                raise ValueError("objective panel identity mismatch")
            cov = identity["coverage"][obj.name] = {}
            for gi, (gparams, hooked, rows) in enumerate(plan):
                for p in params.values():
                    p.requires_grad_(False)
                for p in gparams.values():
                    p.requires_grad_(True)
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                grads, sums, content = _group_pass(
                    model, tokenizer, obj, panel, gparams, hooked, device, fisher
                )
                wall = time.perf_counter() - t0
                payload = {
                    "gradients": {k: v.cpu() for k, v in grads.items()},
                    "factors": {
                        k: {"A": v["A"].cpu(), "G": v["G"].cpu(), "count": v["count"]}
                        for k, v in sums.items()
                    },
                }
                del grads, sums
                fname = f"{obj.name}.g{gi}.pt"
                torch.save(payload, out / fname)
                del payload
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.empty_cache()  # release this group's cached blocks before the next pass
                file_hashes[fname] = hashlib.sha256((out / fname).read_bytes()).hexdigest()
                cov.update(rows)
                identity["panels"][obj.name] = {"panel_id": panel.panel_id, "content_hash": content}
                rec = {"seconds": wall, "rss_peak_bytes": _rss_peak_bytes()}
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    rec["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    rec["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                identity["timing"][f"{obj.name}.g{gi}"] = rec
                if log:
                    log(f"collected {obj.name} group {gi}: {rec}")
    finally:
        for n, p in params.items():
            p.requires_grad_(flags[n])
        for m, mode in training.items():
            m.training = mode
    (out / "manifest.json").write_text(
        json.dumps({"identity": identity, "file_sha256": file_hashes}, indent=2, sort_keys=True)
    )
    return GroupedField(out, identity)


def contract(field: GroupedField, deltas: dict[str, torch.Tensor], device="cpu"):
    """Per objective and tensor: signed <g, D> and q = 1/2 tr(G D A D^T) (None when unsupported, never 0)."""
    result = {}
    for obj in field.coverage:
        rows = result[obj] = {}
        for name, value in deltas.items():
            row = field.coverage[obj].get(name)
            if row is None or not row["gradient"]:
                raise UnsupportedTensor(f"{name}: not in the collected support for {obj}")
            d = value.detach().float().to(device)
            if list(d.shape) != row["shape"]:
                raise ValueError(f"{name}: shape mismatch")
            g = field.gradient(obj, name).to(device)
            signed = float((g * d).sum())
            q = None
            if row["factors"]:
                a, gg = field.kept_factors(obj, row["module"])
                a, gg = a.to(device), gg.to(device)
                if row["kind"] == "bias":
                    a = a[-1:, -1:]
                    matrix = d[:, None]
                else:
                    a = a[: d.shape[-1], : d.shape[-1]]
                    matrix = d
                q = max(0.0, float(0.5 * torch.trace(gg @ matrix @ a @ matrix.T)))
            rows[name] = {
                "signed": signed,
                "q": q,
                "supported": row["factors"],
                "reason": row["reason"],
            }
    return result
