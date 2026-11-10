"""Base-anchored A5.8.1 sensitivity fields (Fisher features, not Hessians).

Scorers are differentiable adapters, called as ``scorer(model, tokenizer, item,
device) -> ScoreBatch``. They supply the *sum* of named scores and its averaging
weight, and logits/mask for a separate model-sampled Fisher pass. A panel factory
must be replayable: collection streams one item (microbatch) per forward, per
objective, without retaining graphs. Fused operators without ordinary Linear
hooks are unsupported and must never be advertised in supported_modules.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Schema 2 (A5.8.1 amendment, 10 Nov 2026): fp64 symmetrisation/eigendecomposition, lambda_max-relative
# negative-eigenvalue tolerance with clamp counts in the coverage block.
SCHEMA_VERSION = 2
SEED = 20261107
DAMPING = 1e-3
ENERGY = 0.99
NEGATIVE_TOLERANCE = 1e-6  # eigenvalues in [-1e-6 x lambda_max, 0) are clamped to 0 and counted
SYMMETRY_TOLERANCE = 1e-6  # max|F - F^T| <= 1e-6 x max|F|
# Harmful and harmless refusal panels remain distinct objectives/roles.
OBJECTIVES = {
    "wikitext_nll": "lower",
    "mmlu_gold_log_prob": "higher",
    "hellaswag_gold_char_log_likelihood": "higher",
    "refusal_margin_harmful": "higher",
    "refusal_margin_harmless": "higher",
}
# Registered field sample sizes (items, prompts or windows) per docs/prereg/a5_manifests/panels.json
# objective. A5.8.1's generic clause is 64 groups per objective; chair ruling A5.3.2(i) (7 Nov 2026,
# Board A 7 Nov 2026) registers the whole unexposed 20-prompt harmful pool, overriding the generic
# clause for that objective only. MMLU-Pro/HellaSwag use exactly the first 64 of their 500 ranked
# groups (A5.3.2(iii)); WikiText uses its 64 windows from whole held-out articles.
REGISTERED_FIELD_ITEMS = {
    "wikitext_nll": 64,
    "mmlu_gold_log_prob": 64,
    "hellaswag_gold_char_log_likelihood": 64,
    "refusal_margin_harmful": 20,
    "refusal_margin_harmless": 64,
    "probe_response": 64,
}
RANKED_RESERVE = 500  # MMLU-Pro and HellaSwag reserved ranking length (all G2 exposure)


def registered_field_items(panels_doc, objective):
    """Field item/window ids of a registered objective from panels.json; fails closed.

    The size check uses REGISTERED_FIELD_ITEMS (20 for refusal_margin_harmful, never 64), takes
    the first 64 of a 500-group ranking without choosing another prefix, refuses a panel whose
    computed certificate is not disjoint, and refuses a harmful panel without the registered
    size exception or with the rejected extraction fill active.
    """
    if objective not in REGISTERED_FIELD_ITEMS:
        raise ValueError(f"{objective!r} is not a registered field objective")
    want = REGISTERED_FIELD_ITEMS[objective]
    try:
        panel = panels_doc["panels"][objective]
    except (KeyError, TypeError):
        raise ValueError(f"{objective!r}: no panel in the manifest") from None
    if not panel.get("certificate", {}).get("computed_disjoint"):
        raise ValueError(f"{objective!r}: computed disjointness certificate missing or failed")
    if objective == "wikitext_nll":
        ids = list(panel.get("window_ids", []))
    elif "field_prefix" in panel:
        if panel["field_prefix"] != want or len(panel.get("item_ids", [])) != RANKED_RESERVE:
            raise ValueError(f"{objective!r}: field must be the first {want} of {RANKED_RESERVE}")
        ids = list(panel["item_ids"][:want])
    else:
        ids = list(panel.get("item_ids", []))
    if objective == "refusal_margin_harmful":
        if panel.get("registered_size") != want or panel.get("n_requested") != want:
            raise ValueError("refusal_margin_harmful: registered 20-prompt size exception missing")
        if "conditional_fill_not_registered" in panel:
            raise ValueError("refusal_margin_harmful: the rejected extraction fill is still active")
    if len(ids) != want or len(set(ids)) != len(ids):
        raise ValueError(f"{objective!r}: {len(ids)} unique field items, registered {want}")
    return ids


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(t.dtype), list(t.shape))).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ObjectiveSpec:
    """Sign is score orientation, never a multiplier on the derivative.

    WikiText: token-weighted NLL; MMLU: equal-item normalized gold-letter
    log-probability; HellaSwag: equal-item gold ending log-likelihood divided by
    character length; refusal: equal-prompt log-mass(refusal)-log-mass(compliance).
    Use distinct harmful/harmless specs. Custom names support CPU experiments.
    """

    name: str
    sign: str
    scorer: Callable
    panel_id: str
    text_group_hash: str

    def __post_init__(self):
        if self.sign not in ("lower", "higher"):
            raise ValueError("sign must be lower or higher")
        if self.name in OBJECTIVES and self.sign != OBJECTIVES[self.name]:
            raise ValueError("objective sign differs from A5.8.1")

    def identity(self):
        import inspect

        try:
            source = inspect.getsource(self.scorer)
        except (OSError, TypeError):
            raise ValueError("scorer must have inspectable source for versioning") from None
        return {k: getattr(self, k) for k in ("name", "sign", "panel_id", "text_group_hash")} | {
            "scorer_sha256": hashlib.sha256(source.encode()).hexdigest()
        }


@dataclass(frozen=True)
class ScoreBatch:
    """Differentiable score sum, positive weight, categorical logits, valid mask.

    Logits have [..., vocabulary] shape. valid_mask covers next-token scoring
    positions and must also match each supported linear's leading dimensions.
    Other layouts require an explicit adapter; they fail rather than guess masks.
    """

    score_sum: torch.Tensor
    weight: float
    logits: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class FieldPanel:
    """Replayable microbatches and a complete exposure inventory.

    Caller supplies *all* development/calibration/sealed scored text groups in
    scored_text_groups. Items should already be hash-selected with
    20261107:field: (64 groups, one item each) and truncated per A5.8.1; small
    panels are allowed for tests. Collection verifies disjointness and hashes
    actual streamed contents, without loading the panel into memory.
    """

    panel_id: str
    items: Callable
    text_groups: tuple[str, ...]
    scored_text_groups: tuple[str, ...]

    @property
    def text_group_hash(self):
        return _hash(sorted(self.text_groups))

    def validate(self):
        if not self.text_groups or len(set(self.text_groups)) != len(self.text_groups):
            raise ValueError("panel needs unique text groups")
        if set(self.text_groups) & set(self.scored_text_groups):
            raise ValueError("auxiliary and scored text groups must be disjoint")


class MissingHookError(RuntimeError):
    """G2 hard failure: supported module has no valid input/gradient hook."""


def spectral_factor(factor):
    """Audit, trace-relative damping, then 99% energy truncation; fp64 eigendecomposition.

    A5.8.1 amendment (10 Nov 2026): the stored fp32 factor is symmetrised and decomposed in float64.
    Symmetry: max|F - F^T| <= SYMMETRY_TOLERANCE x max|F|. Eigenvalues in
    [-NEGATIVE_TOLERANCE x lambda_max, 0) are clamped to 0 and counted (``n_clamped`` and
    ``most_negative_ratio`` = lambda_min / lambda_max, 0.0 when nothing is clamped); anything below
    that, or any negative eigenvalue when lambda_max <= 0, raises. Damping (DAMPING x clamped
    trace / dim) and the 99% energy truncation of the damped spectrum are unchanged. Raw and damped
    traces are retained separately; zero-trace factors remain exactly zero with a flag. The kept
    factor is returned in fp32 (storage dtype).
    """
    f = factor.detach().float().cpu()
    if f.ndim != 2 or f.shape[0] != f.shape[1] or not torch.isfinite(f).all():
        raise ValueError("invalid finite square factor")
    f = f.double()
    largest_entry = f.abs().max().item() if f.numel() else 0.0
    asymmetry = (f - f.T).abs().max().item() if f.numel() else 0.0
    if asymmetry > SYMMETRY_TOLERANCE * largest_entry:
        raise ValueError(
            f"factor symmetry audit failed: max|F-F^T| = {asymmetry:.3e} > "
            f"{SYMMETRY_TOLERANCE:g} x max|F| = {largest_entry:.3e}"
        )
    f = (f + f.T) * 0.5
    values, vectors = torch.linalg.eigh(f)
    lambda_max = values.max().item()
    lambda_min = values.min().item()
    if lambda_min < 0 and (lambda_max <= 0 or lambda_min < -NEGATIVE_TOLERANCE * lambda_max):
        ratio = lambda_min / lambda_max if lambda_max > 0 else float("-inf")
        raise ValueError(
            f"materially negative factor eigenvalue: lambda_min/lambda_max = {ratio:.3e} "
            f"< -{NEGATIVE_TOLERANCE:g}"
        )
    n_clamped = int((values < 0).sum().item())
    most_negative_ratio = lambda_min / lambda_max if n_clamped else 0.0
    values = values.clamp_min(0)
    raw_trace = values.sum().item()
    damping = DAMPING * raw_trace / f.shape[0]
    values = values.flip(0) + damping
    vectors = vectors.flip(1)
    total = values.sum()
    rank = int(torch.searchsorted(values.cumsum(0), ENERGY * total).item()) + 1 if total > 0 else 0
    kept = ((vectors[:, :rank] * values[:rank]) @ vectors[:, :rank].T).float()
    tail = values[rank:].sum().item()
    return kept, {
        "raw_trace": raw_trace,
        "damping": damping,
        "rank": rank,
        "omitted_trace": tail,
        "omitted_energy": tail / total.item() if total > 0 else 0.0,
        "zero_trace": raw_trace == 0,
        "dimension": f.shape[0],
        "lambda_max": lambda_max,
        "n_clamped": n_clamped,
        "most_negative_ratio": most_negative_ratio,
    }


def _clamp_record(pair):
    """Per-factor clamp statistics (A5.8.1 amendment) for the coverage block."""
    return {
        key: {
            "n_clamped": pair[f"{key}_audit"]["n_clamped"],
            "most_negative_ratio": pair[f"{key}_audit"]["most_negative_ratio"],
        }
        for key in ("A", "G")
    }


def _hash_item(item):
    if isinstance(item, torch.Tensor):
        return _tensor_hash(item)
    if isinstance(item, Mapping):
        return {str(k): _hash_item(v) for k, v in item.items()}
    if isinstance(item, (tuple, list)):
        return [_hash_item(v) for v in item]
    return item


@dataclass
class FieldArtifact:
    identity: dict
    gradients: dict
    factors: dict

    @property
    def content_hashes(self):
        hashes = {"identity": _hash(self.identity)}
        for objective, tensors in self.gradients.items():
            for name, tensor in tensors.items():
                hashes[f"gradient/{objective}/{name}"] = _tensor_hash(tensor)
        for objective, modules in self.factors.items():
            for name, pair in modules.items():
                for key in ("A", "G"):
                    hashes[f"factor/{objective}/{name}/{key}"] = _tensor_hash(pair[key])
                hashes[f"metadata/{objective}/{name}"] = _hash(
                    {k: v for k, v in pair.items() if k not in ("A", "G")}
                )
        return hashes

    def save(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        torch.save({"gradients": self.gradients, "factors": self.factors}, root / "tensors.pt")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "identity": self.identity,
                    "content_hashes": self.content_hashes,
                },
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def load(cls, directory, expected_identity=None):
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        identity = manifest["identity"]
        if identity["schema_version"] != SCHEMA_VERSION:
            raise ValueError("field schema version mismatch")
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("field identity mismatch")
        payload = torch.load(root / "tensors.pt", map_location="cpu", weights_only=True)
        field = cls(identity, payload["gradients"], payload["factors"])
        if field.content_hashes != manifest["content_hashes"]:
            raise ValueError("field content hash mismatch")
        return field


def _collect_objective(model, tokenizer, obj, panel, modules, device):
    gradients, sums, handles = {}, {}, []
    active = {"fisher": False, "mask": None}

    def forward(name, module, inputs, output):
        if not active["fisher"]:
            return
        if not output.requires_grad:
            raise MissingHookError(f"{name}: output does not require gradients")
        x = inputs[0].detach().float()
        if module.bias is not None:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)

        def backward(dy):
            mask = active["mask"]
            if mask is None or tuple(mask.shape) != tuple(x.shape[:-1]):
                raise MissingHookError(f"{name}: valid-position mask/layout mismatch")
            xv = x[mask].reshape(-1, x.shape[-1]).cpu()
            gv = dy.detach().float()[mask].reshape(-1, dy.shape[-1]).cpu()
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

    for name, module in modules.items():
        handles.append(
            module.register_forward_hook(lambda m, x, y, name=name: forward(name, m, x, y))
        )
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    total, item_hashes = 0.0, []
    try:
        for item in panel.items():
            item_hashes.append(_hash(_hash_item(item)))
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
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad = parameter.grad.detach().float().cpu()
                    if name not in gradients:
                        gradients[name] = grad.clone()
                    else:
                        gradients[name] += grad
            del batch
            model.zero_grad(set_to_none=True)
            active["fisher"] = True
            batch = obj.scorer(model, tokenizer, item, device)
            mask = batch.valid_mask.to(device=device, dtype=torch.bool)
            active["mask"] = mask
            logits = batch.logits.float()[mask]
            if not len(logits):
                raise ValueError("empty valid-token mask")
            targets = torch.multinomial(logits.detach().softmax(-1).cpu(), 1, generator=generator)
            F.cross_entropy(logits, targets[:, 0].to(device), reduction="sum").backward()
            del batch, logits, targets
            active["fisher"] = False
        if not total:
            raise ValueError("empty objective panel")
        for name in modules:
            if name not in sums:
                raise MissingHookError(f"{name}: supported module without valid hooks")
        for name, pair in sums.items():
            for key in ("A", "G"):
                pair[key] /= pair["count"]
                _, pair[f"{key}_audit"] = spectral_factor(pair[key])
            pair["weight"] = f"{name}.weight" if name else "weight"
            pair["bias"] = (
                (f"{name}.bias" if name else "bias") if modules[name].bias is not None else None
            )
        for name, gradient in gradients.items():
            gradient.div_(total)
            if not torch.isfinite(gradient).all():
                raise ValueError(f"nonfinite gradient: {name}")
        return gradients, sums, _hash(item_hashes)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def collect_field(
    model, tokenizer, objectives, panel, supported_modules, device, dtype_algebra=torch.float32
):
    """Collect sequential objectives with CPU-offloaded fp32 sufficient statistics.

    supported_modules is an iterable of exact module paths or mapping path->role.
    Keep full-attention, GatedDeltaNet and MLP paths/roles distinct. No automatic
    expansion to fused/recurrent/nonlinear tensors is performed. All parameters
    appear in coverage, including unsupported/frozen/unused tensors.
    """
    if dtype_algebra != torch.float32:
        raise ValueError("A5.8.1 requires fp32 algebra")
    objectives = list(objectives)
    if not objectives or len({o.name for o in objectives}) != len(objectives):
        raise ValueError("nonempty unique objective names required")
    known = dict(model.named_modules())
    roles = (
        dict(supported_modules)
        if isinstance(supported_modules, Mapping)
        else {name: name for name in supported_modules}
    )
    modules = {}
    for name in roles:
        if name not in known or not isinstance(known[name], nn.Linear):
            raise MissingHookError(f"{name}: supported module is not a hookable Linear")
        modules[name] = known[name]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "damping": DAMPING,
        "energy": ENERGY,
        "precision": "fp32",
        "spectral": {
            "eigendecomposition": "float64 (symmetrised 0.5(F + F^T)); kept factor stored in fp32",
            "negative_tolerance": f"clamp [-{NEGATIVE_TOLERANCE:g} x lambda_max, 0) to 0; lower fails",
            "symmetry_tolerance": f"max|F - F^T| <= {SYMMETRY_TOLERANCE:g} x max|F|",
        },
        "anchor": "base",
        "fisher": "model-sampled per-valid-token NLL sum; one target/token",
        "objectives": [o.identity() for o in objectives],
        "module_roles": roles,
        "coverage": {},
        "panels": {},
        "code_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent
        ).strip(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_hash": _hash({n: _tensor_hash(t) for n, t in model.state_dict().items()}),
    }
    gradients, factors = {}, {}
    training = {module: module.training for module in model.modules()}
    model.eval()
    try:
        for obj in objectives:
            p = panel[obj.panel_id]
            p.validate()
            if p.panel_id != obj.panel_id or p.text_group_hash != obj.text_group_hash:
                raise ValueError("objective panel identity mismatch")
            grad, pairs, content = _collect_objective(model, tokenizer, obj, p, modules, device)
            gradients[obj.name], factors[obj.name] = grad, pairs
            factored = {v[k]: n for n, v in pairs.items() for k in ("weight", "bias") if v[k]}
            identity["coverage"][obj.name] = {
                n: {
                    "gradient": n in grad,
                    "factors": n in factored,
                    "module": factored.get(n),
                    "shape": list(t.shape),
                    "eigen_clamps": _clamp_record(pairs[factored[n]]) if n in factored else None,
                }
                for n, t in model.named_parameters()
            }
            identity["panels"][obj.name] = {
                "panel_id": p.panel_id,
                "text_group_hash": p.text_group_hash,
                "scored_text_group_hash": _hash(sorted(p.scored_text_groups)),
                "content_hash": content,
            }
    finally:
        for module, mode in training.items():
            module.training = mode
    return FieldArtifact(identity, gradients, factors)


def _delta(field, objective, name, value):
    coverage = field.identity["coverage"][objective]
    if name not in coverage or list(value.shape) != coverage[name]["shape"]:
        raise ValueError(f"unknown tensor or shape mismatch: {name}")
    d = value.detach().float().cpu()
    if not torch.isfinite(d).all():
        raise ValueError(f"nonfinite displacement: {name}")
    return d


def _factor_for_tensor(field, objective, name):
    row = field.identity["coverage"][objective][name]
    if not row["factors"]:
        return None
    pair = field.factors[objective][row["module"]]
    a = spectral_factor(pair["A"])[0]
    g = spectral_factor(pair["G"])[0]
    return (
        a[-1:, -1:]
        if name == pair["bias"]
        else a[
            : field.gradients[objective][name].shape[-1],
            : field.gradients[objective][name].shape[-1],
        ],
        g,
    )


def contract(field, delta_eff_by_tensor):
    """Raw named-score contractions and per-tensor trust-region magnitudes.

    Bias uses its augmented A coordinate. q values are per-tensor features;
    their sum omits weight/bias off-diagonal interactions and is not joint q.
    Unsupported tensors have q=None, never fabricated zero curvature.
    """
    result = {}
    for objective, grads in field.gradients.items():
        result[objective] = {}
        for name, value in delta_eff_by_tensor.items():
            d = _delta(field, objective, name, value)
            if name not in grads:
                raise ValueError(f"missing objective gradient: {name}")
            pair = _factor_for_tensor(field, objective, name)
            q = None
            if pair is not None:
                a, g = pair
                matrix = d[:, None] if d.ndim == 1 else d
                q = max(0.0, (0.5 * torch.trace(g @ matrix @ a @ matrix.T)).item())
            result[objective][name] = {"signed": (grads[name] * d).sum().item(), "q": q}
    return result


def cross_term(field, displacement_by_tensor, delta_eff_by_tensor):
    """Stateless tr(D^T G E A), with full augmented weight/bias coupling."""
    result = {}
    for objective in field.gradients:
        rows = result[objective] = {}
        for name, value in delta_eff_by_tensor.items():
            d = _delta(field, objective, name, value)
            row = field.identity["coverage"][objective][name]
            if not row["factors"]:
                raise ValueError(f"cross term unavailable for unsupported tensor: {name}")
            pair = field.factors[objective][row["module"]]
            a, g = spectral_factor(pair["A"])[0], spectral_factor(pair["G"])[0]
            e = torch.zeros(g.shape[0], a.shape[0])
            for key in ("weight", "bias"):
                tensor = pair[key]
                if tensor in displacement_by_tensor:
                    v = _delta(field, objective, tensor, displacement_by_tensor[tensor])
                    if key == "bias":
                        e[:, -1] = v
                    else:
                        e[:, : v.shape[1]] = v
            transported = g @ e @ a
            part = transported[:, -1] if name == pair["bias"] else transported[:, : d.shape[1]]
            rows[name] = (part * d).sum().item()
    return result


def predecessor_audit(
    model_pred,
    tokenizer,
    objectives,
    panel,
    supported_modules,
    device,
    field,
    displacement_by_tensor,
    delta_eff_by_tensor,
):
    """One exact predecessor audit; caller selects 20 canonical-ID-sorted states.

    Reports the per-objective <=0.1 rule with denominator floor 1e-3 nats.
    G2 requires >=16/20 separately for every objective, not a pooled pass count.
    """
    exact_field = collect_field(model_pred, tokenizer, objectives, panel, supported_modules, device)
    if (
        exact_field.identity["objectives"] != field.identity["objectives"]
        or exact_field.identity["panels"] != field.identity["panels"]
    ):
        raise ValueError("predecessor audit objective/panel identity mismatch")
    exact = contract(exact_field, delta_eff_by_tensor)
    base = contract(field, delta_eff_by_tensor)
    cross = cross_term(field, displacement_by_tensor, delta_eff_by_tensor)
    result = {}
    for objective in exact:
        truth = sum(row["signed"] for row in exact[objective].values())
        estimate = sum(row["signed"] for row in base[objective].values()) + sum(
            cross[objective].values()
        )
        denominator = max(abs(truth), 1e-3)
        error = abs(truth - estimate) / denominator
        result[objective] = {
            "exact": truth,
            "approximate": estimate,
            "denominator": denominator,
            "relative_error": error,
            "passed": error <= 0.1,
        }
    return result
