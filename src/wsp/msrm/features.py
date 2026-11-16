"""x(D): the MSRM-1 v0 input vector of one root edit (SYNTHESIS-r7 Decision 1, "Inputs x(D)").

One code path from tensors to features: :func:`tensor_record` turns one realised tensor increment into a per-tensor
record (optionally taking cached values such as the F12 a_t, q_t or the pinned stats.py statistics), and
:func:`aggregate` turns the records of an edit into the fixed vector ``COLUMNS``:

* ``STATS18`` (18) global means of the ``wsp.features.stats`` delta statistics over changed tensors (the v4 pooled
  global group); ``stable_rank`` is the top-8 truncated stable rank and is labelled ``g_stable_rank_top8``;
* ``FIELD`` (10) a and q overall and per depth bucket (``pool.py`` buckets, 4 over 24 layers);
* ``ROLE_SHARE`` (17) shares of total squared realised-delta norm per ``pool.py:TYPES`` role, in TYPES order;
* ``MISC`` (3) maximum tensor share, fraction of intended nonzeros lost to bf16 rounding, fraction of the 186
  supported matrix slots changed;
* ``SCRIPTED`` (6) editor/world-model WSR-style columns: global relative Frobenius displacement, cosine(D, W),
  realised/intended norm ratio, delta-energy-weighted depth, energy-weighted top-1 singular-energy share, and the
  energy-weighted |cos| of each residual-writer tensor's leading output vector with the norm-scaled unembedding
  mean.

BASE48 = STATS18 + FIELD + ROLE_SHARE + MISC (the representation 48-vector). Every sum uses ``math.fsum`` so the vector
is exactly invariant to tensor order. Every zero denominator has the pinned value in ``ZERO_CONVENTIONS``; with no
changed tensor every column is 0, so x(0) is the zero vector.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from wsp.features.pool import TYPES, classify
from wsp.features.stats import DELTA_STAT_NAMES, delta_stats

N_LAYERS = 24
BUCKETS = 4
N_SLOTS = 186  # supported matrix slots of the qwen3_5 base (response_chart.slot_layout; F12 tokens)
WRITER_ROLES = ("o", "down")  # tensors whose output space is the residual stream

STATS18 = ["g_stable_rank_top8" if n == "stable_rank" else f"g_{n}" for n in DELTA_STAT_NAMES]
FIELD = ["a", "q"] + [f"a_b{b}" for b in range(BUCKETS)] + [f"q_b{b}" for b in range(BUCKETS)]
ROLE_SHARE = [f"share_{t}" for t in TYPES]
MISC = ["max_tensor_share", "rounding_loss_frac", "frac_slots_changed"]
SCRIPTED = [
    "rel_fro_global",
    "cos_DW",
    "realised_over_intended",
    "depth_energy",
    "top1_energy_share",
    "unemb_abscos",
]
BASE48 = STATS18 + FIELD + ROLE_SHARE + MISC
COLUMNS = BASE48 + SCRIPTED
FIELD_COLUMNS = list(FIELD)
NO_FIELD_COLUMNS = [c for c in COLUMNS if c not in FIELD_COLUMNS]

ZERO_CONVENTIONS = {
    "no_changed_tensor": "every column is 0 (x(0) is the zero vector)",
    "rel_fro_global": "sqrt(sum ||D_t||^2) / sqrt(sum ||W_t||^2) over changed tensors; 0 if the base norm is 0",
    "cos_DW": "sum <D_t, W_t> / (||D|| ||W||) over changed tensors; 0 if either norm is 0",
    "realised_over_intended": "||D|| / ||Delta_intended||; 0 if the intended norm is 0 or unknown",
    "rounding_loss_frac": "intended nonzeros that realise to 0 / intended nonzeros; 0 if no intended nonzero",
    "role_and_max_share": "||D_t||^2 / sum ||D||^2; 0 if the total realised energy is 0",
    "depth_energy": "sum ||D_t||^2 (layer_t / 24) / sum ||D_t||^2 (no layer: 0); 0 if total energy is 0",
    "top1_energy_share": "sum sigma1_t^2 / sum ||D_t||^2; 0 if total energy is 0",
    "unemb_abscos": "sum_w ||D_w||^2 |cos(u1_w, ubar)| / sum_w ||D_w||^2 over residual writers; 0 if no writer "
    "energy or ubar is 0",
    "stats18": "fsum mean over changed tensors; 0 if none",
}


def realise(W_bf16: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Realised increment D = fp32(bf16(W + delta)) - fp32(W) (``a5_power_fields.stored_state`` /
    ``rotation.realised_increment``)."""
    if W_bf16.dtype != torch.bfloat16:
        raise TypeError("the base tensor must be bf16")
    return (W_bf16.float() + delta.float()).to(torch.bfloat16).float() - W_bf16.float()


def lora_delta(B: torch.Tensor, A: torch.Tensor, scale: float) -> torch.Tensor:
    """Dense LoRA delta scale * B @ A in fp32 (the realised D is computed from this product)."""
    return (B.float() @ A.float()) * float(scale)


def field_terms(g, G, A, D) -> tuple[float, float]:
    """(a_t, q_t) = (<g, D>, 1/2 tr(G D A D^T)) in fp64 (A5.8.3 per-tensor form; G, A symmetric PSD)."""
    Dd = D.double()
    a = float(torch.sum(g.double() * Dd))
    q = 0.5 * float(torch.sum((G.double() @ Dd) * (Dd @ A.double())))
    return a, max(q, 0.0) if q > -1e-12 * (1.0 + abs(q)) else q


def _sigma1_and_u1(D: torch.Tensor):
    """Largest singular value and its left singular vector of D (fp64, via eigh of D D^T)."""
    Dd = D.double()
    evals, evecs = torch.linalg.eigh(Dd @ Dd.T)
    lam = float(evals[-1])
    return math.sqrt(max(lam, 0.0)), evecs[:, -1]


def tensor_record(
    name: str,
    W: torch.Tensor,
    D: torch.Tensor,
    intended: torch.Tensor | None = None,
    *,
    a: float | None = None,
    q: float | None = None,
    stats: Sequence[float] | None = None,
    sigma1: float | None = None,
    ubar: torch.Tensor | None = None,
    g=None,
    G=None,
    A=None,
) -> dict:
    """Per-tensor record of a realised increment D of base tensor W (bf16). Cached values (a, q, stats, sigma1)
    override the computed ones; the field terms are computed from (g, G, A) when given and not cached."""
    kind, layer = classify(name)
    Dd = D.double()
    Wf = W.double()
    changed = bool(torch.count_nonzero(D))
    rec = {
        "name": name,
        "role": kind,
        "layer": layer,
        "changed": changed,
        "sq_norm": float(torch.sum(Dd * Dd)),
        "base_sq_norm": float(torch.sum(Wf * Wf)),
        "dot_DW": float(torch.sum(Dd * Wf)),
        "intended_sq_norm": 0.0,
        "n_intended_nonzero": 0,
        "n_lost": 0,
        "a": 0.0,
        "q": 0.0,
        "stats": [0.0] * len(DELTA_STAT_NAMES),
        "sigma1_sq": 0.0,
        "unemb_abscos": None,
        "sigma1_computed": None,
    }
    if intended is not None:
        it = intended.double()
        nz = intended != 0
        rec["intended_sq_norm"] = float(torch.sum(it * it))
        rec["n_intended_nonzero"] = int(torch.count_nonzero(nz))
        rec["n_lost"] = int(torch.count_nonzero(nz & (D == 0)))
    if not changed:
        return rec
    if a is None and g is not None:
        a, q = field_terms(g, G, A, D)
    rec["a"] = float(a or 0.0)
    rec["q"] = float(q or 0.0)
    if stats is None:
        Wf32 = W.float()
        stats = delta_stats(Wf32, Wf32 + D.float()).tolist()
    rec["stats"] = [float(s) for s in stats]
    s1, u1 = None, None
    if kind in WRITER_ROLES and ubar is not None and D.ndim == 2 and D.shape[0] == ubar.numel():
        s1, u1 = _sigma1_and_u1(D)
        rec["sigma1_computed"] = s1
        un = float(torch.linalg.norm(ubar.double()))
        rec["unemb_abscos"] = (
            abs(float(u1 @ ubar.double())) / un if un > 0 else 0.0
        )  # u1 has unit norm
    if sigma1 is None:
        sigma1 = s1 if s1 is not None else float(torch.linalg.svdvals(Dd)[0])
    rec["sigma1_sq"] = float(sigma1) ** 2
    return rec


def _div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def bucket_of(layer: int | None) -> int:
    return 0 if layer is None else min(layer * BUCKETS // N_LAYERS, BUCKETS - 1)


def aggregate(records: Iterable[dict], n_slots: int = N_SLOTS) -> np.ndarray:
    """The COLUMNS vector of one edit from its per-tensor records (order-invariant; pinned zero conventions)."""
    all_recs = list(records)
    recs = [r for r in all_recs if r["changed"]]
    for r in recs:
        if not (r["q"] >= 0.0) or not math.isfinite(r["q"]) or not math.isfinite(r["a"]):
            raise ValueError(f"{r['name']}: field terms must be finite with q >= 0 (got {r['q']})")
    out = dict.fromkeys(COLUMNS, 0.0)
    if not recs:
        return np.zeros(len(COLUMNS))
    n = len(recs)
    for j, c in enumerate(STATS18):
        out[c] = math.fsum(r["stats"][j] for r in recs) / n
    out["a"] = math.fsum(r["a"] for r in recs)
    out["q"] = math.fsum(r["q"] for r in recs)
    for b in range(BUCKETS):
        out[f"a_b{b}"] = math.fsum(r["a"] for r in recs if bucket_of(r["layer"]) == b)
        out[f"q_b{b}"] = math.fsum(r["q"] for r in recs if bucket_of(r["layer"]) == b)
    energy = math.fsum(r["sq_norm"] for r in recs)
    for t in TYPES:
        out[f"share_{t}"] = _div(math.fsum(r["sq_norm"] for r in recs if r["role"] == t), energy)
    out["max_tensor_share"] = _div(max(r["sq_norm"] for r in recs), energy)
    n_int = sum(r["n_intended_nonzero"] for r in all_recs)
    out["rounding_loss_frac"] = _div(sum(r["n_lost"] for r in all_recs), n_int)
    out["frac_slots_changed"] = n / n_slots
    base = math.fsum(r["base_sq_norm"] for r in recs)
    out["rel_fro_global"] = math.sqrt(_div(energy, base))
    out["cos_DW"] = _div(math.fsum(r["dot_DW"] for r in recs), math.sqrt(energy * base))
    intended = math.fsum(r["intended_sq_norm"] for r in all_recs)
    out["realised_over_intended"] = math.sqrt(_div(energy, intended))
    out["depth_energy"] = _div(
        math.fsum(r["sq_norm"] * (r["layer"] / N_LAYERS) for r in recs if r["layer"] is not None),
        energy,
    )
    out["top1_energy_share"] = _div(math.fsum(r["sigma1_sq"] for r in recs), energy)
    writers = [r for r in recs if r["unemb_abscos"] is not None]
    out["unemb_abscos"] = _div(
        math.fsum(r["sq_norm"] * r["unemb_abscos"] for r in writers),
        math.fsum(r["sq_norm"] for r in writers),
    )
    x = np.array([out[c] for c in COLUMNS], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite feature")
    return x


def x_zero() -> np.ndarray:
    """x(0): the input of the zero edit (all slots absent), by the pinned conventions."""
    return aggregate([])


# --------------------------------------------------------------------------------------- name-blind

NAME_BLIND_COLUMNS = (
    STATS18
    + ["a", "q"]
    + [f"a_rank{k}" for k in range(BUCKETS)]
    + [f"q_rank{k}" for k in range(BUCKETS)]
    + [f"share_rank{k}" for k in range(len(TYPES))]
    + MISC
    + [c for c in SCRIPTED if c != "depth_energy"]
)


def name_blind(X: np.ndarray) -> np.ndarray:
    """Role and layer identities replaced by opaque, identity-free order statistics: depth buckets ordered by |a_b|
    (descending, ties by q_b descending then a_b), role shares sorted descending, depth-weighted energy dropped."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    idx = {c: i for i, c in enumerate(COLUMNS)}
    out = np.zeros((X.shape[0], len(NAME_BLIND_COLUMNS)))
    for i, x in enumerate(X):
        ab = [x[idx[f"a_b{b}"]] for b in range(BUCKETS)]
        qb = [x[idx[f"q_b{b}"]] for b in range(BUCKETS)]
        order = sorted(range(BUCKETS), key=lambda b: (-abs(ab[b]), -qb[b], -ab[b]))
        shares = sorted((x[idx[c]] for c in ROLE_SHARE), reverse=True)
        v = (
            [x[idx[c]] for c in STATS18]
            + [x[idx["a"]], x[idx["q"]]]
            + [ab[b] for b in order]
            + [qb[b] for b in order]
            + shares
            + [x[idx[c]] for c in MISC]
            + [x[idx[c]] for c in SCRIPTED if c != "depth_energy"]
        )
        out[i] = v
    return out


# ------------------------------------------------------------------------------------- description

FAMILIES = ("chain", "lora_sft", "abliteration", "random_dir")
INNER = ("lora_sft", "abliteration", "random_dir")
DESC_COLUMNS = (
    [f"fam_{f}" for f in FAMILIES]
    + [f"inner_{f}" for f in INNER]
    + [
        "log_lr",
        "log_steps",
        "log_rank",
        "dose_weight",
        "n_directions",
        "abl_min_weight",
        "abl_direction_index",
        "chain_length",
    ]
    + [f"touch_{t}" for t in TYPES]
    + ["layer_mean", "layer_min", "layer_max", "frac_slots_touched"]
)


def _log(v) -> float:
    return math.log(float(v)) if v is not None and float(v) > 0 else 0.0


def description_vector(edit: dict, touched_names: Sequence[str], rank: int | None) -> np.ndarray:
    """Description-only input (family, dose, role, layer, rank): from ``edit.json`` and the names of the tensors the
    edit stores; no weight statistic and no field term. Absent values are 0 (the family indicators disambiguate)."""
    fam = edit.get("family")
    p = edit.get("params") or {}
    inner_fam = p.get("inner_family") if fam == "chain" else None
    ip = (p.get("inner_params") or {}) if fam == "chain" else p
    eff = inner_fam or fam
    d = dict.fromkeys(DESC_COLUMNS, 0.0)
    if fam in FAMILIES:
        d[f"fam_{fam}"] = 1.0
    if inner_fam in INNER:
        d[f"inner_{inner_fam}"] = 1.0
    if eff == "lora_sft":
        d["log_lr"] = _log(ip.get("lr"))
        d["log_steps"] = _log(ip.get("steps"))
    elif eff == "random_dir":
        d["dose_weight"] = float(ip.get("weight") or 0.0)
        d["n_directions"] = float(ip.get("n_directions") or 0.0)
    elif eff == "abliteration":
        d["dose_weight"] = float(ip.get("max_weight") or 0.0)
        d["abl_min_weight"] = float(ip.get("min_weight") or 0.0)
        d["abl_direction_index"] = float(ip.get("direction_index") or 0.0) / N_LAYERS
    d["log_rank"] = _log(rank)
    if fam == "chain":
        d["chain_length"] = float(p.get("length") or 0.0)
    layers = []
    for n in touched_names:
        kind, layer = classify(n)
        d[f"touch_{kind}"] = 1.0
        if layer is not None:
            layers.append(layer / N_LAYERS)
    if layers:
        d["layer_mean"] = math.fsum(layers) / len(layers)
        d["layer_min"], d["layer_max"] = min(layers), max(layers)
    d["frac_slots_touched"] = len(set(touched_names)) / N_SLOTS
    v = np.array([d[c] for c in DESC_COLUMNS], dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite description feature")
    return v


def desc_zero() -> np.ndarray:
    return np.zeros(len(DESC_COLUMNS))


__all__ = [
    "BASE48",
    "COLUMNS",
    "DESC_COLUMNS",
    "FIELD_COLUMNS",
    "NAME_BLIND_COLUMNS",
    "NO_FIELD_COLUMNS",
    "SCRIPTED",
    "ZERO_CONVENTIONS",
    "aggregate",
    "classify",
    "description_vector",
    "field_terms",
    "lora_delta",
    "name_blind",
    "realise",
    "tensor_record",
    "x_zero",
]
