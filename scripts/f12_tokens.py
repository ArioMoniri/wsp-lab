#!/usr/bin/env python3
"""F12 per-tensor field tokens (K+F 39 columns + 3 probe columns) and a, q for the F12 roots (PREREG_H1_TF F12.3, F12.5,
F12.8; F5 schema pins 10 Nov 2026; Board A F12 review conditions 2, 6, 7).

For every row of ``docs/prereg/a5_manifests/f12/population.json`` (all roots: the predecessor is the base, so the frozen
G2 WikiText base field is the exact anchor; no chain_exact path is needed) the realised increment is rebuilt from the
stored zoo bytes (``bytes_via_alias`` holder for the 45 alias rows) exactly as Part A does
(``a5_power_fields.edit_increments``: D_t = fp32(bf16(W_t + Delta_t)) - fp32(W_t)), and:

* **a, q** (A5.8.3): per tensor <g_t, D_t> (fp32, ``response_field.contract`` arithmetic) and
  q_t = 1/2 tr(G_t D_t A_t D_t^T) with the base K-FAC spectral factors, via ``part_a_predictions.contract`` /
  ``contraction_row`` / ``FactorCache`` (bitwise the Part A machinery); a = fsum_t, q = fsum_t.
* **Token per supported matrix tensor** (186 slots of ``wsp.zoo.arch`` qwen3_5, layer-then-role order of
  ``response_chart.slot_layout``), present iff D_t != 0 after rounding; columns (F5 schema pins, K+F mask 0-38):

  - 0-15 vec(M_t) row-major, M_t = (Lambda_G^1/2 U_G^T B)(A U_A Lambda_A^1/2): B = U_r diag(sigma_r^1/2),
    A = diag(sigma_r^1/2) V_r^T from the full thin fp32 SVD of D_t (``torch.linalg.svd(full_matrices=False)``, CPU),
    r the recorded rank; U, Lambda the top-4 eigenpairs of the *raw* (undamped) G2 WikiText K-FAC factors (fp64,
    symmetrised, ``torch.linalg.eigh``, descending with index order on ties, A5.8.1 small-negative clamp, sign: largest
    |entry| positive with lowest index on ties). A cutoff tie (lambda_4 - lambda_5 <= 1e-9 lambda_1) masks the block
    (flag ``k_cutoff_tie``); ties inside the top four are flagged ``k_inner_tie`` and kept;
  - 16-23 log sigma_j(D_t), j = 1..8, untruncated spectrum, 0 where absent (j > min(out, in) or sigma_j = 0);
  - 24-31 1[sigma_j > 1e-3 sigma_1]; 32 sum_{j>8} sigma_j^2 / sum_j sigma_j^2; 33 layer/24; 34 full-attention flag
    (layer type of the tensor's layer); 35 <g_t, D_t> (``wikitext_nll``); 36 q_t; 37 c_t = 0 (roots);
    38 field presence (1 iff 35-37 computed and finite, else 35-38 are 0);
  - 39-41 (F12.3, drafting default): for ``mlp.down_proj`` tokens only, on the G2 probe cache (post-gating
    ``down_proj`` input on the 64 scoring-disjoint probe prompts, base model), 39 = RMS over output coordinates of the
    per-coordinate mean of D_t x, 40 = RMS over output coordinates of the per-coordinate RMS of D_t x, 41 = presence;
    pooling: mean over a prompt's tokens, then equal weight over prompts (the F8a.2(e) activation-pooling convention);
    0 elsewhere.

  Recorded rank r (F5 pins): the adapter ``r`` for lora_sft (``lora/adapter_config.json`` of the bytes holder, equal to
  ``params.rank``) and chain roots with ``inner_family`` lora_sft (``params.inner_params.rank``), 1 for abliteration,
  random_dir and chain roots of those inner families; unknown r excludes the row. r larger than min(out, in) truncates
  at min(out, in) (flag ``rank_exceeds_dims``); a truncation tie (sigma_r - sigma_{r+1} <= 1e-6 sigma_1) is flagged
  ``rank_cutoff_tie`` and kept; omitted energy sum_{j>r} sigma_j^2 / sum_j sigma_j^2 is a diagnostic, not an input.
  Eligibility (set once for every arm): a row with any supported changed tensor lacking r, a finite SVD or a K frame is
  excluded and listed; a changed tensor outside the 186 slots gets no token and is listed.

Outputs (``--out-dir`` under ``/home/ubuntu/wsp-data/runs/a5/f12_inputs``): ``tokens.safetensors`` (``X`` [rows, 186,
42] fp32 in population id order, ``presence`` [rows, 186] uint8, ``row_eligible`` [rows] uint8), ``rows.jsonl`` (one
record per row: a, q, flags, eligibility, timing), ``detail.jsonl`` (per-tensor values and token vectors; also the
resume log), ``aq.json`` (id -> a, q), ``kframes.safetensors`` + ``kframes.json`` (the K frame manifest, per-tensor
sha256), ``pricing.json``. ``--manifest`` (``docs/prereg/a5_manifests/f12/tokens_manifest.json``) records hashes only.
Pricing: after the first row the wall time is projected two ways (n_rows x first-row seconds, and first-row seconds
per supported tensor x each row's stored tensor count), and the run stops (``price_exceeds_cap``) if the larger
projection, plus setup, exceeds ``--max-wall-hours``.

    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python scripts/f12_tokens.py \
        --out-dir /home/ubuntu/wsp-data/runs/a5/f12_inputs/tokens --max-wall-hours 8 [--limit N] [--resume]

CPU only. Reads: population.json, the stored zoo deltas (``delta.safetensors`` or ``lora/``) and ``edit.json``, the
pinned base model, the G2 WikiText field, the G2 probe cache and g2_manifest.json. The intended algorithm opens no
evaluation record, TF-v1 score or outcome; that is algorithmic intent, not access telemetry (``outcomes_read_intent``).

F12-R (PREREG_H1_TF A5 note F12-R; unexposed adoption A10, A11, A19, A21), no numerical change:

* every read goes through ``f12_inputs_guard.ReadBoundary`` (resolved to the leaf; data/evals, Part A
  staging/prediction/calibration and the F12f pilot report refused before opening; only the governed manifests, zoo,
  G2 run, output directory and base-model cache allowed) and the manifest records the exact reads;
* inputs are authenticated: population.json against the governed sha256; components.jsonl against its pin and every
  zoo ``edit.json`` against the snapshot's pinned ``edit_json_index_sha256``; each row's stored bytes against its
  ``resolved_delta_hash`` before loading; probe files are bare names inside the probe cache, hash-checked;
* ``--resume`` only from an atomic ``checkpoint.json`` bound to the code, population/ids/folds, base state, G2
  manifest/field identity, probe cache and threads, and to each done row's authenticated sources and increment hash;
  done ids must be the exact prefix of the population order without duplicates, committed bytes must match their
  checkpointed prefix hashes (an uncommitted tail is truncated), lines must pass schema and finiteness checks; without
  such a checkpoint resume is refused. Completeness is exact id equality;
* the resource authority's cap (4 GB, or a dated chair ruling's cap with ``--chair-ruling``) is enforced at launch
  against the field file size (the field is loaded whole), monitored, and the peak recorded; a fresh run never
  overwrites prior outputs and the manifest is never overwritten.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import a5_power_fields as pf
from scripts import f12_inputs_guard as fg
from scripts import part_a_predictions as pap
from wsp.features import response_chart as rc
from wsp.features import response_field as rf

SCHEMA = "f12-tokens-v1"
OBJECTIVE = pap.OBJECTIVE
DATA_ROOT = pf.DATA_ROOT
ALLOWED_OUT = DATA_ROOT / "runs/a5/f12_inputs"
DEFAULT_POPULATION = ROOT / "docs/prereg/a5_manifests/f12/population.json"
DEFAULT_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/tokens_manifest.json"
G2_MANIFEST_SHA256 = pap.G2_MANIFEST_SHA256
WIKITEXT_IDENTITY_SHA256 = pap.WIKITEXT_IDENTITY_SHA256
ARCH = "qwen3_5"
N_LAYERS = 24
K = 4
N_SIGMA = 8
RANK_TOL = 1e-3  # A5.3.3 rank tolerance: presence flag sigma_j > 1e-3 sigma_1
RANK_TIE = 1e-6  # sigma_r - sigma_{r+1} <= 1e-6 sigma_1
EIG_TIE = 1e-9  # lambda_4 - lambda_5 <= 1e-9 lambda_1 (cutoff); inner ties likewise
N_COLS = 42
COLUMNS = (
    [f"K_M_{i}{j}" for i in range(K) for j in range(K)]
    + [f"log_sigma_{j}" for j in range(1, N_SIGMA + 1)]
    + [f"sigma_present_{j}" for j in range(1, N_SIGMA + 1)]
    + [
        "tail_energy_gt8",
        "layer_over_24",
        "full_attention_layer",
        "field_contraction_wikitext_nll",
        "q",
        "cross_term",
        "field_present",
        "probe_rms_of_mean_Dx",
        "probe_rms_of_rms_Dx",
        "probe_present",
    ]
)
assert len(COLUMNS) == N_COLS
PROBE_ROLE = "mlp.down_proj"
CODE_FILES = (
    "scripts/f12_tokens.py",
    "scripts/f12_inputs_guard.py",
    "scripts/f12_population.py",
    "scripts/part_a_predictions.py",
    "scripts/a5_power_fields.py",
    "src/wsp/features/response_field.py",
    "src/wsp/features/response_chart.py",
    "src/wsp/zoo/apply.py",
    "src/wsp/zoo/arch.py",
)
SCRIPT = "scripts/f12_tokens.py"
# A10 pins: the governed population.json; components.jsonl (the 1,259 zoo ids of the A5 snapshot) and the snapshot's
# content.zoo_bytes.edit_json_index_sha256 (snapshot.json sha256 1b2af25c...), which authenticates every edit.json.
GOVERNED_POPULATION_SHA256 = "07ab85d31cf8595b9b8dcd095a1f272aceb004cfc79d3296c55373db2e233da5"
COMPONENTS_PATH = ROOT / "docs/prereg/a5_manifests/components.jsonl"
COMPONENTS_SHA256 = "6b4eff8d61a4b1c4d83f0c058cb4df5c6c5398e2c6656a8b58b38f8b46251dec"
EDIT_JSON_INDEX_SHA256 = "1c4ba09da3d7821fea8aabcf3561a4cec003297ae3f6adfb621f33663e0a0945"
OUTCOMES_STATEMENT = (
    "none: stored edit bytes, edit.json, the base model, the G2 WikiText field and probe cache only; no evaluation "
    "record, TF-v1 score or outcome is opened"
)
INTENT_LABEL = (
    "algorithmic intent of this code path on authentic inputs; not access telemetry and not a session attestation "
    "(PREREG_H1_TF F12a.1(c); unexposed adoption A11). Actual reads: read_boundary; resume reads: resume"
)
CHECKPOINT = "checkpoint.json"
CHECKPOINT_SCHEMA = "f12-tokens-checkpoint-v1"
ROW_KEYS = (
    "id",
    "family",
    "source",
    "bytes_holder",
    "rank",
    "rank_source",
    "n_edited_tensors",
    "n_tokens",
    "unsupported_changed_tensors",
    "eligibility",
    "a",
    "a_fp64",
    "q",
    "q_partial",
    "field_status",
    "increment_sha256",
    "token_flags",
    "omitted_energy_max",
    "omitted_energy_median",
    "numerical_rank_max",
    "n_probe_tokens",
    "seconds",
)
DETAIL_KEYS = frozenset(("id", "tokens", "per_tensor", "field_per_tensor"))
OUTPUT_NAMES = (
    "tokens.safetensors",
    "rows.jsonl",
    "detail.jsonl",
    CHECKPOINT,
    "aq.json",
    "kframes.json",
    "kframes.safetensors",
    "pricing.json",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


# ------------------------------------------------------------------------------------------- slots


def layer_types_from_coverage(identity, objective=OBJECTIVE, n_layers=N_LAYERS):
    """Layer types from the frozen field's ``module_roles`` (``full_attention:...`` / ``linear_attention:...``)."""
    types = {}
    for module, role in identity["module_roles"].items():
        kind = role.split(":", 1)[0]
        if kind not in ("full_attention", "linear_attention"):
            continue
        layer = int(module.split(".")[2])
        if types.setdefault(layer, kind) != kind:
            raise ValueError(f"layer {layer} has two layer types")
    if sorted(types) != list(range(n_layers)):
        raise ValueError("field module roles do not cover every layer")
    return [types[i] for i in range(n_layers)]


def matrix_slots(layer_types, architecture=ARCH):
    """[(slot index, layer, role, tensor name)] of the registered matrix slots in A5.8.3 order."""
    out = []
    for layer, role, is_matrix in rc.slot_layout(architecture, layer_types):
        if is_matrix:
            out.append((len(out), layer, role, f"model.layers.{layer}.{role}.weight"))
    return out


# ------------------------------------------------------------------------------------------ K frames


def k_frame(factor, k=K, neg_tol=rf.NEGATIVE_TOLERANCE, tie=EIG_TIE):
    """Top-k eigenframe of a raw K-FAC factor (F5 pins "Eigen conventions").

    fp64, symmetrised 1/2(F + F^T), undamped, ``torch.linalg.eigh`` on CPU; eigenvalues in [-neg_tol lambda_max, 0)
    clamped to 0, anything below raises; descending order with index order on exact ties; each vector's largest-|.|
    entry positive (lowest index on ties); Lambda^1/2 of 0 is 0. Returns {U [n, k] fp64, sqrt_lambda [k] fp64,
    lambda [k+1 or fewer], flags}."""
    f = factor.detach().cpu().double()
    if f.ndim != 2 or f.shape[0] != f.shape[1] or not torch.isfinite(f).all():
        raise ValueError("invalid finite square factor")
    f = (f + f.T) * 0.5
    values, vectors = torch.linalg.eigh(f)
    lmax, lmin = values.max().item(), values.min().item()
    if lmin < 0 and (lmax <= 0 or lmin < -neg_tol * lmax):
        raise ValueError(
            f"materially negative factor eigenvalue {lmin:.3e} (lambda_max {lmax:.3e})"
        )
    n_clamped = int((values < 0).sum())
    values = values.clamp_min(0)
    order = torch.argsort(-values, stable=True)
    values, vectors = values[order], vectors[:, order]
    kk = min(k, values.numel())
    U = vectors[:, :kk].clone()
    idx = U.abs().argmax(dim=0)  # first maximal entry: lowest index on ties
    signs = torch.sign(U[idx, torch.arange(kk)])
    U = U * torch.where(signs == 0, 1.0, signs).to(U.dtype)
    lam = values[: kk + 1]
    l1 = values[0].item() if values.numel() else 0.0
    flags = []
    if values.numel() > kk and (values[kk - 1] - values[kk]).item() <= tie * l1:
        flags.append("k_cutoff_tie")
    if any((values[j] - values[j + 1]).item() <= tie * l1 for j in range(kk - 1)):
        flags.append("k_inner_tie")
    return {
        "U": U,
        "sqrt_lambda": values[:kk].sqrt(),
        "lambda": lam,
        "flags": flags,
        "n_clamped": n_clamped,
        "dimension": int(f.shape[0]),
    }


def tensor_frames(raw_pair, shape):
    """(G frame, A frame) of one weight tensor from its module's raw factors; A sliced to the input width."""
    out_dim, in_dim = shape
    A = raw_pair["A"]
    if A.shape[0] != in_dim:
        if raw_pair.get("bias") is None:
            raise ValueError(f"A factor {tuple(A.shape)} does not match input width {in_dim}")
        A = A[:in_dim, :in_dim]  # bias-augmented A: the weight block
    G = raw_pair["G"]
    if G.shape[0] != out_dim:
        raise ValueError(f"G factor {tuple(G.shape)} does not match output width {out_dim}")
    return k_frame(G), k_frame(A)


# ------------------------------------------------------------------------------------------- tokens


def recorded_rank(edit: dict, adapter_config: dict | None = None):
    """(r, source) per the F5 pins, or (None, reason). ``adapter_config`` is the holder's ``lora/adapter_config.json``."""
    fam = edit.get("family")
    params = edit.get("params") or {}
    if fam == "lora_sft":
        pr = params.get("rank")
        if adapter_config is not None:
            if adapter_config.get("rank_pattern"):
                return None, "unknown_rank:adapter_rank_pattern"
            ar = adapter_config.get("r")
            if pr is not None and ar != pr:
                return None, f"unknown_rank:adapter_r_{ar}_differs_from_params_rank_{pr}"
            return int(ar), "adapter_config.r"
        return (
            (int(pr), "params.rank") if pr is not None else (None, "unknown_rank:lora_without_rank")
        )
    if fam in ("abliteration", "random_dir"):
        return 1, f"pinned:{fam}"
    if fam == "chain":
        inner = params.get("inner_family")
        if inner == "lora_sft":
            pr = (params.get("inner_params") or {}).get("rank")
            return (
                (int(pr), "params.inner_params.rank")
                if pr is not None
                else (None, "unknown_rank:chain_lora")
            )
        if inner in ("abliteration", "random_dir"):
            return 1, f"pinned:chain:{inner}"
        return None, f"unknown_rank:chain_inner_{inner}"
    return None, f"unknown_rank:family_{fam}"


def spectrum_columns(S):
    """Columns 16-32 from the descending singular values S (fp32 tensor)."""
    out = np.zeros(17, dtype=np.float64)
    s = S.double()
    s1 = s[0].item() if s.numel() else 0.0
    for j in range(min(N_SIGMA, s.numel())):
        if s[j] > 0:
            out[j] = math.log(s[j].item())
            out[N_SIGMA + j] = float(s[j].item() > RANK_TOL * s1)
    total = (s**2).sum().item()
    out[16] = (s[N_SIGMA:] ** 2).sum().item() / total if total > 0 else 0.0
    return out


def balanced_factors(U, S, Vh, r):
    """B = U_r diag(sigma_r^1/2), A = diag(sigma_r^1/2) V_r^T (rank-r truncation BA), with diagnostics."""
    n = S.numel()
    r_eff = min(int(r), n)
    root = S[:r_eff].clamp_min(0).sqrt()
    B = U[:, :r_eff] * root
    A = root[:, None] * Vh[:r_eff]
    s = S.double()
    total = (s**2).sum().item()
    flags = []
    if r > n:
        flags.append("rank_exceeds_dims")
    if r_eff < n and (s[r_eff - 1] - s[r_eff]).item() <= RANK_TIE * s[0].item():
        flags.append("rank_cutoff_tie")
    omitted = (s[r_eff:] ** 2).sum().item() / total if total > 0 else 0.0
    return B, A, {"r_eff": r_eff, "omitted_energy": omitted, "flags": flags}


def k_block(B, A, frame_G, frame_A):
    """M = (Lambda_G^1/2 U_G^T B)(A U_A Lambda_A^1/2) in fp64 (k x k)."""
    left = frame_G["sqrt_lambda"][:, None] * (frame_G["U"].T @ B.double())
    right = (A.double() @ frame_A["U"]) * frame_A["sqrt_lambda"][None, :]
    return left @ right


def probe_panel(prompts):
    """Pooled probe statistics of one layer from per-prompt inputs [n_p, in] (equal prompt weight).

    Returns (xbar fp64 [in], Xw fp32 [N, in]) with mean_p mean_t x = xbar and sum over rows of (Xw v)^2 =
    mean_p mean_t (x.v)^2 for any v."""
    P = len(prompts)
    xbar = sum(x.double().mean(0) for x in prompts) / P
    Xw = torch.cat([x.float() * math.sqrt(1.0 / (P * x.shape[0])) for x in prompts], dim=0)
    return xbar, Xw


def probe_columns(D, xbar, Xw):
    """(RMS over output coords of the per-coordinate mean of D x, RMS over output coords of the per-coordinate RMS)."""
    m = D.double() @ xbar
    rms_mean = math.sqrt((m**2).mean().item())
    Y = (D.float() @ Xw.T).double()
    meansq = (Y**2).sum(1)
    rms_rms = math.sqrt(meansq.mean().item())
    return rms_mean, rms_rms


def tensor_token(D, *, layer, full_attention, r, frames, field, probe=None):
    """Token vector [42] (fp64 values) and per-tensor info for one nonzero supported matrix increment.

    ``frames``: (G frame, A frame) or None (no K frame: the row is ineligible, raised as ``NoKFrame``); ``field``:
    (signed, q) or None; ``probe``: (xbar, Xw) for a down_proj tensor or None. Raises ``NonFiniteSVD``."""
    tok = np.zeros(N_COLS, dtype=np.float64)
    info = {"flags": []}
    U, S, Vh = torch.linalg.svd(D.float(), full_matrices=False)
    if not (torch.isfinite(U).all() and torch.isfinite(S).all() and torch.isfinite(Vh).all()):
        raise NonFiniteSVD("nonfinite SVD")
    tok[16:33] = spectrum_columns(S)
    tok[33] = layer / N_LAYERS
    tok[34] = float(full_attention)
    info["sigma1"] = S[0].item()
    info["numerical_rank"] = int((S > RANK_TOL * S[0]).sum()) if S[0] > 0 else 0
    if frames is None:
        raise NoKFrame("no K frame")
    B, A, binfo = balanced_factors(U, S, Vh, r)
    info |= {"r_eff": binfo["r_eff"], "omitted_energy": binfo["omitted_energy"]}
    info["flags"] += binfo["flags"]
    fG, fA = frames
    if "k_cutoff_tie" in fG["flags"] or "k_cutoff_tie" in fA["flags"]:
        info["flags"].append("k_cutoff_tie")
    else:
        tok[0:16] = k_block(B, A, fG, fA).reshape(-1).numpy()
    if "k_inner_tie" in fG["flags"] or "k_inner_tie" in fA["flags"]:
        info["flags"].append("k_inner_tie")
    if field is not None:
        signed, q = field
        if signed is not None and q is not None and math.isfinite(signed) and math.isfinite(q):
            tok[35], tok[36], tok[37], tok[38] = signed, q, 0.0, 1.0
    if probe is not None:
        rm, rr = probe_columns(D, *probe)
        if math.isfinite(rm) and math.isfinite(rr):
            tok[39], tok[40], tok[41] = rm, rr, 1.0
    if not np.isfinite(tok).all():
        raise NonFiniteSVD("nonfinite token")
    return tok, info


class NonFiniteSVD(ValueError):
    pass


class NoKFrame(ValueError):
    pass


def row_tokens(D, edited, slots_by_name, layer_types, r, frames, per, probe):
    """Tokens of one row: {slot index: token}, per-tensor info, unsupported changed tensors, exclusion reason or None.

    ``per`` is ``part_a_predictions.contract``'s per-tensor dict (signed, q, nonzero); ``frames`` maps tensor name to
    (G frame, A frame); ``probe`` maps layer to (xbar, Xw)."""
    tokens, info, unsupported, reasons = {}, {}, [], []
    for name in edited:
        d = D[name]
        if not torch.count_nonzero(d):
            continue
        slot = slots_by_name.get(name)
        if slot is None:
            unsupported.append(name)
            continue
        idx, layer, role = slot
        if r is None:
            reasons.append("unknown_rank")
            continue
        row = per.get(name)
        field = (row["signed"], row["q"]) if row is not None else None
        pr = probe.get(layer) if (role == PROBE_ROLE and probe is not None) else None
        try:
            tok, ti = tensor_token(
                d,
                layer=layer,
                full_attention=layer_types[layer] == "full_attention",
                r=r,
                frames=frames.get(name),
                field=field,
                probe=pr,
            )
        except NonFiniteSVD:
            reasons.append(f"nonfinite_svd:{name}")
            continue
        except NoKFrame:
            reasons.append(f"no_k_frame:{name}")
            continue
        tokens[idx] = tok
        info[name] = ti | {
            "slot": idx,
            "a": row["signed"] if row else None,
            "q": row["q"] if row else None,
        }
    reason = None
    if reasons:
        reason = "excluded:" + ";".join(sorted(set(reasons)))
    return tokens, info, sorted(unsupported), reason


def field_contraction(grads, coverage, factors, increments):
    """Part A's contraction once: (field row, per-tensor {a, q}, per-tensor contract dict); nonfinite abstains."""
    try:
        per, uncovered = pap.contract(grads, coverage, factors, increments)
    except ValueError as e:
        if "nonfinite" not in str(e):
            raise
        return (
            {"field_status": pap.ABSTAIN_NONFINITE, "a": None, "q": None, "detail": str(e)},
            {},
            {},
        )
    row = pap.contraction_row(per, uncovered)
    return row, {k: {"a": v["signed"], "q": v["q"]} for k, v in per.items()}, per


def assemble(row_ids, row_tokens_by_id, n_slots, eligible_by_id):
    """X [rows, slots, 42] fp32 (absent tokens zero), presence [rows, slots] uint8, row_eligible [rows] uint8."""
    X = np.zeros((len(row_ids), n_slots, N_COLS), dtype=np.float32)
    P = np.zeros((len(row_ids), n_slots), dtype=np.uint8)
    E = np.zeros(len(row_ids), dtype=np.uint8)
    for i, rid in enumerate(row_ids):
        for idx, tok in row_tokens_by_id.get(rid, {}).items():
            X[i, int(idx)] = np.asarray(tok, dtype=np.float32)
            P[i, int(idx)] = 1
        E[i] = 1 if eligible_by_id.get(rid) else 0
    return X, P, E


def projection(setup_seconds, first_seconds, first_tensors, tensor_counts):
    """(naive, tensor-weighted) projected wall seconds from the first row."""
    naive = setup_seconds + first_seconds * len(tensor_counts)
    per_tensor = first_seconds / max(1, first_tensors)
    weighted = setup_seconds + per_tensor * sum(tensor_counts)
    return naive, weighted


# --------------------------------------------------------------------------------------------- IO


def stored_tensor_count(entry_dir: Path, boundary=None) -> int:
    from safetensors import safe_open

    if boundary is not None:
        for p in fg.entry_files(boundary, entry_dir):
            boundary.resolve(p)
    if (entry_dir / "delta.safetensors").exists():
        with safe_open(str(entry_dir / "delta.safetensors"), "pt") as f:
            return len(list(f.keys()))
    with safe_open(str(entry_dir / "lora/adapter_model.safetensors"), "pt") as f:
        return sum("lora_A" in k for k in f.keys())  # noqa: SIM118 (safe_open is not iterable)


def load_probe(g2_manifest, probe_dir: Path, layers, boundary=None):
    """Hash-checked G2 probe cache -> {layer: (xbar, Xw)} and its identity record (A10: every file a bare name inside
    the cache, read through the boundary and checked against the G2 manifest before loading)."""
    from safetensors.torch import load_file

    boundary = boundary if boundary is not None else fg.ReadBoundary(allowed=[probe_dir])
    rec = g2_manifest["probe_cache"]
    if rec.get("status") != "complete":
        raise ValueError("probe cache is not complete")
    files = rec["files"]
    meta = json.loads(
        boundary.read_bytes(probe_dir / "probe_cache.json", sha256=files["probe_cache.json"])
    )
    per_layer = {layer: [] for layer in layers}
    for row in sorted(meta, key=lambda r: r["index"]):
        name = fg.safe_name(row["file"])
        p = probe_dir / name
        if name not in files:
            raise ValueError(f"{p}: not in the G2 manifest")
        boundary.sha256_file(p, sha256=files[name])
        t = load_file(str(boundary.resolve(p)))
        for layer in layers:
            per_layer[layer].append(t[f"layer.{layer}"])
    if (
        rf._hash([r["content_hashes"] for r in sorted(meta, key=lambda r: r["index"])])
        != rec["content_hash"]
    ):
        raise ValueError("probe cache content hash differs from the G2 manifest")
    probe = {layer: probe_panel(xs) for layer, xs in per_layer.items()}
    ident = {
        "dir": str(probe_dir),
        "content_hash": rec["content_hash"],
        "files_sha256": sha256_bytes(canonical_json(files)),
        "n_prompts": len(meta),
        "n_tokens": sum(r["n_tokens"] for r in meta),
        "hooks": rec["hooks"]["role"],
        "predecessor": rec["predecessor"],
        "pooling": "mean over a prompt's tokens, then equal weight over prompts",
    }
    return probe, ident


def code_files_identity():
    return {f: pf.sha256_file(ROOT / f) for f in CODE_FILES}


def code_identity():
    out = code_files_identity()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        out["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT, env=env
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *CODE_FILES], text=True, cwd=ROOT, env=env
        )
        out["code_files_dirty"] = bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        out["git_sha"] = None
    return out


def check_out_dir(path: Path):
    p = Path(path).resolve()
    if p != ALLOWED_OUT and ALLOWED_OUT not in p.parents:
        raise ValueError(f"{path}: outputs must stay under {ALLOWED_OUT}")


def check_fresh_out(out: Path):
    """A fresh (non-resume) run never overwrites a previous run's outputs (e.g. the frozen token run's)."""
    found = [n for n in OUTPUT_NAMES if (Path(out) / n).exists()]
    if found:
        raise FileExistsError(
            f"{out} holds outputs {found}; use a new --out-dir (or --resume from a checkpoint)"
        )


# ---------------------------------------------------------------------------- F12-R: inputs and resume


def read_boundary() -> fg.ReadBoundary:
    """A10: the governed read boundary of the token build (prohibited paths refused wherever a path resolves)."""
    return fg.ReadBoundary(
        allowed=[
            ROOT / "docs/prereg/a5_manifests/f12",
            COMPONENTS_PATH,
            DATA_ROOT / "zoo/qwen3.5-0.8b",
            DATA_ROOT / "runs/g2",
            ALLOWED_OUT,
            hf_cache_dir(),
        ]
    )


def hf_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def resource_guard(build, threads, cap_bytes, ruling) -> fg.ResourceGuard:
    return fg.ResourceGuard(build=build, threads=threads, cap_bytes=cap_bytes, ruling=ruling)


def launch_estimate(boundary, g2: Path) -> int:
    """A21 launch projection: a lower bound, the G2 field file, which ``FieldArtifact.load`` holds whole."""
    return boundary.resolve(Path(g2) / "fields" / OBJECTIVE / "tensors.pt").stat().st_size


def authenticate_edit_index(boundary, zoo: Path, components_path=COMPONENTS_PATH) -> dict:
    """{id: sha256(edit.json)} for every zoo id of components.jsonl, authenticated as a whole against the A5
    snapshot's pinned ``edit_json_index_sha256`` (``a5_snapshot``: sha256 of json.dumps(sorted([id, sha])))."""
    data = boundary.read_bytes(components_path, sha256=COMPONENTS_SHA256)
    ids = [json.loads(x)["id"] for x in data.decode().splitlines() if x.strip()]
    sha = {}
    for i in ids:
        p = Path(zoo) / i / "edit.json"
        sha[i] = boundary.sha256_file(p, record=False) if p.exists() else None
    index = sha256_bytes(json.dumps(sorted([i, h] for i, h in sha.items())).encode())
    if index != EDIT_JSON_INDEX_SHA256:
        raise fg.AuthenticationError(
            f"edit.json index {index} differs from the snapshot's pinned {EDIT_JSON_INDEX_SHA256}"
        )
    note = (
        f"{len(ids)} edit.json files of components.jsonl, hashed as the snapshot's edit_json_index"
    )
    boundary.reads.append(
        {
            "path": str(Path(zoo) / "*" / "edit.json"),
            "resolved": None,
            "sha256": index,
            "authenticated": True,
            "note": note,
        }
    )
    return sha


def read_edit(boundary, zoo: Path, row_id: str, edit_sha: dict) -> dict:
    return json.loads(
        boundary.read_bytes(Path(zoo) / row_id / "edit.json", sha256=edit_sha[row_id])
    )


def row_source(boundary, zoo: Path, r: dict, edit_sha: dict) -> dict:
    """A row's authenticated sources: its holder's stored bytes (entry_delta_hash = the population's
    resolved_delta_hash) and its own edit.json sha256 (from the authenticated index)."""
    h = fg.authenticate_entry(boundary, Path(zoo) / r["bytes_holder"], r["resolved_delta_hash"])
    return {"entry_delta_hash": h, "edit_json_sha256": edit_sha[r["id"]]}


def resume_binding(code, frozen, base_hash, records, probe_ident, threads) -> dict:
    """A19: everything a resumed row depends on besides its own sources."""
    return {
        "code": dict(code),
        "population_json_sha256": frozen["population_json_sha256"],
        "population_ids_sha256": frozen["population_ids_sha256"],
        "fold_manifest_sha256": frozen["fold_manifest_sha256"],
        "base_state_hash": base_hash,
        "g2_manifest_sha256": records["g2_manifest"]["sha256"],
        "wikitext_field_files": records["wikitext_field"]["files"],
        "wikitext_identity_sha256": records["wikitext_field"]["identity_sha256"],
        "probe_content_hash": probe_ident["content_hash"],
        "probe_files_sha256": probe_ident["files_sha256"],
        "threads": int(threads),
    }


def append_row(rows_path: Path, detail_path: Path, rec: dict, detail: dict, strict=True):
    """Append one row exactly as the build always has (sorted keys, no NaN); (rows line sha256, detail line sha256)."""
    rl = json.dumps(rec, sort_keys=True, allow_nan=not strict) + "\n"
    dl = json.dumps(detail, sort_keys=True, allow_nan=not strict) + "\n"
    with rows_path.open("a") as f:
        f.write(rl)
    with detail_path.open("a") as f:
        f.write(dl)
    return sha256_bytes(rl.encode()), sha256_bytes(dl.encode())


def checkpoint_entry(
    row_id, source, increment_sha256, rows_line_sha256, detail_line_sha256
) -> dict:
    return {
        "id": row_id,
        "source": source,
        "increment_sha256": increment_sha256,
        "rows_line_sha256": rows_line_sha256,
        "detail_line_sha256": detail_line_sha256,
    }


def write_checkpoint(out: Path, binding: dict, entries: list) -> str:
    """Atomic commit point after each row: the binding, the done rows and the committed byte prefixes."""
    doc = {"schema": CHECKPOINT_SCHEMA, "binding": binding, "rows": entries}
    for name in ("rows.jsonl", "detail.jsonl"):
        data = (Path(out) / name).read_bytes()
        doc[name] = {"bytes": len(data), "sha256": sha256_bytes(data)}
    return fg.atomic_write_bytes(
        Path(out) / CHECKPOINT, (json.dumps(doc, sort_keys=True) + "\n").encode()
    )


def _check_row(e, rl, dl):
    if (
        sha256_bytes(rl.encode()) != e["rows_line_sha256"]
        or sha256_bytes(dl.encode()) != e["detail_line_sha256"]
    ):
        raise ValueError(f"{e['id']}: committed line differs from its checkpoint hash")
    rec, det = fg.strict_json_loads(rl), fg.strict_json_loads(dl)
    if rec.get("id") != e["id"] or det.get("id") != e["id"]:
        raise ValueError(f"{e['id']}: line id differs from the checkpoint")
    if set(rec) != set(ROW_KEYS):
        raise ValueError(f"{e['id']}: row schema differs")
    if set(det) != DETAIL_KEYS:
        raise ValueError(f"{e['id']}: detail schema differs")
    for k, v in det["tokens"].items():
        if not isinstance(v, list) or len(v) != N_COLS:
            raise ValueError(f"{e['id']}: token {k} has {len(v)} columns, not {N_COLS}")
        if not all(fg.finite(x) for x in v):
            raise ValueError(f"{e['id']}: nonfinite token {k}")
    if rec["increment_sha256"] != e["increment_sha256"]:
        raise ValueError(f"{e['id']}: row increment differs from the checkpoint")
    return rec, det


def load_checkpoint(out: Path, binding: dict, todo_ids: list, source_fn, boundary=None):
    """A19: (done {id: detail}, rows) from a bound checkpoint, or ValueError. No checkpoint, no resume."""
    out = Path(out)
    boundary = boundary if boundary is not None else fg.ReadBoundary(allowed=[out])
    cp_path = out / CHECKPOINT
    if not cp_path.exists():
        raise ValueError(
            f"{out}: no bound checkpoint; resume without one is disabled for governed builds"
        )
    for name in (CHECKPOINT, "rows.jsonl", "detail.jsonl"):
        if (out / name).is_symlink():
            raise fg.BoundaryError(f"{out / name}: a resume leaf may not be a symlink")
        boundary.resolve(out / name)
    cp = fg.strict_json_loads(boundary.read_bytes(cp_path).decode())
    if cp.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema differs")
    if cp["binding"] != binding:
        keys = sorted(
            k for k in set(binding) | set(cp["binding"]) if binding.get(k) != cp["binding"].get(k)
        )
        raise ValueError(f"resume binding differs: {keys}")
    entries = cp["rows"]
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("checkpoint has duplicate ids")
    if ids != list(todo_ids[: len(ids)]):
        raise ValueError("checkpoint ids are not the exact prefix of the population order")
    lines = {}
    for name in ("rows.jsonl", "detail.jsonl"):
        p, n = out / name, cp[name]["bytes"]
        if fg.prefix_sha256(p, n) != cp[name]["sha256"]:
            raise ValueError(f"{p}: committed prefix differs from the checkpoint")
        os.truncate(p, n)  # drop an uncommitted tail
        lines[name] = p.read_text().splitlines(keepends=True)
        if len(lines[name]) != len(ids):
            raise ValueError(f"{p}: {len(lines[name])} committed lines, checkpoint has {len(ids)}")
    done, recs = {}, []
    for e, rl, dl in zip(entries, lines["rows.jsonl"], lines["detail.jsonl"]):
        rec, det = _check_row(e, rl, dl)
        if source_fn(e["id"]) != e["source"]:
            raise ValueError(f"{e['id']}: row source changed since the checkpoint")
        done[e["id"]] = det
        recs.append(rec)
    return done, recs


def checkpoint_entries(out: Path) -> list:
    return fg.strict_json_loads((Path(out) / CHECKPOINT).read_text())["rows"]


def is_complete(done: dict, rows: list, limit) -> bool:
    """Exact id equality (not a length comparison)."""
    ids = [r["id"] for r in rows]
    return limit is None and len(done) == len(ids) and sorted(done) == sorted(ids)


def load_base(boundary, state_hash):
    cache = hf_cache_dir()
    boundary.resolve(cache)
    note = "base model cache, read by transformers; authenticated by the state hash"
    boundary.reads.append(
        {"path": str(cache), "resolved": None, "sha256": None, "authenticated": False, "note": note}
    )
    return pf.load_base_state(state_hash)


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def load_field_setup(g2: Path, threads: int, boundary=None):
    """Hash-checked G2 WikiText field: (gradients, coverage, FactorCache, K frames by tensor, identity records)."""
    boundary = boundary if boundary is not None else read_boundary()
    g2_bytes = boundary.read_bytes(g2 / "g2_manifest.json", sha256=G2_MANIFEST_SHA256)
    g2_manifest = json.loads(g2_bytes)
    for f in ("manifest.json", "tensors.pt"):
        boundary.check(
            g2 / "fields" / OBJECTIVE / f,
            note="loaded by FieldArtifact.load after its sha256 check",
        )
    frec = g2_manifest["fields"][OBJECTIVE]
    if frec.get("status") != "complete" or frec["identity_sha256"] != WIKITEXT_IDENTITY_SHA256:
        raise ValueError("base WikiText field is not the frozen complete G2 field")
    t0 = time.time()
    field, files = pf.load_field(g2 / "fields" / OBJECTIVE, frec["files"])
    if rf._hash(field.identity) != WIKITEXT_IDENTITY_SHA256:
        raise ValueError("base field identity sha256 differs from the frozen identity")
    coverage = field.identity["coverage"][OBJECTIVE]
    layer_types = layer_types_from_coverage(field.identity)
    grads = field.gradients[OBJECTIVE]
    raw = field.factors[OBJECTIVE]
    identity_state = field.identity["state_hash"]
    del field
    gc.collect()
    load_seconds = time.time() - t0
    log(f"field loaded in {load_seconds:.0f}s")
    slots = matrix_slots(layer_types)
    factors = pap.FactorCache(coverage, raw, drop_raw=True)
    frames, frame_seconds = {}, 0.0
    t1 = time.time()
    for _, _, _, name in slots:
        row = coverage.get(name)
        if row is None or not row["factors"]:
            continue
        module = row["module"]
        if module in factors.kept:
            raise ValueError(f"{module}: raw factors already released before its K frame")
        t = time.time()
        frames[name] = tensor_frames(raw[module], row["shape"])
        frame_seconds += time.time() - t
        factors.module(module)  # spectral factor (Part A path), then the raw pair is released
    spectral_seconds = time.time() - t1 - frame_seconds
    gc.collect()
    log(
        f"K frames {frame_seconds:.0f}s, spectral factors {spectral_seconds:.0f}s for {len(frames)} tensors"
    )
    records = {
        "g2_manifest": {"path": str(g2 / "g2_manifest.json"), "sha256": sha256_bytes(g2_bytes)},
        "wikitext_field": {
            "dir": str(g2 / "fields" / OBJECTIVE),
            "files": files,
            "identity_sha256": frec["identity_sha256"],
            "content_hashes_sha256": frec["content_hashes_sha256"],
            "state_hash": identity_state,
        },
        "timing": {
            "field_load_seconds": load_seconds,
            "k_frame_seconds": frame_seconds,
            "spectral_seconds": spectral_seconds,
        },
    }
    return (
        g2_manifest,
        grads,
        coverage,
        factors,
        frames,
        slots,
        layer_types,
        identity_state,
        records,
    )


def save_frames(frames, out_dir: Path):
    from safetensors.torch import save_file

    tensors, rows = {}, {}
    for name in sorted(frames):
        fG, fA = frames[name]
        for side, f in (("G", fG), ("A", fA)):
            tensors[f"{name}.{side}.U"] = f["U"].contiguous()
            tensors[f"{name}.{side}.sqrt_lambda"] = f["sqrt_lambda"].contiguous()
        rows[name] = {
            side: {
                "U_sha256": rf._tensor_hash(f["U"]),
                "sqrt_lambda_sha256": rf._tensor_hash(f["sqrt_lambda"]),
                "lambda": [float(x) for x in f["lambda"]],
                "flags": f["flags"],
                "n_clamped": f["n_clamped"],
                "dimension": f["dimension"],
            }
            for side, f in (("G", fG), ("A", fA))
        }
    save_file(tensors, str(out_dir / "kframes.safetensors"))
    doc = {
        "schema": "f12-kframes-v1",
        "rule": "F5 schema pins 'Eigen conventions': top-4 eigenpairs of the raw G2 WikiText K-FAC factors, fp64, "
        "symmetrised, undamped, eigh, descending with index order on ties, small-negative clamp, largest-|entry| "
        "positive",
        "tensors": rows,
        "kframes_safetensors_sha256": pf.sha256_file(out_dir / "kframes.safetensors"),
        "counts": dict(Counter(f for r in rows.values() for s in r.values() for f in s["flags"])),
    }
    sha = write_json(out_dir / "kframes.json", doc)
    return doc, sha


# --------------------------------------------------------------------------------------------- run


def run(args):
    t_start = time.time()
    torch.set_num_threads(args.threads)
    check_out_dir(args.out_dir)
    out = Path(args.out_dir)
    if Path(args.manifest).exists():
        raise FileExistsError(f"{args.manifest} exists; a governed manifest is never overwritten")
    if not args.resume:
        check_fresh_out(out)
    out.mkdir(parents=True, exist_ok=True)
    boundary = read_boundary()
    guard = resource_guard(SCRIPT, args.threads, args.rss_cap_bytes, args.chair_ruling)
    guard.launch(launch_estimate(boundary, Path(args.g2)), breach_path=out / "breach.json")
    pop_bytes = boundary.read_bytes(args.population, sha256=GOVERNED_POPULATION_SHA256)
    pop = json.loads(pop_bytes)
    if pop["reproduction"]["status"] != "reproduced":
        raise ValueError("population.json does not reproduce the registered population")
    from scripts import f12_population as fpop

    if (
        fpop.fold_manifest_sha256(pop["folds"]["component_fold"])
        != pop["folds"]["fold_manifest_sha256"]
    ):
        raise ValueError("fold manifest hash differs from population.json")
    rows = sorted(pop["rows"], key=lambda r: r["id"])
    if fpop.ids_sha256([r["id"] for r in rows]) != pop["population_ids_sha256"]:
        raise ValueError("population ids differ from population.json's hash")
    frozen = {
        "population_json_sha256": sha256_bytes(pop_bytes),
        "population_ids_sha256": pop["population_ids_sha256"],
        "fold_manifest_sha256": pop["folds"]["fold_manifest_sha256"],
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"population frozen: {json.dumps(frozen)}")
    zoo = Path(args.zoo)
    edit_sha = authenticate_edit_index(boundary, zoo)
    todo = rows if args.limit is None else rows[: args.limit]
    counts = [stored_tensor_count(zoo / r["bytes_holder"], boundary) for r in todo]

    g2_manifest, grads, coverage, factors, frames, slots, layer_types, state_hash, records = (
        load_field_setup(Path(args.g2), args.threads, boundary)
    )
    guard.check()
    kdoc, k_sha = save_frames(frames, out)
    slots_by_name = {name: (idx, layer, role) for idx, layer, role, name in slots}
    probe, probe_ident = load_probe(
        g2_manifest, Path(args.g2) / "probe_cache", range(N_LAYERS), boundary
    )
    base, base_hash = load_base(boundary, state_hash)
    guard.check()
    from wsp.zoo.apply import load_entry_delta

    setup_seconds = time.time() - t_start
    log(f"setup {setup_seconds:.0f}s; {len(todo)} rows")
    rows_path, detail_path = out / "rows.jsonl", out / "detail.jsonl"
    binding = resume_binding(
        code_files_identity(), frozen, base_hash, records, probe_ident, args.threads
    )
    by_id = {r["id"]: r for r in rows}
    resume = None
    if args.resume:
        done, recs = load_checkpoint(
            out,
            binding,
            [r["id"] for r in todo],
            lambda i: row_source(boundary, zoo, by_id[i], edit_sha),
            boundary,
        )
        entries = checkpoint_entries(out)
        resume = {
            "rows_resumed": len(done),
            "binding_sha256": sha256_bytes(canonical_json(binding)),
        }
        log(f"resume: {len(done)} rows from the bound checkpoint")
    else:
        done, recs, entries = {}, [], []
        rows_path.write_text("")
        detail_path.write_text("")
        write_checkpoint(out, binding, entries)
    pricing = None
    n_new = 0
    for n, (r, count) in enumerate(zip(todo, counts), 1):
        if r["id"] in done:
            continue
        t_row = time.time()
        holder = zoo / r["bytes_holder"]
        src = row_source(
            boundary, zoo, r, edit_sha
        )  # stored bytes and edit.json authenticated first
        edit = read_edit(boundary, zoo, r["id"], edit_sha)
        cfg_path = holder / "lora/adapter_config.json"
        adapter = json.loads(boundary.read_bytes(cfg_path)) if cfg_path.exists() else None
        rank, rank_source = recorded_rank(edit, adapter)
        stored = load_entry_delta(holder, base)
        W, D, raw, edited = pf.edit_increments(base, stored, None)
        del stored, W, raw
        d_edit = {k: D[k] for k in edited}
        frow, fdetail, per = field_contraction(grads, coverage, factors, d_edit)
        toks, info, unsupported, reason = row_tokens(
            D, edited, slots_by_name, layer_types, rank, frames, per, probe
        )
        if rank is None:
            reason = f"excluded:{rank_source}"
        flags = Counter(f for v in info.values() for f in v["flags"])
        om = [v["omitted_energy"] for v in info.values() if "omitted_energy" in v]
        rec = {
            "id": r["id"],
            "family": r["family"],
            "source": r["source"],
            "bytes_holder": r["bytes_holder"],
            "rank": rank,
            "rank_source": rank_source,
            "n_edited_tensors": len(edited),
            "n_tokens": len(toks),
            "unsupported_changed_tensors": unsupported,
            "eligibility": reason or "eligible",
            "a": frow.get("a"),
            "a_fp64": frow.get("a_fp64"),
            "q": frow.get("q"),
            "q_partial": frow.get("q_partial"),
            "field_status": frow.get("field_status"),
            "increment_sha256": pf.tensors_hash(d_edit),
            "token_flags": dict(sorted(flags.items())),
            "omitted_energy_max": max(om) if om else None,
            "omitted_energy_median": float(np.median(om)) if om else None,
            "numerical_rank_max": max((v["numerical_rank"] for v in info.values()), default=None),
            "n_probe_tokens": sum(1 for t in toks.values() if t[41] == 1.0),
        }
        seconds = time.time() - t_row
        rec["seconds"] = seconds
        detail = {
            "id": r["id"],
            "tokens": {str(k): [float(x) for x in v] for k, v in sorted(toks.items())},
            "per_tensor": info,
            "field_per_tensor": fdetail,
        }
        rl, dl = append_row(rows_path, detail_path, rec, detail)
        entries.append(checkpoint_entry(r["id"], src, rec["increment_sha256"], rl, dl))
        write_checkpoint(out, binding, entries)
        recs.append(rec)
        done[r["id"]] = detail
        n_new += 1
        del D, d_edit, per, toks
        gc.collect()
        guard.check()
        log(
            f"{n}/{len(todo)} {r['id']} {r['family']} a={rec['a']} q={rec['q']} tokens={rec['n_tokens']} ({seconds:.1f}s)"
        )
        if n_new == 1 and pricing is None:
            naive, weighted = projection(setup_seconds, seconds, count, counts)
            projected = max(naive, weighted)
            pricing = {
                "setup_seconds": setup_seconds,
                "first_row_id": r["id"],
                "first_row_seconds": seconds,
                "first_row_stored_tensors": count,
                "n_rows": len(todo),
                "stored_tensors_total": sum(counts),
                "projected_wall_seconds_naive": naive,
                "projected_wall_seconds_tensor_weighted": weighted,
                "projected_wall_seconds": projected,
                "projected_wall_hours": projected / 3600,
                "cap_wall_hours": args.max_wall_hours,
                "within_cap": projected <= args.max_wall_hours * 3600,
                "threads": args.threads,
            }
            write_json(out / "pricing.json", pricing)
            log(f"pricing: {json.dumps(pricing)}")
            if not pricing["within_cap"]:
                write_manifest(
                    args,
                    out,
                    "priced_only",
                    frozen,
                    records,
                    probe_ident,
                    kdoc,
                    k_sha,
                    pricing,
                    recs,
                    None,
                    t_start,
                    setup_seconds,
                    base_hash,
                    layer_types,
                    slots,
                    boundary=boundary,
                    guard=guard,
                    resume=resume,
                )
                raise SystemExit(
                    f"price_exceeds_cap: projected {projected / 3600:.2f} h > {args.max_wall_hours} h"
                )
    complete = is_complete(done, rows, args.limit)
    status = "complete" if complete else f"partial ({len(done)}/{len(rows)})"
    write_manifest(
        args,
        out,
        status,
        frozen,
        records,
        probe_ident,
        kdoc,
        k_sha,
        pricing,
        recs,
        done,
        t_start,
        setup_seconds,
        base_hash,
        layer_types,
        slots,
        boundary=boundary,
        guard=guard,
        resume=resume,
    )


def write_manifest(
    args,
    out,
    status,
    frozen,
    records,
    probe_ident,
    kdoc,
    k_sha,
    pricing,
    recs,
    done,
    t_start,
    setup_seconds,
    base_hash,
    layer_types,
    slots,
    boundary=None,
    guard=None,
    resume=None,
):
    from safetensors.torch import save_file

    if Path(args.manifest).exists():
        raise FileExistsError(f"{args.manifest} exists; a governed manifest is never overwritten")
    recs = sorted(recs, key=lambda r: r["id"])
    files = {"kframes.json": k_sha, "kframes.safetensors": kdoc["kframes_safetensors_sha256"]}
    token_record = None
    if done:
        ids = sorted(done)
        X, P, E = assemble(
            ids,
            {i: done[i]["tokens"] for i in ids},
            len(slots),
            {r["id"]: r["eligibility"] == "eligible" for r in recs},
        )
        save_file(
            {
                "X": torch.from_numpy(X),
                "presence": torch.from_numpy(P),
                "row_eligible": torch.from_numpy(E),
            },
            str(out / "tokens.safetensors"),
            metadata={"row_ids": json.dumps(ids), "columns": json.dumps(COLUMNS)},
        )
        aq = {
            r["id"]: {
                k: r[k] for k in ("a", "a_fp64", "q", "q_partial", "field_status", "eligibility")
            }
            for r in recs
        }
        files["aq.json"] = write_json(out / "aq.json", aq)
        files["tokens.safetensors"] = pf.sha256_file(out / "tokens.safetensors")
        token_record = {
            "path": str(out / "tokens.safetensors"),
            "sha256": files["tokens.safetensors"],
            "shape": list(X.shape),
            "row_ids_sha256": sha256_bytes(json.dumps(ids).encode()),
            "X_sha256": rf._tensor_hash(torch.from_numpy(X)),
            "presence_sha256": rf._tensor_hash(torch.from_numpy(P)),
            "n_present_tokens": int(P.sum()),
        }
    for name in ("rows.jsonl", "detail.jsonl", "pricing.json", CHECKPOINT):
        if (out / name).exists():
            files[name] = pf.sha256_file(out / name)
    excl = {r["id"]: r["eligibility"] for r in recs if r["eligibility"] != "eligible"}
    doc = {
        "schema": SCHEMA,
        "status": status,
        "basis": "PREREG_H1_TF F12.3 (K+F tokens 0-38 with K blocks + probe columns 39-41), F12.5 (real token matrix "
        "for the outcome-free simulation), F12.8 (fields and K/SVD/probe extraction, CPU, measured); F5 schema pins "
        "(10 Nov 2026); Board A F12 review conditions 2, 6, 7",
        "ordering_note": "computed on the feat/f12-inputs branch after population.json and its fold manifest were "
        "written and hash-verified (population_frozen), before the F12 C1 commit; the chair decides whether this "
        "run stands as the C2 token output or is recomputed after C1 with the pinned code",
        "population_frozen": frozen,
        "columns": COLUMNS,
        "column_rules": {
            "0-15": "vec(M_t) row-major, M_t = (Lambda_G^1/2 U_G^T B)(A U_A Lambda_A^1/2); zeros with flag "
            "k_cutoff_tie",
            "16-23": "log sigma_j(D_t), natural log, untruncated fp32 thin SVD, 0 where absent",
            "24-31": "1[sigma_j > 1e-3 sigma_1]",
            "32": "sum_{j>8} sigma_j^2 / sum_j sigma_j^2",
            "33": "layer / 24",
            "34": "1 iff the tensor's layer type is full_attention (field module_roles)",
            "35-38": "<g_t, D_t> (wikitext_nll, fp32 as response_field.contract), q_t (base spectral factors), "
            "c_t = 0 (roots), presence",
            "39-41": "mlp.down_proj only: RMS_i of mean(D x)_i, RMS_i of RMS(D x)_i over the G2 probe cache (mean "
            "over a prompt's tokens, equal prompt weight), presence; 0 elsewhere (drafting default)",
            "rank": "adapter r (lora_sft; chain inner lora_sft), 1 (abliteration, random_dir and their chain roots)",
            "presence": "token present iff the tensor is one of the 186 matrix slots and D_t != 0 after rounding",
            "eligibility": "a row with a supported changed tensor lacking r, a finite SVD or a K frame is excluded",
        },
        "slots": {
            "n": len(slots),
            "names_sha256": sha256_bytes(json.dumps([s[3] for s in slots]).encode()),
            "layer_types": layer_types,
        },
        "inputs": records | {"probe_cache": probe_ident, "zoo": str(args.zoo)},
        "base": {
            "model": {"repo": pf.MODEL[0], "revision": pf.MODEL[1], "dtype": "bfloat16"},
            "state_hash": base_hash,
        },
        "kframes": {
            "path": str(out / "kframes.json"),
            "sha256": k_sha,
            "kframes_safetensors_sha256": kdoc["kframes_safetensors_sha256"],
            "n_tensors": len(kdoc["tensors"]),
            "flag_counts": kdoc["counts"],
        },
        "tokens": token_record,
        "files": files,
        "out_dir": str(out),
        "rows_sha256": sha256_bytes(canonical_json(recs)),
        "summary": {
            "n_rows_done": len(recs),
            "n_rows_population": len(json.loads(args.population.read_text())["rows"]),
            "eligible": sum(r["eligibility"] == "eligible" for r in recs),
            "excluded": excl,
            "field_status": dict(Counter(r["field_status"] for r in recs)),
            "q_partial_rows": sum(bool(r["q_partial"]) for r in recs),
            "rows_with_unsupported_changed_tensors": sorted(
                r["id"] for r in recs if r["unsupported_changed_tensors"]
            ),
            "token_flag_rows": dict(Counter(f for r in recs for f in r["token_flags"])),
            "rank_sources": dict(Counter(r["rank_source"] for r in recs)),
            "omitted_energy_gt_0.01_rows": sum((r["omitted_energy_max"] or 0) > 0.01 for r in recs),
            "pricing": pricing,
            "timing": records["timing"]
            | {
                "setup_seconds": setup_seconds,
                "row_seconds_total": sum(r["seconds"] for r in recs),
                "wall_seconds": time.time() - t_start,
                "threads": args.threads,
            },
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "outcomes_read_intent": {"statement": OUTCOMES_STATEMENT, "label": INTENT_LABEL},
        "read_boundary": boundary.record() if boundary is not None else None,
        "resources": guard.stop() if guard is not None else None,
        "resume": resume,
        "code": code_identity(),
    }
    sha = write_json(args.manifest, doc)
    log(f"wrote {args.manifest} ({status}) sha256 {sha}")
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=ALLOWED_OUT / "tokens")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    p.add_argument("--g2", type=Path, default=ROOT / "data/runs/g2")
    p.add_argument("--zoo", default=str(ROOT / "data/zoo/qwen3.5-0.8b"))
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "6")))
    p.add_argument("--max-wall-hours", type=float, default=8.0)
    p.add_argument("--rss-cap-bytes", type=int, default=fg.DEFAULT_CAP_BYTES)
    p.add_argument(
        "--chair-ruling", help="exact heading of the dated chair ruling for a cap above 4 GB"
    )
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
