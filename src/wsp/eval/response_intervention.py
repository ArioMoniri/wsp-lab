"""Part A intervention runner (PREREG_H1_TF A5.3; SYNTHESIS 7 Nov §4.1, gate G3).

1. :func:`select_edits` — A5.3.1 selection: one representative per certified source component (its
   lexicographically smallest eligible canonical id), effective-norm tertiles per family with rank cutpoints
   b1 = floor(N/3), b2 = floor(2N/3) on the (effective_norm, id) order, quotas n // 3 per tertile with the
   remainder given to the lowest tertiles first (chair: 10/10/10 LoRA, 4/3/3 chain); a shortage stops.
2. :func:`build_variants` — for one edit, the rotation R at s = 1 and the scaled edits s·Δ, each applied at
   the actual predecessor and rounded to bf16 (A5.3.3), staged as a zoo entry outside the canonical zoo.
   The rotation pair is audited under ``--audit-rule`` (``f10``, F10.3 remedy d, by default; ``a533``,
   the registered realised audit, on request); the rule is recorded in ``variant.json`` and the manifest.
   ``prepare --rotation-only`` (F10.7) re-stages only the rotations of a staged Part A under F10, re-hashes
   the staged ladders without rebuilding them, keeps the staged manifest as ``manifest_a533.json`` and
   opens no evaluation record.
3. :func:`manifest_for` — the A5.11 pieces computable from the staged zoo.
4. :func:`plan_scoring` / :func:`scoring_summary` — which predecessor and original references are on file
   with the same scoring identity (A5.3.4), and the priced count at 1.04 min per entry.

Staged delta convention. ``wsp.eval.batch`` scores an entry as bf16(base + delta). For a root edit
(predecessor = base) the staged ``delta.safetensors`` is the realised increment bf16(W + X) − W over the
edited tensors. For a chain step (predecessor ≠ base, stored ``dense_cumulative``) the scorer still applies
onto the base, so ``delta.safetensors`` is the realised cumulative state minus base (edited tensors plus the
predecessor's own changed tensors) and the realised increment bf16(W + X) − W over the edited tensors is
written next to it as ``increment.safetensors``. Every written tensor is checked to reproduce the intended
bf16 state bitwise under the scorer's arithmetic.

CLI::

    python -m wsp.eval.response_intervention prepare --zoo Z --features F --sidecar S --out O --manifest-out M
    python -m wsp.eval.response_intervention prepare --zoo Z --selection selection.json \
        --references references.json --text-groups text_groups.json --out O --manifest-out M
    python -m wsp.eval.response_intervention prepare --rotation-only --staged-manifest M_staged \
        --zoo Z --selection selection.json --references references.json --out O --manifest-out M
    python -m wsp.eval.response_intervention score --stage O/<zoo name> [--manifest M] [--dry-run]

``--selection`` stages the registered picks of a committed ``selection.json`` instead of re-running the
selection; ``--references`` names the bytes holder of every alias-only predecessor (its state is loaded from
the holder's stored bytes and the scoring plan checks the holder's bytes against the alias's record);
``--text-groups`` adds the primary-record and 64-window checks of the A5 F3(i) reuse proof to every reused
reference. ``score --manifest`` also scores the references the plan lists as missing, on the source zoo.

``prepare`` stages under ``O/<zoo name>`` so the TF-v1 output directory of ``wsp.eval.batch``
(``data/evals/<zoo name>/tf_v1``) is shared with the canonical references. ``score`` only runs the existing
batch CLI on the staged variants; run it inside ``agent-tmux``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import save_file

from wsp.features import rotation

FAMILIES = ("lora_sft", "chain")
MINUTES_PER_ENTRY = 1.04
ROTATION_DOMAIN = "20261107:rotation:"
NUMERIC_TOL = 1e-5  # A5.3.3 orthogonality / reconstruction residual (chair default)
AUDIT_RULES = ("a533", "f10")  # registered realised audit (A5.3/A5.3.3) or F10.3 (remedy d)
DEFAULT_AUDIT_RULE = "f10"
DEFAULT_CONFIG = "configs/eval_tf_v1.yaml"


class SelectionUnresolved(RuntimeError):
    """A tertile cell is short of its quota: selection stops, no cross-cell replacement (A5.3.1)."""


class SelectionBlocked(RuntimeError):
    """Nonfinite/missing norms or uncertified sources block selection rather than being dropped."""


@dataclass(frozen=True)
class Selection:
    edit_id: str
    component_id: str
    family: str
    tertile: int
    rank: int
    effective_norm: float
    predecessor_id: str


@dataclass
class Variant:
    variant_id: str
    kind: str
    strength: float
    seed: int
    source_id: str
    predecessor_id: str
    entry_dir: Path
    excluded: bool = False
    exclusions: list = field(default_factory=list)
    audit_rule: str | None = None  # rotation variants only
    realised_increment_ladder: bool | None = None  # scale variants only


# ------------------------------------------------------------------------------------------- selection


def tertile_quotas(n):
    """n // 3 per tertile, remainder to the lowest tertiles first (chair decision)."""
    base, rem = divmod(n, 3)
    return [base + (t < rem) for t in range(3)]


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def select_with_ledger(manifest_rows, feature_norms, sidecar, n_lora=30, n_chain=10):
    """A5.3.1 selection plus its ledger (ordered ids, cutpoints, cells, rejected rows).

    ``manifest_rows``: zoo manifest rows (``id``, ``family``, ``parent_id``, ``eligibility``); rows without an
    ``id`` (headers) are ignored. ``feature_norms``: id → effective norm of the full realised incremental edit
    from the pre-outcome feature manifest; an id absent from it has no feature manifest (ineligible), an id
    present with a missing or nonfinite value blocks. ``sidecar``: rows ``{id, component_id,
    closure_certified, exclusion_reason?, canonical_id?}`` from the frozen inventory and exclusion ledger.
    """
    side = {r["id"]: r for r in sidecar}
    rows = [r for r in manifest_rows if "id" in r]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in manifest rows")
    rejected, eligible, blocked = [], [], []
    for row in sorted(rows, key=lambda r: r["id"]):
        i, fam, s = row["id"], row.get("family"), side.get(row["id"], {})
        reason = None
        if fam not in FAMILIES:
            reason = f"unsupported_family:{fam}"
        elif row.get("eligibility", "production") != "production":
            reason = f"eligibility:{row.get('eligibility')}"
        elif s.get("exclusion_reason"):
            reason = s["exclusion_reason"]
        elif s.get("canonical_id") not in (None, i):
            reason = f"alias_of:{s['canonical_id']}"
        elif i not in feature_norms:
            reason = "no_feature_manifest"
        if reason:
            rejected.append({"id": i, "reason": reason})
            continue
        if s.get("closure_certified") is not True or not s.get("component_id"):
            blocked.append(f"{i}: source closure not certified")
        elif not _finite(feature_norms[i]):
            blocked.append(f"{i}: missing or nonfinite effective norm {feature_norms[i]!r}")
        else:
            eligible.append(
                {
                    "id": i,
                    "family": fam,
                    "component": s["component_id"],
                    "norm": float(feature_norms[i]),
                    "predecessor": row.get("parent_id") or "base",
                }
            )
    if blocked:
        raise SelectionBlocked("; ".join(blocked))

    representatives, seen = [], {}
    for r in eligible:  # already sorted by canonical id
        if r["component"] in seen:
            rejected.append(
                {"id": r["id"], "reason": f"component_represented_by:{seen[r['component']]}"}
            )
        else:
            seen[r["component"]] = r["id"]
            representatives.append(r)
    rejected.sort(key=lambda r: r["id"])

    ledger = {"rejected": rejected, "families": {}, "n_eligible": len(eligible)}
    selected = []
    for fam, n in (("lora_sft", n_lora), ("chain", n_chain)):
        ordered = sorted(
            (r for r in representatives if r["family"] == fam), key=lambda r: (r["norm"], r["id"])
        )
        N = len(ordered)
        b1, b2 = N // 3, 2 * N // 3
        cells = [ordered[0:b1], ordered[b1:b2], ordered[b2:N]]
        quotas = tertile_quotas(n)
        ledger["families"][fam] = {
            "N": N,
            "b1": b1,
            "b2": b2,
            "cutpoints": [
                [ordered[b]["norm"], ordered[b]["id"]] if b < N else None for b in (b1, b2)
            ],
            "ordered": [r["id"] for r in ordered],
            "cells": [[r["id"] for r in c] for c in cells],
            "quotas": quotas,
        }
        rank = {r["id"]: k for k, r in enumerate(ordered)}
        for t, (cell, quota) in enumerate(zip(cells, quotas)):
            if len(cell) < quota:
                err = SelectionUnresolved(
                    f"{fam} tertile {t}: {len(cell)} representatives < quota {quota}"
                )
                err.ledger = ledger
                raise err
            for r in sorted(cell, key=lambda r: r["id"])[:quota]:
                selected.append(
                    Selection(
                        r["id"], r["component"], fam, t, rank[r["id"]], r["norm"], r["predecessor"]
                    )
                )
    if len(selected) != n_lora + n_chain or len({s.component_id for s in selected}) != len(
        selected
    ):
        raise AssertionError("selection does not have one edit per component")
    ledger["selected"] = [asdict(s) for s in selected]
    return selected, ledger


def select_edits(manifest_rows, feature_norms, sidecar, n_lora=30, n_chain=10):
    """A5.3.1 selection; see :func:`select_with_ledger`."""
    return select_with_ledger(manifest_rows, feature_norms, sidecar, n_lora, n_chain)[0]


# -------------------------------------------------------------------------------------------- variants


def rotation_seed(edit_id):
    """A5.3.3: SHA256 of ``20261107:rotation:<id>``, first eight bytes big-endian, mod 2^63."""
    digest = hashlib.sha256(f"{ROTATION_DOMAIN}{edit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (1 << 63)


def variant_id(source_id, kind, strength, seed):
    return hashlib.sha256(
        f"a5-part-a:{source_id}:{kind}:{float(strength)!r}:{seed}".encode()
    ).hexdigest()[:16]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(x):
    return float(torch.linalg.vector_norm(x.double()))


def _moments(x, w):
    """Logged, not preserved (S31 §4.1): coordinate moments, sign fractions, cosine to the predecessor."""
    x64, w64 = x.double().flatten(), w.double().flatten()
    denom = float(torch.linalg.vector_norm(x64) * torch.linalg.vector_norm(w64))
    return {
        "mean": float(x64.mean()),
        "std": float(x64.std()) if x64.numel() > 1 else 0.0,
        "frac_pos": float((x64 > 0).double().mean()),
        "frac_nonzero": float((x64 != 0).double().mean()),
        "cos_to_predecessor": float(x64 @ w64) / denom if denom > 0 else 0.0,
    }


def _rotation_diagnostics(D, R_raw, R_eff, W, rows, cols, generator):
    audit = rotation.audit(D, R_eff, rows, cols)
    u, s, vh, rank = rotation.block_svd(D, rows, cols)
    block = D.float()[rows[:, None], cols]
    recon = _norm(u @ torch.diag(s) @ vh - block) / _norm(block)
    replay = torch.Generator()
    replay.set_state(
        generator.get_state()
    )  # rotate_block's draws, replayed for the orthogonality audit
    orth = 0.0
    for n in (len(rows), len(cols)):
        q = rotation.haar_orthogonal(n, replay).double()
        orth = max(orth, _norm(q.T @ q - torch.eye(n, dtype=torch.float64)) / math.sqrt(n))
    d_norm = _norm(D)
    audit.update(
        {
            "orthogonality_error": orth,
            "reconstruction_residual": recon,
            "block_rank": rank,
            "block_shape": [len(rows), len(cols)],
            "prerounding_norm_mismatch": abs(_norm(R_raw) - d_norm) / d_norm,
            "rounding_error_rel": _norm(R_eff - R_raw) / d_norm,
            "nonzero_d": int((D != 0).sum()),
            "nonzero_r": int((R_eff != 0).sum()),
            "logged_d": _moments(D, W),
            "logged_r": _moments(R_eff, W),
        }
    )
    audit["s31_passes"] = audit["passes"]
    audit["passes"] = bool(audit["passes"] and orth <= NUMERIC_TOL and recon <= NUMERIC_TOL)
    return audit


def _edit_audit(audits, D, R):
    if not audits:
        return {"passes": False, "n_tensors": 0}
    d = math.sqrt(sum(_norm(D[k]) ** 2 for k in audits))
    r = math.sqrt(sum(_norm(R[k]) ** 2 for k in audits))
    return {
        "passes": all(a["passes"] for a in audits.values()),
        "n_tensors": len(audits),
        "n_failing": sum(not a["passes"] for a in audits.values()),
        "max_norm_mismatch": max(a["norm_mismatch"] for a in audits.values()),
        "max_spectrum_mismatch": max(a["spectrum_mismatch"] for a in audits.values()),
        "max_abs_rank_difference": max(abs(a["rank_difference"]) for a in audits.values()),
        "max_orthogonality_error": max(a["orthogonality_error"] for a in audits.values()),
        "max_reconstruction_residual": max(a["reconstruction_residual"] for a in audits.values()),
        "pooled_norm_mismatch": abs(r - d) / d,
        "all_support_identical": all(a["support_identical"] for a in audits.values()),
    }


A533_CRITERIA = {  # criterion -> how _rotation_diagnostics decides it (A5.3 tolerances, A5.3.3 numerics)
    "norm": lambda a: a["norm_mismatch"] <= 2e-2,
    "spectrum": lambda a: a["spectrum_mismatch"] <= 5e-2,
    "rank": lambda a: abs(a["rank_difference"]) <= 1,
    "support": lambda a: bool(a["support_identical"]),
    "orthogonality": lambda a: a["orthogonality_error"] <= NUMERIC_TOL,
    "reconstruction": lambda a: a["reconstruction_residual"] <= NUMERIC_TOL,
}
A533_FIELDS = (
    "norm_mismatch",
    "spectrum_mismatch",
    "rank_d",
    "rank_r",
    "rank_difference",
    "support_identical",
    "orthogonality_error",
    "reconstruction_residual",
    "block_rank",
    "prerounding_norm_mismatch",
    "rounding_error_rel",
    "s31_passes",
    "passes",
)


def _a533_diagnostic(D, R_raw, R_eff, W, rows, cols, generator):
    """The complete registered verdict (S31 realised criteria plus fp32 orthogonality and reconstruction,
    exactly :func:`_rotation_diagnostics`) with each criterion's pass/fail. Logged under F10, never a reason."""
    try:
        a = _rotation_diagnostics(D, R_raw, R_eff, W, rows, cols, generator)
    except (ValueError, RuntimeError) as e:
        return {"passes": False, "s31_passes": False, "error": f"{type(e).__name__}: {e}"}
    return {k: a[k] for k in A533_FIELDS} | {
        "criteria": {name: bool(rule(a)) for name, rule in A533_CRITERIA.items()}
    }


def _load_written(path):
    """Tensors reloaded from a written safetensors file; None if missing or unreadable (fails closed)."""
    from safetensors.torch import load_file

    try:
        return load_file(str(path)) if Path(path).exists() else None
    except Exception:  # noqa: BLE001 - any unreadable file is a failed replay, not an error
        return None


def _replay_environment():
    with rotation.torch_threads(rotation.F10_THREADS):
        return rotation.numerical_environment() | {"replay_threads": rotation.F10_THREADS}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


F10_POOLED_DEFINITIONS = {
    "pooled norms": "sqrt of the sum over edited tensors of squared Frobenius norms (D, R, R_eff)",
    "pooled_norm_mismatch_signed": "(pooled ||R_eff|| - pooled ||D||) / pooled ||D|| (F10.3(v) guard)",
    "pooled_energy_ratio": "sum ||R_eff||^2 / sum ||D||^2 (F10.4 rho); log_energy_ratio = ln of it",
    "pooled_rounding_error_rel": "sqrt(sum ||R_eff - R||^2) / pooled ||D|| (pooled eps_R)",
    "pooled_noise_floor": "sqrt(sum over tensors and block entries of ulp_bf16(W)^2 / 12)",
    "pooled_kappa": "pooled ||D|| / sqrt(sum ulp_bf16(W)^2); pooled_eps_hat = min(1, 1/(sqrt(12) kappa))",
    "pooled_eps_D": "roots: sqrt(sum ||D - Delta||^2) / pooled ||D||; non-root chains: unavailable",
    "pooled_spectrum_mismatch": "per-tensor sorted spectra concatenated: sqrt(sum ||s(R_eff)-s(D)||^2)"
    " / sqrt(sum ||s(D)||^2); no pooled matrix is formed",
    "rank sums": "sum over tensors of each tensor's rank (own-sigma_1 cuts); tau stays per tensor",
    "noise_calibrated_passes": "every tensor passes the noise-calibrated audit (F10.5(e))",
}


def _f10_pooled(audits, edited, root):
    """F10.4 pooled per-edit diagnostics; any missing or nonfinite input makes that value None."""
    rows = [audits.get(k) if isinstance(audits.get(k), dict) else {} for k in edited]

    def col(name):
        vals = [a.get(name) for a in rows]
        return vals if vals and all(_num(v) for v in vals) else None

    def ssq(vals):
        return sum(v * v for v in vals) if vals is not None else None

    def ratio(num, den):
        return math.sqrt(num) / math.sqrt(den) if num is not None and den else None

    d, e, r = col("d_norm"), col("r_eff_norm"), col("r_norm")
    d2, e2, r2 = ssq(d), ssq(e), ssq(r)
    eps_r, eps_d, ulp2 = col("rounding_error_rel"), col("eps_D"), col("sum_ulp2_block")
    rho = e2 / d2 if d2 and e2 is not None else None
    signed = (math.sqrt(e2) - math.sqrt(d2)) / math.sqrt(d2) if d2 and e2 is not None else None
    kappa = ratio(d2, sum(ulp2)) if ulp2 is not None and sum(ulp2) > 0 else None
    eps_hat = min(1.0, 1.0 / (math.sqrt(12) * kappa)) if kappa else None
    spec = ratio(ssq(col("spectrum_diff_norm")), ssq(col("spectrum_ref_norm")))
    pre_spec = ratio(
        ssq(col("prerounding_spectrum_diff_norm")), ssq(col("prerounding_spectrum_ref_norm"))
    )

    def total(name):
        vals = col(name)
        return int(sum(vals)) if vals is not None else None

    def worst(name, f=abs):
        vals = col(name)
        return max(f(v) for v in vals) if vals is not None else None

    noise = [a.get("noise_calibrated") or {} for a in rows]
    out = {
        "pooled_d_norm": math.sqrt(d2) if d2 is not None else None,
        "pooled_r_norm": math.sqrt(r2) if r2 is not None else None,
        "pooled_r_eff_norm": math.sqrt(e2) if e2 is not None else None,
        "pooled_norm_mismatch": abs(signed) if signed is not None else None,
        "pooled_norm_mismatch_signed": signed,
        "pooled_prerounding_norm_mismatch": abs(math.sqrt(r2) - math.sqrt(d2)) / math.sqrt(d2)
        if d2 and r2 is not None
        else None,
        "pooled_energy_ratio": rho,
        "log_energy_ratio": math.log(rho) if rho else None,
        "pooled_rounding_error_rel": ratio(
            ssq([x * y for x, y in zip(eps_r, d)]) if eps_r and d else None, d2
        ),
        "pooled_noise_floor": math.sqrt(sum(ulp2) / 12) if ulp2 is not None else None,
        "pooled_kappa": kappa,
        "pooled_eps_hat": eps_hat,
        "pooled_eps_D": ratio(ssq([x * y for x, y in zip(eps_d, d)]), d2)
        if root and eps_d and d
        else None,
        "eps_D_status": "root_stored_delta" if root else "unavailable_non_root",
        "pooled_spectrum_mismatch": spec,
        "pooled_prerounding_spectrum_mismatch": pre_spec,
        "sum_rank_d": total("rank_d"),
        "sum_rank_r": total("rank_r"),
        "sum_prerounding_rank_d": total("prerounding_rank_d"),
        "sum_prerounding_rank_r": total("prerounding_rank_r"),
        "sum_rank_n_near": total("rank_n_near"),
        "max_abs_prerounding_rank_difference": worst("prerounding_rank_difference"),
        "sum_noise_rank_d": total("noise_rank_d"),
        "sum_noise_rank_r": total("noise_rank_r"),
        "max_abs_noise_rank_difference": max(
            abs(a["noise_rank_r"] - a["noise_rank_d"]) for a in rows
        )
        if col("noise_rank_d") is not None and col("noise_rank_r") is not None
        else None,
        "tau_by_tensor": dict(zip(edited, col("tau") or [None] * len(edited))),
        "noise_calibrated_passes": all(a.get("noise_calibrated_passes") is True for a in rows),
        "n_tensors_failing_noise_calibrated": sum(
            a.get("noise_calibrated_passes") is not True for a in rows
        ),
        "noise_calibrated_failures": {
            c: sum(n.get(c) is not True for n in noise) for c in ("norm", "spectrum", "rank")
        },
        "pooled_noise_calibrated_norm_spectrum": signed is not None
        and spec is not None
        and eps_hat is not None
        and abs(signed) <= 2e-2 + eps_hat
        and spec <= 5e-2 + eps_hat,
        "realised_s31_passes": all(a.get("s31_passes") is True for a in rows),
        "max_norm_mismatch": worst("norm_mismatch"),
        "max_spectrum_mismatch": worst("spectrum_mismatch"),
        "max_abs_rank_difference": worst("rank_difference"),
        "max_orthogonality_error": worst("orthogonality_error"),
        "max_reconstruction_residual": worst("reconstruction_residual"),
        "max_prerounding_norm_mismatch": worst("prerounding_norm_mismatch"),
        "max_prerounding_spectrum_mismatch": worst("prerounding_spectrum_mismatch"),
        "max_construction_residual": worst("construction_residual"),
        "max_rounding_bound_ratio": worst("rounding_bound_max_ratio"),
        "max_rounding_excess_fp32_spacings": worst("rounding_bound_max_excess_fp32_spacings"),
        "n_rounding_bound_violations": total("rounding_bound_violations"),
        "n_double_rounding": total("n_double_rounding"),
        "n_exceed_half_ulp_w": total("n_exceed_half_ulp_w"),
        "all_support_identical": all(a.get("support_identical") is True for a in rows),
        "all_replay": all(a.get("replay_passes") is True for a in rows),
    }
    out["in_subset_pooled_norm_2e-2"] = signed is not None and abs(signed) <= 2e-2
    out["in_subset_noise_calibrated"] = out["noise_calibrated_passes"]
    a533 = {k: (audits.get(k) or {}).get("a533") or {} for k in edited}
    out["a533_diagnostic"] = {
        "passes": all(a.get("passes") is True for a in a533.values()),
        "s31_passes": all(a.get("s31_passes") is True for a in a533.values()),
        "n_failing": sum(a.get("passes") is not True for a in a533.values()),
        "failing_tensors": sorted(k for k, a in a533.items() if a.get("passes") is not True),
        "criterion_failures": {
            c: sum((a.get("criteria") or {}).get(c) is not True for a in a533.values())
            for c in A533_CRITERIA
        },
        "would_exclude": [
            {"tensor": k, "reason": "audit_failed"}
            for k, a in a533.items()
            if a.get("passes") is not True
        ],
    }
    out["aggregation"] = F10_POOLED_DEFINITIONS
    return out


def _f10_edit_audit(audits, edited, root, extra=()):
    """F10 per-edit verdict and pooled F10.4/F10.5 diagnostics, failing closed.

    Every edited tensor needs a complete audit (:func:`rotation.f10_verdict`; a missing one fails every
    check) and the pooled half-norm guard needs every tensor's finite norms. ``extra`` adds pair-level
    failures (carried-tensor round trips, unexpected file keys). Returns the per-edit dict; its
    ``exclusions`` list (tensor failures, ``extra``, then ``sub_resolution``) decides the pair."""
    exclusions = []
    for k in edited:
        a = audits.get(k)
        v = rotation.f10_verdict(a)
        if isinstance(a, dict):
            a |= {
                "passes": v["passes"],
                "failed_checks": v["failed_checks"],
                "reason": v["reason"],
                "fail_closed_detail": v["detail"],
            }
        if not v["passes"]:
            x = {"tensor": k, "reason": v["reason"]}
            if v["detail"]:
                x["detail"] = ";".join(v["detail"])
            exclusions.append(x)
    exclusions += list(extra)
    rows = [audits.get(k) if isinstance(audits.get(k), dict) else {} for k in edited]
    guard = rotation.sub_resolution_guard(
        [a.get("d_norm") for a in rows], [a.get("r_eff_norm") for a in rows]
    )
    if not guard["passes"]:
        x = {"tensor": None, "reason": "sub_resolution"}
        if guard["mismatch"] is None:
            x["detail"] = "guard_incomplete"
        exclusions.append(x)
    return {
        "audit_rule": "f10",
        "passes": not exclusions,
        "n_tensors": len(edited),
        "n_audited": sum(isinstance(audits.get(k), dict) for k in edited),
        "n_failing": sum(x["tensor"] in edited for x in exclusions),
        "exclusions": exclusions,
        "sub_resolution_guard": guard,
    } | _f10_pooled(audits, edited, root)


def _stage_header(canonical, edit):
    manifest = canonical / "manifest.jsonl"
    if manifest.exists():
        for line in reversed(manifest.read_text().splitlines()):
            if line.strip() and json.loads(line).get("type") == "header":
                return json.loads(line)
    return {
        "type": "header",
        "base_model": edit.get("base_model"),
        "base_revision": edit["base_revision"],
    }


def _upsert_stage_rows(stage_root, header, rows, remove=()):
    path = stage_root / "manifest.jsonl"
    existing = (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )
    keyed = {r["id"]: r for r in existing if "id" in r}
    keyed.update({r["id"]: r for r in rows})
    for (
        vid
    ) in remove:  # an excluded variant has no staged row (e.g. a rotation re-staged under F10)
        keyed.pop(vid, None)
    lines = [header] + [keyed[k] for k in sorted(keyed)]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))
    tmp.replace(path)


def build_variants(
    entry_dir,
    predecessor_state,
    seed,
    strengths=(0.25, 0.5),
    *,
    stage_root,
    base_state=None,
    predecessor_id=None,
    predecessor_holder=None,
    audit_rule=DEFAULT_AUDIT_RULE,
):
    """Rotation (s = 1) and scaled variants of one edit, staged under ``stage_root/<variant id>``.

    ``predecessor_state`` is the full bf16 state W of the actual predecessor. For a chain step whose stored
    delta is cumulative from the base, ``base_state`` is required (the scorer applies deltas onto the base).
    Root edits scale the raw stored delta (Δ); chain steps scale their realised increment, the only per-step
    object stored. One torch Generator per edit, seeded with ``seed``, is shared across the edited tensors in
    sorted-name order through :func:`rotation.rotate_block` (each tensor starts from its cloned state).
    A tensor with an exclusion reason, or a failed audit, excludes the whole rotation pair: its entry holds
    only ``variant.json`` and is not added to the staged manifest. Never writes into the canonical zoo.
    ``audit_rule`` selects the audit that decides the rotation exclusions (the construction is the same):
    ``a533`` is the registered realised audit (norm, spectrum and 1e-3 rank of R_eff against D_eff, plus
    fp32 orthogonality and reconstruction); ``f10`` is F10.3 (remedy d), with the construction at 4
    threads: the rotation files are written, reloaded and audited per tensor (:func:`rotation.f10_tensor_audit`:
    fp64 construction checks with the near-cut rank budget, support of D, R and R_eff, bit-pattern replay
    from a fresh seeded generator at 4 threads against the reloaded R_eff, the scorer's bf16(base + delta)
    on the reloaded delta including carried predecessor tensors, and the rounding bound with one fp32
    spacing), then per edit the pooled half-norm guard (``sub_resolution``); any missing or nonfinite
    audit item fails closed and an excluded pair's files are removed. The realised statistics and the
    complete a533 verdict are written as diagnostics only. Scale variants of a step
    whose predecessor is not the base scale its realised increment D (no per-step Δ is stored) and carry
    ``realised_increment_ladder: true``.
    ``predecessor_holder`` records the zoo entry whose stored bytes gave ``predecessor_state`` when the
    predecessor is an alias without bytes of its own; the scoring plan checks the holder's bytes.
    """
    from wsp.zoo.apply import load_entry_delta

    if audit_rule not in AUDIT_RULES:
        raise ValueError(f"unknown audit rule {audit_rule!r} (expected one of {AUDIT_RULES})")
    entry_dir, stage_root = Path(entry_dir), Path(stage_root)
    canonical = entry_dir.resolve().parent
    staged = stage_root.resolve()
    if staged == canonical or canonical in staged.parents:
        raise ValueError(f"refusing to stage variants inside the canonical zoo {canonical}")
    edit_text = (entry_dir / "edit.json").read_text()
    edit = json.loads(edit_text)
    source_id = edit.get("id", entry_dir.name)
    parent = edit.get("parent_id")
    cumulative = bool((edit.get("provenance") or {}).get("cumulative_snapshot"))
    if predecessor_id is None:
        predecessor_id = parent or "base"
    elif predecessor_id != (parent or "base"):
        raise ValueError(
            f"{source_id}: predecessor {predecessor_id} differs from parent_id {parent}"
        )
    root = base_state is None
    if root and parent is not None:
        raise ValueError(
            f"{source_id}: a step with a predecessor needs base_state (stored delta is from base)"
        )
    base = predecessor_state if root else base_state
    W = predecessor_state

    stored = load_entry_delta(entry_dir, base)
    keys = set(stored)
    pred_changed = set() if root else {k for k in W if not torch.equal(W[k], base[k])}
    D, S = {}, {}
    for k in sorted(keys | pred_changed):
        S[k] = (base[k].float() + stored[k].float()).to(base[k].dtype) if k in stored else base[k]
        D[k] = rotation.realised_increment(W[k], S[k])
    if root:
        edited = sorted(keys)
        raw = {k: stored[k].float() for k in edited}
        scale_source, convention = "raw_stored_delta", "realised_increment_at_base"
    else:
        edited = sorted(k for k in D if torch.any(D[k] != 0))
        raw = {k: D[k] for k in edited}
        scale_source, convention = "realised_increment", "realised_cumulative_from_base"
    if not edited:
        raise ValueError(f"{source_id}: no edited tensors at the predecessor")

    stage_root.mkdir(parents=True, exist_ok=True)
    stage_meta = stage_root / "stage.json"
    if stage_meta.exists():
        if json.loads(stage_meta.read_text())["source_zoo"] != str(canonical):
            raise ValueError(f"{stage_root} stages a different source zoo")
    else:
        stage_meta.write_text(json.dumps({"source_zoo": str(canonical)}, indent=1, sort_keys=True))
    header = _stage_header(canonical, edit)

    common = {
        "source_id": source_id,
        "predecessor_id": predecessor_id,
        "family": edit.get("family"),
        "seed": seed,
        "stored_cumulative_snapshot": cumulative,
        "edited_tensors": edited,
        "delta_convention": convention,
        "scale_source": scale_source,
        "edited_tensor_rule": "stored delta keys"
        if root
        else "nonzero realised increment at the predecessor",
        "predecessor_bytes_holder": predecessor_holder,
        "lineage": {
            "n_predecessor_changed": len(pred_changed),
            "n_predecessor_changed_carried_bitwise": len(pred_changed - set(edited)),
            "n_stored_tensors": len(keys),
        },
    }

    def write(kind, strength, states, meta, excluded, exclusions, post_write=None):
        """Stage one variant. Without ``post_write`` the registered staging check applies: every written
        tensor must reproduce the intended state under bf16(base + delta), or this raises. With it (F10
        rotations), the files are written, then ``post_write(entry_dir)`` reloads and audits them and
        returns the exclusions; an excluded variant's files are removed and only ``variant.json`` stays."""
        vid = variant_id(source_id, kind, strength, seed)
        d = stage_root / vid
        d.mkdir(exist_ok=True)
        for name in ("delta.safetensors", "increment.safetensors", "edit.json"):
            (d / name).unlink(missing_ok=True)
        rows = []
        if not excluded:
            delta = {k: (states[k].float() - base[k].float()) for k in edited}
            if not root:
                for k in sorted(pred_changed - set(edited)):
                    delta[k] = W[k].float() - base[k].float()
                    states = states | {k: W[k]}
                increment = {k: (states[k].float() - W[k].float()).contiguous() for k in edited}
                save_file(dict(sorted(increment.items())), str(d / "increment.safetensors"))
            if post_write is None:
                # the scorer's arithmetic must reproduce the intended state bitwise
                for k, x in delta.items():
                    if not torch.equal((base[k].float() + x).to(base[k].dtype), states[k]):
                        raise RuntimeError(
                            f"{vid}: {k} does not round-trip through bf16(base + delta)"
                        )
            save_file(
                {k: delta[k].contiguous() for k in sorted(delta)}, str(d / "delta.safetensors")
            )
            shutil.copyfile(entry_dir / "edit.json", d / "edit.json")
            if post_write is not None:
                exclusions = post_write(d)
                if exclusions:
                    excluded = True
                    for name in ("delta.safetensors", "increment.safetensors", "edit.json"):
                        (d / name).unlink(missing_ok=True)
        info = (
            common
            | meta
            | {"kind": kind, "strength": float(strength), "variant_id": vid, "excluded": excluded}
        )
        info["exclusions"] = exclusions
        if not excluded:
            info["delta_sha256"] = _sha256(d / "delta.safetensors")
            rows.append(
                {
                    "id": vid,
                    "family": edit.get("family"),
                    "eligibility": "production",
                    "parent_id": None if predecessor_id == "base" else predecessor_id,
                    "base_revision": edit["base_revision"],
                    "variant_of": source_id,
                    "variant_kind": kind,
                    "strength": float(strength),
                    "seed": seed,
                    "reuse_key": info["delta_sha256"],
                    "stored_delta_format": "dense" if root else "dense_cumulative",
                }
                | ({"predecessor_bytes_holder": predecessor_holder} if predecessor_holder else {})
            )
        strict = (
            info.get("audit_rule") == "f10"
        )  # F10 metadata never carries NaN (fails closed instead)
        if strict:
            info = rotation.json_safe(info)
        (d / "variant.json").write_text(
            json.dumps(info, indent=1, sort_keys=True, allow_nan=not strict)
        )
        _upsert_stage_rows(stage_root, header, rows, remove=() if rows else (vid,))
        return Variant(
            vid,
            kind,
            float(strength),
            seed,
            source_id,
            predecessor_id,
            d,
            excluded,
            exclusions,
            info.get("audit_rule"),
            info.get("realised_increment_ladder"),
        )

    # rotation at s = 1
    exclusions, blocks = [], {}
    for k in edited:
        if not isinstance(D[k], torch.Tensor):
            reason = "not_single_matrix"
        elif D[k].ndim != 2:
            reason = "non_2d"
        else:
            blocks[k] = rotation.changed_block(D[k])
            reason = rotation.exclusion_reason(D[k], *blocks[k])
        if reason:
            exclusions.append({"tensor": k, "reason": reason})
    audits, states, R, R_raw, a533 = {}, {}, {}, {}, {}
    meta = {
        "audit_rule": audit_rule,
        "tensor_audits": audits,
        "edit_audit": {"passes": False, "n_tensors": 0},
        "coverage": {
            "n_edited": len(edited),
            "n_rotated": 0,
            "n_excluded": len(exclusions),
            "fraction": 0.0,
        },
        "rotation_generator": "one torch.Generator per edit seeded with `seed`, shared across tensors in "
        "sorted-name order via rotation.rotate_block(shared_generator=...)",
    }
    post_write = None
    if not exclusions and audit_rule == "a533":
        generator = torch.Generator()
        generator.manual_seed(seed)
        for k in edited:
            rows_k, cols_k = blocks[k]
            R_raw_k = rotation.rotate_block(D[k], rows_k, cols_k, seed, shared_generator=generator)
            states[k], R[k] = rotation.apply_and_round(W[k], R_raw_k)
            audits[k] = _rotation_diagnostics(D[k], R_raw_k, R[k], W[k], rows_k, cols_k, generator)
        exclusions = [
            {"tensor": k, "reason": "audit_failed"} for k, a in audits.items() if not a["passes"]
        ]
        meta["edit_audit"] = _edit_audit(audits, D, R)
        meta["coverage"] = {
            "n_edited": len(edited),
            "n_rotated": len(audits) if not exclusions else 0,
            "n_excluded": len(exclusions),
            "fraction": (len(audits) / len(edited)) if not exclusions else 0.0,
        }
    elif not exclusions:
        generator = torch.Generator()
        generator.manual_seed(seed)
        with rotation.torch_threads(rotation.F10_THREADS):  # the staged construction thread count
            for k in edited:
                rows_k, cols_k = blocks[k]
                R_raw[k] = rotation.rotate_block(
                    D[k], rows_k, cols_k, seed, shared_generator=generator
                )
                states[k], R[k] = rotation.apply_and_round(W[k], R_raw[k])
        for k in edited:  # the complete registered (A5.3/A5.3.3) verdict, kept as a diagnostic
            a533[k] = _a533_diagnostic(D[k], R_raw[k], R[k], W[k], *blocks[k], generator)

        def post_write(entry):
            """F10.3 on the written bytes: reload, replay from a fresh seed, audit (i)-(v)."""
            delta_w = _load_written(entry / "delta.safetensors")
            inc_w = delta_w if root else _load_written(entry / "increment.safetensors")
            frames, extra, carried = {}, [], {}
            for k in edited:
                rows_k, cols_k = blocks[k]
                replayed = rotation.f10_replay(D[k], W[k], rows_k, cols_k, seed, frames)
                a = rotation.f10_tensor_audit(
                    D[k], W[k], R_raw[k], R[k], rows_k, cols_k, seed,
                    written=(inc_w or {}).get(k), scorer=(base[k], (delta_w or {}).get(k)),
                    replayed=replayed, raw_delta=raw[k] if root else None, frames=frames,
                )  # fmt: skip
                if isinstance(a, dict):  # anything else is a missing audit: it fails closed
                    a |= {"nonzero_d": int((D[k] != 0).sum()), "nonzero_r": int((R[k] != 0).sum()),
                          "logged_d": _moments(D[k], W[k]), "logged_r": _moments(R[k], W[k]),
                          "a533": a533[k]}  # fmt: skip
                    audits[k] = a
            for k in sorted(pred_changed - set(edited)):  # carried predecessor tensors, bitwise
                carried[k] = rotation.scorer_roundtrip(base[k], (delta_w or {}).get(k), W[k])
                if not carried[k]["passes"]:
                    extra.append(
                        {"tensor": k, "reason": "replay_failed", "detail": "carried_roundtrip"}
                    )
            want = {"delta.safetensors": set(edited) | pred_changed}
            if not root:
                want["increment.safetensors"] = set(edited)
            for name, f in (("delta.safetensors", delta_w), ("increment.safetensors", inc_w)):
                if name in want and f is not None and set(f) - want[name]:
                    extra.append(
                        {"tensor": None, "reason": "replay_failed", "detail": f"{name}:extra_keys"}
                    )
            audits_safe = rotation.json_safe(audits)
            audits.clear()
            audits.update(audits_safe)
            e = _f10_edit_audit(audits, edited, root, extra)
            meta["edit_audit"] = e
            meta["carried_roundtrip"] = carried
            meta["replay_environment"] = _replay_environment()
            meta["coverage"] = {
                "n_edited": len(edited),
                "n_rotated": len(edited) if e["passes"] else 0,
                "n_excluded": len({x["tensor"] for x in e["exclusions"]} & set(edited)),
                "fraction": 1.0 if e["passes"] else 0.0,
            }
            return e["exclusions"]

    if post_write is not None:
        variants = [write("rotation", 1.0, states, meta, False, [], post_write)]
    else:
        variants = [write("rotation", 1.0, states, meta, bool(exclusions), exclusions)]

    for s in strengths:
        states, stats = {}, {}
        for k in edited:
            states[k], X = rotation.scaled_variant(W[k], raw[k], s)
            target = s * raw[k]
            nonzero = target != 0
            stats[k] = {
                "realised_norm": _norm(X),
                "intended_norm": _norm(target),
                "fraction_nonzero_lost": float((nonzero & (X == 0)).sum())
                / max(int(nonzero.sum()), 1),
            }
        ladder = {"tensor_stats": stats, "realised_increment_ladder": not root}
        variants.append(write("scale", s, states, ladder, False, []))
    return variants


# -------------------------------------------------------------------------------------------- manifest


def _entry_files(d):
    return {
        n: _sha256(d / n)
        for n in ("delta.safetensors", "increment.safetensors")
        if (d / n).exists()
    }


def _variant_row(v):
    row = {
        "variant_id": v.variant_id,
        "kind": v.kind,
        "strength": v.strength,
        "seed": v.seed,
        "source_id": v.source_id,
        "predecessor_id": v.predecessor_id,
        "excluded": v.excluded,
        "exclusions": v.exclusions,
    }
    if v.kind == "rotation":
        row["audit_rule"] = v.audit_rule
    else:
        row["realised_increment_ladder"] = bool(v.realised_increment_ladder)
    return row


def _f10_subsets(variants):
    """F10.5(c), (e): edits whose rotation is in the pooled-norm (<= 2e-2) and noise-calibrated subsets."""
    out = {"pooled_norm_le_2e-2": [], "noise_calibrated": []}
    for v in sorted(variants, key=lambda v: v.source_id):
        if v.kind != "rotation" or v.audit_rule != "f10":
            continue
        e = json.loads((v.entry_dir / "variant.json").read_text())["edit_audit"]
        if e.get("in_subset_pooled_norm_2e-2"):
            out["pooled_norm_le_2e-2"].append(v.source_id)
        if e.get("in_subset_noise_calibrated"):
            out["noise_calibrated"].append(v.source_id)
    return out


def manifest_for(selections, variants, code_sha):
    """A5.11 pieces computable from the staged zoo (no paths, no timestamps: deterministic)."""
    edit_ids = sorted(s.edit_id for s in selections)
    predecessors = sorted(
        {s.predecessor_id for s in selections} | {v.predecessor_id for v in variants}
    )
    by_id = sorted(variants, key=lambda v: v.variant_id)
    rules = sorted({v.audit_rule for v in variants if v.kind == "rotation"})
    if len(rules) > 1:
        raise ValueError(f"rotation variants staged under different audit rules {rules}")
    rule = rules[0] if rules else None
    ladders = sorted(
        v.variant_id for v in by_id if v.kind == "scale" and v.realised_increment_ladder
    )
    return {
        "schema": "a5-part-a-launch-v1",
        "code_sha": code_sha,
        "audit_rule": rule,
        "realised_increment_ladder_variant_ids": ladders,
        "edit_ids": edit_ids,
        "component_ids": sorted(s.component_id for s in selections),
        "predecessor_ids": predecessors,
        "selections": [asdict(s) for s in selections],
        "variant_ids": [v.variant_id for v in by_id if not v.excluded],
        "excluded_variant_ids": [v.variant_id for v in by_id if v.excluded],
        "variants": [_variant_row(v) for v in by_id],
        "delta_sha256": {v.variant_id: _entry_files(v.entry_dir) for v in by_id if not v.excluded},
        "reference_ids_needed": {"predecessors": predecessors, "originals": edit_ids},
        "selected_without_variants": sorted(set(edit_ids) - {v.source_id for v in variants}),
        "variants_without_selection": sorted({v.source_id for v in variants} - set(edit_ids)),
        "minutes_per_entry": MINUTES_PER_ENTRY,
    } | ({"f10_subsets": _f10_subsets(variants)} if rule == "f10" else {})


# ---------------------------------------------------------------------------------------- scoring plan


def _load_config(config):
    if isinstance(config, dict):
        return config
    import yaml

    return yaml.safe_load(Path(config).read_bytes())


def _source_hashes(source):
    """The ``entry_hashes`` that wsp.eval.batch records for a zoo entry."""
    if (source / "delta.safetensors").exists():
        return {"delta.safetensors": _sha256(source / "delta.safetensors")}
    return {
        str(p.relative_to(source)): _sha256(p)
        for p in sorted((source / "lora").glob("*"))
        if p.is_file()
    }


def _reference_on_file(evals_dir, ref_id, source, identity):
    """Same scoring identity (A5.3.4): complete, same config hash, revision, counts and entry bytes."""
    path = Path(evals_dir) / ref_id / "eval.json"
    if not path.exists():
        return False
    p = json.loads(path.read_text())
    ok = (
        p.get("eval_status") == "complete"
        and p.get("config_hash") == identity["config_hash"]
        and p.get("model_revision") == identity["model_revision"]
        and all(p.get("counts", {}).get(k) == n for k, n in identity["counts"].items())
    )
    if ok and source is not None:
        hashes = p.get("entry_hashes") or {}
        ok = bool(hashes) and source.exists() and hashes == _source_hashes(source)
    return bool(ok)


def record_windows(records_path):
    """[(item id, token_ids_hash)] of a TF-v1 record's WikiText windows, in item order."""
    rows = []
    for line in Path(records_path).read_text().splitlines():
        if '"wikitext_ppl"' not in line:
            continue
        r = json.loads(line)
        if r.get("benchmark") == "wikitext_ppl":
            rows.append((str(r["item_id"]), (r.get("extra") or {}).get("token_ids_hash")))
    return sorted(rows, key=lambda x: int(x[0]))


def registered_windows(text_groups):
    """The registered 64 windows of ``text_groups.json`` in the :func:`record_windows` form."""
    doc = json.loads(Path(text_groups).read_text())
    return sorted(
        ((str(w["item_id"]), w["token_ids_hash"]) for w in doc["windows"]), key=lambda x: int(x[0])
    )


def reuse_proof(evals_dir, ref_id, source, identity, windows=None, holder=None):
    """A5 F3(i) reuse proof of one reference record. ``source`` is the zoo entry holding the bytes (the
    alias holder for an alias; None for the base). The verdict needs :func:`_reference_on_file`'s identity
    and entry-bytes checks and, with ``windows``, a primary record whose WikiText windows equal them."""
    d = Path(evals_dir) / ref_id
    out = {"id": ref_id, "bytes_holder": holder, "record": f"{ref_id}/eval.json"}
    if not (d / "eval.json").exists():
        return out | {"verdict": False, "failed_checks": ["no_record"]}
    p = json.loads((d / "eval.json").read_text())
    current = _source_hashes(source) if source is not None and source.exists() else None
    checks = {
        "complete": p.get("eval_status") == "complete",
        "config_hash": p.get("config_hash") == identity["config_hash"],
        "model_revision": p.get("model_revision") == identity["model_revision"],
        "counts": all(p.get("counts", {}).get(k) == n for k, n in identity["counts"].items()),
        "entry_hashes": source is None
        or (bool(p.get("entry_hashes")) and p.get("entry_hashes") == current),
    }
    if windows is not None:
        checks["primary"] = p.get("repeat_id") is None
        checks["windows"] = (d / "records.jsonl").exists() and record_windows(
            d / "records.jsonl"
        ) == list(windows)
    tool = _reference_on_file(evals_dir, ref_id, source, identity)
    core = all(v for k, v in checks.items() if k not in ("primary", "windows"))
    if tool != core:
        raise AssertionError(f"{ref_id}: _reference_on_file={tool} disagrees with checks {checks}")
    return out | {
        "verdict": all(checks.values()),
        "failed_checks": sorted(k for k, v in checks.items() if not v),
        "window_check": windows is not None,
        "run_id": p.get("run_id"),
        "config_hash": p.get("config_hash"),
        "model_revision": p.get("model_revision"),
        "counts": p.get("counts"),
        "recorded_entry_hashes": p.get("entry_hashes"),
        "current_entry_hashes": current,
        "eval_json_sha256": _sha256(d / "eval.json"),
    }


def _stage(staged_zoo, evals_dir, config, windows=None):
    from wsp.eval.batch import effective_config_hash

    staged_zoo = Path(staged_zoo)
    cfg = _load_config(config)
    source_zoo = Path(json.loads((staged_zoo / "stage.json").read_text())["source_zoo"])
    rows = [
        json.loads(x) for x in (staged_zoo / "manifest.jsonl").read_text().splitlines() if x.strip()
    ]
    variants = sorted((r for r in rows if "id" in r), key=lambda r: r["id"])
    counts = {k: v.get("n", v.get("n_windows")) for k, v in cfg["benchmarks"].items()}
    proofs = {}

    def on_file(ref_id, revision, holder=None):
        """Reuse needs the record to match the bytes actually held: an alias's holder's bytes."""
        key = (ref_id, revision)
        if key not in proofs:
            identity = {
                "config_hash": effective_config_hash(cfg),
                "model_revision": revision,
                "counts": counts,
            }
            source = None if ref_id == "base" else source_zoo / (holder or ref_id)
            proofs[key] = reuse_proof(evals_dir, ref_id, source, identity, windows, holder)
        elif proofs[key]["bytes_holder"] != holder:
            raise ValueError(f"{ref_id}: inconsistent bytes holders in the staged manifest")
        return proofs[key]["verdict"]

    on_file.proofs = proofs
    return variants, on_file


def _pred(r):
    return r.get("parent_id") or "base", r["base_revision"], r.get("predecessor_bytes_holder")


def plan_scoring(staged_zoo, evals_dir, config, windows=None):
    """[(variant id, needs_reference)] — whether the variant's predecessor reference lacks an on-file
    record with the same scoring identity (absent identity proof counts as needed)."""
    variants, on_file = _stage(staged_zoo, evals_dir, config, windows)
    return [(r["id"], not on_file(*_pred(r))) for r in variants]


def scoring_summary(staged_zoo, evals_dir, config, windows=None):
    """Plan plus A5.3.4 counts: n_P unique missing predecessors, n_D missing originals, priced minutes."""
    variants, on_file = _stage(staged_zoo, evals_dir, config, windows)
    plan = [(r["id"], not on_file(*_pred(r))) for r in variants]
    preds = sorted({_pred(r) for r in variants}, key=lambda x: (x[0], x[1]))
    origs = sorted({(r["variant_of"], r["base_revision"]) for r in variants})
    missing_p = sorted({p for p, rev, h in preds if not on_file(p, rev, h)})
    missing_d = sorted({o for o, rev in origs if not on_file(o, rev)})
    excluded = sorted(
        json.loads(p.read_text())["variant_id"]
        for p in Path(staged_zoo).glob("*/variant.json")
        if json.loads(p.read_text()).get("excluded")
    )
    n = len(variants) + len(missing_p) + len(missing_d)
    return {
        "plan": plan,
        "n_variants": len(variants),
        "excluded_variants": excluded,
        "predecessors": sorted({p for p, _, _ in preds}),
        "originals": sorted({o for o, _ in origs}),
        "predecessor_bytes_holders": {p: h for p, _, h in preds if h},
        "missing_predecessors": missing_p,
        "missing_originals": missing_d,
        "n_P": len(missing_p),
        "n_D": len(missing_d),
        "references_to_score": len(missing_p) + len(missing_d),
        "n_priced": n,
        "minutes_per_entry": MINUTES_PER_ENTRY,
        "minutes": round(n * MINUTES_PER_ENTRY, 4),
        "reuse_proofs": sorted(on_file.proofs.values(), key=lambda x: x["id"]),
        "window_check": windows is not None,
    }


def _batch_cmd(zoo, config, ids):
    return [
        sys.executable,
        "-m",
        "wsp.eval.batch",
        "--zoo",
        str(zoo),
        "--config",
        str(config),
        "--ids",
        ",".join(ids),
    ]


def score_commands(staged_zoo, config=DEFAULT_CONFIG, references=()):
    """The existing batch CLI on the staged variants (scoring is never reimplemented here), then, if
    ``references`` are given, on those missing reference entries of the source zoo. A reference without
    bytes of its own (an alias) cannot be scored under its id here: that stops for a repriced plan."""
    staged_zoo = Path(staged_zoo)
    rows = [
        json.loads(x) for x in (staged_zoo / "manifest.jsonl").read_text().splitlines() if x.strip()
    ]
    ids = sorted(r["id"] for r in rows if "id" in r)
    cmds = [_batch_cmd(staged_zoo, config, ids)]
    refs = sorted(set(references) - {"base"})
    if "base" in references:
        raise ValueError("the base reference is missing: score it separately and reprice")
    if refs:
        source_zoo = Path(json.loads((staged_zoo / "stage.json").read_text())["source_zoo"])
        no_bytes = [
            i
            for i in refs
            if not (source_zoo / i / "delta.safetensors").exists()
            and not (source_zoo / i / "lora").is_dir()
        ]
        if no_bytes:
            raise ValueError(f"missing references without own bytes (alias; reprice): {no_bytes}")
        cmds.append(_batch_cmd(source_zoo, config, refs))
    return cmds


# ------------------------------------------------------------------------------------------------- CLI


def load_feature_norms(path, rows):
    """id → effective incremental norm. ``path`` is a JSON object, a JSONL of ``{id, effective_norm}``, or a
    features directory of ``<id>/features.json``; there a norm counts only if it is relative to the actual
    predecessor (``delta_relative_to`` base for root edits), otherwise the value is None and blocks."""
    path = Path(path)
    if path.is_file():
        text = path.read_text()
        try:
            data = json.loads(text)
            return {k: v for k, v in data.items()}
        except json.JSONDecodeError:
            return {
                r["id"]: r.get("effective_norm") for r in map(json.loads, text.splitlines()) if r
            }
    norms = {}
    for row in rows:
        if "id" not in row or row.get("family") not in FAMILIES:
            continue
        f = path / row["id"] / "features.json"
        if not f.exists():
            continue
        meta = json.loads(f.read_text())
        rel, parent = meta.get("delta_relative_to"), row.get("parent_id")
        incremental = (rel == "base" and parent is None) or (
            rel in ("parent", "predecessor") and meta.get("parent_id") == parent
        )
        norms[row["id"]] = (meta.get("norms") or {}).get("fro_effective") if incremental else None
    return norms


def load_registered_selection(path, n_lora=30, n_chain=10):
    """The registered picks of a committed ``selection.json`` (``registered`` block), checked, never re-run.

    Returns (selections sorted by edit id, ledger). The ids must equal ``registered.edit_ids`` and, when
    present, the procedural freeze's sorted-id sha256; families, counts and components must be complete.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    reg = doc["registered"]
    if reg.get("status") != "selected":
        raise SelectionBlocked(f"{path}: registered selection status {reg.get('status')!r}")
    rows = sorted(reg["selected"], key=lambda r: r["edit_id"])
    sel = [
        Selection(
            r["edit_id"],
            r["component_id"],
            r["family"],
            int(r["tertile"]),
            int(r["rank"]),
            float(r["effective_norm"]),
            r.get("predecessor_id") or "base",
        )
        for r in rows
    ]
    ids = [s.edit_id for s in sel]
    problems = []
    if ids != sorted(reg["edit_ids"]) or len(set(ids)) != len(ids):
        problems.append("selected rows differ from registered.edit_ids")
    if len({s.component_id for s in sel}) != len(sel):
        problems.append("components not distinct")
    fams = {f: sum(s.family == f for s in sel) for f in FAMILIES}
    if fams != {"lora_sft": n_lora, "chain": n_chain} or len(sel) != n_lora + n_chain:
        problems.append(f"family counts {fams}")
    for r in rows:
        if (
            r.get("tertile_label")
            and ("low", "middle", "high")[int(r["tertile"])] != r["tertile_label"]
        ):
            problems.append(f"{r['edit_id']}: tertile label")
    freeze = doc.get("procedural_freeze") or {}
    ids_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if freeze.get("registered_edit_ids_sorted_sha256") not in (None, ids_sha):
        problems.append("sorted-id sha256 differs from the procedural freeze")
    if problems:
        raise SelectionBlocked(f"{path}: " + "; ".join(problems))
    ledger = {
        "source": "registered selection (selection.json:registered); selection not re-run",
        "selection_json_sha256": _sha256(path),
        "edit_ids_sorted_sha256": ids_sha,
        "status": reg["status"],
        "counts": reg.get("counts"),
        "quotas": reg.get("quotas"),
        "registered_exclusions": reg.get("registered_exclusions"),
        "ruling": reg.get("ruling"),
    }
    return sel, ledger


def load_bytes_holders(references, selections):
    """alias id -> bytes holder for the selected edits and their predecessors, from ``references.json``.

    Every selected edit must have a row whose predecessor equals the selection's; an id whose holder is
    itself is not an alias and is omitted."""
    doc = json.loads(Path(references).read_text())
    by_edit = {r["edit_id"]: r for r in doc["rows"]}
    holders = {}
    for s in selections:
        row = by_edit.get(s.edit_id)
        if row is None or (row.get("predecessor_id") or "base") != s.predecessor_id:
            raise ValueError(f"{s.edit_id}: references row missing or predecessor differs")
        for key in ("original", "predecessor"):
            ref = row.get(key) or {}
            h = ref.get("bytes_holder")
            if h and h != ref["id"]:
                if holders.get(ref["id"], h) != h:
                    raise ValueError(f"{ref['id']}: two bytes holders")
                holders[ref["id"]] = h
    return holders


AUDIT_FIELDS = (
    "block_shape",
    "block_rank",
    "rank_d",
    "rank_r",
    "rank_difference",
    "norm_mismatch",
    "spectrum_mismatch",
    "support_identical",
    "orthogonality_error",
    "reconstruction_residual",
    "prerounding_norm_mismatch",
    "rounding_error_rel",
    "nonzero_d",
    "nonzero_r",
    "s31_passes",
    "passes",
)
F10_AUDIT_FIELDS = (  # written in addition when the rotation was audited under F10
    "failed_checks",
    "reason",
    "fail_closed_detail",
    "checks",
    "construction_items",
    "construction_passes",
    "prerounding_spectrum_mismatch",
    "prerounding_rank_d",
    "prerounding_rank_r",
    "prerounding_rank_difference",
    "rank_cut_d",
    "rank_cut_r",
    "rank_window",
    "rank_n_near",
    "rank_budget_passes",
    "construction_residual",
    "nonzero_d_outside_block",
    "nonzero_r_raw_outside_block",
    "nonzero_r_eff_outside_block",
    "replay_passes",
    "replay_bits_equal",
    "replay_detail",
    "replay_threads",
    "prerounding_replay_bits_equal",
    "scorer_roundtrip_mismatches",
    "scorer_delta_bits_equal",
    "rounding_bound_passes",
    "rounding_bound_violations",
    "rounding_bound_nonfinite",
    "rounding_bound_max_ratio",
    "rounding_bound_max_excess_fp32_spacings",
    "n_double_rounding",
    "n_exceed_half_ulp_w",
    "norm_mismatch_signed",
    "kappa",
    "eps_hat",
    "sum_ulp2_block",
    "eps_D",
    "tau",
    "noise_rank_d",
    "noise_rank_r",
    "noise_calibrated",
    "noise_calibrated_passes",
    "d_norm",
    "r_norm",
    "r_eff_norm",
    "energy_ratio",
    "nonfinite_fields",
)


def _a533_row(a):
    """The complete registered verdict of an F10-audited tensor, flattened (diagnostic, never a reason)."""
    if not isinstance(a, dict):
        return {}
    out = {f"a533_{k}": a.get(k) for k in A533_FIELDS if k in a}
    return out | {"a533_criteria": a.get("criteria"), "a533_error": a.get("error")}


def audit_rows(variants, shapes):
    """One compact row per edited tensor of every rotation variant (A5 F3(ii) per-tensor output), sorted."""
    out = []
    for v in sorted(variants, key=lambda v: v.source_id):
        if v.kind != "rotation":
            continue
        meta = json.loads((v.entry_dir / "variant.json").read_text())
        reasons = {e["tensor"]: e["reason"] for e in meta["exclusions"]}
        for t in meta["edited_tensors"]:
            a = meta["tensor_audits"].get(t, {})
            out.append(
                {
                    "edit_id": v.source_id,
                    "predecessor_id": v.predecessor_id,
                    "variant_id": v.variant_id,
                    "seed": v.seed,
                    "tensor": t,
                    "shape": list(shapes[t]),
                    "audit_rule": meta.get("audit_rule", "a533"),
                    "verdict": reasons.get(t, "pass" if a.get("passes") else "not_audited"),
                }
                | {k: a[k] for k in AUDIT_FIELDS + F10_AUDIT_FIELDS if k in a}
                | _a533_row(a.get("a533"))
            )
    return out


def edit_audit_summary(variants):
    """Per edit: rotation verdict, exclusions, coverage, edit-level audit, unrotated-tensor count."""
    out = {}
    for v in sorted(variants, key=lambda v: v.source_id):
        if v.kind != "rotation":
            continue
        meta = json.loads((v.entry_dir / "variant.json").read_text())
        cov = meta["coverage"]
        out[v.source_id] = {
            "rotation_variant_id": v.variant_id,
            "predecessor_id": v.predecessor_id,
            "predecessor_bytes_holder": meta.get("predecessor_bytes_holder"),
            "seed": v.seed,
            "pair_verdict": "excluded" if v.excluded else "pass",
            "audit_rule": meta.get("audit_rule", "a533"),
            "exclusions": v.exclusions,
            "coverage": cov,
            "unrotated_tensors": cov["n_edited"] - cov["n_rotated"],
            "edit_audit": meta["edit_audit"],
            "delta_convention": meta["delta_convention"],
            "lineage": meta.get("lineage"),
        }
    return out


def _du(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def prepare(args):
    if getattr(args, "rotation_only", False):
        try:
            return prepare_rotation_only(args)
        except RestageRefused as e:
            print(f"rotation-only re-stage refused: {e}", file=sys.stderr)
            return 2
    from wsp.eval.batch import REPO, provenance
    from wsp.features.extract import load_base_state
    from wsp.zoo.apply import load_edited_model

    zoo = Path(args.zoo)
    rows = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    try:
        if args.selection:
            selections, ledger = load_registered_selection(
                args.selection, args.n_lora, args.n_chain
            )
        else:
            if not (args.features and args.sidecar):
                raise SelectionBlocked("--features and --sidecar are required without --selection")
            sidecar = [
                json.loads(x) for x in Path(args.sidecar).read_text().splitlines() if x.strip()
            ]
            norms = load_feature_norms(args.features, rows)
            selections, ledger = select_with_ledger(rows, norms, sidecar, args.n_lora, args.n_chain)
    except (SelectionBlocked, SelectionUnresolved) as e:
        print(f"selection {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    holders = load_bytes_holders(args.references, selections) if args.references else {}
    if any(s.edit_id in holders for s in selections):
        raise ValueError("a selected edit without its own bytes is not supported")
    windows = registered_windows(args.text_groups) if args.text_groups else None
    header = next(r for r in reversed(rows) if r.get("type") == "header")
    revisions = {
        json.loads((zoo / s.edit_id / "edit.json").read_text())["base_revision"] for s in selections
    }
    if len(revisions) != 1:
        raise ValueError(f"selected edits span base revisions {sorted(revisions)}")
    base, _ = load_base_state(header["base_model"], revisions.pop())
    shapes = {k: tuple(v.shape) for k, v in base.items()}
    stage = Path(args.out) / zoo.name
    variants = []
    for s in selections:
        kw = {"stage_root": stage, "predecessor_id": s.predecessor_id}
        if s.predecessor_id == "base":
            pred = base
        else:
            holder = holders.get(s.predecessor_id)
            entry = zoo / (holder or s.predecessor_id)
            pred, kw["base_state"] = load_edited_model(base, entry), base
            kw["predecessor_holder"] = holder
        variants += build_variants(
            zoo / s.edit_id,
            pred,
            rotation_seed(s.edit_id),
            tuple(args.strengths),
            audit_rule=args.audit_rule,
            **kw,
        )
        print(
            f"staged {s.edit_id}: {[v.variant_id for v in variants[-1 - len(args.strengths) :]]}",
            flush=True,
        )
    origin = provenance()
    manifest = manifest_for(selections, variants, origin["git_sha"])
    evals = Path(args.evals) if args.evals else REPO / "data/evals" / zoo.name / "tf_v1"
    summary = scoring_summary(stage, evals, args.config, windows)
    audits = audit_rows(variants, shapes)
    out = Path(args.manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_path = out.with_name("rotation_audits.jsonl")
    audit_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in audits))
    per_edit = edit_audit_summary(variants)
    manifest |= {
        "provenance": origin,
        "selection_ledger": ledger,
        "scoring": summary | {"config": str(args.config), "evals_dir": str(evals)},
        "staged_zoo": str(stage),
        "rotation_seeds": {s.edit_id: rotation_seed(s.edit_id) for s in selections},
        "predecessor_bytes_holders": {
            s.predecessor_id: holders[s.predecessor_id]
            for s in selections
            if s.predecessor_id in holders
        },
        "edit_audits": per_edit,
        "rotation_audits": {
            "file": audit_path.name,
            "sha256": _sha256(audit_path),
            "n_tensors": len(audits),
            "n_pass": sum(r["verdict"] == "pass" for r in audits),
        },
        "coverage": _coverage(per_edit, selections),
        "variant_json_sha256": {
            v.variant_id: _sha256(v.entry_dir / "variant.json")
            for v in sorted(variants, key=lambda v: v.variant_id)
        },
        "disk_bytes": _du(stage),
        "scored": False,
    }
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("plan", "reuse_proofs")}, indent=1
        )
    )
    for vid, need in summary["plan"]:
        print(vid, "needs_reference" if need else "reference_on_file")
    return 0


class RestageRefused(RuntimeError):
    """The rotation-only re-stage found a precondition violated; nothing is regenerated."""


ROTATION_ONLY_CARRIED_SCORING = (
    "config",
    "evals_dir",
    "predecessors",
    "originals",
    "predecessor_bytes_holders",
    "missing_predecessors",
    "missing_originals",
    "n_P",
    "n_D",
    "references_to_score",
    "minutes_per_entry",
    "reuse_proofs",
    "window_check",
)


def _ladder_hashes(stage, staged):
    """Re-hash the staged ladders' files and ``variant.json`` (never rebuilt) against the staged manifest."""
    out = {}
    for v in staged["variants"]:
        if v["kind"] != "scale":
            continue
        vid, d = v["variant_id"], Path(stage) / v["variant_id"]
        files = _entry_files(d)
        vjson = _sha256(d / "variant.json") if (d / "variant.json").exists() else None
        out[vid] = {
            "files": files,
            "variant_json": vjson,
            "equal_staged": files == staged["delta_sha256"].get(vid)
            and vjson == staged["variant_json_sha256"].get(vid),
        }
    return out


def _rotation_only_scoring(staged_scoring, staged_variants, manifest):
    """The staged scoring summary with the rotation set replaced. Reference status, reuse proofs and missing
    references are carried from the staged manifest (they do not depend on the rotations and reading them
    again would open evaluation records); the plan and priced count are recomputed for the new variant set."""
    source = {v["variant_id"]: v["source_id"] for v in staged_variants}
    need = {source[vid]: n for vid, n in staged_scoring["plan"] if vid in source}
    new_source = {v["variant_id"]: v["source_id"] for v in manifest["variants"]}
    plan = [[vid, need[new_source[vid]]] for vid in manifest["variant_ids"]]
    n = len(plan) + staged_scoring["n_P"] + staged_scoring["n_D"]
    return {k: staged_scoring[k] for k in ROTATION_ONLY_CARRIED_SCORING} | {
        "plan": plan,
        "n_variants": len(plan),
        "excluded_variants": manifest["excluded_variant_ids"],
        "n_priced": n,
        "minutes": round(n * MINUTES_PER_ENTRY, 4),
        "carried_from_staged": list(ROTATION_ONLY_CARRIED_SCORING),
        "recomputed": ["plan", "n_variants", "excluded_variants", "n_priced", "minutes"],
    }


def prepare_rotation_only(args):
    """F10.7 rotation-only re-stage: rebuild only the rotation variants under the F10 rule, in the staged
    zoo, at 4 threads. The staged ladders are neither rebuilt nor written: their bytes and ``variant.json``
    are re-hashed against the staged manifest before and after (any difference refuses). The staged manifest
    and audit file are preserved byte for byte as ``manifest_a533.json`` and ``rotation_audits_a533.jsonl``
    beside ``--manifest-out`` (their hashes enter the new manifest), and the new Part A manifest records the
    F10 verdicts. Opens no evaluation record and runs no scorer: reference status and reuse proofs are
    carried from the staged manifest."""
    from wsp.eval.batch import provenance
    from wsp.features.extract import load_base_state
    from wsp.zoo.apply import load_edited_model

    if args.audit_rule != "f10":
        raise RestageRefused("--rotation-only re-stages under the F10 rule only")
    threads_env = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    if any(x != str(rotation.F10_THREADS) for x in threads_env.values()):
        raise RestageRefused(
            f"F10.3(iii) needs OMP_NUM_THREADS = MKL_NUM_THREADS = 4, got {threads_env}"
        )
    if not (args.staged_manifest and args.selection and args.references):
        raise RestageRefused(
            "--rotation-only needs --staged-manifest, --selection and --references"
        )
    staged_path = Path(args.staged_manifest)
    staged_bytes = staged_path.read_bytes()
    staged = json.loads(staged_bytes)
    if staged.get("scored") is not False:
        raise RestageRefused("the staged manifest is not an unscored staging")
    audits_bytes = None  # the staged audit file, or its preserved copy after a previous re-stage
    for name in (staged["rotation_audits"]["file"], "rotation_audits_a533.jsonl"):
        p = staged_path.with_name(name)
        data = p.read_bytes() if p.exists() else b""
        if hashlib.sha256(data).hexdigest() == staged["rotation_audits"]["sha256"]:
            audits_bytes = data
            break
    if audits_bytes is None:
        raise RestageRefused("no audit file beside the staged manifest matches its hash")
    zoo = Path(args.zoo)
    stage = Path(args.out) / zoo.name
    if stage.resolve() != Path(staged["staged_zoo"]).resolve():
        raise RestageRefused(f"{stage} is not the staged zoo {staged['staged_zoo']}")
    selections, ledger = load_registered_selection(args.selection, args.n_lora, args.n_chain)
    if sorted(s.edit_id for s in selections) != staged["edit_ids"]:
        raise RestageRefused("the registered selection differs from the staged edits")
    before = _ladder_hashes(stage, staged)
    if not before or not all(x["equal_staged"] for x in before.values()):
        raise RestageRefused(
            "staged ladder bytes differ from the staged manifest before the re-stage"
        )

    out = Path(args.manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preserved = {}
    for name, data in (("manifest_a533.json", staged_bytes),
                       ("rotation_audits_a533.jsonl", audits_bytes)):  # fmt: skip
        p = out.with_name(name)
        if p.exists() and p.read_bytes() != data:
            raise RestageRefused(f"{p} exists with other content; refusing to overwrite it")
        p.write_bytes(data)
        preserved[name] = {"file": name, "sha256": hashlib.sha256(data).hexdigest()}

    rows = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    header = next(r for r in reversed(rows) if r.get("type") == "header")
    holders = load_bytes_holders(args.references, selections)
    revisions = {
        json.loads((zoo / s.edit_id / "edit.json").read_text())["base_revision"] for s in selections
    }
    if len(revisions) != 1:
        raise RestageRefused(f"selected edits span base revisions {sorted(revisions)}")
    base, _ = load_base_state(header["base_model"], revisions.pop())
    shapes = {k: tuple(v.shape) for k, v in base.items()}
    rotations = []
    for s in selections:
        kw = {"stage_root": stage, "predecessor_id": s.predecessor_id, "audit_rule": "f10"}
        if s.predecessor_id == "base":
            pred = base
        else:
            holder = holders.get(s.predecessor_id)
            pred, kw["base_state"] = (
                load_edited_model(base, zoo / (holder or s.predecessor_id)),
                base,
            )
            kw["predecessor_holder"] = holder
        (v,) = build_variants(zoo / s.edit_id, pred, rotation_seed(s.edit_id), (), **kw)
        rotations.append(v)
        print(
            f"re-staged {s.edit_id}: {v.variant_id} {'excluded' if v.excluded else 'pass'}",
            flush=True,
        )
    after = _ladder_hashes(stage, staged)
    if after != before:
        raise RestageRefused("staged ladder bytes changed during the rotation-only re-stage")

    ladders = [
        Variant(
            v["variant_id"],
            "scale",
            v["strength"],
            v["seed"],
            v["source_id"],
            v["predecessor_id"],
            stage / v["variant_id"],
            v["excluded"],
            v["exclusions"],
            None,
            v["predecessor_id"] != "base",
        )
        for v in staged["variants"]
        if v["kind"] == "scale"
    ]
    origin = provenance()
    manifest = manifest_for(selections, rotations + ladders, origin["git_sha"])
    audits = audit_rows(rotations, shapes)
    audit_path = out.with_name("rotation_audits.jsonl")
    audit_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in audits))
    per_edit = edit_audit_summary(rotations)
    staged_rotations = {v["variant_id"] for v in staged["variants"] if v["kind"] == "rotation"}
    variant_json = {
        v.variant_id: _sha256(v.entry_dir / "variant.json")
        for v in sorted(rotations, key=lambda v: v.variant_id)
    } | {vid: x["variant_json"] for vid, x in after.items()}
    manifest |= {
        "provenance": origin,
        "selection_ledger": ledger,
        "scoring": _rotation_only_scoring(staged["scoring"], staged["variants"], manifest),
        "staged_zoo": staged["staged_zoo"],
        "rotation_seeds": {s.edit_id: rotation_seed(s.edit_id) for s in selections},
        "predecessor_bytes_holders": staged["predecessor_bytes_holders"],
        "edit_audits": per_edit,
        "rotation_audits": {
            "file": audit_path.name,
            "sha256": _sha256(audit_path),
            "n_tensors": len(audits),
            "n_pass": sum(r["verdict"] == "pass" for r in audits),
        },
        "coverage": _coverage(per_edit, selections),
        "variant_json_sha256": dict(sorted(variant_json.items())),
        "disk_bytes": _du(stage),
        "scored": False,
        "restage": {
            "mode": "rotation_only",
            "audit_rule": "f10",
            "preserved": preserved,
            "staged_code_sha": staged.get("code_sha"),
            "ladders_rehashed": {
                "n": len(after),
                "all_equal_staged": all(x["equal_staged"] for x in after.values()),
                "by_variant": after,
            },
            "staged_pass_bitwise": {
                vid: {
                    "staged": staged["delta_sha256"][vid],
                    "restaged": manifest["delta_sha256"].get(vid),
                    "equal": staged["delta_sha256"][vid] == manifest["delta_sha256"].get(vid),
                }
                for vid in sorted(staged_rotations & set(staged["delta_sha256"]))
            },
            "thread_env": threads_env,
            "replay_environment": _replay_environment(),
            "evaluation_reads": "none: no evaluation record opened, no scorer run; reference status "
            "and reuse proofs carried from manifest_a533.json",
        },
    }
    out.write_text(
        json.dumps(rotation.json_safe(manifest), indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    c = manifest["coverage"]
    print(
        f"rotation pairs: {c['n_rotation_pairs_staged']} pass, {c['n_rotation_pairs_excluded']} excluded"
    )
    return 0


def _coverage(per_edit, selections):
    return {
        "n_edits": len(per_edit),
        "n_rotation_pairs_staged": sum(e["pair_verdict"] == "pass" for e in per_edit.values()),
        "n_rotation_pairs_excluded": sum(
            e["pair_verdict"] == "excluded" for e in per_edit.values()
        ),
        "n_tensors_edited": sum(e["coverage"]["n_edited"] for e in per_edit.values()),
        "n_tensors_rotated": sum(e["coverage"]["n_rotated"] for e in per_edit.values()),
        "by_family": {
            f: {
                "edits": sum(s.family == f for s in selections),
                "pairs_excluded": sum(
                    per_edit[s.edit_id]["pair_verdict"] == "excluded"
                    for s in selections
                    if s.family == f
                ),
            }
            for f in FAMILIES
        },
    }


def score(args):
    refs = ()
    if args.manifest:
        sc = json.loads(Path(args.manifest).read_text())["scoring"]
        refs = tuple(sc["missing_originals"]) + tuple(sc["missing_predecessors"])
    cmds = score_commands(args.stage, args.config, refs)
    for cmd in cmds:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m wsp.eval.response_intervention", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "prepare", help="select, build variants, write manifest, print the scoring plan"
    )
    p.add_argument("--zoo", required=True, help="canonical zoo (read only)")
    p.add_argument("--features", help="features dir, JSON {id: norm} or JSONL")
    p.add_argument("--sidecar", help="JSONL {id, component_id, closure_certified, ...}")
    p.add_argument(
        "--selection", help="registered selection.json: stage its picks (selection not re-run)"
    )
    p.add_argument("--references", help="references.json: bytes holders of alias predecessors")
    p.add_argument(
        "--text-groups", help="text_groups.json: add primary/window checks to the reuse proofs"
    )
    p.add_argument("--out", required=True, help="staging root; variants go to OUT/<zoo name>")
    p.add_argument("--manifest-out", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--evals", help="TF-v1 evals dir (default data/evals/<zoo name>/tf_v1)")
    p.add_argument("--n-lora", type=int, default=30)
    p.add_argument("--n-chain", type=int, default=10)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5])
    p.add_argument(
        "--audit-rule",
        choices=AUDIT_RULES,
        default=DEFAULT_AUDIT_RULE,
        help="rotation exclusion audit: a533 (registered realised audit) or f10 (F10.3, remedy d)",
    )
    p.add_argument(
        "--rotation-only",
        action="store_true",
        help="F10.7: re-stage only the rotations under F10 in the staged zoo; ladders re-hashed, not "
        "rebuilt; the staged manifest is kept as manifest_a533.json; no evaluation reads, no scoring",
    )
    p.add_argument("--staged-manifest", help="--rotation-only: the staged Part A manifest.json")
    s = sub.add_parser("score", help="run wsp.eval.batch on the staged zoo (inside agent-tmux)")
    s.add_argument("--stage", required=True, help="staged zoo directory (OUT/<zoo name>)")
    s.add_argument("--config", default=DEFAULT_CONFIG)
    s.add_argument(
        "--manifest", help="prepare's manifest: also score its missing references (source zoo)"
    )
    s.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return {"prepare": prepare, "score": score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
