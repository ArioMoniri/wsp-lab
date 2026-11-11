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
   ``prepare --supplement`` (F11) stages only a frozen list of supplementary root edits (rotation and the
   two ladders, F10 rule, registered seeds) into the staged zoo, re-hashes the staged edits' files before
   and after, carries their manifest entries byte for byte, keeps the staged manifest as
   ``manifest_pre_f11.json`` and merges both sets into one manifest (:func:`prepare_supplement`). The list
   is the frozen ``selection_f11.json`` (:func:`load_supplement_selection`; the F11.2 rule is recomputed
   from its hashed inputs by :func:`f11_rule`). By default it reads no evaluation record: the
   supplementary originals' reuse proofs (:func:`supplement_reference_proofs`) run in ``score
   --manifest`` (:func:`score_reference_proofs`), or at staging only with ``--reference-proofs``. ``slot-mask`` writes the
   ``a5-slot-mask-v1`` mask of a Part A manifest from its pair verdicts (:func:`slot_mask`).
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
    python -m wsp.eval.response_intervention prepare --supplement selection_f11.json \
        --staged-manifest M_staged --zoo Z --selection selection.json --text-groups text_groups.json \
        --out O --manifest-out M [--reference-proofs]
    python -m wsp.eval.response_intervention slot-mask --manifest M --out slot_mask.json
    python -m wsp.eval.response_intervention score --stage O/<zoo name> --manifest M \
        --text-groups text_groups.json [--evals E] --proofs-only
    python -m wsp.eval.response_intervention score --stage O/<zoo name> [--manifest M] \
        [--max-minutes T] [--require-proofs | --no-require-proofs] [--dry-run]

``--selection`` stages the registered picks of a committed ``selection.json`` instead of re-running the
selection; ``--references`` names the bytes holder of every alias-only predecessor (its state is loaded from
the holder's stored bytes and the scoring plan checks the holder's bytes against the alias's record);
``--text-groups`` adds the primary-record and 64-window checks of the A5 F3(i) reuse proof to every reused
reference. ``score --manifest`` also scores the references the plan lists as missing, on the source zoo.
Scoring runs in two phases: ``score --proofs-only`` runs the deferred proof step of the supplementary
originals, writes ``reference_proofs_f11.json`` beside the manifest, prints the repriced count and stops;
the scoring phase then requires that record (``--require-proofs``, the default; :func:`load_reference_proofs`)
and refuses above a stop-and-reprice threshold (``--max-minutes``). ``--no-require-proofs`` keeps the
single call that proves and then scores.

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
import time
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
    source_bytes_holder=None,
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
    ``source_bytes_holder`` does the same for the edit itself (F11: a supplementary root whose canonical
    entry is an alias holding only ``edit.json``): its stored delta is read from the holder's entry in the
    same zoo, everything else (``edit.json``, id, seed, variant ids) is the alias's, and ``variant.json``
    records the holder. Without it nothing in ``variant.json`` changes.
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

    if source_bytes_holder is None:
        stored = load_entry_delta(entry_dir, base)
    else:
        if (entry_dir / "delta.safetensors").exists() or (entry_dir / "lora").exists():
            raise ValueError(f"{source_id}: has bytes of its own but a bytes holder was given")
        stored = load_entry_delta(canonical / source_bytes_holder, base)
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
    if source_bytes_holder is not None:
        common["source_bytes_holder"] = source_bytes_holder

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
    if getattr(args, "supplement", None):
        try:
            return prepare_supplement(args)
        except SupplementRefused as e:
            print(f"supplementary staging refused: {e}", file=sys.stderr)
            return 2
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


# ------------------------------------------------------------------------- F11 supplementary staging


class SupplementRefused(RuntimeError):
    """A precondition of the F11 supplementary staging is violated; the staged forty stay untouched."""


SUPPLEMENT_AMENDMENT = "F11"
SUPPLEMENT_SEED_RULE = (
    "rotation_seed(canonical edit id) of control_seeds.json:rotation (A5.3.3): SHA256 of the UTF-8 bytes "
    "'20261107:rotation:' + id, first eight bytes big-endian, mod 2^63; the registered rule of the forty, "
    "keyed by the supplementary id (no new domain, counter or offset); one torch generator per edit, "
    "shared across its tensors in sorted-name order; the ladders carry the same seed"
)
SUPPLEMENT_CARRIED_KEYS = ("delta_sha256", "variant_json_sha256", "edit_audits", "rotation_seeds")
SUPPLEMENT_CARRIED_SCORING = (
    "config",
    "evals_dir",
    "predecessor_bytes_holders",
    "missing_predecessors",
    "n_P",
    "minutes_per_entry",
    "window_check",
)
SLOT_MASK_SCHEMA = "a5-slot-mask-v1"


def _cell_key(family, tertile):
    return f"{family}:{int(tertile)}"


F11_SELECTION_SCHEMA = "a5-f11-supplement-v1"
F11_SUPPLEMENT_IDS_SHA256 = "1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9"
F11_FROZEN_CELLS = ("lora_sft:2", "chain:0", "chain:2")
F11_PER_CELL = 4
F11_SELECTION_KEYS = frozenset(
    {
        "schema", "k", "per_cell", "rule", "order_key", "frozen_cells", "cells", "supplements",
        "supplement_ids", "supplement_ids_sorted", "supplement_ids_sorted_sha256", "ids_sha256_rule",
        "registered_edit_ids_sorted", "registered_rotation_exclusions", "replaces_registered",
        "no_top_up", "inputs", "outcomes_read", "basis",
    }
)  # fmt: skip
F11_INPUT_KEYS = (
    "selection",
    "components",
    "fields_manifest",
    "registered_donors",
    "part_a_manifest",
)
F11_CELL_KEYS = (
    "cell",
    "family",
    "tertile",
    "quota",
    "registered_picks",
    "registered_rotation_excluded",
    "candidate_order",
    "selected",
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _ids_sha256(ids):
    """F7.1 convention: sha256 of ``json.dumps(sorted(ids))`` (Python default separators)."""
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def f11_depleted_cells(manifest):
    """{"family:tertile": sorted excluded rotation source ids} of a Part A manifest (variants of kind
    ``rotation`` with ``excluded`` true), as ``scripts/a5_f11_supplement.py:depleted_cells``."""
    cell_of = {s["edit_id"]: _cell_key(s["family"], s["tertile"]) for s in manifest["selections"]}
    out = {}
    for v in manifest["variants"]:
        if v.get("kind") == "rotation" and v.get("excluded"):
            out.setdefault(cell_of[v["source_id"]], []).append(v["source_id"])
    return {c: sorted(ids) for c, ids in sorted(out.items())}


def _f11_reasons(eid, rows, donors, pool, registered, used_components):
    """The F11.2 filters of one candidate (empty: it passes). A disagreement between components.jsonl and
    fields_manifest on root-ness or own bytes refuses (as the selection script stops)."""
    r = rows.get(eid)
    if r is None:
        raise SupplementRefused(f"{eid}: no components.jsonl row")
    reasons = []
    if not (
        r.get("eligibility") == "eligible"
        and r.get("closure_certified") is True
        and r.get("component_status") == "development"
    ):
        reasons.append("not_eligible")
    d = donors.get(eid)
    root = r.get("parent_id") is None
    if d is not None and bool(d.get("predecessor_is_base")) != root:
        raise SupplementRefused(f"{eid}: components.jsonl parent_id disagrees with fields_manifest")
    if not root:
        reasons.append("not_root")
    if not (
        d is not None
        and d.get("gradient_source") == "base_field"
        and not d.get("flags")
        and eid in pool
    ):
        reasons.append("not_exact_gradient")
    alias = (not r.get("bytes_present")) or r.get("bytes_via_alias") is not None
    if d is not None and bool(d.get("donor_via_alias")) != alias:
        raise SupplementRefused(f"{eid}: components.jsonl own bytes disagree with fields_manifest")
    if alias:
        reasons.append("alias_only")
    if eid in registered:
        reasons.append("registered_forty")
    if r.get("component_id") in used_components:
        reasons.append("component_already_represented")
    return reasons


def f11_rule(registered_doc, rows, donors, pool, depleted, per_cell=F11_PER_CELL):
    """The registered F11.2 selection, recomputed (PREREG_H1_TF F11.2; scripts/a5_f11_supplement.py).

    For each depleted cell (``depleted``: {"family:tertile": excluded rotation ids}, in
    :data:`F11_FROZEN_CELLS` order), the candidates are ``sorted(cell)[quota:]`` of the registered ledger
    cell (``sorted(cell)[:quota]`` must be the registered picks of that cell), and the first ``per_cell``
    that are eligible development rows with certified closure, roots, exact-gradient donors (fields_manifest
    ``gradient_source`` base_field, no flags, in ``pool`` = power/registered_donors.json), not alias-only
    (``components.jsonl`` ``bytes_present`` true and ``bytes_via_alias`` null; ``donor_via_alias`` false),
    not among the registered forty and of a component not already represented, are selected; later
    candidates are ``reserve_unreached``. A shortage refuses (no cross-cell replacement or top-up).
    ``rows``: components.jsonl rows by id; ``donors``: fields_manifest donors by id. Returns the cell
    records in the ``selection_f11.json:cells`` shape (:data:`F11_CELL_KEYS`)."""
    fams = registered_doc["registered"]["ledger"]["families"]
    reg = set(registered_doc["procedural_freeze"]["registered_edit_ids"])
    used = {rows[i]["component_id"] for i in reg if i in rows}
    out = []
    for name in F11_FROZEN_CELLS:
        family, t = name.split(":")
        t = int(t)
        fam = fams[family]
        cell, quota = sorted(fam["cells"][t]), fam["quotas"][t]
        picks = cell[:quota]
        if picks != sorted(i for i in cell if i in reg):
            raise SupplementRefused(
                f"{name}: registered picks are not the sorted prefix of the cell"
            )
        listed, chosen = [], []
        for pos, eid in enumerate(cell[quota:], start=quota):
            if len(chosen) == per_cell:
                listed.append({"id": eid, "position": pos, "status": "reserve_unreached"})
                continue
            reasons = _f11_reasons(eid, rows, donors, pool, reg, used)
            listed.append({"id": eid, "position": pos, "reasons": reasons,
                           "status": "skipped" if reasons else "selected"})  # fmt: skip
            if not reasons:
                chosen.append(eid)
                used.add(rows[eid]["component_id"])
        if len(chosen) < per_cell:
            raise SupplementRefused(
                f"{name}: only {len(chosen)} of {per_cell} candidates pass F11.2 (no replacement)"
            )
        out.append({"cell": name, "family": family, "tertile": t, "quota": quota,
                    "registered_picks": picks, "registered_rotation_excluded": depleted.get(name),
                    "candidate_order": listed, "selected": chosen})  # fmt: skip
    return out


def load_supplement_selection(
    path, registered_path, staged_bytes, root=REPO_ROOT, expected_ids_sha256=None
):
    """The frozen F11 supplementary list (``selection_f11.json``, schema ``a5-f11-supplement-v1`` as written
    by ``scripts/a5_f11_supplement.py``), checked and never re-selected.

    Refuses unless: the file has exactly the frozen keys (:data:`F11_SELECTION_KEYS`); its recomputed
    ``sha256(json.dumps(sorted(supplement_ids)))`` equals both its recorded ``supplement_ids_sorted_sha256``
    and ``expected_ids_sha256`` (default :data:`F11_SUPPLEMENT_IDS_SHA256`, the Board-A-accepted list);
    ``k`` = 12, four per frozen cell, no top-up, no registered edit replaced; the registered selection
    (``registered_path``) and the staged manifest (``staged_bytes``) are the bytes its ``inputs`` record,
    and its other inputs (components.jsonl, fields_manifest, registered_donors, paths relative to ``root``)
    re-hash to their recorded sha256; its registered ids and rotation exclusions equal the staged
    manifest's; and :func:`f11_rule`, recomputed on those inputs, gives exactly its ``cells`` (every
    candidate's position, status and reasons), ``supplement_ids`` and ``supplements``. Every
    supplementary id is then a root with its own bytes, outside the staged edits and components.
    Returns (selections sorted by id, {"ids", "ids_in_order", "cells", "ids_sha256"})."""
    try:
        doc = json.loads(Path(path).read_text())
        return _check_supplement(
            doc,
            Path(registered_path).read_bytes(),
            staged_bytes,
            Path(root),
            expected_ids_sha256 or F11_SUPPLEMENT_IDS_SHA256,
        )
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
        raise SupplementRefused(f"supplement list malformed: {type(e).__name__}: {e}") from e


def _check_supplement(doc, registered_bytes, staged_bytes, root, expected_sha):
    if not isinstance(doc, dict) or set(doc) != F11_SELECTION_KEYS:
        got = set(doc) if isinstance(doc, dict) else set()
        raise ValueError(
            f"not the frozen selection_f11.json format: missing {sorted(F11_SELECTION_KEYS - got)}, "
            f"unexpected {sorted(got - F11_SELECTION_KEYS)}"
        )
    problems = []
    ids = list(doc["supplement_ids"])
    got_sha = _ids_sha256(ids)
    if doc["schema"] != F11_SELECTION_SCHEMA:
        problems.append(f"schema {doc['schema']!r}")
    if len(set(ids)) != len(ids) or doc["supplement_ids_sorted"] != sorted(ids):
        problems.append(
            "supplement_ids duplicated or supplement_ids_sorted is not their sorted list"
        )
    if doc["supplement_ids_sorted_sha256"] != got_sha:
        problems.append(
            f"recorded ids sha256 {doc['supplement_ids_sorted_sha256']} != recomputed {got_sha}"
        )
    if got_sha != expected_sha:
        problems.append(f"ids sha256 {got_sha} is not the frozen list's {expected_sha}")
    n = F11_PER_CELL * len(F11_FROZEN_CELLS)
    if (doc["k"], doc["per_cell"], len(ids)) != (n, F11_PER_CELL, n):
        problems.append(
            f"k {doc['k']} / per_cell {doc['per_cell']} / {len(ids)} ids, expected {n} / 4"
        )
    if list(doc["frozen_cells"]) != list(F11_FROZEN_CELLS):
        problems.append(f"frozen cells {doc['frozen_cells']}")
    if doc["no_top_up"] is not True or doc["replaces_registered"] is not False:
        problems.append("no_top_up must be true and replaces_registered false")
    inputs = doc["inputs"]
    if sorted(inputs) != sorted(F11_INPUT_KEYS):
        problems.append(f"inputs {sorted(inputs)}")
    held = {"selection": registered_bytes, "part_a_manifest": staged_bytes}
    data = {}
    for k in F11_INPUT_KEYS:
        rec = inputs[k]
        if k in held:
            b = held[k]
        else:
            p = root / rec["path"]
            b = p.read_bytes() if p.is_file() else None
        if b is None or hashlib.sha256(b).hexdigest() != rec["sha256"]:
            what = (
                "--selection"
                if k == "selection"
                else "--staged-manifest"
                if k in held
                else rec["path"]
            )
            problems.append(f"input {k} ({what}) does not hash to the recorded {rec['sha256']}")
        data[k] = b
    if problems:
        raise SupplementRefused("supplement list refused: " + "; ".join(problems))

    registered = json.loads(data["selection"])
    staged = json.loads(data["part_a_manifest"])
    rows = {}
    for line in data["components"].decode().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    donors = json.loads(data["fields_manifest"])["donors"]
    pool = set(json.loads(data["registered_donors"])["donor_ids"])
    reg_ids = sorted(registered["procedural_freeze"]["registered_edit_ids"])
    if not (
        doc["registered_edit_ids_sorted"]
        == reg_ids
        == sorted(registered["registered"]["edit_ids"])
        == staged["edit_ids"]
    ):
        problems.append(
            "registered ids differ between the list, selection.json and the staged manifest"
        )
    depleted = f11_depleted_cells(staged)
    excl = doc["registered_rotation_exclusions"]
    if sorted(depleted) != sorted(F11_FROZEN_CELLS) or excl != {
        "by_cell": depleted,
        "n": sum(map(len, depleted.values())),
    }:
        problems.append(f"rotation exclusions {excl} differ from the staged manifest's {depleted}")
    if problems:
        raise SupplementRefused("supplement list refused: " + "; ".join(problems))

    rule = f11_rule(registered, rows, donors, pool, depleted)
    file_cells = {c["cell"]: c for c in doc["cells"]}
    if sorted(file_cells) != sorted(F11_FROZEN_CELLS) or len(doc["cells"]) != len(F11_FROZEN_CELLS):
        problems.append(f"cells {sorted(file_cells)}")
    for c in rule:
        fc = file_cells.get(c["cell"], {})
        bad = [k for k in F11_CELL_KEYS if fc.get(k) != c[k]]
        if bad:
            problems.append(
                f"{c['cell']}: {bad} differ from the recomputed F11.2 rule ({c['selected']})"
            )
    rule_ids = [i for c in rule for i in c["selected"]]
    if ids != rule_ids:
        problems.append(f"supplement_ids {ids} are not the F11.2 selection {rule_ids}")
    cell_of = {i: c["cell"] for c in rule for i in c["selected"]}
    supp_rows = {s["id"]: s for s in doc["supplements"]}
    if [s["id"] for s in doc["supplements"]] != ids:
        problems.append("supplements rows are not the supplement_ids in order")
    staged_ids, staged_comps = set(staged["edit_ids"]), set(staged["component_ids"])
    fams = registered["registered"]["ledger"]["families"]
    sels = []
    for i in ids:
        r, s = rows.get(i, {}), supp_rows.get(i, {})
        if (s.get("cell"), s.get("component_id"), s.get("effective_norm")) != (
            cell_of.get(i),
            r.get("component_id"),
            r.get("effective_norm"),
        ) or s.get("predecessor_id") != "base":
            problems.append(f"{i}: supplements row differs from components.jsonl / its cell")
        if r.get("parent_id") is not None or not r.get("bytes_present") or r.get("bytes_via_alias"):
            problems.append(f"{i}: not a root with its own bytes")
        if i in staged_ids or r.get("component_id") in staged_comps:
            problems.append(f"{i}: already a staged edit or component")
        if i not in cell_of:
            continue
        family, t = cell_of[i].split(":")
        sels.append(
            Selection(i, r["component_id"], family, int(t), fams[family]["ordered"].index(i),
                      float(r["effective_norm"]), "base")
        )  # fmt: skip
    if len({s.component_id for s in sels}) != len(sels):
        problems.append("supplementary components not distinct")
    if problems:
        raise SupplementRefused("supplement list refused: " + "; ".join(problems))
    cells = {c["cell"]: sorted(c["selected"]) for c in sorted(rule, key=lambda c: c["cell"])}
    info = {"ids": sorted(ids), "ids_in_order": ids, "cells": cells, "ids_sha256": got_sha}
    return sorted(sels, key=lambda s: s.edit_id), info


def _stage_rows(stage):
    path = Path(stage) / "manifest.jsonl"
    rows = (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )
    return {r["id"]: r for r in rows if "id" in r}


def rehash_staged(stage, staged):
    """Re-hash every staged file of a manifest's variants: delta/increment bytes against ``delta_sha256``
    (an excluded variant must hold none), each ``variant.json`` against ``variant_json_sha256``, and the
    staged-zoo ``manifest.jsonl`` row of every staged variant (present, ``reuse_key`` = delta sha256;
    absent for an excluded one). Returns a summary whose ``all_equal`` is the verdict."""
    stage = Path(stage)
    rows, differ, by_variant = _stage_rows(stage), [], {}
    for v in staged["variants"]:
        vid, d = v["variant_id"], stage / v["variant_id"]
        files = _entry_files(d)
        vjson = _sha256(d / "variant.json") if (d / "variant.json").exists() else None
        if files != staged["delta_sha256"].get(vid, {}):
            differ.append(f"{vid}:files")
        if vjson is None or vjson != staged["variant_json_sha256"].get(vid):
            differ.append(f"{vid}:variant.json")
        by_variant[vid] = {"files": files, "variant_json": vjson}
    for vid in staged["variant_ids"]:
        r = rows.get(vid)
        if r is None or r.get("reuse_key") != staged["delta_sha256"][vid].get("delta.safetensors"):
            differ.append(f"{vid}:manifest.jsonl")
    differ += [f"{vid}:manifest.jsonl" for vid in staged["excluded_variant_ids"] if vid in rows]
    carried_rows = [rows[v] for v in sorted(staged["variant_ids"]) if v in rows]
    return {
        "n_variants": len(staged["variants"]),
        "n_files": sum(len(x["files"]) + 1 for x in by_variant.values()),
        "all_equal": not differ,
        "differ": differ,
        "staged_rows_sha256": hashlib.sha256(
            json.dumps(carried_rows, sort_keys=True).encode()
        ).hexdigest(),
        "by_variant_sha256": hashlib.sha256(
            json.dumps(by_variant, sort_keys=True).encode()
        ).hexdigest(),
    }


def supplement_reference_proofs(
    evals_dir, config, zoo, ids, holders, revision, windows=None, log=None
):
    """EVALUATION-RECORD READS, isolated: the A5 F3(i) reuse proofs of the supplementary originals.

    The original staging computed every reference's proof with :func:`reuse_proof`, which opens
    ``<evals>/<id>/eval.json`` (status, config hash, model revision, counts, entry hashes, repeat id, run
    id) and, with ``windows``, ``<evals>/<id>/records.jsonl`` (the WikiText rows, of which it keeps item ids
    and token-id hashes). This computes the same proofs for the supplementary originals only, with an
    alias's bytes holder as the entry-bytes source; the base predecessor's proof is carried, not
    recomputed. Each path it opens or probes is logged through ``log`` and returned. By default (chair
    ruling, F11) it runs at scoring time, in ``score --manifest`` (:func:`score_reference_proofs`), not at
    staging: ``prepare --supplement`` calls it only with the explicit opt-in ``--reference-proofs``, and
    otherwise touches no evaluation path. Returns (proofs, sorted paths touched, identity)."""
    from wsp.eval.batch import effective_config_hash

    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    cfg = _load_config(config)
    identity = {
        "config_hash": effective_config_hash(cfg),
        "model_revision": revision,
        "counts": {k: v.get("n", v.get("n_windows")) for k, v in cfg["benchmarks"].items()},
    }
    proofs, touched = [], []
    for i in ids:
        paths = [Path(evals_dir) / i / "eval.json"]
        if windows is not None:
            paths.append(Path(evals_dir) / i / "records.jsonl")
        for p in paths:
            log(f"reference proof (evaluation record) {p}")
            touched.append(str(p))
        holder = holders.get(i)
        proofs.append(
            reuse_proof(evals_dir, i, Path(zoo) / (holder or i), identity, windows, holder)
        )
    return proofs, sorted(touched), identity


def _supplement_scoring(staged_scoring, merged, new_ids, proofs):
    """The pre-F11 scoring summary extended to the merged set. The forty's plan entries, missing references
    and reuse proofs are carried; the supplementary variants' plan entries use the carried proof of their
    predecessor (base); their originals use ``proofs`` (None, the default: not computed at staging, so each
    is listed in ``unproven_originals`` and priced as an entry "to be proven or scored", the plan's "absent
    identity proof counts as needed"; ``score --manifest`` proves them first and refuses to score while
    any stays unproven)."""
    carried = {p["id"]: p for p in staged_scoring["reuse_proofs"]}
    if "base" not in carried:
        raise SupplementRefused("the staged scoring carries no reuse proof of the base predecessor")
    need = {vid: n for vid, n in staged_scoring["plan"]}
    plan = [
        [vid, need[vid] if vid in need else not carried["base"]["verdict"]]
        for vid in merged["variant_ids"]
    ]
    new_proofs = {p["id"]: p for p in proofs or []}
    unproven = sorted(i for i in new_ids if i not in new_proofs)
    missing_o = sorted(
        set(staged_scoring["missing_originals"])
        | {i for i, p in new_proofs.items() if not p["verdict"]}
    )
    n_P, n_D = staged_scoring["n_P"], len(missing_o)
    n = len(plan) + n_P + n_D + len(unproven)
    return {k: staged_scoring[k] for k in SUPPLEMENT_CARRIED_SCORING} | {
        "plan": plan,
        "n_variants": len(plan),
        "excluded_variants": merged["excluded_variant_ids"],
        "predecessors": sorted(set(staged_scoring["predecessors"]) | {"base"}),
        "originals": sorted(set(staged_scoring["originals"]) | set(new_ids)),
        "missing_originals": missing_o,
        "unproven_originals": unproven,
        "n_D": n_D,
        "references_to_score": n_P + n_D,
        "n_priced": n,
        "minutes": round(n * MINUTES_PER_ENTRY, 4),
        "reuse_proofs": sorted([*carried.values(), *new_proofs.values()], key=lambda p: p["id"]),
        "carried_from_pre_f11": list(SUPPLEMENT_CARRIED_SCORING)
        + ["plan entries, missing_originals and reuse_proofs of the forty"],
        "recomputed": [
            "plan",
            "n_variants",
            "excluded_variants",
            "predecessors",
            "originals",
            "missing_originals",
            "unproven_originals",
            "n_D",
            "references_to_score",
            "n_priced",
            "minutes",
            "reuse_proofs",
        ],
        "priced_unproven_as_references": len(unproven),
        "originals_to_prove_or_score": unproven,
        "unproven_rule": "each unproven original is priced as one entry to be proven or scored; score "
        "--manifest runs the reuse proofs first (proven: reused; failed: scored as a reference) and "
        "refuses to score any variant while its original stays unproven",
    }


def _selection_from_row(r):
    return Selection(
        r["edit_id"], r["component_id"], r["family"], int(r["tertile"]), int(r["rank"]),
        float(r["effective_norm"]), r.get("predecessor_id") or "base",
    )  # fmt: skip


def merge_supplement(staged, selections, variants, holders, code_sha):
    """The merged Part A manifest body: every entry of ``staged`` carried unchanged, the supplementary
    edits added (``supplementary: true`` on their selection rows and edit audits), and the set-level fields
    (ids, variants, exclusions, subsets, coverage) recomputed over the union. Scoring, audit-file and
    supplement records are added by the caller."""
    new = manifest_for(selections, variants, code_sha)
    per_edit = edit_audit_summary(variants)
    for i, e in per_edit.items():
        e |= {"supplementary": True, "source_bytes_holder": holders.get(i)}
    sel_rows = sorted(staged["selections"] + [asdict(s) | {"supplementary": True} for s in selections],
                      key=lambda r: r["edit_id"])  # fmt: skip
    edit_ids = sorted(set(staged["edit_ids"]) | set(new["edit_ids"]))
    rows = sorted(staged["variants"] + new["variants"], key=lambda v: v["variant_id"])
    sources = {v["source_id"] for v in rows}
    preds = sorted(set(staged["predecessor_ids"]) | set(new["predecessor_ids"]))
    edit_audits = dict(sorted((staged["edit_audits"] | per_edit).items()))
    merged = dict(staged) | {
        "code_sha": code_sha,
        "realised_increment_ladder_variant_ids": sorted(
            set(staged["realised_increment_ladder_variant_ids"])
            | set(new["realised_increment_ladder_variant_ids"])
        ),
        "edit_ids": edit_ids,
        "component_ids": sorted(staged["component_ids"] + new["component_ids"]),
        "predecessor_ids": preds,
        "selections": sel_rows,
        "variant_ids": sorted(staged["variant_ids"] + new["variant_ids"]),
        "excluded_variant_ids": sorted(
            staged["excluded_variant_ids"] + new["excluded_variant_ids"]
        ),
        "variants": rows,
        "delta_sha256": dict(sorted((staged["delta_sha256"] | new["delta_sha256"]).items())),
        "reference_ids_needed": {"predecessors": preds, "originals": edit_ids},
        "selected_without_variants": sorted(set(edit_ids) - sources),
        "variants_without_selection": sorted(sources - set(edit_ids)),
        "f10_subsets": {
            k: sorted(set(staged["f10_subsets"][k]) | set(new["f10_subsets"][k]))
            for k in staged["f10_subsets"]
        },
        "rotation_seeds": dict(
            sorted(
                (
                    staged["rotation_seeds"]
                    | {s.edit_id: rotation_seed(s.edit_id) for s in selections}
                ).items()
            )
        ),
        "edit_audits": edit_audits,
        "coverage": _coverage(edit_audits, [_selection_from_row(r) for r in sel_rows]),
        "variant_json_sha256": dict(
            sorted(
                (
                    staged["variant_json_sha256"]
                    | {v.variant_id: _sha256(v.entry_dir / "variant.json") for v in variants}
                ).items()
            )
        ),
    }
    if len(set(merged["component_ids"])) != len(merged["component_ids"]):
        raise SupplementRefused("a supplementary component is already staged")
    if len(set(merged["variant_ids"])) != len(merged["variant_ids"]):
        raise SupplementRefused("a supplementary variant id collides with a staged one")
    return merged


def carried_identical(staged, merged):
    """Every per-entry record of the staged forty serialises identically in the merged manifest."""
    dump = lambda x: json.dumps(x, sort_keys=True)
    bad = [f"{k}:{i}" for k in SUPPLEMENT_CARRIED_KEYS for i, v in staged[k].items()
           if dump(merged[k].get(i)) != dump(v)]  # fmt: skip
    for k in ("selections", "variants"):
        have = {dump(r) for r in merged[k]}
        bad += [f"{k}:{n}" for n, r in enumerate(staged[k]) if dump(r) not in have]
    bad += [k for k in ("restage", "selection_ledger", "predecessor_bytes_holders", "staged_zoo")
            if dump(staged.get(k)) != dump(merged.get(k))]  # fmt: skip
    return bad


def prepare_supplement(args):
    """F11 supplementary staging: stage only the supplementary root edits (rotation at the base and the
    ladders s = 0.25, 0.5, under the F10 rule at 4 threads, registered seeds) into the staged zoo, and
    merge them into the staged Part A manifest.

    The staged forty are never rebuilt or rewritten: all their files are re-hashed against the staged
    manifest before anything is written and again after staging (any difference refuses), and every
    per-entry record of theirs is carried byte for byte (checked before writing). The staged manifest and
    audit file are kept beside ``--manifest-out`` as ``manifest_pre_f11.json`` and
    ``rotation_audits_pre_f11.jsonl``; the merged audit file is the pre-F11 file's bytes followed by the
    supplementary rows. Coverage, exclusions, subsets, the scoring plan and the priced count are
    recomputed over the union; the supplementary selection rows and edit audits carry ``supplementary:
    true``. By default no evaluation record is read: the supplementary originals are priced as entries to
    be proven or scored and their reuse proofs run in ``score --manifest``; ``--reference-proofs`` opts in
    to computing them here (:func:`supplement_reference_proofs`, every path logged)."""
    from wsp.eval.batch import provenance
    from wsp.features.extract import load_base_state

    if getattr(args, "rotation_only", False):
        raise SupplementRefused("--supplement and --rotation-only are exclusive")
    if args.audit_rule != "f10":
        raise SupplementRefused("--supplement stages under the F10 rule only")
    threads_env = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    if any(x != str(rotation.F10_THREADS) for x in threads_env.values()):
        raise SupplementRefused(
            f"F10.3(iii) needs OMP_NUM_THREADS = MKL_NUM_THREADS = 4, got {threads_env}"
        )
    if not (args.staged_manifest and args.selection):
        raise SupplementRefused("--supplement needs --staged-manifest and --selection")
    staged_path = Path(args.staged_manifest)
    staged_bytes = staged_path.read_bytes()
    staged = json.loads(staged_bytes)
    if staged.get("schema") != "a5-part-a-launch-v1" or staged.get("scored") is not False:
        raise SupplementRefused(
            "the staged manifest is not an unscored a5-part-a-launch-v1 staging"
        )
    if staged.get("audit_rule") != "f10":
        raise SupplementRefused("the staged manifest is not under the F10 rule")
    if "supplement" in staged:
        raise SupplementRefused("the staged manifest is already supplemented")
    ladder_strengths = sorted({v["strength"] for v in staged["variants"] if v["kind"] == "scale"})
    if sorted(float(s) for s in args.strengths) != ladder_strengths:
        raise SupplementRefused(f"--strengths must be the staged ladders {ladder_strengths}")
    audits_bytes = None
    for name in (staged["rotation_audits"]["file"], "rotation_audits_pre_f11.jsonl"):
        p = staged_path.with_name(name)
        data = p.read_bytes() if p.exists() else b""
        if hashlib.sha256(data).hexdigest() == staged["rotation_audits"]["sha256"]:
            audits_bytes = data
            break
    if audits_bytes is None:
        raise SupplementRefused("no audit file beside the staged manifest matches its hash")
    zoo = Path(args.zoo)
    stage = Path(args.out) / zoo.name
    if stage.resolve() != Path(staged["staged_zoo"]).resolve():
        raise SupplementRefused(f"{stage} is not the staged zoo {staged['staged_zoo']}")
    reg, _ = load_registered_selection(args.selection, args.n_lora, args.n_chain)
    if sorted(s.edit_id for s in reg) != staged["edit_ids"]:
        raise SupplementRefused("the registered selection differs from the staged edits")
    rows = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines() if x.strip()]
    supp_path = Path(args.supplement)
    selections, info = load_supplement_selection(supp_path, args.selection, staged_bytes)
    ids = [s.edit_id for s in selections]
    zoo_parent = {r["id"]: r.get("parent_id", "absent") for r in rows if "id" in r}
    not_root = [i for i in ids if zoo_parent.get(i, "absent") is not None]
    if not_root:
        raise SupplementRefused(f"supplementary ids not root entries of the zoo: {not_root}")
    # F11.2 admits no alias-only edit, so a valid list never needs a bytes holder for its own delta; the
    # holder path of build_variants stays as defensive code, and an entry without its own bytes refuses.
    holders = {}
    no_bytes = [i for i in ids if not ((zoo / i / "delta.safetensors").exists()
                                       or (zoo / i / "lora").is_dir())]  # fmt: skip
    if no_bytes:
        raise SupplementRefused(
            f"supplementary zoo entries without their own bytes (alias-only, excluded by F11.2): {no_bytes}"
        )
    proofs_wanted = bool(getattr(args, "reference_proofs", False))
    window_check = bool(staged["scoring"].get("window_check"))
    if proofs_wanted and window_check and not args.text_groups:
        raise SupplementRefused(
            "the staged reuse proofs carry the window check: give --text-groups"
        )
    new_vids = {
        variant_id(i, k, st, rotation_seed(i))
        for i in ids
        for k, st in [("rotation", 1.0)] + [("scale", s) for s in ladder_strengths]
    }
    if new_vids & {v["variant_id"] for v in staged["variants"]}:
        raise SupplementRefused("a supplementary variant id collides with a staged one")

    before = rehash_staged(stage, staged)
    if not before["all_equal"]:
        raise SupplementRefused(
            f"staged files of the forty differ from the staged manifest: {before['differ'][:8]}"
        )
    out = Path(args.manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preserved = {}
    for key, name, data in (("manifest", "manifest_pre_f11.json", staged_bytes),
                            ("rotation_audits", "rotation_audits_pre_f11.jsonl", audits_bytes)):  # fmt: skip
        p = out.with_name(name)
        if p.exists() and p.read_bytes() != data:
            raise SupplementRefused(f"{p} exists with other content; refusing to overwrite it")
        p.write_bytes(data)
        preserved[key] = {"file": name, "sha256": hashlib.sha256(data).hexdigest()}

    header = next(r for r in reversed(rows) if r.get("type") == "header")
    revisions = {json.loads((zoo / i / "edit.json").read_text())["base_revision"] for i in ids}
    if len(revisions) != 1:
        raise SupplementRefused(f"supplementary edits span base revisions {sorted(revisions)}")
    revision = revisions.pop()
    if revision != header.get("base_revision"):
        raise SupplementRefused("supplementary edits are not at the zoo's base revision")
    base, _ = load_base_state(header["base_model"], revision)
    shapes = {k: tuple(v.shape) for k, v in base.items()}
    variants = []
    for s in selections:
        t0 = time.monotonic()
        built = build_variants(
            zoo / s.edit_id,
            base,
            rotation_seed(s.edit_id),
            tuple(ladder_strengths),
            stage_root=stage,
            predecessor_id="base",
            audit_rule="f10",
            source_bytes_holder=holders.get(s.edit_id),
        )
        variants += built
        print(
            f"supplement staged {s.edit_id}: {[v.variant_id for v in built]} rotation "
            f"{'excluded' if built[0].excluded else 'pass'} ({time.monotonic() - t0:.1f} s)",
            flush=True,
        )
    after = rehash_staged(stage, staged)
    if after != before:
        raise SupplementRefused(
            "staged files of the forty changed during the supplementary staging"
        )

    proofs, touched, identity = None, [], None
    if proofs_wanted:
        windows = registered_windows(args.text_groups) if window_check else None
        proofs, touched, identity = supplement_reference_proofs(
            args.evals or staged["scoring"]["evals_dir"], args.config, zoo, ids, holders, revision,
            windows,
        )  # fmt: skip
        carried_hash = next(p for p in staged["scoring"]["reuse_proofs"] if p["id"] == "base").get(
            "config_hash"
        )
        if carried_hash not in (None, identity["config_hash"]):
            raise SupplementRefused(
                "--config gives another scoring identity than the staged proofs"
            )

    origin = provenance()
    merged = merge_supplement(staged, selections, variants, holders, origin["git_sha"])
    audits = audit_rows(variants, shapes)
    audit_path = out.with_name("rotation_audits.jsonl")
    audit_bytes = (
        audits_bytes + "".join(json.dumps(r, sort_keys=True) + "\n" for r in audits).encode()
    )
    merged["scoring"] = _supplement_scoring(staged["scoring"], merged, ids, proofs)
    bad = carried_identical(staged, merged)
    if bad:
        raise SupplementRefused(f"merged manifest does not carry the forty unchanged: {bad[:8]}")
    merged |= {
        "provenance": origin,
        "rotation_audits": {
            "file": audit_path.name,
            "sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "n_tensors": staged["rotation_audits"]["n_tensors"] + len(audits),
            "n_pass": staged["rotation_audits"]["n_pass"]
            + sum(r["verdict"] == "pass" for r in audits),
            "layout": "pre-F11 file bytes, then the supplementary rows (edit id order)",
        },
        "disk_bytes": _du(stage),
        "scored": False,
        "supplement": {
            "amendment": SUPPLEMENT_AMENDMENT,
            "selection_file": str(supp_path),
            "sha256": _sha256(supp_path),
            "selection_schema": F11_SELECTION_SCHEMA,
            "ids": ids,
            "ids_in_selection_order": info["ids_in_order"],
            "ids_sorted_sha256": info["ids_sha256"],
            "rule": "F11.2 recomputed from the selection's hashed inputs (f11_rule): equal",
            "cells": info["cells"],
            "n": len(ids),
            "source_bytes_holders": dict(sorted(holders.items())),
            "audit_rule": "f10",
            "strengths": ladder_strengths,
            "seed_rule": SUPPLEMENT_SEED_RULE,
            "pre_f11": preserved
            | {"code_sha": staged.get("code_sha"), "provenance": staged.get("provenance")},
            "carried": {
                "n_edits": len(staged["edit_ids"]),
                "rehash_before": {k: v for k, v in before.items() if k != "differ"},
                "rehash_after": {k: v for k, v in after.items() if k != "differ"},
                "entries_byte_identical": True,
                "keys": list(SUPPLEMENT_CARRIED_KEYS)
                + ["selections", "variants", "restage", "selection_ledger"],
            },
            "reference_proofs": {
                "computed": proofs_wanted,
                "function": "supplement_reference_proofs",
                "identity": identity,
                "paths_touched": touched,
                "carried": ["base (predecessor of every supplementary edit)"],
                "deferred_to": None
                if proofs_wanted
                else "score --manifest (score_reference_proofs)",
            },
            "thread_env": threads_env,
            "replay_environment": _replay_environment(),
            "evaluation_reads": "reuse proofs of the supplementary originals only (--reference-proofs; "
            "paths_touched)"
            if proofs_wanted
            else "none: no evaluation record opened, no scorer run (default; the supplementary "
            "originals' reuse proofs are established at scoring time by score --manifest)",
        },
    }
    audit_path.write_bytes(audit_bytes)
    out.write_text(
        json.dumps(rotation.json_safe(merged), indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    c, sc = merged["coverage"], merged["scoring"]
    print(
        f"merged {len(merged['edit_ids'])} edits: rotation pairs {c['n_rotation_pairs_staged']} pass, "
        f"{c['n_rotation_pairs_excluded']} excluded; priced {sc['n_priced']} entries "
        f"({sc['minutes']} min); unproven originals {len(sc['unproven_originals'])}"
    )
    return 0


def slot_mask(manifest):
    """The slot mask (schema ``a5-slot-mask-v1``: ``schema`` and ``available`` only) of a Part A manifest:
    one entry per ``edit_ids`` id listing ``rotation`` iff its pair verdict is pass, then ``ladder_<s>`` for
    each of its staged scale variants in strength order. The pair verdict must agree with the rotation
    variant's ``excluded`` flag. Used for the F10.6 mask (forty ids) and the F11 mask (fifty-two)."""
    rot, ladders = {}, {i: [] for i in manifest["edit_ids"]}
    for v in manifest["variants"]:
        if v["kind"] == "rotation":
            rot[v["source_id"]] = not v["excluded"]
        elif not v["excluded"]:
            ladders[v["source_id"]].append(float(v["strength"]))
    available = {}
    for i in manifest["edit_ids"]:
        verdict = manifest["edit_audits"][i]["pair_verdict"]
        if (verdict == "pass") is not rot.get(i):
            raise ValueError(f"{i}: pair verdict {verdict!r} disagrees with its rotation variant")
        available[i] = (["rotation"] if verdict == "pass" else []) + [
            f"ladder_{s!r}" for s in sorted(ladders[i])
        ]
    return {"schema": SLOT_MASK_SCHEMA, "available": available}


def slot_mask_text(mask):
    """The committed serialisation of a slot mask (as ``part_a/slot_mask_f10.json``)."""
    return json.dumps(mask, indent=1, sort_keys=True)


def write_slot_mask(args):
    mask = slot_mask(json.loads(Path(args.manifest).read_text()))
    text = slot_mask_text(mask)
    Path(args.out).write_text(text)
    masked = sorted(i for i, a in mask["available"].items() if "rotation" not in a)
    print(
        json.dumps(
            {
                "mask": str(args.out),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "n_ids": len(mask["available"]),
                "rotation_masked": masked,
            },
            indent=1,
        )
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


REFERENCE_PROOFS_SCHEMA = "a5-f11-reference-proofs-v1"
REFERENCE_PROOFS_FILE = "reference_proofs_f11.json"


def score_reference_proofs(manifest_path, stage, config, evals=None, text_groups=None, write=True):
    """The proof step of ``score --manifest`` for the supplementary originals left unproven at staging
    (F11: their reuse proofs are established at scoring time, not by ``prepare --supplement``).

    Runs :func:`supplement_reference_proofs` (EVALUATION-RECORD READS, every path logged) on the
    manifest's ``scoring.unproven_originals``, which must be supplementary ids, under the staged scoring
    identity (``config`` must give the carried base proof's config hash; with the staged window check,
    ``text_groups`` is required). A proven original is reused; a failed proof makes it a reference to
    score. Refuses (``SystemExit``) when the step cannot run, and when any original stays without a
    proof, so no supplementary variant is scored while its original is unproven. The manifest is never
    rewritten: with ``write`` the result goes to ``reference_proofs_f11.json`` beside it (a differing
    earlier file is kept as ``reference_proofs_f11.prev-<sha12>.json``). Returns (originals to score,
    record)."""
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    man = json.loads(manifest_bytes)
    sc, sup = man["scoring"], man.get("supplement") or {}
    unproven = list(sc.get("unproven_originals") or [])
    if not unproven:
        return [], None
    stray = sorted(set(unproven) - set(sup.get("ids") or []))
    if stray:
        raise SystemExit(
            f"refusing to score: unproven originals outside the F11 supplement {stray}"
        )
    windows = None
    if sc.get("window_check"):
        if not text_groups:
            raise SystemExit(
                f"refusing to score: originals {unproven} are unproven and the staged proofs carry the "
                "window check; give --text-groups so score can prove them first"
            )
        windows = registered_windows(text_groups)
    source_zoo = Path(json.loads((Path(stage) / "stage.json").read_text())["source_zoo"])
    revisions = {
        json.loads((source_zoo / i / "edit.json").read_text())["base_revision"] for i in unproven
    }
    if len(revisions) != 1:
        raise SystemExit(
            f"refusing to score: unproven originals span revisions {sorted(revisions)}"
        )
    from wsp.eval.batch import effective_config_hash

    carried = next((p for p in sc["reuse_proofs"] if p["id"] == "base"), {}).get("config_hash")
    if carried not in (None, effective_config_hash(_load_config(config))):
        raise SystemExit(
            "refusing to score: --config gives another identity than the staged proofs"
        )
    proofs, touched, identity = supplement_reference_proofs(
        evals or sc["evals_dir"], config, source_zoo, unproven,
        sup.get("source_bytes_holders") or {}, revisions.pop(), windows,
    )  # fmt: skip
    by_id = {p["id"]: p for p in proofs}
    still = [i for i in unproven if i not in by_id]
    if still:
        raise SystemExit(
            f"refusing to score: originals still unproven after the proof step {still}"
        )
    to_score = sorted(i for i in unproven if not by_id[i]["verdict"])
    record = {
        "schema": REFERENCE_PROOFS_SCHEMA,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "identity": identity,
        "paths_touched": touched,
        "proofs": proofs,
        "proven": sorted(i for i in unproven if by_id[i]["verdict"]),
        "to_score_as_references": to_score,
        "priced": priced_after_proofs(sc, to_score),
    }
    if write:
        text = json.dumps(rotation.json_safe(record), indent=1, sort_keys=True) + "\n"
        out = manifest_path.with_name(REFERENCE_PROOFS_FILE)
        if out.exists() and out.read_text() != text:
            old = out.read_bytes()
            out.rename(
                out.with_name(
                    f"reference_proofs_f11.prev-{hashlib.sha256(old).hexdigest()[:12]}.json"
                )
            )
        out.write_text(text)
    print(
        f"reference proofs: {len(record['proven'])} proven, {len(to_score)} to score as references "
        f"{to_score}",
        flush=True,
    )
    return to_score, record


def priced_after_proofs(sc, to_score):
    """The priced count once the proof step has run: every staged variant, every missing predecessor, and
    the missing originals together with the originals whose proof failed (``to_score``), at the manifest's
    minutes per entry. Retries are not priced."""
    originals = sorted(set(sc["missing_originals"]) | set(to_score))
    refs = len(originals) + len(sc["missing_predecessors"])
    n = sc["n_variants"] + refs
    return {
        "entries": n,
        "minutes": round(n * sc.get("minutes_per_entry", MINUTES_PER_ENTRY), 4),
        "n_variants": sc["n_variants"],
        "references_to_score": refs,
        "failed_proofs": len(to_score),
    }


def load_reference_proofs(manifest_path):
    """``score --require-proofs`` (the default): the retained record of the proofs-only phase instead of a
    second proof step, so the scoring phase reads no evaluation record and scores exactly the accepted,
    repriced plan. Refuses (``SystemExit``) when the manifest leaves originals unproven and the record is
    absent, has another schema, was made for other manifest bytes, does not give every unproven original
    exactly one boolean verdict (and no other id), or lists proven / to-score sets that disagree with its
    verdicts. Returns (originals to score, record); ([], None) when nothing was left unproven."""
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    unproven = list(json.loads(manifest_bytes)["scoring"].get("unproven_originals") or [])
    if not unproven:
        return [], None
    path = manifest_path.with_name(REFERENCE_PROOFS_FILE)
    if not path.exists():
        raise SystemExit(
            f"refusing to score: no proof record {path} for the unproven originals {unproven}; run "
            "score --proofs-only, retain the record and accept the repriced plan first"
        )
    rec = json.loads(path.read_text())
    if rec.get("schema") != REFERENCE_PROOFS_SCHEMA:
        raise SystemExit(f"refusing to score: proof record schema {rec.get('schema')!r}")
    if rec.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise SystemExit("refusing to score: the proof record was made for another manifest")
    proofs = rec.get("proofs") or []
    by_id = {p.get("id"): p for p in proofs}
    bad = sorted(
        {i for i in unproven if not isinstance(by_id.get(i, {}).get("verdict"), bool)}
        | (set(by_id) - set(unproven))
    )
    if bad or len(proofs) != len(unproven):
        raise SystemExit(
            f"refusing to score: originals unproven in the proof record (or not in the manifest) {bad}"
        )
    proven = sorted(i for i in unproven if by_id[i]["verdict"])
    to_score = sorted(i for i in unproven if not by_id[i]["verdict"])
    if rec.get("proven") != proven or rec.get("to_score_as_references") != to_score:
        raise SystemExit(
            "refusing to score: the proof record's proven/to-score lists are inconsistent"
        )
    return to_score, rec


def score(args):
    proofs_only = getattr(args, "proofs_only", False)
    max_minutes = getattr(args, "max_minutes", None)
    if (proofs_only or max_minutes is not None) and not args.manifest:
        raise SystemExit("--proofs-only and --max-minutes need --manifest")
    refs = ()
    if args.manifest:
        sc = json.loads(Path(args.manifest).read_text())["scoring"]
        if proofs_only or not getattr(args, "require_proofs", True):
            proved_missing, _ = score_reference_proofs(
                args.manifest, args.stage, args.config, getattr(args, "evals", None),
                getattr(args, "text_groups", None), write=not args.dry_run,
            )  # fmt: skip
        else:
            proved_missing, _ = load_reference_proofs(args.manifest)
        price = priced_after_proofs(sc, proved_missing)
        print(
            f"PRICE entries={price['entries']} minutes={price['minutes']} "
            f"(variants {price['n_variants']}, references {price['references_to_score']}, "
            f"failed proofs {price['failed_proofs']})",
            flush=True,
        )
        if proofs_only:
            print(
                "proofs only: nothing scored; retain the proof record and accept the repriced plan "
                "before the scoring phase",
                flush=True,
            )
            return 0
        if max_minutes is not None and price["minutes"] > max_minutes:
            raise SystemExit(
                f"refusing to score: priced {price['minutes']} min exceeds the stop-and-reprice "
                f"threshold {max_minutes} min; stop and reprice"
            )
        originals = sorted(set(sc["missing_originals"]) | set(proved_missing))
        refs = tuple(originals) + tuple(sc["missing_predecessors"])
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
    p.add_argument(
        "--staged-manifest", help="--rotation-only / --supplement: the staged Part A manifest.json"
    )
    p.add_argument(
        "--supplement",
        help="F11: frozen supplementary list (selection_f11.json); stage only those roots (F10 rule, "
        "rotation + ladders) into the staged zoo and merge them into --staged-manifest",
    )
    proofs = p.add_mutually_exclusive_group()
    proofs.add_argument(
        "--reference-proofs",
        dest="reference_proofs",
        action="store_true",
        default=False,
        help="--supplement: opt in to computing the supplementary originals' reuse proofs at staging "
        "(reads TF-v1 records; every path logged)",
    )
    proofs.add_argument(
        "--no-reference-proofs",
        dest="reference_proofs",
        action="store_false",
        help="--supplement (default): no evaluation record read at staging; the proofs run in score "
        "--manifest",
    )
    s = sub.add_parser("score", help="run wsp.eval.batch on the staged zoo (inside agent-tmux)")
    s.add_argument("--stage", required=True, help="staged zoo directory (OUT/<zoo name>)")
    s.add_argument("--config", default=DEFAULT_CONFIG)
    s.add_argument(
        "--manifest", help="prepare's manifest: also score its missing references (source zoo)"
    )
    s.add_argument("--evals", help="TF-v1 evals dir for the proof step (default: the manifest's)")
    s.add_argument(
        "--text-groups", help="text_groups.json: window check of the proof step (F11 originals)"
    )
    s.add_argument("--dry-run", action="store_true")
    s.add_argument(
        "--proofs-only",
        action="store_true",
        help="--manifest: run only the deferred proof step of the unproven (F11 supplementary) "
        "originals, write reference_proofs_f11.json beside the manifest, print the repriced count "
        "and exit without scoring",
    )
    req = s.add_mutually_exclusive_group()
    req.add_argument(
        "--require-proofs",
        dest="require_proofs",
        action="store_true",
        default=True,
        help="(default) score from the retained proof record of --proofs-only; refuse if it is "
        "absent, made for another manifest, or leaves any original unproven; no evaluation read",
    )
    req.add_argument(
        "--no-require-proofs",
        dest="require_proofs",
        action="store_false",
        help="prove the unproven originals and score in one call (no stop before GPU work)",
    )
    s.add_argument(
        "--max-minutes",
        type=float,
        help="stop-and-reprice threshold: refuse to score if the priced minutes exceed it",
    )
    m = sub.add_parser("slot-mask", help="write the a5-slot-mask-v1 mask of a Part A manifest")
    m.add_argument("--manifest", required=True, help="Part A manifest.json (pair verdicts)")
    m.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"prepare": prepare, "score": score, "slot-mask": write_slot_mask}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
