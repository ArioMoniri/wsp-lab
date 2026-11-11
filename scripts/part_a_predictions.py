#!/usr/bin/env python3
"""Part A frozen prediction rows for every selected edit (F11: 52 edits) from the frozen G2 WikiText field.

PREREG_H1_TF A5.3 (the registered contrasts), F6.2/F6.3 (exact predecessor gradient for the ten chain picks), F8.3 with
F8a.3 and F9a.2 (what a frozen prediction row carries and its edge values), F11 (the twelve supplementary root edits) and
the pre-scoring inventory gap (``docs/prereg/a5_manifests/part_a/PRESCORE_INVENTORY.md``, gate 5). For every edit of
the execution manifest (``docs/prereg/a5_manifests/part_a/manifest.json``) and each of its staged variants it writes the
registered sign predictions in the analysis' ``--fields`` order [rotation, ladder 0.25, ladder 0.5]:

    rotation   p_rot = <g, D_eff - R_eff>                                  (A5.3 "Null and claim"; F6.3)
    ladder s   p_s   = <g, bf16(W + s * Delta) - W>,  s in (0.25, 0.5)    (A5.3 "Ladder": predicts y(W + s Delta) - y(W))

computed on the realised bf16 increments **as staged**: R_eff and the ladder increments are read from the staged files
under ``data/runs/a5/part_a/qwen3.5-0.8b/<variant id>/`` (``delta.safetensors`` for roots, whose staged delta is the
realised increment at the base; ``increment.safetensors`` for chain steps), each file's sha256 checked against the
manifest's ``delta_sha256`` before use. D_eff = bf16(W + Delta) - W at the actual predecessor is rebuilt from the
stored zoo bytes exactly as ``build_variants`` builds it (``a5_power_fields.edit_increments``); the staged ladders are
re-derived from it and must agree bit for bit. A rotation pair excluded at staging (F10.3; only ``variant.json`` stays)
has no R_eff: its rotation prediction is null with the exclusion as the abstention reason (the analysis accepts a null
only for an excluded contrast).

``g`` is the WikiText gradient (``wikitext_nll``) of the frozen G2 artefacts, never refitted:

* the ten registered chain picks (selection.json): the F6.2 ``chain_exact`` gradient at the actual predecessor
  (``data/runs/g2/chain_exact/<pick>/wikitext_nll``); for the two root picks this is the exact base gradient
  recomputed by the same path. Its support is the pick's nonzero realised increment; any staged increment that is
  nonzero outside it makes the field abstain for that contrast (``abstain:uncovered_tensor``), zero tensors contribute
  nothing;
* every other edit is a root (30 LoRA picks, 12 F11 supplements): the G2 base field ``data/runs/g2/fields/wikitext_nll``,
  the exact gradient at their actual predecessor. A nonroot edit that is not a pick has no validated gradient (F6.1)
  and stops the run.

Per contrast row (F8.3, F8a.3, F9a.2): a = <g, X> (fp32 per tensor as ``response_field.contract``, summed with
math.fsum; an fp64 recomputation is kept for the sign check), q = sum_t 1/2 tr(G_t X_t A_t X_t^T) from the base K-FAC
factors (``base-factor`` for every row; ``q_partial`` when a nonzero tensor has no factor), kappa = q/|a| with F8a.3(b)'s
edge statuses, the row-relative flip point t* = |a|/q (only a < 0 < q), the quadratic-sign forecast sign(a + q), and for
the edit itself (s = 1) a, q, kappa and the nominal s* = |a_1|/q_1. The rotation row also carries the descriptive
quadratic forecast (a_D + q_D) - (a_R + q_R) (F9a.2(b)). q is an unsigned Fisher magnitude; every quadratic quantity is
descriptive and never enters Part A, A5.5 or F5.5. The spectral factors are those of
``response_field._factor_for_tensor`` (``spectral_factor`` of A and G), computed once per module (bitwise the same; tested).

Outputs: ``--out`` (JSON; ``predictions`` is the analysis' ``--fields`` mapping, ``rows`` the full rows, sorted by id,
``rows_sha256`` the sha256 of their canonical JSON), ``<out stem>.rows.jsonl`` (rows as they finish) and
``<out stem>.detail.jsonl`` (per-tensor a and q). Pricing: after the first edit the wall time is projected as setup +
n_edits x first-edit time and the run stops (``price_exceeds_cap``) above ``--max-wall-hours``.
``--compare-fields`` checks the sign predictions against ``data/runs/a5/power/fields.json`` for the edits both cover
(same delta, by tensor hash -> same sign; any disagreement is reported with its cause).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/part_a_predictions.py \
        --out /home/ubuntu/wsp-data/runs/a5/part_a_predictions/predictions_f11.json \
        --compare-fields data/runs/a5/power/fields.json [--limit N] [--max-wall-hours 4]

CPU only. Reads: the manifest, selection.json/components.jsonl, the staged variant files, the stored zoo deltas
(``delta.safetensors`` or ``lora/`` only), the pinned base model and the G2 WikiText artefacts. No evaluation record,
TF-v1 score or sealed outcome is opened; nothing is written under data/ except ``--out``'s directory.
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

import torch
from safetensors.torch import load as load_safetensors_bytes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import a5_power_fields as pf
from wsp.features import response_field as rf
from wsp.features import rotation

SCHEMA = "a5-part-a-predictions-v1"
OBJECTIVE = pf.OBJECTIVE
CONTRASTS = pf.CONTRASTS
LADDERS = {"ladder_0.25": 0.25, "ladder_0.5": 0.5}
KIND = {"rotation": ("rotation", 1.0), "ladder_0.25": ("scale", 0.25), "ladder_0.5": ("scale", 0.5)}
DATA_ROOT = pf.DATA_ROOT
ALLOWED_OUT = DATA_ROOT / "runs/a5/part_a_predictions"
DEFAULT_MANIFEST = ROOT / "docs/prereg/a5_manifests/part_a/manifest.json"
DEFAULT_SELECTION = ROOT / "docs/prereg/a5_manifests/selection.json"
# F8a.3(d): the frozen G2 artefacts (g2_manifest_final.json and the WikiText field identity).
G2_MANIFEST_SHA256 = "c4ec7390aaf118ed785c390ce337a4077936a31204f97da54341bcfa54fd09ff"
WIKITEXT_IDENTITY_SHA256 = "6fab5b29c1fb70f5b08e1ae09f098088467b3bc649a234a5b12f4bc91223a87e"
SOURCE_BASE, SOURCE_EXACT = pf.SOURCE_BASE, pf.SOURCE_EXACT
ABSTAIN_EXCLUDED = "abstain:rotation_pair_excluded_at_staging"
ABSTAIN_UNCOVERED = "abstain:uncovered_tensor"
ABSTAIN_NONFINITE = "abstain:nonfinite"
ABSTAIN_Q_PARTIAL = "abstain:q_partial"
CODE_FILES = (
    "scripts/part_a_predictions.py",
    "scripts/a5_power_fields.py",
    "src/wsp/features/response_field.py",
    "src/wsp/features/rotation.py",
    "src/wsp/eval/response_intervention.py",
    "src/wsp/zoo/apply.py",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sign(x):
    if x is None:
        return None
    return 1 if x > 0 else -1 if x < 0 else 0


# ------------------------------------------------------------------------------------------------ plan


def plan_edits(manifest, selection, components):
    """Per selected edit (sorted by id): predecessor, byte holders, variant ids and gradient source.

    The predecessor must be the lineage parent in components.jsonl (base for a root) and, for a chain pick, the
    selection's ``predecessor_id``. Chain picks use their F6.2 exact gradient; every other edit must be a root
    (base field = exact anchor), else there is no validated gradient (F6.1) and planning stops.
    """
    rows = {r["id"]: r for r in components}
    picks = {s["edit_id"]: s for s in selection["registered"]["selected"] if s["family"] == "chain"}
    supplementary = set((manifest.get("supplement") or {}).get("ids") or [])
    src_holders = (manifest.get("supplement") or {}).get("source_bytes_holders") or {}
    pred_holders = manifest.get("predecessor_bytes_holders") or {}
    by_key = {(v["source_id"], v["kind"], float(v["strength"])): v for v in manifest["variants"]}

    def holder(x):
        r = rows.get(x)
        if r is None:
            raise ValueError(f"{x}: no component row")
        h = r.get("bytes_via_alias") or (x if r.get("bytes_present") else None)
        if h is None:
            raise ValueError(f"{x}: no stored or aliased bytes")
        return h

    plan = []
    for s in sorted(manifest["selections"], key=lambda s: s["edit_id"]):
        i, pred = s["edit_id"], s["predecessor_id"]
        comp = rows.get(i)
        if comp is None:
            raise ValueError(f"{i}: no component row")
        parent = comp.get("parent_id") or "base"
        if parent != pred:
            raise ValueError(
                f"{i}: manifest predecessor {pred} differs from lineage parent {parent}"
            )
        root = pred == "base"
        pick = picks.get(i)
        if pick is not None and (pick["predecessor_id"] or "base") != pred:
            raise ValueError(f"{i}: selection predecessor differs from {pred}")
        if pick is None and not root:
            raise ValueError(
                f"{i}: nonroot edit that is not a registered chain pick has no validated gradient (F6.1)"
            )
        variants = {}
        for c in CONTRASTS:
            v = by_key.get((i, *KIND[c]))
            if v is None:
                raise ValueError(f"{i}: manifest lacks its {c} variant")
            variants[c] = {
                "variant_id": v["variant_id"],
                "excluded": bool(v["excluded"]),
                "exclusions": v.get("exclusions") or [],
            }
        source_bytes = src_holders.get(i) or holder(i)
        pred_bytes = None if root else (pred_holders.get(pred) or holder(pred))
        if not root and pred_holders.get(pred) and pred_holders[pred] != holder(pred):
            raise ValueError(f"{i}: manifest predecessor holder differs from components.jsonl")
        plan.append(
            {
                "edit_id": i,
                "family": s["family"],
                "tertile": s.get("tertile"),
                "predecessor_id": pred,
                "predecessor_is_base": root,
                "supplementary": i in supplementary or bool(s.get("supplementary")),
                "registered_chain_pick": pick is not None,
                "gradient_source": SOURCE_EXACT if pick is not None else SOURCE_BASE,
                "gradient_anchor": "base-exact" if root else "predecessor-exact",
                "factor_anchor": "base-factor",
                "source_bytes": source_bytes,
                "source_via_alias": source_bytes != i,
                "predecessor_bytes": pred_bytes,
                "predecessor_via_alias": bool(pred_bytes and pred_bytes != pred),
                "rotation_seed": (manifest.get("rotation_seeds") or {}).get(i),
                "variants": variants,
            }
        )
    return plan


# ------------------------------------------------------------------------------- staged increments


def load_staged(stage_dir, variant_id, root, manifest):
    """The staged realised increment of one variant, bound to the manifest's file hash.

    Roots: ``delta.safetensors`` (the realised increment at the base). Chain steps: ``increment.safetensors``. The
    bytes are read once, hashed against ``manifest.delta_sha256[variant_id][file]``, then parsed; ``variant.json`` is
    checked against ``manifest.variant_json_sha256``. Returns (increments, file record, variant.json).
    """
    d = Path(stage_dir) / variant_id
    vj_bytes = (d / "variant.json").read_bytes()
    want_vj = (manifest.get("variant_json_sha256") or {}).get(variant_id)
    if want_vj is not None and sha256_bytes(vj_bytes) != want_vj:
        raise ValueError(f"{variant_id}: variant.json differs from the manifest's sha256")
    name = "delta.safetensors" if root else "increment.safetensors"
    want = ((manifest.get("delta_sha256") or {}).get(variant_id) or {}).get(name)
    if want is None:
        raise ValueError(f"{variant_id}: manifest has no {name} sha256")
    data = (d / name).read_bytes()
    got = sha256_bytes(data)
    if got != want:
        raise ValueError(f"{variant_id}: {name} sha256 {got} differs from the manifest's {want}")
    tensors = load_safetensors_bytes(data)
    del data
    return tensors, {"file": name, "sha256": got}, json.loads(vj_bytes)


def variant_json_only(stage_dir, variant_id, manifest):
    d = Path(stage_dir) / variant_id
    vj_bytes = (d / "variant.json").read_bytes()
    want = (manifest.get("variant_json_sha256") or {}).get(variant_id)
    if want is not None and sha256_bytes(vj_bytes) != want:
        raise ValueError(f"{variant_id}: variant.json differs from the manifest's sha256")
    return json.loads(vj_bytes)


# ------------------------------------------------------------------------------------------- field


class FactorCache:
    """``response_field._factor_for_tensor`` with ``spectral_factor`` run once per module.

    Returns the same (A slice, G) pair, bitwise: the spectral factors are a pure function of the stored A and G. With
    ``drop_raw`` the raw factors of a module are released once its spectral pair is kept.
    """

    def __init__(self, coverage, factors, drop_raw=False):
        self.coverage = coverage
        self.raw = factors
        self.kept = {}
        self.drop_raw = drop_raw
        self.seconds = 0.0

    def module(self, module):
        if module not in self.kept:
            t0 = time.time()
            pair = self.raw[module]
            a = rf.spectral_factor(pair["A"])[0]
            g = rf.spectral_factor(pair["G"])[0]
            self.kept[module] = (a, g, pair["bias"])
            if self.drop_raw:
                self.raw[module] = {k: v for k, v in pair.items() if k not in ("A", "G")}
            self.seconds += time.time() - t0
        return self.kept[module]

    def __call__(self, name):
        row = self.coverage[name]
        if not row["factors"]:
            return None
        a, g, bias = self.module(row["module"])
        n = row["shape"][-1]
        return (a[-1:, -1:] if name == bias else a[:n, :n], g)


def contract(grads, coverage, factors, increments, objective=OBJECTIVE, with_q=True):
    """Per-tensor <g, X> and q exactly as ``response_field.contract`` computes them, plus fp64 signed values.

    ``grads`` maps tensor names to the gradient (the base field's, or a chain_exact artefact's support), ``coverage``
    is the base field's coverage block (shapes and factor rows) and ``factors`` a :class:`FactorCache` (or None: no
    q). A tensor without a gradient is uncovered: listed, and recorded as nonzero if any entry is nonzero. A nonfinite
    increment raises ValueError (the caller abstains).
    """
    view = rf.FieldArtifact({"coverage": {objective: coverage}}, {objective: grads}, {})
    per, uncovered = {}, {}
    for name in sorted(increments):
        value = increments[name]
        if name not in grads:
            uncovered[name] = bool(torch.count_nonzero(value))
            continue
        d = rf._delta(view, objective, name, value)
        g = grads[name]
        row = {
            "signed": (g * d).sum().item(),
            "signed64": (g.double() * d.double()).sum().item(),
            "q": None,
            "has_factors": False,
            "nonzero": bool(torch.count_nonzero(d)),
        }
        pair = factors(name) if (with_q and factors is not None) else None
        if pair is not None:
            row["has_factors"] = True
            if row["nonzero"]:
                a, gg = pair
                matrix = d[:, None] if d.ndim == 1 else d
                row["q"] = max(0.0, (0.5 * torch.trace(gg @ matrix @ a @ matrix.T)).item())
            else:
                row["q"] = 0.0
        per[name] = row
    return per, uncovered


def curvature_terms(a, q, q_partial):
    """F8a.3(a)-(c): kappa with edge statuses, the row-relative flip t* and the quadratic-sign forecast."""
    out = {}
    if a is None or q is None or not math.isfinite(a) or not math.isfinite(q):
        for k in ("kappa", "t_star", "forecast_quadratic", "forecast_sign_quadratic"):
            out[k] = None
        out["kappa_status"] = out["t_star_status"] = out["forecast_status"] = ABSTAIN_NONFINITE
        return out
    if q_partial:
        for k in ("kappa", "t_star", "forecast_quadratic", "forecast_sign_quadratic"):
            out[k] = None
        out["kappa_status"] = out["t_star_status"] = out["forecast_status"] = ABSTAIN_Q_PARTIAL
        return out
    if a != 0:
        out["kappa"], out["kappa_status"] = q / abs(a), "ok"
    elif q > 0:
        out["kappa"], out["kappa_status"] = None, "edge:a_zero_q_positive"
    else:
        out["kappa"], out["kappa_status"] = None, "undefined:a_zero_q_zero"
    if a < 0 < q:
        out["t_star"], out["t_star_status"] = abs(a) / q, "ok"
    else:
        out["t_star"], out["t_star_status"] = None, "undefined:outside_a_negative_q_positive"
    out["forecast_quadratic"] = a + q
    out["forecast_sign_quadratic"] = sign(a + q)
    out["forecast_status"] = "ok"
    return out


def contraction_row(per, uncovered):
    """Sum a per-tensor contraction into a field row (a, a_fp64, q, q_partial, field status)."""
    nonzero_uncovered = sorted(k for k, nz in uncovered.items() if nz)
    if nonzero_uncovered:
        return {
            "field_status": ABSTAIN_UNCOVERED,
            "uncovered_nonzero_tensors": nonzero_uncovered,
            "a": None,
            "a_fp64": None,
            "q": None,
            "q_partial": None,
        }
    a = math.fsum(r["signed"] for r in per.values())
    a64 = math.fsum(r["signed64"] for r in per.values())
    q_partial = any(r["nonzero"] and not r["has_factors"] for r in per.values())
    q = math.fsum(r["q"] for r in per.values() if r["q"] is not None)
    row = {
        "field_status": "ok" if math.isfinite(a) else ABSTAIN_NONFINITE,
        "a": a if math.isfinite(a) else None,
        "a_fp64": a64 if math.isfinite(a64) else None,
        "q": q if math.isfinite(q) else None,
        "q_partial": q_partial,
        "n_tensors": len(per),
        "n_uncovered_zero_tensors": len(uncovered),
    }
    if row["a"] is not None and row["a_fp64"] is not None and sign(row["a"]) != sign(row["a_fp64"]):
        row["flags"] = ["fp32-fp64-sign-differs"]
    return row


def field_row(grads, coverage, factors, increments):
    """One contrast's field row and its per-tensor detail; nonfinite increments abstain."""
    try:
        per, uncovered = contract(grads, coverage, factors, increments)
    except ValueError as e:
        if "nonfinite" not in str(e):
            raise
        return {"field_status": ABSTAIN_NONFINITE, "a": None, "q": None, "detail": str(e)}, {}
    row = contraction_row(per, uncovered)
    detail = {k: {"a": r["signed"], "q": r["q"]} for k, r in per.items()}
    return row, detail


# --------------------------------------------------------------------------------------- per edit


def edit_prediction(entry, D, edited, staged, grads, coverage, factors):
    """The frozen prediction row of one edit.

    ``D`` is D_eff over ``edited`` (realised increment at the predecessor); ``staged`` maps each contrast to its
    staged increment dict, or None for an excluded rotation pair. Returns (row, per-tensor detail).
    """
    detail = {}
    d_eff = {k: D[k] for k in edited}
    edit_row, detail["edit"] = field_row(grads, coverage, factors, d_eff)
    edit_row["increment_sha256"] = pf.tensors_hash(d_eff)
    edit_row["sign_first_order"] = sign(edit_row.get("a"))
    curv = curvature_terms(edit_row.get("a"), edit_row.get("q"), edit_row.get("q_partial"))
    edit_row |= {k: curv[k] for k in ("kappa", "kappa_status", "forecast_quadratic")}
    edit_row["forecast_sign_quadratic"] = curv["forecast_sign_quadratic"]
    edit_row["forecast_status"] = curv["forecast_status"]
    edit_row["s_star_nominal"] = curv["t_star"]
    edit_row["s_star_nominal_status"] = curv["t_star_status"]
    rows = {}
    for c in CONTRASTS:
        v = entry["variants"][c]
        inc = staged.get(c)
        row = {"variant_id": v["variant_id"], "strength": KIND[c][1]}
        if inc is None:
            row |= {
                "status": "excluded",
                "exclusion_reasons": sorted({x["reason"] for x in v["exclusions"]}),
                "field_status": ABSTAIN_EXCLUDED,
                "prediction": None,
                "sign": None,
            }
            rows[c] = row
            continue
        if sorted(inc) != list(edited):
            raise ValueError(
                f"{entry['edit_id']} {c}: staged tensors differ from the edited tensors"
            )
        row["status"] = "staged"
        row["increment_sha256"] = pf.tensors_hash(inc)
        fr, detail[c] = field_row(grads, coverage, factors, inc)
        if c == "rotation":
            # A5.3: the registered predictor is <g, D_eff - R_eff>, contracted directly (as a5_power_fields).
            diff = {k: D[k] - inc[k] for k in edited}
            dr, detail["D_minus_R"] = field_row(grads, coverage, None, diff)
            row["field_status"] = dr["field_status"]
            row["prediction"] = dr["a"]
            row["prediction_fp64"] = dr.get("a_fp64")
            row["flags"] = list(dr.get("flags", []))
            row["R_field_status"] = fr["field_status"]
            row["a_R"], row["q_R"], row["q_R_partial"] = (
                fr.get("a"),
                fr.get("q"),
                fr.get("q_partial"),
            )
            a_d, q_d = edit_row.get("a"), edit_row.get("q")
            row["a_D_minus_a_R"] = a_d - row["a_R"] if None not in (a_d, row["a_R"]) else None
            if None in (a_d, q_d, row["a_R"], row["q_R"]):
                status = ABSTAIN_NONFINITE if row["field_status"] == "ok" else row["field_status"]
            elif edit_row["q_partial"] or row["q_R_partial"]:
                status = ABSTAIN_Q_PARTIAL
            else:
                status = "ok"
            qf = (a_d + q_d) - (row["a_R"] + row["q_R"]) if status == "ok" else None
            row |= {
                "forecast_quadratic": qf,
                "forecast_sign_quadratic": sign(qf),
                "forecast_status": status,
            }
        else:
            row |= fr
            row["prediction"] = row["a"]
            row["prediction_fp64"] = row.get("a_fp64")
            row |= curvature_terms(row["a"], row["q"], row["q_partial"])
        row["sign"] = sign(row["prediction"])
        rows[c] = row
    record = {
        k: entry[k]
        for k in (
            "edit_id",
            "family",
            "tertile",
            "predecessor_id",
            "predecessor_is_base",
            "supplementary",
            "registered_chain_pick",
            "gradient_source",
            "gradient_anchor",
            "factor_anchor",
            "source_bytes",
            "source_via_alias",
            "predecessor_bytes",
            "predecessor_via_alias",
            "rotation_seed",
        )
    }
    record["n_edited_tensors"] = len(edited)
    record["edit"] = edit_row
    record["contrasts"] = rows
    record["predictions"] = [rows[c]["prediction"] for c in CONTRASTS]
    record["abstentions"] = {
        c: rows[c]["field_status"] for c in CONTRASTS if rows[c]["prediction"] is None
    }
    return record, detail


def rows_sha256(rows) -> str:
    """sha256 of the canonical JSON of the rows sorted by edit id (keys sorted, no NaN)."""
    return sha256_bytes(canonical_json(sorted(rows, key=lambda r: r["edit_id"])))


def predictions_table(rows) -> dict:
    """The analysis' ``--fields`` mapping: edit id -> [rotation, ladder .25, ladder .5] (A5.3.2 order)."""
    return {r["edit_id"]: list(r["predictions"]) for r in sorted(rows, key=lambda r: r["edit_id"])}


def sign_distribution(rows) -> dict:
    out = {}
    for j, c in enumerate(CONTRASTS):
        vals = [r["predictions"][j] for r in rows]
        out[c] = {
            "positive": sum(v is not None and v > 0 for v in vals),
            "negative": sum(v is not None and v < 0 for v in vals),
            "zero": sum(v == 0 for v in vals if v is not None),
            "null": sum(v is None for v in vals),
        }
    return out


# -------------------------------------------------------------------------------- agreement check


def compare_with_power_fields(rows, fields, power_rows):
    """Sign agreement with ``data/runs/a5/power/fields.json`` for the edits both files cover.

    Same delta (equal tensor hash) and same gradient source must give the same sign; a disagreement is reported
    with its cause: a different staged delta (hash differs: the power run rebuilt R itself and never applied the F10
    exclusion), a different gradient anchoring, or a null here (rotation excluded at staging).
    """
    by_id = {r["edit_id"]: r for r in rows}
    shared = sorted(set(by_id) & set(fields))
    hash_key = {"rotation": "R_eff", "ladder_0.25": "ladder_0.25", "ladder_0.5": "ladder_0.5"}
    out = {
        "n_shared": len(shared),
        "shared_ids": shared,
        "by_contrast": {},
        "disagreements": [],
        "not_comparable": [],
    }
    for j, c in enumerate(CONTRASTS):
        agree = same_value = compared = 0
        for i in shared:
            r = by_id[i]
            mine = r["predictions"][j]
            theirs = fields[i][j] if fields[i] is not None else None
            prow = power_rows.get(i) or {}
            h_mine = r["contrasts"][c].get("increment_sha256")
            h_theirs = (prow.get("hashes") or {}).get(hash_key[c])
            same_delta = h_mine is not None and h_mine == h_theirs
            same_source = prow.get("gradient_source") == r["gradient_source"]
            d_mine, d_theirs = (
                r["edit"]["increment_sha256"],
                (prow.get("hashes") or {}).get("D_eff"),
            )
            if mine is None or theirs is None:
                out["not_comparable"].append(
                    {
                        "edit_id": i,
                        "contrast": c,
                        "here": mine,
                        "fields_json": theirs,
                        "cause": r["contrasts"][c].get("field_status")
                        if mine is None
                        else "null in fields.json",
                    }
                )
                continue
            compared += 1
            same_value += mine == theirs
            if sign(mine) == sign(theirs):
                agree += 1
                continue
            causes = []
            if not same_delta:
                causes.append(
                    f"different staged {hash_key[c]} (here {h_mine}, fields.json {h_theirs})"
                )
            if d_mine != d_theirs:
                causes.append("different D_eff")
            if not same_source:
                causes.append(
                    f"gradient source {r['gradient_source']} here vs {prow.get('gradient_source')}"
                )
            out["disagreements"].append(
                {
                    "edit_id": i,
                    "contrast": c,
                    "here": mine,
                    "fields_json": theirs,
                    "same_delta": same_delta,
                    "same_gradient_source": same_source,
                    "cause": "; ".join(causes)
                    or "UNEXPLAINED: same delta and same gradient source",
                }
            )
        out["by_contrast"][c] = {
            "compared": compared,
            "sign_agree": agree,
            "bitwise_equal_value": same_value,
        }
    hashes = Counter()
    for i in shared:
        prow = power_rows.get(i) or {}
        r = by_id[i]
        hashes["D_eff_equal"] += r["edit"]["increment_sha256"] == (prow.get("hashes") or {}).get(
            "D_eff"
        )
        for c in CONTRASTS:
            hashes[f"{c}_increment_equal"] += r["contrasts"][c].get("increment_sha256") == (
                prow.get("hashes") or {}
            ).get(hash_key[c])
        hashes["gradient_source_equal"] += prow.get("gradient_source") == r["gradient_source"]
    out["hash_equalities"] = dict(sorted(hashes.items()))
    out["unexplained"] = sum(d["cause"].startswith("UNEXPLAINED") for d in out["disagreements"])
    return out


# --------------------------------------------------------------------------------------- loading


def code_identity():
    out = {f: pf.sha256_file(ROOT / f) for f in CODE_FILES}
    try:
        out["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *CODE_FILES], text=True, cwd=ROOT
        )
        out["code_files_dirty"] = bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        out["git_sha"] = None
    return out


def check_out_path(path: Path):
    p = Path(path).resolve()
    if DATA_ROOT in p.parents and ALLOWED_OUT not in p.parents:
        raise ValueError(f"{path}: data outputs must stay under {ALLOWED_OUT}")


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def edited_union(plan, stage_dir, manifest):
    """Union of every edit's edited tensors (from the staged variant.json, hash-checked)."""
    names = set()
    for e in plan:
        vj = variant_json_only(stage_dir, e["variants"]["rotation"]["variant_id"], manifest)
        names |= set(vj["edited_tensors"])
    return names


def run(args):
    t_start = time.time()
    torch.set_num_threads(args.threads)
    check_out_path(args.out)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    comp_path = args.selection.parent / "components.jsonl"
    comp_bytes = comp_path.read_bytes()
    if sha256_bytes(comp_bytes) != selection["components_jsonl_sha256"]:
        raise ValueError("components.jsonl differs from selection.json's pin")
    components = [json.loads(x) for x in comp_bytes.decode().splitlines() if x.strip()]
    plan = plan_edits(manifest, selection, components)
    stage_dir = ROOT / manifest["staged_zoo"] if args.stage is None else Path(args.stage)
    zoo = Path(args.zoo)

    g2 = Path(args.g2)
    g2_manifest_path = g2 / "g2_manifest.json"
    g2_bytes = g2_manifest_path.read_bytes()
    if sha256_bytes(g2_bytes) != G2_MANIFEST_SHA256:
        raise ValueError("g2_manifest.json differs from g2_manifest_final.json (F8a.3(d))")
    g2_manifest = json.loads(g2_bytes)
    frec = g2_manifest["fields"][OBJECTIVE]
    if frec.get("status") != "complete" or frec["identity_sha256"] != WIKITEXT_IDENTITY_SHA256:
        raise ValueError("base WikiText field is not the frozen complete G2 field")
    ce = g2_manifest["chain_exact"]
    ce_rows = {
        r["pick_id"]: r
        for r in ce["rows"]
        if r["objective"] == OBJECTIVE and r.get("status") == "complete"
    }
    from wsp.zoo.apply import load_entry_delta

    log(f"plan: {len(plan)} edits; reading edited-tensor union from staged variant.json")
    union = edited_union(plan, stage_dir, manifest)
    log(f"{len(union)} edited tensors in the union; loading the base WikiText field (hash-checked)")
    t0 = time.time()
    field, base_files = pf.load_field(g2 / "fields" / OBJECTIVE, frec["files"])
    if rf._hash(field.identity) != WIKITEXT_IDENTITY_SHA256:
        raise ValueError("base field identity sha256 differs from the frozen identity")
    if field.identity["state_hash"] != frec["state_hash"]:
        raise ValueError("base field state hash differs from the G2 manifest")
    coverage = field.identity["coverage"][OBJECTIVE]
    base_identity_state = field.identity["state_hash"]
    grads_base = {k: v for k, v in field.gradients[OBJECTIVE].items() if k in union}
    modules = {coverage[k]["module"] for k in union if k in coverage and coverage[k]["factors"]}
    raw_factors = {m: p for m, p in field.factors[OBJECTIVE].items() if m in modules}
    del field
    gc.collect()
    load_seconds = time.time() - t0
    log(f"field loaded in {load_seconds:.0f}s; spectral factors for {len(modules)} modules")
    factors = FactorCache(coverage, raw_factors, drop_raw=True)
    t0 = time.time()
    for m in sorted(modules):
        factors.module(m)
    spectral_seconds = time.time() - t0
    gc.collect()
    log(f"spectral factors in {spectral_seconds:.0f}s; loading the pinned base model state")
    base, base_hash = pf.load_base_state(base_identity_state)
    setup_seconds = time.time() - t_start
    log(f"setup {setup_seconds:.0f}s")

    def load_delta(holder):
        return load_entry_delta(zoo / holder, base)

    out_rows = args.out.with_name(args.out.stem + ".rows.jsonl")
    out_detail = args.out.with_name(args.out.stem + ".detail.jsonl")
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    out_rows.write_text("")
    out_detail.write_text("")
    todo = plan if args.limit is None else plan[: args.limit]
    order = [e for e in todo if e["gradient_source"] == SOURCE_EXACT] + [
        e for e in todo if e["gradient_source"] == SOURCE_BASE
    ]
    rows, pricing = [], None
    for n, e in enumerate(order, 1):
        t_edit = time.time()
        i = e["edit_id"]
        stored = load_delta(e["source_bytes"])
        pred = None if e["predecessor_is_base"] else load_delta(e["predecessor_bytes"])
        W, D, raw, edited = pf.edit_increments(base, stored, pred)
        del stored, pred
        staged, files = {}, {}
        for c in CONTRASTS:
            v = e["variants"][c]
            if v["excluded"]:
                vj = variant_json_only(stage_dir, v["variant_id"], manifest)
                if not vj.get("excluded"):
                    raise ValueError(f"{i} {c}: manifest excludes a variant staged as included")
                staged[c] = None
                files[c] = {"variant_json_sha256": manifest["variant_json_sha256"][v["variant_id"]]}
                continue
            inc, rec, vj = load_staged(
                stage_dir, v["variant_id"], e["predecessor_is_base"], manifest
            )
            if vj["edited_tensors"] != edited:
                raise ValueError(f"{i} {c}: staged edited tensors differ from the rebuilt edit")
            if vj["source_id"] != i or vj["predecessor_id"] != e["predecessor_id"]:
                raise ValueError(f"{i} {c}: staged variant.json lineage differs")
            staged[c] = {k: inc[k].float() for k in edited}
            files[c] = rec | {
                "variant_json_sha256": manifest["variant_json_sha256"][v["variant_id"]]
            }
        ladder_bitwise = {}
        for c, s in LADDERS.items():
            if staged[c] is None:
                continue
            ladder_bitwise[c] = all(
                torch.equal(rotation.scaled_variant(W[k], raw[k], s)[1], staged[c][k])
                for k in edited
            )
            if not ladder_bitwise[c]:
                raise ValueError(f"{i} {c}: staged ladder differs from bf16(W + s*Delta) - W")
        if e["gradient_source"] == SOURCE_EXACT:
            rec = ce_rows.get(i)
            if rec is None or rec["predecessor_id"] != e["predecessor_id"]:
                raise ValueError(f"{i}: no complete chain_exact WikiText row at its predecessor")
            exact, exact_files = pf.load_field(rec["dir"], rec["files"])
            if exact.identity.get("base_state_hash") != base_hash:
                raise ValueError(f"{i}: chain_exact base state hash differs")
            grads = exact.gradients[OBJECTIVE]
            identity = exact.identity
            del exact
            nonzero_D = {k: D[k] for k in edited if torch.count_nonzero(D[k])}
            if identity["increment_sha256"] != pf.tensors_hash(nonzero_D):
                raise ValueError(f"{i}: realised increment differs from chain_exact's")
            if sorted(identity["support"]) != sorted(nonzero_D) or sorted(grads) != sorted(
                nonzero_D
            ):
                raise ValueError(f"{i}: chain_exact support differs from the increment")
            gradient = {
                "source": SOURCE_EXACT,
                "dir": rec["dir"],
                "anchor": "actual predecessor (exact gradient, F6.2)",
                "files": exact_files,
                "identity_sha256": rec["identity_sha256"],
                "recorded_exact_contraction": rec["exact_contraction"],
                "predecessor_state_sha256": identity["predecessor_state_sha256"],
            }
        else:
            grads = grads_base
            gradient = {
                "source": SOURCE_BASE,
                "dir": str(g2 / "fields" / OBJECTIVE),
                "anchor": "base (exact: the predecessor is the base)",
                "files": base_files,
                "identity_sha256": frec["identity_sha256"],
            }
        row, detail = edit_prediction(e, D, edited, staged, grads, coverage, factors)
        if e["gradient_source"] == SOURCE_EXACT and not math.isclose(
            row["edit"]["a"], rec["exact_contraction"], rel_tol=1e-5, abs_tol=1e-9
        ):
            raise ValueError(f"{i}: <g, D> differs from chain_exact's exact_contraction")
        row["gradient"] = gradient
        row["staged_files"] = files
        row["ladder_rebuilt_bitwise"] = ladder_bitwise
        row["edited_tensor_rule"] = (
            "stored delta keys" if e["predecessor_is_base"] else "nonzero realised increment"
        )
        row["scale_source"] = (
            "raw_stored_delta" if e["predecessor_is_base"] else "realised_increment"
        )
        seconds = time.time() - t_edit
        rows.append(row)
        with out_rows.open("a") as f:
            f.write(json.dumps(row | {"seconds": seconds}, sort_keys=True, allow_nan=False) + "\n")
        with out_detail.open("a") as f:
            f.write(json.dumps({"edit_id": i, "per_tensor": detail}, sort_keys=True) + "\n")
        del W, D, raw, staged, grads
        gc.collect()
        log(f"{n}/{len(order)} {i} {e['gradient_source']} {row['predictions']} ({seconds:.0f}s)")
        if n == 1:
            projected = setup_seconds + len(order) * seconds
            pricing = {
                "setup_seconds": setup_seconds,
                "first_edit_id": i,
                "first_edit_seconds": seconds,
                "n_edits": len(order),
                "projected_wall_seconds": projected,
                "cap_wall_hours": args.max_wall_hours,
                "within_cap": projected <= args.max_wall_hours * 3600,
            }
            log(f"pricing: {json.dumps(pricing)}")
            if not pricing["within_cap"]:
                write_json(args.out.with_name(args.out.stem + ".pricing.json"), pricing)
                raise SystemExit(
                    f"price_exceeds_cap: projected {projected / 3600:.2f} h > {args.max_wall_hours} h"
                )
    rows.sort(key=lambda r: r["edit_id"])
    preds = predictions_table(rows)
    comparison = None
    if args.compare_fields:
        fields = json.loads(Path(args.compare_fields).read_text())
        prow_path = Path(args.compare_fields).with_name(
            Path(args.compare_fields).stem + ".rows.jsonl"
        )
        power_rows = {r["id"]: r for r in map(json.loads, prow_path.read_text().splitlines()) if r}
        comparison = compare_with_power_fields(rows, fields, power_rows)
        comparison["fields_json"] = {
            "path": str(args.compare_fields),
            "sha256": pf.sha256_file(args.compare_fields),
        }
        comparison["fields_rows_jsonl_sha256"] = pf.sha256_file(prow_path)
    complete = len(rows) == len(plan)
    doc = {
        "schema": SCHEMA,
        "status": "complete" if complete else f"partial ({len(rows)}/{len(plan)})",
        "basis": "PREREG_H1_TF A5.3 (rotation and ladder contrasts), F6.2/F6.3 (exact predecessor gradient for the "
        "ten chain picks), F8.3 with F8a.3/F9a.2 (row content and edge values), F11 (52 edits); "
        "PRESCORE_INVENTORY gate 5",
        "objective": OBJECTIVE,
        "contrast_order": list(CONTRASTS),
        "predictor": {
            "rotation": "<g, D_eff - R_eff> (A5.3; predicts the sign of y(W + D) - y(W + R))",
            "ladder": "<g, bf16(W + s*Delta) - W>, s in (0.25, 0.5) (A5.3 'Ladder'; predicts the sign of "
            "y(W + s*Delta) - y(W) at the actual predecessor W, as the analysis computes it)",
            "increments": "as staged: R_eff and ladder increments read from the staged files (sha256 = manifest "
            "delta_sha256); D_eff rebuilt from stored zoo bytes as build_variants builds it; staged ladders "
            "re-derived bitwise",
            "contraction": "response_field.contract arithmetic (fp32 per tensor), summed with math.fsum; fp64 "
            "recomputation kept for the sign check",
            "q": "sum over tensors of 1/2 tr(G X A X^T) with the base K-FAC spectral factors (base-factor for every "
            "row, F6.2); unsigned Fisher magnitude; descriptive only (F8a.3(b))",
            "kappa": "q/|a| with F8a.3(b) edge statuses; t* = |a|/q only for a < 0 < q (row-relative); edit "
            "s_star_nominal = |a_1|/q_1 (nominal full-strength-direction forecast, not a ladder-point forecast)",
            "quadratic_forecast": "ladder/edit: sign(a_s + q_s) (F8a.3(a)); rotation: (a_D + q_D) - (a_R + q_R) "
            "(F9a.2(b)); descriptive, never enters Part A, A5.5 or F5.5",
            "gradient_sources": {
                SOURCE_EXACT: "registered chain picks: F6.2 exact WikiText gradient at the actual predecessor "
                "(base for the two root picks)",
                SOURCE_BASE: "root edits (30 LoRA picks, 12 F11 supplements): G2 base field, the exact anchor",
            },
            "abstentions": {
                ABSTAIN_EXCLUDED: "rotation pair excluded at staging (F10.3): no R_eff exists; prediction null",
                ABSTAIN_UNCOVERED: "a nonzero staged tensor outside the gradient's support; prediction null",
                ABSTAIN_NONFINITE: "nonfinite increment or contraction",
                ABSTAIN_Q_PARTIAL: "q incomplete (nonzero tensor without factors): kappa, t*, forecast abstain",
            },
            "chart": "abstain:no_frozen_chart_predictor (F8 spec 3.5); no chart model exists",
        },
        "predictions": preds,
        "predictions_sha256": sha256_bytes(canonical_json(preds)),
        "abstentions": {r["edit_id"]: r["abstentions"] for r in rows if r["abstentions"]},
        "rows": rows,
        "rows_sha256": rows_sha256(rows),
        "rows_sha256_rule": "sha256 of json.dumps(rows sorted by edit_id, sort_keys=True, separators=(',', ':'), "
        "allow_nan=False)",
        "inputs": {
            "manifest": {"path": str(args.manifest), "sha256": sha256_bytes(manifest_bytes)},
            "selection": {"path": str(args.selection), "sha256": sha256_bytes(selection_bytes)},
            "components_jsonl_sha256": sha256_bytes(comp_bytes),
            "staged_zoo": str(stage_dir),
            "zoo": str(zoo),
            "g2_manifest": {"path": str(g2_manifest_path), "sha256": sha256_bytes(g2_bytes)},
            "g2_artefacts": {
                "wikitext_field_dir": str(g2 / "fields" / OBJECTIVE),
                "wikitext_field_files": base_files,
                "wikitext_field_identity_sha256": frec["identity_sha256"],
                "wikitext_field_content_hashes_sha256": frec["content_hashes_sha256"],
                "chain_exact_summary_sha256": ce.get("sha256"),
                "chain_exact_rows": {
                    r["edit_id"]: {
                        "identity_sha256": r["gradient"]["identity_sha256"],
                        "files": r["gradient"]["files"],
                    }
                    for r in rows
                    if r["gradient_source"] == SOURCE_EXACT
                },
            },
        },
        "base": {
            "model": {"repo": pf.MODEL[0], "revision": pf.MODEL[1], "dtype": "bfloat16"},
            "state_hash": base_hash,
        },
        "outcomes_read": "none: no evaluation record, TF-v1 score, rotation/ladder outcome or sealed outcome is "
        "opened; only stored edit bytes, staged variant files, the base model and the G2 WikiText artefacts",
        "nothing_written_under": ["data/zoo", "data/runs/g2", "data/runs/a5/part_a", "data/evals"],
        "code": code_identity(),
        "comparison_with_power_fields": comparison,
        "summary": {
            "n_edits": len(rows),
            "n_plan": len(plan),
            "gradient_source_counts": dict(
                sorted(Counter(r["gradient_source"] for r in rows).items())
            ),
            "n_supplementary": sum(r["supplementary"] for r in rows),
            "sign_distribution": sign_distribution(rows),
            "n_abstaining_contrasts": sum(len(r["abstentions"]) for r in rows),
            "abstention_reasons": dict(
                sorted(Counter(v for r in rows for v in r["abstentions"].values()).items())
            ),
            "q_partial_rows": sum(
                bool(r["contrasts"][c].get("q_partial") or r["contrasts"][c].get("q_R_partial"))
                for r in rows
                for c in CONTRASTS
            ),
            "flags": dict(
                sorted(
                    Counter(
                        f
                        for r in rows
                        for c in CONTRASTS
                        for f in r["contrasts"][c].get("flags", [])
                    ).items()
                )
            ),
            "pricing": pricing,
            "timing": {
                "field_load_seconds": load_seconds,
                "spectral_seconds": spectral_seconds,
                "setup_seconds": setup_seconds,
                "wall_seconds": time.time() - t_start,
                "threads": args.threads,
            },
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "detail_jsonl": {"path": str(out_detail), "sha256": pf.sha256_file(out_detail)},
    }
    sha = write_json(args.out, doc)
    log(f"wrote {args.out} sha256 {sha}; rows_sha256 {doc['rows_sha256']}")
    log(json.dumps(doc["summary"], indent=1))
    if comparison is not None:
        log(
            json.dumps(
                {k: comparison[k] for k in ("by_contrast", "disagreements", "unexplained")},
                indent=1,
            )
        )
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--g2", type=Path, default=ROOT / "data/runs/g2")
    p.add_argument("--zoo", default=str(ROOT / "data/zoo/qwen3.5-0.8b"))
    p.add_argument("--stage", help="staged zoo (default: the manifest's staged_zoo)")
    p.add_argument("--compare-fields", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    p.add_argument("--max-wall-hours", type=float, default=4.0)
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
