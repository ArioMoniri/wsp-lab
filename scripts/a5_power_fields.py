#!/usr/bin/env python3
"""POWER-AND-YIELD field predictions: the registered Part A sign predictor per NULL-SIZE donor pair.

A5.3.2 (POWER-AND-YIELD runs after G2 on development rows, before G3), F2 (the donor pairs and the
[rotation, ladder 0.25, ladder 0.5] contrast order), F6.3 (sign predictor <g_wikitext, D - R>; exact
predecessor gradient for the ten chain picks) and F6.4 (other chain rows keep base-anchored tokens, flagged),
and Board A's F6 review condition 5 (docs/reviews/council/board-a-f6-2026-11-11.md).

For every donor pair of the frozen NULL-SIZE payload (``data/runs/a5/nullsize/inputs.json``; only the donor
ids, families and ``provenance.donors[*].reference_id`` are read, never ``donor_mean``, ``centred_noise`` or
``window_mask``) this writes three finite predictions, in [rotation, ladder 0.25, ladder 0.5] order:

    rotation   p_rot = <g, D_eff - R_eff>
    ladder s   p_s   = <g, bf16(W_pred + s * delta) - W_pred>,   s in (0.25, 0.5)

``g`` is the WikiText gradient (``wikitext_nll``) contracted by :func:`wsp.features.response_field.contract`:

* donors among the ten registered chain picks (selection.json): the F6.2 ``chain_exact`` gradient at the
  actual predecessor, ``data/runs/g2/chain_exact/<pick>/wikitext_nll`` (for the two root picks this is the
  exact gradient at the base, recomputed by the same path);
* root donors (LoRA, and chain edits whose predecessor is the base): the G2 base field
  ``data/runs/g2/fields/wikitext_nll``; the base is their actual predecessor, so this is the exact anchor;
* chain donors that are not picks and whose predecessor is not the base: the base field, flagged
  ``base-approximation-unvalidated`` (F6.4). Board A's condition 5 forbids using base gradients for nonroot
  chains in the primary POWER-AND-YIELD contrast; the flag lets the simulation report that set separately.

D_eff, R_eff and the ladder increments are built as ``wsp.eval.response_intervention.build_variants`` builds
them, without writing a variant: stored bf16 states, realised increments subtracted in fp32 (A5.3.3). Root
edits: D = bf16(base + stored delta) - base over the stored delta keys, the ladder scales the raw stored
delta. Chain steps: the stored cumulative states give D = S - W_pred over the tensors with a nonzero realised
increment, and the ladder scales that realised increment. The rotation R_raw = Q_out D_block Q_in^T uses one
torch generator per edit seeded with ``rotation_seed(canonical id)`` (control_seeds.json ``rotation``), its
initial state cloned for each tensor in sorted-name order (``rotation.rotate_block``); R_eff =
bf16(W_pred + R_raw) - W_pred. Two exact shortcuts keep CPU time bounded and are pinned by tests: equal
block shapes within one edit share frames (each tensor restarts the same generator state, so the Haar draws
are identical), and the ``rank_degenerate`` exclusion of ``rotation.exclusion_reason`` is decided without
its SVD (a finite nonzero block has sigma_1 >= max|entry| > 0, so its numerical rank is at least one).
The A5.3.3 rotation/scaled-edit audits are G3 staging outputs (F3) and are not computed; the payload's audit
flags are not touched. No variant is scored and nothing is written under data/zoo or data/runs/g2.

    python scripts/a5_power_fields.py --inputs data/runs/a5/nullsize/inputs.json --g2 data/runs/g2 \
        --selection docs/prereg/a5_manifests/selection.json --out data/runs/a5/power/fields.json \
        [--manifest-out docs/prereg/a5_manifests/power/fields_manifest.json]

Run it inside ``agent-tmux`` (CPU, about 13 GB peak: the base WikiText field is loaded once with
``FieldArtifact.load`` and its K-FAC factors are dropped; chain_exact artifacts are loaded one pick at a time).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.eval.response_intervention import ROTATION_DOMAIN, rotation_seed
from wsp.features import response_field as rf
from wsp.features import rotation

SCHEMA = "a5-power-fields-v1"
OBJECTIVE = "wikitext_nll"
CONTRASTS = ("rotation", "ladder_0.25", "ladder_0.5")
STRENGTHS = (0.25, 0.5)
MODEL = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
DATA_ROOT = Path("/home/ubuntu/wsp-data")
ALLOWED_OUT = DATA_ROOT / "runs/a5/power"
FLAG_UNVALIDATED = "base-approximation-unvalidated"
SOURCE_BASE, SOURCE_EXACT = "base_field", "chain_exact"
CODE_FILES = (
    "scripts/a5_power_fields.py",
    "src/wsp/features/response_field.py",
    "src/wsp/features/rotation.py",
    "src/wsp/eval/response_intervention.py",
    "src/wsp/zoo/apply.py",
)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tensors_hash(tensors) -> str:
    """The G2 convention (chain_exact ``increment_sha256``): rf._hash of per-tensor rf._tensor_hash."""
    return rf._hash({n: rf._tensor_hash(t) for n, t in tensors.items()})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------- donor plan


def plan_donors(inputs, components, selection):
    """Per donor (payload order): predecessor, byte holders, gradient source and flags.

    Reads only donor ids/families and ``provenance.donors[*].reference_id`` from the payload. The
    predecessor must equal the lineage parent in components.jsonl (base for a root) and, for picks, the
    selection's ``predecessor_id``.
    """
    rows = {r["id"]: r for r in components}
    picks = {s["edit_id"]: s for s in selection["registered"]["selected"] if s["family"] == "chain"}

    def holder(x):
        r = rows.get(x)
        if r is None:
            raise ValueError(f"{x}: no component row")
        h = r.get("bytes_via_alias") or (x if r.get("bytes_present") else None)
        if h is None:
            raise ValueError(f"{x}: no stored or aliased bytes")
        return h

    prov = inputs["provenance"]["donors"]
    plan = []
    for d in inputs["donors"]:
        i = d["id"]
        comp = rows[i]
        ref = prov[i]["reference_id"]
        parent = comp.get("parent_id") or "base"
        if ref != parent:
            raise ValueError(f"{i}: payload reference {ref} differs from lineage parent {parent}")
        root = ref == "base"
        pick = picks.get(i)
        if pick is not None and (pick["predecessor_id"] or "base") != ref:
            raise ValueError(f"{i}: selection predecessor differs from {ref}")
        canonical = comp.get("canonical_id") or i
        flags = []
        if pick is not None:
            source = SOURCE_EXACT
        else:
            source = SOURCE_BASE
            if d["family"] == "chain" and not root:
                flags.append(FLAG_UNVALIDATED)
        donor_bytes = holder(i)
        pred_bytes = None if root else holder(ref)
        if canonical != i:
            flags.append("canonical-id-differs")
        plan.append(
            {
                "id": i,
                "family": d["family"],
                "predecessor_id": ref,
                "predecessor_is_base": root,
                "donor_bytes": donor_bytes,
                "donor_via_alias": donor_bytes != i,
                "predecessor_bytes": pred_bytes,
                "predecessor_via_alias": bool(pred_bytes and pred_bytes != ref),
                "registered_chain_pick": pick is not None,
                "gradient_source": source,
                "canonical_id": canonical,
                "rotation_seed": rotation_seed(canonical),
                "flags": flags,
            }
        )
    missing = sorted(set(picks) - {p["id"] for p in plan})
    if missing:
        raise ValueError(f"registered chain picks without a donor row: {missing}")
    return plan


# ---------------------------------------------------------------------- increments (A5.3.3, build_variants)


def stored_state(base, delta):
    """Stored bf16 state of one tensor: (base + stored delta) rounded to the base dtype."""
    return (base.float() + delta.float()).to(base.dtype)


def edit_increments(base, stored, pred_stored=None):
    """``build_variants``' W, D, scale source and edited tensors for one edit.

    ``base`` maps tensor names to the base bf16 state, ``stored`` is the edit's stored delta (cumulative
    from the base) and ``pred_stored`` the predecessor's (None for a root edit, whose predecessor is the
    base). Returns (W, D, raw, edited) restricted to the tensors the edit or its predecessor touch.
    """
    root = pred_stored is None
    keys = set(stored)
    if root:
        W = {k: base[k] for k in keys}
        pred_changed = set()
    else:
        W = {k: stored_state(base[k], v) for k, v in pred_stored.items()}
        pred_changed = {k for k in W if not torch.equal(W[k], base[k])}
        for k in keys - set(W):
            W[k] = base[k]
    D = {}
    for k in sorted(keys | pred_changed):
        S = stored_state(base[k], stored[k]) if k in stored else base[k]
        D[k] = rotation.realised_increment(W[k], S)
    if root:
        edited = sorted(keys)
        raw = {k: stored[k].float() for k in edited}
    else:
        edited = sorted(k for k in D if torch.any(D[k] != 0))
        raw = {k: D[k] for k in edited}
    if not edited:
        raise ValueError("no edited tensors at the predecessor")
    return W, D, raw, edited


def exclusion_reason(D, rows, cols):
    """``rotation.exclusion_reason`` without its SVD; identical reasons (tested).

    A finite block with a nonzero entry has sigma_1 >= max|entry| > 0, so its numerical rank at any
    relative tolerance below one is at least one: ``rank_degenerate`` cannot follow a passed ``zero_block``.
    """
    if not isinstance(D, torch.Tensor):
        return "not_single_matrix"
    if D.ndim != 2:
        return "non_2d"
    if not torch.isfinite(D).all():
        return "non_finite"
    if len(rows) == 0 or len(cols) == 0 or not torch.any(D[rows[:, None], cols]):
        return "zero_block"
    return None


def rotate_block_cached(D, rows, cols, seed, cache):
    """``rotation.rotate_block(D, rows, cols, seed, shared_generator=<fresh generator(seed)>)``.

    Every tensor of an edit restarts the same initial generator state, so its Haar frames depend only
    on (len(rows), len(cols)); they are drawn once per shape and reused (bitwise identical; tested).
    """
    key = (len(rows), len(cols))
    if key not in cache:
        generator = torch.Generator()
        generator.manual_seed(seed)
        q_out = rotation.haar_orthogonal(len(rows), generator)
        q_in = rotation.haar_orthogonal(len(cols), generator)
        cache[key] = (q_out, q_in)
    q_out, q_in = cache[key]
    block = D.float()[rows[:, None], cols]
    result = torch.zeros_like(D, dtype=torch.float32)
    result[rows[:, None], cols] = q_out @ block @ q_in.T
    return result


def variant_increments(W, D, raw, edited, seed, strengths=STRENGTHS):
    """Realised rotation and ladder increments at the predecessor (no audit, nothing written).

    Returns (R_eff or None, exclusions, {s: X_s}). Any excluded tensor excludes the whole rotation pair.
    """
    exclusions, blocks = [], {}
    for k in edited:
        if not isinstance(D[k], torch.Tensor):
            reason = "not_single_matrix"
        elif D[k].ndim != 2:
            reason = "non_2d"
        else:
            blocks[k] = rotation.changed_block(D[k])
            reason = exclusion_reason(D[k], *blocks[k])
        if reason:
            exclusions.append({"tensor": k, "reason": reason})
    R = None
    if not exclusions:
        R, cache = {}, {}
        for k in edited:
            R_raw = rotate_block_cached(D[k], *blocks[k], seed, cache)
            R[k] = rotation.apply_and_round(W[k], R_raw)[1]
            del R_raw
        del cache
    ladders = {
        s: {k: rotation.scaled_variant(W[k], raw[k], s)[1] for k in edited} for s in strengths
    }
    return R, exclusions, ladders


# ------------------------------------------------------------------------------------ contractions


def gradient_view(field, objective=OBJECTIVE):
    """A factor-free view of one objective for :func:`rf.contract` (signed contraction only).

    The coverage rows carry each gradient's shape with ``factors`` False, so ``contract`` returns
    exactly its ``signed`` value ((g * d).sum() in fp32) and q = None without any eigendecomposition.
    Works for the base field and for chain_exact artifacts (which carry no coverage block).
    """
    grads = field.gradients[objective]
    coverage = {
        n: {"gradient": True, "factors": False, "module": None, "shape": list(g.shape)}
        for n, g in grads.items()
    }
    return rf.FieldArtifact({"coverage": {objective: coverage}}, {objective: grads}, {})


def contraction(view, increments, objective=OBJECTIVE):
    """<g, X> summed over tensors with rf.contract; plus an fp64 recomputation for the sign check."""
    rows = rf.contract(view, increments)[objective]
    value = math.fsum(r["signed"] for r in rows.values())
    grads = view.gradients[objective]
    value64 = math.fsum(
        (grads[n].double() * x.double()).sum().item() for n, x in increments.items()
    )
    return value, value64


def predictions(view, D, R, ladders, edited, objective=OBJECTIVE):
    """[p_rot, p_0.25, p_0.5] and the auxiliary contractions <g, D>, <g, R>."""
    grads = view.gradients[objective]
    missing = [k for k in edited if k not in grads]
    if missing:
        raise ValueError(f"gradient missing for edited tensors {missing[:3]}")
    d_eff = {k: D[k] for k in edited}
    out = {"g_dot_D": contraction(view, d_eff, objective)}
    if R is not None:
        out["g_dot_R"] = contraction(view, R, objective)
        out["rotation"] = contraction(view, {k: D[k] - R[k] for k in edited}, objective)
    for s, X in ladders.items():
        out[f"ladder_{s}"] = contraction(view, X, objective)
    return out


# --------------------------------------------------------------------------------------- per donor


def donor_record(entry, base, load_delta, view, objective=OBJECTIVE, exact_identity=None):
    """Build D/R/ladders for one donor, contract them, and return its manifest row."""
    t0 = time.time()
    stored = load_delta(entry["donor_dir"])
    pred = None if entry["predecessor_is_base"] else load_delta(entry["predecessor_dir"])
    W, D, raw, edited = edit_increments(base, stored, pred)
    del stored, pred
    R, exclusions, ladders = variant_increments(W, D, raw, edited, entry["rotation_seed"])
    nonzero_D = {k: D[k] for k in edited if torch.count_nonzero(D[k])}
    row = {
        k: entry[k]
        for k in (
            "id",
            "family",
            "predecessor_id",
            "predecessor_is_base",
            "donor_bytes",
            "donor_via_alias",
            "predecessor_bytes",
            "predecessor_via_alias",
            "registered_chain_pick",
            "gradient_source",
            "canonical_id",
            "rotation_seed",
        )
    }
    flags = list(entry["flags"])
    row["n_edited_tensors"] = len(edited)
    row["edited_tensor_rule"] = (
        "stored delta keys" if entry["predecessor_is_base"] else "nonzero realised increment"
    )
    row["scale_source"] = (
        "raw_stored_delta" if entry["predecessor_is_base"] else "realised_increment"
    )
    row["hashes"] = {
        "D_eff": tensors_hash({k: D[k] for k in edited}),
        "D_eff_nonzero": tensors_hash(nonzero_D),
        "R_eff": tensors_hash(R) if R is not None else None,
        "ladder_0.25": tensors_hash(ladders[0.25]),
        "ladder_0.5": tensors_hash(ladders[0.5]),
    }
    row["rotation_exclusions"] = exclusions
    if exact_identity is not None:
        # F6.2 artifact: same predecessor and increment (G2's increment_sha256 hashes nonzero tensors).
        if exact_identity["pick"]["predecessor_id"] != entry["predecessor_id"]:
            raise ValueError(f"{entry['id']}: chain_exact predecessor differs")
        if exact_identity["increment_sha256"] != row["hashes"]["D_eff_nonzero"]:
            raise ValueError(f"{entry['id']}: realised increment differs from chain_exact's")
        if sorted(exact_identity["support"]) != sorted(nonzero_D):
            raise ValueError(f"{entry['id']}: chain_exact support differs from the increment")
        # Rotation and ladder supports lie inside D's changed blocks; the zero-D edited tensors of a
        # root pick have no exact gradient but contribute nothing (checked).
        edited_in = [k for k in edited if k in nonzero_D]
        for name, inc in [("R", R or {}), *((f"ladder_{s}", X) for s, X in ladders.items())]:
            stray = [k for k, x in inc.items() if k not in nonzero_D and torch.count_nonzero(x)]
            if stray:
                raise ValueError(f"{entry['id']}: {name} outside the exact-gradient support")
        D = {k: D[k] for k in edited_in}
        R = None if R is None else {k: R[k] for k in edited_in}
        ladders = {s: {k: X[k] for k in edited_in} for s, X in ladders.items()}
        edited = edited_in
    values = predictions(view, D, R, ladders, edited, objective)
    row["contractions_fp32"] = {k: v[0] for k, v in values.items()}
    row["contractions_fp64"] = {k: v[1] for k, v in values.items()}
    if R is None:
        flags.append("rotation-excluded")
        row["predictions"] = None
    else:
        row["predictions"] = [
            values["rotation"][0],
            values["ladder_0.25"][0],
            values["ladder_0.5"][0],
        ]
        if not all(math.isfinite(p) for p in row["predictions"]):
            raise ValueError(f"{entry['id']}: nonfinite prediction")
        signs64 = [values[c][1] for c in ("rotation", "ladder_0.25", "ladder_0.5")]
        if any(
            (a > 0) != (b > 0) or (a < 0) != (b < 0) for a, b in zip(row["predictions"], signs64)
        ):
            flags.append("fp32-fp64-sign-differs")
    row["flags"] = flags
    row["seconds"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------------------- loading


def load_field(directory, want_files=None):
    """FieldArtifact.load (content hashes) after checking the recorded file hashes."""
    directory = Path(directory)
    files = {f: sha256_file(directory / f) for f in ("manifest.json", "tensors.pt")}
    if want_files is not None and files != want_files:
        raise ValueError(f"{directory}: file hashes differ from the G2 manifest")
    field = rf.FieldArtifact.load(directory)
    return field, files


def load_base_state(expected_hash):
    """The pinned base model's bf16 state on CPU (the TF-v1/G2 loading path), checked by state hash."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL[0], revision=MODEL[1], dtype=torch.bfloat16, device_map="cpu"
    )
    state = model.state_dict()
    got = rf._hash({n: rf._tensor_hash(t) for n, t in state.items()})
    if got != expected_hash:
        raise ValueError(f"base state hash {got} differs from the field's {expected_hash}")
    state = {n: t.detach() for n, t in state.items()}
    del model
    return state, got


def code_identity():
    out = {f: sha256_file(ROOT / f) for f in CODE_FILES}
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
    p = path.resolve()
    if DATA_ROOT in p.parents and ALLOWED_OUT not in p.parents:
        raise ValueError(f"{path}: data outputs must stay under {ALLOWED_OUT}")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def sign_distribution(fields):
    out = {}
    for j, c in enumerate(CONTRASTS):
        values = [v[j] for v in fields.values()]
        out[c] = {
            "positive": sum(x > 0 for x in values),
            "negative": sum(x < 0 for x in values),
            "zero": sum(x == 0 for x in values),
        }
    patterns = Counter(
        "".join("+" if x > 0 else "-" if x < 0 else "0" for x in v) for v in fields.values()
    )
    out["patterns"] = dict(sorted(patterns.items()))
    return out


# ------------------------------------------------------------------------------------------- main


def run(args):
    t_start = time.time()
    check_out_path(args.out)
    inputs_bytes = args.inputs.read_bytes()
    inputs = json.loads(inputs_bytes)
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    comp_path = args.selection.parent / "components.jsonl"
    comp_bytes = comp_path.read_bytes()
    if hashlib.sha256(comp_bytes).hexdigest() != selection["components_jsonl_sha256"]:
        raise ValueError("components.jsonl differs from selection.json's pin")
    components = [json.loads(x) for x in comp_bytes.decode().splitlines() if x.strip()]
    plan = plan_donors(inputs, components, selection)
    zoo = Path(args.zoo)
    for e in plan:
        e["donor_dir"] = zoo / e["donor_bytes"]
        e["predecessor_dir"] = None if e["predecessor_is_base"] else zoo / e["predecessor_bytes"]

    g2 = Path(args.g2)
    g2_manifest_path = g2 / "g2_manifest.json"
    g2_manifest = json.loads(g2_manifest_path.read_text())
    frec = g2_manifest["fields"][OBJECTIVE]
    if frec.get("status") != "complete":
        raise ValueError("base WikiText field is not complete in the G2 manifest")
    ce = g2_manifest["chain_exact"]
    ce_rows = {
        r["pick_id"]: r
        for r in ce["rows"]
        if r["objective"] == OBJECTIVE and r.get("status") == "complete"
    }
    from wsp.zoo.apply import load_entry_delta

    log("loading the base WikiText field (file + content hashes)")
    field, base_files = load_field(g2 / "fields" / OBJECTIVE, frec["files"])
    base_identity = field.identity
    if base_identity["state_hash"] != frec["state_hash"]:
        raise ValueError("base field state hash differs from the G2 manifest")
    view = gradient_view(field)
    del field
    gc.collect()
    log("loading the pinned base model state")
    base, base_hash = load_base_state(base_identity["state_hash"])

    def load_delta(d):
        return load_entry_delta(d, base)

    rows = {}
    progress = args.out.with_name(args.out.stem + ".rows.jsonl")  # per-donor rows as they finish
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("")

    def record(row):
        rows[row["id"]] = row
        with progress.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        gc.collect()

    base_plan = [e for e in plan if e["gradient_source"] == SOURCE_BASE]
    exact_plan = [e for e in plan if e["gradient_source"] == SOURCE_EXACT]
    for n, e in enumerate(exact_plan, 1):  # first: their G2 cross-checks fail fast
        rec = ce_rows.get(e["id"])
        if rec is None:
            raise ValueError(f"{e['id']}: no complete chain_exact WikiText row")
        if rec["predecessor_id"] != e["predecessor_id"]:
            raise ValueError(f"{e['id']}: chain_exact predecessor differs")
        exact, files = load_field(rec["dir"], rec["files"])
        if exact.identity.get("base_state_hash") != base_hash:
            raise ValueError(f"{e['id']}: chain_exact base state hash differs")
        ex_view = gradient_view(exact)
        identity = exact.identity
        del exact
        row = donor_record(e, base, load_delta, ex_view, exact_identity=identity)
        if not math.isclose(
            row["contractions_fp32"]["g_dot_D"],
            rec["exact_contraction"],
            rel_tol=1e-5,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{e['id']}: <g, D> differs from chain_exact's exact_contraction")
        row["gradient"] = {
            "source": SOURCE_EXACT,
            "dir": rec["dir"],
            "anchor": "actual predecessor (exact gradient, F6.2)",
            "files": files,
            "identity_sha256": rec["identity_sha256"],
            "recorded_exact_contraction": rec["exact_contraction"],
            "predecessor_state_sha256": identity["predecessor_state_sha256"],
        }
        record(row)
        log(f"exact {n}/{len(exact_plan)} {e['id']} {row['predictions']} ({row['seconds']:.0f}s)")
        del ex_view
    for n, e in enumerate(base_plan, 1):
        row = donor_record(e, base, load_delta, view)
        row["gradient"] = {
            "source": SOURCE_BASE,
            "dir": str(g2 / "fields" / OBJECTIVE),
            "anchor": "base",
            "files": base_files,
        }
        record(row)
        log(f"base {n}/{len(base_plan)} {e['id']} {row['predictions']} ({row['seconds']:.0f}s)")
    del view
    gc.collect()

    excluded = sorted(i for i, r in rows.items() if r["predictions"] is None)
    if excluded:
        raise ValueError(
            f"rotation excluded for {excluded}: POWER-AND-YIELD needs three finite predictions per donor"
        )
    fields = {e["id"]: rows[e["id"]]["predictions"] for e in plan}
    fields_sha = write_json(args.out, dict(sorted(fields.items())))
    counts = Counter(r["gradient_source"] for r in rows.values())
    flagged = Counter(f for r in rows.values() for f in r["flags"])
    manifest = {
        "schema": SCHEMA,
        "basis": "PREREG_H1_TF A5.3.2 POWER-AND-YIELD, F2, F6.3/F6.4; Board A F6 review condition 5",
        "objective": OBJECTIVE,
        "contrast_order": list(CONTRASTS),
        "predictor": {
            "rotation": "<g, D_eff - R_eff>",
            "ladder": "<g, bf16(W_pred + s * delta) - W_pred>, s in (0.25, 0.5); delta = raw stored delta "
            "(root) or realised increment (chain step)",
            "contraction": "wsp.features.response_field.contract (signed, fp32 per tensor), summed with "
            "math.fsum; an fp64 recomputation is recorded for the sign check",
            "gradient_sources": {
                SOURCE_EXACT: "registered chain picks: F6.2 exact WikiText gradient at the actual predecessor",
                SOURCE_BASE: "root donors (predecessor = base): G2 base field (exact anchor); nonroot non-pick "
                f"chain donors: G2 base field, flagged '{FLAG_UNVALIDATED}' (F6.4)",
            },
            "rotation_seed": f"rotation_seed(canonical id): sha256('{ROTATION_DOMAIN}' + id)[:8] big-endian "
            "mod 2^63 (control_seeds.json rotation, A5.3.3)",
            "shortcuts": "equal block shapes within an edit share Haar frames (each tensor restarts the "
            "seeded generator state); rank_degenerate decided without SVD (unreachable for a finite "
            "nonzero block); both bitwise-equal to the registered functions (tests)",
            "audits": "A5.3.3 rotation and scaled-edit audits not computed (G3 staging outputs, F3); the "
            "NULL-SIZE payload's audit flags and masks are unchanged",
        },
        "inputs": {
            "nullsize_inputs": {
                "path": str(args.inputs),
                "sha256": hashlib.sha256(inputs_bytes).hexdigest(),
            },
            "selection": {
                "path": str(args.selection),
                "sha256": hashlib.sha256(selection_bytes).hexdigest(),
            },
            "components_jsonl_sha256": hashlib.sha256(comp_bytes).hexdigest(),
            "g2_manifest": {"path": str(g2_manifest_path), "sha256": sha256_file(g2_manifest_path)},
            "chain_exact_summary_sha256": ce.get("sha256"),
            "zoo": str(zoo),
            "payload_fields_read": "donors[].id/family and provenance.donors[].reference_id only",
        },
        "base": {
            "model": {"repo": MODEL[0], "revision": MODEL[1], "dtype": "bfloat16"},
            "state_hash": base_hash,
            "field_dir": str(g2 / "fields" / OBJECTIVE),
            "field_files": base_files,
            "field_content_hashes_sha256": frec["content_hashes_sha256"],
            "field_identity_sha256": frec["identity_sha256"],
        },
        "outcomes_read": "none: no TF-v1 window value, donor mean/noise, rotation, ladder, G3 or sealed "
        "outcome is read; only stored edit bytes, the base model and the G2 WikiText gradients",
        "nothing_written_under": ["data/zoo", "data/runs/g2"],
        "code": code_identity(),
        "fields_json": {"path": str(args.out), "sha256": fields_sha, "n_donors": len(fields)},
        "summary": {
            "n_donors": len(rows),
            "gradient_source_counts": dict(sorted(counts.items())),
            "flag_counts": dict(sorted(flagged.items())),
            "root_donors_on_base_field": sum(
                r["predecessor_is_base"] and r["gradient_source"] == SOURCE_BASE
                for r in rows.values()
            ),
            "f6_4_literal_non_pick_chain_rows": sum(
                r["family"] == "chain" and not r["registered_chain_pick"] for r in rows.values()
            ),
            "sign_distribution": sign_distribution(fields),
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall_seconds": time.time() - t_start,
        },
        "donors": {i: rows[i] for i in sorted(rows)},
    }
    if args.manifest_out:
        manifest_sha = write_json(args.manifest_out, manifest)
        log(f"manifest {args.manifest_out} sha256 {manifest_sha}")
    log(f"fields {args.out} sha256 {fields_sha}")
    log(json.dumps(manifest["summary"], indent=1))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--g2", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest-out", type=Path)
    p.add_argument("--zoo", default=str(ROOT / "data/zoo/qwen3.5-0.8b"))
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
