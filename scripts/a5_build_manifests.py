#!/usr/bin/env python3
"""A5.11 manifest package for Part A's 40-edit selection (PREREG_H1_TF appendix A5).

Regenerates ``docs/prereg/a5_manifests/`` deterministically from read-only inputs (CPU only, no timestamps,
no absolute paths in the output):

* ``components.jsonl`` — the frozen A5.3.1 inventory: one row per zoo manifest entry with its certified source
  component (docs/prereg/panel_manifest.json), alias-ledger canonical id, lineage exactness, incremental norm and
  outcome-independent eligibility. Its ``id/component_id/closure_certified/exclusion_reason/canonical_id`` fields
  are the sidecar format of :func:`wsp.eval.response_intervention.select_with_ledger`.
* ``selection.json`` — the output of ``select_edits`` (n_lora=30, n_chain=10) on that inventory, with its ledger,
  under the registered chair ruling (positively proven lost-predecessor exclusions); the literal pre-ruling run is
  kept as a superseded audit record. A block/shortage and the A5.3.1 action are recorded, never patched.
* ``references.json`` — per selected edit: predecessor id (or base), TF-v1 primary records on file with the same
  scoring identity (A5.3.4), closure proofs, n_D, n_P and priced minutes at 1.04 min per entry.
* ``text_groups.json`` — the 64 registered WikiText development windows (config + TF-v1 records) and their hash.
* ``code_hashes.json`` — sha256 of the pinned builders, validators/tests, imported selection/grouping/reference
  modules and configs (at ``--ref`` or, with ``--pin-worktree``, the working tree committed with the package),
  the resolved TF-v1 config hash, model/tokenizer revision and Python/dependency versions.
* ``README.md`` — every file with its sha256 and how it was produced.

The chair ruling on A5.3.1 is registered: a missing-norm chain step whose actual predecessor has a positive
lost proof (no stored and no aliased bytes in the artefact_status sidecar, :func:`lost_predecessor_proof`) is
excluded and does not block. ``--procedural-freeze`` (Board A checklist items 1-2) relabels and pins the
committed package and records the dated procedural freeze (inventory, cutpoints, forty ids, code pins); it does
not rebuild components, references or text groups against live data. A later full build must reproduce the
frozen ids and cutpoints (:func:`check_procedural_freeze`).

    python scripts/a5_build_manifests.py [--out docs/prereg/a5_manifests] [--ref main | --pin-worktree]
    python scripts/a5_build_manifests.py --procedural-freeze
    python scripts/a5_build_manifests.py --verify-pins HEAD
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.eval import response_intervention as ri
from wsp.eval.batch import effective_config_hash

N_LORA, N_CHAIN = 30, 10
VARIANTS_PER_EDIT = 3  # rotation at s = 1 and the edit at s = 0.25, 0.5 (S31 §3 G3)
TERTILES = ("low", "middle", "high")
# Pinned source (A5.11; Board A A5 package review item 7): builders, validators/tests, the imported
# selection/grouping/reference logic and the resolved configs. Role is descriptive only.
CODE_FILES = {
    "scripts/a5_build_manifests.py": "builder",
    "scripts/a5_build_panels.py": "builder",
    "scripts/a5_simulation.py": "simulation and NULL-SIZE input validator",
    "scripts/a5_incremental_norms.py": "producer of the incremental-norm input",
    "tests/test_a5_simulation.py": "test",
    "tests/test_a5_build_manifests.py": "test",
    "tests/test_a5_build_panels.py": "test",
    "src/wsp/eval/response_intervention.py": "selection, tertiles, reference identity (imported)",
    "src/wsp/eval/batch.py": "effective_config_hash (imported)",
    "src/wsp/eval/tf_v1.py": "TF-v1 rankings and panels (imported by the panel builder)",
    "src/wsp/eval/subsets.py": "dataset revisions and subset specs (imported by the panel builder)",
    "src/wsp/eval/run.py": "WikiText window construction (perplexity)",
    "src/wsp/features/rotation.py": "A5.3.3 rotation (imported)",
    "src/wsp/features/response_field.py": "G2 field and registered panel-size consumer check",
    "src/wsp/features/response_chart.py": "A5.8.2 chart",
    "configs/eval_tf_v1.yaml": "resolved TF-v1 config",
    "configs/editor_qwen3.5-0.8b.yaml": "model and revision",
}
ENV_PACKAGES = ("numpy", "torch", "transformers", "tokenizers", "datasets", "safetensors", "pyyaml")
PROCEDURAL_FREEZE_DATE = "2026-11-07"
RULING = (
    "REGISTERED: chair ruling A5.3.1 (7 Nov 2026), adopted by Board A "
    "(docs/reviews/council/board-a-a5-package-2026-11-07.md item 1) and the PREREG_H1_TF A5 freeze-prep "
    "amendments (7 Nov 2026): an edit whose actual predecessor has no stored and no aliased bytes (positive "
    "sidecar proof below) has no incremental feature manifest and is ineligible; it does not block selection"
)
# Board A's independent reconstruction (A5 package review item 1): the registered (norm, id) cutpoint keys.
BOARD_A_CUTPOINTS = {
    "lora_sft": [
        [1.0580346482037204, "f4b10903b8e09e24"],
        [2.2141261523655484, "f8c2cf0c6b9ffa8c"],
    ],
    "chain": [[0.7725259468377004, "7af5977862ec021f"], [1.3049782925751559, "14cdaabee246493b"]],
}


# ---------------------------------------------------------------------------------------------- helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def artefact_files(entry_dir: Path) -> list[Path]:
    """Stored bytes of a zoo entry, as scripts/d8_restore_verify.py:artefact_files."""
    dense = entry_dir / "delta.safetensors"
    if dense.exists():
        return [dense]
    return sorted(p for p in (entry_dir / "lora").glob("*") if p.is_file())


def entry_delta_hash(files: list[Path]) -> str | None:
    """Content hash used by docs/prereg/panel_manifest.json source identities (build_h1a_panel.py)."""
    if not files:
        return None
    return sha256_bytes("".join(f"{p.name}:{sha256_file(p)}\n" for p in files).encode())


# -------------------------------------------------------------------------------------------- inventory


def build_inventory(
    manifest_rows, status_rows, norm_rows, panel, feature_ids, bytes_present, root_norms
):
    """One row per manifest entry, sorted by id; eligibility is outcome-independent (A5.3.1).

    Effective norm (A5.3.1: the full realised incremental edit, not the raw adapter norm): for a
    root edit (no parent; predecessor = base) the feature manifest's ``norms.fro_effective`` of the
    base-relative ``effective_bf16`` delta (``root_norms``; the convention of
    ``response_intervention.load_feature_norms``); for a chain step the predecessor-relative
    ``inc_fro`` of scripts/a5_incremental_norms.py. ``inc_fro`` of a LoRA root is the raw
    ``(B @ A) * scaling`` product norm, so it is recorded but never used.

    Reason order follows ``select_with_ledger``: family, manifest eligibility, sidecar exclusion
    (missing eligibility, not a D3 development member, artefact lost), alias, feature manifest; an
    otherwise eligible row without certified closure or with a missing/nonfinite norm blocks.
    """
    status = {r["id"]: r for r in status_rows}
    norms = {r["id"]: r for r in norm_rows}
    comp_of, comp_status, dev = {}, {}, set()
    for c in panel["components"]:
        for m in c["members"]:
            if m in comp_of:
                raise ValueError(f"{m} is a member of two components")
            comp_of[m], comp_status[m] = c["component_id"], c["status"]
        dev.update(c["development_members"])
    canonical = {}
    for _key, g in sorted(panel["alias_ledger"].items()):
        for m in (g["canonical"], *g["aliases"]):
            canonical[m] = g["canonical"]

    rows = []
    for r in sorted((r for r in manifest_rows if "id" in r), key=lambda r: r["id"]):
        i, fam, s, nr = r["id"], r.get("family"), status.get(r["id"], {}), norms.get(r["id"], {})
        own = bool(bytes_present.get(i))
        lost = not own and not s.get("alias_of")
        sidecar_reason = None
        if r.get("eligibility") is None:
            sidecar_reason = (
                "eligibility:missing"  # the tool would default a missing value to production
            )
        elif i not in dev:
            sidecar_reason = f"not_d3_development_member:{comp_status.get(i, 'no_component')}"
        elif lost:
            sidecar_reason = "artefact_lost"
        if r.get("parent_id") is None:
            norm, source = root_norms.get(i), "features.fro_effective"
            cause = "no base-relative effective norm in the feature manifest"
        else:
            norm, source = nr.get("inc_fro"), "a5_incremental_norms.inc_fro"
            cause = nr.get("error", "no incremental norm row")
        if fam not in ri.FAMILIES:
            elig = f"excluded:unsupported_family:{fam}"
        elif r.get("eligibility", "production") != "production":
            elig = f"excluded:eligibility:{r.get('eligibility')}"
        elif sidecar_reason:
            elig = f"excluded:{sidecar_reason}"
        elif canonical.get(i, i) != i:
            elig = f"excluded:alias_of:{canonical[i]}"
        elif i not in feature_ids:
            elig = "excluded:no_feature_manifest"
        elif i not in comp_of:
            elig = "blocked:closure_not_certified"
        elif not finite(norm):
            elig = f"blocked:missing_incremental_norm ({cause})"
        else:
            elig = "eligible"
        rows.append(
            {
                "id": i,
                "family": fam,
                "parent_id": r.get("parent_id"),
                "component_id": comp_of.get(i),
                "component_status": comp_status.get(i),
                "closure_certified": i in comp_of,
                "canonical_id": canonical.get(i, i),
                "d3_development_member": i in dev,
                "bytes_present": own,
                "bytes_via_alias": None if own else s.get("alias_of"),
                "feature_manifest": i in feature_ids,
                "effective_norm": norm,
                "norm_source": source,
                "inc_fro": nr.get("inc_fro"),
                "base_norm": nr.get("base_fro"),
                "exactness": nr.get("exactness"),
                "exclusion_reason": sidecar_reason,
                "eligibility": elig,
            }
        )
    return rows


def lost_predecessor_proof(row, inventory_by_id, status_by_id):
    """Positive proof that ``row``'s actual predecessor has no stored and no aliased bytes, else None.

    Required (chair ruling A5.3.1; Board A A5 package review item 1): the row is a blocked
    missing-norm successor; its predecessor has an inventory row and an artefact_status sidecar row;
    no own bytes on disk; sidecar ``delta_present`` is explicitly false and ``alias_of`` is empty;
    and no alias-ledger holder of the predecessor has bytes. Anything less is not a proof, so the
    row keeps blocking selection. No error string is inspected.
    """
    parent = row.get("parent_id")
    if parent is None or not row["eligibility"].startswith("blocked:missing_incremental_norm"):
        return None
    pred, side = inventory_by_id.get(parent), status_by_id.get(parent)
    if pred is None or side is None:
        return None
    holder = pred["canonical_id"] if pred["canonical_id"] != parent else None
    holder_bytes = bool(holder and inventory_by_id.get(holder, {}).get("bytes_present"))
    if (
        pred["bytes_present"]
        or pred["bytes_via_alias"]
        or side.get("alias_of")
        or side.get("delta_present") is not False
        or holder_bytes
    ):
        return None
    return {
        "predecessor_id": parent,
        "predecessor_bytes_present": False,
        "predecessor_bytes_via_alias": None,
        "sidecar": {
            "delta_present": side.get("delta_present"),
            "alias_of": side.get("alias_of"),
            "artefact_lost": side.get("artefact_lost"),
        },
        "alias_ledger_holder": holder,
        "alias_holder_bytes_present": holder_bytes,
    }


def lost_predecessor_exclusions(inventory, status_rows):
    """{edit id: (exclusion reason, proof)} for every positively proven lost predecessor."""
    by_id = {r["id"]: r for r in inventory}
    status = {r["id"]: r for r in status_rows}
    out = {}
    for r in inventory:
        proof = lost_predecessor_proof(r, by_id, status)
        if proof is not None:
            out[r["id"]] = (f"actual_predecessor_state_lost:{r['parent_id']}", proof)
    return out


# -------------------------------------------------------------------------------------------- selection


def run_selection(manifest_rows, inventory, n_lora, n_chain, extra_exclusions=None):
    """``select_with_ledger`` on the inventory sidecar; blocks and shortages are recorded, never patched."""
    extra = extra_exclusions or {}
    sidecar = [
        {
            "id": r["id"],
            "component_id": r["component_id"],
            "closure_certified": r["closure_certified"],
            "canonical_id": r["canonical_id"],
            **(
                {"exclusion_reason": extra.get(r["id"]) or r["exclusion_reason"]}
                if (extra.get(r["id"]) or r["exclusion_reason"])
                else {}
            ),
        }
        for r in inventory
    ]
    norms = {r["id"]: r["effective_norm"] for r in inventory if r["feature_manifest"]}
    expected_blocked = sorted(
        r["id"]
        for r in inventory
        if r["eligibility"].startswith("blocked") and r["id"] not in extra
    )
    out = {"n_lora": n_lora, "n_chain": n_chain, "quotas": {}}
    out["quotas"] = {"lora_sft": ri.tertile_quotas(n_lora), "chain": ri.tertile_quotas(n_chain)}
    try:
        selected, ledger = ri.select_with_ledger(manifest_rows, norms, sidecar, n_lora, n_chain)
    except ri.SelectionBlocked as e:
        if not expected_blocked:
            raise
        return out | {
            "status": "blocked",
            "error": str(e),
            "blocking_ids": expected_blocked,
            "action": "A5.3.1: missing/nonfinite norms or uncertified sources block selection; "
            "no row is dropped and no selection is made until resolved by amendment/ruling",
        }
    except ri.SelectionUnresolved as e:
        return out | {
            "status": "unresolved_shortage",
            "error": str(e),
            "ledger": e.ledger,
            "action": "A5.3.1: STOP unresolved; no cross-cell replacement, no resampling",
        }
    if expected_blocked:
        raise AssertionError(f"tool did not block on {expected_blocked}")
    assert ri.select_edits(manifest_rows, norms, sidecar, n_lora, n_chain) == selected
    by_id = {r["id"]: r for r in inventory}
    rejected_ids = {r["id"] for r in ledger["rejected"]}
    for r in inventory:  # rows the tool kept are exactly the inventory's eligible rows
        if r["id"] not in rejected_ids and r["eligibility"] != "eligible":
            raise AssertionError(f"{r['id']}: kept by the tool but {r['eligibility']}")
    for rej in ledger["rejected"]:  # the tool's reasons agree with the frozen inventory
        if rej["reason"].startswith("component_represented_by:"):
            continue
        want = f"excluded:{rej['reason']}"
        if extra.get(rej["id"]):
            continue
        if by_id[rej["id"]]["eligibility"] != want:
            raise AssertionError(f"{rej['id']}: tool {rej['reason']} vs {by_id[rej['id']]}")
    rows = []
    for s in selected:
        rows.append(
            asdict(s)
            | {
                "tertile_label": TERTILES[s.tertile],
                "exactness": by_id[s.edit_id]["exactness"],
                "canonical_id": by_id[s.edit_id]["canonical_id"],
                "bytes_present": by_id[s.edit_id]["bytes_present"],
                "bytes_via_alias": by_id[s.edit_id]["bytes_via_alias"],
            }
        )
    counts = Counter((s.family, TERTILES[s.tertile]) for s in selected)
    return out | {
        "status": "selected",
        "selected": rows,
        "edit_ids": sorted(s.edit_id for s in selected),
        "component_ids": sorted(s.component_id for s in selected),
        "counts": {f: {t: counts[(f, t)] for t in TERTILES} for f in ri.FAMILIES},
        "ledger": ledger,
    }


# ------------------------------------------------------------------------------------------- references


def wikitext_windows(records_path: Path):
    rows = []
    for line in records_path.read_text().splitlines():
        if '"wikitext_ppl"' not in line:
            continue
        r = json.loads(line)
        if r.get("benchmark") == "wikitext_ppl":
            rows.append((r["item_id"], (r.get("extra") or {}).get("token_ids_hash")))
    return sorted(rows, key=lambda x: int(x[0]))


def reference_status(evals: Path, ref_id: str, source: Path | None, identity, windows):
    """A5.3.4 reuse proof: the tool's identity check plus primary-record and window identity."""
    d = evals / ref_id
    repeats = sorted(p.name for p in (d / "repeats").glob("*") if p.is_dir())
    out = {"id": ref_id, "record": f"{ref_id}/eval.json", "repeats_on_file": repeats}
    if not (d / "eval.json").exists():
        return out | {"exists": False, "on_file": False, "failed_checks": ["no_record"]}
    p = json.loads((d / "eval.json").read_text())
    current = ri._source_hashes(source) if source is not None and source.exists() else None
    checks = {
        "complete": p.get("eval_status") == "complete",
        "primary": p.get("repeat_id") is None,
        "config_hash": p.get("config_hash") == identity["config_hash"],
        "model_revision": p.get("model_revision") == identity["model_revision"],
        "counts": all(p.get("counts", {}).get(k) == n for k, n in identity["counts"].items()),
        "entry_hashes": source is None
        or (bool(p.get("entry_hashes")) and p.get("entry_hashes") == current),
        "windows": (d / "records.jsonl").exists()
        and wikitext_windows(d / "records.jsonl") == windows,
    }
    tool = ri._reference_on_file(evals, ref_id, source, identity)
    core = all(v for k, v in checks.items() if k not in ("primary", "windows"))
    if tool != core:
        raise AssertionError(f"{ref_id}: tool on_file={tool} disagrees with checks {checks}")
    return out | {
        "exists": True,
        "on_file": all(checks.values()),
        "failed_checks": sorted(k for k, v in checks.items() if not v),
        "run_id": p.get("run_id"),
        "config_hash": p.get("config_hash"),
        "model_revision": p.get("model_revision"),
        "recorded_entry_hashes": p.get("entry_hashes"),
        "current_entry_hashes": current,
        "eval_json_sha256": sha256_file(d / "eval.json"),
        "records_jsonl_sha256": sha256_file(d / "records.jsonl")
        if (d / "records.jsonl").exists()
        else None,
    }


def closure_proof(zoo: Path, entry_id: str, inv_row, panel_ids):
    """Component membership plus whether the current bytes' content hash is a registered source identity."""
    holder = entry_id if inv_row["bytes_present"] else inv_row["bytes_via_alias"]
    h = entry_delta_hash(artefact_files(zoo / holder)) if holder else None
    return {
        "component_id": inv_row["component_id"],
        "closure_certified": inv_row["closure_certified"],
        "bytes_holder": holder,
        "entry_delta_hash": h,
        "hash_in_source_identities": h is not None
        and f"delta:{h}" in panel_ids.get(inv_row["component_id"], ()),
        "exactness": inv_row["exactness"],
    }


def build_references(selection, manifest_rows, inventory, zoo, evals, cfg, panel, windows):
    if selection.get("status") != "selected":
        return None
    by_row = {r["id"]: r for r in manifest_rows if "id" in r}
    inv = {r["id"]: r for r in inventory}
    panel_ids = {c["component_id"]: set(c["source_identities"]) for c in panel["components"]}
    counts = {k: v.get("n", v.get("n_windows")) for k, v in cfg["benchmarks"].items()}
    chash = effective_config_hash(cfg)
    cache = {}

    def status(ref_id, revision):
        if (ref_id, revision) not in cache:
            identity = {"config_hash": chash, "model_revision": revision, "counts": counts}
            holder = None
            if ref_id != "base":  # an alias's current zoo bytes are its holder's bytes
                row = inv[ref_id]
                holder = ref_id if row["bytes_present"] else row["bytes_via_alias"] or ref_id
            source = None if holder is None else zoo / holder
            st = reference_status(evals, ref_id, source, identity, windows)
            cache[(ref_id, revision)] = st | {"bytes_holder": holder}
        return cache[(ref_id, revision)]

    rows, preds, origs = [], set(), set()
    for s in sorted(selection["selected"], key=lambda s: s["edit_id"]):
        e, pred = s["edit_id"], s["predecessor_id"]
        rev = by_row[e].get("base_revision")
        orig = status(e, rev)
        pst = status(pred, rev)
        preds.add((pred, rev))
        origs.add((e, rev))
        rows.append(
            {
                "edit_id": e,
                "family": s["family"],
                "tertile": s["tertile_label"],
                "model_revision": rev,
                "predecessor_id": pred,
                "original": orig,
                "original_missing": not orig["on_file"],
                "predecessor": pst,
                "predecessor_missing": not pst["on_file"],
                "predecessor_bytes_present": pred == "base" or inv[pred]["bytes_present"],
                "closure_proof": closure_proof(zoo, e, inv[e], panel_ids),
                "predecessor_closure_proof": None
                if pred == "base"
                else closure_proof(zoo, pred, inv[pred], panel_ids),
            }
        )
    missing_d = sorted(e for e, rev in origs if not cache[(e, rev)]["on_file"])
    missing_p = sorted({p for p, rev in preds if not cache[(p, rev)]["on_file"]})
    n_var = VARIANTS_PER_EDIT * len(rows)
    n_d, n_p = len(missing_d), len(missing_p)
    return {
        "schema": "a5-references-v1",
        "selection_status": "selected",
        "identity": {"config_hash": chash, "counts": counts},
        "rows": rows,
        "missing_originals": missing_d,
        "missing_predecessors": missing_p,
        "predecessors_without_own_bytes": sorted(
            {r["predecessor_id"] for r in rows if not r["predecessor_bytes_present"]}
        ),
        "n_D": n_d,
        "n_P": n_p,
        "n_variants": n_var,
        "minutes_per_entry": ri.MINUTES_PER_ENTRY,
        "references_to_score": n_d + n_p,
        "additional_minutes": round((n_d + n_p) * ri.MINUTES_PER_ENTRY, 4),
        "total_minutes": round((n_var + n_d + n_p) * ri.MINUTES_PER_ENTRY, 4),
        "not_priced": "preprocessing and retries are charged separately (A5.3.4 stop-and-reprice)",
    }


# ------------------------------------------------------------------------------------------ text groups


def record_dirs(evals: Path) -> list[Path]:
    """TF-v1 primary record directories, listed once per build (records may be added while it runs)."""
    return sorted(
        d
        for d in evals.iterdir()
        if d.is_dir() and (d / "eval.json").exists() and (d / "records.jsonl").exists()
    )


def build_text_groups(evals: Path, cfg, dirs=None):
    spec = cfg["benchmarks"]["wikitext_ppl"]
    n, window, rev = spec["n_windows"], spec["window"], spec.get("revision")
    seen, disagree = {}, []
    for d in record_dirs(evals) if dirs is None else dirs:
        w = wikitext_windows(d / "records.jsonl")
        if w:
            seen[d.name] = w
    distinct = sorted({json.dumps(w) for w in seen.values()})
    if len(distinct) != 1:
        disagree = sorted(seen)
    ref = json.loads(distinct[0]) if distinct else []
    ids_ok = [x[0] for x in ref] == [str(i) for i in range(n)]
    windows = [
        {
            "window_id": f"Salesforce/wikitext:wikitext-2-raw-v1:test@{rev}:window{window}:{i}",
            "item_id": item,
            "token_start": int(item) * window,
            "token_end": (int(item) + 1) * window,
            "token_ids_hash": h,
        }
        for i, (item, h) in enumerate(ref)
    ]
    sorted_ids = sorted(w["window_id"] for w in windows)
    return {
        "schema": "a5-text-groups-v1",
        "source": {
            "config": "configs/eval_tf_v1.yaml",
            "dataset": "Salesforce/wikitext wikitext-2-raw-v1 split=test",
            "revision": rev,
            "window": window,
            "n_windows": n,
            "construction": "tokenizer.encode('\\n\\n'.join(test texts)); window i = tokens "
            "[i*window, (i+1)*window) (src/wsp/eval/run.py:perplexity)",
        },
        "windows": windows,
        "window_ids_sorted_sha256": sha256_bytes(json.dumps(sorted_ids).encode()),
        "token_ids_hashes_sorted_sha256": sha256_bytes(
            json.dumps(sorted(w["token_ids_hash"] for w in windows)).encode()
        ),
        "hash_convention": "sha256 of json.dumps(sorted list of strings), default separators",
        "two_window_blocks": [[2 * k, 2 * k + 1] for k in range(n // 2)],
        "blocks_source": "scripts/a5_simulation.py:contrasts reshapes the 64 windows to (32, 2) in item order",
        "records_checked": sorted(seen),
        "records_consistent": len(distinct) == 1 and ids_ok and len(ref) == n,
        "records_disagreeing": disagree,
        "disjointness": "Every auxiliary gradient, factor and probe panel (A5.2.1; S31 §3) must be "
        "text-group disjoint from these registered development windows, i.e. use none of the token "
        f"spans listed here (test-split tokens [0, {n * window}) at this revision), and from the "
        "calibration and sealed scored items. Those panels are not defined in this package; their "
        "disjointness certificates are still required before G2.",
    }


# ----------------------------------------------------------------------------------------- code hashes


def git_blob_reader(root: Path, ref: str):
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", ref], capture_output=True, text=True, check=True
    ).stdout.strip()

    def read(path):
        cmd = ["git", "-C", str(root), "show", f"{sha}:{path}"]
        res = subprocess.run(cmd, capture_output=True, check=False)
        return res.stdout if res.returncode == 0 else None

    return sha, read


def worktree_reader(root: Path):
    """Pin the working-tree bytes (to be committed unchanged with the package); base = HEAD."""
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    def read(path):
        p = Path(root) / path
        return p.read_bytes() if p.is_file() else None

    return sha or None, read


def environment_identity():
    """Interpreter and dependency versions of the build environment (package metadata, no imports)."""
    from importlib import metadata

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {"python": platform.python_version()} | {p: version(p) for p in ENV_PACKAGES}


def model_identity(root: Path):
    """Model and tokenizer repo/revision from the pinned editor config (the tokenizer ships with it)."""
    import yaml

    p = Path(root) / "configs/editor_qwen3.5-0.8b.yaml"
    if not p.exists():
        return None
    cfg = yaml.safe_load(p.read_bytes())
    ident = {"repo": cfg.get("base_model"), "revision": cfg.get("base_revision")}
    return {
        "model": ident,
        "tokenizer": dict(ident),
        "source": "configs/editor_qwen3.5-0.8b.yaml base_model/base_revision",
    }


def resolved_config(cfg, config_bytes: bytes, records_config_hash):
    """The resolved TF-v1 config: file bytes and effective_config_hash as the TF-v1 records store it."""
    chash = effective_config_hash(cfg)
    return {
        "path": "configs/eval_tf_v1.yaml",
        "sha256": sha256_bytes(config_bytes),
        "effective_config_hash": chash,
        "hash_function": "wsp.eval.batch.effective_config_hash (eval.json config_hash)",
        "records_config_hash": records_config_hash,
        "records_match": records_config_hash == chash,
    }


def build_code_hashes(pinned_from, git_sha, read_blob, *, environment, model, config):
    files = {}
    for path, role in CODE_FILES.items():
        blob = read_blob(path)
        files[path] = {"role": role, "sha256": None if blob is None else sha256_bytes(blob)}
    return {
        "schema": "a5-code-hashes-v2",
        "pinned_from": pinned_from,
        "git_sha": git_sha,
        "files": files,
        "resolved_config": config,
        "model": model,
        "environment": environment,
        "note": "sha256 of the pinned file bytes; a null sha256 means the file is absent. 'working tree' "
        "pins are the bytes committed together with this file (git_sha is their parent): verify with "
        "`git show <commit>:<path> | sha256sum`. No flag here describes any later tree.",
    }


def verify_code_hashes(code, read_blob):
    """Paths whose bytes under ``read_blob`` differ from the pins (empty list = all match)."""
    return sorted(
        p
        for p, e in code["files"].items()
        if (None if (b := read_blob(p)) is None else sha256_bytes(b)) != e["sha256"]
    )


# ------------------------------------------------------------------------------------- registered selection


def with_ruling(selection_run, exclusions):
    """The registered A5.3.1 run with the chair ruling's positively proven exclusions recorded."""
    return {
        "ruling": RULING,
        "registered_exclusions": {
            i: {"reason": reason, "proof": proof}
            for i, (reason, proof) in sorted(exclusions.items())
        },
    } | selection_run


def assemble_selection(inputs, components_sha, literal, registered, freeze=None):
    out = {
        "schema": "a5-selection-v2",
        "algorithm": "wsp.eval.response_intervention.select_with_ledger / select_edits (A5.3.1)",
        "inputs": inputs,
        "inputs_scope": "partial input index, not a full immutable snapshot: feature-file contents and "
        "donor/record contents are not hashed here (Board A A5 package review item 1; checklist item 4)",
        "components_jsonl_sha256": components_sha,
        "registered": registered,
        "pre_ruling_literal": None
        if not registered["registered_exclusions"]
        else {
            "status_label": "SUPERSEDED: literal A5.3.1 run before the chair ruling; audit record only",
        }
        | literal,
    }
    if freeze is not None:
        out["procedural_freeze"] = freeze
    return out


def reproduce_selection(inventory, quotas):
    """Independent A5.3.1 reproduction from components.jsonl (global dedup, (norm, id) cutpoints)."""
    reps = {}
    for r in sorted(
        (r for r in inventory if r["eligibility"] == "eligible"), key=lambda r: r["id"]
    ):
        reps.setdefault(r["component_id"], r)
    out = {}
    for family, quota in quotas.items():
        ordered = sorted(
            (r for r in reps.values() if r["family"] == family),
            key=lambda r: (r["effective_norm"], r["id"]),
        )
        n = len(ordered)
        b1, b2 = n // 3, 2 * n // 3
        cells = [ordered[:b1], ordered[b1:b2], ordered[b2:]]
        picks = [sorted(c, key=lambda r: r["id"])[:q] for c, q in zip(cells, quota)]
        out[family] = {
            "N": n,
            "b1": b1,
            "b2": b2,
            "cutpoints": [
                [ordered[b]["effective_norm"], ordered[b]["id"]] for b in (b1, b2) if b < n
            ],
            "selected": [[r["id"] for r in p] for p in picks],
        }
    return out


def procedural_freeze_block(selection, inventory, code_bytes: bytes):
    """Distinct, dated procedural freeze: inventory, cutpoints, the forty ids and the code pins."""
    reg = selection["registered"]
    if reg.get("status") != "selected":
        raise ValueError(
            f"no procedural freeze without a registered selection ({reg.get('status')})"
        )
    again = reproduce_selection(inventory, reg["quotas"])
    ids = sorted(i for f in again.values() for cell in f["selected"] for i in cell)
    if ids != reg["edit_ids"]:
        raise AssertionError("independent reproduction differs from the registered selection")
    cutpoints = {}
    for family, f in again.items():
        ledger = reg["ledger"]["families"][family]
        if [list(c) for c in ledger["cutpoints"]] != f["cutpoints"]:
            raise AssertionError(f"{family}: ledger cutpoints differ from the reproduction")
        cutpoints[family] = {
            "N_representatives": f["N"],
            "b1": f["b1"],
            "b2": f["b2"],
            "keys": f["cutpoints"],
            "quota": reg["quotas"][family],
        }
    by_id = {r["id"]: r for r in inventory}
    return {
        "label": "procedural freeze",
        "date": PROCEDURAL_FREEZE_DATE,
        "basis": "Board A A5 package review (docs/reviews/council/board-a-a5-package-2026-11-07.md) "
        "item 8, checklist items 1-2; PREREG_H1_TF A5 freeze-prep amendments (7 Nov 2026)",
        "separate_from": "the later pre-G2 evidence freeze (checklist items 3-7: closed G0/G1/G6 snapshot, "
        "regenerated references/text groups/panels, NULL-SIZE payload and report). No evidence hash "
        "exists at this date and none is claimed",
        "inventory": {
            "components_jsonl_sha256": selection["components_jsonl_sha256"],
            "rows": len(inventory),
            "eligible": sum(r["eligibility"] == "eligible" for r in inventory),
            "registered_exclusions": sorted(reg["registered_exclusions"]),
            "inputs": "selection.json:inputs (partial index; see inputs_scope)",
        },
        "cutpoints": cutpoints,
        "cutpoint_rule": "(effective_norm, canonical_edit_id) keys at ranks floor(N/3) and floor(2N/3) of "
        "each family's globally deduplicated representatives; a key equal to a cutpoint belongs to the "
        "higher half-open cell",
        "board_a_cutpoints_match": cutpoints
        and {f: c["keys"] for f, c in cutpoints.items()} == BOARD_A_CUTPOINTS,
        "registered_edit_ids": reg["edit_ids"],
        "registered_edit_ids_sorted_sha256": sha256_bytes(json.dumps(reg["edit_ids"]).encode()),
        "registered_component_ids_sorted_sha256": sha256_bytes(
            json.dumps(reg["component_ids"]).encode()
        ),
        "counts": reg["counts"],
        "retrained_lineage_picks": sorted(
            i for i in reg["edit_ids"] if by_id[i]["exactness"] == "re-trained"
        ),
        "code_hashes_json_sha256": sha256_bytes(code_bytes),
        "not_pinned_at_this_freeze": [
            "feature-file contents and donor/record content hashes (checklist item 4)",
            "closed G0/G1/G6 census, reference and exposure/cost ledger snapshot (checklist item 3)",
            "regenerated references/text_groups/panels against that snapshot (checklist items 4-5)",
            "NULL-SIZE payload, its validation and report (checklist item 6)",
        ],
    }


def check_procedural_freeze(selection, freeze):
    """A later full build must reproduce the frozen forty ids and cutpoints exactly (checklist item 4)."""
    reg = selection["registered"]
    if reg.get("status") != "selected" or reg["edit_ids"] != freeze["registered_edit_ids"]:
        raise AssertionError("registered selection differs from the procedural freeze")
    for family, c in freeze["cutpoints"].items():
        if [list(k) for k in reg["ledger"]["families"][family]["cutpoints"]] != c["keys"]:
            raise AssertionError(f"{family}: cutpoints differ from the procedural freeze")
    if len(set(reg["component_ids"])) != len(reg["edit_ids"]):
        raise AssertionError("selected components are not distinct")


# -------------------------------------------------------------------------------------------------- build

REFERENCES_BASIS = "registered (A5.3.1 with the registered chair ruling; see selection.json)"


def build(
    paths: dict,
    n_lora=N_LORA,
    n_chain=N_CHAIN,
    ref="main",
    git_sha=None,
    read_blob=None,
    *,
    pin_worktree=False,
    environment=None,
    procedural_freeze=None,
):
    """Return {filename: bytes} for the package. ``paths``: zoo, norms, features, evals, panel, config, root.

    ``procedural_freeze``: the committed freeze block; the build must reproduce its ids and cutpoints.
    """
    import yaml

    root, zoo = Path(paths["root"]), Path(paths["zoo"])
    manifest_rows = read_jsonl(zoo / "manifest.jsonl")
    status_rows = read_jsonl(zoo / "artefact_status.jsonl")
    norm_rows = read_jsonl(paths["norms"])
    panel = json.loads(Path(paths["panel"]).read_text())
    config_bytes = Path(paths["config"]).read_bytes()
    cfg = yaml.safe_load(config_bytes)
    features = Path(paths["features"])
    ids = [r["id"] for r in manifest_rows if "id" in r]
    feature_ids = {i for i in ids if (features / i / "features.json").exists()}
    bytes_present = {i: bool(artefact_files(zoo / i)) for i in ids}

    root_norms = {}
    for r in manifest_rows:
        is_root = "id" in r and r.get("parent_id") is None and r.get("family") in ri.FAMILIES
        if is_root and r["id"] in feature_ids:
            meta = json.loads((features / r["id"] / "features.json").read_text())
            ok = meta.get("delta_relative_to") == "base"
            root_norms[r["id"]] = (meta.get("norms") or {}).get("fro_effective") if ok else None
    inventory = build_inventory(
        manifest_rows, status_rows, norm_rows, panel, feature_ids, bytes_present, root_norms
    )
    literal = run_selection(manifest_rows, inventory, n_lora, n_chain)
    exclusions = lost_predecessor_exclusions(inventory, status_rows)
    run = (
        run_selection(
            manifest_rows, inventory, n_lora, n_chain, {i: e[0] for i, e in exclusions.items()}
        )
        if exclusions
        else literal
    )
    registered = with_ruling(run, exclusions)
    evals = Path(paths["evals"])
    dirs = record_dirs(evals)
    record_index = [[d.name, sha256_file(d / "eval.json")] for d in dirs]
    text_groups = build_text_groups(evals, cfg, dirs)
    windows = [(w["item_id"], w["token_ids_hash"]) for w in text_groups["windows"]]
    references = build_references(
        registered, manifest_rows, inventory, zoo, evals, cfg, panel, windows
    )
    if references is not None:
        references["selection_basis"] = REFERENCES_BASIS
    else:
        references = {
            "schema": "a5-references-v1",
            "selection_status": registered["status"],
            "note": "no selection, so no references can be enumerated",
        }

    def rel(p):
        p = Path(p)
        try:
            return str(p.relative_to(root))
        except ValueError:
            return p.name

    inputs = {
        "zoo_manifest": {
            "path": rel(zoo / "manifest.jsonl"),
            "sha256": sha256_file(zoo / "manifest.jsonl"),
        },
        "artefact_status": {
            "path": rel(zoo / "artefact_status.jsonl"),
            "sha256": sha256_file(zoo / "artefact_status.jsonl"),
        },
        "incremental_norms": {"path": rel(paths["norms"]), "sha256": sha256_file(paths["norms"])},
        "panel_manifest": {"path": rel(paths["panel"]), "sha256": sha256_file(paths["panel"])},
        "eval_config": {"path": rel(paths["config"]), "sha256": sha256_file(paths["config"])},
        "feature_manifests": {
            "path": rel(features),
            "n_available": len(feature_ids),
            "ids_sha256": sha256_bytes(json.dumps(sorted(feature_ids)).encode()),
        },
        "tf_v1_records": {
            "path": rel(evals),
            "n_primary_eval_json": len(record_index),
            "index_sha256": sha256_bytes(json.dumps(record_index).encode()),
            "index_convention": "sha256 of json.dumps(sorted [[dir, sha256(eval.json)]]) over dirs "
            "with eval.json and records.jsonl, listed once per build",
        },
    }
    if git_sha is None or read_blob is None:
        git_sha, read_blob = worktree_reader(root) if pin_worktree else git_blob_reader(root, ref)
    code = build_code_hashes(
        "working tree" if pin_worktree else f"git:{ref}",
        git_sha,
        read_blob,
        environment=environment_identity() if environment is None else environment,
        model=model_identity(root),
        config=resolved_config(cfg, config_bytes, effective_config_hash(cfg)),
    )
    files = {
        "components.jsonl": b"".join(
            (json.dumps(r, sort_keys=True) + "\n").encode() for r in inventory
        ),
    }
    selection = assemble_selection(
        inputs, sha256_bytes(files["components.jsonl"]), literal, registered, procedural_freeze
    )
    if procedural_freeze is not None:
        check_procedural_freeze(selection, procedural_freeze)
    files["selection.json"] = dumps(selection)
    files["references.json"] = dumps(references)
    files["text_groups.json"] = dumps(text_groups)
    files["code_hashes.json"] = dumps(code)
    files["README.md"] = readme(files, inventory, selection, references, text_groups)
    return files


def build_procedural_freeze(out: Path, root: Path, status_path: Path, *, environment=None):
    """Checklist items 1-2: registered labels, code/config/model/environment pins and the dated procedural
    freeze, from the COMMITTED package. References, text groups, panels and components are not rebuilt
    against live data; the only data read is the artefact_status sidecar (read-only), whose sha256 must
    equal ``selection.json:inputs.artefact_status``, for the positive lost-predecessor proof.
    Returns {filename: bytes} for selection.json, references.json, code_hashes.json and README.md.
    """
    import yaml

    out, root = Path(out), Path(root)
    comp_bytes = (out / "components.jsonl").read_bytes()
    inventory = [json.loads(x) for x in comp_bytes.decode().splitlines() if x.strip()]
    committed = json.loads((out / "selection.json").read_text())
    references = json.loads((out / "references.json").read_text())
    text_groups = json.loads((out / "text_groups.json").read_text())
    if sha256_bytes(comp_bytes) != committed["components_jsonl_sha256"]:
        raise ValueError("components.jsonl differs from selection.json:components_jsonl_sha256")
    if sha256_file(status_path) != committed["inputs"]["artefact_status"]["sha256"]:
        raise ValueError("artefact_status sidecar differs from the committed input hash; stop")
    if "conditional" in committed:  # a5-selection-v1: the ruling's run was labelled conditional
        literal = committed["registered"]
        cond = committed["conditional"] or {}
        prior = cond.get("extra_exclusions", {})
        body = {k: v for k, v in cond.items() if k not in ("ruling", "extra_exclusions")}
    else:
        literal = {
            k: v for k, v in (committed["pre_ruling_literal"] or {}).items() if k != "status_label"
        } or None
        prior = {
            i: e["reason"] for i, e in committed["registered"]["registered_exclusions"].items()
        }
        body = {
            k: v
            for k, v in committed["registered"].items()
            if k not in ("ruling", "registered_exclusions")
        }
    exclusions = lost_predecessor_exclusions(inventory, read_jsonl(status_path))
    if {i: e[0] for i, e in exclusions.items()} != prior:
        raise AssertionError(f"positive lost-predecessor proof {sorted(exclusions)} != {prior}")
    if literal and set(literal.get("blocking_ids", [])) != set(exclusions):
        raise AssertionError("the registered exclusions do not account for every literal block")
    registered = with_ruling(body, exclusions)
    if references.get("selection_status") == "selected":
        references["selection_basis"] = REFERENCES_BASIS
    cfg = yaml.safe_load((root / "configs/eval_tf_v1.yaml").read_bytes())
    git_sha, read_blob = worktree_reader(root)
    code = build_code_hashes(
        "working tree",
        git_sha,
        read_blob,
        environment=environment_identity() if environment is None else environment,
        model=model_identity(root),
        config=resolved_config(
            cfg,
            (root / "configs/eval_tf_v1.yaml").read_bytes(),
            references.get("identity", {}).get("config_hash"),
        ),
    )
    code_bytes = dumps(code)
    selection = assemble_selection(
        committed["inputs"], committed["components_jsonl_sha256"], literal, registered
    )
    selection["procedural_freeze"] = procedural_freeze_block(selection, inventory, code_bytes)
    files = {
        "components.jsonl": comp_bytes,
        "selection.json": dumps(selection),
        "references.json": dumps(references),
        "text_groups.json": (out / "text_groups.json").read_bytes(),
        "code_hashes.json": code_bytes,
    }
    files["README.md"] = readme(files, inventory, selection, references, text_groups)
    del files["components.jsonl"], files["text_groups.json"]
    return files


def _counts_table(sel):
    lines = ["| family | low | middle | high | total |", "|---|---|---|---|---|"]
    for f in ri.FAMILIES:
        c = sel["counts"][f]
        lines.append(f"| {f} | {c['low']} | {c['middle']} | {c['high']} | {sum(c.values())} |")
    return lines


def readme(files, inventory, selection, references, text_groups):
    shas = {k: sha256_bytes(v) for k, v in sorted(files.items())}
    registered, literal = selection["registered"], selection.get("pre_ruling_literal")
    freeze = selection.get("procedural_freeze")
    excluded = registered.get("registered_exclusions", {})
    supported = [r for r in inventory if r["family"] in ri.FAMILIES]

    def reason(r):
        if r["id"] in excluded:
            return "excluded:actual_predecessor_state_lost"
        return ":".join(
            r["eligibility"].split(" (")[0].split(":")[: 2 if "alias_of" in r["eligibility"] else 3]
        )

    by_reason = Counter(reason(r) for r in supported)
    reason_rows = []
    for k, v in sorted(by_reason.items()):
        rule = k.split(":")[1] if ":" in k else k
        reason_rows.append(
            f"| `{k if len(k) < 60 else k[:57] + '...'}` | {v} | {RULES.get(rule, '')} |"
        )
    root_gap = max(
        (
            abs(r["inc_fro"] - r["effective_norm"]) / r["effective_norm"]
            for r in supported
            if r["eligibility"] == "eligible"
            and r["norm_source"] == "features.fro_effective"
            and finite(r["inc_fro"])
            and r["effective_norm"]
        ),
        default=0.0,
    )
    flagged = Counter(
        (r["family"], r["exactness"]) for r in supported if r["eligibility"] == "eligible"
    )
    out = [
        "# A5.11 manifest package — Part A selection",
        "",
        "Generated by `python scripts/a5_build_manifests.py` (CPU, read-only on `data/`); rerunning it on the same",
        "inputs and git ref reproduces every byte. Do not edit by hand. Inputs and their sha256 are in",
        "`selection.json:inputs` (a partial input index, not a full immutable snapshot); code, config, model and",
        "environment identities are in `code_hashes.json`. `--procedural-freeze` regenerates only the labels,",
        "pins and the dated procedural freeze from the committed files (no rebuild against live data).",
        "",
        "| file | sha256 | produced from |",
        "|---|---|---|",
    ]
    produced = {
        "components.jsonl": "zoo manifest + artefact_status sidecar + panel_manifest components/alias ledger "
        "+ a5_incremental_norms + feature-manifest availability",
        "selection.json": "`select_with_ledger`/`select_edits` on components.jsonl (n_lora=30, n_chain=10) "
        "with the registered lost-predecessor exclusion; procedural freeze",
        "references.json": "TF-v1 records (`data/evals/qwen3.5-0.8b/tf_v1`) checked with "
        "`response_intervention._reference_on_file` + primary/window checks; current zoo bytes",
        "text_groups.json": "`configs/eval_tf_v1.yaml` wikitext_ppl spec + every TF-v1 records.jsonl",
        "code_hashes.json": "pinned file bytes (`pinned_from`), resolved config, model/tokenizer revision, "
        "Python and dependency versions",
    }
    for k in (
        "components.jsonl",
        "selection.json",
        "references.json",
        "text_groups.json",
        "code_hashes.json",
    ):
        out.append(f"| `{k}` | `{shas[k]}` | {produced[k]} |")
    out += [
        "",
        "## Eligibility (A5.3.1; lora_sft and chain rows only — other families are `unsupported_family`)",
        "",
        "| eligibility | rows | rule |",
        "|---|---|---|",
        *reason_rows,
        "",
        "Effective norm (A5.3.1: full realised incremental edit, not the raw adapter norm): root edits use",
        "the feature manifest's `norms.fro_effective` (base-relative `effective_bf16` delta); chain steps",
        "use the predecessor-relative `inc_fro` of `a5_incremental_norms.jsonl`. That file's `inc_fro` for a",
        "LoRA root is the raw `(B @ A) * scaling` norm and is recorded but not used; among eligible roots it",
        f"differs from the effective norm by up to {root_gap:.1%} (relative).",
        "",
        "Lineage exactness among eligible rows (flagged, never an exclusion by itself): "
        + ", ".join(f"{f}/{e}: {n}" for (f, e), n in sorted(flagged.items()))
        + ".",
        "",
        "## Registered selection",
        "",
        f"Status: **{registered['status']}** (A5.3.1 with the registered chair ruling, 7 Nov 2026).",
    ]
    if excluded:
        out += [
            "",
            "Registered exclusions (positive proof: the actual predecessor has no stored and no aliased bytes "
            "in the artefact_status sidecar and no alias-ledger holder with bytes): "
            + ", ".join(f"`{i}` ({e['reason']})" for i, e in sorted(excluded.items()))
            + ".",
        ]
    if registered["status"] == "selected":
        out += ["", *_counts_table(registered)]
    else:
        out += [
            "",
            f"Reason: {registered.get('error')}",
            "",
            f"Action: {registered.get('action')}.",
        ]
    if freeze:
        out += [
            "",
            f"## Procedural freeze ({freeze['date']})",
            "",
            (
                "Distinct from, and earlier than, the pre-G2 evidence freeze; no evidence hash is "
                f"claimed. Forty ids sha256 `{freeze['registered_edit_ids_sorted_sha256']}`; "
                f"`code_hashes.json` sha256 `{freeze['code_hashes_json_sha256']}`."
            ),
            "",
            "| family | N representatives | cutpoint keys (effective_norm, id) | quota |",
            "|---|---|---|---|",
            *(
                f"| {f} | {c['N_representatives']} | "
                + "; ".join(f"({k[0]!r}, `{k[1]}`)" for k in c["keys"])
                + f" | {'/'.join(map(str, c['quota']))} |"
                for f, c in freeze["cutpoints"].items()
            ),
            "",
            f"Board A's reconstructed cutpoints match: {freeze['board_a_cutpoints_match']}. Re-trained lineage "
            "picks (retained, development-exposed): "
            + (", ".join(f"`{i}`" for i in freeze["retrained_lineage_picks"]) or "none")
            + ".",
        ]
    if literal:
        out += [
            "",
            "## Pre-ruling literal run (superseded; audit record only)",
            "",
            f"Status: {literal['status']}. Reason: {literal.get('error')}",
        ]
    if "n_D" in references:
        out += [
            "",
            "## References",
            "",
            f"Selection basis: {references['selection_basis']}.",
            "",
            (
                f"n_D = {references['n_D']} missing original records, n_P = {references['n_P']} missing "
                f"unique predecessors: {references['references_to_score']} references to score, "
                f"{references['additional_minutes']} min at {references['minutes_per_entry']} min/entry; "
                f"total with the {references['n_variants']} variants {references['total_minutes']} min. "
                "Preprocessing and retries are not included. Stale-build estimate, not a final cost."
            ),
            "",
            "Predecessors without their own bytes in the zoo (G3 staging must load the alias holder): "
            + (", ".join(references["predecessors_without_own_bytes"]) or "none")
            + ".",
        ]
    out += [
        "",
        "## Text groups",
        "",
        (
            "64 registered WikiText windows; sha256 of the sorted window ids "
            f"`{text_groups['window_ids_sorted_sha256']}`; records consistent: "
            f"{text_groups['records_consistent']} ({len(text_groups['records_checked'])} TF-v1 records checked)."
        ),
        "Gradient/factor/probe panels must be text-group disjoint from these windows; they are not defined here.",
        "",
        "## Not certified by this package",
        "",
        *(BLOCKED_NOTE if registered["status"] != "selected" else []),
        *NOT_CERTIFIED,
        "",
    ]
    return "\n".join(out).encode()


RULES = {
    "eligible": "passes every A5.3.1 inventory rule",
    "unsupported_family": "A5.3.1 supported family (lora_sft, chain)",
    "eligibility": "zoo manifest eligibility must be `production` (missing/smoke excluded)",
    "not_d3_development_member": "S31 §3 G3 selects D3 edits: only panel development members (D3-evaluated); "
    "confirmatory (sealed, A5.9) and fresh_unselected components are never exposed",
    "artefact_lost": "no stored bytes and no alias holder (D8 artefact_status)",
    "alias_of": "alias-ledger alias: the component is represented through its canonical id",
    "no_feature_manifest": "A5.3.1 pre-outcome feature-manifest availability (`features/<id>/features.json`)",
    "missing_incremental_norm": "A5.3.1: a missing/nonfinite norm of an otherwise eligible, reconstructible "
    "edit BLOCKS selection (not dropped)",
    "actual_predecessor_state_lost": "REGISTERED chair ruling A5.3.1: actual predecessor has no stored and no "
    "aliased bytes (positive sidecar proof); ineligible, does not block",
    "closure_not_certified": "A5.3.1: uncertified source closure BLOCKS selection",
}

BLOCKED_NOTE = [
    "- The registered A5.3.1 run did not select (see above); no G3 launch may proceed until it does.",
]
NOT_CERTIFIED = [
    "- Closure is component membership in `docs/prereg/panel_manifest.json` (its manifest snapshot predates the",
    "  12 heretic_ara entries added since); `hash_in_source_identities` in references.json shows whether current",
    "  bytes match a registered source identity. Re-trained chain bytes cannot match the original identity.",
    "  The G6 closure must be reconciled before the evidence freeze (checklist item 3).",
    "- references.json and text_groups.json reflect the TF-v1 records on file at build time",
    "  (`selection.json:inputs.tf_v1_records`); they are regenerated with this script against the closed",
    "  snapshot after the G6 census and before G3 (checklist items 3-4).",
    "- The eight alias reuse proofs and the tensor-coverage / single-matrix chain eligibility / rotation audits",
    "  (A5.3.3) are G3 staging outputs. Their algorithms, inputs, output schemas, responsible gate and",
    "  fail-closed policy are frozen in A5 (freeze-prep amendments, 7 Nov 2026); the outputs are not here.",
    "- G0/G1 overlap, retries and preprocessing costs (A5.2.2) are not included.",
    "- Gradient/factor/probe panels, disjointness certificates, control seeds and the NULL-SIZE input",
    "  specification are in the companion section below (scripts/a5_build_panels.py); the NULL-SIZE payload,",
    "  its report and Board A acceptance of the pre-G2 evidence are outstanding.",
]


COMPANION_BEGIN, COMPANION_END = b"<!-- a5_build_panels:begin -->", b"<!-- a5_build_panels:end -->"


def keep_companion_section(old: bytes, new: bytes) -> bytes:
    """Carry the README section written by scripts/a5_build_panels.py over a regeneration."""
    if COMPANION_BEGIN not in old or COMPANION_END not in old:
        return new
    start = old.index(COMPANION_BEGIN)
    end = old.index(COMPANION_END) + len(COMPANION_END)
    return new.rstrip(b"\n") + b"\n\n" + old[start:end] + b"\n"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = ROOT / "data"
    p.add_argument("--zoo", default=data / "zoo/qwen3.5-0.8b", type=Path)
    p.add_argument("--norms", default=data / "runs/d8/a5_incremental_norms.jsonl", type=Path)
    p.add_argument("--features", default=data / "features/qwen3.5-0.8b", type=Path)
    p.add_argument("--evals", default=data / "evals/qwen3.5-0.8b/tf_v1", type=Path)
    p.add_argument("--panel", default=ROOT / "docs/prereg/panel_manifest.json", type=Path)
    p.add_argument("--config", default=ROOT / "configs/eval_tf_v1.yaml", type=Path)
    p.add_argument("--out", default=ROOT / "docs/prereg/a5_manifests", type=Path)
    p.add_argument("--ref", default="main", help="git ref whose sha and file bytes are pinned")
    p.add_argument("--n-lora", type=int, default=N_LORA)
    p.add_argument("--n-chain", type=int, default=N_CHAIN)
    p.add_argument(
        "--pin-worktree",
        action="store_true",
        help="pin working-tree bytes (committed with the package) instead of --ref blobs",
    )
    p.add_argument(
        "--procedural-freeze",
        action="store_true",
        help="checklist items 1-2: registered labels, pins and the dated procedural freeze from the "
        "committed package; reads only the artefact_status sidecar; no live rebuild",
    )
    p.add_argument(
        "--verify-pins",
        metavar="REF",
        help="compare code_hashes.json in --out with the file bytes at REF and exit",
    )
    a = p.parse_args(argv)
    out = a.out.resolve()
    if out.is_relative_to(data.resolve()) or out.is_relative_to(Path("/home/ubuntu/wsp-data")):
        p.error("refusing to write under data/ (read-only)")
    paths = {
        "root": ROOT,
        "zoo": a.zoo,
        "norms": a.norms,
        "features": a.features,
        "evals": a.evals,
        "panel": a.panel,
        "config": a.config,
    }
    if a.verify_pins:
        _, read = git_blob_reader(ROOT, a.verify_pins)
        bad = verify_code_hashes(json.loads((out / "code_hashes.json").read_text()), read)
        print("differs: " + ", ".join(bad) if bad else "all pins match")
        return 1 if bad else 0
    if a.procedural_freeze:
        files = build_procedural_freeze(out, ROOT, a.zoo / "artefact_status.jsonl")
        freeze = json.loads(files["selection.json"])["procedural_freeze"]
        code = json.loads(files["code_hashes.json"])
        if not freeze["board_a_cutpoints_match"] or not code["resolved_config"]["records_match"]:
            p.error("cutpoints or resolved config hash do not match the registered values; stop")
    else:
        sel_path = out / "selection.json"
        frozen = (
            json.loads(sel_path.read_text()).get("procedural_freeze") if sel_path.exists() else None
        )
        files = build(
            paths, a.n_lora, a.n_chain, a.ref, pin_worktree=a.pin_worktree, procedural_freeze=frozen
        )
    out.mkdir(parents=True, exist_ok=True)
    old_readme = out / "README.md"
    if old_readme.exists():
        files["README.md"] = keep_companion_section(old_readme.read_bytes(), files["README.md"])
    for name, blob in files.items():
        (out / name).write_bytes(blob)
        print(f"{sha256_bytes(blob)}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
