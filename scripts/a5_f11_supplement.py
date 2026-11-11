"""A5 amendment F11: supplementary procedural selection (k = 12 roots, four per depleted cell).

Applies the chair's frozen F11 rule (PROGRESS 11 Nov 2026, 07:20 UTC; PREREG_H1_TF F11.2) to committed
metadata only. It reads no TF-v1 record, no evaluation file and no G3 outcome, and uses no GPU.

Rule. A depleted cell is a registered (family, tertile) cell in which at least one rotation pair was excluded
at the F10 re-stage (`part_a/manifest.json:variants`, kind ``rotation``, ``excluded`` true). The depleted
cells must equal the frozen set (LoRA-high, chain-low, chain-high). Within each, candidates are taken in the
frozen A5.3.1 order extended past the quota: ``sorted(cell, key=canonical_edit_id)[quota:]``, where ``cell`` is
the registered representative list ``selection.json:registered.ledger.families.<family>.cells[tertile]``.
(The script checks that ``sorted(cell)[:quota]`` is exactly the registered picks of that cell.) The first four
candidates that pass every filter are selected:

* ``eligible``: ``components.jsonl`` row is ``eligibility == "eligible"``, ``closure_certified`` and a
  ``development`` component (the A5.3.1 inventory filter, re-checked);
* ``root``: predecessor is the base (``components.jsonl:parent_id`` null and
  ``fields_manifest.json:donors[id].predecessor_is_base``; a disagreement stops);
* ``exact gradient``: a ``fields_manifest.json`` donor with ``gradient_source == "base_field"``, no flags and
  listed in ``power/registered_donors.json`` (F7.1: the G2 base WikiText gradient is exact at a root);
* ``not alias-only``: the edit's own bytes are stored (``components.jsonl:bytes_present`` true and
  ``bytes_via_alias`` null; ``fields_manifest`` ``donor_via_alias`` false; a disagreement stops);
* not one of the forty registered edits, and a component not already represented.

k is fixed at 12 (four per cell). A cell with fewer than four passing candidates stops the selection: there is
no cross-cell replacement, top-up or reserve use (A5.3.1; F11.2). Candidates after the fourth pick are listed as
``reserve_unreached`` and are not evaluated.

Output (``--out``): the 12 ids by cell, the full candidate order with per-candidate reasons, the rule, the
sha256 of every input file, and ``sha256(json.dumps(sorted(ids)))`` (the F7.1 ``ids_sha256`` convention).
The output carries no timestamp, so reruns on the same inputs are byte-identical.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "a5-f11-supplement-v1"
PER_CELL = 4
FROZEN_CELLS = (("lora_sft", 2), ("chain", 0), ("chain", 2))
TERTILE_LABELS = ("low", "middle", "high")
FAMILY_LABELS = {"lora_sft": "LoRA", "chain": "chain"}
DEFAULTS = {
    "selection": "docs/prereg/a5_manifests/selection.json",
    "components": "docs/prereg/a5_manifests/components.jsonl",
    "fields_manifest": "docs/prereg/a5_manifests/power/fields_manifest.json",
    "registered_donors": "docs/prereg/a5_manifests/power/registered_donors.json",
    "part_a_manifest": "docs/prereg/a5_manifests/part_a/manifest.json",
}
RULE = (
    "F11.2: for each depleted cell (a registered (family, tertile) cell that lost at least one rotation pair at "
    "the F10 re-stage; frozen set LoRA-high, chain-low, chain-high), take the first four candidates in "
    "sorted(cell, key=canonical_edit_id)[quota:] (selection.json:registered.ledger.families.<f>.cells[t]; the "
    "A5.3.1 order extended past the quota) that are eligible development rows with certified closure, root "
    "edits (predecessor = base), exact-gradient donors (fields_manifest gradient_source base_field, no flags, "
    "listed in power/registered_donors.json), not alias-only (own bytes stored), not among the registered "
    "forty and of a component not already represented; k = 12 fixed; a shortage stops (no cross-cell "
    "replacement, no top-up, no reserve use)"
)
ORDER_KEY = (
    "canonical_edit_id (lexicographic), applied to the registered representative list of the cell; the "
    "A5.3.1 selection key sort(cell, key=canonical_edit_id)[0:quota], continued past the quota"
)
HASH_RULE = (
    "sha256 of json.dumps(sorted(ids)) (Python default separators), as power/registered_donors.json"
)


class SupplementError(RuntimeError):
    """The frozen rule cannot be applied to these inputs (stop; no replacement)."""


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids_sha256(ids):
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def _cell_name(family, tertile):
    return f"{family}:{tertile}"


def _load(selection, components, fields_manifest, registered_donors, part_a_manifest):
    sel = json.loads(Path(selection).read_text())
    rows = {}
    for line in Path(components).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    donors = json.loads(Path(fields_manifest).read_text())["donors"]
    pool = set(json.loads(Path(registered_donors).read_text())["donor_ids"])
    manifest = json.loads(Path(part_a_manifest).read_text())
    return sel, rows, donors, pool, manifest


def depleted_cells(manifest):
    """(family, tertile) -> sorted excluded rotation source ids, from the Part A manifest."""
    cell_of = {s["edit_id"]: (s["family"], int(s["tertile"])) for s in manifest["selections"]}
    out = {}
    for v in manifest["variants"]:
        if v.get("kind") == "rotation" and v.get("excluded"):
            sid = v["source_id"]
            if sid not in cell_of:
                raise SupplementError(f"excluded rotation {sid} has no selection row")
            out.setdefault(cell_of[sid], []).append(sid)
    return {c: sorted(ids) for c, ids in out.items()}


def candidate_order(sel, family, tertile):
    """Registered picks and the frozen next-in-cell order for one cell."""
    fam = sel["registered"]["ledger"]["families"][family]
    cell = sorted(fam["cells"][tertile])
    quota = fam["quotas"][tertile]
    registered = set(sel["procedural_freeze"]["registered_edit_ids"])
    picks = cell[:quota]
    in_cell_registered = sorted(i for i in cell if i in registered)
    if picks != in_cell_registered:
        raise SupplementError(
            f"{_cell_name(family, tertile)}: registered picks {in_cell_registered} are not the sorted "
            f"canonical-ID prefix {picks} of the cell; the frozen order does not hold"
        )
    return picks, quota, cell[quota:]


def _reasons(eid, rows, donors, pool, registered, used_components):
    r = rows.get(eid)
    if r is None:
        raise SupplementError(f"{eid}: no components.jsonl row")
    reasons = []
    if not (
        r.get("eligibility") == "eligible"
        and r.get("closure_certified") is True
        and r.get("component_status") == "development"
    ):
        reasons.append("not_eligible")
    d = donors.get(eid)
    root_c = r.get("parent_id") is None
    if d is not None and bool(d.get("predecessor_is_base")) != root_c:
        raise SupplementError(
            f"{eid}: components.jsonl parent_id {r.get('parent_id')!r} disagrees with fields_manifest "
            f"predecessor_is_base {d.get('predecessor_is_base')!r}"
        )
    if not root_c:
        reasons.append("not_root")
    exact = (
        d is not None
        and d.get("gradient_source") == "base_field"
        and not d.get("flags")
        and eid in pool
    )
    if not exact:
        reasons.append("not_exact_gradient")
    alias_c = (not r.get("bytes_present")) or r.get("bytes_via_alias") is not None
    if d is not None and bool(d.get("donor_via_alias")) != alias_c:
        raise SupplementError(
            f"{eid}: components.jsonl bytes_present/bytes_via_alias disagree with fields_manifest "
            f"donor_via_alias {d.get('donor_via_alias')!r}"
        )
    if alias_c:
        reasons.append("alias_only")
    if eid in registered:
        reasons.append("registered_forty")
    if r.get("component_id") in used_components:
        reasons.append("component_already_represented")
    return reasons


def build(
    selection,
    components,
    fields_manifest,
    registered_donors,
    part_a_manifest,
    expected_cells=FROZEN_CELLS,
    per_cell=PER_CELL,
):
    paths = {
        "selection": selection,
        "components": components,
        "fields_manifest": fields_manifest,
        "registered_donors": registered_donors,
        "part_a_manifest": part_a_manifest,
    }
    sel, rows, donors, pool, manifest = _load(**paths)
    registered = sorted(sel["procedural_freeze"]["registered_edit_ids"])
    reg_set = set(registered)
    if sorted(sel["registered"]["edit_ids"]) != registered:
        raise SupplementError(
            "selection.json registered.edit_ids != procedural_freeze.registered_edit_ids"
        )

    depleted = depleted_cells(manifest)
    expected = [tuple(c) for c in expected_cells]
    if sorted(depleted) != sorted(expected):
        raise SupplementError(
            f"depleted cells {sorted(depleted)} from the Part A manifest differ from the frozen set "
            f"{sorted(expected)}"
        )

    used_components = {rows[i]["component_id"] for i in registered if i in rows}
    cells_out, supplements = [], []
    for family, tertile in expected:
        name = _cell_name(family, tertile)
        picks, quota, order = candidate_order(sel, family, tertile)
        listed, chosen = [], []
        for pos, eid in enumerate(order, start=quota):
            if len(chosen) == per_cell:
                listed.append({"position": pos, "id": eid, "status": "reserve_unreached"})
                continue
            reasons = _reasons(eid, rows, donors, pool, reg_set, used_components)
            status = "selected" if not reasons else "skipped"
            listed.append({"position": pos, "id": eid, "status": status, "reasons": reasons})
            if not reasons:
                chosen.append(eid)
                used_components.add(rows[eid]["component_id"])
        if len(chosen) < per_cell:
            raise SupplementError(
                f"{name}: only {len(chosen)} of {per_cell} candidates pass the F11 filters; selection "
                "stops (no cross-cell replacement)"
            )
        cells_out.append(
            {
                "cell": name,
                "family": family,
                "tertile": tertile,
                "label": f"{FAMILY_LABELS.get(family, family)}-{TERTILE_LABELS[tertile]}",
                "quota": quota,
                "registered_picks": picks,
                "registered_rotation_excluded": depleted[(family, tertile)],
                "candidate_order": listed,
                "selected": chosen,
            }
        )
        for eid in chosen:
            r, d = rows[eid], donors[eid]
            supplements.append(
                {
                    "id": eid,
                    "cell": name,
                    "component_id": r["component_id"],
                    "effective_norm": r.get("effective_norm"),
                    "exactness": r.get("exactness"),
                    "predecessor_id": "base",
                    "gradient_source": d["gradient_source"],
                    "in_registered_donor_pool": True,
                }
            )

    ids = [s["id"] for s in supplements]
    if len(ids) != per_cell * len(expected) or len(set(ids)) != len(ids):
        raise SupplementError(f"expected {per_cell * len(expected)} distinct ids, got {ids}")
    n_excluded = sum(len(v) for v in depleted.values())
    return {
        "schema": SCHEMA,
        "basis": (
            "PREREG_H1_TF A5 amendment F11 (F11.2); chair decision PROGRESS 11 Nov 2026 07:20 UTC (Option A); "
            "metrology docs/council/2026-11-11/f11-metrology.md section 2"
        ),
        "rule": RULE,
        "order_key": ORDER_KEY,
        "k": len(ids),
        "per_cell": per_cell,
        "frozen_cells": [_cell_name(f, t) for f, t in expected],
        "registered_rotation_exclusions": {
            "n": n_excluded,
            "by_cell": {_cell_name(f, t): v for (f, t), v in sorted(depleted.items())},
        },
        "cells": cells_out,
        "supplements": supplements,
        "supplement_ids": ids,
        "supplement_ids_sorted": sorted(ids),
        "supplement_ids_sorted_sha256": _ids_sha256(ids),
        "ids_sha256_rule": HASH_RULE,
        "registered_edit_ids_sorted": registered,
        "inputs": {k: {"path": str(v), "sha256": _sha256_file(v)} for k, v in paths.items()},
        "outcomes_read": "none: metadata manifests only (no TF-v1 record, evaluation file or G3 outcome)",
        "no_top_up": True,
        "replaces_registered": False,
    }


def _parse_cells(text):
    out = []
    for item in text.split(","):
        fam, t = item.strip().rsplit(":", 1)
        out.append((fam, int(t)))
    return tuple(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument(
        "--expected-cells",
        default=",".join(_cell_name(f, t) for f, t in FROZEN_CELLS),
        help="frozen depleted cells (family:tertile, comma separated)",
    )
    ap.add_argument("--out", default="docs/prereg/a5_manifests/selection_f11.json")
    a = ap.parse_args(argv)
    try:
        out = build(
            a.selection,
            a.components,
            a.fields_manifest,
            a.registered_donors,
            a.part_a_manifest,
            expected_cells=_parse_cells(a.expected_cells),
        )
    except SupplementError as e:
        print(f"a5_f11_supplement: STOP: {e}", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for c in out["cells"]:
        print(f"{c['cell']} ({c['label']}): {', '.join(c['selected'])}")
    print(f"sorted-id sha256 {out['supplement_ids_sorted_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
