#!/usr/bin/env python3
"""TF-v1 scoring plan with alias-ledger resolution (PREREG_H1_TF amendment A2.5, item 1).

Outcome-free. Reads the sealed partition manifest (docs/prereg/panel_manifest.json), the zoo
manifest and the *membership* of the D3 evaluations (an eval.json exists), never a score.

Why this exists. 101 effective deltas are stored under more than one entry id (a chain step 0 is
byte-identical to a standalone random_dir / abliteration delta). The scored id of a group is its
canonical (lexicographically smallest) id, which for 54 of the 101 groups is a standalone
random_dir / abliteration, not the chain step 0 that is the H1a stratum row. Label and reference
lookups therefore resolve through the alias ledger:

    scored_id(i)    = canonical(i)                       (i itself when not aliased)
    reference(i)    = scored_id(parent_id(i))  for a chain entry with a parent,
                      "base"                    otherwise (convention frozen in the manifest)

Usage: tfv1_scoring_plan.py [--out PATH] [--ids-file-prefix PREFIX]
Prints the dry-run counts; with --ids-file-prefix writes <prefix>.stratum.txt (D3 stratum rows plus
the chain predecessors they need), <prefix>.null.txt (D3 null-family rows) and
<prefix>.fresh_spares.txt (spare components, NOT scored before the section 5 gate).
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/prereg/panel_manifest.json"
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
EVALS = ROOT / "data/evals/qwen3.5-0.8b"
STRATUM = ("lora_sft", "task_vector", "chain")


def canonical_map(alias_ledger):
    """member id -> canonical id for every aliased entry (canonical maps to itself)."""
    out = {}
    for group in alias_ledger.values():
        for member in [group["canonical"], *group["aliases"]]:
            out[member] = group["canonical"]
    return out


def scored_id(entry_id, amap):
    return amap.get(entry_id, entry_id)


def reference_of(entry_id, rows, amap):
    """Scored id of the predecessor-relative reference, or "base"."""
    row = rows[entry_id]
    if row["family"] == "chain" and row.get("parent_id") in rows:
        return scored_id(row["parent_id"], amap)
    return "base"


def load_zoo_rows(zoo=ZOO):
    rows = {}
    for line in (zoo / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "id" in row:
            rows[row["id"]] = row
    return rows


def evaluation_membership(rows, evals=EVALS):
    return {i for i in rows if (evals / i / "eval.json").exists()}


def membership_sha(evaluated):
    return hashlib.sha256(json.dumps(sorted(evaluated)).encode()).hexdigest()


def plan(manifest, rows, evaluated):
    """Deterministic scoring lists. Every list holds scored (canonical) ids, each exactly once."""
    amap = canonical_map(manifest["alias_ledger"])
    components = {c["component_id"]: c for c in manifest["components"]}
    in_status = lambda status: [c for c in components.values() if c["status"] == status]
    d3 = sorted(evaluated)
    d3_stratum = [i for i in d3 if rows[i]["family"] in STRATUM]
    d3_null = [i for i in d3 if rows[i]["family"] not in STRATUM]
    need = {scored_id(i, amap) for i in d3_stratum}
    predecessors = {
        reference_of(i, rows, amap) for i in d3_stratum if rows[i]["family"] == "chain"
    } - {"base"}
    stratum_list = sorted(need | predecessors)
    null_list = sorted({scored_id(i, amap) for i in d3_null} - set(stratum_list))
    confirm = [m for c in in_status("confirmatory") for m in c["members"]]
    spare_members = [
        m for cid in manifest.get("spare_component_ids", []) for m in components[cid]["members"]
    ]
    return {
        "alias_map": amap,
        "d3_entries": d3,
        "d3_stratum_entries": d3_stratum,
        "d3_null_entries": d3_null,
        "predecessor_only": sorted(predecessors - need),
        "stratum_list": stratum_list,
        "null_list": null_list,
        "d3_unique": sorted({scored_id(i, amap) for i in d3} | predecessors),
        "fresh_unique": sorted({scored_id(i, amap) for i in confirm}),
        "fresh_spares_unique": sorted({scored_id(i, amap) for i in spare_members}),
        "fresh_without_spares_unique": sorted(
            {scored_id(i, amap) for i in confirm} - {scored_id(i, amap) for i in spare_members}
        ),
    }


def summary(p, rows):
    fam = lambda ids: dict(sorted(Counter(rows[i]["family"] for i in ids).items()))
    return {
        "d3_entries": len(p["d3_entries"]),
        "d3_stratum_entries": len(p["d3_stratum_entries"]),
        "d3_stratum_by_family": fam(p["d3_stratum_entries"]),
        "d3_null_entries": len(p["d3_null_entries"]),
        "chain_predecessors_not_themselves_d3_stratum": len(p["predecessor_only"]),
        "stratum_list_scored_ids": len(p["stratum_list"]),
        "null_list_scored_ids": len(p["null_list"]),
        "d3_unique_scored_ids": len(p["d3_unique"]),
        "fresh_unique_scored_ids": len(p["fresh_unique"]),
        "fresh_spares_unique_scored_ids": len(p["fresh_spares_unique"]),
        "fresh_without_spares_unique_scored_ids": len(p["fresh_without_spares_unique"]),
        "alias_groups_whose_canonical_is_not_the_stratum_row": sum(
            1 for g in set(p["alias_map"].values()) if rows[g]["family"] not in STRATUM
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--ids-file-prefix", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = load_zoo_rows()
    evaluated = evaluation_membership(rows)
    sealed = manifest["snapshot"]["evaluation_membership_sha256"]
    if membership_sha(evaluated) != sealed:
        raise SystemExit("D3 evaluation membership differs from the sealed snapshot")
    p = plan(manifest, rows, evaluated)
    print(json.dumps(summary(p, rows), indent=1))
    if args.ids_file_prefix:
        prefix = args.ids_file_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for name, ids in (
            ("stratum", p["stratum_list"]),
            ("null", p["null_list"]),
            ("fresh_spares", p["fresh_spares_unique"]),
        ):
            Path(f"{prefix}.{name}.txt").write_text("\n".join(ids) + "\n")


if __name__ == "__main__":
    main()
