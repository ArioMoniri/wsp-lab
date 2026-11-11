#!/usr/bin/env python3
"""F12 population, strata, outer folds and purchase order (PREREG_H1_TF F12.2, F12.5; Board A F12 review 11 Nov 2026).

Executable selector for the F12 rows, read only from the committed A5 manifests
(``docs/prereg/a5_manifests/{components.jsonl,snapshot.json,selection.json,selection_f11.json}``); no evaluation record,
TF-v1 score or outcome is opened (``snapshot.json`` is read for record *membership* and delta hashes only).

* **Part A components** (excluded): ``selection.json`` -> ``registered.component_ids`` plus the components of
  ``selection_f11.json`` -> ``supplement_ids`` (52; F11.5).
* **Existing roots** (source ``existing-TF-v1``): entries with a TF-v1 primary record (``snapshot.json`` ->
  ``content.tf_v1_records.primary``), root (``parent_id`` null), outside the Part A components. Board A adopted the
  270-row reading: the 44 G6-exposed roots of ``confirmatory`` components are reused as development observations
  only (flag ``g6_exposed_confirmatory``); no confirmatory entry is newly scored.
* **Purchasable roots** (source ``purchasable``): no TF-v1 primary record, ``component_status`` ``development``, outside
  Part A, ``bytes_present`` true, root, in a component holding **no** TF-v1 primary record (the mandatory
  component-level filter).
* **Strata**: a component's family; a component holding two families is ``mixed``.
* **Outer folds** (F12.2): within each stratum components ordered by sha256(UTF-8 ``20261111:f12:`` + component id)
  ascending, strata concatenated in the order abliteration, chain, lora_sft, mixed, random_dir, components assigned
  round-robin to folds 0-4 by global position. The component id is the 64-hex ``component_id`` of components.jsonl.
* **Purchase order and S_k** (F12.5): purchasable components ordered by sha256(``20261111:f12:purchase:`` + component
  id) ascending; S_k = existing components + the first k purchasable components, k in {0, 29, 58, 87, 116, 145},
  folds recomputed on S_k by the same rule.
* **Bytes and deltas**: a row's bytes holder is ``bytes_via_alias`` or itself; its delta hash is the holder's
  ``content.zoo_bytes.entry_delta_hash`` (a null own hash resolves through the alias, never pooled as one delta).
  Rows sharing a resolved delta are flagged (``shared_delta_with``) and stay logical rows.

Every count and hash the amendment and the Board A review state is recomputed and compared; a discrepancy is reported
in ``reproduction.discrepancies`` (never repaired) and makes the script exit non-zero unless ``--allow-discrepancy``.

    python scripts/f12_population.py [--out docs/prereg/a5_manifests/f12/population.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "docs/prereg/a5_manifests"
DEFAULT_OUT = MANIFESTS / "f12/population.json"
SCHEMA = "f12-population-v1"
FOLD_DOMAIN = "20261111:f12:"
PURCHASE_DOMAIN = "20261111:f12:purchase:"
N_FOLDS = 5
STRATUM_ORDER = ("abliteration", "chain", "lora_sft", "mixed", "random_dir")
K_GRID = (0, 29, 58, 87, 116, 145)
SOURCE_EXISTING = "existing-TF-v1"
SOURCE_PURCHASABLE = "purchasable"
EXPOSURE_GATES = ("G0", "G1", "G6")

# F12.2 / Board A table (11 Nov 2026): the registered counts and sorted-id hashes this selector must reproduce.
EXPECTED = {
    "existing": {
        "rows": 123,
        "components": 89,
        "families": {"chain": 105, "lora_sft": 10, "abliteration": 4, "random_dir": 4},
        "sha256": "f96e4ea55f4e887abdd713e97b27a06eec2b80fba82a397fbe5d6b9a36ca2160",
    },
    "purchasable": {
        "rows": 147,
        "components": 145,
        "families": {"abliteration": 79, "random_dir": 49, "lora_sft": 19},
        "sha256": "d31b9686178eca25230f6e77ab61676aad2b14296b4f0e9612d9eb9bc003dba0",
    },
    "population": {
        "rows": 270,
        "components": 234,
        "families": {"chain": 105, "abliteration": 83, "random_dir": 53, "lora_sft": 29},
        "sha256": "bd695b4ee6590156d2264b5f4c8dca252a1e63aced35d927a0d940d0eadd6348",
    },
    "literal_226": {
        "rows": 226,
        "components": 197,
        "sha256": "dc687e52f89c01f6f5878f9f150e91b165bb504dc6cf2d0f68c644599b93d5e3",
    },
    "part_a_components": 52,
    "g6_exposed_confirmatory": {"rows": 44, "components": 37},
    "alias_rows": 45,
    "distinct_resolved_deltas": 267,
    "shared_delta_pairs": [
        ["4b67b0aa12cce3e5", "db5c3a7bfc19178c"],
        ["4f31558031fe08c8", "adf416337f5a2277"],
        ["8cd3182b6e55dcbf", "bccb9ab10b5661e0"],
    ],
    "strata_components": {
        "abliteration": 83,
        "chain": 71,
        "random_dir": 50,
        "lora_sft": 29,
        "mixed": 1,
    },
    "existing_with_exposure_row": 117,
    "rows_per_component": {"1": 214, "2": 12, "3": 2, "4": 4, "5": 2},
    "no_component_filter_rows": 194,
    "no_component_filter_components": 179,
    # F12.6 (a): the 16 eligible purchasable lora_sft roots of the additivity panel.
    "f12i_lora_roots": [
        "64b10bbaef2b726d",
        "6cf90a51c4244720",
        "81ffbb2850280a20",
        "823f35704d34f1d8",
        "8289fa52f33a8d0d",
        "89a781074d330a7e",
        "94804c08e9e63a25",
        "a202c469c4f60027",
        "af6664b98264308c",
        "d904e32c5c8bc52e",
        "e0235707664804ef",
        "f3e1be8e4fe2c66b",
        "f4b10903b8e09e24",
        "f66b8fbbfa3e43d0",
        "f8c2cf0c6b9ffa8c",
        "fc796b847b42a34b",
    ],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_sha256(ids) -> str:
    """F12.2 rule: sha256 of the UTF-8 ``json.dumps`` (Python defaults) of the sorted id list."""
    return sha256_bytes(json.dumps(sorted(ids)).encode())


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def order_key(domain: str, component_id: str) -> str:
    return sha256_bytes(f"{domain}{component_id}".encode())


# ------------------------------------------------------------------------------------------ selection


def part_a_components(selection, selection_f11, by_id) -> set:
    comps = set(selection["registered"]["component_ids"])
    for i in selection_f11["supplement_ids"]:
        comps.add(by_id[i]["component_id"])
    return comps


def select(components, primary_ids, excluded_components, component_filter=True):
    """(existing, purchasable) root rows of F12.2. ``component_filter`` False drops only the component-level
    no-primary filter (the Board A sensitivity count; never a population)."""
    primary_ids = set(primary_ids)
    comp_with_primary = {r["component_id"] for r in components if r["id"] in primary_ids}
    existing, purchasable = [], []
    for r in components:
        if r.get("parent_id") is not None or r["component_id"] in excluded_components:
            continue
        if r["id"] in primary_ids:
            existing.append(r)
        elif (
            r.get("component_status") == "development"
            and r.get("bytes_present") is True
            and (not component_filter or r["component_id"] not in comp_with_primary)
        ):
            purchasable.append(r)
    key = lambda r: r["id"]
    return sorted(existing, key=key), sorted(purchasable, key=key)


def strata(rows) -> dict:
    """Component -> stratum: its family, or ``mixed`` if it holds more than one family."""
    fams = {}
    for r in rows:
        fams.setdefault(r["component_id"], set()).add(r["family"])
    out = {}
    for c, f in fams.items():
        if len(f) == 1:
            out[c] = next(iter(f))
        else:
            out[c] = "mixed"
    unknown = set(out.values()) - set(STRATUM_ORDER)
    if unknown:
        raise ValueError(f"strata outside the registered order: {sorted(unknown)}")
    return out


def assign_folds(stratum_of: dict, domain: str = FOLD_DOMAIN, n_folds: int = N_FOLDS):
    """F12.2: hash-sort within stratum, concatenate strata in STRATUM_ORDER, round-robin by global position.

    Returns (fold_of component, ordered list of {component_id, stratum, position, fold, order_key})."""
    ordered = []
    for s in STRATUM_ORDER:
        comps = sorted(
            (c for c, st in stratum_of.items() if st == s), key=lambda c: order_key(domain, c)
        )
        ordered.extend((c, s) for c in comps)
    fold_of, listing = {}, []
    for pos, (c, s) in enumerate(ordered):
        fold_of[c] = pos % n_folds
        listing.append(
            {
                "component_id": c,
                "stratum": s,
                "position": pos,
                "fold": pos % n_folds,
                "order_key": order_key(domain, c),
            }
        )
    return fold_of, listing


def fold_manifest_sha256(fold_of: dict) -> str:
    """sha256 of the canonical JSON of {component_id: fold}."""
    return sha256_bytes(canonical_json(dict(sorted(fold_of.items()))))


def resolve_delta_hashes(rows, by_id, entry_delta_hash):
    """Row -> (bytes holder, resolved delta hash). A null own hash resolves through ``bytes_via_alias``; a row with
    neither stored bytes nor an alias, or an alias without a hash, raises (never pooled under one null)."""
    out = {}
    for r in rows:
        holder = r.get("bytes_via_alias") or (r["id"] if r.get("bytes_present") else None)
        if holder is None:
            raise ValueError(f"{r['id']}: no stored or aliased bytes")
        if holder != r["id"] and by_id[holder]["component_id"] != r["component_id"]:
            raise ValueError(f"{r['id']}: alias holder {holder} lies in another component")
        h = entry_delta_hash.get(r["id"]) or entry_delta_hash.get(holder)
        if h is None:
            raise ValueError(f"{r['id']}: no delta hash for itself or its holder {holder}")
        out[r["id"]] = (holder, h)
    return out


def shared_deltas(resolved) -> list:
    groups = {}
    for i, (_, h) in resolved.items():
        groups.setdefault(h, []).append(i)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def exposure_ids(snapshot) -> tuple[dict, dict]:
    """(id -> sorted gates of ledger rows whose own ``id`` it is, alias -> sorted "gate:row id" of rows listing it
    in ``aliases``) over ``snapshot.json:exposure_ledger`` G0/G1/G6 rows. The F12.2 count (117) is by own id."""
    own, alias = {}, {}
    for row in snapshot["exposure_ledger"]["rows"]:
        gate = row.get("gate")
        if gate not in EXPOSURE_GATES:
            continue
        own.setdefault(row["id"], set()).add(gate)
        for a in row.get("aliases") or []:
            alias.setdefault(a, set()).add(f"{gate}:{row['id']}")
    return {k: sorted(v) for k, v in own.items()}, {k: sorted(v) for k, v in alias.items()}


def set_summary(rows) -> dict:
    return {
        "rows": len(rows),
        "components": len({r["component_id"] for r in rows}),
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        "sha256": ids_sha256([r["id"] for r in rows]),
    }


def fold_summary(rows, fold_of, stratum_of) -> list:
    out = []
    for f in range(N_FOLDS):
        rs = [r for r in rows if fold_of[r["component_id"]] == f]
        comps = {r["component_id"] for r in rs}
        out.append(
            {
                "fold": f,
                "rows": len(rs),
                "components": len(comps),
                "row_families": dict(sorted(Counter(r["family"] for r in rs).items())),
                "component_strata": dict(sorted(Counter(stratum_of[c] for c in comps).items())),
                "row_ids_sha256": ids_sha256([r["id"] for r in rs]),
            }
        )
    return out


def subsets(existing, purchasable, k_grid=K_GRID):
    """F12.5 purchase order and the S_k fold manifests (folds recomputed on each S_k)."""
    p_comps = sorted(
        {r["component_id"] for r in purchasable}, key=lambda c: order_key(PURCHASE_DOMAIN, c)
    )
    order = [
        {"rank": i, "component_id": c, "order_key": order_key(PURCHASE_DOMAIN, c)}
        for i, c in enumerate(p_comps)
    ]
    out = []
    for k in k_grid:
        if k > len(p_comps):
            raise ValueError(f"k={k} exceeds the {len(p_comps)} purchasable components")
        chosen = set(p_comps[:k])
        rows = existing + [r for r in purchasable if r["component_id"] in chosen]
        st = strata(rows)
        fold_of, _ = assign_folds(st)
        out.append(
            {
                "k": k,
                "purchased_components": sorted(chosen),
                "rows": len(rows),
                "purchased_rows": len(rows) - len(existing),
                "components": len(st),
                "row_ids_sha256": ids_sha256([r["id"] for r in rows]),
                "folds": dict(sorted(fold_of.items())),
                "fold_manifest_sha256": fold_manifest_sha256(fold_of),
                "fold_summary": fold_summary(rows, fold_of, st),
            }
        )
    return order, out


def compare(name, got, want, discrepancies):
    if got != want:
        discrepancies.append({"check": name, "expected": want, "got": got})
    return got == want


# ------------------------------------------------------------------------------------------------ build


def build(components, snapshot, selection, selection_f11, inputs=None):
    by_id = {r["id"]: r for r in components}
    primary = set(snapshot["content"]["tf_v1_records"]["primary"])
    edh = snapshot["content"]["zoo_bytes"]["entry_delta_hash"]
    pa = part_a_components(selection, selection_f11, by_id)
    existing, purchasable = select(components, primary, pa)
    population = existing + purchasable
    population.sort(key=lambda r: r["id"])
    stratum_of = strata(population)
    fold_of, fold_listing = assign_folds(stratum_of)
    resolved = resolve_delta_hashes(population, by_id, edh)
    shared = shared_deltas(resolved)
    shared_with = {i: [j for j in g if j != i] for g in shared for i in g}
    exposed, exposed_alias = exposure_ids(snapshot)
    existing_ids = {r["id"] for r in existing}
    confirm = [r for r in existing if r["component_status"] == "confirmatory"]
    literal = [r for r in population if r["component_status"] != "confirmatory"]
    nf_existing, nf_purch = select(components, primary, pa, component_filter=False)
    del nf_existing
    lora_eligible = sorted(
        r["id"]
        for r in purchasable
        if r["family"] == "lora_sft" and r.get("eligibility") == "eligible"
    )
    rows = []
    for r in population:
        holder, h = resolved[r["id"]]
        rows.append(
            {
                "id": r["id"],
                "component_id": r["component_id"],
                "family": r["family"],
                "stratum": stratum_of[r["component_id"]],
                "fold": fold_of[r["component_id"]],
                "source": SOURCE_EXISTING if r["id"] in existing_ids else SOURCE_PURCHASABLE,
                "component_status": r["component_status"],
                "g6_exposed_confirmatory": r["id"] in existing_ids
                and r["component_status"] == "confirmatory",
                "in_literal_226": r["component_status"] != "confirmatory",
                "bytes_present": bool(r.get("bytes_present")),
                "bytes_holder": holder,
                "alias_bytes": holder != r["id"],
                "exactness": r.get("exactness"),
                "resolved_delta_hash": h,
                "shared_delta_with": shared_with.get(r["id"], []),
                "exposure_gates": exposed.get(r["id"], []),
                "exposure_as_alias": exposed_alias.get(r["id"], []),
                "eligibility_components_jsonl": r.get("eligibility"),
                "f12i_lora_root": r["id"] in lora_eligible,
            }
        )
    rpc = Counter(Counter(r["component_id"] for r in population).values())
    mixed = sorted(c for c, s in stratum_of.items() if s == "mixed")
    purchase_order, s_k = subsets(existing, purchasable)

    disc = []
    got = {
        "existing": set_summary(existing),
        "purchasable": set_summary(purchasable),
        "population": set_summary(population),
        "literal_226": set_summary(literal),
    }
    for name in ("existing", "purchasable", "population"):
        compare(f"{name}", got[name], EXPECTED[name], disc)
    lit = {k: got["literal_226"][k] for k in ("rows", "components", "sha256")}
    compare("literal_226", lit, EXPECTED["literal_226"], disc)
    compare("part_a_components", len(pa), EXPECTED["part_a_components"], disc)
    compare(
        "g6_exposed_confirmatory",
        {"rows": len(confirm), "components": len({r["component_id"] for r in confirm})},
        EXPECTED["g6_exposed_confirmatory"],
        disc,
    )
    compare("alias_rows", sum(r["alias_bytes"] for r in rows), EXPECTED["alias_rows"], disc)
    compare(
        "distinct_resolved_deltas",
        len({h for _, h in resolved.values()}),
        EXPECTED["distinct_resolved_deltas"],
        disc,
    )
    compare("shared_delta_pairs", shared, EXPECTED["shared_delta_pairs"], disc)
    compare(
        "strata_components", dict(Counter(stratum_of.values())), EXPECTED["strata_components"], disc
    )
    compare(
        "existing_with_exposure_row",
        sum(1 for r in existing if exposed.get(r["id"])),
        EXPECTED["existing_with_exposure_row"],
        disc,
    )
    compare(
        "rows_per_component",
        {str(k): v for k, v in sorted(rpc.items())},
        EXPECTED["rows_per_component"],
        disc,
    )
    compare("no_component_filter_rows", len(nf_purch), EXPECTED["no_component_filter_rows"], disc)
    compare(
        "no_component_filter_components",
        len({r["component_id"] for r in nf_purch}),
        EXPECTED["no_component_filter_components"],
        disc,
    )
    compare("f12i_lora_roots", lora_eligible, EXPECTED["f12i_lora_roots"], disc)
    for g in shared:
        comps = {by_id[i]["component_id"] for i in g}
        if len(comps) != 1:
            disc.append(
                {"check": "shared_delta_within_component", "expected": 1, "got": sorted(comps)}
            )

    doc = {
        "schema": SCHEMA,
        "basis": "PREREG_H1_TF F12.2 (population, strata, folds), F12.5 (purchase order, S_k), F12.6(a) (F12-I LoRA "
        "roots); Board A F12 review 11 Nov 2026 (270-row reading adopted; mandatory component-level no-primary "
        "filter; alias-resolved delta hashes; mixed stratum; freeze actual fold ids and all S_k fold manifests)",
        "reading": "270/234: no confirmatory or fresh_unselected entry is newly scored; the 44 G6-exposed existing "
        "roots of confirmatory components are reused only as development observations (flag "
        "g6_exposed_confirmatory); the literal 226/197 restriction (flag in_literal_226) is descriptive only",
        "rules": {
            "ids_sha256": "sha256 of UTF-8 json.dumps(sorted(ids)) with Python defaults",
            "component_id": "the 64-hex component_id of components.jsonl",
            "stratum": "component family; a component holding two families is 'mixed'",
            "folds": f"within each stratum components ordered by sha256(UTF-8 '{FOLD_DOMAIN}' + component_id) "
            f"ascending; strata concatenated in the order {list(STRATUM_ORDER)}; round-robin to folds 0-"
            f"{N_FOLDS - 1} by global position",
            "fold_manifest_sha256": "sha256 of canonical JSON (sort_keys, separators (',', ':')) of "
            "{component_id: fold}",
            "purchase_order": f"purchasable components by sha256(UTF-8 '{PURCHASE_DOMAIN}' + component_id) ascending",
            "s_k": f"existing components plus the first k purchasable components, k in {list(K_GRID)}; folds "
            "recomputed on S_k by the same rule",
            "leave_one_family_out": "the mixed component always trains; its rows are excluded from held-family "
            "evaluation (Board A)",
            "delta_hash": "content.zoo_bytes.entry_delta_hash of the row, else of its bytes_via_alias holder",
        },
        "inputs": inputs or {},
        "outcomes_read": "none: snapshot.json is read for TF-v1 primary-record membership, entry delta hashes and "
        "exposure-ledger membership only; no record payload, score or eval file is opened",
        "counts": got
        | {
            "part_a_components": len(pa),
            "g6_exposed_confirmatory": {
                "rows": len(confirm),
                "components": len({r["component_id"] for r in confirm}),
                "ids_sha256": ids_sha256([r["id"] for r in confirm]),
            },
            "alias_rows": sum(r["alias_bytes"] for r in rows),
            "stored_rows": sum(not r["alias_bytes"] for r in rows),
            "distinct_resolved_deltas": len({h for _, h in resolved.values()}),
            "strata_components": dict(sorted(Counter(stratum_of.values()).items())),
            "rows_per_component": {str(k): v for k, v in sorted(rpc.items())},
            "existing_with_exposure_row": sum(1 for r in existing if exposed.get(r["id"])),
            "existing_without_exposure_row": sorted(
                r["id"] for r in existing if not exposed.get(r["id"])
            ),
            "existing_exposed_only_as_alias": sorted(
                r["id"] for r in existing if not exposed.get(r["id"]) and exposed_alias.get(r["id"])
            ),
            "exposure_note": "all existing rows are treated as exposed development outcomes (F12.2; Board A), "
            "including those without an own-id ledger row",
            "no_component_filter_sensitivity": {
                "purchasable_rows": len(nf_purch),
                "purchasable_components": len({r["component_id"] for r in nf_purch}),
                "families": dict(sorted(Counter(r["family"] for r in nf_purch).items())),
                "note": "Board A: the component-level filter is mandatory; this count is reported, never adopted",
            },
        },
        "mixed_components": [
            {
                "component_id": c,
                "rows": sorted(
                    (
                        {"id": r["id"], "family": r["family"]}
                        for r in population
                        if r["component_id"] == c
                    ),
                    key=lambda x: x["id"],
                ),
            }
            for c in mixed
        ],
        "shared_delta_groups": [
            {
                "ids": g,
                "component_id": by_id[g[0]]["component_id"],
                "resolved_delta_hash": resolved[g[0]][1],
                "families": sorted({by_id[i]["family"] for i in g}),
            }
            for g in shared
        ],
        "folds": {
            "n_folds": N_FOLDS,
            "component_fold": dict(sorted(fold_of.items())),
            "fold_manifest_sha256": fold_manifest_sha256(fold_of),
            "order": fold_listing,
            "summary": fold_summary(population, fold_of, stratum_of),
        },
        "purchase_order": purchase_order,
        "s_k": s_k,
        "f12i_lora_roots": lora_eligible,
        "rows": rows,
        "rows_sha256": sha256_bytes(canonical_json(rows)),
        "population_ids_sha256": ids_sha256([r["id"] for r in population]),
        "reproduction": {
            "expected": EXPECTED,
            "discrepancies": disc,
            "status": "reproduced" if not disc else "discrepancy",
        },
    }
    return doc


def load_inputs(manifests: Path):
    paths = {
        "components.jsonl": manifests / "components.jsonl",
        "snapshot.json": manifests / "snapshot.json",
        "selection.json": manifests / "selection.json",
        "selection_f11.json": manifests / "selection_f11.json",
    }
    raw = {k: p.read_bytes() for k, p in paths.items()}
    components = [json.loads(x) for x in raw["components.jsonl"].decode().splitlines() if x.strip()]
    selection = json.loads(raw["selection.json"])
    if selection.get("components_jsonl_sha256") not in (
        None,
        sha256_bytes(raw["components.jsonl"]),
    ):
        raise ValueError("components.jsonl differs from selection.json's pin")
    inputs = {
        k: {
            "path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            "sha256": sha256_bytes(raw[k]),
        }
        for k, p in paths.items()
    }
    return (
        components,
        json.loads(raw["snapshot.json"]),
        selection,
        json.loads(raw["selection_f11.json"]),
        inputs,
    )


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifests", type=Path, default=MANIFESTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--allow-discrepancy", action="store_true")
    args = p.parse_args(argv)
    components, snapshot, selection, selection_f11, inputs = load_inputs(args.manifests)
    doc = build(components, snapshot, selection, selection_f11, inputs)
    sha = write_json(args.out, doc)
    c = doc["counts"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": sha,
                "population": c["population"],
                "fold_manifest_sha256": doc["folds"]["fold_manifest_sha256"],
                "reproduction": doc["reproduction"]["status"],
                "discrepancies": doc["reproduction"]["discrepancies"],
            },
            indent=1,
        )
    )
    if doc["reproduction"]["discrepancies"] and not args.allow_discrepancy:
        sys.exit(1)
    return doc


if __name__ == "__main__":
    main()
