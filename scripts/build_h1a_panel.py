#!/usr/bin/env python3
"""Build and seal the H1a source/ancestry partition (PREREG_H1_TF section 2, items 1-6).

Outcome-free: reads only the zoo manifest, edit.json / generator records, stored delta bytes and
the *membership* of earlier evaluations (which entries have an eval.json). It never reads a score,
KL, accuracy or any other outcome, and it never launches scoring.

Algorithm (frozen by PREREG section 2, seed 20261105):
 1. Snapshot manifest, edit.json files, stored-delta content hashes, generator records, evaluation
    membership (D3) and the pilot / qualification entries; record each hash.
 2. Graph over ALL entries (unlabelled ancestors included). Joins: parent/predecessor edges (chain
    steps, task-vector sources), shared generator (trained adapter / source-run) identity, equal
    stored-delta hashes (exact duplicates). The generation `reuse_key` is a recipe-class key shared by
    every entry of a family configuration (not a source identity), so it is not a join. Transitive closure ->
    connected components. NOT joined (support diagnostics only): shared base model, dataset, the
    refusal-direction fingerprint. Missing sources, cycles, conflicting predecessors, missing
    artefacts, non-production eligibility and the section 8 flagged ARA ids make a whole component
    ineligible.
 3. Development = every component with a D3-evaluated, pilot, timing or qualification member.
 4. Candidates = the previously unevaluated lora_sft / task_vector entries plus chain lineages.
 5. Component id = sha256 of the canonical sorted member / source-identity list; rank = sha256 of
    "20261105:H1a:" + id, ties by id. Select all eligible fresh components containing a candidate
    lora_sft/task_vector entry; if fewer than 100, append eligible fresh chain components in hash
    order until 106 (A2: floor 100 + six spares); if more, retain all. No outcome-based selection.
 6. Write the manifest (ids and hashes only) and its content hash.

Usage: build_h1a_panel.py [--hash-only] [--out PATH]
"""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
EVALS = ROOT / "data/evals/qwen3.5-0.8b"
OUT = ROOT / "data/runs/h1a/panel_manifest.json"
HASH_CACHE = ROOT / "data/runs/h1a/delta_hashes.json"
SEED = 20261105
RANK_DOMAIN = f"{SEED}:H1a:"
FLOOR_COMPONENTS = 100  # PREREG section 2/4: at least this many valid fresh components
# Amendment A2.3: outcome-free closure expansion to the maximum the partition supports (six spare
# fresh chain components after the 100th, in hash order) so one post-freeze exclusion does not
# break the floor.
TARGET_COMPONENTS = 106
REFERENCE_CONVENTION = {
    "chain": "actual predecessor (parent_id); step 0 uses the matched base",
    "lora_sft": "matched base",
    "task_vector": "matched base (the stored delta is the full scaled delta from the base; the parent is a source, not a predecessor model state)",
    "abliteration": "matched base",
    "random_dir": "matched base",
}
STRATUM = ("lora_sft", "task_vector", "chain")  # outcome-free non-null family stratum (PREREG 1)
# PREREG section 8: ara_unconstrained_surrogate_scale diagnostics, always excluded.
FLAGGED_ARA = [
    "bbf6af9653df23c2",
    "770d16bb686c8226",
    "39b0bc05af270314",
    "cb43afcdcf968a92",
    "eeda0ec4d3dc118b",
    "2b3c19e6158e0788",
    "0b5a6ef591cee209",
    "5c1c957dab65753f",
    "d9da6daecd7c8d6d",
    "cec362b7c5eaf0f1",
    "cbc856f1b3ed9f7b",
    "69c7390859177261",
]
# Measured warm TF-v1 cost per scored entry (docs/RESULTS.md, timing pilot round 2).
WARM_MINUTES = 1.805
SCORING_CAP_HOURS = 19.0


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def artefact_files(entry_dir):
    """Stored delta bytes: dense delta.safetensors, or the LoRA adapter files."""
    dense = entry_dir / "delta.safetensors"
    if dense.exists():
        return [dense]
    return sorted(p for p in (entry_dir / "lora").glob("*") if p.is_file())


def entry_delta_hash(entry_dir):
    files = artefact_files(entry_dir)
    if not files:
        return None
    return sha("".join(f"{p.name}:{file_sha(p)}\n" for p in files))


def delta_hashes(ids, cache_path=HASH_CACHE, workers=4):
    """Stored-delta content hash per entry, cached by (size, mtime) per artefact file."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def stamp(i):
        return [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in artefact_files(ZOO / i)]

    todo = [i for i in ids if i not in cache or cache[i]["stamp"] != stamp(i)]

    def work(i):
        return i, {"stamp": stamp(i), "hash": entry_delta_hash(ZOO / i)}

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for n, (i, value) in enumerate(pool.map(work, todo)):
            cache[i] = value
            if n % 100 == 0:
                print(f"delta hash {n}/{len(todo)}", flush=True)
                cache_path.write_text(json.dumps(cache))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return {i: cache[i]["hash"] for i in ids}


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def load_zoo():
    rows = {}
    for line in (ZOO / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "id" in row:
            rows[row["id"]] = row
    edits = {i: json.loads((ZOO / i / "edit.json").read_text()) for i in rows}
    return rows, edits


def membership():
    """Entries that already carry a D3 evaluation (membership only, never a value)."""
    ids = {
        json.loads(line)["id"]
        for line in (ZOO / "manifest.jsonl").read_text().splitlines()
        if '"id"' in line
    }
    return {i for i in ids if (EVALS / i / "eval.json").exists()}


def pilot_ids():
    ids = set((ROOT / "data/runs/d3_shards/controls.txt").read_text().split())
    qual = ROOT / "data/runs/h1a/qualification_panel.json"
    if qual.exists():
        ids |= set(json.loads(qual.read_text())["score_ids"])
    return ids


def entry_audit(i, rows, edits):
    """Entry-level reasons that make the whole component ineligible."""
    reasons, edit, row = [], edits[i], rows[i]
    if i in FLAGGED_ARA:
        reasons.append("flagged_ara_diagnostic")
    if edit.get("eligibility") != "production":
        reasons.append(f"eligibility_{edit.get('eligibility') or 'missing'}")
    if not artefact_files(ZOO / i):
        reasons.append("missing_delta_artefact")
    parent = row.get("parent_id")
    if parent is not None:
        if parent not in rows:
            reasons.append("missing_parent")
        elif edit["parent_id"] != parent:
            reasons.append("conflicting_parent_record")
    if row["family"] == "chain" and parent in rows:
        p, own = edits[parent]["params"], edit["params"]
        if (
            rows[parent]["family"] != "chain"
            or p["step"] != own["step"] - 1
            or p["length"] != own["length"]
        ):
            reasons.append("inconsistent_chain_predecessor")
    if row["family"] == "chain" and parent is None and edit["params"]["step"] != 0:
        reasons.append("chain_step_without_predecessor")
    if row["family"] == "task_vector" and parent in rows:
        key = edit.get("provenance", {}).get("parent_reuse_key")
        if key != edits[parent].get("provenance", {}).get("reuse_key"):
            reasons.append("task_vector_parent_identity_mismatch")
    return reasons


def has_cycle(rows):
    for start in rows:
        seen, node = set(), start
        while node is not None and node in rows:
            if node in seen:
                return True
            seen.add(node)
            node = rows[node].get("parent_id")
    return False


def build(rows, edits, hashes, evaluated, pilots):
    if has_cycle(rows):
        raise ValueError("cycle in the parent graph")
    uf = UnionFind(rows)
    edge_kinds = Counter()
    by_key = defaultdict(list)
    for i, edit in edits.items():
        parent = rows[i].get("parent_id")
        if parent in rows:
            uf.union(i, parent)
            edge_kinds["parent"] += 1
        prov = edit.get("provenance", {})
        for kind, value in (
            ("generator", prov.get("generator_edit_id")),
            ("delta_hash", hashes.get(i)),
        ):
            if value:
                by_key[(kind, value)].append(i)
    for (kind, _), members in by_key.items():
        if len(members) > 1:
            edge_kinds[kind] += len(members) - 1
            for other in members[1:]:
                uf.union(members[0], other)
    groups = defaultdict(list)
    for i in rows:
        groups[uf.find(i)].append(i)
    aliases = {
        value: sorted(m)
        for (kind, value), m in by_key.items()
        if kind == "delta_hash" and len(m) > 1
    }
    components = []
    for members in groups.values():
        members = sorted(members)
        sources = sorted(
            {
                f"generator:{edits[i]['provenance']['generator_edit_id']}"
                for i in members
                if edits[i].get("provenance", {}).get("generator_edit_id")
            }
            | {f"delta:{hashes[i]}" for i in members if hashes.get(i)}
        )
        cid = sha(json.dumps([members, sources]))
        fam = Counter(rows[i]["family"] for i in members)
        development = sorted(i for i in members if i in evaluated or i in pilots)
        reasons = sorted({r for i in members for r in entry_audit(i, rows, edits)})
        components.append(
            {
                "component_id": cid,
                "rank": sha(RANK_DOMAIN + cid),
                "members": members,
                "source_identities": sources,
                "family_counts": dict(sorted(fam.items())),
                "stratum_rows": sorted(i for i in members if rows[i]["family"] in STRATUM),
                "development_members": development,
                "exclusion_reasons": reasons,
                "status": None,
            }
        )
    candidates = {
        i
        for i, r in rows.items()
        if r["family"] in ("lora_sft", "task_vector") and i not in evaluated and i not in pilots
    }
    for c in components:
        if c["development_members"]:
            c["status"] = "development"
        elif c["exclusion_reasons"]:
            c["status"] = "excluded"
        else:
            c["status"] = "fresh"
        c["has_candidate_lora_task"] = bool(candidates & set(c["members"]))
        c["has_chain"] = c["family_counts"].get("chain", 0) > 0
        c["has_stratum"] = bool(c["stratum_rows"])
    order = lambda c: (c["rank"], c["component_id"])
    fresh = sorted(
        (c for c in components if c["status"] == "fresh" and c["has_stratum"]), key=order
    )
    first = [c for c in fresh if c["has_candidate_lora_task"]]
    selected = list(first)
    if len(selected) < TARGET_COMPONENTS:
        for c in fresh:
            if len(selected) >= TARGET_COMPONENTS:
                break
            if c not in selected and c["has_chain"]:
                selected.append(c)
    chosen = {c["component_id"] for c in selected}
    for c in components:
        if c["status"] == "fresh":
            c["status"] = "confirmatory" if c["component_id"] in chosen else "fresh_unselected"
            if c["status"] == "fresh_unselected":
                c["exclusion_reasons"] = [
                    "no_non_null_stratum_row" if not c["has_stratum"] else "beyond_chain_fill_target"
                ]
    components.sort(key=order)
    return components, aliases, dict(edge_kinds), candidates, first


def spare_ids(components, first):
    """Selection order is candidate components (rank order) then chain fill (rank order); the
    components past the floor are the registered spares (amendment A2.3)."""
    order = lambda c: (c["rank"], c["component_id"])
    confirm = [c for c in components if c["status"] == "confirmatory"]
    ordered = list(first) + sorted((c for c in confirm if c not in first), key=order)
    return [c["component_id"] for c in ordered[FLOOR_COMPONENTS:]]


def summarize(
    components, rows, candidates, first, aliases, edge_kinds, evaluated=frozenset(), hashes=None
):
    status = Counter(c["status"] for c in components)
    dev = [c for c in components if c["status"] == "development"]
    dev_members = [i for c in dev for i in c["members"]]
    confirm = [c for c in components if c["status"] == "confirmatory"]
    members = [i for c in confirm for i in c["members"]]
    rows_n = sum(len(c["stratum_rows"]) for c in confirm)
    nonnull = sum(1 for c in confirm if c["has_stratum"])
    hashes = hashes or {}
    unique = lambda ids: len({hashes.get(i) or i for i in ids})  # aliases are scored once
    hours = unique(members) * WARM_MINUTES / 60
    sizes = Counter(len(c["members"]) for c in confirm)
    return {
        "components_total": len(components),
        "components_by_status": dict(sorted(status.items())),
        "confirmatory_components": len(confirm),
        "confirmatory_non_null_components": nonnull,
        "confirmatory_stratum_rows": rows_n,
        "confirmatory_closure_entries": len(members),
        "confirmatory_closure_unique_deltas": unique(members),
        "confirmatory_family_rows": dict(
            sorted(Counter(rows[i]["family"] for i in members).items())
        ),
        "component_size_distribution": {str(k): v for k, v in sorted(sizes.items())},
        "development_components": len(dev),
        "development_non_null_components": sum(1 for c in dev if c["has_stratum"]),
        "development_members": len(dev_members),
        "development_evaluated_members": sum(1 for i in dev_members if i in evaluated),
        "development_unevaluated_members": sum(1 for i in dev_members if i not in evaluated),
        "d3_evaluated_entries": len(evaluated),
        "d3_unique_deltas": unique(evaluated),
        "d3_scoring_hours_at_warm_median": round(unique(evaluated) * WARM_MINUTES / 60, 2),
        "candidate_lora_task_entries": len(candidates),
        "candidate_by_family": dict(Counter(rows[i]["family"] for i in candidates)),
        "components_containing_candidates_fresh": len(first),
        "candidates_in_selected_components": len(candidates & set(members)),
        "met_target": len(confirm) >= FLOOR_COMPONENTS,
        "target_components": TARGET_COMPONENTS,
        "floor_components": FLOOR_COMPONENTS,
        "closure_scoring_hours_at_warm_median": round(hours, 2),
        "warm_minutes_per_entry": WARM_MINUTES,
        "scoring_cap_hours": SCORING_CAP_HOURS,
        "exclusion_reasons_among_non_development": dict(
            Counter(
                r for c in components if c["status"] == "excluded" for r in c["exclusion_reasons"]
            )
        ),
        "delta_hash_alias_groups": len(aliases),
        "join_edges_by_kind": edge_kinds,
    }


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--hash-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows, edits = load_zoo()
    hashes = delta_hashes(sorted(rows))
    if args.hash_only:
        return
    evaluated, pilots = membership(), pilot_ids()
    manifest_bytes = (ZOO / "manifest.jsonl").read_bytes()
    components, aliases, edge_kinds, candidates, first = build(
        rows, edits, hashes, evaluated, pilots
    )
    summary = summarize(components, rows, candidates, first, aliases, edge_kinds, evaluated, hashes)
    partition = {c["component_id"]: c["status"] for c in components}
    out = {
        "version": "h1a-partition-v2",
        "reference_convention": REFERENCE_CONVENTION,
        "seed": SEED,
        "rank_domain": RANK_DOMAIN,
        "registration": {
            "prereg": "docs/PREREG_H1_TF.md",
            "git_head": git("rev-parse", "HEAD"),
            "prereg_last_commit": git("log", "-1", "--format=%H", "--", "docs/PREREG_H1_TF.md"),
        },
        "model_revision": next(iter(edits.values()))["base_revision"],
        "snapshot": {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "entries": len(rows),
            "edit_json_sha256": sha(
                "".join(f"{i}:{file_sha(ZOO / i / 'edit.json')}\n" for i in sorted(rows))
            ),
            "delta_hashes_sha256": sha(json.dumps(hashes, sort_keys=True)),
            "evaluation_membership_sha256": sha(json.dumps(sorted(evaluated))),
            "evaluation_membership_count": len(evaluated),
            "pilot_and_qualification_ids_sha256": sha(json.dumps(sorted(pilots))),
            "flagged_ara_ids": sorted(FLAGGED_ARA),
            "corrected_ara_cohort_cutoff": "none: no corrected ARA rebuild exists in this snapshot",
            "config_sha256": {
                p: file_sha(ROOT / p)
                for p in (
                    "configs/eval_tf_v1.yaml",
                    "configs/eval_subsets/mmlu_pro_tf_v1.txt",
                    "configs/eval_subsets/hellaswag_tf_v1.txt",
                    "configs/eval_subsets/harmless_alpaca_tf_v1.txt",
                )
            },
        },
        "device_assignment": "H0 (single-device scope, DECISIONS D2 addendum)",
        "summary": summary,
        "alias_ledger": {
            h: {"canonical": m[0], "aliases": m[1:]} for h, m in sorted(aliases.items())
        },
        "components": [
            {k: v for k, v in c.items() if k not in ("has_candidate_lora_task",)}
            for c in components
        ],
        "spare_component_ids": spare_ids(components, first),
        "confirmatory_component_ids": [
            c["component_id"] for c in components if c["status"] == "confirmatory"
        ],
        "partition_sha256": sha(json.dumps(partition, sort_keys=True)),
    }
    out["manifest_content_sha256"] = sha(json.dumps(out, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    json.dump(summary, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
