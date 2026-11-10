#!/usr/bin/env python3
"""Closed pre-G2 snapshot for the A5 evidence freeze (Board A A5 package review, checklist item 3).

Writes ``docs/prereg/a5_manifests/snapshot.json`` from read-only inputs (CPU only; no outcome value is
read: TF-v1 records, features and zoo bytes are hashed, never parsed for scores). It records:

* content pins of the closed snapshot: the TF-v1 primary record index (the builder's convention) and a
  content index over ``eval.json`` + ``records.jsonl`` of every primary and repeat record, the
  artefact_status sidecar, zoo manifest, incremental norms, the feature manifest and a content index of
  every feature file, and the stored zoo bytes (``entry_delta_hash`` per entry);
* the G6 census and G1 closure statements, with the run ids found in the records and the logs under
  ``~/agent-logs`` named by file (sha256 of each log);
* the G0 replay / G1 scoring / G6 census exposure-ledger rows (A5.2.1 fields);
* a metadata-level reconciliation of the registered source closure (parent edges, generator identity,
  equal stored bytes; ``scripts/build_h1a_panel.py`` joins) against the current zoo, including the twelve
  heretic_ara entries and the old-seal (confirmatory) components touched by the census;
* re-trained lineage flags and actual alias/predecessor availability of the forty registered picks;
* the cost ledger of the closure work.

The zoo content hashes take ~1 h of disk reads; ``--zoo-hashes`` reads a precomputed JSONL
(``{"id", "entry_delta_hash", "files", "bytes", "edit_json_sha256"}`` per entry, e.g. from
``--write-zoo-hashes``), whose values are spot-checked against the bytes of the forty picks and their
predecessors.

    python scripts/a5_snapshot.py --closed-at 2026-11-08T00:00:00Z --base-ref 19f5cbf \
        --zoo-hashes /tmp/zoo_hashes.jsonl
    python scripts/a5_snapshot.py --write-zoo-hashes /tmp/zoo_hashes.jsonl
    python scripts/a5_snapshot.py --evidence data/runs/a5/nullsize docs/prereg/a5_manifests/nullsize

``--evidence`` adds ``content.a5_pre_g2_evidence`` (final hashes of the package files that do not pin this
snapshot, the NULL-SIZE payload/runs and its reports) to the existing snapshot without changing anything else.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import a5_build_manifests as bm

D8_DOCS = "docs/reviews/P5/d8-regeneration-2026-11-06"
LOG_DIR_LABEL = "~/agent-logs"
# Logs of the closure work, by file name (A5.2.1: failures and retries are recorded too).
LOGS = {
    "g0_replay": {
        "d8-chain-pilot-20261107-055713.log": "chain replay pilot (3 stored chains, verify mode)",
        "d8-chain-pilot2-20261107-082448.log": "pilot re-run with the matrix metric (0.1 GPU-h)",
        "d8-chain-restore-20261107-060330.log": "restore launcher",
        "d8-chain-restore-s0-20261107-060906.log": "restore shard 0",
        "d8-chain-restore-s1-20261107-060906.log": "restore shard 1",
        "d8-chain-restore-s2-20261107-060906.log": "restore shard 2",
        "d8-features-resume-20261107-082521.log": "feature extraction of the 351 installed snapshots",
        "d8-tf-yh2-20261107-082552.log": "TF-v1 (y, h) audit after install (51 entries)",
        "d8-p4-full-20261107-105218.log": "full-set P4 suite (26 runs d3v2f-*)",
    },
    "g1_scoring": {
        "d8-gpu-chain-20261106-045434.log": "G1 12/12 (11 entries + base), run 20261106-062239-29d865d2",
        "d8-g6-census-20261107-102804.log": "FAILED attempt: G1 repeat aborted (repeats/g1 exists), "
        "no record written",
        "d8-g6-census-20261107-102943.log": "G1 repeats label g1b (2 predecessors + base), then the "
        "first 18 census entries; aborted at a legacy-revision row",
        "d8-g1-pred-20261107-105723.log": "G1-primary waiter (scripts/d8_g1_predecessors.sh): empty log",
        "d8-g1-pred-20261107-105724.log": "G1-primary waiter: empty log",
        "d8-g1-pred-20261107-105822.log": "G1-primary waiter: empty log",
    },
    "g6_census": {
        "d8-g6-census-20261107-102943.log": "census entries 1-18 (run 20261107-103429-35d7c436), "
        "aborted at a legacy-revision row",
        "d8-g6-census-20261107-105617.log": "FAILED restart: legacy-revision row, no record written",
        "d8-g6-census-20261107-105822.log": "census entries 19-618 (run 20261107-105825-56ef1597), "
        "including the primaries of b7edbb9f8b339f52 and 86ea1fe0ec777cf8",
    },
}
PREDECESSORS_G1 = ("b7edbb9f8b339f52", "86ea1fe0ec777cf8")
COST_LEDGER = [
    {
        "item": "G0 chain replay (3 GPU shards, LoRA re-training)",
        "amount": 20.3,
        "unit": "run-wall-h",
        "basis": "6,572 + 5,792 + 7,944 s shard wall (chain_replay_restore.json); GPU 40-70% busy",
    },
    {
        "item": "G0 (y, h) audit after install",
        "amount": 0.52,
        "unit": "GPU-h",
        "basis": "51 entries, 08:32-09:03 UTC 7 Nov",
    },
    {"item": "G0 replay pilot re-run", "amount": 0.1, "unit": "GPU-h", "basis": "matrix metric"},
    {
        "item": "G6 census",
        "amount": 11.2,
        "unit": "GPU-h (approx.)",
        "basis": "618 entries, 10:29-10:55 and 10:58-22:09 UTC 7 Nov on H0",
    },
    {
        "item": "G1 primaries of the two regenerated predecessors",
        "amount": 0.05,
        "unit": "GPU-h",
        "basis": "2 entries at ~1.04 min warm (scored inside the census run)",
    },
    {
        "item": "CPU feature extraction (351 installed snapshots)",
        "amount": 5.4,
        "unit": "CPU-h (as charged by the brief)",
        "basis": "log: wall 8,770.8 s = 2.44 h; sum of per-entry times 351 x 99.2 s = 9.67 h",
    },
]


def sha_file(p: Path) -> str:
    return bm.sha256_file(p)


def jsha(obj) -> str:
    return bm.sha256_bytes(json.dumps(obj).encode())


def run_time(run_id: str | None) -> str | None:
    """'20261107-105825-56ef1597' -> '2026-11-07T10:58:25Z' (the run's start, from its id)."""
    if not run_id or len(run_id) < 15:
        return None
    d, t = run_id[:8], run_id[9:15]
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}Z"


# ------------------------------------------------------------------------------------- content pins


def hash_many(paths, workers=6):
    with cf.ThreadPoolExecutor(workers) as ex:
        return dict(zip(paths, ex.map(lambda p: sha_file(p) if p.exists() else None, paths)))


def record_content(evals: Path):
    """Primary and repeat TF-v1 records: {relative dir: [sha eval.json, sha records.jsonl]}."""
    dirs = bm.record_dirs(evals)
    repeat_dirs = sorted(
        d
        for d in evals.glob("*/repeats/*")
        if d.is_dir() and (d / "eval.json").exists() and (d / "records.jsonl").exists()
    )
    files = [f for d in dirs + repeat_dirs for f in (d / "eval.json", d / "records.jsonl")]
    h = hash_many(files)
    primary = {d.name: [h[d / "eval.json"], h[d / "records.jsonl"]] for d in dirs}
    repeats = {
        str(d.relative_to(evals)): [h[d / "eval.json"], h[d / "records.jsonl"]] for d in repeat_dirs
    }
    index = [[d.name, h[d / "eval.json"]] for d in dirs]
    return primary, repeats, index


def feature_content(features: Path):
    ids = sorted(d.name for d in features.iterdir() if d.is_dir())
    names = ("features.json", "features.safetensors", "pooled.safetensors")
    h = hash_many([features / i / n for i in ids for n in names])
    return {i: [h[features / i / n] for n in names] for i in ids}, list(names)


def compute_zoo_hashes(zoo: Path, ids, workers=6):
    def one(i):
        files = bm.artefact_files(zoo / i)
        edit = zoo / i / "edit.json"
        return {
            "id": i,
            "files": [f.name for f in files],
            "bytes": sum(f.stat().st_size for f in files),
            "entry_delta_hash": bm.entry_delta_hash(files),
            "edit_json_sha256": sha_file(edit) if edit.exists() else None,
        }

    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, ids))


# ---------------------------------------------------------------------------------------- closure


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


def reconcile_closure(rows, edits, zoo_hash, panel, selected_ids, census_ids):
    """Registered joins (parent, generator identity, equal stored bytes) on the CURRENT zoo."""
    ids = sorted(rows)
    uf = UnionFind(ids)
    edges = Counter()
    by_key = defaultdict(list)
    for i in ids:
        parent = rows[i].get("parent_id")
        if parent in rows:
            uf.union(i, parent)
            edges["parent"] += 1
        gen = (edits.get(i) or {}).get("provenance", {}).get("generator_edit_id")
        if gen in rows:  # stricter than the registered rule: the generator entry itself joins too
            uf.union(i, gen)
            edges["generator_entry"] += 1
        for kind, value in (("generator", gen), ("delta_hash", zoo_hash.get(i))):
            if value:
                by_key[(kind, value)].append(i)
    for (kind, _), members in sorted(by_key.items()):
        if len(members) > 1:
            edges[kind] += len(members) - 1
            for other in members[1:]:
                uf.union(members[0], other)
    comp_of = {m: c["component_id"] for c in panel["components"] for m in c["members"]}
    status = {c["component_id"]: c["status"] for c in panel["components"]}
    classes = defaultdict(set)
    for i in ids:
        classes[uf.find(i)].add(i)
    # registered components joined into one current class (a merge), or split across classes
    merged = []
    for members in classes.values():
        regs = sorted({comp_of[m] for m in members if m in comp_of})
        if len(regs) > 1:
            merged.append(regs)
    split = sorted(
        c["component_id"]
        for c in panel["components"]
        if len({uf.find(m) for m in c["members"] if m in uf.parent}) > 1
    )
    unregistered = sorted(i for i in ids if i not in comp_of)
    sel_comp = {i: comp_of.get(i) for i in selected_ids}
    sel_classes = {uf.find(i) for i in selected_ids}
    joined = {}
    for i in unregistered:
        regs = sorted({comp_of[m] for m in classes[uf.find(i)] if m in comp_of})
        joined[i] = {
            "family": rows[i].get("family"),
            "joined_registered_components": regs,
            "joins_a_selected_component": bool(set(regs) & set(sel_comp.values())),
        }
    census_comps = Counter(status.get(comp_of.get(i), "no_component") for i in census_ids)
    sealed = sorted(
        {comp_of[i] for i in census_ids if status.get(comp_of.get(i)) == "confirmatory"}
    )
    return {
        "joins": "scripts/build_h1a_panel.py: parent/predecessor edges, shared generator_edit_id "
        "(edit.json provenance; additionally the generator entry itself when it is a zoo entry, which can "
        "only merge more), equal stored bytes (entry_delta_hash of the CURRENT zoo); NOT joined: "
        "base model, dataset, refusal-direction fingerprint (support diagnostics, PREREG section 2)",
        "edge_counts": dict(sorted(edges.items())),
        "n_entries": len(ids),
        "n_registered_components": len(panel["components"]),
        "n_current_classes": len(classes),
        "registered_components_merged": sorted(merged),
        "registered_components_split": split,
        "split_interpretation": "a registered component whose members no longer share a current join "
        "(typically an equal-bytes join of the original snapshot that replayed or lost bytes no longer "
        "satisfy). The registered closure stays authoritative and is coarser, so a split never makes "
        "selected components less independent; only merges would require a prospective amendment",
        "entries_without_registered_component": joined,
        "selected_components_distinct_in_current_closure": len(sel_classes) == len(selected_ids),
        "selected_components_merged_with_any_other_registered_component": sorted(
            {c for regs in merged for c in regs if c in set(sel_comp.values())}
        ),
        "census_rows_by_component_status": dict(sorted(census_comps.items())),
        "old_seal_components_scored_by_census": {
            "n_components": len(sealed),
            "component_ids_sorted_sha256": jsha(sealed),
            "n_entries": census_comps.get("confirmatory", 0),
        },
    }


# ------------------------------------------------------------------------------------------ ledger


def ledger_row(i, *, gate, role, activity, time, inv, comp, canonical, aliases, hashes, extra=None):
    row = inv.get(i, {})
    return {
        "id": i,
        "gate": gate,
        "role": role,
        "activity": activity,
        "time": time,
        "canonical_id": canonical.get(i, i),
        "component_id": comp.get(i, (None, None))[0],
        "component_status": comp.get(i, (None, None))[1],
        "aliases": aliases.get(canonical.get(i, i), []),
        "predecessor": row.get("parent_id") if i != "base" else None,
        "strength": 1.0 if role in ("scoring", "census") else None,
        "fold": None,
        "text_groups": "text_groups.json windows (wikitext_ppl) + TF-v1 development panels"
        if role in ("scoring", "census")
        else None,
        "hashes": hashes,
        **(extra or {}),
    }


# -------------------------------------------------------------------------------------------- build


def build(paths, closed_at, base_ref, zoo_rows):
    root, zoo, evals = Path(paths["root"]), Path(paths["zoo"]), Path(paths["evals"])
    features, runs_d8, logs = Path(paths["features"]), Path(paths["runs_d8"]), Path(paths["logs"])
    out = Path(paths["out"])
    docs = root / D8_DOCS

    manifest_rows = bm.read_jsonl(zoo / "manifest.jsonl")
    rows = {}
    for r in manifest_rows:
        if "id" in r:
            rows.setdefault(r["id"], r)
    status = {r["id"]: r for r in bm.read_jsonl(zoo / "artefact_status.jsonl")}
    norms = {r["id"]: r for r in bm.read_jsonl(paths["norms"])}
    panel = json.loads(Path(paths["panel"]).read_text())
    selection = json.loads((out / "selection.json").read_text())
    picks = {s["edit_id"]: s for s in selection["registered"]["selected"]}
    freeze = selection["procedural_freeze"]
    if sorted(picks) != freeze["registered_edit_ids"]:
        raise AssertionError("committed selection differs from its procedural freeze")

    comp = {m: (c["component_id"], c["status"]) for c in panel["components"] for m in c["members"]}
    canonical, aliases = {}, {}
    for _k, g in sorted(panel["alias_ledger"].items()):
        aliases[g["canonical"]] = sorted(g["aliases"])
        for m in (g["canonical"], *g["aliases"]):
            canonical[m] = g["canonical"]

    # ---- content pins
    primary, repeats, index = record_content(evals)
    feats, feat_names = feature_content(features)
    zoo_hash = {r["id"]: r["entry_delta_hash"] for r in zoo_rows}
    edit_hash = {r["id"]: r["edit_json_sha256"] for r in zoo_rows}
    if sorted(zoo_hash) != sorted(rows):
        raise ValueError("zoo hash list does not cover exactly the manifest ids")
    check_ids = sorted(
        {*picks, *(s["predecessor_id"] for s in picks.values() if s["predecessor_id"] != "base")}
    )
    for i in check_ids:  # spot check the precomputed hashes on the bytes that matter most
        if bm.entry_delta_hash(bm.artefact_files(zoo / i)) != zoo_hash[i]:
            raise ValueError(f"{i}: precomputed zoo hash differs from the bytes on disk")
    edits = {
        i: json.loads((zoo / i / "edit.json").read_text())
        for i in rows
        if (zoo / i / "edit.json").exists()
    }

    content = {
        "tf_v1_records": {
            "path": "data/evals/qwen3.5-0.8b/tf_v1",
            "n_primary": len(primary),
            "index_sha256": jsha(index),
            "index_convention": "a5_build_manifests: sha256 of json.dumps(sorted [[dir, sha256(eval.json)]]) "
            "over dirs with eval.json and records.jsonl",
            "content_index_sha256": jsha(sorted([k, *v] for k, v in primary.items())),
            "content_index_convention": "sha256 of json.dumps(sorted [[dir, sha256(eval.json), "
            "sha256(records.jsonl)]])",
            "primary": primary,
            "n_repeats": len(repeats),
            "repeats_content_index_sha256": jsha(sorted([k, *v] for k, v in repeats.items())),
            "repeats": repeats,
        },
        "artefact_status_sidecar": {
            "path": "data/zoo/qwen3.5-0.8b/artefact_status.jsonl",
            "sha256": sha_file(zoo / "artefact_status.jsonl"),
            "counts": dict(
                sorted(
                    Counter(
                        "aliased"
                        if s.get("alias_of")
                        else "stored"
                        if s.get("delta_present")
                        else "lost"
                        for s in status.values()
                    ).items()
                )
            ),
        },
        "zoo_manifest": {
            "path": "data/zoo/qwen3.5-0.8b/manifest.jsonl",
            "sha256": sha_file(zoo / "manifest.jsonl"),
            "n_entries": len(rows),
        },
        "incremental_norms": {
            "path": "data/runs/d8/a5_incremental_norms.jsonl",
            "sha256": sha_file(Path(paths["norms"])),
        },
        "panel_manifest": {
            "path": "docs/prereg/panel_manifest.json",
            "sha256": sha_file(Path(paths["panel"])),
        },
        "feature_manifest": {
            "path": "data/features/qwen3.5-0.8b/manifest.json",
            "sha256": sha_file(features / "manifest.json"),
            "n_entries": len(feats),
            "ids_sha256": jsha(sorted(feats)),
            "files": feat_names,
            "content_index_sha256": jsha(sorted([i, *v] for i, v in feats.items())),
            "content_index_convention": "sha256 of json.dumps(sorted [[id, sha256 of each of files]])",
            "content": feats,
        },
        "zoo_bytes": {
            "path": "data/zoo/qwen3.5-0.8b/<id>/{delta.safetensors | lora/*}",
            "hash": "a5_build_manifests.entry_delta_hash (sha256 of 'name:sha256(file)\\n' lines); null = "
            "no stored bytes",
            "n_with_bytes": sum(v is not None for v in zoo_hash.values()),
            "content_index_sha256": jsha(sorted([i, h] for i, h in zoo_hash.items())),
            "edit_json_index_sha256": jsha(sorted([i, h] for i, h in edit_hash.items())),
            "entry_delta_hash": dict(sorted(zoo_hash.items())),
            "spot_checked_against_bytes": check_ids,
        },
    }

    # ---- logs
    log_index = {}
    for group, names in LOGS.items():
        log_index[group] = {
            name: {
                "path": f"{LOG_DIR_LABEL}/{name}",
                "description": desc,
                "bytes": (logs / name).stat().st_size if (logs / name).exists() else None,
                "sha256": sha_file(logs / name) if (logs / name).exists() else None,
            }
            for name, desc in names.items()
        }

    # ---- G6 census
    census_ids = (runs_d8 / "g6_ids.txt").read_text().split()
    legacy = json.loads((runs_d8 / "g6_dropped_legacy.json").read_text())
    census_runs, census_missing = Counter(), []
    run_of = {}
    for i in census_ids:
        p = evals / i / "eval.json"
        if not p.exists() or i not in primary:
            census_missing.append(i)
            continue
        e = json.loads(p.read_text())
        run_of[i] = e.get("run_id")
        census_runs[e.get("run_id")] += 1
    chain_ids = sorted(i for i, r in rows.items() if r.get("family") == "chain")
    chain_unscored = {}
    for i in chain_ids:
        if i in primary:
            continue
        s = status.get(i, {})
        chain_unscored[i] = (
            "legacy_revision_dropped"
            if i in {x["id"] for x in legacy}
            else "lost"
            if not s.get("delta_present") and not s.get("alias_of")
            else "alias (scored through its holder)"
            if s.get("alias_of")
            else "no_record"
        )
    census = {
        "statement": f"CLOSED: TF-v1 primary records exist for all {len(census_ids)} census ids "
        f"(stored chain entries without a record at census start); missing: {len(census_missing)}",
        "ids_file": {
            "path": "data/runs/d8/g6_ids.txt",
            "sha256": sha_file(runs_d8 / "g6_ids.txt"),
            "n": len(census_ids),
        },
        "legacy_revision_rows_dropped": {
            "path": "data/runs/d8/g6_dropped_legacy.json",
            "sha256": sha_file(runs_d8 / "g6_dropped_legacy.json"),
            "n": len(legacy),
        },
        "missing": census_missing,
        "runs": {r: {"n_entries": n, "start": run_time(r)} for r, n in sorted(census_runs.items())},
        "logs": log_index["g6_census"],
        "chain_entries": len(chain_ids),
        "chain_entries_with_primary_record": sum(i in primary for i in chain_ids),
        "chain_entries_without_primary_record": dict(
            sorted(Counter(chain_unscored.values()).items())
        ),
        "chain_entries_without_primary_record_ids": chain_unscored,
    }

    # ---- G1
    g1 = json.loads((docs / "g1_report.json").read_text())
    g1b = json.loads((docs / "g1b_report.json").read_text())
    g1_entries = sorted(set(g1["entries"]) | set(g1b["entries"]))

    def g1_row(i):
        e = json.loads((evals / i / "eval.json").read_text())
        reps = {}
        for d in sorted((evals / i / "repeats").glob("*")):
            if (d / "eval.json").exists():
                reps[d.name] = json.loads((d / "eval.json").read_text()).get("run_id")
        return {
            "primary_run_id": e.get("run_id"),
            "primary_start": run_time(e.get("run_id")),
            "repeat_run_ids": reps,
            "records_identical": (g1b["entries"].get(i) or g1["entries"].get(i))[
                "records_identical"
            ],
            "reports": sorted(
                f"{D8_DOCS}/{n}"
                for n, rep in (("g1_report.json", g1), ("g1b_report.json", g1b))
                if i in rep["entries"]
            ),
        }

    g1_rows = {i: g1_row(i) for i in g1_entries}
    pred_runs = {i: g1_rows[i]["primary_run_id"] for i in PREDECESSORS_G1}
    g1_close = {
        "statement": f"CLOSED: {len(g1_entries)} of {len(g1_entries)} G1 entries have bitwise-identical "
        "primary and repeat records (g1: 12 incl. base; g1b: the two regenerated predecessors + base)",
        "all_records_identical": bool(g1["all_records_identical"] and g1b["all_records_identical"]),
        "reports": {
            f"{D8_DOCS}/g1_report.json": sha_file(docs / "g1_report.json"),
            f"{D8_DOCS}/g1b_report.json": sha_file(docs / "g1b_report.json"),
        },
        "entries": g1_rows,
        "predecessor_primaries": {
            "ids": list(PREDECESSORS_G1),
            "run_ids": pred_runs,
            "finding": "the primaries of the two regenerated predecessors carry the census run id "
            f"{sorted(set(pred_runs.values()))} (log d8-g6-census-20261107-105822.log), not a separate "
            "run of scripts/d8_g1_predecessors.sh: its three logs (d8-g1-pred-*.log) are empty. The "
            "regeneration report's wording ('primaries by scripts/d8_g1_predecessors.sh after the census') "
            "does not match the run ids; the G1 comparison itself (g1b_report.json) is complete",
        },
        "logs": log_index["g1_scoring"],
    }

    # ---- G0 replay
    restore = json.loads((docs / "chain_replay_restore.json").read_text())
    install = json.loads((docs / "chain_replay_install.json").read_text())
    pilot = json.loads((docs / "chain_replay_pilot.json").read_text())
    installed = set(install["installed_ids"])
    rejected = {r["id"]: r["reasons"] for r in install["rejected"]}
    yh_ids = (runs_d8 / "tf_yh_ids.txt").read_text().split()
    steps = {}
    for sh in restore["shards"]:
        for ch in sh["chains"]:
            for st in ch["steps"]:
                steps[st["id"]] = st
    pilot_ids = sorted({s["id"] for ch in pilot["chains"] for s in ch["steps"]})
    g0 = {
        "statement": f"CLOSED: {len(steps)} chain steps replayed ({sum(not s.get('stored') for s in steps.values())} "
        f"computed, {sum(bool(s.get('stored')) for s in steps.values())} stored steps recomputed for comparison); "
        f"{len(installed)} installed, {len(rejected)} rejected and still lost; sidecar {install['sidecar']}",
        "reports": {
            f"{D8_DOCS}/{n}": sha_file(docs / n)
            for n in (
                "chain_replay_pilot.json",
                "chain_replay_restore.json",
                "chain_replay_install.json",
            )
        },
        "pilot_step_ids": pilot_ids,
        "rejected": rejected,
        "yh_audit_ids": {
            "path": "data/runs/d8/tf_yh_ids.txt",
            "sha256": sha_file(runs_d8 / "tf_yh_ids.txt"),
            "n": len(yh_ids),
        },
        "p4_full_set": {
            "summary": "docs/reviews/P4/d3v2f-summary-2026-11-08.md",
            "sha256": sha_file(root / "docs/reviews/P4/d3v2f-summary-2026-11-08.md"),
            "rows": 342,
            "clusters": 278,
        },
        "logs": log_index["g0_replay"],
    }

    # ---- exposure ledger
    inv = rows | {"base": {"parent_id": None}}
    common = {"inv": inv, "comp": comp, "canonical": canonical, "aliases": aliases}
    ledger = []
    for i in sorted(steps):
        st = steps[i]
        s = status.get(i, {})
        ledger.append(
            ledger_row(
                i,
                gate="G0",
                role="replay",
                activity="replay+install"
                if i in installed
                else "replay (rejected, still lost)"
                if i in rejected
                else "replay (stored step recomputed for comparison)",
                time=(s.get("regenerated") or {}).get("written_utc") or "2026-11-07T06:09Z/08:22Z",
                hashes={"entry_delta_hash": zoo_hash.get(i), "edit_json_sha256": edit_hash.get(i)},
                extra={
                    "stored_before_replay": bool(st.get("stored")),
                    "bitwise": (st.get("compare") or {}).get("bitwise"),
                    "anchored": st.get("anchored"),
                    "exactness": (norms.get(i) or {}).get("exactness"),
                    "pilot": i in pilot_ids,
                },
                **common,
            )
        )
    for i in sorted(yh_ids):
        ledger.append(
            ledger_row(
                i,
                gate="G0",
                role="replay",
                activity="fingerprint rerun: TF-v1 (y, h) audit after install",
                time="2026-11-07T08:31:53Z/09:02:38Z",
                hashes={"tf_v1_record": primary.get(i)},
                extra={"exactness": (norms.get(i) or {}).get("exactness")},
                **common,
            )
        )
    for i in g1_entries:
        ledger.append(
            ledger_row(
                i,
                gate="G1",
                role="scoring",
                activity="G1 primary + repeat (bitwise identity check)",
                time=g1_rows[i]["primary_start"],
                hashes={
                    "tf_v1_record": primary.get(i),
                    "repeats": {k: v for k, v in repeats.items() if k.startswith(f"{i}/")},
                },
                extra={
                    "run_ids": {
                        "primary": g1_rows[i]["primary_run_id"],
                        **g1_rows[i]["repeat_run_ids"],
                    }
                },
                **common,
            )
        )
    for i in sorted(census_ids):
        ledger.append(
            ledger_row(
                i,
                gate="G6",
                role="census",
                activity="G6 closure census: TF-v1 primary record",
                time=run_time(run_of.get(i)),
                hashes={"tf_v1_record": primary.get(i), "entry_delta_hash": zoo_hash.get(i)},
                extra={
                    "run_id": run_of.get(i),
                    "exactness": (norms.get(i) or {}).get("exactness"),
                },
                **common,
            )
        )

    # ---- closure reconciliation and the forty picks
    closure = reconcile_closure(rows, edits, zoo_hash, panel, sorted(picks), census_ids)

    def availability(i):
        s, r = status.get(i, {}), rows.get(i, {})
        own = bool(bm.artefact_files(zoo / i))
        holder = s.get("alias_of")
        return {
            "id": i,
            "family": r.get("family"),
            "exactness": (norms.get(i) or {}).get("exactness"),
            "regenerated": (s.get("regenerated") or {}).get("exactness"),
            "own_bytes": own,
            "alias_of": holder,
            "alias_holder_bytes": bool(holder and bm.artefact_files(zoo / holder)),
            "available": own or bool(holder and bm.artefact_files(zoo / holder)),
            "entry_delta_hash": zoo_hash.get(i) if own else zoo_hash.get(holder),
            "tf_v1_primary_record": i in primary,
            "component_id": comp.get(i, (None, None))[0],
        }

    forty = []
    for i in sorted(picks):
        s = picks[i]
        pred = s["predecessor_id"]
        forty.append(
            {
                "edit": availability(i),
                "family": s["family"],
                "tertile": s["tertile_label"],
                "predecessor_id": pred,
                "predecessor": {
                    "id": "base",
                    "available": True,
                    "tf_v1_primary_record": "base" in primary,
                }
                if pred == "base"
                else availability(pred),
            }
        )
    flags = Counter(
        (p["edit"]["exactness"], p["predecessor"].get("exactness", "base")) for p in forty
    )

    return {
        "schema": "a5-snapshot-v1",
        "label": "closed pre-G2 snapshot (Board A A5 package review, checklist item 3)",
        "closed_at": closed_at,
        "base_ref": base_ref,
        "procedural_freeze": {
            "date": freeze["date"],
            "registered_edit_ids_sorted_sha256": freeze["registered_edit_ids_sorted_sha256"],
        },
        "no_outcomes_read": "this snapshot hashes records, features and bytes; it reads no score, and no "
        "G3 outcome exists",
        "content": content,
        "g0_replay": g0,
        "g1_scoring": g1_close,
        "g6_census": census,
        "closure_reconciliation": closure,
        "old_seal": {
            "partition_sha256": panel["partition_sha256"],
            "finding": f"the G6 census scored {closure['old_seal_components_scored_by_census']['n_entries']} "
            f"members of {closure['old_seal_components_scored_by_census']['n_components']} confirmatory "
            "(old-seal) components; under A5.2 they are now G6-exposed. A5.9 already bars the old partition "
            "from G7, and A5.3.1 selects only D3 development members, so the forty picks are unaffected; "
            "the 'unspent' wording of A5.9 needs Board A reconciliation",
        },
        "exposure_ledger": {
            "fields": "A5.2.1: canonical id, connected component, aliases, role, time, predecessor, strength, "
            "fold (none assigned at G0/G1/G6), text groups, hashes",
            "counts": dict(sorted(Counter(f"{r['gate']}:{r['role']}" for r in ledger).items())),
            "rows": ledger,
        },
        "forty_picks": {
            "rows": forty,
            "lineage_flags": {f"{e}|{p}": n for (e, p), n in sorted(flags.items(), key=str)},
            "retrained_picks": sorted(
                p["edit"]["id"] for p in forty if p["edit"]["exactness"] == "re-trained"
            ),
            "retrained_predecessors": sorted(
                p["predecessor_id"]
                for p in forty
                if p["predecessor"].get("exactness") == "re-trained"
            ),
            "all_edits_available": all(p["edit"]["available"] for p in forty),
            "all_predecessors_available": all(p["predecessor"]["available"] for p in forty),
            "predecessors_via_alias": sorted(
                p["predecessor_id"]
                for p in forty
                if p["predecessor_id"] != "base" and not p["predecessor"]["own_bytes"]
            ),
        },
        "cost_ledger": COST_LEDGER
        + [
            {
                "item": "full-set P4 suite (26 runs d3v2f-*, CPU)",
                "amount": 36.0,
                "unit": "run-wall-h (sum of 26 runs; 129,597 s)",
                "basis": "log d8-p4-full-20261107-105218.log: 10:52-23:28 UTC 7 Nov = 12.6 h wall, <= 4 "
                "parallel; d3v2f-m3-family exit 1 after 62 s (no metrics)",
            }
        ],
    }


EVIDENCE_MANIFESTS = (  # package files that do not embed the snapshot's own sha256 (no cycle)
    "components.jsonl",
    "references.json",
    "text_groups.json",
    "panels.json",
    "control_seeds.json",
    "nullsize_inputs.json",
)


def evidence_block(out: Path, runs: Path, reports: Path, runs_label="data/runs/a5/nullsize"):
    """Final hashes of the pre-G2 evidence derived from this closed snapshot (checklist items 5-6).

    selection.json, README.md and code_hashes.json are not listed: selection.json and README.md pin this
    file's sha256, so listing them would be circular (they are listed, with their hashes, in README.md).
    The NULL-SIZE payload pins this snapshot by content index (tf_v1_records, zoo_bytes), not by file hash.
    """
    run_files = sorted(f for f in runs.iterdir() if f.is_file() and f.suffix == ".json")
    return {
        "label": "final hashes of the pre-G2 evidence built on this closed snapshot (Board A checklist items "
        "5-6); written after closure by `scripts/a5_snapshot.py --evidence`; the content pins above are "
        "unchanged",
        "not_listed": "selection.json, README.md (both pin this file's sha256) and code_hashes.json; see "
        "docs/prereg/a5_manifests/README.md",
        "manifests": {n: sha_file(out / n) for n in EVIDENCE_MANIFESTS},
        "nullsize_runs": {f"{runs_label}/{f.name}": sha_file(f) for f in run_files},
        "nullsize_reports": {
            f"docs/prereg/a5_manifests/nullsize/{f.name}": sha_file(f)
            for f in sorted(reports.iterdir())
            if f.is_file()
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = ROOT / "data"
    p.add_argument("--zoo", default=data / "zoo/qwen3.5-0.8b", type=Path)
    p.add_argument("--evals", default=data / "evals/qwen3.5-0.8b/tf_v1", type=Path)
    p.add_argument("--features", default=data / "features/qwen3.5-0.8b", type=Path)
    p.add_argument("--norms", default=data / "runs/d8/a5_incremental_norms.jsonl", type=Path)
    p.add_argument("--runs-d8", default=data / "runs/d8", type=Path)
    p.add_argument("--panel", default=ROOT / "docs/prereg/panel_manifest.json", type=Path)
    p.add_argument("--logs", default=Path.home() / "agent-logs", type=Path)
    p.add_argument("--out", default=ROOT / "docs/prereg/a5_manifests", type=Path)
    p.add_argument("--closed-at", help="UTC time of closure (recorded verbatim)")
    p.add_argument("--base-ref", default="HEAD", help="git ref the snapshot is closed against")
    p.add_argument("--zoo-hashes", type=Path, help="precomputed zoo hash JSONL")
    p.add_argument("--write-zoo-hashes", type=Path, help="compute zoo hashes to this path and exit")
    p.add_argument(
        "--evidence",
        nargs=2,
        type=Path,
        metavar=("RUNS", "REPORTS"),
        help="add content.a5_pre_g2_evidence (final hashes of the NULL-SIZE payload, runs, reports and "
        "package files) to the existing snapshot.json and exit; nothing else changes",
    )
    a = p.parse_args(argv)
    if a.evidence:
        out = a.out.resolve()
        if out.is_relative_to(Path("/home/ubuntu/wsp-data")):
            p.error("refusing to write under data/ (read-only)")
        snap = json.loads((out / "snapshot.json").read_text())
        snap["content"]["a5_pre_g2_evidence"] = evidence_block(out, *a.evidence)
        blob = bm.dumps(snap)
        (out / "snapshot.json").write_bytes(blob)
        print(f"{bm.sha256_bytes(blob)}  snapshot.json")
        return 0
    ids = sorted({r["id"] for r in bm.read_jsonl(a.zoo / "manifest.jsonl") if "id" in r})
    if a.write_zoo_hashes:
        if a.write_zoo_hashes.resolve().is_relative_to(Path("/home/ubuntu/wsp-data")):
            p.error("refusing to write under data/ (read-only)")
        rows = compute_zoo_hashes(a.zoo, ids)
        a.write_zoo_hashes.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        return 0
    if not a.closed_at:
        p.error("--closed-at is required")
    out = a.out.resolve()
    if out.is_relative_to(data.resolve()) or out.is_relative_to(Path("/home/ubuntu/wsp-data")):
        p.error("refusing to write under data/ (read-only)")
    zoo_rows = bm.read_jsonl(a.zoo_hashes) if a.zoo_hashes else compute_zoo_hashes(a.zoo, ids)
    sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", a.base_ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    paths = {
        "root": ROOT,
        "zoo": a.zoo,
        "evals": a.evals,
        "features": a.features,
        "norms": a.norms,
        "runs_d8": a.runs_d8,
        "panel": a.panel,
        "logs": a.logs,
        "out": out,
    }
    snap = build(paths, a.closed_at, sha, zoo_rows)
    blob = bm.dumps(snap)
    (out / "snapshot.json").write_bytes(blob)
    print(f"{bm.sha256_bytes(blob)}  snapshot.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
