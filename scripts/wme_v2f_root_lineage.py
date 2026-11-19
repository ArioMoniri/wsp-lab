#!/usr/bin/env python3
"""Hash-only lineage of the bank-root construction shards in ``wme-c1`` (PREREG_WME §II.24.1; Board A confirmation
blocker A). CPU only; writes nothing under data/wme/panels; never opens a split file.

1. Rebuild the c1 inputs exactly as ``scripts/wme_v2e_build_panels.py`` did (``c1_inputs``: the pinned source pools,
   registered sizes, fresh construction minimums and the root-shard items via ``panel_c1.grouped_pool``), load the
   committed allocation record and verify its sha256 against the manifest.
2. Re-run the builder's own deterministic grouping and split plan (``build_c1_panel_v2e(..., lineage=True)``; nothing
   is written) and require it to reproduce the built manifest's hash and group lists for every objective and split
   (hash-list digests compared with the manifest's; the split files themselves are not read).
3. For every original root item: its original group (root sampler id), its final union-policy group, the group's
   owner objective, the written representative and the representative's split; check that every one is in
   construction and that no root item hash or root final group occurs in dev, bank, sealed_test or sealed_gen of
   wme-c1, nor in any split of wme-c2-gen (hash lists).
4. Reconcile 1,761 items / 1,728 original groups with the written representatives; publish the mapping (hashes
   only) and its digest to docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json.

    wme_v2f_root_lineage.py
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json"
EVAL_SPLITS = ("sealed_gen", "sealed_test", "dev", "bank")


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def digest_list(hs) -> str:
    return hashlib.sha256("\n".join(hs).encode()).hexdigest()


def main():
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("b2e", ROOT / "scripts/wme_v2e_build_panels.py")
    b2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b2e)
    from wsp.wm import exposure as X
    from wsp.wm import panels_v2d as V

    man = json.loads((ROOT / "docs/prereg/wme_manifest.json").read_text())
    pools, sizes, cmin, reserved_items, root_groups, _, orig = b2e.c1_inputs(man)
    alloc_path = ROOT / man["exposures.v2"]["mmlu_allocation"].get(
        "file", "data/wme/panels/wme-mmlu-allocation-v2e.json"
    )
    alloc = json.loads(alloc_path.read_text())
    V.verify_allocation(alloc)
    if alloc["sha256"] != man["exposures.v2"]["mmlu_allocation"]["sha256"]:
        raise SystemExit("allocation record does not match the manifest")
    if (
        hashlib.sha256(alloc_path.read_bytes()).hexdigest()
        != man["exposures.v2"]["mmlu_allocation"]["file_sha256"]
    ):
        raise SystemExit("allocation file hash does not match the manifest")
    exp1 = X.load_exposures(ROOT, X.registry_before("wme-c1"))
    plan = V.build_c1_panel_v2e(
        pools,
        sizes,
        b2e.PANELS / "wme-c1",
        exp1,
        alloc,
        reserved_items,
        construction_min=cmin,
        lineage=True,
    )
    c1 = json.loads((ROOT / "data/wme/panels/wme-c1/manifest.json").read_text())
    c2 = json.loads((ROOT / "data/wme/panels/wme-c2-gen/manifest.json").read_text())
    # 2. reproduction of the built manifest (hash lists only)
    mismatch = []
    for o, ss in c1["objectives"].items():
        for s, rec in ss.items():
            if digest_list(sorted(rec["hashes"])) != digest_list(sorted(plan["splits"][s][o])):
                mismatch.append(f"{o}/{s}/hashes")
            if digest_list(sorted(rec["groups"])) != digest_list(sorted(plan["groups"][s][o])):
                mismatch.append(f"{o}/{s}/groups")
    if mismatch:
        raise SystemExit(f"the lineage plan does not reproduce the built manifest: {mismatch[:5]}")
    # 3. root lineage
    written = {}  # hash -> (objective, split)
    wgroup = {}  # final group -> (objective, split, representative hash)
    for o, ss in c1["objectives"].items():
        for s, rec in ss.items():
            for h, g in zip(rec["hashes"], rec["groups"], strict=True):
                written[h] = (o, s)
                wgroup[g] = (o, s, h)
    c2_hashes = {
        h for ss in c2["objectives"].values() for rec in ss.values() for h in rec["hashes"]
    }
    rows, problems = [], []
    # final group(s) of each original group, through its members that are in a wme-c1 source pool
    og_final = collections.defaultdict(set)
    for h in reserved_items:
        g = plan["item_group"].get(h)
        if g is not None:
            og_final[orig[h][1]].add(g)
    for h in sorted(reserved_items):
        o_orig, g_orig = orig[h]
        g = plan["item_group"].get(h)
        in_pool = g is not None
        if not in_pool:
            fs = og_final.get(g_orig, set())
            if len(fs) != 1:
                problems.append(
                    {"item": h, "issue": f"original group maps to {len(fs)} written final groups"}
                )
                continue
            g = next(iter(fs))
        w = wgroup.get(g)
        if w is None:
            problems.append({"item": h, "issue": "final group not written"})
            continue
        rows.append(
            {
                "item": h,
                "original_group": g_orig,
                "original_objective": o_orig,
                "in_wme_c1_source_pool": in_pool,
                "final_group": g,
                "owner": w[0],
                "representative": w[2],
                "role": "representative"
                if w[2] == h
                else ("member" if in_pool else "out_of_pool_member"),
                "split": w[1],
            }
        )
    missing_groups = sorted(set(root_groups) - set(og_final))
    if missing_groups:
        problems.append(
            {
                "issue": "original groups with no member in a wme-c1 source pool",
                "n": len(missing_groups),
            }
        )
    eval_root_items = [
        r["item"] for r in rows if written.get(r["item"], (None, "construction"))[1] in EVAL_SPLITS
    ]
    eval_root_groups = [r for r in rows if r["split"] in EVAL_SPLITS]
    in_c2 = sorted(set(reserved_items) & c2_hashes)
    finals = {r["final_group"] for r in rows}
    reps = {r["representative"] for r in rows}
    role = collections.Counter(r["role"] for r in rows)
    fg_size = collections.Counter(r["final_group"] for r in rows)
    og_per_fg = collections.defaultdict(set)
    for r in rows:
        og_per_fg[r["final_group"]].add(r["original_group"])
    c1_resv = set(plan["reserved_representatives"])
    mapping_digest = hashlib.sha256(_canon(rows).encode()).hexdigest()
    report = {
        "label": "HASH-ONLY LINEAGE of the bank-root construction shards in wme-c1 (PREREG_WME §II.24.1)",
        "inputs": {
            "roots_file_sha256": hashlib.sha256(b2e.ROOTS.read_bytes()).hexdigest(),
            "wme_c1_manifest_sha256": hashlib.sha256(
                (ROOT / "data/wme/panels/wme-c1/manifest.json").read_bytes()
            ).hexdigest(),
            "wme_c2_gen_manifest_sha256": hashlib.sha256(
                (ROOT / "data/wme/panels/wme-c2-gen/manifest.json").read_bytes()
            ).hexdigest(),
            "allocation_sha256": alloc["sha256"],
            "split_files_read": False,
        },
        "reproduction": {
            "builder_plan_matches_manifest": True,
            "units_compared": sum(len(ss) for ss in c1["objectives"].values()),
        },
        "counts": {
            "original_groups": len(root_groups),
            "original_items": len(reserved_items),
            "items_mapped": len(rows),
            "final_union_groups": len(finals),
            "written_representatives": len(reps),
            "c1_manifest_reserved_construction_items": c1.get("reserved_construction_items"),
            "plan_reserved_representatives": len(c1_resv),
            "items_as_representative": role["representative"],
            "items_as_member": role["member"],
            "items_out_of_c1_pools": role["out_of_pool_member"],
            "out_of_pool_objectives": dict(
                collections.Counter(
                    r["original_objective"] for r in rows if not r["in_wme_c1_source_pool"]
                )
            ),
            "original_groups_mapped": len({r["original_group"] for r in rows}),
            "final_groups_holding_more_than_one_root_item": sum(
                1 for v in fg_size.values() if v > 1
            ),
            "final_groups_merging_original_groups": sum(
                1 for v in og_per_fg.values() if len(v) > 1
            ),
            "owner_differs_from_original_objective": sum(
                1 for r in rows if r["owner"] != r["original_objective"]
            ),
            "items_in_later_exposures": len(set(reserved_items) & X.all_exposed(exp1)),
        },
        "checks": {
            "every_item_mapped": not problems and len(rows) == len(reserved_items),
            "every_representative_in_construction": all(r["split"] == "construction" for r in rows),
            "root_item_hashes_in_c1_evaluation_splits": len(eval_root_items),
            "root_final_groups_in_c1_evaluation_splits": len(eval_root_groups),
            "root_item_hashes_in_wme_c2_gen": len(in_c2),
            "representatives_equal_plan_reserved": reps == c1_resv,
        },
        "problems": problems,
        "mapping_sha256": mapping_digest,
        "mapping": rows,
        "seconds": round(time.time() - t0, 1),
    }
    ok = (
        report["checks"]["every_item_mapped"]
        and report["checks"]["every_representative_in_construction"]
        and not eval_root_items
        and not eval_root_groups
        and not in_c2
        and report["checks"]["representatives_equal_plan_reserved"]
    )
    report["clean"] = bool(ok)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(
        json.dumps(
            {k: report[k] for k in ("counts", "checks", "clean", "mapping_sha256", "seconds")},
            indent=1,
        )
    )
    if not ok:
        raise SystemExit("LINEAGE NOT CLEAN")


if __name__ == "__main__":
    main()
