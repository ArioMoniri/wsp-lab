#!/usr/bin/env python3
"""Hash-only exposure audit and split feasibility for ``wme-c1`` (Board A condition 1). CPU only.

Builds the source pools exactly as the pilot-2 panel builder does (pinned revisions, sha256-checked sources), attaches
near-duplicate groups (``wm.exposure``), drops every group touching a registered exposure (manifest hash lists only;
no sealed or listed file is opened) and reports, per objective: pool items, groups, excluded groups/items, available
groups and feasibility of the proposed sizes. No split is assigned or written; no model is loaded; nothing is scored.
Output: counts and digests only (no item text).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import exposure as X
from wsp.wm import panel2 as P2
from wsp.wm import panel_c1 as C1


def _p2_builder():
    spec = importlib.util.spec_from_file_location(
        "wme_p2_build_panel", ROOT / "scripts/wme_p2_build_panel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pools():
    from wsp.wm import p2data as D

    b2 = _p2_builder()
    raw, meta = b2._load_p1_builder().load_raw()
    out = b2.p1_pools(raw)
    syc, syc_meta = D.load_sycophancy_cached()
    math, math_meta = D.load_math_cached()
    out["sycophancy_real"] = P2.syc_items(syc)
    out["mathcomp_gold_lp"] = P2.mathcomp_items(math)
    return out, {
        "p1_sources": meta,
        "sycophancy": syc_meta,
        "math": {k: math_meta[k] for k in ("unique", "dropped_math500_overlap", "n_items")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(ROOT / "docs/reviews/wme/wme-c1-exposure-audit-2026-11-17.json")
    )
    args = ap.parse_args()
    t0 = time.time()
    exposures = X.load_exposures(ROOT)
    pool, meta = pools()
    grouped = C1.grouped_pool(pool)
    report = {
        "label": "hash-only exposure audit (Board A condition 1); no split written, nothing scored",
        "salt": C1.SALT,
        "exposures": {n: {s: len(h) for s, h in e.items()} for n, e in exposures.items()},
        "exposure_registry": X.EXPOSURES,
        "near_duplicate_policy": {
            "shingle_words": X.NEAR_DUP_SHINGLE,
            "jaccard": X.NEAR_DUP_JACCARD,
            "minhash_perms": X.MINHASH_PERMS,
            "bands": X.MINHASH_BANDS,
            "seed": X.MINHASH_SEED,
        },
        "objectives": {},
    }
    owner, available = {}, {}
    order = [o for o in P2.OBJECTIVES if o in grouped]
    for obj in order:
        rows = grouped[obj]
        kept, rep = X.exclude_groups(rows, exposures)
        groups = {r["group"] for r in kept if owner.setdefault(r["group"], obj) == obj}
        available[obj] = len(groups)
        exact = len({r["hash"] for r in rows})
        report["objectives"][obj] = {
            "pool_items": len(rows),
            "pool_unique_hashes": exact,
            "pool_groups": len({r["group"] for r in rows}),
            **rep,
            "available_groups": len(groups),
            "post_exclusion_intersection_with_exposures": X.intersection_audit(
                {"available": {r["hash"] for r in kept}}, exposures
            ),
            "available_digest": hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest(),
        }
    report["feasibility"] = X.feasibility(available, X.DEFAULT_SIZES)
    report["sources"] = meta
    report["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(
        json.dumps(
            {
                o: {k: v[k] for k in ("pool_items", "excluded_groups", "available_groups")}
                for o, v in report["objectives"].items()
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                o: (v["requested"], v["available_groups"], v["feasible"])
                for o, v in report["feasibility"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
