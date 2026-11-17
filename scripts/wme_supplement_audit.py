#!/usr/bin/env python3
"""Hash-only audit of the harmful-refusal supplement sources (RULING-wme-resources §3; PREREG_WME v1 §3.2-3.3).

Verifies each pinned source file (revision + sha256), parses it, and deduplicates it under the frozen v1 policy
against every existing wme-c1 source pool (all objectives) and every exposure hash (pilot panels, sealed sets by
hash only, engineering exposures), and against earlier sources in preference order. Writes counts and hash digests
only; no prompt text, no split and nothing scored. The word-5-gram (0.7) policy of the 17 Nov exposure audit is
reported alongside as a sensitivity count.

    python scripts/wme_supplement_audit.py [--out docs/reviews/wme/wme-supplement-audit-2026-11-17.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import exposure as X
from wsp.wm import supplement as SU


def _audit_module():
    spec = importlib.util.spec_from_file_location(
        "c1audit", ROOT / "scripts/wme_c1_exposure_audit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(ROOT / "docs/reviews/wme/wme-supplement-audit-2026-11-17.json")
    )
    a = ap.parse_args()
    t0 = time.time()
    exposures = X.load_exposures(ROOT)
    exposed = X.all_exposed(exposures)
    pools, _meta = _audit_module().pools()
    existing = {}
    for obj, rows in pools.items():
        for i, r in enumerate(rows):
            existing[f"{obj}:{i:06d}"] = X.group_text(obj, r)
    by_source, pins = {}, {}
    for s in SU.SOURCES:
        p = ROOT / SU.RAW / s["local"]
        pins[s["name"]] = {k: s[k] for k in ("title", "repo", "revision", "path", "url", "sha256")}
        pins[s["name"]]["verified_sha256"] = SU.verify(p, s["sha256"])
        if "subset" in s:
            pins[s["name"]]["subset"] = s["subset"]
        by_source[s["name"]] = SU.parse(s["name"], p)
    res = SU.dedup_sources(by_source, existing, exposed)
    # sensitivity: the 17 Nov word-5-gram / 0.7 policy over the same items (counts only)
    sens = {}
    for name, items in by_source.items():
        texts = {f"new:{i:06d}": r["prompt"] for i, r in enumerate(items)}
        texts.update({f"old:{k}": v for k, v in existing.items() if k.startswith("refusal_margin")})
        rep = X.near_duplicate_groups(texts)
        old_groups = {rep[k] for k in texts if k.startswith("old:")}
        hit = sum(1 for k in texts if k.startswith("new:") and rep[k] in old_groups)
        sens[name] = {"word5_jaccard07_dup_refusal_pool_items": hit}
    c1 = json.loads((ROOT / "docs/reviews/wme/wme-c1-exposure-audit-2026-11-17.json").read_text())
    existing_avail = c1["objectives"]["refusal_margin_harmful"]["available_groups"]
    cum, run = {}, existing_avail
    for s in SU.SOURCES:
        run += res[s["name"]]["counts"]["available_groups"]
        cum[s["name"]] = run
    out = {
        "label": "harmful-refusal supplement audit (hash-only; RULING-wme-resources §3); nothing scored",
        "policy": {
            "char_k": SU.NEAR_DUP_K,
            "jaccard": SU.NEAR_DUP_JACCARD,
            "exact_below_groups": SU.EXACT_LIMIT,
            "normalise": "wm.panel.normalise, punctuation stripped",
            "exact_join": "wm.panel.text_hash",
        },
        "pins": pins,
        "existing_pool_items_all_objectives": len(existing),
        "exposure_hashes": len(exposed),
        "existing_harmful_available_groups": existing_avail,
        "sources": {
            n: {**res[n]["counts"], "available_digest": res[n]["available_digest"]} for n in res
        },
        "cumulative_available_in_preference_order": cum,
        "sensitivity_word5_policy": sens,
        "jbb_origin_counts": {
            o: sum(1 for r in by_source["jbb_behaviors"] if r.get("origin") == o)
            for o in sorted({r.get("origin") for r in by_source["jbb_behaviors"]})
        },
        "available_hashes": {n: res[n]["available"] for n in res},
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "sources",
                    "cumulative_available_in_preference_order",
                    "sensitivity_word5_policy",
                    "seconds",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
