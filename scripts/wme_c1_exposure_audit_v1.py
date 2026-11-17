#!/usr/bin/env python3
"""Exposure re-audit under the frozen v1 near-duplicate policy (RULING-wme-v2 A4-A5). Hash-only; nothing scored,
no split written, no sealed file opened (exposures come from manifest hash lists only).

One joint grouping (``wm.neardup``: exact sha256 joins + exact character-5-gram Jaccard >= 0.8) over every objective's
source pool, the chosen harmful-refusal supplement (AdvBench + HarmBench standard) and the pinned TriviaQA
generation pool. A group is excluded when it contains any exposed hash (pilot 1 and 2 panels including sealed
hashes, the 27B calibration reads, engineering smoke reads). TriviaQA groups are additionally excluded when they
contain any item of any existing source pool (every wme-c1 candidate item). Cross-objective groups keep the
first objective in the registered order, as the wme-c1 builder does.

    python scripts/wme_c1_exposure_audit_v1.py [--out docs/reviews/wme/wme-c1-exposure-audit-v1-2026-11-17.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import exposure as X
from wsp.wm import neardup as ND
from wsp.wm import panel2 as P2
from wsp.wm import supplement as SU
from wsp.wm import triviaqa as TQ

ORDER = list(P2.OBJECTIVES) + ["triviaqa_gen"]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(ROOT / "docs/reviews/wme/wme-c1-exposure-audit-v1-2026-11-17.json")
    )
    ap.add_argument("--pool-out", default=str(ROOT / "data/wme/triviaqa/pool-v1.jsonl"))
    a = ap.parse_args()
    t0 = time.time()
    exposures = X.load_exposures(ROOT)
    exposed = X.all_exposed(exposures)
    pools, _ = _mod("c1audit", "scripts/wme_c1_exposure_audit.py").pools()
    supp = []
    for s in SU.SOURCES[
        :2
    ]:  # AdvBench + HarmBench standard: the chosen sources (track-A report §3)
        p = ROOT / SU.RAW / s["local"]
        SU.verify(p, s["sha256"])
        for r in SU.parse(s["name"], p):
            supp.append(
                {
                    "kind": s["name"],
                    "prompt": r["prompt"],
                    "hash": SU.text_hash(r["prompt"]),
                    "id": f"{s['name']}:{r['source_id']}",
                }
            )
    pools["refusal_margin_harmful"] = list(pools["refusal_margin_harmful"]) + supp
    tq = TQ.items_from_rows(TQ.load_rows(ROOT))
    pools["triviaqa_gen"] = tq
    texts, where = {}, {}
    for obj in ORDER:
        for i, r in enumerate(pools.get(obj, [])):
            k = f"{obj}\x00{i:07d}"
            texts[k] = r["question"] if obj == "triviaqa_gen" else X.group_text(obj, r)
            where[k] = (obj, r)
    t1 = time.time()
    rep = ND.groups(texts)
    t_join = time.time() - t1
    members: dict = {}
    for k, g in rep.items():
        members.setdefault(g, []).append(k)
    bad = {g for g, ks in members.items() if any(where[k][1]["hash"] in exposed for k in ks)}
    pool_items = {g for g, ks in members.items() if any(where[k][0] != "triviaqa_gen" for k in ks)}
    out = {
        "label": "wme-c1 exposure re-audit under the frozen v1 near-duplicate policy (hash-only)",
        "policy": ND.POLICY,
        "exposures": {n: {s: len(h) for s, h in e.items()} for n, e in exposures.items()},
        "n_texts": len(texts),
        "n_groups": len(members),
        "join_seconds": round(t_join, 1),
        "objectives": {},
    }
    owner = {}
    available = {}
    for obj in ORDER:
        ks = [k for k in sorted(texts) if where[k][0] == obj]
        gs = {rep[k] for k in ks}
        excl = {g for g in gs if g in bad}
        if obj == "triviaqa_gen":
            excl |= {g for g in gs if g in pool_items}
        keep = set()
        for g in sorted(gs - excl):
            if owner.setdefault(g, obj) == obj:
                keep.add(g)
        available[obj] = keep
        out["objectives"][obj] = {
            "pool_items": len(ks),
            "pool_groups": len(gs),
            "excluded_groups": len(excl),
            "available_groups": len(keep),
            "available_digest": hashlib.sha256("\n".join(sorted(keep)).encode()).hexdigest(),
        }
    # hash-only intersection check: no available item hash is exposed
    out["post_exclusion_exposed_hashes"] = sum(
        1
        for obj in ORDER
        for k in texts
        if where[k][0] == obj and rep[k] in available[obj] and where[k][1]["hash"] in exposed
    )
    if any(where[k][0] == "sycophancy_real" for k in texts):
        syc = [
            k
            for k in texts
            if where[k][0] == "sycophancy_real" and rep[k] in available["sycophancy_real"]
        ]
        out["sycophancy_real_triviaqa_derived_groups"] = len(
            {rep[k] for k in syc if where[k][1].get("source_dataset") == "trivia_qa"}
        )
    out["triviaqa_pin"] = TQ.PIN
    out["triviaqa_incorrect_rule"] = (
        TQ.__doc__.split("**Incorrect answer")[1].split("* **Scoring")[0].strip()
    )
    seen, pool = set(), []
    for k in sorted(texts):
        if (
            where[k][0] == "triviaqa_gen"
            and rep[k] in available["triviaqa_gen"]
            and rep[k] not in seen
        ):
            seen.add(rep[k])
            pool.append(dict(where[k][1], group=rep[k]))
    Path(a.pool_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.pool_out).write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in pool))
    out["triviaqa_pool_size"] = len(pool)
    out["triviaqa_pool_file_sha256"] = hashlib.sha256(Path(a.pool_out).read_bytes()).hexdigest()
    out["seconds"] = round(time.time() - t0, 1)
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                o: (v["pool_items"], v["excluded_groups"], v["available_groups"])
                for o, v in out["objectives"].items()
            },
            indent=1,
        )
    )
    print(
        "triviaqa pool",
        out["triviaqa_pool_size"],
        "join s",
        out["join_seconds"],
        "rss",
        out["peak_rss_mb"],
    )


if __name__ == "__main__":
    main()
