#!/usr/bin/env python3
"""Build the confirmatory panels ``wme-c1`` and ``wme-c2-gen`` after stage W2 (PREREG_WME §II.23.3; Board A v2
required change 3). CPU only; nothing is scored; no sealed file is ever read back; no ``data/evals``.

1. **Exposures.** ``wm.exposure.registry_before`` (pilots 1-2 with sealed hash lists, the 27B calibration, smokes,
   ``wme-p3s``, ``wme-p3h``, ``wme-w``, ``wme-w2``), from manifest hash lists only.
2. **Bank-root shards.** The item groups the sampled bank roots used (``contrast_source_groups``,
   ``adapter_source_groups`` of ``data/wme/bank_roots/roots-9b-20261117/roots.json``) are mapped to item hashes through
   the same ``panel_c1.grouped_pool`` the root sampler used; they are committed construction items.
3. **Joint MMLU-Pro allocation** (``panels_v2d.allocate``, salt ``wme-mmlupro-v2c``): pool = every MMLU-Pro item of
   either builder's pool; root-shard groups pre-assigned to c1 construction; 232 fresh groups to c1, 3,463 to
   c2-gen (reduced on shortfall), the rest reserved. Written to ``data/wme/panels/wme-mmlu-allocation-v2e.json``.
4. **wme-c1** (``panels_v2d.build_c1_panel_v2e``): the registered sizes; construction = rest + root shards.
5. **wme-c2-gen** (``panels_v2d.build_c2gen_panel``) with ``wme-c1`` as an exposure and every wme-c1 source pool
   (except the TriviaQA question pool, whose 1,000 written items enter step 6) reserved.
6. **Hash-only audit** (``panels_v2d.audit`` and per-source counts) over every (panel, split): cross-panel hash and
   joint-group overlaps; exposure intersections, where the only permitted hits are disclosed root-shard
   construction items. Written to ``docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json`` (counts, hashes, digests).

    wme_v2e_build_panels.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PANELS = ROOT / "data/wme/panels"
ROOTS = ROOT / "data/wme/bank_roots/roots-9b-20261117/roots.json"
ALLOC = PANELS / "wme-mmlu-allocation-v2e.json"
AUDIT = ROOT / "docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def c1_inputs(man: dict):
    """The wme-c1 inputs exactly as the build uses them (also imported by scripts/wme_v2f_root_lineage.py):
    the c1 source pools (+ the harmful supplement and the TriviaQA pool), the registered sizes, the fresh
    construction minimums, and the bank-root shard items mapped through ``panel_c1.grouped_pool``."""
    from wsp.wm import panel_c1 as C1
    from wsp.wm import supplement as SU
    from wsp.wm import triviaqa as TQ

    # ---- pools of wme-c1 (as the v1 audit and the root sampler formed them)
    aud = _mod("c1audit", "scripts/wme_c1_exposure_audit.py")
    base_pools, meta_c1 = aud.pools()
    grouped = C1.grouped_pool(base_pools)
    roots = json.loads(ROOTS.read_text())
    root_groups = set()
    for r in roots["roots"]:
        root_groups |= set(r.get("contrast_source_groups", []) or [])
        root_groups |= set(r.get("adapter_source_groups", []) or [])
    reserved_items, found, root_orig = set(), set(), {}
    for obj, rows in grouped.items():
        for r in rows:
            if r["group"] in root_groups:
                reserved_items.add(r["hash"])
                found.add(r["group"])
                root_orig[r["hash"]] = (obj, r["group"])
    if found != root_groups:
        raise SystemExit(f"{len(root_groups - found)} root-shard groups not found in the pools")
    pools = dict(base_pools)
    supp = []
    for s in SU.SOURCES[:2]:
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
    pools["triviaqa_gen"] = TQ.items_from_rows(TQ.load_rows(ROOT))
    sizes_reg = man["panel.wme_c1.sizes"]
    pools = {o: v for o, v in pools.items() if o in sizes_reg}
    sizes, cmin = {}, {}
    for o, s in sizes_reg.items():
        if o == "triviaqa_gen":
            sizes[o] = {
                "sealed_gen": s["sealed_gen"],
                "sealed_test": 0,
                "dev": 0,
                "bank": 0,
                "construction": 0,
            }
            continue
        sizes[o] = {
            "sealed_gen": s.get("sealed_gen", 0),
            "sealed_test": s["sealed_test"],
            "dev": s["dev"],
            "bank": s.get("bank", 0),
            "construction": "rest",
        }
        cmin[o] = s.get("construction_field", 0) + s.get("construction_e2", 0)

    return pools, sizes, cmin, reserved_items, root_groups, meta_c1, root_orig


def main():
    t0 = time.time()
    from wsp.wm import exposure as X
    from wsp.wm import p3gen as G
    from wsp.wm import panel_c1 as C1
    from wsp.wm import panels_v2d as V
    from wsp.wm import stagew as W
    from wsp.wm import supplement as SU
    from wsp.wm import triviaqa as TQ
    from wsp.wm.panel import text_hash

    for d in ("wme-c1", "wme-c2-gen"):
        if (PANELS / d).exists():
            raise SystemExit(f"{PANELS / d} exists; panels are frozen once built")
    man = json.loads((ROOT / "docs/prereg/wme_manifest.json").read_text())
    log = {"t0": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    pools, sizes, cmin, reserved_items, root_groups, meta_c1, _ = c1_inputs(man)
    sizes_reg = man["panel.wme_c1.sizes"]

    # ---- MMLU-Pro sources of both builders (same hash: text_hash(question || options))
    b1 = _mod("b1", "scripts/wme_build_panel.py")
    raw, meta_raw = b1.load_raw()
    p3 = _mod("p3b", "scripts/wme_p3_build_panels.py")
    psg, sha_sg = p3._hub_file("supergpqa")
    sg = []
    for line in psg.read_text().splitlines():
        r = json.loads(line)
        sg.append(
            {
                "question": r["question"],
                "options": list(r["options"]),
                "answer_index": G.LETTERS.index(r["answer_letter"]),
            }
        )
    mc = W.panel_pools(raw["mmlu_pro"], sg)
    mm_pool = {r["hash"]: r["gtext"] for r in mc["mmlupro_gen"]}
    for r in pools["mmlupro_gold_lp"]:
        mm_pool.setdefault(r["hash"], X.group_text("mmlupro_gold_lp", r))

    # ---- exposures, allocation, wme-c1
    exp1 = X.load_exposures(ROOT, X.registry_before("wme-c1"))
    V.require_exposures(exp1)
    pre = {h: V.C1_ROOT_CONSUMER for h in reserved_items if h in mm_pool}
    eligible = {
        V.CONSUMERS[0][0]: {r["hash"] for r in pools["mmlupro_gold_lp"]},
        V.CONSUMERS[1][0]: {r["hash"] for r in mc["mmlupro_gen"]},
    }
    # c1 must end with 232 usable fresh MMLU-Pro groups (sealed_test 96 + dev 64 + bank 48 + fresh field 24) after
    # its own cross-objective grouping; the allocation's c1 count is raised by the deficit until it does
    # (deterministic, counts only).
    need_c1 = (
        sum(sizes["mmlupro_gold_lp"][s] for s in ("sealed_test", "dev", "bank"))
        + cmin["mmlupro_gold_lp"]
    )
    n_c1 = V.CONSUMERS[0][1]
    for _ in range(20):
        consumers = ((V.CONSUMERS[0][0], n_c1, "raise"), V.CONSUMERS[1])
        alloc = V.allocate(
            mm_pool, X.all_exposed(exp1), consumers, preassigned=pre, eligible=eligible
        )
        got = V.build_c1_panel_v2e(
            pools, sizes, PANELS / "wme-c1", exp1, alloc, reserved_items, dry_run=True
        )["mmlupro_gold_lp"]
        if got >= need_c1:
            break
        n_c1 += need_c1 - got
    else:
        raise SystemExit("could not reach the c1 MMLU-Pro requirement")
    log["c1_mmlu_allocation_n"] = {"registered": V.CONSUMERS[0][1], "dealt": n_c1, "usable": got}
    log["mmlu_pools"] = {k: len(v) for k, v in eligible.items()} | {"union": len(mm_pool)}
    ALLOC.parent.mkdir(parents=True, exist_ok=True)
    ALLOC.write_text(json.dumps(alloc, sort_keys=True))
    log["allocation_s"] = round(time.time() - t0)
    man1, sp1 = V.build_c1_panel_v2e(
        pools, sizes, PANELS / "wme-c1", exp1, alloc, reserved_items, construction_min=cmin
    )
    log["c1_s"] = round(time.time() - t0)

    # ---- wme-c2-gen
    exp2 = X.load_exposures(ROOT, X.registry_before("wme-c2-gen"))
    pins = man["guards.generated"]["pins"]
    from huggingface_hub import hf_hub_download

    gh = pins["guard_harmless"]
    p_or = Path(
        hf_hub_download(gh["repo"], gh["file"], repo_type="dataset", revision=gh["revision"])
    )
    if sha_file(p_or) != gh["sha256"]:
        raise SystemExit("OR-Bench-80k sha256 mismatch")
    with open(p_or, newline="", encoding="utf-8") as f:
        or80 = [r["prompt"] for r in csv.DictReader(f) if r["prompt"].strip()]
    gx = pins["guard_harmful"]
    p_lat = Path(
        hf_hub_download(gx["repo"], gx["file"], repo_type="dataset", revision=gx["revision"])
    )
    if sha_file(p_lat) != gx["sha256"]:
        raise SystemExit("LLM-LAT sha256 mismatch")
    import pandas as pd

    lat = [t for t in pd.read_parquet(p_lat)["prompt"].tolist() if t and t.strip()]

    def prompt(t, prefix):
        t = t.strip()
        h = text_hash(t)
        return {"kind": prefix, "prompt": t, "hash": h, "id": f"{prefix}:{h[:16]}", "gtext": t}

    c2_pools = {
        "mmlupro_gen": mc["mmlupro_gen"],
        "supergpqa_gen": mc["supergpqa_gen"],
        "guard_harmless": [prompt(t, "or80k") for t in or80],
        "guard_harmful": [prompt(t, "llmlat") for t in lat],
    }
    # every written wme-c1 item (all splits, sealed included: held in memory by this build only) and every wme-c1
    # source pool except MMLU-Pro (separated by the joint allocation) and the TriviaQA question pool
    reserved_texts = {
        f"wme-c1-pool:{o}": [C1._text_v1(o, r, X.group_text) for r in rows]
        for o, rows in pools.items()
        if o not in ("triviaqa_gen", "mmlupro_gold_lp")
    }
    reserved_texts["wme-c1-written"] = [
        C1._text_v1(o, r, X.group_text)
        for o, ss in sp1.items()
        for rows in ss.values()
        for r in rows
    ]
    man2 = V.build_c2gen_panel(c2_pools, PANELS / "wme-c2-gen", exp2, alloc, reserved_texts)
    log["c2_s"] = round(time.time() - t0)

    # ---- hash-only audit over every (panel, split)
    texts, panels, source_of = {}, {}, {}
    for o, ss in sp1.items():
        for s, rows in ss.items():
            panels.setdefault(f"wme-c1/{o}", {})[s] = [r["hash"] for r in rows]
            for r in rows:
                texts[r["hash"]] = C1._text_v1(o, r, X.group_text)
                source_of[r["hash"]] = r.get("kind") or r.get("source_dataset") or o
    by_hash = {r["hash"]: r for rows in c2_pools.values() for r in rows}
    for p, ss in man2["objectives"].items():
        for s, rec in ss.items():
            panels.setdefault(f"wme-c2-gen/{p}", {})[s] = rec["hashes"]
            for h in rec["hashes"]:
                texts[h] = by_hash[h]["gtext"]
    a = V.audit(panels, texts, exp1)
    resv = set(reserved_items)
    hits_outside = sum(
        v
        for e in a["exposure"].values()
        for es in e.values()
        for unit, v in es.items()
        if v and not unit.endswith("/construction")
    )
    constr_hit_items = {
        h
        for unit, hs in ((f"{p}/{s}", hs) for p, ss in panels.items() for s, hs in ss.items())
        if unit.endswith("/construction")
        for h in hs
        if h in X.all_exposed(exp1)
    }
    rows = []
    for o, s in sizes_reg.items():
        ex = man1["exclusion"][o]
        for split in ("sealed_gen", "sealed_test", "dev", "bank", "construction"):
            n = man1["objectives"][o][split]["n"]
            req = (
                s.get("construction_field", 0) + s.get("construction_e2", 0)
                if split == "construction"
                else s.get(split, 0)
            )
            if not req and not n:
                continue
            rows.append(
                {
                    "panel": "wme-c1",
                    "source": o,
                    "split": split,
                    "requested": req
                    if split != "construction"
                    else f">= {req} (rest + root shards)",
                    "available_fresh_groups": ex["available_fresh_groups"],
                    "built": n,
                    "hashes_digest": hashlib.sha256(
                        "\n".join(man1["objectives"][o][split]["hashes"]).encode()
                    ).hexdigest(),
                }
            )
    for p, ss in man2["objectives"].items():
        for split, rec in ss.items():
            rows.append(
                {
                    "panel": "wme-c2-gen",
                    "source": p,
                    "split": split,
                    "requested": man2["sizes_required"][p][split],
                    "available_fresh_groups": man2["stats"][p]["usable_groups"],
                    "built": rec["n"],
                    "hashes_digest": hashlib.sha256("\n".join(rec["hashes"]).encode()).hexdigest(),
                }
            )
    report = {
        "label": "HASH-ONLY AUDIT of the confirmatory panels wme-c1 and wme-c2-gen (PREREG_WME §II.23.3)",
        "built": log,
        "exposures": {n: {s: len(h) for s, h in e.items()} for n, e in exp2.items()},
        "allocation": {k: v for k, v in alloc.items() if k != "assignment"}
        | {"file": str(ALLOC.relative_to(ROOT)), "file_sha256": sha_file(ALLOC)},
        "root_shards": {
            "roots_file_sha256": sha_file(ROOTS),
            "groups": len(root_groups),
            "items": len(reserved_items),
            "construction_items_also_in_later_exposures": len(constr_hit_items),
            "all_such_items_are_root_shards": constr_hit_items <= resv,
        },
        "rows": rows,
        "wme_c1": {
            "manifest_sha256": sha_file(PANELS / "wme-c1/manifest.json"),
            "content_hash": man1["content_hash"],
            "split_hashes": man1["split_hashes"],
            "exclusion": man1["exclusion"],
        },
        "wme_c2_gen": {
            "manifest_sha256": sha_file(PANELS / "wme-c2-gen/manifest.json"),
            "content_hash": man2["content_hash"],
            "split_hashes": man2["split_hashes"],
            "shortfall": man2["shortfall"],
            "stats": man2["stats"],
        },
        "cross_panel": {
            "pairs_with_overlap": [
                p for p in a["pairs"] if p["hash_overlap"] or p["group_overlap"]
            ],
            "n_pairs": len(a["pairs"]),
            "audit_sha256": a["sha256"],
        },
        "exposure_hits_outside_root_construction": hits_outside,
        "clean": bool(
            not [p for p in a["pairs"] if p["hash_overlap"] or p["group_overlap"]]
            and hits_outside == 0
            and constr_hit_items <= resv
        ),
        "sources": {
            "c1": meta_c1,
            "mmlu_pro": meta_raw.get("mmlu_pro"),
            "supergpqa": {"pin": p3.PINS["supergpqa"], "sha256": sha_sg},
            "guards": pins,
            "triviaqa": TQ.PIN,
            "supplement": [s["name"] for s in SU.SOURCES[:2]],
        },
        "seconds": round(time.time() - t0),
    }
    AUDIT.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(
        json.dumps(
            {k: report[k] for k in ("clean", "exposure_hits_outside_root_construction", "seconds")}
        )
    )
    for r in rows:
        print(
            r["panel"],
            r["source"],
            r["split"],
            r["requested"],
            r["available_fresh_groups"],
            r["built"],
        )
    if not report["clean"]:
        raise SystemExit("AUDIT NOT CLEAN")


if __name__ == "__main__":
    main()
