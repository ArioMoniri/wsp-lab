#!/usr/bin/env python3
"""Build the WM-E pilot-2 panel ``data/wme/panels/wme-p2`` (EXPLORATORY; CPU only; RULING-wme-pilot1 section 3).

Sources: the pilot-1 sources at their pinned revisions (``scripts/wme_build_panel.py``: WikiText-2, GSM8K, MMLU-Pro,
harmful/harmless prompts) plus the pinned, sha256-verified pilot-2 sources (``wm.p2data``): sycophancy-eval
``answer.jsonl`` and hendrycks MATH levels 4-5 minus MATH-500 (disjointness asserted by problem hash).

Every pilot-1 sealed test hash (from the pilot-1 manifest, which lists hashes only; the sealed file is not opened)
is excluded from every pilot-2 split, so no pilot-1 test item can reach an open pilot-2 split. Assignment is by
sha256(salt:hash) after deduplication; test items are written only under ``sealed_test/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import p2data as D
from wsp.wm import panel as P
from wsp.wm import panel2 as P2

SALT = "wme-p2-panel-20261116"
PILOT1_MANIFEST = ROOT / "data/wme/panels/wme-v0-p1/manifest.json"


def _s(c, v, d, t):
    return {"construction": c, "validation": v, "dev": d, "test": t}


SIZES = {
    "wikitext_nll": _s(64, 64, 64, 48),
    "gsm8k_gold_lp": _s(1024, 64, 128, 128),
    "mathcomp_gold_lp": _s(1024, 64, 128, 128),
    "mmlupro_gold_lp": _s(64, 64, 64, 48),
    "refusal_margin_harmful": _s(64, 64, 64, 48),
    "refusal_margin_harmless": _s(64, 64, 64, 48),
    "sycophancy_real": _s(1024, 64, 128, 128),
    "sycophancy_gap": _s(64, 64, 64, 48),
}


def _load_p1_builder():
    spec = importlib.util.spec_from_file_location(
        "wme_build_panel", ROOT / "scripts/wme_build_panel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def p1_pools(raw):
    """Pilot-1 item construction (``wm.panel.build_panel`` definitions), with the pilot-2 salt for the MMLU split."""
    pools = {}
    wiki = []
    for a in raw["wikitext"]:
        h = P.text_hash(a["text"])
        wiki.append(
            {
                "kind": "wikitext",
                "group": h[:16],
                "text": a["text"],
                "hash": h,
                "id": f"wiki:{h[:16]}",
            }
        )
    pools["wikitext_nll"] = wiki
    gsm = []
    for r in raw["gsm8k"]:
        h = P.text_hash(r["question"])
        gsm.append(
            {
                "kind": "gsm8k",
                "question": r["question"].strip(),
                "gold": P._gsm_gold(r["answer"]),
                "hash": h,
                "group": h[:16],
                "id": f"gsm:{h[:16]}",
            }
        )
    pools["gsm8k_gold_lp"] = gsm
    mm, seen = [], set()
    for r in raw["mmlu_pro"]:
        h = P.text_hash(r["question"] + "||" + "||".join(r["options"]))
        if h in seen:
            continue
        seen.add(h)
        mm.append(
            {
                "kind": "mmlu_pro",
                "question": r["question"],
                "options": list(r["options"]),
                "answer_index": int(r["answer_index"]),
                "hash": h,
                "group": h[:16],
                "id": f"mmlu:{h[:16]}",
            }
        )
    rank = lambda s, h: hashlib.sha256(f"{s}:{h}".encode()).hexdigest()
    mm.sort(key=lambda r: rank(SALT + ":mmlu-objective", r["hash"]))
    pools["mmlupro_gold_lp"] = mm[0::2]
    syc = []
    for r in mm[1::2]:
        wrong = [i for i in range(len(r["options"])) if i != r["answer_index"]]
        pick = wrong[int(rank(SALT + ":wrong", r["hash"]), 16) % len(wrong)]
        syc.append(dict(r, kind="sycophancy", wrong_index=pick, id="syc:" + r["id"].split(":")[1]))
    pools["sycophancy_gap"] = syc
    for src, obj in (
        ("harmful", "refusal_margin_harmful"),
        ("harmless", "refusal_margin_harmless"),
    ):
        rows = []
        for t in raw[src]:
            h = P.text_hash(t)
            rows.append(
                {
                    "kind": src,
                    "prompt": t.strip(),
                    "hash": h,
                    "group": h[:16],
                    "id": f"{src}:{h[:16]}",
                }
            )
        pools[obj] = rows
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-id", default="wme-p2")
    ap.add_argument("--out", default=str(ROOT / "data/wme/panels"))
    args = ap.parse_args()
    out = Path(args.out) / args.panel_id
    if out.exists():
        raise SystemExit(f"{out} exists; panels are frozen once built")
    b1 = _load_p1_builder()
    raw, meta = b1.load_raw()
    pools = p1_pools(raw)
    syc, syc_meta = D.load_sycophancy_cached()
    math, math_meta = D.load_math_cached()
    pools["sycophancy_real"] = P2.syc_items(syc)
    pools["mathcomp_gold_lp"] = P2.mathcomp_items(math)
    p1 = json.loads(PILOT1_MANIFEST.read_text())
    exclude = set(p1["sealed_hashes"])
    man = P2.build_panel2(pools, SIZES, SALT, out, exclude_hashes=exclude)
    # MATH-500 disjointness re-asserted on every written split (open and sealed)
    m500 = set()
    from datasets import load_dataset

    m500 = D.problem_hashes(
        load_dataset(D.MATH500["repo"], revision=D.MATH500["revision"], split="test")["problem"]
    )
    written = {h for s in man["objectives"]["mathcomp_gold_lp"].values() for h in s["hashes"]}
    if written & m500:
        raise SystemExit("MATH-500 overlap in the written panel")
    man["math500_disjoint_checked"] = {"n_written": len(written), "overlap": 0}
    man["sources"] = {**meta, "sycophancy_answer": syc_meta, "hendrycks_math": math_meta}
    man["pilot1_sealed_excluded"] = {
        "manifest": "data/wme/panels/wme-v0-p1/manifest.json",
        "n": len(exclude),
    }
    man["label"] = "EXPLORATORY (WM-E pilot 2)"
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                "panel": str(out),
                "content_hash": man["content_hash"],
                "dedup": man["dedup"],
                "excluded": man["excluded"],
                "math": {k: math_meta[k] for k in ("unique", "dropped_math500_overlap", "n_items")},
                "syc_items": syc_meta["n_items"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
