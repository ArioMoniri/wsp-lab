#!/usr/bin/env python3
"""Build the pilot-3 panels ``data/wme/panels/wme-p3s`` and ``wme-p3h`` (RULING-wme-pilot3 §1; EXPLORATORY; CPU).

Sources are pinned by revision (and, where a file is downloaded, its sha256 is recorded in the manifest). Every
exposure in ``wm.exposure.EXPOSURES`` (pilot-1 and pilot-2 panels including their sealed-test hash lists; sealed
files are never opened) is excluded at near-duplicate-group level; the reserved wme-c1 pools (harmful_behaviors,
AdvBench, HarmBench, JBB, WikiText-2) and their near-duplicates are never used. Nothing is scored here.

    wme_p3_build_panels.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import exposure as X
from wsp.wm import p2data as D2
from wsp.wm import p3gen as G
from wsp.wm import p3panel as PP
from wsp.wm import supplement as SUP
from wsp.wm.panel import _gsm_gold, text_hash

SALT_S = "wme-p3s-panel-20261117"
SALT_H = "wme-p3h-panel-20261117"
PINS = {
    "wikitext103": (
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "b08601e04326c79dfdd32d625aee71d232d685c3",
    ),
    "supergpqa": (
        "m-a-p/SuperGPQA",
        "SuperGPQA-all.jsonl",
        "4430d4458112c7d4497fdcf94d7cc223313d6acf",
    ),
    "orbench_hard": (
        "bench-llm/or-bench",
        "or-bench-hard-1k.csv",
        "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
    ),
    "orbench_toxic": (
        "bench-llm/or-bench",
        "or-bench-toxic.csv",
        "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
    ),
    "llmlat_harmful": (
        "LLM-LAT/harmful-dataset",
        "data/train-00000-of-00001.parquet",
        "8bfba31bc6d93a5b71808fee5275ef4b6330ed91",
    ),
    "alpaca_cleaned": ("yahma/alpaca-cleaned", None, "12567cabf869d7c92e573c7c783905fc160e9639"),
}
WIKI103_CANDIDATES = (
    1500  # salted-rank prefix of WikiText-103 train articles considered (CPU cost of MinHash)
)
S_PANELS = (
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "sycophancy_gap",
)
H_PANELS = (
    "mmlupro_gen",
    "supergpqa_gen",
    "gsm8k_gen",
    "math_gen",
    "ifeval_gen",
    "guard_harmless",
    "guard_harmful",
    "oldguard_harmless",
)
S_SIZES = {p: {"validation": 96, "construction": 96} for p in S_PANELS}
H_SIZES = {
    "mmlupro_gen": {"dev": 1000, "holdout": 256, "construction": 1000},
    "supergpqa_gen": {"dev": 1000, "holdout": 256, "construction": 1000},
    "gsm8k_gen": {"dev": 300, "holdout": 128, "construction": 512},
    "math_gen": {"dev": 300, "holdout": 128, "construction": 512},
    "ifeval_gen": {"dev": 300},
    "guard_harmless": {"dev": None, "holdout": 128},  # dev = the pool rest
    "guard_harmful": {"dev": None, "holdout": 128},
    "oldguard_harmless": {"dev": 64},
}
SPLIT_ORDER = {
    **{p: ("validation", "construction") for p in S_PANELS},
    **{p: tuple(H_SIZES[p]) for p in H_PANELS},
}


def _hub_file(key):
    from huggingface_hub import hf_hub_download

    repo, fname, rev = PINS[key]
    p = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
    return Path(p), SUP.sha256_file(p)


def _rank(salt, h):
    return hashlib.sha256(f"{salt}:{h}".encode()).hexdigest()


def _mc(r, prefix):
    h = text_hash(r["question"] + "||" + "||".join(r["options"]))
    return {
        "kind": prefix,
        "question": r["question"],
        "options": list(r["options"]),
        "answer_index": int(r["answer_index"]),
        "hash": h,
        "id": f"{prefix}:{h[:16]}",
        "gtext": r["question"] + "||" + "||".join(r["options"]),
    }


def _prompt(t, prefix):
    t = t.strip()
    h = text_hash(t)
    return {"kind": prefix, "prompt": t, "hash": h, "id": f"{prefix}:{h[:16]}", "gtext": t}


def load_sources():
    from datasets import load_dataset

    spec = importlib.util.spec_from_file_location("b1", ROOT / "scripts/wme_build_panel.py")
    b1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b1)
    raw, meta = b1.load_raw()
    files = {}
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    repo, cfg, rev = PINS["wikitext103"]
    w103 = load_dataset(repo, cfg, revision=rev, split="train")
    cur, arts = [], []
    for line in w103["text"]:
        if line.startswith(" = ") and not line.startswith(" = = ") and cur:
            t = "".join(cur).strip()
            if len(t) >= 3000:
                arts.append(t)
            cur = []
        cur.append(line)
    arts.sort(key=lambda t: _rank(SALT_S + ":wiki103", text_hash(t)))
    raw["wikitext103"] = arts[:WIKI103_CANDIDATES]
    meta["wikitext103"] = {"repo": repo, "config": cfg, "revision": rev, "articles": len(arts)}
    p, files["supergpqa"] = _hub_file("supergpqa")
    raw["supergpqa"] = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        raw["supergpqa"].append(
            {
                "question": r["question"],
                "options": list(r["options"]),
                "answer_index": G.LETTERS.index(r["answer_letter"]),
            }
        )
    for key in ("orbench_hard", "orbench_toxic"):
        p, files[key] = _hub_file(key)
        with open(p, newline="", encoding="utf-8") as f:
            raw[key] = [r["prompt"] for r in csv.DictReader(f) if r["prompt"].strip()]
    import pandas as pd

    p, files["llmlat_harmful"] = _hub_file("llmlat_harmful")
    raw["llmlat_harmful"] = [t for t in pd.read_parquet(p)["prompt"].tolist() if t and t.strip()]
    repo, _, rev = PINS["alpaca_cleaned"]
    ac = load_dataset(repo, revision=rev, split="train")
    raw["alpaca_cleaned"] = [
        r["instruction"].strip()
        for r in ac
        if not r["input"].strip() and 30 <= len(r["instruction"].strip()) <= 300
    ]
    hl = load_dataset(
        "mlabonne/harmless_alpaca", revision="02c6a92cfcf11bb0c387334f8146d149d65b587f"
    )
    raw["harmless_all"] = [r["text"] for s in hl for r in hl[s]]
    reserved_harmful = list(raw["harmful"])
    for src in SUP.SOURCES:
        path = ROOT / SUP.RAW / src["local"]
        SUP.verify(path, src["sha256"])
        reserved_harmful += [r["prompt"] for r in SUP.parse(src["name"], path)]
    raw["reserved_harmful"] = reserved_harmful
    math_rows, math_meta = D2.load_math_cached()
    raw["math"] = math_rows
    meta.update({"hendrycks_math": math_meta, "pinned_files_sha256": files, "pins": PINS})
    return raw, meta


def families(raw):
    mm = [_mc(r, "mmlu") for r in raw["mmlu_pro"]]
    syc = []
    for r in mm:
        wrong = [i for i in range(len(r["options"])) if i != r["answer_index"]]
        pick = wrong[int(_rank(SALT_S + ":wrong", r["hash"]), 16) % len(wrong)]
        syc.append(dict(r, kind="sycophancy", wrong_index=pick, id="syc:" + r["hash"][:16]))
    gsm = []
    for r in raw["gsm8k"]:
        h = text_hash(r["question"])
        gsm.append(
            {
                "kind": "gsm8k",
                "question": r["question"].strip(),
                "gold": _gsm_gold(r["answer"]),
                "hash": h,
                "id": f"gsm:{h[:16]}",
                "gtext": r["question"],
            }
        )
    wiki = []
    for t in raw["wikitext103"]:
        h = text_hash(t)
        wiki.append({"kind": "wikitext", "text": t, "hash": h, "id": f"wiki:{h[:16]}", "gtext": t})
    math = []
    for r in raw["math"]:
        h = text_hash(r["problem"])
        math.append(
            {
                "kind": "math",
                "problem": r["problem"],
                "gold": r["gold"],
                "level": r["level"],
                "type": r["type"],
                "hash": h,
                "id": f"math:{h[:16]}",
                "gtext": r["problem"],
            }
        )
    ife = []
    for t in raw["alpaca_cleaned"]:
        it = _prompt(t, "ife")
        it["constraints"] = G.make_constraints(it["hash"])
        ife.append(it)
    harmless = [_prompt(t, "harmless") for t in raw["harmless"]]
    return {
        "mc": {
            "pools": {
                "mmlupro_gold_lp": [
                    dict(r, id="mmlu:" + r["hash"][:16], kind="mmlu_pro") for r in mm
                ],
                "sycophancy_gap": syc,
                "mmlupro_gen": [dict(r, id="mmlug:" + r["hash"][:16]) for r in mm],
                "supergpqa_gen": [_mc(r, "sgpqa") for r in raw["supergpqa"]],
            }
        },
        "gsm": {
            "pools": {
                "gsm8k_gold_lp": gsm,
                "gsm8k_gen": [dict(r, id="gsmg:" + r["hash"][:16]) for r in gsm],
            }
        },
        "math": {"pools": {"math_gen": math}},
        "wiki": {
            "pools": {"wikitext_nll": wiki},
            "reserved": {"wikitext2": [a["text"] for a in raw["wikitext"]]},
        },
        "harmful": {
            "pools": {
                "refusal_margin_harmful": [_prompt(t, "harmful") for t in raw["llmlat_harmful"]],
                "guard_harmful": [_prompt(t, "orxtox") for t in raw["orbench_toxic"]],
            },
            "reserved": {"c1_harmful_pools": raw["reserved_harmful"]},
        },
        "harmless": {
            "pools": {
                "refusal_margin_harmless": harmless,
                "oldguard_harmless": [dict(r, id="oldg:" + r["hash"][:16]) for r in harmless],
                "guard_harmless": [_prompt(t, "orhard") for t in raw["orbench_hard"]],
                "ifeval_gen": ife,
            },
            "avoid_texts": {"harmless_alpaca": raw["harmless_all"]},
            "avoid": {"ifeval_gen": ["harmless_alpaca"]},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/wme/panels"))
    a = ap.parse_args()
    out_s, out_h = Path(a.out) / "wme-p3s", Path(a.out) / "wme-p3h"
    if not a.dry_run and (out_s.exists() or out_h.exists()):
        raise SystemExit("pilot-3 panels exist; panels are frozen once built")
    raw, meta = load_sources()
    fams = families(raw)
    exposures = X.load_exposures(ROOT, X.EXPOSURES)
    exposed = X.all_exposed(exposures)
    grouped = PP.group_all(fams)
    sizes = {**S_SIZES, **{p: {s: (n or 0) for s, n in v.items()} for p, v in H_SIZES.items()}}
    res_s = PP.select(fams, S_PANELS, sizes, SALT_S, exposed, SPLIT_ORDER, grouped=grouped)
    s_splits = res_s["splits"]
    s_groups = {r["group"] for ss in s_splits.values() for rows in ss.values() for r in rows}
    kw = {"used_groups": s_groups, "grouped": grouped}
    # guard dev = the pool rest: a first pass with dev = 0 counts the usable groups (no outcome is read)
    probe = PP.select(fams, H_PANELS, sizes, SALT_H, exposed, SPLIT_ORDER, **kw)
    for g in ("guard_harmless", "guard_harmful"):
        sizes[g]["dev"] = probe["stats"][g]["usable_groups"] - sizes[g]["holdout"]
    res_h = PP.select(fams, H_PANELS, sizes, SALT_H, exposed, SPLIT_ORDER, **kw)
    h_splits = res_h["splits"]
    both = {
        **{f"s:{p}": v for p, v in s_splits.items()},
        **{f"h:{p}": v for p, v in h_splits.items()},
    }
    aud = PP.audit(both, exposed)
    from datasets import load_dataset

    m500 = D2.problem_hashes(
        load_dataset(D2.MATH500["repo"], revision=D2.MATH500["revision"], split="test")["problem"]
    )
    math_written = {text_hash(r["problem"]) for rows in h_splits["math_gen"].values() for r in rows}
    summary = {
        "audit": aud,
        "math500_overlap": len(math_written & m500),
        "stats_s": res_s["stats"],
        "stats_h": res_h["stats"],
        "sizes": sizes,
        "n_exposed_hashes": len(exposed),
        "exposures": sorted(X.EXPOSURES),
    }
    print(json.dumps(summary, indent=1))
    if not aud["clean"] or summary["math500_overlap"]:
        raise SystemExit("audit failed")
    if a.dry_run:
        return
    common = {
        "label": "EXPLORATORY (WM-E pilot 3)",
        "plan": "docs/council/2026-11-17/RULING-wme-pilot3.md",
        "sources": meta,
        "exposures_excluded": sorted(X.EXPOSURES),
        "n_exposed_hashes": len(exposed),
        "audit": aud,
    }
    PP.write_panel(
        out_s, s_splits, SALT_S, {**common, "stats": summary["stats_s"], "sizes": S_SIZES}
    )
    PP.write_panel(
        out_h,
        h_splits,
        SALT_H,
        {
            **common,
            "stats": summary["stats_h"],
            "sizes": {p: sizes[p] for p in H_PANELS},
            "math500_overlap": 0,
        },
    )
    print("written", out_s, out_h)


if __name__ == "__main__":
    main()
