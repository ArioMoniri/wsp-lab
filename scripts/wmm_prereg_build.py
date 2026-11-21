#!/usr/bin/env python3
"""PREREG_WMM build (CPU and tokenizer only; no generation, no outcome is read).

Writes:
* ``docs/prereg/wmm_exposure_registry.json``: every Gate 0 / 0b / 1 item, marked EXPOSED;
* ``data/wmm/prereg/panels/*``: the sealed panels (items) and their fixed generation orders (not committed);
* ``docs/prereg/wmm_manifest.json``: salts, dataset pins, per-panel counts, exclusions and sha256 of the item lists,
  panel files and orders; LoRA training pools and their disjointness checks; code hashes.

    wmm_prereg_build.py [--out data/wmm/prereg]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import chat as CH
from wsp.wm import gemma3 as G3
from wsp.wm import wmm_gate0 as W
from wsp.wm.wmm import gate1 as G1
from wsp.wm.wmm import prereg as P

EXPOSURES = {
    "gate0": ROOT / "data/wmm/gate0-2026-11-19/exposure.json",
    "gate0b": ROOT / "data/wmm/gate0b-2026-11-20/medqa_g0b_exposure.json",
    "gate1": ROOT / "data/wmm/gate1-2026-11-20/exposure.json",
}
DATA = {
    "medqa": ("openlifescienceai/medqa", "153e61cdd129eb79d3c27f82cdf3bc5e018c11b0"),
    "medmcqa": ("openlifescienceai/medmcqa", "91c6572c454088bf71b679ad90aa8dffcd0d5868"),
    "medsafety": (
        "israel-adewuyi/med-safety-bench-reproduced",
        "0c3d752deb205f2f672bee4644f22e70e11bd708",
    ),
    "medquad": ("lavita/MedQuAD", "84ea67f83cec9692ad254eaa02c9731b24ecfe4c"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"),
}
FILES = {
    ("medqa", "train"): "data/train-00000-of-00001.parquet",
    ("medqa", "dev"): "data/dev-00000-of-00001.parquet",
    ("medqa", "test"): "data/test-00000-of-00001.parquet",
    ("medmcqa", "validation"): "data/validation-00000-of-00001.parquet",
    ("medsafety", "train"): "data/train-00000-of-00001.parquet",
    ("medsafety", "test"): "data/test-00000-of-00001.parquet",
    ("medquad", "train"): "data/train-00000-of-00001-e36383d177026d53.parquet",
    ("arc", "train"): "ARC-Challenge/train-00000-of-00001.parquet",
    ("arc", "validation"): "ARC-Challenge/validation-00000-of-00001.parquet",
    ("arc", "test"): "ARC-Challenge/test-00000-of-00001.parquet",
}
SIZES = {
    "mm": 2000,
    "ma": 200,
    "benign": 1500,
    "pool_msyc": 128,
    "pool_mte": 1024,
    "pool_mkd": 1024,
    "pool_ma": 256,
}
CODE = [
    "src/wsp/wm/wmm/prereg.py",
    "src/wsp/wm/wmm/powersim.py",
    "src/wsp/wm/wmm/provenance.py",
    "src/wsp/wm/wmm/canary.py",
    "src/wsp/wm/wmm_gate0b.py",
    "src/wsp/wm/wmm/gate1.py",
    "src/wsp/wm/chat.py",
    "src/wsp/wm/gemma3.py",
    "scripts/wmm_prereg_build.py",
    "scripts/wmm_powersim.py",
    "scripts/wmm_gcal_calibration.py",
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parquet(src, split):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, rev = DATA[src]
    return pd.read_parquet(
        hf_hub_download(repo, FILES[(src, split)], repo_type="dataset", revision=rev)
    ).to_dict("records")


def mc_items(src, split):
    rows = parquet(src, split)
    if src == "medqa":
        out = [W.medqa_item(r) for r in rows]
    elif src == "medmcqa":
        out = [x for x in (W.medmcqa_item(r) for r in rows) if x is not None]
    else:
        out = [x for x in (G1.arc_item(r) for r in rows) if x is not None]
    for it in out:
        it["split"] = split
        it["stem"] = P.stem_hash(it["question"])
    return out


def prompt_items(src, split, key, idkey=None):
    out = []
    for i, r in enumerate(parquet(src, split)):
        text = r[key]
        if not text:
            continue
        iid = str(r[idkey]) if idkey else f"{split}-{i}"
        out.append(
            {
                "id": iid,
                "source": src,
                "split": split,
                "prompt": text,
                "hash": W.text_hash(text),
                "stem": P.stem_hash(text),
            }
        )
    return out


def designate(items, source, blocks, excl):
    """Consecutive blocks of the ``wmm-p1`` ranking after removing excluded (hash, (source, id), stem) and duplicate
    items. A block size of None takes everything left."""
    removed = {"exposed_or_sealed": 0, "duplicate": 0}
    seen_h, seen_s, ranked = set(), set(), []
    for it in sorted(items, key=lambda it: P.rank_key(source, it["split"], it["hash"])):
        if (
            it["hash"] in excl["hashes"]
            or (it["source"], it["id"]) in excl["ids"]
            or it["stem"] in excl["stems"]
        ):
            removed["exposed_or_sealed"] += 1
            continue
        if it["hash"] in seen_h or it["stem"] in seen_s:
            removed["duplicate"] += 1
            continue
        seen_h.add(it["hash"])
        seen_s.add(it["stem"])
        ranked.append(it)
    out, pos = {}, 0
    for name, n in blocks:
        take = ranked[pos:] if n is None else ranked[pos : pos + n]
        if n is not None and len(take) < n:
            raise SystemExit(f"{name}: only {len(take)} fresh items, {n} required")
        out[name] = take
        pos += len(take)
    return out, removed, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/wmm/prereg"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)

    # 1. exposure registry
    reg = P.exposure_registry(EXPOSURES)
    reg_path = ROOT / "docs/prereg/wmm_exposure_registry.json"
    reg_doc = {
        "status": "EXPOSED: every item read in study M Gates 0, 0b and 1; development use only, never in a sealed panel",
        "sources": {g: str(p.relative_to(ROOT)) for g, p in EXPOSURES.items()},
        **reg,
    }
    reg_path.write_text(json.dumps(reg_doc, indent=1))
    ex = {
        "hashes": {e["hash"] for e in reg["items"]},
        "ids": {(e["source"], e["id"]) for e in reg["items"]},
        "stems": set(),
    }

    # 2. items
    mq = mc_items("medqa", "dev") + mc_items("medqa", "test")
    mq_train = mc_items("medqa", "train")
    mm = mc_items("medmcqa", "validation")
    arc = mc_items("arc", "test") + mc_items("arc", "validation") + mc_items("arc", "train")
    harm = prompt_items("medsafety", "train", "harmful_medical_request") + prompt_items(
        "medsafety", "test", "harmful_medical_request"
    )
    ben = prompt_items("medquad", "train", "question", "question_id")
    for it in arc:
        it["source"] = "arc_challenge"
    by_hash = {it["hash"]: it for it in mq + mq_train + mm + arc + harm + ben}
    ex["stems"] = {by_hash[h]["stem"] for h in ex["hashes"] if h in by_hash}
    # E11: Gate 0 items carry their own question text (incl. the 32 MedMCQA-train ladder items, whose split the
    # pinned datasets above do not load); add their stems directly.
    g0 = json.loads((ROOT / "data/wmm/gate0-2026-11-19/items.json").read_text())
    g0_stems = {P.stem_hash(it["question"]) for v in g0.values() for it in v}
    g0_mm_train = [it for it in g0["ladder"] if it["source"] == "medmcqa"]
    ex["stems"] |= g0_stems

    # 3. sealed panels
    panels, stats = {}, {}
    for name, (items, src, blocks) in {
        "mq": (mq, "medqa", [("mq", None)]),
        "mm": (mm, "medmcqa", [("mm", SIZES["mm"]), ("ma", SIZES["ma"])]),
        "harm": (harm, "medsafety", [("harm", None)]),
        "benign": (ben, "medquad", [("benign", SIZES["benign"])]),
        "gen": (arc, "arc_challenge", [("gen", None)]),
    }.items():
        got, removed, n_in = designate(items, src, blocks, ex)
        panels.update(got)
        for k in got:
            stats[k] = {"source": src, "pool_in": n_in, "removed": removed, "n": len(got[k])}

    # 4. LoRA training pools: MedQA train, disjoint from exposed and from every sealed item (hash and stem)
    sealed = {
        "hashes": {it["hash"] for v in panels.values() for it in v} | ex["hashes"],
        "stems": {it["stem"] for v in panels.values() for it in v} | ex["stems"],
    }
    excl = {"hashes": sealed["hashes"], "ids": ex["ids"], "stems": sealed["stems"]}
    pools, removed, n_in = designate(
        mq_train,
        "medqa",
        [(k, SIZES[k]) for k in ("pool_msyc", "pool_mte", "pool_mkd", "pool_ma")],
        excl,
    )
    viol = {k: P.disjoint_violations(v, sealed) for k, v in pools.items()}
    assert not any(viol.values()), viol
    pool_ids = [it["id"] for v in pools.values() for it in v]
    assert len(pool_ids) == len(set(pool_ids))

    # 5. fixed generation orders (tokenizer only)
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(
        str(G3.snapshot(G3.TARGET)), local_files_only=True
    )
    variants = {
        "mq": ("neutral", "suggested"),
        "mm": ("neutral",),
        "ma": ("suggested",),
        "harm": ("raw",),
        "benign": ("raw",),
        "gen": ("neutral",),
    }
    prompt_fn = {
        "neutral": P.neutral_prompt,
        "suggested": P.suggested_prompt,
        "raw": lambda it: it["prompt"],
    }
    manifest_panels = {}
    for name, items in panels.items():
        f = out / "panels" / f"{name}.jsonl"
        f.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in items))
        ranks = {
            it["id"]: P.rank_key(stats[name]["source"], it["split"], it["hash"]) for it in items
        }
        orders = {}
        for v in variants[name]:
            lens = {
                it["id"]: len(
                    tok.encode(CH.chat_prompt(tok, prompt_fn[v](it)), add_special_tokens=False)
                )
                for it in items
            }
            order = P.batch_order(lens, ranks)
            of = out / "panels" / f"{name}.{v}.order.json"
            of.write_text(json.dumps(order))
            orders[v] = {
                "sha256": sha(of.read_bytes()),
                "n_batches": len(P.batches(order)),
                "max_prompt_tokens": max(lens.values()),
            }
        ids = sorted(it["id"] for it in items)
        manifest_panels[name] = {
            **stats[name],
            "splits": sorted({it["split"] for it in items}),
            "ids_sha256": sha(json.dumps(ids).encode()),
            "file": f"data/wmm/prereg/panels/{name}.jsonl",
            "file_sha256": sha(f.read_bytes()),
            "orders": orders,
        }
    for name, items in pools.items():
        f = out / "panels" / f"{name}.jsonl"
        f.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in items))
    git = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    manifest = {
        "study": "M′ (PREREG_WMM)",
        "status": "SEALED: built from item text only; no generation; no outcome read",
        "salt": P.SALT,
        "prior_salts_excluded": ["wmm-g0", "wmm-g0b", "wmm-g1"],
        "datasets": DATA,
        "target": {"repo": G3.TARGET, "revision": G3.TARGETS[G3.TARGET]},
        "harness": {
            "prompt": "wmm_gate0b.prompt",
            "extractor": "wmm_gate0b.extract",
            "budget": P.BUDGET,
            "decoding": "greedy",
        },
        "generation": {
            "batch": P.BATCH,
            "order": "ascending prompt tokens, ties by wmm-p1 rank; shared by base and every edit",
        },
        "exposure_registry": {
            "path": str(reg_path.relative_to(ROOT)),
            "sha256": sha(reg_path.read_bytes()),
            "n_entries": reg["n_entries"],
            "n_unique": reg["n_unique"],
        },
        "panels": manifest_panels,
        "training_pools": {
            k: {
                "n": len(v),
                "source": "medqa train",
                "file": f"data/wmm/prereg/panels/{k}.jsonl",
                "ids_sha256": sha(json.dumps(sorted(it["id"] for it in v)).encode()),
                "disjointness_violations": len(viol[k]),
            }
            for k, v in pools.items()
        },
        "gate0_medmcqa_train_stem_check": {
            "n_items": len(g0_mm_train),
            "stem_hits_in_panels_or_pools": sum(
                P.stem_hash(it["question"])
                in (
                    {x["stem"] for v in panels.values() for x in v}
                    | {x["stem"] for v in pools.values() for x in v}
                )
                for it in g0_mm_train
            ),
        },
        "training_pool_exclusions": {
            "pool_in": n_in,
            "removed": removed,
            "rule": "hash, (source, id) or 300-char stem hash shared with any exposed or sealed item",
        },
        "code_sha256": {c: sha((ROOT / c).read_bytes()) for c in CODE if (ROOT / c).exists()},
        "git_sha_at_build": git,
    }
    mp = ROOT / "docs/prereg/wmm_manifest.json"
    mp.write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {k: v["n"] for k, v in manifest_panels.items()} | {k: len(v) for k, v in pools.items()}
        )
    )


if __name__ == "__main__":
    main()
