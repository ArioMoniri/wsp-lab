#!/usr/bin/env python3
"""Study M Gate 0b driver (RULING-wmm-gate0 M1): one attempt of the declared harness on 400 fresh MedQA dev items;
only if MedQA fails, one attempt of the declared MedMCQA fallback on 400 fresh MedMCQA dev items. Rules are in
``wsp.wm.wmm_gate0b``. Local GPU only, <= 16 GB (the Gate 0 loader), stops before 0.95 GPU-h.

    wmm_gate0b.py --out data/wmm/gate0b-YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import gemma3 as G3
from wsp.wm import wmm_gate0 as W
from wsp.wm import wmm_gate0b as B

_spec = importlib.util.spec_from_file_location("wmm_gate0_driver", ROOT / "scripts/wmm_gate0.py")
G0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G0)
G0.GPU_H_CAP = 0.95

EXPOSURE_G0 = ROOT / "data/wmm/gate0-2026-11-19/exposure.json"
DATASETS = {
    "medqa": ("openlifescienceai/medqa", "153e61cdd129eb79d3c27f82cdf3bc5e018c11b0", "dev"),
    "medmcqa": (
        "openlifescienceai/medmcqa",
        "91c6572c454088bf71b679ad90aa8dffcd0d5868",
        "validation",
    ),
}


def load_split(source):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, rev, split = DATASETS[source]
    os.environ["HF_HUB_OFFLINE"] = "0"  # the MedMCQA validation file may not be cached yet
    try:
        df = pd.read_parquet(
            hf_hub_download(
                repo, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=rev
            )
        )
    finally:
        os.environ["HF_HUB_OFFLINE"] = "1"
    rows = df.to_dict("records")
    if source == "medqa":
        return split, [W.medqa_item(r) for r in rows]
    return split, [x for x in (W.medmcqa_item(r) for r in rows) if x is not None]


def attempt(ctx, source, exposed):
    name = f"{source}_g0b"
    got = ctx.read(name)
    if got:
        return got
    split, pool = load_split(source)
    items = B.designate_fresh(pool, source, split, B.N_ITEMS, exposed)
    ctx.write(f"{name}_items", items)
    ctx.write(
        f"{name}_exposure",
        {
            "salt": B.SALT,
            "status": "EXPOSED (study M Gate 0b); never sealed",
            "dataset": DATASETS[source][:2],
            "items": [
                {
                    "source": source,
                    "split": split,
                    "id": it["id"],
                    "hash": it["hash"],
                    "role": "gate0b_card",
                }
                for it in items
            ],
            "disjoint_from_gate0_exposure": True,
            "pool_size": len(pool),
        },
    )
    if not hasattr(ctx, "model"):
        G0.load(ctx)
    ctx.check_cap(name)
    gens = G0.gen_cached(ctx, name, [B.prompt(it) for it in items], B.BUDGET)
    sc = [B.score(it, g["text"], g["hit_max"]) for it, g in zip(items, gens, strict=True)]
    rep = B.verdict(
        sum(s["correct"] for s in sc), len(sc), sum(s["truncated"] for s in sc), B.CARDS[source]
    )
    rules = {}
    for s in sc:
        rules[str(s["rule"])] = rules.get(str(s["rule"]), 0) + 1
    lens = sorted(len(g["ids"]) for g in gens)
    rep.update(
        {
            "source": source,
            "split": split,
            "budget": B.BUDGET,
            "extractor_rules": rules,
            "unparsed_untruncated": sum(1 for s in sc if s["letter"] is None and not s["truncated"])
            / len(sc),
            "length_p50": lens[len(lens) // 2],
            "length_p95": lens[int(0.95 * (len(lens) - 1))],
            "thinking_block_rate": sum(g["text"].lstrip().startswith("<unused94>") for g in gens)
            / len(gens),
        }
    )
    ctx.write(name, rep)
    ctx.log("attempt", **{k: rep[k] for k in ("source", "accuracy", "truncation_rate", "pass")})
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = G0.Ctx(out, False)
    git = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    ctx.write(
        "manifest",
        {
            "study": "M",
            "gate": "0b",
            "ruling": "docs/council/2026-11-20/RULING-wmm-gate0.md M1",
            "target": G3.TARGET,
            "revision": G3.TARGETS[G3.TARGET],
            "git_sha": git,
            "instruction": B.INSTRUCTION,
            "budget": B.BUDGET,
            "cards": B.CARDS,
            "vram_cap_gb": G0.VRAM_CAP_GB,
            "gen_batch": G0.GEN_BATCH,
        },
    )
    exposed = B.read_exposed(EXPOSURE_G0)
    if exposed["n"] != 784:
        raise SystemExit(f"Gate 0 exposure list has {exposed['n']} items, expected 784")
    res = {"medqa": attempt(ctx, "medqa", exposed)}
    if not res["medqa"]["pass"]:
        ex2 = B.read_exposed(out / "medqa_g0b_exposure.json")
        exposed = {
            "hashes": exposed["hashes"] | ex2["hashes"],
            "ids": exposed["ids"] | ex2["ids"],
            "n": exposed["n"] + ex2["n"],
        }
        res["medmcqa"] = attempt(ctx, "medmcqa", exposed)
    final = {
        "medqa_pass": res["medqa"]["pass"],
        "fallback_run": "medmcqa" in res,
        "medmcqa_pass": res.get("medmcqa", {}).get("pass"),
    }
    final["gate0b_pass"] = bool(final["medqa_pass"] or final["medmcqa_pass"])
    final["primary"] = (
        "medqa" if final["medqa_pass"] else ("medmcqa" if final["medmcqa_pass"] else None)
    )
    ctx.write("verdict", final)
    ctx.charge()
    ctx.log("done", gpu_hours=ctx.gpu_seconds() / 3600, **final)


if __name__ == "__main__":
    main()
