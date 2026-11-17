#!/usr/bin/env python3
"""E2 library contrast directions at theta0 and their similarity to bank roots and pilots (PREREG_WME v1 §6.5.2
disclosure; OUTCOME-BLIND: base forward passes on construction prompts only).

The four library contrasts (syc: suggested - neutral sycophancy prompts; gsm: GSM8K - WikiText; mathcomp: MATH -
harmless; trivia: neutral sycophancy prompts - harmless) at layers {20, 22, 24, 26, 28, 30}, each from 32 groups
per source taken from the construction remainder (the wme-c1 v1 rank order reproduced in memory), excluding every
group a bank root's shard used. The library itself is built at the freeze from the registered construction/e2
shard; this audit discloses its expected overlap (J3 |cos|) with the accepted bank roots and the pilots.

    python scripts/wme_e2_library_audit.py --sizes <selected.json> --roots <roots.json> --dirs <directions.pt> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wsp.wm import e2 as E2
from wsp.wm import exposure as X
from wsp.wm import panel as P
from wsp.wm import panel_c1 as C1
from wsp.wm import rootsampler as RS


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.loads(Path(a.sizes).read_text())
    roots = json.loads(Path(a.roots).read_text())
    used = {
        g
        for r in roots["roots"]
        for k in ("contrast_source_groups", "adapter_source_groups")
        for g in (r.get(k) or [])
    }
    aud = _mod("c1audit", "scripts/wme_c1_exposure_audit.py")
    BR = _mod("bankroots", "scripts/wme_bank_roots.py")
    pools, _ = aud.pools()
    grouped = C1.grouped_pool(pools)
    exposures = X.load_exposures(ROOT)
    owner, rem = {}, {}
    for obj, rows in grouped.items():
        kept, _ = X.exclude_groups(rows, exposures)
        by = {}
        for r in sorted(kept, key=lambda r: r["hash"]):
            if owner.setdefault(r["group"], obj) != obj:
                continue
            by.setdefault(r["group"], r)
        rows = sorted(by.values(), key=lambda r: C1._rank(r["group"]))
        front = 24 + d["n_dev"] + d["n_test"] + (d["n_bank"] if obj != "mathcomp_gold_lp" else 0)
        if obj == "sycophancy_real":
            front += d["n_gen_syc"]
        rem[obj] = [dict(r, objective=obj) for r in rows[front:] if r["group"] not in used]
    W1 = _mod("wme_pilot", "scripts/wme_pilot.py")
    from wsp.wm import hf

    model, tok, cfg, base_hash, _ = hf.load_target("Qwen/Qwen3.5-9B", "cuda")
    chat = lambda u: P.chat_prompt(tok, u)  # noqa: E731
    take = lambda o: rem[o][:32]  # noqa: E731
    sets = {
        "syc_suggested": [chat(r["suggested_prompt"]) for r in take("sycophancy_real")],
        "syc_neutral": [chat(r["neutral_prompt"]) for r in take("sycophancy_real")],
        "gsm": [chat(P.gsm8k_user(r)) for r in take("gsm8k_gold_lp")],
        "wiki": [r["text"][:1500] for r in take("wikitext_nll")],
        "mathcomp": [
            chat(r["problem"] + "\nGive only the final answer.") for r in take("mathcomp_gold_lp")
        ],
        "harmless": [chat(r["prompt"]) for r in take("refusal_margin_harmless")],
    }
    means = {s: W1.residual_means(model, tok, v, list(E2.LAYERS))[0] for s, v in sets.items()}
    lib = {}
    for name, (pa, pb) in E2.CONTRASTS.items():
        for L in E2.LAYERS:
            v = (means[pa][L] - means[pb][L]).float().cpu()
            lib[f"e2:{name}@{L}"] = (v / v.norm()).numpy()
    pilot = {}
    for tag, p in BR.PILOT_DIRS.items():
        for k, v in torch.load(ROOT / p, weights_only=False, map_location="cpu")["dirs"].items():
            pilot[f"{tag}:{k}"] = v.float().numpy()
    arte = torch.load(a.dirs, weights_only=False)["artefacts"]
    bank = {
        rid: {L: v.numpy() for L, v in x["directions"].items()}
        for rid, x in arte.items()
        if "directions" in x
    }
    out = {
        "label": "E2 library contrast directions vs bank roots and pilots (theta0; disclosure; no evaluation)",
        "n_library": len(lib),
        "groups_per_source": {s: len(v) for s, v in sets.items()},
        "library_vs_pilots": RS.direction_audit(lib, pilot),
        "bank_vs_library": RS.direction_audit(bank, lib),
        "library_digest": E2.library_digest(),
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    for k in ("library_vs_pilots", "bank_vs_library"):
        w = out[k]["worst"]
        top = max(w.items(), key=lambda kv: kv[1]["cos"])
        print(
            k,
            "pass",
            out[k]["pass"],
            "flagged",
            len(out[k]["flagged"]),
            "max",
            round(top[1]["cos"], 3),
            top,
        )


if __name__ == "__main__":
    sys.exit(main())
