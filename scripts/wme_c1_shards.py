#!/usr/bin/env python3
"""Designate the wme-c1 construction shards (field, construction/e2, e2 contrast sources) by the amendment-G10 rule
(``wsp.wm.c1.shards``). CPU only: reads the open construction split, the committed root-lineage report and the
pinned 9B tokenizer (token counts for the field cap). Writes ids and hashes only.

    wme_c1_shards.py --out docs/reviews/wme/wme-c1-shards-2026-11-19.json
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

from wsp.wm.c1 import shards as S

LINEAGE = ROOT / "docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json"
LINEAGE_SHA = "0df383ae561f81afc42cfd24b0fd337edfb44797be656ef1f10dd3810281b2cf"
PANEL = ROOT / "data/wme/panels/wme-c1"
CONSTRUCTION_SHA = "736c2582142567ad3217e7bdfabd6e8872eeb1820128a8faf36abf499e209c2b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if hashlib.sha256(LINEAGE.read_bytes()).hexdigest() != LINEAGE_SHA:
        raise SystemExit("lineage report hash mismatch")
    cp = PANEL / "open/construction.jsonl"
    if hashlib.sha256(cp.read_bytes()).hexdigest() != CONSTRUCTION_SHA:
        raise SystemExit("construction split hash mismatch")
    rows = [json.loads(x) for x in cp.read_text().splitlines()]
    reps, groups = S.root_reserved(LINEAGE)
    from transformers import AutoTokenizer

    from wsp.wm import hf
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2

    tok = AutoTokenizer.from_pretrained(
        str(hf.snapshot_dir("Qwen/Qwen3.5-9B", hf.TARGETS["Qwen/Qwen3.5-9B"])),
        local_files_only=True,
    )
    cfg = P.PanelConfig()
    n_tokens = lambda o, r: sum(len(x.ids) for x in P2.rows_for(o, r, tok, cfg))
    rec = S.designate(rows, reps, groups, n_tokens)
    S.check_disjoint(rec)
    rec["inputs"] = {
        "construction_sha256": CONSTRUCTION_SHA,
        "lineage_sha256": LINEAGE_SHA,
        "tokenizer": "Qwen/Qwen3.5-9B@" + hf.TARGETS["Qwen/Qwen3.5-9B"],
    }
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "digest": rec["digest"],
                "fresh_counts": rec["fresh_counts"],
                "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
