#!/usr/bin/env python3
"""Build the WM-E v0 lean panel item groups from datasets already in the HF cache (EXPLORATORY; CPU only).

Writes ``data/wme/panels/<panel_id>/{open/{construction,validation,dev}.jsonl, sealed_test/test.jsonl,
manifest.json}``. Assignment is by sha256(salt:normalised-text hash) after deduplication (``wm.panel.build_panel``);
test items are written only under ``sealed_test/`` and are unreadable through ``wm.plan.PlannerItemStore``.
Sources (pinned dataset revisions, offline): WikiText-2 raw (articles from all three splits), GSM8K main (train and
test), MMLU-Pro (test and validation), mlabonne/harmful_behaviors (train and test), mlabonne/harmless_alpaca (test).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import panel as P

SOURCES = {
    "wikitext": (
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        "b08601e04326c79dfdd32d625aee71d232d685c3",
    ),
    "gsm8k": ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "b189ec765aa7ed75c8acfea42df31fdae71f97be"),
    "harmful": ("mlabonne/harmful_behaviors", None, "01cead01398926d81f7c52bdb790ee8cf77ebba7"),
    "harmless": ("mlabonne/harmless_alpaca", None, "02c6a92cfcf11bb0c387334f8146d149d65b587f"),
}
SIZES = {"test": 48, "dev": 64, "validation": 64, "construction": 64}
SALT = "wme-v0-panel-20261116"


def wikitext_articles(ds_dict, min_chars=3000):
    arts = []
    for split in ("train", "validation", "test"):
        cur = []
        for line in ds_dict[split]["text"]:
            if line.startswith(" = ") and not line.startswith(" = = ") and cur:
                text = "".join(cur).strip()
                if len(text) >= min_chars:
                    arts.append({"article": text[:80], "text": text})
                cur = []
            cur.append(line)
        if cur:
            text = "".join(cur).strip()
            if len(text) >= min_chars:
                arts.append({"article": text[:80], "text": text})
    return arts


def load_raw():
    from datasets import load_dataset

    meta = {}

    def ld(key, **kw):
        repo, name, rev = SOURCES[key]
        d = (
            load_dataset(repo, name, revision=rev, **kw)
            if name
            else load_dataset(repo, revision=rev, **kw)
        )
        meta[key] = {
            "repo": repo,
            "config": name,
            "revision": rev,
            "splits": {k: len(v) for k, v in d.items()},
        }
        return d

    wiki = ld("wikitext")
    gsm = ld("gsm8k")
    mm = ld("mmlu_pro")
    hf = ld("harmful")
    hl = ld("harmless")
    raw = {
        "wikitext": wikitext_articles(wiki),
        "gsm8k": [dict(r) for s in ("train", "test") for r in gsm[s]],
        "mmlu_pro": [
            {
                "question_id": r["question_id"],
                "question": r["question"],
                "options": r["options"],
                "answer_index": r["answer_index"],
            }
            for s in ("test", "validation")
            for r in mm[s]
        ],
        "harmful": [r["text"] for s in ("train", "test") for r in hf[s]],
        "harmless": [r["text"] for r in hl["test"]],
    }
    return raw, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-id", default="wme-v0-p1")
    ap.add_argument("--out", default=str(ROOT / "data/wme/panels"))
    args = ap.parse_args()
    out = Path(args.out) / args.panel_id
    if out.exists():
        raise SystemExit(f"{out} exists; panels are frozen once built")
    raw, meta = load_raw()
    man = P.build_panel(raw, SIZES, SALT, out)
    man["sources"] = meta
    man["label"] = "EXPLORATORY (WM-E v0 pilot)"
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(
        json.dumps(
            {"panel": str(out), "content_hash": man["content_hash"], "dedup": man["dedup"]},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
