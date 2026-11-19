"""Designation of the wme-c1 construction shards (amendment G10; outcome-blind, before any data).

The built ``open/construction.jsonl`` is one pool per objective; the builder only checked that enough *fresh*
construction exists for the field and e2 shards. This module designates them deterministically:

* **Fresh construction** = construction items that are not bank-root shard representatives and whose union-policy
  group holds no root member (``docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json``, committed, sha256
  ``0df383ae…``).
* **Order** = sha256(``wme-c1-shards-v1:<group>``) ascending (salted rank, as every wme-c1 split).
* ``field`` (the 24-item base-field shard, §3.4): per trajectory objective, the first 24 fresh items in rank order
  whose teacher-forced rows total <= 1,024 tokens (``panel2.rows_for``; the pilot field rule).
* ``e2`` (construction/e2, §6.5.2): the first 64 fresh sycophancy groups in rank order outside the field shard;
  library directions (``syc``, ``trivia`` positives), library LoRAs and the LoRA-SFT arm use it.
* ``e2_sources`` (the other library contrast sources, which construction/e2 does not contain): the first 64 fresh
  items of GSM8K, WikiText, MATH L4-5 and harmless prompts outside the field shard.

The designation file records item hashes and ids only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SALT = "wme-c1-shards-v1"
FIELD_N = 24
FIELD_MAX_TOKENS = 1024
E2_N = 64
SOURCE_N = 64
FIELD_OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)
SOURCES = {
    "gsm": "gsm8k_gold_lp",
    "wiki": "wikitext_nll",
    "mathcomp": "mathcomp_gold_lp",
    "harmless": "refusal_margin_harmless",
}


def rank(group: str) -> str:
    return hashlib.sha256(f"{SALT}:{group}".encode()).hexdigest()


def root_reserved(lineage_path) -> tuple[set, set]:
    d = json.loads(Path(lineage_path).read_text())
    reps = {m["representative"] for m in d["mapping"]}
    groups = {m["final_group"] for m in d["mapping"]}
    return reps, groups


def fresh_pool(construction_rows, reps: set, root_groups: set) -> dict:
    out: dict = {}
    for r in construction_rows:
        if r["hash"] in reps or r["group"] in root_groups:
            continue
        out.setdefault(r["objective"], []).append(r)
    for o in out:
        out[o].sort(key=lambda r: (rank(r["group"]), r["hash"]))
    return out


def designate(construction_rows, reps, root_groups, n_tokens) -> dict:
    """``n_tokens(objective, item) -> int`` (teacher-forced rows of the item). Returns the designation record."""
    fresh = fresh_pool(construction_rows, reps, root_groups)
    used = set()
    field = {}
    for o in FIELD_OBJECTIVES:
        sel = []
        for r in fresh.get(o, []):
            if n_tokens(o, r) <= FIELD_MAX_TOKENS:
                sel.append(r)
            if len(sel) == FIELD_N:
                break
        if len(sel) < FIELD_N:
            raise ValueError(
                f"{o}: {len(sel)} fresh field items within {FIELD_MAX_TOKENS} tokens < {FIELD_N}"
            )
        field[o] = sel
        used |= {r["hash"] for r in sel}
    e2 = [r for r in fresh.get("sycophancy_real", []) if r["hash"] not in used][:E2_N]
    if len(e2) < E2_N:
        raise ValueError(f"construction/e2: {len(e2)} fresh sycophancy groups < {E2_N}")
    used |= {r["hash"] for r in e2}
    sources = {}
    for name, o in SOURCES.items():
        sel = [r for r in fresh.get(o, []) if r["hash"] not in used][:SOURCE_N]
        if len(sel) < SOURCE_N:
            raise ValueError(f"e2 source {name}: {len(sel)} fresh items < {SOURCE_N}")
        sources[name] = sel
    slim = lambda rs: [{"id": r["id"], "hash": r["hash"], "group": r["group"]} for r in rs]
    rec = {
        "salt": SALT,
        "rule": __doc__.split("\n\n", 1)[1].strip(),
        "field": {o: slim(v) for o, v in field.items()},
        "e2": slim(e2),
        "e2_sources": {k: slim(v) for k, v in sources.items()},
        "fresh_counts": {o: len(v) for o, v in fresh.items()},
    }
    rec["digest"] = hashlib.sha256(
        json.dumps({k: rec[k] for k in ("field", "e2", "e2_sources")}, sort_keys=True).encode()
    ).hexdigest()
    return rec


def check_disjoint(rec: dict):
    seen = {}
    for tag, rows in (
        [("e2", rec["e2"])]
        + [(f"field:{o}", v) for o, v in rec["field"].items()]
        + [(f"src:{k}", v) for k, v in rec["e2_sources"].items()]
    ):
        for r in rows:
            if r["hash"] in seen:
                raise ValueError(f"{r['hash'][:12]} in {seen[r['hash']]} and {tag}")
            seen[r["hash"]] = tag
    return True


def items_for(rec_rows, construction_by_hash) -> list:
    return [construction_by_hash[r["hash"]] for r in rec_rows]
