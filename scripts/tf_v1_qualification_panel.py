#!/usr/bin/env python3
"""Derive the PREREG_H1_TF section 6 ten-entry qualification composition (development only).

Rule (PREREG section 6, "metrology's ten-entry rule"; unspecified choices and ties sorted by ID):
  4 LoRA   two nearest the median legacy KL of the D3-evaluated lora_sft entries, two largest;
  3 chains three D3-evaluated chain entries from distinct lineages, each at step >= 3 (>= 4 nodes
           counting the lineage root), the first three by sorted ID;
  2 task   the first two D3-evaluated task_vector entries by sorted ID;
  1 abliteration  the D3-evaluated abliteration entry with the largest legacy KL.
Legacy KL is the fast-tier `kl_full_sequence_mean` and selects this development pilot only, never
the fresh panel. The predecessor-relative endpoint needs each chain entry's actual predecessor, so
those three parents are listed separately as scoring references (13 entries in total).

PREREG leaves the composition to this rule rather than to the ten D3 controls
(data/runs/d3_shards/controls.txt: 3 random-direction, 3 abliteration, 2 LoRA, 2 task-vector, no
chain), so the controls are not used for G1.
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
EVALS = ROOT / "data/evals/qwen3.5-0.8b"
OUT = ROOT / "data/runs/h1a/qualification_panel.json"


def lineage_root(parents, entry_id):
    while parents[entry_id]["parent_id"] and parents[entry_id]["family"] == "chain":
        entry_id = parents[entry_id]["parent_id"]
    return entry_id


def select(rows, edits, legacy_kl):
    """rows: manifest rows by id; edits: edit.json by id; legacy_kl: fast-tier KL of D3 entries."""
    fam = lambda f: [i for i in legacy_kl if rows[i]["family"] == f]
    lora = fam("lora_sft")
    median = statistics.median(legacy_kl[i] for i in lora)
    near = sorted(lora, key=lambda i: (abs(legacy_kl[i] - median), i))[:2]
    largest = sorted(lora, key=lambda i: (-legacy_kl[i], i))[:2]
    chains, seen = [], set()
    for i in sorted(fam("chain")):
        if edits[i]["params"]["step"] < 3:
            continue
        root = lineage_root(rows, i)
        if root not in seen:
            seen.add(root)
            chains.append(i)
        if len(chains) == 3:
            break
    task = sorted(fam("task_vector"))[:2]
    abliteration = sorted(fam("abliteration"), key=lambda i: (-legacy_kl[i], i))[:1]
    chosen = {
        "lora_nearest_median": near,
        "lora_largest": largest,
        "chain": chains,
        "task_vector": task,
        "abliteration_largest": abliteration,
    }
    ids = [i for group in chosen.values() for i in group]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError(f"composition is not ten distinct entries: {ids}")
    return chosen, median


def main():
    rows = {}
    for line in (ZOO / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "id" in row:
            rows[row["id"]] = row
    edits = {i: json.loads((ZOO / i / "edit.json").read_text()) for i in rows}
    legacy_kl = {}
    for i in rows:
        path = EVALS / i / "eval.json"
        if path.exists():
            legacy_kl[i] = json.loads(path.read_text())["kl_full_sequence_mean"]
    chosen, median = select(rows, edits, legacy_kl)
    ids = [i for group in chosen.values() for i in group]
    predecessors = [rows[i]["parent_id"] for i in chosen["chain"]]
    for i in ids + predecessors:
        if not ((ZOO / i / "delta.safetensors").exists() or any((ZOO / i / "lora").glob("*"))):
            raise FileNotFoundError(f"no delta for {i}")
    result = {
        "rule": __doc__.strip(),
        "composition": chosen,
        "entries": ids,
        "chain_predecessors": predecessors,
        "score_ids": ids + predecessors,
        "legacy_kl": {i: legacy_kl[i] for i in ids},
        "lora_median_legacy_kl": median,
        "chain_steps": {i: edits[i]["params"]["step"] for i in chosen["chain"]},
        "chain_lengths": {i: edits[i]["params"]["length"] for i in chosen["chain"]},
        "panel": "development",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("composition", "chain_predecessors")}, indent=2))


if __name__ == "__main__":
    main()
