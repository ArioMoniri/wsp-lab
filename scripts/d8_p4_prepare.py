#!/usr/bin/env python3
"""D8 step 6a: build the recovered-set inputs for the protocol-v2 P4 suite (no model training here).

1. Evals: expose every evaluated zoo entry to the directory reader as ``<evals>/<id> -> zoo/<id>`` (the last step of
   scripts/backfill_eval_metadata.py, lost with data/evals; symlinks only, nothing is copied or edited).
2. Features view ``<out>/features``: symlinks to the re-extracted feature directories of every entry whose features
   AND the features of every ancestor the dataset needs (chain predecessors back to the root, task-vector sources) exist.
   ``FeatureDataset.chain`` raises on a missing ancestor, so a row with an incomplete history cannot be trained on; this
   is the recovered-set rule and it is the only filter applied. ``manifest.json`` is copied unchanged.
3. Reports the rows the dataset would train on, with the match report (family, device, reason), for the two
   endpoints mmlu_pro and hellaswag.

    d8_p4_prepare.py --out /home/ubuntu/wsp-data/runs/d8/p4_input [--rebind-e871923e]

--rebind-e871923e adds view-only copies of the regenerated e871923e base record under the three lost base run ids that
the 89 chain-predecessor records at that source hash name in ``base_run_id`` (see the report); the canonical record is
never modified. Without the flag the rows that name a lost base run are unmatched.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/ubuntu/wsp-data")
ZOO = DATA / "zoo/qwen3.5-0.8b"
FEATURES = DATA / "features/qwen3.5-0.8b"
EVALS = DATA / "evals/qwen3.5-0.8b"


def ancestors_needed(entry, relations):
    """Entries whose features the dataset reads for ``entry``: itself, its chain predecessors, task-vector sources."""
    need, cur = [], entry
    while cur is not None:
        need.append(cur)
        row = relations.get(cur)
        if row is None:
            return None
        cur = row.get("parent_id") if row["relation"] in ("successor", "source") else None
        if row["relation"] == "source":
            need.append(row["parent_id"])
            cur = None
    return need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rebind-e871923e", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from wsp.predictor.data import FeatureDataset, read_relations

    made = 0
    for f in sorted(ZOO.glob("*/eval.json")):
        link = EVALS / f.parent.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(f.parent.resolve(), target_is_directory=True)
            made += 1
    print(
        f"eval symlinks created: {made}; total evaluated entries {len(list(ZOO.glob('*/eval.json')))}"
    )

    manifest_bytes = (ZOO / "manifest.jsonl").read_bytes()
    relations = read_relations(ZOO, manifest_bytes)
    have = {p.parent.name for p in FEATURES.glob("*/features.json")}
    view = a.out / "features"
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True)
    kept, dropped = [], Counter()
    for i in sorted(have):
        need = ancestors_needed(i, relations)
        if need is None or any(n not in have for n in need):
            dropped[relations[i]["family"] if i in relations else "unknown"] += 1
            continue
        (view / i).symlink_to(FEATURES / i, target_is_directory=True)
        kept.append(i)
    shutil.copy2(FEATURES / "manifest.json", view / "manifest.json")
    print(
        f"features extracted: {len(have)}; kept with complete ancestry: {len(kept)}; dropped by family: {dict(dropped)}"
    )

    evals = EVALS
    rebound = {}
    if a.rebind_e871923e:
        # View-only evals directory: canonical entries are symlinked; base/devices/ is rebuilt with extra directories named
        # after the three lost base run ids that the e871923e records name in ``base_run_id``. Each holds a copy of the
        # regenerated e871923e base record with ``run_id`` set to the lost id and ``rebound_from_run_id`` recording the
        # regenerated run. The canonical records are never modified.
        lost = sorted(
            {
                p["base_run_id"]
                for p in (json.loads(f.read_text()) for f in ZOO.glob("*/eval.json"))
                if p.get("source_hash", "").startswith("e871923e") and p.get("base_run_id")
            }
        )
        regen = next(
            d
            for d in sorted((EVALS / "base/devices/H0").glob("*"))
            if json.loads((d / "eval.json").read_text())["source_hash"].startswith("e871923e")
        )
        evals = a.out / "evals"
        if evals.exists():
            shutil.rmtree(evals)
        evals.mkdir(parents=True)
        for item in EVALS.iterdir():
            if item.name != "base":
                (evals / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
        for dev in (EVALS / "base/devices").iterdir():
            (evals / "base/devices" / dev.name).mkdir(parents=True)
            for run in dev.iterdir():
                (evals / "base/devices" / dev.name / run.name).symlink_to(
                    run.resolve(), target_is_directory=True
                )
        for lost_id in lost:
            d = evals / "base/devices/H0" / lost_id
            d.mkdir(parents=True)
            e = json.loads((regen / "eval.json").read_text())
            e.update(
                {
                    "run_id": lost_id,
                    "rebound_from_run_id": regen.name,
                    "rebound_note": "view-only copy of the regenerated e871923e base (D8 step 6a); the original record is lost",
                }
            )
            (d / "eval.json").write_text(json.dumps(e, indent=2) + "\n")
            (d / "records.jsonl").symlink_to((regen / "records.jsonl").resolve())
            rebound[lost_id] = regen.name
        print("rebound lost base run ids:", rebound)

    ds = FeatureDataset(view, evals, targets=["mmlu_pro", "hellaswag"], zoo=ZOO)
    rep = Counter()
    for (fam, dev, reason), n in ds.match_report.items():
        rep[f"{fam}|{dev}|{reason}"] = n
    out = {
        "rows": len(ds.rows),
        "rows_by_family": dict(Counter(r["family"] for r in ds.rows)),
        "matched_reference_targets": ds.matched_reference_targets,
        "missing_matched_reference": ds.missing_matched_reference,
        "targets_with_inferred_runtime": ds.targets_with_inferred_runtime,
        "match_report": dict(sorted(rep.items())),
        "rebound_base_run_ids": rebound,
        "features_kept": len(kept),
        "features_dropped_by_family": dict(dropped),
    }
    (a.out / "dataset_report.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
