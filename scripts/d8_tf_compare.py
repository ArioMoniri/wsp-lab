#!/usr/bin/env python3
"""D8 step 5 verification: compare the regenerated TF-v1 records with the numbers retained in docs/RESULTS.md.

Fingerprints (all read from the committed document, never from data/):
  * KL cache content root prefix 2ed18bda284b3a89 (round 2 and round 3, registered);
  * pilot table (round 1): TF accuracy over 1,000 MMLU-Pro items and mean gold log-prob, 6 decimals, per entry;
  * round-3 G2 table: MMLU-Pro and HellaSwag accuracies, gold-lp |delta| median / p99 / max, HellaSwag |score delta|;
  * G1: every regenerated repeat bitwise identical to its primary (records equality), 14 originally, here 12 (the
    chain predecessors b7edbb9f8b339f52 and 86ea1fe0ec777cf8 have no delta bytes);
  * round-3 mean per-prompt KL medians by prefix length (round 2 text: 0.00144 / 0.00153 / 0.00160).
Source gate: source_hash a224779bf873, dirty false, device H0, in every eval.json.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / "data/evals/qwen3.5-0.8b/tf_v1"
RESULTS = (ROOT / "docs/RESULTS.md").read_text()
CONTROLS = (ROOT / "data/runs/d3_shards/controls.txt").read_text().split()


def table_after(heading_regex, header_prefix):
    m = re.search(heading_regex, RESULTS)
    block = RESULTS[m.end() :]
    rows, started = [], False
    for line in block.splitlines():
        if line.startswith(header_prefix):
            started = True
            continue
        if started:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def main():
    out = {"gates": {}, "pilot_round1": {}, "kl_prefix_medians": {}, "g1": {}, "g2": {}}
    entries = ["base", *CONTROLS]
    ev = {i: json.loads((TF / i / "eval.json").read_text()) for i in entries}
    out["gates"] = {
        "source_hashes": sorted({e["source_hash"][:12] for e in ev.values()}),
        "dirty": sorted({bool(e["dirty"]) for e in ev.values()}),
        "devices": sorted({e["eval_device_id"] for e in ev.values()}),
        "status": sorted({e["eval_status"] for e in ev.values()}),
        "run_ids": sorted({e["run_id"] for e in ev.values()}),
        "kl_cache_roots": sorted({e["tf_cache"]["base_logits_sha256"][:16] for e in ev.values()}),
        "kl_root_matches_2ed18bda284b3a89": {
            e["tf_cache"]["base_logits_sha256"][:16] for e in ev.values()
        }
        == {"2ed18bda284b3a89"},
        "matched_base_missing": sorted({bool(e["matched_base_missing"]) for e in ev.values()}),
    }
    # round-1 pilot table (run 20261105-131348-277686c3, scorer superseded by the round-2 letter-stop fix, 02c2847): reported for
    # completeness, NOT a fingerprint of the frozen source. Entry | Wall | Load | Peak | TF acc | Fast acc | Gold log-prob | Records
    rows = table_after(r"## TF-v1 timing pilot\n", "| Entry")
    ok = True
    for r in rows:
        i = r[0]
        acc, gold = float(r[4]), float(r[6])
        tf = ev[i]["tf_v1"]["mmlu_pro"]
        got_acc, got_gold = tf["accuracy"], tf["mean_gold_log_prob"]
        match = round(got_acc, 3) == acc and round(got_gold, 6) == gold
        ok &= match
        out["pilot_round1"][i] = {
            "printed": [acc, gold],
            "regenerated": [round(got_acc, 3), round(got_gold, 6)],
            "match": match,
        }
    out["pilot_round1_all_match_superseded_scorer_not_a_gate"] = ok
    for L in (16, 32, 64):
        vals = []
        for i in CONTROLS:
            recs = [json.loads(x) for x in (TF / i / "records.jsonl").read_text().splitlines()]
            kl = [r for r in recs if r["benchmark"] == "kl_to_base"]
            vals.append(float(np.mean([np.mean(r["extra"]["position_kl"][:L]) for r in kl])))
        out["kl_prefix_medians"][str(L)] = float(np.median(vals))
    printed = {"16": 0.00144, "32": 0.00153, "64": 0.00160}
    out["kl_prefix_medians_match_printed_digits"] = all(
        round(v, 5) == printed[k] for k, v in out["kl_prefix_medians"].items()
    )
    g1 = TF / "g1_report.json"
    if g1.exists():
        rep = json.loads(g1.read_text())
        ents = rep["entries"]
        out["g1"] = {
            "n_repeats": len(ents),
            "all_records_identical": all(v["records_identical"] for v in ents.values()),
            "max_abs_over_all": max(
                (m for v in ents.values() for m in v["max_abs"].values()), default=None
            ),
        }
    g2 = TF / "bridge_g2/g2_report.json"
    if g2.exists():
        out["g2_report_keys"] = list(json.loads(g2.read_text()).keys())[:20]
    path = ROOT / "data/runs/d8/tf_compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
