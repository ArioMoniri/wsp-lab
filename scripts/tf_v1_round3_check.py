#!/usr/bin/env python3
"""Round-3 pilot gate (PREREG_H1_TF amendment A2.4): timing and identity against round 2.

Compares the new pilot (data/evals/qwen3.5-0.8b/tf_v1, source after the batching change) with the
archived round-2 primaries (tf_v1_round2_archive, run 20261105-144427-fb58ffe6):

  * warm median wall time <= 1.40 min over the ten edits;
  * MMLU-Pro, KL and WikiText records bitwise identical (every field of every record);
  * HellaSwag deviation <= 0.010 (char-normalised log-lik), phrase log-prob <= 0.39, margin <= 0.30;
  * the KL base-logit cache content root is unchanged (2ed18bda284b3a89...) and every record
    carries the batch it was scored with.

Exit status 0 iff every check passes. Writes tf_v1/round3_check.json. Launches nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
EVALS = ROOT / "data/evals/qwen3.5-0.8b"
NEW, OLD = EVALS / "tf_v1", EVALS / "tf_v1_round2_archive"
OLD_RUN = "20261105-144427-fb58ffe6"
KL_ROOT_PREFIX = "2ed18bda284b3a89"
GATE_MINUTES = 1.40
LIMITS = {"hellaswag": 0.010, "phrase_log_prob": 0.39, "margin": 0.30}
IDENTICAL = ("mmlu_pro", "kl_to_base", "wikitext_ppl")


def records(directory):
    lines = (Path(directory) / "records.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def by_benchmark(rows, name):
    return [r for r in rows if r["benchmark"] == name]


def check_entry(edit_id):
    new, old = records(NEW / edit_id), records(OLD / edit_id)
    out = {"identical": {}, "max_abs": {}}
    for name in IDENTICAL:
        a, b = by_benchmark(new, name), by_benchmark(old, name)
        out["identical"][name] = len(a) == len(b) > 0 and a == b
    a, b = by_benchmark(new, "hellaswag"), by_benchmark(old, "hellaswag")
    diff = [
        abs(x - y)
        for r, q in zip(a, b, strict=True)
        for x, y in zip(
            r["extra"]["raw_option_log_likelihoods"],
            q["extra"]["raw_option_log_likelihoods"],
            strict=True,
        )
    ]
    argmax = lambda v: max(range(len(v)), key=v.__getitem__)
    out["max_abs"]["hellaswag"] = max(diff)
    out["hellaswag_argmax_changes"] = sum(
        argmax(r["extra"]["raw_option_log_likelihoods"])
        != argmax(q["extra"]["raw_option_log_likelihoods"])
        for r, q in zip(a, b, strict=True)
    )
    out["hellaswag_batch"] = sorted({r["extra"].get("scoring_batch") for r in a})
    for name in ("refusal_rate", "over_refusal"):
        a, b = by_benchmark(new, name), by_benchmark(old, name)
        lp = [
            abs(r["extra"]["phrase_log_probs"][k] - q["extra"]["phrase_log_probs"][k])
            for r, q in zip(a, b, strict=True)
            for k in q["extra"]["phrase_log_probs"]
        ]
        mg = [
            abs(r["extra"]["phrase_margin"] - q["extra"]["phrase_margin"])
            for r, q in zip(a, b, strict=True)
        ]
        out["max_abs"][f"{name}.phrase_log_prob"] = max(lp)
        out["max_abs"][f"{name}.margin"] = max(mg)
        out[f"{name}_batch"] = sorted({r["extra"].get("scoring_batch") for r in a})
    return out


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    from tf_v1_pilot_report import report

    ids = ["base", *(ROOT / "data/runs/d3_shards/controls.txt").read_text().split()]
    result = report(gate=(GATE_MINUTES, GATE_MINUTES))
    median = result["warm_median_minutes"]
    gate_ok = median <= GATE_MINUTES
    entries = {i: check_entry(i) for i in ids}
    old_base = json.loads((OLD / "base/eval.json").read_text())
    new_base = json.loads((NEW / "base/eval.json").read_text())
    worst = {}
    for e in entries.values():
        for k, v in e["max_abs"].items():
            worst[k] = max(worst.get(k, 0.0), v)
    limit = lambda k: LIMITS["hellaswag"] if k == "hellaswag" else LIMITS[k.split(".")[1]]
    identical = all(all(e["identical"].values()) for e in entries.values())
    bounded = all(worst[k] <= limit(k) for k in worst)
    root_ok = (
        new_base["tf_cache"]["base_logits_sha256"].startswith(KL_ROOT_PREFIX)
        and new_base["tf_cache"]["base_logits_sha256"] == old_base["tf_cache"]["base_logits_sha256"]
    )
    batches = all(
        e["hellaswag_batch"] == [4]
        and e["refusal_rate_batch"] == [13]
        and e["over_refusal_batch"] == [13]
        for e in entries.values()
    )
    new_run = new_base["run_id"]
    summary = {
        "new_run_id": new_run,
        "old_run_id": old_base["run_id"],
        "old_run_expected": OLD_RUN,
        "new_source_hash": new_base["source_hash"],
        "old_source_hash": old_base["source_hash"],
        "warm_median_minutes": median,
        "warm_range_minutes": result["warm_min_max_minutes"],
        "warm_stage_median_seconds": result["warm_stage_median_seconds"],
        "gate_minutes": GATE_MINUTES,
        "gate_pass": bool(gate_ok),
        "mmlu_kl_wikitext_records_bitwise_identical": bool(identical),
        "secondary_deviations_within_probe_limits": bool(bounded),
        "worst_max_abs_deviation": worst,
        "hellaswag_argmax_changes_total": sum(
            e["hellaswag_argmax_changes"] for e in entries.values()
        ),
        "kl_cache_root_unchanged": bool(root_ok),
        "kl_cache_root": new_base["tf_cache"]["base_logits_sha256"],
        "scoring_batch_recorded": bool(batches),
        "tree_clean": not new_base["dirty"],
        "entries": entries,
    }
    ok = (
        gate_ok
        and identical
        and bounded
        and root_ok
        and batches
        and old_base["run_id"] == OLD_RUN
        and new_base["source_hash"] != old_base["source_hash"]
    )
    summary["all_checks_pass"] = bool(ok)
    (NEW / "round3_check.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
