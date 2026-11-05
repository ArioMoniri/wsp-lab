#!/usr/bin/env python3
"""G2 same-item legacy bridge: TF-v1 scoring of the fast tier's own 100 MMLU-Pro + 100 HellaSwag items.

PREREG_H1_TF section 6: the TF panels deliberately exclude the 300 legacy fast-tier items, so the
pilot could not compare old and new scoring item by item. This script scores the first 100 ids of
configs/eval_subsets/{mmlu_pro,hellaswag}.txt (exactly the fast tier's `n: 100`) with the TF-v1
scorers (`wsp.eval.tf_v1.letter_scores`, `choice_record`; HellaSwag through the final batched
`wsp.eval.tf_v1.hellaswag_scores`, amendment A2.4, four endings in one padded forward, so HellaSwag
is no longer bitwise equal to the fast tier's batch-1 scores) on the 11 pilot records (base + the ten D3 controls), then
compares each item with the fast-tier `records.jsonl` of the same entry.

It lives under scripts/ so that it does not change the TF-v1 `source_hash` of `src/wsp`. Outputs: data/evals/<zoo>/tf_v1/bridge_g2/<entry>/records.jsonl and
bridge_g2/g2_report.json.

Usage: tf_v1_g2_bridge.py [--compare-only]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
EVALS = ROOT / "data/evals/qwen3.5-0.8b"
OUT = EVALS / "tf_v1/bridge_g2"
N_ITEMS = 100
# PREREG_H1_TF section 6 bridge criteria (MMLU-Pro normalized gold-letter log-prob).
MEDIAN_LIMIT, P99_LIMIT, ARGMAX_LIMIT = 0.02, 0.2, 0.99


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def normalized_gold(scores, gold):
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - values.max()
    return float(shifted[gold] - np.log(np.exp(shifted).sum()))


def compare(fast, tf, benchmark):
    """fast: fast-tier records (extra.scores, correct); tf: TF-v1 records (extra.option_log_probs).

    Items are matched by id, in order; any mismatch is an error rather than a silent drop.
    """
    if [r["item_id"] for r in fast] != [r["item_id"] for r in tf]:
        raise ValueError(f"{benchmark}: item ids differ between fast tier and TF-v1")
    raw_fast = [r["extra"]["scores"] for r in fast]
    gold = [r["extra"]["gold_index"] for r in tf]
    pred_fast = np.array([argmax(s) for s in raw_fast])
    pred_tf = np.array([argmax(r["extra"]["raw_option_log_likelihoods"]) for r in tf])
    correct_fast = np.array([r["correct"] for r in fast], dtype=float)
    correct_tf = np.array([r["correct"] for r in tf], dtype=float)
    if not np.array_equal(correct_fast, (pred_fast == np.array(gold)).astype(float)):
        raise ValueError(f"{benchmark}: fast-tier correctness is not the argmax of its scores")
    result = {
        "n": len(fast),
        "accuracy_fast": float(correct_fast.mean()),
        "accuracy_tf_v1": float(correct_tf.mean()),
        "accuracy_abs_diff": float(abs(correct_fast.mean() - correct_tf.mean())),
        "correctness_agreement": float((correct_fast == correct_tf).mean()),
        "argmax_agreement": float((pred_fast == pred_tf).mean()),
        "n_correctness_flips": int((correct_fast != correct_tf).sum()),
    }
    if benchmark == "mmlu_pro":
        diff = np.abs(
            [
                normalized_gold(s, g) - r["extra"]["gold_log_prob"]
                for s, g, r in zip(raw_fast, gold, tf, strict=True)
            ]
        )
    else:
        # HellaSwag: character-normalized log-likelihood of the gold ending.
        diff = np.abs(
            [
                s[g] - r["extra"]["raw_option_log_likelihoods"][g]
                for s, g, r in zip(raw_fast, gold, tf, strict=True)
            ]
        )
    result |= {
        "gold_score_abs_diff_median": float(np.median(diff)),
        "gold_score_abs_diff_p99": float(np.quantile(diff, 0.99)),
        "gold_score_abs_diff_max": float(diff.max()),
        "bitwise_equal_scores": bool(
            all(
                a == b
                for r_fast, r in zip(raw_fast, tf, strict=True)
                for a, b in zip(r_fast, r["extra"]["raw_option_log_likelihoods"], strict=True)
            )
        ),
    }
    return result


def pooled(per_entry):
    out = {}
    for name in ("mmlu_pro", "hellaswag"):
        rows = [e[name] for e in per_entry.values()]
        out[name] = {
            "entries": len(rows),
            "correctness_agreement_mean": float(
                np.mean([r["correctness_agreement"] for r in rows])
            ),
            "correctness_agreement_min": float(min(r["correctness_agreement"] for r in rows)),
            "argmax_agreement_mean": float(np.mean([r["argmax_agreement"] for r in rows])),
            "argmax_agreement_min": float(min(r["argmax_agreement"] for r in rows)),
            "accuracy_abs_diff_mean": float(np.mean([r["accuracy_abs_diff"] for r in rows])),
            "accuracy_abs_diff_max": float(max(r["accuracy_abs_diff"] for r in rows)),
            "gold_score_abs_diff_median_max": float(
                max(r["gold_score_abs_diff_median"] for r in rows)
            ),
            "gold_score_abs_diff_p99_max": float(max(r["gold_score_abs_diff_p99"] for r in rows)),
            "gold_score_abs_diff_max": float(max(r["gold_score_abs_diff_max"] for r in rows)),
            "all_scores_bitwise_equal": all(r["bitwise_equal_scores"] for r in rows),
        }
    m = out["mmlu_pro"]
    out["prereg_mmlu_criteria"] = {
        "median_abs_diff_limit": MEDIAN_LIMIT,
        "p99_abs_diff_limit": P99_LIMIT,
        "argmax_agreement_limit": ARGMAX_LIMIT,
        "pass": bool(
            m["gold_score_abs_diff_median_max"] <= MEDIAN_LIMIT
            and m["gold_score_abs_diff_p99_max"] <= P99_LIMIT
            and m["argmax_agreement_min"] >= ARGMAX_LIMIT
        ),
        "note": "worst entry for each statistic; MMLU only, as registered",
    }
    return out


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def score_entry(runner, edit_id):
    from wsp.eval.subsets import load_items
    from wsp.eval.tf_v1 import choice_record, hellaswag_scores, letter_scores

    out = {}
    for name in ("mmlu_pro", "hellaswag"):
        items, meta = load_items(name, N_ITEMS)
        records = []
        for item_id, row in items:
            if name == "mmlu_pro":
                choices = [chr(65 + i) for i in range(len(row["options"]))]
                context = (
                    row["question"]
                    + "\n"
                    + "\n".join(f"{c}. {o}" for c, o in zip(choices, row["options"]))
                    + "\nAnswer:"
                )
                scores, token_ids = letter_scores(runner, context, choices)
                gold = row["answer_index"]
            else:
                context, choices, gold = row["ctx"], row["endings"], int(row["label"])
                scores, token_ids = hellaswag_scores(runner, context, choices)[0], None
            record = choice_record(edit_id, name, item_id, scores, gold)
            record.extra.update(
                prompt=context,
                choices=choices,
                **({"letter_token_ids": token_ids} if token_ids else {}),
            )
            records.append(record)
        out[name] = (records, meta)
    return out


def run_scoring(ids):
    from dataclasses import asdict

    import torch
    import yaml

    sys.path.insert(0, str(ROOT / "src"))
    from wsp.eval.batch import WeightRestore, provenance
    from wsp.eval.metadata import runtime_metadata
    from wsp.eval.run import HFRunner, apply_zoo_entry
    from wsp.eval.zoo_compat import load_entry_delta
    from wsp.zoo.common import resolve_base_revision

    cfg = yaml.safe_load((ROOT / "configs/eval_tf_v1.yaml").read_bytes())
    header = next(
        r
        for r in reversed(
            [json.loads(x) for x in (ZOO / "manifest.jsonl").read_text().splitlines()]
        )
        if r.get("type") == "header"
    )
    revision = header["base_revision"]
    if len(revision) != 40:
        revision = resolve_base_revision(header["base_model"], revision)
    runner = HFRunner(header["base_model"], dict(cfg["generation"]), revision=revision)
    restore = WeightRestore(runner.model)
    origin = provenance()
    for edit_id in ids:
        started = time.monotonic()
        restore.restore()
        runner.edit_id = edit_id
        if edit_id != "base":
            source = ZOO / edit_id
            delta = load_entry_delta(source, runner.model.state_dict().keys())
            restore.capture(delta)
            del delta
            apply_zoo_entry(runner.model, source)
        scored = score_entry(runner, edit_id)
        dest = OUT / edit_id
        dest.mkdir(parents=True, exist_ok=False)
        for name, (records, _) in scored.items():
            (dest / f"records_{name}.jsonl").write_text(
                "".join(json.dumps(asdict(r)) + "\n" for r in records)
            )
        (dest / "meta.json").write_text(
            json.dumps(
                {
                    **origin,
                    **runtime_metadata(runner),
                    "edit_id": edit_id,
                    "seconds": time.monotonic() - started,
                    "datasets": {k: v[1] for k, v in scored.items()},
                },
                indent=2,
            )
        )
        print(f"G2 scored {edit_id} in {time.monotonic() - started:.1f}s", flush=True)
        torch.cuda.empty_cache()
    restore.restore()


def build_report(ids):
    per_entry = {}
    for edit_id in ids:
        per_entry[edit_id] = {}
        for name in ("mmlu_pro", "hellaswag"):
            fast = [
                r for r in read_jsonl(EVALS / edit_id / "records.jsonl") if r["benchmark"] == name
            ]
            tf = read_jsonl(OUT / edit_id / f"records_{name}.jsonl")
            per_entry[edit_id][name] = compare(fast, tf, name)
    meta = json.loads((OUT / ids[0] / "meta.json").read_text())
    report = {
        "entries": ids,
        "n_items_per_benchmark": N_ITEMS,
        "source_hash": meta["source_hash"],
        "git_sha": meta["git_sha"],
        "dirty": meta["dirty"],
        "eval_device_id": meta["eval_device_id"],
        "per_entry": per_entry,
        "pooled": pooled(per_entry),
    }
    (OUT / "g2_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    ids = ["base", *(ROOT / "data/runs/d3_shards/controls.txt").read_text().split()]
    if not args.compare_only:
        run_scoring(ids)
    print(json.dumps(build_report(ids)["pooled"], indent=2))


if __name__ == "__main__":
    main()
