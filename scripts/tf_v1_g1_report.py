#!/usr/bin/env python3
"""G1 bitwise-repeat report: every `repeats/g1` execution against its primary (PREREG_H1_TF 6).

Compares complete records (every benchmark, every field, per-item logits/log-probs and per-position
KL), not summaries. Writes data/evals/qwen3.5-0.8b/tf_v1/g1_report.json; prints a markdown block.
Never launches evaluations.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / "data/evals/qwen3.5-0.8b/tf_v1"
LABEL = os.environ.get("TF_G1_LABEL", "g1")  # g1b: the two chain predecessors regenerated on 7 Nov
FIELDS = {
    "mmlu_pro": ("gold_log_prob", "option_log_probs", "raw_option_log_likelihoods"),
    "hellaswag": ("gold_log_prob", "option_log_probs", "raw_option_log_likelihoods"),
    "kl_to_base": ("full_sequence_kl", "first_token_kl", "position_kl"),
    "wikitext_ppl": ("mean_nll",),
    "refusal_rate": ("phrase_margin",),
    "over_refusal": ("phrase_margin",),
}


def read(directory):
    lines = (directory / "records.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def flat(value):
    if isinstance(value, list):
        for v in value:
            yield from flat(v)
    else:
        yield value


def compare(primary, repeat):
    if len(primary) != len(repeat):
        raise ValueError("record counts differ")
    out = {"n_records": len(primary), "records_identical": primary == repeat, "max_abs": {}}
    for a, b in zip(primary, repeat, strict=True):
        if (a["benchmark"], a["item_id"]) != (b["benchmark"], b["item_id"]):
            raise ValueError("record order differs")
        for key in ("correct",):
            d = abs(a[key] - b[key])
            out["max_abs"][a["benchmark"]] = max(out["max_abs"].get(a["benchmark"], 0.0), d)
        for key in FIELDS[a["benchmark"]]:
            d = max(abs(x - y) for x, y in zip(flat(a["extra"][key]), flat(b["extra"][key])))
            tag = f"{a['benchmark']}.{key}"
            out["max_abs"][tag] = max(out["max_abs"].get(tag, 0.0), d)
    return out


def main():
    ids = [p.parent.parent.parent.name for p in sorted(TF.glob(f"*/repeats/{LABEL}/records.jsonl"))]
    report = {"label": LABEL, "entries": {}}
    for i in ids:
        d = TF / i
        pe = json.loads((d / "eval.json").read_text())
        re = json.loads((d / "repeats" / LABEL / "eval.json").read_text())
        result = compare(read(d), read(d / "repeats" / LABEL))
        result["provenance"] = {
            "repeat_id": re["repeat_id"],
            "distinct_run": re["run_id"] != pe["run_id"],
            "same_source_hash": re["source_hash"] == pe["source_hash"],
            "repeat_tree_clean": re["dirty"] is False,
            "same_device": re["eval_device_id"] == pe["eval_device_id"],
            "same_batch": re["batch_size"] == pe["batch_size"],
            "repeat_cache_mode": (re.get("tf_cache") or {}).get("mode"),
            "primary_cache_mode": (pe.get("tf_cache") or {}).get("mode"),
        }
        report["entries"][i] = result
    worst = {}
    for r in report["entries"].values():
        for k, v in r["max_abs"].items():
            worst[k] = max(worst.get(k, 0.0), v)
    report["n_entries"] = len(ids)
    report["all_records_identical"] = all(
        r["records_identical"] for r in report["entries"].values()
    )
    report["worst_max_abs"] = worst
    (TF / "g1_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "entries"}, indent=2))
    return report


if __name__ == "__main__":
    main()
