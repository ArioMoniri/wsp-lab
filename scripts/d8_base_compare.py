#!/usr/bin/env python3
"""D8 step 3 verification of the regenerated H0 base records (read-only; writes a JSON report).

For every record under data/evals/qwen3.5-0.8b/base/devices/H0/<run>/ (and the three source hashes of
docs/reviews/P5/board-a-incident-d8-2026-11-06.md section 2):
  * summary scalars against the base values printed in docs/RESULTS.md (gsm8k .16, math500 .16, ifeval .40,
    mmlu_pro .28, hellaswag .39, over-refusal .10, wikitext ppl 22.90), to the printed digits;
  * item identity against the two Lambda base records (same (benchmark, item_id) set, same order);
  * pairwise bitwise identity of the item-level records across the regenerated bases (same device and seed, so
    D2 expects identical records across source trees), and the largest difference otherwise;
  * item-set agreement with the control records of the same source hash (item_order_hash recomputed from records).
Cross-device (H0 vs H1/H2) generative items are expected to differ (D2); likelihood items are reported, not gated.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data/evals/qwen3.5-0.8b"
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
PRINTED = {  # docs/RESULTS.md, base scalars (printed precision)
    "gsm8k": (0.16, 2),
    "math500": (0.16, 2),
    "ifeval": (0.40, 2),
    "mmlu_pro": (0.28, 2),
    "hellaswag": (0.39, 2),
    "over_refusal": (0.10, 2),
    "wikitext_ppl": (22.90, 2),
}


def load(d):
    e = json.loads((d / "eval.json").read_text())
    rows = [json.loads(x) for x in (d / "records.jsonl").read_text().splitlines() if x.strip()]
    return e, rows


def key(r):
    return (r["benchmark"], str(r["item_id"]))


def strip(r):
    return {k: v for k, v in r.items() if k != "edit_id"}


def order_hash(rows):
    return hashlib.sha256(json.dumps([key(r) for r in rows]).encode()).hexdigest()[:16]


def compare(a_rows, b_rows):
    ka, kb = [key(r) for r in a_rows], [key(r) for r in b_rows]
    out = {
        "same_item_set": set(ka) == set(kb),
        "same_order": ka == kb,
        "n_a": len(ka),
        "n_b": len(kb),
    }
    ia, ib = {key(r): strip(r) for r in a_rows}, {key(r): strip(r) for r in b_rows}
    common = [k for k in ka if k in ib]
    ident = sum(ia[k] == ib[k] for k in common)
    out["records_identical"] = f"{ident}/{len(common)}"

    def shared(
        x, y
    ):  # equality on the fields both records carry (a later commit may add a key to `extra`)
        ex, ey = x.get("extra", {}), y.get("extra", {})
        top = {k for k in x if k != "extra"} | {k for k in y if k != "extra"}
        return all(x.get(k) == y.get(k) for k in top) and all(ex[k] == ey[k] for k in ex if k in ey)

    out["records_identical_on_shared_fields"] = (
        f"{sum(shared(ia[k], ib[k]) for k in common)}/{len(common)}"
    )
    keys_only = {
        k2 for k in common for k2 in set(ia[k].get("extra", {})) ^ set(ib[k].get("extra", {}))
    }
    out["extra_keys_present_in_only_one"] = sorted(keys_only)
    per = {}
    for bm in sorted({k[0] for k in common}):
        ks = [k for k in common if k[0] == bm]
        diffs = [abs(float(ia[k]["correct"]) - float(ib[k]["correct"])) for k in ks]
        per[bm] = {
            "n": len(ks),
            "identical": sum(ia[k] == ib[k] for k in ks),
            "max_abs_correct_diff": max(diffs),
        }
    out["per_benchmark"] = per
    return out


def main():
    runs = {}
    for d in sorted((EV / "base/devices/H0").glob("*")):
        e, rows = load(d)
        runs[d.name] = (d, e, rows)
    if not runs:
        sys.exit("no regenerated H0 base records")
    report = {"runs": {}, "pairwise": {}, "lambda": {}}
    for run, (d, e, rows) in runs.items():
        scal = {}
        for bm, (want, nd) in PRINTED.items():
            got = e.get(bm)
            if isinstance(got, dict):
                got = got.get("value", got.get("mean"))
            scal[bm] = {
                "regenerated": got,
                "printed": want,
                "match": got is not None and round(float(got), nd) == want,
            }
        report["runs"][run] = {
            "git_sha": e["git_sha"],
            "source_hash": e["source_hash"][:12],
            "dirty": e.get("dirty"),
            "config_hash": e["config_hash"][:16],
            "batch_size": e.get("batch_size"),
            "eval_status": e.get("eval_status"),
            "minutes": e.get("minutes"),
            "n_records": len(rows),
            "order_hash": order_hash(rows),
            "scalars_vs_RESULTS": scal,
        }
    names = list(runs)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            report["pairwise"][f"{x}|{y}"] = compare(runs[x][2], runs[y][2])
    for ld in sorted(EV.glob("base-lambda-*")):
        _le, lrows = load(ld)
        for run, (_, _, rows) in runs.items():
            report["lambda"][f"{run}|{ld.name}"] = compare(rows, lrows)
    # controls of the same source hash: item identity against their H0 records
    ctl = {}
    for run, (_, e, rows) in runs.items():
        sh = e["source_hash"][:8]
        matches = []
        for f in sorted(ZOO.glob("*/eval.json")):
            p = json.loads(f.read_text())
            if (
                p.get("eval_device_id") == "H0"
                and p.get("source_hash", "")[:8] == sh
                and (f.parent / "records.jsonl").exists()
            ):
                crows = [
                    json.loads(x)
                    for x in (f.parent / "records.jsonl").read_text().splitlines()
                    if x.strip()
                ]
                matches.append(
                    {
                        "id": f.parent.name,
                        "same_items": [key(r) for r in crows] == [key(r) for r in rows],
                        "order_hash_stored": p.get("item_order_hash", "")[:16],
                    }
                )
        ctl[run] = {
            "controls_with_records": len(matches),
            "all_same_items": all(m["same_items"] for m in matches),
            "examples": matches[:3],
        }
    report["controls_same_source"] = ctl
    out = ROOT / "data/runs/d8/base_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
