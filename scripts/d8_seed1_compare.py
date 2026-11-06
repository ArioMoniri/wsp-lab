#!/usr/bin/env python3
"""D8 step 4: compare the regenerated seed-1 repeat panel with the retained numbers (read-only).

Usage: d8_seed1_compare.py <data/runs/p2-batch/<run_id>> [--seed0-base <base H0 run dir for 6f274a83>]

Seed 0 (from the surviving eval.json of the 10 controls and the regenerated 6f274a83 base) against seed 1:
mean / max absolute difference and Spearman over the 11 records per benchmark, the KL maximum, the per-control SD of
the seed noise, and the base values - to be compared with docs/RESULTS.md "Seed-1 repeat panel", the committed audit
log (docs/reviews/P4/paper-v2-audit-2026-11-05.md lines 614-629) and
docs/council/2026-11-05/metrology_budget_output.txt lines 19-27.
Also reports the bitwise state of the likelihood metrics (item-level records, seed 1 vs seed 0) which RESULTS.md calls
identical for mmlu_pro, hellaswag, KL and wikitext ppl (the paper says KL differs by at most 0.0003).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
CONTROLS = (ROOT / "data/runs/d3_shards/controls.txt").read_text().split()
BENCH = [
    "gsm8k",
    "math500",
    "ifeval",
    "over_refusal",
    "refusal_rate",
    "loop_rate",
    "mmlu_pro",
    "hellaswag",
    "kl_to_base",
    "wikitext_ppl",
]
# RESULTS.md (mean abs, max abs, rho), 5 Nov 2026
RETAINED = {
    "gsm8k": (0.055, 0.12, 0.15),
    "math500": (0.036, 0.12, 0.46),
    "ifeval": (0.029, 0.04, 0.10),
    "over_refusal": (0.049, 0.10, 0.67),
    "refusal_rate": (0.009, 0.04, 0.26),
    "loop_rate": (0.006, 0.02, 0.66),
}
RETAINED_SD = {
    "gsm8k": 0.0504,
    "math500": 0.0268,
    "ifeval": 0.0248,
    "refusal_rate": 0.0124,
    "over_refusal": 0.0356,
    "mmlu_pro": 0.0,
    "hellaswag": 0.0,
}
RETAINED_BASE = {"gsm8k": (0.16, 0.10), "math500": (0.16, 0.04), "ifeval": (0.40, 0.36)}
RETAINED_KL_MAX = 0.000334719  # audit log: "KL max abs difference"


def jl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def val(e, bm):
    v = e.get(bm)
    return v.get("value", v.get("mean")) if isinstance(v, dict) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", type=Path)
    ap.add_argument("--seed0-base", type=Path, required=True)
    a = ap.parse_args()
    s1, s0 = {}, {}
    recs1, recs0 = {}, {}
    for eid in CONTROLS:
        d1 = a.panel / eid / "repeats/seed1"
        s1[eid] = json.loads((d1 / "eval.json").read_text())
        recs1[eid] = jl(d1 / "records.jsonl")
        s0[eid] = json.loads((ZOO / eid / "eval.json").read_text())
        recs0[eid] = jl(ZOO / eid / "records.jsonl")
    d1 = a.panel / "base/repeats/seed1"
    s1["base"] = json.loads((d1 / "eval.json").read_text())
    recs1["base"] = jl(d1 / "records.jsonl")
    s0["base"] = json.loads((a.seed0_base / "eval.json").read_text())
    recs0["base"] = jl(a.seed0_base / "records.jsonl")
    ids = CONTROLS + ["base"]
    meta = {
        "panel": str(a.panel),
        "source_hash": s1["base"]["source_hash"][:12],
        "git_sha": s1["base"]["git_sha"],
        "dirty": s1["base"].get("dirty"),
        "config_hash": s1["base"]["config_hash"],
        "batch_size_actual": sorted({s1[i].get("batch_size") for i in ids}),
        "seed": sorted({s1[i].get("seed") for i in ids}),
        "device": sorted({s1[i].get("eval_device_id") for i in ids}),
        "n_records": len(ids),
    }
    table = {}
    for bm in BENCH:
        x0 = np.array([val(s0[i], bm) for i in ids], float)
        x1 = np.array([val(s1[i], bm) for i in ids], float)
        d = np.abs(x1 - x0)
        rho = spearmanr(x0, x1).statistic if np.ptp(x0) and np.ptp(x1) else float("nan")
        table[bm] = {
            "mean_abs": float(d.mean()),
            "max_abs": float(d.max()),
            "rho": float(rho),
            "identical": bool((d == 0).all()),
        }
        if bm in RETAINED:
            m, mx, r = RETAINED[bm]
            table[bm]["retained"] = {"mean_abs": m, "max_abs": mx, "rho": r}
            table[bm]["matches_printed"] = (
                round(table[bm]["mean_abs"], 3),
                round(table[bm]["max_abs"], 2),
                round(float(rho), 2),
            ) == (m, mx, r)
    # per-control paired seed noise on delta vs base: (seed1 - seed0 difference of the change) / sqrt 2
    sd = {}
    for bm in BENCH:
        dy0 = np.array([val(s0[i], bm) - val(s0["base"], bm) for i in CONTROLS], float)
        dy1 = np.array([val(s1[i], bm) - val(s1["base"], bm) for i in CONTROLS], float)
        sd[bm] = float(np.std(dy1 - dy0, ddof=1) / np.sqrt(2)) if len(CONTROLS) > 1 else None
    sdcheck = {
        bm: {"regenerated": round(sd[bm], 4), "retained": v, "match": round(sd[bm], 4) == v}
        for bm, v in RETAINED_SD.items()
    }
    base = {
        bm: {
            "seed0": val(s0["base"], bm),
            "seed1": val(s1["base"], bm),
            "retained": RETAINED_BASE[bm],
        }
        for bm in RETAINED_BASE
    }
    # item-level bitwise state of the likelihood / teacher-forced metrics
    bitwise = {}
    for bm in ("mmlu_pro", "hellaswag", "kl_to_base", "wikitext_ppl"):
        same = 0
        maxdiff = 0.0
        for i in ids:
            r0 = {str(r["item_id"]): r for r in recs0[i] if r["benchmark"] == bm}
            r1 = {str(r["item_id"]): r for r in recs1[i] if r["benchmark"] == bm}
            assert set(r0) == set(r1), (i, bm)
            same += all(
                {k: v for k, v in r0[k].items() if k != "edit_id"}
                == {k: v for k, v in r1[k].items() if k != "edit_id"}
                for k in r0
            )
            maxdiff = max(
                maxdiff, max(abs(float(r0[k]["correct"]) - float(r1[k]["correct"])) for k in r0)
            )
        bitwise[bm] = {
            "records_identical_entries": f"{same}/{len(ids)}",
            "max_abs_correct_diff": maxdiff,
        }
    kl_diff = max(abs(val(s1[i], "kl_to_base") - val(s0[i], "kl_to_base")) for i in ids)
    out = {
        "meta": meta,
        "seed0_vs_seed1": table,
        "seed_noise_sd_over_controls": sdcheck,
        "base": base,
        "bitwise_likelihood_metrics": bitwise,
        "kl_max_abs_diff": {"regenerated": kl_diff, "retained_audit": RETAINED_KL_MAX},
    }
    path = ROOT / "data/runs/d8/seed1_compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
