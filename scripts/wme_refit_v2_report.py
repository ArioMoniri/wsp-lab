#!/usr/bin/env python3
"""Summarise the v2 full-refit validation (RULING-wme-v2 A6 under ruling §4(b)) into the review JSON.

    python scripts/wme_refit_v2_report.py --selected <dir> --next <dir> --out docs/reviews/wme/wme-powersim-refit-v2-<date>.json

Futility rule (deterministic, declared before the stop and disclosed): a cell's alternative arm cannot pass once its
non-rejections exceed ``reps_alt - k_req``, where k_req is the smallest rejection count whose one-sided Clopper-Pearson
lower bound reaches the registered 0.80 at the planned ``reps_alt``. A run stopped under this rule reports the
replicates completed; the null arm's rates are then descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from wsp.wm import powersim_refit_v2 as R2
from wsp.wm import powersim_v1 as PV


def k_required(n: int) -> int:
    return next(k for k in range(n + 1) if PV.cp_lower_one_sided(k, n) >= PV.REQ["power_lb_min"])


def cell(d: Path) -> dict:
    cfg = json.loads((d / "config.json").read_text())
    rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines() if x.strip()]
    s = R2.summarise(rows)
    n_alt = cfg["reps_alt"]
    k = k_required(n_alt)
    alt = [r for r in rows if not r["null"]]
    non = sum(not r["reject"] for r in alt)
    fut = non > n_alt - k
    out = {
        "cell": cfg["cell"],
        "gamma": cfg["gamma"],
        "planned": {"reps_null": cfg["reps_null"], "reps_alt": n_alt},
        "completed": {"null": sum(r["null"] for r in rows), "alt": len(alt)},
        "futility": {
            "k_required_at_planned": k,
            "alt_non_rejections": non,
            "cannot_pass": bool(fut),
        },
        "cpu_h": sum(r["seconds"] for r in rows) / 3600,
        "max_rss_mb": max((r.get("rss_mb", 0) for r in rows), default=0),
        "results_sha256": hashlib.sha256((d / "results.jsonl").read_bytes()).hexdigest(),
        **s,
    }
    for arm in ("null", "alt"):
        rs = [r for r in rows if r["null"] == (arm == "null")]
        if rs and arm in out:
            out[arm]["exact_zero_share"] = float(
                np.mean([r["stat"] == 0.0 and r["ct"] == 0.0 for r in rs])
            )
            out[arm]["ct_quantiles"] = [
                float(q) for q in np.quantile([r["ct"] for r in rs], [0.1, 0.5, 0.9])
            ]
            out[arm]["share_stat_negative"] = float(np.mean([r["stat"] < 0 for r in rs]))
    if fut and "alt" in out:
        out["alt"]["pass"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--next", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cells = {"selected": cell(Path(a.selected)), "next": cell(Path(a.next))}
    passed = {
        k: bool(v.get("null", {}).get("pass") and v.get("alt", {}).get("pass"))
        for k, v in cells.items()
    }
    rep = {
        "label": "PREREG_WME v2 full-refit validation (RULING-wme-v2 A6, ruling §4(b)); alternative not re-tuned",
        "pipeline": "wsp.wm.powersim_refit_v2.replicate: WM-CP vs the R-space static family (S1-S3, inner-fold "
        "choice), component-level sign-flip (10,000 flips, one-sided 0.025), C2-I gate (I1 round trips, I2 "
        "chart-column controls); C2-S/C2-D by construction",
        "requirements": PV.REQ,
        "cells": cells,
        "passed": passed,
        "ht": "powered" if any(passed.values()) else "unresolved — not powered",
        "achieved_ct_alt": {k: v.get("alt", {}).get("mean_ct") for k, v in cells.items()},
        "note": a.note,
    }
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(
        json.dumps({"passed": passed, "ht": rep["ht"], "achieved_ct_alt": rep["achieved_ct_alt"]})
    )


if __name__ == "__main__":
    main()
