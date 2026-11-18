#!/usr/bin/env python3
"""Overlay RULING-wme-prereg-v2-issues (items 1-8) on docs/prereg/wme_manifest.json and regenerate the table.

    python scripts/wme_manifest_prereg_v2c.py --sbon docs/reviews/wme/wme-v2c-sbon-2026-11-18.json \
        --r6 docs/reviews/wme/wme-v2c-r6-2026-11-18.json

Item 3: the registered price uses the bank without the per-state oracle gradients that P1 removed (same pricing code
and unit times) and the S-BoN arm at the parity budget that follows from it. Item 4: S-BoN power at that budget.
Item 6: the R6 full-refit result (quadratic-null size, A7 alternative power). Items 1, 2, 5, 7, 8 are recorded as
registered rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import powersim_v2b as PB
from wsp.wm import prereg_v2 as P

MAN = ROOT / "docs/prereg/wme_manifest.json"
RULING = "docs/council/2026-11-18/RULING-wme-prereg-v2-issues.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict, sbon: dict, r6: dict, sim: dict, stagew: dict) -> dict:
    pv = m["prereg_v2"]
    unit = m["cost.unit_times"]["unit"]
    sel = sim["selected"]
    g = sim["generation"]
    p1 = PB.price(
        sel, dict(unit, oracle_s=0.0), g["greedy_s_per_answer"], g["sampled_s_per_answer"]
    )
    inp = sbon["inputs"]
    assert abs(inp["bank_gpu_h"] - p1["bank_gpu_h"]) < 1e-9, "S-BoN run used a different bank price"
    b_par = P.parity_budget(
        inp["B"], p1["bank_gpu_h"], inp["field_s"], inp["n_episodes"], inp["t_bundle_s"]
    )
    bon_h = P.bon_arm_gpu_h(b_par, inp["n_episodes"], inp["t_bundle_s"])
    te_h = pv["pricing"]["s_te"]["gpu_h"]
    ht3_h = stagew["priced_design"]["ht2"]["gpu_h"]
    reg = P.cap_check(
        {
            "design H-E1 + H-E2 (bank without P1-removed oracle gradients + confirmatory)": p1[
                "total_gpu_h"
            ],
            "S-BoN arm at the P1 parity budget": bon_h,
            "S-TE": te_h,
        }
    )
    reg["remaining_for_ht3_gpu_h"] = P.CAP_TOTAL_GPU_H - reg["total_gpu_h"]
    with_ht3 = P.cap_check(dict(reg["parts_gpu_h"], **{"H-T3 paths (if registered)": ht3_h}))
    pv["pricing"]["v2c"] = {
        "rule": "RULING-wme-prereg-v2-issues item 3: <= 90 GPU-h-eq for bank + confirmatory combined; bank re-priced "
        "without the per-state oracle gradients that P1 removed (powersim_v2b.price, oracle_s = 0, same unit times)",
        "bank_gpu_h": p1["bank_gpu_h"],
        "confirmatory_gpu_h": p1["confirmatory_gpu_h"],
        "design_total_gpu_h": p1["total_gpu_h"],
        "design_usd_9b": p1["usd_9b"],
        "s_bon_gpu_h": bon_h,
        "s_bon_budget": b_par,
        "registered_total": reg,
        "with_ht3": with_ht3,
        "note_on_ruling_arithmetic": "the ruling quotes 76.43 GPU-h-eq 'with no H-T3'; 76.43 is this total with "
        "H-T3 included (v2 report, p1_consistent_with_ht3); without H-T3 it is "
        f"{reg['total_gpu_h']:.2f}, leaving {reg['remaining_for_ht3_gpu_h']:.2f}; H-T3 ({ht3_h:.2f}) fits under "
        "either reading",
    }
    res = sbon["results"]
    pv["claims"]["S-BoN"] = dict(
        pv["claims"]["S-BoN"],
        parity_budget_measured_basis=b_par,
        power_at_measured_parity={
            k: {x: res[k][x] for x in ("rate", "power_lb", "ci95_upper", "alpha", "budget_bon")}
            for k in res
        },
        ruling="RULING-wme-prereg-v2-issues item 4: the measured parity budget; claim stays secondary",
    )
    nq, alt = r6.get("null_quadratic", {}), r6.get("alt", {})
    pv["r6_resimulation"] = {
        "ruling": "RULING-wme-prereg-v2-issues item 6",
        "pipeline": "powersim_refit_v2b.replicate(r6=True): WM-CP and S1-S3 shrunk toward the raw field on the same "
        "selection split, grid {0, .25, .5, .75, 1} and loss; ties within 0.5 % to the field; S1 = ShrunkF3uR",
        "config": r6["config"],
        "stop": r6["stop"],
        "null_quadratic": {
            k: nq.get(k)
            for k in (
                "reps",
                "rate",
                "ci95",
                "coverage_at_zero",
                "size_gate",
                "mean_ct",
                "static_chosen",
                "shrink_wm_counts",
                "shrink_static_counts",
            )
        },
        "alt": {
            k: alt.get(k)
            for k in (
                "reps",
                "rate",
                "power_lb",
                "ci95",
                "power_gate",
                "mean_ct",
                "ct_quantiles",
                "static_chosen",
                "shrink_wm_counts",
                "shrink_static_counts",
            )
        },
        "cpu_h": r6["cpu_h"],
        "consequence": "size > 0.035: R6 must be redesigned before the freeze (ruling item 6); binds only the "
        "descriptive H-T and any H-T3"
        if r6["stop"]["size_fail"] or nq.get("size_gate", "").startswith("FAIL")
        else "size gate met with R6 in place",
    }
    pv["v2c_rules"] = {
        "1": "C2 controls descriptive for every arm; validity preconditions (halt for repair, never scored): inert-"
        "edit bitwise identity, 1-vs-2-GPU bitwise parity, base canary",
        "2": "H-E1 keeps its teacher-forced refusal guards; generated guards descriptive for H-E1, gating S-TE only",
        "3": "<= 90 GPU-h-eq bank + confirmatory combined; P1 bank price registered",
        "4": "S-BoN parity budget measured; power restated; secondary",
        "5": "re-audit at the panel build mandatory; shortfall -> largest fresh count, power restated before any read; "
        "shared MMLU-Pro pool split by a salted deterministic assignment written before the build",
        "6": "R6 re-simulated (pv.r6_resimulation)",
        "7": "S-TE as authored (analytic power, one seed, guards alpha 0.025, N 523/200)",
        "8": "E2 guided by WM-E (the v2b ensemble); WM-CP is the H-T path model",
    }
    pv["rulings_sha256"][RULING] = sha(ROOT / RULING)
    pv["claims"]["D-C2"]["type"] = (
        "descriptive for every arm and analysis (RULING-wme-prereg-v2-issues item 1); validity preconditions halt "
        "the run"
    )
    m["prereg_v2"] = pv
    m["status"] = (
        "DRAFT for PREREG_WME v2 (v2b manifest + PREREG v2 overlay + RULING-wme-prereg-v2-issues); not frozen"
    )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbon", required=True)
    ap.add_argument("--r6", required=True)
    a = ap.parse_args()
    m = json.loads(MAN.read_text())
    sim = json.loads((ROOT / "docs/reviews/wme/wme-powersim-v2b-2026-11-17.json").read_text())
    stagew = json.loads((ROOT / "docs/reviews/wme/wme-stagew-2026-11-18.json").read_text())
    m = build(
        m, json.loads(Path(a.sbon).read_text()), json.loads(Path(a.r6).read_text()), sim, stagew
    )
    m["simulation.report_hash"] = dict(
        m["simulation.report_hash"], sbon_v2c=sha(a.sbon), r6_v2c=sha(a.r6)
    )
    m["generated_from_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()
    MAN.write_text(json.dumps(m, indent=1, sort_keys=True, default=str))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/wme_manifest_table.py")], check=True, cwd=ROOT
    )
    v = m["prereg_v2"]["pricing"]["v2c"]
    print(MAN, sha(MAN))
    print(
        json.dumps(
            {
                "registered": v["registered_total"]["total_gpu_h"],
                "with_ht3": v["with_ht3"]["total_gpu_h"],
            }
        )
    )


if __name__ == "__main__":
    main()
