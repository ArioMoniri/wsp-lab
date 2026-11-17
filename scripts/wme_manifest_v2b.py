#!/usr/bin/env python3
"""Overlay the RULING-wme-v2b fields (A7-A11) on docs/prereg/wme_manifest.json (track A) and regenerate the table.

    python scripts/wme_manifest_v2b.py --sim docs/reviews/wme/wme-powersim-v2b-<date>.json \
        --refit docs/reviews/wme/wme-powersim-refit-v2b-<date>.json --cpsel <cpsel summary> --cal <cal json> \
        --ph100b docs/reviews/wme/wme-ph100b-<date>.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import powersim_refit_v2b as RB
from wsp.wm import powersim_v2b as PB
from wsp.wm import wmcp as CP

MAN = ROOT / "docs/prereg/wme_manifest.json"
RULING = ROOT / "docs/council/2026-11-17/RULING-wme-v2b.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("sim", "refit", "cpsel", "cal", "ph100b"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    m = json.loads(MAN.read_text())
    sim = json.loads(Path(a.sim).read_text())
    rf = json.loads(Path(a.refit).read_text())
    cps = json.loads(Path(a.cpsel).read_text())
    cal = json.loads(Path(a.cal).read_text())
    ph = json.loads(Path(a.ph100b).read_text())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()
    d = sim["selected"]
    m["status"] = (
        "DRAFT for PREREG_WME v2 (RULING-wme-v2 + RULING-wme-v2b overlay on the v1 track-A manifest); not frozen"
    )
    m["generated_from_commit"] = commit
    m["v2b"] = {"ruling": str(RULING.relative_to(ROOT)), "ruling_sha256": sha(RULING)}
    cfg = cps["frozen_config"]
    assert cfg == dataclasses.asdict(CP.CPConfig()), "the frozen selection must be the code default"
    m["model.wmcp.config"] = cfg
    m["model.wmcp.k_rule"] = {
        "rule": "K = 0 if R(E_T) == 0 else min(k_max, max(1, ceil(norm(E_T) / s0)))",
        "k_max": cfg["k_max"],
        "s0": f"{cfg['s0_scale']} x median cumulative norm of training-component h = 1 states, "
        "computed in each fit (training folds only) and recorded with the fitted model",
    }
    m["model.wmcp.selection"] = {
        k: cps[k] for k in ("procedure", "cell", "seeds", "grid", "pooled_median", "picked")
    }
    m["model.wmcp.selection"]["summary_sha256"] = sha(a.cpsel)
    m["ht.alternative"] = {
        "status": "registered (RULING-wme-v2b A7); v1's decomposition-dependent alternative withdrawn as a simulator "
        "specification error",
        "world": "SynthWorld(kind='quadratic'): F(E) = a + q exact",
        "response": "y(E) = F(E) + kappa * s_c * sum_r F(P_r E) * (rho(z_r) - 1); z_r = <u_r, E> / step; "
        "rho(z) = (1 + (z / Z0)^2)^(-1/2); s_c ~ LogNormal(0, sigma) per component",
        "Z0": RB.Z0,
        "strength_sigma": RB.STRENGTH_SIGMA,
        "kappa": cal["kappa_registered"],
        "calibration": {
            "target_median_static_error_se": RB.CAL_TARGET,
            "source": "exposed exploratory pilot-2 median refitted-static error at h >= 3",
            "rule": cal["registered_rule"],
            "reps": cal["crn"]["reps"],
            "pooled_median": cal["crn"]["pooled_median"],
            "seed0": RB.CAL_SEED0,
            "report_sha256": sha(a.cal),
        },
        "nulls": {
            "null_shuffle": "the alternative's residuals beyond the field permuted across trajectories with "
            "random signs",
            "null_quadratic": "the quadratic world itself (A7 paired null)",
        },
        "gates": "both nulls: size <= 0.035 and coverage >= 0.965; alternative: power CP lower bound >= 0.80",
    }
    m["caps"] = dict(
        PB.CAPS,
        source="RULING-wme-v2b A9 (replaces RULING-wme-resources §1)",
        usd_per_gpu_h=PB.USD_PER_GPU_H,
        usd_overhead=PB.USD_OVERHEAD,
    )
    m["selection.priority"] = sim["priority"]
    m["e1.effect"] = {
        "powered_at": "0.5 x the generated (pilot-template) effect: gen 0.025 (alt_gen_half)",
        "sensitivity": "full effect gen 0.05",
        "ruling": "RULING-wme-v2b A10",
    }
    gb = sim["generation"]
    m["gen.config"] = dict(
        m["gen.config"],
        batch=gb["batch"],
        batch_note="registered generation batch (left padding); P-H100b timing and greedy "
        "agreement with batch 8 in docs/reviews/wme/wme-ph100b-2026-11-17.json",
    )
    m["cost.generation"] = {
        "source": "P-H100b",
        "gpu": ph.get("gpu"),
        "batch": gb["batch"],
        "greedy_s_per_answer": gb["greedy_s_per_answer"],
        "sampled_s_per_answer": gb["sampled_s_per_answer"],
        "report_sha256": sha(a.ph100b),
    }
    sizes = json.loads(json.dumps(m["panel.wme_c1.sizes"]))
    for s in sizes.values():
        for k in ("bank", "dev", "sealed_test"):
            if k in s:
                s[k] = {"bank": d["n_bank"], "dev": d["n_dev"], "sealed_test": d["n_test"]}[k]
    sizes["triviaqa_gen"] = {"sealed_gen": d["n_gen_syc"]}
    m["panel.wme_c1.sizes"] = sizes
    m["sim"] = dict(
        m["sim"],
        r_f=d["r_f"],
        n_blk=d["n_blk"],
        n_cal=d["n_cal"],
        n_split=sizes,
        **{
            "n_gen,syc": d["n_gen_syc"],
            "n_rep": d["n_rep"],
            "B": d["B"],
            "n_ep_sec": d["n_ep_sec"],
            "n_rep_sec": d["n_rep_sec"],
            "n_inert": d["n_inert"],
            "pool_size": d["pool_size"],
            "reader_token_cap": sim["reader"]["cap"],
        },
    )
    m["simulation.commit"] = commit
    m["simulation.report_hash"] = {
        "cheap_v2b": sha(a.sim),
        "refit_v2b": sha(a.refit),
        "cal_v2b": sha(a.cal),
        "cpsel_v2b": sha(a.cpsel),
        "ph100b": sha(a.ph100b),
    }
    m["simulation.seeds"] = dict(
        m.get("simulation.seeds", {}), cpsel_seed0=cps["seeds"][0], cal_seed0=RB.CAL_SEED0
    )
    m["claims"] = {
        "run": sim["claims_run"],
        "unresolved_not_powered": sim["unresolved_not_powered"],
        "rank": sim.get("selected_rank"),
        "ht_full_refit": {k: rf[k] for k in ("ht", "achieved_ct_alt", "cells") if k in rf},
        "design_gpu_h": sim["price"]["total_gpu_h"],
        "design_usd_9b": sim["price"]["usd_9b"],
        "rule": "ruling §4(b); RULING-wme-v2b A8 priority, A9 caps, A10 E1 effect, A11",
    }
    MAN.write_text(json.dumps(m, indent=1, sort_keys=True, default=str))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/wme_manifest_table.py")], check=True, cwd=ROOT
    )
    print(MAN, sha(MAN), m["claims"]["run"], m["claims"]["unresolved_not_powered"])


if __name__ == "__main__":
    main()
