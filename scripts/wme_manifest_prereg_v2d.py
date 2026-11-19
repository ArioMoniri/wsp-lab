#!/usr/bin/env python3
"""Overlay Board A's v2 required changes 1-7 (docs/reviews/council/board-a-prereg-wme-v2-2026-11-18.md) on
docs/prereg/wme_manifest.json and regenerate the table.

    python scripts/wme_manifest_prereg_v2d.py --e2sens docs/reviews/wme/wme-v2d-e2sens-2026-11-19.json

Every value comes from code (wsp.wm.e2run, guards_v2, prereg_v2, runguard, arms.sbon, panels_v2d), a committed
report or a ruling. The W2 panel hash is a slot (null) until feat/wme-stagew2 merges.
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
sys.path.insert(0, str(ROOT / "scripts"))

from wme_manifest_prereg_v2c import R6_CONSEQUENCE_ADDENDUM

from wsp.wm import e2run as E2R
from wsp.wm import guards_v2 as GV
from wsp.wm import panels_v2d as PV2
from wsp.wm import prereg_v2 as P
from wsp.wm import runguard as RG
from wsp.wm.arms import sbon as SB

MAN = ROOT / "docs/prereg/wme_manifest.json"
ISSUES = "docs/council/2026-11-18/RULING-wme-prereg-v2-issues.md"
BOARD = "docs/reviews/council/board-a-prereg-wme-v2-2026-11-18.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict, e2sens: dict) -> dict:
    pv = m["prereg_v2"]
    # change 1: digests (the issues ruling now includes its addendum) and the R6 consequence
    old = pv["rulings_sha256"][ISSUES]
    pv["rulings_sha256"][ISSUES] = sha(ROOT / ISSUES)
    pv["rulings_sha256"][BOARD] = sha(ROOT / BOARD)
    pv.setdefault("superseded_digests", {})[ISSUES] = {
        "sha256": old,
        "note": "the ruling before its 18 Nov addendum",
    }
    r6 = pv["r6_resimulation"]
    r6["consequence_superseded"] = r6["consequence"]
    r6["consequence"] = R6_CONSEQUENCE_ADDENDUM
    # change 2: effective post-override configuration
    eff = E2R.effective_config()
    m["e2.config_effective"] = dict(
        eff, entry_point="wsp.wm.e2run.fit_planning_predictors / make_arms"
    )
    m["model.wm.config"] = dataclasses.asdict(E2R.E2_WM_CONFIG)
    sc = m["model.static.configs"]
    for k in ("S2", "S3"):
        sc[k] = dict(sc[k], slope_transport=False)
    sc["note"] = (
        "S1-S3 are the descriptive H-T static family (powersim_refit_v2 R-space members). E2's static arm is "
        "MatchedStatic only (e2.config_effective); slope_transport False everywhere (P1)"
    )
    # change 4: the E2 null and the sensitivity result
    m["inference.e2"] = dict(
        m["inference.e2"],
        null=(
            "conditional on the realised sealed test items, the declared library and the frozen predictors: the 24 "
            "episode paired differences are jointly sign-symmetric about 0 (each symmetric, independent across "
            "episodes given the items); exact under that null. Inference does not generalise over the item population"
        ),
        robustness="signflip.rb_e2_test (RB-E2): family-stratified studentised bootstrap of the same weighted "
        "mean, 9,999 resamples, seed 20261119",
        sensitivity={
            "report": "docs/reviews/wme/wme-v2d-e2sens-2026-11-19.json",
            "sha256": sha(ROOT / "docs/reviews/wme/wme-v2d-e2sens-2026-11-19.json"),
            "decision": e2sens["decision"],
            "rates": {
                k: {x: v[x]["rate"] for x in ("sf", "rb", "iu")}
                for k, v in e2sens["results"].items()
            },
        },
        planning_power_note="0.828 is model-based planning power under an imposed guidance effect "
        "(powersim_v2.sim_e2), not a guarantee for learned search",
    )
    # change 5: S-TE sizing and guards
    pv["claims"]["S-TE"] = dict(
        pv["claims"]["S-TE"],
        sizing_label="analytic benchmark sizing: power 0.8 for one benchmark's paired test at Holm's first level "
        "0.0125, before the conjunctive guards; not the joint S-TE success probability",
        sizing_exact={
            k: dict(v, wilson_upper=P.wilson_upper(v["discordant"], v["n_probe"]))
            for k, v in P.TE_SIZING.items()
        },
    )
    m["guards.generated"] = dict(
        m["guards.generated"],
        test="wsp.wm.guards_v2.guard_test: one-sided alpha "
        f"{GV.GUARD_ALPHA}; bootstrap-t ({GV.N_BOOT} resamples) when >= {GV.SPARSE_MIN_NONZERO} non-zero paired "
        "differences and SE > 0, else the one-sided Clopper-Pearson upper limit of P(d = +1) (sparse/degenerate rule)",
        alpha=GV.GUARD_ALPHA,
        sparse_min_nonzero=GV.SPARSE_MIN_NONZERO,
    )
    # change 6: run guard
    m["harness.runguard"] = {
        "module": "wsp.wm.runguard (passed as harness.run_units(guard=))",
        "cap_gpu_h_combined": RG.CAP_GPU_H,
        "cap_usd_9b": RG.CAP_USD,
        "phase_pause": RG.PHASE_PAUSE,
        "unit_attempts_total_across_relaunches": RG.UNIT_ATTEMPTS,
        "phase_relaunches": RG.PHASE_RELAUNCHES,
        "technical_causes": list(RG.TECHNICAL_CAUSES),
        "canary": {
            "items": "64 exposed wme-p3h holdout MMLU-Pro items (the stage-W canary set), salted rank order",
            "engines": {
                "vllm (S-TE)": "stage-W frozen numerics: prefix caching off, seed 0, gpu_memory_utilization 0.88, "
                "max_num_seqs 256, max_model_len 8192, greedy, max_new_tokens 2048; sha256 of the token ids",
                "hf generate (E1 generation)": "batch 128, left padding, greedy, 128 new tokens, non-thinking; sha256 "
                "of the token ids",
                "teacher-forced (bank/dev/test)": "the 16 construction items x 7 objectives of the P-H100 inert check; "
                "sha256 of the fp32 value bytes",
            },
            "reference": "the first confirmatory instance's start canary per engine, recorded before any outcome",
            "evidence": {
                "stage_w_canary_bitwise_repeat": "b34cc04e97b4d243191b9a9009f30d75bddb2e1ce5d1674c16f9e5deddf1e896",
                "ph100_inert_pass": True,
                "ph100_parity_1v2_gpu_bitwise": "10,080 group means, max relative difference 0",
            },
        },
    }
    # change 7: S-BoN
    pc = SB.plan_counts(pv["claims"]["S-BoN"]["parity_budget_measured_basis"])
    pv["claims"]["S-BoN"] = dict(
        pv["claims"]["S-BoN"],
        runner="wsp.wm.arms.sbon.run_sbon",
        plan_at_planning_budget=pc,
        holm_threshold_power={
            "0.0025 (first of 10)": pv["claims"]["S-BoN"]["power_at_measured_parity"][
                "measured_parity_alpha_0.0025"
            ]["rate"],
            "0.0125 (stage-W reference)": pv["claims"]["S-BoN"]["power_at_measured_parity"][
                "measured_parity"
            ]["rate"],
            "0.025 (last of 10)": pv["claims"]["S-BoN"]["power_at_measured_parity"][
                "measured_parity_alpha_0.025"
            ]["rate"],
        },
    )
    # change 3: exposures, allocation, builders
    ex = m["exposures.v2"]
    ex["required_before_any_build"] = list(PV2.REQUIRED_EXPOSURES)
    ex["wme-w2"] = {
        "status": "pending: registered as wme-w2 on feat/wme-stagew2 (track W2 running); not merged",
        "manifest_sha256": None,
        "outcome": None,
        "ht3_decision": None,
        "fill": "after feat/wme-stagew2 merges, by dated amendment, before any confirmatory panel is built",
    }
    ex["order"] = (
        "W2 merges (wme-w2 in EXPOSURES) -> wme-c1 -> wme-c2-gen -> H-T3 panel (if any) -> wme-c27"
    )
    ex["mmlu_allocation"] = {
        "module": "wsp.wm.panels_v2d.allocate",
        "salt": PV2.ALLOC_SALT,
        "consumers": [list(c) for c in PV2.CONSUMERS],
        "sha256": None,
        "fill": "written at the build, before either builder runs",
    }
    ex["builders"] = {
        "wme-c1": "wsp.wm.panels_v2d.build_c1_panel_v2d",
        "wme-c2-gen": "wsp.wm.panels_v2d.build_c2gen_panel",
        "audit": "wsp.wm.panels_v2d.audit (hash-only, cross-panel)",
    }
    pv["v2d"] = {
        "board_a": BOARD,
        "status": "changes 1, 2, 4, 5, 6, 7 closed in text and code; change 3 code closed, build and audit pending "
        "the W2 merge",
    }
    m["prereg_v2"] = pv
    m["status"] = (
        "DRAFT for PREREG_WME v2 (v2b manifest + PREREG v2 overlay + RULING-wme-prereg-v2-issues + Board A v2 "
        "changes 1-7); not frozen; W2 slots pending"
    )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2sens", required=True)
    a = ap.parse_args()
    m = json.loads(MAN.read_text())
    m = build(m, json.loads(Path(a.e2sens).read_text()))
    m["simulation.report_hash"] = dict(m["simulation.report_hash"], e2sens_v2d=sha(a.e2sens))
    m["generated_from_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()
    MAN.write_text(json.dumps(m, indent=1, sort_keys=True, default=str))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/wme_manifest_table.py")], check=True, cwd=ROOT
    )
    print(MAN, sha(MAN))


if __name__ == "__main__":
    main()
