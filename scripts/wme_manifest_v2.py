#!/usr/bin/env python3
"""Overlay the RULING-wme-v2 fields on docs/prereg/wme_manifest.json (track A) and regenerate the table.

    python scripts/wme_manifest_v2.py --sim docs/reviews/wme/wme-powersim-v2-2026-11-17.json \
        --refit docs/reviews/wme/wme-powersim-refit-v2-2026-11-17.json
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

from wsp.wm import genaudit as GA
from wsp.wm import neardup as ND
from wsp.wm import signflip as SF
from wsp.wm import triviaqa as TQ
from wsp.wm import wmcp as CP

MAN = ROOT / "docs/prereg/wme_manifest.json"
AUDIT = ROOT / "docs/reviews/wme/wme-c1-exposure-audit-v1-2026-11-17.json"
OBJ6 = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed(label):
    return int(hashlib.sha256(f"wme-c1:{label}".encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--refit", required=True)
    a = ap.parse_args()
    m = json.loads(MAN.read_text())
    sim = json.loads(Path(a.sim).read_text())
    rf = json.loads(Path(a.refit).read_text())
    aud = json.loads(AUDIT.read_text())
    d = sim["selected"]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()
    spec9 = CP.spec_9b(
        OBJ6,
        chart_columns=("schema:CHART_SCHEMA_VERSION=2",),
        norm_groups=("writers", "embed_rows"),
    )
    m["status"] = (
        "DRAFT for PREREG_WME v2 (RULING-wme-v2 overlay on the v1 track-A manifest); not frozen"
    )
    m["generated_from_commit"] = commit
    m["v2"] = {
        "ruling": "docs/council/2026-11-17/RULING-wme-v2-design.md",
        "ruling_sha256": sha(ROOT / "docs/council/2026-11-17/RULING-wme-v2-design.md"),
    }
    m["model.wmcp.config"] = dataclasses.asdict(CP.CPConfig())
    m["model.wmcp.k_rule"] = {
        "rule": "K = 0 if R(E_T) == 0 else min(K_MAX, max(1, ceil(norm(E_T) / s0)))",
        "k_max": CP.K_MAX,
        "s0": "median cumulative norm of training-component h = 1 states, "
        "computed in each fit (training folds only) and recorded with the fitted model",
    }
    m["model.representation"] = {
        "spec_9b_columns": list(spec9.names),
        "degrees": list(spec9.degree),
        "builder": "wsp.wm.wmcp.r_from_contractions (collect.contract of the realised "
        "cumulative deltas; unsupported q -> 0, flagged)",
        "chart_schema_version": 2,
    }
    m["model.primary_ht"] = (
        "WM-CP (wsp.wm.wmcp.CanonicalPathWM) vs the R-space static family S1 F3uR, S2 StaticR, "
        "S3 StaticR-aux (sized to WM-CP +-10 %)"
    )
    m["model.descriptive_arms"] = ["WM-E v0 history-dependent on R (wmcp.v0_trajectory / v0_field)"]
    m["controls.c2s_c2d"] = (
        "hold by construction for WM-CP (prediction is a function of R(E_T)); reported, never "
        "counted as evidence"
    )
    m["controls.c2i"] = dict(
        m.get("controls.inert", {}), gate="hard gate (I1 round trips; I2 never-emitted rows)"
    )
    m["inference.ht"] = {
        "test": "component-level sign-flip, one-sided 0.025",
        "n_flip": SF.N_FLIP,
        "seed": seed("signflip:ht"),
        "unit": "component mean of |e_static| - |e_WM| (SE-scaled)",
        "interval": "test inversion",
        "ct": "descriptive, interval by inverting the same test",
    }
    m["inference.e2"] = {
        "test": "family-stratified sign-flip on episode paired differences, one-sided 0.025",
        "n_flip": SF.N_FLIP,
        "seed": seed("signflip:e2"),
        "statistic": "mean over the 3 families of the family-mean difference",
    }
    m["seeds.signflip.ht"] = m["inference.ht"]["seed"]
    m["seeds.signflip.e2"] = m["inference.e2"]["seed"]
    m["gen.pool"] = {
        "source": TQ.PIN,
        "template": "wm.p2data NEUTRAL_TEMPLATE / SUGGESTED_TEMPLATE (unchanged)",
        "incorrect_answer_rule": " ".join(
            aud["triviaqa_incorrect_rule"].split("**", 1)[-1].split()
        ),
        "pool_groups": aud["triviaqa_pool_size"],
        "pool_file_sha256": aud["triviaqa_pool_file_sha256"],
    }
    m["gen.config"] = GA.SPEC_V1
    m["gen.parsers"] = {
        "module": "src/wsp/wm/genaudit.py",
        "sha256": sha(ROOT / "src/wsp/wm/genaudit.py"),
        "functions": ["squad_norm", "classify_v1", "detect_loop_v1", "audit_v1", "audit_v1_single"],
    }
    m["panel.wme_c1.group_policy"] = ND.POLICY
    m["panel.wme_c1.builder"] = (
        "wsp.wm.panel_c1.build_c1_panel_v1 (sealed_gen, sealed_test, dev, bank, construction)"
    )
    m["panel.wme_c1.split_hashes"] = {
        "status": "deferred",
        "binds_at": "freeze build with build_c1_panel_v1 at the sizes below",
    }
    m["audit.exclusion_report_hash"] = {
        "v1_policy": sha(AUDIT),
        **{k: v for k, v in m.get("audit.exclusion_report_hash", {}).items() if k == "supplement"},
    }
    m["audit.available_groups"] = {o: v["available_groups"] for o, v in aud["objectives"].items()}
    m["guards.bounds"] = (
        "item-group bootstrap-t R = 9,999 on both scales (wm.arms.spec.guard_bounds_from_items)"
    )
    sizes = json.loads(json.dumps(m["panel.wme_c1.sizes"]))
    for o, s in sizes.items():
        for k in ("bank", "dev", "sealed_test"):
            if k in s:
                s[k] = {"bank": d["n_bank"], "dev": d["n_dev"], "sealed_test": d["n_test"]}[k]
        if o == "sycophancy_real":
            s.pop("sealed_gen", None)
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
    m["sim"].pop("m_min_HT", None)
    m["sim"]["m_min_HT"] = (
        "sign-flip test valid for any m >= 2; requirement checked per cell (v2 grid)"
    )
    m["simulation.commit"] = commit
    m["simulation.report_hash"] = {"cheap_v2": sha(a.sim), "refit_v2": sha(a.refit)}
    m["simulation.seeds"] = {"grid": 20261117, "refit": 20261117, "signflip": SF.SEED}
    failed = rf["ht"] != "powered"
    run = list(sim["claims_run"])
    unres = list(sim["unresolved_not_powered"])
    if failed and "ht" in run:
        run.remove("ht")
        unres.insert(0, "ht")
    keep = (
        "reps",
        "rate",
        "mean_ct",
        "mean_stat",
        "c2i_pass",
        "pass",
        "coverage_at_zero",
        "power_lb",
    )
    m["claims"] = {
        "run": run,
        "unresolved_not_powered": unres,
        "cheap_grid_powerable": sim["powerable"],
        "cheap_grid_selection": sim["claims_run"],
        "full_refit": {
            c: {
                "cell": v["cell"],
                "completed": v["completed"],
                "futility": v["futility"],
                **{a: {x: v[a][x] for x in keep if x in v[a]} for a in ("null", "alt") if a in v},
            }
            for c, v in rf["cells"].items()
        },
        "ht_outcome": rf["ht"],
        "achieved_ct_alt": rf["achieved_ct_alt"],
        "design_note": "the design is the pre-refit §8.5 selection (ht+e1), kept unchanged; re-applying §8.5 "
        "to the surviving claim set {e1} would select the cheapest e1 design (r_f 8; 40.1 GPU-h-eq): chair's "
        "choice",
        "rule": "ruling §4(b) and RULING-wme-v2 A6",
    }
    MAN.write_text(json.dumps(m, indent=1, sort_keys=True, default=str))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/wme_manifest_table.py")], check=True, cwd=ROOT
    )
    print(MAN, sha(MAN), m["claims"]["run"], m["claims"]["unresolved_not_powered"])


if __name__ == "__main__":
    main()
