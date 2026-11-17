#!/usr/bin/env python3
"""Executable manifest for PREREG_WME DRAFT v1 (track A fields) and the placeholder -> value table for the chair.

Every PREREG v1 ``[MANIFEST: ...]`` field track A owns and every ``[SIM: ...]`` value is filled from a committed
artefact (each with its sha256) or marked ``deferred`` with the exact step that binds it. Track B's fields (arms,
vocabulary-row field, 27B) are left to its fragment. PREREG_WME.md itself is not edited.

    python scripts/wme_manifest.py --sim <report.json> --ph100 <summary.json> --roots <roots.json>
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

from wsp.wm import adapter9b as AD
from wsp.wm import e2 as E2
from wsp.wm import genaudit as GA
from wsp.wm import model as M
from wsp.wm import ph100 as H
from wsp.wm import powersim_v1 as PV
from wsp.wm import rootsampler as RS

OUT = ROOT / "docs/prereg/wme_manifest.json"
TABLE = ROOT / "docs/prereg/wme_manifest_table.md"
OBJ = (
    "sycophancy_real",
    "gsm8k_gold_lp",
    "mathcomp_gold_lp",
    "wikitext_nll",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(label):
    return int(hashlib.sha256(f"wme-c1:{label}".encode()).hexdigest()[:8], 16)


def blocks_and_graph(roots, n_blk, n_cal):
    """Seeded block membership (4 roots from 4 distinct families, a root in at most one block), the component
    graph (blocks; singleton roots with their controls) and the calibration components (stratified, first)."""
    import numpy as np

    rng = np.random.default_rng(seed("blocks"))
    fam = {}
    for r in roots:
        fam.setdefault(r["family"], []).append(r["id"])
    for f in fam:
        fam[f] = sorted(fam[f])
        rng.shuffle(fam[f])
    left = {f: len(v) for f, v in fam.items()}
    blocks = []
    for _ in range(n_blk):
        fs = sorted(fam, key=lambda f: (-left[f], rng.random()))[:4]
        blocks.append(sorted(fam[f].pop() for f in fs))
        for f in fs:
            left[f] -= 1
    in_block = {r for b in blocks for r in b}
    comps = [{"id": f"block:{i:02d}", "stratum": "block", "roots": b} for i, b in enumerate(blocks)]
    comps += [
        {"id": f"single:{r['id']}", "stratum": r["family"], "roots": [r["id"]]}
        for r in roots
        if r["id"] not in in_block
    ]
    strata = sorted({c["stratum"] for c in comps})
    order = rng.permutation(len(comps)).tolist()
    by = {s: [i for i in order if comps[i]["stratum"] == s] for s in strata}
    cal = []
    while len(cal) < n_cal:
        for s in strata:
            if by[s] and len(cal) < n_cal:
                cal.append(comps[by[s].pop(0)]["id"])
    g = {"components": comps, "calibration": sorted(cal)}
    return blocks, g, hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--ph100", required=True)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--refit", default=None)
    a = ap.parse_args()
    sim = json.loads(Path(a.sim).read_text())
    ph = json.loads(Path(a.ph100).read_text())
    roots = json.loads(Path(a.roots).read_text())
    d = sim["selected"]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    blocks, graph, graph_hash = blocks_and_graph(roots["roots"], d["n_blk"], d["n_cal"])
    audit_p = ROOT / "docs/reviews/wme/wme-c1-exposure-audit-2026-11-17.json"
    supp_p = ROOT / "docs/reviews/wme/wme-supplement-audit-2026-11-17.json"
    aud = json.loads(audit_p.read_text())["objectives"]
    supp = json.loads(supp_p.read_text())
    harm = sim["supplement"]
    sizes = {}
    for o in OBJ:
        s = {
            "bank": d["n_bank"],
            "dev": d["n_dev"],
            "sealed_test": d["n_test"],
            "construction_field": 24,
        }
        if o == "mathcomp_gold_lp":
            s.pop("bank")
            s["sealed_gen"] = d["n_gen_math"] if d["n_rep_sec"] else 0
        if o == "gsm8k_gold_lp":
            s["sealed_gen"] = d["n_gen_gsm"] if d["n_rep_sec"] else 0
        if o == "sycophancy_real":
            s["sealed_gen"] = d["n_gen_syc"]
            s["construction_e2"] = d["e2_shard"]
        sizes[o] = s
    episodes = {
        o: E2.episodes(o, d["n_rep"] if o == "sycophancy_real" else d["n_rep_sec"])
        for o in ("sycophancy_real", "gsm8k_gold_lp", "mathcomp_gold_lp")
    }
    cfg = dataclasses.asdict(M.WMConfig())
    m = {
        "status": "DRAFT for PREREG_WME v1 (track A fields); not frozen; no confirmatory outcome exists",
        "generated_from_commit": commit,
        "target.theta0_hash": ph["theta0"],
        "sources": {
            "pilot_pins": "wsp.wm.p2data.SOURCES and scripts/wme_build_panel.py PINS (unchanged from pilot 2)",
            "harmful_supplement": {n: supp["pins"][n] for n in harm["sources"]},
        },
        "panel.wme_c1.minhash_seed": 20261117,
        "panel.wme_c1.group_policy": {
            "id": "wme-c1-group-v1",
            "exact": "sha256 of wm.panel.normalise",
            "near": "character 5-gram Jaccard "
            ">= 0.8, punctuation stripped, exact for sources < 50,000 groups (every source here)",
            "implemented_in": "wsp.wm.supplement (char policy); the 17 Nov exposure audit used the v0 word-5-gram "
            "0.7 policy and must be re-run under this policy at the freeze build",
        },
        "panel.wme_c1.split_hashes": {
            "status": "deferred",
            "binds_at": "freeze build of wme-c1 with the sizes below "
            "(builder must gain the v1 split set gen/test/dev/bank/construction)",
        },
        "panel.wme_c1.sizes": sizes,
        "audit.exclusion_report_hash": {"exposure": sha(audit_p), "supplement": sha(supp_p)},
        "audit.available_groups": {o: aud[o]["available_groups"] for o in OBJ}
        | {"refusal_margin_harmful_with_supplement": harm["pool"], "sycophancy_real_triviaqa": 256},
        "tap.seed": seed("tap"),
        "bank.roots": {
            "identity_digest": roots["identity_digest"],
            "file": "data/wme/bank_roots/<run>/roots.json",
            "n": len(roots["roots"]),
            "feasible": roots["feasible"],
            "smoke": roots["smoke"],
            "control_seeds": [c["control_seed"] for c in roots["controls"]],
        },
        "bank.noise_rho": roots["noise_rho"],
        "bank.contrast_catalogue": {
            "sources": list(RS.SOURCES),
            "n_pairs": len(RS.catalogue()),
            "sha256": hashlib.sha256(json.dumps(RS.catalogue()).encode()).hexdigest(),
            "excluded_sources": ["sycophancy (binding pool)", "wikitext (binding pool)"],
        },
        "bank.root_sampler_seed": RS.seed_for("root_sampler"),
        "bank.blocks": {"seed": seed("blocks"), "blocks": blocks},
        "bank.component_graph_hash": graph_hash,
        "bank.calibration_components": graph["calibration"],
        "bank.folds.outer_seed": seed("folds:outer"),
        "bank.folds.inner_seed": seed("folds:inner"),
        "model.dictionary": {
            "bank_roots": roots["identity_digest"],
            "e2_library": E2.library_digest(),
            "rule": "one coordinate per dose-free bank root and E2 library action (dose = coefficient)",
        },
        "model.wm.config": cfg,
        "model.static.configs": {
            "S1": "F3u least squares (a, q) per objective on training h <= 2",
            "S2": dict(cfg, kind="StaticPredictor", n_seeds=5),
            "S3": dict(cfg, kind="MatchedStatic (StatNN-aux)", static_state=True, n_seeds=5),
            "tie_rule_relative": 0.005,
            "order": ["S1", "S2", "S3"],
        },
        "model.chart_schema_version": 2,
        "model.training_protocol": {
            "train_max_h": 2,
            "inner_folds": 4,
            "outer_folds": 5,
            "shrink_grid": list(M.SHRINK_GRID),
            "epochs": cfg["epochs"],
            "optimiser": "AdamW",
        },
        "model.ensemble_seeds": [1000 + s for s in range(5)],
        "controls.inert": {
            "I1": H.I1_TENSORS,
            "compensated": H.COMPENSATED,
            "I2_factors": list(H.I2_FACTORS),
            "I2_rows": [248077, 248320],
            "n_inert": d["n_inert"],
            "verification": ph.get("inert"),
        },
        "planner.algorithm": "cem",
        "planner.config": dict(
            dataclasses.asdict(AD.PLANNER_CONFIG),
            psi=AD.SUPPORT_PSI,
            support="Mahalanobis excess over the training 95th percentile",
            dose_rule="z in [-1, 1] x max|dose|, snapped to the episode's registered doses",
        ),
        "e2.library": {"digest": E2.library_digest(), "spec": "wsp.wm.e2.library"},
        "e2.library.lora_seeds": [
            p["seed"]
            for p in E2.library()["lora_tv"][:: len(E2.LORA_BLOCKS) * len(E2.LORA_LAMBDAS)]
        ],
        "e2.episodes": {
            o: [{"index": e["index"], "family": e["family"], "seeds": e["seeds"]} for e in v]
            for o, v in episodes.items()
        },
        "e2.winners_manifest_hash": {
            "status": "deferred",
            "binds_at": "after all episodes (wm.e2.winners_manifest, "
            "written under data/wme/winners/wme-c1)",
        },
        "gen.config": GA.SPEC_V1,
        "gen.parsers": {
            "module": "src/wsp/wm/genaudit.py",
            "sha256": sha(ROOT / "src/wsp/wm/genaudit.py"),
        },
        "gen.sampled_seeds": GA.SPEC_V1["sampled_audit"]["seeds"],
        "seeds.bootstrap.guard": seed("boot:guard"),
        "seeds.bootstrap.ht": seed("boot:ht"),
        "seeds.bootstrap.e1": seed("boot:e1"),
        "seeds.bootstrap.e2": seed("boot:e2"),
        "simulation.commit": commit,
        "simulation.seeds": {"grid": PV.SEED, "refit": 20261117},
        "simulation.report_hash": sha(a.sim),
        "cost.unit_times": {
            "source": "P-H100 summary",
            "sha256": sha(a.ph100),
            "unit": sim["inputs"]["unit"],
        },
        "sim": {
            "r_f": d["r_f"],
            "n_blk": d["n_blk"],
            "n_cal": d["n_cal"],
            "m_min_HT": sim["m_min_HT"],
            "n_split": sizes,
            "n_gen,syc": d["n_gen_syc"],
            "n_gen,gsm": d["n_gen_gsm"] if d["n_rep_sec"] else 0,
            "n_gen,math": d["n_gen_math"] if d["n_rep_sec"] else 0,
            "n_rep": d["n_rep"],
            "n_ep_sec": d["n_ep_sec"],
            "n_rep_sec": d["n_rep_sec"],
            "B": d["B"],
            "n_inert": d["n_inert"],
            "pool_size": d["pool_size"],
            "reader_token_cap": sim["reader"]["cap"],
            "shard_contrast": d["shard_contrast"],
            "shard_lora": d["shard_lora"],
        },
        "claims": {
            "run": sim["claims_run"],
            "unresolved_not_powered": sim["unresolved_not_powered"],
        },
        "supplement": harm,
        "outputs": {"runs": "data/wme/runs/wme-c1", "winners": "data/wme/winners/wme-c1"},
    }
    if a.refit:
        rf = json.loads(Path(a.refit).read_text())
        m["simulation.full_refit_hash"] = sha(a.refit)
        sel = rf["cells"]["selected_12_12_12"]
        failed = not (sel["null"]["pass"] and sel["alt"]["pass"])
        m["claims"]["full_refit"] = {
            "selected_cell": {
                k: {x: sel[k][x] for x in ("reps", "rate", "mean_ct", "c2s_n", "c2i_n", "pass")}
                for k in ("null", "alt")
            },
            "next_cell_16_12_12": {
                k: {x: rf["cells"]["next_16_12_12"][k][x] for x in ("reps", "rate", "c2i_n")}
                for k in ("null", "alt")
            },
            "rule": "ruling §4(b): full-refit gates fail where the cheap grid passed -> next larger cell; no cell "
            "passes -> unresolved - not powered",
        }
        if failed:
            m["claims"]["run"] = [c for c in m["claims"]["run"] if c != "ht"]
            m["claims"]["unresolved_not_powered"] = ["ht"] + [
                c for c in m["claims"]["unresolved_not_powered"] if c != "ht"
            ]
    OUT.write_text(json.dumps(m, indent=1, sort_keys=True, default=str))
    print(OUT, sha(OUT))


if __name__ == "__main__":
    main()
