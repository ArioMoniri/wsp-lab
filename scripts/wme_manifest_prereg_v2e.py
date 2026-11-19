#!/usr/bin/env python3
"""Overlay the post-W2 state (PREREG_WME §II.23) on docs/prereg/wme_manifest.json and regenerate the table.

    python scripts/wme_manifest_prereg_v2e.py

Inputs: the merged stage-W2 report and panel manifest, RULING-wme-post-stagew Q4, RULING-wme-e2-scope, the built
panels' manifests (hash lists only) and the hash-only audit docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import prereg_v2 as P

MAN = ROOT / "docs/prereg/wme_manifest.json"
AUDIT = ROOT / "docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json"
W2 = ROOT / "docs/reviews/wme/wme-stagew2-2026-11-18.json"
W2_PANEL = ROOT / "data/wme/panels/wme-w2/manifest.json"
C1 = ROOT / "data/wme/panels/wme-c1/manifest.json"
C2 = ROOT / "data/wme/panels/wme-c2-gen/manifest.json"
RULINGS = [
    "docs/council/2026-11-18/RULING-wme-stagew2.md",
    "docs/council/2026-11-19/RULING-wme-e2-scope.md",
    "docs/reviews/wme/wme-stagew2-2026-11-18.md",
]
NEGATIVE_A = (
    "On Qwen3.5-9B, edit-specific trajectory prediction did not beat path-type averages: stage W2's nested world "
    "model (pooled same-path-type curve + edit-informed residual, shrunk toward 0) did not improve on the pooled curve "
    "(zero-shot mean d 0.003, Holm-adjusted lower bound -0.053; few-shot k = 2 d 0.000; nulls <= 0.011), and stage W's "
    "un-nested world model lost to the pooled curve in every setting. Reported as a negative result for (a) on this "
    "model under these designs; not evidence that no trajectory model could work; no further 9B spending on (a)."
)
E2_WORDING = (
    "the WM-guided minus StatNN-guided paired differences are not sign-symmetric about zero, in the direction "
    "favouring WM, conditional on these test items and this edit library"
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict, audit: dict, w2: dict) -> dict:
    pv = m["prereg_v2"]
    for r in RULINGS:
        pv["rulings_sha256"][r] = sha(ROOT / r)
    p0, p2 = w2["primary"]["0"], w2["primary"]["2"]
    w2p = json.loads(W2_PANEL.read_text())
    m["exposures.v2"]["wme-w2"] = {
        "status": "merged (main 718c48f); registered in wm.exposure.EXPOSURES",
        "manifest_sha256": sha(W2_PANEL),
        "content_hash": w2p["content_hash"],
        "outcome": {
            "verdict": w2["verdict"],
            "zero_shot": {"d_mean": p0["d_mean"], "holm_lb": p0["lb_holm"], "p": p0["p"]},
            "few_shot_k2": {"d_mean": p2["d_mean"], "holm_lb": p2["lb_holm"], "p": p2["p"]},
            "null_sizes": {k: v["size"] for k, v in w2["nulls"].items()},
            "complete_paths": w2["n_paths"],
            "report_json_sha256": sha(W2),
        },
        "ht3_decision": "not registered (W2 fail; RULING-wme-post-stagew Q4)",
    }
    pv["claims"]["H-T3"] = {
        "type": "not registered (stage W2 failed its pre-declared rule; RULING-wme-post-stagew Q4)",
        "planning_gpu_h": None,
        "planning_usd": None,
    }
    pv["claims"]["H-E2"]["supported_wording"] = E2_WORDING
    pv["claims"]["H-E2"]["scope_ruling"] = "docs/council/2026-11-19/RULING-wme-e2-scope.md"
    pv["claims"]["D-HT2"]["type"] = (
        "descriptive (stage W, stage W2); (a) reported as a negative result"
    )
    pv["negative_result_a"] = NEGATIVE_A
    reg = pv["pricing"]["v2c"]["registered_total"]
    pv["pricing"]["v2e"] = {
        "registered_total_gpu_h": reg["total_gpu_h"],
        "registered_total_usd_9b": reg["usd_9b"],
        "parts_gpu_h": reg["parts_gpu_h"],
        "ht3": "not registered; removed from every total",
        "remaining_under_cap_gpu_h": P.CAP_TOTAL_GPU_H - reg["total_gpu_h"],
    }
    for k in ("with_ht3_at_v2b_price", "p1_consistent_with_ht3"):
        pv["pricing"][k] = dict(pv["pricing"][k], status="withdrawn: H-T3 not registered (W2 fail)")
    pv["pricing"]["v2c"]["with_ht3"] = dict(
        pv["pricing"]["v2c"]["with_ht3"], status="withdrawn: H-T3 not registered (W2 fail)"
    )
    # panels
    c1, c2 = json.loads(C1.read_text()), json.loads(C2.read_text())
    m["panel.wme_c1.split_hashes"] = {
        "manifest_sha256": sha(C1),
        "content_hash": c1["content_hash"],
        "files": c1["split_hashes"],
        "built_n": {o: {s: v["n"] for s, v in ss.items()} for o, ss in c1["objectives"].items()},
        "builder": "wsp.wm.panels_v2d.build_c1_panel_v2e (scripts/wme_v2e_build_panels.py)",
    }
    m["panel.wme_c2_gen"] = dict(
        m["panel.wme_c2_gen"],
        split_hashes={
            "manifest_sha256": sha(C2),
            "content_hash": c2["content_hash"],
            "files": c2["split_hashes"],
        },
        built_n={p: {s: v["n"] for s, v in ss.items()} for p, ss in c2["objectives"].items()},
        shortfall=c2["shortfall"],
        builder="wsp.wm.panels_v2d.build_c2gen_panel (families joint, union policy)",
    )
    s = P.TE_SIZING["mmlupro_gen"]
    n_mm = c2["shortfall"]["mmlupro_gen"]["frozen_n"]
    pv["claims"]["S-TE"] = dict(
        pv["claims"]["S-TE"],
        n={"mmlupro_gen": n_mm, "supergpqa_gen": c2["shortfall"]["supergpqa_gen"]["frozen_n"]},
        power_restated={
            "mmlupro_gen": {
                "registered_n": s["n"],
                "frozen_n": n_mm,
                "analytic_power_holm_first_0.0125": P.te_power(
                    n_mm, s["delta"], s["discordant"], s["n_probe"]
                ),
                "analytic_power_holm_second_0.025": P.te_power(
                    n_mm, s["delta"], s["discordant"], s["n_probe"], alpha=0.025
                ),
                "rule": "RULING-wme-prereg-v2-issues item 5 / SYNTHESIS-r11 R1: N drops to the largest fresh "
                "count; power restated before any outcome (prereg_v2.te_power)",
            }
        },
    )
    al = audit["allocation"]
    m["exposures.v2"]["mmlu_allocation"] = dict(
        m["exposures.v2"]["mmlu_allocation"],
        sha256=al["sha256"],
        file_sha256=al["file_sha256"],
        summary={
            k: al[k]
            for k in (
                "pool_items",
                "groups",
                "excluded_groups",
                "fresh_groups",
                "preassigned_groups",
                "preassigned_exposed_groups",
                "reserve_groups",
                "consumers",
            )
        },
    )
    m["exposures.v2"]["order"] = (
        "wme-w2 (merged) -> wme-c1 (built) -> wme-c2-gen (built) -> wme-c27 (later registration)"
    )
    m["exposures.v2"]["registered"] = sorted(
        set(m["exposures.v2"]["registered"]) | {"wme-w2", "wme-c1", "wme-c2-gen"}
    )
    m["audit.panels_v2e"] = {
        "report": str(AUDIT.relative_to(ROOT)),
        "sha256": sha(AUDIT),
        "clean": audit["clean"],
        "cross_panel_pairs": audit["cross_panel"]["n_pairs"],
        "cross_panel_overlaps": len(audit["cross_panel"]["pairs_with_overlap"]),
        "exposure_hits_outside_root_construction": audit["exposure_hits_outside_root_construction"],
        "root_shards": audit["root_shards"],
    }
    m["bank.closure_allocation"] = {
        "component_graph_hash": m["bank.component_graph_hash"],
        "calibration_components_digest": hashlib.sha256(
            json.dumps(m["bank.calibration_components"], sort_keys=True).encode()
        ).hexdigest(),
        "folds": {
            "outer_seed": m["bank.folds.outer_seed"],
            "inner_seed": m["bank.folds.inner_seed"],
        },
        "roots_identity_digest": m["bank.roots"]["identity_digest"],
        "roots_file_sha256": audit["root_shards"]["roots_file_sha256"],
        "root_shards_in_c1_construction": {
            "groups": audit["root_shards"]["groups"],
            "items": audit["root_shards"]["items"],
        },
    }
    pv["v2e"] = {
        "status": "Board A v2 change 3 closed: W2 bound; H-T3 not registered; panels built; audits clean",
    }
    m["status"] = (
        "DRAFT for PREREG_WME v2 (v2b manifest + PREREG v2 overlays through v2e); panels built and audited; not frozen"
    )
    return m


def main():
    m = json.loads(MAN.read_text())
    m = build(m, json.loads(AUDIT.read_text()), json.loads(W2.read_text()))
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
