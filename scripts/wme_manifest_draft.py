#!/usr/bin/env python3
"""Executable manifest DRAFT for PREREG_WME v1: every frozen identity, grid and seed the confirmatory code uses,
collected from the module constants (never retyped), with sha256 of every referenced artefact.

Output: docs/prereg/wme_manifest_draft.json. Status DRAFT: not frozen; the freeze commit hashes the final file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import bank as B
from wsp.wm import confirm_stats as CS
from wsp.wm import controls as C
from wsp.wm import exposure as X
from wsp.wm import genaudit as G
from wsp.wm import harness as H
from wsp.wm import hf
from wsp.wm import model as M
from wsp.wm import p2design as G2
from wsp.wm import panel2 as P2
from wsp.wm import panel_c1 as C1
from wsp.wm import plan as PL
from wsp.wm import planner as PN
from wsp.wm import powersim as PS
from wsp.wm import uncertainty as U
from wsp.wm import wmclosure as W


def sha(path):
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main():
    rec = B.bank_recipe()
    planner_res = json.loads(
        (ROOT / "docs/reviews/wme/wme-planner-synthetic-2026-11-17.json").read_text()
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()
    man = {
        "status": "DRAFT for PREREG_WME v1 - not frozen; no confirmatory outcome exists",
        "generated_from_commit": git,
        "target": {
            "model": "Qwen/Qwen3.5-9B",
            "revision": hf.TARGETS["Qwen/Qwen3.5-9B"],
            "dtype": "bf16",
            "enable_thinking": False,
            "frozen_tensors": ["model.visual.*", "mtp.*"],
        },
        "panel": {
            "id": "wme-c1",
            "salt": C1.SALT,
            "assignment_order": list(C1.SPLITS),
            "split_roles": {k: list(v) for k, v in X.SPLIT_ROLES.items()},
            "separation": X.SEPARATION,
            "sizes": X.DEFAULT_SIZES,
            "sizes_status": "proposed; feasible per the exposure audit; final Ns from the power simulation",
            "grouping": "one group per source question/problem/document/prompt (wm.exposure.group_text); one item per group",
            "near_duplicates": {
                "shingle_words": X.NEAR_DUP_SHINGLE,
                "jaccard": X.NEAR_DUP_JACCARD,
                "minhash_perms": X.MINHASH_PERMS,
                "bands": X.MINHASH_BANDS,
                "seed": X.MINHASH_SEED,
            },
            "exposures": {
                n: dict(v, manifest_sha256=sha(v["manifest"])) for n, v in X.EXPOSURES.items()
            },
            "exclusion": "group-level, hash lists from manifests only",
            "reserve_replication_set": {
                "panel": "wme-p2",
                "file": "sealed_test/test.jsonl",
                "sha256_prefix": "49813882",
                "status": "unread; optional separately registered replication",
            },
            "sealed_reader": "wm.panel_c1.SealedTestReader (one read, frozen winners, manifest hash, O_NOFOLLOW)",
            "exposure_audit": {
                "file": "docs/reviews/wme/wme-c1-exposure-audit-2026-11-17.json",
                "sha256": sha("docs/reviews/wme/wme-c1-exposure-audit-2026-11-17.json"),
            },
        },
        "objectives": {
            "primary_e1": "sycophancy_real",
            "secondary_e1": ["gsm8k_gold_lp", "mathcomp_gold_lp"],
            "trajectory_endpoints": list(P2.T2_OBJECTIVES),
            "orientation": P2.OBJECTIVES,
        },
        "guardrails": {
            "rules": {g: list(v) for g, v in G2.GUARDS.items()},
            "scale": "per-item pp_value before pairing (panel2.pp_value)",
            "simultaneous": "Bonferroni over the guard set",
            "z_95": G2.Z95,
        },
        "bank": {
            "namespace": B.NAMESPACE,
            "recipe_hash": rec["recipe_hash"],
            "recipe_file": {
                "path": "docs/prereg/wme_bank_recipe_draft.json",
                "sha256": sha("docs/prereg/wme_bank_recipe_draft.json"),
            },
            "families": list(B.FAMILIES),
            "roots_per_family": B.ROOTS_PER_FAMILY,
            "support_layers": list(B.SUPPORT_LAYERS),
            "nominals": B.NOMINALS,
            "ladder": list(B.LADDER),
            "ladder_sides": rec["ladder_sides"],
            "stop_rule": rec["stop_rule"],
            "duplicate_rule": rec["duplicate_rule"],
            "composition": rec["composition"],
            "horizon_rule": rec["horizon_rule"],
            "construction_allocation": rec["construction_allocation"],
            "pilot_overlap": rec["pilot_overlap"],
            "n_states_max": rec["n_states_max"],
            "seeds": {k: B.seed_for(k) for k in ("roots", "row_sets", "compositions")},
        },
        "closure": {
            "join_keys": W.JOIN_KEYS,
            "multi_valued": list(W.MULTI_VALUED),
            "registered_policy": "source",
            "sensitivity_policy": "concept",
            "n_components": {p: rec["components"][p]["n_components"] for p in rec["components"]},
            "size_distribution_states": {
                p: rec["components"][p]["size_distribution"] for p in rec["components"]
            },
            "outer_folds": W.N_OUTER_FOLDS,
            "fold_seed": "wme-bank:folds",
            "inner": W.INNER,
            "inner_seed": "wme-bank:inner:<fold>",
            "fold_report": {p: rec["components"][p]["folds"] for p in rec["components"]},
            "direction_similarity_max_cos": 0.5,
        },
        "predictors": {
            "wm_config": dataclasses.asdict(M.WMConfig()),
            "n_members": 5,
            "shrink_grid": list(M.SHRINK_GRID),
            "selection": "component-level, 20% (wm.uncertainty.component_selection_split, seed 20261116)",
            "bootstrap": "component bootstrap per member (seed 20261117)",
            "static_comparators": [
                "MatchedStatic (registered H-T comparator family)",
                "StaticPredictor (StatNN)",
                "refit_f3u (h <= 2 only)",
                "FieldOnly (descriptive)",
                "AnchorShift (planning)",
            ],
            "aux_ablation": "obs_weight = 0 for both arms",
            "train_max_h": B.TRAIN_MAX_H,
        },
        "uncertainty": {
            "target_coverage": U.TARGET_COVERAGE,
            "tolerance_pp": U.COVERAGE_TOL_PP,
            "calibration_max_h": U.CAL_MAX_HORIZON,
            "check_horizons": list(U.CHECK_HORIZONS),
            "floor": "field-residual RMS on the selection split",
            "failure": "ranks only for every arm (planner.penalty_mode)",
        },
        "statistics": {
            "ct_min_clusters": CS.MIN_CLUSTERS,
            "n_boot": CS.N_BOOT,
            "ht_alpha_one_sided": CS.ALPHA_ONE_SIDED,
            "ht_weights": "pooled (component weights as sensitivity)",
            "e1_e2": "two-sided 95% (2.5th percentile LB)",
        },
        "controls": {
            "fixed_endpoint": {
                "horizons": list(C.FIXED_ENDPOINT_HORIZONS),
                "n_alternates": C.N_ALTERNATES,
                "seed": C.CONTROL_SEED,
                "pass_rule": "static invariant and LB(C_T on alternates) > 0",
                "gates": ["H-T", "E2 via WM trajectory model"],
            },
            "corruption_diagnostic": "descriptive only",
            "inert": C.INERT_SPEC,
            "trust_region": {
                "dnll": C.TRUST_REGION_DNLL,
                "nondestructive": C.NONDESTRUCTIVE_DNLL,
                "min_components": C.MIN_STRATUM_COMPONENTS,
                "assignment": "mean paired WikiText ΔNLL on validation items",
            },
            "best_of_n": "two-stage: screen_items/full_items bundle units; survivors gain > 0 and ΔNLL <= 0.01",
            "cost_categories": list(C.COST_CATEGORIES),
        },
        "budget": {
            "B": 32,
            "B_status": "set by the power simulation",
            "bundle_definition": PL.BUNDLE_DEFINITION,
            "kinds": PL.BUNDLE_KINDS,
        },
        "planner": {
            "config": dataclasses.asdict(PN.PlannerConfig()),
            "inner_optimiser": planner_res["decision"]["use"],
            "synthetic_gate": {
                "file": "docs/reviews/wme/wme-planner-synthetic-2026-11-17.json",
                "sha256": sha("docs/reviews/wme/wme-planner-synthetic-2026-11-17.json"),
                "decision": planner_res["decision"],
                "mean_regret": {m: v["mean_regret"] for m, v in planner_res["methods"].items()},
            },
            "incumbents_at": list(PN.INCUMBENT_AT),
        },
        "generation_audit": G.SPEC,
        "harness": {
            "device_cap_bytes": H.DEVICE_CAP_BYTES,
            "unit_key": "sha256(recipe || theta0 || panel)",
        },
        "power_simulation": {
            "requirements": PS.REQUIREMENTS,
            "grid": PS.GRID,
            "scenarios": PS.SCENARIOS,
            "defaults": PS.DEFAULTS,
            "seed": PS.SEED,
            "n_boot": PS.N_BOOT,
            "budget_gpu_h": PS.BUDGET_GPU_H,
            "pricing": {
                "file": "docs/reviews/wme/wme-powersim-pricing-2026-11-17.json",
                "sha256": sha("docs/reviews/wme/wme-powersim-pricing-2026-11-17.json"),
            },
            "status": "built and priced; full run awaits Board A approval",
        },
        "source_sha256": {
            p: sha(p)
            for p in sorted(str(q.relative_to(ROOT)) for q in (ROOT / "src/wsp/wm").glob("*.py"))
        },
    }
    data = json.dumps(man, indent=1, sort_keys=True, default=str)
    out = ROOT / "docs/prereg/wme_manifest_draft.json"
    out.write_text(data)
    print(out, hashlib.sha256(data.encode()).hexdigest())


if __name__ == "__main__":
    main()
