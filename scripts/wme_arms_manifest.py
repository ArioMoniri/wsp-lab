#!/usr/bin/env python3
"""Track-B manifest fragment for PREREG_WME (competitor arms, vocabulary-row field, 27B separation): every frozen
constant the track-B code uses, collected from the module constants (never retyped), with the sha256 of each source
file and referenced artefact. DRAFT: merged into ``docs/prereg/wme_manifest_draft.json`` by the manifest author.

Output: docs/prereg/wme_arms_manifest_fragment.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import sep27 as S27
from wsp.wm import vocabfield as VF
from wsp.wm.arms import direct as D
from wsp.wm.arms import llm as L
from wsp.wm.arms import pool as P
from wsp.wm.arms import readers as R
from wsp.wm.arms import spec as S

SOURCES = (
    "src/wsp/wm/arms/__init__.py",
    "src/wsp/wm/arms/spec.py",
    "src/wsp/wm/arms/pool.py",
    "src/wsp/wm/arms/llm.py",
    "src/wsp/wm/arms/readers.py",
    "src/wsp/wm/arms/direct.py",
    "src/wsp/wm/vocabfield.py",
    "src/wsp/wm/sep27.py",
    "src/wsp/wm/panel_c1.py",
)


def sha(path):
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()
    smoke = "docs/reviews/wme/wme-arms-smoke-2026-11-17.json"
    man = {
        "status": "DRAFT track-B fragment for PREREG_WME v1 - not frozen; no outcome exists",
        "generated_from_commit": git,
        "arms": {
            "registry": S.access_table(),
            "access_vocabulary": S.ACCESS,
            "forbidden": S.FORBIDDEN,
            "exposure": S.EXPOSURE,
            "winner_rule": "spec.select_winner: guard bounds mean + z se <= limit (z = 1.96, one-sided "
            "0.025) and dev LCB > 0; max LCB; ties smaller total relative Frobenius norm, then earlier",
            "z_guard": S.Z_GUARD,
            "arm_seed_rule": S.ARM_SEED_RULE,
            "store": "one store per (arm, episode); owner-only reads (spec.StoreRegistry)",
            "pool": {
                "rounds": P.ROUNDS,
                "max_len": P.MAX_LEN,
                "mutation_fraction": P.MUTATION_FRACTION,
                "n_incumbents": P.N_INCUMBENTS,
                "round_seed_rule": "int(sha256('wme-e2-pool:{episode_seed}:{round}')[:8], 16)",
                "pool_size": "[SIM: pool_size]",
                "failure_rule": P.FAILURE_RULE,
            },
            "readers": {
                "token_cap": S.READER_TOKEN_CAP,
                "reader_max_output_tokens": R.READER_MAX_OUTPUT,
                "role_max_output_tokens": R.ROLE_MAX_OUTPUT,
                "max_revisions": R.MAX_REVISIONS,
                "fugu_roles_access": S.FUGU_ROLES,
                "primary_contrast": R.PRIMARY_CONTRAST,
                "prompt_sha256": {
                    "wsr_system": h(R.WSR_SYSTEM),
                    "thinker_system": h(R.THINKER_SYSTEM),
                    "worker_system": h(R.WORKER_SYSTEM),
                    "verifier_system": h(R.VERIFIER_SYSTEM),
                    "task_text": h(R.ReaderTask().text()),
                },
                "output_contract": "exactly one JSON object (one surrounding code fence allowed); ranking = >= b "
                "distinct pool ids; nothing repaired, defaulted or guessed",
                "scripted_rule": R.ScriptedReader.__doc__ or "see wm.arms.readers module docstring",
            },
            "llm": {
                "providers": {
                    k: {
                        "class": v.__name__,
                        "declared_overhead_tokens": getattr(v, "default_overhead", 0),
                    }
                    for k, v in L.PROVIDERS.items()
                },
                "env_allow": list(L.ENV_ALLOW),
                "truncation_stops": sorted(L.TRUNCATION_STOPS),
                "truncation_rule": "length stop or output tokens >= limit is a failure",
                "smoke_record": {"path": smoke, "sha256": sha(smoke)},
            },
            "wm_greedy": {
                "n_incumbents": D.N_INCUMBENTS,
                "scorer": "wm.planner.Planner.score (shared planner scorer; mode lcb or ranks)",
                "extension": "one more distinct primitive at the end; length <= 4; base always extended",
            },
            "bon_steering": {
                "contrasts": list(D.STEER_CONTRASTS),
                "shifts_residual_norm_fraction": list(D.STEER_SHIFTS),
                "layers": "all (9B: 0-31)",
                "space_size_9b": len(D.steering_space(32)),
                "draw": "B distinct actions uniformly without replacement, seed arm_seed('bon_steering', e)",
            },
            "lora_sft": dict(D.LORA_SFT, seed_rule="arm_seed('lora_sft', episode_seed)"),
        },
        "field": {
            "shared_spec": VF.SHARED_FIELD_SPEC,
            "vocab_rows": {
                "schema": VF.SCHEMA,
                "approximations": VF.APPROXIMATIONS,
                "max_declared_rows": VF.MAX_DECLARED_ROWS,
                "declared_rows_rule": VF.DECLARED_ROWS_RULE,
                "planned_rows": VF.PLANNED_ROWS,
                "unsupported": "signed = q = None, supported = False, with a reason; never 0",
                "tied_embeddings": "refused",
            },
        },
        "c27": {
            "salt": S27.SALT,
            "bank_namespace": S27.BANK_NAMESPACE,
            "model": {"repo": S27.MODEL, "revision": S27.REVISION},
            "experiment": S27.EXPERIMENT,
            "required_exposures": {
                n: dict(S27.EXPOSURES_27[n], manifest_sha256=sha(S27.EXPOSURES_27[n]["manifest"]))
                for n in S27.REQUIRED_EXPOSURES
            },
            "calibration_timing_edits": list(S27.CALIB_TIMING_EDITS),
            "calibration_report_sha256": sha(S27.CALIB_REPORT),
            "root_joins": {
                "J3_max_cos": S27.J3_MAX_COS,
                "J4_row_jaccard": S27.J4_ROW_JACCARD,
                "rule": "J2 and J5 across models; J3 and J4 within the same model; namespace wme-bank27",
            },
            "primary_selection": {
                "min_qualifying": S27.PRIMARY_MIN_QUALIFYING,
                "tie_order": list(S27.PRIMARY_TIE_ORDER),
                "inputs": {"model": S27.MODEL, "panel": S27.SALT, "split": "dev"},
                "relation": S27.RELATION,
            },
            "tuning_guard": {
                "design_elements": list(S27.DESIGN_ELEMENTS),
                "source_kinds": S27.SOURCE_KINDS,
                "labels": S27.LABELS,
                "forbidden_read_roots": list(S27.FORBIDDEN_READ_ROOTS),
            },
            "field": {
                "layers": [24, 25, 26, 27],
                "layers_per_group": S27.FIELD_LAYERS_PER_GROUP,
                "max_item_tokens": S27.FIELD_MAX_ITEM_TOKENS,
                "go_m": S27.GO_M,
            },
            "launch_gate_amendment_fields": list(S27.AMENDMENT_FIELDS),
        },
        "source_sha256": {p: sha(p) for p in SOURCES},
    }
    out = ROOT / "docs/prereg/wme_arms_manifest_fragment.json"
    out.write_text(json.dumps(man, indent=1, sort_keys=True, default=list) + "\n")
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
