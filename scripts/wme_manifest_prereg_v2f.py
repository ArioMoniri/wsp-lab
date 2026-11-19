#!/usr/bin/env python3
"""Overlay PREREG_WME §II.24 (Board A confirmation blockers A and B) on docs/prereg/wme_manifest.json and regenerate
the table.

    python scripts/wme_manifest_prereg_v2f.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import runguard as RG

MAN = ROOT / "docs/prereg/wme_manifest.json"
LIN = ROOT / "docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json"
BOARD = "docs/reviews/council/board-a-prereg-wme-v2-confirm-2026-11-19.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict, lin: dict) -> dict:
    c, k = lin["counts"], lin["checks"]
    m["audit.panels_v2e"]["root_lineage"] = {
        "report": str(LIN.relative_to(ROOT)),
        "sha256": sha(LIN),
        "mapping_sha256": lin["mapping_sha256"],
        "clean": lin["clean"],
        "counts": c,
        "checks": k,
    }
    m["bank.closure_allocation"]["root_shards_in_c1_construction"] = {
        "rule": "group reservation with representative deduplication (PREREG §II.24.1)",
        "original_groups": c["original_groups"],
        "member_items": c["original_items"],
        "final_union_groups": c["final_union_groups"],
        "written_representatives": c["written_representatives"],
        "in_pool_non_representative_members": c["items_as_member"],
        "members_outside_every_panel_source_pool": c["items_out_of_c1_pools"],
        "lineage_mapping_sha256": lin["mapping_sha256"],
    }
    m["exposures.v2"]["construction_only_exception"] = {
        "scope": "wme-c1 construction split only, for the bank-root shard groups committed before the panels",
        "written_construction_items_also_in_later_exploratory_exposures": m["audit.panels_v2e"][
            "root_shards"
        ]["construction_items_also_in_later_exposures"],
        "root_member_hashes_in_later_exposures": c["items_in_later_exposures"],
        "evaluation_exclusions": "absolute: zero exposure, root-item or root-group intersection with dev, bank, "
        "sealed_test or sealed_gen of wme-c1, or any split of wme-c2-gen",
    }
    m["harness.runguard"] = dict(
        m["harness.runguard"],
        module="wsp.wm.runguard.ProductionGuard: the mandatory guard of harness.run_units (keyword-only; an explicit "
        "EngineeringGuard for non-confirmatory runs)",
        launch_attestation="LaunchAttestation: inert identity bitwise, 1-vs-2-GPU parity bitwise, canary reference "
        "sha256, engine-config sha256, evidence; checked at construction",
        persistence="one append-only fsync'd event log + persistent RetryPolicy and CanaryBook, replayed on every "
        "construction (process and instance restarts)",
        eligibility="each attempt stored as <unit>.a<n>; resolve() returns the first canary-eligible attempt in event "
        "order; retained-ineligible attempts kept; pending attempts not complete",
        accounting="failed attempts charged; absolute cap on max(measured unit GPU time, authoritative instance time)",
        unit_attempts_total_across_relaunches=RG.UNIT_ATTEMPTS,
        phase_relaunches=RG.PHASE_RELAUNCHES,
        orchestration_test="tests/test_wme_runguard.py::"
        "test_orchestration_failure_canary_restart_exhaustion_and_first_valid_selection",
    )
    m["prereg_v2"]["rulings_sha256"][BOARD] = sha(ROOT / BOARD)
    m["prereg_v2"]["v2f"] = {
        "status": "Board A confirmation blockers A (root lineage) and B (runtime contract) repaired (technical only)",
    }
    return m


def main():
    m = json.loads(MAN.read_text())
    m = build(m, json.loads(LIN.read_text()))
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
