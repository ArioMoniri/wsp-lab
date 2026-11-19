#!/usr/bin/env python3
"""Overlay PREREG_WME §II.25 (Board A final re-check, blocker B gaps 1-4) on docs/prereg/wme_manifest.json and
regenerate the table.

    python scripts/wme_manifest_prereg_v2g.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "docs/prereg/wme_manifest.json"
BOARD = "docs/reviews/council/board-a-prereg-wme-v2-final-2026-11-19.md"
TESTS = (
    "test_probe_gap1_no_unit_runs_without_an_open_instance",
    "test_probe_gap1_restart_and_reopen_never_reset_instance_time",
    "test_probe_gap2_changed_engine_config_cannot_return_old_payload",
    "test_probe_gap3_lost_end_canary_allows_the_remaining_attempts",
    "test_probe_gap4_changed_fallback_value_is_refused_on_resume",
    "test_orchestration_failure_canary_restart_preemption_exhaustion_and_first_valid_selection",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict) -> dict:
    m["harness.runguard"] = dict(
        m["harness.runguard"],
        instance_lifecycle="units only on a registered active instance (InstanceNotOpen); duplicate opening "
        "idempotent, earliest opening kept; closed ids never reopened; authoritative records (provider / dead-man) "
        "override local times via reconcile_instance",
        contract_binding="sha256 of the full attestation persisted; a different contract on resume raises "
        "ContractMismatch unless a persisted technical ContractTransition is recorded; results and canaries count "
        "only under their own contract (earlier results become superseded_contract, never eligible)",
        lost_instances="declare_lost: persisted technical, outcome-blind disposition; authoritative close at "
        "termination (charged); unbracketed attempts retained-ineligible; no fabricated canary; attempt counts kept",
        fallback_identity="registered fallback payload and sha256 persisted at registration and with every fallback; "
        "a different fallback on resume raises FallbackMismatch",
        regression_tests=[f"tests/test_wme_runguard.py::{t}" for t in TESTS],
    )
    m["harness.runguard"].pop("orchestration_test", None)
    m["prereg_v2"]["rulings_sha256"][BOARD] = sha(ROOT / BOARD)
    m["prereg_v2"]["v2g"] = {
        "status": "Board A final re-check blocker B gaps 1-4 repaired (technical only); blocker A closed by Board A",
    }
    return m


def main():
    m = build(json.loads(MAN.read_text()))
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
