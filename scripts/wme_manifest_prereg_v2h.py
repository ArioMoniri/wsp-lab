#!/usr/bin/env python3
"""Overlay PREREG_WME §II.26 (Board A freeze re-check: clock, contract generations, closed/lost instances; property
test) on docs/prereg/wme_manifest.json and regenerate the table.

    python scripts/wme_manifest_prereg_v2h.py --property docs/reviews/wme/wme-v2h-runguard-property-2026-11-19.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "docs/prereg/wme_manifest.json"
BOARD = "docs/reviews/council/board-a-prereg-wme-v2-freeze-2026-11-19.md"
TESTS = (
    "test_probe_freeze_clock_rollback_cannot_credit_another_instance",
    "test_probe_freeze_contract_rollback_does_not_revive_old_results",
    "test_probe_freeze_canary_after_loss_cannot_revive_an_attempt",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict, prop: dict, prop_path: Path) -> dict:
    rg = dict(m["harness.runguard"])
    rg.update(
        clock="fail closed: every reading through CombinedLedger.tick (new maxima persisted); a backward reading "
        "persists clock_halt and raises ClockInconsistent; halted until authoritative termination of every instance "
        "active at the halt and a clock at or above the recorded maximum (clear_clock_halt); local durations are "
        "ticks (never negative); authoritative records are complete and can only add time (max(local, authoritative)); "
        "no clamping",
        contract_generations="each attestation/transition is a new generation; attempts and canaries bind to "
        "(digest, generation); a transition back to an earlier digest is a new generation and never reactivates "
        "earlier attempts or canaries",
        closed_and_lost_instances="canary() and every unit operation require an active, non-lost instance; closing, "
        "loss or authoritative termination irreversibly disposes of unbracketed attempts (attempt_disposed events); "
        "canaries after an instance's closure never count; run_units validates lifecycle and clock before resolve",
        regression_tests=list(rg.get("regression_tests", []))
        + [f"tests/test_wme_runguard.py::{t}" for t in TESTS],
        property_test={
            "test": "tests/test_wme_runguard_property.py (seeded generator; hypothesis not installed)",
            "invariants": [
                "I1 charged time",
                "I2 no unit without instance or above cap",
                "I3 eligibility irreversibility",
                "I4 canary brackets",
                "I5 resolved-result stability and fallback digest",
                "I6 log replay",
            ],
            "sequences": prop["sequences"],
            "steps_per_sequence": prop["steps"],
            "report": str(prop_path.relative_to(ROOT)),
            "report_sha256": sha(prop_path),
        },
    )
    m["harness.runguard"] = rg
    m["prereg_v2"]["rulings_sha256"][BOARD] = sha(ROOT / BOARD)
    m["prereg_v2"]["v2h"] = {
        "status": "Board A freeze re-check gaps (clock, contract generations, post-loss canary) repaired; adversarial "
        "property test added",
    }
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True)
    a = ap.parse_args()
    pp = ROOT / a.property if not Path(a.property).is_absolute() else Path(a.property)
    m = build(json.loads(MAN.read_text()), json.loads(pp.read_text()), pp)
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
