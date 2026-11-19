#!/usr/bin/env python3
"""Overlay PREREG_WME §II.27 (Board A freeze re-check 2: stale guard objects fenced against the single study log;
property test with retained objects) on docs/prereg/wme_manifest.json and regenerate the table.

    python scripts/wme_manifest_prereg_v2i.py \
        --property docs/reviews/wme/wme-v2i-runguard-property-2026-11-19.json \
        --mutations docs/reviews/wme/wme-v2i-property-mutations-2026-11-19.json
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
BOARD = "docs/reviews/council/board-a-prereg-wme-v2-freeze2-2026-11-19.md"
TESTS = (
    "test_probe_freeze2_stale_guard_cannot_return_a_superseded_result",
    "test_probe_freeze2_stale_same_id_guard_cannot_accept_a_post_loss_canary",
    "test_probe_freeze2_stale_guard_cannot_run_a_fourth_attempt_after_exhaustion",
    "test_probe_freeze2_overlapping_objects_never_duplicate_sequence_numbers",
    "test_unfenced_views_always_act_on_the_current_log",
    "test_lock_serialises_concurrent_processes",
    "test_torn_or_truncated_log_fails_closed_until_reconciled",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict, prop: dict, prop_path: Path, mut_path: Path) -> dict:
    rg = dict(m["harness.runguard"])
    rg.update(
        persistence="one append-only, fsync'd, hash-chained study log per state directory (study.jsonl; global "
        "gap-free sequence; each record carries the previous record's sha256; study.jsonl.head holds the last "
        "sequence and hash); ledger, clock, retry and relaunch counters, canaries, contract generations, lifecycle, "
        "loss dispositions, fallbacks and the owner epoch are all derived by replaying it (StudyState); no other "
        "state files (the pre-§II.27 events/ledger/retry layout is refused; CanaryBook removed)",
        serialisation="every public operation takes an exclusive fcntl.flock on study.jsonl.lock, re-reads and "
        "verifies the tail since its cached sequence, re-validates (owner epoch, contract generation, clock, "
        "lifecycle, cap, retries, in-flight attempt) and appends under the same lock",
        owner_epoch="constructing a ProductionGuard appends owner_epoch (next epoch); every operation of an object "
        "with an older epoch or contract generation raises StaleGuard before reading out or writing anything; "
        "succeeded/failed must match the owner's own in-flight attempt_start",
        log_integrity="a partial last line, invalid record, broken hash chain, log shorter than its head or a changed "
        "already-read record raises LogCorrupt on every operation; repair_torn_tail (partial last line only) "
        "quarantines the torn bytes, cuts back to the last verified record and appends log_repaired, ending the "
        "ownership; any other damage needs audited manual restoration",
        result_identity="attempt_ok records the payload sha256; resolve verifies the stored payload against it "
        "(ResultMismatch); begin refuses a unit with an eligible result or fallback (UnitResolved)",
        regression_tests=list(rg.get("regression_tests", []))
        + [f"tests/test_wme_runguard.py::{t}" for t in TESTS],
        property_test={
            "test": "tests/test_wme_runguard_property.py (seeded generator; hypothesis not installed)",
            "invariants": [
                "I0 gap-free unique sequence, hash chain, head",
                "I1 charged time",
                "I2 no unit under a stale owner, without instance, halted, above cap, beyond 3 attempts, or after "
                + "resolution",
                "I3 eligibility irreversibility",
                "I4 canary brackets and payload digest",
                "I5 returned results equal the model; stability; fallback digest",
                "I6 replay and model agreement",
                "stale objects refused without writing",
            ],
            "oracle": "independent replay of the raw log bytes (Model), sharing no runguard code",
            "retained_objects": "every constructed guard kept in a pool and reused at random",
            "sequences": prop["sequences"],
            "steps_per_sequence": prop["steps"],
            "temporary_storage": prop.get("temporary_storage"),
            "report": str(prop_path.relative_to(ROOT)),
            "report_sha256": sha(prop_path),
            "mutations_report": str(mut_path.relative_to(ROOT)),
            "mutations_report_sha256": sha(mut_path),
        },
    )
    m["harness.runguard"] = rg
    m["prereg_v2"]["rulings_sha256"][BOARD] = sha(ROOT / BOARD)
    m["prereg_v2"]["v2i"] = {
        "status": "Board A freeze re-check 2 root cause (stale objects not fenced against shared persisted state) "
        "repaired structurally: single hash-chained log, OS lock with re-read and re-validation, owner epoch, "
        "torn/truncated logs fail closed; property test with retained objects",
    }
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True)
    ap.add_argument("--mutations", required=True)
    a = ap.parse_args()
    pp, mp_ = (
        ROOT / x if not Path(x).is_absolute() else Path(x) for x in (a.property, a.mutations)
    )
    m = build(json.loads(MAN.read_text()), json.loads(pp.read_text()), pp, mp_)
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
