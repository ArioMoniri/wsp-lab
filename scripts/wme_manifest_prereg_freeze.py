#!/usr/bin/env python3
"""Overlay PREREG_WME §II.28 (the freeze: Board A freeze re-check 3 conditions, chair decisions, launch preconditions)
on docs/prereg/wme_manifest.json, mark it frozen and regenerate the table.

    python scripts/wme_manifest_prereg_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "docs/prereg/wme_manifest.json"
BOARD = "docs/reviews/council/board-a-prereg-wme-v2-freeze3-2026-11-19.md"
FROZEN = "FROZEN 19 Nov 2026"

# Chair decisions, verbatim (binding; §II.28.2)
CHAIR_1 = (
    "The 9B confirmatory run uses ONLY the supported serialized execution path: one controller host owning the "
    "single authoritative local study log, one serialized guard call stream (no shared-object threads), one active "
    'confirmatory GPU instance at a time; the "up to 4 concurrent instances" plan is withdrawn for this freeze. Any '
    "parallel adapter would require a dated amendment with atomic aggregate reservations and Board A check before use."
)
CHAIR_2 = (
    "Dead-man allowance for the single instance equals the remaining combined budget; no other confirmatory instance "
    "may run."
)
BOARD_A_CHAIN = [
    (
        "docs/reviews/council/board-a-prereg-wme-2026-11-17.md",
        "v1",
        "ACCEPT WITH CONDITIONS (14 changes)",
    ),
    (
        "docs/reviews/council/board-a-prereg-wme-v2-2026-11-18.md",
        "v2",
        "ACCEPT WITH CONDITIONS (changes 1-7)",
    ),
    (
        "docs/reviews/council/board-a-prereg-wme-v2-confirm-2026-11-19.md",
        "confirmation",
        "ACCEPT WITH CONDITIONS (blockers A, B)",
    ),
    (
        "docs/reviews/council/board-a-prereg-wme-v2-final-2026-11-19.md",
        "final re-check",
        "ACCEPT WITH CONDITIONS; freeze blocked by B (A closed)",
    ),
    (
        "docs/reviews/council/board-a-prereg-wme-v2-freeze-2026-11-19.md",
        "freeze re-check",
        "REJECT (B open)",
    ),
    (
        "docs/reviews/council/board-a-prereg-wme-v2-freeze2-2026-11-19.md",
        "freeze re-check 2",
        "REJECT (B open)",
    ),
    (
        BOARD,
        "freeze re-check 3",
        "ACCEPT WITH CONDITIONS (B closed for the serialized production path)",
    ),
]
LAUNCH_PRECONDITIONS = [
    (
        "pinned-engine inert identity: inert-edit identity and output checks, bitwise, on the actual pinned confirmatory "
        "engines and configuration, retained"
    ),
    "real 1-vs-2-GPU parity, bitwise, on the actual pinned confirmatory engines and configuration, retained",
    (
        "committed per-engine canary references; start, periodic and closing canaries under the registered eligibility "
        "rules"
    ),
    (
        "evidence and fallback binding: real evidence paths and digests, the engine configuration and the registered "
        "fallbacks bound in the LaunchAttestation and study log before the first confirmatory unit"
    ),
    (
        "unique physical-instance identities: a bound mapping of physical instances, phases, attempts, contract "
        "generations and canary references to controller records; remote outputs admitted only through the current "
        "owner, with stale completion refused; a controller takeover stops or accounts for old outstanding work"
    ),
    (
        "deployed persistence and repair: one authoritative study log on the controller host's local storage; deployed "
        "restart and replacement persistence, partial-write handling and retention of every charged attempt "
        "demonstrated; verified fresh replay on every restart and recovery; outcome-blind technical repair records; fail "
        "closed on loss of controller, state integrity, ownership or accounting connectivity, retaining termination and "
        "accounting evidence; deployment restart, takeover and worker loss demonstrated with CPU fixtures before any "
        "confirmatory outcome"
    ),
    (
        "authoritative time and dead-man reconciliation: provider or dead-man instance times reconciled into the "
        "controller ledger; all previously spent bank and study time and every active instance in the combined charge "
        "(setup, downloads, idle, canaries, failures, retries and replacements included); worker sub-ledgers are "
        "evidence, never cap authorities; the single instance's dead-man allowance equals the remaining combined budget"
    ),
    (
        "ProductionGuard only: every confirmatory unit runs through harness.run_units with runguard.ProductionGuard; "
        "EngineeringGuard results never enter confirmatory evidence"
    ),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(m: dict) -> dict:
    rg = dict(m["harness.runguard"])
    rg.update(
        serialisation="every public operation takes an exclusive fcntl.flock on study.jsonl.lock, re-reads and "
        "verifies the tail since its cached sequence, re-validates and appends under the same lock. Scope "
        "(§II.28.1): separate objects and separate processes are serialized by the OS lock; calls sharing one "
        "StudyLog object (one guard and its ledger and retry views) must themselves be serialized, including across "
        "threads (the object-wide re-entrancy depth is not a thread-aware lock); no shared-object concurrent entry",
        log_integrity="a cached refresh verifies the last cached record, the appended tail and the head; it raises "
        "LogCorrupt on a partial last line, an invalid appended record, a broken sequence or hash chain in the tail, "
        "a log shorter than already read or than its head, or a changed last cached record. Earlier cached records "
        "are verified only by a fresh replay (a new guard or read_study), which verifies the whole chain. Scope "
        "(§II.28.1): immutable-prefix storage is an operating assumption; a verified fresh replay is required on every "
        "restart and recovery; the hash chain and head are not protection against a coordinated rollback of the log "
        "and its head. repair_torn_tail (partial last line only) quarantines the torn bytes, cuts back to the last "
        "verified record and appends log_repaired, ending the ownership; any other damage needs audited manual "
        "restoration",
        supported_execution={
            "chair_decision_1": CHAIR_1,
            "chair_decision_2": CHAIR_2,
            "board_a": BOARD,
            "parallel_adapter": "not registered; requires a dated amendment with atomic aggregate reservations and "
            "a Board A check before use",
        },
        launch_preconditions=LAUNCH_PRECONDITIONS,
    )
    m["harness.runguard"] = rg
    caps = dict(m["caps"])
    caps["concurrency"] = (
        "one active confirmatory GPU instance at a time (chair decision 1, §II.28.2); supersedes §II.13's at most 4 "
        "concurrent instances"
    )
    caps["dead_man_allowance"] = (
        "the single instance's dead-man allowance equals the remaining combined budget "
        "(CombinedLedger.remaining_usd(); chair decision 2, §II.28.2)"
    )
    m["caps"] = caps
    pv = m["prereg_v2"]
    pv["rulings_sha256"][BOARD] = sha(ROOT / BOARD)
    pv["status"] = (
        f"{FROZEN}: PREREG_WME v2 (Part II through §II.28); registered claim set of RULING-wme-post-stagew Q3"
    )
    pv["freeze"] = {
        "date": "2026-11-19",
        "section": "PREREG_WME §II.28",
        "board_a_chain": [
            {"review": p, "stage": s, "verdict": v, "sha256": sha(ROOT / p)}
            for p, s, v in BOARD_A_CHAIN
        ],
        "chair_decisions": [CHAIR_1, CHAIR_2],
        "launch_preconditions_mandatory_before_first_confirmatory_unit": LAUNCH_PRECONDITIONS,
        "registered_total_gpu_h": pv["pricing"]["v2e"]["registered_total_gpu_h"],
        "registered_total_usd_9b": pv["pricing"]["v2e"]["registered_total_usd_9b"],
    }
    m["status"] = (
        f"{FROZEN}: PREREG_WME v2 manifest (v2b manifest + PREREG v2 overlays through §II.28); panels built and "
        "audited; Board A freeze re-check 3 ACCEPT WITH CONDITIONS, recorded in §II.28"
    )
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
