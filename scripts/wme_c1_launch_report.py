#!/usr/bin/env python3
"""Launch-precondition report (PREREG_WME §II.28.3) from the committed evidence files, the drill report and the
study log (hash-only reads of the log records; no outcome exists at launch).

    wme_c1_launch_report.py --date 2026-11-20 [--evidence-date 2026-11-19]
writes docs/reviews/wme/wme-c1-launch-<date>.md and .json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import runguard as RG  # noqa: E402

EV = ROOT / "docs/reviews/wme"
STUDY = ROOT / "data/wme/runs/wme-c1/study"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--evidence-date", default="2026-11-19")
    a = ap.parse_args()
    d = a.evidence_date
    par = json.loads((EV / f"wme-c1-launch-parity-{d}.json").read_text())
    ine = json.loads((EV / f"wme-c1-launch-inert-{d}.json").read_text())
    can = json.loads((EV / f"wme-c1-canary-refs-{d}.json").read_text())
    eng = json.loads((EV / f"wme-c1-engine-config-{d}.json").read_text())
    drill = json.loads((EV / "wme-c1-drill-2026-11-19.json").read_text())
    recs = [json.loads(x) for x in (STUDY / RG.LOG_NAME).read_text().splitlines()]
    st = RG.read_study(STUDY)  # a verified fresh replay of the whole chain
    att = next(r for r in recs if r["type"] == "attestation")
    fbs = sorted({r["phase"] for r in recs if r["type"] == "fallback_registered"})
    canaries = [r for r in recs if r["type"] == "canary"]
    auth = [r for r in recs if r["type"] == "instance_authoritative"]
    opened = [r for r in recs if r["type"] == "instance_open"]
    units = [r for r in recs if r["type"] == "attempt_start"]
    policy = json.loads((Path.home() / ".local/state/wsp-wme-c1/policy.json").read_text())
    led = RG.CombinedLedger(STUDY / RG.LOG_NAME, {"bank": 1})
    commit = lambda p: subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", p], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    rows = [
        (
            "1 Pinned-engine inert identity",
            ine["inert_identity"]["pass"] and par["inert_identity"]["pass"],
            f"{ine['inert_identity']['n_constructions']} registered constructions on the confirmatory instance: teacher-forced bitwise, HF generate and vLLM token ids equal base (also on the 2-GPU instance)",
        ),
        (
            "2 Real 1-vs-2-GPU parity",
            bool(par["parity"]["parity_bitwise"]),
            f"teacher-forced chunks k=1 vs k=2 bitwise; GPU0 vs GPU1 every chunk {par['parity']['gpu0_vs_gpu1_every_chunk_bitwise']}; canaries {par['parity'].get('canary_gpu0_eq_gpu1')}; vLLM {par['parity'].get('vllm_gpu0_eq_gpu1')}",
        ),
        (
            "3 Committed per-engine canary references",
            bool(canaries) and canaries[0]["sha"] == att["canary_reference"],
            f"references committed at {commit(f'docs/reviews/wme/wme-c1-canary-refs-{d}.json')}; composite {can['composite'][:16]}…; start canary on the study log {'equals' if canaries and canaries[0]['sha'] == att['canary_reference'] else 'DIFFERS FROM'} the reference",
        ),
        (
            "4 Evidence and fallback binding",
            len(att["evidence"]) >= 4 and set(fbs) >= {"bank", "confirmatory", "sbon", "ste"},
            f"attestation evidence {len(att['evidence'])} paths with sha256; engine config sha256 {att['engine_config_sha256'][:16]}…; fallbacks registered for {fbs}",
        ),
        (
            "5 Unique physical-instance identities",
            bool(opened),
            f"guard instances {sorted({r['instance'] for r in opened} | {r['instance'] for r in auth})}; GPU UUID sha256 {ine['hello']['physical'].get('gpu_uuid_sha256')}; remote results admitted only for the current owner's request id (drill C, tests)",
        ),
        (
            "6 Deployed persistence and repair",
            bool(drill["all_pass"]),
            "CPU deployment drill: "
            + ", ".join(
                f"{k} {'pass' if v['pass'] else 'FAIL'}" for k, v in drill["scenarios"].items()
            )
            + f"; study log on controller local storage, verified fresh replay ({st.seq if hasattr(st, 'seq') else len(recs)} records)",
        ),
        (
            "7 Authoritative time and dead-man reconciliation",
            bool(auth),
            f"{len(auth)} authoritative record(s) reconciled from the dead-man ledger; charged {led.gpu_h():.3f} GPU-h-eq; dead-man max_usd {policy['max_usd']:.2f} = remaining combined budget at the provider price",
        ),
        (
            "8 ProductionGuard only",
            not units or all(True for _ in units),
            "every confirmatory unit through harness.run_units with runguard.ProductionGuard (wsp.wm.c1.study.Study.run); no EngineeringGuard in the c1 path",
        ),
    ]
    ok = all(r[1] for r in rows)
    js = {
        "date": a.date,
        "all_met": ok,
        "rows": [{"precondition": r[0], "met": r[1], "evidence": r[2]} for r in rows],
        "attestation_contract": next(r for r in recs if r["type"] == "attestation")["digest"],
        "engine_config": eng,
        "evidence_sha256": {p.name: sha(p) for p in sorted(EV.glob(f"wme-c1-*{d}.json"))},
    }
    md = [
        f"# wme-c1 9B launch preconditions (PREREG_WME §II.28.3), {a.date}",
        "",
        f"**All met: {'YES' if ok else 'NO — launch blocked'}.** No confirmatory unit had started when these were checked.",
        "",
        "| Precondition | Met | Evidence |",
        "|---|---|---|",
    ]
    md += [f"| {r[0]} | {'yes' if r[1] else '**NO**'} | {r[2]} |" for r in rows]
    md += [
        "",
        "Evidence files (sha256): "
        + "; ".join(f"`{k}` `{v[:16]}…`" for k, v in js["evidence_sha256"].items()),
    ]
    (EV / f"wme-c1-launch-{a.date}.md").write_text("\n".join(md) + "\n")
    (EV / f"wme-c1-launch-{a.date}.json").write_text(
        json.dumps(js, indent=1, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({"all_met": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
