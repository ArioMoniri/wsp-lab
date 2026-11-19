#!/usr/bin/env python3
"""CPU deployment drill of the wme-c1 controller path (PREREG_WME §II.28.3 precondition 6; §II.28.1 operating
rules). Real code: ``wsp.wm.c1.ctl`` / ``study`` with ``ProductionGuard`` on a scratch study state directory, the
real worker loop in a separate process (fixture handlers, no model), ``LocalTransport``. Nothing confirmatory.

Scenarios (each checked against the raw study log): (A) bracketed units become eligible; (B) worker loss mid-unit
is a charged technical failure and the restarted worker's late result for the dead attempt is never admitted;
(C) controller kill mid-unit and takeover: fresh verified replay, new owner epoch, fence moved, the old in-flight
request is refused/aborted and the unit re-attempted; (D) a torn log tail fails closed until the audited repair;
(E) instance loss: declare_lost disposes unbracketed attempts, the registered relaunch continues on a new instance id;
(F) dead-man reconciliation adds authoritative time; (G) fail closed on a corrupted earlier record (fresh replay).

    wme_c1_drill.py --dir data/wme/runs/smoke/drill --out docs/reviews/wme/wme-c1-drill-2026-11-19.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import harness as H  # noqa: E402
from wsp.wm import runguard as RG  # noqa: E402
from wsp.wm.c1 import ctl as C  # noqa: E402
from wsp.wm.c1 import drillworker as DW  # noqa: E402
from wsp.wm.c1 import study as SD  # noqa: E402
from wsp.wm.c1 import transport as T  # noqa: E402

PANEL = "drill"
THETA0 = "d" * 64


def att():
    return RG.LaunchAttestation(
        True, True, C.composite_canary(DW.ENGINES), "e" * 64, ["drill fixture"]
    )


def study(d):
    s = SD.Study(d / "study", d / "runs", att(), THETA0, PANEL)
    s.priced = {"bank": 100.0, "confirmatory": 100.0, "sbon": 100.0, "ste": 100.0}
    return s


def start_worker(d):
    return subprocess.Popen(
        [sys.executable, "-m", "wsp.wm.c1.drillworker", "--root", str(d / "w")],
        stdout=subprocess.DEVNULL,
        stderr=open(d / "worker.log", "a"),
    )


def units(lo, hi, sleep=0.0, steps=1):
    return [
        (
            {"kind": "square", "args": {"x": i, "sleep": sleep, "steps": steps}, "label": f"u{i}"},
            f"u{i}",
        )
        for i in range(lo, hi)
    ]


def records(d):
    return [json.loads(x) for x in (d / "study" / RG.LOG_NAME).read_text().splitlines()]


def controller_child(d, inst, lo, hi, sleep, steps):
    s = study(d)
    tr = T.LocalTransport(d / "w")
    rem = C.Remote(tr, d / "ctl", poll_s=0.05, dead_after_s=5)
    g = s.guard("bank", inst)
    tr.set_fence(g.epoch, inst)
    g.open_instance(1)
    s.run(g, rem, tr, units(lo, hi, sleep, steps), chunk=50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", nargs="*")
    a = ap.parse_args()
    d = Path(a.dir)
    if a.child:
        inst, lo, hi, sleep, steps = a.child
        controller_child(d, inst, int(lo), int(hi), float(sleep), int(steps))
        return
    if d.exists():
        import shutil

        shutil.rmtree(d)
    d.mkdir(parents=True)
    rep = {
        "label": "wme-c1 CPU deployment drill (fixtures; engineering only; not confirmatory)",
        "scenarios": {},
    }
    wp = start_worker(d)
    s = study(d)
    tr = T.LocalTransport(d / "w")
    procs = {"p": wp}

    def revive():
        procs["p"] = start_worker(d)

    rem = C.Remote(tr, d / "ctl", poll_s=0.05, dead_after_s=5, revive=revive)
    inst1 = "lambda:drill-1"
    # A: bracketed units
    g = s.guard("bank", inst1)
    tr.set_fence(g.epoch, inst1)
    g.open_instance(1)
    out = s.run(g, rem, tr, units(0, 4), chunk=2)
    rep["scenarios"]["A_bracketed_eligible"] = {
        "pass": [out[f"u{i}"]["sq"] for i in range(4)] == [0, 1, 4, 9],
        "canaries": sum(r["type"] == "canary" for r in records(d)),
    }
    # B: worker loss mid-unit (killed while the request runs); controller sees WorkerDead -> charged technical
    #    failure; the worker is restarted; the retry (new id) is admitted, the dead attempt's late result is not
    g = s.guard("bank", inst1)
    tr.set_fence(g.epoch, inst1)

    def killer():
        time.sleep(1.0)
        wp.send_signal(signal.SIGKILL)
        os.utime(d / "w" / "alive", (time.time() - 60, time.time() - 60))

    import threading

    threading.Thread(target=killer, daemon=True).start()
    out = s.run(g, rem, tr, units(4, 5, sleep=0.5, steps=8), chunk=4)
    recs = records(d)
    u4 = H.unit_key(units(4, 5, 0.5, 8)[0][0], THETA0, PANEL)
    starts = [r for r in recs if r["type"] == "attempt_start" and r["unit"] == u4]
    fails = [r for r in recs if r["type"] == "attempt_fail" and r["unit"] == u4]
    oks = [r for r in recs if r["type"] == "attempt_ok" and r["unit"] == u4]
    revivals = [
        json.loads(x)
        for x in (d / "ctl" / "dispatch.jsonl").read_text().splitlines()
        if '"revive"' in x
    ]
    rep["scenarios"]["B_worker_loss"] = {
        "pass": out["u4"]["sq"] == 16 and len(revivals) == 1 and len(oks) == 1 and len(starts) == 1,
        "revivals": len(revivals),
        "attempts": len(starts),
        "failed_attempts_charged_s": [f["seconds"] for f in fails],
        "fail_causes": [f["cause"] for f in fails],
    }
    wp = procs["p"]
    # C: controller killed mid-unit; a new controller takes over (fresh replay, new epoch, fence)
    child = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "--dir",
            str(d),
            "--out",
            a.out,
            "--child",
            inst1,
            "5",
            "6",
            "0.5",
            "20",
        ],
        stderr=open(d / "child.log", "a"),
    )
    time.sleep(4.0)
    child.send_signal(signal.SIGKILL)
    child.wait()
    s2 = study(d)
    g2 = s2.guard("bank", inst1)
    tr.set_fence(g2.epoch, inst1)
    out = s2.run(g2, rem, tr, units(5, 6, 0.1, 2), chunk=4)
    recs = records(d)
    u5 = H.unit_key(units(5, 6, 0.1, 2)[0][0], THETA0, PANEL)
    u5b = H.unit_key(units(5, 6, 0.5, 20)[0][0], THETA0, PANEL)
    prog = [json.loads(x) for x in (d / "w" / "progress.jsonl").read_text().splitlines()]
    rep["scenarios"]["C_takeover"] = {
        "pass": out["u5"]["sq"] == 25
        and any(p["event"] in ("fenced", "rejected_stale") for p in prog),
        "owner_epochs": sorted({r["epoch"] for r in recs if r["type"] == "owner_epoch"}),
        "killed_controller_attempt_retained": any(
            r["type"] == "attempt_start" and r["unit"] == u5b for r in recs
        ),
        "worker_events": sorted({p["event"] for p in prog}),
    }
    try:
        g.canary(C.composite_canary(DW.ENGINES))
        stale_refused = False
    except RG.StaleGuard:
        stale_refused = True
    rep["scenarios"]["C_takeover"]["old_owner_refused"] = stale_refused
    rep["scenarios"]["C_takeover"]["pass"] &= stale_refused
    # D: torn tail fails closed until the audited repair
    log = d / "study" / RG.LOG_NAME
    with open(log, "ab") as f:
        f.write(b'{"seq": 9999, "type": "torn')
    try:
        study(d).guard("bank", inst1)
        refused = False
    except RG.LogCorrupt:
        refused = True
    RG.repair_torn_tail(d / "study", "drill operator", "crash mid-append (fixture)")
    g3 = study(d).guard("bank", inst1)
    tr.set_fence(g3.epoch, inst1)
    out = (
        study(d).run(g3, rem, tr, units(6, 7), chunk=4)
        if False
        else s.run(g3, rem, tr, units(6, 7), chunk=4)
    )
    rep["scenarios"]["D_torn_tail"] = {
        "pass": refused and out["u6"]["sq"] == 36,
        "refused_before_repair": refused,
        "repaired": any(r["type"] == "log_repaired" for r in records(d)),
    }
    # E: instance loss with a pending attempt; registered relaunch on a new instance id
    g4 = s.guard("bank", inst1)
    tr.set_fence(g4.epoch, inst1)
    rec7 = units(7, 8)[0]
    H.run_units(
        s.store("bank"),
        [rec7],
        lambda r: rem.call("square", g4.epoch, inst1, s.key(r), "u7", r["args"])[0],
        THETA0,
        PANEL,
        guard=g4,
    )
    g4.declare_lost(
        inst1, "preemption_or_capacity", "drill: provider terminated the instance", "drill000", True
    )
    inst2 = "lambda:drill-2"
    g4.relaunch("preemption_or_capacity", inst2)
    g5 = s.guard("bank", inst2)
    tr.set_fence(g5.epoch, inst2)
    g5.open_instance(1)
    out = s.run(g5, rem, tr, [rec7], chunk=4)
    recs = records(d)
    u7 = s.key(rec7[0])
    rep["scenarios"]["E_instance_loss"] = {
        "pass": out["u7"]["sq"] == 49
        and any(r["type"] == "attempt_disposed" and r["unit"] == u7 for r in recs),
        "attempts_u7": sum(r["type"] == "attempt_start" and r["unit"] == u7 for r in recs),
        "relaunches": [r for r in recs if r["type"] == "relaunch"][-1]["n"],
    }
    try:
        g5.relaunch("infrastructure", "lambda:drill-3")
        second = False
    except RG.RetryExhausted:
        second = True
    rep["scenarios"]["E_instance_loss"]["second_relaunch_refused"] = second
    # F: dead-man reconciliation adds authoritative time (never removes)
    now = time.time()
    before = g5.ledger.instance_gpu_h()
    g5.reconcile_instance(
        now - 7200, now - 10, 2, "drill dead-man ledger", instance="lambda:drill-parity"
    )
    after = g5.ledger.instance_gpu_h()
    rep["scenarios"]["F_reconcile"] = {"pass": after - before >= 3.9, "added_gpu_h": after - before}
    # G: a changed earlier record is caught by the fresh verified replay required on restart
    raw = log.read_bytes().splitlines(keepends=True)
    i = next(k for k, line in enumerate(raw) if b"owner_epoch" in line)
    bad = raw[i].replace(b'"bank"', b'"bonk"', 1)  # same length: a changed value in an early record
    assert bad != raw[i] and len(bad) == len(raw[i])
    backup = log.read_bytes()
    log.write_bytes(b"".join(raw[:i] + [bad] + raw[i + 1 :]))
    try:
        RG.read_study(d / "study")
        caught = False
    except RG.LogCorrupt:
        caught = True
    log.write_bytes(backup)
    rep["scenarios"]["G_fresh_replay_detects_change"] = {"pass": caught}
    rep["log"] = {"records": len(records(d)), "types": sorted({r["type"] for r in records(d)})}
    rep["all_pass"] = all(v["pass"] for v in rep["scenarios"].values())
    (d / "w" / "STOP").write_text("stop")
    time.sleep(0.5)
    procs["p"].terminate()
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: v["pass"] for k, v in rep["scenarios"].items()} | {"all_pass": rep["all_pass"]}
        )
    )


if __name__ == "__main__":
    main()
