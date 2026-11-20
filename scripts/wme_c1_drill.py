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


def study(d, sub="", health=False):
    # A-G exercise the registered guard mechanics (automatic technical retries); H runs the §II.31 health layer
    s = SD.Study(
        d / f"study{sub}",
        d / f"runs{sub}",
        att(),
        THETA0,
        PANEL,
        health=health,
        health_order=(("drill", ("square",)),),
    )
    s.priced = {"bank": 100.0, "confirmatory": 100.0, "sbon": 100.0, "ste": 100.0}
    return s


def start_worker(d):
    return subprocess.Popen(
        [sys.executable, "-m", "wsp.wm.c1.drillworker", "--root", str(d / "w")],
        stdout=subprocess.DEVNULL,
        stderr=open(d / "worker.log", "a"),
    )


def units(lo, hi, sleep=0.0, steps=1, **extra):
    return [
        (
            {
                "kind": "square",
                "args": {"x": i, "sleep": sleep, "steps": steps, **extra},
                "label": f"u{i}",
            },
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
    from wsp.wm.c1 import phasel as PL

    rep = {
        "label": "wme-c1 CPU deployment drill (fixtures; engineering only; not confirmatory)",
        # §II.31.9 N5: the code the drill ran on
        "commit": PL.git("rev-parse", "HEAD"),
        "tracked_tree_clean": PL.git("status", "--porcelain", "--untracked-files=no") == "",
        "scenarios": {},
    }
    wp = start_worker(d)
    s = study(d)
    tr = T.LocalTransport(d / "w")
    procs = {"p": wp}

    def revive():
        procs["p"] = start_worker(d)

    def restart():  # §II.31.8 R6: kill the worker process and start it again
        procs["p"].kill()
        procs["p"].wait(timeout=10)
        procs["p"] = start_worker(d)

    rem = C.Remote(tr, d / "ctl", poll_s=0.05, dead_after_s=5, revive=revive, restart=restart)
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
    # H (§II.31.5, §II.31.8 R3 R4 R6): run health on the real worker loop.
    # H1 R3: a technical failure keeps its registered cause, halts (no further attempt), resumes after resolution;
    #    a non-technical failure halts and is retried only after a repair record.
    sh = study(d, "_h", health=True)

    def guard_h():
        g_ = sh.guard("bank", "lambda:drill-h")
        tr.set_fence(g_.epoch, "lambda:drill-h")
        g_.open_instance(1)
        return g_

    def run_h(g_, us):
        try:
            return sh.run(g_, rem, tr, us, chunk=1), None
        except Exception as exc:  # noqa: BLE001
            return None, type(exc).__name__

    def hrecs():
        return [json.loads(x) for x in (d / "study_h" / RG.LOG_NAME).read_text().splitlines()]

    gh = guard_h()
    ut = units(20, 21, fail_once="technical")
    _, e_t = run_h(gh, ut)
    k20 = sh.key(ut[0][0])
    fail20 = [r["cause"] for r in hrecs() if r["type"] == "attempt_fail" and r["unit"] == k20]
    starts20_halt = sum(r["type"] == "attempt_start" and r["unit"] == k20 for r in hrecs())
    gh.resolve_incidents("drill: injected infrastructure error, diagnosed", "drill report (fixture)")
    gh = guard_h()
    out_t, e_t2 = run_h(gh, ut)
    un = units(21, 22, fail_once="defect")
    _, e_n = run_h(gh, un)
    k21 = sh.key(un[0][0])
    gh.resolve_incidents("drill: injected defect, diagnosed", "drill report (fixture)")
    gh = guard_h()
    _, e_n2 = run_h(gh, un)  # refused: no repair yet
    gh.retry.repair("code_defect_crash", "drill: defect fixed", "drill00", outcome_blind=True)
    out_n, e_n3 = run_h(gh, un)
    hr = hrecs()
    summ = (d / "runs_h" / "health" / "summary.jsonl").read_text().splitlines()
    rep["scenarios"]["H_run_health_halt"] = {
        "pass": e_t == "HealthHalt"
        and fail20 == ["infrastructure"]
        and starts20_halt == 1
        and e_t2 is None
        and out_t["u20"]["sq"] == 400
        and e_n == "WorkerError"
        and [r["cause"] for r in hr if r["type"] == "attempt_fail" and r["unit"] == k21] == [None]
        and e_n2 == "RepairRequired"
        and e_n3 is None
        and out_n["u21"]["sq"] == 441
        and sum(r["type"] == "incident" for r in hr) == 2
        and sum(r["type"] == "incidents_resolved" for r in hr) == 2
        and len(summ) >= 2,
        "technical_cause_recorded": fail20,
        "errors": [e_t, e_t2, e_n, e_n2, e_n3],
        "attempts_u20": sum(r["type"] == "attempt_start" and r["unit"] == k20 for r in hr),
        "attempts_u21": sum(r["type"] == "attempt_start" and r["unit"] == k21 for r in hr),
    }
    # H (addendum 2, answer 3): the GPU-idle rule binds GPU kinds only; CPU kinds are checked by progress. Fixture
    #    classes {square: gpu, plan: cpu}, scripted utilisation, limits shortened to 1.5 s (registered: 15 min).
    s2 = SD.Study(
        d / "study_h2",
        d / "runs_h2",
        att(),
        THETA0,
        PANEL,
        health=True,
        health_order=(("drill", ("square", "plan")),),
        health_kinds={"square": "gpu", "plan": "cpu", "canary": "gpu"},
        health_limits={"idle_s": 1.5, "progress_s": 1.5, "poll_s": 0.0},
    )
    s2.priced = dict(s.priced)
    util = d / "w" / "gpu_util"

    def plan_units(x, **kw):
        return [({"kind": "plan", "args": {"x": x, **kw}, "label": f"p{x}"}, f"p{x}")]

    def worker_idle(timeout=30.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            hb_ = tr.heartbeat() or {}
            if hb_.get("inflight") is None:
                return True
            time.sleep(0.1)
        return False

    def guard2():
        g_ = s2.guard("bank", "lambda:drill-h2")
        tr.set_fence(g_.epoch, "lambda:drill-h2")
        g_.open_instance(1)
        return g_

    def attempt(g_, us):
        try:
            return s2.run(g_, rem, tr, us, chunk=1), None
        except Exception as exc:  # noqa: BLE001
            return None, type(exc).__name__

    def incidents2():
        recs = [json.loads(x) for x in (d / "study_h2" / RG.LOG_NAME).read_text().splitlines()]
        return [r["detail"]["inflight"]["kind"] + ":" + r["kind"] for r in recs if r["type"] == "incident"]

    util.write_text("5")  # the GPU idles throughout (a)-(b)
    g2 = guard2()
    # (a) a CPU kind progressing on an idle GPU for 3 s (> the 1.5 s limit): no incident
    out_a, err_a = attempt(g2, plan_units(30, steps=20, sleep=0.15))
    inc_a = incidents2()
    # (b) R4 + R6: a GPU kind progressing on an idle GPU trips gpu_idle three times (each withdrawn by fencing,
    #     resolved); the aborts never count, then with the GPU busy it completes
    trips_b = []
    for i in range(3):
        _o, e = attempt(g2, units(31, 32, sleep=0.15, steps=20))
        trips_b.append(e)
        idle_ok = worker_idle()
        g2.resolve_incidents(f"drill: GPU idle trip {i} (scripted)", "drill report (fixture)")
        g2 = guard2()
    err_b, idle_b = trips_b[0], idle_ok
    util.write_text("90")
    out_b2, err_b2 = attempt(g2, units(31, 32, sleep=0.15, steps=20))
    k31 = s2.key(units(31, 32, sleep=0.15, steps=20)[0][0])
    # (d) R6: a hung GPU op (never reaches ctx.check) on an idle GPU: fencing cannot stop it -> killed, restarted
    util.write_text("5")
    rem.withdraw_grace_s = 0.5
    _o, err_d = attempt(g2, units(33, 34, hang_once=4.0))
    rem.withdraw_grace_s = 120.0
    util.write_text("90")
    restarted = '"how": "restarted"' in (d / "ctl" / "dispatch.jsonl").read_text()
    idle_d = worker_idle()
    g2.resolve_incidents("drill: hung op, worker restarted", "drill report (fixture)")
    g2 = guard2()
    out_d, err_d2 = attempt(g2, units(33, 34, hang_once=4.0))
    # (c) a CPU kind without progress for 3 s: no_progress incident; resolved, it completes
    out_c, err_c = attempt(g2, plan_units(32, stall_once=3.0, steps=2, sleep=0.05))
    idle_c = worker_idle()
    g2.resolve_incidents("drill: planner stalled (scripted), diagnosed", "drill report (fixture)")
    g2 = guard2()
    out_c2, err_c2 = attempt(g2, plan_units(32, stall_once=3.0, steps=2, sleep=0.05))
    inc = incidents2()
    h2 = [json.loads(x) for x in (d / "study_h2" / RG.LOG_NAME).read_text().splitlines()]
    aborted31 = sum(r["type"] == "attempt_aborted" and r["unit"] == k31 for r in h2)
    fails_h2 = [r for r in h2 if r["type"] == "attempt_fail"]
    rep["scenarios"]["H_withdrawal_and_no_attempt_burn"] = {
        "pass": trips_b == ["HealthIncident"] * 3
        and aborted31 == 3
        and not fails_h2
        and not [r for r in h2 if r["type"] == "fallback"]
        and err_b2 is None
        and err_d == "HealthIncident"
        and restarted
        and idle_d
        and err_d2 is None
        and out_d["u33"]["sq"] == 1089
        and '"how": "fenced"' in (d / "ctl" / "dispatch.jsonl").read_text(),
        "aborted_attempts_u31": aborted31,
        "failures": len(fails_h2),
        "restarted": restarted,
        "trips": trips_b,
    }
    rep["scenarios"]["H_gpu_and_cpu_kinds"] = {
        "pass": err_a is None
        and out_a["p30"]["plan"] == 1030
        and inc_a == []
        and err_b == "HealthIncident"
        and idle_b
        and err_b2 is None
        and out_b2["u31"]["sq"] == 961
        and err_c == "HealthIncident"
        and idle_c
        and err_c2 is None
        and out_c2["p32"]["plan"] == 1032
        and inc == ["square:gpu_idle"] * 4 + ["plan:no_progress"],
        "incidents": inc,
        "cpu_kind_on_idle_gpu_incidents": inc_a,
    }
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
