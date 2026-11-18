#!/usr/bin/env python3
"""RULING-wme-prereg-v2-issues item 6: re-simulate the H-T quadratic-null size and the A7 alternative with R6 symmetric
shrinkage in place (CPU only, synthetic data only).

    wme_v2c_r6.py run     --out D [--cell r_f=16,n_blk=6,n_cal=12] [--kappa 0.8] [--reps 600] [--procs 7]
    wme_v2c_r6.py summary --out D

Pipeline: ``powersim_refit_v2b.replicate(..., r6=True)``, the v2b full refit (frozen ``CPConfig()``, 10,000 flips,
C2-I gate) with every arm shrunk toward the raw field on the same split, grid and loss, ties within 0.5 % to the field.
Cell, kappa and seeds are v2b's (reference cell, kappa 0.80, validation replicates 0, 1, ...).

Stopping rules (written to config.json before any replicate runs; deterministic, outcome counts only):
* **size fail**: the quadratic-null rejection count exceeds floor(0.035 * planned reps); the size can no longer be
  <= 0.035, so every remaining job is cancelled and the run reports "R6 quadratic-null size > 0.035" (the ruling:
  stop and report, no redesign here);
* **alternative futility**: alternative non-rejections exceed planned - k_required (the v2b refit rule); no further
  alternative replicates are submitted;
* otherwise every planned replicate runs.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCENARIOS = ("null_quadratic", "alt")
SIZE_MAX = 0.035


def _cell(v: str) -> dict:
    return {k: int(x) for k, x in (kv.split("=") for kv in v.split(","))}


def _job(t):
    from wsp.wm import powersim_refit_v2b as RB

    r = RB.replicate(t["rep"], t["cell"], t["scenario"], t["kappa"], r6=True)
    r["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return r


def k_required(n: int) -> int:
    from wsp.wm import powersim_v1 as PV

    return next(k for k in range(n + 1) if PV.cp_lower_one_sided(k, n) >= PV.REQ["power_lb_min"])


def limits(n: int) -> dict:
    return {
        "null_max_rejections": math.floor(SIZE_MAX * n),
        "alt_max_non_rejections": n - k_required(n),
    }


def state(rows, n):
    lim = limits(n)
    nq = [r for r in rows if r["scenario"] == "null_quadratic"]
    al = [r for r in rows if r["scenario"] == "alt"]
    nq_rej = sum(r["reject"] for r in nq)
    al_non = sum(not r["reject"] for r in al)
    return {
        "size_fail": nq_rej > lim["null_max_rejections"],
        "alt_futile": al_non > lim["alt_max_non_rejections"],
        "null_rejections": nq_rej,
        "null_done": len(nq),
        "alt_non_rejections": al_non,
        "alt_done": len(al),
    }


def _rows(out: Path):
    f = out / "results.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cell = _cell(a.cell)
    cfgp = out / "config.json"
    conf = {
        "cell": cell,
        "kappa": a.kappa,
        "reps": a.reps,
        "procs": a.procs,
        "scenarios": SCENARIOS,
        "r6": True,
        "limits": limits(a.reps),
        "seeds": f"validation replicates 0..{a.reps - 1} (powersim_v1.SEED; v2b)",
        "ruling": "docs/council/2026-11-18/RULING-wme-prereg-v2-issues.md item 6",
    }
    if cfgp.exists() and json.loads(cfgp.read_text()) != conf:
        raise SystemExit("config.json differs from this invocation; refusing to mix runs")
    cfgp.write_text(json.dumps(conf, indent=1))
    rows = _rows(out)
    done = {(r["rep"], r["scenario"]) for r in rows}
    todo = [
        {"rep": i, "cell": cell, "scenario": sc, "kappa": a.kappa}
        for i in range(a.reps)
        for sc in SCENARIOS
        if (i, sc) not in done
    ]
    prog = out / "progress.jsonl"
    t0 = time.time()
    with (
        mp.get_context("spawn").Pool(a.procs, maxtasksperchild=10) as pool,
        open(out / "results.jsonl", "a") as f,
    ):
        pending = []
        it = iter(todo)
        stop = False
        while True:
            st = state(rows, a.reps)
            if st["size_fail"]:
                stop = True
            while not stop and len(pending) < a.procs:
                t = next(it, None)
                if t is None:
                    break
                if t["scenario"] == "alt" and st["alt_futile"]:
                    continue
                pending.append(pool.apply_async(_job, (t,)))
            if not pending or stop:
                break
            time.sleep(2)
            for p in [p for p in pending if p.ready()]:
                pending.remove(p)
                r = p.get()
                rows.append(r)
                f.write(json.dumps(r, default=float) + "\n")
                f.flush()
                with open(prog, "a") as g:
                    g.write(json.dumps({"t": time.time() - t0, **state(rows, a.reps)}) + "\n")
        if stop:
            pool.terminate()
    s = summary(out)
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    with open(prog, "a") as g:
        g.write(json.dumps({"t": time.time() - t0, "state": "finished"}) + "\n")
    print(json.dumps({k: s[k] for k in ("stop", "null_quadratic", "alt") if k in s}, default=float))


def summary(out: Path) -> dict:
    import numpy as np

    from wsp.wm import powersim_refit_v2b as RB
    from wsp.wm import powersim_v1 as PV

    conf = json.loads((out / "config.json").read_text())
    rows = _rows(out)
    n = conf["reps"]
    st = state(rows, n)
    s = RB.summarise(rows)
    for sc in SCENARIOS:
        rs = [r for r in rows if r["scenario"] == sc]
        if not rs:
            continue
        s[sc]["ct_quantiles"] = [
            float(q) for q in np.quantile([r["ct"] for r in rs], [0.1, 0.5, 0.9])
        ]
        s[sc]["median_wm_err"] = float(np.median([r["median_wm_err"] for r in rs]))
        sw = [x for r in rs for x in r["shrink_wm"]]
        ss = [x for r in rs for x in r["shrink_static"] if not math.isnan(x)]
        s[sc]["shrink_wm_counts"] = {str(v): sw.count(v) for v in sorted(set(sw))}
        s[sc]["shrink_static_counts"] = {str(v): ss.count(v) for v in sorted(set(ss))}
    if "null_quadratic" in s:
        k, m = st["null_rejections"], st["null_done"]
        s["null_quadratic"]["size_point"] = k / m
        s["null_quadratic"]["size_ci95"] = PV.clopper_pearson(k, m)
        s["null_quadratic"]["size_gate"] = (
            "FAIL (rejections exceed floor(0.035 x planned))"
            if st["size_fail"]
            else ("pass" if m >= n and k / m <= SIZE_MAX else "incomplete")
        )
    if "alt" in s:
        s["alt"]["power_gate"] = (
            "FAIL (futility: cannot reach the CP lower bound 0.80)"
            if st["alt_futile"]
            else ("pass" if s["alt"]["pass"] and s["alt"]["reps"] >= n else "incomplete or fail")
        )
    return {
        "config": conf,
        "stop": st,
        "cpu_h": sum(r["seconds"] for r in rows) / 3600,
        "max_rss_mb": max((r.get("rss_mb", 0) for r in rows), default=0),
        **s,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--cell", default="r_f=16,n_blk=6,n_cal=12")
    r.add_argument("--kappa", type=float, default=0.8)
    r.add_argument("--reps", type=int, default=600)
    r.add_argument("--procs", type=int, default=7)
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        print(json.dumps(summary(Path(a.out)), indent=1, default=float))


if __name__ == "__main__":
    main()
