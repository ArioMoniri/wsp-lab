#!/usr/bin/env python3
"""Full-refit validation of the selected v1 simulation cell (RULING-wme-resources §4(b)). CPU only.

    run --cell '<json>' --reps-null N --reps-alt N --kappa K --procs P --out <dir>   resumable (results.jsonl)
    summary --out <dir>                                                              rates, CIs, pass/fail

Each replicate refits the whole nested pipeline (``wm.powersim_refit.replicate``); one replicate per process at a
time, torch single-threaded (< 4 GB RSS per worker).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mod(version):
    if version == "v2":
        from wsp.wm import powersim_refit_v2 as R
    else:
        from wsp.wm import powersim_refit as R
    return R


def _one(args):
    rep, cell, null, kappa, gamma, version = args
    R = _mod(version)
    r = (
        R.replicate(rep, cell, null, gamma=gamma)
        if version == "v2"
        else R.replicate(rep, cell, null, kappa, gamma=gamma)
    )
    r["version"] = version
    r["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return r


def cmd_run(a):
    out = Path(a.out)
    (out / "progress").mkdir(parents=True, exist_ok=True)
    cell = json.loads(a.cell)
    res = out / "results.jsonl"
    done = set()
    if res.exists():
        for line in res.read_text().splitlines():
            r = json.loads(line)
            done.add((r["rep"], r["null"]))
    todo = [
        (i, cell, True, a.kappa, a.gamma, a.version)
        for i in range(a.reps_null)
        if (i, True) not in done
    ]
    todo += [
        (i, cell, False, a.kappa, a.gamma, a.version)
        for i in range(a.reps_alt)
        if (i, False) not in done
    ]
    todo.sort(key=lambda t: (t[0], t[2]))  # interleave null and alternative
    t0, n = time.time(), 0
    (out / "config.json").write_text(
        json.dumps(
            {
                "cell": cell,
                "kappa": a.kappa,
                "gamma": a.gamma,
                "reps_null": a.reps_null,
                "reps_alt": a.reps_alt,
                "procs": a.procs,
                "version": a.version,
            }
        )
    )
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=20) as pool, open(res, "a") as f:
        for r in pool.imap_unordered(_one, todo, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            n += 1
            (out / "progress" / "refit.json").write_text(
                json.dumps({"done": n, "todo": len(todo), "elapsed_s": time.time() - t0})
            )
    (out / "progress" / "refit.json").write_text(
        json.dumps({"done": n, "state": "finished", "elapsed_s": time.time() - t0})
    )


def cmd_summary(a):
    R = _mod(a.version)
    rows = [json.loads(x) for x in (Path(a.out) / "results.jsonl").read_text().splitlines()]
    s = R.summarise(rows)
    s["cpu_h"] = sum(r["seconds"] for r in rows) / 3600
    s["max_rss_mb"] = max(r.get("rss_mb", 0) for r in rows)
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--cell", default='{"r_f": 8, "n_blk": 6, "n_cal": 12}')
    ap.add_argument("--reps-null", type=int, default=600)
    ap.add_argument("--reps-alt", type=int, default=600)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument(
        "--gamma", type=float, default=0.0, help="alternative path effect (SE units), calibrated"
    )
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
