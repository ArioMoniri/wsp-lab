#!/usr/bin/env python3
"""RULING-wme-v2b A7/A11 simulation jobs (CPU only; resumable ``results.jsonl`` per output directory).

    cpsel   --out D [--reps N] [--procs P]          WM-CP config/K-rule selection on native synthetic data only
    cpsel-summary --out D                           pick the config (pooled median error; 2 % tie -> earlier)
    cal     --out D --kappas 0.5,1,2 [--reps N]     static held-out error at h = 3-4 under the alternative
    cal-summary --out D                             pooled median per kappa; kappa* by interpolation to 2.17
    refit   --out D --cell JSON --kappa K --reps N  full refit, the three scenarios interleaved per replicate
    refit-summary --out D                           rates, coverage, achieved C_T, futility (A11, ruling §4(b))

One single-threaded job per worker. The config used by ``cal`` and ``refit`` is the frozen selection
(``--cfg JSON``, as written by cpsel-summary).
"""

from __future__ import annotations

import argparse
import dataclasses
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

SEL_CELL = {
    "r_f": 16,
    "n_blk": 6,
    "n_cal": 12,
}  # reference cell for selection and calibration (v2 selection)
CPSEL_SEED0 = 50_000
CPSEL_TIE = 0.02
CPSEL_GRID = (
    ("C0", {}),
    ("C1", {"k_max": 32, "s0_scale": 0.5}),
    ("C2", {"hidden": 128, "latent": 64}),
    ("C3", {"obs_weight": 0.1}),
)


def _arg(v: str):
    """JSON argument, inline or ``@path`` (the Lambda driver passes arguments unquoted)."""
    if not v:
        return {}
    if v.startswith("@"):
        return json.loads(Path(v[1:]).read_text())
    if "=" in v and not v.lstrip().startswith(
        "{"
    ):  # r_f=16,n_blk=6,n_cal=12 (no braces: survives unquoted shells)
        return {k: int(x) for k, x in (kv.split("=") for kv in v.split(","))}
    return json.loads(v)


def _cfg(d):
    from wsp.wm import wmcp as CP

    return CP.CPConfig(**(d or {}))


def _job(args):
    kind, payload = args
    import numpy as np

    from wsp.wm import powersim_refit_v2b as RB

    t0 = time.time()
    if kind == "cpsel":
        e = RB.cp_errors(payload["rep"], SEL_CELL, _cfg(payload["cfg"]))
        r = {
            "name": payload["name"],
            "rep": payload["rep"],
            "median": float(np.median(e)),
            "errors": e.tolist(),
        }
    elif kind == "cal":
        e = RB.static_errors(payload["rep"], SEL_CELL, payload["kappa"], _cfg(payload["cfg"]))
        r = {
            "kappa": payload["kappa"],
            "rep": payload["rep"],
            "median": float(np.median(e)),
            "errors": e.tolist(),
        }
    else:
        r = RB.replicate(
            payload["rep"],
            payload["cell"],
            payload["scenario"],
            payload["kappa"],
            _cfg(payload["cfg"]),
        )
    r["seconds"] = r.get("seconds", time.time() - t0)
    r["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return r


def _run(out, kind, todo, procs, key):
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    done = set()
    if res.exists():
        done = {key(json.loads(x)) for x in res.read_text().splitlines() if x.strip()}
    todo = [t for t in todo if key(t) not in done]
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=10) as pool, open(res, "a") as f:
        for r in pool.imap_unordered(_job, [(kind, t) for t in todo], chunksize=1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            n += 1
            (out / "progress.json").write_text(
                json.dumps({"done": n, "todo": len(todo), "s": time.time() - t0})
            )
    (out / "progress.json").write_text(
        json.dumps({"done": n, "state": "finished", "s": time.time() - t0})
    )


def _rows(out):
    return [
        json.loads(x) for x in (Path(out) / "results.jsonl").read_text().splitlines() if x.strip()
    ]


def cmd_cpsel(a):
    from wsp.wm import powersim_refit_v2b as RB  # noqa: F401  (import check before spawning)

    todo = [
        {"name": n, "cfg": c, "rep": CPSEL_SEED0 + i} for i in range(a.reps) for n, c in CPSEL_GRID
    ]
    (Path(a.out)).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "config.json").write_text(
        json.dumps(
            {
                "grid": CPSEL_GRID,
                "cell": SEL_CELL,
                "reps": a.reps,
                "seed0": CPSEL_SEED0,
                "tie": CPSEL_TIE,
            }
        )
    )
    _run(Path(a.out), "cpsel", todo, a.procs, lambda t: (t["name"], t["rep"]))


def cpsel_pick(rows):
    import numpy as np

    med = {}
    for n, _c in CPSEL_GRID:
        e = [x for r in rows if r["name"] == n for x in r["errors"]]
        if e:
            med[n] = float(np.median(e))
    best = min(med.values())
    pick = next(n for n, _c in CPSEL_GRID if n in med and med[n] <= best * (1 + CPSEL_TIE))
    return pick, med


def cmd_cpsel_summary(a):
    from wsp.wm import wmcp as CP

    rows = _rows(a.out)
    pick, med = cpsel_pick(rows)
    cfg = dataclasses.asdict(_cfg(dict(CPSEL_GRID)[pick]))
    s = {
        "procedure": "WM-CP config and K rule chosen on synthetic data only (A7): native SynthWorld('saturating') "
        "outcomes + noise, never the registered alternative; candidates CPSEL_GRID; criterion the pooled median of "
        "WM-CP's own held-out SE-scaled |error| at h = 3-4 over the 5 outer folds of each replicate; ties within 2 % "
        "relative go to the earlier (simpler) candidate",
        "cell": SEL_CELL,
        "seeds": sorted({r["rep"] for r in rows}),
        "grid": dict(CPSEL_GRID),
        "pooled_median": med,
        "picked": pick,
        "frozen_config": cfg,
        "k_max_default": CP.K_MAX,
        "reps_per_config": {n: sum(r["name"] == n for r in rows) for n in med},
        "cpu_h": sum(r["seconds"] for r in rows) / 3600,
    }
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def cmd_cal(a):
    from wsp.wm import powersim_refit_v2b as RB

    ks = [float(x) for x in a.kappas.split(",")]
    cfg = _arg(a.cfg)
    todo = [
        {"kappa": k, "rep": RB.CAL_SEED0 + a.rep0 + i, "cfg": cfg}
        for i in range(a.reps)
        for k in ks
    ]
    _run(Path(a.out), "cal", todo, a.procs, lambda t: (t["kappa"], t["rep"]))


def cmd_cal_summary(a):
    import numpy as np

    from wsp.wm import powersim_refit_v2b as RB

    rows = _rows(a.out)
    ks = sorted({r["kappa"] for r in rows})
    med = {
        k: float(np.median([x for r in rows if r["kappa"] == k for x in r["errors"]])) for k in ks
    }
    reps = {k: sum(r["kappa"] == k for r in rows) for k in ks}
    m = [med[k] for k in ks]
    star = None
    for i in range(len(ks) - 1):
        if (m[i] - RB.CAL_TARGET) * (m[i + 1] - RB.CAL_TARGET) <= 0 and m[i] != m[i + 1]:
            star = ks[i] + (RB.CAL_TARGET - m[i]) * (ks[i + 1] - ks[i]) / (m[i + 1] - m[i])
            break
    s = {
        "target": RB.CAL_TARGET,
        "pooled_median": med,
        "reps": reps,
        "kappa_star": star,
        "cpu_h": sum(r["seconds"] for r in rows) / 3600,
    }
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


CAL_RULE = (
    "kappa* = linear interpolation, at the first crossing of 2.17, of the pooled median static held-out SE-scaled "
    "|error| (h = 3-4, all objectives) over every calibration replicate at each kappa (seeds CAL_SEED0 + rep)"
)


def cal_combine(rows):
    import numpy as np

    from wsp.wm import powersim_refit_v2b as RB

    ks = sorted({float(r["kappa"]) for r in rows})
    med = {
        k: float(np.median([x for r in rows if float(r["kappa"]) == k for x in r["errors"]]))
        for k in ks
    }
    reps = {k: len({r["rep"] for r in rows if float(r["kappa"]) == k}) for k in ks}
    star = None
    for i in range(len(ks) - 1):
        m0, m1 = med[ks[i]], med[ks[i + 1]]
        if (m0 - RB.CAL_TARGET) * (m1 - RB.CAL_TARGET) <= 0 and m0 != m1:
            star = ks[i] + (RB.CAL_TARGET - m0) * (ks[i + 1] - ks[i]) / (m1 - m0)
            break
    return {
        "target": RB.CAL_TARGET,
        "rule": CAL_RULE,
        "pooled_median": med,
        "reps": reps,
        "kappa_star": star,
    }


def cmd_cal_combine(a):
    rows = [r for d in a.out.split(",") for r in _rows(d)]
    s = cal_combine(rows)
    s["kappa_star_registered"] = None if s["kappa_star"] is None else round(s["kappa_star"], 3)
    s["dirs"] = a.out.split(",")
    s["cpu_h"] = sum(r["seconds"] for r in rows) / 3600
    print(json.dumps(s, indent=1))


def cmd_refit(a):
    from wsp.wm import powersim_refit_v2b as RB

    cell = _arg(a.cell)
    cfg = _arg(a.cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "cell": cell,
                "kappa": a.kappa,
                "reps": a.reps,
                "cfg": cfg,
                "scenarios": RB.SCENARIOS,
                "procs": a.procs,
            }
        )
    )
    todo = [
        {"rep": i, "cell": cell, "scenario": sc, "kappa": a.kappa, "cfg": cfg}
        for i in range(a.rep0, a.rep0 + a.reps)
        for sc in RB.SCENARIOS
    ]
    _run(out, "refit", todo, a.procs, lambda t: (t["rep"], t["scenario"]))


def k_required(n):
    from wsp.wm import powersim_v1 as PV

    return next(k for k in range(n + 1) if PV.cp_lower_one_sided(k, n) >= PV.REQ["power_lb_min"])


def refit_summary(d: Path) -> dict:
    import numpy as np

    from wsp.wm import powersim_refit_v2b as RB

    cfg = json.loads((d / "config.json").read_text())
    rows = _rows(d)
    s = RB.summarise(rows)
    n = cfg["reps"]
    k = k_required(n)
    alt = [r for r in rows if r["scenario"] == "alt"]
    non = sum(not r["reject"] for r in alt)
    fut = non > n - k
    if fut and "alt" in s:
        s["alt"]["pass"] = False
    for sc in RB.SCENARIOS:
        rs = [r for r in rows if r["scenario"] == sc]
        if rs:
            s[sc]["ct_quantiles"] = [
                float(q) for q in np.quantile([r["ct"] for r in rs], [0.1, 0.5, 0.9])
            ]
            s[sc]["median_wm_err"] = float(np.median([r["median_wm_err"] for r in rs]))
    return {
        "cell": cfg["cell"],
        "kappa": cfg["kappa"],
        "cfg": cfg["cfg"],
        "planned_reps": n,
        "completed": {sc: sum(r["scenario"] == sc for r in rows) for sc in RB.SCENARIOS},
        "futility": {
            "k_required_at_planned": k,
            "alt_non_rejections": non,
            "cannot_pass": bool(fut),
        },
        "passed": bool(all(s.get(sc, {}).get("pass") for sc in RB.SCENARIOS)),
        "cpu_h": sum(r["seconds"] for r in rows) / 3600,
        "max_rss_mb": max((r.get("rss_mb", 0) for r in rows), default=0),
        **s,
    }


def cmd_refit_summary(a):
    s = refit_summary(Path(a.out))
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(s, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "cpsel",
            "cpsel-summary",
            "cal",
            "cal-summary",
            "cal-combine",
            "refit",
            "refit-summary",
        ],
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--kappas", default="1,2,4")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--cell", default=json.dumps(SEL_CELL))
    ap.add_argument("--cfg", default="")
    a = ap.parse_args()
    {
        "cpsel": cmd_cpsel,
        "cpsel-summary": cmd_cpsel_summary,
        "cal": cmd_cal,
        "cal-summary": cmd_cal_summary,
        "cal-combine": cmd_cal_combine,
        "refit": cmd_refit,
        "refit-summary": cmd_refit_summary,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
