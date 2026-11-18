#!/usr/bin/env python3
"""Stage-W2 analysis (RULING-wme-stagew2 §7-§11). EXPLORATORY; CPU; run in the LC-PFN venv (Python 3.10).

    wme_w2_analyse.py codecheck --stagew-run DIR --out OUT.json [--reps 200] [--procs 6]
    wme_w2_analyse.py analyse --slot a=DIR --slot b=DIR --slot c=DIR --out OUT.json [--reps 1000] [--procs 6]

``codecheck`` runs the nested model, its nulls and the comparator merge on the stage-W run thinned to the W2 grid
(non-evidential, §10). ``analyse`` is the pre-declared W2 analysis: the primary contrast and pass rule (§7), the three
null sizes (§8; replicates split across processes, seeded per replicate), the secondary comparators (§9) and, on a
pass, the H-T3 sizing (§11). Every output file is followed by its sha256 in ``<out>.sha256`` after its final write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wsp.wm import stagew as W
from wsp.wm import stagew2 as W2


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def _write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1))
    tmp.replace(path)
    Path(str(path) + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
    )


def _lcpfn():
    try:
        return W.LCPFN()
    except ImportError:
        return None


_WORLD = None
_CAL = None


def _init(world, cal):
    global _WORLD, _CAL
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    _WORLD, _CAL = world, cal


def _chunk(args):
    kind, idx, seed = args
    return W2.null_decisions(_WORLD, kind, idx, seed=seed, cal=_CAL)


def nulls(world, reps, procs, seed=0):
    cal = W2.calibrate_n3(world)
    out = {}
    chunks = [list(range(i, min(reps, i + 25))) for i in range(0, reps, 25)]
    with mp.get_context("fork").Pool(procs, initializer=_init, initargs=(world, cal)) as pool:
        for kind in ("n1", "n2", "n3"):
            rows = [r for part in pool.map(_chunk, [(kind, c, seed) for c in chunks]) for r in part]
            rows.sort(key=lambda r: r["rep"])
            out[kind] = W2.summarize_null(rows)
            out[kind]["rows"] = [
                {"rep": r["rep"], "pass": r["pass"], "reject": r["reject"]} for r in rows
            ]
    out["n3"]["tau"] = cal["tau"]
    return out


def comparators(world, A, nested, lc):
    comp = W.analyse(world, n_flip=10_000, lcpfn=lc, inner=True, ks=W2.KS)
    return comp, W2.comparator_contrasts(nested, comp, A)


def cmd_codecheck(a):
    t0 = time.time()
    w = W2.thin_world(W.load_run(a.stagew_run))
    A = W2.arrays(w)
    res = W2.analyse(A)
    nl = nulls(w, a.reps, a.procs)  # before torch is imported (forked workers)
    lc = _lcpfn()
    comp, contrasts = comparators(w, A, res, lc)
    out = {
        "label": "NON-EVIDENTIAL code check on stage-W data (RULING-wme-stagew2 §10); 4 runs cannot pass",
        "stagew_run": str(a.stagew_run),
        "paths": A["ids"],
        "nested": res,
        "nulls": nl,
        "comparators": contrasts,
        "comparator_mean_E": {k: comp["settings"][k]["mean_E"] for k in comp["settings"]},
        "lcpfn": None if lc is None else lc.name,
        "cpu_seconds": time.time() - t0,
    }
    _write(a.out, out)
    print(json.dumps(_jsonable({k: v["mean_E"] for k, v in res["settings"].items()})))


def cmd_analyse(a):
    t0 = time.time()
    dirs = dict(s.split("=", 1) for s in a.slot)
    w = W2.load_runs(dirs)
    A = W2.arrays(w)
    res = W2.analyse(A)
    print(
        "primary",
        json.dumps(
            _jsonable({k: (v["d_mean"], v["p"], v["lb_holm"]) for k, v in res["settings"].items()})
        ),
        flush=True,
    )
    nl = nulls(w, a.reps, a.procs)
    print("nulls", json.dumps({k: v["size"] for k, v in nl.items()}), flush=True)
    sizes_ok = all(nl[k]["size"] <= W2.NULL_MAX for k in ("n1", "n2", "n3"))
    verdict = bool(res["enough_paths"] and res["pass_primary"] and sizes_ok)
    lc = _lcpfn()
    comp, contrasts = comparators(w, A, res, lc)
    out = {
        "label": "EXPLORATORY stage W2 (RULING-wme-stagew2)",
        "slots": dirs,
        "n_paths": len(A["ids"]),
        "paths": A["ids"],
        "runs": A["run_ids"],
        "incomplete": w["incomplete"],
        "base_slot": w["base_slot"],
        "slot_bases": w["slot_bases"],
        "slot_of_run": w["slot_of_run"],
        "base_y": w["base"]["y"],
        "base_rerun_y": w.get("base_rerun", {}).get("y"),
        "base_rerun_discordance": w.get("base_rerun_discordance"),
        "N": w["N"],
        "features": W.FEATURES,
        "endpoints": W.ENDPOINTS,
        "members": {
            k: {"spec": W.unkey(k), "phi": m["phi"], "y": m["y"]} for k, m in w["members"].items()
        },
        "nested": res,
        "nulls": nl,
        "rule": {
            "enough_paths": res["enough_paths"],
            "primary_pass": res["pass_primary"],
            "null_sizes_ok": sizes_ok,
            "null_max": W2.NULL_MAX,
            "PASS": verdict,
        },
        "comparators": contrasts,
        "comparator_raw": {"shrink": comp["shrink"], "gate": comp["gate"]},
        "lcpfn": None if lc is None else lc.name,
    }
    if verdict:
        ks = [k for k, st in res["settings"].items() if st["pass"]]
        powers = {}
        for k in ks:
            d = np.asarray(res["settings"][k]["E"]["pooled"]) - np.asarray(
                res["settings"][k]["E"]["nested"]
            )
            powers[k] = {R: W2.ht3_power(d, A["runs"], R, seed=int(k)) for R in W2.R_GRID}
        chosen = next((R for R in W2.R_GRID if all(powers[k][R] >= W2.POWER_MIN for k in ks)), None)
        out["ht3"] = {"settings": ks, "power": powers, "R": chosen}
    out["cpu_seconds"] = time.time() - t0
    _write(a.out, out)
    print("PASS" if verdict else "FAIL", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("codecheck")
    s.add_argument("--stagew-run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--reps", type=int, default=200)
    s.add_argument("--procs", type=int, default=6)
    s = sub.add_parser("analyse")
    s.add_argument("--slot", action="append", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--reps", type=int, default=W2.NULL_REPS)
    s.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    {"codecheck": cmd_codecheck, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
