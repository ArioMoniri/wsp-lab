#!/usr/bin/env python3
"""Stage-W analysis and power simulations (RULING-wme-stagew §6-§8). EXPLORATORY; CPU; run in the LC-PFN venv.

    wme_w_analyse.py analyse --run DIR --out OUT.json          # H-T2 on stage-W data, N1/N2 nulls, merges
    wme_w_analyse.py ht2sim  --run DIR --out OUT.json [--procs 7] [--reps 400]
    wme_w_analyse.py e2sim   --out OUT.json [--reps 1000]

``ht2sim`` simulates P = 40 first for every world (alt, N1, N2, N3); if the alternative is futile at P = 40
(every setting's one-sided 95 % upper bound < 0.8) the smaller P are not simulated (plan §7).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wsp.wm import powersim_w as PW
from wsp.wm import stagew as W

N_NULL_PERMS = 20


def _lcpfn():
    try:
        return W.LCPFN()
    except ImportError:
        return None


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


def cmd_analyse(a):
    t0 = time.time()
    w = W.load_run(a.run)
    lc = _lcpfn()
    res = W.analyse(w, n_flip=10_000, lcpfn=lc, inner=True)
    nulls = {}
    for name, fn in (
        ("n1_shuffled_features", W.null_shuffle_features),
        ("n2_path_permutation", W.null_permute_paths),
    ):
        rows = []
        for s in range(N_NULL_PERMS):
            r = W.analyse(fn(w, seed=s), n_flip=10_000, lcpfn=lc, inner=True)
            rows.append(
                {
                    "seed": s,
                    "reject": {k: v["reject"] for k, v in r["settings"].items()},
                    "p_iu": {k: v["p_iu"] for k, v in r["settings"].items()},
                    "C": {k: v["C"] for k, v in r["settings"].items()},
                    "mean_E_wm": {k: v["mean_E"]["wm"] for k, v in r["settings"].items()},
                }
            )
        nulls[name] = {
            "rows": rows,
            "reject_rate": {
                k: float(np.mean([r["reject"][k] for r in rows])) for k in res["settings"]
            },
            "mean_C": {
                k: {
                    b: float(np.mean([r["C"][k][b] for r in rows]))
                    for b in res["settings"][k]["baselines"]
                }
                for k in res["settings"]
            },
        }
    out = {
        "run": str(a.run),
        "lcpfn": None if lc is None else lc.name,
        "n_layers": w["n_layers"],
        "N": w["N"],
        "paths": [p["id"] for p in w["paths"]],
        "incomplete": w["incomplete"],
        "base_y": w["base"]["y"],
        "base_rerun_y": w.get("base_rerun", {}).get("y"),
        "base_rerun_discordance": w.get("base_rerun_discordance"),
        "members": {
            k: {"spec": W.unkey(k), "phi": m["phi"], "y": m["y"]} for k, m in w["members"].items()
        },
        "features": W.FEATURES,
        "endpoints": W.ENDPOINTS,
        "ht2": res,
        "nulls": nulls,
        "merges_zero_shot": W.merges_zero_shot(w),
        "cpu_seconds": time.time() - t0,
    }
    Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))
    print(
        json.dumps(_jsonable({k: out["ht2"]["settings"][k]["C"] for k in out["ht2"]["settings"]}))
    )


_CAL = None
_LC = None


def _init(run):
    global _CAL, _LC
    import torch

    torch.set_num_threads(1)
    _CAL = PW.calibrate(W.load_run(run))
    _LC = _lcpfn()


def _rep(args):
    P, kind, seed = args
    t = time.time()
    r = PW.replicate(_CAL, P, kind, seed, lcpfn=_LC, n_flip=10_000)
    r.update(P=P, kind=kind, seed=seed, seconds=time.time() - t)
    return r


def cmd_ht2sim(a):
    t0 = time.time()
    w = W.load_run(a.run)
    cal = PW.calibrate(w)
    res = json.loads(Path(a.analysis).read_text())
    st = res["ht2"]["settings"]
    E = {int(k): {arm: np.array(v) for arm, v in s["E"].items()} for k, s in st.items()}
    bl = {int(k): s["baselines"] for k, s in st.items()}
    out = {
        "calibration": {
            "tau": cal["tau"],
            "offsets_raw": cal["offsets_raw"],
            "runs": sorted(cal["traj"]),
        },
        "grid": {},
        "empirical": {},
        "reps": a.reps,
    }
    prog = Path(a.out + ".progress")
    with mp.get_context("spawn").Pool(a.procs, initializer=_init, initargs=(a.run,)) as pool:
        order = list(PW.P_GRID[::-1])
        for P in order:
            cells = {}
            for kind in PW.KINDS:
                rows = pool.map(_rep, [(P, kind, s) for s in range(a.reps)], chunksize=1)
                cnt = {k: int(sum(r["reject"][k] for r in rows)) for k in rows[0]["reject"]}
                cells[kind] = {
                    "rate": {k: c / a.reps for k, c in cnt.items()},
                    "counts": cnt,
                    "power_lb": {k: PW.cp_lower(c, a.reps) for k, c in cnt.items()},
                    "upper": {k: PW.cp_upper(c, a.reps) for k, c in cnt.items()},
                    "mean_C": {
                        k: {
                            b: float(np.mean([r["C"][k][b] for r in rows])) for b in rows[0]["C"][k]
                        }
                        for k in rows[0]["C"]
                    },
                    "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
                }
                prog.write_text(json.dumps({"P": P, "kind": kind, "elapsed_s": time.time() - t0}))
            out["grid"][P] = cells
            out["empirical"][P] = PW.empirical_power(E, bl, P, reps=2000, seed=P)
            Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))
            if P == max(PW.P_GRID) and PW.futile(cells["alt"]["counts"], a.reps):
                out["futility"] = {"at_P": P, "stopped": True, "counts": cells["alt"]["counts"]}
                break
    out["cpu_wall_seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))


def cmd_e2sim(a):
    t0 = time.time()
    inp = PW.e2_inputs()
    rho = inp["rho_hat"]
    runs = {
        "planning_parity": {"rho": rho, "budget_bon": inp["budget_bon"]},
        "null_exchangeable": {"rho": 0.0, "budget_bon": inp["B"], "exchangeable": True},
        "rho_half_parity": {"rho": 0.5 * rho, "budget_bon": inp["budget_bon"]},
        "rho_sqrt_parity": {"rho": float(np.sqrt(rho)), "budget_bon": inp["budget_bon"]},
        "planning_no_charge": {"rho": rho, "budget_bon": inp["B"]},
        "perfect_prior_parity": {"rho": 1.0, "budget_bon": inp["budget_bon"]},
    }
    out = {
        "inputs": {k: v for k, v in inp.items() if k != "pools"},
        "pools": inp["pools"],
        "results": {},
    }
    for i, (name, kw) in enumerate(runs.items()):
        out["results"][name] = PW.e2_power(
            inp, reps=a.reps, seed=1000 + i, alpha=0.0125, n_flip=10_000, **kw
        )
        Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))
    out["results"]["planning_parity_alpha_0.025"] = PW.e2_power(
        inp,
        rho=rho,
        budget_bon=inp["budget_bon"],
        reps=a.reps,
        seed=2000,
        alpha=0.025,
        n_flip=10_000,
    )
    out["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps({k: v["rate"] for k, v in out["results"].items()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyse")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("ht2sim")
    s.add_argument("--run", required=True)
    s.add_argument("--analysis", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--procs", type=int, default=7)
    s.add_argument("--reps", type=int, default=PW.REPS)
    s = sub.add_parser("e2sim")
    s.add_argument("--out", required=True)
    s.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    {"analyse": cmd_analyse, "ht2sim": cmd_ht2sim, "e2sim": cmd_e2sim}[a.cmd](a)


if __name__ == "__main__":
    main()
