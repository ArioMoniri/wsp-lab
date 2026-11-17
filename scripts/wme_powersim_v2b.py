#!/usr/bin/env python3
"""Cheap grid and A8 selection under RULING-wme-v2b (A7-A10). CPU only.

    grid    --inputs <v1 inputs.json> --out <dir> --claims e1|ht [--ct-alt X] [--procs P] [--reps 1000]
    select  --inputs ... --out <dir> --v2 <v2 grid results.jsonl> --ph100b <genbatch.json> --gen-batch B
            --refit <refit summary json> --report <json>

H-T cells use the v2b scenarios (``powersim_v2b.ht_scenarios``) with the alternative's C_T measured by full-refit
replicates on calibration seeds (``--ct-alt``); E1 cells run at 0.5x the generated effect (``alt_gen_half``, A10)
with the full effect as a sensitivity, over ``powersim_v2b.GEN_GRID``; E2 rows are v2's (unchanged model and
inference). Results within 2 Monte Carlo SE of a threshold are re-run at 4,000 replicates. Selection: A8 priority,
caps A9, pricing with the P-H100b generation seconds at the registered batch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wsp.wm import powersim_v1 as PV
from wsp.wm import powersim_v2 as P2
from wsp.wm import powersim_v2b as PB

AUDIT_V1 = ROOT / "docs/reviews/wme/wme-c1-exposure-audit-v1-2026-11-17.json"


def _key(t):
    return json.dumps([t[0], sorted(t[1].items()), t[2]])


def tasks(claims):
    g = P2.GRID
    out = []
    if "ht" in claims:
        for r_f, nb, nc, nbank in itertools.product(g["r_f"], g["n_blk"], g["n_cal"], g["n_bank"]):
            if PV.graph(r_f, nb) is None:
                continue
            for s in PB.HT_SELECT:
                out.append(("ht", {"r_f": r_f, "n_blk": nb, "n_cal": nc, "n_bank": nbank}, s))
    if "e1" in claims:
        for nr, B, nd, nt, ng in itertools.product(
            g["n_rep"], g["B"], g["n_dev"], g["n_test"], PB.GEN_GRID
        ):
            for s in PB.E1_SELECT + PB.E1_SENSITIVITY:
                out.append(
                    ("e1", {"n_rep": nr, "B": B, "n_dev": nd, "n_test": nt, "n_gen_syc": ng}, s)
                )
    return out


_INP, _CT = None, None


def _init(inp, ct):
    import torch

    torch.set_num_threads(1)
    global _INP, _CT
    _INP, _CT = dict(inp, se_ratio=np.asarray(inp["se_ratio"])), ct


def _run(args):
    t, reps, n_flip = args
    r = PB.run_cell(t[0], t[1], t[2], reps, _INP, n_flip=n_flip, ct_alt=_CT)
    r["key"] = _key(t)
    return r


def cmd_grid(a):
    inp = json.loads(Path(a.inputs).read_text())
    claims = a.claims.split(",")
    if "ht" in claims and a.ct_alt is None:
        raise SystemExit("--ct-alt is required for the H-T grid")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_p = out / "results.jsonl"
    done = {}
    if res_p.exists():
        for line in res_p.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    todo = [(t, a.reps, P2.N_FLIP_GRID) for t in tasks(claims) if _key(t) not in done]
    prog, t0, n = out / f"progress-{a.claims}.json", time.time(), 0
    with (
        mp.get_context("spawn").Pool(a.procs, initializer=_init, initargs=(inp, a.ct_alt)) as pool,
        open(res_p, "a") as f,
    ):
        for r in pool.imap_unordered(_run, todo, chunksize=4):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            done[r["key"]] = r
            n += 1
            if n % 50 == 0:
                prog.write_text(
                    json.dumps({"done": n, "todo": len(todo), "elapsed_s": time.time() - t0})
                )
        near = []
        for r in done.values():
            if r["claim"] in claims and r["reps"] < PV.REQ["rerun_reps"] and PV.near_threshold(r):
                k = json.loads(r["key"])
                near.append(((k[0], dict(k[1]), k[2]), PV.REQ["rerun_reps"], P2.N_FLIP_GRID))
        for r in pool.imap_unordered(_run, near, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            n += 1
    prog.write_text(json.dumps({"done": n, "state": "finished", "elapsed_s": time.time() - t0}))


def _load(path):
    best = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["key"] not in best or r["reps"] >= best[r["key"]]["reps"]:
            best[r["key"]] = r
    return best


def _table(res, claim, scen, keys, passes):
    tab = {}
    for r in res.values():
        if r["claim"] == claim:
            tab.setdefault(tuple(r["cell"][k] for k in keys), {})[r["scenario"]] = r
    return {
        k: {"pass": all(passes(v[s]) for s in scen), "rows": v}
        for k, v in tab.items()
        if all(s in v for s in scen)
    }


def pool_demand_v2(d):
    dem = PV.pool_demand(d)
    dem["sycophancy_real"] -= d["n_gen_syc"]
    dem["triviaqa_gen"] = d["n_gen_syc"]
    return dem


def gen_seconds(ph100b: dict, batch: int) -> tuple[float, float]:
    r = ph100b["runs"]
    return r[f"greedy_b{batch}"]["s_per_answer"], r[f"sampled_b{batch}"]["s_per_answer"]


def cmd_select(a):
    inp = json.loads(Path(a.inputs).read_text())
    res = _load(Path(a.out) / "results.jsonl")
    v2 = _load(a.v2)
    aud = json.loads(AUDIT_V1.read_text())["objectives"]
    avail = {o: v["available_groups"] for o, v in aud.items()}
    ph = json.loads(Path(a.ph100b).read_text())
    gg, gs = gen_seconds(ph, a.gen_batch)
    HT = _table(res, "ht", PB.HT_SELECT, ("r_f", "n_blk", "n_cal", "n_bank"), PB.passes)
    if a.ht_refit_failed:  # ruling §4(b): the full refit failed at the selected and next cells
        HT = {k: dict(v, pass_cheap=v["pass"]) for k, v in HT.items()}
        for v in HT.values():
            v["pass"] = False
    E2 = _table(v2, "e2", P2.E2_SELECT, ("n_rep", "B", "n_dev", "n_test"), P2.passes)
    E1 = _table(res, "e1", PB.E1_SELECT, ("n_rep", "B", "n_dev", "n_test", "n_gen_syc"), PV.passes)
    unit = inp["unit"]
    fixed = {"shard_contrast": 16, "shard_lora": 64, "e2_shard": 64, "pool_size": 64, "n_inert": 12}
    g = P2.GRID
    cands, n_cons = [], 0
    for (r_f, nb, nc, nbank), ht in HT.items():
        m = PV.graph(r_f, nb)["n_components"] - nc
        for nr, B, nd, nt, ngs in itertools.product(
            g["n_rep"], g["B"], g["n_dev"], g["n_test"], PB.GEN_GRID
        ):
            e1 = E1.get((nr, B, nd, nt, ngs), {"pass": False})
            e2 = E2.get((nr, B, nd, nt), {"pass": False})
            d = dict(
                fixed,
                r_f=r_f,
                n_blk=nb,
                n_cal=nc,
                n_bank=nbank,
                n_rep=nr,
                B=B,
                n_dev=nd,
                n_test=nt,
                n_gen_syc=ngs,
                n_rep_sec=0,
                n_gen_gsm=200,
                n_gen_math=200,
                n_ep_sec="all",
            )
            n_cons += 1
            dem = pool_demand_v2(d)
            if any(
                n > avail.get(o, 10**9) for o, n in dem.items() if o != "construction_contrast_lora"
            ):
                continue
            pr = PB.price(d, unit, gg, gs)
            if not pr["within_caps"]:
                d["n_ep_sec"] = "half"
                pr = PB.price(d, unit, gg, gs)
            cands.append((d, pr, {"ht": ht["pass"], "e1": e1["pass"], "e2": e2["pass"]}, m, nt))
    feas = [c for c in cands if c[1]["within_caps"]]
    sel = PB.select(cands)
    per_rank = {}
    for cs in PB.PRIORITY:
        pool = [c for c in feas if all(c[2][x] for x in cs)]
        if pool:
            d, pr = min(((c[0], c[1]) for c in pool), key=lambda t: t[1]["total_gpu_h"])
            per_rank["+".join(cs)] = {
                "gpu_h": pr["total_gpu_h"],
                "usd_9b": pr["usd_9b"],
                "design": d,
            }
        else:
            per_rank["+".join(cs)] = None
    if sel is None:
        run, design, price = (), None, None
    else:
        _i, run, design, price = sel
    claims_all = ("ht", "e1", "e2")
    powerable = {
        c: any(c in cs and per_rank["+".join(cs)] is not None for cs in PB.PRIORITY)
        for c in claims_all
    }
    out = {
        "label": "PREREG_WME v2b (RULING-wme-v2b) cheap grid and A8 selection",
        "requirements": PV.REQ,
        "caps": PB.CAPS,
        "usd_per_gpu_h": PB.USD_PER_GPU_H,
        "usd_overhead": PB.USD_OVERHEAD,
        "priority": ["+".join(cs) for cs in PB.PRIORITY],
        "gen_grid": list(PB.GEN_GRID),
        "generation": {
            "batch": a.gen_batch,
            "greedy_s_per_answer": gg,
            "sampled_s_per_answer": gs,
            "source": "P-H100b genbatch",
        },
        "ht_ct_alt": next(iter(r["ct_alt"] for r in res.values() if r["claim"] == "ht"), None),
        "ht_refit_failed": bool(a.ht_refit_failed),
        "n_designs_considered": n_cons,
        "n_designs_feasible_within_caps": len(feas),
        "pass_counts": {
            "ht": sum(v["pass"] for v in HT.values()),
            "ht_cells": len(HT),
            "e1": sum(v["pass"] for v in E1.values()),
            "e1_cells": len(E1),
            "e2": sum(v["pass"] for v in E2.values()),
            "e2_cells": len(E2),
        },
        "cheapest_per_rank": per_rank,
        "powerable": powerable,
        "claims_run": list(run),
        "unresolved_not_powered": [c for c in claims_all if c not in run] + ["e1_secondaries"],
        "selected": design,
        "price": price,
        "m_heldout_components": None
        if design is None
        else PV.graph(design["r_f"], design["n_blk"])["n_components"] - design["n_cal"],
        "pools": None if design is None else {"demand": pool_demand_v2(design), "available": avail},
    }
    if design is not None:
        sel_rows = {}
        for nm, tab, keys in (
            ("ht", HT, ("r_f", "n_blk", "n_cal", "n_bank")),
            ("e2", E2, ("n_rep", "B", "n_dev", "n_test")),
            ("e1", E1, ("n_rep", "B", "n_dev", "n_test", "n_gen_syc")),
        ):
            ck = tuple(design[x] for x in keys)
            sel_rows[nm] = {
                s: {k: v for k, v in r.items() if k not in ("cell", "key")}
                for s, r in tab.get(ck, {"rows": {}})["rows"].items()
            }
        e1c = tuple(design[x] for x in ("n_rep", "B", "n_dev", "n_test", "n_gen_syc"))
        sens = res.get(
            _key(
                (
                    "e1",
                    dict(zip(("n_rep", "B", "n_dev", "n_test", "n_gen_syc"), e1c, strict=True)),
                    "alt",
                )
            )
        )
        out["selected_cell_results"] = sel_rows
        out["e1_sensitivity_full_effect"] = (
            None if sens is None else {k: sens[k] for k in ("rate", "power_lb", "reps")}
        )
        out["reader"] = {
            "cap": PV.reader_cap(design["pool_size"], design["B"]),
            "usd": {
                mm: PV.reader_usd(design["pool_size"], design["B"], 3 * design["n_rep"], mm)
                for mm in PV.CLAUDE_PRICES
            },
        }
    e1_by_n = {}
    for (nr, B, nd, nt, ng), v in E1.items():
        if v["pass"]:
            e1_by_n.setdefault(ng, 0)
            e1_by_n[ng] += 1
    out["e1_passing_cells_by_n_gen"] = e1_by_n
    Path(a.report).write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "claims_run",
                    "unresolved_not_powered",
                    "selected",
                    "price",
                    "pass_counts",
                    "cheapest_per_rank",
                    "e1_passing_cells_by_n_gen",
                )
            },
            indent=1,
            default=float,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "select"])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--claims", default="e1")
    ap.add_argument("--ct-alt", type=float, default=None)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--v2", default=str(ROOT / "data/wme/powersim_v2/run-20261117/results.jsonl"))
    ap.add_argument("--ph100b", default="")
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--refit", default="")
    ap.add_argument("--ht-refit-failed", action="store_true")
    ap.add_argument(
        "--report", default=str(ROOT / "docs/reviews/wme/wme-powersim-v2b-2026-11-17.json")
    )
    a = ap.parse_args()
    {"grid": cmd_grid, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
