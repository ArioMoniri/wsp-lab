#!/usr/bin/env python3
"""Cheap grid and §8.5 selection under RULING-wme-v2 (A2-A4, A6). CPU only.

    grid    --inputs <v1 inputs.json> --v1 <v1 results.jsonl> --out <dir> --procs 6 [--reps 1000]
    select  --inputs ... --v1 ... --out <dir> --report <json>

H-T and E2 cells are re-simulated with the v2 inference (sign-flip tests); E1 cells are simulated only for the new
generation sizes (the E1 model and inputs are unchanged, so v1's results for n_gen <= 250 and the E1 secondaries are
reused, not re-run). Availability: the v1-policy re-audit (harmful with the AdvBench + HarmBench supplement; the
TriviaQA generation pool for n_gen,syc, which no longer draws on the sycophancy_real pool). Inputs, alternatives and
pricing are v1's (not re-tuned).
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

AUDIT_V1 = ROOT / "docs/reviews/wme/wme-c1-exposure-audit-v1-2026-11-17.json"
GUARDS = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mathcomp_gold_lp",
    "mmlupro_gold_lp",
    "syc_neutral",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
)


def _key(t):
    return json.dumps([t[0], sorted(t[1].items()), t[2]])


def tasks():
    g = P2.GRID
    out = []
    for r_f, nb, nc, nbank in itertools.product(g["r_f"], g["n_blk"], g["n_cal"], g["n_bank"]):
        if PV.graph(r_f, nb) is None:
            continue
        for s in P2.HT_SELECT:
            out.append(("ht", {"r_f": r_f, "n_blk": nb, "n_cal": nc, "n_bank": nbank}, s))
    for nr, B, nd, nt in itertools.product(g["n_rep"], g["B"], g["n_dev"], g["n_test"]):
        for s in P2.E2_SELECT:
            out.append(("e2", {"n_rep": nr, "B": B, "n_dev": nd, "n_test": nt}, s))
    new_gen = [n for n in P2.GEN_GRID if n not in PV.GRID["n_gen_syc"]]
    for nr, B, nd, nt, ng in itertools.product(
        g["n_rep"], g["B"], g["n_dev"], g["n_test"], new_gen
    ):
        for s in PV.E1_SELECT:
            out.append(("e1", {"n_rep": nr, "B": B, "n_dev": nd, "n_test": nt, "n_gen_syc": ng}, s))
    return out


_INP = None


def _init(inp):
    import torch

    torch.set_num_threads(1)
    global _INP
    _INP = dict(inp, se_ratio=np.asarray(inp["se_ratio"]))


def _run(args):
    t, reps, n_flip = args
    r = P2.run_cell(t[0], t[1], t[2], reps, _INP, n_flip=n_flip)
    r["key"] = _key(t)
    return r


def cmd_grid(a):
    inp = json.loads(Path(a.inputs).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_p = out / "results.jsonl"
    done = {}
    if res_p.exists():
        for line in res_p.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    todo = [(t, a.reps, P2.N_FLIP_GRID) for t in tasks() if _key(t) not in done]
    prog, t0, n = out / "progress.json", time.time(), 0
    with (
        mp.get_context("spawn").Pool(a.procs, initializer=_init, initargs=(inp,)) as pool,
        open(res_p, "a") as f,
    ):
        for r in pool.imap_unordered(_run, todo, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            done[r["key"]] = r
            n += 1
            prog.write_text(
                json.dumps({"done": n, "todo": len(todo), "elapsed_s": time.time() - t0})
            )
        near = []
        for r in done.values():
            if r["reps"] < PV.REQ["rerun_reps"] and PV.near_threshold(r):
                k = json.loads(r["key"])
                near.append(((k[0], dict(k[1]), k[2]), PV.REQ["rerun_reps"], P2.N_FLIP_GRID))
        for r in pool.imap_unordered(_run, near, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            n += 1
            prog.write_text(
                json.dumps(
                    {
                        "done": n,
                        "todo": len(todo) + len(near),
                        "phase": "rerun",
                        "elapsed_s": time.time() - t0,
                    }
                )
            )
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
    dem["sycophancy_real"] -= d["n_gen_syc"]  # generation now draws on the TriviaQA pool
    dem["triviaqa_gen"] = d["n_gen_syc"]
    return dem


def cmd_select(a):
    inp = json.loads(Path(a.inputs).read_text())
    v2 = _load(Path(a.out) / "results.jsonl")
    v1 = _load(a.v1)
    aud = json.loads(AUDIT_V1.read_text())["objectives"]
    avail = {o: v["available_groups"] for o, v in aud.items()}
    HT = _table(v2, "ht", P2.HT_SELECT, ("r_f", "n_blk", "n_cal", "n_bank"), P2.passes)
    E2 = _table(v2, "e2", P2.E2_SELECT, ("n_rep", "B", "n_dev", "n_test"), P2.passes)
    e1rows = {
        **{k: r for k, r in v1.items() if r["claim"] == "e1"},
        **{k: r for k, r in v2.items() if r["claim"] == "e1"},
    }
    E1 = _table(
        e1rows, "e1", PV.E1_SELECT, ("n_rep", "B", "n_dev", "n_test", "n_gen_syc"), PV.passes
    )
    E1S = {
        o: _table(
            v1,
            f"e1s:{o}",
            tuple(PV.E1S_SCENARIOS),
            (
                "n_rep_sec",
                "B",
                "n_dev",
                "n_test",
                "n_gen_gsm" if o == "gsm8k_gold_lp" else "n_gen_math",
            ),
            PV.passes,
        )
        for o in ("gsm8k_gold_lp", "mathcomp_gold_lp")
    }
    unit = inp["unit"]
    fixed = {"shard_contrast": 16, "shard_lora": 64, "e2_shard": 64, "pool_size": 64, "n_inert": 12}
    g = P2.GRID
    best, n_cons, n_feas = {}, 0, 0
    claim_sets = [
        ("ht", "e1", "e2"),
        ("ht", "e2"),
        ("ht", "e1"),
        ("e1", "e2"),
        ("ht",),
        ("e1",),
        ("e2",),
        (),
    ]
    for (r_f, nb, nc, nbank), ht in HT.items():
        for nr, B, nd, nt, ngs in itertools.product(
            g["n_rep"], g["B"], g["n_dev"], g["n_test"], g["n_gen_syc"]
        ):
            e1 = E1.get((nr, B, nd, nt, ngs), {"pass": False})
            e2 = E2.get((nr, B, nd, nt), {"pass": False})
            for rs in g["n_rep_sec"]:
                gg = (
                    [(200, 200)]
                    if rs == 0
                    else list(itertools.product(g["n_gen_gsm"], g["n_gen_math"]))
                )
                for ngg, ngm in gg:
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
                        n_rep_sec=rs,
                        n_gen_gsm=ngg,
                        n_gen_math=ngm,
                        n_ep_sec="all",
                    )
                    n_cons += 1
                    dem = pool_demand_v2(d)
                    if any(
                        n > avail.get(o, 10**9)
                        for o, n in dem.items()
                        if o != "construction_contrast_lora"
                    ):
                        continue
                    pr = PV.price(d, unit)
                    if not pr["within_caps"]:
                        d["n_ep_sec"] = "half"
                        pr = PV.price(d, unit)
                        if not pr["within_caps"]:
                            continue
                    n_feas += 1
                    sec_ok = rs == 0 or all(
                        E1S[o].get(
                            (rs, B, nd, nt, ngg if o == "gsm8k_gold_lp" else ngm), {"pass": False}
                        )["pass"]
                        for o in E1S
                    )
                    passed = {"ht": ht["pass"], "e1": e1["pass"], "e2": e2["pass"]}
                    m = PV.graph(r_f, nb)["n_components"] - nc
                    for cs in claim_sets:
                        if all(passed[c] for c in cs) and sec_ok:
                            key = (cs, rs > 0)
                            rank = (pr["total_gpu_h"], -m, -nt)
                            if key not in best or rank < best[key][0]:
                                best[key] = (rank, d, pr)
    powerable = {c: any(c in cs for (cs, _s) in best) for c in ("ht", "e1", "e2")}
    target = tuple(c for c in ("ht", "e1", "e2") if powerable[c])
    key = (target, True) if (target, True) in best else (target, False)
    if key not in best:
        key = max(best, key=lambda k: (len(k[0]), -best[k][0][0]))
    rank, design, price = best[key]
    run = key[0]
    sel = {}
    for nm, tab, keys in (
        ("ht", HT, ("r_f", "n_blk", "n_cal", "n_bank")),
        ("e2", E2, ("n_rep", "B", "n_dev", "n_test")),
        ("e1", E1, ("n_rep", "B", "n_dev", "n_test", "n_gen_syc")),
    ):
        ck = tuple(design[x] for x in keys)
        sel[nm] = {
            s: {k: v for k, v in r.items() if k not in ("cell", "key")}
            for s, r in tab.get(ck, {"rows": {}})["rows"].items()
        }
    best_e1 = max(E1.items(), key=lambda kv: kv[1]["rows"]["alt"]["power_lb"])
    report = {
        "label": "PREREG_WME v2 (RULING-wme-v2) cheap grid and §8.5 selection; alternatives not re-tuned",
        "inputs": inp,
        "requirements": PV.REQ,
        "alpha_one_sided": PV.ALPHA,
        "caps": PV.CAPS,
        "n_flip_grid": P2.N_FLIP_GRID,
        "gen_grid": list(P2.GEN_GRID),
        "availability_v1_policy": avail,
        "n_designs_considered": n_cons,
        "n_designs_feasible_within_caps": n_feas,
        "powerable": powerable,
        "claims_run": list(run),
        "e1_secondaries_run": bool(key[1]),
        "unresolved_not_powered": [c for c in ("ht", "e1", "e2") if c not in run]
        + ([] if key[1] else ["e1_secondaries"]),
        "selected": design,
        "price": price,
        "m_heldout_components": PV.graph(design["r_f"], design["n_blk"])["n_components"]
        - design["n_cal"],
        "pass_counts": {
            "ht": sum(v["pass"] for v in HT.values()),
            "ht_cells": len(HT),
            "e1": sum(v["pass"] for v in E1.values()),
            "e1_cells": len(E1),
            "e2": sum(v["pass"] for v in E2.values()),
            "e2_cells": len(E2),
        },
        "selected_cell_results": sel,
        "best_e1_cell": {
            "cell": dict(zip(("n_rep", "B", "n_dev", "n_test", "n_gen_syc"), best_e1[0])),
            "rows": {
                s: {k: r[k] for k in ("rate", "power_lb", "reps")}
                for s, r in best_e1[1]["rows"].items()
            },
        },
        "cheapest_per_claim_set": {
            "+".join(cs) + ("+sec" if s else ""): {"gpu_h": v[0][0], "design": v[1]}
            for (cs, s), v in best.items()
        },
        "pools": {"demand": pool_demand_v2(design), "available": avail},
        "reader": {
            "cap": PV.reader_cap(design["pool_size"], design["B"]),
            "usd": {
                m: PV.reader_usd(design["pool_size"], design["B"], 3 * design["n_rep"], m)
                for m in PV.CLAUDE_PRICES
            },
        },
    }
    Path(a.report).write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "powerable",
                    "claims_run",
                    "unresolved_not_powered",
                    "selected",
                    "price",
                    "pass_counts",
                    "best_e1_cell",
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
    ap.add_argument("--v1", default=str(ROOT / "data/wme/powersim_v1/run-20261117/results.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument(
        "--report", default=str(ROOT / "docs/reviews/wme/wme-powersim-v2-2026-11-17.json")
    )
    a = ap.parse_args()
    {"grid": cmd_grid, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
