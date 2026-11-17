#!/usr/bin/env python3
"""PREREG_WME v1 §8 outcome-blind simulation: inputs, the cheap grid and the §8.5 selection. CPU only.

    inputs  --ph100 <summary.json> --out <dir>/inputs.json      noise and rate inputs (P-H100 + pilot-2 E2 spread)
    grid    --inputs ... --out <dir> --procs 6 [--reps 1000]   every grid cell x selection scenario (resumable)
    select  --inputs ... --out <dir> [--report <json>]          per-claim pass tables, feasibility, pricing, choice,
                                                                 descriptive scenarios and the supplement choice

Noise inputs are P-H100 paired SEs (validation n = 64, H100) and pilot-2 exploratory quantities used only as noise
inputs (the de-noised MoM SD over all 32 sycophancy candidates; MAD as sensitivity). Pool availability comes from the
hash-only exposure and supplement audits. No wme-c1 outcome exists or is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wsp.wm import powersim_v1 as PV

AUDIT = ROOT / "docs/reviews/wme/wme-c1-exposure-audit-2026-11-17.json"
SUPP = ROOT / "docs/reviews/wme/wme-supplement-audit-2026-11-17.json"
PILOT2 = ROOT / "docs/reviews/wme/wme-pilot2-2026-11-16.json"
TRIVIAQA_AVAILABLE = (
    256  # sycophancy groups from TriviaQA after exclusion (hash-only count, 17 Nov)
)
GUARDS = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mathcomp_gold_lp",
    "mmlupro_gold_lp",
    "syc_neutral",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
)


def _mom(rows):
    g = np.array([r["gain"] for r in rows])
    se = np.array([r["se"] for r in rows])
    return float(math.sqrt(max(0.0, g.var(ddof=1) - np.mean(se**2)))), float(
        1.4826 * np.median(np.abs(g - np.median(g)))
    )


def cmd_inputs(a):
    """Noise inputs: SEs scale with the edit, so the candidate-level SEs come from the pilot-2 E2 candidates (dev,
    n = 128; the same edit families as the E2 library) and the H-T static scale from the pilot-2 T2 states, both
    converted to the H100 platform by the measured same-edit SE ratio H100 / RTX (``--ratio``); native guard SEs
    and every rate come from P-H100 directly."""
    s = json.loads(Path(a.ph100).read_text())
    ratio = {
        k: v["median"]
        for k, v in json.loads(Path(a.ratio).read_text())[
            "se_ratio_h100_over_rtx_same_edits"
        ].items()
    }
    p2 = json.loads(PILOT2.read_text())
    e2 = p2["E2"]["9b"]["objectives"]
    sd, mad = _mom(e2["sycophancy_real"]["rows"])
    med = lambda rows: float(np.median([r["se"] for r in rows]))  # noqa: E731
    syc_se_64 = med(e2["sycophancy_real"]["rows"]) * math.sqrt(2) * ratio["sycophancy_real"]
    sec = {}
    for o in ("gsm8k_gold_lp", "mathcomp_gold_lp"):
        sd_o, mad_o = _mom(e2[o]["rows"])
        sec[o] = {
            "se_64": med(e2[o]["rows"]) * math.sqrt(2) * ratio[o],
            "sd_cand": sd_o,
            "sd_cand_mad": mad_o,
        }
    gpp = {}
    for r in e2["sycophancy_real"]["rows"]:
        for g, v in r["guards"].items():
            gpp.setdefault(g, []).append(v["se"])
    se_h = {k: v["median_se_n64"] for k, v in s["se"].items()}
    guard = {}
    for g in GUARDS:
        pp = (
            float(np.median(gpp[g])) * math.sqrt(2) * ratio.get(g, 1.0)
            if g in gpp
            else (s["se_pp"].get(g, {}).get("median_se_n64", 0.001))
        )
        guard[g] = {"native": se_h[g], "pp": pp}
    ex = s.get("extras", {})
    ut = s["unit_times"]["validation"]
    allw = ex.get("oracle_grad_allwriters", {})
    g4 = ex.get("oracle_grad_group4", {})
    oracle = allw["seconds"] if allw.get("ok") else 8 * g4.get("seconds", float("nan"))
    field = ex.get("field_kfac_group4", {}).get("seconds", float("nan"))
    gen = {o: v["s_per_answer"] for o, v in ex.get("generation", {}).items()}
    loc = p2["ledger"]["rows"]
    t2_local = next(r["wall_seconds"] for r in loc if r["phase"] == "t2") / p2["T2"]["n_states"]
    h_state = sum(ut["per_item_s"][o] * 64 for o in PV.OBJ6) + ut["overhead_per_unit_s"] * 6
    speed = h_state / t2_local
    lora_local = [r["wall_seconds"] for r in loc if "lora" in r["phase"]]
    lora_s = (float(np.median(lora_local)) if lora_local else 900.0) * speed
    mean_item = float(np.mean(list(ut["per_item_s"].values())))
    inp = {
        "label": "v1 simulation inputs (P-H100 rates and native SEs; pilot-2 exploratory SEs and spread used only "
        "as noise inputs, converted to H100 by the measured same-edit SE ratio)",
        "ph100_summary": Path(a.ph100).name,
        "se_ratio_h100_over_rtx": ratio,
        # multiplier of the pilot-2 static |error| scale in SE units: a smaller SE on H100 means larger SE units
        "se_ratio": [1.0 / ratio[o] for o in PV.OBJ6],
        "syc_se_64": syc_se_64,
        "sd_cand": sd,
        "sd_cand_mad": mad,
        "guard_se_64": guard,
        "guard_point_sd_item": guard["gsm8k_gold_lp"]["pp"] * 8.0,
        "secondary": sec,
        "unit": {
            "per_item_s": ut["per_item_s"],
            "overhead_s": ut["overhead_per_unit_s"],
            "oracle_s": oracle,
            "field_s": 8 * field,
            "dirs_s_per_root": 2 * 32 * mean_item,
            "lora_s_per_root": lora_s,
            "library_s": 24 * 2 * 128 * mean_item + 4 * 3 * lora_s,
            "lora_sft_s": lora_s,
            "gen_s": gen,
            "speed_h100_over_local_panel": speed,
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(inp, indent=1, sort_keys=True))
    print(
        json.dumps(
            {k: inp[k] for k in ("se_ratio", "syc_se_64", "sd_cand", "sd_cand_mad", "unit")},
            indent=1,
        )
    )


def _np_inp(inp):
    return dict(inp, se_ratio=np.asarray(inp["se_ratio"]))


def tasks():
    g = PV.GRID
    out = []
    for r_f, nb, nc, nbank in itertools.product(g["r_f"], g["n_blk"], g["n_cal"], g["n_bank"]):
        if PV.graph(r_f, nb) is None:
            continue
        for s in PV.HT_SELECT:
            out.append(("ht", {"r_f": r_f, "n_blk": nb, "n_cal": nc, "n_bank": nbank}, s))
    for nr, B, nd, nt, ng in itertools.product(
        g["n_rep"], g["B"], g["n_dev"], g["n_test"], g["n_gen_syc"]
    ):
        for s in PV.E1_SELECT:
            out.append(("e1", {"n_rep": nr, "B": B, "n_dev": nd, "n_test": nt, "n_gen_syc": ng}, s))
    for nr, B, nd, nt in itertools.product(g["n_rep"], g["B"], g["n_dev"], g["n_test"]):
        for s in PV.E2_SELECT:
            out.append(("e2", {"n_rep": nr, "B": B, "n_dev": nd, "n_test": nt}, s))
    for o in ("gsm8k_gold_lp", "mathcomp_gold_lp"):
        gk = "n_gen_gsm" if o == "gsm8k_gold_lp" else "n_gen_math"
        for rs, B, nd, nt, ng in itertools.product((1, 2), g["B"], g["n_dev"], g["n_test"], g[gk]):
            for s in PV.E1S_SCENARIOS:
                out.append(
                    (f"e1s:{o}", {"n_rep_sec": rs, "B": B, "n_dev": nd, "n_test": nt, gk: ng}, s)
                )
    return out


def _key(t):
    return json.dumps([t[0], sorted(t[1].items()), t[2]])


_INP = None


def _init(inp):
    global _INP
    import torch

    torch.set_num_threads(1)
    _INP = _np_inp(inp)


def _run(args):
    t, reps = args
    claim, cell, scen = t
    if claim.startswith("e1s:"):
        o = claim.split(":", 1)[1]
        c = dict(cell)
        c.setdefault("n_gen_gsm", 200)
        c.setdefault("n_gen_math", 200)
        r = PV.run_secondary_cell(o, c, scen, reps, _INP)
        r["cell"] = cell
    else:
        r = PV.run_cell(claim, cell, scen, reps, _INP)
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
    ts = tasks()
    todo = [(t, a.reps) for t in ts if _key(t) not in done or done[_key(t)]["reps"] < a.reps]
    prog = out / "progress.json"
    t0 = time.time()
    n = 0
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
        # rerun cells within 2 MC SE of a threshold at 4,000 replicates (§8.4)
        near = [
            (json.loads(r["key"]), PV.REQ["rerun_reps"])
            for r in done.values()
            if r["reps"] < PV.REQ["rerun_reps"] and PV.near_threshold(r)
        ]
        near = [((k[0], dict(k[1]), k[2]), reps) for k, reps in near] if not a.no_rerun else []
        for r in pool.imap_unordered(_run, near, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            n += 1
            prog.write_text(
                json.dumps(
                    {
                        "done": n,
                        "todo": len(todo) + len(near),
                        "elapsed_s": time.time() - t0,
                        "phase": "rerun",
                    }
                )
            )
    prog.write_text(json.dumps({"done": n, "elapsed_s": time.time() - t0, "state": "finished"}))


def load_results(path):
    best = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["key"] not in best or r["reps"] >= best[r["key"]]["reps"]:
            best[r["key"]] = r
    return best


def pass_table(res, claim, scenarios, keys):
    tab = {}
    for r in res.values():
        if r["claim"] != claim:
            continue
        ck = tuple(r["cell"][k] for k in keys)
        tab.setdefault(ck, {})[r["scenario"]] = r
    out = {}
    for ck, sc in tab.items():
        if all(s in sc for s in scenarios):
            out[ck] = {"pass": all(PV.passes(sc[s]) for s in scenarios), "rows": sc}
    return out


def available():
    au = json.loads(AUDIT.read_text())["objectives"]
    av = {o: v["available_groups"] for o, v in au.items()}
    supp = json.loads(SUPP.read_text())
    return av, supp


def cmd_select(a):
    inp = json.loads(Path(a.inputs).read_text())
    res = load_results(Path(a.out) / "results.jsonl")
    av, supp = available()
    harm_max = supp["cumulative_available_in_preference_order"]["jbb_behaviors"]
    HT = pass_table(res, "ht", PV.HT_SELECT, ("r_f", "n_blk", "n_cal", "n_bank"))
    E1 = pass_table(res, "e1", PV.E1_SELECT, ("n_rep", "B", "n_dev", "n_test", "n_gen_syc"))
    E2 = pass_table(res, "e2", PV.E2_SELECT, ("n_rep", "B", "n_dev", "n_test"))
    E1S = {
        o: pass_table(
            res,
            f"e1s:{o}",
            tuple(PV.E1S_SCENARIOS),
            (
                "n_rep_sec",
                "B",
                "n_dev",
                "n_test",
                "n_gen_gsm" if o == "gsm8k_gold_lp" else "n_gen_math",
            ),
        )
        for o in ("gsm8k_gold_lp", "mathcomp_gold_lp")
    }
    c2i = {n: PV.c2i_rates(n) for n in PV.GRID["n_inert"]}
    n_inert = min(n for n, r in c2i.items() if r["pass_requirement"])
    c14 = {n: PV.c14_rates(n) for n in PV.GRID["n_cal"]}
    unit = inp["unit"]
    fixed = {
        "shard_contrast": 16,
        "shard_lora": 64,
        "e2_shard": 64,
        "pool_size": 64,
        "n_inert": n_inert,
    }

    def feasible(d):
        if d["n_gen_syc"] > TRIVIAQA_AVAILABLE:
            return False, "triviaqa"
        dem = PV.pool_demand(d)
        avail = dict(av, refusal_margin_harmful=harm_max)
        for o, n in dem.items():
            if o == "construction_contrast_lora":
                continue
            if n > avail[o]:
                return False, o
        return True, None

    best = {}
    n_cons = n_feas = 0
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
    g = PV.GRID
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
                    ok, _ = feasible(d)
                    if not ok:
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
    sec_key = (target, True) if (target, True) in best else (target, False)
    chosen = best.get(sec_key)
    if (
        chosen is None
    ):  # the powerable claims are not jointly achievable: largest achievable subset, then cost
        cands = sorted(((len(cs), -v[0][0]), cs, s) for (cs, s), v in best.items())
        _, cs, s = cands[-1]
        sec_key = (cs, s)
        chosen = best[sec_key]
    rank, design, price = chosen
    run = sec_key[0]
    report = {
        "label": "PREREG_WME v1 §8 outcome-blind simulation and feasibility (cheap grid + §8.5 selection)",
        "inputs": inp,
        "requirements": PV.REQ,
        "alpha_one_sided": PV.ALPHA,
        "caps": PV.CAPS,
        "assumptions": PV.ASSUMPTIONS,
        "n_designs_considered": n_cons,
        "n_designs_feasible_within_caps": n_feas,
        "powerable": powerable,
        "claims_run": list(run),
        "e1_secondaries_run": bool(sec_key[1]),
        "unresolved_not_powered": [c for c in ("ht", "e1", "e2") if c not in run]
        + ([] if sec_key[1] else ["e1_secondaries"]),
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
            **{f"e1s:{o}": sum(v["pass"] for v in t.values()) for o, t in E1S.items()},
        },
        "c2i": c2i,
        "c14": c14,
        "pools": {
            "available": av,
            "harmful_with_all_supplements": harm_max,
            "triviaqa_available": TRIVIAQA_AVAILABLE,
            "demand": PV.pool_demand(design),
        },
        "reader": {
            "cap": PV.reader_cap(design["pool_size"], design["B"]),
            "tokens_per_episode": PV.reader_tokens(design["pool_size"], design["B"]),
            "usd": {
                m: PV.reader_usd(design["pool_size"], design["B"], 3 * design["n_rep"], m)
                for m in PV.CLAUDE_PRICES
            },
        },
    }
    dem_h = PV.pool_demand(design)["refusal_margin_harmful"]
    from wsp.wm import supplement as SU

    report["supplement"] = SU.choose(
        supp["existing_harmful_available_groups"],
        dem_h,
        {n: {"available_groups": v["available_groups"]} for n, v in supp["sources"].items()},
    )
    # m_min_HT: smallest held-out component count at which every simulated cell meets the coverage requirement
    cov = sorted(
        (PV.graph(k[0], k[1])["n_components"] - k[2], v["rows"]["null_nobetter"]["coverage"])
        for k, v in HT.items()
    )
    m_min = None
    for m, _c in cov:
        if all(c >= PV.REQ["coverage_min"] for mm, c in cov if mm >= m):
            m_min = m
            break
    report["m_min_HT"] = m_min
    # the selected cell's per-scenario results
    sel = {}
    for k, tab, keys in (
        ("ht", HT, ("r_f", "n_blk", "n_cal", "n_bank")),
        ("e2", E2, ("n_rep", "B", "n_dev", "n_test")),
        ("e1", E1, ("n_rep", "B", "n_dev", "n_test", "n_gen_syc")),
    ):
        ck = tuple(design[x] for x in keys)
        sel[k] = {
            s: {kk: r[kk] for kk in r if kk not in ("cell", "key")}
            for s, r in tab.get(ck, {"rows": {}})["rows"].items()
        }
    report["selected_cell_results"] = sel
    # nearest powered alternatives per claim (cost-ordered) for the record
    report["cheapest_per_claim_set"] = {
        "+".join(cs) + ("+sec" if s else ""): {"gpu_h": v[0][0], "design": v[1]}
        for (cs, s), v in best.items()
    }
    rp = Path(a.report)
    rp.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
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
                    "m_min_HT",
                    "supplement",
                    "pass_counts",
                )
            },
            indent=1,
            default=float,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inputs", "grid", "select"])
    ap.add_argument("--ph100")
    ap.add_argument("--ratio", default=str(ROOT / "data/wme/ph100/platform_ratio.json"))
    ap.add_argument("--inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument(
        "--no-rerun",
        action="store_true",
        help="skip the 4,000-replicate reruns (pipeline smoke only)",
    )
    ap.add_argument(
        "--report", default=str(ROOT / "docs/reviews/wme/wme-powersim-v1-2026-11-17.json")
    )
    a = ap.parse_args()
    {"inputs": cmd_inputs, "grid": cmd_grid, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
