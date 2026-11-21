#!/usr/bin/env python3
"""PREREG_WMM power simulation (CPU). Inputs: Gate 1 dev pairs (exposed data, sizing only) and the sealed panel sizes
in ``docs/prereg/wmm_manifest.json``. Runs the registered claim functions (``wmm.prereg``) on simulated data and the
round-11 analytic sizing. Writes ``docs/prereg/wmm_powersim.json``.

    wmm_powersim.py [--reps 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wsp.wm.wmm import gate1 as G1
from wsp.wm.wmm import powersim as S
from wsp.wm.wmm import prereg as P

GATE1 = ROOT / "data/wmm/gate1-2026-11-20"


def rd(name):
    return json.loads((GATE1 / f"{name}.json").read_text())


def gate1_inputs():
    bs = rd("state_base_s")["sets"]
    b = rd("state_base")["sets"]
    s1 = rd("state_syc_lora@1")["sets"]
    ab, ae = np.array(bs["mq_syc"]["adopted"], int), np.array(s1["mq_syc"]["adopted"], int)
    cb = np.concatenate([np.array(bs[n]["correct"], int) for n in ("mq", "mm")])
    ce = np.concatenate([np.array(s1[n]["correct"], int) for n in ("mq", "mm")])
    sq = []
    for n in ("mq", "mm"):
        for xb, xe, yb, ye in zip(
            bs[n]["conf"], s1[n]["conf"], bs[n]["correct"], s1[n]["correct"], strict=True
        ):
            if xb is not None and xe is not None:
                sq.append((xe - ye) ** 2 - (xb - yb) ** 2)
    hb, he = np.array(b["harm"]["refused"], int), np.array(s1["harm"]["refused"], int)
    bb, be = np.array(b["benign"]["refused"], int), np.array(s1["benign"]["refused"], int)
    gb, ge = np.array(b["general"]["correct"], int), np.array(s1["general"]["correct"], int)
    triples = tuple(
        np.concatenate([np.array(bs[n][k], int) for n in ("mq", "mm")])
        for k in ("correct", "length", "truncated")
    )
    # M-TE discordance references at budget 1,024 (edits that shorten answers)
    lowd = {}
    for tag in ("interp@0.25", "tv@0.9", "h1_lora@1"):
        e = rd(f"state_{tag}")["sets"]
        a = np.concatenate(
            [
                P.correct_at(bs[n]["correct"], bs[n]["length"], bs[n]["truncated"], 1024)
                for n in ("mq", "mm")
            ]
        )
        c = np.concatenate(
            [
                P.correct_at(e[n]["correct"], e[n]["length"], e[n]["truncated"], 1024)
                for n in ("mq", "mm")
            ]
        )
        lowd[tag] = {"k": int(np.sum(a != c)), "n": len(a), "gain": float(np.mean(c - a))}
    return {
        "adoption": {
            "k": int(np.sum(ab != ae)),
            "n": len(ab),
            "m": float(np.mean(ae - ab)),
            "base_rate": float(ab.mean()),
            "unparsed": float(np.mean([x is None for x in bs["mq_syc"]["letter"]])),
        },
        "neutral": {
            "k": int(np.sum(cb != ce)),
            "n": len(cb),
            "m": float(np.mean(ce - cb)),
            "base_rate": float(cb.mean()),
        },
        "brier": {"n": len(sq), "m": float(np.mean(sq)), "sd": float(np.std(sq, ddof=1))},
        "harm_compliance": {
            "k": int(np.sum(hb != he)),
            "n": len(hb),
            "m": float(np.mean((1 - he) - (1 - hb))),
            "base_rate": float(1 - hb.mean()),
        },
        "benign_over_refusal": {
            "k": int(np.sum(bb != be)),
            "n": len(bb),
            "m": float(np.mean(be - bb)),
            "base_rate": float(bb.mean()),
        },
        "general_accuracy": {
            "k": int(np.sum(gb != ge)),
            "n": len(gb),
            "m": float(np.mean(ge - gb)),
            "base_rate": float(gb.mean()),
        },
        "triples": triples,
        "low_budget_1024": lowd,
    }


CALIB = ROOT / "docs/prereg/wmm_gcal_calibration.json"


def gcal_spec(man, calib: dict, m: float) -> dict:
    """C4: G-CAL over all mq + mm neutral items (no parse condition), SD = the 97.5 % upper bound s_U from the
    committed calibration run, mean m (0 at null, the dev ΔBS in the dev scenario, +0.01 for the sensitivity check)."""
    return {
        "n": man["panels"]["mq"]["n"] + man["panels"]["mm"]["n"],
        "m": m,
        "sd": calib["sd_dBS_upper_97_5"],
    }


def guards_spec(inp, man, calib, null: bool, gcal_m=None):
    n = {
        k: man["panels"][p]["n"]
        for k, p in (
            ("harm_compliance", "harm"),
            ("benign_over_refusal", "benign"),
            ("general_accuracy", "gen"),
        )
    }
    g = {}
    for k in ("harm_compliance", "benign_over_refusal", "general_accuracy"):
        x = inp[k]
        d = S.wilson_upper(x["k"], x["n"])
        m = 0.0 if null else x["m"]
        # conservative: keep the Wilson-upper discordance; raise the base rate to the smallest feasible value
        g[k] = {"n": n[k], "d": d, "m": m, "base_rate": max(x["base_rate"], (d - m) / 2 + 1e-9)}
    if gcal_m is None:
        gcal_m = 0.0 if null else calib["dev_mean_dBS"]
    g["gcal"] = gcal_spec(man, calib, gcal_m)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--ma-reps", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    man = json.loads((ROOT / "docs/prereg/wmm_manifest.json").read_text())
    calib = json.loads(CALIB.read_text())
    inp = gate1_inputs()
    n_mq, n_mm = man["panels"]["mq"]["n"], man["panels"]["mm"]["n"]
    res = {
        "status": "EXPLORATORY M′ (RULING-wmm-prereg); alpha_f fixed at 0.025/3",
        "inputs": {k: v for k, v in inp.items() if k != "triples"},
        "gcal_calibration": {k: calib[k] for k in ("sd_dBS", "sd_dBS_upper_97_5", "dev_mean_dBS")},
        "panel_n": {k: v["n"] for k, v in man["panels"].items()},
        "alpha_f": P.ALPHA_F,
    }
    ad, ne = inp["adoption"], inp["neutral"]
    d_ad = S.wilson_upper(ad["k"], ad["n"])
    d_ne = S.wilson_upper(ne["k"], ne["n"])
    res["round11_sizing"] = {
        "msyc_adoption": {
            "d_wilson_upper": d_ad,
            "dev_gain": ad["m"],
            "N_half_gain": G1.sizing_n(d_ad, abs(ad["m"]) / 2, alpha=P.ALPHA_F),
            "N_full_gain": G1.sizing_n(d_ad, abs(ad["m"]), alpha=P.ALPHA_F),
            "available": n_mq,
        },
        "mte_low_budget_1024": {
            tag: {
                "d_wilson_upper": S.wilson_upper(x["k"], x["n"]),
                "dev_gain": x["gain"],
                "N_delta_0.05_holm_first": G1.sizing_n(
                    S.wilson_upper(x["k"], x["n"]), 0.05, alpha=P.ALPHA_F / 2
                ),
            }
            for tag, x in inp["low_budget_1024"].items()
        }
        | {"available": n_mq + n_mm},
    }

    def msyc_spec(m_ad, null, gcal_m=None, neutral_m=None):
        return {
            "adoption": {
                "n": n_mq,
                "d": d_ad,
                "m": m_ad,
                "base_rate": ad["base_rate"],
                "unparsed": ad["unparsed"],
            },
            "neutral": {
                "n": n_mq + n_mm,
                "d": d_ne,
                "m": (0.0 if null else ne["m"]) if neutral_m is None else neutral_m,
                "base_rate": ne["base_rate"],
            },
            "guards": guards_spec(inp, man, calib, null, gcal_m),
            "alpha_f": P.ALPHA_F,
        }

    msyc = {}
    for label, m_ad, null in (
        ("half_effect_null_guards", ad["m"] / 2, True),
        ("dev_effect_null_guards", ad["m"], True),
        ("half_effect_dev_guards", ad["m"] / 2, False),
        ("dev_effect_dev_guards", ad["m"], False),
    ):
        msyc[label] = S.msyc_power(msyc_spec(m_ad, null), reps=a.reps, seed=1)
    msyc["sensitivity_gcal_at_margin"] = S.msyc_power(
        msyc_spec(ad["m"] / 2, True, gcal_m=0.01), reps=a.reps, seed=1
    )
    res["msyc_power"] = msyc
    res["gcal_checks"] = {
        "pass_at_null": msyc["half_effect_null_guards"]["gcal"],
        "fail_at_margin": 1 - msyc["sensitivity_gcal_at_margin"]["gcal"],
        "required": {"pass_at_null": 0.95, "fail_at_margin": 0.97},
    }

    p = float(np.mean(inp["triples"][0]))
    q = d_ne / (2 * p * (1 - p))
    mte = {}
    for mu in (-0.5, -1.0, -1.5):
        for dfull in (0.0, -0.01):
            spec = {
                "n": n_mq + n_mm,
                "mu": mu,
                "sd": 0.58,
                "q": q,
                "delta_full": dfull,
                "guards": guards_spec(inp, man, calib, True),
                "alpha_f": P.ALPHA_F,
            }
            mte[f"mu{mu}|dfull{dfull}"] = S.mte_power(
                spec, inp["triples"], reps=max(500, a.reps // 2), seed=2
            )
    res["mte_power"] = {"q_redraw": q, "log_len_sd": 0.58, "scenarios": mte}

    ma = {}
    for spread in (0.5, 1.0, 2.0):
        for r in (0.3, 0.5, 0.7):
            ma[f"spread{spread}|r{r}"] = S.ma_power(
                n_items=man["panels"]["ma"]["n"],
                d_item=0.18,
                spread_se=spread,
                r_feature=r,
                reps=a.ma_reps,
                seed=3,
            )
    res["ma_power"] = ma
    res["seconds"] = time.time() - t0
    out = ROOT / "docs/prereg/wmm_powersim.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(
        json.dumps(
            {
                "msyc": {
                    k: {q: v[q] for q in ("endpoint_only", "pass", "gcal")} for k, v in msyc.items()
                },
                "gcal_checks": res["gcal_checks"],
                "mte": {k: {q: v[q] for q in ("endpoint_only", "pass")} for k, v in mte.items()},
                "ma": {k: v["power"] for k, v in ma.items()},
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
