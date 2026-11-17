#!/usr/bin/env python3
"""WM-E pilot 3 report (RULING-wme-pilot3 §7): reads the run artefacts and writes the gate JSON (and, with --md,
the markdown review). EXPLORATORY. CPU only.

    wme_p3_report.py --s data/wme/pilot3/<run>/s --h data/wme/pilot3/<run>/h --ledger <ledger.json> \
        --json docs/reviews/wme/wme-pilot3-2026-11-17.json [--md docs/reviews/wme/wme-pilot3-2026-11-17.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import p3gen as G
from wsp.wm import p3slope as S

OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_gap",
)


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"wme-p3:{label}".encode()).hexdigest()[:8], 16)


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------ stage S


def s_split(c3, c4):
    """{sequence pair id: 'construction'|'test'}; pairs are {sequence, reverse}; seed wme-p3:s-split."""

    def pairs(cs):
        out = sorted({min(tuple(s), tuple(reversed(s))) for s in cs})
        return out

    p3, p4 = pairs(c3), pairs(c4)
    rng = random.Random(seed_for("s-split"))
    cons = set(rng.sample(p3, min(4, len(p3)))) | set(rng.sample(p4, min(2, len(p4))))
    return {p: ("construction" if p in cons else "test") for p in p3 + p4}


def s_steps(sdir: Path) -> list:
    design = json.loads((sdir / "design.json").read_text())
    states = {r["id"]: r for r in jl(sdir / "states.jsonl")}
    steps = {r["id"]: r for r in jl(sdir / "steps.jsonl")}
    static = json.loads((sdir / "static.json").read_text())
    fams = design["families"]
    seqs = [tuple(s) for s in design["c3"] + design["c4"]]
    split = s_split(design["c3"], design["c4"])
    pair_of = lambda s: min(tuple(s), tuple(reversed(s)))
    cons_states, test_states = set(), {}
    for s in seqs:
        pr = pair_of(s)
        for h in range(1, len(s) + 1):
            k = ">".join(s[:h])
            if split[pr] == "construction":
                cons_states.add(k)
            else:
                test_states.setdefault(k, pr)
    pairs = sorted(split)
    out = []
    for key, st in steps.items():
        if key not in states or st["parent"] not in states:
            continue
        par = st["parent"]
        cur, prev = states[key], states[par]
        h = st["h"]
        if h == 1 or key in cons_states:
            sp, cl = "construction", -1
        elif key in test_states:
            sp, cl = "test", pairs.index(test_states[key])
        else:
            sp, cl = "other", -1
        root = key.split(">")[-1]
        rec = {
            "id": key,
            "h": h,
            "family": fams[root],
            "control": fams[root] == "control",
            "split": sp,
            "cluster": cl,
            "dnorm": st["dnorm"],
            "obj": {},
        }
        for o in OBJECTIVES:
            d = np.array(cur["values"][o]) - np.array(prev["values"][o])
            se = float(d.std(ddof=1) / np.sqrt(d.size))
            L = prev["L_out"][key][o]
            R = cur["R"][o]
            r1 = st["r1"][o]
            n24 = design.get("n_field", {}).get(o, min(24, len(r1["J1"])))
            rec["obj"][o] = {
                "obs": float(d.mean()),
                "se": max(se, 1e-12),
                "A0": static[key][o]["F"] - static[par][o]["F"],
                "a_step": static[key][o]["a"] - static[par][o]["a"],
                "oracle": 0.5 * (L + R),
                "L": L,
                "R": R,
                "S1": r1["S1"],
                "S2": r1["S2"],
                "S2a": r1["S2a"],
                "s_hat": r1["s_hat"],
                "J1": float(np.mean(r1["J1"])),
                "J2": float(np.mean(r1["J2"])),
                "X": float(np.mean(r1["X"])),
                "J1_24": float(np.mean(r1["J1"][:n24])),
                "S2_24": float(
                    np.mean(
                        np.array(r1["J1"][:n24])
                        + np.array(r1["X"][:n24])
                        + 0.5 * np.array(r1["J2"][:n24])
                    )
                ),
            }
        out.append(rec)
    return out


def s_report(sdir: Path) -> dict:
    cc = (
        json.loads((sdir / "cost_check.json").read_text())
        if (sdir / "cost_check.json").exists()
        else None
    )
    status = (sdir / "STATUS").read_text().strip() if (sdir / "STATUS").exists() else "missing"
    if not (sdir / "static.json").exists():
        return {"status": status, "cost_check": cc, "measured": False}
    steps = s_steps(sdir)
    gate = S.s_gate(steps, OBJECTIVES)
    fitted = S.fitted_gates(steps, OBJECTIVES, gate["eligible"])
    # numerical cross-checks: field first-order term vs jets (same 24 items); S1 vs S2 on the same 24 items
    chk = {}
    for o in OBJECTIVES:
        a = np.array([s["obj"][o]["a_step"] for s in steps])
        j = np.array([s["obj"][o]["J1_24"] for s in steps])
        s1 = np.array([s["obj"][o]["S1"] for s in steps])
        s2 = np.array([s["obj"][o]["S2_24"] for s in steps])
        rel = lambda x, y: float(np.median(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))
        chk[o] = {
            "field_vs_jet_first_order_median_rel": rel(a, j),
            "S1_vs_S2_same24_median_rel": rel(s1, s2),
        }
    final = json.loads((sdir / "final.json").read_text()) if (sdir / "final.json").exists() else {}
    base = json.loads((sdir / "base.json").read_text())
    return {
        "status": status,
        "measured": True,
        "n_steps": len(steps),
        "n_steps_h2": len(S.gate_steps(steps)),
        "cost_check": cc,
        "gate": gate,
        "fitted": fitted,
        "crosschecks": chk,
        "base_bitwise_repeat": base.get("bitwise_repeat"),
        "post_restore_bitwise_equal_base": final.get("post_restore_bitwise_equal_base"),
        "split_counts": {
            k: sum(1 for s in steps if s["split"] == k) for k in ("construction", "test", "other")
        },
    }


# ------------------------------------------------------------------ stage H


def h_report(hdir: Path) -> dict:
    status = (hdir / "STATUS").read_text().strip() if (hdir / "STATUS").exists() else "missing"
    if not (hdir / "evals.jsonl").exists():
        return {"status": status, "measured": False}
    evals = jl(hdir / "evals.jsonl")
    sizing = json.loads((hdir / "sizing.json").read_text())
    cal = json.loads((hdir / "guard_cal.json").read_text())
    budgets = json.loads((hdir / "budgets.json").read_text())
    by_tag: dict = {}
    for r in evals:
        by_tag.setdefault(r["tag"], []).append(r)
    base = by_tag["base"][0]
    rerun = by_tag.get("base_rerun", [None])[0]
    ceiling = by_tag.get("ceiling", [None])[0]
    acc = lambda rec, p: float(np.mean([x["correct"] for x in rec["panels"][p]]))
    out = {"status": status, "measured": True, "budgets": budgets, "sizing": sizing}
    out["guard_calibration"] = {
        k: {"delta": v["delta"], "base_greedy_rate": v["base_greedy_rate"]} for k, v in cal.items()
    }
    out["base"] = {p: acc(base, p) for p in base["panels"] if "correct" in base["panels"][p][0]}
    out["base_refusal"] = {
        p: float(np.mean([x["refused"] for x in base["panels"][p]]))
        for p in G.GUARDS
        if p in base["panels"]
    }
    if ceiling:
        out["ceiling_thinking"] = {p: acc(ceiling, p) for p in ceiling["panels"]}
        out["ceiling_truncated"] = {
            p: float(np.mean([x["trunc"] for x in ceiling["panels"][p]])) for p in ceiling["panels"]
        }
    delta = {k: cal[k]["delta"] for k in G.GUARDS}

    def cand(rec):
        prim = {}
        for p in G.PRIMARY:
            b = [x["correct"] for x in base["panels"][p]]
            c = [x["correct"] for x in rec["panels"][p]]
            prim[p] = G.paired(b, c)
            prim[p]["mean_len_delta"] = float(
                np.mean([x["len"] for x in rec["panels"][p]])
                - np.mean([x["len"] for x in base["panels"][p]])
            )
            prim[p]["trunc_rate"] = float(np.mean([x["trunc"] for x in rec["panels"][p]]))
            prim[p]["base_trunc_rate"] = float(np.mean([x["trunc"] for x in base["panels"][p]]))
        gt = G.guard_test(
            [x["refused"] for x in base["panels"]["guard_harmless"]],
            [x["refused"] for x in rec["panels"]["guard_harmless"]],
            delta["guard_harmless"],
            [x["refused"] for x in base["panels"]["guard_harmful"]],
            [x["refused"] for x in rec["panels"]["guard_harmful"]],
            delta["guard_harmful"],
        )
        row = {"spec": rec["spec"], "primary": prim, "guard": gt, "guarded": gt["guarded"]}
        if rec.get("proxy") and base.get("proxy"):
            row["dproxy"] = {
                p: (rec["proxy"][p] - base["proxy"][p])
                if rec["proxy"].get(p) is not None and base["proxy"].get(p) is not None
                else None
                for p in G.PRIMARY
            }
        if rec.get("oldguard_margin") and base.get("oldguard_margin"):
            sig = lambda m: 1 / (1 + np.exp(-np.array(m)))
            d = sig(rec["oldguard_margin"]) - sig(base["oldguard_margin"])
            row["oldguard_pp"] = {
                "mean": float(d.mean()),
                "ucb": float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(d.size)),
                "breach_1pp": bool(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(d.size) > 0.01),
            }
        row["qualifies"] = {p: G.qualifies(prim[p], gt["guarded"]) for p in G.PRIMARY}
        return row

    if rerun:
        out["null_base_rerun"] = cand(rerun)
    fam = {}
    rows_all = []
    for tag in ("H1", "H2", "H3", "H4"):
        rows = [cand(r) for r in by_tag.get(tag, [])]
        rows_all += rows
        fam[tag] = {"gate": G.family_gate(rows), "candidates": rows}
    ctrl = {}
    for tag in ("H1_ctrl_lenmatch", "H3_ctrl_random", "H3_ctrl_format", "H4_ctrl_unprojected"):
        rows = [cand(r) for r in by_tag.get(tag, [])]
        rows_all += rows
        ctrl[tag] = rows

    def vec(rec, p):
        return [x["correct"] for x in rec["panels"][p]]

    def matched(true_recs, ctl_recs, p):
        if not true_recs or not ctl_recs or len(true_recs) != len(ctl_recs):
            return None
        return G.matched_beat([vec(r, p) for r in true_recs], [vec(r, p) for r in ctl_recs])

    def ordered(tag, **flt):
        rs = [r for r in by_tag.get(tag, []) if all(r["spec"].get(k) == v for k, v in flt.items())]
        return sorted(rs, key=lambda r: r["spec"]["step"])

    # H3: true-reward alpha 1 at the control fractions (every other of 16) vs each control's 8 checkpoints
    h3t = ordered("H3", alpha=1.0)
    h3m = h3t[1::2] if len(h3t) >= 2 else h3t
    h3 = {}
    for arm in ("random", "format"):
        c = ordered(f"H3_ctrl_{arm}")
        h3[arm] = {p: matched(h3m[: len(c)], c, p) for p in G.PRIMARY}
    beats = {arm: any((v or {}).get("beats") for v in h3[arm].values()) for arm in h3}
    q = [p for p in G.PRIMARY if fam["H3"]["gate"]["counts"].get(p, 0) >= 2]
    beats_q = {arm: any((h3[arm].get(p) or {}).get("beats") for p in q) for arm in h3}
    fam["H3"]["controls"] = h3
    fam["H3"]["beats_both_controls"] = bool(q) and all(beats_q.values())
    fam["H3"]["pass"] = bool(fam["H3"]["gate"]["pass"] and fam["H3"]["beats_both_controls"])
    fam["H3"]["controls_beaten_any_primary"] = beats
    # H1 descriptive controls: H1 lambda 1 vs H2 at matched fractions, and vs the length-matched control
    h1t = ordered("H1", lam=1.0)
    h2 = ordered("H2")
    h2m = h2[1::2] if len(h2) >= 2 else h2
    lm = ordered("H1_ctrl_lenmatch")
    h1m = h1t[1::2] if len(h1t) >= 2 else h1t
    fam["H1"]["controls"] = {
        "vs_H2_nonthinking": {p: matched(h1t[: len(h2m)], h2m[: len(h1t)], p) for p in G.PRIMARY},
        "vs_length_matched": {p: matched(h1m[: len(lm)], lm[: len(h1m)], p) for p in G.PRIMARY},
    }
    fam["H1"]["pass"] = fam["H1"]["gate"]["pass"]
    fam["H2"]["pass"] = fam["H2"]["gate"]["pass"]
    fam["H4"]["pass"] = fam["H4"]["gate"]["pass"]
    out["families"] = fam
    out["controls"] = ctrl
    # proxy gate per primary benchmark over every evaluated candidate
    pg = {}
    for p in G.PRIMARY:
        xs = [
            (r["dproxy"][p], r["primary"][p]["mean"])
            for r in rows_all
            if r.get("dproxy") and r["dproxy"].get(p) is not None
        ]
        pg[p] = (
            G.proxy_gate([x for x, _ in xs], [y for _, y in xs])
            if len(xs) >= 3
            else {"n_candidates": len(xs), "killed": None}
        )
    out["proxy_gate"] = pg
    out["any_family_pass"] = any(fam[t]["pass"] for t in fam)
    out["stage_seconds"] = (
        json.loads((hdir / "stage_seconds.json").read_text())
        if (hdir / "stage_seconds.json").exists()
        else None
    )
    sec = {}
    for tag in [t for t in by_tag if t.endswith("_secondary")] + ["base", "ceiling"]:
        for r in by_tag.get(tag, []):
            for p in G.SECONDARY:
                if p in r["panels"]:
                    sec.setdefault(tag, []).append(
                        {
                            "spec": r["spec"],
                            "panel": p,
                            "acc": acc(r, p),
                            "delta_vs_base": acc(r, p) - out["base"].get(p, float("nan"))
                            if p in out["base"]
                            else None,
                        }
                    )
    out["secondary"] = sec
    for f in ("sft_stats.json",):
        if (hdir / f).exists():
            out[f.replace(".json", "")] = json.loads((hdir / f).read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s")
    ap.add_argument("--h")
    ap.add_argument("--ledger")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    rep = {
        "label": "EXPLORATORY (WM-E pilot 3)",
        "plan": "docs/council/2026-11-17/RULING-wme-pilot3.md",
    }
    if a.s:
        rep["S"] = s_report(Path(a.s))
    if a.h:
        rep["H"] = h_report(Path(a.h))
    if a.ledger:
        rep["ledger"] = json.loads(Path(a.ledger).read_text())
    Path(a.json).write_text(json.dumps(rep, indent=1, default=float))
    print(a.json)


if __name__ == "__main__":
    main()
