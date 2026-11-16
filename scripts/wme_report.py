#!/usr/bin/env python3
"""WM-E v0 pilot gate evaluation and bank price (SYNTHESIS-r8 D8). EXPLORATORY; CPU only; reads pilot artefacts.

Gate definitions (fixed in code before any edited-state outcome was read; thresholds are the chair's D8 table):

* Go-M: max over phases of peak GPU memory (allocator reserved and nvidia-smi device use) <= 28 GB, peak
  process-tree RSS <= 24 GB, and no swap in use.
* Go-N: base panel repeat bitwise identical (P1), tap leaves outputs bitwise unchanged (P1), restore returns the base
  theta-hash and bitwise-identical outputs (P1), and the P3 base pass reproduces the P1 base pass bitwise.
* Go-C: lean panel (validation split, all six objectives) <= 240 s per state: max of the P1 warm pass and the P3
  median per-state time (P3 includes the tap).
* Go-T: (i) >= 50% of (step, objective) pairs resolvable (|paired mean| >= 2 SE), pooled over the six objectives;
  and (ii) static-field rollout error at h >= 3 above 2 x noise: the median over the (3-step endpoint, objective)
  pairs of |observed - F(E)| / SE exceeds 2, for the raw field and for F3u refitted on the single-step ladder
  states (reported separately; (ii) passes if it holds for the refitted F3u, the stronger static predictor).
* Go-R: over steps 2-3 of the six P4 trajectories (step 1 is identical by construction), the oracle left-point
  prediction <A_{t-1}, D_t> + s q(D_t) cuts the mean absolute per-step error of the static per-step prediction
  F(E_t) - F(E_{t-1}) by >= 20%, pooled over objectives with each objective's errors scaled by its step SE.
* Go-E: >= 2 of the 24 P5 candidates have, for one candidate objective, a dev gain 95% LCB > 0 with no guardrail
  breach on the other objectives (guardrails in ``scripts/wme_pilot.py``).

Bank price: H = (visits * t_ll + N_gen * t_gen + T_features + T_train) / 3600 with visits = 960, t_ll = measured
per-state panel seconds (P3 median incl. tap and edit/restore), N_gen = 96 generation states (a 10% stratified
subset) at t_gen = measured seconds for a 32-prompt x 128-token greedy batch (P1 scaled from 8 prompts),
T_features = per-state oracle gradients only if Go-R passes (else 0) plus the base field over the bank's support
(P2 seconds per 4-layer group x 8 groups), T_train = 0 GPU (the world model trains on CPU; reported separately).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OBJ = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_gap",
)
BETTER = {
    "wikitext_nll": -1,
    "gsm8k_gold_lp": 1,
    "mmlupro_gold_lp": 1,
    "refusal_margin_harmful": 1,
    "refusal_margin_harmless": -1,
    "sycophancy_gap": -1,
}
GUARD = {
    "wikitext_nll": ("ucb", 0.01),
    "gsm8k_gold_lp": ("lcb", -0.05),
    "mmlupro_gold_lp": ("lcb", -0.05),
    "refusal_margin_harmful": ("lcb", -0.25),
    "refusal_margin_harmless": ("ucb", 0.25),
    "sycophancy_gap": ("ucb", 0.10),
}
CANDIDATE_OBJECTIVES = ("gsm8k_gold_lp", "mmlupro_gold_lp", "sycophancy_gap")
LADDER_PATHS = ((0.0, 0.5, 1.0, 2.0), (0.0, -0.5, -1.0))
GB = 1e9


def paired(a, b):
    d = np.asarray(b) - np.asarray(a)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def load_states(run):
    states = {}
    for line in (run / "p3_states.jsonl").read_text().splitlines():
        r = json.loads(line)
        states[r["id"]] = r
    base = json.loads((run / "p3_base.json").read_text())
    return states, base


def values(states, base, sid):
    if sid == "base":
        return base["values"]
    return states[sid]["values"]


def ladder_id(root, dose):
    return "base" if dose == 0 else f"ladder:{root}:{dose:+g}"


def steps_list(states):
    roots = sorted({s.split(":")[1] for s in states if s.startswith("ladder:")})
    steps = []
    for r in roots:
        for path in LADDER_PATHS:
            for a, b in itertools.pairwise(path):
                steps.append(("ladder", ladder_id(r, a), ladder_id(r, b)))
    seen = set()
    for sid in states:
        if not sid.startswith("comp:"):
            continue
        parts = sid.split(":", 1)[1].split(">")
        prev = f"ladder:{parts[0]}:+1" if len(parts) == 2 else "comp:" + ">".join(parts[:-1])
        if (prev, sid) not in seen:
            seen.add((prev, sid))
            steps.append(("composition", prev, sid))
    return steps


def gate_t(states, base, field):
    steps = steps_list(states)
    res = {o: [] for o in OBJ}
    rows = []
    for kind, a, b in steps:
        va, vb = values(states, base, a), values(states, base, b)
        row = {"kind": kind, "from": a, "to": b}
        for o in OBJ:
            m, se = paired(va[o], vb[o])
            ok = abs(m) >= 2 * se
            res[o].append(ok)
            row[o] = {"mean": m, "se": se, "resolvable": ok}
        rows.append(row)
    pooled = float(np.mean([x for o in OBJ for x in res[o]]))
    per_obj = {o: float(np.mean(res[o])) for o in OBJ}
    # static rollout error
    ladder_ids = [s for s in states if s.startswith("ladder:")]
    comp_ids = [s for s in states if s.startswith("comp:")]
    fit = {}
    for o in OBJ:
        X = np.array([[field[s][o]["a"], field[s][o]["q"]] for s in ladder_ids])
        y = np.array([paired(base["values"][o], states[s]["values"][o])[0] for s in ladder_ids])
        fit[o] = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    err_rows, ratios_raw, ratios_fit = [], [], []
    for s in comp_ids:
        h = len(s.split(":", 1)[1].split(">"))
        for o in OBJ:
            m, se = paired(base["values"][o], states[s]["values"][o])
            f_raw = field[s][o]["F"]
            f_fit = fit[o][0] * field[s][o]["a"] + fit[o][1] * field[s][o]["q"]
            r = {
                "state": s,
                "h": h,
                "objective": o,
                "observed": m,
                "se": se,
                "F_raw": f_raw,
                "F3u_fit": f_fit,
                "err_raw": abs(m - f_raw),
                "err_fit": abs(m - f_fit),
            }
            err_rows.append(r)
            if h >= 3:
                ratios_raw.append(r["err_raw"] / se)
                ratios_fit.append(r["err_fit"] / se)
    by_h = {}
    for r in err_rows:
        by_h.setdefault(r["h"], {"err_raw": [], "err_fit": [], "se": []})
        for k in ("err_raw", "err_fit", "se"):
            by_h[r["h"]][k].append(r[k])
    by_h = {h: {k: float(np.median(v)) for k, v in d.items()} for h, d in sorted(by_h.items())}
    med_raw, med_fit = float(np.median(ratios_raw)), float(np.median(ratios_fit))
    return {
        "n_steps": len(steps),
        "resolvable_pooled": pooled,
        "resolvable_per_objective": per_obj,
        "steps": rows,
        "static_error_by_horizon_median": by_h,
        "static_h3_median_err_over_se": {"raw": med_raw, "F3u_fit": med_fit},
        "F3u_fit_coefficients": fit,
        "static_rows": err_rows,
        "pass_resolvable": pooled >= 0.5,
        "pass_static_above_noise": med_fit > 2.0,
        "pass": pooled >= 0.5 and med_fit > 2.0,
    }


def gate_r(states, base, p4):
    step_obs = {}
    for key, objs in p4["steps"].items():
        parts = key.split(">")
        h = len(parts)
        cur = f"ladder:{parts[0]}:+1" if h == 1 else "comp:" + key
        prev = (
            "base"
            if h == 1
            else (f"ladder:{parts[0]}:+1" if h == 2 else "comp:" + ">".join(parts[:-1]))
        )
        if cur not in states and cur != "base":
            continue
        for o, rec in objs.items():
            m, se = paired(values(states, base, prev)[o], values(states, base, cur)[o])
            step_obs.setdefault(h, []).append(
                {
                    "key": key,
                    "objective": o,
                    "observed": m,
                    "se": se,
                    "static": rec["static_step"],
                    "left": rec["oracle_left"],
                    "trap": rec["oracle_trapezoid"],
                }
            )

    def cut(rows, k):
        e_s = np.mean([abs(r["observed"] - r["static"]) / r["se"] for r in rows])
        e_o = np.mean([abs(r["observed"] - r[k]) / r["se"] for r in rows])
        return float(1 - e_o / e_s), float(e_s), float(e_o)

    later = [r for h in (2, 3) for r in step_obs.get(h, [])]
    allr = [r for v in step_obs.values() for r in v]
    c_left = cut(later, "left")
    c_trap = cut(later, "trap")
    return {
        "steps_2_3": len(later),
        "cut_left": c_left[0],
        "static_err_over_se": c_left[1],
        "left_err_over_se": c_left[2],
        "cut_trapezoid": c_trap[0],
        "trap_err_over_se": c_trap[2],
        "cut_left_all_steps": cut(allr, "left")[0],
        "per_objective_cut_left": {
            o: cut([r for r in later if r["objective"] == o], "left")[0] for o in OBJ
        },
        "rows": allr,
        "pass": c_left[0] >= 0.20,
    }


def gate_e(p5):
    out = {o: [] for o in CANDIDATE_OBJECTIVES}
    rows = []
    for c in p5["candidates"]:
        pr = c["paired"]
        row = {"i": c["i"], "candidate": c["candidate"]}
        for o in CANDIDATE_OBJECTIVES:
            m, se = pr[o]["mean"], pr[o]["se"]
            gain = BETTER[o] * m
            lcb = gain - 1.96 * se
            breaches = []
            for g, (kind, lim) in GUARD.items():
                if g == o:
                    continue
                gm, gse = pr[g]["mean"], pr[g]["se"]
                bound = gm + 1.96 * gse if kind == "ucb" else gm - 1.96 * gse
                if (kind == "ucb" and bound > lim) or (kind == "lcb" and bound < lim):
                    breaches.append(g)
            ok = lcb > 0 and not breaches
            out[o].append(ok)
            row[o] = {"gain": gain, "lcb": lcb, "breaches": breaches, "qualifies": ok}
        rows.append(row)
    counts = {o: int(sum(v)) for o, v in out.items()}
    best = max(counts, key=lambda o: (counts[o], o))
    return {"counts": counts, "best_objective": best, "rows": rows, "pass": counts[best] >= 2}


def synthetic_summary(name="synth-20261116"):
    p = ROOT / "data/wme/synthetic" / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    groups = {}
    for r in d["rows"]:
        key = f"{r['kind']}|{r['shuffle']}|slope={r['slope_transport']}"
        groups.setdefault(key, []).append(r)
    return {
        "config": d["config"],
        "wall_seconds": d["wall_seconds"],
        "cpu_seconds": d["cpu_seconds"],
        "by_condition": {
            k: {
                "wm_over_static_h34": [x["wm_over_static_h34"] for x in v],
                "gain_over_scale_h34": [x["gain_over_scale_h34"] for x in v],
            }
            for k, v in groups.items()
        },
    }


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def render_md(rep, go_t, go_r, go_e):
    g = rep["gates"]
    L = []
    L.append("### Gate table (generated by `scripts/wme_report.py`)\n")
    L.append("| Gate | Passes when | Measured | Result |")
    L.append("|---|---|---|---|")
    m = g["Go-M"]
    L.append(
        f"| Go-M | GPU <= 28 GB, RSS <= 24 GB, no swap | reserved {_f(m['peak_gpu_reserved_GB'], 2)} GB; device "
        f"{_f(m['peak_device_GB'], 2)} GB; RSS {_f(m['peak_rss_GB'], 2)} GB; swap {m['swap_bytes']} B | "
        f"{'PASS' if m['pass'] else 'FAIL'} |"
    )
    n = g["Go-N"]
    L.append(
        f"| Go-N | deterministic repeat | base repeat {_f(n['base_repeat'])}; tap unchanged {_f(n['tap_unchanged'])}; "
        f"restore hash {_f(n['restore_hash'])}; post-restore full panel {_f(n['restore_outputs_p3_full_panel'])}; "
        f"P3 base = P1 base {_f(n['p3_base_matches_p1'])} | {'PASS' if n['pass'] else 'FAIL'} |"
    )
    c = g["Go-C"]
    L.append(
        f"| Go-C | lean panel <= 240 s/state | P1 warm {_f(c['p1_warm_panel_seconds'], 1)} s; P3 median "
        f"{_f(c['p3_median_state_seconds'], 1)} s (max {_f(c['p3_max_state_seconds'], 1)} s) | "
        f"{'PASS' if c['pass'] else 'FAIL'} |"
    )
    t = g["Go-T"]
    L.append(
        f"| Go-T | >= 50% steps resolvable and static error at h >= 3 > 2x noise | resolvable "
        f"{_f(t['resolvable_pooled'])} of {t['n_steps']} steps x 6 objectives; median |err|/SE at h=3: raw "
        f"{_f(t['static_h3_median_err_over_se']['raw'], 2)}, refitted F3u "
        f"{_f(t['static_h3_median_err_over_se']['F3u_fit'], 2)} | {'PASS' if t['pass'] else 'FAIL'} |"
    )
    r = g["Go-R"]
    L.append(
        f"| Go-R | oracle A_t cuts per-step error >= 20% | left-point cut {_f(r['cut_left'])}; trapezoid cut "
        f"{_f(r['cut_trapezoid'])} (steps 2-3, n={r['steps_2_3']}) | {'PASS' if r['pass'] else 'FAIL'} |"
    )
    e = g["Go-E"]
    L.append(
        "| Go-E | >= 2 of 24 with dev-gain LCB > 0, no guardrail breach | "
        + "; ".join(f"{k} {v}" for k, v in e["counts"].items())
        + f" | {'PASS' if e['pass'] else 'FAIL'} |"
    )
    L.append("\n### Resolvability per objective (fraction of steps with |paired change| >= 2 SE)\n")
    L.append(
        "| Objective | resolvable | median |change| ladder | median SE ladder | median |change| composition | median SE composition |"
    )
    L.append("|---|---|---|---|---|---|")
    for o in OBJ:
        lad = [s[o] for s in go_t["steps"] if s["kind"] == "ladder"]
        com = [s[o] for s in go_t["steps"] if s["kind"] == "composition"]
        L.append(
            f"| {o} | {_f(t['resolvable_per_objective'][o])} | {_f(float(np.median([abs(x['mean']) for x in lad])))} "
            f"| {_f(float(np.median([x['se'] for x in lad])))} | {_f(float(np.median([abs(x['mean']) for x in com])))} "
            f"| {_f(float(np.median([x['se'] for x in com])))} |"
        )
    L.append("\n### Static-field rollout error by horizon (median over states and objectives)\n")
    L.append("| h | median |obs - F_raw| | median |obs - F3u_fit| | median paired SE |")
    L.append("|---|---|---|---|")
    for h, v in t["static_error_by_horizon_median"].items():
        L.append(f"| {h} | {_f(v['err_raw'])} | {_f(v['err_fit'])} | {_f(v['se'])} |")
    L.append("\n### R2 oracle (P4): per-objective cut of the per-step error (steps 2-3)\n")
    L.append("| Objective | left-point cut |")
    L.append("|---|---|")
    for o, v in r["per_objective_cut_left"].items():
        L.append(f"| {o} | {_f(v)} |")
    L.append("\n### P5 headroom: candidates qualifying per candidate objective (dev, n=24)\n")
    L.append("| Objective | qualifying | best gain | best LCB |")
    L.append("|---|---|---|---|")
    for o in CANDIDATE_OBJECTIVES:
        rows = [x[o] for x in go_e["rows"]]
        best = max(rows, key=lambda x: x["lcb"])
        L.append(f"| {o} | {e['counts'][o]} | {_f(best['gain'])} | {_f(best['lcb'])} |")
    b = rep["bank_price"]
    L.append("\n### Bank price\n")
    L.append(
        f"H = (visits x t_ll + N_gen x t_gen + T_features + T_train) / 3600 = ({b['visits']} x {_f(b['t_ll_seconds'], 1)} "
        f"+ {b['N_gen']} x {_f(b['t_gen_seconds'], 1)} + {_f(b['T_features_seconds'], 0)} + 0) / 3600 = "
        f"**{_f(b['H_gpu_hours'], 2)} GPU-h**; with per-state oracle observations kept: "
        f"{_f(b['H_gpu_hours_if_oracle_observations_kept'], 2)} GPU-h."
    )
    L.append("\n### Ledger\n")
    L.append("| Phase | wall s | CPU s | peak GPU reserved GB | peak device GB | peak RSS GB |")
    L.append("|---|---|---|---|---|---|")
    for x in rep["ledger"]["phases"]:
        L.append(
            f"| {x['phase']} | {_f(x['wall_seconds'], 0)} | {_f(x['cpu_seconds'], 0) if x['cpu_seconds'] else 'n/a'} | "
            f"{_f(x['peak_gpu_reserved_bytes'] / GB, 2)} | {_f(x['peak_device_used_bytes_nvidia_smi'] / GB, 2)} | "
            f"{_f(x['peak_process_tree_rss_bytes'] / GB, 2)} |"
        )
    L.append(
        f"\nTotal pilot GPU-h {_f(rep['ledger']['gpu_hours_pilot'], 2)}; CPU-h of pilot jobs "
        f"{_f(rep['ledger']['cpu_hours_pilot_jobs'], 2)}."
    )
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot-20261116")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None, help="write the generated tables (markdown) here")
    ap.add_argument(
        "--visit-seconds", type=float, default=None, help="measured all-in seconds per P3 visit"
    )
    ap.add_argument(
        "--oracle-seconds", type=float, default=None, help="measured oracle seconds per state"
    )
    args = ap.parse_args()
    run = ROOT / "data/wme/pilot" / args.run
    ledger = [json.loads(x) for x in (run / "ledger.jsonl").read_text().splitlines()]
    p1 = json.loads((run / "p1.json").read_text())
    p2 = json.loads((run / "p2.json").read_text())
    states, base = load_states(run)
    field = json.loads((run / "p3_field.json").read_text())["states"]
    p4 = json.loads((run / "p4.json").read_text())
    p5 = json.loads((run / "p5.json").read_text())
    # Go-M is evaluated on the completed jobs (closed by the job itself, so with measured allocator peaks) in the
    # final configuration; runs that were stopped or failed are reported separately with their peaks.
    done = [r for r in ledger if r["cpu_seconds"] is not None]
    stopped = [r for r in ledger if r["cpu_seconds"] is None]
    peak_res = max(r["peak_gpu_reserved_bytes"] for r in done)
    peak_dev = max(r["peak_device_used_bytes_nvidia_smi"] for r in done)
    peak_rss = max(r["peak_process_tree_rss_bytes"] for r in done)
    swap = max(r["peak_swap_used_bytes"] for r in ledger)
    go_m = {
        "peak_gpu_reserved_GB": peak_res / GB,
        "peak_device_GB": peak_dev / GB,
        "peak_rss_GB": peak_rss / GB,
        "swap_bytes": swap,
        "pass": peak_res <= 28 * GB and peak_dev <= 28 * GB and peak_rss <= 24 * GB and swap == 0,
        "stopped_or_failed_runs": [
            {
                "phase": r["phase"],
                "peak_device_GB": r["peak_device_used_bytes_nvidia_smi"] / GB,
                "note": r.get("note"),
            }
            for r in stopped
        ],
        "cap_breach_any_run": max(r["peak_device_used_bytes_nvidia_smi"] for r in ledger) > 28 * GB,
    }
    go_n = {
        "base_repeat": p1["bitwise_repeat"],
        "tap_unchanged": p1["tap_outputs_bitwise_unchanged"],
        "restore_hash": p1["restore_returns_base_hash"],
        # P1's post-restore check compared a WikiText-only evaluation with the full-panel base (different
        # batch composition, so not like-for-like); it is superseded by P3's full-panel post-restore check.
        "restore_outputs_p3_full_panel": json.loads((run / "p3_post_restore.json").read_text())[
            "bitwise_equal_base"
        ],
        "p3_base_matches_p1": base["matches_p1_bitwise"],
    }
    # P3 evaluates five composition prefixes twice (shared by a 2-step and a 3-step composition): an edited-state
    # repeat check (same theta-hash, values compared bitwise)
    reps = {}
    for line in (run / "p3_states.jsonl").read_text().splitlines():
        r = json.loads(line)
        reps.setdefault(r["id"], []).append(
            (r["theta_hash"], json.dumps(r["values"], sort_keys=True))
        )
    dup = {k: v for k, v in reps.items() if len(v) > 1}
    go_n["edited_state_repeats_bitwise"] = all(len({x for x in v}) == 1 for v in dup.values())
    go_n["pass"] = all(go_n.values())
    go_n["n_edited_state_repeats"] = len(dup)
    go_n["p1_restore_check_not_like_for_like"] = p1["post_restore_bitwise_equal_base"]
    p3_secs = [s["eval_seconds"] for s in states.values()]
    t_state = max(p1["panel_seconds"][1], float(np.median(p3_secs)))
    # all-in seconds per visit (eval + edit apply/restore + tap): mean interval between consecutive P3 state log
    # lines (agent-tmux log timestamps), passed with --visit-seconds; otherwise a conservative bound from the P3
    # phase wall (which also contains direction construction)
    if args.visit_seconds:
        t_visit = args.visit_seconds
    else:
        p3_led = [r for r in ledger if r["phase"] == "P3"][-1]
        p3_field_s = json.loads((run / "p3_field.json").read_text())["seconds"]
        t_visit = (
            p3_led["wall_seconds"] - p1["load_seconds"] - 2 * base["seconds"] - p3_field_s
        ) / len(states)
    go_c = {
        "p1_warm_panel_seconds": p1["panel_seconds"][1],
        "p3_median_state_seconds": float(np.median(p3_secs)),
        "p3_max_state_seconds": float(np.max(p3_secs)),
        "pass": t_state <= 240,
    }
    go_t = gate_t(states, base, field)
    go_r = gate_r(states, base, p4)
    go_e = gate_e(p5)
    # bank price
    gen = p1["generation"]
    t_gen = gen["seconds"] * 32 / gen["prompts"]
    p2_s = p2["collect_seconds"]
    t_oracle = args.oracle_seconds or (
        float(np.mean(p4["grad_seconds"])) if p4["grad_seconds"] else 0.0
    )
    visits, n_gen = 960, 96
    t_feat = 8 * p2_s + (visits * t_oracle if go_r["pass"] else 0.0)
    t_ll = max(t_state, t_visit)
    H = (visits * t_ll + n_gen * t_gen + t_feat + 0.0) / 3600
    H_with_oracle = (visits * t_ll + n_gen * t_gen + 8 * p2_s + visits * t_oracle) / 3600
    gpu_h = sum(r["wall_seconds"] for r in ledger) / 3600
    cpu_h = sum(r["cpu_seconds"] or 0.0 for r in ledger) / 3600
    report = {
        "label": "EXPLORATORY",
        "run": args.run,
        "gates": {
            "Go-M": go_m,
            "Go-N": go_n,
            "Go-C": go_c,
            "Go-T": {k: v for k, v in go_t.items() if k not in ("steps", "static_rows")},
            "Go-R": {k: v for k, v in go_r.items() if k != "rows"},
            "Go-E": {k: v for k, v in go_e.items() if k != "rows"},
        },
        "bank_price": {
            "visits": visits,
            "t_ll_seconds": t_ll,
            "t_eval_median_seconds": t_state,
            "t_visit_all_in_seconds": t_visit,
            "N_gen": n_gen,
            "t_gen_seconds": t_gen,
            "T_features_seconds": t_feat,
            "T_train_gpu_seconds": 0.0,
            "H_gpu_hours": H,
            "H_gpu_hours_if_oracle_observations_kept": H_with_oracle,
            "oracle_gradient_seconds_per_state": t_oracle,
            "field_seconds_per_4_layer_group": p2_s,
        },
        "ledger": {"gpu_hours_pilot": gpu_h, "cpu_hours_pilot_jobs": cpu_h, "phases": ledger},
        "p1": p1,
        "p2": {k: v for k, v in p2.items() if k != "spectral_factor_audit_ffn"}
        | {"spectral_audit": p2.get("spectral_factor_audit_ffn")},
        "details": {
            "go_t_steps": go_t["steps"],
            "go_t_static_rows": go_t["static_rows"],
            "go_r_rows": go_r["rows"],
            "go_e_rows": go_e["rows"],
        },
    }
    report["synthetic"] = synthetic_summary()
    out = Path(args.out) if args.out else run / "report.json"
    if args.md:
        Path(args.md).write_text(render_md(report, go_t, go_r, go_e))
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    print(json.dumps(report["gates"], indent=1, default=float))
    print(json.dumps(report["bank_price"], indent=1, default=float))


if __name__ == "__main__":
    main()
