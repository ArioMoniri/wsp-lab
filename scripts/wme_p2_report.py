#!/usr/bin/env python3
"""WM-E pilot-2 gate evaluation (EXPLORATORY; CPU only; reads ``data/wme/pilot2/<run>``).

Gates and consequences are computed by ``wm.p2design`` exactly as fixed in ``RULING-wme-pilot1.md`` (section 3-4);
see that module's docstring for the implementation choices. Anything else in the output is labelled descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import p2design as G
from wsp.wm import panel2 as P2

T2O = P2.T2_OBJECTIVES


def paired(a, b):
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def jsonl(p):
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def ledger(run):
    rows = []
    for p in sorted(run.rglob("ledger.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            r["_file"] = str(p.relative_to(run))
            rows.append(r)
    return rows


# ------------------------------------------------------------------ T2


def t2_eval(run):
    t2 = json.loads((run / "t2.json").read_text())
    states = jsonl(run / "t2_states.jsonl")
    base = t2["base_values"]
    fam = t2["families"]

    def vals(sid):
        return base if sid == "base" else states[sid]["values"]

    visited = {r: {"pos": v["pos"], "neg": v["neg"]} for r, v in t2["visited"].items()}
    comp_ids = [s for s in states if states[s]["kind"] == "composition"]
    steps = G.steps_list(visited, comp_ids, fam)
    rows, desc_rows = [], []
    for s in steps:
        for o in T2O + P2.DESCRIPTIVE:
            m, se = paired(vals(s["from"])[o], vals(s["to"])[o])
            r = {**s, "objective": o, "mean": m, "se": se, "resolvable": G.is_resolvable(m, se)}
            (rows if o in T2O else desc_rows).append(r)
    res = G.resolvability(rows)

    def group(s):
        if s["kind"] == "composition":
            return "composition"
        return s["to"].split(":")[1]

    by_group = {}
    for r in rows:
        by_group.setdefault((group(r), r["control"]), []).append(r["resolvable"])
    per_root = {
        g: {
            "family": fam.get(g, "composition"),
            "control": c,
            "resolvable": float(np.mean(v)),
            "n": len(v),
        }
        for (g, c), v in sorted(by_group.items())
    }
    per_obj = {
        o: float(
            np.mean([r["resolvable"] for r in rows if r["objective"] == o and not r["control"]])
        )
        for o in T2O
    }
    # static endpoint (pilot-1 aggregate)
    field = t2["field"]
    ladder_rows, ladder_rows_nc, comp_rows = [], [], []
    for sid, st in states.items():
        for o in T2O:
            m, se = paired(base[o], st["values"][o])
            f = field[sid][o]
            if st["kind"] == "ladder":
                r = {"state": sid, "objective": o, "a": f["a"], "q": f["q"], "observed": m}
                ladder_rows.append(r)
                if fam[sid.split(":")[1]] != "control":
                    ladder_rows_nc.append(r)
            elif st["kind"] == "composition":
                comp_rows.append(
                    {
                        "state": sid,
                        "objective": o,
                        "h": st["h"],
                        "a": f["a"],
                        "q": f["q"],
                        "F": f["F"],
                        "observed": m,
                        "se": se,
                    }
                )
    st_main = G.static_endpoint(ladder_rows, comp_rows, T2O)
    st_nc = G.static_endpoint(ladder_rows_nc, comp_rows, T2O)
    raw = [abs(r["observed"] - r["F"]) / r["se"] for r in comp_rows if r["h"] >= 3]
    gate = G.go_t2(res["noncontrol"], st_main["median_err_over_se_h3plus"])
    ladders = {
        r: {
            side: [
                {k: w[k] for k in ("dose", "dnll", "endpoint", "label")}
                for w in t2["visited"][r][side + "_walk"]
            ]
            for side in ("pos", "neg")
        }
        for r in t2["visited"]
    }
    return {
        "design": t2["design"],
        "compositions_3": t2["compositions_3"],
        "compositions_4": t2["compositions_4"],
        "n_states": len(states),
        "n_steps": len(steps),
        "n_steps_control": sum(s["control"] for s in steps),
        "resolvability": res,
        "resolvability_per_root_or_group": per_root,
        "resolvability_per_objective_noncontrol": per_obj,
        "descriptive_proxy_resolvability_noncontrol": float(
            np.mean([r["resolvable"] for r in desc_rows if not r["control"]])
        ),
        "static_endpoint_median_err_over_se_h3plus": st_main["median_err_over_se_h3plus"],
        "static_n_pairs": st_main["n_pairs"],
        "F3u_coef": st_main["coef"],
        "descriptive": {
            "per_objective_median_h3plus": st_main["per_objective_median_descriptive"],
            "median_of_per_objective_medians": float(
                np.median(list(st_main["per_objective_median_descriptive"].values()))
            ),
            "raw_field_median_h3plus": float(np.median(raw)),
            "fit_without_control_ladders_median_h3plus": st_nc["median_err_over_se_h3plus"],
        },
        "ladders": ladders,
        "base_matches_field_phase": t2["base_matches_field_phase"],
        "post_restore_bitwise_equal_base": t2["post_restore_bitwise_equal_base"],
        "gate": gate,
    }


# ------------------------------------------------------------------ R2


def r2_eval(run):
    t2 = json.loads((run / "t2.json").read_text())
    states = jsonl(run / "t2_states.jsonl")
    r2 = jsonl(run / "r2_states.jsonl")
    base = t2["base_values"]
    field = t2["field"]
    seqs = [tuple(s) for s in t2["compositions_3"] + t2["compositions_4"]]

    def sid_of(key):
        parts = key.split(">")
        return G.ladder_id(parts[0], 1.0) if len(parts) == 1 else "comp:" + key

    def vals(sid):
        return base if sid == "base" else states[sid]["values"]

    def F(sid, o):
        return 0.0 if sid == "base" else field[sid][o]["F"]

    rows, seen = [], set()
    missing = []
    for s in seqs:
        for h in range(1, len(s) + 1):
            key = ">".join(s[:h])
            prev = ">".join(s[: h - 1]) if h > 1 else "base"
            if (prev, key) in seen:
                continue
            seen.add((prev, key))
            if key not in r2 or prev not in r2:
                missing.append(key)
                continue
            a, b = (sid_of(prev) if prev != "base" else "base"), sid_of(key)
            for o in T2O:
                m, se = paired(vals(a)[o], vals(b)[o])
                left = r2[prev]["out"][key][o]
                right = r2[key]["in"][o]
                rows.append(
                    {
                        "key": key,
                        "h": h,
                        "objective": o,
                        "observed": m,
                        "se": se,
                        "static": F(b, o) - F(a, o),
                        "left": left,
                        "right": right,
                        "trap": 0.5 * (left + right),
                    }
                )
    gate = G.go_r2(rows)
    per_obj = {o: G.go_r2([r for r in rows if r["objective"] == o])["cut"] for o in T2O}
    grad_s = [r["grad_seconds"] for r in r2.values() if "grad_seconds" in r]
    return {
        "gate": gate,
        "per_objective_cut_descriptive": per_obj,
        "n_oracle_states": len(grad_s),
        "mean_grad_seconds": float(np.mean(grad_s)) if grad_s else None,
        "missing": missing,
        "rows": rows,
    }


# ------------------------------------------------------------------ E2


def e2_eval(run, model_key):
    d = run / f"e2-{model_key}"
    if not (d / "e2_meta.json").exists():
        return None
    base = json.loads((d / "base_dev.json").read_text())
    cands = json.loads((d / "candidates.json").read_text())
    scores = jsonl(d / "e2_scores.jsonl")
    meta = json.loads((d / "e2_meta.json").read_text())
    by_id = {o: {r["id"]: r for r in recs} for o, recs in base.items()}

    def stats(rec, o, scale):
        e = {r["id"]: r for r in rec["records"][o]}
        if scale == "value":
            a = [r["value"] for r in base[o]]
            b = [e[r["id"]]["value"] for r in base[o]]
        else:
            a = [P2.pp_value(o, r) for r in base[o]]
            b = [P2.pp_value(o, e[r["id"]]) for r in base[o]]
        m, se = paired(a, b)
        return {"mean": m, "se": se}

    out = {"model": meta["model"], "revision": meta["revision"], "objectives": {}}
    counts, ncand = {}, {}
    for o in P2.E2_OBJECTIVES:
        rows = []
        for c in cands[o]:
            if G.candidate_key(c) not in scores:
                continue  # incomplete run: the gate is not computed
            rec = scores[G.candidate_key(c)]
            g = stats(rec, o, "value")
            guards = {
                k: stats(rec, k, G.GUARDS[k][0]) for k in rec["records"] if k in G.GUARDS and k != o
            }
            br = G.guard_breaches(guards, o)
            lcb = G.gain_lcb(o, g["mean"], g["se"])
            rows.append(
                {
                    "candidate": G.candidate_key(c),
                    "gain": G.ORIENT[o] * g["mean"],
                    "se": g["se"],
                    "lcb": lcb,
                    "breaches": br,
                    "qualifies": G.qualifies(o, g["mean"], g["se"], br),
                    "guards": guards,
                }
            )
        counts[o] = sum(r["qualifies"] for r in rows)
        ncand[o] = len(rows)
        best = max(rows, key=lambda r: r["lcb"])
        clean = [r for r in rows if not r["breaches"]]
        lora = [r for r in rows if r["candidate"].startswith("lora:")]
        out["objectives"][o] = {
            "qualifying": counts[o],
            "n_candidates": len(rows),
            "best_lcb": best["lcb"],
            "best_lcb_candidate": best["candidate"],
            "best_lcb_breaches": best["breaches"],
            "best_lcb_without_breach": max((r["lcb"] for r in clean), default=None),
            "n_without_breach": len(clean),
            "n_lcb_positive": sum(r["lcb"] > 0 for r in rows),
            "breach_counts": {g: sum(g in r["breaches"] for r in rows) for g in G.GUARDS},
            "lora": [
                {k: r[k] for k in ("candidate", "gain", "lcb", "breaches", "qualifies")}
                for r in lora
            ],
            "rows": rows,
        }
    # DESCRIPTIVE sensitivity (not the gate): guardrails read on each objective's own value scale with the
    # pilot-1 exploratory margins (scripts/wme_pilot.py GUARD; analogues for the two new objectives)
    p1_margins = {
        "wikitext_nll": ("ucb", 0.01),
        "gsm8k_gold_lp": ("lcb", -0.05),
        "mathcomp_gold_lp": ("lcb", -0.05),
        "mmlupro_gold_lp": ("lcb", -0.05),
        "refusal_margin_harmful": ("lcb", -0.25),
        "refusal_margin_harmless": ("ucb", 0.25),
        "sycophancy_real": ("ucb", 0.10),
    }
    sens = {}
    for o in P2.E2_OBJECTIVES:
        n = 0
        for c in cands[o]:
            if G.candidate_key(c) not in scores:
                continue
            rec = scores[G.candidate_key(c)]
            g = stats(rec, o, "value")
            bad = False
            for k, (kind, lim) in p1_margins.items():
                if k == o or k not in rec["records"]:
                    continue
                st = stats(rec, k, "value")
                b = (
                    st["mean"] + G.Z95 * st["se"]
                    if kind == "ucb"
                    else st["mean"] - G.Z95 * st["se"]
                )
                bad |= (kind == "ucb" and b > lim) or (kind == "lcb" and b < lim)
            n += G.gain_lcb(o, g["mean"], g["se"]) > 0 and not bad
        sens[o] = n
    out["descriptive_qualifying_with_pilot1_value_scale_margins"] = sens
    if all(n == G.E2_N for n in ncand.values()):
        out["gate"] = G.go_e2(counts, ncand)
    else:
        out["gate"] = {"incomplete": ncand, "any_pass": None}
    out["base_dev_mean"] = {o: float(np.mean([r["value"] for r in v])) for o, v in base.items()}
    out["post_restore_bitwise_equal_base"] = meta["post_restore_bitwise_equal_base"]
    out["unique_evaluations"] = meta["unique_evaluations"]
    out["lora_logs"] = {}
    import torch

    for o in P2.E2_OBJECTIVES:
        lg = torch.load(d / f"lora_{o}.pt", weights_only=False)["log"]
        out["lora_logs"][o] = {
            k: lg[k]
            for k in (
                "steps",
                "seconds",
                "stopped_by_cap",
                "n_rows",
                "epochs",
                "lr",
                "batch_rows",
                "rank",
                "alpha",
            )
        } | {
            "loss_first20": float(np.mean(lg["loss"][:20])),
            "loss_last20": float(np.mean(lg["loss"][-20:])),
        }
    del by_id
    return out


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float) and (math.isnan(x)):
        return "nan"
    if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def render_md(rep) -> str:
    L = []
    t, r = rep["T2"], rep["R2"]
    L.append("| Gate | Rule (pre-declared) | Measured | Result |")
    L.append("|---|---|---|---|")
    if t:
        L.append(
            f"| Go-T2 (i) | non-control steps resolvable >= 0.50 | {_f(t['resolvability']['noncontrol'])} of "
            f"{t['resolvability']['n_noncontrol']} (step, objective) pairs | "
            f"{'PASS' if t['gate']['pass_resolvable'] else 'FAIL'} |"
        )
        L.append(
            f"| Go-T2 (ii) primary | six-objective median refitted-F3u static error at h >= 3 > 2 SE | "
            f"{_f(t['static_endpoint_median_err_over_se_h3plus'], 2)} SE ({t['static_n_pairs']} pairs) | "
            f"{'PASS' if t['gate']['pass_static'] else 'FAIL'} |"
        )
        L.append(f"| **Go-T2** | (i) and (ii) | | **{'PASS' if t['gate']['pass'] else 'FAIL'}** |")
    if r:
        g = r["gate"]
        L.append(
            f"| **Go-R2** | trapezoid oracle cuts per-step error >= 20% vs A0 (steps h >= 2) | cut {_f(g['cut'])} "
            f"(static {_f(g['static_err_over_se'], 2)} SE -> trapezoid {_f(g['trap_err_over_se'], 2)} SE; n = {g['n']}) "
            f"| **{'PASS' if g['pass'] else 'FAIL'}** |"
        )
    for k, e in rep["E2"].items():
        if not e:
            continue
        for o, v in e["objectives"].items():
            gp = e["gate"].get("pass", {}).get(o)
            L.append(
                f"| Go-E2 {k.upper()} {o} | >= 2 of 32 with dev-gain LCB > 0 and no guardrail breach | "
                f"{v['qualifying']} of {v['n_candidates']} (best LCB {_f(v['best_lcb'], 4)}) | "
                f"{'PASS' if gp else 'FAIL'} |"
            )
    L.append("")
    if t:
        L.append(
            "Resolvability per root (ladder steps) and for compositions; the negative control is separate:\n"
        )
        L.append("| Root / group | Family | Control | Resolvable | n pairs |")
        L.append("|---|---|---|---|---|")
        for g_, v in t["resolvability_per_root_or_group"].items():
            L.append(
                f"| {g_} | {v['family']} | {_f(v['control'])} | {_f(v['resolvable'])} | {v['n']} |"
            )
        L.append("")
        L.append("| Objective | resolvable (non-control) | static median h>=3 (descriptive) |")
        L.append("|---|---|---|")
        for o, v in t["resolvability_per_objective_noncontrol"].items():
            L.append(
                f"| {o} | {_f(v)} | {_f(t['descriptive']['per_objective_median_h3plus'].get(o), 2)} |"
            )
        L.append("")
        L.append("Ladders (dose: ΔNLL; * = labelled endpoint, side stopped):\n")
        L.append("| Root | positive side | negative side |")
        L.append("|---|---|---|")
        for root, sides in t["ladders"].items():
            cell = lambda ws: ", ".join(
                f"{w['dose']:+g}: {w['dnll']:.3f}{'*' if w['endpoint'] else ''}" for w in ws
            )
            L.append(f"| {root} | {cell(sides['pos'])} | {cell(sides['neg'])} |")
        L.append("")
    for k, e in rep["E2"].items():
        if not e:
            continue
        L.append(f"E2 on {e['model']} @ `{e['revision'][:12]}` (dev items only):\n")
        L.append(
            "| Objective | qualifying / 32 | LCB > 0 (any breach) | no breach | best LCB (candidate) | best LCB without breach | breaches by guard |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for o, v in e["objectives"].items():
            bc = ", ".join(f"{g}: {n}" for g, n in v["breach_counts"].items() if n)
            L.append(
                f"| {o} | {v['qualifying']} | {v['n_lcb_positive']} | {v['n_without_breach']} | "
                f"{_f(v['best_lcb'], 4)} ({v['best_lcb_candidate']}) | {_f(v['best_lcb_without_breach'], 4)} | {bc} |"
            )
        L.append("")
        L.append("| LoRA candidate | gain | LCB | breaches | qualifies |")
        L.append("|---|---|---|---|---|")
        for o, v in e["objectives"].items():
            for x in v["lora"]:
                L.append(
                    f"| {x['candidate']} | {_f(x['gain'], 4)} | {_f(x['lcb'], 4)} | {', '.join(x['breaches']) or '-'} "
                    f"| {_f(x['qualifies'])} |"
                )
        L.append("")
    led = rep["ledger"]
    L.append(
        "| Ledger file | Phase | wall s (= GPU s) | CPU s | peak device GB | peak RSS GB | swap B |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for x in led["rows"]:
        L.append(
            f"| {x['_file']} | {x['phase']} | {_f(x['wall_seconds'], 0)} | {_f(x['cpu_seconds'], 0)} | "
            f"{_f((x['peak_device_used_bytes_nvidia_smi'] or 0) / 1e9, 2)} | "
            f"{_f((x['peak_process_tree_rss_bytes'] or 0) / 1e9, 2)} | {x['peak_swap_used_bytes']} |"
        )
    L.append(f"\nTotal GPU-h {led['gpu_hours']:.2f}; CPU-h {led['cpu_hours']:.2f}.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot2-20261116")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = ROOT / "data/wme/pilot2" / args.run
    rep = {"label": "EXPLORATORY", "run": args.run}
    rep["panel_manifest"] = {
        k: v
        for k, v in json.loads((ROOT / "data/wme/panels/wme-p2/manifest.json").read_text()).items()
        if k
        in (
            "content_hash",
            "files",
            "sizes",
            "excluded",
            "dedup",
            "sources",
            "math500_disjoint_checked",
            "pilot1_sealed_excluded",
            "salt",
        )
    }
    rep["T2"] = t2_eval(run) if (run / "t2.json").exists() else None
    rep["R2"] = r2_eval(run) if (run / "r2_states.jsonl").exists() else None
    rep["E2"] = {k: e2_eval(run, k) for k in ("9b", "4b")}
    e9, e4 = rep["E2"]["9b"], rep["E2"]["4b"]
    if rep["T2"] and e9:
        e9p = e9["gate"]["any_pass"]
        e4p = None if e4 is None else e4["gate"]["any_pass"]
        try:
            rep["consequence"] = G.consequence(rep["T2"]["gate"]["pass"], e9p, e4p)
        except ValueError as exc:
            rep["consequence"] = {"row": None, "text": f"pending: {exc}"}
    led = ledger(run)
    rep["ledger"] = {
        "rows": [
            {
                k: r.get(k)
                for k in (
                    "_file",
                    "phase",
                    "wall_seconds",
                    "cpu_seconds",
                    "peak_gpu_reserved_bytes",
                    "peak_device_used_bytes_nvidia_smi",
                    "peak_process_tree_rss_bytes",
                    "peak_swap_used_bytes",
                    "git_sha",
                )
            }
            for r in led
        ],
        "gpu_hours": sum(r["wall_seconds"] for r in led) / 3600,
        "cpu_hours": sum(r["cpu_seconds"] for r in led) / 3600,
        "peak_device_GB": max((r["peak_device_used_bytes_nvidia_smi"] for r in led), default=0)
        / 1e9,
        "peak_rss_GB": max((r["peak_process_tree_rss_bytes"] for r in led), default=0) / 1e9,
        "peak_swap_bytes": max((r["peak_swap_used_bytes"] for r in led), default=0),
    }
    out = Path(args.out) if args.out else run / "report.json"
    out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=float))
    out.with_suffix(".md").write_text(render_md(rep) + "\n")
    s = {
        "T2": rep["T2"]
        and {
            "resolvable_noncontrol": rep["T2"]["resolvability"]["noncontrol"],
            "control": rep["T2"]["resolvability"]["control"],
            "static": rep["T2"]["static_endpoint_median_err_over_se_h3plus"],
            "pass": rep["T2"]["gate"]["pass"],
        },
        "R2": rep["R2"] and {k: rep["R2"]["gate"][k] for k in ("cut", "n", "pass")},
        "E2": {k: v and v["gate"] for k, v in rep["E2"].items()},
        "consequence": rep.get("consequence"),
        "gpu_hours": rep["ledger"]["gpu_hours"],
    }
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
