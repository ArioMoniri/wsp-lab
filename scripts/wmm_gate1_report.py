#!/usr/bin/env python3
"""Study M Gate 1 report: reads a ``scripts/wmm_gate1.py`` run directory (and the guard audit's judge files) and
writes the verdict JSON plus markdown tables. Rules: ``wsp.wm.wmm.gate1`` (plan wmm-gate1-plan-2026-11-20.md).

    wmm_gate1_report.py RUN_DIR OUT_JSON OUT_TABLES_MD
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import itertools

import numpy as np

from wsp.wm.wmm import gate1 as G1


def rd(run, name, default=None):
    p = run / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else default


def pooled(rec, names, key="correct"):
    return [int(x) for n in names for x in rec["sets"][n][key]]


def stats_for(base, rec, names):
    b = {"correct": pooled(base, names), "truncated": pooled(base, names, "truncated")}
    e = {"correct": pooled(rec, names), "truncated": pooled(rec, names, "truncated")}
    return {
        "at_budget": G1.paired(e["correct"], b["correct"]),
        "untruncated_in_both": G1.untruncated_in_both(b, e),
    }


def sizing(disc_k, n, gain):
    _lo, hi = G1.wilson(disc_k, n)
    out = {"d": disc_k / n, "d_wilson_upper": hi, "N": {}}
    for delta in (0.02, 0.03, 0.05):
        out["N"][f"delta_{delta}"] = G1.sizing_n(hi, delta)
    if gain is not None and gain > 0:
        out["N"]["delta_half_gain"] = G1.sizing_n(hi, gain / 2)
        out["half_gain"] = gain / 2
    return out


def agreement_perm(items_orig, rec_o, rec_p, name_o, name_p, perm_items):
    """Share of items (parsed in both) whose chosen option text is the same on the original and permuted copy."""
    lo, lp = rec_o["sets"][name_o]["letter"], rec_p["sets"][name_p]["letter"]
    same, n = 0, 0
    for it, ip, a, b in zip(items_orig, perm_items, lo, lp, strict=True):
        if a is None or b is None:
            continue
        n += 1
        same += G1.option_text(it, a) == G1.option_text(ip, b)
    return {"n": n, "agreement": same / n if n else None}


def audit(run):
    d = run / "audit"
    if not (d / "judge_a.json").exists() or not (d / "judge_b.json").exists():
        return None
    rule = json.loads((d / "rule_labels.json").read_text())
    a = json.loads((d / "judge_a.json").read_text())
    b = json.loads((d / "judge_b.json").read_text())
    keys = sorted(rule)
    agree = [k for k in keys if a.get(k) == b.get(k) and a.get(k) in ("refuse", "comply")]
    ok = [k for k in agree if rule[k] == a[k]]
    by = {}
    for grp in ("harm", "benign"):
        ks = [k for k in agree if k.startswith(grp)]
        by[grp] = {
            "n": len(ks),
            "rule_agreement": sum(rule[k] == a[k] for k in ks) / len(ks) if ks else None,
            "audit_refusal_rate": sum(a[k] == "refuse" for k in ks) / len(ks) if ks else None,
            "rule_refusal_rate": sum(rule[k] == "refuse" for k in ks) / len(ks) if ks else None,
        }
    return {
        "n_items": len(keys),
        "judges_agree": len(agree),
        "judge_agreement": len(agree) / len(keys),
        "rule_vs_audit_agreement": len(ok) / len(agree) if agree else None,
        "by_group": by,
        "rule_disagreements": [k for k in agree if rule[k] != a[k]],
    }


def main():
    run, out_json, out_md = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    items = rd(run, "items")
    base, base_s = rd(run, "state_base"), rd(run, "state_base_s")
    summ = rd(run, "summary", {})
    states = {}
    for fam, dose in G1.ORDER:
        tag = f"{fam}@{dose:g}"
        r = rd(run, f"state_{tag}")
        if r and "mq" in r["sets"]:
            states[tag] = r
    perm_items = {k: [G1.permuted(it) for it in items[k]] for k in ("medqa", "medmcqa")}
    res = {
        "study": "M",
        "gate": 1,
        "run_dir": "data/wmm/" + run.name,
        "manifest": rd(run, "manifest"),
    }

    # base
    bstat = {}
    for key, name in (("medqa", "mq"), ("medmcqa", "mm")):
        for tag, rec in (("pass1", base), ("base_s", base_s)):
            s = rec["sets"][name]
            bstat.setdefault(key, {})[tag] = {
                "accuracy": float(np.mean(s["correct"])),
                "truncation": float(np.mean(s["truncated"])),
                "b_star": G1.b_star(s["length"], s["truncated"]),
                "truncation_by_budget": {
                    b: G1.truncation_at(s["length"], s["truncated"], b) for b in G1.BUDGETS
                },
                "rules": {str(k): s["rule"].count(k) for k in set(s["rule"])},
            }
        nd = G1.paired(base_s["sets"][name]["correct"], base["sets"][name]["correct"])
        bstat[key]["null_composition"] = {
            "discordance": nd["discordance"],
            "mean": nd["mean"],
            "text_identical": None,
        }
    for name in ("mq_syc", "mq_perm", "mm_perm"):
        s = base_s["sets"][name]
        bstat[name] = {
            "accuracy": float(np.mean(s["correct"])),
            "truncation": float(np.mean(s["truncated"])),
        }
    bstat["mq_syc"]["adoption"] = float(np.mean(base_s["sets"]["mq_syc"]["adopted"]))
    for name in ("harm", "benign"):
        bstat[name] = {
            "refusal_rate": float(np.mean(base["sets"][name]["refused"])),
            "truncation": float(np.mean(base["sets"][name]["truncated"])),
        }
    bstat["general"] = {
        "accuracy": float(np.mean(base["sets"]["general"]["correct"])),
        "truncation": float(np.mean(base["sets"]["general"]["truncated"])),
    }
    conf = [c for n in ("mq", "mm") for c in base_s["sets"][n]["conf"]]
    corr = [y for n in ("mq", "mm") for y in base_s["sets"][n]["correct"]]
    cc = [(c, y) for c, y in zip(conf, corr, strict=True) if c is not None]
    bstat["calibration"] = {
        "n": len(cc),
        "brier": G1.brier(*zip(*cc, strict=True)),
        "ece": G1.ece(*zip(*cc, strict=True)),
    }
    bstat["perm_agreement"] = {
        "medqa": agreement_perm(
            items["medqa"], base_s, base_s, "mq", "mq_perm", perm_items["medqa"]
        ),
        "medmcqa": agreement_perm(
            items["medmcqa"], base_s, base_s, "mm", "mm_perm", perm_items["medmcqa"]
        ),
    }
    res["base"] = bstat

    # edits
    edits = {}
    for tag, rec in states.items():
        prim = [n for n in ("mq", "mm") if n in rec["sets"]]
        e = {
            "primaries": prim,
            "pooled": stats_for(base_s, rec, prim),
            "medqa": stats_for(base_s, rec, ["mq"]),
            "medmcqa": stats_for(base_s, rec, ["mm"]) if "mm" in prim else None,
            "accuracy": {
                k: float(np.mean(rec["sets"][n]["correct"]))
                for k, n in (("medqa", "mq"), ("medmcqa", "mm"))
                if n in prim
            },
            "truncation": {
                k: float(np.mean(rec["sets"][n]["truncated"]))
                for k, n in (("medqa", "mq"), ("medmcqa", "mm"))
                if n in prim
            },
            "timing": rec["timing"],
            "tokens": {n: rec["sets"][n]["tokens"] for n in rec["sets"]},
        }
        e["positive_untruncated"] = G1.positive(e["pooled"]["untruncated_in_both"])
        if "mq_perm" in rec["sets"]:
            e["permuted"] = stats_for(
                {"sets": {n: base_s["sets"][f"{n}_perm"] for n in prim}},
                {"sets": {n: rec["sets"][f"{n}_perm"] for n in prim}},
                prim,
            )
            e["positive_permuted"] = G1.positive(e["permuted"]["untruncated_in_both"])
            e["perm_agreement"] = {
                "medqa": agreement_perm(
                    items["medqa"], rec, rec, "mq", "mq_perm", perm_items["medqa"]
                ),
                "medmcqa": agreement_perm(
                    items["medmcqa"], rec, rec, "mm", "mm_perm", perm_items["medmcqa"]
                )
                if "mm_perm" in rec["sets"]
                else None,
            }
        if "mq_syc" in rec["sets"]:
            e["sycophancy"] = {
                "adoption": float(np.mean(rec["sets"]["mq_syc"]["adopted"])),
                "accuracy": float(np.mean(rec["sets"]["mq_syc"]["correct"])),
                "d_adoption": G1.paired(
                    [int(x) for x in rec["sets"]["mq_syc"]["adopted"]],
                    [int(x) for x in base_s["sets"]["mq_syc"]["adopted"]],
                ),
                "d_accuracy_suggested": G1.paired(
                    [int(x) for x in rec["sets"]["mq_syc"]["correct"]],
                    [int(x) for x in base_s["sets"]["mq_syc"]["correct"]],
                ),
                "d_neutral_accuracy_medqa": e["medqa"]["at_budget"],
            }
        pb = e["pooled"]["at_budget"]
        e["sizing"] = sizing(round(pb["discordance"] * pb["n"]), pb["n"], pb["mean"])
        edits[tag] = e
    res["edits"] = edits

    # M2: LoRA target-behaviour dose ladders (post hoc)
    m2 = {}
    for fam, metric in (("h1_lora", "accuracy"), ("syc_lora", "adoption")):
        pts = []
        for d in G1.GRID[fam]:
            tag = f"{fam}@{d:g}"
            if tag not in edits:
                continue
            if metric == "accuracy":
                pts.append((d, edits[tag]["medqa"]["at_budget"]["mean"]))  # MedQA at every dose
            else:
                pts.append((d, edits[tag]["sycophancy"]["d_adoption"]["mean"]))
        vals = [0.0] + [v for _d, v in pts]
        m2[fam] = {
            "metric": "MedQA accuracy change" if metric == "accuracy" else "adoption change",
            "ladder": pts,
            "monotone_nonincreasing": all(b <= a for a, b in itertools.pairwise(vals)),
            "monotone_nondecreasing": all(b >= a for a, b in itertools.pairwise(vals)),
        }
    res["m2_lora_dose"] = m2

    guards = rd(run, "guards", {})
    res["guards"] = guards
    res["audit"] = audit(run)

    both = summ.get("positive_on_both") or [
        t for t, e in edits.items() if e.get("positive_untruncated") and e.get("positive_permuted")
    ]
    syc_drop = summ.get("syc_drop_lambda1")
    if syc_drop is None and "syc_lora@1" in edits:
        syc_drop = -edits["syc_lora@1"]["sycophancy"]["d_adoption"]["mean"]
    bstar = {k: bstat[k]["base_s"]["b_star"] for k in ("medqa", "medmcqa")}
    qgf = {t: guards[t]["fail_any"] for t in both if t in guards}
    agreement = res["audit"]["rule_vs_audit_agreement"] if res["audit"] else None
    res["kill"] = G1.kill_verdict(both, bstar, syc_drop, qgf, agreement)
    res["kill"]["guards_missing_for"] = [t for t in both if t not in guards]
    res["qualifiers"] = summ.get("qualifiers")
    res["stopped_by_cap"] = summ.get("stopped_by_cap")
    res["final"] = rd(run, "final")
    res["lora"] = {f: rd(run, f"lora_{f}") for f in ("h1_lora", "syc_lora")}
    res["gpu_hours"] = rd(run, "accounting")["gpu_hours"]
    # per-member timing (prices the r13 R5 bank): seconds per screened state on mq+mm
    tim = []
    for e in edits.values():
        t = e["timing"]
        tim.append(
            t.get("apply_seconds", 0)
            + t.get("restore_seconds", 0)
            + t.get("mq_seconds", 0)
            + t.get("mm_seconds", 0)
        )
    res["member_timing"] = {
        "n": len(tim),
        "mean_seconds": float(np.mean(tim)) if tim else None,
        "max_seconds": float(np.max(tim)) if tim else None,
    }
    out_json.write_text(json.dumps(res, indent=1, default=float))
    out_md.write_text(tables(res))
    print(json.dumps({"kill": res["kill"], "gpu_hours": res["gpu_hours"]}, indent=1, default=float))


def f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def tables(res) -> str:
    L = [
        "| edit | acc MedQA | acc MedMCQA | Δ at budget [95% CI] | Δ untrunc-in-both [95% CI] (n) | positive | Δ permuted, untrunc [95% CI] | trunc MQ/MM | discordance (Wilson UB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    b = res["base"]
    L.append(
        f"| base (base_s) | {f(b['medqa']['base_s']['accuracy'])} | {f(b['medmcqa']['base_s']['accuracy'])} | — | — | — | — | {f(b['medqa']['base_s']['truncation'])}/{f(b['medmcqa']['base_s']['truncation'])} | — |"
    )
    for tag, e in res["edits"].items():
        a, u = e["pooled"]["at_budget"], e["pooled"]["untruncated_in_both"]
        p = e.get("permuted", {}).get("untruncated_in_both")
        ps = f"{f(p['mean'])} [{f(p['lb95'])}, {f(p['ub95'])}]" if p else "—"
        L.append(
            f"| {tag} | {f(e['accuracy']['medqa'])} | {f(e['accuracy'].get('medmcqa'))} | {f(a['mean'])} [{f(a['lb95'])}, {f(a['ub95'])}] | "
            f"{f(u['mean'])} [{f(u['lb95'])}, {f(u['ub95'])}] ({u['n']}) | {'yes' if e['positive_untruncated'] else 'no'} | {ps} | "
            f"{f(e['truncation']['medqa'])}/{f(e['truncation'].get('medmcqa'))} | {f(e['sizing']['d'])} ({f(e['sizing']['d_wilson_upper'])}) |"
        )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
