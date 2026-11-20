#!/usr/bin/env python3
"""Study M Gate 0 report: reads a ``scripts/wmm_gate0.py`` run directory and writes the verdict JSON and markdown
tables (dose verdicts by the pre-specified rules in ``wsp.wm.wmm_gate0``).

    wmm_gate0_report.py RUN_DIR OUT_JSON [OUT_TABLES_MD]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wsp.wm import wmm_gate0 as W


def rd(run, name):
    return json.loads((run / f"{name}.json").read_text())


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, se = W.mean_se(d)
    return {"mean": m, "se": se, "lb95": m - 1.96 * se, "ub95": m + 1.96 * se}


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    inert, card, base, final = (
        rd(run, "inert"),
        rd(run, "card"),
        rd(run, "ladder_base_nll"),
        rd(run, "final"),
    )
    arch, acct, man = rd(run, "architecture"), rd(run, "accounting"), rd(run, "manifest")
    fams, dose = {}, {}
    for fam in W.FAMILIES:
        rep = rd(run, f"ladder_{fam}")
        st = rep["states"]
        drift = {float(k): v["drift"] for k, v in st.items()}
        dose[fam] = W.dose_verdict(fam, drift)
        rows = []
        for k in sorted(st, key=float):
            v = st[k]
            r = {
                "dose": float(k),
                "drift_mean": float(np.mean(v["drift"])),
                "drift_se": W.mean_se(v["drift"])[1],
                "accuracy": float(np.mean(v["correct"])),
                "d_accuracy": paired(v["correct"], base["correct"]),
                "truncation": float(np.mean(v["truncated"])),
                "edit_realised_norm": v["edit"].get("realised_norm"),
            }
            if "syc_adopt" in v:
                r["syc_adopt"] = float(np.mean(v["syc_adopt"]))
                r["d_syc_adopt"] = paired(v["syc_adopt"], base["syc_adopt"])
                r["syc_correct"] = float(np.mean(v["syc_correct"]))
            rows.append(r)
        restored = (
            rep.get("restored_hash") == inert["state_hash_start"]
            if "restored_hash" in rep
            else None
        )
        fams[fam] = {"ladder": rows, "dose_verdict": dose[fam], "restored_bitwise": restored}
        if fam.endswith("lora"):
            fams[fam]["training"] = rd(run, f"lora_{fam}")
    inert_pass = bool(inert["pass"] and final["canary_match"] and final["state_hash_matches_start"])
    kill = W.kill_verdict(inert_pass, dose, card)
    res = {
        "study": "M",
        "gate": 0,
        "run_dir": str(run),
        "manifest": man,
        "architecture_same": arch["same"],
        "architecture": {
            k: {kk: v[kk] for kk in ("n_text_tensors", "n_text_params", "shapes_sha256", "tied")}
            for k, v in arch["signatures"].items()
        },
        "inert": inert,
        "final": final,
        "inert_pass_including_canary": inert_pass,
        "card": card,
        "base_ladder": {
            "n": len(base["correct"]),
            "accuracy": float(np.mean(base["correct"])),
            "truncation": float(np.mean(base["truncated"])),
            "syc_n": len(base["syc_adopt"]),
            "syc_adopt": float(np.mean(base["syc_adopt"])),
            "syc_correct": float(np.mean(base["syc_correct"])),
            "budget": base["budget"],
        },
        "families": fams,
        "kill": kill,
        "gpu_hours": acct["gpu_hours"],
    }
    out.write_text(json.dumps(res, indent=1, default=float))
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(tables(res))
    print(
        json.dumps(
            {"kill": kill, "gpu_hours": acct["gpu_hours"], "card": card["accuracy"]}, indent=1
        )
    )


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def tables(res) -> str:
    """Markdown tables of the dose ladders and verdicts (pasted into the review)."""
    out = []
    for fam, v in res["families"].items():
        dv = v["dose_verdict"]
        out.append(f"#### {fam} ({W.FAMILIES[fam]['param']}; base {W.FAMILIES[fam]['base']})\n")
        syc = any("syc_adopt" in r for r in v["ladder"])
        head = "| dose | drift (nats/token) ± SE | accuracy (64) | Δacc [95% CI] | truncated |"
        sep = "|---|---|---|---|---|"
        if syc:
            head += " syc adopt | Δadopt [95% CI] | syc correct |"
            sep += "---|---|---|"
        out += [head, sep]
        for r in v["ladder"]:
            d = r["d_accuracy"]
            row = (
                f"| {r['dose']:g} | {f(r['drift_mean'], 4)} ± {f(r['drift_se'], 4)} | {f(r['accuracy'])} | "
                f"{f(d['mean'])} [{f(d['lb95'])}, {f(d['ub95'])}] | {f(r['truncation'])} |"
            )
            if syc:
                a = r["d_syc_adopt"]
                row += f" {f(r['syc_adopt'])} | {f(a['mean'])} [{f(a['lb95'])}, {f(a['ub95'])}] | {f(r['syc_correct'])} |"
            out.append(row)
        arms = "; ".join(
            f"arm {a['doses']}: monotone {a['monotone']}, top mean {f(a['top_mean'], 4)}, LB {f(a['top_lb95'], 4)}, "
            f"non-flat {a['non_flat']}"
            for a in dv["arms"]
        )
        out.append(f"\nVerdict: **{'PASS' if dv['pass'] else 'FAIL'}** ({arms}).\n")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
