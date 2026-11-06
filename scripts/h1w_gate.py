#!/usr/bin/env python3
"""H1w: the registered section 5 / A2.3 gate on the WikiText log-PPL D3 out-of-fold losses (PREREG A4.3-A4.5).

CPU only; launches no scoring. Run through `agent-tmux`:

    python scripts/h1w_gate.py --workers 8            # registered scale: 4,000 / 4,000 / 4,000
    python scripts/h1w_gate.py --panels 40 --joint-panels 20 --draws 400 --workers 4   # smoke

The pipeline is `tfv1_requalify.py` unchanged (frozen section 3 fit: nuisance OLS with intercept,
family dummies and log1p(incremental effective norm); StatNN seeds 0,1,2 averaged before the loss;
isotonic on raw incremental norm; five hashed inner folds; row-summed theta = 1 - sum A / sum B;
1e-12 denominator tolerance; bootstrap-t / BCa / percentile intervals, conditional and joint; the same
empirical, item-noise-inflated and cluster-merged simulation scenarios on the sealed component sizes)
with y replaced by the paired mean of WikiText token-weighted log-perplexity (nats). The joint
procedure resamples the endpoint's two-window blocks (A2.1), so each item is the mean of a block.

Output: docs/reviews/P5/h1w-gate-<date>.md and data/runs/h1w/gate_report.json (plus a committed copy of
the JSON next to the document). PASS/FAIL per Board A's criteria (A4.5 and its reversal evidence).
"""

import os

# one BLAS thread per process (parallelism comes from --workers): set before numpy loads
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import h1a_companions as hc  # noqa: E402
import tfv1_requalify as rq  # noqa: E402

ENDPOINT = "wikitext_log_ppl"
SIX = [f"{w}_{m}" for w in ("conditional", "joint") for m in ("bootstrap_t", "bca", "percentile")]
SCENARIOS = {
    "empirical": (1.0, False),
    "item_noise_x1.5": (1.5, False),
    "components_merged_in_pairs": (1.0, True),
}
GPU_COST_H = 1.9  # Board A (derived, not measured): ~0.4 min per delta x 281 deltas


def wording(intervals):
    """A4.4 wording rule on the six intervals of theta."""
    lows = {k: intervals[k][0] for k in SIX}
    return {
        "lower_bounds": lows,
        "all_six_lower_bounds_positive": bool(all(v > 0 for v in lows.values())),
        "phrase": (
            "distinguishable from zero"
            if all(v > 0 for v in lows.values())
            else "not distinguishable from zero"
        ),
    }


def interval_vs_simulation(intervals, results):
    """Each interval next to its simulated null size (theta = 0) and coverage (theta = .15)."""
    emp = results["empirical"]
    rows = []
    for key in SIX:
        world, method = key.split("_", 1)
        rows.append(
            {
                "interval": key,
                "theta_interval": intervals[key],
                "null_size": emp["theta_0.0"][world][method]["rejection"],
                "coverage_at_0.15": emp["theta_0.15"][world][method]["coverage95"],
            }
        )
    return rows


def criteria(results, verdict, qualified, flex_theta, panels, joint_panels, draws):
    """Board A / A4.5: (a)-(e) with the A2.3 primary or BCa fallback; the reversal evidence (power >= .80
    at theta = .15 for both procedures, null size <= .035, coverage >= .93, denominator failures
    < 0.1%, and flexible-nuisance theta_hat > 0)."""
    registered = panels >= 4000 and joint_panels >= 4000 and draws >= 4000
    out = {"registered_scale": registered, "per_method": {}}
    for method in ("bootstrap_t", "bca", "percentile"):
        m = {}
        for world in ("conditional", "joint"):
            eff = results["empirical"]["theta_0.15"][world][method]
            m[f"(a) power lower95 >= 0.80 at theta 0.15, {world}"] = {
                "value": eff["rejection_wilson_lower95"],
                "pass": bool(eff["rejection_wilson_lower95"] >= 0.80),
            }
        for scen, by in results.items():
            for world in ("conditional", "joint"):
                null, eff = by["theta_0.0"][world][method], by["theta_0.15"][world][method]
                m[f"(b) null size <= 0.035, {scen}, {world}"] = {
                    "value": null["rejection"],
                    "pass": bool(null["rejection"] <= 0.035),
                }
                m[f"(b) coverage >= 0.93 at theta 0.15, {scen}, {world}"] = {
                    "value": eff["coverage95"],
                    "pass": bool(eff["coverage95"] >= 0.93),
                }
                fails = max(
                    null["denominator_failure_fraction"], eff["denominator_failure_fraction"]
                )
                m[f"(c) denominator failures < 0.1%, {scen}, {world}"] = {
                    "value": fails,
                    "pass": bool(fails < 0.001),
                }
        m["all_pass"] = bool(all(v["pass"] for v in m.values()))
        out["per_method"][method] = m
    out["(d) qualification A2.1 met (WikiText)"] = bool(qualified)
    out["(e) P(decision pass) at 0.15 / 0.20 / 0.25 (conditional, joint; bootstrap-t)"] = {
        f"theta_{t}": {
            w: results["empirical"][f"theta_{t}"][w]["bootstrap_t"]["p_decision_pass"]
            for w in ("conditional", "joint")
        }
        for t in rq.THETA_TARGETS
    }
    chosen = verdict["method_chosen"]
    out["method_chosen_by_A2.3_order"] = chosen
    out["a4_5_gate_met"] = bool(chosen and qualified and registered)
    out["flexible_nuisance_theta_hat"] = flex_theta
    out["flexible_nuisance_theta_above_zero"] = bool(flex_theta > 0)
    out["board_reversal_criteria_met"] = bool(out["a4_5_gate_met"] and flex_theta > 0)
    return out


def pf(flag):
    return "PASS" if flag else "FAIL"


def ci(v):
    return hc.ci(v)


def render(rep):
    q, g, c = rep["qualification"][ENDPOINT], rep["gate"], rep["companions"]
    fit, crit = g["fit"], g["criteria"]
    lines = [
        f"# H1w gate: the section 5 / A2.3 procedure on WikiText log-PPL D3 out-of-fold losses ({rep['date']})\n",
        *(
            [
                "**SMOKE RUN below the registered scale: INDICATIVE ONLY, not the registered gate.**\n"
            ]
            if not crit["registered_scale"]
            else []
        ),
        f"Generated by `scripts/h1w_gate.py` (PREREG_H1_TF amendment A4, committed `{rep['a4_commit']}` before any WikiText out-of-fold loss, theta_hat or interval was computed). "
        f"Source `source_hash` `{rep['source_hash'][:12]}`, partition seal `{rep['partition_sha256'][:12]}`, {rep['n_stratum']} D3 stratum entries ({rep['family_counts']}) in {c['n_components']} components, device H0 only. "
        f"Development panel; nothing here is a fresh outcome, no GPU was used, no sealed or fresh component was scored. "
        "**H1w is a separately named exploratory claim; it is not H1a and may not be cited as H1a (A4.2).**\n",
        f"Disclosure (A4.6): H1w was proposed because WikiText qualified and MMLU-Pro did not, with the MMLU D3 theta_hat = 0.129 known. The WikiText D3 theta_hat was not computed before A4 was committed; the first WikiText theta is in this report.\n",
        "## 1. Verdict\n",
        f"- **A4.5 gate (a)-(e) with the A2.3 primary or BCa fallback: {pf(crit['a4_5_gate_met'])}** (interval procedure chosen on simulated data: `{crit['method_chosen_by_A2.3_order']}`; registered scale: {crit['registered_scale']}).",
        f"- **Board A reversal criteria (gate met *and* flexible-nuisance theta_hat > 0): {pf(crit['board_reversal_criteria_met'])}** (flexible-nuisance theta_hat = {hc.g(crit['flexible_nuisance_theta_hat'])}, family x natural spline; n = {c['n_rows']} rows, {c['n_components']} components).",
        f"- **H1w status: EXPLORATORY** (A4.5; it could become eligible for confirmatory scoring only if the gate is met and a sealed WikiText window panel and the A3 bitwise re-check are committed first; neither exists).",
        f"- **Wording rule (A4.4):** the lower bounds of the six intervals are {', '.join(hc.g(v) for v in g['wording']['lower_bounds'].values())}; WikiText theta_hat = {hc.g(fit['theta_hat_stratum'])} is therefore **{g['wording']['phrase']}** (n = {fit['n_stratum']} rows, {c['n_components']} components). No pass/kill label.\n",
        "## 2. Qualification recap (A2.1, already met; recomputed from the same artefacts)\n",
        f"wikitext_log_ppl, delta = {hc.g(q['delta'])}, {q['n_entries']} entries, {q['n_items']} windows in blocks of 2.\n",
        "| Condition | Value | Limit | Result |",
        "|---|---:|---:|---|",
        f"| (i) panel-mean half-width, upper 97.5% | {hc.g(q['i_panel_mean']['upper97_5'])} | {hc.g(q['i_panel_mean']['limit_delta_over_2'])} | {pf(q['i_panel_mean']['pass'])} |",
        f"| (ii) max half-width | {hc.g(q['ii_ceiling']['max_halfwidth'])} | {hc.g(q['ii_ceiling']['limit_delta'])} | {pf(q['ii_ceiling']['pass'])} |",
        f"| (iii) tripwire P, upper 97.5% (point {hc.g(q['iii_tripwire']['P_point'])}) | {hc.g(q['iii_tripwire']['P_upper97_5'])} | 0.2 | {pf(q['iii_tripwire']['pass'])} |",
        f"| repeat max discrepancy | {hc.g(q['repeat']['max_abs_discrepancy'])} | {q['repeat']['limit_max']} | {pf(q['repeat']['pass_max'])} |",
        f"| reliability lower 95% (point {hc.g(q['reliability']['reliability'])}) | {hc.g(q['reliability']['reliability_ci95'][0])} | 0.9 | {pf(q['reliability']['pass_lower_bound'])} |",
        f"\n**wikitext_log_ppl: {'QUALIFIED' if q['qualified'] else 'NOT QUALIFIED'}**; entries with h_i > delta/2: {q['n_entries_over_half_delta']} of {q['n_entries']}.\n",
        "## 3. Estimate and the six intervals, each next to its simulated null size and coverage\n",
        f"D3 fit: {fit['n_rows']} development rows ({fit['n_stratum']} stratum + {fit['n_null']} null-family controls; the 160 registered controls minus the three legacy exclusions = 157), five inner folds, StatNN seeds 0,1,2 averaged before the loss, isotonic on incremental norm. "
        f"Empirical OOF row-summed ratio on the stratum rows: **theta_hat = {hc.g(fit['theta_hat_stratum'])}**; n_eff = {hc.g(fit['n_eff'])}; top-five component share of sum B = {hc.g(fit['top5_share'])}.\n",
        "Null size = simulated one-sided rejection at theta = 0 (limit 0.035); coverage = simulated two-sided 95% coverage at theta = 0.15 (limit 0.93); both in the empirical scenario of this gate.\n",
        "| Interval | theta interval | simulated null size | simulated coverage at 0.15 |",
        "|---|---|---:|---:|",
        *[
            f"| {r['interval']} | {ci(r['theta_interval'])} | {hc.g(r['null_size'])} | {hc.g(r['coverage_at_0.15'])} |"
            for r in g["interval_table"]
        ],
        "",
        "A2.1 (iv) leave-out (rows with h_i > delta/2; the table is identical to all rows when none exceed):\n",
        "| Rows | n rows | theta_hat | cond. bootstrap-t | joint bootstrap-t | cond. BCa | joint BCa |",
        "|---|---:|---:|---|---|---|---|",
        *[
            f"| {label} | {v['n_rows']} | {hc.g(v['theta_hat'])} | {ci(v['conditional_bootstrap_t'])} | {ci(v['joint_bootstrap_t'])} | {ci(v['conditional_bca'])} | {ci(v['joint_bca'])} |"
            for label, v in fit["leave_out_sensitivity"].items()
        ],
        "",
        "## 4. Gate criteria, PASS/FAIL (simulated data only)\n",
        f"Simulation scale: {rep['panels']} conditional panels, {rep['joint_panels']} joint panels, {rep['draws']} bootstrap draws (registered: 4,000 / 4,000 / 4,000), on the sealed component sizes (106 components, metadata only) with donor D3 components; scenarios empirical, item noise x 1.5, components merged in pairs.\n",
        "| Scenario | theta (true) | procedure | method | rejection | power lower 95% | coverage | mean half-width | P(decision pass) | denom. failures |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scen, by in g["results"].items():
        for entry in by.values():
            for world in ("conditional", "joint"):
                for method in ("bootstrap_t", "bca", "percentile"):
                    s = entry[world][method]
                    lines.append(
                        f"| {scen} | {hc.g(entry['theta_true_population'], 3)} | {world} | {method} | {hc.g(s['rejection'])} | {hc.g(s['rejection_wilson_lower95'])} | {hc.g(s['coverage95'])} | {hc.g(s['mean_half_width'])} | {hc.g(s['p_decision_pass'])} | {hc.g(s['denominator_failure_fraction'])} |"
                    )
    lines += [
        "",
        "Criteria by interval method (A2.3 order: bootstrap-t, then BCa; percentile is reported alongside and cannot choose):\n",
    ]
    for method, m in crit["per_method"].items():
        lines.append(f"- `{method}`: all criteria {pf(m['all_pass'])}")
        lines += [
            f"  - {k}: {hc.g(v['value'])} {pf(v['pass'])}" for k, v in m.items() if k != "all_pass"
        ]
    lines += [
        "",
        f"(d) qualification (A2.1) met for WikiText: {pf(crit['(d) qualification A2.1 met (WikiText)'])}.",
        "(e) P(decision pass) at theta = 0.15 / 0.20 / 0.25 (bootstrap-t; conditional, joint): "
        + "; ".join(
            f"{k.replace('theta_', '')}: {hc.g(v['conditional'])}, {hc.g(v['joint'])}"
            for k, v in crit[
                "(e) P(decision pass) at 0.15 / 0.20 / 0.25 (conditional, joint; bootstrap-t)"
            ].items()
        )
        + " (the rows of the table above list all three methods).\n",
        "## 5. Mandatory descriptive companions (A4.3), WikiText log-PPL\n",
        "Everything in this section is descriptive and post hoc; it does not replace the registered theta_hat above. Strata are outcome-free (family, incremental-norm tertile).\n",
        *hc.render_companions(c, level="###"),
        "## 6. What fresh WikiText scoring would cost\n",
        f"No GPU was used here. Board A's derivation (not a measurement): about 0.4 min per delta x 281 unique fresh deltas (255 plus the six spares, as in the sealed partition) = **about {GPU_COST_H} GPU-h**, plus the engineering of a sealed WikiText window panel disjoint from the 64 development windows and the A3 bitwise re-check (about 0.2 h). "
        + (
            "The gate is met, so this run would be eligible; it is not authorised by this report."
            if crit["a4_5_gate_met"]
            else "The gate is not met, so it is **not recommended**: a fresh run would yield another exploratory estimate of the same kind."
        ),
        "",
        "Limitation: the D3 fit uses the 229 stratum rows plus the 157 null-family controls (160 registered less three legacy exclusions), a small training set for the StatNN-versus-isotonic comparison. Residual learners are trained on inner-OOF residuals (the registered nested-fold design).",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panels", type=int, default=4000)
    parser.add_argument("--joint-panels", type=int, default=None)
    parser.add_argument("--draws", type=int, default=rq.DRAWS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-splits", type=int, default=hc.N_RANDOM_SPLITS)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--out-doc", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--tf-dir", type=Path, default=ROOT / "data/evals/qwen3.5-0.8b/tf_v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/prereg/panel_manifest.json")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/qwen3.5-0.8b")
    args = parser.parse_args()
    args.restrict_ids = None
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    joint_panels = args.joint_panels or args.panels
    report, ctx = rq.assemble(args, log)
    arrays = report.pop("_arrays")
    q = report["qualification"][ENDPOINT]
    half = arrays[ENDPOINT][2]
    pop, info, _ = rq.d3_fit(ctx, args, log, ENDPOINT)
    over = {d["entry"] for d in q["iv_disclosure_over_half_delta"]}
    keep = np.array([i not in over for i in ctx["stratum"]])
    comp_idx = {c: k for k, c in enumerate(sorted(set(ctx["codes"])))}
    row_comp = np.array([comp_idx[c] for c in ctx["codes"]])
    sens = {}
    for label, sel in (("all_rows", np.ones(len(keep), bool)), ("without_over_half_delta", keep)):
        rows_sel = np.flatnonzero(sel)
        _, comp_sel = np.unique(row_comp[rows_sel], return_inverse=True)
        sens[label] = rq.observed_intervals(pop, rows_sel, comp_sel, 1.0, args.draws)
        sens[label]["n_rows"] = int(len(rows_sel))
    log(f"observed WikiText theta_hat = {sens['all_rows']['theta_hat']:.4f}")
    results = rq.run_scenarios(
        pop, ctx["manifest"], args.panels, joint_panels, args.draws, args.workers, SCENARIOS, log
    )
    verdict = rq.gate_verdict(results, args.panels, joint_panels, args.draws, q["qualified"])
    comp, _ = hc.compute_companions(ctx, args, ENDPOINT, half, log, args.draws, args.random_splits)
    flex = comp["flexible_nuisance"]["family_x_natural_spline"]["theta"]
    info["leave_out_sensitivity"] = sens
    crit = criteria(results, verdict, q["qualified"], flex, args.panels, joint_panels, args.draws)
    wd = wording(sens["all_rows"])
    a4 = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", "docs/PREREG_H1_TF.md"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    report.update(
        {
            "endpoint": ENDPOINT,
            "a4_commit": a4,
            "panels": args.panels,
            "joint_panels": joint_panels,
            "draws": args.draws,
            "companions": comp,
            "gate": {
                "fit": info,
                "results": results,
                "verdict": verdict,
                "criteria": crit,
                "wording": wd,
                "interval_table": interval_vs_simulation(sens["all_rows"], results),
            },
        }
    )
    out_json = args.out_json or ROOT / "data/runs/h1w/gate_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            report,
            indent=1,
            sort_keys=True,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )
    doc = args.out_doc or ROOT / f"docs/reviews/P5/h1w-gate-{args.date}.md"
    doc.write_text(render(report))
    log(f"wrote {doc} and {out_json}")


if __name__ == "__main__":
    main()
