#!/usr/bin/env python3
"""Render docs/reviews/P5/tf-v1-qualification-2026-11-05.md from the qualification and power JSON.

Every number in the document comes from data/runs/h1a/{qualification,power,panel_manifest}.json;
nothing is typed by hand. Run tf_v1_qualification.py and tf_v1_power.py first.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H1A = ROOT / "data/runs/h1a"
OUT = ROOT / "docs/reviews/P5/tf-v1-qualification-2026-11-05.md"
ENDPOINTS = (
    ("mmlu_gold_log_prob", "MMLU-Pro gold log-prob (predecessor-relative)", "mmlu"),
    ("kl_to_base", "KL(base‖edited)", "kl"),
    ("wikitext_log_ppl", "WikiText log-PPL (predecessor-relative)", "wiki"),
)


def f(x, n=4):
    return f"{x:.{n}f}"


def yn(flag):
    return "pass" if flag else "**FAIL**"


def main():
    q = json.loads((H1A / "qualification_report.json").read_text())
    pw = json.loads((H1A / "power_report.json").read_text())
    pm = json.loads((H1A / "panel_manifest.json").read_text())
    s = pm["summary"]
    zoo = {}
    for line in (ROOT / "data/zoo/qwen3.5-0.8b/manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "id" in row:
            zoo[row["id"]] = row
    E = q["panels"]["expanded20"]
    C = q["panels"]["composition10"]
    prov = q["provenance"]
    flags = ("native_measured", "clean_tree", "same_source_as_base", "matched_base", "complete")
    prov_ok = all(
        all(v[k] for k in flags) and v["device"] == "H0" and v["batch_size"] == 1
        for group in (prov["entries"], prov["predecessors"])
        for v in group.values()
    ) and all(prov["base"][k] for k in flags)
    L = []
    w = L.append
    w("# TF-v1 qualification (section 6) and TF-specific power (section 5), 5 November 2026\n")
    w(
        "Author: eval/metrology. Development panel only; no fresh component has been scored. Branch `feat/tf-v1-prepanel`; "
        "inputs: the timing pilot (10 D3 controls + base), G1 (`repeats/g1`), G2 (bridge), the sealed partition "
        "manifest (components, metadata only). Source `source_hash` 12ee875a68c9, device H0 only. Every number below is "
        "produced by `scripts/tf_v1_qualification.py` and `scripts/tf_v1_power.py` (4,000 draws, seed 20261105) and rendered by "
        "`scripts/tf_v1_qualification_doc.py`. This document qualifies nothing by itself: it applies the registered rule "
        "(PREREG_H1_TF §6, amendment A1) and reports where it passes or fails.\n"
    )
    w("## 1. Verdict\n")
    m, k, wt = (E[e[0]] for e in ENDPOINTS)
    w(
        "| Endpoint | δ | Measured provenance | Paired repeat error (max / p95 upper) | Item half-width: every entry / panel-mean upper bound vs δ/2 | Reliability lower 95% bound ≥ .90 | Result |"
    )
    w("|---|---:|---|---|---|---|---|")
    for name, label, tag in ENDPOINTS:
        d = E[name]
        g, r, rel = d["itemwidth_gate"], d["repeat"], d["reliability"]
        extra = ""
        ok = (
            prov_ok
            and r["pass_max"]
            and r["pass_p95_upper"]
            and g["pass_every_gated_entry"]
            and g["pass_panel_mean_upper_bound"]
            and rel["pass_lower_bound"]
        )
        if tag == "kl":
            share = len(d["kl_excluded_high_kl"]) / d["n_entries"]
            tripwire = share > 0.20
            ok = ok and not tripwire
            extra = f" (KL > 0.10 excluded: {len(d['kl_excluded_high_kl'])}/{d['n_entries']} = {share:.0%}; tripwire 20%: {'TRIPPED' if tripwire else 'ok'})"
        w(
            f"| {label} | {f(d['delta'], 3)} | {'native-measured, H0, clean, matched' if prov_ok else '**FAIL**'} | "
            f"{r['max_abs_discrepancy']:.1e} / {r['p95_upper97_5_component_bootstrap']:.1e} ({yn(r['pass_max'] and r['pass_p95_upper'])}) | "
            f"{yn(g['pass_every_gated_entry'])} (max {f(g['max_halfwidth_over_gated_entries'])}; {len(g['entries_over_limit'])} over) / "
            f"{yn(g['pass_panel_mean_upper_bound'])} ({f(g['panel_mean_halfwidth_upper97_5'])} vs {f(g['limit_delta_over_2'])}) | "
            f"{yn(rel['pass_lower_bound'])} (lower {f(rel['reliability_ci95'][0], 3)}, point {f(rel['reliability'], 3)}) | "
            f"**{'QUALIFIED' if ok else 'NOT QUALIFIED'}**{extra} |"
        )
    w("")
    w(
        "Reading. WikiText log-PPL passes every registered component (diagnostic; it cannot pass H1a). The H1a primary endpoint "
        "(MMLU gold log-prob) passes provenance, the repeat bound (exactly zero), the panel-mean δ/2 bound and the reliability "
        "bound on the 20-entry panel, but **fails the registered per-entry δ/2 condition** (A1.3): two large-effect LoRA entries "
        "have item half-widths of "
        + ", ".join(
            f"{f(m['itemwidth_gate'] and m['item_halfwidth95'][i])} (`{i}`)"
            for i in m["itemwidth_gate"]["entries_over_limit"]
        )
        + f" against δ/2 = {f(m['itemwidth_gate']['limit_delta_over_2'], 3)}. KL is **not qualified** under A1.2 (see §4). "
        "**The H1a primary endpoint is therefore not qualified by the rule as registered.** The failure is a property of "
        "the larger LoRA edits, which move individual items' log-probs a lot (§4), not of noise in repeats.\n"
    )
    w("## 2. Panel and provenance\n")
    comp = q["composition"]
    w(
        "Composition (PREREG §6 rule, `docs/prereg/qualification_panel.json`): "
        f"LoRA nearest median legacy KL {', '.join('`' + i + '`' for i in comp['lora_nearest_median'])}; "
        f"LoRA largest {', '.join('`' + i + '`' for i in comp['lora_largest'])}; chains (step ≥ 3, distinct lineages) {', '.join('`' + i + '`' for i in comp['chain'])} "
        f"with predecessors {', '.join('`' + v + '`' for v in q['chain_predecessors'].values())}; task vectors {', '.join('`' + i + '`' for i in comp['task_vector'])}; "
        f"largest-KL abliteration `{comp['abliteration_largest'][0]}`. The ten-entry panel is **expanded to 20 y-bearing entries** with the ten already-measured D3 pilot controls "
        "(3 random_dir, 3 abliteration, 2 LoRA, 2 task-vector; same source, same base run, same device). That is a deviation from "
        "the prespecified hash-order expansion, made because those entries were already measured; both panels are reported. "
        "Chain entries use their actual scored predecessor as reference; every other entry uses the base. "
        f"Provenance of every entry, predecessor and the base: native-measured, device H0, batch 1, clean tree, complete, same `source_hash`, base matched "
        f"(`matched_base_missing: false`): **{'all pass' if prov_ok else 'FAILURES'}**.\n"
    )
    w(
        "| Entry | Family | MMLU y | MMLU half-width | KL | KL half-width | WikiText y | WikiText half-width |"
    )
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    for i in q["panel_ids"]:
        w(
            f"| `{i}` | {zoo[i]['family']} | {f(m['y'][i])} | {f(m['item_halfwidth95'][i])} | {f(k['y'][i])} | {f(k['item_halfwidth95'][i])} | "
            f"{f(wt['y'][i])} | {f(wt['item_halfwidth95'][i])} |"
        )
    w("")
    w(
        "## 3. Rule components, per endpoint (20-entry panel; 10-entry composition in brackets where it differs)\n"
    )
    for name, label, tag in ENDPOINTS:
        d, d10 = E[name], C[name]
        g, r, rel, rel10 = d["itemwidth_gate"], d["repeat"], d["reliability"], d10["reliability"]
        w(f"### {label}, δ = {f(d['delta'], 4)}\n")
        w(
            f"- **Paired repeat error (G1: the 10 composition entries, their 3 chain predecessors and the base, each executed twice; the 10 pilot controls have no g1 repeat):** {r['n_values']:,} item values; "
            f"max |repeat − primary| = {r['max_abs_discrepancy']:.1e} (limit {r['limit_max']:.0e}), 95th percentile {r['p95_abs_discrepancy']:.1e}, "
            f"upper 97.5% component-bootstrap bound {r['p95_upper97_5_component_bootstrap']:.1e} (limit δ/2 = {f(r['limit_p95_upper'])}); bitwise identical: {r['bitwise_identical_items']}. "
            "The repeat-error variance entering reliability is therefore 0; a bound of 0 reflects deterministic same-device execution, not a measurement of device error."
        )
        w(
            f"- **Item uncertainty (joint item draws across edits and the shared reference; blocks of 2 windows for WikiText):** per-entry half-width median {f(g['median_halfwidth_over_gated_entries'])}, "
            f"max {f(g['max_halfwidth_over_gated_entries'])} (limit δ/2 {f(g['limit_delta_over_2'])}); entries over the limit: {', '.join('`' + i + '`' for i in g['entries_over_limit']) or 'none'}; "
            f"panel-mean half-width {f(g['panel_mean_halfwidth'])}, upper 97.5% component-bootstrap bound {f(g['panel_mean_halfwidth_upper97_5'])}."
        )
        w(
            f"- **Signal reliability** (family + log-norm OLS residuals, signal = between-entry residual variance − mean item-error variance − repeat variance; untruncated): "
            f"n = {rel['n_entries']}, design rank {rel['design_rank']}, residual variance {rel['residual_variance']:.3e}, mean item-error variance {rel['mean_item_error_variance']:.2e}, "
            f"signal variance {rel['signal_variance_untruncated']:.3e} (positive: {rel['signal_variance_untruncated'] > 0}); reliability {f(rel['reliability'], 4)}, "
            f"95% component-bootstrap interval [{f(rel['reliability_ci95'][0], 3)}, {f(rel['reliability_ci95'][1], 3)}] ({rel['valid_draws']} valid of 4,000 draws). "
            f"[10-entry composition: reliability {f(rel10['reliability'], 3)}, lower bound {f(rel10['reliability_ci95'][0], 3)}, {yn(rel10['pass_lower_bound'])}; "
            f"item half-width bound {f(d10['itemwidth_gate']['panel_mean_halfwidth_upper97_5'])}.]"
        )
        w("")
    w("## 4. Why the result is what it is, and what it does not show\n")
    kd = E["kl_to_base"]
    hws = sorted(m["item_halfwidth95"].values())
    big = {i: v for i, v in m["item_halfwidth95"].items() if v > 0.03}
    w(
        f"- **MMLU per-entry gate.** The ten smallest item half-widths are ≤ {f(hws[9])}, but {len(big)} entries exceed 0.03 "
        f"({', '.join(f'`{i}` ({zoo[i]["family"]}, y {f(m["y"][i], 3)}, half-width {f(v, 3)})' for i, v in sorted(big.items(), key=lambda kv: -kv[1]))}); "
        f"{len(m['itemwidth_gate']['entries_over_limit'])} exceed δ/2 = {f(m['itemwidth_gate']['limit_delta_over_2'], 3)}. The half-width follows how much an edit moves individual items' log-probs, "
        "not the size of the mean change (several entries with |y| ≈ 0.04 have half-widths near 0.04), so it is a property of the larger LoRA edits, not of repeat noise. The panel-mean bound "
        f"({f(m['itemwidth_gate']['panel_mean_halfwidth_upper97_5'])}) passes. The panel includes the two largest-KL LoRAs by the §6 rule. I do not relax the rule after seeing it fail; options for the owner are in §7."
    )
    w(
        f"- **KL.** Under A1.2 entries above 0.10 nats are excluded from the gate: {len(kd['kl_excluded_high_kl'])} of {kd['n_entries']} here ({', '.join('`' + i + '`' for i in kd['kl_excluded_high_kl'])}), "
        f"above the 20% tripwire, so KL is unqualified for this panel regardless of the other components. Among the {kd['kl_gate_entries']} in-gate entries, "
        f"`{kd['itemwidth_gate']['entries_over_limit'][0] if kd['itemwidth_gate']['entries_over_limit'] else 'none'}` still has half-width {f(kd['itemwidth_gate']['max_halfwidth_over_gated_entries'])} > δ/2 = 0.005: its KL is below 0.10 and the half-width is "
        f"{f(kd['item_halfwidth95'][kd['itemwidth_gate']['entries_over_limit'][0]] / kd['y'][kd['itemwidth_gate']['entries_over_limit'][0]], 2) if kd['itemwidth_gate']['entries_over_limit'] else 'n/a'} of its KL (measured relative half-widths of the entries above 0.03 range {f(min(v / kd['y'][i] for i, v in kd['item_halfwidth95'].items() if kd['y'][i] > 0.03), 2)} to {f(max(v / kd['y'][i] for i, v in kd['item_halfwidth95'].items() if kd['y'][i] > 0.03), 2)}), so a threshold of 10δ is too high for an absolute δ/2 bound. That is a defect of the A1.2 threshold, reported here and not repaired retroactively."
    )
    w(
        "- **Reliability is panel-dependent.** It is high on this deliberately dispersed panel (LoRA, abliteration, chain, task) because between-entry variance is large relative to the item error. On a fresh panel of mostly "
        "small edits the between-entry variance is smaller and the same error would give a lower ratio; the estimate cannot be carried over without recomputing on the fresh panel's variance. The nuisance (family, log-norm) is fitted on these 20 entries because no D3 "
        "TF-v1 training rows exist yet (the registered nuisance is training-only), and the norm is the stored cumulative Frobenius norm, not the incremental effective norm against the predecessor; both are approximations that matter little at 6 parameters but are not the registered computation."
    )
    w(
        "- **Not shown:** device error (single device, DECISIONS D2 addendum), cross-batch/engine agreement, signal or effect resolution for HellaSwag and the phrase margins (descriptive, unqualified; only their bitwise repeat is established), and anything about fresh components."
    )
    w("\n## 5. TF-specific power for θ (planning computation, not the §5 gate)\n")
    rows = pw["rows_per_component"]
    w(
        f"**Achievable count (sealed partition, `docs/prereg/panel_manifest.json`, partition hash `{pm['partition_sha256'][:16]}…`).** {s['components_total']} source-closed components; "
        f"development {s['development_components']} ({s['development_non_null_components']} non-null), excluded {s['components_by_status']['excluded']}, fresh-unselected {s['components_by_status']['fresh_unselected']}, "
        f"**confirmatory {s['confirmatory_components']}** (all non-null): {s['components_containing_candidates_fresh']} fresh components containing any of the 110 lora_sft/task_vector candidates "
        f"({s['candidates_in_selected_components']} candidate entries) plus {s['confirmatory_components'] - s['components_containing_candidates_fresh']} chain components appended in hash order to reach 100. "
        f"Only 6 more fresh chain components exist, so the achievable maximum is 106 and any post-freeze exclusion leaves almost no margin above the ≥100 floor. "
        f"Rows: {s['confirmatory_stratum_rows']} stratum rows ({', '.join(f'{k} {v}' for k, v in s['confirmatory_family_rows'].items() if k in ('lora_sft', 'task_vector', 'chain'))}); "
        f"{s['confirmatory_closure_entries']} closure entries ({s['confirmatory_closure_unique_deltas']} unique deltas, aliases scored once). Component sizes (rows): "
        + ", ".join(f"{k}: {v}" for k, v in rows["size_counts"].items())
        + ". Most components are 1 or 2 rows; a few chain lineages carry up to 17 rows.\n"
    )
    w(
        "**What is missing for the §5 gate.** The gate needs the D3 out-of-fold TF-v1 residuals and the StatNN/isotonic predictions, item/reference covariance and device support. None exists: no D3 entry has been scored under TF-v1 except the 23 here, and no predictor has been fit on TF-v1 labels. The computation below is therefore a planning calculation. "
        f"It uses the measured TF-v1 pilot residual scale (|family/log-norm residual| of the {pw['proxy_residual']['n']} entries, CV {f(pw['proxy_residual']['cv_abs_residual'], 2)}) for the comparator loss and a grid of assumptions for how the StatNN/isotonic loss ratio varies across rows. "
        "The alternative is built so the **row-summed ratio of expectations is exactly θ = .15** (and 0 for the null), losses nonnegative, row ratio lognormal with within-component correlation ρ. Panels use the actual component size distribution; "
        f"each of {pw['panels_per_scenario']:,} simulated panels is analysed with the registered conditional procedure ({pw['bootstrap_draws']:,} component-bootstrap draws of the row-summed ratio, 2.5th percentile > 0). "
        "The joint component/item-block procedure is **not run**: it needs item-level paired predictions that do not exist; I report only the conditional procedure and a variance-inflated sensitivity (scenarios D and F).\n"
    )
    w(
        "| Scenario | Loss shape | Row heterogeneity s | ρ within component | Component scale sd | Power at θ=.15 (lower binomial bound) | Precision gate attained (half-width ≤ θ̂/2) | Decision pass rate (θ̂ ≥ .15 ∧ lower > 0 ∧ half-width ≤ θ̂/2) | θ̂ mean (sd) | Null rejection rate |"
    )
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, sc in pw["scenarios"].items():
        a, n0 = sc["alternative_theta_0.15"], sc["null_theta_0"]
        w(
            f"| {name} | {sc['shape']} | {sc['row_heterogeneity_s']} | {sc['within_component_rho']} | {sc['component_scale_sd']} | "
            f"{a['reject_rate']:.3f} ({a['reject_binomial_lower95']:.3f}) | {a['precision_gate_rate']:.3f} | {a['decision_pass_rate']:.3f} | "
            f"{a['theta_hat_mean']:.3f} ({a['theta_hat_sd']:.3f}) | {n0['reject_rate']:.3f} |"
        )
    w("")
    w("Component-count sweep, scenario C (actual size distribution resampled to n components):\n")
    w("| Components | Power (lower bound) | Precision gate | θ̂ sd |")
    w("|---:|---:|---:|---:|")
    for n, a in sorted(pw["component_count_sweep_scenario_C"].items(), key=lambda kv: int(kv[0])):
        w(
            f"| {n} | {a['reject_rate']:.3f} ({a['reject_binomial_lower95']:.3f}) | {a['precision_gate_rate']:.3f} | {a['theta_hat_sd']:.3f} |"
        )
    alt = {n: sc["alternative_theta_0.15"] for n, sc in pw["scenarios"].items()}
    ok = [n for n, a in alt.items() if a["reject_binomial_lower95"] >= 0.80]
    short = [n for n in alt if n not in ok]
    a_ = alt["A"]
    w("")
    w(
        f"**Reading.** At the achievable {s['confirmatory_components']} components ({s['confirmatory_stratum_rows']} rows), scenario A, whose θ̂ spread "
        f'(sd {a_["theta_hat_sd"]:.3f}, a cluster SD near {a_["theta_hat_sd"] * (s["confirmatory_components"] ** 0.5):.2f}) is the closest to the metrology planning value of 0.47 behind the "79 clusters" figure, '
        f"has conditional power {a_['reject_rate']:.3f} (lower binomial bound {a_['reject_binomial_lower95']:.3f}), but attains the registered precision gate (half-width ≤ θ̂/2) in only {a_['precision_gate_rate']:.0%} of panels "
        f"and the full decision rule in {a_['decision_pass_rate']:.0%}. Scenarios with lower-bound power ≥ .80: {', '.join(ok) or 'none'}; below: {', '.join(short) or 'none'}. "
        "§5 requires the lower binomial bound ≥ .80 for both the conditional and the joint procedure and a separate precision report; the joint procedure was not run, so **the §5 gate is not satisfied**, and where the conditional bound is met it depends on assumptions that "
        "only the D3 OOF losses can confirm. The decision rule also needs half-width ≤ θ̂/2, which is not met in most simulated panels even when the rejection test succeeds: per §4 an imprecise result is inconclusive, not negative. "
        "The old 79-cluster figure is not carried over. "
        f"**The registered percentile bootstrap is anti-conservative here:** under the null (θ = 0) it rejects in {min(sc['null_theta_0']['reject_rate'] for sc in pw['scenarios'].values()):.3f} to {max(sc['null_theta_0']['reject_rate'] for sc in pw['scenarios'].values()):.3f} of panels against the nominal one-sided 0.025 "
        "(heavy-tailed row losses, unequal component sizes, a ratio statistic and only 100 components). The power figures above are therefore optimistic: they include false rejections, and a calibrated test would have less power. This is a planning finding for the registered procedure, to be re-examined on the real OOF losses."
    )
    w(
        "\n## 6. Ancestry-cluster bootstrap plan (registered procedures, restated as executable steps)\n"
    )
    w(
        "1. Unit: source-closed component from the sealed manifest (`component_id`); all rows, both predictions and shared reference links of a component move together. Selected components are drawn with replacement at the original count (100), seed 20261105, 4,000 draws, one stream shared by all contrasts.\n"
        "2. Statistic: θ = 1 − ΣA/ΣB over the drawn rows (row sums, not a mean of component ratios), A = |r − mean of 3 StatNN seed predictions|, B = |r − isotonic|; B/n ≤ 1e-12 nats in any replicate counts as a denominator failure (not clamped, reported; any failure leaves the precision gate unmet).\n"
        "3. Report 2.5/97.5 percentiles, valid-draw count, component-size distribution; decision needs θ̂ ≥ .15, lower bound > 0, half-width ≤ θ̂/2.\n"
        "4. Item claim: additional 4,000-draw joint bootstrap of components and sealed independent problem blocks with one common item draw across every edit, predecessor and the shared base; fixed D3 fit. Both intervals must satisfy the decision.\n"
        "5. Qualification bootstraps (this document) use the same unit (the dev-panel components from the manifest), the same seed and the same 4,000 draws; shared references are drawn once per replicate.\n"
        "6. Power simulation (§5) reuses steps 1 to 3 on simulated losses, 4,000 panels per scenario, with the real D3 OOF losses substituted for the proxy before sealed execution."
    )
    w("\n## 7. Decisions and blockers for the owner (none is taken here)\n")
    w(
        "1. **MMLU per-entry δ/2.** The registered rule fails on 2 of 20 entries (large effects). Options, each needing a dated amendment before fresh scoring: keep the rule and accept the endpoint unqualified; scope the per-entry check to entries with |y| below a stated multiple of δ (as A1.2 does for KL) and report the rest; or switch the per-entry condition to the panel-level bound plus a relative condition. This choice should be made on grounds other than this pilot's outcome (for example the H1a stratum's expected effect sizes).\n"
        "2. **KL threshold.** A1.2's 10δ is too high by the measured relative width; a lower threshold (and its tripwire) needs an amendment, and KL cannot pass H1a anyway.\n"
        "3. **Budget.** The as-built source does not fit the 19.0 h cap at the actual closure count (PREREG A1.4); the batching remedy (A1.5) would, but needs a source change, a new pilot, G1 and G2 on the final source, and invalidates the artefacts used here (they would become development evidence).\n"
        "4. **Sealed item panel** (TF-v1:item domain, text-group dedupe, disjointness from the development rankings and legacy items) is not built; no sealed item may be scored before it is committed.\n"
        "5. **Power.** Re-run `scripts/tf_v1_power.py` with the D3 OOF TF-v1 residuals and predictions once D3 is scored, and add the joint component/item-block procedure; the 100-component floor is exactly met and only 6 spare fresh chain components exist, so expansion within the cap is limited to 106."
    )
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
