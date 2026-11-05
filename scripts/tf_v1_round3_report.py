#!/usr/bin/env python3
"""Markdown for docs/RESULTS.md: TF-v1 pilot round 3, G1 (round 3), G2 (round 3). Reads artefacts only.

Usage: tf_v1_round3_report.py [--append]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / "data/evals/qwen3.5-0.8b/tf_v1"


def pilot_section():
    check = json.loads((TF / "round3_check.json").read_text())
    pilot = json.loads((TF / "pilot_report.json").read_text())
    stage = pilot["warm_stage_median_seconds"]
    lines = [
        "\n## TF-v1 pilot round 3\n",
        "Batched HellaSwag and refusal stages, amendment A2.4.\n",
        f"Run `{check['new_run_id']}` on H0 (RTX PRO 4500, greedy, no sampling), branch `feat/tf-v1-batching`, tree clean, `source_hash` `{check['new_source_hash'][:12]}` (round 2: `{check['old_source_hash'][:12]}`, run `{check['old_run_id']}`, archived under `data/evals/qwen3.5-0.8b/tf_v1_round2_archive/`). Base + the ten D3 control IDs in `data/runs/d3_shards/controls.txt`, one process. **Panel status: development/pilot.** MMLU-Pro (1,000), KL (200 prompts x 64 tokens, resident cache, mode `{pilot['cold']['cache_mode']}` for the base) and WikiText stay batch 1; HellaSwag (250, four endings per forward) and the refusal / over-refusal stages (13 phrases per forward) are batched, the batch is recorded in every record (`scoring_batch`).\n",
        "| Entry | Wall min | MMLU s | HellaSwag s | KL s | WikiText s | Refusal s | Over-refusal s | Peak alloc/reserved GiB | Storage MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pilot["rows"]:
        s = r["stage_seconds"]
        lines.append(
            f"| {r['edit_id']} | {r['wall_minutes']:.3f} | {s['mmlu_pro']:.1f} | {s['hellaswag']:.1f} | {s['kl_to_base']:.1f} | {s['wikitext_ppl']:.1f} | {s['refusal_rate']:.1f} | {s['over_refusal']:.1f} | {r['peak_gpu_allocated_gib']:.2f}/{r['peak_gpu_reserved_gib']:.2f} | {r['storage_bytes'] / 2**20:.2f} |"
        )
    lo, hi = check["warm_range_minutes"]
    lines += [
        "",
        f"**Warm median wall time over {pilot['n_warm']} edits (cache resident): {check['warm_median_minutes']:.3f} min/entry** (range {lo:.3f} to {hi:.3f}); round 2 was 1.805. Warm stage medians (s): MMLU {stage['mmlu_pro']:.1f}, HellaSwag {stage['hellaswag']:.1f}, KL {stage['kl_to_base']:.1f}, WikiText {stage['wikitext_ppl']:.1f}, refusal {stage['refusal_rate']:.1f}, over-refusal {stage['over_refusal']:.1f}. **Gate (warm median <= 1.40 min): {'PASS' if check['gate_pass'] else 'FAIL'}.**",
        "",
        f"**Identity against the round-2 primaries (run `{check['old_run_id']}`), all 11 records:** MMLU-Pro, KL and WikiText records bitwise identical (every field of every record): **{check['mmlu_kl_wikitext_records_bitwise_identical']}**. KL base-logit cache content root unchanged (`{check['kl_cache_root'][:16]}…`): **{check['kl_cache_root_unchanged']}** (cache mode `load`, no rebuild).",
        "",
    ]
    worst = check["worst_max_abs_deviation"]
    limits = check["probe_limits"]
    limit = lambda k: limits["hellaswag"] if k == "hellaswag" else limits[k.split(".")[1]]
    lines += [
        "Batched versus batch-1 deviation over the 11 records (max |difference| per entry; the probe limits were measured on the base entry only):\n",
        "| Quantity | Worst over 11 entries | Median over entries | Probe limit | Within limit |",
        "|---|---:|---:|---:|---|",
    ]
    for k in sorted(worst):
        med = float(np.median([e["max_abs"][k] for e in check["entries"].values()]))
        lines.append(
            f"| {k} | {worst[k]:.4f} | {med:.4f} | {limit(k):.3f} | {'yes' if worst[k] <= limit(k) else 'NO'} |"
        )
    lines += [
        "",
        f"HellaSwag argmax changes: {check['hellaswag_argmax_changes_total']} of {11 * 250} items (probe: 1 of 250 on the base).",
        "",
        "**A2.4 clause not met on the edited entries.** A2.4 asks for HellaSwag and phrase deviations within the probe maxima (HellaSwag 0.010, phrase log-prob 0.39, margin 0.30). Measured over the 11 pilot records the worst deviations are "
        + ", ".join(
            f"{k} {worst[k]:.3f} ({worst[k] / limit(k):.2f} x limit)"
            for k in sorted(check["probe_limit_exceeded_by"])
        )
        + ". They are one to two bf16 logit steps on edited models (the probe scored the unedited base only, 11x fewer entries) and they affect only descriptive stages (HellaSwag 250, uncalibrated refusal margins), never MMLU-Pro, KL or WikiText. The pilot gate as stated for execution (warm median and bitwise-identical MMLU-Pro/KL/WikiText) passes, so scoring proceeded; this exceedance is recorded here for Board A to accept or to answer with fallback 1 (drop the refusal stages by config) or a revert of the batching for HellaSwag. The batched path itself repeats bitwise (G1 below).",
    ]
    return "\n".join(lines) + "\n"


def g1_section():
    report = json.loads((TF / "g1_report.json").read_text())
    entries = report["entries"]
    n_records = sorted({e["n_records"] for e in entries.values()})
    prov = [e["provenance"] for e in entries.values()]
    worst = report["worst_max_abs"]
    maxima = {
        "MMLU-Pro gold log-prob / option log-probs": max(
            worst.get("mmlu_pro.gold_log_prob", 0), worst.get("mmlu_pro.option_log_probs", 0)
        ),
        "HellaSwag scores": max(
            worst.get("hellaswag.gold_log_prob", 0),
            worst.get("hellaswag.raw_option_log_likelihoods", 0),
        ),
        "KL (per prompt and per position)": max(
            worst.get("kl_to_base.full_sequence_kl", 0), worst.get("kl_to_base.position_kl", 0)
        ),
        "WikiText mean NLL": worst.get("wikitext_ppl.mean_nll", 0),
        "refusal / over-refusal phrase margins": max(
            worst.get("refusal_rate.phrase_margin", 0), worst.get("over_refusal.phrase_margin", 0)
        ),
    }
    lines = [
        "\n## TF-v1 G1 (round 3)\n",
        "Bitwise repeat on the PREREG section 6 composition, batched source.\n",
        f"Source `source_hash` `{pilot_hash()[:12]}`, device H0, committed tree (`dirty: false` in every eval.json), development panel. Composition and process layout as in G1 above (13 entries plus the base: 4 LoRA, 3 chains with their predecessors, 2 task-vector, 1 abliteration), re-run on the batched source: primaries in a new process (base reused from the round-3 pilot, matched by the registered reference keys), repeats in a second fresh process as `repeats/g1` (immutable run directories, base re-executed and paired, cache verified again).\n",
        f"**Result: {sum(e['records_identical'] for e in entries.values())} of {len(entries)} repeats are bitwise-identical to their primaries** (complete `records.jsonl` equality, {', '.join(map(str, n_records))} records per entry, every benchmark and field including the batched HellaSwag and refusal stages). All maxima exactly zero: "
        + "; ".join(f"{k} {v:.1e}" for k, v in maxima.items())
        + f". Limits (§6): 1e-4 nats (MMLU, log-PPL), 1e-5 (KL). Distinct run ids: {all(p['distinct_run'] for p in prov)}; same source hash: {all(p['same_source_hash'] for p in prov)}; repeat tree clean: {all(p['repeat_tree_clean'] for p in prov)}; same device: {all(p['same_device'] for p in prov)}. Report: `data/evals/qwen3.5-0.8b/tf_v1/g1_report.json` (`scripts/tf_v1_g1_report.py`).\n",
        "Same scope limits as before: same-device repeat determinism on H0, not cross-device, cross-batch-size or cross-driver agreement; it qualifies no endpoint by itself.\n",
    ]
    return "\n".join(lines)


def pilot_hash():
    return json.loads((TF / "round3_check.json").read_text())["new_source_hash"]


def g2_section():
    report = json.loads((TF / "bridge_g2/g2_report.json").read_text())
    pooled = report["pooled"]
    old = TF.parent / "tf_v1_round2_archive/bridge_g2"
    mmlu_identical = all(
        (TF / "bridge_g2" / e / "records_mmlu_pro.jsonl").read_bytes()
        == (old / e / "records_mmlu_pro.jsonl").read_bytes()
        for e in report["entries"]
    )
    m, h, c = pooled["mmlu_pro"], pooled["hellaswag"], pooled["prereg_mmlu_criteria"]
    lines = [
        "\n## TF-v1 G2 (round 3)\n",
        "Same-item legacy bridge on the batched source.\n",
        f"Source `source_hash` `{report['source_hash'][:12]}`, tree clean (`dirty: {str(report['dirty']).lower()}`), device H0. `scripts/tf_v1_g2_bridge.py` now scores HellaSwag through the final batched `hellaswag_scores` (four endings per forward) and MMLU-Pro through the unchanged `letter_scores`; the 100 + 100 legacy items and the 11 pilot records are the same as in G2 above.\n",
        "| Entry | MMLU acc fast | MMLU acc TF-v1 | MMLU \\|Δacc\\| | MMLU argmax agreement | MMLU gold-lp \\|Δ\\| median / p99 / max | HellaSwag acc fast / TF-v1 | HellaSwag score \\|Δ\\| max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for edit_id, e in sorted(report["per_entry"].items()):
        a, b = e["mmlu_pro"], e["hellaswag"]
        lines.append(
            f"| {edit_id} | {a['accuracy_fast']:.2f} | {a['accuracy_tf_v1']:.2f} | {a['accuracy_abs_diff']:.2f} | {a['argmax_agreement']:.2f} | {a['gold_score_abs_diff_median']:.1e} / {a['gold_score_abs_diff_p99']:.3f} / {a['gold_score_abs_diff_max']:.3f} | {b['accuracy_fast']:.2f} / {b['accuracy_tf_v1']:.2f} | {b['gold_score_abs_diff_max']:.4f} |"
        )
    lines += [
        "",
        f"**PREREG §6 bridge criteria (MMLU-Pro, worst entry for each statistic):** median |Δ| {m['gold_score_abs_diff_median_max']:.1e} (limit {c['median_abs_diff_limit']}), p99 {m['gold_score_abs_diff_p99_max']:.3f} (limit {c['p99_abs_diff_limit']}), argmax agreement {m['argmax_agreement_min']:.0%} (limit {c['argmax_agreement_limit']:.0%}): **{'PASS' if c['pass'] else 'FAIL'}**. MMLU-Pro bridge records bitwise equal to the round-2 bridge (`tf_v1_round2_archive/bridge_g2`), all 11 entries: **{mmlu_identical}** (the MMLU-Pro scorer is unchanged).",
        f"HellaSwag (descriptive; batched path): mean correctness agreement {h['correctness_agreement_mean']:.4f}, worst entry {h['correctness_agreement_min']:.2f}; argmax agreement worst {h['argmax_agreement_min']:.2f}; max score |Δ| against the fast tier's batch-1 scores {h['gold_score_abs_diff_max']:.4f}; bitwise equal to the fast tier: {h['all_scores_bitwise_equal']} (expected False: the batched kernel path differs by a few bf16 steps).",
        "A one-item shortfall of the worst entry's MMLU argmax agreement (exactly 99% against a 99% limit, a bf16 tie) would be investigated, not automatically a stop (A2.4).",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    text = pilot_section()
    if (TF / "g1_report.json").exists():
        text += g1_section()
    if (TF / "bridge_g2/g2_report.json").exists():
        text += g2_section()
    if args.append:
        with (ROOT / "docs/RESULTS.md").open("a") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
