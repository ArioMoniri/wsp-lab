#!/usr/bin/env python3
"""Report measured TF-v1 round-2 pilot artifacts; never launch evaluations.

Usage: tf_v1_pilot_report.py [--load-probe LABEL --load-probe-id ID] [--append]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/evals/qwen3.5-0.8b"
GATE_PASS, GATE_REVISE = 2.0, 3.0  # minutes, PREREG_H1_TF section 9


def load(directory):
    payload = json.loads((directory / "eval.json").read_text())
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    return payload, records


def prefix_kl(records, length):
    kl = [r for r in records if r["benchmark"] == "kl_to_base"]
    return float(np.mean([np.mean(r["extra"]["position_kl"][:length]) for r in kl]))


def report(load_probe=None, probe_id=None):
    out = ZOO / "tf_v1"
    ids = ["base", *(ROOT / "data/runs/d3_shards/controls.txt").read_text().split()]
    rows, base_payload = [], None
    for edit_id in ids:
        directory = out / edit_id
        payload, records = load(directory)
        assert payload["eval_status"] == "complete" and payload["repeat_id"] is None
        assert not payload["matched_base_missing"] and payload["panel"] == "development"
        if edit_id == "base":
            base_payload = payload
        assert payload["source_hash"] == base_payload["source_hash"]
        assert payload["run_id"] == base_payload["run_id"] and not payload["dirty"]
        kl = [r for r in records if r["benchmark"] == "kl_to_base"]
        assert {r["extra"]["base_logits_sha256"] for r in kl} == {
            payload["tf_cache"]["base_logits_sha256"]
        }
        rows.append(
            {
                "edit_id": edit_id,
                "wall_minutes": payload["wall_seconds"] / 60,
                "stage_seconds": payload["stage_seconds"],
                "cache_mode": payload["tf_cache"]["mode"],
                "cache_seconds": payload["tf_cache"]["seconds"],
                "peak_gpu_allocated_gib": payload["peak_gpu_allocated_bytes"] / 2**30,
                "peak_gpu_reserved_gib": payload["peak_gpu_reserved_bytes"] / 2**30,
                "storage_bytes": sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
                "records_bytes": (directory / "records.jsonl").stat().st_size,
                "n_records": len(records),
                "mmlu_acc": payload["tf_v1"]["mmlu_pro"]["accuracy"],
                "gold_log_prob": payload["tf_v1"]["mmlu_pro"]["mean_gold_log_prob"],
                "kl": payload["tf_v1"]["kl_to_base"]["full_sequence_kl"],
                "kl_prefix": {n: prefix_kl(records, n) for n in (16, 32, 64)},
            }
        )
    edits = rows[1:]
    warm = [r["wall_minutes"] for r in edits if r["cache_mode"] == "resident"]
    cache_dir = (
        ZOO
        / "base/tf_cache"
        / base_payload["eval_device_id"]
        / base_payload["tf_cache"]["cache_hash"]
    )
    cache_bytes = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
    stages = sorted({k for r in edits for k in r["stage_seconds"]})
    result = {
        "run_id": base_payload["run_id"],
        "git_sha": base_payload["git_sha"],
        "source_hash": base_payload["source_hash"],
        "panel": base_payload["panel"],
        "rows": rows,
        "n_warm": len(warm),
        "warm_median_minutes": float(np.median(warm)),
        "warm_min_max_minutes": [float(min(warm)), float(max(warm))],
        "warm_stage_median_seconds": {
            k: float(np.median([r["stage_seconds"][k] for r in edits])) for k in stages
        },
        "cold": {
            "base_wall_minutes": rows[0]["wall_minutes"],
            "base_cache_build_seconds": base_payload["tf_cache"]["seconds"],
            "cache_mode": base_payload["tf_cache"]["mode"],
        },
        "model_load_seconds": base_payload["model_load_seconds"],
        "cache_bytes": cache_bytes,
        "cache_manifest_bytes": base_payload["tf_cache"]["cache_bytes"],
        "base_logits_sha256": base_payload["tf_cache"]["base_logits_sha256"],
        "median_entry_storage_bytes": float(np.median([r["storage_bytes"] for r in edits])),
        "median_records_bytes": float(np.median([r["records_bytes"] for r in edits])),
        "gate1": {
            "pass_line_minutes": GATE_PASS,
            "revise_line_minutes": GATE_REVISE,
            "status": "PASS"
            if np.median(warm) <= GATE_PASS
            else "REVISE (reduce scope, re-pilot)"
            if np.median(warm) <= GATE_REVISE
            else "FAIL (remove optional scope, re-pilot)",
        },
    }
    if load_probe:
        probe, probe_records = load(out / probe_id / "repeats" / load_probe)
        primary = load(out / probe_id)[1]
        kl_diff = max(
            abs(a["extra"]["full_sequence_kl"] - b["extra"]["full_sequence_kl"])
            for a, b in zip(probe_records, primary, strict=True)
            if a["benchmark"] == "kl_to_base"
        )
        gold_diff = max(
            abs(a["extra"]["gold_log_prob"] - b["extra"]["gold_log_prob"])
            for a, b in zip(probe_records, primary, strict=True)
            if a["benchmark"] == "mmlu_pro"
        )
        result["load_probe"] = {
            "entry": probe_id,
            "cache_mode": probe["tf_cache"]["mode"],
            "cache_load_verify_seconds": probe["tf_cache"]["seconds"],
            "wall_minutes": probe["wall_seconds"] / 60,
            "model_load_seconds": probe["model_load_seconds"],
            "max_abs_kl_diff_vs_primary": kl_diff,
            "max_abs_gold_log_prob_diff_vs_primary": gold_diff,
            "records_identical": [r for r in probe_records] == primary,
        }
    (out / "pilot_report.json").write_text(json.dumps(result, indent=2))
    return result


def markdown(result):
    rows = result["rows"]
    stage = result["warm_stage_median_seconds"]
    lines = [
        "\n## TF-v1 timing pilot (round 2)\n",
        f"Run `{result['run_id']}` on H0 (RTX PRO 4500, batch 1, greedy, no sampling), committed source `{result['git_sha'][:7]}` (`source_hash` {result['source_hash'][:12]}, tree clean). Base + the ten D3 control IDs in `data/runs/d3_shards/controls.txt`, one process; native metadata and same-run base matching in each eval.json. **Panel status: development/pilot** (items scored here are barred from the sealed panel). Config: MMLU-Pro 1,000, HellaSwag 250 (nested prefix of the 500-item ranking), KL 200 prompts x 64 greedy tokens (exact full vocabulary), WikiText 64, refusal 50 + 50 continuation-phrase margins. Round 1 (`20261105-131348-277686c3`, gate 1 failed at 4.525 min) is archived under `data/evals/qwen3.5-0.8b/tf_v1_round1_archive/`.\n",
        "| Entry | Wall min | MMLU s | HellaSwag s | KL s | WikiText s | Refusal s | Over-refusal s | Peak alloc/reserved GiB | Storage MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["stage_seconds"]
        lines.append(
            f"| {r['edit_id']} | {r['wall_minutes']:.3f} | {s['mmlu_pro']:.1f} | {s['hellaswag']:.1f} | {s['kl_to_base']:.1f} | {s['wikitext_ppl']:.1f} | {s['refusal_rate']:.1f} | {s['over_refusal']:.1f} | {r['peak_gpu_allocated_gib']:.2f}/{r['peak_gpu_reserved_gib']:.2f} | {r['storage_bytes'] / 2**20:.2f} |"
        )
    g = result["gate1"]
    lines += [
        "",
        f"**Warm median wall time over {result['n_warm']} edits (cache resident, loaded and verified once in the process): {result['warm_median_minutes']:.3f} min/entry** (range {result['warm_min_max_minutes'][0]:.3f} to {result['warm_min_max_minutes'][1]:.3f}). Warm stage medians (s): MMLU {stage['mmlu_pro']:.1f}, HellaSwag {stage['hellaswag']:.1f}, KL {stage['kl_to_base']:.1f}, WikiText {stage['wikitext_ppl']:.1f}, refusal {stage['refusal_rate']:.1f}, over-refusal {stage['over_refusal']:.1f}. **Gate 1 ({g['pass_line_minutes']:.1f} min pass line, {g['revise_line_minutes']:.1f} min revision line): {g['status']}.**",
        "",
        f"Cold cost: base entry {result['cold']['base_wall_minutes']:.2f} min wall including cold greedy-continuation cache construction ({result['cold']['base_cache_build_seconds']:.0f} s, mode `{result['cold']['cache_mode']}`); model load {result['model_load_seconds']:.1f} s once per process.",
    ]
    if "load_probe" in result:
        p = result["load_probe"]
        lines.append(
            f"New-process cache cost (separate process, entry `{p['entry']}`, label `load-probe`): cache {p['cache_mode']} plus content verification **{p['cache_load_verify_seconds']:.1f} s**, entry wall {p['wall_minutes']:.2f} min. Repeat vs primary: max |ΔKL| = {p['max_abs_kl_diff_vs_primary']:.2e}, max |Δgold log-prob| = {p['max_abs_gold_log_prob_diff_vs_primary']:.2e}, records identical: {p['records_identical']}."
        )
    lines += [
        "",
        f"Storage: median {result['median_entry_storage_bytes'] / 2**20:.2f} MiB per entry (records {result['median_records_bytes'] / 2**20:.2f} MiB, including per-position KL); shared base-logit cache {result['cache_bytes'] / 2**30:.2f} GiB on disk (manifest bytes {result['cache_manifest_bytes'] / 2**30:.2f} GiB, {len(rows[0]['kl_prefix'])} prefix lengths recorded), content root `{result['base_logits_sha256'][:16]}…` recorded in every KL record, datasets and eval.json. Peak GPU allocated {max(r['peak_gpu_allocated_gib'] for r in rows):.2f} GiB with the resident cache.",
        "",
        "Mean per-prompt KL(base‖edited) by continuation prefix length across the ten edits (descriptive; for choosing the token cap): "
        + ", ".join(
            f"{n} tokens: median entry value {np.median([r['kl_prefix'][n] for r in rows[1:]]):.5f}"
            for n in (16, 32, 64)
        )
        + ".",
        "",
        "Not run: G1 bitwise repeat on two entries, same-item G2 bridge, G3 second device, the §6 ten-entry composition (these are the ten D3 controls: 3 random-direction, 3 abliteration, 2 LoRA, 2 task-vector), and the reliability procedure. Endpoints remain unqualified.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--load-probe")
    parser.add_argument("--load-probe-id")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    result = report(args.load_probe, args.load_probe_id)
    text = markdown(result)
    if args.append:
        with (ROOT / "docs/RESULTS.md").open("a") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
