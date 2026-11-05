#!/usr/bin/env python3
"""Report measured primary TF-v1 pilot artifacts; never launch evaluations."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def report():
    zoo = ROOT / "data/evals/qwen3.5-0.8b"
    ids = ["base", *(ROOT / "data/runs/d3_shards/controls.txt").read_text().splitlines()]
    measured_source = json.loads((zoo / "tf_v1/base/eval.json").read_text())["source_hash"]
    rows = []
    for edit_id in ids:
        directory = zoo / "tf_v1" / edit_id
        payload = json.loads((directory / "eval.json").read_text())
        legacy = json.loads((zoo / edit_id / "eval.json").read_text())
        assert payload["eval_status"] == "complete" and payload["repeat_id"] is None
        assert payload["source_hash"] == measured_source
        assert not payload["matched_base_missing"]
        assert payload["counts"]["mmlu_pro"] == 1000 and legacy["counts"]["mmlu_pro"] == 100
        from wsp.eval.tf_v1 import summarize
        from wsp.schema import EvalRecord

        records = [
            EvalRecord(**json.loads(line))
            for line in (directory / "records.jsonl").read_text().splitlines()
        ]
        assert len(records) == 1864
        assert summarize(records) == payload["tf_v1"]
        assert len({(r.benchmark, r.item_id) for r in records}) == len(records)
        tf = payload["tf_v1"]["mmlu_pro"]
        cache_hash = next(r for r in records if r.benchmark == "kl_to_base").extra["cache_hash"]
        rows.append(
            {
                "edit_id": edit_id,
                "run_id": payload["run_id"],
                "wall_minutes": payload["wall_seconds"] / 60,
                "model_load_seconds": payload["model_load_seconds"],
                "peak_gpu_allocated_gib": payload["peak_gpu_allocated_bytes"] / 2**30,
                "peak_gpu_reserved_gib": payload["peak_gpu_reserved_bytes"] / 2**30,
                "storage_bytes": sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
                "records_bytes": (directory / "records.jsonl").stat().st_size,
                "accuracy_1000": tf["accuracy"],
                "accuracy_100": legacy["mmlu_pro"],
                "gold_log_prob": tf["mean_gold_log_prob"],
                "legacy_run_id": legacy["run_id"],
                "stage_seconds": payload["stage_seconds"],
                "kl_cache_hash": cache_hash,
            }
        )
    # Cold entry = first completed entry per base-cache hash (it builds the cache).
    seen = set()
    for r in sorted(
        rows, key=lambda r: (zoo / "tf_v1" / r["edit_id"] / "eval.json").stat().st_mtime
    ):
        r["cache_rebuilt"] = r["kl_cache_hash"] not in seen
        seen.add(r["kl_cache_hash"])
    warm_rows = [r for r in rows[1:] if not r["cache_rebuilt"]]
    result = {
        "rows": rows,
        "warm_median_minutes": float(np.median([r["wall_minutes"] for r in warm_rows])),
        "warm_median_minutes_all_10": float(np.median([r["wall_minutes"] for r in rows[1:]])),
        "n_warm": len(warm_rows),
        "cold_entries": [r["edit_id"] for r in rows if r["cache_rebuilt"]],
        "median_abs_accuracy_delta": float(
            np.median([abs(r["accuracy_1000"] - r["accuracy_100"]) for r in rows])
        ),
    }
    for label, panel in [("all_11", rows), ("edits_10", rows[1:])]:
        x = [r["gold_log_prob"] for r in panel]
        y = [r["accuracy_1000"] for r in panel]
        result[label] = {
            "gold_log_prob_vs_accuracy_pearson": float(pearsonr(x, y).statistic),
            "gold_log_prob_vs_accuracy_spearman": float(spearmanr(x, y).statistic),
        }
    caches = {h: zoo / "base/tf_cache/H0" / h for h in {r["kl_cache_hash"] for r in rows}}
    sizes = {
        h: sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) for h, d in caches.items()
    }
    result["shared_cache_bytes"] = sizes[rows[0]["kl_cache_hash"]]
    result["duplicate_cache_bytes"] = sum(sizes.values()) - result["shared_cache_bytes"]
    result["median_entry_storage_bytes"] = float(np.median([r["storage_bytes"] for r in rows[1:]]))
    out = zoo / "tf_v1/pilot_report.json"
    out.write_text(json.dumps(result, indent=2))
    from wsp.schema import write_manifest

    write_manifest(
        out.parent / "manifest.json",
        **{
            k: payload[k]
            for k in ("seed", "config_hash", "git_sha", "source_hash", "model_revision", "run_id")
        },
        entries=ids,
        report=out.name,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].scatter([r["gold_log_prob"] for r in rows], [100 * r["accuracy_1000"] for r in rows])
    axes[0].set(xlabel="Mean normalized gold-letter log-prob (nats)", ylabel="TF-v1 accuracy (%)")
    axes[1].scatter(
        [100 * r["accuracy_100"] for r in rows], [100 * r["accuracy_1000"] for r in rows]
    )
    axes[1].set(xlabel="Legacy accuracy, 100 items (%)", ylabel="TF-v1 accuracy, 1000 items (%)")
    fig.suptitle(f"TF-v1 timing pilot: {rows[0]['run_id']} (11 records; descriptive)")
    fig.savefig(out.parent / "bridge.png", dpi=160)
    plt.close(fig)

    lines = [
        "\n## TF-v1 timing pilot\n",
        f"Measured run `{rows[0]['run_id']}` on H0: base + the ten fixed IDs in `data/runs/d3_shards/controls.txt`; primary records (`repeat_id=null`), batch 1, no sampling. Native runtime/source/item metadata and same-device base matching are in each eval.json. Source may be dirty; source hashes identify the measured code.\n",
        "| Entry | Wall min | Load s (shared once) | Peak allocated/reserved GiB | TF acc (1000) | Fast acc (100) | Gold log-prob | Records MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['edit_id']} | {r['wall_minutes']:.3f} | {r['model_load_seconds']:.2f} | {r['peak_gpu_allocated_gib']:.3f}/{r['peak_gpu_reserved_gib']:.3f} | {r['accuracy_1000']:.3f} | {r['accuracy_100']:.3f} | {r['gold_log_prob']:.6f} | {r['records_bytes'] / 2**20:.3f} |"
        )
    warm = result["warm_median_minutes"]
    rebuilt = [r["edit_id"] for r in rows[1:] if r["cache_rebuilt"]]
    lines.extend(
        [
            "",
            f"Median warm entry storage (records + eval/summary/manifest): {result['median_entry_storage_bytes'] / 2**20:.3f} MiB; shared base logits cache: {result['shared_cache_bytes'] / 2**30:.3f} GiB (not duplicated per entry). The pilot also left an orphan byte-identical duplicate cache of {result['duplicate_cache_bytes'] / 2**30:.3f} GiB (see caveat).",
            "",
            f"Warm median wall time over {result['n_warm']} edits that reused a cache: **{warm:.3f} min/entry** (all ten edits: {result['warm_median_minutes_all_10']:.3f}); entry `{', '.join(rebuilt)}` hit a cache-key defect (git sha/dirty flag in the key; a WIP commit during the run changed them) that rebuilt a byte-identical base cache (KL stage ~10 min) and is excluded from the warm median; the following three edits reused that duplicate. Pilot code `source_hash` {measured_source[:12]}; the key is fixed afterwards in a later commit.  Model load: **{rows[0]['model_load_seconds']:.2f} s** once per batch invocation (not charged eleven times). Base includes cold greedy-continuation cache construction. The council warm timing pass criterion is ≤2.0 minutes; >3.0 minutes calls for revision and re-pilot. This fixed-count pilot does not silently reduce the requested item counts. Per-stage times are in `data/evals/qwen3.5-0.8b/tf_v1/pilot_report.json`; wall includes weight restore/application but excludes shared model load and result serialization.",
            "",
            f"Legacy bridge (11 paired records): median |accuracy(1000 new) − accuracy(100 legacy)| = **{result['median_abs_accuracy_delta']:.4f}** ({100 * result['median_abs_accuracy_delta']:.2f} percentage points). Legacy run `{rows[0]['legacy_run_id']}`. This compares different item sets; it is not a same-item numerical-parity or qualification test.",
            "",
            f"Gold log-prob versus TF accuracy across all 11 entries: Pearson r={result['all_11']['gold_log_prob_vs_accuracy_pearson']:.4f}, Spearman rho={result['all_11']['gold_log_prob_vs_accuracy_spearman']:.4f}; excluding base (10 edits): r={result['edits_10']['gold_log_prob_vs_accuracy_pearson']:.4f}, rho={result['edits_10']['gold_log_prob_vs_accuracy_spearman']:.4f}. Descriptive point estimates only: related controls are not independent components; no confirmatory CI or readability claim.",
            "",
            "TF-v1 stores normalized MMLU letter vectors, per-character HellaSwag scores, paired item IDs, KL continuation IDs/cache hashes, window NLL/token counts, and refusal/compliance token sets and margins. The lexical first-token sets are frozen in the config, exploratory and not calibrated on a separate prompt panel. Refusal uses the requested 50+50 prompts, rather than the council proposal’s 200+200. This control panel contains 3 random-direction, 3 abliteration, 2 LoRA and 2 task-vector edits, with no chains; it replaces the council proposal’s stratified panel at the user’s explicit request. No gold-solution NLL, independent repeat, cross-device execution, fresh component qualification or same-item legacy bridge was requested or performed. Endpoints remain unqualified under Board A’s prospective rule.",
        ]
    )
    with (ROOT / "docs/RESULTS.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    report()
