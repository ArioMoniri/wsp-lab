#!/usr/bin/env python3
"""PREREG_H1_TF section 6 qualification statistics for the TF-v1 endpoints (development panel).

Reads only existing artefacts under data/evals/qwen3.5-0.8b/tf_v1 (pilot entries, the section 6
composition + chain predecessors, their `repeats/g1` executions) and the sealed-partition manifest
(components). Launches nothing. Writes data/runs/h1a/qualification_report.json.

Per endpoint (MMLU-Pro predecessor-relative gold log-prob delta .10; KL delta .01; WikiText
log-PPL delta log 1.05) it reports: measured provenance, paired repeat error (max and 95th
percentile with a component-bootstrap upper bound), item-bootstrap half-widths with joint item draws
across edits and the shared reference, the delta/2 gates, and the signal-reliability lower bound.
All bootstraps use 4,000 draws and seed 20261105.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
TF = ROOT / "data/evals/qwen3.5-0.8b/tf_v1"
PANEL = ROOT / "data/runs/h1a/panel_manifest.json"
QUAL = ROOT / "data/runs/h1a/qualification_panel.json"
OUT = ROOT / "data/runs/h1a/qualification_report.json"
SEED, DRAWS = 20261105, 4000
LABEL = "g1"
KL_HIGH_FACTOR = 10  # entries with KL > 10 delta are outside the KL gate (PREREG amendment)

ENDPOINTS = {
    # name: (delta, repeat-discrepancy limit, record benchmark, item field, relative to)
    "mmlu_gold_log_prob": (0.10, 1e-4, "mmlu_pro"),
    "kl_to_base": (0.01, 1e-5, "kl_to_base"),
    "wikitext_log_ppl": (math.log(1.05), 1e-4, "wikitext_ppl"),
}


def read_records(directory):
    path = Path(directory) / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def item_values(records, endpoint):
    """Per-item/prompt/window values of the endpoint, in panel order."""
    if endpoint == "mmlu_gold_log_prob":
        return np.array(
            [r["extra"]["gold_log_prob"] for r in records if r["benchmark"] == "mmlu_pro"]
        )
    if endpoint == "kl_to_base":
        return np.array(
            [r["extra"]["full_sequence_kl"] for r in records if r["benchmark"] == "kl_to_base"]
        )
    return np.array(
        [r["extra"]["mean_nll"] for r in records if r["benchmark"] == "wikitext_ppl"]
    )  # equal-length windows: the mean is the token-weighted log-PPL


def item_ids(records, endpoint):
    name = ENDPOINTS[endpoint][2]
    return [r["item_id"] for r in records if r["benchmark"] == name]


def block_size(endpoint):
    return 2 if endpoint == "wikitext_log_ppl" else 1  # contiguous window blocks, as repeats.py


def draw_indices(n_items, endpoint, rng):
    """One joint item/block draw shared by every edit and reference (DRAWS x n_items)."""
    size = block_size(endpoint)
    n_blocks = n_items // size
    picks = rng.integers(n_blocks, size=(DRAWS, n_blocks))
    return (picks[:, :, None] * size + np.arange(size)).reshape(DRAWS, -1)


def provenance(directory):
    p = json.loads((Path(directory) / "eval.json").read_text())
    return p


def components_of(ids):
    manifest = json.loads(PANEL.read_text())
    where = {m: c["component_id"] for c in manifest["components"] for m in c["members"]}
    return [where[i] for i in ids]


def reliability(y_resid, error_var):
    """(signal variance, reliability) for one resample; untruncated."""
    total = float(np.var(y_resid, ddof=1))
    signal = total - float(np.mean(error_var))
    return signal, signal / total if total > 0 else float("nan")


def nuisance_residuals(y, families, log_norm):
    """OLS with intercept, treatment-coded family dummies (sorted, first level reference), log-norm."""
    levels = sorted(set(families))
    columns = [np.ones(len(y)), log_norm] + [
        np.array([f == lv for f in families], float) for lv in levels[1:]
    ]
    x = np.column_stack(columns)
    beta = np.linalg.pinv(x) @ y
    return y - x @ beta, int(np.linalg.matrix_rank(x))


def analyse(endpoint, entries, refs, families, log_norm, clusters, composition, rng):
    """entries: id -> {'primary': records, 'repeat': records|None}; refs: id -> reference key."""
    delta, repeat_limit, _ = ENDPOINTS[endpoint]
    ids = list(entries)
    n_items = len(item_values(entries[ids[0]]["primary"], endpoint))
    index = draw_indices(n_items, endpoint, rng)
    base_kl = endpoint == "kl_to_base"

    def diff(side, edit_id):
        edit = item_values(entries[edit_id][side], endpoint)
        if base_kl:
            return edit  # KL is base-to-edited; the base reference is exactly zero
        reference = item_values(refs[edit_id][side], endpoint)
        return edit - reference

    primary = {i: diff("primary", i) for i in ids}
    y = np.array([primary[i].mean() for i in ids])
    boot = {i: primary[i][index].mean(1) for i in ids}
    half = np.array([(np.quantile(boot[i], 0.975) - np.quantile(boot[i], 0.025)) / 2 for i in ids])
    se = np.array([boot[i].std(ddof=1) for i in ids])
    result = {
        "delta": delta,
        "n_entries": len(ids),
        "n_items": n_items,
        "y": dict(zip(ids, y.tolist())),
        "item_se": dict(zip(ids, se.tolist())),
        "item_halfwidth95": dict(zip(ids, half.tolist())),
    }
    in_gate = np.ones(len(ids), bool)
    if base_kl:
        in_gate = y <= KL_HIGH_FACTOR * delta
        result["kl_gate_entries"] = int(in_gate.sum())
        result["kl_excluded_high_kl"] = [i for i, g in zip(ids, in_gate) if not g]
    # delta/2 gates on item uncertainty: per entry, and the panel mean with a component-bootstrap bound.
    comp = np.array(clusters)
    uniq = sorted(set(clusters))
    pick = np.random.default_rng(SEED).integers(len(uniq), size=(DRAWS, len(uniq)))
    masks = {c: np.flatnonzero(comp == c) for c in uniq}

    def component_bootstrap(stat, mask=None):
        values = []
        for row in pick:
            members = np.concatenate([masks[uniq[k]] for k in row])
            if mask is not None:
                members = members[mask[members]]
            values.append(stat(members) if len(members) else np.nan)
        return np.array(values)

    mean_half = component_bootstrap(lambda m: half[m].mean(), in_gate)
    result["itemwidth_gate"] = {
        "limit_delta_over_2": delta / 2,
        "max_halfwidth_over_gated_entries": float(half[in_gate].max()),
        "median_halfwidth_over_gated_entries": float(np.median(half[in_gate])),
        "panel_mean_halfwidth": float(half[in_gate].mean()),
        "panel_mean_halfwidth_upper97_5": float(np.nanquantile(mean_half, 0.975)),
        "entries_over_limit": [i for i, h, g in zip(ids, half, in_gate) if g and h > delta / 2],
        "pass_every_gated_entry": bool((half[in_gate] <= delta / 2).all()),
        "pass_panel_mean_upper_bound": bool(np.nanquantile(mean_half, 0.975) <= delta / 2),
    }
    # Paired repeat error, only for entries with a g1 repeat.
    rep_ids = [i for i in ids if entries[i]["repeat"] is not None]
    if rep_ids:
        diffs = {i: diff("repeat", i) - primary[i] for i in rep_ids}
        flat = np.concatenate([np.abs(diffs[i]) for i in rep_ids])
        entry_max = np.array([np.abs(diffs[i]).max() for i in rep_ids])
        entry_y = np.array([diffs[i].mean() for i in rep_ids])
        rep_comp = np.array([clusters[ids.index(i)] for i in rep_ids])
        rep_uniq = sorted(set(rep_comp))
        rep_pick = np.random.default_rng(SEED + 1).integers(
            len(rep_uniq), size=(DRAWS, len(rep_uniq))
        )
        rep_masks = {c: np.flatnonzero(rep_comp == c) for c in rep_uniq}
        p95 = []
        for row in rep_pick:
            members = np.concatenate([rep_masks[rep_uniq[k]] for k in row])
            p95.append(
                np.quantile(np.concatenate([np.abs(diffs[rep_ids[m]]) for m in members]), 0.95)
            )
        result["repeat"] = {
            "entries": rep_ids,
            "n_values": int(flat.size),
            "max_abs_discrepancy": float(flat.max()),
            "max_abs_entry_mean_discrepancy": float(np.abs(entry_y).max()),
            "bitwise_identical_items": bool((flat == 0).all()),
            "p95_abs_discrepancy": float(np.quantile(flat, 0.95)),
            "p95_upper97_5_component_bootstrap": float(np.quantile(p95, 0.975)),
            "limit_max": repeat_limit,
            "limit_p95_upper": delta / 2,
            "pass_max": bool(flat.max() <= repeat_limit),
            "pass_p95_upper": bool(np.quantile(p95, 0.975) <= delta / 2),
            "per_entry_max": dict(zip(rep_ids, entry_max.tolist())),
        }
    # Signal reliability on family/log-norm residual effects, error = item + repeat variance.
    if len(ids) >= 4:
        resid, rank = nuisance_residuals(y, families, log_norm)
        result["family_norm_residual"] = dict(zip(ids, resid.tolist()))
        repeat_var = np.zeros(len(ids))
        if rep_ids:
            for i in rep_ids:
                repeat_var[ids.index(i)] = float((diff("repeat", i) - primary[i]).mean() ** 2) / 2
        error_var = se**2 + repeat_var
        sel = in_gate if base_kl else np.ones(len(ids), bool)
        signal, rel = reliability(resid[sel], error_var[sel])
        draws = []
        for row in pick:
            members = np.concatenate([masks[uniq[k]] for k in row])
            members = members[sel[members]]
            if len(members) > 2:
                draws.append(reliability(resid[members], error_var[members])[1])
        draws = np.array(draws)
        result["reliability"] = {
            "n_entries": int(sel.sum()),
            "design_rank": rank,
            "residual_variance": float(np.var(resid[sel], ddof=1)),
            "mean_item_error_variance": float((se[sel] ** 2).mean()),
            "mean_repeat_error_variance": float(repeat_var[sel].mean()),
            "signal_variance_untruncated": signal,
            "reliability": rel,
            "reliability_ci95": [
                float(np.nanquantile(draws, 0.025)),
                float(np.nanquantile(draws, 0.975)),
            ],
            "valid_draws": int(np.isfinite(draws).sum()),
            "lower_bound_threshold": 0.90,
            "pass_lower_bound": bool(np.nanquantile(draws, 0.025) >= 0.90 and signal > 0),
        }
    return result


def load_entries(ids, with_repeat):
    entries, meta = {}, {}
    for i in ids:
        d = TF / i
        entries[i] = {"primary": read_records(d), "repeat": None}
        r = d / "repeats" / LABEL
        if with_repeat and (r / "records.jsonl").exists():
            entries[i]["repeat"] = read_records(r)
        meta[i] = provenance(d)
    return entries, meta


def provenance_checks(meta, base_meta):
    rows = {}
    for i, p in meta.items():
        rows[i] = {
            "native_measured": p.get("metadata_provenance") == "native-measured",
            "device": p["eval_device_id"],
            "clean_tree": p["dirty"] is False,
            "source_hash": p["source_hash"][:12],
            "same_source_as_base": p["source_hash"] == base_meta["source_hash"],
            "matched_base": p.get("matched_base_missing") is False,
            "complete": p["eval_status"] == "complete",
            "batch_size": p["batch_size"],
            "run_id": p["run_id"],
        }
    return rows


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    qual = json.loads(QUAL.read_text())
    controls = (ROOT / "data/runs/d3_shards/controls.txt").read_text().split()
    manifest = {}
    for line in (ZOO / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "id" in row:
            manifest[row["id"]] = row
    edit = lambda i: json.loads((ZOO / i / "edit.json").read_text())
    composition = qual["entries"]
    predecessors = dict(zip(qual["composition"]["chain"], qual["chain_predecessors"], strict=True))
    base_entries, base_meta = load_entries(["base"], True)
    base = base_entries["base"]
    panel_ids = composition + controls  # y-bearing entries: ten composition + ten pilot controls
    entries, meta = load_entries(panel_ids, True)
    pred_entries, pred_meta = load_entries(list(predecessors.values()), True)
    refs = {}
    for i in panel_ids:
        source = pred_entries[predecessors[i]] if i in predecessors else base
        refs[i] = source
    families = [manifest[i]["family"] for i in panel_ids]
    log_norm = np.log1p([edit(i)["delta_fro_norm"] for i in panel_ids])
    clusters = components_of(panel_ids)

    def run(ids, name):
        sel = [panel_ids.index(i) for i in ids]
        sub = {i: entries[i] for i in ids}
        return analyse(
            name,
            sub,
            {i: refs[i] for i in ids},
            [families[k] for k in sel],
            log_norm[sel],
            [clusters[k] for k in sel],
            ids,
            np.random.default_rng(SEED),
        )

    report = {
        "seed": SEED,
        "draws": DRAWS,
        "composition": qual["composition"],
        "chain_predecessors": predecessors,
        "panel_ids": panel_ids,
        "provenance": {
            "base": provenance_checks({"base": base_meta["base"]}, base_meta["base"])["base"],
            "entries": provenance_checks(meta, base_meta["base"]),
            "predecessors": provenance_checks(pred_meta, base_meta["base"]),
        },
        "panels": {},
    }
    for label, ids in (("composition10", composition), ("expanded20", panel_ids)):
        report["panels"][label] = {name: run(ids, name) for name in ENDPOINTS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
