"""Paired reliability summaries; base and edited panels are analysed separately."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REFERENCE_KEYS = (
    "eval_device_id",
    "gpu_class",
    "engine",
    "torch_version",
    "cuda_version",
    "transformers_version",
    "dtype",
    "attention_implementation",
    "batch_size",
    "item_order_hash",
    "item_id_hash",
    "generation",
    "model_revision",
    "source_hash",
    "config_hash",
    "seed",
)


def known(value):
    return value is not None and value != "" and value != "unknown" and value != {} and value != []


def matched_reference(observation, candidates):
    """Require known, equal device/runtime/config/items/actual batch, including seed.

    Partial-inferred backfills are eligible only with identical inferred values:
    every inferred runtime field is a REFERENCE_KEY and is compared exactly below.
    Callers producing predictor manifests must disclose inferred target counts.
    """
    if any(not known(observation.get(k)) for k in REFERENCE_KEYS):
        return None
    return next(
        (
            c
            for c in candidates
            if all(known(c.get(k)) and c[k] == observation[k] for k in REFERENCE_KEYS)
        ),
        None,
    )


def rank_icc(x):
    n = len(x)
    if n < 2 or any(np.ptp(x[:, j]) == 0 for j in range(2)):
        return [float("nan")] * 2
    ranks = np.array([rankdata(x[:, j]) for j in range(2)])
    rho = np.corrcoef(ranks)[0, 1]
    rows, cols, grand = x.mean(1), x.mean(0), x.mean()
    msr = 2 * ((rows - grand) ** 2).sum() / (n - 1)
    msc = n * ((cols - grand) ** 2).sum()
    mse = ((x - rows[:, None] - cols + grand) ** 2).sum() / (n - 1)
    denominator = msr + mse + 2 * (msc - mse) / n
    return [float(rho), float((msr - mse) / denominator) if denominator else float("nan")]


def summary(x, bootstrap, seed, groups=None):
    d = x[:, 1] - x[:, 0]
    n = len(d)
    groups = np.arange(n) if groups is None else np.array(groups)
    clusters = list(dict.fromkeys(groups))
    draws = np.random.default_rng(seed).integers(len(clusters), size=(bootstrap, len(clusters)))
    indices = [
        np.concatenate([np.flatnonzero(groups == clusters[j]) for j in draw]) for draw in draws
    ]
    ranks = np.array([rank_icc(x[i]) for i in indices])
    intervals = []
    for j in range(2):
        finite = ranks[:, j][np.isfinite(ranks[:, j])]
        intervals.append(
            np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None]
        )
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    rho, icc = rank_icc(x)
    centered = d - d.mean()
    tail_draws = [np.quantile(abs(centered[i]), 0.95) for i in indices]
    return {
        "n": n,
        "centered_difference_rmse": float(np.sqrt(np.mean(centered**2))),
        "centered_absolute_p95": float(np.quantile(abs(centered), 0.95)),
        "centered_absolute_p95_ci95": np.quantile(tail_draws, [0.025, 0.975]).tolist(),
        "tail_estimation": "weak for small edit panels; also inspect maximum absolute difference",
        "paired_difference": float(d.mean()),
        "mean_absolute_difference": float(abs(d).mean()),
        "rmse": float(np.sqrt((d * d).mean())),
        "median_absolute_difference": float(np.median(abs(d))),
        "max_absolute_difference": float(abs(d).max()),
        "sd_difference": sd,
        "bland_altman_limits": [float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)],
        "bootstrap_ci95": np.quantile([d[i].mean() for i in indices], [0.025, 0.975]).tolist(),
        "spearman": rho,
        "icc_2_1": icc,
        "spearman_ci95": intervals[0],
        "icc_2_1_ci95": intervals[1],
        "rank_bootstrap_valid": np.isfinite(ranks).sum(0).tolist(),
    }


def wilson(p, n):
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - half), float(center + half)]


def item_records(payload):
    path = payload.get("path")
    if not path:
        return None
    path = Path(path)
    path = path.parent if path.is_file() else path
    file = path / "records.jsonl"
    return (
        [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        if file.exists()
        else None
    )


def paired_draws(edits, bootstrap, seed, block_size=1, references=None, groups=None):
    """Hierarchical paired bootstrap: edits, then paired items/contiguous blocks.

    The same reference array object denotes a shared base: draw it once per
    replicate, preserving covariance between all deltas that use that base.
    Blocks are nonoverlapping contiguous windows in execution order.
    """
    rng = np.random.default_rng(seed)
    groups = list(range(len(edits))) if groups is None else groups
    clusters = list(dict.fromkeys(groups))

    def sample(values):
        blocks = [values[i : i + block_size] for i in range(0, len(values), block_size)]
        selected = rng.integers(len(blocks), size=len(blocks))
        return float(np.concatenate([blocks[i] for i in selected]).mean())

    draws = []
    for _ in range(bootstrap):
        reference_draws = {}
        if references is not None:
            for ref in references:
                if id(ref) not in reference_draws:
                    reference_draws[id(ref)] = sample(ref)
        selected = rng.integers(len(clusters), size=len(clusters))
        means = []
        for group in selected:
            members = [i for i, g in enumerate(groups) if g == clusters[group]]
            means.append(
                np.mean(
                    [
                        sample(edits[i])
                        - (reference_draws[id(references[i])] if references is not None else 0)
                        for i in members
                    ]
                )
            )
        draws.append(np.mean(means))
    return np.array(draws)


def paired_items(a, b, benchmark):
    rows = [item_records(p) for p in (a, b)]
    if any(r is None for r in rows):
        return None
    rows = [[r for r in side if r["benchmark"] == benchmark] for side in rows]
    if not rows[0] or [r["item_id"] for r in rows[0]] != [r["item_id"] for r in rows[1]]:
        return None
    x = np.array([[r["correct"] for r in side] for side in rows], dtype=float)
    return np.log(x) if benchmark == "wikitext_ppl" else x


def item_analysis(pairs, benchmark, binary, bootstrap, seed, references=None):
    arrays, ref_arrays, groups = [], [], []
    cache = {}
    for i, (a, b) in enumerate(pairs):
        x = paired_items(a, b, benchmark)
        if x is None:
            continue
        if references is not None:
            ra, rb = references[i]
            key = tuple(
                (
                    r.get("run_id"),
                    r.get("path"),
                    r.get("edit_id"),
                    id(r) if not r.get("run_id") and not r.get("path") else None,
                )
                for r in (ra, rb)
            )
            if key not in cache:
                ref = paired_items(ra, rb, benchmark)
                cache[key] = None if ref is None else ref[1] - ref[0]
            if cache[key] is None:
                continue
            ref_arrays.append(cache[key])
        arrays.append(x)
        groups.append(a["edit_id"])
    out = {"available_pairs": len(arrays), "missing_pairs": len(pairs) - len(arrays)}
    if not arrays:
        return out
    differences = [x[1] - x[0] for x in arrays]
    block_size = 2 if benchmark == "wikitext_ppl" else 1
    draws = paired_draws(
        differences,
        bootstrap,
        seed,
        block_size,
        ref_arrays if references is not None else None,
        groups,
    )
    out.update(
        n_items=sum(x.shape[1] for x in arrays),
        paired_item_se=float(draws.std(ddof=1)) if bootstrap > 1 else None,
        paired_item_ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        resampling_unit="edit cluster then paired items"
        if block_size == 1
        else "edit cluster then contiguous nonoverlapping window blocks",
        window_block_size=block_size,
        shared_reference_resampled=references is not None,
    )
    levels = np.concatenate(arrays, axis=1)
    out["level_mean"] = levels.mean(axis=1).tolist()
    out["level_median"] = np.median(levels, axis=1).tolist()
    out["level_quantiles"] = np.quantile(levels, [0, 0.25, 0.5, 0.75, 0.95, 1], axis=1).T.tolist()
    if benchmark == "wikitext_ppl":
        out["ppl_ratio_ci95"] = np.exp(out["paired_item_ci95"]).tolist()
        out["population_inference"] = (
            "window-block diagnostic; short panels give weak population inference"
        )
    if binary:
        out["n01"] = int(((levels[0] == 0) & (levels[1] == 1)).sum())
        out["n10"] = int(((levels[0] == 1) & (levels[1] == 0)).sum())
        out["disagreement_count"] = out["n01"] + out["n10"]
    return out


def compare_pairs(
    primary,
    repeat,
    bootstrap=10000,
    seed=0,
    primary_references=(),
    repeat_references=(),
    allow_batch_mismatch=False,
):
    """Edit-cluster bootstrap; no base in ranks; optional matched-reference delta panels.

    Input paths locate records.jsonl automatically. Continuous KL uses prompt scores;
    WikiText uses log-ppl and window blocks, never a binomial approximation.
    """

    def read(x):
        if isinstance(x, (str, Path)):
            return json.loads(Path(x).read_text()) | {"path": str(Path(x).parent)}
        return x

    left, right = list(map(read, primary)), list(map(read, repeat))
    if len(left) != len(right) or not left or bootstrap < 1:
        raise ValueError("panels must be nonempty and aligned; bootstrap must be positive")
    for a, b in zip(left, right):
        for key in ("edit_id", *REFERENCE_KEYS):
            if key == "eval_device_id":
                continue  # Device identity may differ for host reliability panels.
            if key == "batch_size" and allow_batch_mismatch:
                continue
            if not known(a.get(key)) or not known(b.get(key)) or a[key] != b[key]:
                raise ValueError(f"known matching {key} required")
        if a.get("seed") != b.get("seed"):
            raise ValueError("matching seed required")
    refs = [list(map(read, primary_references)), list(map(read, repeat_references))]
    result = {}
    benchmarks = set().union(*(p.get("counts", {}) for p in left))
    benchmarks |= {
        k for k in ("kl_first_token_mean", "kl_full_sequence_mean") if any(k in p for p in left)
    }
    for benchmark in sorted(benchmarks):
        pairs = [(a, b) for a, b in zip(left, right) if benchmark in a and benchmark in b]
        if not pairs:
            continue
        binary = not (
            benchmark.startswith("kl_") or benchmark in ("wikitext_ppl", "log_ppl", "mean_nll")
        )

        def value(p, benchmark=benchmark):
            return np.log(p[benchmark]) if benchmark == "wikitext_ppl" else p[benchmark]

        edits = [(a, b) for a, b in pairs if a["edit_id"] != "base"]
        x = np.array([[value(a), value(b)] for a, b in edits])
        report = (
            summary(x, bootstrap, seed, [a["edit_id"] for a, _ in edits]) if edits else {"n": 0}
        )
        report["base"] = [[value(a), value(b)] for a, b in pairs if a["edit_id"] == "base"]
        report["scale"] = "log-ppl" if benchmark == "wikitext_ppl" else "original"
        report["binomial_se_method"] = (
            "sqrt(p(1-p)/n); Wilson half-width/1.96 at boundaries" if binary else None
        )
        report["binomial_se"], report["wilson_ci95"] = [], []
        for pair in pairs:
            intervals, ses = [], []
            for p in pair:
                count = p.get("counts", {}).get(benchmark)
                ci = wilson(p[benchmark], count) if binary and count else None
                intervals.append(ci)
                ses.append(
                    float(np.sqrt(p[benchmark] * (1 - p[benchmark]) / count))
                    if ci and 0 < p[benchmark] < 1
                    else (ci[1] - ci[0]) / 3.92
                    if ci
                    else None
                )
            report["binomial_se"].append(ses)
            report["wilson_ci95"].append(intervals)
        report["batch_size_mismatch"] = any(
            a.get("batch_size") != b.get("batch_size") for a, b in pairs
        )
        report["item_discordance"] = item_analysis(edits, benchmark, binary, bootstrap, seed)
        deltas, delta_groups, delta_pairs, delta_refs = [], [], [], []
        for a, b in edits:
            matches = [
                matched_reference(
                    p,
                    [
                        r
                        for r in candidates
                        if r.get("edit_id") == p.get("reference_edit_id", "base")
                    ],
                )
                for p, candidates in zip((a, b), refs)
            ]
            if all(r is not None and benchmark in r for r in matches):
                deltas.append([value(p) - value(r) for p, r in zip((a, b), matches)])
                delta_groups.append(a["edit_id"])
                delta_pairs.append((a, b))
                delta_refs.append(matches)
        report["matched_reference_deltas"] = (
            summary(np.array(deltas), bootstrap, seed, delta_groups) if deltas else {"n": 0}
        )
        report["matched_reference_deltas"]["item_uncertainty"] = item_analysis(
            delta_pairs, benchmark, binary, bootstrap, seed, delta_refs
        )
        if benchmark == "wikitext_ppl":
            for panel in (report, report["matched_reference_deltas"]):
                if "bootstrap_ci95" in panel:
                    panel["ppl_ratio_ci95"] = np.exp(panel["bootstrap_ci95"]).tolist()
        report["qualification_analysis"] = "blocked: preregistered host-by-edit analysis pending"
        report["missing_matched_reference"] = len(edits) - len(deltas)
        result[benchmark] = report
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", nargs="+", required=True, type=Path)
    parser.add_argument("--repeat", nargs="+", required=True, type=Path)
    parser.add_argument("--primary-references", nargs="*", default=[], type=Path)
    parser.add_argument("--repeat-references", nargs="*", default=[], type=Path)
    parser.add_argument("--allow-batch-mismatch", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(compare_pairs(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
