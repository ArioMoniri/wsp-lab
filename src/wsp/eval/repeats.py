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
)


def known(value):
    return value is not None and value != "" and value != "unknown" and value != {} and value != []


def matched_reference(observation, candidates):
    """Require known, equal device/runtime/config/items/actual batch, including seed."""
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


def item_analysis(pairs, benchmark, binary, bootstrap, seed):
    diffs, levels = [], [[], []]
    n01 = n10 = n = available = 0
    for a, b in pairs:
        records = [item_records(p) for p in (a, b)]
        if any(r is None for r in records):
            continue
        maps = [
            {r["item_id"]: r["correct"] for r in rows if r["benchmark"] == benchmark}
            for rows in records
        ]
        if not maps[0] or maps[0].keys() != maps[1].keys():
            continue
        available += 1
        for key in maps[0]:
            u, v = maps[0][key], maps[1][key]
            if benchmark == "wikitext_ppl":
                u, v = np.log(u), np.log(v)
            levels[0].append(u)
            levels[1].append(v)
            diffs.append(v - u)
            n += 1
            n01 += int(u == 0 and v == 1)
            n10 += int(u == 1 and v == 0)
    out = {"available_pairs": available, "missing_pairs": len(pairs) - available}
    if n:
        d = np.array(diffs)
        idx = np.random.default_rng(seed).integers(n, size=(bootstrap, n))
        out.update(
            n_items=n,
            paired_item_se=float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            paired_item_ci95=np.quantile(d[idx].mean(1), [0.025, 0.975]).tolist(),
            resampling_unit="paired item (WikiText window block)",
            level_quantiles=[
                np.quantile(v, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist() for v in levels
            ],
        )
        out["level_mean"] = [float(np.mean(v)) for v in levels]
        out["level_median"] = [float(np.median(v)) for v in levels]
        out["outlier_count_iqr"] = []
        for v in levels:
            q1, q3 = np.quantile(v, [0.25, 0.75])
            out["outlier_count_iqr"].append(
                int(
                    np.sum(
                        (np.array(v) < q1 - 1.5 * (q3 - q1)) | (np.array(v) > q3 + 1.5 * (q3 - q1))
                    )
                )
            )
        if benchmark == "wikitext_ppl":
            out["population_inference"] = (
                "window-block diagnostic; adjacent windows give weak population inference"
            )
        if binary:
            out.update(n01=n01, n10=n10, disagreement_count=n01 + n10)
    return out


def compare_pairs(
    primary, repeat, bootstrap=10000, seed=0, primary_references=(), repeat_references=()
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
        for key in ("edit_id", "item_id_hash", "item_order_hash"):
            if not known(a.get(key)) or not known(b.get(key)) or a[key] != b[key]:
                raise ValueError(f"known matching {key} required")
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
        report["item_discordance"] = item_analysis(pairs, benchmark, binary, bootstrap, seed)
        deltas, delta_groups = [], []
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
        report["matched_reference_deltas"] = (
            summary(np.array(deltas), bootstrap, seed, delta_groups) if deltas else {"n": 0}
        )
        report["missing_matched_reference"] = len(edits) - len(deltas)
        result[benchmark] = report
    return result
