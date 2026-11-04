"""Paired reliability summaries for aligned entry panels (base analysed separately)."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def compare_pairs(primary, repeat, bootstrap=10000, seed=0):
    """Compare aligned payloads or eval.json paths; never infer missing legacy metadata.

    Returns per-benchmark levels, signed repeat-primary bootstrap CI, MAD,
    Spearman, absolute-agreement ICC(2,1), item-count binomial SE and batch flags.
    """

    def read(x):
        return json.loads(Path(x).read_text()) if isinstance(x, (str, Path)) else x

    left, right = list(map(read, primary)), list(map(read, repeat))
    if len(left) != len(right) or not left:
        raise ValueError("panels must be nonempty and aligned")
    for a, b in zip(left, right):
        if a.get("edit_id") != b.get("edit_id"):
            raise ValueError("entry IDs must match")
        if a.get("item_id_hash") != b.get("item_id_hash"):
            raise ValueError("item sets must match")
    result = {}
    for benchmark in left[0].get("counts", {}):
        pairs = [(a, b) for a, b in zip(left, right) if benchmark in a and benchmark in b]
        if not pairs:
            continue
        x = np.array([[a[benchmark], b[benchmark]] for a, b in pairs])
        d = x[:, 1] - x[:, 0]
        n = len(x)
        ci = np.quantile(
            np.random.default_rng(seed).choice(d, (bootstrap, n)).mean(1), [0.025, 0.975]
        )
        icc = float("nan")
        variable = n > 1 and all(np.ptp(x[:, j]) > 0 for j in range(2))
        if variable:
            rows, cols, grand = x.mean(1), x.mean(0), x.mean()
            msr = 2 * ((rows - grand) ** 2).sum() / (n - 1)
            msc = n * ((cols - grand) ** 2).sum()
            mse = ((x - rows[:, None] - cols + grand) ** 2).sum() / (n - 1)
            denominator = msr + mse + 2 * (msc - mse) / n
            icc = float((msr - mse) / denominator) if denominator else float("nan")
        binary = benchmark not in ("kl_to_base", "wikitext_ppl")
        result[benchmark] = {
            "n": n,
            "mean_absolute_difference": float(abs(d).mean()),
            "paired_difference": float(d.mean()),
            "bootstrap_ci95": ci.tolist(),
            "spearman": float(spearmanr(x[:, 0], x[:, 1]).statistic) if variable else float("nan"),
            "icc_2_1": icc,
            "binomial_se": [
                [
                    float(np.sqrt(p[benchmark] * (1 - p[benchmark]) / p["counts"][benchmark]))
                    if binary
                    else None
                    for p in pair
                ]
                for pair in pairs
            ],
            "batch_size_mismatch": any(
                a.get("batch_size") != b.get("batch_size") for a, b in pairs
            ),
        }
    return result


def matched_reference(observation, candidates):
    """Return an exactly matched reference, or None (including legacy unknowns).

    Callers select base for single steps or predecessor for successor chains first.
    Never substitute an unmatched reference when deriving target differences.
    """
    keys = (
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
    )
    if any(k not in observation for k in keys):
        return None
    return next(
        (c for c in candidates if all(k in c and c[k] == observation[k] for k in keys)), None
    )
