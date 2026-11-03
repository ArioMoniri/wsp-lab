"""Stable seed-0 positional IDs in the published evaluation splits.

Regenerate without network access: python -m wsp.eval.subsets --regen.
Dataset identities and split cardinalities are checked when loading the battery.
"""

import argparse
import random
from pathlib import Path

SPECS = {
    "gsm8k": ("openai/gsm8k", "main", 1319, 200, "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, 500, 100, "test"),
    "ifeval": ("google/IFEval", None, 541, 100, "train"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, 12032, 300, "test"),
    "hellaswag": ("Rowan/hellaswag", None, 10042, 300, "validation"),
}
REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "ifeval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "mmlu_pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
}
ROOT = Path(__file__).resolve().parents[3] / "configs/eval_subsets"


def subset_ids(size, n, seed=0):
    return [str(i) for i in sorted(random.Random(seed).sample(range(size), n))]


def load_items(name, n=None):
    from datasets import load_dataset

    repo, config, size, _count, split = SPECS[name]
    ds = load_dataset(repo, config, split=split, revision=REVISIONS[name])
    if len(ds) != size:
        raise ValueError(f"{name}: split size changed: {len(ds)} != {size}")
    ids = (ROOT / f"{name}.txt").read_text().splitlines()
    return [(i, ds[int(i)]) for i in ids[:n]], {
        "repo": repo,
        "revision": REVISIONS[name],
        "config": config,
        "split": split,
        "fingerprint": ds._fingerprint,
        "subset_ids": ids[:n],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--regen", action="store_true", required=True)
    parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (_, _, size, n, _) in SPECS.items():
        (ROOT / f"{name}.txt").write_text("\n".join(subset_ids(size, n)) + "\n")


if __name__ == "__main__":
    main()
