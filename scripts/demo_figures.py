#!/usr/bin/env python
"""Write one demo figure per wsp.plots panel into paper/figures/demo_*.png (+ .pdf).

Each figure shows the intended JEPAWG look (docs/FIGURE_STYLE.md) on toy data,
so the style can be checked before any real zoo/eval run exists. Deterministic
for a given --seed. Refuses to finish if any PNG exceeds 300 kB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, CPU only

import numpy as np
from matplotlib import pyplot as plt

from wsp.plots import (
    latent_scatter,
    method_bars,
    pred_vs_measured,
    savefig,
    small_multiples,
    sweep_with_ci,
)
from wsp.plots.style import FIGURES_DIR, OKABE_ITO

PNG_BUDGET = 300 * 1024


def _toy_latent(rng):
    fams = ["random_dir", "abliteration", "heretic_ara", "lora_sft"]
    centers = {
        "random_dir": (0.0, 0.0),
        "abliteration": (1.6, -0.4),
        "heretic_ara": (1.1, 1.2),
        "lora_sft": (-1.2, 1.0),
    }
    z, fam, score = [], [], []
    for f in fams:
        zz = np.asarray(centers[f]) + rng.normal(size=(35, 2)) * 0.45
        z.append(zz)
        fam += [f] * 35
        score += list(np.clip(0.62 - 0.18 * zz[:, 0] + 0.03 * rng.normal(size=35), 0.0, 1.0))
    return np.vstack(z), np.asarray(score), fam


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(FIGURES_DIR), help="output dir (default: paper/figures)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    written: list[Path] = []

    z, score, fam = _toy_latent(rng)
    written += savefig(
        latent_scatter(z, score, fam, title="latent (2-D PCA), toy zoo"),
        "demo_latent_scatter",
        outdir=out,
    )

    x = np.linspace(0.0, 1.0, 9)
    runs = 0.06 - 0.16 * x[:, None] + 0.025 * rng.normal(size=(9, 5))
    written += savefig(
        sweep_with_ci(x, runs, "SSM predictor", color=OKABE_ITO["blue"], ylabel="Δ MATH-500"),
        "demo_sweep_with_ci",
        outdir=out,
    )

    s = np.linspace(0.0, 1.0, 8)
    pred = -0.30 * s**1.2
    meas = pred + 0.02 * rng.normal(size=8)
    written += savefig(
        pred_vs_measured(
            pred, meas, s, "abliteration", title="abliteration: predicted vs. measured"
        ),
        "demo_pred_vs_measured",
        outdir=out,
    )

    methods = ["Δnorm linear", "StatNN", "MLP φ_svd", "Transformer", "Mamba-2 SSM", "JEPA"]
    scores = [0.09, 0.15, 0.24, 0.27, 0.36, 0.34]
    cis = [(0.06, 0.12), (0.12, 0.18), (0.21, 0.27), (0.24, 0.30), (0.33, 0.39), (0.31, 0.37)]
    mask = [True, True, False, False, False, False]
    written += savefig(
        method_bars(methods, scores, cis, mask, ylabel="Spearman ρ on MATH-500"),
        "demo_method_bars",
        outdir=out,
    )

    xm = np.linspace(0.0, 1.0, 11)
    series = {
        "GSM8K": (xm, 0.05 * xm[:, None] + 0.02 * rng.normal(size=(11, 4))),
        "MATH-500": (xm, -0.22 * xm[:, None] + 0.03 * rng.normal(size=(11, 4))),
        "IFEval": (xm, -0.10 * xm[:, None] + 0.02 * rng.normal(size=(11, 4))),
        "MMLU-Pro": (xm, -0.04 * xm[:, None] + 0.015 * rng.normal(size=(11, 4))),
    }
    written += savefig(
        small_multiples(series, xlabel="abliteration strength"),
        "demo_small_multiples",
        outdir=out,
    )
    plt.close("all")

    for p in written:
        print(f"{p}  {p.stat().st_size / 1024:7.1f} kB")
    oversize = [p for p in written if p.suffix == ".png" and p.stat().st_size >= PNG_BUDGET]
    if oversize:
        print(f"FAIL: PNG(s) over the 300 kB budget: {oversize}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
