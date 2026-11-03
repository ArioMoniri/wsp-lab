"""Figure panels for wsp-lab, in the JEPAWG style (docs/FIGURE_STYLE.md).

Every function builds and returns a `matplotlib.figure.Figure`; nothing is
written to disk here — call `wsp.plots.style.savefig` on the result.

Conventions (JEPAWG, arXiv 2607.07127 / 2610.30847):
- thin per-seed lines (alpha ~0.35) behind a thick mean line;
- shaded bootstrap 95 % band over the run/edit axis;
- latent scatter coloured by a measured score (viridis) with marker shape per
  edit family and a tight right-hand colour bar;
- measured overlays as open black circles;
- dashed reference line at the known critical value (here: Δscore = 0);
- hatched bars for baselines in method comparisons;
- small multiples: one near-square panel per system, title above, no frames.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .style import (
    CATEGORICAL_CYCLE,
    COL_WIDTH,
    DBL_WIDTH,
    EDIT_FAMILY_COLORS,
    EDIT_FAMILY_MARKERS,
    OKABE_ITO,
    ORDERED_CMAP,
    apply_style,
)

__all__ = [
    "latent_scatter",
    "method_bars",
    "pred_vs_measured",
    "small_multiples",
    "sweep_with_ci",
]


def _check_family(family: str) -> None:
    if family not in EDIT_FAMILY_COLORS:
        known = ", ".join(sorted(EDIT_FAMILY_COLORS))
        raise ValueError(f"unknown edit family {family!r}; known families: {known}")


def _bootstrap_ci(
    y: np.ndarray, *, n_boot: int, seed: int, lo_pct: float = 2.5, hi_pct: float = 97.5
) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise bootstrap CI over the run axis of `y` (m points × n runs).

    Whole runs are resampled jointly (paired bootstrap), so run-level structure
    survives; deterministic for a given `seed`.
    """
    m, n = y.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = y[:, idx].mean(axis=-1)  # (m, n_boot)
    lo, hi = np.percentile(means, [lo_pct, hi_pct], axis=1)
    return lo, hi


def _sweep_into_ax(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str | None,
    color: str,
    seed: int,
    n_boot: int,
    legend: bool = True,
) -> None:
    """Mean line + bootstrap band (+ thin per-run lines when n_runs ≤ 10)."""
    m, n = y.shape
    mean = y.mean(axis=1)
    if n >= 2:
        lo, hi = _bootstrap_ci(y, n_boot=n_boot, seed=seed)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1)
    if 2 <= n <= 10:  # markers for individual seeds when n ≤ 10
        for j in range(n):
            ax.plot(
                x,
                y[:, j],
                color=color,
                linewidth=0.6,
                alpha=0.35,
                marker="o",
                markersize=1.8,
                markerfacecolor=color,
                markeredgecolor="none",
                zorder=2,
            )
    ax.plot(
        x,
        mean,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=3,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=label,
        zorder=3,
    )
    if legend and label:
        ax.legend()


def latent_scatter(
    z2d: Sequence[Sequence[float]],
    color: Sequence[float],
    marker_family: str | Sequence[str],
    *,
    color_label: str = "measured MATH-500",
    title: str | None = None,
) -> plt.Figure:
    """2-D latent (PCA) scatter: colour = measured score (viridis), shape = edit family.

    The wsp analogue of JEPAWG's latent PCA plots: one marker family per edit
    family, points coloured by the measured quantity, tight right colour bar.
    """
    apply_style()
    z = np.asarray(z2d, dtype=float)
    c = np.asarray(color, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"z2d must have shape (n, 2), got {z.shape}")
    if len(c) != len(z):
        raise ValueError(f"color has {len(c)} entries for {len(z)} points")
    if isinstance(marker_family, str):
        fams = np.array([marker_family] * len(z))
    else:
        fams = np.asarray(marker_family)
        if len(fams) != len(z):
            raise ValueError(f"marker_family has {len(fams)} entries for {len(z)} points")
    for f in fams:
        _check_family(str(f))

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.55))
    norm = plt.Normalize(c.min(), c.max())
    order = list(dict.fromkeys(fams.tolist()))  # first-appearance order
    last = None
    for f in order:
        mask = fams == f
        last = ax.scatter(
            z[mask, 0],
            z[mask, 1],
            c=c[mask],
            cmap=ORDERED_CMAP,
            norm=norm,
            marker=EDIT_FAMILY_MARKERS[f],
            s=9,
            alpha=0.9,
            edgecolors="none",
        )
    cbar = fig.colorbar(last, ax=ax, fraction=0.055, pad=0.03, aspect=18)
    cbar.set_label(color_label)
    handles = [
        Line2D(
            [],
            [],
            color="0.3",
            marker=EDIT_FAMILY_MARKERS[f],
            linestyle="none",
            markersize=4,
            markerfacecolor="0.3",
            label=str(f),
        )
        for f in order
    ]
    ax.legend(handles=handles)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if title:
        ax.set_title(title)
    return fig


def sweep_with_ci(
    x: Sequence[float],
    y_runs: Sequence[Sequence[float]],
    label: str,
    *,
    color: str | None = None,
    seed: int = 0,
    n_boot: int = 1000,
    xlabel: str = "edit strength",
    ylabel: str = "Δscore",
    title: str | None = None,
) -> plt.Figure:
    """Parameter sweep: thick mean line with a bootstrap 95 % band over runs.

    `y_runs` is (m sweep points × n runs); individual runs are drawn as thin
    low-alpha lines with markers when n ≤ 10 (JEPAWG seed convention).
    """
    apply_style()
    xv = np.asarray(x, dtype=float)
    y = np.asarray(y_runs, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if xv.ndim != 1 or len(xv) != y.shape[0]:
        raise ValueError(f"x has shape {xv.shape} but y_runs gives {y.shape[0]} sweep points")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.3))
    _sweep_into_ax(
        ax,
        xv,
        y,
        label=str(label),
        color=color or OKABE_ITO["blue"],
        seed=seed,
        n_boot=n_boot,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig


def pred_vs_measured(
    pred: Sequence[float],
    meas: Sequence[float],
    strength: Sequence[float],
    family: str,
    *,
    title: str | None = None,
) -> plt.Figure:
    """Predicted Δscore vs. edit strength for one edit family, measured overlaid.

    Our analogue of JEPAWG's "derivative of decoded weights vs. parameter"
    response plot: the predictor's curve in the family colour, the measured
    Δscore as open black circles, and a dashed reference line at Δscore = 0.
    """
    apply_style()
    _check_family(family)
    p = np.asarray(pred, dtype=float)
    m = np.asarray(meas, dtype=float)
    s = np.asarray(strength, dtype=float)
    if not (len(p) == len(m) == len(s)):
        raise ValueError("pred, meas and strength must have the same length")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))
    ax.axhline(0.0, color="0.55", linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
    ax.plot(
        s,
        p,
        color=EDIT_FAMILY_COLORS[family],
        linewidth=1.2,
        marker=EDIT_FAMILY_MARKERS[family],
        markersize=3,
        markerfacecolor=EDIT_FAMILY_COLORS[family],
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=f"predicted ({family})",
        zorder=3,
    )
    ax.plot(
        s,
        m,
        linestyle="none",
        marker="o",
        markersize=4,
        markerfacecolor="none",
        markeredgecolor="0.15",
        markeredgewidth=0.6,
        label="measured",
        zorder=4,
    )
    ax.legend()
    ax.set_xlabel("edit strength")
    ax.set_ylabel("Δscore")
    if title:
        ax.set_title(title)
    return fig


def method_bars(
    methods: Sequence[str],
    scores: Sequence[float],
    cis: Sequence[Sequence[float]],
    baseline_mask: Sequence[bool],
    *,
    ylabel: str = "Δscore",
    title: str | None = None,
) -> plt.Figure:
    """Method comparison bars with 95 % CI whiskers; baselines are hatched.

    `cis` holds absolute (lo, hi) bounds per method; entries where
    `baseline_mask` is True are drawn grey with a `///` hatch.
    """
    apply_style()
    k = len(methods)
    if not (len(scores) == len(cis) == len(baseline_mask) == k):
        raise ValueError("methods, scores, cis and baseline_mask must have equal lengths")
    scores_a = np.asarray(scores, dtype=float)
    lo = np.array([float(c[0]) for c in cis])
    hi = np.array([float(c[1]) for c in cis])
    err = np.vstack([np.clip(scores_a - lo, 0, None), np.clip(hi - scores_a, 0, None)])

    width = COL_WIDTH if k <= 4 else DBL_WIDTH
    fig, ax = plt.subplots(figsize=(width, 2.3))
    xs = np.arange(k)
    edge_kw = {"edgecolor": "0.25", "linewidth": 0.5}
    error_kw = {"ecolor": "0.2", "elinewidth": 0.6, "capsize": 1.5, "capthick": 0.6}
    main_i = 0
    for i, (name, score, is_base) in enumerate(zip(methods, scores_a, baseline_mask)):
        yerr = np.array([[err[0, i]], [err[1, i]]])  # (2, 1): lo and hi offsets for one bar
        if is_base:
            ax.bar(xs[i], score, width=0.62, facecolor="0.82", hatch="///", yerr=yerr,
                   error_kw=error_kw, **edge_kw)
        else:
            color = CATEGORICAL_CYCLE[main_i % len(CATEGORICAL_CYCLE)]
            main_i += 1
            ax.bar(xs[i], score, width=0.62, facecolor=color, yerr=yerr, error_kw=error_kw,
                   **edge_kw)
    ax.set_xticks(xs, [str(m) for m in methods], rotation=20, rotation_mode="anchor", ha="right")
    ax.yaxis.grid(True, color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig


def small_multiples(
    dict_of_series: dict[str, tuple[Sequence[float], Sequence[float] | Sequence[Sequence[float]]]],
    *,
    xlabel: str = "edit strength",
    ylabel: str = "Δscore",
    seed: int = 0,
    n_boot: int = 1000,
) -> plt.Figure:
    """One panel per series (JEPAWG small multiples).

    Each value is `(x, y)`; `y` is either a 1-D curve (single line) or an
    (m × n) run matrix, which is drawn as mean + bootstrap band like
    `sweep_with_ci`. Panel titles are the dict keys; only the leftmost panel
    carries the y-label.
    """
    apply_style()
    if not dict_of_series:
        raise ValueError("dict_of_series is empty")
    keys = list(dict_of_series)
    width = COL_WIDTH if len(keys) == 1 else DBL_WIDTH
    fig, axes = plt.subplots(1, len(keys), figsize=(width, 2.1))
    axes = np.atleast_1d(axes)
    for i, (key, ax) in enumerate(zip(keys, axes)):
        x, y = dict_of_series[key]
        xv = np.asarray(x, dtype=float)
        yv = np.asarray(y, dtype=float)
        if yv.ndim == 1:
            color = CATEGORICAL_CYCLE[i % len(CATEGORICAL_CYCLE)]
            ax.plot(xv, yv, color=color, linewidth=1.0, marker="o", markersize=2.5, zorder=3)
        else:
            _sweep_into_ax(
                ax,
                xv,
                yv,
                label=None,
                color=CATEGORICAL_CYCLE[i % len(CATEGORICAL_CYCLE)],
                seed=seed,
                n_boot=n_boot,
                legend=False,
            )
        ax.set_title(str(key))
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
    return fig
