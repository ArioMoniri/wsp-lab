"""Figure style for wsp-lab: JEPAWG look, house rules in docs/FIGURE_STYLE.md.

Copied from the JEPAWG papers (arXiv 2607.07127 / 2610.30847): white background,
one panel per idea, thin full-box axes with outward ticks, small filled markers,
a tight right-hand colour bar on coloured scatters, thin per-seed lines behind a
thick mean line, shaded bands over seeds, dashed reference lines at known
critical values, frameless legends, no grid.

House rules layered on top (docs/FIGURE_STYLE.md): Okabe–Ito for categories,
viridis for ordered variables, TMLR column widths 3.25 / 6.75 in, PDF + PNG
export, bootstrap 95 % bands.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# TMLR column widths, inches.
COL_WIDTH = 3.25
DBL_WIDTH = 6.75

# Where savefig() drops figures by default (repo-root/paper/figures).
FIGURES_DIR = Path(__file__).resolve().parents[3] / "paper" / "figures"

# Okabe–Ito colour-blind-safe categorical palette.
OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Edit families (plus the untouched base) mapped to palette entries. `chain`
# keeps the yellow slot; it composes other families, so it rarely colours a
# panel, and plot helpers give it a dark edge when it does.
EDIT_FAMILY_COLORS: dict[str, str] = {
    "llm_edit": OKABE_ITO["sky_blue"],
    "base": OKABE_ITO["black"],
    "abliteration": OKABE_ITO["vermillion"],
    "heretic_ara": OKABE_ITO["reddish_purple"],
    "lora_sft": OKABE_ITO["blue"],
    "task_vector": OKABE_ITO["bluish_green"],
    "random_dir": OKABE_ITO["sky_blue"],
    "public_variant": OKABE_ITO["orange"],
    "chain": OKABE_ITO["yellow"],
}

# Marker shape per edit family (latent plots: shape = family, colour = score).
EDIT_FAMILY_MARKERS: dict[str, str] = {
    "llm_edit": "*",
    "base": "o",
    "abliteration": "^",
    "heretic_ara": "v",
    "lora_sft": "s",
    "task_vector": "D",
    "random_dir": "*",
    "public_variant": "P",
    "chain": "X",
}

# Ordered variables (edit strength, KL target) use viridis.
ORDERED_CMAP = mpl.colormaps["viridis"]

# Cycle for method comparisons and small multiples, in a fixed order.
CATEGORICAL_CYCLE = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["orange"],
]

RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.titlepad": 3.5,
    "axes.grid": False,
    "axes.spines.top": True,  # JEPAWG draws full boxes
    "axes.spines.right": True,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "legend.handlelength": 1.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def apply_style() -> None:
    """Apply the wsp figure style (idempotent)."""
    mpl.rcParams.update(RC_PARAMS)


def ordered_color(t: float) -> str:
    """Hex colour for an ordered variable at t ∈ [0, 1] (viridis)."""
    return mpl.colors.to_hex(ORDERED_CMAP(float(np.clip(t, 0.0, 1.0))))


def savefig(
    fig: plt.Figure,
    name: str,
    *,
    outdir: str | Path | None = None,
    dpi: int = 200,
) -> list[Path]:
    """Export a figure as PDF + PNG into paper/figures (or `outdir`).

    Returns the two written paths, PDF first. `name` must be a bare file stem;
    pass a subdirectory through `outdir` instead.
    """
    if not _SAFE_NAME.match(name):
        raise ValueError(f"figure name {name!r} must be a bare file stem (no path separators)")
    out = Path(outdir) if outdir is not None else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, format=ext, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths
