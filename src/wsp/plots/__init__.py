"""wsp.plots: JEPAWG-style figures for the paper (docs/FIGURE_STYLE.md)."""

from .panels import latent_scatter, method_bars, pred_vs_measured, small_multiples, sweep_with_ci
from .style import (
    CATEGORICAL_CYCLE,
    COL_WIDTH,
    DBL_WIDTH,
    EDIT_FAMILY_COLORS,
    EDIT_FAMILY_MARKERS,
    FIGURES_DIR,
    OKABE_ITO,
    ORDERED_CMAP,
    apply_style,
    ordered_color,
    savefig,
)

apply_style()  # `import wsp.plots` styles matplotlib

__all__ = [
    "CATEGORICAL_CYCLE",
    "COL_WIDTH",
    "DBL_WIDTH",
    "EDIT_FAMILY_COLORS",
    "EDIT_FAMILY_MARKERS",
    "FIGURES_DIR",
    "OKABE_ITO",
    "ORDERED_CMAP",
    "apply_style",
    "latent_scatter",
    "method_bars",
    "ordered_color",
    "pred_vs_measured",
    "savefig",
    "small_multiples",
    "sweep_with_ci",
]
