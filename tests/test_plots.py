"""wsp.plots: JEPAWG-style figure panels, palettes and savefig (docs/FIGURE_STYLE.md)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, CPU-only test tier

import runpy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from wsp.plots import panels, style
from wsp.schema import EDIT_FAMILIES

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- style ----


def test_importing_wsp_plots_applies_house_style():
    rc = plt.rcParams
    assert rc["figure.dpi"] == 200
    assert rc["font.size"] == 8
    assert rc["axes.linewidth"] == 0.6
    assert rc["legend.frameon"] is False
    assert rc["savefig.bbox"] == "tight"
    assert rc["axes.grid"] is False
    assert rc["figure.facecolor"] == "white"
    assert rc["font.family"] == ["sans-serif"]


def test_column_widths_match_tmlr():
    assert style.COL_WIDTH == 3.25
    assert style.DBL_WIDTH == 6.75


def test_categorical_palette_is_okabe_ito_and_covers_families():
    assert style.OKABE_ITO == {
        "black": "#000000",
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
    }
    families = {"base"} | set(EDIT_FAMILIES)
    assert set(style.EDIT_FAMILY_COLORS) == families
    assert set(style.EDIT_FAMILY_MARKERS) == families
    assert set(style.EDIT_FAMILY_COLORS.values()) <= set(style.OKABE_ITO.values())


def test_ordered_palette_is_viridis():
    assert style.ORDERED_CMAP.name == "viridis"
    assert style.ordered_color(0.0) == "#440154"
    assert style.ordered_color(1.0) == "#fde725"
    assert style.ordered_color(-0.5) == style.ordered_color(0.0)


def test_savefig_exports_pdf_and_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = style.savefig(fig, "unit_probe", outdir=tmp_path)
    assert [p.suffix for p in out] == [".pdf", ".png"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in out)
    assert style.FIGURES_DIR.name == "figures" and style.FIGURES_DIR.parent.name == "paper"
    plt.close(fig)


def test_savefig_rejects_unsafe_names(tmp_path):
    fig, _ax = plt.subplots()
    with pytest.raises(ValueError):
        style.savefig(fig, "../escape", outdir=tmp_path)
    with pytest.raises(ValueError):
        style.savefig(fig, "a/b", outdir=tmp_path)
    with pytest.raises(ValueError):
        style.savefig(fig, "", outdir=tmp_path)
    plt.close(fig)


# ---------------------------------------------------------------- panels ----


def _toy_latent(n_per_family=30, seed=0):
    rng = np.random.default_rng(seed)
    fams = ["random_dir", "abliteration", "lora_sft"]
    z, fam = [], []
    for k, f in enumerate(fams):
        z.append(rng.normal(size=(n_per_family, 2)) * 0.5 + k * np.array([1.4, -0.6]))
        fam += [f] * n_per_family
    z = np.vstack(z)
    color = np.clip(0.55 + 0.25 * z[:, 0] + 0.05 * rng.normal(size=len(z)), 0, 1)
    return z, color, fam


def test_latent_scatter_returns_figure_with_colorbar_and_family_markers():
    z, color, fam = _toy_latent()
    fig = panels.latent_scatter(z, color, fam)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.collections) == 3  # one scatter per edit family
    assert ax.get_xlabel() == "PC 1" and ax.get_ylabel() == "PC 2"
    assert len(fig.axes) == 2  # panel + colour bar
    assert fig.axes[1].get_ylabel() == "measured MATH-500"
    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == set(fam)
    assert fig.get_size_inches()[0] == pytest.approx(style.COL_WIDTH)


def test_latent_scatter_accepts_single_family_string():
    z, color, _ = _toy_latent(n_per_family=8)
    fig = panels.latent_scatter(z[:8], color[:8], "lora_sft")
    legend = fig.axes[0].get_legend()
    assert {t.get_text() for t in legend.get_texts()} == {"lora_sft"}


def test_latent_scatter_rejects_unknown_family():
    z, color, _ = _toy_latent(n_per_family=4)
    with pytest.raises(ValueError, match="family"):
        panels.latent_scatter(z[:4], color[:4], ["magic"] * 4)


def test_latent_scatter_rejects_empty_or_non_finite_data():
    with pytest.raises(ValueError, match="shape"):
        panels.latent_scatter([], [], [])
    with pytest.raises(ValueError, match="finite"):
        panels.latent_scatter([[0.0, np.nan]], [0.5], "base")


def test_sweep_with_ci_draws_bootstrap_band_and_seed_markers():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, 9)
    y_runs = 0.1 - 0.2 * x[:, None] + 0.02 * rng.normal(size=(9, 5))
    fig = panels.sweep_with_ci(x, y_runs, "SSM")
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    lines = [ln for ln in ax.get_lines() if ln.get_linestyle() != "None"]
    assert len(lines) == 6  # mean + 5 individual runs (n_runs <= 10)
    mean = max(lines, key=lambda ln: ln.get_linewidth())
    assert np.allclose(mean.get_ydata(), y_runs.mean(axis=1), atol=1e-12)
    assert mean.get_marker() == "o"  # markers for individual seeds
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(bands) == 1
    legend = ax.get_legend()
    assert legend is not None and "SSM" in {t.get_text() for t in legend.get_texts()}


def test_sweep_with_ci_band_is_deterministic_and_hides_runs_above_ten():
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 1.0, 12)
    y_runs = 0.05 * x[:, None] + 0.03 * rng.normal(size=(12, 12))
    f1 = panels.sweep_with_ci(x, y_runs, "a", seed=3)
    f2 = panels.sweep_with_ci(x, y_runs, "a", seed=3)
    b1 = next(c for c in f1.axes[0].collections if isinstance(c, PolyCollection))
    b2 = next(c for c in f2.axes[0].collections if isinstance(c, PolyCollection))
    v1 = b1.get_paths()[0].vertices
    v2 = b2.get_paths()[0].vertices
    assert np.array_equal(v1, v2)
    assert (v1[:, 1].max() - v1[:, 1].min()) > 0  # the band has width
    assert len(f1.axes[0].get_lines()) == 1  # n_runs > 10: mean only, no seed spaghetti


def test_sweep_with_ci_band_covers_mean_of_many_runs():
    rng = np.random.default_rng(4)
    x = np.linspace(0.0, 1.0, 15)
    y_runs = 0.2 - 0.3 * x[:, None] + 0.01 * rng.normal(size=(15, 40))
    fig = panels.sweep_with_ci(x, y_runs, "b")
    band = next(c for c in fig.axes[0].collections if isinstance(c, PolyCollection))
    verts = band.get_paths()[0].vertices[:, 1]
    mean = y_runs.mean(axis=1)
    assert (verts.max() >= mean.max() - 1e-9) and (verts.min() <= mean.min() + 1e-9)


def test_sweep_with_ci_rejects_invalid_run_matrix():
    with pytest.raises(ValueError, match="shape"):
        panels.sweep_with_ci([0, 1], np.zeros((2, 1, 1)), "bad")


def test_pred_vs_measured_overlays_measured_points():
    rng = np.random.default_rng(5)
    strength = np.linspace(0.0, 1.0, 8)
    pred = 0.02 - 0.08 * strength
    meas = pred + 0.01 * rng.normal(size=8)
    fig = panels.pred_vs_measured(pred, meas, strength, "abliteration")
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    solid = [ln for ln in ax.get_lines() if ln.get_linestyle() == "-"]
    assert any(ln.get_color() == style.EDIT_FAMILY_COLORS["abliteration"] for ln in solid)
    measured = [ln for ln in ax.get_lines() if ln.get_linestyle() == "None"]
    assert len(measured) == 1 and measured[0].get_marker() == "o"
    assert measured[0].get_markerfacecolor() in ("none", "None")
    assert "strength" in ax.get_xlabel() and "score" in ax.get_ylabel()
    texts = {t.get_text() for t in ax.get_legend().get_texts()}
    assert any("predicted" in t for t in texts) and any("measured" in t for t in texts)
    zero = [ln for ln in ax.get_lines() if np.allclose(np.asarray(ln.get_ydata(), dtype=float), 0)]
    assert zero, "expected a dashed reference line at Δscore = 0"


def test_pred_vs_measured_rejects_unknown_family():
    with pytest.raises(ValueError, match="family"):
        panels.pred_vs_measured(np.zeros(4), np.zeros(4), np.linspace(0, 1, 4), "magic")


def test_method_bars_hatch_only_baselines_with_error_bars():
    methods = ["Δnorm linear", "StatNN", "MLP φ_svd", "Transformer", "Mamba-2 SSM", "JEPA"]
    scores = [0.11, 0.18, 0.29, 0.33, 0.41, 0.40]
    cis = [(0.08, 0.14), (0.15, 0.21), (0.26, 0.32), (0.30, 0.36), (0.38, 0.44), (0.37, 0.43)]
    mask = [True, True, False, False, False, False]
    fig = panels.method_bars(methods, scores, cis, mask)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    bars = ax.patches
    assert len(bars) == len(methods)
    for bar, is_base in zip(bars, mask):
        assert (bar.get_hatch() == "///") == is_base
    from matplotlib.container import BarContainer

    for c in ax.containers:
        if isinstance(c, BarContainer):
            assert c.errorbar is not None
    assert fig.get_size_inches()[0] == pytest.approx(style.DBL_WIDTH)  # 6 methods


def test_method_bars_rejects_length_mismatch():
    with pytest.raises(ValueError):
        panels.method_bars(["a", "b"], [0.1], [(0.0, 0.2)], [True, False])


def test_small_multiples_one_panel_per_series():
    x = np.linspace(0.0, 1.0, 10)
    series = {
        "abliteration": (x, 0.1 - 0.2 * x),
        "lora_sft": (x, 0.05 * x),
        "task_vector": (x, 0.02 + 0.01 * x),
        "random_dir": (x, np.zeros(10)),
    }
    fig = panels.small_multiples(series)
    assert isinstance(fig, plt.Figure)
    axes = fig.axes
    assert len(axes) == len(series)
    assert [ax.get_title() for ax in axes] == list(series)
    assert all(len(ax.get_lines()) >= 1 for ax in axes)
    assert fig.get_size_inches()[0] == pytest.approx(style.DBL_WIDTH)


def test_small_multiples_accept_run_matrices_and_reject_empty():
    x = np.linspace(0.0, 1.0, 7)
    rng = np.random.default_rng(6)
    runs = 0.1 * x[:, None] + 0.02 * rng.normal(size=(7, 4))
    fig = panels.small_multiples({"gsm8k": (x, runs), "math500": (x, runs * 2)})
    for ax in fig.axes:
        assert any(isinstance(c, PolyCollection) for c in ax.collections)
    with pytest.raises(ValueError, match="empty"):
        panels.small_multiples({})


# ----------------------------------------------------------------- demo ----


def test_demo_figures_script_writes_one_example_per_panel(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["demo_figures.py", "--out", str(tmp_path), "--seed", "0"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPTS / "demo_figures.py"), run_name="__main__")
    assert exc.value.code == 0
    names = [
        "demo_latent_scatter",
        "demo_sweep_with_ci",
        "demo_pred_vs_measured",
        "demo_method_bars",
        "demo_small_multiples",
    ]
    for name in names:
        png = tmp_path / f"{name}.png"
        pdf = tmp_path / f"{name}.pdf"
        assert png.is_file() and png.stat().st_size > 0, name
        assert pdf.is_file() and pdf.stat().st_size > 0, name
        assert png.stat().st_size < 300 * 1024, f"{name} exceeds 300 kB"
