#!/usr/bin/env python
"""Render the README fallback animation with Matplotlib and Pillow."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from PIL import Image

FPS = 30
FRAMES = 100
COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


def render(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1200)
    ax.set_ylim(300, 0)
    ax.set_axis_off()

    ax.add_patch(
        Rectangle(
            (45, 112),
            160,
            76,
            facecolor="white",
            edgecolor=COLORS["black"],
            linewidth=2,
            joinstyle="round",
        )
    )
    ax.text(125, 154, "BASE MODEL", ha="center", va="center", fontsize=16, weight="bold")

    edits = [
        ("abliterate", "vermillion", (350, 78), (278, 97)),
        ("LoRA", "blue", (350, 150), (278, 150)),
        ("merge", "bluish_green", (350, 222), (278, 204)),
    ]
    arrows = []
    edit_labels = []
    for label, color_name, end, label_position in edits:
        arrow = ax.annotate(
            "",
            xy=end,
            xytext=(205, 150),
            arrowprops={"arrowstyle": "-|>", "color": COLORS[color_name], "lw": 2.2},
        )
        arrow.set_alpha(0)
        arrows.append(arrow)
        text = ax.text(
            *label_position,
            label,
            color=COLORS[color_name],
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )
        text.set_alpha(0)
        edit_labels.append(text)

    heights = [20, 34, 48, 28, 41]
    token_colors = ["orange", "blue", "bluish_green", "reddish_purple", "sky_blue"]
    bars = []
    for i, (height, color_name) in enumerate(zip(heights, token_colors)):
        bar = Rectangle(
            (412 + i * 15, 188 - height),
            10,
            height,
            facecolor=COLORS[color_name],
            edgecolor="none",
        )
        bar.set_height(0)
        ax.add_patch(bar)
        bars.append(bar)
    token_label = ax.text(
        450, 225, "DELTA TOKENS", ha="center", va="center", fontsize=11, weight="bold"
    )
    token_label.set_alpha(0)

    ax.annotate(
        "",
        xy=(590, 160),
        xytext=(500, 160),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["black"], "lw": 1.8},
    )
    ax.add_patch(
        Rectangle(
            (605, 109),
            170,
            82,
            facecolor="white",
            edgecolor=COLORS["blue"],
            linewidth=2,
            joinstyle="round",
        )
    )
    ssm_label = ax.text(
        690, 145, "SSM", color=COLORS["blue"], ha="center", va="center", fontsize=19, weight="bold"
    )
    ssm_label.set_alpha(0)
    ax.plot([620, 760], [170, 170], color=COLORS["black"], alpha=0.25, linewidth=1)
    state_dots = ax.scatter(
        [620, 650, 680, 710],
        [170] * 4,
        s=42,
        c=[COLORS[name] for name in ("vermillion", "orange", "bluish_green", "sky_blue")],
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    state_dots.set_alpha(0)

    ax.annotate(
        "",
        xy=(905, 150),
        xytext=(785, 150),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["black"], "lw": 1.8},
    )
    curve_x = np.linspace(935, 1150, 80)
    curve_t = (curve_x - curve_x[0]) / (curve_x[-1] - curve_x[0])
    curve_y = 200 - 118 * curve_t**1.25
    (forecast,) = ax.plot([], [], color=COLORS["blue"], linewidth=4, solid_capstyle="round")
    endpoint = ax.scatter([], [], s=38, color=COLORS["vermillion"], zorder=5)
    forecast_label = ax.text(
        1040,
        235,
        "FORECAST",
        color=COLORS["blue"],
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    forecast_label.set_alpha(0)
    ax.plot([80, 1120], [276, 276], color="#D9D9D9", linewidth=2)
    progress_marker = ax.scatter([80], [276], s=18, color=COLORS["orange"], zorder=6)

    def update(frame: int) -> list:
        progress = frame / (FRAMES - 1)
        progress_marker.set_offsets([[80 + progress * 1040, 276]])
        for i, (arrow, label) in enumerate(zip(arrows, edit_labels)):
            alpha = np.clip((progress - (0.05 + i * 0.1)) / 0.08, 0, 1)
            arrow.set_alpha(float(alpha))
            label.set_alpha(float(alpha))

        token_progress = np.clip((progress - 0.28) / 0.12, 0, 1)
        for bar, height in zip(bars, heights):
            grown = height * token_progress
            bar.set_height(grown)
            bar.set_y(188 - grown)
        token_label.set_alpha(float(token_progress))

        state_progress = float(np.clip((progress - 0.45) / 0.1, 0, 1))
        state_dots.set_alpha(state_progress)
        positions = [620 + ((frame * 2.7 + i * 38) % 140) for i in range(4)]
        state_dots.set_offsets(np.column_stack([positions, [170] * 4]))
        ssm_label.set_alpha(state_progress)

        curve_progress = np.clip((progress - 0.66) / 0.16, 0, 1)
        count = max(2, int(len(curve_x) * curve_progress))
        forecast.set_data(curve_x[:count], curve_y[:count])
        endpoint.set_offsets([[curve_x[count - 1], curve_y[count - 1]]])
        endpoint.set_alpha(float(curve_progress))
        forecast_label.set_alpha(float(curve_progress))
        return [
            *arrows,
            *edit_labels,
            *bars,
            token_label,
            state_dots,
            ssm_label,
            forecast,
            endpoint,
            forecast_label,
            progress_marker,
        ]

    animation = FuncAnimation(
        fig,
        update,
        frames=FRAMES,
        interval=1000 / FPS,
        blit=True,
        repeat=True,
    )
    animation.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    with Image.open(out) as rendered:
        frames = []
        for frame in range(rendered.n_frames):
            rendered.seek(frame)
            frames.append(rendered.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE))
    durations = [30 if frame % 3 != 2 else 40 for frame in range(len(frames))]
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "assets" / "pipeline.gif"
    parser.add_argument("--out", type=Path, default=default, help="output GIF path")
    args = parser.parse_args(argv)
    render(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
