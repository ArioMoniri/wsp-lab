#!/usr/bin/env python
"""Generate the compact, native-shape Bodymovin pipeline animation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FPS = 30
FRAMES = 100
WIDTH = 1200
HEIGHT = 300

COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


def _rgb(value: str) -> list[float]:
    value = COLORS.get(value, value)
    return [round(int(value[i : i + 2], 16) / 255, 3) for i in (1, 3, 5)]


def _static(value: Any) -> dict:
    return {"a": 0, "k": value}


def _fade(start: int) -> dict:
    easing = {"x": [0.333], "y": [0.0]}
    ending = {"x": [0.667], "y": [1.0]}
    frames = []
    if start:
        frames.append({"t": 0, "s": [0], "e": [0], "o": easing, "i": ending})
    frames.extend(
        [
            {"t": start, "s": [0], "e": [100], "o": easing, "i": ending},
            {"t": min(start + 8, FRAMES), "s": [100]},
        ]
    )
    return {"a": 1, "k": frames}


def _path(
    vertices: list[list[float]],
    *,
    closed: bool = False,
    incoming: list[list[float]] | None = None,
    outgoing: list[list[float]] | None = None,
) -> dict:
    zeros = [[0, 0] for _ in vertices]
    return {
        "i": incoming if incoming is not None else zeros,
        "o": outgoing if outgoing is not None else zeros,
        "v": vertices,
        "c": closed,
    }


def _shape_group(items: list[dict], name: str) -> dict:
    type_names = {
        "sh": "ADBE Vector Shape - Group",
        "rc": "ADBE Vector Shape - Rect",
        "el": "ADBE Vector Shape - Ellipse",
        "fl": "ADBE Vector Graphic - Fill",
        "st": "ADBE Vector Graphic - Stroke",
    }
    for index, item in enumerate(items, 1):
        item.setdefault("ix", index)
        item.setdefault("mn", type_names.get(item["ty"], "ADBE Vector Graphic"))
        item.setdefault("hd", False)
    transform = {
        "ty": "tr",
        "a": _static([0, 0]),
        "p": _static([0, 0]),
        "s": _static([100, 100]),
        "r": _static(0),
        "o": _static(100),
        "sk": _static(0),
        "sa": _static(0),
        "nm": "Transform",
        "ix": len(items) + 1,
        "mn": "ADBE Vector Transform Group",
        "hd": False,
    }
    return {
        "ty": "gr",
        "it": [*items, transform],
        "nm": name,
        "np": len(items),
        "cix": 2,
        "bm": 0,
        "ix": 1,
        "mn": "ADBE Vector Group",
        "hd": False,
    }


def _line_shapes(
    vertices: list[list[float]],
    color: str,
    *,
    width: float = 2,
    closed: bool = False,
    fill: bool = False,
    incoming: list[list[float]] | None = None,
    outgoing: list[list[float]] | None = None,
) -> list[dict]:
    shapes = [
        {
            "ty": "sh",
            "ks": {
                "a": 0,
                "k": _path(vertices, closed=closed, incoming=incoming, outgoing=outgoing),
            },
            "nm": "vector path",
            "ind": 1,
        }
    ]
    if fill:
        shapes.append(
            {
                "ty": "fl",
                "c": _static(_rgb(color)),
                "o": _static(100),
                "r": 1,
                "bm": 0,
                "nm": "fill",
            }
        )
    else:
        shapes.append(
            {
                "ty": "st",
                "c": _static(_rgb(color)),
                "o": _static(100),
                "w": _static(width),
                "lc": 2,
                "lj": 2,
                "ml": 4,
                "bm": 0,
                "nm": "stroke",
            }
        )
    return [_shape_group(shapes, "vector path")]


def _arrow_shapes(start: tuple[float, float], end: tuple[float, float], color: str) -> list[dict]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base = (end[0] - ux * 10, end[1] - uy * 10)
    head = [
        [end[0], end[1]],
        [base[0] + px * 5, base[1] + py * 5],
        [base[0] - px * 5, base[1] - py * 5],
    ]
    return _line_shapes([list(start), list(end)], color) + _line_shapes(
        head, color, closed=True, fill=True
    )


def _rect_shapes(
    width: float,
    height: float,
    color: str,
    *,
    position: tuple[float, float] = (0, 0),
    radius: float = 7,
    stroke: str | None = None,
) -> list[dict]:
    shapes = [
        {
            "ty": "rc",
            "d": 1,
            "s": _static([width, height]),
            "p": _static(list(position)),
            "r": _static(radius),
            "nm": "rounded rectangle",
        },
        {
            "ty": "fl",
            "c": _static(_rgb(color)),
            "o": _static(100),
            "r": 1,
            "bm": 0,
            "nm": "fill",
        },
    ]
    if stroke:
        shapes.append(
            {
                "ty": "st",
                "c": _static(_rgb(stroke)),
                "o": _static(100),
                "w": _static(1.5),
                "lc": 2,
                "lj": 2,
                "ml": 4,
                "bm": 0,
                "nm": "outline",
            }
        )
    return [_shape_group(shapes, "rounded rectangle")]


def _shape_layer(
    index: int,
    name: str,
    shapes: list[dict],
    *,
    position: tuple[float, float] = (0, 0),
    visible_at: int = 0,
) -> dict:
    for shape_index, shape in enumerate(shapes, 1):
        shape["ix"] = shape_index
    return {
        "ddd": 0,
        "ind": index,
        "ty": 4,
        "nm": name,
        "sr": 1,
        "ks": {
            "o": _fade(visible_at),
            "r": _static(0),
            "p": _static([position[0], position[1], 0]),
            "a": _static([0, 0, 0]),
            "s": _static([100, 100, 100]),
        },
        "ao": 0,
        "shapes": shapes,
        "ip": 0,
        "op": FRAMES,
        "st": 0,
        "bm": 0,
    }


def _text_layer(
    index: int,
    name: str,
    text: str,
    position: tuple[float, float],
    *,
    color: str = "black",
    size: int = 16,
    visible_at: int = 0,
) -> dict:
    return {
        "ddd": 0,
        "ind": index,
        "ty": 5,
        "nm": name,
        "sr": 1,
        "ks": {
            "o": _fade(visible_at),
            "r": _static(0),
            "p": _static([position[0], position[1], 0]),
            "a": _static([0, 0, 0]),
            "s": _static([100, 100, 100]),
        },
        "ao": 0,
        "t": {
            "d": {
                "a": [],
                "k": [
                    {
                        "s": {
                            "s": size,
                            "f": "ArialMT",
                            "t": text,
                            "j": 2,
                            "tr": 0,
                            "lh": size * 1.2,
                            "ls": 0,
                            "fc": _rgb(color),
                        },
                        "t": 0,
                    }
                ],
            },
            "p": {},
            "m": {"g": 1, "a": _static([0, 0])},
            "a": [],
        },
        "ip": 0,
        "op": FRAMES,
        "st": 0,
        "bm": 0,
    }


def build_animation() -> dict:
    layers: list[dict] = []

    def add_shape(name: str, shapes: list[dict], **kwargs: object) -> None:
        layers.append(_shape_layer(len(layers) + 1, name, shapes, **kwargs))

    def add_text(name: str, text: str, position: tuple[float, float], **kwargs: object) -> None:
        layers.append(_text_layer(len(layers) + 1, name, text, position, **kwargs))

    add_shape(
        "base-model-box",
        _rect_shapes(160, 76, "#FFFFFF", stroke=COLORS["black"]),
        position=(125, 150),
    )
    add_text("base-model-label", "BASE MODEL", (125, 146), size=17)

    edits = [
        ("abliterate", "vermillion", (350, 78), (276, 95)),
        ("LoRA", "blue", (350, 150), (276, 150)),
        ("merge", "bluish_green", (350, 222), (276, 205)),
    ]
    for offset, (label, color_name, end, label_position) in enumerate(edits):
        color = COLORS[color_name]
        add_shape(
            f"edit-arrow-{label.lower()}",
            _arrow_shapes((204, 150), end, color),
            visible_at=5 + offset * 10,
        )
        add_text(
            f"edit-label-{label.lower()}",
            label,
            label_position,
            color=color_name,
            size=14,
            visible_at=5 + offset * 10,
        )

    bar_specs = [
        (-34, 20, "orange"),
        (-17, 34, "blue"),
        (0, 48, "bluish_green"),
        (17, 28, "reddish_purple"),
        (34, 41, "sky_blue"),
    ]
    token_shapes: list[dict] = []
    for x, height, color_name in bar_specs:
        token_shapes.extend(
            _rect_shapes(10, height, COLORS[color_name], position=(x, 20 - height / 2), radius=2)
        )
    add_shape("delta-token-bars", token_shapes, position=(445, 160), visible_at=30)
    add_text("delta-token-label", "DELTA TOKENS", (445, 222), size=13, visible_at=30)
    add_shape(
        "tokens-to-ssm-arrow",
        _arrow_shapes((500, 160), (590, 160), COLORS["black"]),
        visible_at=35,
    )

    add_shape(
        "ssm-block",
        _rect_shapes(170, 82, "#FFFFFF", stroke=COLORS["blue"]),
        position=(690, 150),
        visible_at=40,
    )
    add_text("ssm-label", "SSM", (690, 145), color="blue", size=19, visible_at=40)
    add_shape(
        "ssm-state-track",
        _line_shapes([[615, 170], [765, 170]], COLORS["black"], width=1.2),
        visible_at=42,
    )
    for dot, color_name in enumerate(["vermillion", "orange", "bluish_green", "sky_blue"]):
        layer = _shape_layer(
            len(layers) + 1,
            f"ssm-state-flow-{dot + 1}",
            [
                _shape_group(
                    [
                        {
                            "ty": "el",
                            "d": 1,
                            "p": _static([0, 0]),
                            "s": _static([11, 11]),
                            "nm": "flowing state",
                        },
                        {
                            "ty": "fl",
                            "c": _static(_rgb(COLORS[color_name])),
                            "o": _static(100),
                            "r": 1,
                            "bm": 0,
                            "nm": "state colour",
                        },
                    ],
                    "state dot",
                )
            ],
            visible_at=48,
        )
        start = 620 + dot * 24
        end = 755 + dot * 24
        layer["ks"]["p"] = {
            "a": 1,
            "k": [
                {
                    "t": 0,
                    "s": [start, 170, 0],
                    "e": [end, 170, 0],
                    "o": {"x": [0.333], "y": [0]},
                    "i": {"x": [0.667], "y": [1]},
                },
                {"t": FRAMES, "s": [end, 170, 0]},
            ],
        }
        layers.append(layer)

    add_shape(
        "ssm-to-forecast-arrow",
        _arrow_shapes((785, 150), (905, 150), COLORS["black"]),
        visible_at=58,
    )
    curve = _line_shapes(
        [[935, 200], [1035, 160], [1150, 82]],
        COLORS["blue"],
        width=4,
        incoming=[[0, 0], [-32, 18], [-36, 8]],
        outgoing=[[35, -8], [35, -24], [0, 0]],
    )
    add_shape("forecast-curve", curve, visible_at=68)
    add_shape(
        "forecast-endpoint",
        [
            _shape_group(
                [
                    {
                        "ty": "el",
                        "d": 1,
                        "p": _static([0, 0]),
                        "s": _static([12, 12]),
                        "nm": "forecast point",
                    },
                    {
                        "ty": "fl",
                        "c": _static(_rgb(COLORS["vermillion"])),
                        "o": _static(100),
                        "r": 1,
                        "bm": 0,
                        "nm": "forecast colour",
                    },
                ],
                "forecast endpoint",
            )
        ],
        position=(1150, 82),
        visible_at=76,
    )
    add_text(
        "forecast-label",
        "FORECAST",
        (1040, 235),
        color="blue",
        size=14,
        visible_at=68,
    )

    return {
        "v": "5.7.4",
        "fr": FPS,
        "ip": 0,
        "op": FRAMES,
        "w": WIDTH,
        "h": HEIGHT,
        "nm": "wsp weight-edit prediction pipeline",
        "ddd": 0,
        "assets": [],
        "fonts": {
            "list": [
                {
                    "fName": "ArialMT",
                    "fFamily": "Arial",
                    "fStyle": "Regular",
                    "ascent": 75,
                }
            ]
        },
        "layers": layers,
        "markers": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "assets" / "lottie"
    parser.add_argument("--out", type=Path, default=default, help="output directory")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "pipeline.json"
    output.write_text(json.dumps(build_animation(), separators=(",", ":")) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
