#!/usr/bin/env python
"""Generate matching SVG and font-independent Bodymovin from one vector scene.

The forecast is a schematic, not an experimental result. Lottie labels are
outlined locally with the locally installed Liberation Sans; no runtime fonts needed.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath, TextToPath

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
INK = "#25313B"
GRAY = "#64717C"
RULE = "#C8D0D6"
WHITE = "#FFFFFF"


def prop(k):
    return {"a": 0, "k": k}


def animated(values):
    return {
        "a": 1,
        "k": [
            {
                "t": t,
                "s": [v] if isinstance(v, (int, float)) else v,
                "o": {"x": [0.333], "y": [0.333]},
                "i": {"x": [0.667], "y": [0.667]},
            }
            for t, v in values
        ],
    }


def rgb(c):
    return [round(int(c[i : i + 2], 16) / 255, 4) for i in (1, 3, 5)]


class Scene:
    def __init__(self):
        self.svg = []
        self.layers = []
        self.assets = []
        self.glyphs = {}

    def layer(self, name, items, motion=None):
        items.append(
            {
                "ty": "tr",
                "p": prop([0, 0]),
                "a": prop([0, 0]),
                "s": prop([100, 100]),
                "r": prop(0),
                "o": prop(100),
            }
        )
        self.layers.append(
            {
                "ddd": 0,
                "ind": len(self.layers) + 1,
                "ty": 4,
                "nm": name,
                "sr": 1,
                "ip": 0,
                "op": 120,
                "st": 0,
                "bm": 0,
                "ks": {
                    "p": motion or prop([0, 0, 0]),
                    "a": prop([0, 0, 0]),
                    "s": prop([100, 100, 100]),
                    "r": prop(0),
                    "o": prop(100),
                },
                "shapes": [{"ty": "gr", "nm": name, "it": items}],
            }
        )

    @staticmethod
    def paint(fill, stroke):
        out = []
        if fill:
            out.append({"ty": "fl", "c": prop(rgb(fill)), "o": prop(100), "r": 1})
        if stroke:
            out.append(
                {
                    "ty": "st",
                    "c": prop(rgb(stroke)),
                    "o": prop(100),
                    "w": prop(1.2),
                    "lc": 2,
                    "lj": 2,
                }
            )
        return out

    def rect(self, name, x, y, w, h, fill=WHITE, stroke=RULE, radius=8):
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke or "none"}"/>'
        )
        self.layer(
            name,
            [
                {
                    "ty": "rc",
                    "d": 1,
                    "p": prop([x + w / 2, y + h / 2]),
                    "s": prop([w, h]),
                    "r": prop(radius),
                }
            ]
            + self.paint(fill, stroke),
        )

    def path(
        self,
        name,
        vertices,
        stroke=RULE,
        fill=None,
        closed=False,
        incoming=None,
        outgoing=None,
        flow=False,
        draw=False,
    ):
        zeros = [[0, 0] for _ in vertices]
        incoming, outgoing = incoming or zeros, outgoing or zeros
        d = f"M{vertices[0][0]},{vertices[0][1]}"
        for j in range(1, len(vertices)):
            a, b = vertices[j - 1], vertices[j]
            o, i = outgoing[j - 1], incoming[j]
            d += f" C{a[0] + o[0]},{a[1] + o[1]} {b[0] + i[0]},{b[1] + i[1]} {b[0]},{b[1]}"
        if closed:
            d += " Z"
        attrs, anim = "", ""
        if flow:
            attrs = ' stroke-dasharray="4 7"'
            anim = '<animate attributeName="stroke-dashoffset" from="0" to="-44" dur="4s" repeatCount="indefinite"/>'
        if draw:
            attrs = ' stroke-dasharray="240"'
            anim = '<animate attributeName="stroke-dashoffset" values="240;0;0;240" keyTimes="0;.4;.85;1" dur="4s" repeatCount="indefinite"/>'
        self.svg.append(
            f'<path d="{d}" fill="{fill or "none"}" stroke="{stroke or "none"}"{attrs}>{anim}</path>'
        )
        items = [
            {"ty": "sh", "ks": prop({"v": vertices, "i": incoming, "o": outgoing, "c": closed})}
        ]
        items += self.paint(fill, stroke)
        if flow:
            items[-1]["d"] = [
                {"n": "d", "v": prop(4)},
                {"n": "g", "v": prop(7)},
                {"n": "o", "v": animated([(0, 0), (120, -44)])},
            ]
        if draw:
            items.append(
                {
                    "ty": "tm",
                    "s": prop(0),
                    "e": animated([(0, 0), (48, 100), (102, 100), (120, 0)]),
                    "o": prop(0),
                    "m": 1,
                }
            )
        self.layer(name, items)

    def text(self, text, x, y, size=16, color=INK):
        self.svg.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(text)}</text>'
        )
        glyph_layers = []
        cursor = x
        for ch in text:
            if ch != " ":
                key = (ch, color)
                if key not in self.glyphs:
                    gid = "glyph-" + str(len(self.glyphs))
                    self.glyphs[key] = gid
                    self.layer(gid, self.glyph(ch, color))
                    self.assets.append(
                        {"id": gid, "w": 200, "h": 200, "layers": [self.layers.pop()]}
                    )
                glyph_layers.append(
                    {
                        "ty": 0,
                        "refId": self.glyphs[key],
                        "w": 200,
                        "h": 200,
                        "ind": len(glyph_layers) + 1,
                        "ip": 0,
                        "op": 120,
                        "ks": {
                            "o": prop(100),
                            "r": prop(0),
                            "a": prop([0, 0, 0]),
                            "p": prop([round(cursor, 2), y - size, 0]),
                            "s": prop([size, size, 100]),
                        },
                    }
                )
            cursor += TextToPath().get_text_width_height_descent(
                ch, FontProperties(family="Liberation Sans", size=size), False
            )[0]
        aid = "text-" + str(len(self.assets))
        self.assets.append({"id": aid, "w": 1200, "h": 320, "layers": glyph_layers})
        self.layer("label-" + text, [])
        layer = self.layers[-1]
        del layer["shapes"]
        layer.update(ty=0, refId=aid, w=1200, h=320)

    def glyph(self, text, color):
        # Convert quadratic glyph contours to cubic paths understood by Bodymovin.
        tp = TextPath((0, -100), text, size=100, prop=FontProperties(family="Liberation Sans"))
        items, vs, ins, outs = [], [], [], []

        def finish():
            if vs:
                items.append(
                    {
                        "ty": "sh",
                        "ks": prop({"v": vs.copy(), "i": ins.copy(), "o": outs.copy(), "c": True}),
                    }
                )

        def point(a):
            return [round(float(a[0])), round(-float(a[1]))]

        for coords, code in tp.iter_segments():
            if code == 1:
                finish()
                vs, ins, outs = [point(coords)], [[0, 0]], [[0, 0]]
            elif code == 2:
                vs.append(point(coords))
                ins.append([0, 0])
                outs.append([0, 0])
            elif code == 3:
                q, end = point(coords[:2]), point(coords[2:])
                outs[-1] = [round((q[k] - vs[-1][k]) * 2 / 3) for k in (0, 1)]
                vs.append(end)
                ins.append([round((q[k] - end[k]) * 2 / 3) for k in (0, 1)])
                outs.append([0, 0])
            elif code == 4:
                c1, c2, end = point(coords[:2]), point(coords[2:4]), point(coords[4:])
                outs[-1] = [round(c1[k] - vs[-1][k]) for k in (0, 1)]
                vs.append(end)
                ins.append([round(c2[k] - end[k]) for k in (0, 1)])
                outs.append([0, 0])
            elif code == 79:
                finish()
                vs, ins, outs = [], [], []
        finish()
        return items + self.paint(color, None)

    def arrow(self, name, points, color=RULE):
        self.path(name, points, color, flow=True)
        x, y = points[-1]
        self.path(name + "-head", [[x - 5, y - 3], [x, y], [x - 5, y + 3]], color)


def build_scene():
    s = Scene()
    s.rect("background", 0, 0, 1200, 320, stroke=None, radius=0)
    for x, number, title in [
        (24, "01", "Edit weights"),
        (440, "02", "Encode deltas"),
        (656, "03", "Carry state"),
        (920, "04", "Forecast"),
    ]:
        s.text(number, x, 40, 14, BLUE)
        s.text(title, x + 30, 40, 18)
    s.path("header-rule", [[24, 62], [1176, 62]])
    s.rect("base-model", 24, 130, 168, 86)
    s.text("Base model", 40, 157, 16, GRAY)
    s.text("Qwen3.5-0.8B", 40, 187, 17)
    for label, color, y in [("abliterate", ORANGE, 92), ("LoRA", BLUE, 154), ("merge", GREEN, 216)]:
        s.arrow(
            "edit-arrow-" + label.lower(),
            [[192, 173], [220, 173], [220, y + 20], [248, y + 20]],
            color,
        )
        s.rect("edit-" + label, 248, y, 134, 40, stroke=color)
        s.text(label, 264, y + 26, 17)
        s.path(
            "edit-to-tokens-" + label,
            [[382, y + 20], [408, y + 20], [408, 173], [428, 173]],
            color,
            flow=True,
        )
    s.rect("token-panel", 440, 92, 158, 164)
    s.text("Delta tokens", 456, 120, 18)
    s.text("canonical SVD", 456, 144, 15, GRAY)
    for j, h in enumerate([20, 40, 58, 46, 32, 16]):
        s.rect(
            "delta-token-bars" if j == 0 else f"token-{j}",
            458 + j * 21,
            226 - h,
            9,
            h,
            [ORANGE, BLUE, GREEN][j % 3],
            None,
            2,
        )
    s.arrow("tokens-to-ssm", [[598, 173], [644, 173]], GRAY)
    s.rect("ssm-block", 656, 92, 208, 164, stroke=BLUE)
    s.text("State-space", 672, 122, 18)
    s.text("predictor", 672, 146, 18)
    s.text("edit sequence → state", 672, 173, 14, GRAY)
    s.path("state-track", [[680, 214], [840, 214]], BLUE)
    for x in [688, 760, 832]:
        s.rect("state-cell-" + str(x), x - 7, 207, 14, 14, WHITE, BLUE, 4)
    s.svg.append(
        '<circle cx="688" cy="214" r="4" fill="#0072B2"><animate attributeName="cx" values="688;832;688" dur="4s" repeatCount="indefinite"/><animate attributeName="opacity" values=".4;1;.4" dur="4s" repeatCount="indefinite"/></circle>'
    )
    s.layer(
        "ssm-state-flow",
        [
            {
                "ty": "el",
                "p": prop([0, 214]),
                "s": animated([(0, [6, 6]), (60, [9, 9]), (120, [6, 6])]),
            }
        ]
        + s.paint(BLUE, None),
        animated([(0, [688, 0, 0]), (60, [832, 0, 0]), (120, [688, 0, 0])]),
    )
    s.arrow("ssm-to-forecast", [[864, 173], [908, 173]], GRAY)
    s.rect("forecast-panel", 920, 92, 256, 164)
    s.text("Benchmark change", 936, 120, 18)
    s.path("forecast-axes", [[942, 142], [942, 205], [1154, 205]], RULE)
    s.path(
        "forecast-band",
        [
            [950, 190],
            [1006, 162],
            [1070, 151],
            [1152, 130],
            [1152, 154],
            [1070, 175],
            [1006, 184],
            [950, 202],
        ],
        None,
        "#D9EAF3",
        True,
    )
    s.path(
        "forecast-curve",
        [[950, 196], [1006, 173], [1070, 163], [1152, 142]],
        BLUE,
        incoming=[[0, 0], [-20, 4], [-24, 0], [-28, 0]],
        outgoing=[[22, -2], [24, -4], [28, -2], [0, 0]],
        draw=True,
    )
    s.text("MATH · GSM8K", 936, 226, 15)
    s.text("IFEval · refusal", 936, 247, 15)
    s.text("Schematic · illustrative forecast band", 920, 292, 14, GRAY)
    return s


def svg_document(scene):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 320" '
        'role="img" aria-labelledby="title desc" font-family="Inter, system-ui, sans-serif" '
        'stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">\n'
        '<title id="title">From weight edits to benchmark forecasts</title>\n'
        '<desc id="desc">Qwen3.5-0.8B fans into abliterate, LoRA and merge edits. '
        "Canonical SVD delta tokens feed a state-space predictor and a schematic "
        "forecast with an illustrative confidence band for MATH, GSM8K, IFEval and refusal.</desc>\n"
        + "\n".join(scene.svg)
        + "\n</svg>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write both assets to this directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    json_dir = args.out or root / "assets/lottie"
    svg_dir = args.out or root / "assets"
    json_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    anim = {
        "v": "5.7.4",
        "fr": 30,
        "ip": 0,
        "op": 120,
        "w": 1200,
        "h": 320,
        "nm": "Weight edits to benchmark forecasts (schematic)",
        "ddd": 0,
        "assets": scene.assets,
        "layers": list(reversed(scene.layers)),
        "markers": [],
    }
    (json_dir / "pipeline.json").write_text(json.dumps(anim, separators=(",", ":")) + "\n")
    (svg_dir / "hero.svg").write_text(svg_document(scene))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
