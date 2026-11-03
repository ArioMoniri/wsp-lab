"""Assets: the Lottie pipeline animation, its GIF fallback, and where they are wired in."""

import json
from pathlib import Path

from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOTTIE = ROOT / "assets" / "lottie" / "pipeline.json"
GIF = ROOT / "assets" / "pipeline.gif"
SCRIPTS = ROOT / "scripts"

OKABE_RGB = {
    "black": (0.0, 0.0, 0.0),
    "orange": (0.902, 0.624, 0.0),
    "sky_blue": (0.337, 0.706, 0.914),
    "bluish_green": (0.0, 0.620, 0.451),
    "yellow": (0.941, 0.894, 0.259),
    "blue": (0.0, 0.447, 0.698),
    "vermillion": (0.835, 0.369, 0.0),
    "reddish_purple": (0.8, 0.471, 0.655),
}


def _layer_names_and_texts(anim):
    names = [layer.get("nm", "") for layer in anim["layers"]]
    for layer in anim["layers"]:
        if layer.get("ty") == 5:  # text layer
            doc = layer["t"]["d"]["k"][0]["s"]
            names.append(doc[1])
    return names


def _animation_colors(anim):
    """All fill/stroke colours as (r, g, b) rounded to 3 dp."""
    out = set()

    def walk(node):
        if isinstance(node, dict):
            if node.get("ty") in ("fl", "st") and isinstance(node.get("c"), dict):
                k = node["c"].get("k")
                if isinstance(k, list) and len(k) >= 3 and not isinstance(k[0], dict):
                    out.add(tuple(round(v, 3) for v in k[:3]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(anim)
    return out


def test_lottie_is_valid_bodymovin_json_under_150kb():
    assert LOTTIE.is_file()
    assert LOTTIE.stat().st_size <= 150 * 1024
    anim = json.loads(LOTTIE.read_text())
    for key in ("v", "fr", "ip", "op", "w", "h", "layers"):
        assert key in anim
    assert anim["fr"] == 30
    assert 90 <= anim["op"] - anim["ip"] <= 120  # 3–4 s loop at 30 fps
    assert anim["w"] > 0 and anim["h"] > 0
    assert isinstance(anim["layers"], list) and anim["layers"]


def test_lottie_layers_are_well_formed_and_in_range():
    anim = json.loads(LOTTIE.read_text())
    for layer in anim["layers"]:
        assert layer["ip"] >= anim["ip"] and layer["op"] <= anim["op"]
        assert "ks" in layer and all(k in layer["ks"] for k in ("o", "p", "a", "s"))
        if layer.get("ty") == 4:
            assert layer.get("shapes"), f"shape layer {layer.get('nm')} has no shapes"
        if layer.get("ty") == 5:
            assert "t" in layer and "d" in layer["t"]


def test_lottie_covers_the_pipeline_stages_in_okabe_ito():
    anim = json.loads(LOTTIE.read_text())
    blob = " ".join(_layer_names_and_texts(anim)).lower()
    for stage in ("base", "abliterate", "lora", "merge", "token", "ssm", "forecast"):
        assert stage in blob, f"stage {stage!r} missing from the animation"
    colors = _animation_colors(anim)
    assert len(colors) >= 5
    for name in ("black", "vermillion", "blue", "bluish_green", "reddish_purple", "sky_blue"):
        assert OKABE_RGB[name] in colors, f"Okabe–Ito {name} not used in the animation"
    for rgb in colors:
        assert all(0.0 <= c <= 1.0 for c in rgb)


def test_lottie_generator_reproduces_committed_json(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["make_lottie.py", "--out", str(tmp_path)])
    import runpy  # noqa: E402

    runpy.run_path(str(SCRIPTS / "make_lottie.py"), run_name="__main__")
    regen = json.loads((tmp_path / "pipeline.json").read_text())
    assert regen == json.loads(LOTTIE.read_text())


def test_gif_fallback_is_an_animated_3_to_4_second_loop():
    assert GIF.is_file()
    assert GIF.stat().st_size < 1_000_000
    with Image.open(GIF) as img:
        assert getattr(img, "is_animated", False)
        n = img.n_frames
        per_frame_ms = img.info.get("duration", 0)
        assert 90 <= n <= 120
        assert per_frame_ms == 33  # 30 fps
        assert 2.7 <= n * per_frame_ms / 1000 <= 4.4
        assert 4.0 <= img.size[0] / img.size[1] <= 6.5  # hero banner aspect


def test_readme_embeds_gif_directly_under_hero_and_site_loads_lottie():
    readme = (ROOT / "README.md").read_text().splitlines()
    hero = next(i for i, line in enumerate(readme) if "assets/hero.svg" in line)
    gif = next(i for i, line in enumerate(readme) if "assets/pipeline.gif" in line)
    assert hero < gif <= hero + 4
    html = (ROOT / "site" / "index.html").read_text()
    assert "assets/lottie/pipeline.json" in html
