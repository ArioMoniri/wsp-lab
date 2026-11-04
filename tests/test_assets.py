"""Contracts for the portable hero and matching Pages animation."""

import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOTTIE = ROOT / "assets/lottie/pipeline.json"


def test_svg_is_self_contained_readable_and_animated():
    root = ET.parse(ROOT / "assets/hero.svg").getroot()
    assert root.attrib["viewBox"] == "0 0 1200 320"
    ns = {"s": "http://www.w3.org/2000/svg"}
    labels = " ".join(root.itertext())
    for label in (
        "Qwen3.5-0.8B",
        "abliterate",
        "LoRA",
        "merge",
        "canonical SVD",
        "State-space",
        "MATH",
        "GSM8K",
        "IFEval",
        "refusal",
        "Schematic",
    ):
        assert label in labels
    assert root.find("s:title", ns) is not None
    assert root.find("s:desc", ns) is not None
    assert len(root.findall(".//s:animate", ns)) >= 3
    assert all(el.attrib.get("dur") == "4s" for el in root.findall(".//s:animate", ns))
    for el in root.iter():
        assert el.tag.split("}")[-1] not in ("image", "script", "foreignObject")
        assert not any("href" in k for k in el.attrib)


def test_lottie_is_portable_four_second_bodymovin():
    anim = json.loads(LOTTIE.read_text())
    assert LOTTIE.stat().st_size <= 150_000
    assert (anim["w"], anim["h"], anim["fr"], anim["ip"], anim["op"]) == (1200, 320, 30, 0, 120)
    assert anim["assets"] and all("layers" in a and "p" not in a for a in anim["assets"])
    assets = {a["id"] for a in anim["assets"]}
    assert len(assets) == len(anim["assets"])
    for composition in anim["assets"]:
        for child in composition["layers"]:
            if child["ty"] == 0:
                assert child["refId"] in assets
                assert child["w"] > 0 and child["h"] > 0
                assert "r" in child["ks"]  # required by the Pages WASM renderer
    assert "fonts" not in anim  # labels are vector outlines for the canvas/WASM renderer
    names = {layer["nm"] for layer in anim["layers"]}
    for name in (
        "base-model",
        "delta-token-bars",
        "ssm-block",
        "forecast-curve",
        "forecast-band",
        "ssm-state-flow",
        "label-Qwen3.5-0.8B",
        "label-MATH · GSM8K",
        "label-IFEval · refusal",
    ):
        assert name in names
    assert sum(name.startswith("edit-arrow-") and not name.endswith("-head") for name in names) == 3
    for layer in anim["layers"]:
        assert layer["ty"] in (0, 4)
        assert layer["ip"] == 0 and layer["op"] == 120
        assert layer["ks"]["o"] == {"a": 0, "k": 100}  # labels never disappear
        if layer["ty"] == 4:
            assert all(group["it"][-1]["ty"] == "tr" for group in layer["shapes"])
        else:
            assert layer["refId"] in assets
    state = next(l for l in anim["layers"] if l["nm"] == "ssm-state-flow")
    assert state["ks"]["p"]["a"] == 1
    curve = next(l for l in anim["layers"] if l["nm"] == "forecast-curve")
    assert any(s["ty"] == "tm" and s["e"]["a"] == 1 for g in curve["shapes"] for s in g["it"])


def test_generator_reproduces_both_assets(tmp_path):
    import pytest

    with pytest.raises(SystemExit) as exc:
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["make_lottie.py", "--out", str(tmp_path)]):
            runpy.run_path(str(ROOT / "scripts/make_lottie.py"), run_name="__main__")
    assert exc.value.code == 0
    assert (tmp_path / "pipeline.json").read_bytes() == LOTTIE.read_bytes()
    assert (tmp_path / "hero.svg").read_bytes() == (ROOT / "assets/hero.svg").read_bytes()


def test_readme_uses_only_svg_and_site_keeps_lottie():
    assert not (ROOT / "assets/pipeline.gif").exists()
    readme = (ROOT / "README.md").read_text()
    assert "assets/hero.svg" in readme and "assets/pipeline.gif" not in readme
    assert "assets/lottie/pipeline.json" in (ROOT / "site/index.html").read_text()
