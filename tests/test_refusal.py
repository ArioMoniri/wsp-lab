import pytest
import torch

from wsp.edits.kernels import heretic_abliteration_kernel
from wsp.edits.refusal import interpolate_direction


def test_interpolation_and_full_ablation():
    dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    d = interpolate_direction(dirs, -0.5)
    assert torch.allclose(d, torch.tensor([2**-0.5, 2**-0.5]))
    w = torch.randn(2, 4)
    result, _ = heretic_abliteration_kernel({"w": w}, dirs, {0: ["w"]}, -0.5, 1.0, 0.0, 0.0, 1.0)
    assert torch.allclose(d @ result["w"], torch.zeros(4), atol=1e-6)


@pytest.mark.gpu
def test_extraction_and_abliteration_roundtrip(cuda, tmp_path):
    from pathlib import Path

    from wsp.edits.refusal import extract_directions
    from wsp.zoo.apply import load_edited_model
    from wsp.zoo.build import _load_base_state, _write_entry, build_abliteration, load_config

    config = load_config(Path("configs/zoo_qwen3.5-0.8b.yaml"))
    config["refusal_n"] = 8
    directions = extract_directions(config, tmp_path)
    assert directions.ndim == 2 and torch.isfinite(directions).all()
    base = _load_base_state(config)
    desc, state, touched = build_abliteration(
        config, config["families"]["abliteration"], 0, base, tmp_path
    )[0]
    delta = {k: state[k] - base[k] for k in touched}
    _write_entry(tmp_path, desc, delta, torch.bfloat16)
    restored = load_edited_model(base, tmp_path / desc.id)
    assert all(torch.equal(restored[k], base[k] + delta[k]) for k in touched)
