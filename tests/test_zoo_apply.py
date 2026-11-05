"""Tests for applying zoo deltas to a base model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from wsp.schema import EditDescriptor
from wsp.zoo.apply import load_edited_model


def _write_entry(
    tmp_path: Path, base: dict[str, torch.Tensor], delta: dict[str, torch.Tensor]
) -> Path:
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir(parents=True)
    desc = EditDescriptor(
        family="random_dir",
        params={"layer": 0, "weight": 1.0},
        seed=0,
        base_model="toy",
        touched_tensors=list(delta.keys()),
        delta_fro_norm=float(sum(t.norm().item() ** 2 for t in delta.values()) ** 0.5),
    )
    (entry_dir / "edit.json").write_text(desc.to_json())
    save_file({k: v.contiguous() for k, v in delta.items()}, entry_dir / "delta.safetensors")
    return entry_dir


def test_load_edited_model_from_state_dict_cpu(tmp_path, toy_weights):
    delta = {k: v * 0.1 for k, v in toy_weights.items()}
    entry_dir = _write_entry(tmp_path, toy_weights, delta)
    out = load_edited_model(toy_weights, entry_dir)
    for k in toy_weights:
        assert torch.allclose(out[k], toy_weights[k] + delta[k])


@pytest.mark.gpu
@pytest.mark.slow
def test_load_edited_model_bf16_roundtrip_exact_on_cuda(cuda, tmp_path):
    """Applying a bf16 delta to a bf16 base state must be bit-exact."""
    base = {
        "a": torch.randn(64, 64, dtype=torch.bfloat16, device=cuda),
        "b": torch.randn(32, 32, dtype=torch.bfloat16, device=cuda),
    }
    delta = {k: torch.randn_like(v) * 0.01 for k, v in base.items()}
    # Keep delta as bf16 on cpu to mimic storage
    delta_cpu = {k: v.cpu() for k, v in delta.items()}
    entry_dir = _write_entry(tmp_path, base, delta_cpu)

    reconstructed = load_edited_model(base, entry_dir)
    expected = {k: base[k] + delta[k] for k in base}
    for k in base:
        assert reconstructed[k].dtype == torch.bfloat16
        assert reconstructed[k].device.type == cuda.type
        assert torch.equal(reconstructed[k], expected[k])


def test_wrapped_name_dense_delta_applies_like_text_name_delta_and_leaves_file_untouched(tmp_path):
    base = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 6),
        "model.layers.1.mlp.down_proj.weight": torch.randn(4, 6),
        "lm_head.weight": torch.randn(5, 4),
    }
    delta = {"model.layers.1.mlp.down_proj.weight": torch.randn(4, 6) * 0.1}
    saved = {k.replace("model.", "model.language_model.", 1): v for k, v in delta.items()}
    entry = _write_entry(tmp_path, base, saved)
    before = (entry / "delta.safetensors").read_bytes()
    out = load_edited_model(base, entry)
    k = "model.layers.1.mlp.down_proj.weight"
    assert torch.allclose(out[k], base[k] + delta[k])
    assert set(out) == set(base)  # no stray wrapped-name tensor was added
    assert (entry / "delta.safetensors").read_bytes() == before  # D1: artefact immutable


def test_load_entry_delta_reports_the_mapping_applied(tmp_path):
    from wsp.zoo.apply import load_entry_delta

    key = "model.layers.0.mlp.down_proj.weight"
    base = {key: torch.zeros(2, 2)}
    wrapped = _write_entry(
        tmp_path / "w",
        base,
        {"model.language_model.layers.0.mlp.down_proj.weight": torch.ones(2, 2)},
    )
    plain = _write_entry(tmp_path / "p", base, {key: torch.ones(2, 2)})
    assert load_entry_delta(wrapped, base).key_mapping == "language_model_prefix_v1"
    assert load_entry_delta(plain, base).key_mapping is None
    with pytest.raises(KeyError, match="not in the base state"):
        load_entry_delta(wrapped, {"other.weight": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="shape"):
        load_entry_delta(wrapped, {key: torch.zeros(3, 3)})
