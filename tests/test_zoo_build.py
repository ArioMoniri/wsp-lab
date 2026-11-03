"""Tests for the edit-zoo builder (CPU)."""

from __future__ import annotations

import json

import pytest
import yaml

from wsp.zoo.build import build_zoo, sample_param


def test_sample_param_uniform():
    rng = __import__("random").Random(0)
    spec = {"min": 0.0, "max": 1.0}
    vals = [sample_param(rng, spec) for _ in range(20)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    # Deterministic: a fresh RNG with the same seed produces the same sequence.
    rng2 = __import__("random").Random(0)
    assert vals == [sample_param(rng2, spec) for _ in range(20)]


def test_sample_param_choices():
    rng = __import__("random").Random(1)
    spec = {"choices": ["a", "b", "c"]}
    vals = [sample_param(rng, spec) for _ in range(30)]
    assert set(vals) <= {"a", "b", "c"}


def test_sample_param_log():
    rng = __import__("random").Random(2)
    spec = {"log_min": 1e-5, "log_max": 1e-3}
    vals = [sample_param(rng, spec) for _ in range(20)]
    assert all(1e-5 <= v <= 1e-3 for v in vals)


def test_dry_run_writes_manifest_without_model(tmp_path):
    """Dry run must produce manifest.jsonl and never load a model or write deltas."""
    zoo_dir = tmp_path / "zoo"
    config = {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "base_revision": "main",
        "dtype": "bfloat16",
        "out_dir": str(zoo_dir),
        "seed": 0,
        "families": {
            "random_dir": {
                "n": 3,
                "layers": [0, 1],
                "tensors": ["mlp.down_proj"],
                "weight": {"min": 0.25, "max": 1.0},
                "n_directions": {"choices": [1]},
            }
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    manifest = build_zoo(cfg_path, dry_run=True, git_sha="testsha")

    assert len(manifest) == 3
    assert all(m["family"] == "random_dir" for m in manifest)
    assert all(m["git_sha"] == "testsha" for m in manifest)
    assert all("id" in m and "seed" in m and "config_hash" in m for m in manifest)

    manifest_path = zoo_dir / "manifest.dryrun.jsonl"
    assert manifest_path.exists()
    lines = [json.loads(line) for line in manifest_path.read_text().strip().split("\n")]
    assert len(lines) == 3
    assert lines == manifest

    assert build_zoo(cfg_path, dry_run=True, git_sha="testsha") == manifest
    assert not (zoo_dir / "manifest.jsonl").exists()

    # No per-entry artefacts and no delta files in dry run
    assert not list(zoo_dir.glob("*/edit.json"))
    assert not list(zoo_dir.glob("*/delta.safetensors"))


def test_build_unknown_family_raises(tmp_path):
    cfg = {
        "base_model": "x",
        "out_dir": str(tmp_path / "zoo"),
        "seed": 0,
        "families": {"magic": {"n": 1}},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unknown edit family"):
        build_zoo(p, dry_run=True)


def test_sample_param_accepts_component_list():
    import random

    assert sample_param(random.Random(0), ["attn", "mlp", "both"]) in {"attn", "mlp", "both"}
