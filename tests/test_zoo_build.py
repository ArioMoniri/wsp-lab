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
    assert [x for x in lines if x.get("type") != "header"] == manifest
    assert sum(x.get("type") == "header" for x in lines) == 1

    before = manifest_path.read_text()
    assert build_zoo(cfg_path, dry_run=True, git_sha="testsha") == manifest
    assert manifest_path.read_text() == before  # idempotent: no duplicate rows or headers
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


# ---- family policy, manifest safety, provenance ------------------------------------------------


def _cfg(tmp_path, families, **extra):
    config = {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "base_revision": "main",
        "dtype": "float32",
        "out_dir": str(tmp_path / "zoo"),
        "seed": 0,
        "families": families,
        **extra,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(config))
    return p


RANDOM_DIR = {
    "n": 2,
    "layers": [0, 1],
    "tensors": ["mlp.down_proj"],
    "weight": {"min": 0.25, "max": 1.0},
    "n_directions": {"choices": [1]},
}
FAKE_FAMILIES = {
    "random_dir": RANDOM_DIR,
    "heretic_ara": {"n": 1, "preserve_good_behavior_weight": {"choices": [1]}},
    "lora_sft": {
        "n": 1,
        "datasets": ["x"],
        "rank": {"choices": [4]},
        "lr": {"min": 1e-4, "max": 2e-4},
        "steps": {"choices": [5]},
    },
    "task_vector": {"n": 3, "scale": {"min": 0.5, "max": 1.0}},
    "chain": {"n": 2, "length": {"min": 2, "max": 3}},
}


@pytest.fixture
def fake_base(monkeypatch):
    import torch

    import wsp.zoo.build as B

    g = torch.Generator().manual_seed(0)
    state = {
        f"model.layers.{i}.mlp.down_proj.weight": torch.randn(8, 8, generator=g) for i in range(2)
    }
    calls = []

    def load(config):
        calls.append(1)
        return {k: v.clone() for k, v in state.items()}

    monkeypatch.setattr(B, "_load_base_state", load)
    return calls


@pytest.fixture
def families_missing(monkeypatch):
    """Make the optional family modules unimportable (sys.modules[name] = None)."""
    import sys

    for fam in ("heretic_ara", "lora_sft"):
        monkeypatch.setitem(sys.modules, f"wsp.zoo.{fam}", None)


def test_missing_families_fail_unless_allowed_and_nothing_is_loaded(
    tmp_path, fake_base, families_missing
):
    from wsp.zoo.build import MissingFamiliesError

    cfg = _cfg(tmp_path, FAKE_FAMILIES)
    for dry in (True, False):
        with pytest.raises(MissingFamiliesError, match="heretic_ara"):
            build_zoo(cfg, dry_run=dry)
    assert not fake_base  # failed before loading the model
    assert not (tmp_path / "zoo" / "manifest.jsonl").exists()


def test_allowed_missing_families_are_recorded_in_header_and_every_chain_edit_json(
    tmp_path, fake_base, families_missing
):
    cfg = _cfg(tmp_path, FAKE_FAMILIES)
    build_zoo(cfg, allow_missing_families=True, git_sha="s")
    zoo = tmp_path / "zoo"
    lines = [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines()]
    (header,) = [x for x in lines if x.get("type") == "header"]
    expect = {
        "families_requested": ["random_dir", "heretic_ara", "lora_sft", "task_vector", "chain"],
        "families_available": ["random_dir", "task_vector", "chain"],
        "skipped_families": ["heretic_ara", "lora_sft"],
    }
    for k, v in expect.items():
        assert header[k] == v
    assert header["allow_missing_families"] is True and header["config_hash"]
    chains = [x for x in lines if x.get("family") == "chain"]
    assert chains
    for row in chains:
        edit = json.loads((zoo / row["id"] / "edit.json").read_text())
        for k, v in expect.items():
            assert edit["provenance"][k] == v
        assert edit["params"]["inner_family"] == "random_dir"  # only available single family


def test_dry_run_reports_same_policy_as_real_build(tmp_path, fake_base, families_missing):
    cfg = _cfg(tmp_path, FAKE_FAMILIES)
    build_zoo(cfg, dry_run=True, allow_missing_families=True)
    header = json.loads((tmp_path / "zoo" / "manifest.dryrun.jsonl").read_text().splitlines()[0])
    assert header["dry_run"] is True and header["skipped_families"] == ["heretic_ara", "lora_sft"]


def test_real_build_is_idempotent_and_repairs_polluted_manifest(tmp_path, fake_base):
    cfg = _cfg(tmp_path, {k: FAKE_FAMILIES[k] for k in ("random_dir", "task_vector", "chain")})
    zoo = tmp_path / "zoo"
    zoo.mkdir()
    ghost = {"id": "ghostghostghost0", "family": "random_dir", "parent_id": None, "seed": 9}
    (zoo / "manifest.jsonl").write_text((json.dumps(ghost) + "\n") * 2)  # dup + no artefacts
    rows = build_zoo(cfg, git_sha="s")
    assert "ghostghostghost0" not in {r["id"] for r in rows}
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert all(
        (zoo / i / "edit.json").exists() and (zoo / i / "delta.safetensors").exists() for i in ids
    )
    on_disk = (zoo / "manifest.jsonl").read_text()
    assert "ghostghostghost0" not in on_disk
    # task vectors only ever pick parents that have artefacts
    for r in rows:
        if r["family"] == "task_vector":
            assert (zoo / r["parent_id"] / "delta.safetensors").exists()
    rows2 = build_zoo(cfg, git_sha="s")
    assert [r["id"] for r in rows2] == ids  # same entries, no duplicates
    lines = (zoo / "manifest.jsonl").read_text().splitlines()
    rows_only = [json.loads(x) for x in lines if '"type"' not in x]
    assert len(rows_only) == len(ids)


def test_dry_run_never_pollutes_production_manifest(tmp_path, fake_base):
    cfg = _cfg(tmp_path, {"random_dir": RANDOM_DIR})
    build_zoo(cfg, dry_run=True)
    assert not (tmp_path / "zoo" / "manifest.jsonl").exists()
    build_zoo(cfg)
    prod = (tmp_path / "zoo" / "manifest.jsonl").read_text()
    build_zoo(cfg, dry_run=True)
    assert (tmp_path / "zoo" / "manifest.jsonl").read_text() == prod


def test_heretic_ara_params_are_labelled_dense_and_lora_sft_lora():
    from wsp.zoo.build import _with_lora_extras

    assert _with_lora_extras("heretic_ara", {}, {})["delta_format"] == "dense"
    assert _with_lora_extras("lora_sft", {}, {})["delta_format"] == "lora"
