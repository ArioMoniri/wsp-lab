"""Unit tests for scripts/d8_chain_replay.py (graph handling and the sampler patch)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "d8_chain_replay.py"
spec = importlib.util.spec_from_file_location("d8_chain_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
sys.modules["d8_chain_replay"] = replay
spec.loader.exec_module(replay)  # type: ignore[union-attr]

from wsp.zoo import build


def _edit(i, step, parent, fam="random_dir"):
    return {
        "id": i,
        "parent_id": parent,
        "params": {"step": step, "inner_family": fam, "length": 3},
    }


def test_chain_groups_orders_by_step_and_checks_links():
    edits = {
        "r": _edit("r", 0, None),
        "b": _edit("b", 2, "a"),
        "a": _edit("a", 1, "r"),
        "s": _edit("s", 0, None),
    }
    groups = replay.chain_groups(edits)
    assert groups == {"r": ["r", "a", "b"], "s": ["s"]}


def test_chain_groups_rejects_gaps_and_bad_parents():
    with pytest.raises(ValueError):
        replay.chain_groups({"r": _edit("r", 0, None), "b": _edit("b", 2, "r")})
    with pytest.raises(ValueError):
        replay.chain_groups(
            {"r": _edit("r", 0, None), "a": _edit("a", 1, "r"), "b": _edit("b", 2, "r")}
        )
    with pytest.raises(KeyError):
        replay.chain_groups({"a": _edit("a", 1, "missing")})


def test_chain_groups_allows_branching_roots():
    """A root re-built at another commit shares its root: two step-1 children are a tree, not an error."""
    edits = {
        "r": _edit("r", 0, None),
        "a": _edit("a", 1, "r"),
        "b": _edit("b", 1, "r"),
        "c": _edit("c", 2, "b"),
    }
    groups = replay.chain_groups(edits)
    assert groups == {"r": ["r", "a", "b", "c"]}


def test_storage_of_prefers_own_bytes_then_alias():
    sidecar = {
        "x": {"delta_present": True, "alias_of": None},
        "y": {"delta_present": False, "alias_of": "x"},
        "z": {"delta_present": False, "alias_of": None},
    }
    zoo = Path("/zoo")
    assert replay.storage_of("x", sidecar, zoo) == zoo / "x"
    assert replay.storage_of("y", sidecar, zoo) == zoo / "x"
    assert replay.storage_of("z", sidecar, zoo) is None


def test_select_chains():
    groups = {"r": ["r", "a"], "s": ["s"]}
    sidecar = {
        "r": {"delta_present": True, "alias_of": None},
        "a": {"delta_present": False, "alias_of": None},
        "s": {"delta_present": True, "alias_of": None},
    }
    assert replay.select_chains(groups, sidecar, "verify", None) == ["s"]
    assert replay.select_chains(groups, sidecar, "restore", None) == ["r"]
    assert replay.select_chains(groups, sidecar, "restore", {"a"}) == ["r"]


def test_recorded_sampler_reproduces_build_random_dir():
    """Replaying the recorded parameters gives the same state as the builder's own sampled build."""
    config = {
        "base_model": "x",
        "base_revision": "main",
        "dtype": "bfloat16",
        "seed": 0,
        "families": {
            "random_dir": {
                "layers": [8],
                "tensors": ["mlp.down_proj", "self_attn.o_proj"],
                "weight": {"min": 0.25, "max": 1.0},
                "n_directions": {"choices": [1, 2]},
            }
        },
        "_arch": None,
        "_overrides": [],
        "_include_smoke": False,
    }
    g = torch.Generator().manual_seed(1)
    base = {
        "model.layers.8.mlp.down_proj.weight": torch.randn(16, 8, generator=g).to(torch.bfloat16),
        "model.layers.8.self_attn.o_proj.weight": torch.randn(16, 8, generator=g).to(
            torch.bfloat16
        ),
        "model.layers.9.mlp.down_proj.weight": torch.randn(16, 8, generator=g).to(torch.bfloat16),
    }
    seed = 7
    ((desc, want, touched),) = build.build_random_dir(
        config, config["families"]["random_dir"], seed, base
    )
    got, got_touched, _ = replay.replay_step(
        "random_dir", desc.params, seed, config, base, Path("/tmp")
    )
    assert sorted(got_touched) == sorted(touched)
    for k in want:
        assert torch.equal(got[k], want[k]), k
    # the patch is undone
    assert build.sample_random_dir_params is not replay.RecordedSampler
    d = replay.chain_delta(got, base, got_touched, torch.bfloat16)
    assert set(d) == set(touched)
    assert replay.compare(d, d)["bitwise"]
