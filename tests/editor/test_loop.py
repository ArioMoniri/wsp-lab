import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from wsp.editor.api import EditorAPI
from wsp.editor.driver import run_episode
from wsp.editor.episode import Episode
from wsp.editor.providers import ScriptedEditor, provider
from wsp.editor.sandbox import run


class TinyCausal(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        h = torch.nn.functional.one_hot(input_ids % 4, 4).float()
        return SimpleNamespace(logits=h, hidden_states=(h,))


def test_hidden_prefill_decode_meter():
    api = EditorAPI({"w": torch.ones(2, 2)}, TinyCausal(), 25, hidden_layers=(0,))
    assert "hidden_states" in api.forward([[1, 2]] * 3, hidden_layers=(0,))
    api.forward([[2]] * 2)
    api.forward([[3]])
    assert api.used == 6


@pytest.mark.parametrize("limit", [10, 25, 50])
def test_actual_budget_counts(limit):
    api = EditorAPI({"w": torch.ones(2, 2)}, TinyCausal(), limit)
    for _ in range(limit):
        api.forward([[1]])
    with pytest.raises(ValueError):
        api.forward([[1]])
    assert api.used == limit


def test_subprocess_isolation_timeout_and_protocol():
    api = EditorAPI({"w": torch.ones(2, 2)}, TinyCausal(), 10)
    assert run("call('probe.forward', input_ids=[[1]])", api).valid
    for code in [
        "open('/etc/passwd').read()",
        "import socket; socket.socket()",
        "import os; assert os.system('true') == 0",
        "while True: pass",
    ]:
        assert not run(code, api, wall_seconds=0.2).valid
    assert not run("import os; assert 'HF_TOKEN' in os.environ", api).valid
    assert not run("call('state_dict')", api).valid


def test_loop_replay_manifest_d1_and_failures(tmp_path):
    from wsp.eval.run import apply_delta
    from wsp.schema import EditDescriptor

    code = "call('delta.set', name='weight', tensor=[[0.02,0],[0,0.03]], rationale='test')"
    config = {"base_model": "tiny", "base_revision": "a" * 40, "budget": 10, "seed": 0}
    ids = []
    for episode_id in ["one", "two"]:
        episode = Episode(tmp_path / "runs", episode_id)
        result = run_episode(
            ScriptedEditor([code, "raise ValueError()"]),
            config,
            {"weight": torch.ones(2, 2).bfloat16()},
            TinyCausal(),
            [],
            episode,
            tmp_path / "zoo",
            proposals=2,
        )
        assert result[0] and result[1] is None
        ids.append(result[0])
    assert ids[0] == ids[1]
    entry = tmp_path / "zoo" / ids[0]
    desc = EditDescriptor.from_json((entry / "edit.json").read_text())
    assert desc.parent_id is None and desc.family == "llm_edit"
    model = torch.nn.Linear(2, 2, bias=False).bfloat16()
    model.weight.data.fill_(1)
    delta = load_file(entry / "delta.safetensors")
    apply_delta(model, delta)
    assert torch.equal(model.weight.float() - 1, delta["weight"])
    events = [json.loads(x) for x in (tmp_path / "runs/one/episode.jsonl").read_text().splitlines()]
    assert any(x["event"] == "failed" for x in events)
    with pytest.raises(ValueError):
        provider("paid-unregistered")


def test_training_pool_and_gemma_config():
    import yaml

    from wsp.editor.driver import validate_pool

    with pytest.raises(ValueError):
        validate_pool([{"id": "secret", "split": "test", "prompt": "sealed"}])
    for model in ["gemma-3-270m", "qwen3.5-0.8b"]:
        with open(f"configs/editor_{model}.yaml") as stream:
            cfg = yaml.safe_load(stream)
        assert cfg["budgets"] == [10, 25, 50]
        assert cfg["editable_tensors"]


def test_distinct_proposals_and_sealed_absent(tmp_path):
    episode = Episode(tmp_path / "runs", "distinct")
    codes = [f"call('delta.set', name='w', tensor=[[{v}]], rationale='test')" for v in (0.1, 0.2)]
    config = {
        "base_model": "tiny",
        "base_revision": "a" * 40,
        "seed": 0,
        "budget": 10,
        "sealed_reference": "sealed-marker-do-not-disclose",
        "editable_tensors": ["w"],
    }
    ids = run_episode(
        ScriptedEditor(codes),
        config,
        {"w": torch.ones(1, 1)},
        TinyCausal(),
        [{"id": "train-1", "source": "gsm8k", "split": "train", "prompt": "1+1?"}],
        episode,
        tmp_path / "zoo",
        proposals=2,
    )
    assert ids[0] and ids[1] and ids[0] != ids[1]
    assert "sealed-marker" not in episode.path.read_text()


def test_partial_ipc_and_resource_exits():
    api = EditorAPI({"w": torch.ones(1)}, TinyCausal(), 10)
    result = run(
        "import os; os.write(1, b'partial'); exec('while True: pass')", api, wall_seconds=0.2
    )
    assert not result.valid and "TimeoutError" in result.error
    assert not run("x = bytearray(1024**3)", api).valid
    assert not run("call('weights.read', name='sealed')", api).valid


def test_build_registry_replays_scripted_candidate(tmp_path):
    from wsp.zoo.build import _build_single_family

    cfg = {"base_model": "tiny", "base_revision": "a" * 40}
    family = {
        "provider": "scripted",
        "provider_options": {
            "cells": ["call('delta.set', name='w', tensor=[[0.1]], rationale='test')"]
        },
        "editor_config": {"budget": 10},
        "episode_root": str(tmp_path / "episodes"),
    }
    result = _build_single_family(
        "llm_edit", cfg, family, 0, {"w": torch.ones(1, 1).bfloat16()}, tmp_path, [], False
    )
    assert result[0][0].family == "llm_edit"
    assert result[0][0].parent_id is None
