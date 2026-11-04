import json
import os
import time
from types import SimpleNamespace

import pytest
import torch

from wsp.editor.api import EditorAPI
from wsp.editor.driver import run_episode, validate_config
from wsp.editor.episode import Episode, digest
from wsp.editor.providers import REGISTRY, ScriptedEditor, provider
from wsp.editor.sandbox import run


class DictModel:
    config = SimpleNamespace(vocab_size=8, hidden_size=4)

    def __call__(self, **kw):
        b, s = kw["input_ids"].shape
        return {
            "logits": torch.arange(b * s * 8).reshape(b, s, 8).float(),
            "hidden_states": (torch.ones(b, s, 4), torch.ones(b, s, 4) * 2),
        }


def config(**kw):
    cfg = {
        "base_model": "tiny",
        "base_revision": "a" * 40,
        "budget": 10,
        "seed": 0,
        "editable_tensors": ["w"],
        "development_pool_hash": digest([]),
    }
    cfg.update(kw)
    return cfg


def test_modeloutput_reduced_positions_layers_and_topk():
    api = EditorAPI({"w": torch.ones(2, 2)}, DictModel(), 10, hidden_layers=(1,))
    out = api.forward([[1, 2, 3]], hidden_layers=[1], top_k=2)
    assert out["type"] == "probe.v1"
    assert out["positions"] == [2]
    assert len(out["token_ids"][0][0]) == 2
    assert out["hidden_states"]["1"]["mean"] == [[2.0]]
    assert "values" not in out["hidden_states"]["1"]
    assert api.forward([[1, 2]])["logits"] == [[[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]]


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 65}, {"positions": [0, 1, 2]}, {"hidden_layers": [0, 1]}, {"positions": [99]}],
)
def test_probe_caps_prelaunch(kwargs):
    api = EditorAPI(
        {"w": torch.ones(1)}, DictModel(), 10, hidden_layers=(0, 1), max_positions=2, max_layers=1
    )
    with pytest.raises(ValueError):
        api.forward([[1, 2, 3]], **kwargs)
    assert api.used == 0


def test_byte_sequence_caps_and_unlaunched_free():
    api = EditorAPI({"w": torch.ones(100)}, DictModel(), 10, max_bytes=100, max_sequence=2)
    for kwargs in [{"input_ids": [[1, 2, 3]]}, {"input_ids": [[1]]}]:
        with pytest.raises(ValueError):
            api.forward(**kwargs)
    with pytest.raises(ValueError):
        api.dispatch({"version": "editor_api.v1", "op": "weights.read", "args": {"name": "w"}})
    assert api.used == 0
    api = EditorAPI({"w": torch.ones(1)}, None, 10)
    with pytest.raises(ValueError):
        api.forward([[1]])
    assert api.used == 0


def test_add_exact_shape():
    api = EditorAPI({"w": torch.ones(2, 2)}, None, 10)
    with pytest.raises(ValueError):
        api.add("w", [[0.1, 0.1]], "reason")


def test_config_allowlist_revision_and_pool_pin():
    for change in [
        {"editable_tensors": None},
        {"editable_tensors": ["bad"]},
        {"base_revision": "main"},
        {"development_pool_hash": "0" * 64},
    ]:
        cfg = config()
        cfg.update(change)
        with pytest.raises(ValueError):
            validate_config(cfg, {"w": torch.ones(1)}, [])
    validate_config(config(development_pool_hash=digest([])), {"w": torch.ones(1)}, [])


def test_production_provider_gate():
    REGISTRY["paid"] = ScriptedEditor
    try:
        for cfg in [
            config(),
            config(production_enabled=True, development_pool_hash=None),
            config(
                production_enabled=True,
                development_pool_hash=digest([]),
                require_separate_principal=True,
            ),
        ]:
            with pytest.raises(ValueError):
                provider("paid", config=cfg, cells=[])
        assert provider(
            "paid",
            config=config(production_enabled=True, development_pool_hash=digest([])),
            cells=[],
        )
    finally:
        del REGISTRY["paid"]


def test_stat_oracle_and_stdout_not_ipc(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.write_text("secret")
    code = f"""
import os
errors = []
for path in [{str(sealed)!r}, {str(sealed) + "-missing"!r}]:
    try: os.stat(path); errors.append('reachable')
    except OSError as e: errors.append(e.errno)
assert errors == [1,1]
try: os.write(1, b'{{"op":"injected"}}\\n')
except OSError: pass
call('budget')
"""
    result = run(code, EditorAPI({"w": torch.ones(1)}, None, 10))
    assert result.valid, result.error
    assert len(result.transcript) == 1


def test_logged_replay_delta_meter_and_array_hashes(tmp_path):
    code = "call('probe.forward', input_ids=[[1,2]], top_k=2); call('delta.set', name='w', tensor=[[0.1]], rationale='test')"
    rows = []
    ids = []
    for name in ["first", "replay"]:
        ep = Episode(tmp_path, name)
        ids.append(
            run_episode(
                ScriptedEditor([code]),
                config(),
                {"w": torch.ones(1, 1)},
                DictModel(),
                [],
                ep,
                tmp_path / name / "zoo",
            )[0]
        )
        events = [json.loads(s) for s in ep.path.read_text().splitlines()]
        rows.append(next(e for e in events if e["event"] == "sandbox"))
        code = next(e["code"] for e in events if e["event"] == "completion")
    assert ids[0] and ids[0] == ids[1]
    from hashlib import sha256

    assert (
        sha256((tmp_path / "first" / "zoo" / ids[0] / "delta.safetensors").read_bytes()).hexdigest()
        == sha256(
            (tmp_path / "replay" / "zoo" / ids[1] / "delta.safetensors").read_bytes()
        ).hexdigest()
    )
    assert rows[0]["budget"] == rows[1]["budget"]
    assert {k: v for k, v in rows[0]["meter"].items() if k != "wall_seconds"} == {
        k: v for k, v in rows[1]["meter"].items() if k != "wall_seconds"
    }
    response = rows[0]["transcript"][0]["response"]["result"]
    assert set(response["token_ids"]) == {"sha256", "shape"}
    assert rows[0]["meter"]["tokens"] == rows[1]["meter"]["tokens"] == 2


def assert_sealed_absent(text, markers):
    assert all(marker not in text for marker in markers)


def test_sealed_store_unreachable_and_detector_catches_injection(tmp_path):
    sealed = tmp_path / "sealed-items.json"
    sealed.write_text(json.dumps({"id": "sealed-item-719", "content": "SEALED_CONTENT_719"}))
    markers = ["SEALED_CONTENT_719", "sealed-item-719"]
    ep = Episode(tmp_path, "safe")
    code = f"""
try:
    open({str(sealed)!r}).read()
    raise AssertionError('reachable')
except PermissionError: pass
call('delta.set', name='w', tensor=[[0.1]], rationale='test')
"""
    assert run_episode(
        ScriptedEditor([code]),
        config(),
        {"w": torch.ones(1, 1)},
        DictModel(),
        [],
        ep,
        tmp_path / "zoo",
    )[0]
    assert_sealed_absent(ep.path.read_text(), markers)
    for marker in markers + [str(sealed)]:
        with pytest.raises(AssertionError):
            assert_sealed_absent(ep.path.read_text() + marker, [marker])
    # Positive control: the same code fails when an ordinary accessible store is injected.
    with pytest.raises(AssertionError, match="reachable"):
        exec(code, {"call": lambda *a, **k: None})  # noqa: S102 - positive leakage control


class HungModel(DictModel):
    def __call__(self, **kw):
        time.sleep(5)


def test_probe_hard_timeout_kills_worker():
    api = EditorAPI({"w": torch.ones(1)}, HungModel(), 10, call_seconds=0.1)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        api.forward([[1]])
    assert time.monotonic() - start < 1
    assert api.used == 1


def test_episode_meter_across_proposals(tmp_path):
    ep = Episode(tmp_path, "meter")
    code = "call('probe.forward', input_ids=[[1,2]]); call('delta.set', name='w', tensor=[[0.1]], rationale='test')"
    cfg = config(episode_limits={"calls": 1, "sequences": 10, "tokens": 2, "wall_seconds": 5})
    ids = run_episode(
        ScriptedEditor([code, code]),
        cfg,
        {"w": torch.ones(1, 1)},
        DictModel(),
        [],
        ep,
        tmp_path / "zoo",
        proposals=2,
    )
    assert ids[0] and ids[1] is None
    events = [json.loads(s) for s in ep.path.read_text().splitlines()]
    assert [e["meter"]["calls"] for e in events if e["event"] == "sandbox"] == [1, 1]


@pytest.mark.skipif(os.geteuid() != 0, reason="setuid requires root")
def test_separate_principal_uid():
    api = EditorAPI({"w": torch.ones(1)}, None, 10)
    assert run(
        "import os; assert os.getuid()==65534", api, uid=65534, require_separate_principal=True
    ).valid


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_checkpoint_dtype_effective_delta(tmp_path, dtype):
    from safetensors.torch import load_file

    from wsp.editor.zoo import register
    from wsp.features.extract import effective_delta

    base = torch.tensor([[1.003]], dtype=dtype)
    api = EditorAPI({"w": base}, None, 10)
    api.set("w", [[0.002]], "test")
    provenance = {"code_hash": "x", "provider": "scripted", "model": "tiny", "version": "1"}
    entry = register(api, config(), provenance, tmp_path)
    assert torch.equal(
        load_file(tmp_path / entry / "delta.safetensors")["w"],
        effective_delta(base, api.deltas["w"])[0],
    )


@pytest.mark.parametrize(
    "limits",
    [
        {"calls": 0},
        {"sequences": 0},
        {"tokens": 1},
        {"wall_seconds": 0},
    ],
)
def test_episode_each_limit_prevents_launch(limits):
    from wsp.editor.api import EpisodeMeter

    meter = EpisodeMeter(**{"max_" + k: v for k, v in limits.items()})
    api = EditorAPI({"w": torch.ones(1)}, DictModel(), 10, meter=meter)
    with pytest.raises(ValueError):
        api.forward([[1, 2]])
    assert api.used == meter.calls == meter.sequences == meter.tokens == 0


def test_separate_principal_refuses_unavailable():
    from wsp.editor.driver import validate_production

    cfg = config(
        production_enabled=True,
        development_pool_hash=digest([]),
        require_separate_principal=True,
        sandbox={"uid": os.getuid()},
    )
    with pytest.raises(ValueError):
        validate_production(cfg)
    with pytest.raises(ValueError):
        run("pass", EditorAPI({"w": torch.ones(1)}, None, 10), require_separate_principal=True)


def test_load_config_rejects_before_provider_and_arrays_scrubbed(tmp_path):
    from wsp.editor.driver import load_config

    path = tmp_path / "config.yaml"
    path.write_text("base_revision: main\n")
    with pytest.raises(ValueError):
        load_config(path, {"w": torch.ones(1)}, [])
    ep = Episode(tmp_path, "arrays")
    ep.append("test", payload=[[1.0, 2.0], [3.0, 4.0]])
    payload = json.loads(ep.path.read_text())["payload"]
    assert payload == {"sha256": digest([[1.0, 2.0], [3.0, 4.0]]), "shape": [2, 2]}


def test_episode_deadline_kills_probe_and_launch_failure_free(monkeypatch):
    import multiprocessing

    from wsp.editor.api import EpisodeMeter

    api = EditorAPI({"w": torch.ones(1)}, HungModel(), 10, meter=EpisodeMeter(max_wall_seconds=0.1))
    with pytest.raises(TimeoutError):
        api.forward([[1]])
    assert not multiprocessing.active_children()
    api = EditorAPI({"w": torch.ones(1)}, DictModel(), 10)

    def fail_start(self):
        raise OSError("cannot launch")

    monkeypatch.setattr(multiprocessing.context.ForkServerProcess, "start", fail_start)
    with pytest.raises(OSError):
        api.forward([[1]])
    assert api.used == api.meter.calls == api.meter.tokens == 0
