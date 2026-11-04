import json
import threading
import time
import tracemalloc

import pytest
import torch

from wsp.editor.api import EditorAPI
from wsp.editor.sandbox import run


@pytest.mark.parametrize("mode", ["write", "sleep"])
def test_wall_monitor_kills_blocked_worker(monkeypatch, mode):
    from wsp.editor import sandbox

    original = sandbox.subprocess.Popen
    timers = []

    def launch(*args, **kwargs):
        proc = original(*args, **kwargs)
        timer = threading.Timer(2, lambda: proc.poll() is None and proc.kill())
        timer.start()
        timers.append(timer)
        return proc

    monkeypatch.setattr(sandbox.subprocess, "Popen", launch)
    code = "import _thread; _thread.allocate_lock().acquire(); lock=_thread.allocate_lock(); lock.acquire(); lock.acquire()"
    if mode == "write":
        code = (
            "import json; transport=__import__('sys')._getframe().f_back.f_globals['transport_out']; transport.write(json.dumps(dict(version='editor_api.v1',op='weights.read',args=dict(name='w')))+'\\n'); transport.flush(); "
            + code
        )
    if mode == "sleep":
        monkeypatch.setattr(sandbox, "WORKER", "import time; time.sleep(5)\n" + sandbox.WORKER)
    start = time.monotonic()
    try:
        result = run(code, EditorAPI({"w": torch.ones(25000)}, None, 10), wall_seconds=0.3)
        assert not result.valid
        assert time.monotonic() - start < 1.3
        assert "wall time" in result.error
        if mode == "write":
            assert result.transcript[0]["request"]["op"] == "weights.read"
            assert result.transcript[0]["response"]["result"]["shape"] == [25000]
    finally:
        for timer in timers:
            timer.cancel()


def test_weight_cap_before_index_allocation():
    api = EditorAPI({"w": torch.ones(1024, 3584)}, None, 10, max_bytes=1024**2)
    indices = [0] * 100000
    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
        ) as prof,
        pytest.raises(ValueError, match="byte cap"),
    ):
        api.read("w", indices)
    assert sum(max(0, e.cpu_memory_usage) for e in prof.events()) == 0


def test_weight_storage_slice_allocation():
    api = EditorAPI({"w": torch.ones(1024, 3584)}, None, 10)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as prof:
        result = api.read("w", __import__("builtins").slice(0, 21))
    tracemalloc.start()
    api.read("w", __import__("builtins").slice(0, 21))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.numel() * 4 == 301056
    assert max(e.cpu_memory_usage for e in prof.events()) <= 301056
    assert peak < 1024**2


class ParallelModel:
    def __call__(self, input_ids, **kwargs):
        a = torch.ones(512, 512)
        value = (a @ a)[0, :8]
        return {"logits": value.expand(*input_ids.shape, 8)}


def test_three_probes_after_parent_parallel_torch():
    torch.set_num_threads(4)
    a = torch.ones(512, 512)
    a @ a
    api = EditorAPI({"w": torch.ones(1)}, ParallelModel(), 10, max_vocab=8, call_seconds=15)
    for _ in range(3):
        assert api.forward([[1]])["logits"][0][0] == [512.0] * 8
        a @ a
    assert api.used == 3


def test_done_requires_eof_and_strings_bounded():
    from wsp.editor.sandbox import audit

    code = 'import sys; out=sys._getframe().f_back.f_globals[\'transport_out\']; out.write(\'{"done":true,"version":"editor_api.v1"}\\n{}\\n\'); out.flush()'
    result = run(code, EditorAPI({"w": torch.ones(1)}, None, 10))
    assert not result.valid
    summary = audit("x" * 900000)
    assert len(json.dumps(summary)) < 5000
    assert summary["sha256"]


def test_pool_disjointness_normalised_and_manifest(tmp_path):
    from wsp.editor.driver import qualify_pools
    from wsp.editor.episode import Episode, digest

    dev = [{"id": "train:1", "source": "gsm8k", "split": "train", "prompt": "Hello  WORLD"}]
    sealed = [{"id": "test:1", "prompt": " hello world "}]
    with pytest.raises(ValueError, match="overlap"):
        qualify_pools(dev, sealed)
    sealed[0]["prompt"] = "Different"
    ep = Episode(tmp_path, "pool")
    hashes = qualify_pools(dev, sealed, ep)
    assert hashes["development_pool_hash"] == digest(dev)
    manifest = json.loads((ep.path.parent / "manifest.json").read_text())
    assert manifest["pools"] == hashes
    assert "Different" not in json.dumps(manifest)


def test_seccomp_denials_audited_and_eperm():
    code = """
import os, socket
for operation in [lambda: open('/etc/passwd'), lambda: os.stat('/etc'), lambda: os.listdir('/'), lambda: socket.socket()]:
    try:
        operation()
        raise AssertionError('allowed')
    except OSError as e:
        assert e.errno == 1
call('budget')
"""
    result = run(code, EditorAPI({"w": torch.ones(1)}, None, 10))
    assert result.valid, result.error
    assert result.denial_audit["enabled"]
    assert sum(result.denial_audit["denied_syscalls"].values()) >= 4
    assert result.denial_audit["error"] is None


def test_hf_runner_explicit_device_and_reference_kernels(monkeypatch):
    from types import SimpleNamespace

    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    from wsp.eval.run import HFRunner

    seen = {}

    def load(*args, **kwargs):
        seen.update(kwargs)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr("wsp.zoo.common.resolve_base_revision", lambda *a: "a" * 40)
    originals = {
        name: getattr(qwen, name)
        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        )
    }
    # Keep global changes local to this test; workers each own their kernel policy.
    for name, function in originals.items():
        monkeypatch.setattr(qwen, name, function)
    runner = HFRunner("Qwen/Qwen3.5-0.8B", {}, revision="a" * 40, device="cpu", kernels="reference")
    assert seen["device_map"] == "cpu"
    assert runner.revision == "a" * 40
    import inspect

    for name, original in originals.items():
        function = original
        if isinstance(function, torch.nn.Module):
            function = inspect.getclosurevars(function.forward).nonlocals["func"]
        assert getattr(qwen, name) is inspect.unwrap(function)
        assert not isinstance(getattr(qwen, name), torch.nn.Module)
    x, w = torch.ones(1, 2, 4), torch.ones(2, 3)
    assert qwen.causal_conv1d_fn(x, w).device.type == "cpu"


def test_code_and_rationale_caps(tmp_path):
    from wsp.editor.driver import run_episode
    from wsp.editor.episode import Episode, digest
    from wsp.editor.providers import ScriptedEditor

    ep = Episode(tmp_path, "bounded")
    cfg = {
        "base_model": "tiny",
        "base_revision": "a" * 40,
        "budget": 10,
        "editable_tensors": ["w"],
        "development_pool_hash": digest([]),
    }
    code = "#" + "x" * 900000
    assert run_episode(
        ScriptedEditor([code]), cfg, {"w": torch.ones(1)}, None, [], ep, tmp_path / "zoo"
    ) == [None]
    assert ep.path.stat().st_size < 10000
    completion = next(
        json.loads(line)
        for line in ep.path.read_text().splitlines()
        if json.loads(line)["event"] == "completion"
    )
    assert completion["code"]["sha256"] == digest(code)
    with pytest.raises(ValueError, match="rationale"):
        EditorAPI({"w": torch.ones(1)}, None, 10).set("w", [0.1], "x" * 4097)


def test_episode_pool_preflight_enforced_before_editor(tmp_path):
    from wsp.editor.driver import run_episode
    from wsp.editor.episode import Episode, digest
    from wsp.editor.providers import ScriptedEditor

    dev = [{"id": "train:1", "source": "gsm8k", "split": "train", "prompt": "visible"}]
    sealed = [{"id": "test:1", "prompt": "hidden"}]
    cfg = {
        "base_model": "tiny",
        "base_revision": "a" * 40,
        "budget": 10,
        "seed": 0,
        "editable_tensors": ["w"],
        "development_pool_hash": digest(dev),
        "sealed_pool_hash": digest(sealed),
    }
    ep = Episode(tmp_path, "preflight")
    args = (ScriptedEditor([]), cfg, {"w": torch.ones(1)}, None, dev, ep, tmp_path / "zoo")
    with pytest.raises(ValueError, match="sealed pool required"):
        run_episode(*args)
    with pytest.raises(ValueError, match="hash mismatch"):
        run_episode(*args, sealed_pool=[])
    cfg["sealed_pool_hash"] = digest([{"id": "train:1", "prompt": "other"}])
    with pytest.raises(ValueError, match="overlap"):
        run_episode(*args, sealed_pool=[{"id": "train:1", "prompt": "other"}])
    cfg["sealed_pool_hash"] = digest(sealed)
    run_episode(*args, sealed_pool=sealed, proposals=0)
    manifest = json.loads((ep.path.parent / "manifest.json").read_text())
    assert manifest["pools"]["disjoint"]
    assert manifest["model_revision"] == cfg["base_revision"]
    assert manifest["config_hash"] == digest(cfg)
    assert "hidden" not in json.dumps(manifest)
