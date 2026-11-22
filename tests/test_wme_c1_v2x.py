"""RULING-wme-c1-bank-halt addendum 6 (run7 canary OOM), v2x: ``Runtime.release()`` drops every GPU reference (the
R cache, the bank world, the field tensors, the model) BEFORE ``gc.collect()`` and ``torch.cuda.empty_cache()``; the
parent's free GPU memory is checked and recorded before any child process starts, and a shortfall refuses the child.
CPU only; fixtures (the local-GPU release smoke test is ``scripts/wme_c1_release_smoke.py``)."""

import gc
import json
import weakref

import pytest
import torch

from wsp.wm.c1 import childmem as CM
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import hfchild as HC
from wsp.wm.c1 import serve as S
from wsp.wm.c1 import steops as SO
from wsp.wm.c1 import vllmops as VO


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


class Blob:
    """A stand-in for a GPU tensor holder (weak-referenceable)."""


class FakeModel:
    def __init__(self):
        self.weights = Blob()
        self.released = []

    def release(self, free=True):
        self.released.append(free)
        self.weights = None


class FakeField:
    def __init__(self):
        self._kept = {"k": Blob()}
        self._grad = {"g": Blob()}


def runtime():
    rt = S.Runtime.__new__(S.Runtime)
    rt.cfg = None
    rt.m, rt._field, rt._rc, rt._bank = FakeModel(), FakeField(), Blob(), Blob()
    rc_model = (
        rt.m
    )  # the R cache holds the model and the field, as F.RCache does (a reference cycle)
    rt._rc.model, rt._rc.field, rt._rc.me = rc_model, rt._field, rt._rc
    return rt


def test_release_drops_every_gpu_reference_before_gc_and_empty_cache(monkeypatch):
    rt = runtime()
    refs = {
        "rc": weakref.ref(rt._rc),
        "bank": weakref.ref(rt._bank),
        "field": weakref.ref(rt._field),
        "field_kept": weakref.ref(rt._field._kept["k"]),
        "field_grad": weakref.ref(rt._field._grad["g"]),
        "model": weakref.ref(rt.m),
        "weights": weakref.ref(rt.m.weights),
    }
    model = refs["model"]()
    order = []
    real_collect = gc.collect

    def collect(*a, **k):
        order.append(("gc", {n: r() is not None for n, r in refs.items()}))
        return real_collect(*a, **k)

    def empty_cache():
        order.append(("empty_cache", {n: r() is not None for n, r in refs.items()}))

    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    del model
    rt.release()
    assert [o[0] for o in order] == [
        "gc",
        "empty_cache",
    ]  # one collection, then one cache release, in that order
    alive_at_empty = order[1][1]
    assert not any(alive_at_empty.values()), (
        alive_at_empty
    )  # nothing GPU-holding survives to empty_cache
    assert rt.m is None and rt._rc is None and rt._bank is None and rt._field is None


def test_the_model_release_inside_runtime_release_does_not_empty_the_cache_early(monkeypatch):
    rt = runtime()
    m = rt.m
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    rt.release()
    assert m.released == [False] and calls == ["empty_cache"]


def test_modelctx_release_alone_still_frees_by_default():
    from wsp.wm.c1 import modelops as MO

    assert "free" in MO.ModelCtx.release.__code__.co_varnames


# ------------------------------------------------------------------ the parent's free memory before a child


GIB = 2**30


def test_require_records_and_refuses_a_shortfall(monkeypatch):
    monkeypatch.setattr(
        CM, "free_info", lambda: None
    )  # "no CUDA" below, whatever the test machine has
    rec = CM.require("hf", required_gib=60.0, info=(70 * GIB, 80 * GIB))
    assert rec["pass"] is True and rec["free_gib"] == 70.0 and rec["required_gib"] == 60.0
    with pytest.raises(CM.ChildMemoryInsufficient, match="56.00 GiB free < 60.00 GiB"):
        CM.require("hf", required_gib=60.0, info=(56 * GIB, 80 * GIB))
    rec = CM.require("vllm", util=0.88, info=(75 * GIB, 80 * GIB))
    assert rec["required_gib"] == pytest.approx(70.4) and rec["pass"]
    with pytest.raises(CM.ChildMemoryInsufficient):
        CM.require("vllm", util=0.88, info=(55 * GIB, 80 * GIB))
    assert CM.require("hf", required_gib=60.0, info=None)["checked"] is False  # no CUDA (CPU tests)
    assert "out of memory" not in str(CM.ChildMemoryInsufficient("x")).lower()


@pytest.fixture
def short(monkeypatch):
    monkeypatch.setattr(CM, "free_info", lambda: (10 * GIB, 80 * GIB))


def test_no_hf_child_starts_on_a_shortfall(tmp_path, short):
    marker = tmp_path / "started"
    argv = ["python", "-c", f"open({str(marker)!r}, 'w').write('1')"]
    with pytest.raises(CM.ChildMemoryInsufficient):
        HC.spawn({"smoke": False, "ops": []}, tmp_path / "w", argv=argv)
    assert not marker.exists()
    rec = json.loads((tmp_path / "w" / "child_memory.json").read_text())
    assert rec["child"] == "hf" and rec["pass"] is False and rec["free_gib"] == 10.0


def test_no_vllm_or_training_child_starts_on_a_shortfall(tmp_path, short, monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("a child process was started")
    )
    with pytest.raises(CM.ChildMemoryInsufficient):
        VO.spawn({"smoke": False}, tmp_path / "v")
    with pytest.raises(CM.ChildMemoryInsufficient):
        SO.spawn_train({"smoke": False}, tmp_path / "t")
    assert json.loads((tmp_path / "v" / "child_memory.json").read_text())["child"] == "vllm"
    assert json.loads((tmp_path / "t" / "child_memory.json").read_text())["child"] == "train"


def test_a_child_with_enough_memory_starts_and_its_record_is_returned(tmp_path, monkeypatch):
    monkeypatch.setattr(CM, "free_info", lambda: (79 * GIB, 80 * GIB))
    op = tmp_path / "w" / "hf_out.json"
    argv = ["python", "-c", f"import json; json.dump({{'ok': 1}}, open({str(op)!r}, 'w'))"]
    out = HC.spawn({"smoke": False, "ops": []}, tmp_path / "w", argv=argv)
    assert (
        out["ok"] == 1
        and out["parent_memory"]["pass"] is True
        and out["parent_memory"]["free_gib"] == 79.0
    )


def test_a_canary_child_shortfall_is_an_incident_through_k2():
    """The canary's worker error (no child started) reaches the controller as a WorkerError; ``ctl.run_canary``
    records it as a ``worker_error`` incident before the run halts (K2). The error names the free and required
    memory and is not classified as an out-of-memory retry."""
    from wsp.wm.c1 import worker as W

    exc = CM.ChildMemoryInsufficient(
        "insufficient free GPU memory for the hf child: 10.00 GiB free < 60.00 GiB"
    )
    assert W.classify(exc) is None
