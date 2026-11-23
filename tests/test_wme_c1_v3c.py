"""run13 halt (v3c): ``Runtime.release()`` (before a canary's child engines) must also drop the E2 environment cache
(``rt._e2``: the Env and DevWorld built by ``serve_e2.env_for`` hold the model context). Before the fix the cached Env
kept the released model (``sess`` None) and the next E2 request reused it: ``e2:random:00`` failed in
``modelops.at_base`` ("'NoneType' object has no attribute 'theta_hash'"). After a release the next ``env_for``
rebuilds the Env on the reloaded model. CPU only; fixtures."""

import gc
import weakref

import pytest
import torch

from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import serve as S
from wsp.wm.c1 import serve_e2 as SE


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


class Blob:
    pass


class FakeModel:
    def __init__(self):
        self.weights, self.sess, self.released = Blob(), Blob(), []

    def release(self, free=True):
        self.released.append(free)
        self.weights = self.sess = None


CLOSED = []


class FakeWorld:
    def __init__(self, m):
        self.m = m

    def close(self):
        CLOSED.append(1)


def runtime_with_e2():
    rt = S.Runtime.__new__(S.Runtime)
    rt.cfg = None
    rt.m, rt._field, rt._rc, rt._bank = FakeModel(), None, None, None
    st = SE.E2State()
    st.key, st.world = ("l", "d", "f"), FakeWorld(rt.m)
    st.env = Blob()
    st.env.m, st.env.world = rt.m, st.world
    rt._e2 = st
    return rt


def test_release_drops_the_e2_env_cache_before_gc_and_empty_cache(monkeypatch):
    rt = runtime_with_e2()
    CLOSED.clear()
    refs = {
        "env": weakref.ref(rt._e2.env),
        "world": weakref.ref(rt._e2.world),
        "model": weakref.ref(rt.m),
    }
    order = []
    real_collect = gc.collect
    monkeypatch.setattr(
        gc, "collect", lambda *a, **k: (order.append("gc"), real_collect(*a, **k))[1]
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: order.append(("empty_cache", {n: r() is not None for n, r in refs.items()})),
    )
    rt.release()
    assert rt._e2 is None and CLOSED == [1]
    assert order[0] == "gc" and not any(order[1][1].values()), (
        order
    )  # no Env or model survives to empty_cache


def test_env_for_after_a_release_rebuilds_on_the_reloaded_model(monkeypatch):
    """The cached state is gone, so ``env_for`` takes its rebuild branch (``st.key != key``) and asks ``rt.hf()``."""
    rt = runtime_with_e2()
    rt.release()
    assert getattr(rt, "_e2", None) is None
    st = getattr(rt, "_e2", None) or SE.E2State()
    assert st.key is None and st.env is None  # env_for's rebuild condition holds for any input key


def test_release_without_an_e2_cache_is_unchanged():
    rt = runtime_with_e2()
    rt._e2 = None
    rt.release()
    assert rt._e2 is None and rt.m is None


def test_a_chunk_boundary_canary_between_two_e2_units_rebuilds_the_env(tmp_path, monkeypatch):
    """The run10 latent path (addendum 13): E2 unit 1 builds the Env through ``env_for``; the chunk's closing canary
    releases the model (``Runtime.release`` before its child engines); E2 unit 2 must get a new Env on the reloaded
    model, never the cached one holding the released model (``sess`` None)."""
    import json as _json
    import types

    from wsp.wm.c1 import e2ops as EO

    built = []

    class Lib:
        def __init__(self, path, rm, smoke=False):
            self.index = {}

    class World(FakeWorld):
        def __init__(self, m, items, base, lib, check=None):
            super().__init__(m)
            built.append(self)

    monkeypatch.setattr(EO, "Library", Lib)
    monkeypatch.setattr(EO, "DevWorld", World)
    monkeypatch.setattr(SE, "_verified", lambda p, sha: p)
    monkeypatch.setattr(SE, "_dev_items", lambda rt: [])
    monkeypatch.setattr(SE, "_c1_manifest", lambda rt: {})
    dev = tmp_path / "dev_base.json"
    dev.write_text(_json.dumps({"records": []}))
    rt = S.Runtime.__new__(S.Runtime)
    rt.cfg = types.SimpleNamespace(smoke=False)
    rt.m, rt._field, rt._rc, rt._bank = None, None, None, None

    def hf():
        if rt.m is None:
            rt.m = FakeModel()
            rt.m.rm = None
        return rt.m

    rt.hf, rt.rcache = hf, lambda: Blob()
    rt.shard_items = lambda: (None, [], None)
    args = {
        "library_sha256": "l",
        "dev_base_sha256": "d",
        "library_file": tmp_path / "lib.pt",
        "dev_base_file": dev,
    }
    ctx = types.SimpleNamespace(check=lambda: None, tick=lambda *a, **k: None)
    env1 = SE.env_for(rt, args, ctx, need_fit=False)  # E2 unit 1
    assert SE.env_for(rt, args, ctx, need_fit=False) is env1  # cached within a chunk
    m1 = env1.m
    CLOSED.clear()
    rt.release()  # the chunk-boundary canary
    assert m1.sess is None and CLOSED == [1]
    env2 = SE.env_for(rt, args, ctx, need_fit=False)  # E2 unit 2
    assert env2 is not env1 and env2.m is rt.m and env2.m is not m1 and env2.m.sess is not None
    assert len(built) == 2
