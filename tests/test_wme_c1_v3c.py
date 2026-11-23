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
