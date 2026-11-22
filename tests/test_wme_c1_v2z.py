"""RULING-wme-c1-bank-halt addendum 8 (run9: compensated inerts outside the field support), v2z: R1 R(E) of an inert
unit is contracted over the supported tensors only (the field's writers and the vocabulary rows) and the unit records
``r_omitted``; every other unit still raises on an unsupported tensor. R2 a chunk that executed no unit gets no
closing canary. CPU only; fixtures."""

import types

import pytest
import torch

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import field as F
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import protocol as P
from wsp.wm.collect import UnsupportedTensor

ROOT_UP = "model.layers.12.mlp.up_proj.weight"
DOWN = "model.layers.12.mlp.down_proj.weight"
EMBED = "model.embed_tokens.weight"


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


class Act:
    def __init__(self, tensor, v):
        self.tensor, self.v, self.family = tensor, v, "scale"

    def key(self):
        return (self.tensor, self.v)

    def param(self, name):
        return None


class Sess:
    def __init__(self):
        self.d = {}

    def apply(self, a):
        self.d[a.tensor] = self.d.get(a.tensor, 0.0) + a.v

    def displacement(self, name):
        return torch.full((2, 2), float(self.d.get(name, 0.0)))

    def restore(self):
        self.d = {}


class FakeField:
    """The field's support: the writers in ``param_group`` (no up_proj), the vocabulary rows by ``contract_embed``."""

    def __init__(self):
        self.param_group = {DOWN: (0, {"factors": False})}

    def contract_tensor(self, name, D, objs):
        if name not in self.param_group:
            raise UnsupportedTensor(f"{name}: not in the collected support")
        return {o: (float(D.sum()), None) for o in objs}

    def contract_embed(self, name, D, objs):
        return {o: (float(D.sum()) * 0.5, None) for o in objs}


def rcache():
    m = types.SimpleNamespace(sess=Sess(), embed=EMBED, at_base=lambda: None)
    return F.RCache(m, FakeField())


COMP = [
    [Act(ROOT_UP, 2.0), Act(DOWN, -1.0)]
]  # a compensated pair: up_proj scaled, down_proj compensating


def test_a_non_inert_unit_still_raises_on_an_unsupported_tensor():
    rc = rcache()
    with pytest.raises(UnsupportedTensor):
        rc.r_of(COMP)
    with pytest.raises(UnsupportedTensor):
        rc.r_of_many([COMP])


def test_support_only_r_of_a_compensated_inert_omits_up_proj_and_equals_the_supported_part():
    rc = rcache()
    r, flags = rc.r_of(COMP, omit_unsupported=True)
    r_sup, flags_sup = rcache().r_of([[Act(DOWN, -1.0)]])  # R(E) over the supported tensor alone
    assert list(r) == list(r_sup) and list(flags) == list(flags_sup)
    assert rc.omitted_of(COMP) == [ROOT_UP]
    rows = rcache().r_of_many([COMP, [[Act(DOWN, -1.0)]]], omit_unsupported=True)
    assert list(rows[0][0]) == list(r_sup)
    assert (
        rcache().omitted_of([[Act(DOWN, 1.0), Act(EMBED, 1.0)]]) == []
    )  # embed rows are supported


def bankworld(rc):
    from wsp.wm.c1 import bankops as B

    bw = B.BankWorld.__new__(B.BankWorld)
    bw.rc, bw.check = rc, (lambda: None)
    bw.m = types.SimpleNamespace(
        at_base=lambda: None, restore=lambda: None, base_hash="b" * 64,
        sess=types.SimpleNamespace(params={}, apply_bundle=lambda b: None),
    )  # fmt: skip
    return bw


def test_the_inert_unit_contracts_support_only_and_records_r_omitted(monkeypatch):
    from wsp.wm.c1 import bankops as B

    bw = bankworld(rcache())
    monkeypatch.setattr(B, "inert_steps", lambda spec, m: [COMP[0]])
    monkeypatch.setattr(
        B.BankWorld, "_score", lambda self, b: {"values": {"o": 1.0}, "theta_hash": "b" * 64}
    )
    out = bw.inert({"kind": "comp", "index": 0}, {"o": 1.0})
    assert out["r_omitted"] == [ROOT_UP]
    assert out["states"][0]["r_omitted"] == [ROOT_UP] and "R" in out["states"][0]


def test_an_inert_unit_without_unsupported_tensors_keeps_its_registered_payload_shape(monkeypatch):
    from wsp.wm.c1 import bankops as B

    bw = bankworld(rcache())
    monkeypatch.setattr(B, "inert_steps", lambda spec, m: [[Act(DOWN, 1.0)], [Act(DOWN, -1.0)]])
    monkeypatch.setattr(
        B.BankWorld, "_score", lambda self, b: {"values": {"o": 1.0}, "theta_hash": "b" * 64}
    )
    out = bw.inert({"kind": "i1", "shape": "2step"}, {"o": 1.0})
    assert "r_omitted" not in out and all("r_omitted" not in s for s in out["states"])


def test_a_bank_side_or_composition_still_raises(monkeypatch):
    bw = bankworld(rcache())
    with pytest.raises(UnsupportedTensor):
        bw._with_r([{"h": 1}], [COMP])  # the default path of every non-inert unit


# ------------------------------------------------------------------ R2: no closing canary for an empty chunk

REF = "a" * 64
ATT = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])


def guard(tmp_path):
    g = RG.ProductionGuard(
        tmp_path / "s", "bank", {"bank": 500.0}, ATT, "lambda:i1", 0.01, technical=C.technical,
        fallback_value={"rule": "x"},
    )  # fmt: skip
    g.open_instance(1)
    return g


def test_a_chunk_that_executed_no_unit_gets_no_closing_canary(tmp_path):
    g = guard(tmp_path)
    store = H.UnitStore(tmp_path / "u")
    calls = []

    def canary():
        calls.append("canary")
        g.canary(REF)
        return {"pass": True, "composite": REF}

    units = [({"u": i}, f"u{i}") for i in range(6)]
    fn = lambda r: {"v": r["u"]}
    C.run_bracketed(g, store, units[:4], fn, "t" * 64, "p", canary, chunk=2)
    assert calls == ["canary"] * 3  # start + 2 chunks
    calls.clear()
    out = C.run_bracketed(g, store, units, fn, "t" * 64, "p", canary, chunk=2)
    # start canary; chunks u0-u1 and u2-u3 executed nothing (eligible): no canary; chunk u4-u5 ran: one canary
    assert calls == ["canary"] * 2
    assert [out[f"u{i}"]["v"] for i in range(6)] == list(range(6))


def test_a_failed_attempt_counts_as_executed_and_keeps_its_canary(tmp_path):
    g = guard(tmp_path)
    store = H.UnitStore(tmp_path / "u")
    calls = []

    def canary():
        calls.append(1)
        g.canary(REF)
        return {"pass": True, "composite": REF}

    state = {"n": 0}

    def flaky(r):
        state["n"] += 1
        if state["n"] == 1:
            raise P.TransportError("technical: infrastructure")  # retried in the same chunk
        return {"v": 1}

    C.run_bracketed(g, store, [({"u": 0}, "u0")], flaky, "t" * 64, "p", canary, chunk=1)
    assert len(calls) == 2  # start + the chunk that executed (a failed then a successful attempt)
