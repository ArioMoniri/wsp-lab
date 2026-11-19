"""wme-c1 controller-worker protocol, fencing and bracketed production runs (CPU fixtures, engineering).

PREREG_WME §II.28.3 preconditions 5 (remote outputs only through the current owner; stale completion refused;
takeover stops old work) and 8 (ProductionGuard only)."""

import json
import threading
import time

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import protocol as P
from wsp.wm.c1 import transport as T
from wsp.wm.c1 import worker as W

ENG = {"tf": "1" * 64, "hf_gen": "2" * 64, "vllm": "3" * 64}
REF = C.composite_canary(ENG)


def attest(ref=REF):
    return RG.LaunchAttestation(
        inert_identity_bitwise=True,
        gpu_parity_bitwise=True,
        canary_reference=ref,
        engine_config_sha256="c" * 64,
        evidence=["fixture"],
    )


def guard(state, phase="bank", instance="lambda:i1", fallback=None):
    return RG.ProductionGuard(
        state,
        phase,
        {"bank": 10.0, "confirmatory": 50.0},
        attest(),
        instance,
        est_unit_gpu_h=0.01,
        technical=C.technical,
        fallback_value=fallback or {"missing": True},
    )


class Box:
    engines = dict(ENG)


def handlers(box=Box):
    def square(req, ctx):
        ctx.check()
        x = req["args"]["x"]
        out = ctx.scratch / "sq.json"
        out.write_text(json.dumps({"x": x, "sq": x * x}))
        return {"sq": x * x}, {"sq.json": out}

    def boom(req, ctx):
        raise MemoryError("CUDA out of memory (fixture)")

    def crash(req, ctx):
        raise KeyError("a code defect")

    def slow(req, ctx):
        for _ in range(200):
            time.sleep(0.01)
            ctx.check()
        return {"slow": True}, {}

    def canary(req, ctx):
        return {"engines": dict(box.engines)}, {}

    return {"square": square, "boom": boom, "crash": crash, "slow": slow, "canary": canary}


@pytest.fixture
def rig(tmp_path):
    root = tmp_path / "worker"
    wk = W.Worker(root, handlers(), poll_s=0.01)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            wk.touch()
            todo = wk.pending()
            if todo:
                wk.handle(todo[0])
            else:
                time.sleep(0.005)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    tr = T.LocalTransport(root)
    rem = C.Remote(tr, tmp_path / "ctl", poll_s=0.01, dead_after_s=30)
    yield {"root": root, "wk": wk, "tr": tr, "rem": rem, "tmp": tmp_path}
    stop.set()
    th.join(2)


def test_request_roundtrip_files_verified(rig):
    payload, d = rig["rem"].call("square", 1, "lambda:i1", "k" * 64, "u0", {"x": 7})
    assert payload == {"sq": 49}
    assert json.loads((d / "sq.json").read_text())["sq"] == 49
    lines = [json.loads(x) for x in (rig["root"] / "progress.jsonl").read_text().splitlines()]
    assert lines[-1]["event"] == "ok"


def test_result_for_another_request_is_never_admitted(rig, tmp_path):
    rem = rig["rem"]
    _, d = rem.call("square", 1, "lambda:i1", "k" * 64, "u0", {"x": 3})
    other = P.make_request("square", 1, "lambda:i1", "k" * 64, "u0", {"x": 3})
    with pytest.raises(P.ProtocolError, match="does not answer"):
        P.verify_result(d, other)
    # a tampered payload file is refused too
    req = P.make_request("square", 1, "lambda:i1", "q" * 64, "u1", {"x": 2})
    rig["tr"].put(req)
    while not rig["tr"].ready(req["id"]):
        time.sleep(0.01)
    dd = rig["tr"].fetch(req["id"], tmp_path / "f")
    (dd / "sq.json").write_text("{}")
    with pytest.raises(P.ProtocolError, match="sha256"):
        P.verify_result(dd, req)


def test_worker_errors_are_classified(rig):
    with pytest.raises(P.WorkerError) as e:
        rig["rem"].call("boom", 1, "lambda:i1", "b" * 64, "b", {})
    assert C.technical(e.value) == "out_of_memory"
    with pytest.raises(P.WorkerError) as e2:
        rig["rem"].call("crash", 1, "lambda:i1", "c" * 64, "c", {})
    assert C.technical(e2.value) is None  # unrecognised: the controller halts
    assert C.technical(P.TransportError("x")) == "infrastructure"
    assert C.technical(ValueError("x")) is None


def test_stale_request_is_rejected_and_takeover_aborts_running_work(rig, tmp_path):
    tr, wk = rig["tr"], rig["wk"]
    tr.set_fence(5, "lambda:i1")
    old = P.make_request("square", 4, "lambda:i1", "s" * 64, "old", {"x": 1})
    tr.put(old)
    deadline = time.time() + 5
    while not (rig["root"] / "rejected" / f"{old['id']}.json").exists():
        assert time.time() < deadline
        time.sleep(0.01)
    assert not tr.ready(old["id"])
    # a running request is aborted when the fence moves past it: no result is written
    run = P.make_request("slow", 5, "lambda:i1", "r" * 64, "run", {})
    tr.put(run)
    time.sleep(0.2)
    tr.set_fence(6, "lambda:i1")
    deadline = time.time() + 5
    while not (rig["root"] / "done" / f"{run['id']}.json").exists():
        assert time.time() < deadline
        time.sleep(0.01)
    assert not tr.ready(run["id"])
    ev = [json.loads(x) for x in (rig["root"] / "progress.jsonl").read_text().splitlines()]
    assert {"rejected_stale", "fenced"} <= {e["event"] for e in ev}


def _units(n):
    return [({"unit": i, "x": i + 1}, f"u{i}") for i in range(n)]


def test_bracketed_production_run_returns_only_eligible_results(rig, tmp_path):
    state = tmp_path / "study"
    g = guard(state)
    g.open_instance(1)
    rig["tr"].set_fence(g.epoch, g.instance)
    store = H.UnitStore(tmp_path / "units")
    fn = lambda r: rig["rem"].call("square", g.epoch, g.instance, H.unit_key(r, "t0", "p"), "u", {"x": r["x"]})[0]
    out = C.run_bracketed(g, store, _units(5), fn, "t0", "p", lambda: C.run_canary(g, rig["rem"]), chunk=2)
    assert [out[f"u{i}"]["sq"] for i in range(5)] == [1, 4, 9, 16, 25]
    rep = g.report()
    assert all(st == "eligible" for u in rep["units"].values() for _, st in u["attempts"])
    # 1 start + 3 closing canaries for chunks of 2, 2, 1
    assert sum(1 for e in g.events if e["type"] == "canary") == 4


def test_failed_closing_canary_halts_and_leaves_results_ineligible(rig, tmp_path):
    state = tmp_path / "study"
    g = guard(state)
    g.open_instance(1)
    store = H.UnitStore(tmp_path / "units")
    calls = {"n": 0}

    class FlipBox:
        engines = dict(ENG)

    def canary():
        calls["n"] += 1
        eng = dict(ENG) if calls["n"] == 1 else dict(ENG, tf="9" * 64)
        return {"composite": C.composite_canary(eng), "pass": g.canary(C.composite_canary(eng))}

    fn = lambda r: rig["rem"].call("square", g.epoch, g.instance, H.unit_key(r, "t0", "p"), "u", {"x": r["x"]})[0]
    with pytest.raises(RG.CanaryMismatch):
        C.run_bracketed(g, store, _units(2), fn, "t0", "p", canary, chunk=8)
    for recipe, _ in _units(2):
        assert g.resolve(store, H.unit_key(recipe, "t0", "p")) is None


def test_takeover_makes_the_old_controller_stale_and_resumes_without_recompute(rig, tmp_path):
    state = tmp_path / "study"
    g1 = guard(state)
    g1.open_instance(1)
    store = H.UnitStore(tmp_path / "units")
    fn1 = lambda r: rig["rem"].call("square", g1.epoch, g1.instance, H.unit_key(r, "t0", "p"), "u", {"x": r["x"]})[0]
    C.run_bracketed(g1, store, _units(2), fn1, "t0", "p", lambda: C.run_canary(g1, rig["rem"]))
    g2 = guard(state)  # a controller restart: verified fresh replay, new owner epoch
    rig["tr"].set_fence(g2.epoch, g2.instance)
    with pytest.raises(RG.StaleGuard):
        g1.canary(REF)
    computed = []

    def fn2(r):
        computed.append(r["unit"])
        return rig["rem"].call("square", g2.epoch, g2.instance, H.unit_key(r, "t0", "p"), "u", {"x": r["x"]})[0]

    out = C.run_bracketed(g2, store, _units(3), fn2, "t0", "p", lambda: C.run_canary(g2, rig["rem"]))
    assert computed == [2] and out["u0"]["sq"] == 1 and out["u2"]["sq"] == 9
    # a request issued under the old epoch is refused by the worker
    old = P.make_request("square", g1.epoch, "lambda:i1", "z" * 64, "late", {"x": 1})
    rig["tr"].put(old)
    deadline = time.time() + 5
    while not (rig["root"] / "rejected" / f"{old['id']}.json").exists():
        assert time.time() < deadline
        time.sleep(0.01)


def test_dead_worker_ends_the_attempt_without_waiting_forever(tmp_path):
    root = tmp_path / "w"
    tr = T.LocalTransport(root)
    (root / "alive").write_text("x")
    import os

    os.utime(root / "alive", (time.time() - 5000, time.time() - 5000))
    rem = C.Remote(tr, tmp_path / "c", poll_s=0.01, dead_after_s=60)
    with pytest.raises(C.WorkerDead):
        rem.call("square", 1, "lambda:i1", "d" * 64, "d", {"x": 1})
    assert C.technical(C.WorkerDead("x")) == "infrastructure"


def test_lost_instance_ends_the_attempt(tmp_path):
    root = tmp_path / "w"
    tr = T.LocalTransport(root)
    rem = C.Remote(tr, tmp_path / "c", poll_s=0.01, lost=lambda: True)
    with pytest.raises(C.InstanceLost):
        rem.call("square", 1, "lambda:i1", "d" * 64, "d", {"x": 1})
