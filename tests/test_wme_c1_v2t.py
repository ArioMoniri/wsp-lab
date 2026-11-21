"""Chair item FF (after the run4 resume halt, 21 Nov 2026): fail fast on a rejected or unanswered request, and the
driver version in the launch / resume evidence. CPU only; fixtures.

* A request the worker moves to ``rejected/`` (stale epoch, fenced while running, malformed) is detected by the
  waiting controller within one poll interval and becomes a run-health incident naming the reason.
* Every request has a bounded response deadline per kind (``wsp.wm.c1.deadlines``): short for hello, canary and
  timing, generous for units. A passed deadline is an incident; the request is withdrawn.
* Both are ``IncidentAbort``: an in-flight unit attempt is recorded ``attempt_aborted`` (charged, never counted).
"""

import importlib.util
import json
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import deadlines as DL
from wsp.wm.c1 import health as HL
from wsp.wm.c1 import protocol as P
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import transport as T
from wsp.wm.c1 import worker as W

ROOT = Path(__file__).resolve().parents[1]
REF = "a" * 64
ATTEST = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])
POLL = 10.0


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    from wsp.wm.c1 import genengine as _GE

    saved = _GE.global_state()
    _GE.restore_state(_GE.BASELINE_STATE)
    yield
    _GE.restore_state(saved)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def open_guard(tmp_path, clk, inst="lambda:i1"):
    g = RG.ProductionGuard(
        tmp_path / "s",
        "e2",
        {"e2": 500.0},
        ATTEST,
        inst,
        0.01,
        technical=C.technical,
        fallback_value="base",
        clock=clk,
        cap=SD.CAP,
    )
    g.open_instance(1)
    g.canary(REF)
    return g


def recs(tmp_path, t=None):
    rs = [json.loads(x) for x in (tmp_path / "s" / RG.LOG_NAME).read_text().splitlines()]
    return rs if t is None else [r for r in rs if r["type"] == t]


def real_rig(tmp_path, clk, handlers=None):
    """The real worker and LocalTransport, stepped synchronously: every controller poll sleep advances the clock by
    the poll interval and lets the worker handle what is in its inbox."""
    root = tmp_path / "w"
    wk = W.Worker(root, handlers or {"square": lambda req, ctx: ({"y": req["args"]["x"] ** 2}, {})})
    tr = T.LocalTransport(root)
    (root / "alive").touch()

    def sleep(s):
        clk.t += s
        for p in wk.pending():
            wk.handle(p)

    rem = C.Remote(tr, tmp_path / "ctl", poll_s=POLL, sleep=sleep, clock=clk)
    return wk, tr, rem


def unit_through_guard(tmp_path, g, rem, kind="square", args=None):
    """One unit through the harness and the run-health wrapper, as ``Study.run`` issues it."""
    mon = HL.HealthMonitor(tmp_path / "runs")
    store = H.UnitStore(tmp_path / "u")

    def fn(recipe):
        return rem.call(kind, g.epoch, g.instance, "k" * 64, "u1", dict(args or {"x": 3}))[0]

    return H.run_units(store, [({"u": 1}, "u1")], mon.wrap_fn(g, fn), "t" * 64, "p", guard=g)


# ------------------------------------------------------------------ rejected requests


def test_ff_stale_request_rejected_by_the_worker_is_an_incident_within_one_poll_no_attempt(
    tmp_path,
):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    _wk, tr, rem = real_rig(tmp_path, clk)
    C.take_over(tr, rem, g)
    tr.set_fence(
        g.epoch + 1, g.instance
    )  # a newer owner fenced the worker: this owner's requests are stale
    t0 = clk.t
    with pytest.raises(C.RequestRejected) as ei:
        unit_through_guard(tmp_path, g, rem)
    assert clk.t - t0 <= POLL  # detected at the first poll after the worker moved it
    assert ei.value.incident_kind == "request_rejected"
    inc = recs(tmp_path, "incident")
    assert [i["kind"] for i in inc] == ["request_rejected"]
    d = inc[0]["detail"]
    assert d["reason"] == "stale_epoch" and d["request_epoch"] == g.epoch
    assert d["fence_epoch"] == g.epoch + 1 and d["kind"] == "square" and d["label"] == "u1"
    assert len(recs(tmp_path, "attempt_aborted")) == 1  # charged, never counted
    assert not recs(tmp_path, "attempt_fail") and not recs(tmp_path, "fallback")
    # the worker's evidence stays: the request in rejected/ with its reason
    rid = d["id"]
    assert (tr.root / "rejected" / f"{rid}.json").exists()
    assert (
        json.loads((tr.root / "rejected" / P.reason_name(rid)).read_text())["reason"]
        == "stale_epoch"
    )


def test_ff_the_run4_hang_now_fails_fast(tmp_path):
    """The run4 defect itself: a request at epoch 0 behind a fence at the owner epoch (outside any unit)."""
    clk = Clock()
    g = open_guard(tmp_path, clk)
    _wk, tr, rem = real_rig(tmp_path, clk, {"hello": lambda req, ctx: ({"ok": 1}, {})})
    C.take_over(tr, rem, g)
    t0 = clk.t
    with pytest.raises(C.RequestRejected, match="stale_epoch"):
        rem.call("hello", 0, g.instance, "m" * 64, "hello", {})
    assert clk.t - t0 <= POLL
    assert [i["kind"] for i in recs(tmp_path, "incident")] == ["request_rejected"]
    assert not recs(tmp_path, "attempt_start")  # not a unit: no attempt at all


def test_ff_malformed_request_is_rejected_with_its_reason(tmp_path):
    """A controller and worker that disagree on the protocol (a stale bundle): the worker refuses the request as
    malformed; the controller names the reason instead of waiting."""
    clk = Clock()
    g = open_guard(tmp_path, clk)
    _wk, tr, rem = real_rig(tmp_path, clk)
    C.take_over(tr, rem, g)
    put = tr.put

    def put_other_schema(req):
        put(req)
        p = tr.root / "inbox" / f"{req['id']}.json"
        p.write_text(json.dumps(dict(req, schema="wme-c1-rpc-v0")))

    tr.put = put_other_schema
    with pytest.raises(C.RequestRejected) as ei:
        unit_through_guard(tmp_path, g, rem)
    d = recs(tmp_path, "incident")[0]["detail"]
    assert d["reason"] == "malformed" and "schema" in d["detail"]
    assert "malformed" in str(ei.value)
    assert len(recs(tmp_path, "attempt_aborted")) == 1 and not recs(tmp_path, "attempt_fail")


def test_ff_fenced_while_running_is_detected_by_the_waiting_controller(tmp_path):
    """Another owner takes over while this owner's request runs: the worker aborts it (no result) and records why;
    the old owner fails fast instead of waiting for its deadline."""
    clk = Clock()
    g = open_guard(tmp_path, clk)

    def takeover_midway(req, ctx):
        T.LocalTransport(tmp_path / "w").set_fence(req["epoch"] + 1, req["instance"])
        ctx.check()
        return {"never": 1}, {}

    _wk, tr, rem = real_rig(tmp_path, clk, {"square": takeover_midway})
    C.take_over(tr, rem, g)
    t0 = clk.t
    with pytest.raises(C.RequestRejected):
        unit_through_guard(tmp_path, g, rem)
    assert clk.t - t0 <= POLL
    d = recs(tmp_path, "incident")[0]["detail"]
    assert d["reason"] == "fenced_while_running" and d["fence_epoch"] == g.epoch + 1
    assert (
        tr.root / "done" / f"{d['id']}.json"
    ).exists()  # the request file keeps its place (test_wme_c1_rpc)
    assert len(recs(tmp_path, "attempt_aborted")) == 1 and not recs(tmp_path, "attempt_fail")


# ------------------------------------------------------------------ unanswered requests


class SilentTransport:
    """A worker that is alive (fresh ``alive``, a heartbeat) but never answers and never rejects."""

    def __init__(self):
        self.puts, self.cancelled, self.fences = [], [], []

    def put(self, req):
        self.puts.append(req)

    def status(self, rid):
        return "pending"

    def rejection(self, rid):
        return None

    def ready(self, rid):
        return False

    def alive_age(self):
        return 1.0

    def heartbeat(self):
        return {"t": 0, "gpu": [], "inflight": None}

    def cancel(self, rid):
        self.cancelled.append(rid)
        return True

    def set_fence(self, epoch, instance):
        self.fences.append(epoch)


def silent_rig(tmp_path, clk):
    tr = SilentTransport()

    def sleep(s):
        clk.t += s

    return tr, C.Remote(tr, tmp_path / "ctl", poll_s=POLL, sleep=sleep, clock=clk)


@pytest.mark.parametrize("kind", ["bank_side", "e1_gen", "ste_train"])
def test_ff_an_unanswered_unit_is_a_deadline_incident_within_the_bound_no_attempt(tmp_path, kind):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    tr, rem = silent_rig(tmp_path, clk)
    C.take_over(tr, rem, g)
    t0 = clk.t
    with pytest.raises(C.RequestDeadline) as ei:
        unit_through_guard(tmp_path, g, rem, kind=kind)
    bound = DL.deadline_s(kind)
    assert bound <= clk.t - t0 <= bound + POLL
    assert ei.value.incident_kind == "request_deadline"
    inc = recs(tmp_path, "incident")
    assert [i["kind"] for i in inc] == ["request_deadline"]
    assert inc[0]["detail"]["kind"] == kind and inc[0]["detail"]["deadline_s"] == bound
    assert len(recs(tmp_path, "attempt_aborted")) == 1  # charged, never counted
    assert not recs(tmp_path, "attempt_fail") and not recs(tmp_path, "fallback")
    # the request was withdrawn (R6): taken out of the inbox, fenced above its epoch
    assert tr.cancelled == [tr.puts[0]["id"]] and tr.fences[-1] == g.epoch + 1


@pytest.mark.parametrize("kind", ["hello", "canary", "timing"])
def test_ff_hello_canary_and_timing_have_short_deadlines(tmp_path, kind):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    tr, rem = silent_rig(tmp_path, clk)
    C.take_over(tr, rem, g)
    t0 = clk.t
    with pytest.raises(C.RequestDeadline):
        rem.call(kind, g.epoch, g.instance, "m" * 64, kind, {})
    assert clk.t - t0 <= DL.deadline_s(kind) + POLL <= 1800 + POLL
    assert [i["kind"] for i in recs(tmp_path, "incident")] == ["request_deadline"]
    assert not recs(tmp_path, "attempt_start")


def test_ff_a_canary_deadline_halts_the_bracketed_run_before_any_unit(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    tr, rem = silent_rig(tmp_path, clk)
    C.take_over(tr, rem, g)
    store = H.UnitStore(tmp_path / "u")
    ran = []
    with pytest.raises(C.RequestDeadline):
        C.run_bracketed(
            g, store, [({"u": 1}, "u1")], ran.append, "t" * 64, "p", lambda: C.run_canary(g, rem)
        )
    assert not ran and not recs(tmp_path, "attempt_start")
    assert [i["kind"] for i in recs(tmp_path, "incident")] == ["request_deadline"]  # recorded once


def test_ff_without_a_bound_guard_the_failure_is_still_immediate(tmp_path):
    """The parity instance and the first-launch checks run before any guard exists: the same failure is raised at
    once (the command then terminates or halts), and it is journalled."""
    clk = Clock()
    _tr, rem = silent_rig(tmp_path, clk)
    with pytest.raises(C.RequestDeadline):
        rem.call("hello", 0, "lambda:p", "m" * 64, "hello", {})
    assert clk.t - 1000.0 <= DL.deadline_s("hello") + POLL
    ev = [json.loads(x) for x in (tmp_path / "ctl" / "dispatch.jsonl").read_text().splitlines()]
    assert any(e["event"] == "incident" and e["incident"] == "request_deadline" for e in ev)


def test_ff_an_explicit_timeout_is_a_deadline_incident_too(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    tr, rem = silent_rig(tmp_path, clk)
    C.take_over(tr, rem, g)
    with pytest.raises(C.RequestDeadline):
        rem.call("bank_side", g.epoch, g.instance, "k" * 64, "x", {}, timeout_s=60)
    assert clk.t - 1000.0 <= 60 + POLL


def test_ff_a_healthy_request_is_unaffected(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    _wk, tr, rem = real_rig(tmp_path, clk)
    C.take_over(tr, rem, g)
    out = unit_through_guard(tmp_path, g, rem)
    assert out["u1"]["status"] == "pending_canary" and not recs(tmp_path, "incident")


# ------------------------------------------------------------------ the deadline table


def test_ff_every_worker_handler_kind_has_a_deadline_short_checks_generous_units():
    import importlib

    from wsp.wm.c1 import serve as S

    importlib.import_module("wsp.wm.c1.serve_more")
    kinds = set(S.HANDLERS)
    assert kinds <= set(DL.DEADLINE_S), sorted(kinds - set(DL.DEADLINE_S))
    # derived from the registered per-unit estimates of the guard (study.EST_UNIT_GPU_H)
    assert DL.EST_UNIT_GPU_H == SD.EST_UNIT_GPU_H
    for kind, phase in DL.UNIT_PHASE.items():
        assert DL.deadline_s(kind) == DL.UNIT_FACTOR * SD.EST_UNIT_GPU_H[phase] * 3600
    assert set(DL.UNIT_PHASE) | set(DL.CHECK_S) == set(DL.DEADLINE_S)
    assert not set(DL.UNIT_PHASE) & set(DL.CHECK_S)
    for k in ("hello", "canary", "timing"):
        assert DL.deadline_s(k) <= 1800
    assert max(DL.CHECK_S.values()) < min(DL.deadline_s(k) for k in DL.UNIT_PHASE)
    # every unit kind of the §II.18 stages is a generous unit deadline
    assert set(HL.STAGE_OF) <= set(DL.UNIT_PHASE)
    with pytest.raises(KeyError):
        DL.deadline_s("no_such_kind")


# ------------------------------------------------------------------ transports


def test_ff_local_transport_status_and_rejection(tmp_path):
    tr = T.LocalTransport(tmp_path / "w")
    assert tr.status("x") == "pending" and tr.rejection("x") is None
    (tr.root / "rejected").mkdir(exist_ok=True)
    (tr.root / "rejected" / P.reason_name("x")).write_text(json.dumps({"reason": "stale_epoch"}))
    assert tr.status("x") == "rejected" and tr.rejection("x") == {"reason": "stale_epoch"}
    (tr.root / "rejected" / "y.json").write_text(
        "{}"
    )  # a worker without reason files (older bundle)
    assert tr.status("y") == "rejected" and tr.rejection("y") == {"reason": "unknown"}
    (tr.root / "outbox" / "z").mkdir()
    (tr.root / "outbox" / "z" / P.RESULT).write_text("{}")
    assert tr.status("z") == "ready"


def test_ff_ssh_transport_status_is_one_round_trip(tmp_path):
    seen = []

    def runner(argv, stdin=None, timeout=None):
        seen.append(argv[-1])
        if "cat " in argv[-1]:
            return json.dumps({"reason": "malformed", "detail": "schema"})
        return "rejected\n"

    tr = T.SSHTransport("a", tmp_path / "cfg", "/r/w", runner=runner)
    assert tr.status("e00001-x") == "rejected" and len(seen) == 1
    assert "outbox/e00001-x/result.json" in seen[0] and "rejected/" in seen[0]
    assert tr.rejection("e00001-x")["reason"] == "malformed"


# ------------------------------------------------------------------ every owner takeover binds the guard


def test_ff_every_fence_site_binds_the_guard():
    """An owner takeover sets the worker fence and binds the guard to the remote in one step (``ctl.take_over``),
    so every request failure is recorded with the current owner. No other fence call in the controller paths."""
    for rel in ("scripts/wme_c1_ctl.py", "src/wsp/wm/c1/controller.py", "src/wsp/wm/c1/study.py"):
        src = (ROOT / rel).read_text()
        assert ".set_fence(" not in src, rel
    src = (ROOT / "src/wsp/wm/c1/study.py").read_text()
    assert "bind(guard)" in src  # Study.run binds the phase guard before its first request


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2t", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ driver version in the evidence


def test_driver_version_is_recorded_and_compared_with_the_parity_instance_never_a_stop(monkeypatch):
    mod = load_ctl()
    monkeypatch.setattr(
        mod, "instance_record", lambda iid: {"bundle_sha256": "b" * 64, "gate": {"pass": True}}
    )
    from wsp.wm.c1 import phasel as PL

    monkeypatch.setattr(PL, "code_identity", lambda c, b: {"pass": True})
    monkeypatch.setattr(PL, "worker_state_check", lambda s, e: {"pass": True})
    hello = {"code_sha": "a" * 40, "physical": {"driver": "570.148.08", "gpu_names": ["H100"]}}
    par = {"hello": {"physical": {"driver": "580.105.08"}}}
    cs = mod.code_and_state(hello, "X", par)
    assert cs["driver"]["version"] == "570.148.08"
    assert cs["driver"]["parity_instance"] == "580.105.08"
    assert cs["driver"]["same_as_parity"] is False
    assert "canary" in cs["driver"]["binding_test"]
    mod.require_code_and_state(cs)  # a different driver alone is not a stop
    cs = mod.code_and_state(hello, "X")  # the parity instance itself
    assert cs["driver"]["version"] == "570.148.08" and cs["driver"]["parity_instance"] is None


def test_driver_is_in_the_launch_and_resume_evidence():
    src = (ROOT / "scripts/wme_c1_ctl.py").read_text()
    for fn in ("def cmd_phasel_main(", "def cmd_phasel_resume("):
        body = src[src.index(fn) : src.index("\ndef ", src.index(fn) + 10)]
        assert "code_and_state(hello, a.iid, par)" in body, fn
