"""Board A conditions on PREREG_WME §II.31 (docs/reviews/council/board-a-prereg-wme-v2l-2026-11-19.md), §II.31.8:
R1 the Phase L pass rule / selection executed by the driver; R2 canary set, run-tagged non-overwriting evidence and
engine-config binding; R3 the failure cause preserved and non-technical failures retried only after a repair; R4
non-failure incidents never consume an attempt; R5 C1(b) parser input = HF ids and HF decoding; R6 real withdrawal
and the shared skip declaration (live worker); R7 CPU-kind progress emission (worker and LLM-reader waits); R8 the
mechanical cap raise; R9 the gen.parsers digests. CPU only; fixtures."""

import hashlib
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import health as HL
from wsp.wm.c1 import phasel as PL
from wsp.wm.c1 import protocol as P
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import transport as T
from wsp.wm.c1 import worker as W

ROOT = Path(__file__).resolve().parents[1]
REF = "a" * 64
ATTEST = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])
A, B_ = "1" * 64, "2" * 64


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    """R11 (SYNTHESIS-r13 addendum item 5): every test here starts from torch's process defaults (no deterministic mode, default SDPA
    backends, no CUBLAS workspace variable), whatever an earlier test in any module left behind (e.g.
    ``wsp.predictor.models.seed_all``), and the caller's state is restored afterwards."""
    from wsp.wm.c1 import genengine as _GE

    saved = _GE.global_state()
    _GE.restore_state(_GE.BASELINE_STATE)
    yield
    _GE.restore_state(saved)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def guard(state, clk, cap=SD.CAP, priced=None, inst="i1"):
    return RG.ProductionGuard(
        state,
        "e2",
        priced or {"e2": 500.0},
        ATTEST,
        inst,
        0.01,
        technical=C.technical,
        fallback_value="base",
        clock=clk,
        cap=cap,
    )


def open_guard(tmp_path, clk, **kw):
    g = guard(tmp_path / "s", clk, **kw)
    g.open_instance(1)
    g.canary(REF)
    return g


def recs(tmp_path, t=None):
    rs = [json.loads(x) for x in (tmp_path / "s" / RG.LOG_NAME).read_text().splitlines()]
    return rs if t is None else [r for r in rs if r["type"] == t]


# ------------------------------------------------------------------ R1: the pass rule and selection, executed


def checks(rep=(A, A, A), g0=A, g1=A, inert_hf=True, tf_ok=True):
    return {
        "parity": {
            "parity_bitwise": tf_ok,
            "gpu0_vs_gpu1_every_chunk_bitwise": True,
            "canary_gpu0": {"tf": "t" * 64, "hf_gen": g0},
            "canary_gpu1": {"tf": "t" * 64, "hf_gen": g1},
            "vllm_gpu0_eq_gpu1": True,
        },
        "inert_identity": {
            "pass": inert_hf,
            "rows": [
                {"spec": {"kind": "i1"}, "hf_sha256": g0, "hf_bitwise_base": inert_hf},
                {"spec": {"kind": "i2"}, "hf_sha256": g0, "hf_bitwise_base": True},
                {"spec": {"kind": "i3"}, "hf_sha256": None, "hf_bitwise_base": None},
            ],
            "vllm_rows": [{"vllm_bitwise_base": True}],
        },
        "gen_repeat": {
            "deterministic": [{"sha256": h, "seconds": 10.0} for h in rep],
            "registered": {"sha256": "r" * 64, "seconds": 9.0},
            "t_det_median_s": 10.0,
        },
    }


def test_r1_run_checks_issues_gen_repeat_only_under_c1a():
    seen = []

    def call(kind, args):
        seen.append((kind, args))
        return {}

    PL.run_checks(call, "hf_deterministic")
    assert seen == [("parity", {}), ("inert_identity", {}), ("gen_repeat", {"n": 3})]
    seen.clear()
    PL.run_checks(call, "vllm")
    assert [k for k, _ in seen] == ["parity", "inert_identity"]


def test_r1_evaluation_pass_and_each_failure_item():
    ok = PL.evaluate(checks(), "hf_deterministic", 3.8, 110.0, 0.545)
    assert ok["proceed"] is True and ok["selected_engine"] == "hf_deterministic"
    assert ok["pass_rule"] == {
        "repeat_3_of_3": True,
        "gpu0_eq_gpu1": True,
        "inert_hf_bitwise": True,
        "pass": True,
    }
    assert ok["inputs"]["S_L"] == 3.8 and ok["inputs"]["C"] == 110.0
    for bad, item in (
        (checks(rep=(A, A, B_)), "repeat_3_of_3"),
        (checks(g1=B_), "gpu0_eq_gpu1"),  # the field that was false at the failed launch
        (checks(inert_hf=False), "inert_hf_bitwise"),
    ):
        ev = PL.evaluate(bad, "hf_deterministic", 3.8, 110.0, 0.545)
        assert ev["pass_rule"][item] is False
        assert ev["selected_engine"] == "vllm" and ev["proceed"] is False
    # another precondition failing stops the run even with the engine selected
    ev = PL.evaluate(checks(tf_ok=False), "hf_deterministic", 3.8, 110.0, 0.545)
    assert ev["selected_engine"] == "hf_deterministic" and ev["proceed"] is False


def test_r1_evaluation_under_active_c1b():
    c = checks()
    c["parity"].update(vllm_gen_gpu0=A, vllm_gen_gpu1=A)
    c["gen_repeat"] = None
    ev = PL.evaluate(c, "vllm", 3.8, 110.0, 0.545)
    assert ev["proceed"] is True and ev["selected_engine"] == "vllm"
    c["parity"]["vllm_gen_gpu1"] = B_
    ev = PL.evaluate(c, "vllm", 3.8, 110.0, 0.545)
    assert ev["proceed"] is False and ev["selected_engine"] is None


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2n", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRemote:
    def __init__(self, payloads):
        self.payloads, self.calls = payloads, []

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        self.calls.append((kind, args))
        p = self.payloads[kind]
        return (p(args) if callable(p) else p), None


PASSING_CS = {
    "code_identity": {"pass": True, "controller_commit": "fixture"},
    "worker_global_state": {"pass": True},
    "gate": {"pass": True},
}


def drive_parity(tmp_path, monkeypatch, c):
    mod = load_ctl()
    monkeypatch.setattr(
        mod, "code_and_state", lambda hello, iid, parity=None: PASSING_CS
    )  # v2o tests cover it
    rem = FakeRemote(
        {
            "hello": {"gen_engine": GE.config(), "theta0": "t", "versions": {}, "physical": {}},
            "parity": c["parity"],
            "inert_identity": c["inert_identity"],
            "gen_repeat": c["gen_repeat"],
        }
    )
    terminated = []
    monkeypatch.setattr(mod, "remote_for", lambda kind, iid: (None, rem))
    monkeypatch.setattr(mod, "EVID", tmp_path)
    monkeypatch.setattr(
        mod, "RUNS", tmp_path / "runs"
    )  # a first launch: no study (never the real data dir)
    monkeypatch.setattr(mod, "cap_state", lambda: (110.0, 106.2, 0))
    dm = types.SimpleNamespace(state=lambda: {"instances": {}})
    monkeypatch.setattr(mod.LO, "deadman", lambda *a, **k: dm)
    monkeypatch.setattr(mod.LO, "deadman_gpu_h", lambda st: 3.81)
    monkeypatch.setattr(
        mod.LO, "terminate_and_confirm", lambda dm_, iid, why: terminated.append(iid) or 1.0
    )
    a = types.SimpleNamespace(iid="X1", tag="r2")
    return mod, a, rem, terminated


def test_r1_phasel_parity_driver_both_outcomes(tmp_path, monkeypatch):
    mod, a, rem, term = drive_parity(tmp_path / "ok", monkeypatch, checks())
    mod.cmd_phasel_parity(a)
    assert [k for k, _ in rem.calls] == ["hello", "parity", "inert_identity", "gen_repeat"]
    assert ("gen_repeat", {"n": 3}) in rem.calls and term == ["X1"]
    ev = json.loads(next((tmp_path / "ok").glob("wme-c1-launch-parity-*-r2.json")).read_text())
    assert ev["evaluation"]["proceed"] is True
    assert ev["engine_config_sha256"] == mod.ec_sha()
    assert ev["evaluation"]["inputs"]["S_L"] == 3.81 and ev["evaluation"]["inputs"]["C"] == 110.0
    # failing outcome: evidence written, instance terminated, non-zero exit naming the selected engine
    mod, a, rem, term = drive_parity(tmp_path / "bad", monkeypatch, checks(g1=B_))
    with pytest.raises(SystemExit, match="selected engine vllm"):
        mod.cmd_phasel_parity(a)
    assert term == ["X1"]
    ev = json.loads(next((tmp_path / "bad").glob("wme-c1-launch-parity-*-r2.json")).read_text())
    assert ev["evaluation"]["selected_engine"] == "vllm"
    # R2: the same run tag never overwrites the committed evidence
    with pytest.raises(FileExistsError):
        mod.cmd_phasel_parity(a)


# ------------------------------------------------------------------ R2


def test_r2_evidence_never_overwritten_and_bound_to_the_engine_config(tmp_path):
    p = tmp_path / "e.json"
    h = PL.write_new(p, {"engine_config_sha256": "e" * 64})
    assert h == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        PL.write_new(p, {"x": 1})
    PL.check_binding({"engine_config_sha256": "e" * 64}, "e" * 64)
    with pytest.raises(RuntimeError, match="engine config"):
        PL.check_binding({"engine_config_sha256": "f" * 64}, "e" * 64)
    with pytest.raises(RuntimeError):
        PL.check_binding(
            {}, "e" * 64
        )  # evidence from before the binding (the failed launch) is refused


def test_r2_phasel_main_refuses_unbound_evidence_and_uses_the_engine_canary_set(
    tmp_path, monkeypatch
):
    mod = load_ctl()
    monkeypatch.setattr(mod, "EVID", tmp_path)
    monkeypatch.setattr(mod, "RUNS", tmp_path / "runs")  # a first launch: no existing study
    monkeypatch.setattr(mod, "STUDY", tmp_path / "runs" / "study")
    ev = {
        "engine_config_sha256": "0" * 64,
        "evaluation": {"proceed": True, "selected_engine": GE.GEN_ENGINE},
    }
    (tmp_path / "par.json").write_text(json.dumps(ev))
    monkeypatch.setattr(
        mod, "remote_for", lambda *a: pytest.fail("no instance contact before the binding")
    )
    a = types.SimpleNamespace(iid="X", tag="r2", parity_evidence=str(tmp_path / "par.json"))
    with pytest.raises(RuntimeError, match="engine config"):
        mod.cmd_phasel_main(a)
    # bound evidence, C1(b) active: the canary set {tf, vllm_gen, vllm} is accepted (stops at the next request)
    monkeypatch.setattr(GE, "GEN_ENGINE", "vllm")
    from wsp.wm.c1 import phasel as PL

    ev = {
        "engine_config_sha256": mod.ec_sha(),
        "evaluation": {"proceed": True, "selected_engine": "vllm"},
        "code_identity": {"controller_commit": PL.git("rev-parse", "HEAD")},
        "parity": {"canary_gpu0": {"tf": A}, "vllm_gpu0": A, "vllm_gen_gpu0": A},
    }
    (tmp_path / "par.json").write_text(json.dumps(ev))
    monkeypatch.setattr(mod, "code_and_state", lambda hello, iid, parity=None: PASSING_CS)

    class Stop(Exception):
        pass

    def timing(args):
        raise Stop

    rem = FakeRemote(
        {
            "hello": {
                "gen_engine": GE.config("vllm"),
                "theta0": "t",
                "versions": {},
                "physical": {},
            },
            "inert_identity": {"pass": True},
            "canary": {"engines": {"tf": A, "vllm_gen": A, "vllm": A}},
            "timing": timing,
        }
    )
    monkeypatch.setattr(mod, "remote_for", lambda *a: (None, rem))
    with pytest.raises(Stop):
        mod.cmd_phasel_main(a)


# ------------------------------------------------------------------ R3: the cause is preserved


def test_r3_worker_error_keeps_its_registered_cause_and_halts_without_another_attempt(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)
    store = H.UnitStore(tmp_path / "u")

    def oom(r):
        raise P.WorkerError("OutOfMemoryError", "CUDA out of memory", "out_of_memory")

    with pytest.raises(RG.HealthHalt):
        H.run_units(store, [({"u": 1}, "u1")], mon.wrap_fn(g, oom), "t" * 64, "p", guard=g)
    fails = recs(tmp_path, "attempt_fail")
    assert [f["cause"] for f in fails] == ["out_of_memory"]  # the registered cause, not null
    assert len(recs(tmp_path, "attempt_start")) == 1  # the halt refused the retry before an attempt
    g.resolve_incidents("OOM: micro-batch", "chair report 1")
    g.canary(REF)
    H.run_units(
        store, [({"u": 1}, "u1")], mon.wrap_fn(g, lambda r: {"v": 1}), "t" * 64, "p", guard=g
    )
    assert [s["cause"] for s in recs(tmp_path, "attempt_start")] == [None, "out_of_memory"]


def test_r3_non_technical_failure_is_retried_only_after_a_repair_record(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)
    store = H.UnitStore(tmp_path / "u")

    def bug(r):
        raise ValueError("a code defect")

    with pytest.raises(ValueError):
        H.run_units(store, [({"u": 1}, "u1")], mon.wrap_fn(g, bug), "t" * 64, "p", guard=g)
    assert [f["cause"] for f in recs(tmp_path, "attempt_fail")] == [None]
    g.resolve_incidents("code defect diagnosed", "chair report 2")
    g.canary(REF)
    ok = mon.wrap_fn(g, lambda r: {"v": 1})
    with pytest.raises(RG.RepairRequired):
        H.run_units(store, [({"u": 1}, "u1")], ok, "t" * 64, "p", guard=g)
    assert len(recs(tmp_path, "attempt_start")) == 1
    g.retry.repair("code_defect_crash", "fix the defect", "abcdef1", outcome_blind=True)
    H.run_units(store, [({"u": 1}, "u1")], ok, "t" * 64, "p", guard=g)
    assert len(recs(tmp_path, "attempt_start")) == 2


# ------------------------------------------------------------------ R4: incidents never consume attempts


def test_r4_repeated_incident_trips_never_reach_retry_exhaustion_or_the_fallback(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)
    store = H.UnitStore(tmp_path / "u")

    def idle(r):  # the watch abort of an in-flight attempt (GPU idle)
        g.incident("gpu_idle", {"low_for_s": 901})
        raise HL.HealthIncident("gpu_idle", "gpu_idle")

    for i in range(4):  # more trips than the 3-attempt limit
        with pytest.raises(HL.HealthIncident):
            H.run_units(store, [({"u": 1}, "u1")], mon.wrap_fn(g, idle), "t" * 64, "p", guard=g)
        g.resolve_incidents(f"trip {i}", "chair report")
        g.canary(REF)
    assert len(recs(tmp_path, "attempt_aborted")) == 4
    assert not recs(tmp_path, "attempt_fail") and not recs(tmp_path, "fallback")
    out = H.run_units(
        store, [({"u": 1}, "u1")], mon.wrap_fn(g, lambda r: {"v": 9}), "t" * 64, "p", guard=g
    )
    assert out["u1"]["status"] == "pending_canary"  # ran: not exhausted, no fallback
    # an order incident is raised before begin: no attempt at all
    n = len(recs(tmp_path, "attempt_start"))
    with pytest.raises(HL.HealthIncident):
        H.run_units(
            store,
            [({"kind": "ste_train", "u": 2}, "u2")],
            mon.wrap_fn(g, lambda r: {}),
            "t" * 64,
            "p",
            guard=g,
            before_begin=mon.before_begin(g),
        )
    assert len(recs(tmp_path, "attempt_start")) == n and g.open_incidents()[-1]["kind"] == "order"


def test_r4_aborts_do_not_count_but_real_failures_still_do(tmp_path):
    rp = RG.RetryPolicy(tmp_path / "r.jsonl")
    for _ in range(5):
        rp.attempt("u")
        with rp._op():
            rp._append("attempt_aborted", unit="u", n=0, incident="gpu_idle")
    assert rp.attempt("u") == 6  # unique attempt numbers; 0 counted so far
    rp.attempt("u", "out_of_memory")
    with pytest.raises(
        RG.RetryExhausted
    ):  # counted: 6 and 7; 8 is the 3rd counted; the next one exhausts
        rp.attempt("u", "out_of_memory")
        rp.attempt("u", "out_of_memory")


# ------------------------------------------------------------------ R5: C1(b) parser input


class HFTok:
    eos_token_id = 1

    def convert_tokens_to_ids(self, t):
        return 2 if t == "<|im_end|>" else -1

    def encode(self, t, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) for c in t]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i in (1, 2)))

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking=None):
        return f"<{msgs[0]['content']}>"


def test_r5_vllm_gets_hf_token_ids_and_the_text_is_hf_decoded(monkeypatch):
    from wsp.wm.c1 import vllmops as VO

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    seen = {}

    class LLM:
        def generate(self, inputs, sp, use_tqdm=False):
            seen["inputs"], seen["sp"] = inputs, sp
            outs = []
            for x in inputs:
                ids = [104, 105, 2]  # "hi" + <|im_end|>
                c = types.SimpleNamespace(
                    token_ids=ids, text="vLLM-detok-differs", finish_reason="stop"
                )
                outs.append(types.SimpleNamespace(outputs=[c]))
            return outs

    eng = VO.Engine.__new__(VO.Engine)
    eng.llm, eng.tok = LLM(), HFTok()
    tok = HFTok()
    b = GE.vllm_block("x", tok, ["ab", "c"])
    res = eng.gen(
        b["prompts"],
        b["max_tokens"],
        None,
        b["stop_token_ids"],
        b["prompt_token_ids"],
        b["hf_text"],
    )
    # the ids vLLM receives are exactly the HF path's encoding of the chat prompt
    assert seen["inputs"] == [
        {"prompt_token_ids": [ord(c) for c in "<ab>"]},
        {"prompt_token_ids": [ord(c) for c in "<c>"]},
    ]
    assert seen["sp"]["stop_token_ids"] == [1, 2]
    # parser input = HF decode of the ids (trailing stop stripped), not vLLM's own text
    assert [r["text"] for r in res] == ["hi", "hi"] and [r["ids"] for r in res] == [[104, 105]] * 2
    # the stage-W canary path is unchanged (text prompts, vLLM text)
    res2 = eng.gen(["<x>"], 16)
    assert seen["inputs"] == ["<x>"] and res2[0]["text"] == "vLLM-detok-differs"


# ------------------------------------------------------------------ R6: real withdrawal (live worker)


class LiveWorker:
    def __init__(self, root, handlers, sampler=None):
        self.w = W.Worker(root, handlers, poll_s=0.02, sampler=sampler)
        self.th = threading.Thread(target=self.w.serve, daemon=True)
        self.th.start()

    def stop(self):
        (self.w.root / "STOP").write_text("x")
        self.th.join(timeout=10)


def wait_inflight(tr, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = tr.heartbeat() or {}
        if hb.get("inflight"):
            return hb["inflight"]
        time.sleep(0.02)
    raise AssertionError("never in flight")


def test_r6_withdrawal_fences_a_checking_handler_and_the_worker_keeps_serving(tmp_path):
    seen = {"fenced": False}

    def slow(req, ctx):
        try:
            for _ in range(500):
                time.sleep(0.02)
                ctx.check()
        except W.Fenced:
            seen["fenced"] = True
            raise
        return {"done": True}, {}

    def quick(req, ctx):
        return {"ok": 1}, {}

    sampler = HL.Sampler(tmp_path / "w", period_s=0.05, sample=lambda: None)
    tr = T.LocalTransport(tmp_path / "w")
    lw = LiveWorker(tmp_path / "w", {"slow": slow, "quick": quick}, sampler)
    restarts = []
    state = {"armed": False}

    def watch():
        if state["armed"]:
            raise HL.HealthIncident("gpu_idle", "gpu_idle")
        if (tr.heartbeat() or {}).get("inflight"):
            state["armed"] = True

    rem = C.Remote(tr, tmp_path / "ctl", poll_s=0.02, watch=watch, restart=lambda: restarts.append(1),
                   withdraw_grace_s=5.0)  # fmt: skip
    with pytest.raises(HL.HealthIncident):
        rem.call("slow", 1, "i1", "k" * 64, "slow", {})
    assert seen["fenced"] is True and restarts == []
    j = (tmp_path / "ctl" / "dispatch.jsonl").read_text()
    assert '"how": "fenced"' in j
    # the loop survived; the next owner epoch is served
    rem.watch = None
    p, _ = rem.call("quick", 2, "i1", "q" * 64, "quick", {})
    assert p == {"ok": 1}
    lw.stop()


def test_r6_a_handler_that_never_checks_is_killed_and_restarted(tmp_path):
    def stuck(req, ctx):
        time.sleep(1.5)  # a hung GPU op: never reaches ctx.check
        return {"late": True}, {}

    def quick(req, ctx):
        return {"ok": 1}, {}

    sampler = HL.Sampler(tmp_path / "w", period_s=0.05, sample=lambda: None)
    tr = T.LocalTransport(tmp_path / "w")
    lw = LiveWorker(tmp_path / "w", {"stuck": stuck, "quick": quick}, sampler)
    restarts = []
    state = {"armed": False}

    def watch():
        if state["armed"]:
            raise HL.HealthIncident("gpu_idle", "gpu_idle")
        if (tr.heartbeat() or {}).get("inflight"):
            state["armed"] = True

    rem = C.Remote(tr, tmp_path / "ctl", poll_s=0.02, watch=watch, restart=lambda: restarts.append(1),
                   withdraw_grace_s=0.3)  # fmt: skip
    with pytest.raises(HL.HealthIncident):
        rem.call("stuck", 1, "i1", "k" * 64, "stuck", {})
    assert restarts == [1]
    assert '"how": "restarted"' in (tmp_path / "ctl" / "dispatch.jsonl").read_text()
    # the in-thread worker (not killable here) finishes the withdrawn request without crashing its loop
    rem.watch = None
    p, _ = rem.call("quick", 2, "i1", "q" * 64, "quick", {})
    assert p == {"ok": 1} and lw.th.is_alive()
    lw.stop()


def test_r6_worker_tolerates_a_withdrawn_inbox_file(tmp_path):
    w = W.Worker(tmp_path, {})
    assert w.handle(tmp_path / "inbox" / "gone.json") == "withdrawn"


def test_r6_skip_declared_on_another_monitor_is_seen_by_the_running_one(tmp_path):
    """Board A probe P2: Study.run's long-lived monitor and the controller's skip declaration."""
    clk = Clock()
    g = open_guard(tmp_path, clk)
    run_mon = HL.HealthMonitor(tmp_path / "runs")
    for k in ("field_group", "bank_side", "fit", "e2_arm", "sbon_arm", "sealed_eval"):
        run_mon.check_order(g, {"kind": k})
    HL.HealthMonitor(tmp_path / "runs").skip_stage(
        "e1_gen", "E1 not designated"
    )  # controller.stage_gen
    run_mon.check_order(g, {"kind": "ste_calibrate"})  # no false order incident
    run_mon.chunk_summary(g)  # a later save does not erase the declaration
    assert "e1_gen" in json.loads((tmp_path / "runs/health/state.json").read_text())["skipped"]
    assert not g.open_incidents()


def test_r6_controller_declares_the_skip_through_the_study():
    src = (ROOT / "src/wsp/wm/c1/controller.py").read_text()
    assert 'self.study.skip_stage("e1_gen"' in src and "HealthMonitor(self.study.runs)" not in src


# ------------------------------------------------------------------ R7: CPU-kind progress emission


def test_r7_kind_table_covers_every_stage_kind():
    assert set(HL.STAGE_OF) <= set(HL.KIND_CLASS)
    assert {k for k, c in HL.KIND_CLASS.items() if c == "cpu"} >= {"e2_arm", "fit"}


def test_r7_e2_runners_and_world_emit_progress():
    from wsp.wm.c1 import e2arms as EA
    from wsp.wm.c1 import e2ops as EO

    ticks = []
    env = EA.Env.__new__(EA.Env)
    env.rc = types.SimpleNamespace(r_of=lambda b: ([0.0], set()))
    env.lib = types.SimpleNamespace(bundles=lambda p: p)
    env.tick = lambda e, **kw: ticks.append(e)
    env.r_of_pids([1, 2])
    assert ticks == ["e2_r_of"]
    world = EO.DevWorld.__new__(EO.DevWorld)
    world.m = types.SimpleNamespace(
        at_base=lambda: None,
        sess=types.SimpleNamespace(
            apply_bundle=lambda b: None, touched=[], displacement=None, original=None
        ),
        theta_hash=lambda: "h",
        evaluate=lambda items: ({}, None),
        restore=lambda: None,
    )
    world.check = lambda: ticks.append("check")
    world.tick = lambda e, **kw: ticks.append(e)
    world._apply_eval([], {})
    assert ticks[-2:] == ["check", "e2_dev_eval"]


def test_r7_ctx_tick_moves_the_heartbeat_and_throttles_the_file(tmp_path):
    s = HL.Sampler(tmp_path, sample=lambda: None)
    w = W.Worker(tmp_path, {}, sampler=s)
    s.set_inflight({"id": "r", "kind": "e2_arm"})
    ctx = W.Ctx(w, {"id": "r", "kind": "e2_arm", "epoch": 0}, tmp_path)
    t0 = s.inflight["last_progress"]
    time.sleep(0.01)
    for _ in range(50):
        ctx.tick("e2_r_of")
    assert s.inflight["last_progress"] > t0
    lines = (tmp_path / "progress.jsonl").read_text().splitlines()
    assert len(lines) == 1  # one file line for 50 ticks (30 s throttle)


def test_r7_llm_reader_waits_are_controller_side_progress(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)
    events = []

    class Slow:
        def complete(self, system, user, max_output_tokens):
            clk.t += 600.0  # a 10-min LLM wait
            return "ok"

    pc = SD.ProgressClient(
        Slow(), "reader", lambda e, **kw: (events.append(e), mon.note_progress(e)), mon, g
    )

    def unit(recipe):
        for _ in range(3):  # 30 min of LLM calls in total, each completion is progress
            pc.complete("s", "u", 10)
        return {"v": 1}

    assert mon.wrap_fn(g, unit)({"kind": "e2_arm", "label": "wsr:0"}) == {"v": 1}
    assert events == ["llm_completion"] * 3 and not g.open_incidents()

    def stalled(recipe):
        clk.t += 901.0  # no progress for > 15 min, then the next call checks first
        pc.complete("s", "u", 10)

    with pytest.raises(HL.HealthIncident):
        mon.wrap_fn(g, stalled)({"kind": "e2_arm", "label": "wsr:1"})
    assert g.open_incidents()[-1]["kind"] == "no_progress"
    assert (tmp_path / "runs/health/controller_progress.jsonl").exists()


# ------------------------------------------------------------------ R8: the mechanical cap raise


def test_r8_cap_raise_is_mechanical_from_ledger_seconds(tmp_path):
    clk = Clock()
    g = guard(tmp_path / "s", clk, priced={"e2": 120.0, "ste": 20.0})
    g.ledger.add("e2", 109.0 * 3600)  # charged 109 of the 110 cap; priced remainder 11 + 20 = 31
    g.ledger.check_before("e2", 2.0)  # would cross 110: R8 raises instead of stopping
    caps = g.ledger.cap_generations()
    assert len(caps) == 2
    c = caps[1]
    assert c["rule"] == "R8" and c["decided_by"] == "chair" and c["outcome_blind"] is True
    assert c["basis"] == {
        "charged_gpu_h": pytest.approx(109.0),
        "priced_remainder_gpu_h": pytest.approx(31.0),
        "previous_cap_gpu_h": 110.0,
    }
    assert c["cap_gpu_h"] == pytest.approx(110.0 + 1.25 * 31.0)
    assert c["announced_to_user"] is False  # a notification owed, not a gate
    # combined spend + priced remainder above $1,000: no raise, the run stops at the cap
    g2 = guard(tmp_path / "s2", clk, priced={"e2": 400.0})
    g2.ledger.add("e2", 109.0 * 3600)
    with pytest.raises(RG.CapExceeded):
        g2.ledger.check_before("e2", 2.0)
    assert len(g2.ledger.cap_generations()) == 1
    # the ceiling: never above $1,000 at the convention
    basis = {"charged_gpu_h": 150.0, "priced_remainder_gpu_h": 30.0, "previous_cap_gpu_h": 160.0}
    assert RG.r8_cap(basis) == pytest.approx(RG.CAP_USD / RG.USD_PER_GPU_H)


def test_r8_plain_registered_ledger_has_no_raise(tmp_path):
    led = RG.CombinedLedger(tmp_path / "l.jsonl", {"bank": 3.0, "e2": 80.0})
    led.add("e2", 89.5 * 3600)
    with pytest.raises(RG.CapExceeded):
        led.check_before("e2", 0.6)


# ------------------------------------------------------------------ R9: gen.parsers digests


def test_r9_parser_check_passes_on_the_current_file_and_catches_changes(tmp_path):
    ok = PL.parser_check()
    assert ok["pass"] and ok["functions_unchanged"] and ok["file_history"]
    src = (ROOT / "src/wsp/wm/genaudit.py").read_text()
    bad = tmp_path / "g.py"
    bad.write_text(src.replace("def squad_norm(", "def squad_norm(  ", 1))
    r = PL.parser_check(bad)
    assert r["functions_unchanged"] is False and r["pass"] is False
    other = tmp_path / "h.py"
    other.write_text(src + "\n# a non-parser change\n")
    r = PL.parser_check(other)
    assert r["functions_unchanged"] is True and r["file_history"] is None and r["pass"] is False


def test_r9_registered_file_hash_history():
    import subprocess

    for commit, want in (("2737136", "43ec3beb"), ("63ca07f", "52ad2bbd")):
        data = subprocess.run(
            ["git", "show", f"{commit}:src/wsp/wm/genaudit.py"],
            capture_output=True,
            cwd=ROOT,
            check=True,
        ).stdout
        h = hashlib.sha256(data).hexdigest()
        assert h.startswith(want) and h in PL.GENAUDIT_FILES
        assert PL.parser_digests(data.decode()) == PL.PARSER_DIGESTS


def test_r9_post_freeze_chat_change_is_qwen_equivalent():
    """46967f8 (study M) had moved the stop ids and the chat prompt into wsp.wm.chat (since reverted on main by
    d3a2d01); wsp.wm.chat itself remains Qwen-equivalent to the registered code for the pinned tokenizer."""
    from wsp.wm import chat as CH
    from wsp.wm import hf
    from wsp.wm import p3gen as G
    from wsp.wm import panel as Pn
    from wsp.wm.c1 import modelops as MO

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            str(hf.snapshot_dir(MO.REPO, hf.TARGETS[MO.REPO])), local_files_only=True
        )
    except Exception:  # noqa: BLE001
        pytest.skip("the pinned 9B tokenizer is not in this cache")

    def old_chat(u):
        return tok.apply_chat_template(
            [{"role": "user", "content": u}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    users = [G.user_prompt("mmlupro_gen", it) for it in MO.gen_canary_items()]
    assert all(Pn.chat_prompt(tok, u) == old_chat(u) == G.chat(tok, u) for u in users)
    old_stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    assert CH.stop_ids(tok) == old_stop


def test_r7_a_long_worker_subrequest_is_not_a_controller_stall(tmp_path):
    """Progress is noted after every worker sub-request, so a 20-min GPU sub-request does not trip the
    controller-side rule at the next LLM call; the controller rule is never part of Remote.watch."""
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)
    events = []
    progress = lambda e, **kw: (events.append(e), mon.note_progress(e))
    pc = SD.ProgressClient(
        types.SimpleNamespace(complete=lambda *a: "ok"), "reader", progress, mon, g
    )

    def unit(recipe):
        pc.complete("s", "u", 10)
        clk.t += 1200.0  # a 20-min dev_eval sub-request on the worker
        progress("worker_subrequest")
        pc.complete("s", "u", 10)
        return {"v": 1}

    assert mon.wrap_fn(g, unit)({"kind": "e2_arm", "label": "wsr:2"}) == {"v": 1}
    assert not g.open_incidents()
    src = (ROOT / "src/wsp/wm/c1/study.py").read_text()
    assert "remote.watch = lambda: mon.poll(guard)" in src


def test_r10_concurrent_heartbeats_never_fail(tmp_path):
    """Found by the R10 drill repeats: the sampler thread and the worker thread raced on one temp file and the
    worker loop died (FileNotFoundError in set_inflight)."""
    s = HL.Sampler(tmp_path, sample=lambda: None)
    errors = []

    def hammer(k):
        try:
            for i in range(300):
                if k % 2:
                    s.beat()
                else:
                    s.set_inflight({"id": f"r{i}", "kind": "fit"} if i % 2 else None)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ths = [threading.Thread(target=hammer, args=(k,)) for k in range(6)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert errors == []
    json.loads((tmp_path / HL.HEARTBEAT).read_text())
    assert not list(tmp_path.glob(".heartbeat.json*.tmp"))


# ------------------------------------------------------------------ R11 (SYNTHESIS-r13 addendum item 5): global state


def _set_global(det: bool):
    import torch

    torch.use_deterministic_algorithms(det, warn_only=False)
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = not det


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("mode", ["hf_deterministic", "hf_registered"])
def test_r11_every_hf_generate_call_restores_the_previous_global_state(monkeypatch, prior, mode):
    """Another module (e.g. wsp.predictor.models.seed_all) may have switched deterministic mode on globally; an HF
    generate call must leave every global setting exactly as it found it, also when generation raises."""
    import torch

    from wsp.wm import genaudit as GA

    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)
    monkeypatch.setattr(
        GE, "kernel_bindings", lambda: dict.fromkeys(GE.LINEAR_ATTN_FUNCS, GE.QWEN35_MODULE)
    )
    model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa"))
    _set_global(prior)
    torch.backends.cuda.enable_flash_sdp(not prior)  # a non-default SDPA flag set by someone else
    before = GE.global_state()
    inside = {}

    def gen(m, tok, prompts, n, *a, batch=8, device="cuda"):
        inside["det"] = torch.are_deterministic_algorithms_enabled()
        inside["flash"] = torch.backends.cuda.flash_sdp_enabled()
        return [{"ids": [1], "text": "", "hit_max": False}]

    monkeypatch.setattr(GA, "generate_greedy", gen)
    GE.hf_generate(model, None, ["p"], 8, 128, "cpu", mode=mode)
    assert GE.global_state() == before
    if mode == "hf_deterministic":
        assert inside == {"det": True, "flash": False}
    else:
        assert inside == {"det": prior, "flash": not prior}  # the registered path changes nothing

    def boom(*a, **k):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(GA, "generate_greedy", boom)
    with pytest.raises(RuntimeError, match="generation failed"):
        GE.hf_generate(model, None, ["p"], 8, 128, "cpu", mode=mode)
    assert GE.global_state() == before


def test_r11_restore_state_round_trip(monkeypatch):
    monkeypatch.delenv(GE.CUBLAS_ENV, raising=False)
    s0 = GE.global_state()
    _set_global(True)
    import os as _os

    _os.environ[GE.CUBLAS_ENV] = ":16:8"
    assert GE.global_state() != s0
    GE.restore_state(s0)
    assert GE.global_state() == s0 and GE.CUBLAS_ENV not in _os.environ


def test_r11_tests_start_from_a_clean_baseline():
    """The autouse fixture: whatever an earlier test (in any module) left behind, a c1 test starts at the baseline."""
    assert GE.global_state() == GE.BASELINE_STATE


def test_r9_launch_requires_the_registered_generation_file_restored_on_main():
    """main d3a2d01 restored genaudit.py (and panel, p3gen, actions, controls, hf) to the pre-46967f8 bytes: the launch
    check now accepts exactly the registered file 52ad2bbd… (63ca07f); the reverted 46967f8 version no longer passes."""
    import subprocess

    ok = PL.parser_check()
    assert ok["pass"] and ok["file_sha256"] == PL.GENAUDIT_LAUNCH_SHA256
    assert PL.GENAUDIT_LAUNCH_SHA256.startswith("52ad2bbd")
    old = subprocess.run(
        ["git", "show", "46967f8:src/wsp/wm/genaudit.py"], capture_output=True, cwd=ROOT, check=True
    ).stdout
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "genaudit.py"
        p.write_bytes(old)
        r = PL.parser_check(p)
    assert r["functions_unchanged"] is True and r["file_history"] and r["pass"] is False


def test_registered_wm_files_are_at_the_frozen_bytes():
    """N2 (Board A v2n re-check): the C1-registered wm files equal their bytes at the freeze tag
    prereg-wme-v2-frozen (af4face), so the C1 test command covers what study M's test covers."""
    import subprocess

    for f in ("genaudit", "panel", "p3gen", "actions", "controls", "hf"):
        frozen = subprocess.run(
            ["git", "show", f"af4face:src/wsp/wm/{f}.py"], capture_output=True, cwd=ROOT, check=True
        ).stdout
        assert (ROOT / "src/wsp/wm" / f"{f}.py").read_bytes() == frozen, (
            f"{f}.py differs from af4face"
        )
