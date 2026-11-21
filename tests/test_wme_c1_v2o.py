"""Board A re-check of v2n (docs/reviews/council/board-a-prereg-wme-v2n-2026-11-20.md), conditions N1-N6, and the
§II.31.9 registration (P1). CPU only; fixtures."""

import hashlib
import importlib.util
import json
import subprocess
import types
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import health as HL
from wsp.wm.c1 import phasel as PL
from wsp.wm.c1 import study as SD

ROOT = Path(__file__).resolve().parents[1]
REF = "a" * 64
ATTEST = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])
A, B_ = "1" * 64, "2" * 64


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def open_guard(tmp_path, clk):
    g = RG.ProductionGuard(
        tmp_path / "s",
        "confirmatory",
        {"confirmatory": 500.0},
        ATTEST,
        "i1",
        0.01,
        technical=C.technical,
        fallback_value="base",
        clock=clk,
        cap=SD.CAP,
    )
    g.open_instance(1)
    g.canary(REF)
    return g


def recs(tmp_path, t):
    rs = [json.loads(x) for x in (tmp_path / "s" / RG.LOG_NAME).read_text().splitlines()]
    return [r for r in rs if r["type"] == t]


class FakeT:
    def __init__(self, log=""):
        self.log = log

    def heartbeat(self):
        return None

    def read_log(self, off):
        return self.log[off:], len(self.log)


# ------------------------------------------------------------------ N1: no uncounted redraw after completion


def test_n1_a_traceback_after_completion_is_a_counted_failure_not_a_free_redraw(tmp_path):
    """A nondeterministic reader-arm unit completes; a traceback then appears in the worker log. The completed
    payload is discarded as a counted attempt_fail (cause None): halt, repair record, counted retry. No uncounted
    redraw exists, so a reader arm cannot be resampled by post-completion aborts."""
    clk = Clock()
    g = open_guard(tmp_path, clk)
    t = FakeT()
    mon = HL.HealthMonitor(tmp_path / "runs", t, clock=clk)
    store = H.UnitStore(tmp_path / "u")
    draws = iter(range(100))

    def reader_arm(recipe):  # nondeterministic: every draw differs
        k = next(draws)
        if k == 0:
            t.log += "Traceback (most recent call last):\nRuntimeError: reader sub-request\n"
        return {"winner": f"draw-{k}"}

    unit = [({"kind": "e2_arm", "arm": "wsr_reader", "episode": 0}, "wsr:0")]
    fn = mon.wrap_fn(g, reader_arm)
    with pytest.raises(HL.PostCompletionError):
        H.run_units(store, unit, fn, "t" * 64, "p", guard=g)
    fails = recs(tmp_path, "attempt_fail")
    assert [f["cause"] for f in fails] == [None] and not recs(tmp_path, "attempt_aborted")
    assert g.open_incidents()[-1]["kind"] == "worker_error"
    g.resolve_incidents("sub-request traceback diagnosed", "chair report")
    g.canary(REF)
    with pytest.raises(RG.RepairRequired):  # never a free redraw
        H.run_units(store, unit, fn, "t" * 64, "p", guard=g)
    g.retry.repair("code_defect_crash", "reader sub-request fixed", "abcdef1", outcome_blind=True)
    H.run_units(store, unit, fn, "t" * 64, "p", guard=g)
    starts = recs(tmp_path, "attempt_start")
    assert len(starts) == 2  # the redraw is the counted 2nd attempt of 3
    # a traceback inside the payload itself: the same counted disposition
    g.canary(REF)
    unit2 = [({"kind": "e2_arm", "arm": "fugu_team", "episode": 1}, "fugu:1")]
    with pytest.raises(HL.PostCompletionError):
        H.run_units(
            store, unit2, mon.wrap_fn(g, lambda r: {"x": "Traceback (most recent call last): y"}),
            "t" * 64, "p", guard=g,
        )  # fmt: skip
    assert [f["cause"] for f in recs(tmp_path, "attempt_fail")] == [None, None]


def test_n1_mid_flight_incidents_stay_uncounted_aborts(tmp_path):
    """R4 is unchanged for incidents raised while the unit is in flight (no payload exists)."""
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)

    def idle(r):
        g.incident("gpu_idle", {})
        raise HL.HealthIncident("gpu_idle", "gpu_idle")

    with pytest.raises(HL.HealthIncident):
        H.run_units(
            H.UnitStore(tmp_path / "u"), [({"u": 1}, "u1")], mon.wrap_fn(g, idle), "t" * 64, "p",
            guard=g,
        )  # fmt: skip
    assert len(recs(tmp_path, "attempt_aborted")) == 1 and not recs(tmp_path, "attempt_fail")


# ------------------------------------------------------------------ N3: main-instance canaries = parity GPU 0


PAR = {"parity": {"canary_gpu0": {"tf": A, "hf_gen": A}, "vllm_gpu0": A, "vllm_gen_gpu0": A}}


def test_n3_cross_instance_canaries():
    ok = PL.cross_instance_canaries({"tf": A, "hf_gen": A, "vllm": A}, PAR)
    assert ok["pass"] and all(p["equal"] for p in ok["pairs"].values())
    bad = PL.cross_instance_canaries({"tf": A, "hf_gen": B_, "vllm": A}, PAR)
    assert bad["pass"] is False and bad["pairs"]["hf_gen"] == {
        "main": B_,
        "parity_gpu0": A,
        "equal": False,
    }
    assert (
        PL.cross_instance_canaries({"tf": A}, {"parity": {}})["pass"] is False
    )  # missing reference
    assert PL.cross_instance_canaries({"tf": A, "vllm_gen": A, "vllm": A}, PAR)["pass"]  # C1(b) set


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2o", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRemote:
    def __init__(self, payloads):
        self.payloads, self.calls = payloads, []

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        self.calls.append(kind)
        return self.payloads[kind], None


PASSING_CS = {
    "code_identity": {"pass": True},
    "worker_global_state": {"pass": True},
    "gate": {"pass": True},
}


def main_setup(tmp_path, monkeypatch, engines, commit=None):
    mod = load_ctl()
    monkeypatch.setattr(mod, "EVID", tmp_path)
    monkeypatch.setattr(mod, "RUNS", tmp_path / "runs")  # a first launch: no existing study
    monkeypatch.setattr(mod, "STUDY", tmp_path / "runs" / "study")
    ev = {
        "engine_config_sha256": mod.ec_sha(),
        "evaluation": {"proceed": True, "selected_engine": GE.GEN_ENGINE},
        "code_identity": {"controller_commit": commit or PL.git("rev-parse", "HEAD")},
        **PAR,
    }
    (tmp_path / "par.json").write_text(json.dumps(ev))
    rem = FakeRemote(
        {
            "hello": {"gen_engine": GE.config(), "theta0": "t", "versions": {}, "physical": {}},
            "inert_identity": {"pass": True},
            "canary": {"engines": engines},
        }
    )
    monkeypatch.setattr(mod, "remote_for", lambda *a: (None, rem))
    monkeypatch.setattr(mod, "code_and_state", lambda hello, iid: PASSING_CS)
    monkeypatch.setattr(
        C, "composite_canary", lambda e: pytest.fail("composite before the N3 check")
    )
    a = types.SimpleNamespace(iid="X", tag="t", parity_evidence=str(tmp_path / "par.json"))
    return mod, a, rem


def test_n3_phasel_main_refuses_main_canaries_that_differ_from_the_parity_instance(
    tmp_path, monkeypatch
):
    mod, a, rem = main_setup(tmp_path, monkeypatch, {"tf": A, "hf_gen": B_, "vllm": A})
    with pytest.raises(SystemExit, match="main vs parity canaries"):
        mod.cmd_phasel_main(a)
    assert "timing" not in rem.calls  # refused before anything else, before composite_canary


def test_p2_phasel_main_uses_only_parity_evidence_of_its_own_commit(tmp_path, monkeypatch):
    mod, a, rem = main_setup(
        tmp_path, monkeypatch, {"tf": A, "hf_gen": A, "vllm": A}, commit="0" * 40
    )
    with pytest.raises(SystemExit, match="not produced by this commit"):
        mod.cmd_phasel_main(a)
    assert rem.calls == []  # no instance contact


# ------------------------------------------------------------------ N4 (a): the before_begin contract


def test_n4a_a_before_begin_raise_leaves_no_attempt(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    store = H.UnitStore(tmp_path / "u")
    n0 = len(g.events)
    for exc in (HL.HealthIncident("order", "order"), ValueError("any exception")):

        def bb(recipe, label, exc=exc):
            raise exc

        with pytest.raises(type(exc)):
            H.run_units(store, [({"u": 1}, "u1")], lambda r: pytest.fail("fn ran"), "t" * 64, "p",
                        guard=g, before_begin=bb)  # fmt: skip
    assert not recs(tmp_path, "attempt_start") and len(g.events) == n0
    # the production hook only reads/advances the monitor stage and records incidents (no unit operation)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)
    with pytest.raises(HL.HealthIncident):
        mon.before_begin(g)({"kind": "ste_train"}, "x")
    new = [e["type"] for e in g.events[n0:]]
    assert new == ["incident"]


# ------------------------------------------------------------------ N4 (b): code identity and worker globals


def test_n4b_code_identity_needs_a_clean_tree_the_same_commit_and_the_bundle(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "x@y")
    run("config", "user.name", "x")
    (repo / "f.txt").write_text("1")
    run("add", "f.txt")
    run("commit", "-q", "-m", "c")
    head = PL.git("rev-parse", "HEAD", cwd=repo)
    bundle = hashlib.sha256(b"bundle").hexdigest()
    assert PL.code_identity(head, bundle, cwd=repo)["pass"] is True
    assert PL.code_identity("0" * 40, bundle, cwd=repo)["pass"] is False
    assert PL.code_identity(head, None, cwd=repo)["pass"] is False
    (repo / "untracked.json").write_text("{}")  # Phase L evidence before its commit: allowed
    assert PL.code_identity(head, bundle, cwd=repo)["pass"] is True
    (repo / "f.txt").write_text("2")  # a modified tracked file: refused
    r = PL.code_identity(head, bundle, cwd=repo)
    assert r["pass"] is False and r["controller_tree_clean"] is False


def test_n4b_worker_globals_and_hello(monkeypatch):
    want = PL.expected_worker_state("hf_deterministic")
    assert want[GE.CUBLAS_ENV] == ":4096:8" and want["deterministic_algorithms"] is False
    assert PL.worker_state_check(want, "hf_deterministic")["pass"]
    leaked = json.loads(json.dumps(want))
    leaked["deterministic_algorithms"] = True
    assert PL.worker_state_check(leaked, "hf_deterministic")["pass"] is False
    assert PL.expected_worker_state("vllm")[GE.CUBLAS_ENV] is None
    from wsp.wm.c1 import modelops as MO
    from wsp.wm.c1 import serve as S

    monkeypatch.setattr(MO, "physical_identity", dict)
    monkeypatch.setattr(MO, "versions", dict)
    rt = types.SimpleNamespace(
        hf=lambda: types.SimpleNamespace(base_hash="h", load_s=1.0),
        cfg=types.SimpleNamespace(repo="r", smoke=True),
    )
    payload, _ = S.h_hello(rt, {}, None)
    assert payload["global_state"] == GE.global_state() and "code_sha" in payload


# ------------------------------------------------------------------ N6: the gate


def test_n6_gate_records_counts_and_exit_code():
    ok = PL.run_gate(runner=lambda argv, env: (0, "....\n137 passed, 1 skipped in 70.0s\n"))
    assert ok["pass"] and ok["counts"] == {"passed": 137, "skipped": 1} and ok["exit_code"] == 0
    assert any(f.startswith("tests/test_wme_c1_") for f in ok["files"])
    bad = PL.run_gate(runner=lambda argv, env: (1, "F..\n1 failed, 136 passed in 70s\n"))
    assert bad["pass"] is False and bad["counts"]["failed"] == 1
    assert PL.run_gate(runner=lambda argv, env: (0, "no tests ran"))["pass"] is False


def test_n6_up_refuses_before_any_spend(tmp_path, monkeypatch):
    import tarfile

    mod = load_ctl()
    monkeypatch.setattr(
        mod.LO, "deadman", lambda *a, **k: pytest.fail("no dead-man / launch before the gate")
    )
    bundle = tmp_path / "b.tgz"
    stage = tmp_path / "wmec1"
    stage.mkdir()
    (stage / "CODE_SHA").write_text("a" * 40)
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(stage, arcname="wmec1")
    a = types.SimpleNamespace(
        bundle=str(bundle), kind="parity", tag="", t_max_hours=None, poll_max_min=1
    )
    calls = {"git": []}

    def fake_git(*args, cwd=None, head="a" * 40, dirty=""):
        calls["git"].append(args)
        return head if args[0] == "rev-parse" else dirty

    monkeypatch.setattr(PL, "git", lambda *x, **k: fake_git(*x, dirty=" M src/x.py"))
    with pytest.raises(SystemExit, match="uncommitted"):
        mod.cmd_up(a)
    monkeypatch.setattr(PL, "git", lambda *x, **k: fake_git(*x, head="b" * 40))
    with pytest.raises(SystemExit, match="bundle was built from"):
        mod.cmd_up(a)
    monkeypatch.setattr(PL, "git", lambda *x, **k: fake_git(*x))
    monkeypatch.setattr(
        PL, "run_gate", lambda: {"pass": False, "counts": {"failed": 1}, "exit_code": 1}
    )
    with pytest.raises(SystemExit, match="gate failed"):
        mod.cmd_up(a)


def test_n6_ruff_is_clean_so_precommit_reaches_pytest():
    import shutil

    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff not on PATH in this venv")
    for argv in ([ruff, "check", "src", "tests"], [ruff, "format", "--check", "src", "tests"]):
        p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stdout[-2000:]


# ------------------------------------------------------------------ found by the v2o full suite: UnitStore race


def test_unitstore_concurrent_writers_of_one_unit_never_fail(tmp_path):
    """test_wm_dp::test_kill_and_resume failed under full-suite load: after a killed worker's claim expired, two
    resumed workers wrote the same unit through one shared '<key>.tmp' and one os.replace raised. Writers now use
    their own temporary file; the record stays a single verified file."""
    import threading

    store = H.UnitStore(tmp_path / "u")
    errors = []

    def write(k):
        try:
            for _ in range(200):
                store.write("ab" * 32, {"v": 1}, theta_after="t" * 64)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ths = [threading.Thread(target=write, args=(k,)) for k in range(6)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert errors == []
    assert store.read("ab" * 32) == {"v": 1}
    assert not list((tmp_path / "u").glob("**/*.tmp"))
