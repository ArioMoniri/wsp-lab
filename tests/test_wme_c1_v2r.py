"""Chair ruling docs/council/2026-11-21/RULING-wme-c1-bank-halt.md: K1 (HF canaries and every non-unit generation
check in a fresh child process), K2 (a canary or child error is a run-health incident before the halt), K3 (the
outcome-blind, bitwise field repair). CPU only; fixtures."""

import json
import sys
import types
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import fieldrepair as FR
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import hfchild as HC
from wsp.wm.c1 import launchops as LO
from wsp.wm.c1 import protocol as P
from wsp.wm.c1 import serve as S
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import transport as T
from wsp.wm.c1 import worker as W

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


class Ctx:
    def __init__(self, scratch):
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)

    def check(self):
        pass

    def progress(self, *a, **k):
        pass


class NoInProcessModel:
    """The parent's model: any in-process canary or generation is a K1 violation."""

    tok = None

    def __getattr__(self, name):
        if name.startswith("canary") or "generate" in name:
            pytest.fail(f"in-process {name} in the worker parent (K1)")
        raise AttributeError(name)

    def at_base(self):
        pass


class FakeRT:
    def __init__(self, tmp_path, vllm=False):
        self.cfg = types.SimpleNamespace(
            root=tmp_path, smoke=True, repo="Qwen/Qwen3.5-0.8B", vllm=vllm
        )
        self.events = []
        self.m = NoInProcessModel()
        self.manifest = {}

    def hf(self):
        self.events.append("hf")
        return self.m

    def release(self):
        self.events.append("release")


def capture_spawn(monkeypatch, result, events=None):
    jobs = []

    def spawn(j, workdir, **kw):
        if events is not None:
            events.append("spawn")
        jobs.append(j)
        return result(j) if callable(result) else result

    monkeypatch.setattr(HC, "spawn", spawn)
    return jobs


# ------------------------------------------------------------------ K1


def test_k1_canary_runs_in_a_child_after_the_parent_released_its_gpu_state(tmp_path, monkeypatch):
    rt = FakeRT(tmp_path)
    jobs = capture_spawn(monkeypatch, {"canary": {"tf": A, "hf_gen": B_}}, rt.events)
    payload, _ = S.h_canary(rt, {"args": {}}, Ctx(tmp_path / "s"))
    assert payload["engines"] == {"tf": A, "hf_gen": B_}
    assert jobs[0]["ops"] == [{"op": "canary", "hf": True}]
    assert rt.events.index("release") < rt.events.index("spawn")  # parent freed first
    assert payload["detail"]["child"] is True


def test_k1_inert_identity_and_gen_repeat_use_the_child(tmp_path, monkeypatch):
    rt = FakeRT(tmp_path)
    inert = {
        "base": {"tf": A, "hf_gen": B_},
        "rows": [{"spec": {"kind": "i1"}, "theta_is_base": True, "tf_bitwise_base": True,
                  "hf_bitwise_base": True}],
        "vllm_states": [],
    }  # fmt: skip
    jobs = capture_spawn(monkeypatch, lambda j: {"inert": inert})
    payload, _ = LO.h_inert_identity(rt, {"args": {}}, Ctx(tmp_path / "s"))
    assert jobs[0]["ops"][0]["op"] == "inert" and payload["pass"] is True
    assert payload["base"] == {"tf": A, "hf_gen": B_}
    rep = {
        "gen_repeat": {
            "deterministic": [{"sha256": A, "seconds": 10.0}] * 3,
            "registered": {"sha256": B_, "seconds": 8.0},
        }
    }
    jobs = capture_spawn(monkeypatch, rep, rt.events)
    payload, _ = LO.h_gen_repeat(rt, {"args": {"n": 3}}, Ctx(tmp_path / "s2"))
    assert jobs[0]["ops"] == [{"op": "gen_repeat", "n": 3}]
    assert payload["repeat_bitwise"] is True and payload["throughput_ratio"] == pytest.approx(1.25)
    assert rt.events.index("release") < rt.events.index("spawn")


def test_k1_parity_gpu0_canaries_come_from_the_child(tmp_path, monkeypatch):
    rt = FakeRT(tmp_path)
    monkeypatch.setattr(LO, "_chunks", lambda: [0, 1])
    monkeypatch.setattr(LO, "score_chunks", lambda m, idx: {i: f"c{i}" for i in idx})
    g1 = {"chunks": {"0": "c0", "1": "c1"}, "canary": {"tf": A, "hf_gen": B_}}
    monkeypatch.setattr(
        LO.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=json.dumps(g1), stderr=""),
    )  # fmt: skip
    jobs = capture_spawn(monkeypatch, {"canary": {"tf": A, "hf_gen": B_}}, rt.events)
    payload, _ = LO.h_parity(rt, {"args": {}}, Ctx(tmp_path / "s"))
    assert jobs[0]["ops"] == [{"op": "canary", "hf": True}]
    assert payload["canary_gpu0"] == {"tf": A, "hf_gen": B_}
    assert payload["canary_gpu0_eq_gpu1"] == {"tf": True, "hf_gen": True}


def _body(src, name):
    i = src.index(f"def {name}(")
    j = src.find("\ndef ", i + 5)
    k = src.find("\n@register", i + 5)
    ends = [e for e in (j, k) if e != -1]
    return src[i : min(ends)] if ends else src[i:]


def test_k1_no_in_process_hf_canary_remains_in_the_worker_handlers():
    """The worker parent's handlers call no canary / generation in-process: those run in hfchild (the GPU-1 parity
    helper `_gpu1_main` is itself a separate process; `inert_hf_rows` runs inside the child)."""
    calls = ("canary_hf(", "canary_tf(", "canary_hf_timed(", "_hf_at_state(", "hf_generate(")
    serve = (ROOT / "src/wsp/wm/c1/serve.py").read_text()
    launch = (ROOT / "src/wsp/wm/c1/launchops.py").read_text()
    for src, name in ((serve, "h_canary"), (launch, "h_inert_identity"), (launch, "h_parity"),
                      (launch, "h_gen_repeat")):  # fmt: skip
        body = _body(src, name)
        assert "HC.spawn(" in body, name
        for c in calls:
            assert c not in body, f"{c} in {name}"


def test_k1_child_runs_the_requested_operations(tmp_path, monkeypatch):
    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)

    class M:
        def canary_tf(self, items):
            return A

        def canary_hf(self, items):
            return B_

        def canary_hf_timed(self, items, mode=None):
            return (A if mode == "hf_deterministic" else B_), 1.0

    from wsp.wm.c1 import modelops as MO

    monkeypatch.setattr(MO, "tf_canary_items", lambda *a: {})
    monkeypatch.setattr(MO, "gen_canary_items", lambda *a: [])
    monkeypatch.setattr(
        LO, "inert_hf_rows", lambda rt, scratch: {"rows": [], "base": {}, "vllm_states": []}
    )
    rt = types.SimpleNamespace(hf=lambda: M())
    out = HC.run_job(
        {"ops": [{"op": "canary", "hf": True}, {"op": "gen_repeat", "n": 2},
                 {"op": "inert", "scratch": str(tmp_path)}]},
        runtime=rt,
    )  # fmt: skip
    assert out["canary"] == {"tf": A, "hf_gen": B_}
    assert [r["sha256"] for r in out["gen_repeat"]["deterministic"]] == [A, A]
    assert out["gen_repeat"]["registered"]["sha256"] == B_ and "inert" in out
    with pytest.raises(ValueError):
        HC.job(types.SimpleNamespace(cfg=types.SimpleNamespace(root=tmp_path, smoke=False, repo="r")),
               [{"op": "generate_anything"}])  # fmt: skip


def test_k1_a_child_failure_reaches_the_parent_with_its_cause(tmp_path):
    argv = [sys.executable, "-c",
            "import sys; sys.stderr.write('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate'); sys.exit(1)"]  # fmt: skip
    with pytest.raises(RuntimeError, match="CUDA out of memory") as ei:
        HC.spawn({"ops": []}, tmp_path / "c", argv=argv)
    assert W.classify(ei.value) == "out_of_memory"
    assert "CUDA out of memory" in (tmp_path / "c" / "hf_child.log").read_text()


# ------------------------------------------------------------------ K2


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def open_guard(tmp_path, clk, phase="bank"):
    g = RG.ProductionGuard(
        tmp_path / "s", phase, {phase: 100.0}, ATTEST, "i1", 0.01, technical=C.technical,
        fallback_value="base", clock=clk, cap=SD.CAP,
    )  # fmt: skip
    g.open_instance(1)
    g.canary(REF)
    return g


class FailingRemote:
    def call(self, kind, *a, **k):
        raise P.WorkerError("OutOfMemoryError", "CUDA out of memory (HF child)", "out_of_memory")


def test_k2_a_canary_error_is_a_run_health_incident_before_the_halt(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    with pytest.raises(P.WorkerError):
        C.run_canary(g, FailingRemote())
    inc = g.open_incidents()
    assert [i["kind"] for i in inc] == ["worker_error"]
    assert inc[0]["detail"]["where"] == "canary" and "out of memory" in inc[0]["detail"]["message"]
    with pytest.raises(RG.HealthHalt):  # nothing is issued until the incident is resolved
        H.run_units(
            H.UnitStore(tmp_path / "u"), [({"u": 1}, "u1")], lambda r: {}, "t" * 64, "p", guard=g
        )


def test_k2_the_chunk_boundary_canary_path_records_it_too(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    canary = lambda: C.run_canary(g, FailingRemote())
    with pytest.raises(P.WorkerError):
        C.run_bracketed(
            g, H.UnitStore(tmp_path / "u"), [({"u": 1}, "u1")], lambda r: {}, "t" * 64, "p", canary
        )
    assert g.open_incidents()[-1]["detail"]["where"] == "canary"


# ------------------------------------------------------------------ K3


def field_payloads():
    out = {}
    for gi in range(8):
        out[f"field:g{gi}"] = {
            "group": gi,
            "manifest_sha256": f"m{gi}" * 8,
            "file_sha256": {f"obj.g{gi}.pt": f"f{gi}" * 8},
            "kept_sha256": {f"obj.g{gi}.kept.pt": f"k{gi}" * 8},
            "collect_seconds": 1.0,
        }
    out["field:vocab"] = {"manifest_sha256": "v" * 64, "seconds": 2.0}
    return out


def test_k3_plan_verifies_data_hashes_and_routes_the_vocab_unit_to_k3b():
    pl = field_payloads()
    plan = FR.plan(
        {k: ("field_vocab" if k.endswith("vocab") else "field_group", v) for k, v in pl.items()}
    )
    assert plan["indirect"] == ["field:vocab"] and plan["unverifiable"] == []
    assert plan["expected"]["field:g3"] == {
        "file_sha256": pl["field:g3"]["file_sha256"],
        "kept_sha256": pl["field:g3"]["kept_sha256"],
    }
    newer = dict(pl["field:vocab"], file_sha256={"obj.vocab.pt": "z" * 64})
    assert FR.expected("field_vocab", newer) == {"file_sha256": {"obj.vocab.pt": "z" * 64}}
    bad = dict(pl["field:g2"])
    del bad["kept_sha256"]
    assert FR.plan({"field:g2": ("field_group", bad)})["unverifiable"] == ["field:g2"]


def test_k3b_vocab_dependency_is_read_from_the_code_path():
    roots = {"row_scale:002": {"family": "row_scale"}, "steer:005": {"family": "steer"}}
    side = lambda r: {"kind": "bank_side", "args": {"root": r}}
    assert FR.vocab_dependent(side("row_scale:002"), roots) is True  # edits embed_tokens rows
    assert FR.vocab_dependent(side("steer:005"), roots) is False  # a layer writer only
    assert (
        FR.vocab_dependent({"kind": "bank_inert", "args": {"spec": {"kind": "i2"}}}, roots) is True
    )
    assert (
        FR.vocab_dependent({"kind": "bank_inert", "args": {"spec": {"kind": "i1"}}}, roots) is False
    )
    comp = {"kind": "bank_comp", "args": {"order": ["steer:005", "row_scale:002"]}}
    assert FR.vocab_dependent(comp, roots) is True


def test_k3b_timing_free_hash_ignores_only_timings():
    a = {"r_seconds": 1.0, "states": [{"R": [0.1, 0.2], "eval_seconds": 3.0, "h": 1}]}
    b = {"r_seconds": 9.0, "states": [{"R": [0.1, 0.2], "eval_seconds": 7.0, "h": 1}]}
    c = {"r_seconds": 1.0, "states": [{"R": [0.1, 0.2000000001], "eval_seconds": 3.0, "h": 1}]}
    assert FR.timing_free_sha(a) == FR.timing_free_sha(b) != FR.timing_free_sha(c)


ROOTS = {
    "roots": [
        {"id": "scale:001", "family": "scale"},
        {"id": "row_scale:002", "family": "row_scale", "rows": [1, 2]},
    ],
    "controls": [],
}
MAN = {"bank.blocks": {"blocks": []}}
BASE = {"values": {"wikitext_nll": [1.0, 2.0]}, "sketch": {}, "theta_hash": "t" * 64}


def side_payload(root, side, k=0):
    return {
        "root": root,
        "doses": [0.25, 0.5],
        "r_seconds": 12.0 + k,
        "states": [{"h": 1, "R": [0.5, -0.25], "R_flags": [], "eval_seconds": 3.0 + k,
                    "values": {"wikitext_nll": [1.0]}, "theta_hash": "s" * 64}],
    }  # fmt: skip


def make_study(tmp_path):
    st = SD.Study(
        tmp_path / "study", tmp_path / "runs", ATTEST, "t" * 64, "panel", log=lambda *a: None
    )
    st.priced = {"bank": 100.0, "confirmatory": 100.0, "sbon": 100.0, "ste": 100.0}
    return st


def eligible_bank(st, tr, inst, field_pl, with_dependent=True):
    from wsp.wm.c1 import bankops as BO

    g = st.guard("bank", inst)
    tr.set_fence(g.epoch, inst)
    g.open_instance(1)
    g.canary(REF)
    units = SD.bank_units({}, {}, range(8), True)
    units.append(({"kind": "bank_base", "args": {}, "label": "bank:base"}, "bank:base"))
    pls = dict(field_pl, **{"bank:base": BASE})
    H.run_units(st.store("bank"), units, lambda r: pls[r["label"]], st.theta0, st.panel, guard=g)
    g.canary(REF)
    old = BO.inert_specs
    BO.inert_specs = lambda man: []
    try:
        su = SD.bank_state_units(MAN, ROOTS, BASE, None, True)
    finally:
        BO.inert_specs = old
    done = [u for u in su if u[1].startswith("side:scale")]
    if with_dependent:
        done += [u for u in su if u[1] == "side:row_scale:002:pos"]
    H.run_units(
        st.store("bank"), done,
        lambda r: side_payload(r["args"]["root"], r["label"].rsplit(":", 1)[1]),
        st.theta0, st.panel, guard=g,
    )  # fmt: skip
    g.canary(REF)


class RepairRemote:
    def __init__(self, payloads, tamper=None, tamper_r=False):
        self.payloads, self.tamper, self.tamper_r, self.calls = payloads, tamper, tamper_r, []

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        self.calls.append((kind, label, args))
        if kind == "bank_verify":
            p = side_payload(args["args"]["root"], "pos", k=5)  # different timings, same data
            if self.tamper_r:
                p["states"][0]["R"][0] = 0.5000001
            return p, None
        p = json.loads(json.dumps(self.payloads[label]))
        p["manifest_sha256"] = (
            "different-timing" * 4
        )  # the manifest embeds wall times: never compared
        if label == "field:vocab":
            p["file_sha256"] = {
                "obj.vocab.pt": "n" * 64
            }  # re-collected vocab: recorded in the repair
        if self.tamper == label:
            p["kept_sha256"] = {"obj.kept.pt": "x" * 64}
        return p, None


def records(tmp_path):
    return [json.loads(x) for x in (tmp_path / "study" / RG.LOG_NAME).read_text().splitlines()]


def controller(tmp_path, st, rem, tr, inst):
    from wsp.wm.c1 import controller as CO

    return CO.Controller(st, rem, tr, inst, 1, log=lambda *a: None)


@pytest.fixture
def no_inert(monkeypatch):
    from wsp.wm.c1 import bankops as BO

    monkeypatch.setattr(BO, "inert_specs", lambda man: [])


def test_k3b_vocab_is_verified_through_a_completed_dependent_unit(tmp_path, no_inert):
    st, tr, inst = make_study(tmp_path), T.LocalTransport(tmp_path / "w"), "lambda:x"
    pl = field_payloads()  # the vocab payload records only its timing-bearing manifest hash
    eligible_bank(st, tr, inst, pl)
    before = records(tmp_path)
    rem = RepairRemote(pl)
    out = controller(tmp_path, st, rem, tr, inst).repair_field(MAN, ROOTS, commit="abcdef1")
    kinds = [c[0] for c in rem.calls]
    assert kinds == ["field_repair"] * 9 + ["bank_verify"]
    assert (
        rem.calls[-1][1] == "verify:side:row_scale:002:pos"
    )  # the earliest completed dependent unit
    v = rem.calls[-1][2]
    assert v["kind"] == "bank_side" and v["args"]["root"] == "row_scale:002"
    assert v["args"]["base_wikitext"] == [
        1.0,
        2.0,
    ]  # the recorded unit's exact args (unit key asserted)
    assert out["verified"] == 8 and out["vocab"]["verifying_unit"] == "side:row_scale:002:pos"
    new = [r["type"] for r in records(tmp_path)[len(before) :]]
    assert "attempt_start" not in new and "attempt_ok" not in new  # no unit result
    rep = [r for r in records(tmp_path) if r["type"] == "repair"]
    assert len(rep) == 1 and "side:row_scale:002:pos" in rep[0]["description"]


def test_k3b_a_dependent_unit_mismatch_halts_and_substitutes_nothing(tmp_path, no_inert):
    st, tr, inst = make_study(tmp_path), T.LocalTransport(tmp_path / "w"), "lambda:x"
    pl = field_payloads()
    eligible_bank(st, tr, inst, pl)
    rem = RepairRemote(pl, tamper_r=True)
    with pytest.raises(FR.FieldRepairMismatch, match="side:row_scale:002:pos"):
        controller(tmp_path, st, rem, tr, inst).repair_field(MAN, ROOTS, commit="abcdef1")
    inc = [r for r in records(tmp_path) if r["type"] == "incident"]
    assert inc[-1]["kind"] == "field_repair_mismatch"
    assert inc[-1]["detail"]["label"] == "side:row_scale:002:pos"
    assert not [r for r in records(tmp_path) if r["type"] == "repair"]


def test_k3b_no_completed_dependent_unit_is_an_incident_before_any_gpu_work(tmp_path, no_inert):
    st, tr, inst = make_study(tmp_path), T.LocalTransport(tmp_path / "w"), "lambda:x"
    pl = field_payloads()
    eligible_bank(st, tr, inst, pl, with_dependent=False)
    rem = RepairRemote(pl)
    with pytest.raises(FR.FieldRepairUnverifiable, match="no completed"):
        controller(tmp_path, st, rem, tr, inst).repair_field(MAN, ROOTS, commit="abcdef1")
    assert rem.calls == []
    assert [r for r in records(tmp_path) if r["type"] == "incident"][-1]["kind"] == (
        "field_repair_unverifiable"
    )


def test_k3_a_field_group_mismatch_halts(tmp_path, no_inert):
    st, tr, inst = make_study(tmp_path), T.LocalTransport(tmp_path / "w"), "lambda:x"
    pl = field_payloads()
    eligible_bank(st, tr, inst, pl)
    rem = RepairRemote(pl, tamper="field:g4")
    with pytest.raises(FR.FieldRepairMismatch, match="field:g4"):
        controller(tmp_path, st, rem, tr, inst).repair_field(MAN, ROOTS, commit="abcdef1")
    assert [c[0] for c in rem.calls].count("bank_verify") == 0
    assert not [r for r in records(tmp_path) if r["type"] == "repair"]


def test_k3_a_direct_vocab_hash_needs_no_dependent_unit(tmp_path, no_inert):
    st, tr, inst = make_study(tmp_path), T.LocalTransport(tmp_path / "w"), "lambda:x"
    pl = field_payloads()
    pl["field:vocab"]["file_sha256"] = {
        "obj.vocab.pt": "n" * 64
    }  # future payloads (v2r) carry data hashes
    eligible_bank(st, tr, inst, pl, with_dependent=False)
    rem = RepairRemote(pl)
    out = controller(tmp_path, st, rem, tr, inst).repair_field(MAN, ROOTS, commit="abcdef1")
    assert out["verified"] == 9 and "bank_verify" not in [c[0] for c in rem.calls]


def test_k3_worker_field_repair_request_is_registered_and_gpu_class():
    from wsp.wm.c1 import health as HL

    for k in ("field_repair", "bank_verify"):
        assert k in S.HANDLERS and HL.KIND_CLASS[k] == "gpu"
        assert k not in HL.STAGE_OF  # never a unit kind


def test_k3_new_vocab_payloads_record_their_file_hashes():
    src = (ROOT / "src/wsp/wm/c1/serve.py").read_text()
    body = src[src.index("def h_field_vocab(") : src.index("def h_bank_base(")]
    assert '"file_sha256"' in body
