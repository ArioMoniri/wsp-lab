"""Board A quick check of v2r (docs/reviews/council/board-a-wme-c1-v2r-2026-11-21.md): B1 (phasel-main --resume keeps
the persisted contract; a plain phasel-main refuses on an existing study before any write; attestation.json is never
rewritten), C2 (hf_gen == the committed reference at parity and at resume), C3 (K3b over every dependent unit, row
coverage, R-flag watch), C4 (quarantine of a mismatched field), C5 (the field gate). CPU only; fixtures."""

import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import fieldrepair as FR
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import health as HL
from wsp.wm.c1 import phasel as PL
from wsp.wm.c1 import protocol as P
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import transport as T

ROOT = Path(__file__).resolve().parents[1]
REF_HF = PL.COMMITTED_HF
ENGINES = {"tf": "1" * 64, "hf_gen": REF_HF, "vllm": "3" * 64}
COMPOSITE = C.composite_canary(ENGINES)


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2s", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ C2


def checks(g0=REF_HF, g1=REF_HF):
    return {
        "parity": {
            "parity_bitwise": True,
            "gpu0_vs_gpu1_every_chunk_bitwise": True,
            "canary_gpu0": {"tf": "1" * 64, "hf_gen": g0},
            "canary_gpu1": {"tf": "1" * 64, "hf_gen": g1},
            "vllm_gpu0_eq_gpu1": True,
            "vllm_gpu0": "3" * 64,
        },
        "inert_identity": {
            "pass": True,
            "rows": [{"spec": {"kind": "i1"}, "hf_sha256": g0, "hf_bitwise_base": True}],
            "vllm_rows": [{"vllm_bitwise_base": True}],
        },
        "gen_repeat": {
            "deterministic": [{"sha256": g0, "seconds": 10.0}] * 3,
            "registered": {"sha256": "r" * 64, "seconds": 9.0},
            "t_det_median_s": 10.0,
        },
    }


def test_c2_parity_must_reproduce_the_committed_hf_gen_reference():
    ok = PL.evaluate(checks(), "hf_deterministic", 4.0, 110.0, 0.545, committed_hf=REF_HF)
    assert ok["pass_rule"]["hf_gen_eq_committed"] is True and ok["proceed"] is True
    drift = "d" * 64  # a drifted but self-consistent pair of GPUs (3/3 and GPU0 == GPU1)
    bad = PL.evaluate(
        checks(drift, drift), "hf_deterministic", 4.0, 110.0, 0.545, committed_hf=REF_HF
    )
    assert bad["pass_rule"]["repeat_3_of_3"] and bad["pass_rule"]["gpu0_eq_gpu1"]
    assert bad["pass_rule"]["hf_gen_eq_committed"] is False and bad["proceed"] is False
    first = PL.evaluate(
        checks(drift, drift), "hf_deterministic", 4.0, 110.0, 0.545
    )  # first launch: no reference
    assert "hf_gen_eq_committed" not in first["pass_rule"]


def test_c2_committed_references_come_from_the_attestation_evidence(tmp_path):
    f = tmp_path / "docs/reviews/wme/wme-c1-canary-refs-x.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"engines": ENGINES}))
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    att = {
        "evidence": [f"docs/reviews/wme/wme-c1-canary-refs-x.json sha256:{h}"],
        "canary_reference": COMPOSITE,
    }
    assert PL.committed_references(att, tmp_path)["engines"] == ENGINES
    f.write_text(json.dumps({"engines": dict(ENGINES, tf="9" * 64)}))  # tampered
    with pytest.raises(RuntimeError, match="sha256"):
        PL.committed_references(att, tmp_path)


def test_c2_resume_canaries_need_the_committed_references_and_the_parity_instance():
    par = {"parity": {"canary_gpu0": {"tf": "1" * 64, "hf_gen": REF_HF}, "vllm_gpu0": "3" * 64}}
    ok = PL.resume_canaries(ENGINES, par, ENGINES, COMPOSITE)
    assert ok["pass"] and ok["hf_gen_eq_committed"] and ok["composite_eq_persisted"]
    drift = dict(ENGINES, hf_gen="d" * 64)
    par_d = {"parity": {"canary_gpu0": {"tf": "1" * 64, "hf_gen": "d" * 64}, "vllm_gpu0": "3" * 64}}
    bad = PL.resume_canaries(drift, par_d, ENGINES, COMPOSITE)  # main == new parity, both drifted
    assert bad["pass"] is False and bad["hf_gen_eq_committed"] is False


# ------------------------------------------------------------------ B1


ATTEST = RG.LaunchAttestation(
    True, True, COMPOSITE, "c" * 64, ["docs/reviews/wme/refs.json sha256:" + "0" * 64]
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def existing_study(tmp_path, monkeypatch):
    """A first-launch study: attestation.json, a log with 2 eligible bank units on an old (closed) instance."""
    mod = load_ctl()
    runs = tmp_path / "runs"
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "STUDY", runs / "study")
    monkeypatch.setattr(mod, "EVID", tmp_path / "evid")
    runs.mkdir(parents=True)
    (runs / "attestation.json").write_text(json.dumps(RG._att_fields(ATTEST), indent=1))
    st = SD.Study(runs / "study", runs, ATTEST, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    g = st.guard("bank", "lambda:old")
    g.open_instance(1)
    g.canary(COMPOSITE)
    units = [
        ({"kind": "field_group", "args": {"group": i}, "label": f"field:g{i}"}, f"field:g{i}")
        for i in (0, 1)
    ]
    H.run_units(
        st.store("bank"), units, lambda r: {"v": r["args"]["group"]}, st.theta0, st.panel, guard=g
    )
    g.canary(COMPOSITE)
    g.close_instance()
    return mod, runs, st, units


def statuses(runs):
    s = RG.read_study(runs / "study")
    return sorted(s.status(a) for v in s.ok.values() for a in v), s.contract, s.generation


def test_b1_plain_phasel_main_refuses_on_an_existing_study_before_any_write(tmp_path, monkeypatch):
    mod, runs, _st, _u = existing_study(tmp_path, monkeypatch)
    before = (runs / "attestation.json").read_bytes()
    log_before = (runs / "study" / RG.LOG_NAME).read_bytes()
    monkeypatch.setattr(mod, "remote_for", lambda *a: pytest.fail("no instance contact"))
    a = types.SimpleNamespace(iid="X", tag="t", parity_evidence="p.json", resume=False)
    with pytest.raises(SystemExit, match="--resume"):
        mod.cmd_phasel_main(a)
    assert (runs / "attestation.json").read_bytes() == before
    assert (runs / "study" / RG.LOG_NAME).read_bytes() == log_before
    with pytest.raises(SystemExit, match="never rewrites"):
        mod.write_attestation_once(ATTEST)
    assert (runs / "attestation.json").read_bytes() == before


class FakeRemote:
    def __init__(self, payloads):
        self.payloads, self.calls = payloads, []

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        self.calls.append(kind)
        p = self.payloads[kind]
        return (p() if callable(p) else p), None


def resume_setup(tmp_path, monkeypatch, engines):
    mod, runs, _st, _units = existing_study(tmp_path, monkeypatch)
    par = {
        "engine_config_sha256": "c" * 64,
        "code_identity": {"controller_commit": "a" * 40},
        "evaluation": {"proceed": True, "selected_engine": GE.GEN_ENGINE, "pass_rule": {"hf_gen_eq_committed": True}},
        "parity": {"canary_gpu0": {"tf": "1" * 64, "hf_gen": REF_HF}, "vllm_gpu0": "3" * 64},
    }  # fmt: skip
    (tmp_path / "par.json").write_text(json.dumps(par))
    monkeypatch.setattr(mod, "ec_sha", lambda: "c" * 64)
    monkeypatch.setattr(PL, "git", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(PL, "committed_references", lambda att, root=None: {"engines": ENGINES})
    monkeypatch.setattr(PL, "parser_check", lambda: {"pass": True})
    monkeypatch.setattr(mod, "MO_theta0", lambda: "t" * 64)
    monkeypatch.setattr(
        mod, "code_and_state", lambda hello, iid, parity=None: {"code_identity": {"pass": True}}
    )
    monkeypatch.setattr(mod, "require_code_and_state", lambda cs: None)
    commits = []
    monkeypatch.setattr(mod, "commit_evidence", lambda paths, msg: commits.append(paths))
    dm = tmp_path / "dm"
    dm.mkdir()
    (dm / "state.json").write_text(json.dumps({"instances": {}}))
    monkeypatch.setattr(mod.LO, "STATE", dm)
    rem = FakeRemote(
        {
            "hello": {"gen_engine": GE.config(), "theta0": "t", "versions": {}, "physical": {}},
            "inert_identity": {"pass": True},
            "canary": {"engines": engines},
            "timing": {"eigh_4096_s": 1.0},
        }
    )
    tr = T.LocalTransport(tmp_path / "w")
    monkeypatch.setattr(mod, "remote_for", lambda kind, iid: (tr, rem))
    a = types.SimpleNamespace(
        iid="NEW", tag="r1", parity_evidence=str(tmp_path / "par.json"), resume=True
    )
    return mod, runs, a, rem, commits


def test_b1_resume_keeps_the_contract_and_every_eligible_unit(tmp_path, monkeypatch):
    mod, runs, a, rem, commits = resume_setup(tmp_path, monkeypatch, ENGINES)
    before_att = (runs / "attestation.json").read_bytes()
    st0, contract0, gen0 = statuses(runs)
    assert st0 == ["eligible", "eligible"]
    mod.cmd_phasel_main(a)
    assert (runs / "attestation.json").read_bytes() == before_att  # never rewritten
    assert list(runs.glob("attestation.json.bak-*"))  # backed up first
    st1, contract1, gen1 = statuses(runs)
    assert (
        st1 == ["eligible", "eligible"] and contract1 == contract0 and gen1 == gen0
    )  # no transition
    recs = [json.loads(x) for x in (runs / "study" / RG.LOG_NAME).read_text().splitlines()]
    assert not [r for r in recs if r["type"] == "contract_transition"]
    rep = [r for r in recs if r["type"] == "repair"]
    assert len(rep) == 1 and "persisted contract" in rep[0]["description"]
    assert [r for r in recs if r["type"] == "instance_open" and r["instance"] == "lambda:NEW"]
    cans = [r for r in recs if r["type"] == "canary" and r["instance"] == "lambda:NEW"]
    assert cans and cans[-1]["sha"] == COMPOSITE  # the start canary against the persisted reference
    assert commits and rem.calls == ["hello", "inert_identity", "canary", "timing", "canary"]


def test_b1_a_failed_resume_check_leaves_attestation_json_unchanged(tmp_path, monkeypatch):
    drift = dict(ENGINES, hf_gen="d" * 64)
    mod, runs, a, _rem, commits = resume_setup(tmp_path, monkeypatch, drift)
    before_att = (runs / "attestation.json").read_bytes()
    with pytest.raises(SystemExit, match="INCIDENT"):
        mod.cmd_phasel_main(a)
    assert (runs / "attestation.json").read_bytes() == before_att
    recs = [json.loads(x) for x in (runs / "study" / RG.LOG_NAME).read_text().splitlines()]
    assert [r["kind"] for r in recs if r["type"] == "incident"] == ["canary_mismatch"]
    assert not [r for r in recs if r["type"] == "repair"] and not commits
    assert statuses(runs)[0] == ["eligible", "eligible"]


def test_b1_resume_refuses_a_contract_or_engine_config_drift_offline(tmp_path, monkeypatch):
    mod, _runs, a, rem, _commits = resume_setup(tmp_path, monkeypatch, ENGINES)
    monkeypatch.setattr(mod, "ec_sha", lambda: "e" * 64)
    with pytest.raises(SystemExit, match="engine config"):
        mod.cmd_phasel_main(a)
    assert rem.calls == []


# ------------------------------------------------------------------ C3, C4, C5 (field repair / gate)


ROOTS = {
    "roots": [
        {"id": "scale:001", "family": "scale"},
        {"id": "row_scale:002", "family": "row_scale", "rows": [1, 2]},
        {"id": "row_scale:003", "family": "row_scale", "rows": [3, 4]},
    ],
    "controls": [],
}
MAN = {"bank.blocks": {"blocks": []}, "controls.inert": {"I2_rows": [10, 12]}}
BASE = {"values": {"wikitext_nll": [1.0, 2.0]}}


def field_payloads(vocab_hash=False):
    out = {
        f"field:g{gi}": {"group": gi, "file_sha256": {f"g{gi}.pt": f"f{gi}" * 8},
                         "kept_sha256": {f"g{gi}.kept.pt": f"k{gi}" * 8}}
        for gi in range(8)
    }  # fmt: skip
    out["field:vocab"] = {"manifest_sha256": "v" * 64}
    if vocab_hash:
        out["field:vocab"]["file_sha256"] = {"vocab.pt": "n" * 64}
    return out


def side_payload(root, k=0):
    return {"root": root, "r_seconds": 1.0 + k,
            "states": [{"h": 1, "R": [0.5], "R_flags": [], "eval_seconds": 2.0 + k}]}  # fmt: skip


@pytest.fixture
def no_inert(monkeypatch):
    from wsp.wm.c1 import bankops as BO

    monkeypatch.setattr(BO, "inert_specs", lambda man: [])


def make_bank(tmp_path, inst="lambda:x", dependents=("side:row_scale:002:pos", "side:row_scale:003:neg"),
              vocab_hash=False):  # fmt: skip
    st = SD.Study(
        tmp_path / "study", tmp_path / "runs", ATTEST, "t" * 64, "panel", log=lambda *a: None
    )
    st.priced = {"bank": 100.0, "confirmatory": 100.0, "sbon": 100.0, "ste": 100.0}
    tr = T.LocalTransport(tmp_path / "w")
    g = st.guard("bank", inst)
    tr.set_fence(g.epoch, inst)
    g.open_instance(1)
    g.canary(COMPOSITE)
    pl = field_payloads(vocab_hash)
    units = SD.bank_units({}, {}, range(8), True) + [
        ({"kind": "bank_base", "args": {}, "label": "bank:base"}, "bank:base")
    ]
    pls = dict(pl, **{"bank:base": BASE})
    H.run_units(st.store("bank"), units, lambda r: pls[r["label"]], st.theta0, st.panel, guard=g)
    g.canary(COMPOSITE)
    su = [u for u in SD.bank_state_units(MAN, ROOTS, BASE, None, True) if u[1] in dependents]
    H.run_units(
        st.store("bank"),
        su,
        lambda r: side_payload(r["args"]["root"]),
        st.theta0,
        st.panel,
        guard=g,
    )
    g.canary(COMPOSITE)
    return st, tr, pl


class RepairRemote:
    def __init__(self, payloads, tamper_verify=None, disk=None):
        self.payloads, self.tamper_verify, self.calls = payloads, tamper_verify, []
        self.disk = disk

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        self.calls.append((kind, label))
        if kind == "field_quarantine":
            return {"moved": {"g0": {"file_sha256": {}}}, "dest": "/q/x"}, None
        if kind == "field_status":
            return {"field": self.disk}, None
        if kind == "bank_verify":
            p = side_payload(args["args"]["root"], k=3)
            if self.tamper_verify and label.endswith(self.tamper_verify):
                p["states"][0]["R"] = [0.5000001]
            return p, None
        p = json.loads(json.dumps(self.payloads[label]))
        if label == "field:vocab":
            p["file_sha256"] = {"vocab.pt": "n" * 64}
        return p, None


def controller(st, rem, tr, inst="lambda:x"):
    from wsp.wm.c1 import controller as CO

    return CO.Controller(st, rem, tr, inst, 1, log=lambda *a: None)


def recs(tmp_path):
    return [json.loads(x) for x in (tmp_path / "study" / RG.LOG_NAME).read_text().splitlines()]


def test_c3_k3b_verifies_every_dependent_unit_and_records_row_coverage(tmp_path, no_inert):
    st, tr, pl = make_bank(tmp_path)
    rem = RepairRemote(pl)
    out = controller(st, rem, tr).repair_field(MAN, ROOTS, commit="abcdef1")
    ver = [lab for k, lab in rem.calls if k == "bank_verify"]
    assert ver == ["verify:side:row_scale:002:pos", "verify:side:row_scale:003:neg"]
    cov = out["vocab"]["coverage"]
    assert cov == {
        "covered_rows": 4,
        "declared_rows": 6,
        "partial": True,
    }  # rows 1-4 of {1,2,3,4,10,11}
    rep = [r for r in recs(tmp_path) if r["type"] == "repair"]
    assert (
        "side:row_scale:002:pos" in rep[0]["description"]
        and "side:row_scale:003:neg" in rep[0]["description"]
    )
    assert "PARTIAL" in rep[0]["description"]
    assert st.vocab_partially_verified() is True


def test_c3_after_a_partial_check_a_later_r_flag_is_a_counted_incident(tmp_path):
    clk = Clock()
    g = RG.ProductionGuard(tmp_path / "s", "bank", {"bank": 100.0}, ATTEST, "i1", 0.01,
                           technical=C.technical, fallback_value="base", clock=clk, cap=SD.CAP)  # fmt: skip
    g.open_instance(1)
    g.canary(COMPOSITE)
    mon = HL.HealthMonitor(tmp_path / "runs", clock=clk)
    mon.r_flag_watch = True
    flagged = {"states": [{"h": 2, "R": [0.1], "R_flags": ["rows_outside_declared_support"]}]}
    with pytest.raises(HL.PostCompletionError):
        H.run_units(H.UnitStore(tmp_path / "u"), [({"kind": "bank_side", "u": 1}, "side:x")],
                    mon.wrap_fn(g, lambda r: flagged), "t" * 64, "p", guard=g)  # fmt: skip
    rs = [json.loads(x) for x in (tmp_path / "s" / RG.LOG_NAME).read_text().splitlines()]
    assert [r["cause"] for r in rs if r["type"] == "attempt_fail"] == [None]
    assert g.open_incidents()[-1]["kind"] == "r_flag_after_partial_vocab_check"
    mon2 = HL.HealthMonitor(
        tmp_path / "runs2", clock=clk
    )  # no partial check: flags are not incidents
    assert mon2.r_flag_watch is False


def test_c4_a_mismatch_quarantines_the_field_before_the_incident(tmp_path, no_inert):
    st, tr, pl = make_bank(tmp_path)
    rem = RepairRemote(pl, tamper_verify="row_scale:003:neg")
    with pytest.raises(FR.FieldRepairMismatch):
        controller(st, rem, tr).repair_field(MAN, ROOTS, commit="abcdef1")
    kinds = [k for k, _ in rem.calls]
    assert kinds[0] == "field_quarantine"  # anything on disk moved aside before re-collecting
    assert (
        kinds[-1] == "field_quarantine"
    )  # the mismatched re-collection moved aside before the incident
    inc = [r for r in recs(tmp_path) if r["type"] == "incident"][-1]
    assert inc["kind"] == "field_repair_mismatch" and inc["detail"]["quarantine"]["dest"] == "/q/x"
    assert not [r for r in recs(tmp_path) if r["type"] == "repair"]


def disk_of(pl, vocab=("vocab.pt", "n" * 64)):
    d = {FR.dir_of(k): {"file_sha256": v["file_sha256"], "kept_sha256": v["kept_sha256"]}
         for k, v in pl.items() if k != "field:vocab"}  # fmt: skip
    d["vocab"] = {"file_sha256": {vocab[0]: vocab[1]}, "kept_sha256": None}
    return d


def test_c5_no_field_reading_unit_on_a_new_instance_before_the_repair(tmp_path, no_inert):
    st, tr, pl = make_bank(tmp_path, inst="lambda:old")
    g_old = st.guard("bank", "lambda:old")
    g_old.close_instance()
    co = controller(st, RepairRemote(pl, disk=disk_of(pl)), tr, inst="lambda:new")
    co.field_inputs = (MAN, ROOTS)
    g = co.guard("bank")
    n = len([r for r in recs(tmp_path) if r["type"] == "attempt_start"])
    with pytest.raises(FR.FieldRepairUnverifiable, match="neither produced nor repaired"):
        co.require_field(g)
    assert g.open_incidents()[-1]["kind"] == "field_unverified"
    assert (
        len([r for r in recs(tmp_path) if r["type"] == "attempt_start"]) == n
    )  # no attempt consumed
    g.resolve_incidents("expected: no repair yet", "test")
    g.canary(COMPOSITE)
    co.repair_field(MAN, ROOTS, commit="abcdef1")  # on lambda:new
    g = co.guard("bank")
    assert co.require_field(g)["mode"] == "repaired_here"
    co.remote.disk = disk_of(
        pl, vocab=("vocab.pt", "z" * 64)
    )  # the on-disk vocabulary changed afterwards
    with pytest.raises(FR.FieldRepairUnverifiable, match="differs"):
        co.require_field(g)


def test_c5_a_field_produced_on_this_instance_passes_and_the_gate_is_wired():
    src = (ROOT / "src/wsp/wm/c1/controller.py").read_text()
    for stage in ("def stage_bank(", "def stage_e2(", "def stage_sbon("):
        body = src[src.index(stage) : src.index("\n    def ", src.index(stage) + 10)]
        assert "self.require_field(" in body, stage
        assert body.index("self.require_field(") < body.rindex("self.run(g"), stage
    ctl = (ROOT / "scripts/wme_c1_ctl.py").read_text()
    assert "co.field_inputs = (man," in ctl


def test_c5_produced_here_mode(tmp_path, no_inert):
    st, tr, pl = make_bank(tmp_path, inst="lambda:x")
    co = controller(st, RepairRemote(pl, disk=disk_of(pl)), tr, inst="lambda:x")
    co.field_inputs = (MAN, ROOTS)
    assert co.require_field(co.guard("bank"))["mode"] == "produced_here"


# ------------------------------------------------------------------ v2t: the resume run4 halt (fencing)


class FencedRemote(FakeRemote):
    """As the real worker does: a request whose epoch is below the worker fence is rejected (never answered)."""

    def __init__(self, payloads, transport):
        super().__init__(payloads)
        self.t, self.epochs = transport, []

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        fp = self.t.root / P.FENCE
        fence = int(json.loads(fp.read_text()).get("epoch", -1)) if fp.exists() else -1
        self.epochs.append((kind, epoch, fence))
        if fence > int(epoch):  # worker.Worker.handle: moved to rejected/, never answered
            raise AssertionError(
                f"{kind} at epoch {epoch} behind the worker fence {fence}: rejected_stale, no result"
            )
        return super().call(kind, epoch, inst, uk, label, args, timeout_s)


def test_v2t_every_resume_request_carries_the_new_owner_epoch(tmp_path, monkeypatch):
    """run4 halt (21 Nov 2026): resume set the worker fence to the new owner epoch, then sent hello at epoch 0; the
    worker rejected it as stale and the controller waited forever. Every resume request must carry the owner epoch."""
    mod, runs, a, rem, commits = resume_setup(tmp_path, monkeypatch, ENGINES)
    tr = T.LocalTransport(tmp_path / "w")
    frem = FencedRemote(rem.payloads, tr)
    monkeypatch.setattr(mod, "remote_for", lambda kind, iid: (tr, frem))
    mod.cmd_phasel_main(a)
    epochs = {e for _k, e, _f in frem.epochs}
    assert len(epochs) == 1 and all(e >= f for _k, e, f in frem.epochs)
    assert [k for k, _e, _f in frem.epochs] == [
        "hello",
        "inert_identity",
        "canary",
        "timing",
        "canary",
    ]
    assert statuses(runs)[0] == ["eligible", "eligible"] and commits
