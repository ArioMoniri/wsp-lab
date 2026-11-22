"""T4 of RULING-wme-c1-bank-halt addendum 4 (after the D1 diagnostic and the T2 probe): item 1, field and vocabulary
collection under the registered deterministic context (and nothing else); item 2, the contract transition
(supersedes every bank-phase and field unit, resolves incident seq 277, per-file hashes in the new field payloads);
item 3's tool (``field-timing``); item 4, the recovery path (lost instance -> resume -> deterministic re-collection
-> bitwise K3). CPU only; fixtures."""

import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest
import torch

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import deadlines as DL
from wsp.wm.c1 import fieldrepair as FR
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import health as HL
from wsp.wm.c1 import phasel as PL
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import transport as T

ROOT = Path(__file__).resolve().parents[1]
REF_HF = PL.COMMITTED_HF
ENGINES = {"tf": "1" * 64, "hf_gen": REF_HF, "vllm": "3" * 64}
COMPOSITE = C.composite_canary(ENGINES)
TORCH_PATH = dict.fromkeys(GE.LINEAR_ATTN_FUNCS, GE.QWEN35_MODULE)


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2u", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model(impl="sdpa"):
    return types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation=impl))


# ------------------------------------------------------------------ item 1: deterministic field collection only


def test_field_collection_is_the_registered_deterministic_context_and_restores(monkeypatch):
    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)
    before = GE.global_state()
    with GE.field_collection(model(), TORCH_PATH):
        inside = GE.global_state()
    assert GE.global_state() == before
    assert inside["deterministic_algorithms"] is True and inside["warn_only"] is False
    assert inside["cudnn_deterministic"] is True and inside["cudnn_benchmark"] is False
    assert inside["sdp"] == {"flash": False, "mem_efficient": False, "math": True, "cudnn": False}
    with pytest.raises(ValueError), GE.field_collection(model(), TORCH_PATH):
        raise ValueError("a failing collection")
    assert GE.global_state() == before  # restored on an exception too (R11)
    assert GE.FIELD_MODE == "deterministic"


def test_engine_config_records_the_field_mode_and_the_pre_transition_config_excludes_it():
    mod = load_ctl()
    ec = mod.engine_config()
    assert ec["field"] == GE.field_config()
    assert GE.field_config()["mode"] == "deterministic"
    assert GE.field_config()["deterministic"] == GE.config("hf_deterministic")["deterministic"] | {
        "scope": GE.field_config()["deterministic"]["scope"]
    }
    legacy = mod.engine_config(field=False)
    assert "field" not in legacy and {k: v for k, v in ec.items() if k != "field"} == legacy
    assert mod.ec_sha() != mod.ec_sha(field=False)
    # the generation engine's own recorded settings are unchanged by the field entry
    assert ec["hf_generate"]["engine"] == GE.config()


class FakeM:
    def __init__(self):
        self.model, self.tok, self.base_hash = model(), object(), "t" * 64
        self.rm = types.SimpleNamespace(n_layers=8, name=lambda L, r: f"model.layers.{L}.{r}")
        self.embed = "model.embed_tokens.weight"

    def at_base(self):
        pass


class FakeRT:
    def __init__(self, root):
        self.root, self.m = Path(root), FakeM()
        self._field = self._rc = None

    def hf(self):
        return self.m

    def shard_items(self):
        return {"wikitext_nll": [1], "mmlupro_gold_lp": [2]}, None, None

    def scored_hashes(self):
        return ()

    def field_dir(self):
        d = self.root / "field"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def declared_rows(self, m):
        return [1, 2]


@pytest.fixture
def field_rig(tmp_path, monkeypatch):
    """The field handlers on a fake runtime: every collection records the global state it ran under."""
    from wsp.wm import collect as CO
    from wsp.wm import panel2 as P2
    from wsp.wm import scorers as Sc
    from wsp.wm import vocabfield as VF
    from wsp.wm.c1 import field as F

    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)
    monkeypatch.setattr(GE, "kernel_bindings", lambda: TORCH_PATH)
    seen = []
    monkeypatch.setattr(Sc, "field_panel", lambda name, its, scored: name)
    monkeypatch.setattr(
        P2,
        "objective_specs",
        lambda objs, panels, cfg: [types.SimpleNamespace(name=o) for o in objs],
    )

    def collect(model_, tok, specs, panels, groups, device, out, **kw):
        seen.append(("collect", GE.global_state()))
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fh = {f"{s.name}.g0.pt": hashlib.sha256(s.name.encode()).hexdigest() for s in specs}
        timing = {f"{s.name}.g0": {"seconds": 1.0} for s in specs}
        (out / "manifest.json").write_text(
            json.dumps({"identity": {"timing": timing}, "file_sha256": fh})
        )

    def kept(d, device):
        seen.append(("kept", GE.global_state()))
        return {"sha256": {"k.pt": "k" * 64}, "audits": {"m": {"A": {"rank": 1}, "G": {"rank": 1}}}}

    def vocab(model_, tok, specs, panels, rows, device, out, **kw):
        seen.append(("vocab", GE.global_state()))
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({"file_sha256": {"vocab.pt": "n" * 64}}))

    monkeypatch.setattr(CO, "collect_grouped", collect)
    monkeypatch.setattr(F, "compute_kept", kept)
    monkeypatch.setattr(VF, "collect_vocab_rows", vocab)
    ctx = types.SimpleNamespace(check=lambda: None, scratch=tmp_path / "scratch")
    return FakeRT(tmp_path / "w"), ctx, seen


def test_field_and_vocab_collection_run_deterministic_and_restore_kept_runs_registered(field_rig):
    from wsp.wm.c1 import serve as S

    rt, ctx, seen = field_rig
    before = GE.global_state()
    pg, _ = S.h_field_group(rt, {"args": {"group": 0}}, ctx)
    pv, _ = S.h_field_vocab(rt, {"args": {}}, ctx)
    pr, _ = S.h_field_repair(rt, {"args": {"group": 0}}, ctx)  # K3 re-collection: the same handler
    assert GE.global_state() == before
    by = {}
    for what, st in seen:
        by.setdefault(what, []).append(st)
    for st in by["collect"] + by["vocab"]:
        assert (
            st["deterministic_algorithms"] is True and st["sdp"]["math"] and not st["sdp"]["flash"]
        )
    for st in by["kept"]:  # the kept-factor eigendecompositions keep their registered settings
        assert st == before
    # item 2: the new field payloads record per-file hashes (groups and vocabulary): K3 can verify them bitwise
    pl = FR.plan({"field:g0": ("field_group", pg), "field:vocab": ("field_vocab", pv)})
    assert (
        set(pl["expected"]) == {"field:g0", "field:vocab"}
        and not pl["indirect"]
        and not pl["unverifiable"]
    )
    assert pr["file_sha256"] == pg["file_sha256"]


def test_only_field_and_vocab_collection_enter_the_field_context():
    """Every other unit and engine keeps its registered settings: the field context is entered in exactly two
    places (field groups, vocabulary rows) and the timing request of item 3; nowhere else in the worker code."""
    import re

    users = {}
    for p in sorted((ROOT / "src/wsp/wm/c1").glob("*.py")):
        if p.name == "genengine.py":
            continue
        src = p.read_text()
        for m in re.finditer(r"field_collection\(", src):
            fn = src[: m.start()].rsplit("\ndef ", 1)[-1].split("(", 1)[0]
            users.setdefault(p.name, set()).add(fn)
    assert users == {"serve.py": {"h_field_group", "h_field_vocab", "h_field_timing"}}


def test_other_units_run_under_the_registered_state_after_a_field_unit(field_rig, monkeypatch):
    from wsp.wm.c1 import serve as S

    rt, ctx, _seen = field_rig
    S.h_field_group(rt, {"args": {"group": 0}}, ctx)
    # any later handler (a bank unit, a canary child, a generate call outside its own context) sees the baseline
    assert GE.global_state() == GE.BASELINE_STATE | {GE.CUBLAS_ENV: GE.CUBLAS_WORKSPACE}
    assert not torch.are_deterministic_algorithms_enabled()


def test_field_timing_collects_registered_and_deterministic_into_scratch_only(field_rig):
    from wsp.wm.c1 import serve as S

    rt, ctx, seen = field_rig
    p, files = S.h_field_timing(rt, {"args": {"group": 0}}, ctx)
    assert [r["mode"] for r in p["runs"]] == ["registered", "deterministic", "registered"]
    dets = [st for what, st in seen if what == "collect"]
    assert [s["deterministic_algorithms"] for s in dets] == [False, True, False]
    assert not any(rt.field_dir().iterdir())  # nothing written to the live field
    assert files == {}
    assert set(p["runs"][0]["seconds"]) == {"wikitext_nll.g0", "mmlupro_gold_lp.g0"}


def test_field_timing_is_a_classified_gpu_request_with_a_deadline():
    assert HL.KIND_CLASS["field_timing"] == "gpu"
    assert DL.deadline_s("field_timing") <= 3600


# ------------------------------------------------------------------ item 2: the contract transition

OLD_EC = "c" * 64


def att_old(ev=("docs/reviews/wme/refs.json sha256:" + "0" * 64,)):
    return RG.LaunchAttestation(True, True, COMPOSITE, OLD_EC, list(ev))


def study_with_halt(tmp_path, monkeypatch, mod):
    """The run5 state: attestation.json (generation 0), eligible bank units on an old instance, an open
    field_repair_mismatch incident."""
    runs = tmp_path / "runs"
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "STUDY", runs / "study")
    monkeypatch.setattr(mod, "EVID", tmp_path / "evid")
    runs.mkdir(parents=True)
    att = att_old()
    (runs / "attestation.json").write_text(json.dumps(RG._att_fields(att), indent=1))
    st = SD.Study(runs / "study", runs, att, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    st.priced = {"bank": 100.0, "confirmatory": 100.0, "sbon": 100.0, "ste": 100.0}
    g = st.guard("bank", "lambda:old")
    g.open_instance(1)
    g.canary(COMPOSITE)
    units = SD.bank_units({}, {}, range(2), False)
    H.run_units(st.store("bank"), units, lambda r: {"group": r["args"]["group"], "file_sha256": {"a": "0" * 64},
                "kept_sha256": {"k": "0" * 64}}, st.theta0, st.panel, guard=g)  # fmt: skip
    g.canary(COMPOSITE)
    g.incident("field_repair_mismatch", {"label": "field:g0"})
    g.close_instance()
    return runs, st, units


class CanaryRemote:
    def __init__(self, engines=ENGINES):
        self.engines, self.calls, self.guard = engines, [], None

    def bind(self, g):
        self.guard = g

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        self.calls.append((kind, epoch))
        if kind == "canary":
            return {"engines": self.engines}, None
        if kind == "field_quarantine":
            return {"moved": {}, "dest": None}, None
        if kind == "hello":  # Board A v2u C3: the worker runs the parity commit's code
            return {"code_sha": "a" * 40, "physical": {"driver": "x"}, "versions": {}}, None
        raise AssertionError(f"unexpected request {kind}")


def fake_git(*a, **k):
    """HEAD is "a"*40, the tracked tree is clean, history since an ancestor is evidence-only; any other commit is
    not an ancestor (as git merge-base --is-ancestor reports)."""
    import subprocess

    if a and a[0] in ("status", "diff"):
        return ""
    if a and a[0] == "merge-base" and a[2] != "a" * 40:
        raise subprocess.CalledProcessError(1, ["git", *a])
    return "a" * 40


def transition_setup(tmp_path, monkeypatch):
    mod = load_ctl()
    runs, _st, _units = study_with_halt(tmp_path, monkeypatch, mod)
    new_ec = "e" * 64
    monkeypatch.setattr(mod, "ec_sha", lambda field=True: new_ec if field else OLD_EC)
    # a clean tracked tree and evidence-only history (Board A v2u C2; tested in test_wme_c1_v2v)
    monkeypatch.setattr(PL, "git", fake_git)
    monkeypatch.setattr(mod, "MO_theta0", lambda: "t" * 64)
    commits = []
    monkeypatch.setattr(mod, "commit_evidence", lambda paths, msg: commits.append((paths, msg)))
    ev = tmp_path / "ev"
    ev.mkdir()
    par = {"engine_config_sha256": new_ec, "code_identity": {"controller_commit": "a" * 40},
           "evaluation": {"proceed": True, "selected_engine": GE.GEN_ENGINE, "pass_rule": {"hf_gen_eq_committed": True}}}  # fmt: skip
    tim = {"engine_config_sha256": new_ec, "commit": "a" * 40, "group": 0, "instance": "lambda:NEW",
           "vocab_probe": {"ok": True},  # Board A v2u C3, C4 (tested in test_wme_c1_v2v)
           "runs": [{"mode": "registered"}, {"mode": "deterministic"}, {"mode": "registered"}]}  # fmt: skip
    (ev / "par.json").write_text(json.dumps(par))
    (ev / "tim.json").write_text(json.dumps(tim))
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for rel, text in ((mod.TRANSITION_RULING, "addendum 4"), (mod.TRANSITION_PREP, "{}")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    rem = CanaryRemote()
    tr = T.LocalTransport(tmp_path / "w")
    monkeypatch.setattr(mod, "remote_for", lambda kind, iid: (tr, rem))
    a = types.SimpleNamespace(
        iid="NEW", tag="t4", parity_evidence="ev/par.json", timing_evidence="ev/tim.json"
    )
    return mod, runs, a, rem, commits


def test_transition_supersedes_every_bank_unit_and_resolves_the_open_incident(
    tmp_path, monkeypatch
):
    mod, runs, a, rem, commits = transition_setup(tmp_path, monkeypatch)
    before = (runs / "attestation.json").read_bytes()
    s0 = RG.read_study(runs / "study")
    assert [i["kind"] for i in s0.incidents] == ["field_repair_mismatch"]
    inc_seq = s0.incidents[0]["seq"]
    out = mod.cmd_transition(a)
    s = RG.read_study(runs / "study")
    # a new generation under the registered guard mechanism, decided outcome-blind, naming the ruling and commit
    tr_rec = [r for r in s.contracts if r["type"] == "contract_transition"]
    assert len(tr_rec) == 1 and tr_rec[0]["generation"] == 1 and tr_rec[0]["outcome_blind"] is True
    assert tr_rec[0]["cause"] in RG.TECHNICAL_CAUSES and tr_rec[0]["commit"] == "a" * 40
    assert "addendum 4" in tr_rec[0]["description"] and str(inc_seq) in tr_rec[0]["description"]
    assert (
        tr_rec[0]["engine_config_sha256"] == "e" * 64 and tr_rec[0]["canary_reference"] == COMPOSITE
    )
    # every bank-phase and field unit superseded
    sts = [s.status(x) for v in s.ok.values() for x in v]
    assert sts and set(sts) == {"superseded_contract"}
    # the open incident resolved, citing the ruling and the transition record
    recs = [json.loads(x) for x in (runs / "study" / RG.LOG_NAME).read_text().splitlines()]
    res = [r for r in recs if r["type"] == "incidents_resolved"]
    assert (
        len(res) == 1
        and "addendum 4" in res[0]["report"]
        and str(tr_rec[0]["seq"]) in res[0]["diagnosis"]
    )
    assert not s.incidents
    # the start canary under the new contract passed; attestation.json (generation 0) is never rewritten
    assert out["canary"]["pass"] is True and ("canary", out["epoch"]) in rem.calls
    assert (runs / "attestation.json").read_bytes() == before
    assert (
        json.loads((runs / "attestation-g1.json").read_text())["engine_config_sha256"] == "e" * 64
    )
    # the current contract is read from the study log from now on
    assert mod.load_att().engine_config_sha256 == "e" * 64
    assert RG.contract_digest(mod.load_att()) == s.contract
    assert len([e for e in mod.load_att().evidence if "canary-refs" in e or "refs.json" in e]) == 1
    assert commits


def test_transition_refuses_without_bound_evidence_or_when_nothing_changes(tmp_path, monkeypatch):
    mod, runs, a, rem, _commits = transition_setup(tmp_path, monkeypatch)
    log0 = (runs / "study" / RG.LOG_NAME).read_bytes()
    bad = json.loads((tmp_path / "ev/tim.json").read_text())
    (tmp_path / "ev/tim2.json").write_text(json.dumps(dict(bad, engine_config_sha256="f" * 64)))
    with pytest.raises(SystemExit, match="timing"):
        mod.cmd_transition(types.SimpleNamespace(**{**vars(a), "timing_evidence": "ev/tim2.json"}))
    (tmp_path / "ev/par2.json").write_text(json.dumps({**json.loads((tmp_path / "ev/par.json").read_text()),
                                                       "code_identity": {"controller_commit": "b" * 40}}))  # fmt: skip
    with pytest.raises(SystemExit, match="parity"):
        mod.cmd_transition(types.SimpleNamespace(**{**vars(a), "parity_evidence": "ev/par2.json"}))
    assert (
        runs / "study" / RG.LOG_NAME
    ).read_bytes() == log0 and not rem.calls  # nothing written, no contact
    mod.cmd_transition(a)
    with pytest.raises(SystemExit, match="nothing to transition"):
        mod.cmd_transition(a)


def test_transition_refuses_an_unexpected_open_incident(tmp_path, monkeypatch):
    mod, runs, a, _rem, _commits = transition_setup(tmp_path, monkeypatch)
    st = SD.Study(runs / "study", runs, att_old(), "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    g = st.guard("bank", "lambda:old2")
    g.incident("gpu_idle", {"x": 1})
    with pytest.raises(SystemExit, match="gpu_idle"):
        mod.cmd_transition(a)


def test_bank_and_field_repair_refuse_until_the_transition(tmp_path, monkeypatch):
    mod, _runs, _a, _rem, _commits = transition_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mod, "remote_for", lambda *a: pytest.fail("no instance contact before the transition")
    )
    with pytest.raises(SystemExit, match="transition"):
        mod.cmd_run(types.SimpleNamespace(iid="NEW", stages="bank"))
    with pytest.raises(SystemExit, match="transition"):
        mod.cmd_repair_field(types.SimpleNamespace(iid="NEW"))


def test_resume_accepts_only_the_pre_transition_engine_config_difference(tmp_path, monkeypatch):
    mod, _runs, _a, _rem, _commits = transition_setup(tmp_path, monkeypatch)
    assert mod.contract_state() == "pre_transition"
    monkeypatch.setattr(mod, "ec_sha", lambda field=True: "9" * 64)
    with pytest.raises(SystemExit, match="engine config"):
        mod.contract_state()


# ------------------------------------------------------------------ item 4: the recovery path


def test_recovery_after_a_lost_instance_is_resume_then_deterministic_recollection_and_bitwise_k3(
    tmp_path, monkeypatch
):
    from wsp.wm.c1 import controller as CO

    mod, runs, a, _rem, _commits = transition_setup(tmp_path, monkeypatch)
    mod.cmd_transition(a)
    att1 = mod.load_att()
    # the field collected once under the new contract on the live instance (deterministic payloads with hashes)
    st = SD.Study(runs / "study", runs, att1, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    st.priced = {"bank": 100.0, "confirmatory": 100.0, "sbon": 100.0, "ste": 100.0}
    g = st.guard("bank", "lambda:NEW")
    g.open_instance(1)
    g.canary(COMPOSITE)
    units = SD.bank_units({}, {}, range(2), True)
    pls = {f"field:g{i}": {"group": i, "file_sha256": {f"g{i}.pt": f"{i}" * 64},
                           "kept_sha256": {f"g{i}.k": f"{i}" * 64}} for i in range(2)}  # fmt: skip
    pls["field:vocab"] = {"manifest_sha256": "m" * 64, "file_sha256": {"vocab.pt": "v" * 64}}
    H.run_units(st.store("bank"), units, lambda r: pls[r["label"]], st.theta0, st.panel, guard=g)
    g.canary(COMPOSITE)
    s = RG.read_study(runs / "study")
    assert {s.status(x) for v in s.ok.values() for x in v if x["generation"] == 1} == {"eligible"}
    # the instance is lost; a new one resumes under the SAME (new) contract: no transition pending any more
    assert mod.contract_state() == "current"

    class Recollect:
        def __init__(self, tamper=None):
            self.tamper, self.calls = tamper, []

        def bind(self, g):
            pass

        def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
            self.calls.append((kind, label))
            if kind == "field_quarantine":
                return {"moved": {}, "dest": None}, None
            p = json.loads(
                json.dumps(pls[label])
            )  # the deterministic re-collection reproduces the hashes
            if label == self.tamper:
                p["file_sha256"] = {k: "x" * 64 for k in p["file_sha256"]}
            return p, None

    rc = Recollect()
    co = CO.Controller(
        st, rc, T.LocalTransport(tmp_path / "w2"), "lambda:NEW2", 1, log=lambda *a: None
    )
    res = co.repair_field({}, {"roots": [], "controls": []}, range(2), True, commit="a" * 40)
    assert res["verified"] == 3 and not [
        c for c in rc.calls if c[0] == "bank_verify"
    ]  # no K3b needed
    assert [c[1] for c in rc.calls if c[0] == "field_repair"] == [
        "field:g0",
        "field:g1",
        "field:vocab",
    ]
    s = RG.read_study(runs / "study")
    assert any("lambda:NEW2" in r["description"] for r in s.repairs)
    # a re-collection that is not bitwise equal still halts (nothing substituted)
    rc2 = Recollect(tamper="field:g1")
    co2 = CO.Controller(
        st, rc2, T.LocalTransport(tmp_path / "w3"), "lambda:NEW3", 1, log=lambda *a: None
    )
    with pytest.raises(FR.FieldRepairMismatch):
        co2.repair_field({}, {"roots": [], "controls": []}, range(2), True, commit="a" * 40)
