"""Board A quick check of the T4 transition (docs/reviews/council/board-a-wme-c1-v2u-2026-11-21.md), conditions C1-C4
and the C6 record: C1 the attempt budget per contract generation; C2 a robust P2 rule across evidence-only commits;
C3 the transition bound to the live instance and the worker's code; C4 a scratch-only deterministic vocabulary probe
inside ``field_timing``; C6 the deterministic mmlupro hash compared with T2's (report only). CPU only; fixtures."""

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
from wsp.wm.c1 import phasel as PL
from wsp.wm.c1 import protocol as P
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import transport as T

ROOT = Path(__file__).resolve().parents[1]
REF = "a" * 64
ATT0 = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])
ATT1 = RG.LaunchAttestation(True, True, REF, "d" * 64, ["fixture", "t4"])
TR = RG.ContractTransition(
    "driver_or_cuda", "T4 transition (fixture)", "a" * 40, outcome_blind=True
)


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def guard(tmp_path, att, clk, inst="lambda:i1", transition=None):
    g = RG.ProductionGuard(
        tmp_path / "s", "bank", {"bank": 500.0}, att, inst, 0.01, technical=C.technical,
        fallback_value={"rule": "unlabelled"}, clock=clk, transition=transition,
    )  # fmt: skip
    g.open_instance(1)
    g.canary(REF)
    return g


def recs(tmp_path, t=None):
    rs = [json.loads(x) for x in (tmp_path / "s" / RG.LOG_NAME).read_text().splitlines()]
    return rs if t is None else [r for r in rs if r["type"] == t]


UNIT = [({"u": 1}, "u1")]


def boom(r):
    raise P.TransportError("worker unreachable")  # technical: infrastructure (retried)


def ok(r):
    return {"v": 1}


def run(tmp_path, g, fn, units=UNIT):
    return H.run_units(H.UnitStore(tmp_path / "u"), units, fn, "t" * 64, "p", guard=g)


# ------------------------------------------------------------------ C1: the attempt budget per contract generation


def test_c1_a_superseded_unit_gets_three_fresh_attempts_in_the_new_generation_then_its_fallback(
    tmp_path,
):
    clk = Clock()
    g0 = guard(tmp_path, ATT0, clk)
    run(tmp_path, g0, ok)
    g0.canary(REF)
    g1 = guard(tmp_path, ATT1, clk, transition=TR)
    out = run(tmp_path, g1, boom)
    assert out["u1"]["fallback"] is True
    starts = [(r["n"], r["generation"]) for r in recs(tmp_path, "attempt_start")]
    assert starts == [(1, 0), (2, 1), (3, 1), (4, 1)]  # numbers stay global (.a2-.a4): keys unique
    assert len(recs(tmp_path, "fallback")) == 1


def test_c1_the_limit_of_three_within_one_generation_is_unchanged(tmp_path):
    clk = Clock()
    g0 = guard(tmp_path, ATT0, clk)
    out = run(tmp_path, g0, boom)
    assert out["u1"]["fallback"] is True
    assert [r["n"] for r in recs(tmp_path, "attempt_start")] == [1, 2, 3]


def test_c1_an_earlier_generations_failure_cause_neither_blocks_nor_licenses(tmp_path):
    clk = Clock()
    g0 = guard(tmp_path, ATT0, clk)

    def defect(r):
        raise ValueError("a code defect")  # no registered cause: retried only after a repair

    with pytest.raises(ValueError):
        run(tmp_path, g0, defect)
    with pytest.raises(RG.RepairRequired):
        run(tmp_path, g0, ok)
    g1 = guard(tmp_path, ATT1, clk, transition=TR)
    out = run(tmp_path, g1, ok)  # generation 1: the generation-0 failure does not block
    assert out["u1"]["status"] == "pending_canary"
    # and a generation-0 repair record cannot license a generation-1 non-technical failure either
    g1.canary(REF)
    with pytest.raises(ValueError):
        run(tmp_path, g1, defect, units=[({"u": 2}, "u2")])
    with pytest.raises(RG.RepairRequired):
        run(tmp_path, g1, ok, units=[({"u": 2}, "u2")])


def test_c1_incident_aborts_still_never_count_and_fallback_needs_three_counted_in_the_generation(
    tmp_path,
):
    from wsp.wm.c1 import health as HL

    clk = Clock()
    g0 = guard(tmp_path, ATT0, clk)
    run(tmp_path, g0, ok)
    g0.canary(REF)
    g1 = guard(tmp_path, ATT1, clk, transition=TR)

    def idle(r):
        g1.incident("gpu_idle", {"low_for_s": 901})
        raise HL.HealthIncident("gpu_idle", "gpu_idle")

    for i in range(3):
        with pytest.raises(HL.HealthIncident):
            run(tmp_path, g1, idle)
        g1.resolve_incidents(f"trip {i}", "chair report")
        g1.canary(REF)
    with pytest.raises(ValueError, match="attempts remain"):
        g1.fallback("unit-key-does-not-matter", "u1")  # nothing counted for an untouched unit
    out = run(tmp_path, g1, boom)
    assert out["u1"]["fallback"] is True
    gen1 = [r["n"] for r in recs(tmp_path, "attempt_start") if r["generation"] == 1]
    assert gen1 == [2, 3, 4, 5, 6, 7]  # 3 aborted (not counted) + 3 counted failures


def test_c1_a_transition_refuses_when_the_superseded_generation_holds_a_unit_fallback(tmp_path):
    clk = Clock()
    g0 = guard(tmp_path, ATT0, clk)
    run(tmp_path, g0, boom)
    before = (tmp_path / "s" / RG.LOG_NAME).read_bytes()
    with pytest.raises(RG.TransitionRefused, match="fallback"):
        RG.ProductionGuard(
            tmp_path / "s", "bank", {"bank": 500.0}, ATT1, "lambda:i1", 0.01, technical=C.technical,
            fallback_value={"rule": "unlabelled"}, clock=clk, transition=TR,
        )  # fmt: skip
    assert (tmp_path / "s" / RG.LOG_NAME).read_bytes() == before  # refused before any write


def test_c1_prereg_note_is_appended():
    txt = (ROOT / "docs/PREREG_WME.md").read_text()
    sec = txt[txt.index("### II.31.12") :]
    assert "per contract generation" in sec and "board-a-wme-c1-v2u-2026-11-21.md" in sec


# ------------------------------------------------------------------ C2: P2 across evidence-only commits


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2v", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(repo, *a):
    return subprocess.run(
        ["git", *a], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    for rel, text in {
        "src/a.py": "x = 1\n",
        "docs/reviews/wme/evidence.json": "{}\n",
        "docs/prereg/wme_manifest.json": "{}\n",
        "docs/reviews/wme/wme-c1-shards-2026-11-19.json": "{}\n",
    }.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    git(r, "init", "-q")
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base", "--no-verify")
    mod = load_ctl()
    monkeypatch.setattr(mod, "ROOT", r)
    return r, mod, git(r, "rev-parse", "HEAD")


def commit(r, msg):
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", msg, "--no-verify")


def test_c2_evidence_only_commits_pass(repo):
    r, mod, base = repo
    (r / "docs/reviews/wme/evidence2.json").write_text("{}\n")
    commit(r, "evidence")
    assert mod.same_code(base) is True and mod.same_code(git(r, "rev-parse", "HEAD")) is True


def test_c2_a_rename_from_src_to_docs_is_code(repo):
    r, mod, base = repo
    git(r, "mv", "src/a.py", "docs/reviews/wme/a.py")
    commit(r, "move")
    assert mod.same_code(base) is False


@pytest.mark.parametrize(
    "rel", ["docs/prereg/wme_manifest.json", "docs/reviews/wme/wme-c1-shards-2026-11-19.json"]
)
def test_c2_bundled_docs_inputs_are_code(repo, rel):
    r, mod, base = repo
    (r / rel).write_text('{"changed": 1}\n')
    commit(r, "input")
    assert mod.same_code(base) is False


def test_c2_every_bundled_docs_path_is_treated_as_code():
    mod = load_ctl()
    docs = [p for p in mod.BUNDLE_FILES if p.startswith("docs/")]
    assert docs and all(not mod._evidence_only_path(p) for p in docs)
    assert not mod._evidence_only_path("docs/prereg/anything.json")
    assert mod._evidence_only_path("docs/reviews/wme/wme-c1-field-timing-2026-11-22-x.json")


def test_c2_a_non_ancestor_is_not_the_same_code(repo):
    r, mod, _base = repo
    git(r, "checkout", "-q", "-b", "side")
    (r / "docs/reviews/wme/side.json").write_text("{}\n")
    commit(r, "side")
    side = git(r, "rev-parse", "HEAD")
    git(r, "checkout", "-q", "-")
    assert mod.same_code(side) is False


def test_c2_field_timing_and_transition_refuse_a_dirty_tree(repo):
    r, mod, _base = repo
    (r / "src/a.py").write_text("x = 2\n")
    monkeypatch_remote = lambda *a: pytest.fail("no instance contact with a dirty tree")
    mod.remote_for = monkeypatch_remote
    with pytest.raises(SystemExit, match="clean"):
        mod.require_clean_tree("field-timing")
    with pytest.raises(SystemExit, match="clean"):
        mod.cmd_field_timing(types.SimpleNamespace(iid="X", tag="t"))
    (r / "src/a.py").write_text("x = 1\n")
    mod.require_clean_tree("transition")  # clean again


# ------------------------------------------------------------------ C3: instance and worker-code binding

ENGINES = {"tf": "1" * 64, "hf_gen": PL.COMMITTED_HF, "vllm": "3" * 64}
COMPOSITE = C.composite_canary(ENGINES)
OLD_EC, NEW_EC = "c" * 64, "e" * 64


def fake_git(*a, **k):
    if a and a[0] in ("status", "diff"):
        return ""
    return "a" * 40


class LiveRemote:
    """The live instance: hello (code identity), canary, quarantine."""

    def __init__(self, code_sha="a" * 40):
        self.code_sha, self.calls = code_sha, []

    def bind(self, g):
        pass

    def call(self, kind, epoch, inst, uk, label, args, timeout_s=None):
        self.calls.append((kind, epoch))
        if kind == "hello":
            return {
                "code_sha": self.code_sha,
                "physical": {"driver": "570.148.08"},
                "versions": {"torch": "2.14"},
            }, None
        if kind == "canary":
            return {"engines": ENGINES}, None
        if kind == "field_quarantine":
            return {"moved": {}, "dest": None}, None
        raise AssertionError(kind)


def transition_rig(
    tmp_path, monkeypatch, timing_instance="lambda:NEW", code_sha="a" * 40, vocab_ok=True
):
    mod = load_ctl()
    runs = tmp_path / "runs"
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "STUDY", runs / "study")
    monkeypatch.setattr(mod, "EVID", tmp_path / "evid")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    runs.mkdir(parents=True)
    att = RG.LaunchAttestation(
        True, True, COMPOSITE, OLD_EC, ["docs/reviews/wme/refs.json sha256:" + "0" * 64]
    )
    (runs / "attestation.json").write_text(json.dumps(RG._att_fields(att), indent=1))
    st = SD.Study(runs / "study", runs, att, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    g = st.guard("bank", "lambda:old")
    g.open_instance(1)
    g.canary(COMPOSITE)
    g.incident("field_repair_mismatch", {"label": "field:g0"})
    g.close_instance()
    st.guard(
        "bank", "lambda:NEW"
    )  # the field-timing owner on the live instance (its epoch is the worker fence)
    monkeypatch.setattr(mod, "ec_sha", lambda field=True: NEW_EC if field else OLD_EC)
    monkeypatch.setattr(PL, "git", fake_git)
    monkeypatch.setattr(mod, "MO_theta0", lambda: "t" * 64)
    monkeypatch.setattr(mod, "commit_evidence", lambda paths, msg: None)
    (tmp_path / "ev").mkdir()
    par = {"engine_config_sha256": NEW_EC, "code_identity": {"controller_commit": "a" * 40},
           "evaluation": {"proceed": True, "selected_engine": GE.GEN_ENGINE, "pass_rule": {"hf_gen_eq_committed": True}}}  # fmt: skip
    tim = {"engine_config_sha256": NEW_EC, "commit": "a" * 40, "instance": timing_instance, "group": 0,
           "hello": {"code_sha": code_sha}, "vocab_probe": {"ok": vocab_ok},
           "runs": [{"mode": "registered"}, {"mode": "deterministic"}, {"mode": "registered"}]}  # fmt: skip
    (tmp_path / "ev/par.json").write_text(json.dumps(par))
    (tmp_path / "ev/tim.json").write_text(json.dumps(tim))
    for rel in (mod.TRANSITION_RULING, mod.TRANSITION_PREP):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    rem = LiveRemote(code_sha)
    tr = T.LocalTransport(tmp_path / "w")
    monkeypatch.setattr(mod, "remote_for", lambda kind, iid: (tr, rem))
    a = types.SimpleNamespace(
        iid="NEW", tag="t4", parity_evidence="ev/par.json", timing_evidence="ev/tim.json"
    )
    return mod, runs, a, rem


def test_c3_transition_binds_the_timing_instance_before_any_write(tmp_path, monkeypatch):
    mod, runs, a, rem = transition_rig(tmp_path, monkeypatch, timing_instance="lambda:OTHER")
    log0 = (runs / "study" / RG.LOG_NAME).read_bytes()
    with pytest.raises(SystemExit, match="instance"):
        mod.cmd_transition(a)
    assert (runs / "study" / RG.LOG_NAME).read_bytes() == log0 and not rem.calls


def test_c3_transition_checks_the_worker_code_by_hello_before_any_write(tmp_path, monkeypatch):
    mod, runs, a, rem = transition_rig(tmp_path, monkeypatch, code_sha="b" * 40)
    log0 = (runs / "study" / RG.LOG_NAME).read_bytes()
    with pytest.raises(SystemExit, match="worker"):
        mod.cmd_transition(a)
    assert (runs / "study" / RG.LOG_NAME).read_bytes() == log0
    assert [k for k, _e in rem.calls] == [
        "hello"
    ]  # at the current fence epoch: no new owner was written
    assert rem.calls[0][1] == RG.read_study(runs / "study").epoch


def test_c3_c4_transition_records_the_hello_identity_and_needs_the_vocab_probe(
    tmp_path, monkeypatch
):
    mod, _runs, a, rem = transition_rig(tmp_path, monkeypatch, vocab_ok=False)
    with pytest.raises(SystemExit, match="vocab"):
        mod.cmd_transition(a)
    mod, _runs, a, rem = transition_rig(tmp_path / "2", monkeypatch)
    out = mod.cmd_transition(a)
    assert out["hello"]["code_sha"] == "a" * 40 and out["hello"]["driver"] == "570.148.08"
    assert [k for k, _e in rem.calls][:1] == ["hello"] and out["generation"] == 1


# ------------------------------------------------------------------ C4, C6: field_timing


@pytest.fixture
def timing_rig(tmp_path, monkeypatch):
    import hashlib

    from wsp.wm import collect as CO
    from wsp.wm import panel2 as P2
    from wsp.wm import scorers as Sc
    from wsp.wm import vocabfield as VF
    from wsp.wm.c1 import field as F

    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)
    monkeypatch.setattr(
        GE, "kernel_bindings", lambda: dict.fromkeys(GE.LINEAR_ATTN_FUNCS, GE.QWEN35_MODULE)
    )
    monkeypatch.setattr(Sc, "field_panel", lambda name, its, scored: name)
    monkeypatch.setattr(
        P2,
        "objective_specs",
        lambda objs, panels, cfg: [types.SimpleNamespace(name=o) for o in objs],
    )
    seen = []

    def collect(model_, tok, specs, panels, groups, device, out, **kw):
        seen.append(("collect", GE.global_state()["deterministic_algorithms"]))
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fh = {f"{s.name}.g0.pt": hashlib.sha256(s.name.encode()).hexdigest() for s in specs}
        (out / "manifest.json").write_text(json.dumps({"identity": {"timing": {f"{s.name}.g0": {"seconds": 1.0}
                                                                                for s in specs}}, "file_sha256": fh}))  # fmt: skip

    state = {"raise": None}

    def vocab(model_, tok, specs, panels, rows, device, out, **kw):
        seen.append(("vocab", GE.global_state()["deterministic_algorithms"]))
        if state["raise"]:
            raise RuntimeError(state["raise"])
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({"file_sha256": {"vocab.pt": "n" * 64}}))

    monkeypatch.setattr(CO, "collect_grouped", collect)
    monkeypatch.setattr(VF, "collect_vocab_rows", vocab)
    m = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa")),
                              tok=object(), base_hash="t" * 64, embed="model.embed_tokens.weight",
                              rm=types.SimpleNamespace(n_layers=8, name=lambda L, r: f"model.layers.{L}.{r}"),
                              at_base=lambda: None)  # fmt: skip
    fdir = tmp_path / "field"
    fdir.mkdir()
    rt = types.SimpleNamespace(hf=lambda: m, shard_items=lambda: ({"mmlupro_gold_lp": [1], "wikitext_nll": [2]}, 0, 0),
                               scored_hashes=lambda: (), field_dir=lambda: fdir, declared_rows=lambda m: [1, 2],
                               _field=None, _rc=None)  # fmt: skip
    ctx = types.SimpleNamespace(check=lambda: None, scratch=tmp_path / "scratch")
    assert F.T2_OBJECTIVES
    return rt, ctx, seen, state, fdir


def test_c4_the_vocab_probe_runs_first_deterministic_scratch_only(timing_rig):
    from wsp.wm.c1 import serve as S

    rt, ctx, seen, _state, fdir = timing_rig
    p, _ = S.h_field_timing(rt, {"args": {"group": 0}}, ctx)
    assert seen[0] == ("vocab", True)
    assert p["vocab_probe"]["ok"] is True and p["vocab_probe"]["file_sha256"] == {
        "vocab.pt": "n" * 64
    }
    assert "seconds" in p["vocab_probe"] and not any(fdir.iterdir())
    assert [r["mode"] for r in p["runs"]] == ["registered", "deterministic", "registered"]


def test_c4_a_raising_vocab_kernel_is_reported_and_stops_the_timing(timing_rig):
    from wsp.wm.c1 import serve as S

    rt, ctx, seen, state, _fdir = timing_rig
    state["raise"] = "index_add_cuda_ does not have a deterministic implementation"
    p, _ = S.h_field_timing(rt, {"args": {"group": 0}}, ctx)
    assert (
        p["vocab_probe"]["ok"] is False
        and "index_add_cuda_" in p["vocab_probe"]["error"]["message"]
    )
    assert p["runs"] == [] and [w for w, _ in seen] == [
        "vocab"
    ]  # halt: no timing run, nothing substituted


def test_c6_field_timing_evidence_compares_the_deterministic_mmlupro_hash_with_t2(
    tmp_path, monkeypatch
):
    mod = load_ctl()
    assert mod.T2_MMLUPRO_DET.startswith("af39a617") and len(mod.T2_MMLUPRO_DET) == 64
    runs = [{"mode": "deterministic", "file_sha256": {"mmlupro_gold_lp.g0.pt": mod.T2_MMLUPRO_DET}}]
    assert mod.c6_record(runs)["equal"] is True
    runs = [{"mode": "deterministic", "file_sha256": {"mmlupro_gold_lp.g0.pt": "f" * 64}}]
    rec = mod.c6_record(runs)
    assert rec["equal"] is False and "one live instance" in rec["note"]
