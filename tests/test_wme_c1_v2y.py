"""RULING-wme-c1-bank-halt addendum 7 (the bank PhasePause), v2y: the registered resumption of a paused phase (§II.13
item 9, §II.14: "the chair may resume from ledger seconds only, within the absolute cap"). A dated ``phase_resume``
record (chair, ruling, ledger-seconds basis, outcome-blind) lifts the 1.25x pause for that phase only; it is refused
unless the pause is pending; the absolute cap and R8 still apply. CPU only; fixtures."""

import importlib.util
import json
import types
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import study as SD

ROOT = Path(__file__).resolve().parents[1]
REF = "a" * 64
ATT = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])
RULING = "RULING-wme-c1-bank-halt.md addendum 7"


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
        return self.t


def guard(tmp_path, clk, phase="bank", priced=None, est=1.0, inst="lambda:i1"):
    g = RG.ProductionGuard(
        tmp_path / "s", phase, priced or {"bank": 4.0, "e2": 100.0}, ATT, inst, est, technical=C.technical,
        fallback_value={"rule": "x"}, clock=clk, cap=SD.CAP,
    )  # fmt: skip
    g.open_instance(1)
    g.canary(REF)
    return g


def recs(tmp_path, t=None):
    rs = [json.loads(x) for x in (tmp_path / "s" / RG.LOG_NAME).read_text().splitlines()]
    return rs if t is None else [r for r in rs if r["type"] == t]


def quick(r):
    return {"v": r["u"]}


def units(lo, hi):
    return [({"u": i}, f"u{i}") for i in range(lo, hi)]


def paused(tmp_path):
    """Measured bank time 4.5 GPU-h (priced 4.0, so the pause threshold is 5.0; the next unit's estimate is 1.0):
    the next bank unit pauses."""
    clk = Clock()
    g = guard(tmp_path, clk)
    store = H.UnitStore(tmp_path / "u")
    H.run_units(store, units(0, 1), quick, "t" * 64, "p", guard=g)
    g.canary(REF)
    g.ledger.add("bank", 4.5 * 3600, note="fixture: measured bank time")
    with pytest.raises(RG.PhasePause):
        H.run_units(store, units(1, 2), quick, "t" * 64, "p", guard=g)
    return clk, g, store


def test_a_pending_pause_is_resumed_by_a_dated_chair_record_and_units_continue(tmp_path):
    _clk, g, store = paused(tmp_path)
    n_ok = len(recs(tmp_path, "attempt_ok"))
    rec = g.resume_phase("bank", RULING)
    assert (
        rec["type"] == "phase_resume"
        and rec["resumed_phase"] == "bank"
        and rec["decided_by"] == "chair"
    )
    assert rec["ruling"] == RULING and rec["outcome_blind"] is True and rec["date"]
    b = rec["basis"]
    assert (
        b["threshold_gpu_h"] == pytest.approx(5.0)
        and b["measured_unit_gpu_h"] + b["next_unit_estimate_gpu_h"] > 5.0
    )
    H.run_units(store, units(20, 21), quick, "t" * 64, "p", guard=g)  # no PhasePause any more
    assert len(recs(tmp_path, "attempt_ok")) == n_ok + 1
    assert (
        RG.read_study(tmp_path / "s").phase_resumes["bank"]["seq"] == rec["seq"]
    )  # a fresh replay agrees


def test_resume_is_refused_unless_a_pause_is_pending_and_only_once(tmp_path):
    clk = Clock()
    g = guard(tmp_path, clk)
    with pytest.raises(ValueError, match="no PhasePause pending"):
        g.resume_phase("bank", RULING)
    with pytest.raises(ValueError):
        g.resume_phase("bank", "")
    tp = tmp_path / "p2"
    tp.mkdir()
    _clk2, g2, _store = paused(tp)
    g2.resume_phase("bank", RULING)
    with pytest.raises(ValueError, match="already resumed"):
        g2.resume_phase("bank", RULING)


def test_the_resumption_lifts_only_that_phase_and_never_the_absolute_cap(tmp_path):
    clk = Clock()
    g = guard(tmp_path, clk, priced={"bank": 4.0, "e2": 1.0})
    g.ledger.add("bank", 4.5 * 3600, note="fixture")
    g.resume_phase("bank", RULING)
    assert set(RG.read_study(tmp_path / "s").phase_resumes) == {"bank"}
    g.ledger.check_before("bank", 1.0)  # resumed: no pause
    with pytest.raises(RG.PhasePause):
        g.ledger.check_before("e2", 2.0)  # e2: 0 + 2 > 1.25 x 1.0, still paused by its own rule
    with pytest.raises(RG.CapExceeded):
        g.ledger.check_before("bank", 500.0)  # the absolute cap still applies to the resumed phase


def test_a_paused_phase_without_a_resumption_still_pauses(tmp_path):
    _clk, g, store = paused(tmp_path)
    with pytest.raises(RG.PhasePause):
        H.run_units(store, units(30, 31), quick, "t" * 64, "p", guard=g)


# ------------------------------------------------------------------ the command


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2y", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cli_study(tmp_path, monkeypatch, measured_h):
    mod = load_ctl()
    runs = tmp_path / "runs"
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "STUDY", runs / "study")
    runs.mkdir()
    (runs / "attestation.json").write_text(json.dumps(RG._att_fields(ATT)))
    (runs / "bank_price.json").write_text(json.dumps({"P_bank": 4.0}))
    st = SD.Study(runs / "study", runs, ATT, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    g = st.guard("bank", "lambda:x")
    g.ledger.add("bank", measured_h * 3600, note="fixture")
    return mod, runs


def test_resume_phase_command_records_the_decision(tmp_path, monkeypatch):
    mod, runs = cli_study(tmp_path, monkeypatch, measured_h=4.9)  # 4.9 + 0.35 > 5.0
    out = mod.cmd_resume_phase(types.SimpleNamespace(phase="bank", ruling=RULING))
    assert (
        out["resumed_phase"] == "bank"
        and out["ruling"] == RULING
        and out["basis"]["threshold_gpu_h"] == 5.0
    )
    s = RG.read_study(runs / "study")
    assert s.phase_resumes["bank"]["decided_by"] == "chair"


def test_resume_phase_command_refuses_before_any_write_without_a_pending_pause(
    tmp_path, monkeypatch
):
    mod, runs = cli_study(tmp_path, monkeypatch, measured_h=1.0)
    log0 = (runs / "study" / RG.LOG_NAME).read_bytes()
    with pytest.raises(SystemExit, match="no PhasePause pending"):
        mod.cmd_resume_phase(types.SimpleNamespace(phase="bank", ruling=RULING))
    assert (runs / "study" / RG.LOG_NAME).read_bytes() == log0
