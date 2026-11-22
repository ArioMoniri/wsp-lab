"""RULING-wme-c1-bank-halt addendum 5 (run6 bank-start halt), v2w: (1) the §II.18 order monitor scoped to the
contract generation; (2) ``resolve-incident`` (one incident, its diagnosis and the ruling it cites); (3) heartbeat
hardening, the ``preflight`` step and the two-heartbeat gate before any request to a live instance; (4) drill
scenario I is in the drill (tested by running the drill). CPU only; fixtures."""

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import health as HL
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import worker as W

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


def guard(tmp_path, att, inst="lambda:i1", transition=None):
    g = RG.ProductionGuard(
        tmp_path / "s", "bank", {"bank": 500.0}, att, inst, 0.01, technical=C.technical,
        fallback_value={"rule": "unlabelled"}, transition=transition,
    )  # fmt: skip
    g.open_instance(1)
    g.canary(REF)
    return g


def recs(tmp_path, t=None):
    rs = [json.loads(x) for x in (tmp_path / "s" / RG.LOG_NAME).read_text().splitlines()]
    return rs if t is None else [r for r in rs if r["type"] == t]


def load_ctl():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v2w", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 1. the order monitor per contract generation


def test_a_transition_after_the_bank_stage_then_a_field_unit_gives_no_order_incident(tmp_path):
    mon = HL.HealthMonitor(tmp_path / "runs")
    g0 = guard(tmp_path, ATT0)
    mon.check_order(g0, {"kind": "field_group"})
    mon.check_order(
        g0, {"kind": "bank_side"}
    )  # generation 0 reached bank_states (the run3 position)
    g1 = guard(tmp_path, ATT1, transition=TR)
    mon.check_order(
        g1, {"kind": "field_group"}
    )  # generation 1 starts at the first registered stage
    mon.check_order(g1, {"kind": "bank_side"})
    assert not recs(tmp_path, "incident")
    st = json.loads((tmp_path / "runs/health/state.json").read_text())
    assert st["generation"] == 1 and st["stage"] == HL.STAGE_OF["bank_side"]
    assert st["history"] == [
        {"generation": 0, "stage": HL.STAGE_OF["bank_side"], "skipped": [], "skip_reasons": {}}
    ]


def test_the_run6_state_without_a_generation_is_generation_0(tmp_path):
    d = tmp_path / "runs/health"
    d.mkdir(parents=True)
    (d / "state.json").write_text(
        json.dumps({"chunks": 1, "log_offset": 11976, "skipped": [], "stage": 1})
    )
    mon = HL.HealthMonitor(tmp_path / "runs")
    guard(tmp_path, ATT0)
    g1 = guard(tmp_path, ATT1, transition=TR)
    mon.check_order(g1, {"kind": "field_group"})
    assert not recs(tmp_path, "incident")
    st = json.loads((d / "state.json").read_text())
    assert st["log_offset"] == 11976 and st["chunks"] == 1  # the other monitor state is kept


def test_within_one_generation_the_order_rule_is_unchanged(tmp_path):
    mon = HL.HealthMonitor(tmp_path / "runs")
    g0 = guard(tmp_path, ATT0)
    mon.check_order(g0, {"kind": "field_group"})
    mon.check_order(g0, {"kind": "bank_side"})
    with pytest.raises(HL.HealthIncident):
        mon.check_order(g0, {"kind": "field_group"})  # going back within the generation
    assert [r["kind"] for r in recs(tmp_path, "incident")] == ["order"]


def test_skip_declarations_are_scoped_to_their_generation(tmp_path):
    runs = tmp_path / "runs"
    st0 = SD.Study(tmp_path / "s", runs, ATT0, "t" * 64, "p", log=lambda *a: None)
    st0.guard("bank", "lambda:i1")
    st0.skip_stage("e1_gen", "E1 not designated (generation 0)")
    assert json.loads((runs / "health/state.json").read_text())["skipped"] == ["e1_gen"]
    st1 = SD.Study(tmp_path / "s", runs, ATT1, "t" * 64, "p", log=lambda *a: None)
    g1 = st1.guard("bank", "lambda:i1", transition=TR)
    g1.open_instance(1)
    mon = HL.HealthMonitor(runs)
    mon.check_order(g1, {"kind": "field_group"})
    st = json.loads((runs / "health/state.json").read_text())
    assert st["generation"] == 1 and st["skipped"] == []  # generation 0's skip does not carry over
    st1.skip_stage("ste_calibrate", "fixture")
    assert json.loads((runs / "health/state.json").read_text())["skipped"] == ["ste_calibrate"]


# ------------------------------------------------------------------ 2. resolve-incident


def test_resolve_incident_closes_exactly_one_incident_with_diagnosis_and_ruling(tmp_path):
    g = guard(tmp_path, ATT0)
    a = g.incident("order", {"x": 1})
    b = g.incident("worker_error", {"y": 2})
    g.resolve_incident(
        a["seq"], "order pointer not generation-scoped, fixed in v2w", "RULING addendum 5"
    )
    assert [i["seq"] for i in g.open_incidents()] == [b["seq"]]
    r = recs(tmp_path, "incident_resolved")[-1]
    assert (
        r["incident_seq"] == a["seq"]
        and "v2w" in r["diagnosis"]
        and r["ruling"] == "RULING addendum 5"
    )
    with pytest.raises(ValueError, match="already resolved"):
        g.resolve_incident(a["seq"], "again", "RULING addendum 5")
    with pytest.raises(ValueError, match="no incident"):
        g.resolve_incident(10_000, "unknown", "RULING addendum 5")
    with pytest.raises(ValueError, match="no incident"):
        g.resolve_incident(1, "seq 1 is not an incident record", "RULING addendum 5")
    with pytest.raises(ValueError):
        g.resolve_incident(b["seq"], "", "RULING addendum 5")
    g.resolve_incident(b["seq"], "close canary against a terminated instance", "RULING addendum 5")
    assert not g.open_incidents()
    assert RG.read_study(tmp_path / "s").incidents == []  # a fresh replay agrees


def test_resolve_incident_refuses_an_incident_closed_by_a_bulk_resolution(tmp_path):
    g = guard(tmp_path, ATT0)
    a = g.incident("order", {"x": 1})
    g.resolve_incidents("bulk diagnosis", "bulk report")
    with pytest.raises(ValueError, match="already resolved"):
        g.resolve_incident(a["seq"], "late", "RULING addendum 5")


def test_resolve_incident_command(tmp_path, monkeypatch):
    mod = load_ctl()
    runs = tmp_path / "runs"
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "STUDY", runs / "study")
    runs.mkdir()
    (runs / "attestation.json").write_text(json.dumps(RG._att_fields(ATT0)))
    st = SD.Study(runs / "study", runs, ATT0, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    g = st.guard("bank", "lambda:x")
    inc = g.incident("order", {"kind": "field_group"})
    log0 = (runs / "study" / RG.LOG_NAME).read_bytes()
    a = types.SimpleNamespace(
        seq=inc["seq"] + 1, diagnosis="d", ruling="RULING-wme-c1-bank-halt.md addendum 5"
    )
    with pytest.raises(SystemExit, match="no incident"):
        mod.cmd_resolve_incident(a)
    assert (runs / "study" / RG.LOG_NAME).read_bytes() == log0  # refused before any write
    a = types.SimpleNamespace(seq=inc["seq"], diagnosis="order pointer not generation-scoped, fixed in v2w",
                              ruling="RULING-wme-c1-bank-halt.md addendum 5")  # fmt: skip
    out = mod.cmd_resolve_incident(a)
    assert out["incident_seq"] == inc["seq"] and out["kind"] == "order" and out["open_after"] == []
    with pytest.raises(SystemExit, match="already resolved"):
        mod.cmd_resolve_incident(a)


# ------------------------------------------------------------------ 3. heartbeat hardening, preflight, the gate


def run_hb(env, *args):
    return subprocess.run(
        ["bash", str(ROOT / "scripts/wme_c1_hb.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )


def test_hb_script_fails_loudly_without_its_python(tmp_path):
    env = dict(os.environ, WME_HB_PYTHON=str(tmp_path / "no-python"))
    p = run_hb(env, "dry", str(tmp_path / "state"), "/nonexistent")
    assert p.returncode != 0 and "FATAL" in (p.stdout + p.stderr)


def test_hb_script_runs_once_against_a_dry_target(tmp_path):
    env = dict(os.environ, WME_HB_PYTHON=sys.executable)
    p = run_hb(env, "dry", str(tmp_path / "state"), "/nonexistent")
    assert p.returncode == 0 and "no live instance" in p.stdout, p.stdout + p.stderr


def test_hb_script_records_each_progress_heartbeat():
    src = (ROOT / "scripts/wme_c1_hb.sh").read_text()
    assert "heartbeats.jsonl" in src and '"source": "progress"' in src and "$ALIAS" in src


@pytest.fixture
def dm_state(tmp_path, monkeypatch):
    mod = load_ctl()
    st = tmp_path / "dm"
    st.mkdir()
    monkeypatch.setattr(mod.LO, "STATE", st)
    return mod, st


def write_dm(st, launched=1000.0, last_tick=None, actions=()):
    (st / "state.json").write_text(json.dumps({
        "instances": {"IID": {"launched_at": launched, "confirmed_terminated_at": None}},
        **({"last_tick": last_tick} if last_tick is not None else {}),
    }))  # fmt: skip
    (st / "actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in actions))


def test_two_progress_heartbeats_from_the_live_worker_are_required(dm_state):
    mod, st = dm_state
    write_dm(st, launched=1000.0)
    hb = st / "heartbeats.jsonl"
    hb.write_text(
        json.dumps({"t": 900.0, "alias": "wmec1", "source": "progress"}) + "\n"
    )  # before the launch
    hb.write_text(
        hb.read_text() + json.dumps({"t": 1100.0, "alias": "wmec1p", "source": "progress"}) + "\n"
    )
    hb.write_text(
        hb.read_text() + json.dumps({"t": 1200.0, "alias": "wmec1", "source": "progress"}) + "\n"
    )
    with pytest.raises(SystemExit, match="heartbeat"):
        mod.require_heartbeats("IID", "wmec1", wait_s=0)
    hb.write_text(
        hb.read_text() + json.dumps({"t": 1260.0, "alias": "wmec1", "source": "progress"}) + "\n"
    )
    assert mod.require_heartbeats("IID", "wmec1", wait_s=0) == 2


def test_every_request_to_an_instance_passes_the_heartbeat_gate(dm_state, monkeypatch):
    mod, _st = dm_state
    seen = []
    monkeypatch.setattr(
        mod, "require_heartbeats", lambda iid, alias, **k: seen.append((iid, alias)) or 2
    )
    monkeypatch.setattr(
        mod.LO, "deadman", lambda: types.SimpleNamespace(state=lambda: {"instances": {}})
    )
    mod.remote_for("main", "IID")
    assert seen == [("IID", "wmec1")]
    monkeypatch.setattr(
        mod, "require_heartbeats", lambda *a, **k: (_ for _ in ()).throw(SystemExit("no heartbeat"))
    )
    with pytest.raises(SystemExit, match="heartbeat"):
        mod.remote_for("parity", "IID")


def test_tick_health(dm_state):
    mod, st = dm_state
    write_dm(st, last_tick=100.0, actions=[{"event": "start", "t": 50}])
    ticks = iter([130.0, 160.0])

    def advance(_s):
        write_dm(st, last_tick=next(ticks), actions=[{"event": "start", "t": 50}])

    ok = mod.tick_health(st, watcher_running=True, need=2, wait_s=60, sleep=advance, poll_s=1)
    assert ok["pass"] is True and ok["ticks"] == 2
    write_dm(st, last_tick=100.0, actions=[{"event": "start", "t": 50}])
    assert (
        mod.tick_health(st, watcher_running=True, need=2, wait_s=0, sleep=lambda s: None)["pass"]
        is False
    )
    write_dm(
        st, last_tick=100.0, actions=[{"event": "start", "t": 50}, {"event": "error", "t": 60}]
    )
    ticks = iter([130.0, 160.0])
    bad = mod.tick_health(st, watcher_running=True, need=2, wait_s=60, sleep=lambda s: write_dm(
        st, last_tick=next(ticks), actions=[{"event": "start", "t": 50}, {"event": "error", "t": 60}]), poll_s=1)  # fmt: skip
    assert bad["pass"] is False and bad["errors_since_start"] == 1
    assert (
        mod.tick_health(st, watcher_running=False, need=2, wait_s=0, sleep=lambda s: None)["pass"]
        is False
    )


def test_preflight_checks_the_venv_link_and_the_dry_heartbeat(tmp_path, monkeypatch):
    mod = load_ctl()
    fake = tmp_path / "wt"
    (fake / "scripts").mkdir(parents=True)
    for n in ("wme_c1_hb.sh",):
        (fake / "scripts" / n).write_text((ROOT / "scripts" / n).read_text())
    monkeypatch.setattr(mod, "ROOT", fake)
    monkeypatch.setattr(mod, "tick_health", lambda *a, **k: {"pass": True})
    monkeypatch.setattr(mod, "watcher_running", lambda: True)
    rep = mod.preflight_checks(wait_s=0)
    assert rep["venv"]["pass"] is False and rep["pass"] is False
    (fake / ".venv/bin").mkdir(parents=True)
    os.symlink(sys.executable, fake / ".venv/bin/python")
    (fake / "scripts/lambda_deadman.py").symlink_to(ROOT / "scripts/lambda_deadman.py")
    rep = mod.preflight_checks(wait_s=0)
    assert rep["venv"]["pass"] is True
    assert rep["heartbeat_dry_run"]["pass"] is True, rep["heartbeat_dry_run"]
    assert rep["pass"] is True


def test_up_runs_the_preflight_before_any_poll():
    src = (ROOT / "scripts/wme_c1_ctl.py").read_text()
    body = src[src.index("def cmd_up(") : src.index("\ndef ", src.index("def cmd_up(") + 10)]
    assert "preflight_checks(" in body and body.index("preflight_checks(") < body.index(
        "wme_lambda_poll.py"
    )


def test_the_worker_records_its_start_as_progress(tmp_path):
    wk = W.Worker(tmp_path / "w", {}, poll_s=0.01)
    wk.serve(max_idle_s=0.0)
    ev = [json.loads(x) for x in (tmp_path / "w/progress.jsonl").read_text().splitlines()]
    assert ev[0]["event"] == "worker_start"


# ------------------------------------------------------------------ 4. drill scenario I


def test_the_drill_covers_a_transition_followed_by_bank_unit_issue():
    src = (ROOT / "scripts/wme_c1_drill.py").read_text()
    assert '"I_transition_then_bank"' in src
    from wsp.wm.c1 import drillworker as DWK

    assert {"field_group", "bank_side"} <= set(DWK.HANDLERS)
