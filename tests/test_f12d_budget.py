"""F12d: Board A F12c condition 7, auditable budget stops.

A launch / checkpoint barrier on cumulative spent CPU (parent and worker process tree, every kind of work, failed and
aborted attempts) plus projected remaining CPU against the chair's 190 CPU-h ceiling; a projection from separate stress
and alternative costs with a disclosed conservative extrapolation, noise and overhead; the timing sidecar's schedule
projection for an alternative-only pilot; the reconciled resource-plan status. Fabricated timings only."""

import json
import subprocess
import sys
import time

import pytest

from scripts import f12_budget as fb
from scripts import f12_simulation as fs

PLAN = {
    "basis": "test plan",
    "items_cpu_h": {"noise_check": 0.01, "pilot": 0.6, "stage1": 30.0, "target": 12.0, "stage2": 140.0,
                    "overhead": 0.5},
}  # fmt: skip


def _meter(ledger, kind="stage1", units=480, cpu=None, **kw):
    clock = {"cpu": 0.0}
    m = fb.Meter(ledger, PLAN, job=f"{kind}-job", kind=kind, planned_units=units, operator="chair",
                 ceiling_cpu_h=190.0, cpu=cpu or (lambda: clock["cpu"]), **kw)  # fmt: skip
    return m, clock


def test_launch_barrier_counts_every_spent_attempt(tmp_path):
    L = fb.Ledger(tmp_path / "ledger.jsonl")
    m, _ = _meter(L)
    d = m.launch()
    assert d["projected_total_cpu_h"] == pytest.approx(sum(PLAN["items_cpu_h"].values()))
    assert d["ceiling_cpu_h"] == 190.0 and d["operator"] == "chair"
    # an aborted attempt of 9 CPU-h is spent: 9 + 183.11 > 190 -> the relaunch is refused
    m.end("aborted", cpu_s=9 * 3600)
    m2, _ = _meter(L)
    with pytest.raises(fb.BudgetStop, match="190"):
        m2.launch()
    evs = L.events()
    assert evs[-1]["event"] == "stop" and evs[-1]["status"] == "stopped_budget"
    assert L.spent_cpu_h() == pytest.approx(9.0)
    assert {e["kind"] for e in evs} == {"stage1"}


def test_failed_noise_and_pilot_attempts_and_overhead_are_spent(tmp_path):
    L = fb.Ledger(tmp_path / "ledger.jsonl")
    for kind, h, status in (("noise_check", 0.01, "failed"), ("noise_check", 0.01, "completed"),
                            ("pilot", 0.7, "completed"), ("assembly", 0.2, "completed"),
                            ("verification", 0.1, "completed")):  # fmt: skip
        m, _ = _meter(L, kind=kind, units=1)
        m.launch()
        m.end(status, cpu_s=h * 3600)
    assert L.spent_cpu_h() == pytest.approx(1.02)
    m, _ = _meter(L)
    d = m.launch()
    # completed kinds are not projected again; overhead projection is reduced by the spent overhead
    want = 1.02 + 30.0 + 12.0 + 140.0 + (0.5 - 0.3)
    assert d["projected_total_cpu_h"] == pytest.approx(want)


def test_checkpoint_stops_before_a_projected_breach_without_changing_counts(tmp_path):
    L = fb.Ledger(tmp_path / "ledger.jsonl")
    m, clock = _meter(L, kind="stage2", units=2200, every=10, max_seconds=10_000)
    m.launch()
    # 400 CPU-s per replicate: projected stage-2 remaining 2190 x 400 s = 243 CPU-h -> stop at the first checkpoint
    for done in range(1, 10):
        clock["cpu"] = done * 400.0
        m.tick(done)  # only every tenth replicate is a checkpoint
    assert [e["event"] for e in L.events()] == ["launch"]
    clock["cpu"] = 10 * 400.0
    with pytest.raises(fb.BudgetStop) as exc:
        m.tick(10)
    assert "no count or draw is reduced" in str(exc.value) and "no decision" in str(exc.value)
    assert m.planned_units == 2200
    stop = L.events()[-1]
    assert stop["event"] == "stop" and stop["status"] == "stopped_budget"
    assert stop["checkpoint_every_units"] == 10 and stop["done_units"] == 10
    assert stop["projected_total_cpu_h"] > 190.0 and stop["cpu_s"] == pytest.approx(4000.0)


def test_checkpoint_passes_when_within_the_ceiling(tmp_path):
    L = fb.Ledger(tmp_path / "ledger.jsonl")
    m, clock = _meter(L, kind="stage2", units=2200, every=10)
    m.launch()
    for done in range(1, 101):
        clock["cpu"] = (
            done * 200.0
        )  # below the planned 229 CPU-s per replicate: the plan rate is kept
        m.tick(done)
    cps = [e for e in L.events() if e["event"] == "checkpoint"]
    assert [e["done_units"] for e in cps] == list(range(10, 101, 10))
    assert all(e["projected_total_cpu_h"] <= 190.0 for e in cps)
    m.end("completed", cpu_s=None)
    ev = L.events()
    assert ev[-1]["event"] == "end" and ev[-1]["cpu_s"] == pytest.approx(20_000.0)


def test_time_based_checkpoint(tmp_path):
    L = fb.Ledger(tmp_path / "ledger.jsonl")
    m, _ = _meter(L, kind="stage1", every=1000, max_seconds=0.0)
    m.launch()
    m.tick(1)
    assert [e["event"] for e in L.events()] == ["launch", "checkpoint"]


def test_process_tree_cpu_includes_live_children():
    code = ("import time, sys\nt = time.process_time()\nwhile time.process_time() - t < 0.6: pass\n"
            "sys.stdout.write('x'); sys.stdout.flush(); time.sleep(30)")  # fmt: skip
    before = fb.process_tree_cpu_s()
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    try:
        assert p.stdout.read(1) == b"x"
        time.sleep(0.05)
        assert fb.process_tree_cpu_s() - before >= 0.5
    finally:
        p.kill()
        p.wait()


def test_simulate_stops_at_a_checkpoint_callback(monkeypatch):
    done = []

    def fake_task(t):
        done.append(t)
        return {"scenario": t[0], "k": t[1], "r": t[2]}, (0.0, 1.0, 1.0)

    monkeypatch.setattr(fs, "_task", fake_task)
    monkeypatch.setattr(fs, "_init_worker", lambda *a, **k: None)

    def on_done(n):
        if n == 3:
            raise fb.BudgetStop("projected breach")

    tasks = [("alt", 145, r) for r in range(10)]
    with pytest.raises(fb.BudgetStop):
        fs.simulate(None, None, fs.Config(), tasks=tasks, workers=1, on_done=on_done)
    assert len(done) == 3


def test_governed_real_token_runs_require_ledger_plan_and_operator(tmp_path):
    base = ["run", "--tokens", str(tmp_path / "t.npz"), "--out", str(tmp_path / "o"),
            "--restricted-payload", str(tmp_path / "p.json")]  # fmt: skip
    for name in ("pilot", "stage1"):
        with pytest.raises(SystemExit, match="budget"):
            fs.main(base + ["--config", str(fs.F12C_CONFIGS[name])])
    with pytest.raises(SystemExit, match="budget"):
        fs.main(["noisecheck", "--tokens", str(tmp_path / "t.npz"), "--restricted-payload",
                 str(tmp_path / "p.json"), "--out", str(tmp_path / "n")])  # fmt: skip


# --------------------------------------------------------------------------- projection


def test_project_cpu_separate_stress_and_alternative_with_noise_and_overhead():
    pr = fs.project_cpu({"145": {"alt": 219.0}}, noise_cpu_h=0.004, overhead_cpu_h=0.05)
    assert pr["stress_timing"].startswith("extrapolated")
    assert any("stress" in x and "alternative" in x for x in pr["extrapolations"])
    assert any("k = 145" in x for x in pr["extrapolations"])
    s2 = pr["stage2_by_k"]["0"]
    assert (
        s2["null_s"] == s2["alt_s"] == s2["null_var2_s"] == 219.0
    )  # conservative: as costly as k = 145
    assert pr["stage1_cpu_h"] == pytest.approx(480 * 219 / 3600)
    assert pr["all_in_worst_cpu_h"] == pytest.approx((480 + 2200 + 200 + 10) * 219 / 3600 + 0.054)
    assert "not measured stress timing" in pr["label"]
    both = fs.project_cpu(
        {"145": {"null": 202.0, "alt": 219.0}}, noise_cpu_h=0.0, overhead_cpu_h=0.0
    )
    assert both["stress_timing"] == "measured"
    assert both["stage2_by_k"]["145"]["null_s"] == 202.0
    assert both["stage2_by_k"]["145"]["null_var2_s"] == 219.0  # max(stress, alternative), disclosed
    stress_only = fs.project_cpu({"145": {"null": 202.0}}, noise_cpu_h=0.0, overhead_cpu_h=0.0)
    assert stress_only["stage2_by_k"]["145"]["alt_s"] == pytest.approx(
        202.0 / fs.SYNTH_STRESS_ALT_RATIO_MIN
    )
    assert stress_only["alternative_timing"].startswith("extrapolated")
    default = fs.project_cpu({"145": {"alt": 219.0}})
    assert default["noise_cpu_h_source"].startswith("projected") and default["noise_cpu_h"] > 0
    assert (
        default["overhead_cpu_h_source"].startswith("projected") and default["overhead_cpu_h"] > 0
    )


def test_write_timing_projects_an_alternative_only_pilot(tmp_path):
    class A:
        workers = 2

    (tmp_path / "report.json").write_text("{}")
    ru0 = fs._rusage_cpu()
    recs = [{"k": 145, "scenario": "alt", "r": 3000 + i} for i in range(3)]
    times = [(10.0, 219.0, 800.0)] * 3
    fs._write_timing(
        tmp_path, A(), recs, times, 30.0, ru0, [145], started_utc="2026-11-12T10:00:00Z"
    )
    t = json.loads((tmp_path / "report.time").read_text())
    assert t["schema"] == fs.TIMING_SCHEMA
    assert t["projection_f12c_schedule"] is not None
    assert t["projection_f12c_schedule"]["stress_timing"].startswith("extrapolated")
    assert t["started_utc"] == "2026-11-12T10:00:00Z" and t["finished_utc"] >= t["started_utc"]
    assert t["report_sha256"] == fs.sha256_file(tmp_path / "report.json")


# ------------------------------------------------------------------ resource plan


def test_resource_plan_status_reconciled_with_the_chair_ruling():
    plan = json.loads((fs.ROOT / "docs/prereg/f12_manifests/f12c_resource_plan.json").read_text())
    assert (
        "requesting" not in plan["status"] and "acceptance of the repricing" not in plan["status"]
    )
    acc = plan["ceiling_acceptance"]
    assert acc["all_in_ceiling_cpu_h"] == 190 and acc["utc"] == "2026-11-11T18:55Z"
    assert "chair ruling on F12c" in acc["source"]
    assert plan["pilot_acceptance"]["status"].startswith("not written")
    old = plan["status_history"][0]
    assert "chair's acceptance of the repricing" in old["status"] and old[
        "superseded_by"
    ].startswith("F12d")
    assert set(plan["items_by_kind_cpu_h"]) == set(fb.PLAN_KINDS)
    assert plan["budget_barrier"]["checkpoint_every_units"] == fb.CHECKPOINT_EVERY_UNITS
    assert plan["budget_barrier"]["checkpoint_max_seconds"] == fb.CHECKPOINT_MAX_SECONDS
    assert plan["budget_barrier"]["ceiling_cpu_h"] == 190.0
