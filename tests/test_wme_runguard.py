"""Validity/repair/stopping contract (wsp.wm.runguard; Board A v2 required change 6)."""

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG


def test_combined_cap_stops_before_any_unit_that_would_cross_it(tmp_path):
    led = RG.CombinedLedger(tmp_path / "l.jsonl", {"bank": 3.0, "e2": 80.0})
    led.add("e2", 89.5 * 3600)
    with pytest.raises(RG.CapExceeded):
        led.check_before("e2", 0.6)
    led2 = RG.CombinedLedger(tmp_path / "l.jsonl", {"bank": 3.0, "e2": 80.0})  # persisted
    assert led2.gpu_h() == pytest.approx(89.5)
    assert led2.remaining_usd() == pytest.approx(0.5 * RG.USD_PER_GPU_H)


def test_phase_pause_at_125x(tmp_path):
    led = RG.CombinedLedger(tmp_path / "l.jsonl", {"bank": 3.0})
    led.add("bank", 3.7 * 3600)
    with pytest.raises(RG.PhasePause):
        led.check_before("bank", 0.1)


def test_run_units_honours_the_guard(tmp_path):
    led = RG.CombinedLedger(tmp_path / "l.jsonl", {"x": 1.0})
    store = H.UnitStore(tmp_path / "u")
    g = RG.UnitGuard(led, "x", est_unit_gpu_h=0.4)
    done = H.run_units(
        store, [({"a": 1}, "u1"), ({"a": 2}, "u2")], lambda r: r, "t" * 64, "p", guard=g
    )
    assert set(done) == {"u1", "u2"} and len(led.rows) == 2
    with pytest.raises(RG.PhasePause):  # 0 + 0.4 + 0.4 + 0.4 > 1.25 x 1.0
        H.run_units(
            store,
            [({"a": 3}, "u3"), ({"a": 4}, "u4"), ({"a": 5}, "u5"), ({"a": 6}, "u6")],
            lambda r: r,
            "t" * 64,
            "p",
            guard=RG.UnitGuard(led, "x", est_unit_gpu_h=1.3),
        )


def test_attempts_and_relaunches_are_separate_counters():
    rp = RG.RetryPolicy()
    assert rp.attempt("ep3:wm") == 1
    assert rp.attempt("ep3:wm", "out_of_memory") == 2
    rp.relaunch("e2", "infrastructure")  # the attempt count survives the relaunch
    assert rp.attempt("ep3:wm", "infrastructure") == 3
    with pytest.raises(RG.RetryExhausted):
        rp.attempt("ep3:wm", "infrastructure")
    with pytest.raises(RG.RetryExhausted):
        rp.relaunch("e2", "infrastructure")
    with pytest.raises(ValueError):
        rp.attempt("ep4:wm", "winner looked bad")


def test_repairs_must_be_technical_and_outcome_blind():
    rp = RG.RetryPolicy()
    rp.repair("out_of_memory", "micro-batch 8 -> 4", "abcdef1", outcome_blind=True)
    with pytest.raises(ValueError):
        rp.repair("improve_score", "retune", "abcdef1", outcome_blind=True)
    with pytest.raises(ValueError):
        rp.repair("out_of_memory", "x", "abcdef1", outcome_blind=False)


def test_units_before_a_failed_end_canary_are_retained_ineligible_and_rerun_by_time_order():
    cb = RG.CanaryBook("REF")
    cb.canary("i1", 0, "REF")
    cb.canary("i1", 10, "BAD")  # end canary fails
    cb.canary("i2", 20, "REF")
    cb.canary("i2", 40, "REF")
    assert cb.status("i1", 5) == "retained_ineligible"
    assert cb.status("i2", 25) == "eligible"
    # first canary-valid attempt wins regardless of value
    attempts = [("i1", 5, 0.9), ("i2", 30, 0.1), ("i2", 35, 0.8)]
    assert cb.eligible_result(attempts) == 0.1
    assert cb.status("i2", 50) == "retained_ineligible"  # no closing canary yet
