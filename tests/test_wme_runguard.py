"""Validity/repair/stopping contract enforced in the production path (wsp.wm.runguard, harness.run_units;
Board A v2 change 6 and confirmation blocker B)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG

REF = "a" * 64
BAD = "b" * 64
ATTEST = RG.LaunchAttestation(
    inert_identity_bitwise=True,
    gpu_parity_bitwise=True,
    canary_reference=REF,
    engine_config_sha256="c" * 64,
    evidence=["docs/reviews/wme/wme-ph100-2026-11-17.json"],
)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_combined_cap_stops_before_any_unit_that_would_cross_it(tmp_path):
    led = RG.CombinedLedger(tmp_path / "l.jsonl", {"bank": 3.0, "e2": 80.0})
    led.add("e2", 89.5 * 3600)
    with pytest.raises(RG.CapExceeded):
        led.check_before("e2", 0.6)
    led2 = RG.CombinedLedger(tmp_path / "l.jsonl", {"bank": 3.0, "e2": 80.0})  # persisted
    assert led2.gpu_h() == pytest.approx(89.5)
    assert led2.remaining_usd() == pytest.approx(0.5 * RG.USD_PER_GPU_H)


def test_instance_time_is_authoritative_for_the_absolute_cap(tmp_path):
    clk = Clock()
    led = RG.CombinedLedger(tmp_path / "l.jsonl", {"e2": 200.0}, clock=clk)
    led.instance_open("i1", gpus=1)
    led.add("e2", 3600.0)  # one measured GPU-hour of units
    clk.t = 89.8 * 3600  # setup, downloads and idle make the instance time much larger
    assert led.gpu_h() == pytest.approx(89.8) and led.unit_gpu_h() == pytest.approx(1.0)
    with pytest.raises(RG.CapExceeded):
        led.check_before("e2", 0.5)
    led.instance_close("i1")
    assert RG.CombinedLedger(
        tmp_path / "l.jsonl", {"e2": 200.0}, clock=clk
    ).instance_gpu_h() == pytest.approx(89.8)


def test_phase_pause_at_125x(tmp_path):
    led = RG.CombinedLedger(tmp_path / "l.jsonl", {"bank": 3.0})
    led.add("bank", 3.7 * 3600)
    with pytest.raises(RG.PhasePause):
        led.check_before("bank", 0.1)


def test_counters_persist_across_restarts(tmp_path):
    rp = RG.RetryPolicy(tmp_path / "r.jsonl")
    assert rp.attempt("u") == 1 and rp.attempt("u", "out_of_memory") == 2
    rp.relaunch("e2", "infrastructure")
    rp2 = RG.RetryPolicy(tmp_path / "r.jsonl")  # a new process
    assert rp2.attempt("u", "infrastructure") == 3
    with pytest.raises(RG.RetryExhausted):
        rp2.attempt("u", "infrastructure")
    with pytest.raises(RG.RetryExhausted):
        rp2.relaunch("e2", "infrastructure")
    with pytest.raises(ValueError):
        rp2.attempt("v", "winner looked bad")
    # one log: the ledger and the counters are views of the same hash-chained records (§II.27)
    led = RG.CombinedLedger(tmp_path / "r.jsonl", {"e2": 1.0})
    led.add("e2", 60.0)
    rp3 = RG.RetryPolicy(tmp_path / "r.jsonl")
    assert rp3.attempts == {"u": 3} and rp3.relaunches == {"e2": 1}
    assert led.unit_gpu_h() == pytest.approx(60 / 3600)
    assert [r["seq"] for r in led.rows] == list(range(1, len(led.rows) + 1))


def test_repairs_must_be_technical_and_outcome_blind():
    rp = RG.RetryPolicy()
    rp.repair("out_of_memory", "micro-batch 8 -> 4", "abcdef1", outcome_blind=True)
    with pytest.raises(ValueError):
        rp.repair("improve_score", "retune", "abcdef1", outcome_blind=True)
    with pytest.raises(ValueError):
        rp.repair("out_of_memory", "x", "abcdef1", outcome_blind=False)


def test_launch_requires_attestations_and_run_units_requires_a_guard(tmp_path):
    bad = RG.LaunchAttestation(True, False, REF, "c" * 64, ["x"])
    with pytest.raises(RG.NotAttested):
        RG.ProductionGuard(tmp_path, "e2", {"e2": 1.0}, bad, "i1", 0.01)
    with pytest.raises(RG.NotAttested):
        RG.ProductionGuard(
            tmp_path,
            "e2",
            {"e2": 1.0},
            RG.LaunchAttestation(True, True, "x", "c" * 64, ["e"]),
            "i1",
            0.01,
        )
    with pytest.raises(ValueError):
        H.run_units(
            H.UnitStore(tmp_path / "u"), [({"a": 1}, "u1")], lambda r: r, "t" * 64, "p", guard=None
        )
    with pytest.raises(TypeError):
        H.run_units(H.UnitStore(tmp_path / "u"), [({"a": 1}, "u1")], lambda r: r, "t" * 64, "p")


def _technical(exc):
    return "out_of_memory" if isinstance(exc, MemoryError) else None


H1 = 3600.0
UNITS = [({"u": 1}, "u1"), ({"u": 2}, "u2"), ({"u": 3}, "u3")]


def _key(i):
    return H.unit_key({"u": i}, "t" * 64, "p")


def _mk(state, inst, clk, att=ATTEST, fallback="base", **kw):
    return RG.ProductionGuard(
        state,
        "e2",
        {"e2": 200.0},
        att,
        inst,
        0.01,
        technical=_technical,
        fallback_value=fallback,
        clock=clk,
        **kw,
    )


def _ok(recipe):
    return {"unit": f"u{recipe['u']}"}


# ------------------------------------------------------------------ Board A final re-check probes (gaps 1-4)


def test_probe_gap1_no_unit_runs_without_an_open_instance(tmp_path):
    clk = Clock()
    g = _mk(tmp_path / "s", "i1", clk)
    with pytest.raises(
        RG.InstanceNotOpen
    ):  # no canary either without a registered instance (§II.26)
        g.canary(REF)
    clk.t = 91 * H1  # 91 hours pass with no registered instance
    with pytest.raises(RG.InstanceNotOpen):
        H.run_units(H.UnitStore(tmp_path / "u"), UNITS[:1], _ok, "t" * 64, "p", guard=g)
    assert g.ledger.unit_gpu_h() == 0
    # with the instance registered from launch, the same 91 h trip the absolute cap
    g2 = _mk(tmp_path / "s2", "i1", Clock())
    g2.open_instance()
    g2.canary(REF)
    g2.clock.t = 91 * H1
    with pytest.raises(RG.CapExceeded):
        H.run_units(H.UnitStore(tmp_path / "u2"), UNITS[:1], _ok, "t" * 64, "p", guard=g2)


def test_probe_gap1_restart_and_reopen_never_reset_instance_time(tmp_path):
    clk = Clock()
    g = _mk(tmp_path / "s", "i1", clk)
    g.open_instance()
    clk.t = 89 * H1
    g2 = _mk(tmp_path / "s", "i1", clk)  # process restart
    g2.open_instance()  # duplicate opening: idempotent, the earliest opening is kept
    assert g2.ledger.instance_gpu_h() == pytest.approx(89.0)
    assert [e["type"] for e in g2.events if e["type"].startswith("instance_")] == [
        "instance_open",
        "instance_resume",
    ]
    # an authoritative record from the terminator/provider overrides local times (earlier launch, termination)
    g2.reconcile_instance(
        launched_at=-1 * H1, terminated_at=89.5 * H1, gpus=1, source="dead-man ledger"
    )
    clk.t = 100 * H1
    assert g2.ledger.instance_gpu_h() == pytest.approx(90.5)
    assert g2.ledger.instance_state("i1") == "closed"
    with pytest.raises(RG.InstanceNotOpen):
        g2.begin(_key(1), "u1")
    with pytest.raises(ValueError):  # a closed id cannot be reopened, so its time can never restart
        g2.open_instance()
    with pytest.raises(
        ValueError
    ):  # an authoritative launch later than the recorded opening is refused
        g2.ledger.reconcile_instance("i1", 5 * H1, None, 1, "x")


def test_probe_gap2_changed_engine_config_cannot_return_old_payload(tmp_path):
    clk, store = Clock(), H.UnitStore(tmp_path / "u")
    g = _mk(tmp_path / "s", "i1", clk)
    g.open_instance()
    g.canary(REF)
    H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=g)
    g.canary(REF)
    assert g.resolve(store, _key(1)) == {"unit": "u1"}  # eligible under the registered contract
    changed = RG.LaunchAttestation(True, True, REF, "d" * 64, ATTEST.evidence)
    with pytest.raises(RG.ContractMismatch):  # refuse to run or return results
        _mk(tmp_path / "s", "i1", clk, att=changed)
    with pytest.raises(ValueError):  # a transition must be technical and outcome-blind
        _mk(
            tmp_path / "s",
            "i1",
            clk,
            att=changed,
            transition=RG.ContractTransition("better scores", "x", "abcdef1", True),
        )
    tr = RG.ContractTransition("driver_or_cuda", "driver 580 -> 581 after a crash", "abcdef1", True)
    g2 = _mk(tmp_path / "s", "i1", clk, att=changed, transition=tr)
    assert (
        g2.resolve(store, _key(1)) is None
    )  # the old result is not eligible under the new contract
    assert g2.unit_status(_key(1))["attempts"][0][1] == "superseded_contract"
    with pytest.raises(RG.CanaryMismatch):  # the new contract needs its own canary
        g2.begin(_key(1), "u1")
    g2.canary(REF)
    out = H.run_units(
        store, UNITS[:1], lambda r: {"unit": "u1", "rerun": True}, "t" * 64, "p", guard=g2
    )
    assert out["u1"]["status"] == "pending_canary"
    g3 = _mk(
        tmp_path / "s", "i1", clk, att=changed
    )  # resume under the transitioned contract: no new transition
    assert [e["type"] for e in g3.events].count("contract_transition") == 1
    with pytest.raises(
        RG.ContractMismatch
    ):  # going back to the old contract also needs a transition
        _mk(tmp_path / "s", "i1", clk)


def test_probe_gap3_lost_end_canary_allows_the_remaining_attempts(tmp_path):
    clk, store = Clock(), H.UnitStore(tmp_path / "u")
    g = _mk(tmp_path / "s", "i1", clk)
    g.open_instance()
    g.canary(REF)
    H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=g)  # pending; then i1 is preempted
    clk.t = 2 * H1
    g2 = _mk(tmp_path / "s", "i1", clk)
    g2.relaunch("preemption_or_capacity", "i2")
    g2.open_instance()
    assert g2.canary(REF)
    with pytest.raises(
        RG.CanaryMismatch
    ):  # the old pending attempt still blocks, until a disposition
        g2.begin(_key(1), "u1")
    n_canaries_i1 = sum(1 for e in g2.events if e["type"] == "canary" and e["instance"] == "i1")
    g2.declare_lost(
        "i1",
        "preemption_or_capacity",
        "provider preempted i1",
        "abcdef1",
        True,
        terminated_at=1.5 * H1,
    )
    assert (
        sum(1 for e in g2.events if e["type"] == "canary" and e["instance"] == "i1")
        == n_canaries_i1
    )
    assert g2.unit_status(_key(1))["attempts"][0][1] == "retained_ineligible"
    assert g2.ledger.instance_state("i1") == "closed"
    # i1 is charged max(local time to the loss declaration, 2.0 h; its termination record, 1.5 h); i2 just opened
    assert g2.ledger.instance_gpu_h() == pytest.approx(2.0)
    out = H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=g2)
    assert out["u1"]["attempt_key"].endswith(".a2")  # the attempt count is kept
    with pytest.raises(ValueError):
        g2.declare_lost("i3", "winner looked bad", "x", "abcdef1", True)


def test_probe_gap4_changed_fallback_value_is_refused_on_resume(tmp_path):
    clk, store = Clock(), H.UnitStore(tmp_path / "u")
    g = _mk(tmp_path / "s", "i1", clk, fallback={"winner": "base"})
    g.open_instance()
    g.canary(REF)

    def boom(recipe):
        raise MemoryError("OOM")

    out = H.run_units(store, UNITS[:1], boom, "t" * 64, "p", guard=g)
    assert out["u1"]["fallback"] and out["u1"]["value"] == {"winner": "base"}
    with pytest.raises(RG.FallbackMismatch):
        _mk(tmp_path / "s", "i1", clk, fallback={"winner": "other"})
    g2 = _mk(tmp_path / "s", "i1", clk, fallback={"winner": "base"})
    r = g2.resolve(store, _key(1))
    assert r["value"] == {"winner": "base"} and r["payload_sha256"] == g2.fallback_sha256


def test_orchestration_failure_canary_restart_preemption_exhaustion_and_first_valid_selection(
    tmp_path,
):
    """One run through every registered rule: a missing instance and canary refused; a technical failure retried
    and charged; a unit exhausting 3 attempts and taking its registered fallback; a failed end canary; a process
    restart that reloads every counter without resetting instance time; a preempted instance with no closing canary
    declared lost (no canary fabricated); the registered relaunch; and the first canary-valid attempt selected on
    resume without recomputation."""
    state, store = tmp_path / "state", H.UnitStore(tmp_path / "units")
    calls = {"u1": 0, "u2": 0, "u3": 0}

    def fn(recipe):
        lab = f"u{recipe['u']}"
        calls[lab] += 1
        if lab == "u3" or (lab == "u2" and calls["u2"] == 1):
            raise MemoryError("OOM")  # technical
        return {"unit": lab, "call": calls[lab]}

    clk = Clock()
    g = _mk(state, "i1", clk)
    with pytest.raises(RG.InstanceNotOpen):
        H.run_units(store, UNITS, fn, "t" * 64, "p", guard=g)
    g.open_instance()
    with pytest.raises(RG.CanaryMismatch):  # no start canary yet
        H.run_units(store, UNITS, fn, "t" * 64, "p", guard=g)
    assert g.canary(REF)
    out = H.run_units(store, UNITS, fn, "t" * 64, "p", guard=g)
    assert out["u1"]["status"] == out["u2"]["status"] == "pending_canary"
    assert out["u3"]["fallback"] and out["u3"]["value"] == "base" and calls["u3"] == 3
    assert (
        len([r for r in g.ledger.rows if r["type"] == "unit_time" and r["status"] == "failed"]) == 4
    )
    assert not g.canary(BAD)  # failed end canary: u1.a1 and u2.a2 are retained-ineligible
    g.retry.repair(
        "canary_mismatch", "re-pinned the CUDA library path", "abcdef1", outcome_blind=True
    )
    clk.t = 3 * H1

    g2 = _mk(state, "i1", clk)  # process restart on the same instance
    g2.open_instance()  # resume: idempotent
    assert g2.ledger.instance_gpu_h() == pytest.approx(3.0)
    assert g2.retry.attempts[_key(2)] == 2 and g2.retry.attempts[_key(3)] == 3
    assert g2.canary(REF)
    out2 = H.run_units(store, UNITS, fn, "t" * 64, "p", guard=g2)
    assert calls == {"u1": 2, "u2": 3, "u3": 3}  # u3 not re-run: its persisted fallback is returned
    assert out2["u1"]["status"] == out2["u2"]["status"] == "pending_canary"
    clk.t = 4 * H1  # i1 is preempted before its closing canary

    g3 = _mk(state, "i1", clk)
    with pytest.raises(RG.CanaryMismatch):
        g3.begin(_key(1), "u1")
    g3.declare_lost(
        "i1",
        "preemption_or_capacity",
        "provider preempted i1",
        "abcdef1",
        True,
        terminated_at=4 * H1,
    )
    g3.relaunch("preemption_or_capacity", "i2")
    g3.open_instance()
    assert g3.canary(REF)
    out3 = H.run_units(store, UNITS, fn, "t" * 64, "p", guard=g3)
    assert calls == {
        "u1": 3,
        "u2": 3,
        "u3": 3,
    }  # u1 its 3rd attempt; u2 exhausted -> fallback; u3 fallback
    assert out3["u2"]["fallback"] and out3["u2"]["value"] == "base"
    assert g3.canary(REF)  # closing canary on i2
    with pytest.raises(RG.RetryExhausted):  # the one registered relaunch is used
        g3.relaunch("infrastructure", "i3")
    clk.t = 5 * H1

    g4 = _mk(state, "i2", clk)  # resume in a fourth process: nothing recomputed
    out4 = H.run_units(store, UNITS, fn, "t" * 64, "p", guard=g4)
    assert out4["u1"] == {
        "unit": "u1",
        "call": 3,
    }  # the first eligible attempt, not an earlier value
    assert out4["u2"]["value"] == "base" and out4["u3"]["value"] == "base"
    assert calls == {"u1": 3, "u2": 3, "u3": 3}
    assert [s for _, s in g4.unit_status(_key(1))["attempts"]] == [
        "retained_ineligible",  # before the failed end canary
        "retained_ineligible",  # on the lost instance
        "eligible",
    ]
    rep = g4.report()
    assert rep["lost_instances"] == ["i1"] and rep["relaunches"] == {"e2": 1}
    assert rep["instance_gpu_h"] == pytest.approx(4.0 + 1.0)  # i1 to its termination + i2 so far


def test_non_technical_failure_halts_but_is_charged(tmp_path):
    clk = Clock()
    g = _mk(tmp_path / "s", "i1", clk)
    g.open_instance()
    g.canary(REF)

    def fn(recipe):
        raise ValueError("a bug")

    with pytest.raises(ValueError):
        H.run_units(H.UnitStore(tmp_path / "u"), [({"u": 9}, "u9")], fn, "t" * 64, "p", guard=g)
    assert [r["status"] for r in g.ledger.rows if r["type"] == "unit_time"] == ["failed"]


# ------------------------------------------------------------------ Board A freeze re-check probes (§II.26)


def test_probe_freeze_clock_rollback_cannot_credit_another_instance(tmp_path):
    clk = Clock()
    clk.t = 100 * H1
    g = _mk(tmp_path / "s", "i1", clk)
    g.open_instance()
    clk.t = 10 * H1  # the wall clock moves backward 90 h
    with pytest.raises(RG.ClockInconsistent):  # fail closed: no negative duration is ever recorded
        g.close_instance()
    with pytest.raises(
        RG.ClockInconsistent
    ):  # halted: nothing proceeds until authoritative reconciliation
        g.relaunch("infrastructure", "i2")
    assert g.ledger.halted()
    with pytest.raises(RG.ClockInconsistent):  # clearing needs the authoritative termination of i1
        g.ledger.clear_clock_halt("ntp", "clock re-synchronised")
    g.reconcile_instance(
        launched_at=100 * H1, terminated_at=101 * H1, gpus=1, source="provider API"
    )
    with pytest.raises(RG.ClockInconsistent):  # and a clock at or above the recorded maximum
        g.ledger.clear_clock_halt("ntp", "clock re-synchronised")
    clk.t = 101 * H1
    g.ledger.clear_clock_halt("ntp", "clock re-synchronised")
    charges = g.ledger.instance_charges()
    assert all(v >= 0 for v in charges.values()) and charges["i1"] == pytest.approx(1.0)
    g.relaunch("infrastructure", "i2")
    g.open_instance()
    g.canary(REF)
    clk.t = 192 * H1  # i2 open 91 h: the cap stops the next unit (no credit from i1)
    with pytest.raises(RG.CapExceeded):
        H.run_units(H.UnitStore(tmp_path / "u"), UNITS[:1], _ok, "t" * 64, "p", guard=g)


def test_probe_freeze_contract_rollback_does_not_revive_old_results(tmp_path):
    clk, store = Clock(), H.UnitStore(tmp_path / "u")
    g = _mk(tmp_path / "s", "i1", clk)
    g.open_instance()
    g.canary(REF)
    H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=g)
    g.canary(REF)
    assert g.resolve(store, _key(1)) == {"unit": "u1"}
    b = RG.LaunchAttestation(True, True, REF, "d" * 64, ATTEST.evidence)
    tr = RG.ContractTransition("driver_or_cuda", "driver change", "abcdef1", True)
    gb = _mk(tmp_path / "s", "i1", clk, att=b, transition=tr)
    assert gb.generation == 1 and gb.resolve(store, _key(1)) is None
    ga = _mk(tmp_path / "s", "i1", clk, att=ATTEST, transition=tr)  # back to the original digest
    assert ga.generation == 2 and ga.contract == g.contract
    assert ga.resolve(store, _key(1)) is None  # the generation-0 result stays superseded
    assert ga.unit_status(_key(1))["attempts"][0][1] == "superseded_contract"
    called = []

    def fn(recipe):
        called.append(1)
        return {"unit": "u1", "generation": 2}

    with pytest.raises(RG.CanaryMismatch):  # generation-0 canaries do not count for generation 2
        H.run_units(store, UNITS[:1], fn, "t" * 64, "p", guard=ga)
    ga.canary(REF)
    H.run_units(store, UNITS[:1], fn, "t" * 64, "p", guard=ga)
    assert called == [1]


def test_probe_freeze_canary_after_loss_cannot_revive_an_attempt(tmp_path):
    clk, store = Clock(), H.UnitStore(tmp_path / "u")
    g = _mk(tmp_path / "s", "i1", clk)
    g.open_instance()
    g.canary(REF)
    H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=g)  # a1 pending; i1 is then lost
    clk.t = H1
    g.declare_lost("i1", "preemption_or_capacity", "preempted", "abcdef1", True)
    g.relaunch("preemption_or_capacity", "i2")
    g.open_instance()
    g.canary(REF)
    H.run_units(store, UNITS[:1], lambda r: {"unit": "u1", "on": "i2"}, "t" * 64, "p", guard=g)
    g.canary(REF)
    stale = _mk(tmp_path / "s", "i1", clk)  # a stale guard with the lost instance's id
    with pytest.raises(RG.InstanceNotOpen):  # no canary for a closed or lost instance
        stale.canary(REF)
    with pytest.raises(RG.InstanceNotOpen):  # run_units validates before resolving anything
        H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=stale)
    assert [s for _, s in stale.unit_status(_key(1))["attempts"]] == [
        "retained_ineligible",
        "eligible",
    ]
    assert any(e["type"] == "attempt_disposed" for e in stale.events)
    assert stale.resolve(store, _key(1)) == {"unit": "u1", "on": "i2"}


# ------------------------------------------------------------------ Board A freeze re-check 2 probes (§II.27)


def test_probe_freeze2_stale_guard_cannot_return_a_superseded_result(tmp_path):
    clk, store, st = Clock(), H.UnitStore(tmp_path / "u"), tmp_path / "s"
    g0 = _mk(st, "i1", clk)
    g0.open_instance()
    g0.canary(REF)
    H.run_units(store, UNITS[:1], lambda r: {"generation": 0}, "t" * 64, "p", guard=g0)
    g0.canary(REF)
    assert g0.resolve(store, _key(1)) == {"generation": 0}
    b = RG.LaunchAttestation(True, True, REF, "d" * 64, ATTEST.evidence)
    tr = RG.ContractTransition("driver_or_cuda", "driver change", "abcdef1", True)
    g1 = _mk(st, "i1", clk, att=b, transition=tr)
    g2 = _mk(st, "i1", clk, att=ATTEST, transition=tr)  # A -> B -> A: generation 2
    for _ in range(3):
        g2 = _mk(st, "i1", clk)  # repeated reconstruction keeps generation 2
    assert g2.generation == 2 and g2.resolve(store, _key(1)) is None
    g2.canary(REF)
    H.run_units(store, UNITS[:1], lambda r: {"generation": 2}, "t" * 64, "p", guard=g2)
    g2.canary(REF)
    assert g2.resolve(store, _key(1)) == {"generation": 2}

    def never(recipe):
        raise AssertionError("a resolved unit is never recomputed")

    n = len(g2.events)
    for stale in (g0, g1):  # the retained generation-0 and generation-1 objects
        with pytest.raises(RG.StaleGuard):
            H.run_units(store, UNITS[:1], never, "t" * 64, "p", guard=stale)
        with pytest.raises(RG.StaleGuard):
            stale.resolve(store, _key(1))
        with pytest.raises(RG.StaleGuard):
            stale.canary(REF)
    assert len(g2.events) == n  # the stale objects wrote nothing
    assert H.run_units(store, UNITS[:1], never, "t" * 64, "p", guard=g2) == {
        "u1": {"generation": 2}
    }


def test_probe_freeze2_stale_same_id_guard_cannot_accept_a_post_loss_canary(tmp_path):
    clk, store, st = Clock(), H.UnitStore(tmp_path / "u"), tmp_path / "s"
    live = _mk(st, "i1", clk)
    live.open_instance()
    live.canary(REF)
    H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=live)  # .a1 pending its closing canary
    other = _mk(st, "i1", clk)  # a second object for the same state directory and instance id
    clk.t = H1
    other.declare_lost("i1", "preemption_or_capacity", "preempted", "abcdef1", True)
    n = len(other.events)
    with pytest.raises(RG.StaleGuard):  # the retained object can no longer record evidence
        live.canary(REF)
    with pytest.raises(RG.StaleGuard):
        H.run_units(store, UNITS[:1], _ok, "t" * 64, "p", guard=live)
    assert len(other.events) == n
    assert other.unit_status(_key(1))["attempts"] == [(f"{_key(1)}.a1", "retained_ineligible")]
    with pytest.raises(RG.InstanceNotOpen):  # and the current owner refuses a canary for a lost id
        other.canary(REF)
    fresh = _mk(st, "i1", clk)
    with pytest.raises(RG.InstanceNotOpen):
        fresh.canary(REF)
    assert fresh.resolve(store, _key(1)) is None


def test_probe_freeze2_stale_guard_cannot_run_a_fourth_attempt_after_exhaustion(tmp_path):
    clk, store, st = Clock(), H.UnitStore(tmp_path / "u"), tmp_path / "s"
    g0 = _mk(st, "i1", clk)
    g0.open_instance()
    g0.canary(REF)
    stale = _mk(st, "i1", clk)
    g = _mk(st, "i1", clk)
    calls = []

    def fail(recipe):
        calls.append("failure")
        raise MemoryError("OOM")

    def succeed(recipe):
        calls.append("fourth execution")
        return {"unit": "u1"}

    out = H.run_units(store, UNITS[:1], fail, "t" * 64, "p", guard=g)
    assert out["u1"]["fallback"] and calls == ["failure"] * 3
    with pytest.raises(RG.StaleGuard):
        H.run_units(store, UNITS[:1], succeed, "t" * 64, "p", guard=stale)
    assert calls == ["failure"] * 3  # never executed a fourth time
    assert H.run_units(store, UNITS[:1], succeed, "t" * 64, "p", guard=g)["u1"]["value"] == "base"
    g2 = _mk(st, "i1", clk)
    assert H.run_units(store, UNITS[:1], succeed, "t" * 64, "p", guard=g2)["u1"]["value"] == "base"
    with pytest.raises(RG.UnitResolved):
        g2.begin(_key(1), "u1")
    assert calls == ["failure"] * 3
    assert RG.RetryPolicy(st / RG.LOG_NAME).attempts == {_key(1): 3}
    assert sum(1 for e in g2.events if e["type"] == "fallback") == 1


def test_probe_freeze2_overlapping_objects_never_duplicate_sequence_numbers(tmp_path):
    clk, st = Clock(), tmp_path / "s"
    a = _mk(st, "i1", clk)
    a.open_instance()
    b = _mk(st, "i1", clk)  # overlapping owner
    with pytest.raises(RG.StaleGuard):
        a.canary(REF)
    assert b.canary(REF)
    led_a = RG.CombinedLedger(st / RG.LOG_NAME, {"e2": 200.0}, clock=clk)  # unfenced views
    led_b = RG.CombinedLedger(st / RG.LOG_NAME, {"e2": 200.0}, clock=clk)
    for _ in range(5):
        led_a.add("e2", 1.0)
        led_b.add("e2", 1.0)
        b.canary(REF)
    b.retry.repair("infrastructure", "x", "abcdef1", outcome_blind=True)
    recs = [json.loads(x) for x in (st / RG.LOG_NAME).read_text().splitlines()]
    assert [r["seq"] for r in recs] == list(range(1, len(recs) + 1))
    assert all(r["prev"] == (recs[i - 1]["hash"] if i else RG.GENESIS) for i, r in enumerate(recs))
    assert all(r["hash"] == RG.record_hash(r) for r in recs)
    assert led_a.unit_gpu_h() == led_b.unit_gpu_h() == pytest.approx(10 / 3600)
    assert RG.read_study(st).snapshot() == RG.StudyLog(st / RG.LOG_NAME).state.snapshot()
    assert b.events == recs


def test_unfenced_views_always_act_on_the_current_log(tmp_path):
    p = tmp_path / RG.LOG_NAME
    led1, led2 = RG.CombinedLedger(p, {"e2": 200.0}), RG.CombinedLedger(p, {"e2": 200.0})
    led1.add("e2", 89.5 * 3600)
    with pytest.raises(RG.CapExceeded):  # led2 re-reads the tail under the lock: no stale snapshot
        led2.check_before("e2", 0.6)
    r1, r2 = RG.RetryPolicy(p), RG.RetryPolicy(p)
    assert [r1.attempt("u", None if i == 0 else "out_of_memory") for i in range(3)] == [1, 2, 3]
    with pytest.raises(RG.RetryExhausted):
        r2.attempt("u", "out_of_memory")


_APPEND = """
import sys
from wsp.wm import runguard as RG
led = RG.CombinedLedger(sys.argv[1], {"e2": 1e6})
for _ in range(int(sys.argv[2])):
    led.add("e2", 1.0)
"""


def test_lock_serialises_concurrent_processes(tmp_path):
    path = tmp_path / RG.LOG_NAME
    env = {**os.environ, "PYTHONPATH": str(Path(RG.__file__).parents[2])}
    procs = [
        subprocess.Popen([sys.executable, "-c", _APPEND, str(path), "40"], env=env)
        for _ in range(4)
    ]
    assert [p.wait(120) for p in procs] == [0, 0, 0, 0]
    st = RG.StudyLog(path).state  # verifies the whole chain
    assert st.n == 160 and st.unit_s["e2"] == pytest.approx(160.0)


def test_torn_or_truncated_log_fails_closed_until_reconciled(tmp_path):
    clk, st = Clock(), tmp_path / "s"
    g = _mk(st, "i1", clk)
    g.open_instance()
    g.canary(REF)
    log = st / RG.LOG_NAME
    torn = b'{"seq": 99, "type": "can'
    with open(log, "ab") as f:  # a crash in the middle of an append
        f.write(torn)
    with pytest.raises(RG.LogCorrupt):
        g.canary(REF)
    with pytest.raises(RG.LogCorrupt):
        _mk(st, "i1", clk)
    with pytest.raises(RG.LogCorrupt):
        RG.read_study(st)
    rec = RG.repair_torn_tail(st, "operator", "crash during an append; torn bytes quarantined")
    assert rec["type"] == "log_repaired" and (st / rec["quarantine"]).read_bytes() == torn
    with pytest.raises(RG.StaleGuard):  # the repair ends the ownership
        g.canary(REF)
    g2 = _mk(st, "i1", clk)
    assert g2.canary(REF)
    data = log.read_bytes()
    log.write_bytes(
        data[: data.rstrip(b"\n").rfind(b"\n") + 1]
    )  # a complete record lost from the end
    with pytest.raises(RG.LogCorrupt):
        g2.canary(REF)
    with pytest.raises(RG.LogCorrupt):  # detected against the head record on a fresh load
        RG.read_study(st)
    with pytest.raises(ValueError):  # not a torn tail: no automatic repair
        RG.repair_torn_tail(st, "operator", "x")
    lines = data.splitlines(keepends=True)
    i = next(k for k, x in enumerate(lines) if b'"instance_open"' in x)
    lines[i] = lines[i].replace(b'"gpus":1', b'"gpus":8')  # an altered middle record
    log.write_bytes(b"".join(lines))
    with pytest.raises(RG.LogCorrupt):
        RG.read_study(st)
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "events.jsonl").write_text("")
    with pytest.raises(ValueError):  # the pre-§II.27 multi-file layout is refused
        _mk(tmp_path / "old", "i1", clk)
