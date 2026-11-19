"""Validity/repair/stopping contract enforced in the production path (wsp.wm.runguard, harness.run_units;
Board A v2 change 6 and confirmation blocker B)."""

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
    cb = RG.CanaryBook(REF, tmp_path / "c.jsonl")
    cb.canary("i1", 1, REF)
    assert RG.CanaryBook(REF, tmp_path / "c.jsonl").last("i1") == REF


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
    assert g2.ledger.instance_gpu_h() == pytest.approx(
        1.5 + 0.0
    )  # i1 charged to its termination; i2 just opened
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
    assert len([r for r in g.ledger.rows if r.get("status") == "failed"]) == 4
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
    assert [r["status"] for r in g.ledger.rows if r.get("kind") == "unit"] == ["failed"]
