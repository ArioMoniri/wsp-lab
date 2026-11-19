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


def test_orchestration_failure_canary_restart_exhaustion_and_first_valid_selection(tmp_path):
    """One run through every registered rule: a technical failure retried and charged; a failed end canary making
    earlier results retained-ineligible; a process restart and an instance relaunch reloading every counter; a unit
    exhausting its 3 attempts and taking its fallback; and the first canary-valid attempt selected on resume."""
    state, store = tmp_path / "state", H.UnitStore(tmp_path / "units")
    units = [({"u": 1}, "u1"), ({"u": 2}, "u2"), ({"u": 3}, "u3")]
    calls = {"u1": 0, "u2": 0, "u3": 0}

    def fn(recipe):
        lab = f"u{recipe['u']}"
        calls[lab] += 1
        if lab == "u3" or (lab == "u2" and calls["u2"] == 1):
            raise MemoryError("OOM")  # technical
        return {"unit": lab, "call": calls[lab]}

    clk = Clock()
    mk = lambda inst: RG.ProductionGuard(
        state,
        "e2",
        {"e2": 10.0},
        ATTEST,
        inst,
        0.01,
        technical=_technical,
        fallback_value="base",
        clock=clk,
    )
    g = mk("i1")
    g.open_instance()
    with pytest.raises(RG.CanaryMismatch):  # no start canary yet
        H.run_units(store, units, fn, "t" * 64, "p", guard=g)
    assert g.canary(REF)
    out = H.run_units(store, units, fn, "t" * 64, "p", guard=g)
    assert out["u1"]["status"] == out["u2"]["status"] == "pending_canary"
    assert out["u3"] == {"fallback": True, "value": "base"} and calls["u3"] == 3
    failed = [r for r in g.ledger.rows if r.get("status") == "failed"]
    assert len(failed) == 4  # u2 once, u3 three times: every failed attempt is charged
    assert not g.canary(BAD)  # the end canary fails
    assert (
        g.unit_status(H.unit_key({"u": 1}, "t" * 64, "p"))["attempts"][0][1]
        == "retained_ineligible"
    )
    clk.t += 3600
    g.close_instance()

    # process restart: a new guard object reloads attempts, canaries, fallbacks and the ledger from disk
    g2 = mk("i1")
    k1, k2, k3 = (H.unit_key({"u": i}, "t" * 64, "p") for i in (1, 2, 3))
    assert g2.retry.attempts[k2] == 2 and g2.retry.attempts[k3] == 3
    assert g2.resolve(store, k1) is None  # retained-ineligible results are not complete
    assert g2.resolve(store, k3) == {"fallback": True, "value": "base"}
    g2.relaunch("infrastructure", "i2")
    g2.open_instance()
    assert g2.canary(REF)
    out2 = H.run_units(store, units, fn, "t" * 64, "p", guard=g2)
    assert calls == {
        "u1": 2,
        "u2": 3,
        "u3": 3,
    }  # u1 attempt 2, u2 attempt 3, u3 not re-run (fallback)
    assert out2["u3"]["fallback"] and out2["u1"]["status"] == "pending_canary"
    with pytest.raises(RG.CanaryMismatch):  # a pending attempt must be closed by a canary first
        g2.begin(k1, "u1")
    assert g2.canary(REF)
    with pytest.raises(RG.RetryExhausted):  # the one registered relaunch is used
        g2.relaunch("infrastructure", "i3")

    # resume in a third process: only the first eligible attempts are returned; nothing is recomputed
    g3 = mk("i2")
    out3 = H.run_units(store, units, fn, "t" * 64, "p", guard=g3)
    assert out3["u1"] == {"unit": "u1", "call": 2}  # the i2 attempt, not the earlier i1 value
    assert out3["u2"] == {"unit": "u2", "call": 3}
    assert out3["u3"] == {"fallback": True, "value": "base"}
    assert calls == {"u1": 2, "u2": 3, "u3": 3}
    st1 = g3.unit_status(k1)["attempts"]
    assert [s for _, s in st1] == ["retained_ineligible", "eligible"]  # retained for transparency
    assert store.done(f"{k1}.a1") and store.done(f"{k1}.a2")
    rep = g3.report()
    assert rep["relaunches"] == {"e2": 1} and rep["charged_gpu_h"] >= rep["unit_gpu_h"]


def test_non_technical_failure_halts_but_is_charged(tmp_path):
    clk = Clock()
    g = RG.ProductionGuard(
        tmp_path / "s", "e2", {"e2": 10.0}, ATTEST, "i1", 0.01, technical=_technical, clock=clk
    )
    g.canary(REF)

    def fn(recipe):
        raise ValueError("a bug")

    with pytest.raises(ValueError):
        H.run_units(H.UnitStore(tmp_path / "u"), [({"u": 9}, "u9")], fn, "t" * 64, "p", guard=g)
    assert [r["status"] for r in g.ledger.rows if r.get("kind") == "unit"] == ["failed"]
