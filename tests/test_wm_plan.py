"""wm.plan: equal-budget ledger, identical B per arm, sealed test items unreachable (D6, D8)."""

import json
import os

import pytest

from wsp.wm import plan as P


def test_every_arm_gets_identical_budget_and_overdraw_raises():
    led = P.BudgetLedger(budget=4, arms=["wm", "statnn", "random"])
    assert {a: led.remaining(a) for a in led.arms} == {"wm": 4, "statnn": 4, "random": 4}
    for _ in range(4):
        led.charge("wm", candidate="c", seconds=1.5)
    with pytest.raises(P.BudgetExceeded):
        led.charge("wm", candidate="c2", seconds=1.0)
    with pytest.raises(KeyError):
        led.charge("unknown", candidate="c", seconds=1.0)
    rep = led.report()
    # Board A condition 10: the budget is identical by construction, usage is reported separately
    assert rep["budget"] == 4 and rep["budget_identical"] is True and rep["equal_usage"] is False
    assert rep["used"] == {"wm": 4, "statnn": 0, "random": 0}
    assert rep["seconds"]["wm"] == 6.0


def test_rejected_and_intermediate_evaluations_are_charged():
    led = P.BudgetLedger(budget=3, arms=["a"])
    led.charge("a", candidate="x", seconds=0.1, kind="rejected")
    led.charge("a", candidate="y", seconds=0.1, kind="intermediate")
    assert led.remaining("a") == 1
    assert [e["kind"] for e in led.entries("a")] == ["rejected", "intermediate"]


def test_ledger_rejects_unequal_budget_construction():
    with pytest.raises(ValueError):
        P.BudgetLedger(budget=0, arms=["a"])
    with pytest.raises(ValueError):
        P.BudgetLedger(budget=2, arms=["a", "a"])


def _write_items(root):
    open_dir = root / "open"
    sealed = root / "sealed_test"
    open_dir.mkdir(parents=True)
    sealed.mkdir()
    (open_dir / "dev.jsonl").write_text(json.dumps({"id": "d1"}) + "\n")
    (sealed / "test.jsonl").write_text(json.dumps({"id": "t1"}) + "\n")
    return open_dir, sealed


def test_planner_store_reads_dev_but_never_test(tmp_path):
    _open_dir, sealed = _write_items(tmp_path)
    store = P.PlannerItemStore(tmp_path)
    assert store.read("dev")[0]["id"] == "d1"
    for split in ("test", "../sealed_test/test", "sealed_test/test"):
        with pytest.raises(P.SealedError):
            store.read(split)
    with pytest.raises(P.SealedError):
        store.read_path(sealed / "test.jsonl")


def test_boundary_resolves_symlinks_before_refusing(tmp_path):
    open_dir, sealed = _write_items(tmp_path)
    link = open_dir / "innocent.jsonl"
    os.symlink(sealed / "test.jsonl", link)
    store = P.PlannerItemStore(tmp_path)
    with pytest.raises(P.SealedError):
        store.read("innocent")
    dlink = open_dir / "dirlink"
    os.symlink(sealed, dlink)
    with pytest.raises(P.SealedError):
        store.read_path(dlink / "test.jsonl")


def test_planner_store_records_every_read(tmp_path):
    _write_items(tmp_path)
    store = P.PlannerItemStore(tmp_path)
    store.read("dev")
    assert len(store.reads) == 1 and store.reads[0]["sha256"]


def test_sealed_reader_is_separate_and_requires_a_frozen_winner(tmp_path):
    _write_items(tmp_path)
    with pytest.raises(P.SealedError):
        P.read_sealed_test(tmp_path, frozen_winners=None)
    rows = P.read_sealed_test(tmp_path, frozen_winners={"wm": "cand-1"})
    assert rows[0]["id"] == "t1"


def test_random_arm_draws_reproducible_candidates():
    space = P.CandidateSpace(
        layers=[4, 8], directions=["d0", "d1", "d2"], doses=[0.5, 1.0], max_actions=2
    )
    a = space.sample(5, seed=1)
    b = space.sample(5, seed=1)
    assert a == b and len(a) == 5
    assert all(1 <= len(c) <= 2 for c in a)
    assert space.sample(5, seed=2) != a


# ------------------------------------------------------------------ bundle definition (Board A condition 10)


def test_bundle_kinds_have_registered_unit_rules():
    led = P.BudgetLedger(budget=4, arms=["a"])
    for kind in ("evaluated", "intermediate", "checkpoint", "validation", "rejected"):
        assert P.BUNDLE_KINDS[kind] == 1.0
    with pytest.raises(ValueError):
        led.charge("a", "x", 1.0, kind="screen")  # a screen needs its fractional units
    with pytest.raises(ValueError):
        led.charge("a", "x", 1.0, kind="oracle")  # oracle derivatives are priced explicitly
    with pytest.raises(ValueError):
        led.charge("a", "x", 1.0, kind="generation")  # never part of B
    with pytest.raises(ValueError):
        led.charge("a", "x", 1.0, kind="evaluated", units=0.5)  # full kinds are one bundle
    led.charge("a", "x", 1.0, kind="screen", units=0.25)
    led.charge("a", "x", 1.0, kind="oracle", units=0.5)
    assert led.remaining("a") == pytest.approx(3.25)
    assert "full search-development panel" in P.BUNDLE_DEFINITION


def test_repeat_of_an_evaluated_theta_is_served_from_cache_at_zero_units():
    led = P.BudgetLedger(budget=2, arms=["a", "b"])
    led.charge("a", "x", 1.0, theta_hash="t" * 64)
    led.charge("a", "x", 0.0, theta_hash="t" * 64)
    assert led.remaining("a") == 1 and led.entries("a")[1]["kind"] == "cache"
    led.charge("b", "x", 1.0, theta_hash="t" * 64)  # another arm pays for its own evaluation
    assert led.remaining("b") == 1


def test_parity_requires_every_arm_closed_at_equal_usage():
    led = P.BudgetLedger(budget=3, arms=["wm", "statnn"])
    for _ in range(3):
        led.charge("wm", "c", 1.0)
    led.charge("statnn", "c", 1.0)
    with pytest.raises(P.ParityError):
        led.verify_parity()
    led.close("statnn", reason="base fallback: no candidate passes")
    led.close("wm", reason="budget spent")
    rep = led.verify_parity()
    assert rep["equal_usage"] is False and rep["parity"] is True
    assert rep["forfeited"] == {"wm": 0.0, "statnn": 2.0}
    with pytest.raises(P.BudgetExceeded):
        led.charge("statnn", "c", 1.0)  # closed


def test_report_by_kind_and_seconds():
    led = P.BudgetLedger(budget=4, arms=["a"])
    led.charge("a", "x", 2.0)
    led.charge("a", "y", 1.0, kind="screen", units=0.5)
    rep = led.report()
    assert rep["by_kind"]["a"] == {"evaluated": 1.0, "screen": 0.5}
    assert rep["units"]["a"] == 1.5 and rep["seconds"]["a"] == 3.0
