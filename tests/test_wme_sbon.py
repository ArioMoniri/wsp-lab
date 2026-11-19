"""The registered two-rung S-BoN runner on CPU fixtures at the non-integer budget 86.7218 (Board A v2 change 7)."""

import hashlib
import math

import numpy as np
import pytest

from wsp.wm import e2 as E2
from wsp.wm import plan as PL
from wsp.wm import powersim_w as PW
from wsp.wm.arms import sbon as SB

B = 86.7218


class FakeWorld:
    """Deterministic fixture: each candidate's true gain from its primitive set; realised theta depends on the
    set only (so permutations are the same state: cache hits); screened groups are reused by ``complete``."""

    def __init__(self, tie=False):
        self.n_screen, self.n_complete, self.tie = 0, 0, tie
        self.screened = set()

    def _g(self, pids):
        h = int(hashlib.sha256("|".join(sorted(pids)).encode()).hexdigest()[:8], 16)
        return 0.05 if self.tie else (h % 1000) / 1000 * 0.2 - 0.05

    def theta_hash(self, pids):
        return "|".join(sorted(pids))

    def screen(self, pids):
        self.n_screen += 1
        self.screened.add(self.theta_hash(pids))
        return {"gain16": self._g(pids), "guard16_ok": self._g(pids) > -0.04, "seconds": 1.0}

    def complete(self, pids):
        assert self.theta_hash(pids) in self.screened  # promotion reuses the screened groups
        self.n_complete += 1
        g16 = self._g(pids)
        g48 = self._g(pids) + 0.01
        return {
            "gain": (16 * g16 + 48 * g48) / 64,
            "gain_se": 0.02,
            "guards_pass": True,
            "rel_norm": 1.0,
            "seconds": 3.0,
        }


def _episode(n=48, index=3):
    return {"index": index, "subsample": [f"p{i:02d}" for i in range(n)]}


def test_plan_counts_at_the_noninteger_budget():
    pc = SB.plan_counts(B)
    c16, _ = PW.sh_costs(SB.T_BUNDLE_S, SB.OVERHEAD_S)
    assert pc["screen"] == pytest.approx(c16)
    assert pc["promote"] == pytest.approx(1 - c16 + SB.OVERHEAD_S / SB.T_BUNDLE_S)
    k, p = pc["K"], pc["P"]
    assert p == math.ceil(k / 4)
    assert k * pc["screen"] + p * pc["promote"] <= B
    k1 = k + 1
    assert k1 * pc["screen"] + math.ceil(k1 / 4) * pc["promote"] > B  # K is the largest
    assert 0 <= pc["residual"] < pc["screen"] + pc["promote"]


def test_runner_spends_exactly_the_plan_and_forfeits_the_residual():
    w = FakeWorld()
    led = PL.BudgetLedger(B, ["sbon"])
    out = SB.run_sbon(w, _episode(), B, led)
    pc = out["plan"]
    assert out["screened"] == pc["K"] == w.n_screen
    assert out["promoted"] == pc["P"] == w.n_complete
    assert out["units_used"] == pytest.approx(pc["planned_units"])
    assert out["forfeited"] == pytest.approx(pc["residual"])
    ents = led.entries("sbon")
    tot = sum(e["units"] for e in ents)
    assert tot == pytest.approx(pc["planned_units"]) and tot <= B
    assert led.report()  # closed and consistent
    assert out["winner"]["fallback"] is False


def test_runner_is_deterministic_and_caches_repeats_at_zero_cost():
    a = SB.run_sbon(FakeWorld(), _episode(n=5), B, PL.BudgetLedger(B, ["sbon"]))
    b = SB.run_sbon(FakeWorld(), _episode(n=5), B, PL.BudgetLedger(B, ["sbon"]))
    assert a["winner"]["candidate"] == b["winner"]["candidate"]
    # with 5 primitives there are only 31 distinct sets: permutations and repeats are cache hits (0 units) and
    # the stream runs dry before K screens; the rest of the budget is forfeited, never re-spent
    assert a["cache_hits"] > 0 and a["screened"] <= 31 < a["plan"]["K"]
    assert a["forfeited"] > a["plan"]["residual"]


def test_ties_go_to_the_earlier_draw():
    w = FakeWorld(tie=True)
    out = SB.run_sbon(w, _episode(), B, PL.BudgetLedger(B, ["sbon"]))
    rng = np.random.default_rng(SB.stream_seed(3))
    first = SB.draw_candidate(rng, _episode()["subsample"])
    assert out["winner"]["candidate"] == first and out["winner"]["order"] == 0
    assert E2.candidate_key(out["winner"]["candidate"]) == E2.candidate_key(first)
