"""H-E2 sensitivity machinery and the RB-E2 robustness test (Board A v2 required change 4)."""

import json
from pathlib import Path

import numpy as np

from wsp.wm import powersim_e2sens as ES
from wsp.wm import signflip as SF

ROOT = Path(__file__).resolve().parents[1]


def test_rb_e2_rejects_a_clear_advantage_and_not_zero():
    fam = np.arange(24) % 3
    rng = np.random.default_rng(0)
    assert SF.rb_e2_test(1.0 + 0.1 * rng.standard_normal(24), fam, n_boot=499)["reject"]
    assert not SF.rb_e2_test(np.zeros(24), fam, n_boot=499)["reject"]
    assert not SF.rb_e2_test(rng.standard_normal(24) - 0.5, fam, n_boot=499)["reject"]


def test_scenarios_have_the_registered_population_means():
    inp = json.loads((ROOT / "docs/reviews/wme/wme-powersim-v2-2026-11-17.json").read_text())[
        "inputs"
    ]
    for name in ("asym_fallback_heavy", "sym_fallback_heavy", "item_ix_025"):
        r = ES.run(name, 30, inp, n_flip=999, n_boot=199)
        assert set(r) >= {"sf", "rb", "iu", "mean_d"} and r["reps"] == 30
    assert set(ES.MEAN_NULL_GATE) <= set(ES.SCENARIOS)
    assert all(ES.SCENARIOS[s].get("null") for s in ES.MEAN_NULL_GATE)


def test_decision_rule():
    ok = {s: {"sf": {"rate": 0.02}, "rb": {"rate": 0.02}} for s in ES.MEAN_NULL_GATE}
    assert "stands as registered" in ES.decide(ok)["verdict"]
    one = dict(ok, asym_extreme={"sf": {"rate": 0.05}, "rb": {"rate": 0.02}})
    assert "IU" in ES.decide(one)["verdict"]
    both = dict(ok, asym_extreme={"sf": {"rate": 0.05}, "rb": {"rate": 0.04}})
    assert "withdrawn" in ES.decide(both)["verdict"]
