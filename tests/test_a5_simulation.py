"""Synthetic calibration only: no development or project data access."""

import numpy as np
import pytest

from scripts.a5_simulation import (
    cp_interval,
    donor_cells,
    ordered_decision,
    resolution,
    selection_cells,
    simulate,
)


def toy(n=40):
    rows = [
        {
            "id": f"e{i:03}",
            "component": f"c{i:03}",
            "family": "LoRA",
            "norm": i,
            "residuals": np.tile([2.0, 1.0, 0.0], (64, 1)).tolist(),
        }
        for i in range(n)
    ]
    return {"inventory": rows, "donors": rows}, {r["id"]: [1, 1, 1] for r in rows}


def test_cp_thresholds_and_boundaries():
    assert cp_interval(12, 15)[0] == pytest.approx(0.5191, abs=5e-5)
    assert cp_interval(27, 40)[0] == pytest.approx(0.5087, abs=5e-5)
    assert cp_interval(11, 15)[0] < 0.5
    assert cp_interval(26, 40)[0] < 0.5
    assert cp_interval(0, 0) == (0, 1)
    assert cp_interval(0, 15)[0] == 0
    assert cp_interval(15, 15)[1] == 1
    assert ordered_decision([40, 0, 40], [40, 40, 40]) == "alternative"
    assert ordered_decision([27, 30, 30], [40, 40, 40]) == "go"


def test_fair_null_size_and_determinism():
    payload, _ = toy()
    kwargs = {"replicates": 2000, "quotas": {"LoRA": (13, 13, 14)}}
    first = simulate(payload, **kwargs)
    assert first == simulate(payload, **kwargs)
    null = first["reports"]["null_size"]
    assert 0.03 <= null["two_sided_rejection_rates"][0] <= 0.07
    assert null["accepted"]


def test_strong_alternative_power():
    payload, fields = toy(120)
    result = simulate(payload, fields, replicates=1000, quotas={"LoRA": (40, 40, 40)})
    assert result["reports"]["power_yield"]["ordered_go_rate"] > 0.9


def test_donor_cell_mapping_and_shortage():
    payload, _ = toy()
    _, edges = selection_cells(payload["inventory"], {"LoRA": (13, 13, 14)})
    extra = dict(payload["donors"][0], id="a", component="c000")
    cells = donor_cells([*payload["donors"], extra], edges)
    assert cells["LoRA", 0][0]["id"] == "a"
    with pytest.raises(ValueError, match="empty donor"):
        donor_cells([extra], edges)


def test_resolution_matches_direct_paired_bootstrap():
    blocks = np.random.default_rng(8).normal(size=(32, 2, 3))
    actual = resolution(blocks, np.random.default_rng(9))
    indices = np.random.default_rng(9).integers(0, 32, size=(4000, 32))
    expected = np.quantile(blocks[indices].mean(axis=1), [0.025, 0.975], axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
