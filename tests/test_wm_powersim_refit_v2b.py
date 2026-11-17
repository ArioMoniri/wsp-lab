"""RULING-wme-v2b A7: the physically possible H-T alternative, the quadratic-world null and the calibration."""

import numpy as np
import pytest

from wsp.wm import powersim_refit as R1
from wsp.wm import powersim_refit_v2 as R2
from wsp.wm import powersim_refit_v2b as RB
from wsp.wm import powersim_v1 as PV
from wsp.wm import synthetic as S
from wsp.wm import wmcp as CP


@pytest.fixture(scope="module")
def bank():
    w = S.SynthWorld(kind="quadratic", seed=3, n_dirs=8, n_obj=6)
    g = PV.graph(8, 6)
    return w, g, RB.build_bank_v2b(w, g, 5)


def test_bank_is_the_v2_bank_with_roots_recorded(bank):
    w, g, b = bank
    a = R2.build_bank_v2(w, g, 5)
    assert all(np.array_equal(x["x"], y["x"]) for x, y in zip(a, b, strict=True))
    assert all(len(t["roots"]) == len(g["components"][t["comp"]]["families"]) for t in b)


def test_alternative_is_path_independent(bank):
    w, g, b = bank
    st = RB.strengths(len(g["components"]), 1)
    comp = [t for t in b if t["kind"] == "composition"]
    assert comp, "fixture needs a block component"
    t1, t2 = comp[0], comp[1]
    assert t1["comp"] == t2["comp"]
    np.testing.assert_allclose(t1["x"][-1, : w.J], t2["x"][-1, : w.J], atol=1e-12)
    y1 = RB.alt_outcome(w, t1["x"][-1, : w.J], t1["roots"], st[t1["comp"]], 1.3)
    y2 = RB.alt_outcome(w, t2["x"][-1, : w.J], t2["roots"], st[t2["comp"]], 1.3)
    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_second_order_near_zero_and_normalisation_type(bank):
    w, _g, b = bank
    u = b[0]["roots"][0]
    d = [np.abs(RB.departure(w, t * w.step * u, [u], 1.0)).max() / t**2 for t in (0.1, 0.05, 0.025)]
    assert d[1] < 0.6 * d[0] and d[2] < 0.6 * d[1]  # o(|E|^2): the field is exact to second order
    for z in (0.5, 2.0, 4.0):
        E = z * w.step * u
        np.testing.assert_allclose(
            RB.alt_outcome(w, E, [u], 1.0, 1.0), S.raw_field(w, E) * RB.rho(z), atol=1e-12
        )
    grow = [np.abs(RB.departure(w, z * w.step * u, [u], 1.0)).max() for z in (0.5, 1.0, 2.0)]
    assert grow[0] < grow[1] < grow[2]


def test_strengths_are_independent_per_component():
    a, b = RB.strengths(500, 1), RB.strengths(500, 2)
    assert a.shape == (500,) and np.all(a > 0) and not np.allclose(a, b)
    assert np.median(a) == pytest.approx(1.0, abs=0.1)
    np.testing.assert_array_equal(a, RB.strengths(500, 1))


def test_scenarios(bank):
    w, g, b = bank
    out = {sc: RB.label(b, w, sc, kappa=2.0, seed=7) for sc in RB.SCENARIOS}
    se = out["alt"][1]
    for sc in RB.SCENARIOS:
        np.testing.assert_array_equal(
            out[sc][1], se
        )  # one SE for all scenarios (quadratic h <= 2 outcomes)
    st = RB.strengths(len(g["components"]), 7)
    for t, ta, tq in zip(
        b, out["alt"][0], out["null_quadratic"][0], strict=True
    ):  # same noise draws
        for h in range(t["horizon"] + 1):
            E = t["x"][h, : w.J]
            np.testing.assert_allclose(
                ta["y"][h] - tq["y"][h] + S.raw_field(w, E),
                RB.alt_outcome(w, E, t["roots"], st[t["comp"]], 2.0),
                atol=1e-9,
            )
    # quadratic null: field + noise only; observations consistent with the outcome (loss block)
    tq = out["null_quadratic"][0][3]
    nq = tq["y"] - S.raw_field(w, tq["x"][:, : w.J])
    assert np.allclose(nq[0], 0) and np.abs(nq[1:]).max() < 6 * se.max()
    ta, t0 = out["alt"][0][3], b[3]
    n_sk, n_it = len(w.probes) * w.P.shape[1], len(w.obs_items)
    dep = RB.alt_outcome(w, t0["x"][2, : w.J], t0["roots"], st[t0["comp"]], 2.0) - S.raw_field(
        w, t0["x"][2, : w.J]
    )
    np.testing.assert_allclose(
        ta["obs"][2, n_sk : n_sk + w.K * n_it] - t0["obs"][2, n_sk : n_sk + w.K * n_it],
        np.repeat(dep, n_it),
        atol=1e-9,
    )
    # shuffled null: residuals beyond the field are permuted, sign-flipped alternative residuals
    rs = [t["y"] - S.raw_field(w, t["x"][:, : w.J]) for t in out["null_shuffle"][0]]
    ra = [t["y"] - S.raw_field(w, t["x"][:, : w.J]) for t in out["alt"][0]]
    assert not all(np.allclose(x, y) for x, y in zip(rs, ra, strict=True))


def test_replicate_runs_every_scenario():
    cell = {"r_f": 8, "n_blk": 6, "n_cal": 12}
    cfg = CP.CPConfig(epochs=3)
    for sc in RB.SCENARIOS:
        r = RB.replicate(0, cell, sc, kappa=1.0, cfg=cfg, n_flip=199)
        assert r["scenario"] == sc and r["decomposition_gap_max"] <= 1e-12 and r["m"] >= 2
        assert np.isfinite(r["ct"]) and isinstance(r["c2i"], bool)
    e = RB.static_errors(0, cell, kappa=1.0, cfg=cfg)
    assert e.ndim == 1 and e.size > 0 and np.all(e >= 0)


def test_summarise_gates_both_nulls():
    base = {
        "lb": 0.0,
        "ct": 0.0,
        "stat": 0.0,
        "c2i": True,
        "decomposition_gap_max": 0.0,
        "static_chosen": ["S2"],
        "seconds": 1.0,
        "reject": False,
    }
    rows = [dict(base, rep=i, scenario=sc) for i in range(10) for sc in RB.SCENARIOS]
    s = RB.summarise(rows)
    assert set(s) == set(RB.SCENARIOS)
    assert s["null_shuffle"]["pass"] and s["null_quadratic"]["pass"] and not s["alt"]["pass"]
    assert R1  # imported for the bank helpers
