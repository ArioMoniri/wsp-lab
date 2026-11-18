"""SYNTHESIS-r11 R6 (flag 1) symmetric shrinkage: tie rule, the shrunk S1, and the opt-in full-refit path."""

import numpy as np

from wsp.wm import powersim_refit_v2 as R2
from wsp.wm import powersim_refit_v2b as RB
from wsp.wm import wmcp as CP


def test_tie_rule_zero_is_the_original_rule():
    errs = {0.0: 1.0, 0.25: 0.9, 0.5: 0.9, 0.75: 1.1, 1.0: 1.2}
    assert CP._tie_to_field(errs, 0.0) == min(errs, key=lambda lam: (errs[lam], lam)) == 0.25


def test_tie_rule_goes_to_the_field_within_tolerance():
    errs = {0.0: 1.004, 0.25: 1.002, 0.5: 1.0, 0.75: 1.3, 1.0: 1.5}
    assert CP._tie_to_field(errs, 0.005) == 0.0
    assert CP._tie_to_field(errs, 0.001) == 0.5


def _spec(K=3, J=4):
    names = (
        tuple(f"a{k}" for k in range(K))
        + tuple(f"q{k}" for k in range(K))
        + tuple(f"e{j}" for j in range(J))
        + ("norm",)
    )
    return CP.RSpec(
        names=names,
        degree=(1,) * K + (2,) * K + (1,) * J + (1,),
        n_obj=K,
        norm_index=len(names) - 1,
    )


def _states(spec, f_scale, n_comp=12, per=4, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_comp):
        for _ in range(per):
            r = rng.normal(size=spec.dim)
            y = f_scale * CP.field(r, spec) + noise * rng.normal(size=spec.n_obj)
            out.append({"r": r, "y": y, "obs": np.zeros(2), "h": 1, "comp": f"c{c}"})
    return out


def test_shrunk_f3u_reduces_to_the_field_when_the_field_is_exact():
    spec = _spec()
    st = _states(spec, 1.0)
    m = CP.ShrunkF3uR(spec, RB.R6_TIE_REL).fit(st, [s["comp"] for s in st])
    assert m.shrink == 0.0
    r = np.random.default_rng(5).normal(size=spec.dim)
    assert np.allclose(m.predict(r)[0], CP.field(r, spec))


def test_shrunk_f3u_keeps_a_real_departure():
    spec = _spec()
    st = _states(spec, 2.0)
    m = CP.ShrunkF3uR(spec, RB.R6_TIE_REL).fit(st, [s["comp"] for s in st])
    assert m.shrink == 1.0
    r = np.random.default_rng(6).normal(size=spec.dim)
    assert np.allclose(m.predict(r)[0], 2.0 * CP.field(r, spec))


def test_members_r6_have_the_tie_rule_and_the_same_names():
    spec = _spec()
    cfg = CP.CPConfig(epochs=2)
    mem = RB.members_r6(cfg, spec, np.zeros(2), None)
    assert set(mem) == set(R2._members(cfg, spec, np.zeros(2), None)) == {"S1", "S2", "S3"}
    assert isinstance(mem["S1"](), CP.ShrunkF3uR)
    assert mem["S2"]().shrink_tie_rel == RB.R6_TIE_REL
    assert mem["S3"]().shrink_tie_rel == RB.R6_TIE_REL


def test_replicate_r6_runs_and_records_shrinkage():
    cell = {"r_f": 8, "n_blk": 6, "n_cal": 12}
    r = RB.replicate(
        0, cell, "null_quadratic", kappa=0.8, cfg=CP.CPConfig(epochs=3), n_flip=199, r6=True
    )
    assert r["r6"] is True and len(r["shrink_wm"]) == len(r["shrink_static"]) == R2.K_OUT
    assert all(x in CP.SHRINK_GRID for x in r["shrink_wm"])
    assert np.isfinite(r["ct"]) and r["decomposition_gap_max"] <= 1e-12
