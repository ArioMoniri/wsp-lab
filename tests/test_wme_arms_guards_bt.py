"""Guard bounds and the dev gain LCB through the registered item-group bootstrap-t (RULING-wme-v2 A5)."""

import numpy as np
import pytest

from wsp.wm import e2 as E2
from wsp.wm.arms import spec as S


def _items(rng, n=64, shift=0.0):
    groups = [f"g{i // 2}" for i in range(n)]
    gi = {}
    for g in E2.GUARD_SETS["sycophancy_real"]:
        gi[g] = {
            sc: (rng.standard_t(3, n) * 0.004 + shift).tolist() for sc, _s, _l in E2.MARGINS[g]
        }
    return gi, groups


def test_guard_ok_uses_bootstrap_t_on_both_scales():
    rng = np.random.default_rng(0)
    gi, groups = _items(rng)
    rec = {"objective": "sycophancy_real", "guard_items": gi, "item_groups": groups}
    ok = S.guard_ok(rec)
    bounds = S.guard_bounds_from_items("sycophancy_real", gi, groups)
    assert ok == E2.guards_pass("sycophancy_real", bounds)["pass"]
    assert (
        rec["guard_bounds"] == bounds
        and rec["guard_method"] == "item-group bootstrap-t (R = 9,999)"
    )
    gi2, _ = _items(rng, shift=-0.2)
    assert not S.guard_ok(
        {"objective": "sycophancy_real", "guard_items": gi2, "item_groups": groups}
    )


def test_gain_lcb_bootstrap_t_when_items_present():
    rng = np.random.default_rng(1)
    d = rng.normal(0.05, 0.2, 128)
    groups = list(range(128))
    lcb = S.gain_lcb(
        {"gain": float(d.mean()), "se": 0.02, "gain_items": d.tolist(), "item_groups": groups}
    )
    assert lcb == pytest.approx(
        E2.bootstrap_t_bounds(d, groups, n_boot=S.N_BOOT_T, seed=S.BOOT_SEED)["lcb"]
    )
    assert S.gain_lcb({"gain": 0.1, "se": 0.01}) == pytest.approx(
        0.1 - S.Z_GUARD * 0.01
    )  # synthetic only
