"""PREREG_WME v2 arithmetic (wsp.wm.prereg_v2): secondary pricing, guard N, parity budget, cap check."""

import math

import pytest

from wsp.wm import p3gen as G
from wsp.wm import prereg_v2 as P


def test_guard_n_matches_p3gen_at_one_sided_005():
    # at z = 1.645 the generalised rule reproduces pilot 3's guard_n exactly
    for disc, delta, pool in (
        (0.1015625, 0.0390625, 5000),
        (0.0078125, 0.0234375, 5000),
        (0.3, 0.02, 400),
    ):
        assert P.guard_n(disc, delta, pool, z_alpha=1.645) == G.guard_n(disc, delta, pool)


def test_guard_n_at_one_sided_0025():
    n = P.guard_n(0.1015625, 0.0390625, 10_000, z_alpha=P.Z_0975)
    assert n == math.ceil((P.Z_0975 + P.Z_080) ** 2 * 0.1015625 / 0.0390625**2)
    # the floor binds for a rare-discordance guard
    assert P.guard_n(0.0078125, 0.0234375, 10_000, z_alpha=P.Z_0975) == 200
    # the pool caps it
    assert P.guard_n(0.1015625, 0.0390625, 300, z_alpha=P.Z_0975) == 300


def test_parity_budget_reproduces_stage_w():
    b = P.parity_budget(
        48,
        bank_gpu_h=12.076729130256979,
        field_s=2782.4204444885254,
        n_episodes=24,
        t_bundle_s=14.579798132181168,
    )
    assert b == pytest.approx(180.19960504605555, rel=1e-9)


def test_bon_arm_gpu_h_reproduces_stage_w():
    h = P.bon_arm_gpu_h(180.19960504605555, n_episodes=24, t_bundle_s=14.579798132181168)
    assert h == pytest.approx(17.51515910046843, rel=1e-9)


def test_te_price_components_are_positive_and_sum():
    rates = {
        "s_per_item_2048": 0.1,
        "s_per_item_4096": 0.3,
        "lora_s_per_step": 2.5,
        "setup_s": 500.0,
        "boot_s": 600.0,
    }
    p = P.te_price(
        {"mmlupro_gen": 10, "supergpqa_gen": 20},
        {"guard_harmless": 5, "guard_harmful": 3},
        n_cal=4,
        cal_samples=8,
        lora_steps=100,
        canary_items=2,
        rates=rates,
    )
    parts = p["seconds"]
    assert all(v > 0 for v in parts.values())
    assert p["gpu_h"] == pytest.approx(sum(parts.values()) / 3600)
    assert parts["eval_2048"] == pytest.approx(2 * 30 * 0.1)
    assert parts["eval_4096"] == pytest.approx(2 * 30 * 0.3)
    # guards: base + edit on the test splits, base-only calibration (greedy + samples), E1 winner + base descriptive
    assert parts["guards"] == pytest.approx((2 * 8 + 2 * 4 * (1 + 8) + 2 * 8) * 0.1)


def test_lora_steps():
    assert P.lora_steps(828, rows_per_step=8, epochs=2) == 208


def test_usd():
    assert P.usd(10.0) == pytest.approx(10.0 * 4.89 * 1.10)


def test_cap_check():
    ok = P.cap_check({"a": 50.0, "b": 30.0})
    assert ok["total_gpu_h"] == pytest.approx(80.0) and ok["within"]
    bad = P.cap_check({"a": 50.0, "b": 41.0})
    assert not bad["within"]
    with pytest.raises(ValueError):
        P.cap_check({"a": -1.0})
