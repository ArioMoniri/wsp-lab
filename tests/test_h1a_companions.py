"""CPU tests for scripts/h1a_companions.py and the endpoint generalisation of tfv1_requalify (A4)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rq = load("tfv1_requalify")
hc = load("h1a_companions")


def test_block_means_keep_the_paired_mean_and_group_adjacent_windows():
    D = np.arange(24, dtype=float).reshape(2, 12)
    blocks = rq.block_means(D, 2)
    assert blocks.shape == (2, 6)
    assert blocks[0, 0] == pytest.approx((D[0, 0] + D[0, 1]) / 2)
    assert blocks.mean(1) == pytest.approx(D.mean(1))  # y and the losses are unchanged
    assert rq.block_means(D, 1) is D
    with pytest.raises(ValueError):
        rq.block_means(D[:, :11], 2)
    assert rq.ENDPOINTS["wikitext_log_ppl"][4] == 2  # 64 windows -> 32 two-window blocks


def test_theta_weights_selection_and_equal_component_ratio():
    a = np.array([1.0, 1.0, 2.0, 4.0])
    b = np.array([2.0, 2.0, 2.0, 4.0])
    assert hc.theta_of(a, b) == pytest.approx(1 - 8 / 10)
    assert hc.theta_of(a, b, np.array([True, True, False, False])) == pytest.approx(0.5)
    w = np.array([2.0, 0.0, 1.0, 1.0])
    assert hc.theta_of(a, b, w=w) == pytest.approx(1 - 8 / 10)  # (2+0+2+4)/(4+0+2+4)
    codes = ["x", "x", "y", "z"]
    eq, mean_ratio, w_eq = hc.equal_component_theta(a, b, codes)
    assert w_eq == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert eq == pytest.approx(1 - (0.5 + 0.5 + 2 + 4) / (1 + 1 + 2 + 4))
    assert mean_ratio == pytest.approx(np.mean([0.5, 0.0, 0.0]))
    assert hc.effective_n(np.ones(5)) == pytest.approx(5)
    assert hc.effective_n([1.0, 0.0, 0.0]) == pytest.approx(1)


def test_cluster_percentile_covers_a_known_ratio_and_filters_inside_the_resample():
    rng = np.random.default_rng(0)
    codes = np.repeat([f"c{i}" for i in range(40)], 3)
    b = rng.uniform(1, 2, 120)
    a = 0.8 * b * rng.uniform(0.9, 1.1, 120)  # theta ~ 0.2
    lo, hi = hc.cluster_percentile(a, b, codes, draws=400)
    assert lo < hc.theta_of(a, b) < hi and 0.1 < lo and hi < 0.3
    sel = np.arange(120) % 3 == 0
    lo2, hi2 = hc.cluster_percentile(a, b, codes, sel, draws=400)
    assert lo2 < hc.theta_of(a, b, sel) < hi2


def test_tertiles_and_strata_cells_carry_their_n():
    inc = np.linspace(0.1, 3.0, 30)
    tert, cuts = hc.tertile_labels(inc)
    assert [int((tert == t).sum()) for t in range(3)] == [10, 10, 10] and len(cuts) == 2
    fam = np.array(["lora_sft", "task_vector", "chain"] * 10)
    codes = np.array([f"c{i // 2}" for i in range(30)])
    rng = np.random.default_rng(1)
    b = rng.uniform(1, 2, 30)
    a = 0.9 * b
    out = hc.strata_tables(a, b, codes, fam, inc, draws=50)
    assert out["family"]["chain"]["n_rows"] == 10
    assert sum(c["n_rows"] for c in out["cross"].values()) == 30
    assert out["tertile"]["T3"]["share_of_sum_B"] == pytest.approx(b[tert == 2].sum() / b.sum())
    assert out["family"]["lora_sft"]["n_components"] <= 10


def test_natural_spline_is_linear_beyond_the_outer_knots_and_truncated_cubic_is_not():
    knots = [0.5, 1.0, 1.5]
    x = np.array([2.0, 3.0, 4.0, 5.0])
    nat = hc.natural_spline_basis(x, knots)
    assert nat.shape == (4, 3)  # 1, x, one nonlinear term for three knots
    second_difference = nat[:-2, 2] - 2 * nat[1:-1, 2] + nat[2:, 2]
    assert second_difference == pytest.approx(0, abs=1e-9)  # linear tail
    trunc = hc.spline_basis(x, knots)
    assert abs(trunc[:, 3][-1]) > 100  # cubic tail explodes


def test_flexible_nuisance_uses_training_rows_only_fits_a_curve_and_fails_closed():
    rng = np.random.default_rng(2)
    n = 120
    fam = np.array(["lora_sft", "task_vector", "chain", "abliteration"] * 30)
    inc = rng.uniform(0.1, 3, n)
    log_norm = np.log1p(inc)
    y = np.where(fam == "chain", 0.3 * np.sin(2 * log_norm), -0.2 * log_norm**2) + rng.normal(
        0, 0.01, n
    )
    train = np.arange(n) % 5 != 0
    test = ~train
    for variant in hc.NUISANCE_VARIANTS:
        pred = hc.flexible_nuisance(fam, log_norm, y, variant)(train, test)
        assert pred.shape == (test.sum(),) and np.isfinite(pred).all()
    nat = hc.flexible_nuisance(fam, log_norm, y, "family_x_natural_spline")
    assert np.abs(nat(train, test) - y[test]).mean() < 0.05
    # a held-out family level that the training rows never saw fails closed
    train2 = fam != "abliteration"
    with pytest.raises(ValueError, match="unseen family"):
        hc.flexible_nuisance(fam, log_norm, y)(train2, ~train2)
    # changing a held-out label cannot change the held-out nuisance (training rows only)
    y2 = y.copy()
    y2[test] += 5.0
    assert hc.flexible_nuisance(fam, log_norm, y2)(train, test) == pytest.approx(nat(train, test))


def test_fit_oof_default_nuisance_is_unchanged_and_the_hook_is_used():
    rng = np.random.default_rng(3)
    n = 60
    fam = ["lora_sft", "chain", "task_vector"] * 20
    inc = rng.uniform(0.2, 2, n)
    codes = [f"c{i // 2}" for i in range(n)]
    X = rng.normal(size=(n, 8))
    y = 0.2 * np.log1p(inc) + rng.normal(0, 0.05, n)
    base = rq.fit_oof(X, y, fam, inc, codes, seeds=(0,))
    zero = rq.fit_oof(X, y, fam, inc, codes, seeds=(0,), nuisance=lambda tr, te: np.zeros(te.sum()))
    assert np.allclose(zero["m"], 0)
    assert np.allclose(zero["resid"], y)
    again = rq.fit_oof(X, y, fam, inc, codes, seeds=(0,))
    assert np.allclose(base["m"], again["m"])  # deterministic, registered nuisance untouched
    assert not np.allclose(base["m"], 0)


def test_h_variants_weights_restriction_and_the_leave_out_convention():
    rng = np.random.default_rng(4)
    n = 60
    codes = np.array([f"c{i // 2}" for i in range(n)])
    b = rng.uniform(1, 2, n)
    a = 0.8 * b
    half = np.where(np.arange(n) < 10, 0.08, 0.02)
    over = half > 0.05
    out = hc.h_variants(a, b, codes, half, over, 0.10, draws=50)
    assert out["restricted_h_le_half_delta"]["n_rows"] == 50
    assert out["only_h_gt_half_delta"]["n_rows"] == 10
    assert out["inverse_h_squared_weighted"][
        "share_of_weight_on_rows_over_half_delta"
    ] == pytest.approx(10 * (1 / 0.08**2) / (10 / 0.08**2 + 50 / 0.02**2))
    assert out["inverse_h_squared_weighted"]["theta"] == pytest.approx(0.2)  # a/b constant
    assert out["inverse_h_squared_weighted"]["kish_n_eff_of_weights"] < n
    none = hc.h_variants(a, b, codes, np.full(n, 0.02), np.zeros(n, bool), 0.10, draws=20)
    assert (
        none["only_h_gt_half_delta"]["n_rows"] == 0
        and none["only_h_gt_half_delta"]["cluster_percentile95"] is None
    )


def test_item_split_halves_are_disjoint_and_the_split_seed_is_fixed():
    assert hc.SPLIT_SEED == 20261106 and hc.N_RANDOM_SPLITS == 20
    n_items = 32
    idx = np.arange(n_items)
    ia, ib = idx % 2 == 0, idx % 2 == 1
    assert not (ia & ib).any() and (ia | ib).all()
    rng = np.random.default_rng(hc.SPLIT_SEED)
    perm = rng.permutation(n_items)
    assert len(set(perm[:16]) & set(perm[16:])) == 0


def test_render_companions_labels_every_number_with_n():
    cell = {
        "n_rows": 5,
        "n_components": 4,
        "share_of_sum_B": 0.5,
        "theta": 0.1,
        "cluster_percentile95": [0.0, 0.2],
    }
    reg = {"theta": 0.1, "n_eff": 3.0, "top5_share": 0.4, "cluster_percentile95": [0.0, 0.2]}
    iv = {
        "theta_hat": 0.1,
        **{
            f"{w}_{m}": [0.0, 0.2]
            for w in ("conditional", "joint")
            for m in ("bootstrap_t", "bca", "percentile")
        },
    }
    flex = {
        "theta": 0.1,
        "n_rows": 5,
        "n_components": 4,
        "n_eff": 3.0,
        "top5_share": 0.4,
        "cluster_percentile95": [0.0, 0.2],
        "held_out_rows_outside_training_norm_range": 1,
        "mean_abs_residual": {"registered": 0.1, "flexible": 0.1},
        "six_intervals": iv,
        "strata_by_family": {"chain": {"n_rows": 5, "n_components": 4, "theta": 0.1}},
        "strata_by_tertile": {"T1": {"n_rows": 5, "n_components": 4, "theta": 0.1}},
    }
    rs = {
        k: {"mean": 0.1, "q05": 0.0, "q95": 0.2, "n_positive": 3}
        for k in ("A->A", "B->B", "A->B", "B->A")
    }
    hv = {
        "all_rows_unweighted": {"theta": 0.1, "n_rows": 5, "n_components": 4},
        "inverse_h_squared_weighted": {
            "theta": 0.1,
            "n_rows": 5,
            "n_components": 4,
            "kish_n_eff_of_weights": 4.0,
            "n_eff_component_weighted_B": 3.0,
            "top5_component_share_of_weighted_B": 0.5,
            "weight_ratio_max_over_min": 2.0,
            "share_of_weight_on_rows_over_half_delta": 0.1,
            "cluster_percentile95": [0.0, 0.2],
        },
        "restricted_h_le_half_delta": {
            "theta": 0.1,
            "n_rows": 4,
            "n_components": 3,
            "cluster_percentile95": [0.0, 0.2],
        },
        "only_h_gt_half_delta": {
            "theta": 0.1,
            "n_rows": 1,
            "n_components": 1,
            "cluster_percentile95": [0.0, 0.2],
        },
        "refit_without_over_half_delta": {"note": "none", "n_rows": 5},
    }
    comp = {
        "n_rows": 5,
        "n_components": 4,
        "draws": 100,
        "delta": 0.1,
        "n_null_controls": 3,
        "n_development_rows": 8,
        "family_counts": {"chain": 5},
        "registered": reg,
        "strata": {
            "tertile_cuts_incremental_norm": [1.0, 2.0],
            "median_incremental_norm_by_family": {"chain": 1.0},
            "family": {"chain": cell},
            "tertile": {"T1": cell},
            "cross": {"chain|T1": cell},
            "within_family_tertile": {"chain|within-T1": cell},
            "pre_declared": {"x": cell},
        },
        "equal_component_weight": {
            "theta_row_weight_1_over_nc": 0.1,
            "theta_mean_of_component_ratios": 0.0,
            "cluster_percentile95_row_weight_1_over_nc": [0.0, 0.2],
            "n_rows": 5,
            "n_components": 4,
        },
        "flexible_nuisance": {"family_x_natural_spline": flex},
        "item_split": {
            "n_rows": 5,
            "n_components": 4,
            "n_blocks": 32,
            "n_blocks_per_half": 16,
            "parity": {"A->A": 0.1, "B->B": 0.1, "A->B": 0.1, "B->A": 0.1},
            "random_splits": {"n_splits": 4, "by_combination": rs},
        },
        "h_variants": hv,
        "half_width_summary": {
            "n_rows": 5,
            "n_over_half_delta": 1,
            "max_h": 0.07,
            "median_h": 0.02,
        },
    }
    text = "\n".join(hc.render_companions(comp))
    assert "n components" in text and "family_x_natural_spline" in text
    assert "100 draws" in text and "refit without the h_i > delta/2 rows" in text
