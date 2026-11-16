"""MSRM-1 development v0: nested fitting contract (F3u, residual heads, arms on identical rows/folds).

Synthetic data only; exploratory development code.
"""

import dataclasses

import numpy as np
import pytest

from wsp.msrm import evaluate as ev
from wsp.msrm import models as md

FAST = dataclasses.replace(md.FitConfig(), max_epochs=25, patience=6, tree_grid=(50, 150))


def _synthetic(n_comp=36, per=2, d=6, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_comp * per
    comps = np.repeat([f"c{i:03d}" for i in range(n_comp)], per)
    a = rng.normal(size=n)
    q = rng.gamma(2.0, 1.0, size=n)
    X = rng.normal(size=(n, d))
    y = 0.3 + 1.5 * a - 0.4 * q + signal * np.tanh(2 * X[:, 0]) + 0.05 * rng.normal(size=n)
    data = md.Dataset(
        row_ids=np.array([f"r{i:03d}" for i in range(n)]),
        comps=comps,
        y=y,
        a=a,
        q=q,
        inputs={"x": (X, np.zeros(d)), "desc": (rng.normal(size=(n, 3)), np.zeros(3))},
    )
    folds = {c: i % 5 for i, c in enumerate(sorted(set(comps)))}
    return data, np.array([folds[c] for c in comps])


def test_f3u_is_unpenalised_ols():
    rng = np.random.default_rng(1)
    a, q = rng.normal(size=30), rng.gamma(2.0, size=30)
    y = 0.7 + 2.0 * a + 3.0 * q
    beta = md.f3u_fit(a, q, y)
    np.testing.assert_allclose(beta, [0.7, 2.0, 3.0], atol=1e-10)  # no ridge shrinkage
    yn = y + rng.normal(size=30)
    ref = np.linalg.lstsq(np.column_stack([np.ones(30), a, q]), yn, rcond=None)[0]
    np.testing.assert_allclose(md.f3u_fit(a, q, yn), ref, atol=1e-12)


def test_mlp_head_has_401_parameters_at_48_inputs():
    assert md.mlp_param_count(48) == 401
    assert md.mlp_param_count(54) == 54 * 8 + 8 + 8 + 1


@pytest.mark.parametrize("head", ["mlp", "ridge", "tree"])
def test_residual_correction_is_zero_at_the_zero_edit(head):
    data, fold = _synthetic()
    X, x0 = data.inputs["x"]
    X = X.copy()
    X[0] = x0  # row 0 is the zero edit (a = q = 0)
    data.a[0], data.q[0] = 0.0, 0.0
    data.inputs["x"] = (X, x0)
    arm = md.ArmSpec("msrm", base="f3u", inputs="x", head=head)
    test = np.flatnonzero(fold == fold[0])
    train = np.flatnonzero(fold != fold[0])
    pred, info = md.fit_predict_arm(arm, data, train, test, FAST)
    beta = md.f3u_fit(data.a[train], data.q[train], data.y[train])
    assert pred[list(test).index(0)] == beta[0]
    assert info["correction_at_zero"] == 0.0


def test_preprocessing_is_isolated_inside_each_training_fold():
    data, fold = _synthetic(seed=2)
    X, x0 = data.inputs["x"]
    t = md.InputTransform().fit(X[:40])
    t2 = md.InputTransform().fit(np.vstack([X[:40]]))
    np.testing.assert_array_equal(t.transform(X), t2.transform(X))
    test = np.flatnonzero(fold == 1)
    train = np.flatnonzero(fold != 1)
    arm = md.ArmSpec("msrm", base="f3u", inputs="x", head="ridge")
    p1, _ = md.fit_predict_arm(arm, data, train, test, FAST)
    # held-out outcomes and other held-out inputs cannot move a held-out prediction
    data2 = dataclasses.replace(data, y=data.y.copy(), inputs=dict(data.inputs))
    data2.y[test] = 1e6
    X2 = X.copy()
    X2[test[1:]] += 100.0
    data2.inputs["x"] = (X2, x0)
    p2, _ = md.fit_predict_arm(arm, data2, train, test, FAST)
    assert p2[0] == p1[0]
    np.testing.assert_array_equal(p2[0], p1[0])


def test_inner_folds_are_component_disjoint_and_within_training():
    data, fold = _synthetic(seed=3)
    train = np.flatnonzero(fold != 0)
    inner = md.inner_fold_of_rows(data.comps[train])
    for j in range(4):
        va = set(data.comps[train][inner == j])
        tr = set(data.comps[train][inner != j])
        assert va and not (va & tr)


def test_learned_head_finds_a_nonlinear_residual_signal():
    data, fold = _synthetic(signal=1.0, seed=4)
    arms = [md.ArmSpec("f3u", base="f3u"), md.ArmSpec("msrm", base="f3u", inputs="x", head="tree")]
    oof = md.run_cv(arms, data, fold, FAST)
    c = ev.contrast(data.y, oof["msrm"], oof["f3u"])
    assert c > 0.2


def test_shuffled_outcome_control_shows_no_gain():
    data, fold = _synthetic(signal=1.0, seed=5)
    shuffled = md.shuffle_outcomes(data, seed=0)
    assert sorted(shuffled.y) == sorted(data.y) and not np.array_equal(shuffled.y, data.y)
    arms = [md.ArmSpec("f3u", base="f3u"), md.ArmSpec("msrm", base="f3u", inputs="x", head="ridge")]
    oof = md.run_cv(arms, shuffled, fold, FAST)
    assert ev.contrast(shuffled.y, oof["msrm"], oof["f3u"]) < 0.05


def test_all_arms_receive_identical_eligible_rows():
    data, fold = _synthetic(seed=6)
    D, d0 = data.inputs["desc"]
    D = D.copy()
    D[5, 1] = np.nan  # a missing required input excludes the row from every arm
    data.inputs["desc"] = (D, d0)
    elig = md.eligible_rows(data)
    assert not elig[5] and elig.sum() == len(elig) - 1
    arms = [
        md.ArmSpec("f3u", base="f3u"),
        md.ArmSpec("zero_a", base="zero_a"),
        md.ArmSpec("msrm", base="f3u", inputs="x", head="ridge"),
        md.ArmSpec("desc", base="const", inputs="desc", head="ridge"),
    ]
    oof = md.run_cv(arms, data, fold, FAST)
    rows = {k: np.flatnonzero(np.isfinite(v)) for k, v in oof.items()}
    assert all(np.array_equal(rows["f3u"], r) for r in rows.values())
    assert 5 not in rows["f3u"]
    np.testing.assert_array_equal(oof["zero_a"][elig], data.a[elig])


def test_head_class_selection_is_one_of_the_candidates():
    data, _ = _synthetic(seed=7)
    rows = np.arange(len(data.y))
    choice, table = md.select_head_class(data, rows, "x", FAST, candidates=("ridge", "tree"))
    assert choice in ("ridge", "tree")
    assert set(table) == {"ridge", "tree"}
    assert table[choice] == min(table.values())


@pytest.mark.parametrize("head", ["ridge", "tree", "mlp"])
def test_zero_edit_anchor_is_part_of_the_fit_not_an_offset(head):
    """x(0) far from every training row: the fitted function must be the evaluated h(x) − h(x0), not a fit
    whose constant offset −h(x0) is added afterwards (the dry-run ridge failure)."""
    data, fold = _synthetic(seed=8, d=4)
    X, _ = data.inputs["x"]
    X = X + 6.0  # no training row is near x0 = 0
    data.inputs["x"] = (X, np.zeros(4))
    data.y = 0.3 + 1.5 * data.a - 0.4 * data.q + 0.2 * (X[:, 0] - 6.0)
    arms = [md.ArmSpec("f3u", base="f3u"), md.ArmSpec("m", base="f3u", inputs="x", head=head)]
    oof = md.run_cv(arms, data, fold, FAST)
    assert ev.contrast(data.y, oof["m"], oof["f3u"]) > -0.1
    assert abs(float(np.mean(data.y - oof["m"]))) < 0.1
