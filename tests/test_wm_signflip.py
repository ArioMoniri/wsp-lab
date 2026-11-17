"""Sign-flip randomisation tests for H-T (component level) and H-E2 (family-stratified episodes) (RULING-wme-v2 A2-A3)."""

import numpy as np
import pytest

from wsp.wm import signflip as SF


def test_flip_matrix_is_seeded_and_fixed():
    a = SF.flips(7, n_flip=1000, seed=3)
    assert a.shape == (1000, 7) and set(np.unique(a)) <= {-1.0, 1.0}
    assert np.array_equal(a, SF.flips(7, n_flip=1000, seed=3))


def test_pvalue_exact_small_case():
    d = np.array([1.0, 2.0, 3.0])
    r = SF.test(d, n_flip=None)  # exact enumeration of the 8 sign patterns
    assert r["p"] == pytest.approx(1 / 8)  # only the identity attains the observed mean
    assert SF.test(-d, n_flip=None)["p"] == pytest.approx(1.0)


def test_lower_bound_by_inversion_brackets_the_mean_and_is_monotone():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 40)
    r = SF.test(d, n_flip=10_000, seed=1)
    lb = r["lb"]
    assert lb < d.mean()
    # inverting: the shifted data are just not rejected at the bound, rejected below it
    assert SF.test(d - (lb + 1e-3), n_flip=10_000, seed=1)["p"] > SF.ALPHA
    assert SF.test(d - (lb - 0.05), n_flip=10_000, seed=1)["p"] <= SF.ALPHA


def test_size_under_symmetric_null():
    rng = np.random.default_rng(1)
    rej = sum(
        SF.test(rng.standard_t(3, 20), n_flip=999, seed=s)["p"] <= SF.ALPHA for s in range(400)
    )
    assert rej / 400 <= 0.05


def test_ht_component_means_and_ct_interval():
    rng = np.random.default_rng(2)
    comps = np.repeat(np.arange(30), 8)
    e_st = np.abs(rng.normal(0, 2.0, comps.size))
    e_wm = 0.8 * e_st
    r = SF.ht_test(e_wm, e_st, comps, n_flip=4000, seed=0)
    assert r["n_components"] == 30
    assert r["ct"] == pytest.approx(0.2)
    assert r["p"] <= SF.ALPHA and r["lb"] > 0
    assert r["ct_lb"] < r["ct"] and r["ct_lb"] > 0
    null = SF.ht_test(e_st, e_st, comps, n_flip=4000, seed=0)
    assert null["p"] > SF.ALPHA and null["ct"] == 0.0


def test_e2_family_stratified():
    rng = np.random.default_rng(3)
    fam = np.arange(24) % 3
    d = rng.normal(0.3, 0.5, 24)
    r = SF.e2_test(d, fam, n_flip=4000, seed=0)
    assert r["stat"] == pytest.approx(np.mean([d[fam == f].mean() for f in range(3)]))
    assert r["lb"] < r["stat"]
    with pytest.raises(ValueError):
        SF.e2_test(d, np.zeros(24), n_flip=100, seed=0)  # needs every family
