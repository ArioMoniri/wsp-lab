"""scripts/wme_w2_model_search.py: the post-hoc W2 model search (SYNTHESIS-r13 addendum 3). Synthetic arrays only."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from wsp.wm import stagew as W
from wsp.wm import stagew2 as W2

P = Path(__file__).resolve().parents[1] / "scripts" / "wme_w2_model_search.py"
spec = importlib.util.spec_from_file_location("wme_w2_model_search", P)
MS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MS)

J = 6


def synth(
    n_runs=8, seed=0, dose_slope=0.05, j1_slope=0.0, carry=0.0, noise=0.01, N=500, grow=False
):
    """3 types x n_runs paths. y = type curve + dose_slope * D + j1_slope * J1G[..., 0] (+ a path effect that is
    visible at every point when ``carry``) + noise."""
    rng = np.random.default_rng(seed)
    runs = np.repeat([f"r{i:02d}" for i in range(n_runs)], 3)
    types = np.tile(list("SLM"), n_runs)
    P_ = len(runs)
    lr = np.repeat(rng.uniform(-1, 1, n_runs), 3)
    D = lr[:, None] + np.linspace(-1, 0, J)[None, :] + 0.05 * rng.normal(size=(P_, J))
    J1G = rng.normal(size=(P_, J, 24))
    R1S = rng.normal(size=(P_, J, 3))
    curve = {
        t: 0.3 + 0.1 * i + 0.02 * np.arange(J)[:, None] * np.ones((1, 4))
        for i, t in enumerate("SLM")
    }
    eff = carry * rng.normal(size=(P_, 1, 4))
    if grow:
        eff = eff * np.arange(1, J + 1)[None, :, None] / 2
    Y = np.stack([curve[t] for t in types]) + dose_slope * D[..., None] + j1_slope * J1G[..., :1]
    Y = np.clip(Y + eff + noise * rng.normal(size=Y.shape), 0.02, 0.98)
    return {
        "Y": Y,
        "SE": W.se_of(Y, N),
        "types": types,
        "runs": runs,
        "run_ids": list(dict.fromkeys(runs.tolist())),
        "ids": [f"{t}:{r}" for t, r in zip(types, runs)],
        "N": N,
        "D": D,
        "J1G": J1G,
        "R1S": R1S,
    }


# ------------------------------------------------------------------ features


def test_j1_groups_sum_to_the_w2_member_features():
    W.configure_layers(32)
    rng = np.random.default_rng(1)
    j1raw = {("r01", 8): rng.normal(size=(3, 5, 32))}
    for lam, groups in ((1.0, 8), (0.5, 8), (1.0, 3)):
        sp = W.member("r01", 8, lam=lam, groups=groups)
        g = MS.j1_groups(sp, {k: v.mean(1) for k, v in j1raw.items()})
        assert g.shape == (24,)
        ref = W.member_features(sp, {("r01", 8): np.ones(32)}, {}, j1raw)[4:7]
        assert np.allclose(g.reshape(3, 8).sum(1), ref)
        assert np.all(g.reshape(3, 8)[:, groups:] == 0)


def test_dose_is_the_log_total_r0_norm():
    phi = np.array([[3.0, 4.0, 0.0, 0.0, 9, 9, 9]])
    assert np.allclose(MS.dose_of(phi), math.log(5.0))


# ------------------------------------------------------------------ metric and baselines


def test_scores_reproduce_w2_score_and_the_pooled_curve():
    A = synth()
    tr = A["run_ids"][1:]
    pred = MS.base_predict(A, np.isin(A["runs"], tr), "pooled")
    for t in "SLM":
        ix = np.isin(A["runs"], tr) & (A["types"] == t)
        assert np.allclose(pred[A["types"] == t], A["Y"][ix].mean(0))
    for k in (0, 2):
        pk = MS.offset(pred, A["Y"], k, 1.0)
        S = MS.endpoint_scores(pk, A, k)
        assert S.shape == (len(A["ids"]), 4)
        for i in range(3):
            assert np.isclose(S[i].mean(), W2.score(pk[i], A, i, k))


def test_offset_weight():
    A = synth()
    pred = np.zeros_like(A["Y"])
    assert np.array_equal(MS.offset(pred, A["Y"], 0, 1.0), pred)
    o = MS.offset(pred, A["Y"], 2, 0.5)
    assert np.allclose(o, 0.5 * A["Y"][:, :2].mean(1, keepdims=True))


def test_dose_baseline_fits_an_exact_linear_dose_world():
    A = synth(noise=0.0)
    tr = A["run_ids"][1:]
    pred = MS.base_predict(A, np.isin(A["runs"], tr), "dose")
    te = A["runs"] == A["run_ids"][0]
    assert np.allclose(pred[te], A["Y"][te], atol=1e-9)
    lp = MS.base_predict(A, np.isin(A["runs"], tr), "logit_dose")
    assert np.all((lp > 0) & (lp < 1))


def test_crossfit_never_uses_the_path_own_run():
    A = synth()
    tr = A["run_ids"][:5]
    cf = MS.crossfit(A, tr, "pooled")
    for q in tr:
        ix = A["runs"] == q
        other = np.isin(A["runs"], [r for r in tr if r != q])
        for t in "SLM":
            assert np.allclose(
                cf[ix & (A["types"] == t)], A["Y"][other & (A["types"] == t)].mean(0)
            )
    assert np.all(np.isnan(cf[~np.isin(A["runs"], tr)]))


# ------------------------------------------------------------------ candidates


@pytest.mark.parametrize("cand", ["ridge", "gp", "per_endpoint"])
def test_first_grid_entry_is_the_dose_baseline(cand):
    A = synth()
    tr = A["run_ids"][1:]
    for k in (0, 2):
        g = MS.grid_preds(A, tr, k, cand)
        first = MS.GRIDS[cand](k)[0]
        base = MS.offset(MS.base_predict(A, np.isin(A["runs"], tr), "dose"), A["Y"], k, 1.0)
        assert np.allclose(g[first], base)


def test_ridge_finds_a_planted_j1_signal_and_tunes_toward_it():
    A = synth(n_runs=10, j1_slope=0.03, noise=0.002)
    res = MS.run_candidate(A, "ridge", ks=(0,), workers=1)
    dose = MS.run_candidate(A, "dose", ks=(0,), workers=1)
    assert res["E"][0].mean() < 0.75 * dose["E"][0].mean()
    assert all(h != MS.GRIDS["ridge"](0)[0] for h in res["chosen"][0].values())


def test_tie_rule_prefers_the_earlier_grid_entry():
    assert MS.choose([1.0, 0.999, 0.5 + 0.6], ["a", "b", "c"]) == "a"
    assert MS.choose([1.0, 0.9, 0.95], ["a", "b", "c"]) == "b"


def test_per_endpoint_picks_per_endpoint():
    A = synth(n_runs=10, j1_slope=0.0, noise=0.002)
    # plant a J1 signal in truncation only
    A["Y"][..., 2:] += 0.03 * A["J1G"][..., :1]
    A["SE"] = W.se_of(A["Y"], A["N"])
    res = MS.run_candidate(A, "per_endpoint", ks=(0,), workers=1)
    ch = list(res["chosen"][0].values())
    assert all(c[2][0].startswith("ridge") and c[3][0].startswith("ridge") for c in ch)


def test_sequence_off_entry_is_the_baseline_and_models_count_parameters():
    A = synth(n_runs=5)
    tr = A["run_ids"][1:]
    g = MS.grid_preds(A, tr, 2, "gru", seeds=1, steps=5)
    base = MS.offset(MS.base_predict(A, np.isin(A["runs"], tr), "dose"), A["Y"], 2, 1.0)
    te = A["runs"] == A["run_ids"][0]
    assert np.allclose(g["off"][te], base[te])
    assert MS.n_params("gru", 4) == 248 and MS.n_params("gru", 8) == 588
    assert MS.n_params("ssm", 4) == 240 and MS.n_params("ssm", 8) == 424


@pytest.mark.parametrize("arch", ["gru", "ssm"])
def test_sequence_model_extrapolates_a_growing_path_effect_at_k2(arch):
    # a path effect that grows along the path: the 2 given points reveal it, the mean offset under-corrects it, and
    # a sequence model can learn to extrapolate it
    A = synth(n_runs=12, carry=0.02, noise=0.002, seed=3, grow=True)
    tr = A["run_ids"][1:]
    te = A["runs"] == A["run_ids"][0]
    g = MS.grid_preds(A, tr, 2, arch, seeds=1, steps=300)
    S_off = MS.endpoint_scores(g["off"], A, 2)[te].mean()
    S_fit = MS.endpoint_scores(g[(8, 0.001)], A, 2)[te].mean()
    assert S_fit < 0.8 * S_off


# ------------------------------------------------------------------ tests and nulls


def test_holm_family_and_selection_rule():
    rows = {
        "ridge": {
            0: {"p": 0.001, "d_mean": 0.1, "lb": 0.05},
            2: {"p": 0.5, "d_mean": 0.0, "lb": -0.1},
        },
        "gp": {
            0: {"p": 0.004, "d_mean": 0.2, "lb": 0.02},
            2: {"p": 0.003, "d_mean": 0.2, "lb": 0.01},
        },
        "gru": {
            0: {"p": 0.9, "d_mean": -0.1, "lb": -0.3},
            2: {"p": 0.9, "d_mean": -0.1, "lb": -0.3},
        },
    }
    dec = MS.holm_family({(c, k): r[k]["p"] for c, r in rows.items() for k in r})
    assert (
        dec[("ridge", 0)]["reject"] and dec[("gp", 0)]["reject"] and not dec[("gru", 0)]["reject"]
    )
    assert np.isclose(dec[("ridge", 0)]["alpha"], 0.025 / 6)
    beats = {c: True for c in ("ridge", "gp")}
    assert MS.select_frozen(rows, beats) == "gp"
    assert MS.select_frozen(rows, {}) is None


def test_null_n1_shuffles_only_non_dose_features_within_type():
    A = synth()
    B = MS.null_arrays(A, "n1", seed=0)
    assert np.array_equal(B["D"], A["D"]) and np.array_equal(B["Y"], A["Y"])
    for t in "SLM":
        ix = A["types"] == t
        a = A["J1G"][ix].reshape(ix.sum(), -1)
        b = B["J1G"][ix].reshape(ix.sum(), -1)
        assert sorted(map(tuple, a)) == sorted(map(tuple, b))
        assert not any(np.array_equal(a[i], b[i]) for i in range(len(a)))
    C = MS.null_arrays(A, "n1_dose", seed=0)
    assert not np.array_equal(C["D"], A["D"]) and np.array_equal(C["J1G"], A["J1G"])


def test_null_n2_deranges_outcomes_within_type():
    A = synth()
    B = MS.null_arrays(A, "n2", seed=0)
    assert np.array_equal(B["J1G"], A["J1G"])
    for t in "SLM":
        ix = A["types"] == t
        assert sorted(map(tuple, A["Y"][ix].reshape(ix.sum(), -1))) == sorted(
            map(tuple, B["Y"][ix].reshape(ix.sum(), -1))
        )
    assert np.allclose(B["SE"], W.se_of(B["Y"], A["N"]))


def test_contrast_reports_d_p_and_lb():
    A = synth(n_runs=6)
    out = MS.contrast(
        MS.run_candidate(A, "dose", ks=(0,), workers=1),
        MS.run_candidate(A, "pooled", ks=(0,), workers=1),
        A,
    )
    assert set(out[0]) >= {"d_mean", "p", "lb", "mean_E", "mean_E_ref"}
    assert out[0]["d_mean"] > 0
