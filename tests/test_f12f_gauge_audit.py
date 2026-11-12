"""F12f step 2: the F5 gauge audit in the simulation runner (chair ruling on F12e (a); F12.3; F12a.2(i); F12a.5(d)), and
the real-matrix schema (chair ruling (b): the 18-column synthetic v4 stand-in is for synthetic runs only). Synthetic
tensors and toy noise only; nothing under data/ is opened."""

import argparse
import dataclasses

import numpy as np
import pytest

from scripts import f12_simulation as fs
from tests.test_f12_simulation import TINY, small_matrix, toy_payload


def _with_gauge(m, perturb=None):
    m = dict(m)
    g = np.array(m["tokens"], dtype=np.float64, copy=True)
    if perturb is not None:
        perturb(g, m)
    m["tokens_gauge"] = g
    return m


# --------------------------------------------------------------------------------------------- tolerance


def test_tolerance_is_the_f5_pin_per_row_in_nats():
    assert (fs.GAUGE_ATOL, fs.GAUGE_RTOL) == (1e-5, 1e-4)
    zero = np.zeros(2)
    assert fs.gauge_within_tolerance(zero, np.array([1e-5, -1e-5])).all()  # equality passes
    assert not fs.gauge_within_tolerance(zero, np.array([1.0000001e-5, -1.0000001e-5])).any()
    f = np.array([2.0, -3.0, 1e3])
    tol = 1e-5 + 1e-4 * np.abs(f)
    assert fs.gauge_within_tolerance(f, f + 0.999 * tol).all()
    assert not fs.gauge_within_tolerance(f, f - 1.001 * tol).any()
    assert not fs.gauge_within_tolerance(
        np.array([1.0]), np.array([np.nan])
    ).any()  # nonfinite fails


# ------------------------------------------------------------------------------------------- audit unit


def _fitted(m, cfg=TINY):
    data = fs.HeadData(m)
    rows = np.arange(data.n)
    st = data.stats(rows)
    plist = [fs._seeded(s) for s in cfg.seeds]
    y = np.random.default_rng(0).normal(size=data.n)
    _, P = fs.train_heads(data, plist, [st] * len(plist), [y] * len(plist), [rows] * len(plist), cfg,
                          epochs=[2] * len(plist))  # fmt: skip
    return data, P, st


def test_identical_gauge_tokens_give_no_discrepancy():
    m = _with_gauge(small_matrix())
    data, P, st = _fitted(m)
    dg = fs.HeadData(m, tokens_key="tokens_gauge")
    test = np.arange(0, data.n, 3)
    a = fs.gauge_audit(data, dg, P, st, 0.7, test, (0, 1, 2)[: P["w"].shape[0]], fold=2)
    assert a["violations"] == 0 and a["discrepancies"] == []
    assert a["fits"] == [
        {"fold": 2, "seed": 0, "rows": len(test), "violations": 0, "max_abs_delta": 0.0}
    ]


def test_perturbed_gauge_token_is_a_violation_and_every_discrepancy_is_kept():
    base = small_matrix()
    row = int(base["token_row"][0])

    def bump(g, m):
        g[m["token_row"] == row, 0:16] += 50.0

    m = _with_gauge(base, bump)
    data, P, st = _fitted(m)
    dg = fs.HeadData(m, tokens_key="tokens_gauge")
    test = np.arange(data.n)
    a = fs.gauge_audit(data, dg, P, st, 1.3, test, (0,), fold=0)
    assert a["violations"] >= 1
    assert {d["row"] for d in a["discrepancies"]} == {row}
    d = a["discrepancies"][0]
    assert set(d) >= {"fold", "seed", "row", "f", "f_gauge", "delta", "tolerance", "violation"}
    assert d["violation"] is True and d["delta"] == pytest.approx(abs(d["f_gauge"] - d["f"]))
    assert d["tolerance"] == pytest.approx(1e-5 + 1e-4 * abs(d["f"]))
    # f is the seed's head prediction in nats (scaled back by the residual-target SD)
    ref = fs.predict_heads(data, P, [st], test)[0] * 1.3
    assert d["f"] == pytest.approx(ref[list(test).index(row)])


def test_small_discrepancy_within_tolerance_is_retained_not_a_violation():
    base = small_matrix()
    row = int(base["token_row"][0])

    def nudge(g, m):
        sel = m["token_row"] == row
        g[sel, 0:16] *= 1 + 1e-5

    m = _with_gauge(base, nudge)
    data, P, st = _fitted(m)
    dg = fs.HeadData(m, tokens_key="tokens_gauge")
    a = fs.gauge_audit(data, dg, P, st, 1.0, np.arange(data.n), (0,), fold=1)
    assert a["violations"] == 0
    assert all(d["violation"] is False for d in a["discrepancies"])
    assert {d["row"] for d in a["discrepancies"]} <= {row}


# ------------------------------------------------------------------------------------------- replicate


def _ctx(m, k=10):
    return fs.Context(m, toy_payload(n_comp=10, per=3, seed=5), k, TINY)


def test_replicate_audits_every_seed_and_outer_fold_without_changing_the_fits():
    plain = small_matrix()
    cfg = dataclasses.replace(TINY, seeds=(0, 1))
    pay = toy_payload(n_comp=10, per=3, seed=5)
    rec0 = fs.run_replicate(fs.Context(plain, pay, 10, cfg), "null", 0)
    rec1 = fs.run_replicate(fs.Context(_with_gauge(plain), pay, 10, cfg), "null", 0)
    assert (
        rec0["gauge_audit"]["applied"] is False
        and "no gauge-transformed tokens" in rec0["gauge_audit"]["reason"]
    )
    g = rec1["gauge_audit"]
    assert g["applied"] is True and g["violations"] == 0
    folds = {f["fold"] for f in g["fits"]}
    assert len(g["fits"]) == len(folds) * 2 and {f["seed"] for f in g["fits"]} == {0, 1}
    assert len(folds) == 5
    assert (
        g["rows_audited"]
        == sum(f["rows"] for f in g["fits"])
        == 2 * len(fs.Context(plain, pay, 10, cfg).ids)
    )
    for key in ("A", "decision", "valid", "C_S_F3"):
        assert rec0[key] == rec1[key]


def test_replicate_with_a_gauge_violation_is_invalid_and_wmr_contrasts_undefined():
    base = small_matrix()
    m = _with_gauge(
        base, lambda g, mm: g.__setitem__((slice(None), slice(0, 16)), g[:, 0:16] + 25.0)
    )
    rec = fs.run_replicate(_ctx(m), "null", 0)
    g = rec["gauge_audit"]
    assert g["applied"] and g["violations"] > 0
    assert rec["valid"] is False and rec["decision"] == "unresolved_validity"
    assert rec["failure"].startswith("gauge audit")
    for c in ("C_WM", "C_WM_S"):
        assert rec[c]["valid"] is False and rec[c]["undefined"] == "WM-R gauge violation (F5 pins)"
    # a gauge failure counts as a replicate failure in the rates (F12a.5(d))
    rates = fs.f12a_rates([rec], dataclasses.replace(TINY, gating="f12a"))
    assert rates["power"]["count"] == 0 and any(
        "gauge audit" in k for k in rates["invalid_by_cause"]
    )


# ------------------------------------------------------------------------------------------ real matrix


def _real_like(m, v4_cols=None):
    """A non-synthetic matrix with the real layout (v4 width from the pinned pooling code)."""
    n = len(m["row_ids"])
    out = _with_gauge(m)
    out["synthetic"] = False
    out["v4"] = np.zeros((n, fs.V4_REAL_COLS if v4_cols is None else v4_cols))
    out["x0_v4"] = np.zeros(fs.V4_REAL_COLS)
    return out


def test_real_v4_width_is_the_pinned_pooling_code():
    from wsp.features import pool

    s, p = pool.pooled_dims(4)
    assert fs.V4_REAL_COLS == s + p == 1378
    assert fs.V4_COLS == 18


def test_non_synthetic_matrix_refuses_the_synthetic_stand_in_and_needs_gauge_and_x0():
    m = small_matrix()
    fs.validate_matrix(_real_like(m))
    with pytest.raises(ValueError, match="synthetic"):
        fs.validate_matrix(_real_like(m, v4_cols=fs.V4_COLS))
    no_gauge = _real_like(m)
    del no_gauge["tokens_gauge"]
    with pytest.raises(ValueError, match="gauge"):
        fs.validate_matrix(no_gauge)
    no_x0 = _real_like(m)
    del no_x0["x0_v4"]
    with pytest.raises(ValueError, match="x0"):
        fs.validate_matrix(no_x0)
    bad = _real_like(m)
    bad["tokens_gauge"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fs.validate_matrix(bad)
    bad = _real_like(m)
    bad["v4"][0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        fs.validate_matrix(bad)
    bad = _real_like(m)
    bad["x0_v4"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fs.validate_matrix(bad)
    short = _real_like(m)
    short["tokens_gauge"] = short["tokens_gauge"][:-1]
    with pytest.raises(ValueError, match="gauge"):
        fs.validate_matrix(short)


def test_matrix_hash_binds_gauge_tokens_and_x0_and_keeps_the_synthetic_hash():
    # the synthetic matrix's identity is unchanged by F12f (no optional keys)
    assert fs.matrix_sha256(fs.synthetic_token_matrix()) == (
        "ff59266763d1fb4c8ea4f5a802f5282ee1236130fc67cb142feb7f4c912db1a1"
    )
    m = _real_like(small_matrix())
    h = fs.matrix_sha256(m)
    g = dict(m, tokens_gauge=m["tokens_gauge"].copy())
    g["tokens_gauge"][0, 0] += 1e-9
    assert fs.matrix_sha256(g) != h
    x = dict(m, x0_v4=m["x0_v4"].copy())
    x["x0_v4"][3] = 1.0
    assert fs.matrix_sha256(x) != h


def test_save_load_and_subset_carry_the_gauge_tokens(tmp_path):
    m = _real_like(small_matrix())
    m["tokens_gauge"] = m["tokens_gauge"] + np.arange(len(m["tokens_gauge"]))[:, None] * 1e-3
    path = tmp_path / "m.npz"
    fs.save_matrix(path, m)
    back = fs.load_matrix(path)
    assert fs.matrix_sha256(back) == fs.matrix_sha256(m) and back["synthetic"] is False
    rows = fs.s_k_rows(back, 5)
    sub = fs.subset_matrix(back, rows)
    # each gauge token stays aligned with its token
    for t, gt, r, s in zip(sub["tokens"], sub["tokens_gauge"], sub["token_row"], sub["token_slot"]):
        i = int(np.flatnonzero((back["token_row"] == rows[r]) & (back["token_slot"] == s))[0])
        assert np.array_equal(t, back["tokens"][i]) and np.array_equal(gt, back["tokens_gauge"][i])
    assert np.array_equal(sub["x0_v4"], back["x0_v4"])
    F = fs.statnn_features(sub)
    assert F.shape == (len(rows), fs.V4_REAL_COLS + fs.N_SLOTS * (fs.N_COLS + 1))


def test_governed_runs_refuse_the_synthetic_stand_in(tmp_path):
    a = argparse.Namespace(synthetic=True, tokens=None)
    with pytest.raises(SystemExit, match="synthetic"):
        fs._load_matrix_arg(a, governed=True)
    with pytest.raises(SystemExit, match="synthetic"):
        fs.main(["noisecheck", "--synthetic", "--restricted-payload", str(tmp_path / "p.json"),
                 "--out", str(tmp_path / "o"), "--budget-ledger", str(tmp_path / "l.jsonl"),
                 "--budget-plan", str(tmp_path / "plan.json"), "--operator", "x"])  # fmt: skip


def test_governed_real_matrix_must_equal_the_committed_matrix_manifest(tmp_path, monkeypatch):
    m = _real_like(small_matrix())
    path = tmp_path / "m.npz"
    fs.save_matrix(path, m)
    man = tmp_path / "matrix_manifest.json"
    import json

    man.write_text(
        json.dumps({"matrix_sha256": fs.matrix_sha256(m), "file_sha256": fs.sha256_file(path)})
    )
    monkeypatch.setattr(fs, "MATRIX_MANIFEST", man)
    a = argparse.Namespace(synthetic=False, tokens=path)
    _mat, src = fs._load_matrix_arg(a, governed=True)
    assert src["matrix_manifest"]["sha256"] == fs.sha256_file(man)
    assert src["matrix_sha256"] == fs.matrix_sha256(m)
    man.write_text(json.dumps({"matrix_sha256": "0" * 64, "file_sha256": fs.sha256_file(path)}))
    with pytest.raises(SystemExit, match="matrix manifest"):
        fs._load_matrix_arg(a, governed=True)


def test_rules_text_states_the_gauge_audit():
    eff = fs.rules_text(fs.load_config(fs.F12C_CONFIGS["pilot"])[0])["effective"]
    text = eff["gauge_audit"]
    assert "1e-5 + 1e-4 |f|" in text and "each seed" in text and "outer fold" in text
    assert "undefined" in text and "synthetic" in text
    assert "v4" in eff["StatNN"] and str(fs.V4_REAL_COLS) in eff["StatNN"]
