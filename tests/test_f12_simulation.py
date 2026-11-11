"""F12.5 simulation: synthetic tensors only (no outcome, no data/evals). The real-payload check reads
only the pinned, outcome-free F8a.2 noise payload and is skipped when it is absent."""

import dataclasses
import json
from collections import Counter

import numpy as np
import pytest
import torch

from scripts import f12_simulation as fs

TINY = dataclasses.replace(
    fs.Config(),
    seeds=(0,),
    max_epochs=4,
    patience=2,
    tree_grid=(5,),
    draws=200,
    cal_draws=20,
    val_draws=20,
)


def toy_payload(n_comp=12, per=3, common=1.0, seed=0):
    """Centred donor noise with a shared window pattern (the base reference) + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=64)
    donors, noise = [], []
    for c in range(n_comp):
        for j in range(per):
            raw = common * shared + rng.normal(size=64)
            noise.append((raw - raw.mean()) * 0.01)
            donors.append({"id": f"d{c}-{j}", "component": f"pc{c:02}", "family": "chain"})
    return fs.NoisePayload(noise, np.ones((len(donors), 64), bool).tolist(), donors)


def small_matrix(n_existing=24, n_purchasable=20):
    rows = fs.synthetic_population()
    comps = []
    for origin, n in (("existing", n_existing), ("purchasable", n_purchasable)):
        seen = []
        for r in rows:
            if r["origin"] == origin and r["component"] not in seen:
                seen.append(r["component"])
        comps += seen[:n]
    keep = [r for r in rows if r["component"] in set(comps)]
    return fs.synthetic_token_matrix(keep)


# ------------------------------------------------------------------------------- schema


def test_synthetic_population_matches_f12_counts():
    rows = fs.synthetic_population()
    assert len(rows) == 270 and len({r["component"] for r in rows}) == 234
    assert Counter(r["family"] for r in rows) == {
        "chain": 105, "abliteration": 83, "random_dir": 53, "lora_sft": 29
    }  # fmt: skip
    ex = [r for r in rows if r["origin"] == "existing"]
    assert len(ex) == 123 and len({r["component"] for r in ex}) == 89
    sizes = Counter(Counter(r["component"] for r in rows).values())
    assert sizes == {1: 214, 2: 12, 3: 2, 4: 4, 5: 2}
    strata = Counter(
        fs.component_strata([r["component"] for r in rows], [r["family"] for r in rows]).values()
    )
    assert strata == {"abliteration": 83, "chain": 71, "random_dir": 50, "lora_sft": 29, "mixed": 1}


def test_token_schema_and_keying(tmp_path):
    m = fs.synthetic_token_matrix()
    fs.validate_matrix(m)
    assert fs.N_SLOTS == 186 and m["tokens"].shape[1] == 42
    assert fs.head_param_count() == 1185 and fs.head_param_count(39) == 1137
    down = np.array([fs.ROLES[r] == "mlp.down_proj" for r in m["token_role"]])
    assert np.all(m["tokens"][~down, 39:42] == 0) and np.all(m["tokens"][down, 41] == 1)
    assert np.all(m["tokens"][:, 37] == 0) and np.all(m["tokens"][:, 38] == 1)
    sums = np.bincount(m["token_row"], m["tokens"][:, 35], len(m["row_ids"]))
    np.testing.assert_allclose(sums, m["sum_a"])
    # keyed by component: a subset regenerates identical rows
    sub = small_matrix()
    idx = [list(m["row_ids"]).index(i) for i in sub["row_ids"]]
    ref = fs.subset_matrix(m, idx)
    assert fs.matrix_sha256(ref) == fs.matrix_sha256(sub)
    path = tmp_path / "t.npz"
    fs.save_matrix(path, sub)
    assert fs.matrix_sha256(fs.load_matrix(path)) == fs.matrix_sha256(sub)
    bad = dict(sub, token_role=sub["token_role"] + 1)
    with pytest.raises(ValueError):
        fs.validate_matrix(bad)


def test_folds_and_purchase_order():
    m = fs.synthetic_token_matrix()
    order = fs.purchase_order(m)
    assert len(order) == 145
    assert order == sorted(order, key=lambda c: fs.sha_key("20261111:f12:purchase:", c))
    for k in fs.K_GRID:
        rows = fs.s_k_rows(m, k)
        comps = set(m["component"][rows])
        assert len(comps) == 89 + k
    comps, fams = m["component"].tolist(), m["family"].tolist()
    folds = fs.outer_folds(comps, fams)
    strata = fs.component_strata(comps, fams)
    seq = []
    for s in fs.STRATA:
        seq += sorted(
            [c for c, t in strata.items() if t == s], key=lambda c: fs.sha_key("20261111:f12:", c)
        )
    assert [folds[c] for c in seq] == [i % 5 for i in range(len(seq))]
    inner = fs.inner_folds(seq[:40])
    assert sorted(Counter(inner.values()).values()) == [10, 10, 10, 10]


# ------------------------------------------------------------------------------- models


def test_f3_operator_is_ridge_one():
    rng = np.random.default_rng(1)
    a, q = rng.normal(size=50), rng.gamma(1.0, size=50)
    y = 0.3 + 2 * a + 0.5 * q + rng.normal(size=50) * 0.1
    tr, te = np.arange(40), np.arange(40, 50)
    X = np.column_stack([a, q])
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z = (X[tr] - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + np.eye(2), Z.T @ (y[tr] - y[tr].mean()))
    ref = y[tr].mean() + ((X[te] - mu) / sd) @ beta
    np.testing.assert_allclose(fs.f3_fit_predict(a, q, y, tr, te), ref, atol=1e-12)
    fold = np.arange(50) % 5
    P = fs.oof_operator(a, q, fold)
    for f in range(5):
        t = fold == f
        np.testing.assert_allclose(
            (P @ y)[t],
            fs.f3_fit_predict(a, q, y, np.flatnonzero(~t), np.flatnonzero(t)),
            atol=1e-12,
        )


def test_wmr_nests_f3_when_h_is_zero():
    m = small_matrix()
    data = fs.HeadData(m)
    rows = np.arange(data.n)
    st = data.stats(rows)
    p = fs.init_params(0)
    p["w"] = torch.zeros_like(p["w"])  # h == 0 for every input
    P = fs.stack_params([p])
    out = fs.predict_heads(data, P, [st], rows)
    assert np.all(out == 0.0)
    # zero edit -> f(D) - f(0) = 0 exactly for any parameters
    zero = dict(m)
    zero_rows = [{"id": "z", "component": "zc", "family": "chain", "origin": "existing"}]
    zero = fs.assemble_matrix(zero_rows, [([], np.zeros((0, fs.N_COLS)))], synthetic=True)
    zd = fs.HeadData(zero)
    out0 = fs.predict_heads(zd, fs.stack_params([fs.init_params(3)]), [st], [0])
    assert out0[0, 0] == 0.0
    # WM-R prediction = F3 + mean_s h: with h zeroed the fold prediction is F3's exactly
    rng = np.random.default_rng(0)
    y = m["sum_a"] + rng.normal(size=data.n) * 1e-3
    train, test = rows[: data.n - 10], rows[data.n - 10 :]
    f3 = fs.f3_fit_predict(m["sum_a"], m["sum_q"], y, train, test)
    np.testing.assert_array_equal(f3 + out[0, test], f3)


def test_batched_adamw_matches_torch():
    m = small_matrix()
    data = fs.HeadData(m)
    rows = np.arange(data.n)
    st = data.stats(rows)
    y = np.random.default_rng(2).normal(size=data.n) * 0.01
    cfg = dataclasses.replace(TINY, max_epochs=1)
    _, P = fs.train_heads(data, [fs._seeded(5)], [st], [y], [rows], cfg, epochs=[2])
    # reference: the same batches through torch.optim.AdamW on a KFHead
    torch.manual_seed(5)
    head = fs.KFHead()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    gen = torch.Generator().manual_seed(5)
    mu, sd = torch.tensor(st[0], dtype=torch.float32), torch.tensor(st[1], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(2):
        perm = torch.randperm(data.n, generator=gen)
        for s in range(0, data.n, 16):
            idx = perm[s : s + 16]
            x, role, mask = data.batch(idx[None], mu[None], sd[None])
            loss = (head(x[0], role[0], mask[0]) - yt[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    np.testing.assert_allclose(
        P["W1"][0].detach().numpy(), head.lin1.weight.detach().T.numpy(), atol=2e-6
    )
    np.testing.assert_allclose(
        P["w"][0].detach().numpy(), head.out.weight.detach()[0].numpy(), atol=2e-6
    )


# -------------------------------------------------------------------------------- noise


def test_repaired_noise_has_panel_mean_variance_and_clustering():
    pay = toy_payload()
    comp = np.repeat(np.arange(20), 2)  # 20 simulated components of two rows
    rng = np.random.default_rng(0)
    draws = [fs.draw_panel_noise(pay, comp, rng) for _ in range(800)]
    eps = np.stack([d[0].mean(1) for d in draws])
    assert np.all(eps.std(0) > 1e-4)  # non-zero panel-mean variance
    # without the common block resampling the centred donors' means vanish
    assert np.abs(np.stack([pay.blocks[d[1]].mean(1) for d in draws])).max() < 1e-12
    # both rows of a simulated component draw from one payload component
    for _, donors, _ in draws[:50]:
        pc = [pay.donors[j]["component"] for j in donors]
        assert all(pc[2 * c] == pc[2 * c + 1] for c in range(20))
    # common-item dependence across components (shared window pattern)
    C = np.corrcoef(eps.T)
    assert C[0, 2] > 0.3 and C[1, 39] > 0.3
    indep = toy_payload(common=0.0, seed=1)
    rng = np.random.default_rng(1)
    e2 = np.stack([fs.draw_panel_noise(indep, comp, rng)[0].mean(1) for _ in range(800)])
    assert abs(np.corrcoef(e2.T)[0, 39]) < 0.15


@pytest.mark.skipif(not fs.DEFAULT_PAYLOAD.exists(), reason="noise payload absent")
def test_real_payload_allowlist_and_panel_mean():
    pay = fs.NoisePayload.load()
    assert pay.provenance["sha256"] == fs.PAYLOAD_SHA256
    assert set(pay.restricted()) == {
        "schema",
        "provenance",
        "window_ids",
        "donors",
        "centred_noise",
        "window_mask",
    }
    assert all(set(d) <= set(fs.DONOR_FIELDS) for d in pay.donors)
    assert len(pay.donors) == 141 and len(pay.members) == 63
    diag = fs.noise_diagnostics(pay, np.arange(30), seed=0, draws=300)
    assert diag["panel_mean_sd_median"] > 1e-4
    assert diag["max_abs_mean_without_common_resampling"] < 1e-12


# ---------------------------------------------------------------------------- calibration


def test_gamma_calibration_hits_oracle_c():
    rng = np.random.default_rng(3)
    n = 400
    comp = np.arange(n)
    a = rng.normal(size=n) * 0.05
    q = np.abs(a) * 0.02
    h = rng.normal(size=n)
    h /= h.std()
    P = fs.oof_operator(a, q, comp % 5)
    pay = toy_payload(n_comp=30, per=4, common=0.3, seed=4)
    rc, rv = np.random.default_rng(10), np.random.default_rng(11)
    # large calibration sample: the common panel-mean term makes panel-level variance large, so
    # +-0.005 needs thousands of panels, not the drafting default's 100 (reported by the run)
    cal = np.stack([fs.draw_panel_noise(pay, comp, rc)[0].mean(1) for _ in range(4000)])
    val = np.stack([fs.draw_panel_noise(pay, comp, rv)[0].mean(1) for _ in range(4000)])
    gamma, info = fs.calibrate_gamma(P, a, h, cal, val, fs.Config())
    assert info["status"] == "ok" and gamma > 0
    assert abs(info["C_calibration"] - 0.15) <= 1e-6
    rt = np.random.default_rng(12)
    test = np.stack([fs.draw_panel_noise(pay, comp, rt)[0].mean(1) for _ in range(4000)])
    assert abs(fs.oracle_c(P, a, h, test, gamma) - 0.15) <= 0.005
    assert abs(info["C_validation"] - 0.15) <= 0.005 and info["C_validation_se"] < 0.005
    # no bracket -> failed qualification, never a substitute
    g2, i2 = fs.calibrate_gamma(P, a, h, cal, val, dataclasses.replace(fs.Config(), gamma_max=1e-7))
    assert g2 is None and i2["status"].startswith("failed")


# ---------------------------------------------------------------------------- intervals


def test_a542_contrast_guards_and_direction():
    rng = np.random.default_rng(0)
    n, comp = 60, np.repeat(np.arange(30), 2)
    D = rng.normal(size=(n, 1)) * 0.1 + rng.normal(size=(n, 32)) * 0.01
    y = D.mean(1)
    mults = fs.a542_mults(30, TINY)
    qb = y + rng.normal(size=n) * 0.05
    same = fs.a542_contrast(D, qb, qb, comp, mults)  # L = 0 is valid, but its studentiser is 0
    assert (
        same["C"] == 0.0 and not same["valid"] and "conditional draws invalid" in same["failures"]
    )
    good = fs.a542_contrast(D, y + rng.normal(size=n) * 0.005, qb, comp, mults)
    assert good["valid"] and good["C"] > 0.5
    assert (
        good["intervals"]["conditional_bootstrap_t"][0] > 0
        and good["intervals"]["joint_bootstrap_t"][0] > 0
    )
    assert set(good["intervals"]) == {
        f"{p}_{m}" for p in ("conditional", "joint") for m in ("bootstrap_t", "bca", "percentile")
    }
    exact = fs.a542_contrast(D, y.copy(), qb, comp, mults)  # A_cand = 0: guard fails, no clamp
    assert not exact["valid"] and "point guard" in exact["failures"]
    one = fs.a542_contrast(D, qb, y + 1, np.zeros(n, int), fs.a542_mults(1, TINY))
    assert not one["valid"]


def test_decision_order():
    def c(C, lo, hi, valid=True):
        iv = {f"{p}_bootstrap_t": [lo, hi] for p in ("conditional", "joint")}
        return {"valid": valid, "C": C, "intervals": iv}

    assert fs.f12_decision(c(0.2, 0.1, 0.3, False), c(0.2, 0.1, 0.3)) == "unresolved_validity"
    assert fs.f12_decision(c(0.2, 0.1, 0.3), c(0.2, 0.01, 0.3)) == "correction_detected"
    assert fs.f12_decision(c(0.2, 0.1, 0.3), c(0.04, 0.01, 0.3)) == "unresolved"
    assert fs.f12_decision(c(0.0, -0.1, 0.049), c(-0.3, -0.5, 0.0)) == "none_detectable_bounds"
    assert fs.f12_decision(c(0.0, -0.1, 0.05), c(-0.3, -0.5, 0.0)) == "unresolved"


# --------------------------------------------------------------------------- Monte Carlo


def test_monte_carlo_bounds_and_gates():
    cfg = fs.Config()
    assert cfg.bonferroni_m == 24
    b = fs.mc_bounds(850, 1000, cfg)
    assert b["bonferroni_lower"] >= 0.80 and b["cp95"][0] < 0.85 < b["cp95"][1]
    assert fs.mc_bounds(810, 1000, cfg)["bonferroni_lower"] < 0.80  # point .81 does not release
    assert fs.mc_bounds(800, 1000, cfg)["bonferroni_lower"] < 0.80
    assert fs.mc_bounds(40, 1000, cfg)["bonferroni_upper"] <= 0.07
    assert fs.mc_bounds(60, 1000, cfg)["bonferroni_upper"] > 0.07  # point .06 does not pass
    a1 = 0.05 / 24
    n90 = fs.replicates_needed(0.90, 0.80, "power", a1)
    n85 = fs.replicates_needed(0.85, 0.80, "power", a1)
    assert n90 < n85 <= 1000 and fs.replicates_needed(0.80, 0.80, "power", a1) is None
    assert fs.replicates_needed(0.05, 0.07, "size", a1) > fs.replicates_needed(
        0.03, 0.07, "size", a1
    )

    def rec(scen, decision, lo_c, lo_j, valid=True):
        iv = {
            f"{p}_{m}": [-1.0, 1.0]
            for p in ("conditional", "joint")
            for m in ("bootstrap_t", "bca", "percentile")
        }
        iv["conditional_bootstrap_t"] = [lo_c, 1.0]
        iv["joint_bootstrap_t"] = [lo_j, 1.0]
        c = {"valid": valid, "C": 0.1, "intervals": iv}
        return {"k": 0, "scenario": scen, "valid": valid, "decision": decision, "C_WM": c, "C_WM_S": c,
                "A": {"F3": 1.0, "WM": 0.9, "S": 0.95, "oracle": 0.85}}  # fmt: skip

    recs = [rec("null", "unresolved", -0.1, -0.1)] * 1000
    recs += [rec("alt", "correction_detected", 0.1, 0.1)] * 900 + [
        rec("alt", "unresolved", -1, -1, False)
    ] * 100
    s = fs.summarise(recs, cfg)
    assert s["0"]["gates"] == {"size_pass": True, "size_point_pass": True, "power_pass": True,
                               "power_point_pass": True, "both_pass": True}  # fmt: skip
    assert s["0"]["alt"]["invalid"] == 100 and s["0"]["alt"]["detected"]["rate"] == 0.9
    rel = fs.release_decision(s, cfg, [0], 1000, True)
    assert not rel["binding"] and rel["outcome"] == "no gate decision"  # partial k grid
    rel = fs.release_decision(s, cfg, list(fs.K_GRID), 1000, True)
    assert rel["binding"] and rel["outcome"] == "release S_0"


# ------------------------------------------------------------------ end to end, determinism


def test_statnn_tree_count_chosen_on_inner_folds():
    m = small_matrix()
    F = fs.statnn_features(m)
    assert F.shape[1] == fs.V4_COLS + 186 * 43
    y = m["sum_a"] + np.random.default_rng(0).normal(size=len(F)) * 1e-3
    comps = m["component"]
    train, test = np.arange(len(F) - 8), np.arange(len(F) - 8, len(F))
    cfg = dataclasses.replace(TINY, tree_grid=(3, 6))
    pred, info = fs.fit_statnn_fold(F, m["sum_a"], m["sum_q"], y, comps, train, test, cfg)
    assert info["trees"] in (3, 6) and pred.shape == (8,) and np.all(np.isfinite(pred))


def test_null_size_reduced_and_determinism():
    m = small_matrix()
    pay = toy_payload(n_comp=10, per=3, seed=5)
    k = 10
    recs, _ = fs.simulate(m, pay, TINY, [k], ["null", "alt"], replicates=3, workers=1)
    assert len(recs) == 6
    null = [r for r in recs if r["scenario"] == "null"]
    assert all(r["gamma"] == 0.0 for r in null)
    assert sum(r["decision"] == "correction_detected" for r in null) <= 1
    alt = [r for r in recs if r["scenario"] == "alt"]
    assert all(r["gamma"] > 0 for r in alt if r["valid"])
    again, _ = fs.simulate(m, pay, TINY, [k], ["null", "alt"], replicates=3, workers=2)
    dump = lambda x: json.dumps(x, sort_keys=True, default=fs._json_default)
    assert dump(recs) == dump(again)
    s = fs.summarise(recs, TINY)
    assert s[str(k)]["null"]["n"] == 3 and "power_pass" in s[str(k)]["gates"]
