"""F12b repairs to the F12 simulation (PREREG_H1_TF A5 amendment F12b; docs/design/F12B_REPAIRS.md).

Synthetic tensors and toy noise only: no outcome, no data/evals, no Part A payload."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import f12_simulation as fs
from scripts.tfv1_requalify import joint_replicates
from tests.test_f12_simulation import TINY, small_matrix, toy_payload

# One intra-op thread, as in every simulation worker (_init_worker): multi-threaded accumulation of the
# role-embedding gradient is not bitwise reproducible.
torch.set_num_threads(1)

F12B = fs.load_config(fs.F12B_CONFIGS["stage2"])[0]
TINY_B = dataclasses.replace(
    F12B, seeds=(0,), max_epochs=4, patience=2, tree_grid=(5,), draws=200, cal_draws=200,
    val_draws=200,
)  # fmt: skip


# ------------------------------------------------------------------------------- config


def test_registered_configs_are_committed_and_pinned():
    for stage in ("stage1", "stage2"):
        path = fs.F12B_CONFIGS[stage]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == fs.F12B_CONFIG_SHA256[stage]
        cfg, sha = fs.load_config(path)
        assert sha == fs.F12B_CONFIG_SHA256[stage]
        assert fs.registered_name(cfg) == f"f12b_{stage}"
    s1 = fs.load_config(fs.F12B_CONFIGS["stage1"])[0]
    s2 = F12B
    # the registered (stage-2) run keeps every F12/F12a setting not repaired by F12b
    assert s2.draws == 4000 and s2.boot_seed == 20261107 and s2.seeds == (0, 1, 2)
    assert s2.tree_grid == (100, 300, 600) and s2.max_epochs == 300 and s2.patience == 30
    assert (s2.lr, s2.weight_decay, s2.batch, s2.ridge_lambda) == (1e-3, 1e-2, 16, 1.0)
    assert (s2.size_limit, s2.power_limit, s2.mc_alpha, s2.bonferroni_m) == (0.07, 0.80, 0.05, 36)
    assert s2.head_zero_init and s2.target_standardise
    assert s2.joint_studentiser == "floor_vcomp" and s2.calibration == "f12b"
    assert s2.cal_tol == 0.02 and s2.noise_draw == "f12a" and s2.gating == "f12a"
    assert s2.cal_monotone == "convex" and s2.cal_halvings == 50
    assert s2.cal_draws == 4000 and s2.val_draws == 4000 and s2.cal_bisect_tol == 0.001
    assert s2.replicates == 1000 and s2.seed_tag == "sim"
    # the pilot differs only in draws, replicates, seed tag and stage
    diff = {f.name for f in dataclasses.fields(fs.Config)} - {"name"}
    changed = {k for k in diff if getattr(s1, k) != getattr(s2, k)}
    assert changed == {"replicates", "seed_tag", "stage", "replicate_offset"}
    assert s1.draws == 4000 and s1.stage == 1 and s2.stage == 2 and s1.replicate_offset == 1000
    # F12b.5 cheaper fits: one-thread LightGBM (identical trees), single-seed tree-count selection
    assert dict(s2.lgbm)["n_jobs"] == 1 and s2.statnn_select_seeds == (0,)
    # the drafted defaults are unchanged and are not a registered F12b config
    assert fs.registered_name(fs.Config()) is None and fs.Config().bonferroni_m == 24
    with pytest.raises(ValueError):
        fs.config_from_dict({**s2.as_dict(), "joint_studentiser": "clamp"})


# --------------------------------------------------------------------------- head repair


def test_zero_initialised_head_is_f3_until_trained_and_zero_at_zero_edit():
    m = small_matrix()
    data = fs.HeadData(m)
    rows = np.arange(data.n)
    st = data.stats(rows)
    p = fs._seeded(0, zero_out=True)
    assert torch.all(p["w"] == 0) and torch.all(p["b0"] == 0)
    ref = fs.init_params(0)
    assert torch.equal(p["W1"], ref["W1"]) and torch.equal(p["E"], ref["E"])  # rest PyTorch default
    # zero residual target: L1 subgradient 0 at 0, weight decay keeps w = 0, so h == 0 after training
    _, P = fs.train_heads(data, [p], [st], [np.zeros(data.n)], [rows], TINY, epochs=[3])
    assert np.all(fs.predict_heads(data, P, [st], rows) == 0.0)
    # a trained zero-initialised head still predicts exactly 0 at the zero edit
    y = np.random.default_rng(0).normal(size=data.n)
    _, P = fs.train_heads(data, [fs._seeded(0, zero_out=True)], [st], [y], [rows], TINY, epochs=[3])
    assert torch.any(P["w"] != 0)
    zero_rows = [{"id": "z", "component": "zc", "family": "chain", "origin": "existing"}]
    zd = fs.HeadData(
        fs.assemble_matrix(zero_rows, [([], np.zeros((0, fs.N_COLS)))], synthetic=True)
    )
    assert fs.predict_heads(zd, P, [st], [0])[0, 0] == 0.0


def test_target_standardisation_is_training_only_and_inverted():
    m = small_matrix()
    data = fs.HeadData(m)
    a, q, comps = m["sum_a"], m["sum_q"], m["component"]
    n = data.n
    y = a + np.random.default_rng(1).normal(size=n) * 1e-3
    test = np.arange(n - 8, n)
    train = np.arange(n - 8)
    cfg = dataclasses.replace(TINY_B, max_epochs=3, patience=2)
    p1, i1 = fs.fit_wmr_fold(data, a, q, y, comps, train, test, cfg)
    # scale equivariance: residual / training SD is identical for 2^10 y, so h scales back exactly
    p2, i2 = fs.fit_wmr_fold(data, a, q, 1024.0 * y, comps, train, test, cfg)
    np.testing.assert_allclose(p2, 1024.0 * p1, rtol=1e-9, atol=0)
    assert i2["target_sd"] == pytest.approx(1024.0 * i1["target_sd"], rel=1e-12)
    ref = np.std((y - fs.f3_operator(a, q, train, np.arange(n)) @ y[train])[train])
    assert i1["target_sd"] == pytest.approx(ref, rel=1e-12)
    # held-out rows never enter the fit: perturbing their outcome changes nothing
    y3 = y.copy()
    y3[test] += 5.0
    p3, _ = fs.fit_wmr_fold(data, a, q, y3, comps, train, test, cfg)
    np.testing.assert_array_equal(p3, p1)
    # without the repair the old behaviour is kept (no scaling recorded)
    _, i0 = fs.fit_wmr_fold(
        data, a, q, y, comps, train, test, dataclasses.replace(TINY, max_epochs=3)
    )
    assert "target_sd" not in i0
    # a degenerate residual fails closed
    with pytest.raises(ValueError, match="target SD"):
        fs.target_scale(np.zeros(n), train)


# ------------------------------------------------------------------- joint studentiser


def _contrast_inputs(seed, n_comp=25, scale_b=1.0, close=None):
    """close: qb = qa + N(0, close^2) (near-identical predictors, as F3 vs an untrained WM-R)."""
    rng = np.random.default_rng(seed)
    comp = np.repeat(np.arange(n_comp), 2)
    n = len(comp)
    D = rng.normal(size=(n, 1)) * 0.1 + rng.normal(size=(n, 32)) * 0.01
    y = D.mean(1)
    qa = y + rng.normal(size=n) * 0.01
    qb = y + rng.normal(size=n) * 0.01 * scale_b
    if close is not None:
        qb = qa + rng.normal(size=n) * close
    return D, qa, qb, comp


def test_joint_parts_reproduce_the_a542_two_way_studentiser():
    D, qa, qb, comp = _contrast_inputs(0)
    mults = fs.a542_mults(25, TINY)
    parts = fs.joint_parts(D, qa, qb, comp, mults[1])
    pop = {"D": D, "qa": qa, "qb": qb}
    ln, se = joint_replicates(pop, np.arange(len(comp)), comp, 25, np.ones(32), 1.0, None, 0,
                              mult=mults[1])  # fmt: skip
    var = parts["v_comp"] + parts["v_item"] - parts["v_both"]
    np.testing.assert_allclose(parts["ln"], ln, rtol=1e-12)
    ok = var > 0
    np.testing.assert_allclose(np.sqrt(var[ok]), se[ok], rtol=1e-10)
    assert np.all(np.isnan(se[~ok]))
    # the point decomposition: V_comp equals the conditional (component) linearised variance
    pt = fs.joint_parts(D, qa, qb, comp, (np.ones((1, 25)), np.ones((1, 32))))
    ca = np.bincount(comp, np.abs(D.mean(1) - qa), 25)
    cb = np.bincount(comp, np.abs(D.mean(1) - qb), 25)
    from scripts.tfv1_requalify import linearised_se

    assert pt["v_comp"][0] == pytest.approx(float(linearised_se(ca, cb)) ** 2, rel=1e-10)


def test_floor_rule_is_max_of_two_way_and_one_way_terms():
    parts = {
        "v_comp": np.array([1.0, 1.0, 2.0, 1.0]),
        "v_item": np.array([0.5, 3.0, 0.5, 0.0]),
        "v_both": np.array([0.2, 5.0, 3.0, 2.0]),
    }
    counts = {"raw_nonpositive": 3, "raw_nonfinite": 0, "vcomp_invalid": 0, "floor_binding": 3}
    var, info = fs.joint_variance(parts, "floor_one_way")
    np.testing.assert_allclose(var, [1.3, 3.0, 2.0, 1.0])
    assert info == counts
    var, info = fs.joint_variance(parts, "floor_vcomp")
    np.testing.assert_allclose(var, [1.3, 1.0, 2.0, 1.0])
    assert info == counts
    raw, info = fs.joint_variance(parts, "strict")  # F12c.2: invalid draws are NaN, never values
    np.testing.assert_allclose(raw, [1.3, np.nan, np.nan, np.nan])
    assert info["raw_nonpositive"] == 3 and info["floor_binding"] == 0
    with pytest.raises(ValueError):
        fs.joint_variance(parts, "drop")


def test_a542_contrast_floor_rule_resolves_nonpositive_draws_only():
    # find toy inputs whose raw two-way variance is non-positive in some draw
    mults = fs.a542_mults(25, dataclasses.replace(TINY, draws=2000))
    found = None
    for seed in range(60):
        D, qa, qb, comp = _contrast_inputs(seed, close=1e-4)
        parts = fs.joint_parts(D, qa, qb, comp, mults[1])
        if np.any(parts["v_comp"] + parts["v_item"] - parts["v_both"] <= 0):
            found = (D, qa, qb, comp)
            break
    assert found is not None, "no toy input with a non-positive raw joint studentiser"
    D, qa, qb, comp = found
    strict = fs.a542_contrast(D, qa, qb, comp, mults, rule="strict")
    floor = fs.a542_contrast(D, qa, qb, comp, mults, rule="floor_one_way")
    assert not strict["valid"] and "joint draws invalid" in strict["failures"]
    assert floor["valid"] and floor["C"] == strict["C"]
    st = floor["studentiser"]
    assert st["rule"] == "floor_one_way" and st["raw_nonpositive"] >= 1
    assert st["floor_binding"] >= st["raw_nonpositive"]
    # the conditional procedure is unchanged by the joint rule
    assert floor["intervals"]["conditional_bootstrap_t"] == strict.get("intervals", {}).get(
        "conditional_bootstrap_t", floor["intervals"]["conditional_bootstrap_t"]
    )
    ok = fs.a542_contrast(*_contrast_inputs(1, scale_b=3.0), mults, rule="strict")
    fl = fs.a542_contrast(*_contrast_inputs(1, scale_b=3.0), mults, rule="floor_one_way")
    assert ok["valid"] and fl["valid"]
    assert fl["intervals"]["conditional_bootstrap_t"] == ok["intervals"]["conditional_bootstrap_t"]


# --------------------------------------------------------------------- noise draw order


def reference_f12a_noise(payload, comp_codes, row_ids, rng):
    """F12a.4(b) literally: (i) one integers(0, n_payload_comp) per simulated component in order;
    (ii) per simulated component, rows by ascending id, choice(n_c, m, replace=m > n_c) over the
    payload component's donors sorted by id; (iii) one integers(0, 32, size=32)."""
    comps = sorted(set(comp_codes.tolist()))
    pcomp = sorted({d["component"] for d in payload.donors})
    pick = [int(rng.integers(0, len(pcomp))) for _ in comps]
    donors = np.zeros(len(comp_codes), dtype=np.int64)
    for c, p in zip(comps, pick):
        members = sorted(
            (j for j, d in enumerate(payload.donors) if d["component"] == pcomp[p]),
            key=lambda j: payload.donors[j]["id"],
        )
        rows = sorted(np.flatnonzero(comp_codes == c), key=lambda i: row_ids[i])
        sel = rng.choice(len(members), size=len(rows), replace=len(rows) > len(members))
        donors[rows] = np.asarray(members)[sel]
    b = rng.integers(0, 32, size=32)
    return payload.blocks[donors][:, b], donors, b


def test_noise_draw_order_follows_f12a():
    pay = toy_payload(n_comp=7, per=3, seed=2)
    # shuffle donor order so sorting by id matters
    perm = np.random.default_rng(0).permutation(len(pay.donors))
    pay = fs.NoisePayload(pay.noise[perm], pay.mask[perm], [pay.donors[j] for j in perm])
    comp = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4, 4, 5])  # one component with more rows than donors
    row_ids = np.array([f"r{i:02d}" for i in (5, 1, 9, 3, 8, 0, 2, 7, 4, 6, 10)])
    for s in range(5):
        got = fs.draw_panel_noise(
            pay, comp, np.random.default_rng(s), order="f12a", row_ids=row_ids
        )
        ref = reference_f12a_noise(pay, comp, row_ids, np.random.default_rng(s))
        np.testing.assert_array_equal(got[1], ref[1])
        np.testing.assert_array_equal(got[2], ref[2])
        np.testing.assert_array_equal(got[0], ref[0])
    with pytest.raises(ValueError):
        fs.draw_panel_noise(pay, comp, np.random.default_rng(0), order="f12a")  # row ids required


# --------------------------------------------------------------------------- calibration


def _cal_problem(n=200, seed=3):
    rng = np.random.default_rng(seed)
    comp = np.arange(n)
    a = rng.normal(size=n) * 0.05
    q = np.abs(a) * 0.02
    h = rng.normal(size=n)
    h /= h.std()
    P = fs.oof_operator(a, q, comp % 5)
    pay = toy_payload(n_comp=30, per=4, common=0.3, seed=4)
    return P, a, h, pay, comp


def test_noise_tau_is_rms_of_plug_in_panel_mean_sds():
    pay = toy_payload(n_comp=5, per=2, seed=1)
    e = pay.blocks  # donor block means
    ref = np.sqrt(np.mean((e**2).sum(1) / 32 / 32))
    assert fs.noise_tau(pay) == pytest.approx(ref, rel=1e-12)


def test_f12b_calibration_grid_bisection_and_validation_tolerance():
    P, a, h, pay, comp = _cal_problem()
    ids = np.array([f"r{i:04d}" for i in range(len(a))])
    tau = fs.noise_tau(pay)
    big = dataclasses.replace(TINY_B, cal_draws=600, val_draws=600)
    eps_c = fs.calibration_panels(pay, comp, ids, big.cal_draws, "cal", 0, 0, big)
    eps_v = fs.calibration_panels(pay, comp, ids, big.val_draws, "calval", 0, 0, big)
    gamma, info = fs.calibrate_gamma_f12b(P, a, h, eps_c, eps_v, tau, big)
    assert info["status"] == "ok" and gamma > 0
    # this problem dips below C(0) before rising: F12a.4(h)'s check would fail it, F12b does not
    assert info["dip_below_C0"] and not info["f12a_grid_nondecreasing"]
    g_f12a, i_f12a = fs.calibrate_gamma_f12b(
        P, a, h, eps_c, eps_v, tau, dataclasses.replace(big, cal_monotone="f12a")
    )
    assert g_f12a is None and i_f12a["status"] == "failed: non-monotone bracket"
    # uniqueness: the curve crosses the target once, at the bisection root
    gg = np.linspace(0, info["grid"][-1], 4001)
    cc = fs.oracle_curve(P, a, h, eps_c)(gg)
    assert np.sum(np.diff(cc >= 0.15) != 0) == 1
    assert info["grid"][0] == 0.0 and info["grid"][1] == pytest.approx(tau * 2.0**-8)
    assert len(info["grid"]) == 22 and info["grid"][-1] == pytest.approx(tau * 2.0**12)
    assert abs(info["C_calibration"] - 0.15) <= big.cal_bisect_tol
    assert info["halvings"] <= 50
    assert abs(info["C_validation"] - 0.15) <= 0.02 and info["C_validation_se"] > 0
    # per-draw seeds: panel d is reproducible on its own
    one = fs.calibration_panels(pay, comp, ids, 1, "cal", 0, 0, big, start=5)
    np.testing.assert_array_equal(one[0], eps_c[5])
    # validation outside +-cal_tol fails qualification (no substitute h*, no retry)
    strict = dataclasses.replace(big, cal_tol=1e-6)
    g2, i2 = fs.calibrate_gamma_f12b(P, a, h, eps_c, eps_v, tau, strict)
    assert g2 is None and i2["status"] == "failed: validation tolerance"
    # no bracket on the grid -> failed qualification
    g3, i3 = fs.calibrate_gamma_f12b(P, a, 0.0 * h, eps_c, eps_v, tau, big)
    assert g3 is None and i3["status"].startswith("failed")


# ------------------------------------------------------------------- gates and selection


def _rec(scen, decision, lo, valid=True, k=0):
    iv = {f"{p}_{m}": [-1.0, 1.0] for p in ("conditional", "joint") for m in ("bootstrap_t", "bca", "percentile")}  # fmt: skip
    iv["conditional_bootstrap_t"] = [lo, 1.0]
    iv["joint_bootstrap_t"] = [lo, 1.0]
    c = {"valid": valid, "C": 0.1, "intervals": iv if valid else {}}
    return {"k": k, "scenario": scen, "valid": valid, "decision": decision, "C_WM": c,
            "C_WM_S": c, "A": {"F3": 1.0, "WM": 0.9, "S": 0.95, "oracle": 0.85}}  # fmt: skip


def test_f12a_gates_six_rates_bonferroni_36_invalid_counted():
    cfg = F12B
    assert fs.mc_bounds(838, 1000, cfg)["bonferroni_lower"] >= 0.80
    assert fs.mc_bounds(837, 1000, cfg)["bonferroni_lower"] < 0.80
    assert fs.mc_bounds(46, 1000, cfg)["bonferroni_upper"] <= 0.07
    assert fs.mc_bounds(47, 1000, cfg)["bonferroni_upper"] > 0.07
    # 40 valid rejections + 6 invalid null replicates = 46 rejections: passes; one more fails
    null = [_rec("null", "unresolved", 0.1)] * 40 + [
        _rec("null", "unresolved_validity", 0, False)
    ] * 6
    null += [_rec("null", "unresolved", -0.1)] * (1000 - 46)
    alt = [_rec("alt", "correction_detected", 0.1)] * 838 + [_rec("alt", "unresolved", -1)] * 162
    s = fs.summarise(null + alt, cfg)["0"]
    assert set(s["null"]["stress_rates"]) == {
        "C_WM_cond_lower", "C_WM_joint_lower", "C_WM_S_cond_lower", "C_WM_S_joint_lower", "detected"
    }  # fmt: skip
    assert s["null"]["stress_rates"]["C_WM_cond_lower"]["count"] == 46
    assert s["null"]["stress_rates"]["detected"]["count"] == 6  # invalid counted as rejection
    assert s["gates"]["stress_pass"] and s["gates"]["power_pass"] and s["gates"]["both_pass"]
    null2 = null[:-1] + [_rec("null", "unresolved_validity", 0, False)]
    s2 = fs.summarise(null2 + alt, cfg)["0"]
    assert (
        s2["null"]["stress_rates"]["C_WM_cond_lower"]["count"] == 47
        and not s2["gates"]["stress_pass"]
    )


def test_stage2_k_selection_rule():
    cfg = fs.load_config(fs.F12B_CONFIGS["stage1"])[0]
    n = cfg.replicates

    def summary(power_counts, size_counts):
        out = {}
        for k, pc, sc in zip(fs.K_GRID, power_counts, size_counts):
            recs = [_rec("alt", "correction_detected", 0.1, k=k)] * pc
            recs += [_rec("alt", "unresolved", -1, k=k)] * (n - pc)
            recs += [_rec("null", "unresolved", 0.1, k=k)] * sc
            recs += [_rec("null", "unresolved", -0.1, k=k)] * (n - sc)
            out.update(fs.summarise(recs, cfg))
        return out

    need = int(np.ceil(cfg.stage1_power_min * n))
    cap = int(np.floor(cfg.stage1_size_max * n + 1e-9))
    s = summary([0, 0, need - 1, need, n, n], [0] * 6)
    sel = fs.select_stage2_k(s, cfg)
    assert sel["k"] == [87] and sel["rule_met_at"] == 87
    s = summary([0, 0, need, need, n, n], [0, 0, cap + 1, cap, 0, 0])
    assert fs.select_stage2_k(s, cfg)["k"] == [87]
    s = summary([0, 0, need, need, n, n], [0, 0, cap + 1, cap + 1, 0, 0])
    assert fs.select_stage2_k(s, cfg)["k"] == [116]
    s = summary([0] * 6, [0] * 6)
    sel = fs.select_stage2_k(s, cfg)
    assert sel["k"] == [145] and sel["rule_met_at"] is None
    with pytest.raises(ValueError):
        fs.select_stage2_k({str(k): v for k, v in list(s.items())[:5]}, cfg)  # all six k required


def test_f12b_configs_are_superseded_and_never_eligible():
    """F12c: the F12b configs remain committed but are not governed; their reports never become
    numerically eligible (the F12c execution validator rejects their config)."""
    cfg = F12B
    recs = [_rec("null", "unresolved", -0.1, k=116)] * 1000
    recs += [_rec("alt", "correction_detected", 0.1, k=116)] * 1000
    s = fs.summarise(recs, cfg)
    assert s["116"]["gates"]["both_pass"]
    assert fs.governed_name(cfg) is None and fs.registered_name(cfg) == "f12b_stage2"
    report = {"config": cfg.as_dict(), "config_file": {"sha256": fs.F12B_CONFIG_SHA256["stage2"]},
              "identities": {"tokens_synthetic": False}, "k_grid": [116], "records": recs, "summary": s}  # fmt: skip
    v = fs.validate_execution(report, selection=None, prereqs={})
    assert not v["passed"] and "config is not the registered F12c stage-2 config" in v["failures"]
    rel = fs.release_decision(s, cfg, v)
    assert rel["eligible"] is False and rel["field"] == fs.RELEASE_FIELD


# ------------------------------------------------------------------------- end to end


def test_f12b_end_to_end_null_and_alt_deterministic():
    m = small_matrix()
    pay = toy_payload(n_comp=10, per=3, seed=5)
    k = 10
    recs, _ = fs.simulate(m, pay, TINY_B, [k], ["null", "alt"], replicates=2, workers=1)
    assert len(recs) == 4
    for r in recs:
        assert r["seed"] == fs.domain_seed(fs.FOLD_DOMAIN, f"sim:{r['scenario']}:{k}:{r['r']}")
    null = [r for r in recs if r["scenario"] == "null"]
    assert all(r["gamma"] == 0.0 for r in null)
    for r in null:
        if r["valid"]:
            st = r["C_WM"]["studentiser"]
            assert st["rule"] == "floor_vcomp"
            # descriptive sensitivity: the other rules on the same draws
            assert set(st["sensitivity"]) == {"strict", "floor_one_way"}
            # F12c.2: each sensitivity record carries its own binding / raw counts
            assert set(st["sensitivity"]["strict"]) >= {
                "valid",
                "joint_bootstrap_t",
                "floor_binding",
            }
            assert all("target_sd" in f for f in r["fit"]["wmr"])
    alt = [r for r in recs if r["scenario"] == "alt"]
    for r in alt:
        assert r["calibration"]["grid"][0] == 0.0
    again, _ = fs.simulate(m, pay, TINY_B, [k], ["null", "alt"], replicates=2, workers=2)
    dump = lambda x: json.dumps(x, sort_keys=True, default=fs._json_default)
    assert dump(recs) == dump(again)
    pilot = dataclasses.replace(TINY_B, seed_tag="f12b-stage1", stage=1, replicate_offset=1000)
    p, _ = fs.simulate(m, pay, pilot, [k], ["null"], replicates=1, workers=1)
    assert p[0]["seed"] == fs.domain_seed(fs.FOLD_DOMAIN, f"f12b-stage1:null:{k}:1000")


# ------------------------------------------------------------------ restricted payload


def test_restricted_payload_round_trip_and_pin(tmp_path):
    pay = toy_payload(n_comp=4, per=2, seed=3)
    for d in pay.donors:
        d["eligible"] = True  # F12d: the restricted copy carries the original eligible flag
    pay.provenance = {"sha256": fs.PAYLOAD_SHA256, "source": "toy"}
    path = tmp_path / "noise_payload_restricted.json"
    path.write_text(json.dumps(pay.restricted(), sort_keys=True))
    pin = fs.restricted_pin(
        path,
        extractor={"script": "toy", "commit": "0" * 40, "sha256": "1" * 64, "functions": "toy"},
    )
    back = fs.NoisePayload.load_restricted(path, pin)
    np.testing.assert_array_equal(back.blocks, pay.blocks)
    assert back.donors == pay.donors and back.provenance["restricted_copy_sha256"]
    obj = json.loads(path.read_text())
    for bad in (dict(obj, donor_mean=[0.0]), dict(obj, schema="x"),
                dict(obj, provenance={"sha256": "0" * 64})):  # fmt: skip
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):  # by digest, and by content against a pin of the bad bytes
            fs.NoisePayload.load_restricted(path, pin)
        with pytest.raises(ValueError):
            fs.NoisePayload.load_restricted(
                path, fs.restricted_pin(path, extractor=pin["extractor"])
            )
