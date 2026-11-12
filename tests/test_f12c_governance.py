"""F12c: Board A F12b review conditions 2-7 (PREREG_H1_TF A5 note F12c).

Synthetic tensors, toy noise and fabricated replicate records only: no outcome, no data/evals, no raw
noise payload (the restricted-copy check reads only the pinned restricted copy and skips when absent)."""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import f12_simulation as fs
from tests.test_f12_simulation import small_matrix, toy_payload

torch.set_num_threads(1)

S1, S1_SHA = fs.load_config(fs.F12C_CONFIGS["stage1"])
S2, S2_SHA = fs.load_config(fs.F12C_CONFIGS["stage2"])
TG, TG_SHA = fs.load_config(fs.F12C_CONFIGS["target"])
PI, PI_SHA = fs.load_config(fs.F12C_CONFIGS["pilot"])
TINY_C = dataclasses.replace(
    S2, seeds=(0,), max_epochs=4, patience=2, tree_grid=(5,), draws=200, cal_draws=200,
    val_draws=200, schedule=(("null", 1), ("alt", 1), ("null_var2", 1)),
)  # fmt: skip
PINNED_RESTRICTED = json.loads(fs.RESTRICTED_PIN.read_text())["restricted_copy"]["sha256"]
IDS = {
    "tokens_matrix_sha256": "a" * 64,
    "tokens_synthetic": False,
    "payload_restricted_sha256": PINNED_RESTRICTED,
    "code_sha256": {
        "scripts/f12_simulation.py": fs.sha256_file(fs.ROOT / "scripts/f12_simulation.py")
    },
}


# ----------------------------------------------------------------------- configs, schedule


def test_f12c_configs_pinned_with_exact_schedules():
    for name, path in fs.F12C_CONFIGS.items():
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == fs.F12C_CONFIG_SHA256[name]
        cfg, _ = fs.load_config(path)
        assert fs.registered_name(cfg) == f"f12c_{name}"
    assert S2.schedule == (("null", 1000), ("alt", 1000), ("null_var2", 200))
    assert S2.k_rule == "selected" and S2.replicate_offset == 0 and S2.seed_tag == "sim"
    assert S1.schedule == (("null", 40), ("alt", 40)) and S1.k_rule == "all"
    assert S1.replicate_offset == 1000 and S1.seed_tag == "f12b-stage1"
    assert TG.schedule == (("null", 200),) and TG.k_rule == "selected"
    assert TG.replicate_offset == 2000 and TG.seed_tag == "target"
    assert PI.schedule == (("alt", 10),) and PI.k_rule == "145" and PI.seed_key == "{tag}:{r}"
    assert PI.replicate_offset == 3000 and PI.seed_tag == "pilot"
    # alpha = 0.05/36 kept; every training / inference setting equal to F12b's stage-2 config
    f12b = fs.load_config(fs.F12B_CONFIGS["stage2"])[0]
    same = {"seeds", "max_epochs", "patience", "lr", "weight_decay", "batch", "tree_grid", "draws",
            "boot_seed", "cal_draws", "val_draws", "cal_tol", "ridge_lambda", "size_limit",
            "power_limit", "mc_alpha", "bonferroni_m", "lgbm", "head_zero_init",
            "target_standardise", "joint_studentiser", "calibration", "cal_bisect_tol",
            "cal_halvings", "cal_monotone", "noise_draw", "gating", "statnn_select_seeds"}  # fmt: skip
    for cfg in (S1, S2, TG, PI):
        assert {k for k in same if getattr(cfg, k) != getattr(f12b, k)} == set()
    assert S2.bonferroni_m == 36 and S2.mc_alpha == 0.05
    assert S2.cov_se_mult == 2.0 and S2.cov_error_limit == 0.07
    # F12b configs are superseded: not registered for governed execution
    assert fs.registered_name(f12b) == "f12b_stage2" and not fs.governed_name(f12b)
    assert fs.governed_name(S2) == "f12c_stage2"


def test_planned_ids_are_distinct_with_disjoint_seed_streams():
    p2 = fs.planned_ids(S2, [87])
    assert len(p2) == 2200 and len(set(p2)) == 2200
    assert sum(s == "null_var2" for s, _, _ in p2) == 200
    p1 = fs.planned_ids(S1, fs.K_GRID)
    assert len(p1) == 480 and {r for _, _, r in p1} == set(range(1000, 1040))
    pt = fs.planned_ids(TG, [87])
    pp = fs.planned_ids(PI, [145])
    assert len(pt) == 200 and len(pp) == 10
    seeds = {}
    for cfg, ids in ((S1, p1), (S2, p2), (TG, pt), (PI, pp)):
        for s, k, r in ids:
            seeds.setdefault(fs.replicate_seed(cfg, s, k, r), []).append((cfg.name, s, k, r))
    assert len(seeds) == 480 + 2200 + 200 + 10  # every planned panel has its own stream
    assert fs.replicate_seed(PI, "alt", 145, 3000) == fs.domain_seed(fs.FOLD_DOMAIN, "pilot:3000")
    assert fs.replicate_seed(S2, "null", 87, 5) == fs.domain_seed(fs.FOLD_DOMAIN, "sim:null:87:5")
    # h* streams (keyed by r) of stage 1, stage 2, target and pilot are disjoint
    rs = [{r for _, _, r in x} for x in (p1, p2, pt, pp)]
    assert all(not (a & b) for i, a in enumerate(rs) for b in rs[i + 1 :])


# ------------------------------------------------------------- condition 2: studentiser


def test_joint_variance_rejects_nonfinite_terms_and_nonpositive_vcomp():
    parts = {
        # draw: 0 ok-raw, 1 floor binds, 2 v_item nan, 3 v_both inf, 4 v_comp <= 0 with raw > 0,
        #       5 v_comp nan, 6 raw <= 0 (floor binds), 7 v_comp = 0 and raw <= 0
        "v_comp": np.array([1.0, 1.0, 1.0, 1.0, -0.5, np.nan, 2.0, 0.0]),
        "v_item": np.array([0.5, 0.1, np.nan, 1.0, 3.0, 1.0, 0.5, 0.0]),
        "v_both": np.array([0.2, 0.3, 0.1, np.inf, 0.1, 0.1, 3.0, 1.0]),
    }
    var, info = fs.joint_variance(parts, "floor_vcomp")
    np.testing.assert_allclose(var[[0, 1, 6]], [1.3, 1.0, 2.0])
    assert np.all(np.isnan(var[[2, 3, 4, 5, 7]]))  # rejected, never replaced by V_comp
    assert info["raw_nonfinite"] == 3  # draws 2, 3, 5
    assert info["vcomp_invalid"] == 3  # draws 4, 5, 7
    assert info["raw_nonpositive"] == 2  # finite raw <= 0: draws 6, 7
    assert info["floor_binding"] == 2  # draws 1 and 6 (valid draws only)
    var1, info1 = fs.joint_variance(parts, "floor_one_way")
    np.testing.assert_allclose(var1[[0, 1, 6]], [1.3, 1.0, 2.0])
    assert np.all(np.isnan(var1[[2, 3, 4, 5, 7]])) and info1["floor_binding"] == 2
    raw, info0 = fs.joint_variance(parts, "strict")
    assert np.all(np.isnan(raw[[2, 3, 5, 6, 7]])) and raw[4] == pytest.approx(2.4)
    assert info0["floor_binding"] == 0 and info0["raw_nonfinite"] == 3


def _contrast_inputs(seed, n_comp=25):
    rng = np.random.default_rng(seed)
    comp = np.repeat(np.arange(n_comp), 2)
    n = len(comp)
    D = rng.normal(size=(n, 1)) * 0.1 + rng.normal(size=(n, 32)) * 0.01
    y = D.mean(1)
    return D, y + rng.normal(size=n) * 0.01, y + rng.normal(size=n) * 0.03, comp


def test_a542_contrast_enforces_vcomp_at_point_and_every_draw(monkeypatch):
    D, qa, qb, comp = _contrast_inputs(0)
    mults = fs.a542_mults(25, dataclasses.replace(S2, draws=300))
    real = fs.joint_parts
    good = fs.a542_contrast(D, qa, qb, comp, mults, rule="floor_vcomp")
    assert good["valid"]

    def corrupt_draw(D, qa, qb, comp, mult):
        out = real(D, qa, qb, comp, mult)
        if len(mult[0]) > 1:  # a joint draw (not the point): V_comp <= 0 but V_item large
            out["v_comp"][7] = -1e-9
            out["v_item"][7] = 1e3
        return out

    monkeypatch.setattr(fs, "joint_parts", corrupt_draw)
    bad = fs.a542_contrast(D, qa, qb, comp, mults, rule="floor_vcomp")
    assert not bad["valid"] and "joint draws invalid" in bad["failures"]
    assert bad["failure_reasons"]["joint"]["v_comp_nonpositive_or_nonfinite"] == 1
    assert bad["studentiser"]["vcomp_invalid"] == 1

    def corrupt_point(D, qa, qb, comp, mult):
        out = real(D, qa, qb, comp, mult)
        if len(mult[0]) == 1:
            out["v_comp"][0] = np.nan
        return out

    monkeypatch.setattr(fs, "joint_parts", corrupt_point)
    bad = fs.a542_contrast(D, qa, qb, comp, mults, rule="floor_vcomp")
    assert not bad["valid"] and "joint point V_comp" in bad["failures"]

    def nonfinite_item(D, qa, qb, comp, mult):
        out = real(D, qa, qb, comp, mult)
        if len(mult[0]) > 1:
            out["v_both"][3] = np.inf
        return out

    monkeypatch.setattr(fs, "joint_parts", nonfinite_item)
    bad = fs.a542_contrast(D, qa, qb, comp, mults, rule="floor_vcomp")
    assert not bad["valid"]  # a non-finite term is rejected, not floored to V_comp
    assert bad["failure_reasons"]["joint"]["two_way_term_nonfinite"] == 1
    assert bad["studentiser"]["raw_nonfinite"] == 1


def _iv(lo, hi=1.0):
    return {f"{p}_{m}": [lo, hi] for p in ("conditional", "joint") for m in ("bootstrap_t", "bca", "percentile")}  # fmt: skip


def _contrast(valid=True, lo=-0.1, hi=0.3, C=0.1, st=None, failures=None, reasons=None):
    c = {"valid": valid, "C": C, "intervals": _iv(lo, hi) if valid else {}}
    if st is not None:
        c["studentiser"] = st
    if not valid:
        c["failures"] = failures or ["joint draws invalid"]
        c["failure_reasons"] = reasons or {
            "conditional": {"loss_guard": 0},
            "joint": {"v_comp_nonpositive_or_nonfinite": 2, "loss_guard": 0},
        }
    return c


def _st(binding, raw_np=10, raw_nf=0, vci=0, point=False, strict_valid=False, ow_lo=-0.1):
    return {
        "rule": "floor_vcomp", "draws": 4000, "floor_binding": binding, "raw_nonpositive": raw_np,
        "raw_nonfinite": raw_nf, "vcomp_invalid": vci, "point_floor_binding": point,
        "point_raw_nonpositive": False,
        "sensitivity": {
            "strict": {"valid": strict_valid, "joint_bootstrap_t": [-0.2, 0.2] if strict_valid else None,
                       "floor_binding": 0, "raw_nonpositive": raw_np, "raw_nonfinite": raw_nf,
                       "point_floor_binding": False},
            "floor_one_way": {"valid": True, "joint_bootstrap_t": [ow_lo, 0.2], "floor_binding": 2 * binding,
                              "raw_nonpositive": raw_np, "raw_nonfinite": raw_nf, "point_floor_binding": True},
        },
    }  # fmt: skip


def rec(
    cfg, scen, k, r, decision="unresolved", lo=-0.1, hi=0.3, valid=True, C=0.1, st=None, A=None
):
    out = {"k": k, "scenario": scen, "r": r, "seed": fs.replicate_seed(cfg, scen, k, r), "valid": valid,
           "decision": decision, "C_WM": _contrast(valid, lo, hi, C, st), "C_WM_S": _contrast(valid, lo, hi, C, st),
           "A": A or {"F3": 1.0, "WM": 0.9, "S": 0.95, "oracle": 0.85}}  # fmt: skip
    if not valid:
        out["decision"] = "unresolved_validity"
    return out


def test_studentiser_summary_by_contrast_planned_denominator():
    k = 87
    recs = [rec(S2, "null", k, 0, st=_st(400, point=True, strict_valid=True, ow_lo=0.01)),
            rec(S2, "null", k, 1, st=_st(800)),
            rec(S2, "null", k, 2, valid=False)]  # fmt: skip
    recs.append({"k": k, "scenario": "null", "r": 3, "seed": 0, "valid": False,
                 "decision": "unresolved_validity", "failure": "fit: residual target SD 0 <= 1e-12"})  # fmt: skip
    s = fs.studentiser_summary(recs, S2)
    cw = s["C_WM"]
    assert cw["planned"] == 4
    assert cw["amended_valid"]["count"] == 2 and cw["amended_valid"]["rate"] == 0.5
    assert cw["strict_valid"]["count"] == 1 and cw["strict_valid"]["rate"] == 0.25
    assert cw["one_way_max_valid"]["count"] == 2
    assert cw["raw_nonpositive_draws"]["total"] == 20 and cw["raw_nonfinite_draws"]["total"] == 0
    assert cw["floor_binding_fraction"]["n"] == 2
    assert cw["floor_binding_fraction"]["min"] == pytest.approx(0.1)
    assert cw["floor_binding_fraction"]["max"] == pytest.approx(0.2)
    assert cw["point_binding"]["count"] == 1 and cw["point_binding"]["n"] == 4
    assert cw["sensitivity_joint_lower_positive"]["strict"] == 0
    assert cw["sensitivity_joint_lower_positive"]["floor_one_way"] == 1
    assert cw["sensitivity_floor_binding_fraction"]["floor_one_way"]["max"] == pytest.approx(0.4)
    g = cw["failed_guards_by_cause"]
    assert g["joint: v_comp_nonpositive_or_nonfinite"] == 2
    assert g["replicate: fit: residual target SD 0 <= 1e-12"] == 1
    assert g["contrast: joint draws invalid"] == 1


# ------------------------------------------- conditions 3, 4: target, coverage, wording


def test_expected_row_sum_target_with_mc_se_and_fail_closed():
    planned = fs.planned_ids(TG, [87])
    rng = np.random.default_rng(0)
    recs = []
    for s, k, r in planned:
        f3 = 1.0 + rng.random()
        recs.append(
            rec(
                TG,
                s,
                k,
                r,
                A={"F3": f3, "WM": 0.97 * f3 + 0.01 * rng.normal(), "S": f3, "oracle": f3},
            )
        )
    t = fs.expected_row_sum_target(recs, planned, TG)
    wm = np.array([x["A"]["WM"] for x in recs])
    f3 = np.array([x["A"]["F3"] for x in recs])
    assert t["status"] == "ok" and t["n"] == 200
    assert t["C_WM"]["target"] == pytest.approx(1 - wm.sum() / f3.sum(), rel=1e-12)
    R = wm.sum() / f3.sum()
    se = np.sqrt((((wm - R * f3) / f3.mean()) ** 2).sum() / (200 * 199))
    assert t["C_WM"]["se"] == pytest.approx(se, rel=1e-10)
    assert t["C_WM"]["lo"] == pytest.approx(t["C_WM"]["target"] - 2 * se)
    assert t["C_WM"]["hi"] == pytest.approx(t["C_WM"]["target"] + 2 * se)
    assert t["C_WM_S"]["target"] == pytest.approx(1 - wm.sum() / f3.sum())
    # fail closed: a missing planned replicate, a duplicate, or a fit failure leaves no target
    assert fs.expected_row_sum_target(recs[:-1], planned, TG)["status"].startswith("failed")
    assert fs.expected_row_sum_target(recs + recs[:1], planned, TG)["status"].startswith("failed")
    broken = copy.deepcopy(recs)
    del broken[5]["A"]
    broken[5]["failure"] = "fit: empty inner fold"
    assert fs.expected_row_sum_target(broken, planned, TG)["status"].startswith("failed")


def _target(t=0.02, se=0.005):
    one = {"target": t, "se": se, "lo": t - 2 * se, "hi": t + 2 * se}
    return {"status": "ok", "n": 200, "C_WM": dict(one), "C_WM_S": dict(one)}


def test_coverage_of_six_intervals_at_the_expected_row_sum_target():
    k, n = 87, 1000
    target = _target()
    recs = []
    for r in range(n):
        if r < 40:  # lower bound above the conservative target (target - 2 SE = 0.01): error
            recs.append(rec(S2, "null", k, r, lo=0.015, hi=0.3))
        elif r < 45:  # invalid: an error in both tails
            recs.append(rec(S2, "null", k, r, valid=False))
        elif r < 75:  # upper bound below target + 2 SE = 0.03: upper-tail error
            recs.append(rec(S2, "null", k, r, lo=-0.3, hi=0.025))
        else:
            recs.append(rec(S2, "null", k, r, lo=-0.1, hi=0.2))
    cov = fs.coverage_summary(recs, target, S2)
    cw = cov["C_WM"]
    assert set(cw) == {
        f"{p}_{m}" for p in ("conditional", "joint") for m in ("bootstrap_t", "bca", "percentile")
    }
    jb = cw["joint_bootstrap_t"]
    assert jb["lower_tail_error"]["count"] == 45 and jb["upper_tail_error"]["count"] == 35
    # coverage at the point target 0.02: 40 (lo 0.015 <= 0.02) cover, 30 (hi 0.025 >= 0.02) cover
    assert jb["covers_target"]["count"] == 1000 - 5
    assert jb["covers_target"]["n"] == 1000
    g = cov["gates"]
    assert g["n_bounds"] == 8 and g["alpha_each"] == pytest.approx(0.05 / 36)
    assert g["lower_tail_pass"]  # 45/1000: Bonferroni upper 0.0675 <= 0.07
    assert g["upper_tail_pass"]
    recs[100] = rec(S2, "null", k, 100, lo=0.5, hi=0.6)
    recs[101] = rec(S2, "null", k, 101, lo=0.5, hi=0.6)
    assert not fs.coverage_summary(recs, target, S2)["gates"]["lower_tail_pass"]  # 47/1000
    none = fs.coverage_summary(recs, {"status": "failed: fit failure"}, S2)
    assert none["gates"]["lower_tail_pass"] is False and "target" in none["status"]


def test_summary_reports_ratio_of_sums_stress_label_and_marginal_rates():
    k = 87
    recs = [rec(S2, "null", k, r, lo=-0.1) for r in range(10)]
    recs += [
        rec(S2, "alt", k, r, decision="correction_detected", lo=0.01, C=0.2) for r in range(10)
    ]
    s = fs.summarise(recs, S2, target=_target())[str(k)]
    assert s["null"]["label"] == fs.STRESS_LABEL == "linear-signal / ridge-bias stress scenario"
    assert "stress_rates" in s["null"] and "size_rates" not in s["null"]
    assert set(s["gates"]) >= {"stress_pass", "power_pass", "both_pass", "coverage_lower_pass",
                               "coverage_upper_pass", "qualification_pass"}  # fmt: skip
    learned = s["alt"]["learned_C"]
    assert learned["estimand"].startswith("ratio of expected row sums")
    assert "valid-only" in learned["conditioning"]
    assert not any(key.endswith("_mean") for key in learned)
    md = s["alt"]["marginal_detection"]
    assert md["C_WM"]["count"] == 10 and md["C_WM_S"]["count"] == 10 and md["C_WM"]["n"] == 10
    assert "descriptive" in md["note"]
    assert "coverage" in s["null"] and s["alt"]["coverage"]["status"].startswith("not computed")
    assert fs.FAILURE_WORDING == (
        "The registered two-comparator decision did not meet its power qualification under this "
        "alternative; it does not establish absence of a K+F correction."
    )


def test_upper_bound_decision_code_is_provisional():
    def c(C, lo, hi):
        return {
            "valid": True,
            "C": C,
            "intervals": {f"{p}_bootstrap_t": [lo, hi] for p in ("conditional", "joint")},
        }

    assert fs.f12_decision(c(0.0, -0.1, 0.049), c(-0.3, -0.5, 0.0)) == "none_detectable_provisional"


# ------------------------------------------------------------- conditions 5, 6: selection


def _stage1_records(power=(0, 0, 36, 40, 40, 40), size=(0,) * 6):
    recs = []
    for k, pc, sc in zip(fs.K_GRID, power, size):
        for i, r in enumerate(range(1000, 1040)):
            recs.append(rec(S1, "null", k, r, lo=0.1 if i < sc else -0.1))
            ok = i < pc
            recs.append(rec(S1, "alt", k, r, decision="correction_detected" if ok else "unresolved",
                            lo=0.1 if ok else -0.1, C=0.2 if ok else 0.0))  # fmt: skip
    return recs


def _report(cfg, sha, records, k_grid, ids=IDS, **extra):
    return fs.make_report(cfg=cfg, config_file={"path": "x", "sha256": sha}, records=records,
                          k_grid=k_grid, identities=ids, **extra)  # fmt: skip


def _write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=1, default=fs._json_default))
    return path


def test_stage1_report_verified_and_selection_frozen(tmp_path):
    rep = _report(S1, S1_SHA, _stage1_records(), list(fs.K_GRID))
    p = _write(tmp_path / "stage1.json", rep)
    sel = fs.verify_stage1_report(p, identities=IDS)
    assert sel["k_star"] == 58 and sel["selection"]["rule_met_at"] == 58
    assert sel["stage1_report_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert sel["stage1_config_sha256"] == S1_SHA and sel["synthetic"] is False
    assert sel["identities"] == IDS
    out = tmp_path / "f12c_stage2_selection.json"
    fs.write_selection(sel, out)
    assert fs.verify_selection(out, p, identities=IDS) == sel
    with pytest.raises(ValueError, match="immutable"):
        fs.write_selection({**sel, "k_star": 145}, out)
    tampered = json.loads(out.read_text())
    tampered["k_star"] = 145
    _write(out, tampered)
    with pytest.raises(ValueError, match="selection"):
        fs.verify_selection(out, p, identities=IDS)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["records"].pop(), "planned"),
        (lambda r: r["records"].append(copy.deepcopy(r["records"][0])), "planned"),
        (lambda r: r["records"][3].update(seed=1), "seed"),
        (lambda r: r["records"][3].pop("decision"), "complete"),
        (lambda r: r["summary"]["0"]["alt"]["power"].update(count=40), "summary"),
        (
            lambda r: r.update(config=fs.load_config(fs.F12B_CONFIGS["stage1"])[0].as_dict()),
            "config",
        ),
        (lambda r: r["config_file"].update(sha256="0" * 64), "config"),
        (lambda r: r["identities"].update(tokens_synthetic=True), "synthetic"),
        (lambda r: r["identities"].update(tokens_matrix_sha256="f" * 64), "identit"),
        (
            lambda r: r["identities"]["code_sha256"].update(
                {"scripts/f12_simulation.py": "e" * 64}
            ),
            "identit",
        ),
        (lambda r: r.update(k_grid=[0, 29]), "k grid"),
    ],
)
def test_stage1_verifier_rejects_wrong_provenance_or_counts(tmp_path, mutate, match):
    rep = json.loads(
        json.dumps(
            _report(S1, S1_SHA, _stage1_records(), list(fs.K_GRID)), default=fs._json_default
        )
    )
    mutate(rep)
    p = _write(tmp_path / "stage1.json", rep)
    with pytest.raises(ValueError, match=match):
        fs.verify_stage1_report(p, identities=IDS)


def test_synthetic_stage1_never_selects_a_governed_run(tmp_path):
    ids = {**IDS, "tokens_synthetic": True}
    p = _write(
        tmp_path / "s1.json", _report(S1, S1_SHA, _stage1_records(), list(fs.K_GRID), ids=ids)
    )
    with pytest.raises(ValueError, match="synthetic"):
        fs.verify_stage1_report(p, identities=ids)
    sel = fs.verify_stage1_report(p, identities=ids, allow_synthetic=True)
    assert sel["synthetic"] is True
    with pytest.raises(ValueError, match="governed"):
        fs.write_selection(sel, fs.STAGE2_SELECTION)


# --------------------------------------------------- condition 6: execution validator


def _stage2_records(k, detected=900, stress=10):
    recs = []
    for s, kk, r in fs.planned_ids(S2, [k]):
        if s == "alt":
            ok = r < detected
            recs.append(rec(S2, s, kk, r, decision="correction_detected" if ok else "unresolved",
                            lo=0.1 if ok else -0.1, C=0.2))  # fmt: skip
        else:
            recs.append(rec(S2, s, kk, r, lo=0.05 if r < stress else -0.1))
    return recs


def _governed(tmp_path, k=58):
    """F12d: the bound prerequisites (noise check with code identity, pilot report / timing / acceptance,
    coverage target) and a complete committed C1 with a Board A record in a temporary repository."""
    from tests.test_f12d_prereqs import governed

    return governed(tmp_path, k)


def test_execution_validator_passes_only_a_complete_governed_run(tmp_path):
    report, sel, prereqs, kw = _governed(tmp_path)
    v = fs.validate_execution(report, selection=sel, prereqs=prereqs, **kw)
    assert v["passed"], v["failures"]
    rel = fs.release_decision(report["summary"], S2, v)
    assert rel["field"] == "numerical eligibility pending Board A acceptance"
    assert rel["eligible"] is True and rel["numerical_eligibility"].startswith("eligible: S_58")
    assert "binding" not in rel and "authority" in rel and "Board A" in rel["authority"]
    # gates failing -> not eligible, with the registered failure wording
    bad = json.loads(json.dumps(_report(S2, S2_SHA, _stage2_records(58, detected=800), [58],
                                        selection=sel, target=report["coverage_target"]),
                                default=fs._json_default))  # fmt: skip
    v2 = fs.validate_execution(bad, selection=sel, prereqs=prereqs, **kw)
    rel2 = fs.release_decision(bad["summary"], S2, v2)
    assert v2["passed"] and not rel2["eligible"] and fs.FAILURE_WORDING in rel2["wording"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r, s, p: r["identities"].update(tokens_synthetic=True), "synthetic"),
        (lambda r, s, p: r["records"].pop(), "planned"),
        (lambda r, s, p: r["records"].append(copy.deepcopy(r["records"][-1])), "planned"),
        (
            lambda r, s, p: r.update(
                records=[x for x in r["records"] if x["scenario"] != "null_var2"]
            ),
            "planned",
        ),
        (lambda r, s, p: r.update(k_grid=[87]), "k"),
        (lambda r, s, p: r["config_file"].update(sha256="0" * 64), "config"),
        (lambda r, s, p: r.update(config_registered=None, config=fs.Config().as_dict()), "config"),
        (lambda r, s, p: r["summary"]["58"]["alt"]["power"].update(count=1000), "summary"),
        (lambda r, s, p: p.pop("noise_check"), "noise_check"),
        (lambda r, s, p: p["noise_check"].update(f12a4d_check=False, panels=500), "noise_check"),
        (lambda r, s, p: p["pilot_acceptance"].update(chair_accepted=False), "pilot"),
        (lambda r, s, p: p["noise_check"].update(code_sha256={}), "noise_check"),
        (lambda r, s, p: p["coverage_target"]["records"].pop(), "coverage_target"),
        (lambda r, s, p: r["coverage_target"]["C_WM"].update(target=0.5), "coverage target"),
        (lambda r, s, p: s.update(synthetic=True), "selection"),
        (lambda r, s, p: r["stage1_selection"].update(k_star=145), "selection"),
        (lambda r, s, p: r["identities"].update(payload_restricted_sha256="9" * 64), "identit"),
    ],
)
def test_execution_validator_fails_closed(tmp_path, mutate, match):
    report, sel, prereqs, kw = _governed(tmp_path)
    mutate(report, sel, prereqs)
    v = fs.validate_execution(report, selection=sel, prereqs=prereqs, **kw)
    assert not v["passed"] and any(match in f for f in v["failures"]), v["failures"]
    rel = fs.release_decision(report["summary"], S2, v)
    assert rel["eligible"] is False and rel["numerical_eligibility"].startswith("none")


def test_report_assembly_across_invocations_preserves_provenance(tmp_path):
    report, sel, _, _ = _governed(tmp_path)
    recs = report["records"]
    parts = []
    for scen in ("null", "alt", "null_var2"):
        part = _report(S2, S2_SHA, [x for x in recs if x["scenario"] == scen], [58], selection=sel,
                       target=report["coverage_target"], part=scen)  # fmt: skip
        parts.append(_write(tmp_path / f"part_{scen}.json", part))
    whole = fs.assemble_reports(parts)
    assert whole["records"] == json.loads(json.dumps(recs, default=fs._json_default))
    assert [x["sha256"] for x in whole["parts"]] == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in parts
    ]
    assert whole["summary"] == report["summary"] and whole["part"] is None
    # overlapping planned ids, foreign provenance -> refused
    dup = _write(tmp_path / "dup.json", json.loads(parts[0].read_text()))
    with pytest.raises(ValueError, match="more than once"):
        fs.assemble_reports(parts + [dup])
    other = json.loads(parts[1].read_text())
    other["identities"]["tokens_matrix_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="provenance"):
        fs.assemble_reports([parts[0], _write(tmp_path / "o.json", other), parts[2]])
    other = json.loads(parts[1].read_text())
    other["config_file"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        fs.assemble_reports([parts[0], _write(tmp_path / "o2.json", other), parts[2]])
    # a part's own release field is never eligible
    assert json.loads(parts[0].read_text())["release"]["eligible"] is False


def test_partial_or_synthetic_run_never_eligible():
    recs = _stage2_records(58)
    rep = _report(S2, S2_SHA, recs, [58], ids={**IDS, "tokens_synthetic": True}, target=_target())
    assert rep["release"]["eligible"] is False
    assert rep["release"]["field"] == "numerical eligibility pending Board A acceptance"
    assert rep["status"] == "synthetic dry run, not a gate"


# ------------------------------------------------------------------ CLI conformance


def test_cli_refuses_ungoverned_paths_before_any_compute(tmp_path):
    base = ["run", "--synthetic", "--out", str(tmp_path / "o")]
    reg = ["--config", str(fs.F12C_CONFIGS["stage2"])]
    with pytest.raises(SystemExit, match="restricted"):  # raw payload is no governed fallback
        fs.main(base + reg)
    rp = ["--restricted-payload", str(tmp_path / "missing.json")]
    with pytest.raises(SystemExit, match="k grid"):
        fs.main(base + reg + rp + ["--k-grid", "87"])
    with pytest.raises(SystemExit, match="selection"):
        fs.main(base + reg + rp)
    with pytest.raises(SystemExit, match="workers"):
        fs.main(base + ["--workers", "4"])
    with pytest.raises(SystemExit, match="override"):
        fs.main(base + reg + rp + ["--replicates", "5"])


def test_schedule_run_end_to_end_reduced(tmp_path):
    m = small_matrix()
    pay = toy_payload(n_comp=10, per=3, seed=5)
    k = 10
    tasks = fs.planned_ids(TINY_C, [k])
    recs, _ = fs.simulate(m, pay, TINY_C, tasks=tasks, workers=1)
    assert [(x["scenario"], x["k"], x["r"]) for x in recs] == tasks
    for x in recs:
        assert x["seed"] == fs.replicate_seed(TINY_C, x["scenario"], k, x["r"])
        if x["valid"]:
            sens = x["C_WM"]["studentiser"]["sensitivity"]
            assert set(sens) == {"strict", "floor_one_way"}
            for v in sens.values():
                assert {"valid", "joint_bootstrap_t", "floor_binding", "raw_nonpositive",
                        "raw_nonfinite", "point_floor_binding"} <= set(v)  # fmt: skip
            assert x["C_WM"]["studentiser"]["draws"] == TINY_C.draws
    s = fs.summarise(recs, TINY_C)[str(k)]
    assert set(s) >= {"null", "alt", "null_var2", "gates"}
    assert s["null_var2"]["label"].startswith("variance")


def test_timing_projection_follows_the_f12c_schedule():
    by_k = {"0": {"null": 121.0, "alt": 136.0}, "87": {"null": 190.0, "alt": 208.0},
            "145": {"null": 202.0, "alt": 219.0}}  # fmt: skip
    pr = fs.project_cpu(by_k)
    w = pr["stage2_by_k"]["145"]
    # F12d: unmeasured variance-x2 is projected at max(stress, alternative), not at the stress time
    assert w["cpu_h"] == pytest.approx((1000 * 202 + 1000 * 219 + 200 * 219) / 3600)
    assert w["null_var2_from"] == "max(stress, alternative)"
    assert pr["stage2_worst_cpu_h"] == pytest.approx(129.1111, abs=1e-3)
    assert pr["stage1_cpu_h"] == pytest.approx(23.78, abs=0.01)
    assert pr["coverage_target_worst_cpu_h"] == pytest.approx(200 * 202 / 3600)
    assert pr["pilot_cpu_h"] == pytest.approx(10 * 219 / 3600)
    assert "1000 x 6" not in json.dumps(pr)


def test_rules_text_states_only_the_effective_rules():
    r = fs.rules_text(S2)
    assert set(r) == {"effective"}
    text = json.dumps(r)
    for stale in ("gamma-cal", "geomspace", "+-0.005", "0.05/24", "cyclic", "permutation",
                  "size:", "exact null", "100 calibration"):  # fmt: skip
        assert stale not in text, stale
    eff = r["effective"]
    assert fs.STRESS_LABEL in eff["scenarios"] and "1000" in eff["schedule"]
    assert "max(V_comp + V_item - V_both, V_comp)" in eff["joint_studentiser"]
    assert "provisional" in eff["decision"] and "0.05/36" in eff["gates"]
    assert "seed 0" in eff["StatNN"] and "zero" in eff["WM-R"]
    drafted = fs.rules_text(fs.Config())
    assert set(drafted) == {"drafted_not_registered"}


# ------------------------------------------------------------- restricted payload pin


def _toy_restricted(tmp_path):
    pay = toy_payload(n_comp=4, per=2, seed=3)
    for d in pay.donors:
        d["eligible"] = True  # F12d: the restricted copy carries the original flag
    pay.provenance = {
        "sha256": fs.PAYLOAD_SHA256,
        "source": "toy",
        "fields_read": list(fs.PAYLOAD_FIELDS),
    }
    pay.window_ids = [str(i) for i in range(64)]
    path = tmp_path / "noise_payload_restricted.json"
    path.write_bytes(json.dumps(pay.restricted(), sort_keys=True).encode())
    pin = fs.restricted_pin(
        path,
        extractor={"script": "toy", "commit": "0" * 40, "sha256": "1" * 64, "functions": "toy"},
    )
    return path, pin


def test_restricted_loader_verifies_pinned_bytes_contents_and_extractor(tmp_path):
    path, pin = _toy_restricted(tmp_path)
    back = fs.NoisePayload.load_restricted(path, pin)
    assert back.provenance["restricted_copy_sha256"] == pin["restricted_copy"]["sha256"]
    assert pin["contents"]["n_donors"] == 8 and pin["contents"]["n_components"] == 4
    raw = path.read_bytes()
    obj = json.loads(raw)
    obj["centred_noise"][0][0] += 1e-6  # any other bytes fail the digest
    path.write_bytes(json.dumps(obj, sort_keys=True).encode())
    with pytest.raises(ValueError, match="digest"):
        fs.NoisePayload.load_restricted(path, pin)
    path.write_bytes(raw)
    wrong = copy.deepcopy(pin)
    wrong["contents"]["donor_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="donor"):
        fs.NoisePayload.load_restricted(path, wrong)
    # bytes that the pinned extractor would not have written (re-serialisation differs) fail
    indented = json.dumps(json.loads(raw), sort_keys=True, indent=1).encode()
    path.write_bytes(indented)
    pin2 = fs.restricted_pin(path, extractor=pin["extractor"])
    with pytest.raises(ValueError, match="extractor"):
        fs.NoisePayload.load_restricted(path, pin2)
    with pytest.raises(TypeError):
        fs.NoisePayload.load_restricted(path)  # the pin is mandatory


def test_committed_restricted_pin_matches_the_restricted_copy():
    pin = json.loads(fs.RESTRICTED_PIN.read_text())
    assert pin["status"] == "draft, pending Board A C1 verification"
    assert (
        pin["schema"] == fs.RESTRICTED_PIN_SCHEMA
    )  # F12d: the v2 pin (eligible flag); v1 superseded
    assert pin["source"]["sha256"] == fs.PAYLOAD_SHA256
    assert pin["contents"]["n_donors"] == 141 and pin["contents"]["n_components"] == 63
    assert pin["contents"]["windows"] == 64
    path = Path(pin["restricted_copy"]["path"])
    if not path.exists():
        pytest.skip("restricted copy absent")
    pay = fs.NoisePayload.load_restricted(path, pin)
    assert len(pay.donors) == 141 and len(pay.members) == 63
    diag = fs.noise_diagnostics(pay, np.arange(30), seed=0, draws=300)
    assert diag["panel_mean_sd_median"] > 1e-4
    assert diag["max_abs_mean_without_common_resampling"] < 1e-12


# ----------------------------------------------------- condition 7: F12a.4(d) noise check


def _check_inputs():
    comp = np.repeat(np.arange(12), [1, 2, 3, 4, 1, 1, 2, 5, 1, 1, 2, 3])  # rows > donors in some
    ids = np.array([f"r{i:03d}" for i in range(len(comp))])
    return comp, ids


def test_noise_check_matches_plug_ins_within_four_se():
    pay = toy_payload(n_comp=7, per=3, common=1.0, seed=2)
    comp, ids = _check_inputs()
    out = fs.noise_check(pay, comp, ids, k=145, panels=3000)
    assert out["schema"] == fs.NOISE_CHECK_SCHEMA and out["status"] == "pass", out["failures"]
    assert out["f12a4d_check"] is False  # 3,000 panels: a smoke, never the registered check
    for key in ("panel_mean_variance", "within_component_covariance", "cross_component_covariance"):
        q = out["checks"][key]
        assert q["mc_se"] > 0 and abs(q["empirical"] - q["plug_in"]) <= 4 * q["mc_se"]
    assert out["checks"]["cross_component_covariance"]["plug_in"] > 0  # common-item dependence
    assert out["checks"]["within_component_covariance"]["plug_in"] > 0
    assert out["donor_plug_in_variance_min"] > 0
    assert out["seeds"] == "seed('20261111:f12:', 'noisecheck:<d>'), d = 0..2999"
    # panel d is drawn from its own noisecheck:<d> stream in the F12a.4(b) order
    rng = np.random.default_rng(fs.domain_seed(fs.FOLD_DOMAIN, "noisecheck:7"))
    ref = fs.draw_panel_noise(pay, comp, rng, order="f12a", row_ids=ids)[0].mean(1)
    np.testing.assert_array_equal(fs.noise_check_panel(pay, comp, ids, 7), ref)


def test_noise_check_fails_closed():
    comp, ids = _check_inputs()
    pay = toy_payload(n_comp=7, per=3, seed=2)
    zero = pay.noise.copy()
    zero[4] = 0.0  # a donor whose plug-in panel-mean variance is 0
    pz = fs.NoisePayload(zero, pay.mask, pay.donors)
    out = fs.noise_check(pz, comp, ids, k=145, panels=400)
    assert out["status"] == "fail" and any("zero plug-in" in f for f in out["failures"])

    def independent_items(payload, comp_codes, row_ids, rng):  # breaks the common block index
        _, donors, _ = fs.draw_panel_noise(payload, comp_codes, rng, order="f12a", row_ids=row_ids)
        b = rng.integers(0, 32, size=(len(comp_codes), 32))
        return payload.blocks[donors[:, None], b].mean(1)

    out = fs.noise_check(pay, comp, ids, k=145, panels=3000, draw=independent_items)
    assert out["status"] == "fail"
    assert any("cross_component_covariance" in f for f in out["failures"])


# ------------------------------------------------------------------------ C1 draft pin


def test_code_hashes_c1_draft_pins_current_bytes():
    obj = json.loads(fs.CODE_HASHES_C1.read_text())
    assert obj["schema"] == fs.C1_SCHEMA
    # the draft pin until the C1 freeze; the governed item 11 (f12_c1.build_code_hashes) once MANIFEST_C1.sha256
    # exists. This check also catches `f12_simulation.py c1-draft` overwriting the governed item 11 after the freeze.
    from scripts import f12_c1 as c1

    assert c1.code_pin_failures(fs.ROOT) == []
    root = fs.ROOT
    listed = set(obj["files"])
    for pat in (
        "scripts/f12*.py",
        "tests/test_f12*.py",
        "docs/prereg/f12_manifests/f12*_config.json",
    ):
        for p in root.glob(pat):
            assert str(p.relative_to(root)) in listed, p
    for dep in ("scripts/a5_build_panels.py", "scripts/a5_simulation.py", "scripts/tfv1_requalify.py",
                "docs/prereg/f12_manifests/noise_payload_restricted_pin.json"):  # fmt: skip
        assert dep in listed
    assert fs.verify_code_hashes(obj) == []
    pin = json.loads(fs.RESTRICTED_PIN.read_text())
    assert obj["restricted_payload"]["sha256"] == pin["restricted_copy"]["sha256"]
    assert set(obj["versions"]) >= {"python", "numpy", "torch", "lightgbm", "scipy"}
