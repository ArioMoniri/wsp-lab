"""CPU tests for scripts/tfv1_requalify.py on synthetic data (no scoring artefacts needed)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load():
    spec = importlib.util.spec_from_file_location("tfv1_requalify", SCRIPTS / "tfv1_requalify.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["tfv1_requalify"] = module
    spec.loader.exec_module(module)
    return module


rq = load()


def test_item_halfwidth_matches_a_hand_bootstrap_and_uses_one_common_draw():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 1, size=(3, 200))
    half, se = rq.item_halfwidths(diffs, 1, seed=5, draws=500)
    index = rq.draw_indices(200, 1, np.random.default_rng(5), 500)
    for i in range(3):
        boot = diffs[i][index].mean(1)
        assert half[i] == pytest.approx((np.quantile(boot, 0.975) - np.quantile(boot, 0.025)) / 2)
        assert se[i] == pytest.approx(boot.std(ddof=1))
    # half-width is ~1.96 se of the mean (sd 1, n 200)
    assert half[0] == pytest.approx(1.96 / math.sqrt(200), rel=0.25)
    # wikitext blocks of two windows keep adjacent windows together
    idx = rq.draw_indices(8, 2, np.random.default_rng(0), 10)
    assert (idx[:, 0::2] + 1 == idx[:, 1::2]).all()


def test_stratum_weights_and_tripwire_arithmetic():
    manifest = {
        "components": [
            {"status": "confirmatory", "family_counts": {"chain": 187, "abliteration": 3}},
            {"status": "confirmatory", "family_counts": {"lora_sft": 55, "task_vector": 25}},
            {"status": "development", "family_counts": {"chain": 999}},
        ]
    }
    w, rows = rq.stratum_weights(manifest)
    assert rows == {"chain": 187, "lora_sft": 55, "task_vector": 25}
    assert w["chain"] == pytest.approx(187 / 267) and sum(w.values()) == pytest.approx(1)
    fam = ["lora_sft"] * 6 + ["task_vector"] * 2 + ["chain"] * 12
    exceed = np.array([True, True] + [False] * 18)
    point = rq.tripwire_P(exceed, fam, w)
    assert point == pytest.approx(w["lora_sft"] * 2 / 6)  # the Board's interim 20-entry arithmetic
    codes = [f"c{i // 2}" for i in range(20)]
    assert rq.tripwire_upper(exceed, fam, codes, w, draws=300) >= point * 0.5


def test_a2_conditions_pass_on_clean_and_fail_on_a_large_halfwidth_entry():
    rng = np.random.default_rng(1)
    fams = ["chain"] * 30 + ["lora_sft"] * 10
    codes = [f"c{i}" for i in range(40)]
    w = {"chain": 0.7, "lora_sft": 0.2, "task_vector": 0.1}
    clean = rng.normal(0, 0.3, size=(40, 400))  # half-width ~ 0.03
    res = rq.a2_item_condition(clean, fams, codes, w, 0.10, 1, draws=400)[0]
    assert res["pass"] and res["n_entries_over_half_delta"] == 0
    noisy = clean.copy()
    noisy[3] = rng.normal(0, 3.0, size=400)  # half-width ~ 0.3 > delta
    res2, *_ = rq.a2_item_condition(noisy, fams, codes, w, 0.10, 1, draws=400)
    assert not res2["ii_ceiling"]["pass"] and not res2["pass"]
    assert res2["iv_disclosure_over_half_delta"][0]["index"] == 3


def test_reliability_refits_the_nuisance_inside_every_draw():
    rng = np.random.default_rng(2)
    n = 60
    fams = (["chain"] * 30) + (["lora_sft"] * 30)
    log_norm = rng.uniform(0, 2, n)
    y = 0.5 * log_norm + rng.normal(0, 1.0, n)  # real signal after the nuisance
    codes = [f"c{i}" for i in range(n)]
    res = rq.reliability_refit(y, fams, log_norm, np.full(n, 0.01), codes, draws=300)
    assert res["reliability"] > 0.95 and res["pass_lower_bound"]
    assert res["design_rank"] == 3
    noise_only = rq.reliability_refit(y, fams, log_norm, np.full(n, 5.0), codes, draws=300)
    assert noise_only["signal_variance_untruncated"] < 0 and not noise_only["pass_lower_bound"]


def test_repeat_bounds_zero_and_nonzero():
    prim = np.zeros((4, 50))
    out = rq.repeat_bounds(prim, prim.copy(), ["a", "b", "c", "d"], 0.10, 1e-4, draws=100)
    assert out["bitwise_identical_items"] and out["pass_max"] and out["pass_p95_upper"]
    rep = prim.copy()
    rep[0, 0] = 1e-3
    out = rq.repeat_bounds(prim, rep, ["a", "b", "c", "d"], 0.10, 1e-4, draws=100)
    assert not out["pass_max"] and out["max_abs_discrepancy"] == pytest.approx(1e-3)


def test_inner_folds_are_hash_ordered_round_robin_and_deterministic():
    comps = [f"comp{i}" for i in range(23)]
    folds = rq.inner_folds(comps)
    assert sorted(set(folds.values())) == [0, 1, 2, 3, 4]
    counts = np.bincount(list(folds.values()))
    assert counts.max() - counts.min() <= 1
    assert folds == rq.inner_folds(list(reversed(comps)))
    import hashlib

    first = min(comps, key=lambda c: hashlib.sha256(f"{rq.SEED}:inner:{c}".encode()).hexdigest())
    assert folds[first] == 0


def test_linearised_se_matches_the_registered_formula():
    ca, cb = np.array([1.0, 2.0, 4.0, 3.0]), np.array([2.0, 2.5, 5.0, 4.0])
    u = ca / ca.sum() - cb / cb.sum()
    assert float(rq.linearised_se(ca, cb)) == pytest.approx(math.sqrt(4 / 3 * (u**2).sum()))
    mult = np.array([2.0, 0.0, 1.0, 1.0])
    ua = ca / (mult @ ca) - cb / (mult @ cb)
    assert float(rq.linearised_se(ca, cb, mult)) == pytest.approx(
        math.sqrt(4 / 3 * (mult * ua**2).sum())
    )


def synthetic_population(n_rows=60, J=120, seed=0, signal=0.15):
    rng = np.random.default_rng(seed)
    comp_id = np.repeat(np.arange(n_rows // 2), 2)
    truth = rng.normal(0, 0.3, n_rows)
    D = truth[:, None] + rng.normal(0, 0.5, size=(n_rows, J))
    shared = rng.normal(0, 0.1, J)
    D += shared[None, :]
    m = np.zeros(n_rows)
    return {
        "D": D,
        "qa": m + truth * (1 - signal),
        "qb": m + truth * 0.3,
        "codes": [f"c{c}" for c in comp_id],
        "families": ["chain" if c % 2 else "lora_sft" for c in comp_id],
    }


def test_conditional_interval_is_calibrated_under_the_null_on_a_synthetic_population():
    pop = synthetic_population()
    sizes = np.array([2] * 30)
    classes = ["mixed"] * 30
    pools = rq.donor_pools(pop["codes"], pop["families"])
    kappa = rq.calibrate_kappa(pop, 0.0)
    outs = [
        rq.one_panel((pop, sizes, classes, pools, kappa, False, 1000 + k, 300, "conditional"))
        for k in range(150)
    ]
    summary = rq.summarise(outs, "bootstrap_t", 0.0)
    assert summary["panels"] == 150
    assert summary["rejection"] <= 0.12  # nominal 0.025; loose at 150 panels
    assert summary["denominator_failure_fraction"] == 0


def test_power_rises_with_theta_and_joint_runs():
    pop = synthetic_population(seed=3)
    sizes = np.array([2] * 30)
    classes = ["mixed"] * 30
    pools = rq.donor_pools(pop["codes"], pop["families"])
    rates = {}
    for theta in (0.0, 0.4):
        kappa = rq.calibrate_kappa(pop, theta)
        outs = [
            rq.one_panel((pop, sizes, classes, pools, kappa, False, 50 + k, 200, "conditional"))
            for k in range(60)
        ]
        rates[theta] = rq.summarise(outs, "percentile", theta)["rejection"]
    assert rates[0.4] > rates[0.0] + 0.2
    kappa = rq.calibrate_kappa(pop, 0.2)
    joint = [
        rq.one_panel((pop, sizes, classes, pools, kappa, False, 7 + k, 40, "joint"))
        for k in range(3)
    ]
    assert all(np.isfinite(o["theta_hat"]) and o["n_failed"] <= 40 for o in joint)


def test_joint_studentiser_tracks_the_two_way_bootstrap_spread_and_component_se_matches():
    # Two-way pigeonhole replicates double-count the cell-noise term, and the replicate se* (with
    # replicate multiplicities) carries the same inflation, so t* is pivotal: mean se* ~ sd(ln R*).
    pop = synthetic_population(seed=5, J=200)
    rows, comp = np.arange(60), np.repeat(np.arange(30), 2)
    kappa = rq.calibrate_kappa(pop, 0.1)
    ones = np.ones(200)
    rng = np.random.default_rng(0)
    ln_star, se_star = rq.joint_replicates(pop, rows, comp, 30, ones, kappa, rng, 600)
    assert np.nanmean(se_star) == pytest.approx(np.nanstd(ln_star), rel=0.3)
    ca, cb, cn = rq.joint_point(pop, rows, comp, 30, ones, kappa)
    cond_ln, _ = rq.conditional_replicates(ca, cb, cn, rng, 1500)
    assert float(rq.linearised_se(ca, cb)) == pytest.approx(np.nanstd(cond_ln), rel=0.15)
    _, se_obs = rq.joint_replicates_once(pop, rows, comp, ones, kappa)
    assert 0 < se_obs < np.nanstd(ln_star)  # single-counted observed variance < pigeonhole spread


def test_bca_and_wilson_basics():
    assert rq.wilson_lower(90, 100) == pytest.approx(0.826, abs=0.01)
    ln_star = np.log(np.linspace(0.6, 1.0, 400))
    lo, hi = rq.bca_levels(ln_star, math.log(0.8), np.log(np.array([0.78, 0.8, 0.82, 0.79])))
    assert lo < math.log(0.8) < hi


def test_gate_verdict_requires_the_registered_scale_and_is_exploratory_by_default():
    def cell(rej=0.03, cover=0.95, low=0.9):
        d = {
            "rejection": rej,
            "rejection_wilson_lower95": low,
            "coverage95": cover,
            "denominator_failure_fraction": 0.0,
        }
        return {"bootstrap_t": d, "bca": d}

    by = {"theta_0.0": {"conditional": cell(), "joint": cell()}}
    by["theta_0.15"] = {"conditional": cell(), "joint": cell()}
    results = {"empirical": by, "item": by, "merged": by}
    small = rq.gate_verdict(results, 100, 100, 100, True)
    assert small["status"].startswith("INDICATIVE") and not small["confirmatory"]
    full = rq.gate_verdict(results, 4000, 4000, 4000, True)
    assert full["confirmatory"] and full["method_chosen"] == "bootstrap_t"
    bad = {
        "theta_0.0": {"conditional": cell(rej=0.06), "joint": cell()},
        "theta_0.15": by["theta_0.15"],
    }
    fallback = rq.gate_verdict(
        {"empirical": bad, "item": bad, "merged": bad}, 4000, 4000, 4000, True
    )
    assert not fallback["confirmatory"] and fallback["status"].startswith("EXPLORATORY")
    assert not rq.gate_verdict(results, 4000, 4000, 4000, False)["confirmatory"]


def test_d3_minutes_logger_counts_only_complete_entries_with_the_expected_source(
    tmp_path, monkeypatch, capsys
):
    import json

    spec = importlib.util.spec_from_file_location("tfv1_d3_minutes", SCRIPTS / "tfv1_d3_minutes.py")
    minutes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(minutes)
    tf = tmp_path / "tf"

    def entry(name, source, status="complete", wall=62.0, dirty=False):
        (tf / name).mkdir(parents=True)
        (tf / name / "eval.json").write_text(
            json.dumps(
                {
                    "eval_status": status,
                    "source_hash": source,
                    "dirty": dirty,
                    "wall_seconds": wall,
                    "minutes": 1.0,
                    "stage_seconds": {},
                    "tf_cache": {"mode": "resident"},
                    "run_id": "r",
                    "family": "chain",
                }
            )
        )

    entry("base", "S")
    entry("a", "S")
    entry("b", "S", wall=66.0)
    entry("c", "OTHER")  # wrong source: never counted
    entry("d", "S", dirty=True)  # dirty tree: never counted
    ids = tmp_path / "ids"
    Path(f"{ids}.stratum.txt").write_text("a\nb\nc\nd\ne\n")
    monkeypatch.setattr(minutes, "TF", tf)
    monkeypatch.setattr(minutes, "IDS", ids)
    monkeypatch.setattr(minutes, "LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr("sys.argv", ["x", "--check"])
    with pytest.raises(SystemExit) as stop:
        minutes.main()
    assert stop.value.code == 1  # incomplete: c, d wrong, e missing
    out = json.loads(capsys.readouterr().out)
    assert out["done"] == 2 and out["remaining"] == 3
    assert sorted(out["incomplete_or_wrong_source"]) == ["c", "d"]
    assert out["median_wall_minutes"] == pytest.approx((62 + 66) / 2 / 60)
    logged = [json.loads(x)["edit_id"] for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert sorted(logged) == ["a", "b"]


def reference_se(pop, rows, comp_of_row, mult_c, mult_j, kappa):
    """Direct (loop) evaluation of the two-way studentiser for ONE replicate, from the definitions:
    V_comp + V_item - V_both with cell influences xi_cj = sum_{i in c} [s^a (d - qa)/SA - s^b (d - qb)/SB]."""
    D = pop["D"][rows]
    J = D.shape[1]
    qa, qb = pop["qa"][rows], pop["qb"][rows]
    comps = sorted(set(comp_of_row))
    n = len(comps)
    ybar = D @ mult_j / J
    a, b = ybar - qa, ybar - qb
    row_m = np.array([mult_c[comps.index(c)] for c in comp_of_row])
    sa, sb = (row_m * np.abs(a)).sum(), (row_m * np.abs(b)).sum()
    u = {
        c: np.abs(a)[comp_of_row == c].sum() / sa - np.abs(b)[comp_of_row == c].sum() / sb
        for c in comps
    }
    v_comp = n / (n - 1) * sum(mult_c[k] * u[c] ** 2 for k, c in enumerate(comps))
    g = np.sign(a) / sa - np.sign(b) / sb
    lam = (row_m * g) @ D
    p = mult_j / J
    v_item = (p * (lam - (p * lam).sum()) ** 2).sum() / (J - 1)
    v_both = 0.0
    for k, c in enumerate(comps):
        mem = comp_of_row == c
        xi = (np.sign(a)[mem, None] * (D[mem] - qa[mem, None])).sum(0) / sa - (
            np.sign(b)[mem, None] * (D[mem] - qb[mem, None])
        ).sum(0) / sb
        v_both += mult_c[k] * (mult_j * xi**2).sum() / J**2
    nn = n * J
    v_both *= nn / (nn - 1)
    return math.sqrt(v_comp + v_item - v_both), math.log(kappa * sa / sb)


def test_pairwise_gram_studentiser_equals_the_direct_cell_definition():
    pop = synthetic_population(seed=11, n_rows=40, J=50)
    rows = np.array([3, 5, 7, 8, 9, 11, 12, 20, 21, 22, 30, 31])
    comp = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 5])
    kappa = rq.calibrate_kappa(pop, 0.1)
    rng = np.random.default_rng(3)
    mult_c = rng.multinomial(6, np.full(6, 1 / 6), size=7).astype(float)
    mult_j = rng.multinomial(50, np.full(50, 1 / 50), size=7).astype(float)
    ln, se = rq.joint_replicates(
        pop, rows, comp, 6, np.ones(50), kappa, None, 7, mult=(mult_c, mult_j)
    )
    for r in range(7):
        ref_se, ref_ln = reference_se(pop, rows, comp, mult_c[r], mult_j[r], kappa)
        assert ln[r] == pytest.approx(ref_ln)
        assert se[r] == pytest.approx(ref_se, rel=1e-9)


def test_joint_replicates_resample_the_observed_items_not_the_full_item_set():
    pop = synthetic_population(seed=12, n_rows=20, J=40)
    rows, comp = np.arange(20), np.repeat(np.arange(10), 2)
    w_obs = np.zeros(40)
    w_obs[:5] = 8.0  # the observed panel contains only the first five items
    kappa = rq.calibrate_kappa(pop, 0.1)
    ln_star, _ = rq.joint_replicates(
        pop, rows, comp, 10, w_obs, kappa, np.random.default_rng(0), 300
    )
    ln_obs, _ = rq.joint_replicates_once(pop, rows, comp, w_obs, kappa)
    # replicates centre on the observed-sample statistic, not on the full-set one
    full, _ = rq.joint_replicates_once(pop, rows, comp, np.ones(40), kappa)
    assert abs(np.nanmean(ln_star) - ln_obs) < abs(np.nanmean(ln_star) - full) + 0.15
    assert np.isfinite(ln_star).all()


def test_panel_calibration_hits_the_target_theta_of_the_simulated_dgp():
    pop = synthetic_population(seed=8)
    sizes, classes = np.array([2, 2, 3, 1] * 8), ["mixed", "chain"] * 16
    pools = rq.donor_pools(pop["codes"], pop["families"])
    for theta in (0.0, 0.15, 0.25):
        kappa = rq.calibrate_kappa_panels(pop, sizes, classes, pools, False, theta, 5, 300)
        assert rq.population_theta(
            pop, sizes, classes, pools, kappa, False, 5, 300
        ) == pytest.approx(theta)
