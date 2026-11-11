"""Synthetic-outcome tests for the Part A analysis (A5.3/A5.5 registered outputs, F10.5 reports).

Every outcome here is synthetic (synthetic ledgers, fields, Part B results and masked-calibration
reports only). The only repository files read are committed, outcome-free metadata: the Part A
staging manifest, the NULL-SIZE report, text_groups.json, the F10.5 specification and the
registration text (for the F10b.1 hash and the verbatim texts). Nothing under data/ is opened.
"""

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from wsp.eval import part_a_analysis as pa

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/prereg/a5_manifests/part_a/manifest.json"
NULLSIZE = ROOT / "docs/prereg/a5_manifests/nullsize/report.json"
SPEC = ROOT / "docs/design/F10_SENSITIVITY_SPEC.md"
REGISTRATION = ROOT / "docs/PREREG_H1_TF.md"
TEXT_GROUPS = ROOT / "docs/prereg/a5_manifests/text_groups.json"
DRAWS = 200  # API override for speed; the CLI always uses the registered 4,000


@pytest.fixture(scope="module")
def manifest():
    return pa.load_manifest(MANIFEST)


@pytest.fixture(scope="module")
def nullsize():
    return pa.load_nullsize_report(NULLSIZE)


TG = pa.load_text_groups(TEXT_GROUPS)
PROV = {k: hashlib.sha256(k.encode()).hexdigest() for k in pa.RESTAGE_PROVENANCE_KEYS}


# ------------------------------------------------------------------------------ synthetic builder

MOMENTS = ("mean", "std", "frac_pos", "frac_nonzero", "cos_to_predecessor")


def _moments(rng, scale=1.0):
    return {k: float(rng.normal(0, scale)) for k in MOMENTS}


def build(
    manifest,
    *,
    seed=0,
    d_mean=None,
    noise=1e-3,
    predictions=None,
    excluded=None,
    missing=None,
    restage_rho=None,
    drop_covariate=None,
    ladder_mean=None,
    ladder_pred=None,
):
    """Synthetic sealed ledger + fields for the forty registered ids (no real outcome)."""
    rng = np.random.default_rng(seed)
    ids = sorted(s["edit_id"] for s in manifest["selections"])
    sel = {s["edit_id"]: s for s in manifest["selections"]}
    var_ids = {
        (v["source_id"], v["kind"], v["strength"]): v["variant_id"] for v in manifest["variants"]
    }
    d_mean = d_mean or {}
    predictions = predictions or {}
    excluded = excluded or {}
    missing = missing or {}
    restage_rho = restage_rho or {}
    drop_covariate = drop_covariate or {}
    ladder_mean = ladder_mean if ladder_mean is not None else {}
    ladder_pred = ladder_pred if ladder_pred is not None else {}
    windows = list(TG["window_ids"])
    refs, variants, restage, fields = {}, {}, {}, {}

    def series(base, mean):
        return list(base + mean + rng.normal(0, noise, 64))

    for i in ids:
        pred_id = sel[i]["predecessor_id"]
        if pred_id not in refs:
            refs[pred_id] = {"mean_nll": list(3.0 + rng.normal(0, 0.05, 64))}
        base = np.asarray(refs[pred_id]["mean_nll"])
        orig = base + 0.02 + rng.normal(0, noise, 64)
        refs[i] = {"mean_nll": list(orig)}
        d = d_mean.get(i, 0.01)
        rot = orig - d + rng.normal(0, noise, 64)
        lm = ladder_mean.get(i, 0.005)
        variants[i] = {
            "rotation": {"variant_id": var_ids[(i, "rotation", 1.0)], "mean_nll": list(rot)},
            "ladder_0.25": {
                "variant_id": var_ids[(i, "scale", 0.25)],
                "mean_nll": series(base, lm),
            },
            "ladder_0.5": {
                "variant_id": var_ids[(i, "scale", 0.5)],
                "mean_nll": series(base, 2 * lm),
            },
        }
        staged = 2 * math.log1p(manifest["edit_audits"][i]["edit_audit"]["pooled_norm_mismatch"])
        rho = restage_rho.get(i, math.exp(staged))
        tensors = {
            f"t{k}": {
                "shape": [8, 4 + k],
                "logged_d": _moments(rng),
                "logged_r": _moments(rng),
            }
            for k in range(2)
        }
        restage[i] = {
            "f10_3_verdict": "pass",
            "pre_audit_exclusions": [],
            "f10_3_exclusions": [],
            "registered_realised_verdict": False,
            "diagnostics": {"pooled_rho": rho, "epsilon_R_max": 0.1},
            "variant_json_sha256": hashlib.sha256(i.encode()).hexdigest(),
            "pooled_rho": rho,
            "tensors": tensors,
        }
        code = excluded.get(i)
        if code in pa.PRE_AUDIT_CODES:
            restage[i]["pre_audit_exclusions"] = [code]
        elif code is not None:
            restage[i]["f10_3_exclusions"] = [code]
        if code is not None:
            restage[i]["f10_3_verdict"] = "excluded"
        if code is not None:
            variants[i]["rotation"] = {
                "variant_id": var_ids[(i, "rotation", 1.0)],
                "excluded": True,
            }
        p = predictions.get(i, 0.5 if d >= 0 else -0.5)
        fields[i] = [p, ladder_pred.get(i, 0.3), ladder_pred.get(i, 0.3)]
        col = drop_covariate.get(i)
        if col == "ln_rho":
            restage[i]["pooled_rho"] = None
        elif col == "d_mean":
            restage[i]["tensors"]["t0"]["logged_r"]["mean"] = float("nan")
    for i, arm in missing.items():
        if arm in ("original",):
            refs[i]["mean_nll"][5] = None
        elif arm == "rotation_nonfinite":
            variants[i]["rotation"]["mean_nll"][7] = float("nan")
        elif arm == "rotation_absent":
            variants[i]["rotation"].pop("mean_nll")
    fields_bytes = json.dumps(fields, sort_keys=True).encode()
    ledger = {
        "schema": pa.LEDGER_SCHEMA,
        "sealed": True,
        "manifest_sha256": pa.STAGED_MANIFEST_SHA256,
        "fields_sha256": hashlib.sha256(fields_bytes).hexdigest(),
        "text_groups_sha256": TG["sha256"],
        "window_ids": windows,
        "two_window_blocks": copy.deepcopy(TG["two_window_blocks"]),
        "window_token_ids_hashes": list(TG["token_ids_hashes"]),
        "restage_provenance": dict(PROV),
        "references": refs,
        "variants": variants,
        "restage": restage,
    }
    return ledger, fields, fields_bytes


def masked_for(ledger, manifest, passed=True, power_passed=True):
    """Synthetic ``a5_simulation.py --slot-mask`` output matching the ledger's exclusions."""
    audit = pa.restage_audit(ledger, manifest)
    excluded = sorted(i for i, a in audit.items() if a["status"] != "pass")
    canonical = {
        i: (["rotation"] if audit[i]["status"] == "pass" else []) + ["ladder_0.25", "ladder_0.5"]
        for i in sorted(audit)
    }
    mapping = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    rep = {
        "replicates": 10000,
        "seed": 20261107,
        "slot_mask": {
            "label": "slot-masked (F10.6)",
            "schema": "a5-slot-mask-v1",
            "sha256": "ab" * 32,
            "mapping_sha256": mapping,
            "masked_ids": {"rotation": excluded, "ladder_0.25": [], "ladder_0.5": []},
            "gates": {
                "label": "slot-masked (F10.6)",
                "size": {"passed": passed, "ordered_go_rate": 0.015},
                "power": {"passed": power_passed, "ordered_go_rate": 0.83},
                "passed": passed and power_passed,
            },
        },
        "reports": {
            "null_size": {
                "decisions": {"go": 150, "alternative": 1800, "unresolved": 8050},
                "ordered_go_rate": 0.015,
                "ordered_go_cp95": [0.0127, 0.0176],
            }
        },
    }
    raw = json.dumps(rep, sort_keys=True).encode()
    return rep, hashlib.sha256(raw).hexdigest(), raw


def valid_for(ledger, masked_sha=None):
    """Step-1 predicates that pass and link exactly the required pins."""
    b = pa.validity_bindings(
        pa.SPEC_SHA256, ledger["fields_sha256"], ledger.get("restage_provenance"), masked_sha
    )
    out = {}
    for k in pa.VALIDITY_PREDICATES:
        ev = [{"path": f"synthetic/{k}/{n}", "sha256": h} for n, h in enumerate(b.get(k, []))]
        out[k] = {"pass": True, "evidence": ev or [{"path": f"synthetic/{k}", "sha256": "cd" * 32}]}
    return out


def analyse(manifest, nullsize, ledger, fields, **kw):
    """API wrapper: synthetic masked calibration when the ledger has exclusions; 'auto' validity."""
    kw.setdefault("bootstrap_draws", DRAWS)
    if "masked_calibration" not in kw:
        audit = pa.restage_audit(ledger, manifest)
        if any(a["status"] != "pass" for a in audit.values()):
            kw["masked_calibration"], kw["masked_calibration_sha256"], _ = masked_for(
                ledger, manifest
            )
    if isinstance(kw.get("validity"), str):
        kw["validity"] = valid_for(ledger, kw.get("masked_calibration_sha256"))
    if kw.get("part_b") == "ok":
        kw["part_b"] = PB()
    return pa.analyse(ledger, manifest, fields, nullsize, **kw)


EV = [{"path": "synthetic/evidence.json", "sha256": "0" * 64}]
VALID = {k: {"pass": True, "evidence": EV} for k in pa.VALIDITY_PREDICATES}


def PB(c=0.2, lo=0.01, hi=0.3, ctrl_lo=0.01, eligible=True, power=0.9, ctrl_joint_lo=None):
    """Synthetic, step-1-valid Part B results (never computed here)."""
    iv = {"conditional": [lo, hi], "joint": [lo, hi]}
    ctrl = {
        "conditional": [ctrl_lo, 0.5],
        "joint": [ctrl_lo if ctrl_joint_lo is None else ctrl_joint_lo, 0.5],
    }
    diag = {
        p: {"draws": 4000, "failed_draws": 0, "min_distinct_components": 12} for p in pa.PROCEDURES
    }
    return {
        "points": {"C": c, "C_abl": 0.1, "C_rand": 0.1, "C_perm": 0.1},
        "intervals": {
            "C": iv,
            "C_abl": copy.deepcopy(ctrl),
            "C_rand": copy.deepcopy(ctrl),
            "C_perm": copy.deepcopy(ctrl),
        },
        "valid": dict.fromkeys(pa.PART_B_CONTRASTS, True),
        "diagnostics": {n: copy.deepcopy(diag) for n in pa.PART_B_CONTRASTS},
        "destruction_controls_eligible": eligible,
        "evidence": [{"path": "synthetic/part_b.json", "sha256": "ef" * 32}],
        "power_at_0_15": {"conditional": power, "joint": power},
    }


# ------------------------------------------------------------------------------ frozen constants


def test_frozen_subset_hashes_and_complements(manifest):
    assert pa.sorted_sha(pa.S30_IDS) == pa.S30_SHA256
    assert pa.S30_SHA256.startswith("3da5f44f") and pa.S30_SHA256.endswith("267c")
    assert pa.sorted_sha(pa.S34_IDS) == pa.S34_SHA256
    assert pa.S34_SHA256.startswith("fb1c34e6") and pa.S34_SHA256.endswith("dfa0")
    ids = set(manifest["edit_ids"])
    assert pa.sorted_sha(ids - set(pa.S30_IDS)) == (
        "7ce32f86ac171dde7da77c4c37c4b18c9e8f18b0c851d4f3e87b826a56b52280"
    )
    assert pa.sorted_sha(ids - set(pa.S34_IDS)) == (
        "b926f27079df5720ecc1fce07c97b3e223069f920a361cb3cf7f89027eb198aa"
    )
    assert len(set(pa.S30_IDS) & set(pa.S34_IDS)) == 29


def test_s30_recomputes_from_staged_manifest(manifest):
    s30 = [
        i
        for i, a in manifest["edit_audits"].items()
        if a["edit_audit"]["pooled_norm_mismatch"] <= 2e-2
    ]
    assert sorted(s30) == sorted(pa.S30_IDS)


def test_norm_tertiles_from_manifest_match_frozen_hashes(manifest):
    cells = pa.norm_tertiles(manifest)
    assert [len(cells[t]["pooled"]) for t in range(3)] == [14, 13, 13]
    for t in range(3):
        assert pa.sorted_sha(cells[t]["pooled"]) == pa.NORM_TERTILE_SHA256[t]
        assert len(cells[t]["lora_sft"]) == 10
    assert [len(cells[t]["chain"]) for t in range(3)] == [4, 3, 3]


def test_lnrho_tertiles_frozen_cuts_and_members(manifest):
    cells, cuts = pa.lnrho_tertiles(manifest)
    assert cuts == pa.LNRHO_CUTS
    assert cuts[0] == (0.01010406252666643, "057dcfbedea3251f")
    assert cuts[1] == (0.03502604499592402, "22659cefc904e29f")
    assert [len(c) for c in cells] == [13, 13, 14]
    assert [pa.sorted_sha(c) for c in cells] == list(pa.LNRHO_CELL_SHA256)
    assert cells[0][0] == "15a53586efc72a50" and cells[2][-1] == "61d1f61e25212068"


def test_frozen_constants_fail_closed_on_drift(manifest, monkeypatch):
    monkeypatch.setattr(pa, "NORM_TERTILE_SHA256", ("0" * 64,) + pa.NORM_TERTILE_SHA256[1:])
    with pytest.raises(pa.AnalysisRefused, match="norm tertile"):
        pa.norm_tertiles(manifest)


def _spec_text():
    return SPEC.read_text()


def _norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def test_verbatim_texts_match_spec():
    spec = _spec_text()
    block = spec.split("## 5. Limitations paragraph")[1].split("\n---")[0]
    quoted = " ".join(line[2:] for line in block.splitlines() if line.startswith("> "))
    assert _norm_ws(quoted) == _norm_ws(pa.LIMITATIONS)
    assert _norm_ws(pa.COVARIATE_SENTENCE.replace("*", "")) in _norm_ws(spec.replace("*", ""))
    assert _norm_ws(pa.NO_SUPERIORITY_STATEMENT) in _norm_ws(spec.replace("*", ""))


def test_spec_hash_recorded():
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == pa.SPEC_SHA256


# ------------------------------------------------------------------------------ CP intervals


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (0, 15), (15, 15), (12, 15), (27, 40), (7, 13)])
def test_cp_matches_beta_quantiles(k, n):
    from scipy.stats import beta

    lo, hi = pa.cp(k, n)
    exp_lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    exp_hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    assert lo == pytest.approx(exp_lo, abs=1e-12)
    assert hi == pytest.approx(exp_hi, abs=1e-12)


def test_cp_registered_examples_and_empty():
    assert pa.cp(12, 15)[0] == pytest.approx(0.5191, abs=1e-4)
    assert pa.cp(27, 40)[0] == pytest.approx(0.5087, abs=1e-4)
    assert pa.cp(0, 0) is None
    row = pa.proportion(0, 0)
    assert row["interval"] == "no resolved pairs"


# ------------------------------------------------------------------------------ ridge logistic


def test_ridge_matches_sklearn_objective():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    Z = rng.normal(size=(30, 8))
    a = (Z[:, 0] + rng.normal(size=30) > 0).astype(float)
    b0, b, info = pa.fit_ridge_logistic(Z, a)
    sk = LogisticRegression(C=1.0, fit_intercept=True, tol=1e-12, max_iter=10_000).fit(Z, a)
    assert b0 == pytest.approx(sk.intercept_[0], abs=1e-6)
    np.testing.assert_allclose(b, sk.coef_[0], atol=1e-6)
    assert info["converged"] is True and info["iterations"] <= 100
    assert info["gradient_max_abs"] <= 1e-10
    p = 1 / (1 + np.exp(-(b0 + Z @ b)))
    grad = np.concatenate(([np.sum(p - a)], Z.T @ (p - a) + b))
    assert np.max(np.abs(grad)) <= 1e-10


def test_ridge_one_class_is_failure():
    with pytest.raises(pa.FitFailure, match="one class"):
        pa.fit_ridge_logistic(np.zeros((20, 2)), np.ones(20))


def test_train_only_standardisation_and_degenerate_column():
    X_train = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    X_held = np.array([[7.0, 9.0]])
    Zt, Zh, degenerate = pa.standardise(X_train, X_held, ["c0", "c1"])
    mu, sd = 3.0, math.sqrt(8 / 3)  # population SD, ddof 0
    np.testing.assert_allclose(Zt[:, 0], (X_train[:, 0] - mu) / sd)
    assert Zh[0, 0] == pytest.approx((7 - mu) / sd)  # held-out row uses training moments only
    assert degenerate == ["c1"]
    assert np.all(Zt[:, 1] == 0) and Zh[0, 1] == 0


def test_bootstrap_indices_registered_stream():
    idx = pa.bootstrap_indices(17, 4000)
    ref = np.random.Generator(np.random.PCG64(20261107)).integers(0, 17, size=(4000, 17))
    np.testing.assert_array_equal(idx, ref)


# ------------------------------------------------------------------------------ registered contrast


def test_all_agree_goes_and_comparator_reported(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    rep = analyse(manifest, nullsize, ledger, fields, validity="auto", part_b="ok")
    rot = rep["registered"]["contrasts"]["rotation"]
    assert rot["resolved"] == 40 and rot["agreements"] == 40
    assert rot["units"] == "nats per valid next token"
    # ladder: predictions positive, ladder means positive -> agreement, no alternative
    dec = rep["registered"]["decision"]
    assert dec["step_2_ladder_alternative"] is False
    assert dec["part_a"] == "go"
    cmp_ = rep["f10_5_sensitivity"]["constant_sign_comparator"]
    assert cmp_["authority"] == "none (descriptive)"
    assert cmp_["main_contrast"]["comparator"]["k"] == 40
    assert pa.NO_SUPERIORITY_STATEMENT in cmp_["statement"]


def test_zero_prediction_counts_as_disagreement_and_sign_table(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    preds = {ids[0]: 0.0, ids[1]: -0.4}
    ledger, fields, _ = build(manifest, predictions=preds)
    rep = analyse(manifest, nullsize, ledger, fields)
    rot = rep["registered"]["contrasts"]["rotation"]
    assert rot["agreements"] == 38 and rot["disagreements"] == 2
    assert rot["zero_prediction_disagreements"] == 1
    tab = rot["sign_table"]
    assert tab["zero"] == {
        "n": 1,
        "agree": 0,
        "disagree": 1,
        "observed_positive": 1,
        "observed_negative": 0,
    }
    assert tab["negative"]["disagree"] == 1
    assert tab["positive"]["agree"] == 38
    assert rot["negative_prediction_share"] == pytest.approx(1 / 40)
    # comparator: every d_hat > 0 -> 40/40 regardless of predictions
    assert rot["comparator"]["k"] == 40


def test_exclusions_missing_and_unresolved_stay_in_ledger(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(
        manifest,
        excluded={ids[0]: "sub_resolution", ids[1]: "non_2d"},
        missing={ids[2]: "original", ids[3]: "rotation_nonfinite", ids[4]: "rotation_absent"},
        d_mean={ids[5]: 0.0},
    )
    rep = analyse(manifest, nullsize, ledger, fields)
    led = rep["registered"]["contrasts"]["rotation"]["ledger"]
    assert led["selected"]["n"] == 40
    assert led["pre_audit_excluded"]["by_code"] == {"non_2d": [ids[1]]}
    assert led["f10_3_excluded"]["by_code"] == {"sub_resolution": [ids[0]]}
    assert led["missing_score"]["by_arm"] == {"original": [ids[2]], "rotation": [ids[3], ids[4]]}
    assert ids[5] in led["unresolved"]["ids"]
    assert led["resolved"]["n"] == 40 - 5 - len(led["unresolved"]["ids"])
    # denominator excludes everything above; every id appears exactly once
    flat = (
        led["pre_audit_excluded"]["ids"]
        + led["f10_3_excluded"]["ids"]
        + led["missing_score"]["ids"]
        + led["unresolved"]["ids"]
        + led["resolved"]["ids"]
    )
    assert sorted(flat) == ids


def test_resolution_is_deterministic_and_uses_common_draws(manifest, nullsize):
    ledger, fields, _ = build(manifest, noise=2e-2, seed=5)
    r1 = analyse(manifest, nullsize, ledger, fields)
    r2 = analyse(manifest, nullsize, ledger, fields)
    assert r1["registered"] == r2["registered"]
    rec = r1["registered"]["pairs"][min(manifest["edit_ids"])]["rotation"]
    assert rec["interval"][0] <= rec["d_hat"] <= rec["interval"][1]
    assert r1["registered"]["resolution"]["draws"] == 4000
    assert r1["registered"]["resolution"]["seed"] == 20261107


def test_ladder_alternative_overrides_rotation_go(manifest, nullsize):
    ids = manifest["edit_ids"]
    # ladder predictions all negative while ladder responses positive -> 0 agreement at s=.25
    ledger, fields, _ = build(manifest, ladder_pred={i: -0.3 for i in ids})
    rep = analyse(manifest, nullsize, ledger, fields, validity="auto", part_b="ok")
    dec = rep["registered"]["decision"]
    assert rep["registered"]["contrasts"]["rotation"]["agreements"] == 40
    assert dec["step_2_ladder_alternative"] is True
    assert dec["gate"] == "alternative"
    assert dec["part_a"] == "not reached (alternative overrides rotation go)"


def test_realised_increment_ladder_flags(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    rep = analyse(manifest, nullsize, ledger, fields)
    flagged = [
        i
        for i, p in rep["registered"]["pairs"].items()
        if p["ladder_0.25"]["realised_increment_ladder"]
    ]
    assert sorted(flagged) == sorted(pa.REALISED_INCREMENT_LADDER_IDS)
    assert len(flagged) == 8


# ------------------------------------------------------------------------------ decision function


def _summary(k, n):
    return {"resolved": n, "agreements": k, "cp95": pa.cp(k, n) if n else None}


@pytest.mark.parametrize(
    "rot,lad,expected",
    [
        ((12, 15), (10, 14), "go"),  # exactly 15 resolved meets the floor; ladder below its floor
        ((11, 15), (10, 14), "unresolved: interval includes 0.5"),
        ((14, 14), (0, 0), "unresolved: fewer than 15 resolved rotation pairs"),
        (
            (2, 20),
            (0, 0),
            "unresolved: interval entirely below 0.5 (observed disagreement reported)",
        ),
    ],
)
def test_part_a_step3(rot, lad, expected):
    d = pa.decision(VALID, _summary(*rot), _summary(*lad), part_b=PB())
    assert d["part_a"] == expected


def test_ladder_upper_exactly_point_six_meets_rule():
    lad = {"resolved": 15, "agreements": 5, "cp95": (0.2, 0.6)}
    d = pa.decision(VALID, _summary(40, 40), lad, part_b=PB())
    assert d["gate"] == "alternative"
    lad["cp95"] = (0.2, 0.6000001)
    assert pa.decision(VALID, _summary(40, 40), lad, part_b=PB())["gate"] != "alternative"
    lad = {"resolved": 14, "agreements": 0, "cp95": pa.cp(0, 14)}  # below the 15 floor
    assert pa.decision(VALID, _summary(40, 40), lad, part_b=PB())["gate"] == "Part A go"


def test_validity_failure_stops_everything():
    bad = dict(VALID)
    bad["null_size"] = {"pass": False, "evidence": "x"}
    d = pa.decision(bad, _summary(40, 40), _summary(0, 20), part_b=PB())
    assert d["gate"] == "unresolved (validity)" and d["part_a"] == "unresolved (validity)"
    assert d["step_2_ladder_alternative"] == "unresolved (validity)"
    assert d["step_4_part_b_eligibility"] == "unresolved (validity)"
    assert d["funding"] == "unresolved (validity)"
    d = pa.decision(None, _summary(40, 40), _summary(0, 20), part_b=PB())
    assert d["gate"] == "unresolved (validity)"
    assert "predicates not supplied" in d["validity"]["problems"]["external_predicates"]


def test_funding_branches():
    go = (VALID, _summary(40, 40), _summary(0, 0))
    assert pa.decision(*go, part_b=PB())["funding"] == "go"
    assert pa.decision(*go, part_b=PB(c=0.1, lo=-0.1))["funding"] == "screen"
    assert pa.decision(*go, part_b=PB(c=0.05, lo=-0.1))["funding"] == "screen"
    nogo = PB(c=0.0, lo=-0.1, hi=0.04)
    assert pa.decision(*go, part_b=nogo)["funding"] == "no-go"
    nogo_eq = PB(c=0.0, lo=-0.1, hi=0.05)  # upper exactly 0.05 does not meet no-go
    assert pa.decision(*go, part_b=nogo_eq)["funding"] == "unresolved"
    assert pa.decision(*go, part_b=PB(ctrl_lo=0.0))["funding"].startswith("unresolved")
    # Part A not go -> funding unresolved regardless of C
    d = pa.decision(VALID, _summary(10, 20), _summary(0, 0), part_b=PB())
    assert d["funding"].startswith("unresolved")


# ------------------------------------------------------------------------------ covariate analysis


def _rows(n, seed=0, p_agree=0.7):
    rng = np.random.default_rng(seed)
    ids = [f"{k:016x}" for k in range(n)]
    X = rng.normal(size=(n, 8))
    X[:, 0] = (np.arange(n) % 3 == 0).astype(float)
    a = (rng.random(n) < p_agree).astype(float)
    a[0], a[1] = 1.0, 0.0
    return ids, X, a


def test_covariate_not_estimable_below_floor_or_one_class():
    ids, X, a = _rows(14)
    out = pa.covariate_models(ids, X, a, draws=DRAWS)
    assert (
        out["M0"]["status"].startswith("not estimable") and "n_cc = 14 < 15" in out["M0"]["status"]
    )
    assert out["M1"]["status"] == out["M0"]["status"]
    assert "coefficients" not in out["M1"]
    ids, X, a = _rows(20)
    out = pa.covariate_models(ids, X, np.ones(20), draws=DRAWS)
    assert "one agreement class" in out["M0"]["status"]


def test_covariate_models_estimates_and_draw_failures():
    ids, X, a = _rows(16, seed=1, p_agree=0.85)
    out = pa.covariate_models(ids, X, a, draws=DRAWS)
    assert out["M0"]["status"] == "estimated" and out["M1"]["status"] == "estimated"
    assert len(out["M0"]["coefficients"]) == 7 and len(out["M1"]["coefficients"]) == 8
    assert "ln_rho" in out["M1"]["coefficients"]
    assert out["M1"]["theta_adj"]["point"] == pytest.approx(
        1 / (1 + math.exp(-out["M1"]["intercept"]))
    )
    assert out["delta_theta"]["point"] == pytest.approx(
        out["M1"]["theta_adj"]["point"] - out["M0"]["theta_adj"]["point"]
    )
    # with 2 disagreements in 16 rows, some bootstrap draws have one class -> counted failures
    failed = out["M0"]["failed_draws"]
    assert failed > 0
    assert out["M0"]["theta_adj"]["label"] == f"diagnostic ({failed} of {DRAWS} draws failed)"
    assert out["loo"]["M0"]["failed_folds"] >= 0
    for m in ("intercept_only", "M0", "M1"):
        assert {"log_loss", "brier", "failed_folds"} <= set(out["loo"][m])


def test_covariate_loo_held_out_uses_training_moments(monkeypatch):
    ids, X, a = _rows(16, seed=2)
    seen = []
    real = pa.standardise

    def spy(train, apply, names):
        seen.append((train.shape[0], None if apply is None else apply.shape[0]))
        return real(train, apply, names)

    monkeypatch.setattr(pa, "standardise", spy)
    pa.covariate_models(ids, X, a, draws=3)
    assert (15, 1) in seen  # LOO: 15 training rows, 1 held-out row standardised with them


def test_covariate_report_drops_missing_from_both_models(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(
        manifest, drop_covariate={ids[0]: "ln_rho", ids[1]: "d_mean"}, predictions={ids[2]: -1.0}
    )
    rep = analyse(manifest, nullsize, ledger, fields)
    cov = rep["f10_5_sensitivity"]["covariate"]
    assert cov["authority"] == "none (descriptive)"
    dropped = {d["id"]: d["reason"] for d in cov["dropped_rows"]}
    assert dropped == {ids[0]: "covariate_missing:ln_rho", ids[1]: "covariate_missing:d_mean"}
    assert cov["n_cc"] == 38
    assert cov["theta_raw"]["k"] == 37 and cov["comparator"]["k"] == 38
    assert cov["full_analysis_set_agreement"]["n"] == 40
    assert cov["sentence"] == pa.COVARIATE_SENTENCE
    assert cov["columns"] == list(pa.COVARIATE_COLUMNS)


def test_moment_differences_are_element_weighted():
    tensors = {
        "a": {
            "shape": [2, 3],
            "logged_d": dict.fromkeys(MOMENTS, 0.0),
            "logged_r": dict.fromkeys(MOMENTS, 1.0),
        },
        "b": {
            "shape": [4],
            "logged_d": dict.fromkeys(MOMENTS, 0.0),
            "logged_r": dict.fromkeys(MOMENTS, -1.0),
        },
    }
    out = pa.moment_differences(tensors)
    assert out["d_mean"] == pytest.approx((6 * 1 + 4 * -1) / 10)


def test_lnrho_restage_drift_flag_and_disclosure(manifest, nullsize):
    low = "15a53586efc72a50"  # frozen low-energy cell
    ledger, fields, _ = build(manifest, restage_rho={low: math.exp(0.05)})
    rep = analyse(manifest, nullsize, ledger, fields)
    rows = rep["f10_5_sensitivity"]["tertiles"]["ln_rho"]
    low_row = rows[0]
    assert low in low_row["ids"]  # never moved
    assert low_row["cell_drift"] == [low]
    assert rep["f10_5_sensitivity"]["covariate"]["max_abs_lnrho_restage_minus_staged"] > 0.04


# ------------------------------------------------------------------------------ tertiles and subsets


def test_tertile_rows_labelled_and_complete(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    rep = analyse(manifest, nullsize, ledger, fields)
    t = rep["f10_5_sensitivity"]["tertiles"]
    assert t["authority"] == "none (descriptive)"
    assert len(t["norm"]) == 9 and len(t["ln_rho"]) == 3
    for row in t["norm"] + t["ln_rho"]:
        assert row["label"] == "below floor (descriptive)"
        assert row["authority"] == "none (descriptive)"
        assert {"field", "comparator", "sign_table", "ledger"} <= set(row)
    assert {r["cell"] for r in t["norm"]} >= {
        "low (0) pooled",
        "low (0) lora_sft",
        "high (2) chain",
    }


def test_subset_intersection_disclosed_never_replaced(manifest, nullsize):
    in_s30 = "0027efdf2eae7405"
    ledger, fields, _ = build(
        manifest,
        excluded={in_s30: "replay_failed"},
        missing={"00f8111cbad8218f": "original"},
        d_mean={"014f8a9ec78db2be": 0.0},
    )
    rep = analyse(manifest, nullsize, ledger, fields)
    subs = rep["f10_5_sensitivity"]["subsets"]
    assert subs["authority"] == "none (descriptive)"
    s30 = subs["S30"]
    assert s30["size"] == 30 and s30["sha256"] == pa.S30_SHA256
    assert s30["intersection_with_X"] == [
        {"id": "0027efdf2eae7405", "reason": "replay_failed"},
        {"id": "00f8111cbad8218f", "reason": "missing_score"},
    ]
    assert s30["resolved_n"] == 30 - 2 - len(s30["unresolved_ids"])
    assert s30["label"] == "resolved (n ≥ 15)"
    assert "never replaced" in s30["disclosure"]
    s34 = subs["S34"]
    assert s34["size"] == 34
    assert {d["id"] for d in s34["intersection_with_X"]} == {"0027efdf2eae7405"}


def test_subset_hash_mismatch_fails_closed(manifest, nullsize, monkeypatch):
    monkeypatch.setattr(pa, "S30_IDS", pa.S30_IDS[:-1] + ("0a841d4082532e8b",))
    ledger, fields, _ = build(manifest)
    with pytest.raises(pa.AnalysisRefused, match="S30"):
        analyse(manifest, nullsize, ledger, fields)


def test_subset_below_floor_label(manifest, nullsize):
    ids = sorted(pa.S30_IDS)[:16]
    ledger, fields, _ = build(manifest, excluded={i: "support_failed" for i in ids})
    rep = analyse(manifest, nullsize, ledger, fields)
    assert rep["f10_5_sensitivity"]["subsets"]["S30"]["label"] == "below floor (n < 15)"


# ------------------------------------------------------------------------------ report structure


def test_authority_separation_and_texts(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    rep = analyse(manifest, nullsize, ledger, fields)
    sens = rep["f10_5_sensitivity"]
    assert sens["authority"] == "none (descriptive)"
    for key in ("covariate", "tertiles", "subsets", "constant_sign_comparator"):
        assert sens[key]["authority"] == "none (descriptive)"
    assert rep["registered"]["authority"].startswith("registered")
    assert rep["limitations"] == pa.LIMITATIONS
    assert rep["units"] == "nats per valid next token"
    f57 = rep["f5_7_null_branch_table"]
    assert next(r["scenario"] for r in f57["rows"]) == "registered"
    reg = f57["rows"][0]
    assert (reg["alternative_count"], reg["go_count"], reg["unresolved_count"]) == (1939, 180, 7881)
    assert rep["spec"]["sha256"] == pa.SPEC_SHA256


def test_markdown_rendering_contains_required_text(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    md = pa.render_markdown(analyse(manifest, nullsize, ledger, fields))
    assert pa.LIMITATIONS in md
    assert pa.NO_SUPERIORITY_STATEMENT in md
    assert pa.COVARIATE_SENTENCE in md
    assert "nats per valid next token" in md
    assert "authority: none (descriptive)" in md
    assert md.index("Registered") < md.index("F10.5")
    assert "| predicted sign (3 × 2) | n predicted | agree | disagree |" in md


# ------------------------------------------------------------------------------ fail-closed inputs


def test_missing_prediction_row_refused(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    fields.pop(min(fields))
    with pytest.raises(pa.AnalysisRefused, match="prediction"):
        analyse(manifest, nullsize, ledger, fields)


def test_null_prediction_only_for_excluded_variant(manifest, nullsize):
    i = min(manifest["edit_ids"])
    ledger, fields, _ = build(manifest, excluded={i: "sub_resolution"})
    fields[i][0] = None
    analyse(manifest, nullsize, ledger, fields)  # excluded rotation: accepted
    ledger, fields, _ = build(manifest)
    fields[i][0] = None
    with pytest.raises(pa.AnalysisRefused, match="prediction"):
        analyse(manifest, nullsize, ledger, fields)


def test_unsealed_or_bad_ledger_refused(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    bad = copy.deepcopy(ledger)
    bad["sealed"] = False
    with pytest.raises(pa.AnalysisRefused, match="sealed"):
        analyse(manifest, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    bad["window_ids"] = bad["window_ids"][:63]
    with pytest.raises(pa.AnalysisRefused, match="64"):
        analyse(manifest, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    i = min(bad["variants"])
    bad["variants"][i]["ladder_0.5"]["variant_id"] = "ffffffffffffffff"
    with pytest.raises(pa.AnalysisRefused, match="variant id"):
        analyse(manifest, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    bad["restage"][i]["f10_3_exclusions"] = ["audit_failed"]
    with pytest.raises(pa.AnalysisRefused, match="reason code"):
        analyse(manifest, nullsize, bad, fields)


def test_missing_restage_audit_fails_closed(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    i = min(ledger["restage"])
    del ledger["restage"][i]
    rep = analyse(manifest, nullsize, ledger, fields)
    led = rep["registered"]["contrasts"]["rotation"]["ledger"]
    assert led["f10_3_excluded"]["by_code"] == {"audit_missing": [i]}


def test_manifest_hash_mismatch_refused(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_bytes(MANIFEST.read_bytes() + b"\n")
    with pytest.raises(pa.AnalysisRefused, match="manifest sha256"):
        pa.load_manifest(bad)


def _write_inputs(tmp_path, manifest, **kw):
    ledger, _fields, fields_bytes = build(manifest, **kw)
    (tmp_path / "fields.json").write_bytes(fields_bytes)
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    return tmp_path / "ledger.json", tmp_path / "fields.json"


def test_cli_writes_json_and_markdown(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr(pa, "BOOTSTRAP_DRAWS", DRAWS)
    led, fld = _write_inputs(tmp_path, manifest)
    out, md = tmp_path / "r.json", tmp_path / "r.md"
    rc = pa.main(
        ["--ledger", str(led), "--fields", str(fld), "--output", str(out), "--markdown", str(md)]
        + ["--spec-sha256", pa.SPEC_SHA256]
    )
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["registered"]["contrasts"]["rotation"]["resolved"] == 40
    assert rep["inputs"]["manifest_sha256"] == pa.STAGED_MANIFEST_SHA256
    assert pa.LIMITATIONS in md.read_text()


def test_cli_refuses_missing_prediction_and_bad_manifest(tmp_path, manifest, capsys):
    led, fld = _write_inputs(tmp_path, manifest)
    fields = json.loads(fld.read_text())
    fields.pop(min(fields))
    fld.write_text(json.dumps(fields))
    out = tmp_path / "r.json"
    args = [
        "--ledger",
        str(led),
        "--fields",
        str(fld),
        "--output",
        str(out),
        "--markdown",
        str(tmp_path / "r.md"),
        "--spec-sha256",
        pa.SPEC_SHA256,
    ]
    assert pa.main(args) == 2  # fields hash no longer matches the ledger either way: refused
    assert not out.exists()
    led, fld = _write_inputs(tmp_path, manifest)
    bad = tmp_path / "m.json"
    bad.write_bytes(MANIFEST.read_bytes().replace(b'"scored": false', b'"scored": true'))
    args[3] = str(fld)
    assert pa.main(args + ["--manifest", str(bad)]) == 2
    assert "manifest sha256" in capsys.readouterr().err
    assert not out.exists()


def test_cli_refuses_fields_without_row_even_if_hash_matches(tmp_path, manifest):
    ledger, fields, _ = build(manifest)
    fields.pop(min(fields))
    fb = json.dumps(fields).encode()
    ledger["fields_sha256"] = hashlib.sha256(fb).hexdigest()
    (tmp_path / "f.json").write_bytes(fb)
    (tmp_path / "l.json").write_text(json.dumps(ledger))
    rc = pa.main(
        [
            "--ledger",
            str(tmp_path / "l.json"),
            "--fields",
            str(tmp_path / "f.json"),
            "--output",
            str(tmp_path / "r.json"),
            "--markdown",
            str(tmp_path / "r.md"),
            "--spec-sha256",
            pa.SPEC_SHA256,
        ]
    )
    assert rc == 2 and not (tmp_path / "r.json").exists()


def test_records_parser(tmp_path):
    rows = [
        {"benchmark": "wikitext_ppl", "item_id": "1", "extra": {"mean_nll": 2.5}},
        {"benchmark": "mmlu_pro", "item_id": "1", "extra": {"mean_nll": 9.0}},
        {"benchmark": "wikitext_ppl", "item_id": "0", "extra": {"mean_nll": 2.0}},
    ]
    p = tmp_path / "records.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert pa.window_nll_from_records(p, ["0", "1", "2"]) == [2.0, 2.5, None]


# ============================================================================== Board A condition 6
# Deliberate-failure tests for the six corrections (Board A, Part A analysis review, 11 Nov 2026).

# ------------------------------------------------------------------------------ (1) validity first

ALT = {"resolved": 20, "agreements": 2, "cp95": pa.cp(2, 20)}  # ladder alternative would fire


def _invalid_b():
    b = PB()
    b["valid"]["C"] = False
    return b


def _is_validity_unresolved(d):
    return (
        d["gate"].startswith("unresolved (validity")
        and d["step_1_validity"] == "unresolved"
        and d["step_2_ladder_alternative"] == "unresolved (validity)"
        and d["part_a"] == "unresolved (validity)"
        and d["step_4_part_b_eligibility"] == "unresolved (validity)"
        and d["funding"] == "unresolved (validity)"
    )


def test_invalid_part_b_overrides_ladder_alternative():
    d = pa.decision(VALID, _summary(40, 40), ALT, part_b=_invalid_b())
    assert _is_validity_unresolved(d)
    assert d["gate"] != "alternative"
    assert any("C invalid" in p for p in d["validity"]["problems"]["part_b_inference_and_controls"])


def test_invalid_part_b_overrides_rotation_go():
    d = pa.decision(VALID, _summary(40, 40), _summary(0, 0), part_b=_invalid_b())
    assert _is_validity_unresolved(d)


def test_invalid_part_b_when_part_a_would_be_unresolved():
    d = pa.decision(VALID, _summary(10, 20), _summary(0, 0), part_b=_invalid_b())
    assert _is_validity_unresolved(d)
    assert d["part_a"] == "unresolved (validity)"  # not the step-3 'interval includes 0.5'


def _bad_interval(kind):
    b = PB()
    iv = b["intervals"]["C"]
    if kind == "missing":
        del iv["joint"]
    elif kind == "nan":
        iv["conditional"] = [float("nan"), 0.3]
    elif kind == "inf":
        iv["joint"] = [0.01, float("inf")]
    elif kind == "reversed":
        iv["conditional"] = [0.3, 0.01]
    elif kind == "malformed":
        iv["joint"] = [0.01]
    elif kind == "string":
        iv["joint"] = "0.01..0.3"
    elif kind == "control_nan":
        b["intervals"]["C_rand"]["joint"] = [float("nan"), 0.5]
    elif kind == "point_nan":
        b["points"]["C"] = float("nan")
    elif kind == "point_missing":
        del b["points"]["C_abl"]
    return b


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "nan",
        "inf",
        "reversed",
        "malformed",
        "string",
        "control_nan",
        "point_nan",
        "point_missing",
    ],
)
def test_malformed_part_b_fails_validity_despite_true_flags(kind):
    b = _bad_interval(kind)
    assert all(b["valid"].values())
    for lad in (ALT, _summary(0, 0)):
        d = pa.decision(VALID, _summary(40, 40), lad, part_b=b)
        assert _is_validity_unresolved(d), kind
        assert d["funding"] != "screen"


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda b: b["diagnostics"]["C"]["joint"].update(failed_draws=3), "failed draws"),
        (lambda b: b["diagnostics"]["C_abl"]["conditional"].update(draws=3999), "draws"),
        (
            lambda b: b["diagnostics"]["C_perm"]["joint"].update(min_distinct_components=1),
            "distinct components",
        ),
        (lambda b: b["diagnostics"].pop("C_rand"), "diagnostics missing"),
        (lambda b: b.update(destruction_controls_eligible=False), "destruction controls"),
        (lambda b: b.pop("destruction_controls_eligible"), "destruction controls"),
        (lambda b: b.update(evidence=[]), "Part B evidence"),
        (lambda b: b.update(power_at_0_15={"conditional": float("nan"), "joint": 0.9}), "power"),
    ],
)
def test_part_b_diagnostics_and_eligibility_enter_step_1(mutate, needle):
    b = PB()
    mutate(b)
    d = pa.decision(VALID, _summary(40, 40), ALT, part_b=b)
    assert _is_validity_unresolved(d)
    probs = d["validity"]["problems"]["part_b_inference_and_controls"]
    assert any(needle in p for p in probs), probs


def test_missing_part_b_is_not_evidence_of_eligibility():
    d = pa.decision(VALID, _summary(40, 40), _summary(0, 0))
    assert _is_validity_unresolved(d)
    assert d["part_b_reporting"]["status"] == "not supplied to this tool"
    assert "not evidence of eligibility" in d["part_b_reporting"]["note"]


@pytest.mark.parametrize(
    "evidence",
    [None, [], "synthetic", [{"path": "x"}], [{"path": "", "sha256": "0" * 64}], [{"sha256": "z"}]],
)
def test_missing_or_malformed_evidence_does_not_certify_step_1(evidence):
    v = copy.deepcopy(VALID)
    if evidence is None:
        del v["qualification"]["evidence"]
    else:
        v["qualification"]["evidence"] = evidence
    d = pa.decision(v, _summary(40, 40), _summary(0, 0), part_b=PB())
    assert _is_validity_unresolved(d)
    assert "evidence" in d["validity"]["problems"]["qualification"]


def test_evidence_must_link_required_pins():
    bindings = {"registration_pinned": ["1" * 64]}
    d = pa.decision(VALID, _summary(40, 40), _summary(0, 0), part_b=PB(), bindings=bindings)
    assert _is_validity_unresolved(d)
    assert "required pin" in d["validity"]["problems"]["registration_pinned"]
    v = copy.deepcopy(VALID)
    v["registration_pinned"]["evidence"].append({"path": "pin", "sha256": "1" * 64})
    d = pa.decision(v, _summary(40, 40), _summary(0, 0), part_b=PB(), bindings=bindings)
    assert d["gate"] == "Part A go"


def test_g2_predecessor_failure_is_hard_stop():
    v = copy.deepcopy(VALID)
    v["g2_predecessor_audit"]["pass"] = False
    d = pa.decision(v, _summary(40, 40), _summary(0, 0), part_b=PB())
    assert _is_validity_unresolved(d)
    assert d["validity"]["g2_hard_stop"] is True
    assert "hard stop of G2 pending a new amendment" in d["scientific"]
    assert "hard stop" in d["gate"]
    assert d["validity"]["problems"]["g2_predecessor_audit"] == pa.G2_HARD_STOP


def test_internal_check_failure_enters_validity():
    internal = {"null_size_report": {"pass": False, "reason": "not PASS"}}
    d = pa.decision(VALID, _summary(40, 40), ALT, part_b=PB(), internal=internal)
    assert _is_validity_unresolved(d)


def test_part_b_reported_on_every_branch():
    b = PB()
    branches = {
        "go": pa.decision(VALID, _summary(40, 40), _summary(0, 0), part_b=b),
        "part_a_unresolved": pa.decision(VALID, _summary(10, 20), _summary(0, 0), part_b=b),
        "alternative": pa.decision(VALID, _summary(40, 40), ALT, part_b=b),
        "validity": pa.decision(None, _summary(40, 40), _summary(0, 0), part_b=b),
        "funding_unresolved": pa.decision(
            VALID, _summary(40, 40), _summary(0, 0), part_b=PB(ctrl_lo=-0.01)
        ),
    }
    assert branches["go"]["gate"] == "Part A go"
    assert branches["part_a_unresolved"]["gate"] == "Part A unresolved"
    assert branches["alternative"]["gate"] == "alternative"
    assert branches["validity"]["gate"] == "unresolved (validity)"
    assert branches["funding_unresolved"]["funding"].startswith("unresolved")
    for name, d in branches.items():
        pbr = d["part_b_reporting"]
        assert pbr["status"] == "supplied", name
        assert pbr["computed_by_this_tool"] is False
        assert pbr["supplied"]["intervals"] == (
            b["intervals"] if name != "funding_unresolved" else PB(ctrl_lo=-0.01)["intervals"]
        )
    # an invalid Part B is still reported, never suppressed
    bad = _invalid_b()
    d = pa.decision(VALID, _summary(40, 40), _summary(0, 0), part_b=bad)
    assert d["part_b_reporting"]["supplied"] is bad
    assert d["part_b_reporting"]["validity"]["status"] == "invalid"


def test_funding_equalities_and_independent_control_failures():
    go = (VALID, _summary(40, 40), _summary(0, 0))
    assert pa.decision(*go, part_b=PB(c=0.15))["funding"] == "go"  # C exactly 0.15 meets
    assert pa.decision(*go, part_b=PB(c=0.15, lo=0.0))["funding"] == "screen"  # touching zero
    nogo = PB(c=0.0, lo=-0.1, hi=0.04, power=0.80)  # power exactly 0.80 meets
    assert pa.decision(*go, part_b=nogo)["funding"] == "no-go"
    nogo = PB(c=0.0, lo=-0.1, hi=0.04, power=0.7999)
    assert pa.decision(*go, part_b=nogo)["funding"] == "unresolved"
    no_power = PB(c=0.0, lo=-0.1, hi=0.04)
    del no_power["power_at_0_15"]
    assert pa.decision(*go, part_b=no_power)["funding"] == "unresolved"
    # conditional-only and joint-only control failures, each alone, block funding (step 4)
    cond_only = PB(ctrl_lo=0.0, ctrl_joint_lo=0.02)
    joint_only = PB(ctrl_lo=0.02, ctrl_joint_lo=0.0)
    for b, proc in ((cond_only, "conditional"), (joint_only, "joint")):
        d = pa.decision(*go, part_b=b)
        assert d["step_1_validity"] == "pass" and d["part_a"] == "go"
        assert d["funding"].startswith("unresolved") and proc in d["funding"]
        assert d["step_4_part_b_eligibility"].startswith("not eligible")
    one = PB()
    one["intervals"]["C_abl"]["joint"] = [0.0, 0.5]  # C_abl alone, joint alone
    assert "C_abl joint" in pa.decision(*go, part_b=one)["funding"]


def test_validity_first_in_full_analysis(manifest, nullsize):
    ids = manifest["edit_ids"]
    ledger, fields, _ = build(manifest, ladder_pred={i: -0.3 for i in ids})
    rep = analyse(manifest, nullsize, ledger, fields, validity="auto", part_b=_invalid_b())
    dec = rep["registered"]["decision"]
    assert rep["registered"]["contrasts"]["ladder_0.25"]["agreements"] == 0
    assert _is_validity_unresolved(dec)
    assert rep["registered"]["part_b"]["supplied"]["valid"]["C"] is False
    rep = analyse(manifest, nullsize, ledger, fields, validity="auto", part_b="ok")
    assert rep["registered"]["decision"]["gate"] == "alternative"
    assert rep["registered"]["part_b"]["status"] == "supplied"


# ------------------------------------------------------------------------------ (2) spec hash


def test_f10b_note_records_the_spec_hash():
    assert pa.f10b_spec_hash(REGISTRATION) == pa.SPEC_SHA256
    chk = pa.verify_spec(SPEC, REGISTRATION, pa.SPEC_SHA256)
    assert chk["verified"] and chk["sha256_file"] == chk["sha256_f10b_note"] == pa.SPEC_SHA256


def test_spec_missing_changed_or_wrong_supplied_hash_refused(tmp_path):
    with pytest.raises(pa.AnalysisRefused, match="missing"):
        pa.verify_spec(tmp_path / "absent.md", REGISTRATION)
    changed = tmp_path / "spec.md"
    changed.write_bytes(SPEC.read_bytes() + b"\n")
    with pytest.raises(pa.AnalysisRefused, match="differs from the F10b.1"):
        pa.verify_spec(changed, REGISTRATION)
    with pytest.raises(pa.AnalysisRefused, match="--spec-sha256"):
        pa.verify_spec(SPEC, REGISTRATION, "0" * 64)


def test_registration_note_hash_mismatch_or_absent_refused(tmp_path):
    reg = tmp_path / "prereg.md"
    text = REGISTRATION.read_text()
    reg.write_text(text.replace(f"**sha256 `{pa.SPEC_SHA256}`**", "**sha256 `" + "1" * 64 + "`**"))
    with pytest.raises(pa.AnalysisRefused, match="F10b.1 records"):
        pa.verify_spec(SPEC, reg)
    reg.write_text(text.replace("**F10b.1", "**F10x.1"))
    with pytest.raises(pa.AnalysisRefused, match="F10b.1"):
        pa.verify_spec(SPEC, reg)


def test_analyse_refuses_unverified_spec(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    with pytest.raises(pa.AnalysisRefused, match="specification"):
        analyse(manifest, nullsize, ledger, fields, spec_check={"verified": False})
    rep = analyse(manifest, nullsize, ledger, fields)  # default: hashes the file at run time
    v = rep["spec"]["verification"]
    assert v["sha256_file"] == v["sha256_f10b_note"] == pa.SPEC_SHA256


def _cli_args(tmp_path, led, fld, *extra):
    return [
        "--ledger",
        str(led),
        "--fields",
        str(fld),
        "--output",
        str(tmp_path / "r.json"),
        "--markdown",
        str(tmp_path / "r.md"),
        *extra,
    ]


def test_cli_refuses_spec_problems_and_writes_nothing(tmp_path, manifest, capsys):
    led, fld = _write_inputs(tmp_path, manifest)
    changed = tmp_path / "spec.md"
    changed.write_bytes(SPEC.read_bytes().replace(b"F10.5", b"F10.6", 1))
    cases = [
        ["--spec-sha256", pa.SPEC_SHA256, "--spec", str(tmp_path / "missing.md")],
        ["--spec-sha256", pa.SPEC_SHA256, "--spec", str(changed)],
        ["--spec-sha256", "0" * 64],
    ]
    for extra in cases:
        assert pa.main(_cli_args(tmp_path, led, fld, *extra)) == 2
        assert "refused" in capsys.readouterr().err
        assert not (tmp_path / "r.json").exists() and not (tmp_path / "r.md").exists()
    with pytest.raises(SystemExit) as e:  # --spec-sha256 is required
        pa.main(_cli_args(tmp_path, led, fld))
    assert e.value.code == 2
    assert not (tmp_path / "r.json").exists()


# ------------------------------------------------------------------------------ (3) windows, re-stage


@pytest.mark.parametrize(
    "mutate",
    [
        lambda L: L.update(window_ids=L["window_ids"][::-1]),  # reversed
        lambda L: L["window_ids"].__setitem__(5, "64"),  # substituted
        lambda L: L["window_ids"].pop(),  # missing
        lambda L: L["window_ids"].__setitem__(1, "0"),  # duplicate
        lambda L: L.update(window_ids=[int(x) for x in L["window_ids"]]),  # wrong type
        lambda L: L["window_ids"].__setitem__(slice(0, 2), ["1", "0"]),  # swapped in-block
        lambda L: L["two_window_blocks"].__setitem__(0, [0, 2]),  # wrong blocks
        lambda L: L.pop("two_window_blocks"),
        lambda L: L["window_token_ids_hashes"].__setitem__(3, "f" * 64),  # scoring identity
        lambda L: L.pop("window_token_ids_hashes"),
        lambda L: L.update(text_groups_sha256="0" * 64),
        lambda L: L.pop("text_groups_sha256"),
    ],
)
def test_window_order_and_blocks_must_match_text_groups(manifest, nullsize, mutate):
    ledger, fields, _ = build(manifest)
    mutate(ledger)
    with pytest.raises(pa.AnalysisRefused, match="window|block|text_groups|token"):
        analyse(manifest, nullsize, ledger, fields)


def test_text_groups_file_tamper_refused(tmp_path):
    bad = tmp_path / "tg.json"
    bad.write_bytes(TEXT_GROUPS.read_bytes().replace(b'"item_id": "0"', b'"item_id": "00"'))
    with pytest.raises(pa.AnalysisRefused, match="text_groups.json sha256"):
        pa.load_text_groups(bad)
    tg = pa.load_text_groups(TEXT_GROUPS)
    assert tg["window_ids"] == [str(k) for k in range(64)]
    assert tg["two_window_blocks"][0] == [0, 1] and tg["two_window_blocks"][-1] == [62, 63]


def _restage_status(manifest, nullsize, ledger, fields, i):
    rep = analyse(manifest, nullsize, ledger, fields)
    return rep["registered"]["pairs"][i]["rotation"], rep


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda r: r.clear(), "no re-stage entry"),
        (lambda r: r.pop("f10_3_verdict"), "lacks ['f10_3_verdict']"),
        (lambda r: r.update(f10_3_verdict="ok"), "verdict 'ok'"),
        (lambda r: r.pop("diagnostics"), "lacks ['diagnostics']"),
        (lambda r: r.update(diagnostics={}), "diagnostics empty"),
        (lambda r: r.pop("f10_3_exclusions"), "lacks ['f10_3_exclusions']"),
        (lambda r: r.pop("pre_audit_exclusions"), "lacks ['pre_audit_exclusions']"),
        (lambda r: r.pop("registered_realised_verdict"), "registered_realised_verdict"),
        (lambda r: r.update(registered_realised_verdict="no"), "not a boolean"),
        (lambda r: r.pop("variant_json_sha256"), "variant_json_sha256"),
        (lambda r: r.update(variant_json_sha256="abc"), "not a sha256"),
        (lambda r: r.update(f10_3_exclusions="sub_resolution"), "not a list"),
    ],
)
def test_incomplete_restage_entry_is_audit_missing(manifest, nullsize, mutate, needle):
    ledger, fields, _ = build(manifest)
    i = min(ledger["restage"])
    mutate(ledger["restage"][i])
    rec, rep = _restage_status(manifest, nullsize, ledger, fields, i)
    assert rec["status"] == "f10_3_excluded" and rec["reasons"] == ["audit_missing"]
    assert needle in rec["detail"]
    led = rep["registered"]["contrasts"]["rotation"]["ledger"]
    assert led["f10_3_excluded"]["by_code"] == {"audit_missing": [i]}
    assert rep["f10_6_calibration"]["excluded_ids"] == [i]


def test_missing_restage_provenance_fails_every_pair_closed(manifest, nullsize):
    for mutate in (
        lambda L: L.pop("restage_provenance"),
        lambda L: L["restage_provenance"].pop("audit_file_sha256"),
        lambda L: L["restage_provenance"].update(readme_sha256="not-a-hash"),
    ):
        ledger, fields, _ = build(manifest)
        mutate(ledger)
        rep = analyse(manifest, nullsize, ledger, fields)
        led = rep["registered"]["contrasts"]["rotation"]["ledger"]
        assert led["f10_3_excluded"]["by_code"] == {"audit_missing": sorted(manifest["edit_ids"])}
        assert rep["registered"]["contrasts"]["rotation"]["resolved"] == 0
        assert rep["restage"]["yield"]["passing"] == 0
        # ladders unaffected
        assert rep["registered"]["contrasts"]["ladder_0.25"]["resolved"] == 40


def test_restage_hashes_must_be_linked_in_validity_evidence(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    v = valid_for(ledger)
    v["selection_and_audits"]["evidence"] = v["selection_and_audits"]["evidence"][:2]
    rep = analyse(manifest, nullsize, ledger, fields, validity=v, part_b="ok")
    dec = rep["registered"]["decision"]
    assert _is_validity_unresolved(dec)
    assert PROV["readme_sha256"] in dec["validity"]["problems"]["selection_and_audits"]
    v = valid_for(ledger)
    v["registration_pinned"]["evidence"] = [
        e for e in v["registration_pinned"]["evidence"] if e["sha256"] != ledger["fields_sha256"]
    ]
    rep = analyse(manifest, nullsize, ledger, fields, validity=v, part_b="ok")
    assert "registration_pinned" in rep["registered"]["decision"]["validity"]["problems"]
    rep = analyse(manifest, nullsize, ledger, fields, validity="auto", part_b="ok")
    assert rep["registered"]["decision"]["gate"] == "Part A go"


def test_contradictory_restage_verdict_refused(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    i = min(ledger["restage"])
    ledger["restage"][i]["f10_3_exclusions"] = ["sub_resolution"]  # verdict still "pass"
    with pytest.raises(pa.AnalysisRefused, match="contradicts"):
        analyse(manifest, nullsize, ledger, fields)
    ledger, fields, _ = build(manifest)
    ledger["restage"][i]["f10_3_verdict"] = "excluded"  # no reason code
    with pytest.raises(pa.AnalysisRefused, match="contradicts"):
        analyse(manifest, nullsize, ledger, fields)


def test_complete_case_exclusion_is_not_an_audit_failure(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(manifest, drop_covariate={ids[0]: "ln_rho"})
    rep = analyse(manifest, nullsize, ledger, fields)
    assert rep["registered"]["pairs"][ids[0]]["rotation"]["status"] == "resolved"
    cov = rep["f10_5_sensitivity"]["covariate"]
    assert [d["id"] for d in cov["dropped_rows"]] == [ids[0]]
    assert ids[0] in cov["resolved_ids"] and ids[0] not in cov["complete_case_ids"]
    assert rep["f10_6_calibration"]["status"] == "not required (40/40)"


# ------------------------------------------------------------------------------ (4) report contents


def test_every_pair_listed_with_status_and_cause(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(
        manifest,
        excluded={ids[0]: "replay_failed"},
        missing={ids[1]: "original"},
        d_mean={ids[2]: 0.0},
        noise=2e-3,
    )
    rep = analyse(manifest, nullsize, ledger, fields)
    for c in pa.CONTRASTS:
        row = rep["registered"]["contrasts"][c]
        assert [p["id"] for p in row["pair_status"]] == ids
        assert all(p["status"] and p["detail"] for p in row["pair_status"])
        causes = row["ledger"]["unresolved"]["causes"]
        assert set(causes) == set(row["ledger"]["unresolved"]["ids"])
        assert all("does not strictly exclude zero" in x for x in causes.values())
    rot = rep["registered"]["contrasts"]["rotation"]
    assert ids[2] in rot["ledger"]["unresolved"]["causes"]
    status = {p["id"]: p for p in rot["pair_status"]}
    assert status[ids[0]]["detail"] == "f10_3_excluded: replay_failed"
    assert status[ids[1]]["detail"] == "missing or nonfinite score: original"
    assert sorted(rot["agreement_ids"] + rot["disagreement_ids"]) == rot["resolved_ids"]


def test_sign_tables_per_contrast_and_family(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(manifest, predictions={ids[0]: 0.0, ids[1]: -0.2})
    rep = analyse(manifest, nullsize, ledger, fields)
    sel = {s["edit_id"]: s["family"] for s in manifest["selections"]}
    for c in pa.CONTRASTS:
        row = rep["registered"]["contrasts"][c]
        assert set(row["sign_table"]) == {"negative", "zero", "positive"}
        for s in pa.SIGNS:
            cell = row["sign_table"][s]
            assert cell["n"] == cell["agree"] + cell["disagree"] == row["predicted_sign_counts"][s]
        fams = row["by_family"]
        assert set(fams) == {"lora_sft", "chain"}
        assert fams["lora_sft"]["n_selected"] == 30 and fams["chain"]["n_selected"] == 10
        for s in pa.SIGNS:
            assert (
                fams["lora_sft"]["predicted_sign_counts"][s]
                + fams["chain"]["predicted_sign_counts"][s]
                == row["predicted_sign_counts"][s]
            )
    rot = rep["registered"]["contrasts"]["rotation"]
    assert rot["predicted_sign_counts"] == {"negative": 1, "zero": 1, "positive": 38}
    fam0 = sel[ids[0]]
    assert rot["by_family"][fam0]["zero_prediction_ids"] == [ids[0]]


def test_covariate_and_subset_id_accounting(manifest, nullsize):
    in_s30 = "0027efdf2eae7405"
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(
        manifest,
        excluded={in_s30: "replay_failed"},
        predictions={"014f8a9ec78db2be": 0.0},
        drop_covariate={ids[-1]: "d_mean"},
    )
    rep = analyse(manifest, nullsize, ledger, fields)
    cov = rep["f10_5_sensitivity"]["covariate"]
    assert cov["selected_ids"] == ids
    assert in_s30 not in cov["resolved_ids"]
    assert cov["complete_case_ids"] == cov["complete_case_agreement"]["ids"]
    assert cov["complete_case_agreement"]["ledger"]["selected"]["n"] == cov["n_cc"]
    assert cov["zero_prediction_disagreements"] == 1
    assert cov["zero_prediction_ids"] == ["014f8a9ec78db2be"]
    assert sorted(cov["agreement_ids"] + cov["disagreement_ids"]) == cov["complete_case_ids"]
    assert cov["full_analysis_set_ledger"]["f10_3_excluded"]["ids"] == [in_s30]
    assert set(cov["units"]) >= {"coefficients", "ln_rho"}
    s30 = rep["f10_5_sensitivity"]["subsets"]["S30"]
    assert s30["ledger"]["selected"]["n"] == 30
    assert s30["ledger"]["f10_3_excluded"]["by_code"] == {"replay_failed": [in_s30]}
    assert s30["zero_prediction_ids"] == ["014f8a9ec78db2be"]
    assert s30["zero_prediction_disagreements"] == 1
    assert sorted(s30["agreement_ids"] + s30["disagreement_ids"]) == s30["resolved_ids"]
    assert len(s30["pair_status"]) == 30


def test_markdown_carries_all_required_tables(manifest, nullsize):
    low = "15a53586efc72a50"
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(
        manifest, restage_rho={low: math.exp(0.05)}, predictions={ids[3]: 0.0}
    )
    rep = analyse(manifest, nullsize, ledger, fields, validity="auto", part_b="ok")
    md = pa.render_markdown(rep)
    n_tables = md.count("| predicted sign (3 × 2) |")
    # 3 contrasts × (pooled + 2 families) + covariate θ_raw + 12 tertile cells + 2 subsets
    assert n_tables == 9 + 1 + 12 + 2
    cuts = rep["f10_5_sensitivity"]["tertiles"]["lnrho_cuts"]
    assert f"cut 1 = ({cuts[0][0]!r}, `{cuts[0][1]}`)" in md
    assert "| frozen member (order) | staged ln ρ | re-stage ln ρ | cell_drift |" in md
    for cell in pa.LNRHO_CELLS:
        for i in cell:
            assert f"| `{i}` | " in md
    assert f"| `{low}` | " in md and "yes (not moved)" in md
    cov = rep["f10_5_sensitivity"]["covariate"]
    for text in cov["rules"].values():
        assert text in md
    assert cov["uncertainty"] in md
    assert "log-odds per training-SD of the column" in md
    assert "#### Fit diagnostics (full fits)" in md and "#### Bootstrap diagnostics" in md
    assert "#### Leave-one-pair-out folds" in md and "Per-fold standardisation parameters" in md
    assert "- zero predictions (" in md  # subset zero-prediction accounting
    assert "### Part B (reported on every branch)" in md
    assert "Specification hash verified at run time" in md
    assert "### Pair status (every pair, every contrast)" in md
    for i in ids:
        assert f"| `{i}` | resolved" in md
    assert pa.F10_6_ALL_PASS in md


def test_full_fit_diagnostics_recorded():
    ids, X, a = _rows(20, seed=4, p_agree=0.6)
    out = pa.covariate_models(ids, X, a, draws=DRAWS)
    for m in ("M0", "M1"):
        fit = out[m]["fit"]
        assert fit["converged"] is True and fit["iterations"] >= 1
        assert fit["gradient_max_abs"] <= 1e-10
        cols = out[m]["columns"]
        st = fit["standardisation"]
        assert list(st["mean"]) == cols and st["n_train"] == 20
        Xs = X[np.argsort(ids)][:, : len(cols)]
        np.testing.assert_allclose(list(st["mean"].values()), Xs.mean(axis=0))
        np.testing.assert_allclose(list(st["sd"].values()), Xs.std(axis=0, ddof=0))
        bd = out[m]["bootstrap_diagnostics"]
        assert bd["draws"] == DRAWS and bd["iterations_max"] >= 1
        assert bd["gradient_max_abs_max"] <= 1e-10


def test_loo_held_out_values_are_deterministic_train_only():
    ids, X, a = _rows(16, seed=2)
    out = pa.covariate_models(ids, X, a, draws=3)
    j = 3
    fold = out["loo"]["M1"]["fold_records"][j]
    assert fold["held_out"] == ids[j] and fold["status"] == "fitted"
    tr = np.arange(16) != j
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0, ddof=0)
    z = (X[j] - mu) / sd
    np.testing.assert_allclose(list(fold["z_held_out"].values()), z, rtol=1e-12)
    np.testing.assert_allclose(list(fold["standardisation"]["mean"].values()), mu)
    np.testing.assert_allclose(list(fold["standardisation"]["sd"].values()), sd)
    Zt = (X[tr] - mu) / sd
    b0, b, _ = pa.fit_ridge_logistic(Zt, a[tr])
    p = 1 / (1 + math.exp(-(b0 + z @ b)))
    assert fold["p_held_out"] == pytest.approx(p, abs=1e-12)
    # the held-out row never enters its own fold's moments
    assert not np.allclose(mu, X.mean(axis=0))
    io = out["loo"]["intercept_only"]["fold_records"][j]
    assert io["p_held_out"] == pytest.approx(a[tr].mean())


def test_loo_one_minority_row_fold_fails():
    ids, X, _ = _rows(16, seed=6)
    a = np.ones(16)
    a[7] = 0.0  # one minority row: its LOO training fold is one-class
    out = pa.covariate_models(ids, X, a, draws=DRAWS)
    assert out["M0"]["status"] == "estimated"
    for m in ("intercept_only", "M0", "M1"):
        loo = out["loo"][m]
        assert loo["failed_folds"] == 1
        failed = [f for f in loo["fold_records"] if f["status"].startswith("failed")]
        assert [f["held_out"] for f in failed] == [ids[7]]
        assert "one class" in failed[0]["status"]
    reasons = out["M0"]["bootstrap_diagnostics"]["failure_reasons"]
    assert reasons.get("one class in the draw", 0) == out["M0"]["failed_draws"] > 0


def test_degenerate_columns_recorded_in_loo_and_bootstrap():
    ids, X, a = _rows(16, seed=7)
    X[:, 3] = 0.0
    X[5, 3] = 1.0  # d_mean is constant whenever row 5 is absent
    out = pa.covariate_models(ids, X, a, draws=DRAWS)
    col = pa.COVARIATE_COLUMNS[3]
    for m in ("M0", "M1"):
        assert out["loo"][m]["degenerate_column_events"][col] == 1
        fold5 = out["loo"][m]["fold_records"][5]
        assert col in fold5["standardisation"]["degenerate_columns"]
        assert fold5["z_held_out"][col] == 0.0
        bd = out[m]["bootstrap_diagnostics"]
        expected = int(sum(5 not in row for row in pa.bootstrap_indices(16, DRAWS)))
        assert bd["degenerate_column_events"][col] == expected > 0
        assert bd["fits_with_degenerate_columns"] >= expected


def test_non_convergence_is_reported_not_retried(monkeypatch):
    ids, X, a = _rows(20, seed=4, p_agree=0.6)
    monkeypatch.setattr(pa, "NEWTON_MAX_ITER", 1)
    out = pa.covariate_models(ids, X, a, draws=20)
    for m in ("M0", "M1"):
        assert out[m]["status"].startswith("not estimable: full fit failed (no convergence")
        fit = out[m]["fit"]
        assert fit["converged"] is False and fit["iterations"] == 1
        assert fit["gradient_max_abs"] > 1e-10
        assert "standardisation" in fit
        bd = out[m]["bootstrap_diagnostics"]
        assert bd["failure_reasons"].get("no convergence in 1 iterations", 0) > 0
        assert out["loo"][m]["failed_folds"] == 20
    assert out["delta_theta"] == {"status": "not estimable"}


def test_forced_fit_failure_counts_every_draw_and_fold(monkeypatch):
    ids, X, a = _rows(18, seed=8)

    def boom(Z, a, *args, **kw):
        raise pa.FitFailure("forced", {"iterations": 0, "gradient_max_abs": 1.0})

    monkeypatch.setattr(pa, "fit_ridge_logistic", boom)
    out = pa.covariate_models(ids, X, a, draws=25)
    for m in ("M0", "M1"):
        assert out[m]["status"] == "not estimable: full fit failed (forced)"
        assert out[m]["bootstrap_diagnostics"]["failed_draws"] == 25
        assert out["loo"][m]["failed_folds"] == 18
        assert all(f["status"] == "failed: forced" for f in out["loo"][m]["fold_records"])
    assert out["loo"]["intercept_only"]["failed_folds"] == 0
    md_rep = {"models": out}
    assert md_rep["models"]["M1"]["fit"]["reason"] == "forced"


# ------------------------------------------------------------------------------ (5) F10.6 masked


def test_exclusion_without_masked_calibration_refused(manifest, nullsize):
    ledger, fields, _ = build(manifest, excluded={min(manifest["edit_ids"]): "sub_resolution"})
    with pytest.raises(pa.AnalysisRefused, match="F10.6"):
        analyse(manifest, nullsize, ledger, fields, masked_calibration=None)


def test_masked_row_added_without_replacement(manifest, nullsize):
    i = min(manifest["edit_ids"])
    ledger, fields, _ = build(manifest, excluded={i: "sub_resolution"})
    masked, sha, _ = masked_for(ledger, manifest)
    rep = analyse(
        manifest,
        nullsize,
        ledger,
        fields,
        masked_calibration=masked,
        masked_calibration_sha256=sha,
        validity=valid_for(ledger, sha),
        part_b="ok",
    )
    rows = rep["f5_7_null_branch_table"]["rows"]
    base = pa.f57_table(nullsize)["rows"]
    assert len(rows) == 6 and rows[:5] == base
    assert [r["scenario"] for r in rows[:5]] == list(pa.F57_SCENARIOS)
    assert rows[5]["scenario"] == "slot_masked_null_size"
    assert rows[5]["label"].startswith("slot-masked (F10.6)")
    assert rows[5]["alternative_count"] == 1800 and rows[5]["replicates"] == 10000
    f106 = rep["f10_6_calibration"]
    assert f106["required"] and f106["verdict"] == "PASS" and f106["passed"] is True
    assert f106["sha256"] == sha and f106["slot_mask_file_sha256"] == "ab" * 32
    assert f106["masked_ids"]["rotation"] == [i]
    assert f106["restage_yield"] == {"passing": 39, "selected": 40}
    assert rep["registered"]["decision"]["step_1_validity"] == "pass"
    md = pa.render_markdown(rep)
    assert "slot-masked (F10.6)" in md and sha in md and "verdict **PASS**" in md


@pytest.mark.parametrize("size_ok,power_ok", [(False, True), (True, False)])
def test_failed_masked_calibration_makes_step_1_unresolved(manifest, nullsize, size_ok, power_ok):
    ledger, fields, _ = build(manifest, excluded={min(manifest["edit_ids"]): "replay_failed"})
    masked, sha, _ = masked_for(ledger, manifest, passed=size_ok, power_passed=power_ok)
    rep = analyse(
        manifest,
        nullsize,
        ledger,
        fields,
        masked_calibration=masked,
        masked_calibration_sha256=sha,
        validity=valid_for(ledger, sha),
        part_b="ok",
    )
    dec = rep["registered"]["decision"]
    assert _is_validity_unresolved(dec)
    assert dec["validity"]["problems"]["f10_6_calibration"] == pa.F10_6_FAILED
    assert rep["f10_6_calibration"]["verdict"] == "FAIL"
    assert pa.F10_6_FAILED in rep["f10_6_calibration"]["statement"]
    assert len(rep["f5_7_null_branch_table"]["rows"]) == 6  # still disclosed


def test_masked_calibration_must_match_exclusions(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _ = build(manifest, excluded={ids[0]: "sub_resolution"})
    masked, sha, _ = masked_for(ledger, manifest)
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["masked_ids"]["rotation"] = [ids[1]]
    with pytest.raises(pa.AnalysisRefused, match="masked ids"):
        analyse(
            manifest,
            nullsize,
            ledger,
            fields,
            masked_calibration=bad,
            masked_calibration_sha256=sha,
        )
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["mapping_sha256"] = "0" * 64
    with pytest.raises(pa.AnalysisRefused, match="mapping sha256"):
        analyse(
            manifest,
            nullsize,
            ledger,
            fields,
            masked_calibration=bad,
            masked_calibration_sha256=sha,
        )
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["label"] = "unmasked"
    with pytest.raises(pa.AnalysisRefused, match="slot-masked"):
        analyse(
            manifest,
            nullsize,
            ledger,
            fields,
            masked_calibration=bad,
            masked_calibration_sha256=sha,
        )
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["gates"]["passed"] = True
    bad["slot_mask"]["gates"]["size"]["passed"] = False
    with pytest.raises(pa.AnalysisRefused, match="gates"):
        analyse(
            manifest,
            nullsize,
            ledger,
            fields,
            masked_calibration=bad,
            masked_calibration_sha256=sha,
        )
    with pytest.raises(pa.AnalysisRefused, match="sha256"):
        analyse(
            manifest,
            nullsize,
            ledger,
            fields,
            masked_calibration=masked,
            masked_calibration_sha256=None,
        )


def test_masked_hash_must_be_linked_in_null_size_evidence(manifest, nullsize):
    ledger, fields, _ = build(manifest, excluded={min(manifest["edit_ids"]): "sub_resolution"})
    masked, sha, _ = masked_for(ledger, manifest)
    rep = analyse(
        manifest,
        nullsize,
        ledger,
        fields,
        masked_calibration=masked,
        masked_calibration_sha256=sha,
        validity=valid_for(ledger),  # links the NULL-SIZE report but not the masked run
        part_b="ok",
    )
    dec = rep["registered"]["decision"]
    assert _is_validity_unresolved(dec)
    assert sha in dec["validity"]["problems"]["null_size"]


def test_full_yield_states_simulations_stand_and_refuses_masked_run(manifest, nullsize):
    ledger, fields, _ = build(manifest)
    rep = analyse(manifest, nullsize, ledger, fields)
    f106 = rep["f10_6_calibration"]
    assert f106["required"] is False and f106["statement"] == pa.F10_6_ALL_PASS
    assert "stand without a new run" in f106["statement"]
    assert len(rep["f5_7_null_branch_table"]["rows"]) == 5
    masked, sha, _ = masked_for(ledger, manifest)
    with pytest.raises(pa.AnalysisRefused, match="all 40 pairs pass"):
        analyse(
            manifest,
            nullsize,
            ledger,
            fields,
            masked_calibration=masked,
            masked_calibration_sha256=sha,
        )


# ------------------------------------------------------------------------------ CLI end to end


def test_cli_full_inputs_end_to_end(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr(pa, "BOOTSTRAP_DRAWS", DRAWS)
    i = min(manifest["edit_ids"])
    ledger, _fields, fields_bytes = build(manifest, excluded={i: "rounding_bound_failed"})
    _masked, msha, mraw = masked_for(ledger, manifest)
    (tmp_path / "fields.json").write_bytes(fields_bytes)
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "masked.json").write_bytes(mraw)
    (tmp_path / "validity.json").write_text(json.dumps(valid_for(ledger, msha)))
    (tmp_path / "part_b.json").write_text(json.dumps(PB()))
    args = _cli_args(
        tmp_path,
        tmp_path / "ledger.json",
        tmp_path / "fields.json",
        "--spec-sha256",
        pa.SPEC_SHA256,
        "--validity",
        str(tmp_path / "validity.json"),
        "--part-b",
        str(tmp_path / "part_b.json"),
    )
    assert pa.main(args) == 2  # masked calibration required (39/40) but not supplied
    assert not (tmp_path / "r.json").exists()
    assert pa.main(args + ["--masked-calibration", str(tmp_path / "masked.json")]) == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    dec = rep["registered"]["decision"]
    assert dec["step_1_validity"] == "pass" and dec["gate"] == "Part A go"
    assert dec["funding"] == "go"
    assert (
        rep["registered"]["part_b"]["sha256"]
        == hashlib.sha256((tmp_path / "part_b.json").read_bytes()).hexdigest()
    )
    assert rep["inputs"]["masked_calibration"]["sha256"] == msha
    assert rep["inputs"]["spec"]["sha256"] == pa.SPEC_SHA256
    assert rep["inputs"]["spec_sha256_supplied"] == pa.SPEC_SHA256
    assert rep["spec"]["verification"]["sha256_f10b_note"] == pa.SPEC_SHA256
    assert rep["inputs"]["text_groups_sha256"] == pa.TEXT_GROUPS_SHA256
    md = (tmp_path / "r.md").read_text()
    assert "Gate: **Part A go**" in md and msha in md
