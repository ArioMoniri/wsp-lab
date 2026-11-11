"""Synthetic-outcome tests for the Part A analysis (A5.3/A5.5 registered outputs, F10.5 reports).

Every outcome here is synthetic (synthetic ledgers, fields, Part B results and masked-calibration
reports only). The only repository files read are committed, outcome-free metadata: the preserved
a533 Part A staging manifest (the frozen anchor), the committed execution manifest (one binding
smoke test; its hash is not pinned here), the NULL-SIZE report, text_groups.json, the F10.5
specification and the registration text (for the F10b.1 hash and the verbatim texts). Execution
manifests are otherwise synthetic: the frozen manifest plus synthetic F10 verdicts and pooled rho.
Nothing under data/ is opened.
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
FROZEN = ROOT / "docs/prereg/a5_manifests/part_a/manifest_a533.json"
EXECUTION = ROOT / "docs/prereg/a5_manifests/part_a/manifest.json"
NULLSIZE = ROOT / "docs/prereg/a5_manifests/nullsize/report.json"
SPEC = ROOT / "docs/design/F10_SENSITIVITY_SPEC.md"
REGISTRATION = ROOT / "docs/PREREG_H1_TF.md"
TEXT_GROUPS = ROOT / "docs/prereg/a5_manifests/text_groups.json"
DRAWS = 200  # API override for speed; the CLI always uses the registered 4,000


FROZEN_M = pa.load_frozen_manifest(FROZEN)


@pytest.fixture(scope="module")
def manifest():
    """The frozen (a533) staging manifest: registered ids, cuts, cells, staged values."""
    return FROZEN_M


@pytest.fixture(scope="module")
def nullsize():
    return pa.load_nullsize_report(NULLSIZE)


TG = pa.load_text_groups(TEXT_GROUPS)
PROV = {k: hashlib.sha256(k.encode()).hexdigest() for k in pa.RESTAGE_PROVENANCE_KEYS}


def manifest_bytes(m):
    return json.dumps(m, sort_keys=True).encode()


def exec_sha(m):
    """sha256 of the bytes a synthetic execution manifest is written with."""
    return hashlib.sha256(manifest_bytes(m)).hexdigest()


def bind(ledger, execution):
    """Record ``execution``'s hash in the ledger (after a deliberate execution-manifest edit)."""
    h = exec_sha(execution)
    ledger["manifest_sha256"] = h
    if isinstance(ledger.get("restage_provenance"), dict):
        ledger["restage_provenance"]["manifest_sha256"] = h
    return ledger


F11_SELECTION = ROOT / "docs/prereg/a5_manifests/selection_f11.json"
# The frozen F11.2 list (committed, outcome-free metadata): ids, cells and effective norms, in the
# file's cell order (LoRA-high, chain-low, chain-high).
F11_ROWS = json.loads(F11_SELECTION.read_text())["supplements"]
F11_IDS = sorted(r["id"] for r in F11_ROWS)


def supplementary_rows(k, order=None):
    """The first k frozen F11.2 supplementary root edits (synthetic staging): selection rows with
    their cells and norms, and three variants each. ``order`` reorders the twelve first."""
    rows, variants = [], []
    src = [next(r for r in F11_ROWS if r["id"] == i) for i in order] if order else F11_ROWS
    for n, r in enumerate(src[:k]):
        i = r["id"]
        family, tertile = r["cell"].split(":")
        rows.append(
            {
                "component_id": r["component_id"],
                "edit_id": i,
                "effective_norm": r["effective_norm"],
                "family": family,
                "predecessor_id": "base",
                "rank": 16,
                "supplementary": True,
                "tertile": int(tertile),
            }
        )
        for kind, strength, tag in (
            ("rotation", 1.0, "r"),
            ("scale", 0.25, "a"),
            ("scale", 0.5, "b"),
        ):
            variants.append(
                {
                    "excluded": False,
                    "exclusions": [],
                    "kind": kind,
                    "predecessor_id": "base",
                    "source_id": i,
                    "strength": strength,
                    "variant_id": f"e{tag}{n:014x}",
                }
            )
    return rows, variants


def execution_manifest(frozen, codes, rhos, supplementary=0, ladder_excluded=None, order=None):
    """Synthetic F10 execution manifest: the frozen manifest plus F10 verdicts and pooled rho.

    Supplementary edits also get their own staging values (pooled norm mismatch from rho, the two
    subset flags); ``ladder_excluded`` {id: [contrast, ...]} marks staged ladder exclusions."""
    m = dict(frozen)
    rows, variants = supplementary_rows(supplementary, order)
    supp = {r["edit_id"] for r in rows}
    strength = {"ladder_0.25": 0.25, "ladder_0.5": 0.5}
    lad = {(i, strength[c]) for i, cs in (ladder_excluded or {}).items() for c in cs}
    m["selections"] = list(frozen["selections"]) + rows
    m["variants"] = [
        dict(v, excluded=True, exclusions=["construction_failed"])  # a copy: frozen stays intact
        if v["kind"] == "scale" and (v["source_id"], float(v["strength"])) in lad
        else v
        for v in list(frozen["variants"]) + variants
    ]
    m["audit_rule"] = "f10"
    m["edit_audits"] = {}
    for s in m["selections"]:
        i = s["edit_id"]
        code = codes.get(i)
        m["edit_audits"][i] = {
            "audit_rule": "f10",
            "pair_verdict": "excluded" if code else "pass",
            "exclusions": [{"reason": code, "tensor": "model.layers.0.mlp.up_proj.weight"}]
            if code
            else [],
            "edit_audit": {"pooled_energy_ratio": rhos.get(i)},
        }
        if i in supp and rhos.get(i) is not None:
            mismatch = abs(math.sqrt(rhos[i]) - 1.0)
            m["edit_audits"][i]["edit_audit"] |= {
                "pooled_norm_mismatch": mismatch,
                "in_subset_pooled_norm_2e-2": mismatch <= 2e-2,
                "in_subset_noise_calibrated": True,
            }
    return m


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
    supplementary=0,
    ladder_excluded=None,
    order=None,
):
    """Synthetic sealed ledger, fields and execution manifest (no real outcome).

    ``manifest`` is the frozen a533 manifest; the forty registered ids plus ``supplementary``
    synthetic supplementary edits are selected in the returned execution manifest, whose hash the
    ledger records.
    """
    rng = np.random.default_rng(seed)
    d_mean = d_mean or {}
    predictions = predictions or {}
    excluded = excluded or {}
    missing = missing or {}
    restage_rho = restage_rho or {}
    drop_covariate = drop_covariate or {}
    ladder_mean = ladder_mean if ladder_mean is not None else {}
    ladder_pred = ladder_pred if ladder_pred is not None else {}
    frozen_ids = [s["edit_id"] for s in manifest["selections"]]
    supp_ids = [r["edit_id"] for r in supplementary_rows(supplementary, order)[0]]
    ladder_excluded = ladder_excluded or {}
    staged = {
        i: 2 * math.log1p(manifest["edit_audits"][i]["edit_audit"]["pooled_norm_mismatch"])
        for i in frozen_ids
    }
    rhos = {i: restage_rho.get(i, math.exp(staged.get(i, 0.02))) for i in frozen_ids + supp_ids}
    for i, col in drop_covariate.items():
        if col == "ln_rho":
            rhos[i] = None
    execution = execution_manifest(manifest, excluded, rhos, supplementary, ladder_excluded, order)
    ids = sorted(s["edit_id"] for s in execution["selections"])
    sel = {s["edit_id"]: s for s in execution["selections"]}
    var_ids = {
        (v["source_id"], v["kind"], v["strength"]): v["variant_id"] for v in execution["variants"]
    }
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
        rho = rhos[i]
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
        for c in ladder_excluded.get(i, ()):
            variants[i][c] = {
                "variant_id": variants[i][c]["variant_id"],
                "excluded": True,
                "exclusions": ["construction_failed"],
            }
        p = predictions.get(i, 0.5 if d >= 0 else -0.5)
        fields[i] = [p, ladder_pred.get(i, 0.3), ladder_pred.get(i, 0.3)]
        col = drop_covariate.get(i)
        if col == "d_mean":
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
        "manifest_sha256": exec_sha(execution),
        "fields_sha256": hashlib.sha256(fields_bytes).hexdigest(),
        "text_groups_sha256": TG["sha256"],
        "window_ids": windows,
        "two_window_blocks": copy.deepcopy(TG["two_window_blocks"]),
        "window_token_ids_hashes": list(TG["token_ids_hashes"]),
        "restage_provenance": dict(PROV, manifest_sha256=exec_sha(execution)),
        "references": refs,
        "variants": variants,
        "restage": restage,
    }
    return ledger, fields, fields_bytes, execution


def masked_for(
    ledger,
    execution,
    passed=True,
    power_passed=True,
    *,
    power_rate=None,
    size_rate=None,
    two_sided=(0.03, 0.03, 0.03),
):
    """Synthetic ``a5_simulation.py --slot-mask [--supplement]`` output matching the ledger's
    per-contrast exclusions (the simulator's canonical availability mapping)."""
    audit = pa.restage_audit(ledger, execution)
    supp = sorted(s["edit_id"] for s in execution["selections"] if s.get("supplementary"))
    avail = pa.contrast_availability(ledger, audit, audit)
    mapping = hashlib.sha256(json.dumps(avail, sort_keys=True).encode()).hexdigest()
    label = "slot-masked (F10.6); supplemented (F11.4)" if supp else "slot-masked (F10.6)"
    power_rate = power_rate if power_rate is not None else (0.83 if power_passed else 0.7514)
    size_rate = size_rate if size_rate is not None else (0.015 if passed else 0.081)
    rep = {
        "replicates": 10000,
        "seed": 20261107,
        "slot_mask": {
            "label": label,
            "schema": "a5-slot-mask-v1",
            "sha256": "ab" * 32,
            "mapping_sha256": mapping,
            "masked_ids": {c: sorted(i for i in avail if c not in avail[i]) for c in pa.CONTRASTS},
            "slots": [
                {"slot": n, "edit_id": i, "available": avail[i]} for n, i in enumerate(avail)
            ],
            "gates": {
                "label": label,
                "size": {
                    "passed": passed,
                    "ordered_go_rate": size_rate,
                    "two_sided_rejection_rates": list(two_sided),
                },
                "power": {"passed": power_passed, "ordered_go_rate": power_rate},
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
    if supp:
        rep["supplement"] = {
            "label": "supplemented (F11.4)",
            "schema": "a5-f11-supplement-v1",
            "k": len(supp),
            "ids_sorted_sha256": pa.sorted_sha(supp),
            "n_slots": len(avail),
            "sha256": "ce" * 32,
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


def analyse(execution, nullsize, ledger, fields, **kw):
    """API wrapper: synthetic masked calibration when the ledger has exclusions; 'auto' validity.

    ``execution`` is the (synthetic) execution manifest; its sha256 is that of its written bytes
    and the frozen a533 manifest is the verified committed file, unless overridden.
    """
    kw.setdefault("bootstrap_draws", DRAWS)
    kw.setdefault("manifest_sha256", exec_sha(execution))
    kw.setdefault("frozen_manifest", FROZEN_M)
    if "masked_calibration" not in kw:
        audit = pa.restage_audit(ledger, execution)
        supp = any(s.get("supplementary") for s in execution["selections"])
        if supp or any(a["status"] != "pass" for a in audit.values()):
            kw["masked_calibration"], kw["masked_calibration_sha256"], _ = masked_for(
                ledger, execution
            )
    if isinstance(kw.get("validity"), str):
        kw["validity"] = valid_for(ledger, kw.get("masked_calibration_sha256"))
    if kw.get("part_b") == "ok":
        kw["part_b"] = PB()
    return pa.analyse(ledger, execution, fields, nullsize, **kw)


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
    ledger, fields, _, execution = build(manifest)
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
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
    ledger, fields, _, execution = build(manifest, predictions=preds)
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(
        manifest,
        excluded={ids[0]: "sub_resolution", ids[1]: "non_2d"},
        missing={ids[2]: "original", ids[3]: "rotation_nonfinite", ids[4]: "rotation_absent"},
        d_mean={ids[5]: 0.0},
    )
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(manifest, noise=2e-2, seed=5)
    r1 = analyse(execution, nullsize, ledger, fields)
    r2 = analyse(execution, nullsize, ledger, fields)
    assert r1["registered"] == r2["registered"]
    rec = r1["registered"]["pairs"][min(manifest["edit_ids"])]["rotation"]
    assert rec["interval"][0] <= rec["d_hat"] <= rec["interval"][1]
    assert r1["registered"]["resolution"]["draws"] == 4000
    assert r1["registered"]["resolution"]["seed"] == 20261107


def test_ladder_alternative_overrides_rotation_go(manifest, nullsize):
    ids = manifest["edit_ids"]
    # ladder predictions all negative while ladder responses positive -> 0 agreement at s=.25
    ledger, fields, _, execution = build(manifest, ladder_pred={i: -0.3 for i in ids})
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
    dec = rep["registered"]["decision"]
    assert rep["registered"]["contrasts"]["rotation"]["agreements"] == 40
    assert dec["step_2_ladder_alternative"] is True
    assert dec["gate"] == "alternative"
    assert dec["part_a"] == "not reached (alternative overrides rotation go)"


def test_realised_increment_ladder_flags(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(
        manifest, drop_covariate={ids[0]: "ln_rho", ids[1]: "d_mean"}, predictions={ids[2]: -1.0}
    )
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(manifest, restage_rho={low: math.exp(0.05)})
    rep = analyse(execution, nullsize, ledger, fields)
    rows = rep["f10_5_sensitivity"]["tertiles"]["ln_rho"]
    low_row = rows[0]
    assert low in low_row["ids"]  # never moved
    assert low_row["cell_drift"] == [low]
    assert rep["f10_5_sensitivity"]["covariate"]["max_abs_lnrho_restage_minus_staged"] > 0.04


# ------------------------------------------------------------------------------ tertiles and subsets


def test_tertile_rows_labelled_and_complete(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(
        manifest,
        excluded={in_s30: "replay_failed"},
        missing={"00f8111cbad8218f": "original"},
        d_mean={"014f8a9ec78db2be": 0.0},
    )
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(manifest)
    with pytest.raises(pa.AnalysisRefused, match="S30"):
        analyse(execution, nullsize, ledger, fields)


def test_subset_below_floor_label(manifest, nullsize):
    ids = sorted(pa.S30_IDS)[:16]
    ledger, fields, _, execution = build(manifest, excluded={i: "support_failed" for i in ids})
    rep = analyse(execution, nullsize, ledger, fields)
    assert rep["f10_5_sensitivity"]["subsets"]["S30"]["label"] == "below floor (n < 15)"


# ------------------------------------------------------------------------------ report structure


def test_authority_separation_and_texts(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(manifest)
    md = pa.render_markdown(analyse(execution, nullsize, ledger, fields))
    assert pa.LIMITATIONS in md
    assert pa.NO_SUPERIORITY_STATEMENT in md
    assert pa.COVARIATE_SENTENCE in md
    assert "nats per valid next token" in md
    assert "authority: none (descriptive)" in md
    assert md.index("Registered") < md.index("F10.5")
    assert "| predicted sign (3 × 2) | n predicted | agree | disagree |" in md


# ------------------------------------------------------------------------------ fail-closed inputs


def test_missing_prediction_row_refused(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    fields.pop(min(fields))
    with pytest.raises(pa.AnalysisRefused, match="prediction"):
        analyse(execution, nullsize, ledger, fields)


def test_null_prediction_only_for_excluded_variant(manifest, nullsize):
    i = min(manifest["edit_ids"])
    ledger, fields, _, execution = build(manifest, excluded={i: "sub_resolution"})
    fields[i][0] = None
    analyse(execution, nullsize, ledger, fields)  # excluded rotation: accepted
    ledger, fields, _, execution = build(manifest)
    fields[i][0] = None
    with pytest.raises(pa.AnalysisRefused, match="prediction"):
        analyse(execution, nullsize, ledger, fields)


def test_unsealed_or_bad_ledger_refused(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    bad = copy.deepcopy(ledger)
    bad["sealed"] = False
    with pytest.raises(pa.AnalysisRefused, match="sealed"):
        analyse(execution, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    bad["window_ids"] = bad["window_ids"][:63]
    with pytest.raises(pa.AnalysisRefused, match="64"):
        analyse(execution, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    i = min(bad["variants"])
    bad["variants"][i]["ladder_0.5"]["variant_id"] = "ffffffffffffffff"
    with pytest.raises(pa.AnalysisRefused, match="variant id"):
        analyse(execution, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    bad["restage"][i]["f10_3_exclusions"] = ["audit_failed"]
    with pytest.raises(pa.AnalysisRefused, match="reason code"):
        analyse(execution, nullsize, bad, fields)


def test_missing_restage_audit_fails_closed(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    i = min(ledger["restage"])
    del ledger["restage"][i]
    rep = analyse(execution, nullsize, ledger, fields)
    led = rep["registered"]["contrasts"]["rotation"]["ledger"]
    assert led["f10_3_excluded"]["by_code"] == {"audit_missing": [i]}


def test_frozen_manifest_hash_mismatch_refused(tmp_path):
    bad = tmp_path / "manifest_a533.json"
    bad.write_bytes(FROZEN.read_bytes() + b"\n")
    with pytest.raises(pa.AnalysisRefused, match="frozen staged manifest sha256"):
        pa.load_frozen_manifest(bad)
    with pytest.raises(pa.AnalysisRefused, match="missing or unreadable"):
        pa.load_frozen_manifest(tmp_path / "absent.json")
    # The regenerated manifest is never accepted as the frozen anchor.
    with pytest.raises(pa.AnalysisRefused, match="frozen staged manifest sha256"):
        pa.load_frozen_manifest(EXECUTION)
    assert pa.sha256_file(FROZEN) == pa.FROZEN_STAGED_MANIFEST_SHA256
    assert pa.FROZEN_STAGED_MANIFEST_SHA256.startswith("d7785b35")
    assert pa.DEFAULT_FROZEN_MANIFEST == FROZEN


def test_execution_manifest_hash_mismatch_refused(tmp_path, manifest):
    *_, execution = build(manifest)
    path = tmp_path / "execution.json"
    path.write_bytes(manifest_bytes(execution))
    sha = exec_sha(execution)
    assert pa.load_execution_manifest(path, sha) == json.loads(path.read_text())
    with pytest.raises(pa.AnalysisRefused, match="does not match the supplied --manifest-sha256"):
        pa.load_execution_manifest(path, "0" * 64)
    path.write_bytes(manifest_bytes(execution) + b"\n")
    with pytest.raises(pa.AnalysisRefused, match="does not match the supplied --manifest-sha256"):
        pa.load_execution_manifest(path, sha)
    for absent in (None, "", "abc", sha.upper()):
        with pytest.raises(pa.AnalysisRefused, match="is required"):
            pa.load_execution_manifest(path, absent)
    with pytest.raises(pa.AnalysisRefused, match="missing or unreadable"):
        pa.load_execution_manifest(tmp_path / "absent.json", sha)


def test_api_requires_execution_manifest_hash(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    for absent in (None, "not-a-hash"):
        with pytest.raises(pa.AnalysisRefused, match="sha256 is required"):
            analyse(execution, nullsize, ledger, fields, manifest_sha256=absent)


def test_ledger_provenance_mismatch_refused(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    bad = copy.deepcopy(ledger)
    bad["manifest_sha256"] = pa.FROZEN_STAGED_MANIFEST_SHA256  # the old staged binding
    with pytest.raises(pa.AnalysisRefused, match="ledger manifest_sha256"):
        analyse(execution, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    bad.pop("manifest_sha256")
    with pytest.raises(pa.AnalysisRefused, match="ledger manifest_sha256"):
        analyse(execution, nullsize, bad, fields)
    bad = copy.deepcopy(ledger)
    bad["restage_provenance"]["manifest_sha256"] = "1" * 64
    with pytest.raises(pa.AnalysisRefused, match=r"restage_provenance\.manifest_sha256"):
        analyse(execution, nullsize, bad, fields)
    # A supplied hash differing from the ledger's record refuses even if the ledger is intact.
    with pytest.raises(pa.AnalysisRefused, match="ledger manifest_sha256"):
        analyse(execution, nullsize, ledger, fields, manifest_sha256="1" * 64)
    # An edited execution manifest no longer matches the hash the ledger sealed.
    edited = dict(execution, scored=True)
    with pytest.raises(pa.AnalysisRefused, match="ledger manifest_sha256"):
        analyse(edited, nullsize, ledger, fields)


def _without(execution, **kw):
    return dict(execution, **kw)


def test_execution_manifest_must_contain_the_registered_ids(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    i = min(manifest["edit_ids"])
    cases = [
        (
            _without(
                execution, selections=[s for s in execution["selections"] if s["edit_id"] != i]
            ),
            "lacks registered id",
        ),
        (
            _without(
                execution,
                variants=[
                    v
                    for v in execution["variants"]
                    if v["variant_id"] != manifest["variant_ids"][0]
                ],
            ),
            "lacks a533 variant id",
        ),
        (
            _without(execution, selections=execution["selections"] + execution["selections"][:1]),
            "more than once",
        ),
    ]
    rows = copy.deepcopy(execution["selections"])
    rows[0]["tertile"] = (rows[0]["tertile"] + 1) % 3
    cases.append((_without(execution, selections=rows), "differs from the frozen"))
    rows = copy.deepcopy(execution["selections"])
    rows[0]["effective_norm"] *= 1.5
    cases.append((_without(execution, selections=rows), "differs from the frozen"))
    rows = copy.deepcopy(execution["selections"])
    rows[0]["supplementary"] = True
    cases.append((_without(execution, selections=rows), "is flagged supplementary"))
    extra, extra_variants = supplementary_rows(1)
    extra[0].pop("supplementary")
    cases.append(
        (
            _without(
                execution,
                selections=execution["selections"] + extra,
                variants=execution["variants"] + extra_variants,
            ),
            "neither registered nor flagged",
        )
    )
    for bad, needle in cases:
        with pytest.raises(pa.AnalysisRefused, match=needle):
            analyse(
                bad, nullsize, bind(copy.deepcopy(ledger), bad), fields, masked_calibration=None
            )
    rows = copy.deepcopy(execution["selections"])
    rows[0]["supplementary"] = False  # an explicit false flag on a registered row is accepted
    ok = _without(execution, selections=rows)
    analyse(ok, nullsize, bind(copy.deepcopy(ledger), ok), fields)


def test_frozen_manifest_must_carry_the_forty_registered_ids(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    short = dict(manifest, selections=manifest["selections"][1:])
    with pytest.raises(pa.AnalysisRefused, match="forty registered ids"):
        analyse(execution, nullsize, ledger, fields, frozen_manifest=short)
    other = dict(manifest, selection_ledger={"edit_ids_sorted_sha256": "0" * 64})
    with pytest.raises(pa.AnalysisRefused, match="forty registered ids"):
        analyse(execution, nullsize, ledger, fields, frozen_manifest=other)


def test_frozen_cuts_come_from_a533_not_from_the_execution_manifest(manifest, nullsize):
    """Re-stage ρ in the execution manifest moves no frozen id; staged values stay a533's."""
    ids = sorted(manifest["edit_ids"])
    shifted = {i: 1.5 for i in ids}  # every re-stage ln ρ = 0.405, far above both cuts
    ledger, fields, _, execution = build(manifest, restage_rho=shifted)
    rep = analyse(execution, nullsize, ledger, fields)
    t = rep["f10_5_sensitivity"]["tertiles"]
    assert t["lnrho_cuts"] == [list(c) for c in pa.LNRHO_CUTS]
    rows = t["ln_rho"]
    assert [tuple(r["ids_in_frozen_order"]) for r in rows] == list(pa.LNRHO_CELLS)
    assert rows[0]["cell_drift"] == list(pa.LNRHO_CELLS[0])  # flagged, never moved
    assert rows[1]["cell_drift"] == list(pa.LNRHO_CELLS[1])
    assert rows[2]["cell_drift"] == []
    staged = pa.staged_ln_rho(manifest)
    assert rows[0]["staged_ln_rho"] == {i: staged[i] for i in pa.LNRHO_CELLS[0]}
    assert rows[0]["restage_ln_rho"][pa.LNRHO_CELLS[0][0]] == pytest.approx(math.log(1.5))
    assert [len(r["ids"]) for r in t["norm"][::3]] == [14, 13, 13]
    subs = rep["f10_5_sensitivity"]["subsets"]
    assert subs["S30"]["sha256"] == pa.S30_SHA256 and subs["S34"]["sha256"] == pa.S34_SHA256
    assert rep["manifests"]["frozen"]["sha256"] == pa.FROZEN_STAGED_MANIFEST_SHA256
    assert rep["manifests"]["execution"]["sha256"] == exec_sha(execution)


def test_verdicts_and_rho_are_read_from_the_execution_manifest(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    i, j = ids[0], ids[1]
    ledger, fields, _, execution = build(manifest, excluded={i: "replay_failed"})
    rep = analyse(execution, nullsize, ledger, fields)
    assert rep["registered"]["pairs"][i]["rotation"]["status"] == "f10_3_excluded"
    assert rep["restage"]["audit"][i]["reasons"] == ["replay_failed"]
    # The ledger's re-stage entry must agree with the execution manifest.
    bad = copy.deepcopy(ledger)
    bad["restage"][i].update(f10_3_verdict="pass", f10_3_exclusions=[])
    bad["variants"][i]["rotation"].pop("excluded")
    with pytest.raises(pa.AnalysisRefused, match="contradicts the execution manifest"):
        analyse(execution, nullsize, bad, fields, masked_calibration=None)
    bad = copy.deepcopy(ledger)
    bad["restage"][i]["f10_3_exclusions"] = ["rounding_bound_failed"]
    with pytest.raises(pa.AnalysisRefused, match="contradicts the execution manifest"):
        analyse(execution, nullsize, bad, fields, masked_calibration=None)
    bad = copy.deepcopy(ledger)
    bad["restage"][j]["pooled_rho"] *= 1.01
    with pytest.raises(pa.AnalysisRefused, match="pooled_rho"):
        analyse(execution, nullsize, bad, fields, masked_calibration=None)
    # A ledger without pooled_rho takes ρ from the execution manifest (no covariate dropped).
    led2 = copy.deepcopy(ledger)
    led2["restage"][j].pop("pooled_rho")
    rep = analyse(execution, nullsize, led2, fields)
    assert j in rep["f10_5_sensitivity"]["covariate"]["complete_case_ids"]
    # Execution-manifest evidence: unknown codes and contradictions refuse ...
    for ea_mutate, needle in (
        (lambda ea: ea["exclusions"].append({"reason": "audit_failed"}), "unknown execution"),
        (lambda ea: ea.update(pair_verdict="pass"), "contradicts its reason codes"),
    ):
        ex = dict(execution, edit_audits=copy.deepcopy(execution["edit_audits"]))
        ea_mutate(ex["edit_audits"][i])
        with pytest.raises(pa.AnalysisRefused, match=needle):
            analyse(ex, nullsize, bind(copy.deepcopy(ledger), ex), fields, masked_calibration=None)
    # ... and missing or malformed evidence fails the pair closed as audit_missing.
    for ea_mutate, needle in (
        (lambda eas: eas.pop(j), "no edit_audits entry"),
        (lambda eas: eas[j].pop("pair_verdict"), "pair_verdict None"),
        (lambda eas: eas[j].update(audit_rule="a533"), "audit_rule 'a533'"),
        (lambda eas: eas[j].update(exclusions="none"), "exclusions malformed"),
    ):
        ex = dict(execution, edit_audits=copy.deepcopy(execution["edit_audits"]))
        ea_mutate(ex["edit_audits"])
        rep = analyse(ex, nullsize, bind(copy.deepcopy(ledger), ex), fields)
        rec = rep["registered"]["pairs"][j]["rotation"]
        assert rec["status"] == "f10_3_excluded" and rec["reasons"] == ["audit_missing"]
        assert needle in rec["detail"]
    # Ladder exclusion flags must agree with the execution manifest's variants.
    lad = next(
        v["variant_id"]
        for v in execution["variants"]
        if v["source_id"] == j and v["kind"] == "scale" and v["strength"] == 0.5
    )
    ex = dict(
        execution,
        variants=[
            dict(v, excluded=True) if v["variant_id"] == lad else v for v in execution["variants"]
        ],
    )
    with pytest.raises(pa.AnalysisRefused, match="exclusion flag contradicts"):
        analyse(ex, nullsize, bind(copy.deepcopy(ledger), ex), fields)


LORA_HIGH = sorted(r["id"] for r in F11_ROWS if r["cell"] == "lora_sft:2")
CHAIN_LOW = sorted(r["id"] for r in F11_ROWS if r["cell"] == "chain:0")
CHAIN_HIGH = sorted(r["id"] for r in F11_ROWS if r["cell"] == "chain:2")
# F11.1(a): the nine registered rotation exclusions of the F10 re-stage, with their reasons.
F10_EXCLUSIONS = {
    "15a53586efc72a50": "rounding_bound_failed",
    "2233954b0cc3edc0": "rounding_bound_failed",
    "72d3756e968e3209": "rounding_bound_failed",
    "057dcfbedea3251f": "replay_failed",
    "0bc0a0c4ad0520e0": "replay_failed",
    "0d055869527177b4": "replay_failed",
    "0027efdf2eae7405": "replay_failed",
    "05531ce1db4df63f": "replay_failed",
    "14cdaabee246493b": "replay_failed",
}


def _ids_in(obj):
    return set(re.findall(r"[0-9a-f]{16}", json.dumps(obj, default=str)))


def test_supplementary_edits_join_f10_5_reports_with_frozen_cells_unchanged(manifest, nullsize):
    """Correction 1: supplements join the covariate model, sign tables, comparator and tertile
    reports (frozen cut keys, marked supplementary); the forty's cells and S30/S34 are unchanged."""
    ledger, fields, _, execution = build(
        manifest, supplementary=3, excluded={LORA_HIGH[1]: "replay_failed"}
    )
    supp = LORA_HIGH[:3]
    registered = sorted(manifest["edit_ids"])
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
    assert rep["manifests"]["supplementary_ids"] == supp
    assert rep["manifests"]["registered_ids"] == registered
    assert rep["manifests"]["supplementary_matches_f11_2_frozen_list"] is False  # 3 of 12
    # Registered analysis: 43 selected, the supplementary exclusion counted like any other.
    rot = rep["registered"]["contrasts"]["rotation"]
    assert rot["ledger"]["selected"]["n"] == 43 and rot["resolved"] == 42
    assert rot["ledger"]["f10_3_excluded"]["by_code"] == {"replay_failed": [supp[1]]}
    assert rep["registered"]["contrasts"]["ladder_0.25"]["resolved"] == 43
    bs = rot["by_selection"]
    assert bs["registered_40"]["n_selected"] == 40 and bs["supplementary"]["resolved"] == 2
    assert rep["registered"]["decision"]["step_1_validity"] == "pass"
    sens = rep["f10_5_sensitivity"]
    assert sens["supplementary_ids"] == supp
    # Covariate model on the supplemented set, supplements marked.
    cov = sens["covariate"]
    assert cov["selected_ids"] == sorted(registered + supp)
    assert cov["supplementary_ids"] == supp
    assert cov["complete_case_supplementary_ids"] == [supp[0], supp[2]]
    assert set(cov["complete_case_ids"]) == set(registered) | {supp[0], supp[2]}
    assert cov["n_cc"] == 42 and cov["full_analysis_set_agreement"]["n"] == 42
    assert cov["sign_table"] == cov["complete_case_agreement"]["sign_table"]
    assert sum(r["n"] for r in cov["sign_table"].values()) == 42
    # Tertiles: frozen cells unchanged; supplements added by the frozen cut keys, marked.
    frozen_cells = pa.norm_tertiles(manifest)
    for r in sens["tertiles"]["norm"]:
        t = pa.TERTILE_NAMES.index(r["cell"].split(" ")[0])
        group = r["cell"].split(" ")[-1]
        assert r["frozen_ids"] == sorted(frozen_cells[t][group])
        want = supp if (t == 2 and group in ("pooled", "lora_sft")) else []
        assert r["supplementary_ids"] == want, r["cell"]
        assert r["ids"] == sorted(r["frozen_ids"] + want)
        assert ("frozen_40" in r) == bool(want)
    staged = pa.supplementary_staged_ln_rho(execution, supp)
    placed = {}
    for k, r in enumerate(sens["tertiles"]["ln_rho"]):
        assert tuple(r["ids_in_frozen_order"]) == pa.LNRHO_CELLS[k]
        assert pa.sorted_sha(r["frozen_ids"]) == pa.LNRHO_CELL_SHA256[k]
        for i in r["supplementary_ids"]:
            placed[i] = k
            assert r["staged_ln_rho"][i] == staged[i]
    assert placed == {i: pa._lnrho_cell(staged[i], i) for i in supp}
    pred = sens["supplementary_predicates"]
    assert set(pred) == set(supp) and all(p["supplementary"] and p["root"] for p in pred.values())
    # S30/S34 stay the registered-40 lists; supplements' predicate values shown beside.
    for name, ids_ in (("S30", pa.S30_IDS), ("S34", pa.S34_IDS)):
        assert sens["subsets"][name]["ids"] == sorted(ids_)
        assert not set(supp) & _ids_in(sens["subsets"][name])
    sp = sens["subsets"]["supplementary_predicates"]
    assert set(sp) == set(supp) and not any(x["enters_subsets"] for x in sp.values())
    for i in supp:
        m = execution["edit_audits"][i]["edit_audit"]["pooled_norm_mismatch"]
        assert sp[i]["s30"]["value"] == m and sp[i]["s30"]["verdict"] == (m <= 2e-2)
        assert sp[i]["s34"]["verdict"] is True
    # Constant-sign comparator: main contrast on the supplemented set, split by selection.
    comp = sens["constant_sign_comparator"]
    assert comp["main_contrast"]["comparator"] == rot["comparator"]
    assert comp["by_selection"]["supplementary"]["n_selected"] == 3
    md = pa.render_markdown(rep)
    assert all(i in md for i in supp) and "supplementary ids (3" in md
    assert "(supplementary)" in md and "values of the S30/S34 predicates" in md
    # A supplementary edit still needs its frozen prediction row.
    fields.pop(supp[0])
    with pytest.raises(pa.AnalysisRefused, match="prediction"):
        analyse(execution, nullsize, ledger, fields)


def test_frozen_forty_reports_identical_with_and_without_supplements(manifest, nullsize):
    """Supplements never move a frozen member: the forty's cell lists, cut keys and S30/S34
    reports (on the same synthetic forty) are identical with and without supplements."""
    led0, f0, _, ex0 = build(manifest, seed=3)
    led1, f1, _, ex1 = build(manifest, seed=3, supplementary=12)
    r0 = analyse(ex0, nullsize, led0, f0)
    r1 = analyse(ex1, nullsize, led1, f1)
    t0, t1 = r0["f10_5_sensitivity"]["tertiles"], r1["f10_5_sensitivity"]["tertiles"]
    assert t0["lnrho_cuts"] == t1["lnrho_cuts"]
    for a, b in zip(t0["norm"] + t0["ln_rho"], t1["norm"] + t1["ln_rho"], strict=True):
        assert a["frozen_ids"] == b["frozen_ids"] and a["cell"] == b["cell"]
    added = sorted(i for r in t1["ln_rho"] for i in r["supplementary_ids"])
    assert added == F11_IDS  # every supplement placed once by the frozen ln ρ cuts
    for name in ("S30", "S34"):
        assert (
            r0["f10_5_sensitivity"]["subsets"][name]["ids"]
            == (r1["f10_5_sensitivity"]["subsets"][name]["ids"])
        )
    assert r0["registered_40_sensitivity"]["applicable"] is False


@pytest.mark.parametrize(
    "target,mutate,needle",
    [
        (
            "chain_low",
            lambda r: r.update(tertile=2),
            "differs from the frozen A5.3.1 norm cut keys",
        ),
        ("lora_high", lambda r: r.update(predecessor_id="b7edbb9f8b339f52"), "is not a root"),
        ("lora_high", lambda r: r.update(tertile=0, effective_norm=0.5), "not an F11.2 depleted"),
        ("lora_high", lambda r: r.update(family="dpo"), "not an F11.2 depleted cell"),
        ("chain_high", lambda r: r.update(effective_norm=None), "norm cut keys"),
    ],
)
def test_supplementary_cell_and_lineage_fail_closed(manifest, nullsize, target, mutate, needle):
    ledger, fields, _, execution = build(manifest, supplementary=12)
    i = {"lora_high": LORA_HIGH, "chain_low": CHAIN_LOW, "chain_high": CHAIN_HIGH}[target][0]
    rows = copy.deepcopy(execution["selections"])
    mutate(next(r for r in rows if r["edit_id"] == i))
    bad = _without(execution, selections=rows)
    with pytest.raises(pa.AnalysisRefused, match=needle):
        analyse(bad, nullsize, bind(copy.deepcopy(ledger), bad), fields)


def test_frozen_norm_cut_keys_bind_to_a533(manifest, nullsize):
    pa.verify_norm_cut_keys(manifest)
    bad = copy.deepcopy(manifest)
    bad["selection_ledger"]["selection_json_sha256"] = "0" * 64
    with pytest.raises(pa.AnalysisRefused, match="selection.json"):
        pa.verify_norm_cut_keys(bad)
    bad = copy.deepcopy(manifest)
    next(r for r in bad["selections"] if r["tertile"] == 0)["effective_norm"] = 100.0
    with pytest.raises(pa.AnalysisRefused, match="do not reproduce"):
        pa.verify_norm_cut_keys(bad)
    sel = json.loads((ROOT / "docs/prereg/a5_manifests/selection.json").read_text())
    assert pa.sha256_file(ROOT / "docs/prereg/a5_manifests/selection.json") == (
        pa.SELECTION_JSON_SHA256
    )
    for fam, keys in pa.NORM_CUT_KEYS.items():
        assert [tuple(k) for k in sel["procedural_freeze"]["cutpoints"][fam]["keys"]] == list(keys)
    for r in F11_ROWS:
        fam, t = r["cell"].split(":")
        assert pa.norm_tertile(fam, r["effective_norm"], r["id"]) == int(t)


def test_supplementary_predicate_contradiction_and_lnrho_drift(manifest, nullsize):
    i, j = LORA_HIGH[:2]
    ledger, fields, _, execution = build(manifest, supplementary=2)
    bad = copy.deepcopy(execution)
    ea = bad["edit_audits"][i]["edit_audit"]
    ea["in_subset_pooled_norm_2e-2"] = not ea["in_subset_pooled_norm_2e-2"]
    with pytest.raises(pa.AnalysisRefused, match="contradicts its pooled norm mismatch"):
        analyse(bad, nullsize, bind(copy.deepcopy(ledger), bad), fields)
    # Staged ln ρ (own staging record) in the low cell, re-stage ln ρ in the high cell: the
    # supplement is placed by the staged value under the frozen cuts and flagged, never moved.
    rho = math.exp(0.05)  # ln ρ 0.05 > cut 2 (0.035)
    ledger, fields, _, execution = build(manifest, supplementary=2, restage_rho={i: rho})
    drift = copy.deepcopy(execution)
    drift["edit_audits"][i]["edit_audit"]["pooled_norm_mismatch"] = 1e-4  # staged ln ρ 2e-4
    drift["edit_audits"][i]["edit_audit"]["in_subset_pooled_norm_2e-2"] = True
    for k in ("pooled_norm_mismatch", "in_subset_pooled_norm_2e-2", "in_subset_noise_calibrated"):
        del drift["edit_audits"][j]["edit_audit"][k]
    rep = analyse(drift, nullsize, bind(copy.deepcopy(ledger), drift), fields)
    t = rep["f10_5_sensitivity"]["tertiles"]
    low, high = t["ln_rho"][0], t["ln_rho"][2]
    assert low["supplementary_ids"] == [i] and i not in high["supplementary_ids"]
    assert i in low["cell_drift"]
    assert low["staged_ln_rho"][i] == 2 * math.log1p(1e-4)
    assert math.isclose(low["restage_ln_rho"][i], 0.05)
    assert t["lnrho_unassigned_supplementary_ids"] == [j]
    assert j not in {x for r in t["ln_rho"] for x in r["ids"]}
    sp = rep["f10_5_sensitivity"]["subsets"]["supplementary_predicates"][j]
    assert sp["s30"]["verdict"] is None and sp["s34"]["verdict"] is None
    assert sp["s34"]["source"] == "not evaluated in the staging record"
    md = pa.render_markdown(rep)
    assert "without a staged ln ρ" in md


def test_registered_40_report_and_lineage_splits(manifest, nullsize):
    """Correction 2: the registered-40 report on the forty alone (no decision authority), with
    availability distinct from resolution, root/non-root splits and full accounting."""
    reg = sorted(manifest["edit_ids"])
    disagree = reg[:12]  # 28/40 agree on the forty
    preds = {i: -0.5 for i in disagree} | {i: -0.5 for i in F11_IDS}
    ledger, fields, _, execution = build(
        manifest, supplementary=12, excluded=F10_EXCLUSIONS, predictions=preds
    )
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
    r40 = rep["registered_40_sensitivity"]
    assert r40["authority"] == pa.AUTHORITY_NONE and r40["label"] == pa.REGISTERED_40_LABEL
    assert r40["ids"] == reg and not set(F11_IDS) & _ids_in(r40)
    av = r40["availability"]
    assert (av["rotation"]["selected"], av["rotation"]["available"]) == (40, 31)
    assert av["ladder_0.25"]["available"] == av["ladder_0.5"]["available"] == 40
    assert av["rotation"]["resolved"] == r40["contrasts"]["rotation"]["resolved"]
    assert "Availability is not resolution" in av["rotation"]["rule"]
    reg_av = rep["registered"]["contrasts"]["rotation"]["availability"]
    assert (reg_av["selected"], reg_av["available"]) == (52, 43)
    # Decision on the forty is descriptive and never replaces the registered one.
    d40, dec = r40["decision"], rep["registered"]["decision"]
    assert d40["authority"] == pa.AUTHORITY_NONE and dec["authority"] == pa.AUTHORITY_REGISTERED
    rot40 = r40["contrasts"]["rotation"]
    assert rot40["resolved"] == 31
    assert "part_b_reporting" not in d40
    reg_rot = rep["registered"]["contrasts"]["rotation"]
    wrong40 = len([i for i in disagree if i not in F10_EXCLUSIONS])
    assert reg_rot["resolved"] == 43 and rot40["agreements"] == 31 - wrong40
    assert reg_rot["agreements"] == rot40["agreements"]  # the twelve supplements disagree
    assert reg_rot["field"]["cp95"] != rot40["field"]["cp95"]
    # Root versus non-root chain splits, per family, in both reports.
    lin = rep["registered"]["contrasts"]["rotation"]["by_lineage"]
    assert lin["authority"] == pa.AUTHORITY_NONE
    assert lin["chain"]["root"]["ids"] == sorted(
        ["014f8a9ec78db2be", "0559051354691a70"] + CHAIN_LOW + CHAIN_HIGH
    )
    assert lin["chain"]["non_root"]["ids"] == sorted(pa.REALISED_INCREMENT_LADDER_IDS)
    assert lin["lora_sft"]["non_root"]["n_selected"] == 0
    assert lin["lora_sft"]["root"]["n_selected"] == 34
    lin40 = rot40["by_lineage"]
    assert lin40["chain"]["root"]["ids"] == ["014f8a9ec78db2be", "0559051354691a70"]
    md = pa.render_markdown(rep)
    assert "## Registered-40 sensitivity (F11.5) — no decision authority" in md
    assert "Registered-40 A5.5 ordered decision (descriptive; no decision authority)" in md
    assert "`chain` non-root" in md and "`chain` root" in md
    sec = md.split("## Registered-40 sensitivity")[1].split("## F10.5 sensitivity")[0]
    assert not set(F11_IDS) & set(re.findall(r"[0-9a-f]{16}", sec))
    for i in F10_EXCLUSIONS:
        assert i in sec  # exclusion ids listed in the forty's ledger


def test_registered_40_decision_differs_and_has_no_authority(manifest, nullsize):
    """A go on the forty alone with an unresolved registered (supplemented) result: reported as
    a descriptive disagreement only."""
    reg = sorted(manifest["edit_ids"])
    preds = {i: -0.5 for i in reg[:12]} | {i: -0.5 for i in F11_IDS}
    ledger, fields, _, execution = build(manifest, supplementary=12, predictions=preds)
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
    d40 = rep["registered_40_sensitivity"]["decision"]
    dec = rep["registered"]["decision"]
    assert d40["part_a"] == "go" and d40["authority"] == pa.AUTHORITY_NONE
    assert dec["part_a"].startswith("unresolved") and dec["gate"] == "Part A unresolved"
    assert dec["funding"].startswith("unresolved")


def _ladder_case(manifest):
    a, b, c = LORA_HIGH[0], CHAIN_LOW[0], CHAIN_HIGH[0]
    excluded = dict(F10_EXCLUSIONS) | {c: "replay_failed"}
    lad = {a: ["ladder_0.25"], b: ["ladder_0.5"], c: ["ladder_0.25"]}
    return (a, b, c), build(manifest, supplementary=12, excluded=excluded, ladder_excluded=lad)


def test_per_contrast_masks_with_supplementary_ladder_exclusions(manifest, nullsize):
    """Correction 3: supplementary ladder exclusions at each strength are masked per contrast; a
    valid ladder is kept after a rotation failure; the denominators are separate."""
    (a, b, c), (ledger, fields, _, execution) = _ladder_case(manifest)
    masked, sha, _ = masked_for(ledger, execution)
    assert masked["slot_mask"]["masked_ids"]["ladder_0.25"] == sorted([a, c])
    assert masked["slot_mask"]["masked_ids"]["ladder_0.5"] == [b]
    rep = analyse(
        execution,
        nullsize,
        ledger,
        fields,
        masked_calibration=masked,
        masked_calibration_sha256=sha,
        validity=valid_for(ledger, sha),
        part_b="ok",
    )
    f = rep["f10_6_calibration"]
    assert f["route"] == "decisive" and f["supplemented"] is True
    assert f["masked_ids"]["rotation"] == sorted(list(F10_EXCLUSIONS) + [c])
    assert f["available_counts"] == {"rotation": 42, "ladder_0.25": 50, "ladder_0.5": 51}
    assert f["mapping"][c] == ["ladder_0.5"] and f["mapping"][a] == ["rotation", "ladder_0.5"]
    con = rep["registered"]["contrasts"]
    assert con["ladder_0.25"]["ledger"]["excluded"]["ids"] == sorted([a, c])
    assert con["ladder_0.5"]["ledger"]["excluded"]["ids"] == [b]
    assert (
        con["ladder_0.5"]["pair_status"][
            [p["id"] for p in con["ladder_0.5"]["pair_status"]].index(c)
        ]["status"]
        == "resolved"
    )
    assert (con["rotation"]["resolved"], con["ladder_0.25"]["resolved"]) == (42, 50)
    assert con["ladder_0.5"]["resolved"] == 51
    md = pa.render_markdown(rep)
    assert "Masked ladder_0.25 slots (2; available 50/52)" in md


def test_per_contrast_mask_mismatches_refused(manifest, nullsize):
    (a, b, _c), (ledger, fields, _, execution) = _ladder_case(manifest)
    masked, sha, _ = masked_for(ledger, execution)

    def run(m):
        return analyse(
            execution,
            nullsize,
            ledger,
            fields,
            masked_calibration=m,
            masked_calibration_sha256=sha,
        )

    cases = []
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["masked_ids"]["ladder_0.25"] = [a]  # omits c's ladder mask
    cases.append((bad, "masked ids of ladder_0.25"))
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["masked_ids"]["ladder_0.5"] = [b, min(manifest["edit_ids"])]
    cases.append((bad, "masked ids of ladder_0.5"))  # a ladder mask on a registered id
    bad = copy.deepcopy(masked)
    old = masked_for(*build(manifest, supplementary=12, excluded=F10_EXCLUSIONS)[::3])[0]
    bad["slot_mask"]["mapping_sha256"] = old["slot_mask"]["mapping_sha256"]  # rotation-only map
    cases.append((bad, "mapping sha256"))
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["slots"][0]["available"] = ["rotation"]
    cases.append((bad, "slot table"))
    bad = copy.deepcopy(masked)
    bad["supplement"]["ids_sorted_sha256"] = pa.sorted_sha(F11_IDS[:11])
    cases.append((bad, "supplement block"))
    bad = copy.deepcopy(masked)
    del bad["supplement"]
    cases.append((bad, "supplement block"))
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["label"] = "slot-masked (F10.6)"
    cases.append((bad, "slot-masked block"))
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["label"] = f"slot-masked (F10.6); {pa.REGISTERED_40_LABEL}"
    cases.append((bad, "slot-masked block"))
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["gates"]["decision_authority"] = False
    cases.append((bad, "registered-40 sensitivity run"))
    for m, needle in cases:
        with pytest.raises(pa.AnalysisRefused, match=re.escape(needle)):
            run(m)
    run(masked)  # the correct mask is accepted


def test_registered_ladder_exclusion_cannot_be_masked(manifest, nullsize):
    i = min(manifest["edit_ids"])
    for supp in (0, 12):
        ledger, fields, _, execution = build(
            manifest,
            supplementary=supp,
            excluded={max(manifest["edit_ids"]): "replay_failed"},
            ladder_excluded={i: ["ladder_0.25"]},
        )
        with pytest.raises(pa.AnalysisRefused, match="cannot be masked"):
            analyse(execution, nullsize, ledger, fields)


def test_supplemented_analysis_always_needs_the_f11_4_run(manifest, nullsize):
    """F11.4: a supplemented analysis needs the supplemented calibration even at full yield."""
    ledger, fields, _, execution = build(manifest, supplementary=2)
    with pytest.raises(pa.AnalysisRefused, match="F11.4"):
        analyse(execution, nullsize, ledger, fields, masked_calibration=None)
    rep = analyse(execution, nullsize, ledger, fields)
    f = rep["f10_6_calibration"]
    assert f["required"] and f["slot_mask_label"] == pa.SUPPLEMENT_SLOT_MASK_LABEL
    assert all(f["masked_ids"][c] == [] for c in pa.CONTRASTS)
    # A supplemented run supplied to a forty-only analysis is refused.
    led0, f0, _, ex0 = build(manifest, excluded={min(manifest["edit_ids"]): "replay_failed"})
    m0, sha0, _ = masked_for(led0, ex0)
    m0["supplement"] = {"label": pa.SUPPLEMENT_LABEL}
    with pytest.raises(pa.AnalysisRefused, match="selects no supplementary edit"):
        analyse(ex0, nullsize, led0, f0, masked_calibration=m0, masked_calibration_sha256=sha0)


def _screen(manifest, nullsize, *, accept=True, link=True, preds=None, ladder_mean=None, **kw):
    ledger, fields, _, execution = build(
        manifest,
        supplementary=12,
        excluded=F10_EXCLUSIONS,
        predictions=preds,
        ladder_mean=ladder_mean,
        ladder_pred=kw.pop("ladder_pred", None),
    )
    masked, sha, _ = masked_for(ledger, execution, power_passed=False)
    validity = valid_for(ledger, sha)
    if accept:
        ev = [{"path": "docs/reviews/council/board-a-f11-calibration.md", "sha256": "12" * 32}]
        if link:
            ev.append({"path": "power_slotmask_f11.json", "sha256": sha})
        validity[pa.SCREEN_ROUTE_KEY] = {"pass": True, "evidence": ev}
    return analyse(
        execution,
        nullsize,
        ledger,
        fields,
        masked_calibration=masked,
        masked_calibration_sha256=sha,
        validity=validity,
        part_b="ok",
        **kw,
    )


def _f11_7_quote():
    text = REGISTRATION.read_text()
    block = text[text.index("**F11.7 — Fallback (Option B).**") : text.index("**F11.8")]
    quote = [ln[2:] for ln in block.splitlines() if ln.startswith("> ")]
    return " ".join(" ".join(quote).split())


def test_power_only_failure_runs_as_screen(manifest, nullsize):
    """Correction 4: power-only F11.4 failure with Board A acceptance → screen route; the rotation
    contrast is not decisive and the A5.5 validity row is worded as F11.7 states."""
    assert " ".join(pa.F11_7_VALIDITY_ROW.split()) == _f11_7_quote()
    rep = _screen(manifest, nullsize)
    f = rep["f10_6_calibration"]
    assert f["route"] == "screen" and f["verdict"] == "FAIL" and f["passed"] is False
    assert pa.F11_7_VALIDITY_ROW in f["statement"]
    dec = rep["registered"]["decision"]
    assert dec["step_1_validity"] == "pass"
    assert dec["route"] == "screen (F11.7 Option B; F11a.1)"
    assert dec["rotation_contrast"] == "not decisive (A5.3.2; F11.7)"
    assert dec["validity"]["validity_row"] == pa.F11_7_VALIDITY_ROW
    assert dec["validity"]["screen_rule"] == pa.F11A_SCREEN_RULE
    assert dec["part_a"] == pa.SCREEN_POSITIVE
    assert dec["gate"] == "Part A screen (rotation not decisive)" and "go" not in dec["gate"]
    assert dec["funding"].startswith("unresolved") and "cannot fund" in dec["funding"]
    assert dec["step_4_part_b_eligibility"].startswith("not eligible")
    assert "no G7 contribution" in dec["g7"]
    sc = dec["screen_calibration"]
    assert sc["power"]["ordered_go_rate"] == 0.7514 and sc["size"]["passed"] is True
    assert "0.7514" in sc["statement"] and "[0.03, 0.03, 0.03]" in sc["statement"]
    assert rep["registered"]["part_b"]["status"] == "supplied"  # Part B still reported
    d40 = rep["registered_40_sensitivity"]["decision"]
    assert d40["authority"] == pa.AUTHORITY_NONE and d40["part_a"] != "go"
    md = pa.render_markdown(rep)
    assert "route **screen**" in md and pa.F11_7_VALIDITY_ROW in md
    assert "Gate: **Part A screen (rotation not decisive)**" in md


def test_screen_route_keeps_ladder_alternative_priority_and_floor(manifest, nullsize):
    # Ladder s = .25 resolves negative while predicted positive: alternative fires first.
    rep = _screen(
        manifest, nullsize, ladder_mean=dict.fromkeys(F11_IDS + sorted(manifest["edit_ids"]), -0.01)
    )
    dec = rep["registered"]["decision"]
    assert dec["step_2_ladder_alternative"] is True and dec["gate"] == "alternative"
    assert dec["funding"].startswith("stop")
    # Zero predictions disagree: no positive screen.
    many = sorted(manifest["edit_ids"]) + F11_IDS
    rep = _screen(manifest, nullsize, preds=dict.fromkeys(many[:30], 0.0))
    assert rep["registered"]["decision"]["part_a"].startswith("unresolved")
    d = pa.decision(VALID, _summary(10, 14), _summary(0, 0), part_b=PB(), route="screen")
    assert d["part_a"] == "unresolved: fewer than 15 resolved rotation pairs"
    d = pa.decision(VALID, _summary(8, 20), _summary(0, 0), part_b=PB(), route="screen")
    assert d["part_a"] == "unresolved: interval includes 0.5"


@pytest.mark.parametrize("accept,link", [(False, False), (True, False)])
def test_screen_route_needs_linked_acceptance(manifest, nullsize, accept, link):
    rep = _screen(manifest, nullsize, accept=accept, link=link)
    dec = rep["registered"]["decision"]
    assert _is_validity_unresolved(dec)
    assert "F11a.1(d)" in dec["validity"]["problems"][pa.SCREEN_ROUTE_KEY]
    assert dec["route"].startswith("screen")


@pytest.mark.parametrize("supp", [0, 12])
@pytest.mark.parametrize("power_ok", [True, False])
def test_size_failure_refuses_with_priority(manifest, nullsize, supp, power_ok):
    ledger, fields, _, execution = build(manifest, supplementary=supp, excluded=F10_EXCLUSIONS)
    masked, sha, _ = masked_for(
        ledger, execution, passed=False, power_passed=power_ok, two_sided=(0.03, 0.08, 0.03)
    )
    with pytest.raises(pa.AnalysisRefused, match=re.escape(pa.SIZE_HOLD)):
        analyse(
            execution,
            nullsize,
            ledger,
            fields,
            masked_calibration=masked,
            masked_calibration_sha256=sha,
            validity=valid_for(ledger, sha),
            part_b="ok",
        )


def test_screen_route_evidence_and_gate_consistency_fail_closed(manifest, nullsize):
    ledger, fields, _, execution = build(manifest, supplementary=12, excluded=F10_EXCLUSIONS)

    def run(m, sha):
        return analyse(
            execution, nullsize, ledger, fields, masked_calibration=m, masked_calibration_sha256=sha
        )

    m, sha, _ = masked_for(ledger, execution, power_passed=False)
    bad = copy.deepcopy(m)
    del bad["slot_mask"]["gates"]["size"]["two_sided_rejection_rates"]
    with pytest.raises(pa.AnalysisRefused, match=r"F11a\.1\(c\)"):
        run(bad, sha)
    bad = copy.deepcopy(m)
    bad["slot_mask"]["gates"]["size"]["two_sided_rejection_rates"] = [0.03, 0.09, 0.03]
    with pytest.raises(pa.AnalysisRefused, match="size flag vs size rates"):
        run(bad, sha)  # flagged as passing but a two-sided rate exceeds 0.07
    bad = copy.deepcopy(m)
    bad["slot_mask"]["gates"]["power"]["ordered_go_rate"] = 0.85
    with pytest.raises(pa.AnalysisRefused, match="power flag vs ordered power"):
        run(bad, sha)
    bad = copy.deepcopy(m)
    bad["slot_mask"]["gates"]["size"]["two_sided_rejection_rates"] = [0.03, 0.03]
    with pytest.raises(pa.AnalysisRefused, match="two-sided size rates malformed"):
        run(bad, sha)


def test_decisive_route_is_unchanged():
    for rot, lad in (
        ((40, 40), (0, 0)),
        ((10, 20), (0, 0)),
        ((40, 40), (2, 20)),
        ((5, 10), (0, 0)),
    ):
        a = pa.decision(VALID, _summary(*rot), _summary(*lad), part_b=PB())
        b = pa.decision(VALID, _summary(*rot), _summary(*lad), part_b=PB(), route="decisive")
        assert a == b and "route" not in a
    with pytest.raises(pa.AnalysisRefused, match="unknown decision route"):
        pa.decision(VALID, _summary(40, 40), _summary(0, 0), part_b=PB(), route="waived")


def test_cli_end_to_end_f11_twelve_supplements(tmp_path, manifest, monkeypatch):
    """Correction 5 (CPU end-to-end): the twelve frozen supplements, the nine preserved original
    exclusions, independent supplementary rotation/ladder exclusions, a closed ledger, complete
    supplemented and registered-40 reports, the decisive and the screen route, refusals."""
    monkeypatch.setattr(pa, "BOOTSTRAP_DRAWS", DRAWS)
    a, b, c = LORA_HIGH[0], CHAIN_LOW[1], CHAIN_HIGH[2]
    ledger, _fields, fields_bytes, execution = build(
        manifest,
        supplementary=12,
        excluded=dict(F10_EXCLUSIONS) | {b: "replay_failed"},
        ladder_excluded={a: ["ladder_0.5"], c: ["ladder_0.25", "ladder_0.5"]},
    )
    (tmp_path / "fields.json").write_bytes(fields_bytes)
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "execution.json").write_bytes(manifest_bytes(execution))
    (tmp_path / "part_b.json").write_text(json.dumps(PB()))
    base = _cli_args(
        tmp_path,
        tmp_path / "ledger.json",
        tmp_path / "fields.json",
        "--spec-sha256",
        pa.SPEC_SHA256,
        "--part-b",
        str(tmp_path / "part_b.json"),
    )

    def run(masked_raw, validity):
        (tmp_path / "masked.json").write_bytes(masked_raw)
        (tmp_path / "validity.json").write_text(json.dumps(validity))
        for f in ("r.json", "r.md"):
            (tmp_path / f).unlink(missing_ok=True)
        return pa.main(
            base
            + ["--validity", str(tmp_path / "validity.json")]
            + ["--masked-calibration", str(tmp_path / "masked.json")]
        )

    # Decisive route.
    _m, msha, mraw = masked_for(ledger, execution)
    assert run(mraw, valid_for(ledger, msha)) == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["manifests"]["supplementary_ids"] == F11_IDS
    assert rep["manifests"]["supplementary_matches_f11_2_frozen_list"] is True
    con = rep["registered"]["contrasts"]
    assert [con[k]["availability"]["available"] for k in pa.CONTRASTS] == [42, 51, 50]
    assert con["rotation"]["ledger"]["selected"]["n"] == 52
    assert sorted(con["rotation"]["ledger"]["f10_3_excluded"]["ids"]) == sorted(
        list(F10_EXCLUSIONS) + [b]
    )
    r40 = rep["registered_40_sensitivity"]
    assert [r40["availability"][k]["available"] for k in pa.CONTRASTS] == [31, 40, 40]
    sens = rep["f10_5_sensitivity"]
    assert sens["covariate"]["supplementary_ids"] == F11_IDS
    assert sorted(i for r in sens["tertiles"]["ln_rho"] for i in r["supplementary_ids"]) == F11_IDS
    pooled = {r["cell"]: r for r in sens["tertiles"]["norm"] if r["cell"].endswith("pooled")}
    assert pooled["high (2) pooled"]["supplementary_ids"] == sorted(LORA_HIGH + CHAIN_HIGH)
    assert pooled["low (0) pooled"]["supplementary_ids"] == CHAIN_LOW
    assert pooled["middle (1) pooled"]["supplementary_ids"] == []
    assert set(sens["subsets"]["supplementary_predicates"]) == set(F11_IDS)
    dec = rep["registered"]["decision"]
    assert dec["step_1_validity"] == "pass" and dec["gate"] == "Part A go" and "route" not in dec
    assert rep["f10_6_calibration"]["masked_ids"]["ladder_0.5"] == sorted([a, c])
    md = (tmp_path / "r.md").read_text()
    for needle in (
        "## Registered-40 sensitivity (F11.5) — no decision authority",
        "values of the S30/S34 predicates",
        "slot-masked (F10.6); supplemented (F11.4)",
        "Masked ladder_0.5 slots (2; available 50/52)",
        "(supplementary)",
    ):
        assert needle in md, needle
    # Screen route (power-only failure, acceptance linked).
    _m, msha, mraw = masked_for(ledger, execution, power_passed=False)
    v = valid_for(ledger, msha)
    v[pa.SCREEN_ROUTE_KEY] = {"pass": True, "evidence": [{"path": "accept.md", "sha256": msha}]}
    assert run(mraw, v) == 0
    dec = json.loads((tmp_path / "r.json").read_text())["registered"]["decision"]
    assert dec["part_a"] == pa.SCREEN_POSITIVE and dec["funding"].startswith("unresolved")
    # Size failure: refused, nothing written.
    _m, msha, mraw = masked_for(ledger, execution, passed=False, power_passed=False)
    assert run(mraw, valid_for(ledger, msha)) == 2
    assert not (tmp_path / "r.json").exists() and not (tmp_path / "r.md").exists()
    # Wrong per-contrast mask (rotation-only mapping): refused, nothing written.
    led_r, _, _, ex_r = build(
        manifest, supplementary=12, excluded=dict(F10_EXCLUSIONS) | {b: "replay_failed"}
    )
    _m, msha, mraw = masked_for(led_r, ex_r)
    assert run(mraw, valid_for(ledger, msha)) == 2
    assert not (tmp_path / "r.json").exists()


def test_committed_execution_manifest_binds_to_the_frozen_a533():
    """The committed regenerated manifest (hash computed here, not pinned) binds to the a533."""
    sha = pa.sha256_file(EXECUTION)
    assert sha != pa.FROZEN_STAGED_MANIFEST_SHA256
    m = pa.load_execution_manifest(EXECUTION, sha)
    b = pa.bind_manifests(FROZEN_M, m, sha)
    assert b["registered_ids"] == sorted(FROZEN_M["edit_ids"])
    assert set(b["registered_ids"]) <= set(b["selected_ids"])
    assert set(b["supplementary_ids"]) == set(b["selected_ids"]) - set(b["registered_ids"])
    for i in b["selected_ids"]:
        assert not isinstance(pa._manifest_verdict(m, i), list), i  # F10 verdict present
    rho = pa.restage_rho(m)
    assert all(rho[i] is not None for i in b["registered_ids"])


def _write_inputs(tmp_path, manifest, **kw):
    ledger, _fields, fields_bytes, execution = build(manifest, **kw)
    (tmp_path / "fields.json").write_bytes(fields_bytes)
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "execution.json").write_bytes(manifest_bytes(execution))
    return tmp_path / "ledger.json", tmp_path / "fields.json"


def _manifest_args(tmp_path, path=None, sha=None):
    path = path or tmp_path / "execution.json"
    return ["--manifest", str(path), "--manifest-sha256", sha or pa.sha256_file(path)]


def test_cli_writes_json_and_markdown(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr(pa, "BOOTSTRAP_DRAWS", DRAWS)
    led, fld = _write_inputs(tmp_path, manifest)
    out, md = tmp_path / "r.json", tmp_path / "r.md"
    rc = pa.main(_cli_args(tmp_path, led, fld, "--spec-sha256", pa.SPEC_SHA256))
    assert rc == 0
    rep = json.loads(out.read_text())
    sha = pa.sha256_file(tmp_path / "execution.json")
    assert rep["registered"]["contrasts"]["rotation"]["resolved"] == 40
    assert rep["inputs"]["manifest"] == {"path": str(tmp_path / "execution.json"), "sha256": sha}
    assert rep["inputs"]["manifest_sha256_supplied"] == sha
    assert rep["inputs"]["frozen_manifest"]["sha256"] == pa.FROZEN_STAGED_MANIFEST_SHA256
    assert rep["manifests"]["execution"]["sha256"] == sha
    assert pa.LIMITATIONS in md.read_text() and sha in md.read_text()


def test_cli_refuses_missing_prediction_and_bad_manifests(tmp_path, manifest, capsys, monkeypatch):
    monkeypatch.setattr(pa, "BOOTSTRAP_DRAWS", DRAWS)
    led, fld = _write_inputs(tmp_path, manifest)
    fields = json.loads(fld.read_text())
    fields.pop(min(fields))
    fld.write_text(json.dumps(fields))
    out, md = tmp_path / "r.json", tmp_path / "r.md"
    spec = ["--spec-sha256", pa.SPEC_SHA256]
    assert pa.main(_cli_args(tmp_path, led, fld, *spec)) == 2  # fields hash no longer matches
    assert not out.exists()
    led, fld = _write_inputs(tmp_path, manifest)
    capsys.readouterr()
    sha = pa.sha256_file(tmp_path / "execution.json")
    tampered = tmp_path / "m.json"
    tampered.write_bytes(
        (tmp_path / "execution.json").read_bytes().replace(b'"scored": false', b'"scored": true')
    )
    frozen_bad = tmp_path / "a533.json"
    frozen_bad.write_bytes(FROZEN.read_bytes().replace(b'"scored": false', b'"scored": true'))
    led_bad = tmp_path / "ledger_bad.json"
    lb = json.loads(led.read_text())
    lb["restage_provenance"]["manifest_sha256"] = "1" * 64
    led_bad.write_text(json.dumps(lb))
    base = ["--ledger", str(led), "--fields", str(fld), "--output", str(out)]
    base += ["--markdown", str(md), *spec]
    cases = [
        (_manifest_args(tmp_path, tampered, sha), "execution manifest sha256"),
        (_manifest_args(tmp_path, sha="0" * 64), "execution manifest sha256"),
        (_manifest_args(tmp_path, sha="xyz"), "is required"),
        (_manifest_args(tmp_path, FROZEN), "ledger manifest_sha256"),  # a533 as execution
        (
            _manifest_args(tmp_path) + ["--frozen-manifest", str(frozen_bad)],
            "frozen staged manifest sha256",
        ),
        (
            _manifest_args(tmp_path) + ["--frozen-manifest", str(EXECUTION)],
            "frozen staged manifest sha256",
        ),
        (
            _manifest_args(tmp_path) + ["--ledger", str(led_bad)],
            "restage_provenance.manifest_sha256",
        ),
    ]
    for extra, needle in cases:
        assert pa.main(base + extra) == 2
        assert needle in capsys.readouterr().err
        assert not out.exists() and not md.exists()
    for drop in ("--manifest", "--manifest-sha256"):  # both are required
        args = _manifest_args(tmp_path)
        k = args.index(drop)
        with pytest.raises(SystemExit) as e:
            pa.main(base + args[:k] + args[k + 2 :])
        assert e.value.code == 2
    assert not out.exists()
    assert pa.main(base + _manifest_args(tmp_path)) == 0  # the untampered inputs run


def test_cli_refuses_fields_without_row_even_if_hash_matches(tmp_path, manifest):
    ledger, fields, _, execution = build(manifest)
    fields.pop(min(fields))
    fb = json.dumps(fields).encode()
    ledger["fields_sha256"] = hashlib.sha256(fb).hexdigest()
    (tmp_path / "f.json").write_bytes(fb)
    (tmp_path / "l.json").write_text(json.dumps(ledger))
    (tmp_path / "execution.json").write_bytes(manifest_bytes(execution))
    rc = pa.main(
        _cli_args(
            tmp_path, tmp_path / "l.json", tmp_path / "f.json", "--spec-sha256", pa.SPEC_SHA256
        )
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
    ledger, fields, _, execution = build(manifest, ladder_pred={i: -0.3 for i in ids})
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b=_invalid_b())
    dec = rep["registered"]["decision"]
    assert rep["registered"]["contrasts"]["ladder_0.25"]["agreements"] == 0
    assert _is_validity_unresolved(dec)
    assert rep["registered"]["part_b"]["supplied"]["valid"]["C"] is False
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
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
    ledger, fields, _, execution = build(manifest)
    with pytest.raises(pa.AnalysisRefused, match="specification"):
        analyse(execution, nullsize, ledger, fields, spec_check={"verified": False})
    rep = analyse(execution, nullsize, ledger, fields)  # default: hashes the file at run time
    v = rep["spec"]["verification"]
    assert v["sha256_file"] == v["sha256_f10b_note"] == pa.SPEC_SHA256


def _cli_args(tmp_path, led, fld, *extra):
    """CLI arguments with the execution manifest written by :func:`_write_inputs`."""
    return [
        "--ledger",
        str(led),
        "--fields",
        str(fld),
        "--output",
        str(tmp_path / "r.json"),
        "--markdown",
        str(tmp_path / "r.md"),
        *_manifest_args(tmp_path),
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
    ledger, fields, _, execution = build(manifest)
    mutate(ledger)
    with pytest.raises(pa.AnalysisRefused, match="window|block|text_groups|token"):
        analyse(execution, nullsize, ledger, fields)


def test_text_groups_file_tamper_refused(tmp_path):
    bad = tmp_path / "tg.json"
    bad.write_bytes(TEXT_GROUPS.read_bytes().replace(b'"item_id": "0"', b'"item_id": "00"'))
    with pytest.raises(pa.AnalysisRefused, match="text_groups.json sha256"):
        pa.load_text_groups(bad)
    tg = pa.load_text_groups(TEXT_GROUPS)
    assert tg["window_ids"] == [str(k) for k in range(64)]
    assert tg["two_window_blocks"][0] == [0, 1] and tg["two_window_blocks"][-1] == [62, 63]


def _restage_status(execution, nullsize, ledger, fields, i):
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(manifest)
    i = min(ledger["restage"])
    mutate(ledger["restage"][i])
    rec, rep = _restage_status(execution, nullsize, ledger, fields, i)
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
        ledger, fields, _, execution = build(manifest)
        mutate(ledger)
        rep = analyse(execution, nullsize, ledger, fields)
        led = rep["registered"]["contrasts"]["rotation"]["ledger"]
        assert led["f10_3_excluded"]["by_code"] == {"audit_missing": sorted(manifest["edit_ids"])}
        assert rep["registered"]["contrasts"]["rotation"]["resolved"] == 0
        assert rep["restage"]["yield"]["passing"] == 0
        # ladders unaffected
        assert rep["registered"]["contrasts"]["ladder_0.25"]["resolved"] == 40


def test_restage_hashes_must_be_linked_in_validity_evidence(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    v = valid_for(ledger)
    v["selection_and_audits"]["evidence"] = v["selection_and_audits"]["evidence"][:2]
    rep = analyse(execution, nullsize, ledger, fields, validity=v, part_b="ok")
    dec = rep["registered"]["decision"]
    assert _is_validity_unresolved(dec)
    assert PROV["readme_sha256"] in dec["validity"]["problems"]["selection_and_audits"]
    v = valid_for(ledger)
    v["registration_pinned"]["evidence"] = [
        e for e in v["registration_pinned"]["evidence"] if e["sha256"] != ledger["fields_sha256"]
    ]
    rep = analyse(execution, nullsize, ledger, fields, validity=v, part_b="ok")
    assert "registration_pinned" in rep["registered"]["decision"]["validity"]["problems"]
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
    assert rep["registered"]["decision"]["gate"] == "Part A go"


def test_contradictory_restage_verdict_refused(manifest, nullsize):
    ledger, fields, _, execution = build(manifest)
    i = min(ledger["restage"])
    ledger["restage"][i]["f10_3_exclusions"] = ["sub_resolution"]  # verdict still "pass"
    with pytest.raises(pa.AnalysisRefused, match="contradicts"):
        analyse(execution, nullsize, ledger, fields)
    ledger, fields, _, execution = build(manifest)
    ledger["restage"][i]["f10_3_verdict"] = "excluded"  # no reason code
    with pytest.raises(pa.AnalysisRefused, match="contradicts"):
        analyse(execution, nullsize, ledger, fields)


def test_complete_case_exclusion_is_not_an_audit_failure(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _, execution = build(manifest, drop_covariate={ids[0]: "ln_rho"})
    rep = analyse(execution, nullsize, ledger, fields)
    assert rep["registered"]["pairs"][ids[0]]["rotation"]["status"] == "resolved"
    cov = rep["f10_5_sensitivity"]["covariate"]
    assert [d["id"] for d in cov["dropped_rows"]] == [ids[0]]
    assert ids[0] in cov["resolved_ids"] and ids[0] not in cov["complete_case_ids"]
    assert rep["f10_6_calibration"]["status"] == "not required (40/40)"


# ------------------------------------------------------------------------------ (4) report contents


def test_every_pair_listed_with_status_and_cause(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _, execution = build(
        manifest,
        excluded={ids[0]: "replay_failed"},
        missing={ids[1]: "original"},
        d_mean={ids[2]: 0.0},
        noise=2e-3,
    )
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(manifest, predictions={ids[0]: 0.0, ids[1]: -0.2})
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(
        manifest,
        excluded={in_s30: "replay_failed"},
        predictions={"014f8a9ec78db2be": 0.0},
        drop_covariate={ids[-1]: "d_mean"},
    )
    rep = analyse(execution, nullsize, ledger, fields)
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
    ledger, fields, _, execution = build(
        manifest, restage_rho={low: math.exp(0.05)}, predictions={ids[3]: 0.0}
    )
    rep = analyse(execution, nullsize, ledger, fields, validity="auto", part_b="ok")
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
    ledger, fields, _, execution = build(
        manifest, excluded={min(manifest["edit_ids"]): "sub_resolution"}
    )
    with pytest.raises(pa.AnalysisRefused, match="F10.6"):
        analyse(execution, nullsize, ledger, fields, masked_calibration=None)


def test_masked_row_added_without_replacement(manifest, nullsize):
    i = min(manifest["edit_ids"])
    ledger, fields, _, execution = build(manifest, excluded={i: "sub_resolution"})
    masked, sha, _ = masked_for(ledger, execution)
    rep = analyse(
        execution,
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
    """Forty only (F10.6): a power failure leaves step 1 unresolved; a size failure now refuses
    the run outright (F11a.1(b): size first, scoring hold, a new amendment needed)."""
    ledger, fields, _, execution = build(
        manifest, excluded={min(manifest["edit_ids"]): "replay_failed"}
    )
    masked, sha, _ = masked_for(ledger, execution, passed=size_ok, power_passed=power_ok)
    kw = {
        "masked_calibration": masked,
        "masked_calibration_sha256": sha,
        "validity": valid_for(ledger, sha),
        "part_b": "ok",
    }
    if not size_ok:
        with pytest.raises(pa.AnalysisRefused, match=re.escape(pa.SIZE_HOLD)):
            analyse(execution, nullsize, ledger, fields, **kw)
        return
    rep = analyse(execution, nullsize, ledger, fields, **kw)
    dec = rep["registered"]["decision"]
    assert _is_validity_unresolved(dec)
    assert dec["validity"]["problems"]["f10_6_calibration"] == pa.F10_6_FAILED
    assert rep["f10_6_calibration"]["verdict"] == "FAIL"
    assert rep["f10_6_calibration"]["route"] == "f10_6_failed"
    assert pa.F10_6_FAILED in rep["f10_6_calibration"]["statement"]
    assert len(rep["f5_7_null_branch_table"]["rows"]) == 6  # still disclosed


def test_masked_calibration_must_match_exclusions(manifest, nullsize):
    ids = sorted(manifest["edit_ids"])
    ledger, fields, _, execution = build(manifest, excluded={ids[0]: "sub_resolution"})
    masked, sha, _ = masked_for(ledger, execution)
    bad = copy.deepcopy(masked)
    bad["slot_mask"]["masked_ids"]["rotation"] = [ids[1]]
    with pytest.raises(pa.AnalysisRefused, match="masked ids"):
        analyse(
            execution,
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
            execution,
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
            execution,
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
            execution,
            nullsize,
            ledger,
            fields,
            masked_calibration=bad,
            masked_calibration_sha256=sha,
        )
    with pytest.raises(pa.AnalysisRefused, match="sha256"):
        analyse(
            execution,
            nullsize,
            ledger,
            fields,
            masked_calibration=masked,
            masked_calibration_sha256=None,
        )


def test_masked_hash_must_be_linked_in_null_size_evidence(manifest, nullsize):
    ledger, fields, _, execution = build(
        manifest, excluded={min(manifest["edit_ids"]): "sub_resolution"}
    )
    masked, sha, _ = masked_for(ledger, execution)
    rep = analyse(
        execution,
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
    ledger, fields, _, execution = build(manifest)
    rep = analyse(execution, nullsize, ledger, fields)
    f106 = rep["f10_6_calibration"]
    assert f106["required"] is False and f106["statement"] == pa.F10_6_ALL_PASS
    assert "stand without a new run" in f106["statement"]
    assert len(rep["f5_7_null_branch_table"]["rows"]) == 5
    masked, sha, _ = masked_for(ledger, execution)
    with pytest.raises(pa.AnalysisRefused, match="all 40 pairs pass"):
        analyse(
            execution,
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
    ledger, _fields, fields_bytes, execution = build(
        manifest, excluded={i: "rounding_bound_failed"}
    )
    _masked, msha, mraw = masked_for(ledger, execution)
    (tmp_path / "fields.json").write_bytes(fields_bytes)
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "masked.json").write_bytes(mraw)
    (tmp_path / "validity.json").write_text(json.dumps(valid_for(ledger, msha)))
    (tmp_path / "part_b.json").write_text(json.dumps(PB()))
    (tmp_path / "execution.json").write_bytes(manifest_bytes(execution))
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
    assert rep["inputs"]["manifest"]["sha256"] == exec_sha(execution)
    md = (tmp_path / "r.md").read_text()
    assert "Gate: **Part A go**" in md and msha in md
