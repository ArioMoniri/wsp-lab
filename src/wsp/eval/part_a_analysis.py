"""Part A single analysis: registered A5.3/A5.5 outputs and the F10.5 sensitivity reports.

CPU/numpy only. Reads only the files it is given. Implements, without tuning:

* the **registered** Part A analysis (PREREG_H1_TF A5.3, A5.3.1, A5.3.2; F10.3 set; F10.9 flags):
  per-pair paired block-bootstrap resolution (32 two-window blocks, 4,000 draws, one common
  index array from ``numpy.random.Generator(numpy.random.PCG64(20261107))`` shared by every pair,
  arm and contrast; linear .025/.975 percentiles; resolved = interval strictly excludes zero),
  sign agreement (a zero prediction disagrees), exact CP 95 % intervals, the forty-ID ledger, and
  the A5.5 ordered decision (validity → ladder alternative → Part A evidence → Part B eligibility →
  funding), with the F5.7 null-branch table beside it;
* the **F10.5 reports** frozen by ``docs/design/F10_SENSITIVITY_SPEC.md`` (commit ``e577c67``):
  §1 covariate models M0/M1, §2 norm and ln ρ tertiles, §3 the S30/S34 subsets, §4 the
  constant-sign comparator, §5 the verbatim limitations paragraph, §6 denominators/units/methods.
  Every F10.5 output carries ``authority: none (descriptive)`` and lives under its own key, apart
  from the registered decision.

The paired-block helpers, the resolution draws and the CP intervals are reused from
``scripts/a5_simulation.py`` (``donor_block``, ``resolution``, ``cp_interval``).

Sealed ledger (JSON, schema ``a5-part-a-sealed-ledger-v1``)::

    {"schema": ..., "sealed": true,
     "manifest_sha256": <staged Part A manifest sha256>, "fields_sha256": <fields file sha256>,
     "window_ids": [64 registered WikiText item ids, in registered order],
     "references": {<edit id (original W+D) | predecessor id | "base">: {"mean_nll": [64]}},
     "variants": {<edit id>: {"rotation" | "ladder_0.25" | "ladder_0.5":
                  {"variant_id": ..., "mean_nll": [64] (absent if not scored),
                   "excluded": bool, "exclusions": [codes]}}},
     "restage": {<edit id>: {"pre_audit_exclusions": [codes], "f10_3_exclusions": [codes],
                 "registered_realised_verdict": bool, "pooled_rho": float,
                 "tensors": {<name>: {"shape": [...], "logged_d": {5 moments}, "logged_r": {...}}}}}}

``mean_nll`` holds TF-v1 WikiText ``extra.mean_nll`` per window (nats per valid next token),
``null`` where missing; :func:`window_nll_from_records` builds it from a ``records.jsonl``.
Fields JSON maps each selected edit id to three predictions in [rotation, ladder .25, ladder .5]
order (A5.3.2 convention), optionally wrapped as ``{"predictions": {...}}``.

    python scripts/part_a_analyse.py --ledger LEDGER.json --fields FIELDS.json \
        --output REPORT.json --markdown REPORT.md [--validity V.json] [--part-b B.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
try:
    from scripts import a5_simulation as sim
except ImportError:  # pragma: no cover - when run without the repository root on sys.path
    sys.path.insert(0, str(ROOT))
    from scripts import a5_simulation as sim

# ---------------------------------------------------------------------------- frozen constants

LEDGER_SCHEMA = "a5-part-a-sealed-ledger-v1"
REPORT_SCHEMA = "a5-part-a-report-v1"
UNITS = "nats per valid next token"
AUTHORITY_NONE = "none (descriptive)"
AUTHORITY_REGISTERED = "registered (A5.3, A5.5; decision on the F10.3 set)"
SEED = 20261107
RESOLUTION_DRAWS = 4000
BOOTSTRAP_DRAWS = 4000
WINDOWS, BLOCKS = 64, 32
FLOOR = 15
CONTRASTS = ("rotation", "ladder_0.25", "ladder_0.5")
LADDER_STRENGTH = {"ladder_0.25": 0.25, "ladder_0.5": 0.5}

DEFAULT_MANIFEST = ROOT / "docs/prereg/a5_manifests/part_a/manifest.json"
DEFAULT_NULLSIZE = ROOT / "docs/prereg/a5_manifests/nullsize/report.json"
STAGED_MANIFEST_SHA256 = "d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6"
NULLSIZE_REPORT_SHA256 = "1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6"
SPEC = {
    "path": "docs/design/F10_SENSITIVITY_SPEC.md",
    "commit": "e577c67",
    "sha256": "8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4",
}
SPEC_SHA256 = SPEC["sha256"]

PRE_AUDIT_CODES = ("non_2d", "non_finite", "zero_block", "rank_degenerate", "not_single_matrix")
F10_3_CODES = (
    "construction_failed",
    "support_failed",
    "replay_failed",
    "rounding_bound_failed",
    "sub_resolution",
)
AUDIT_MISSING = "audit_missing"  # F10.3: "A missing audit fails closed (F3(ii))"

# F10.9: the 8 non-root chain picks whose ladders scale the realised increment D_t.
REALISED_INCREMENT_LADDER_IDS = (
    "0027efdf2eae7405",
    "00f8111cbad8218f",
    "05531ce1db4df63f",
    "057dcfbedea3251f",
    "0bc0a0c4ad0520e0",
    "0d055869527177b4",
    "13b02fe2673cc296",
    "14cdaabee246493b",
)

# Spec §2.1: pooled sorted-id sha256 of the registered norm tertiles (manifest selections[].tertile).
NORM_TERTILE_SHA256 = (
    "64d442b072f81da1fdfaf4a7ee8a0c05f97c762904e42561b9711cc9845c0317",
    "041b916bf2fabb6d68f5b31c4981553a50f23061a43ff4c1653618be32822bbd",
    "7c663b27747ebb9f8f70490caa40a994cf06f94add40a3654465551d666a95a1",
)
TERTILE_NAMES = ("low", "middle", "high")

# Spec §2.2: frozen ln ρ cut keys and cell membership (ascending staged ln ρ).
LNRHO_CUTS = (
    (0.01010406252666643, "057dcfbedea3251f"),
    (0.03502604499592402, "22659cefc904e29f"),
)
LNRHO_CELLS = (
    (
        "15a53586efc72a50",
        "34e407b92a902282",
        "72d3756e968e3209",
        "8cfc1393b97300ab",
        "0027efdf2eae7405",
        "2233954b0cc3edc0",
        "2971f30e5276edd9",
        "5abd76a640cc2b9d",
        "14cdaabee246493b",
        "0559051354691a70",
        "00f8111cbad8218f",
        "07f9a50b29a7ef5c",
        "0712caa1d248514a",
    ),
    (
        "057dcfbedea3251f",
        "85366c308158ea48",
        "5f80bd482d092938",
        "05531ce1db4df63f",
        "014f8a9ec78db2be",
        "7d03d89c0b0299e6",
        "3996bea032c24200",
        "13b02fe2673cc296",
        "19d9a98cfb2db6e2",
        "2d7f5f78fac4a9ae",
        "2f61a4086395b719",
        "4f918f6bbd264b34",
        "2fb8d03d2547f357",
    ),
    (
        "22659cefc904e29f",
        "3499f7dcc5968d96",
        "983c5e13c59e0022",
        "1b2b671e5da5d625",
        "2d1de332e36d8f00",
        "453d8c08966438e0",
        "455b881d5de4e6e0",
        "1fc51f7164052d50",
        "0d055869527177b4",
        "649bd035e79488e5",
        "1ab850a41051fe45",
        "0a841d4082532e8b",
        "0bc0a0c4ad0520e0",
        "61d1f61e25212068",
    ),
)
LNRHO_CELL_SHA256 = (
    "2cc773dd85e33d3e36f068d97829960a2e9b43e4f90cdf47b56956ea7d630e96",
    "37f9eacdba5885cc88c487285233896cc131dbb662a87d4845665388c6c54980",
    "b3decd34fa74e474c3e83df5f4f8253f06d76d5cc9ee6ac4760722aa1afad6c6",
)
LNRHO_CELL_NAMES = ("low energy", "middle energy", "high energy")

# Spec §3: the two pre-listed subsets, frozen by id list and sorted-id sha256.
S30_IDS = (
    "0027efdf2eae7405",
    "00f8111cbad8218f",
    "014f8a9ec78db2be",
    "05531ce1db4df63f",
    "0559051354691a70",
    "057dcfbedea3251f",
    "0712caa1d248514a",
    "07f9a50b29a7ef5c",
    "13b02fe2673cc296",
    "14cdaabee246493b",
    "15a53586efc72a50",
    "19d9a98cfb2db6e2",
    "1b2b671e5da5d625",
    "2233954b0cc3edc0",
    "22659cefc904e29f",
    "2971f30e5276edd9",
    "2d7f5f78fac4a9ae",
    "2f61a4086395b719",
    "2fb8d03d2547f357",
    "3499f7dcc5968d96",
    "34e407b92a902282",
    "3996bea032c24200",
    "4f918f6bbd264b34",
    "5abd76a640cc2b9d",
    "5f80bd482d092938",
    "72d3756e968e3209",
    "7d03d89c0b0299e6",
    "85366c308158ea48",
    "8cfc1393b97300ab",
    "983c5e13c59e0022",
)
S30_SHA256 = "3da5f44f0c10307c2af7336c23fad430d13f6eb9347976e34de5d0810277267c"
S34_IDS = (
    "0027efdf2eae7405",
    "014f8a9ec78db2be",
    "05531ce1db4df63f",
    "0559051354691a70",
    "057dcfbedea3251f",
    "0712caa1d248514a",
    "07f9a50b29a7ef5c",
    "0d055869527177b4",
    "13b02fe2673cc296",
    "14cdaabee246493b",
    "15a53586efc72a50",
    "19d9a98cfb2db6e2",
    "1b2b671e5da5d625",
    "1fc51f7164052d50",
    "2233954b0cc3edc0",
    "22659cefc904e29f",
    "2971f30e5276edd9",
    "2d1de332e36d8f00",
    "2d7f5f78fac4a9ae",
    "2f61a4086395b719",
    "2fb8d03d2547f357",
    "3499f7dcc5968d96",
    "34e407b92a902282",
    "3996bea032c24200",
    "453d8c08966438e0",
    "455b881d5de4e6e0",
    "4f918f6bbd264b34",
    "5abd76a640cc2b9d",
    "5f80bd482d092938",
    "72d3756e968e3209",
    "7d03d89c0b0299e6",
    "85366c308158ea48",
    "8cfc1393b97300ab",
    "983c5e13c59e0022",
)
S34_SHA256 = "fb1c34e68d91c03b67b90ae1c0e867eaad32edadebf2ec4b615ce13515aedfa0"

MOMENT_KEYS = ("mean", "std", "frac_pos", "frac_nonzero", "cos_to_predecessor")
COVARIATE_COLUMNS = (
    "chain",
    "log1p_effective_norm",
    "d_mean",
    "d_std",
    "d_frac_pos",
    "d_frac_nonzero",
    "d_cos_to_predecessor",
    "ln_rho",
)
M0_COLUMNS = COVARIATE_COLUMNS[:7]
LAMBDA = 1.0
NEWTON_TOL, NEWTON_MAX_ITER = 1e-10, 100

# Spec §6.4 / F5.7 row order.
F57_SCENARIOS = (
    "registered",
    "variance_x2",
    "unavailable_audit_fails",
    "constant_positive_mean",
    "cell_mask_noise",
)

# A5.5.1 step-1 validity predicates decided outside this tool (each must link its evidence).
# The Part B numerical/destruction-control predicates enter step 4 through the Part B input.
VALIDITY_PREDICATES = (
    "registration_pinned",
    "source_closure",
    "qualification",
    "allocation",
    "g2_predecessor_audit",
    "selection_and_audits",
    "null_size",
)

COLLINEARITY_DISCLOSURE = (
    "Outcome-free collinearity disclosure (spec §1.1, staged values, 40 pairs): Pearson correlation "
    "of staged ln ρ with ln(1 + effective norm) is −0.735, Spearman −0.832. The ln ρ coefficient "
    "is therefore not separable from the norm term on these data; β̂_ρ is not an energy effect "
    "isolated from norm."
)
SUBSET_DISCLOSURE = (
    "A pair in S ∩ X is never replaced by another pair; S is never redefined (by re-stage values, "
    "new thresholds or any other list), and no further subset is reported under F10.5. The subset "
    "result is reported beside the F10.3 primary contrast and cannot replace its denominator "
    '(spec §3.3; Board A: subsets do not "authorize choosing another primary denominator after '
    'outcomes"). S30 ∩ S34 has 29 pairs and the two complements together hold 11: the subsets '
    'are not two independent checks. Board A: "The 30-pair subset removes ten low-tertile edits; '
    'it is not an independent confirmation."'
)
AUTHORITY_STATEMENT = (
    "None of the F10.5 reports has A5.5 or F5.5 authority. They do not enter A5.5's validity "
    "step, the step-2 ladder alternative, the step-3 Part A evidence rule or Part B eligibility, "
    "the F5.5 joint line-stop rule, any funding branch, or any gate of F10.6 or A5.3.2. A5.5 is "
    "evaluated once, on the F10.3 set. A sensitivity that disagrees with the main contrast is "
    "reported as a disagreement; it does not reverse, qualify or delay the decision (spec §6.5)."
)

# Verbatim texts of the specification (§5, §1.7, §4.3).
LIMITATIONS = (
    "**Limitations of the Part A rotation contrast and its F10.5 sensitivities.** (1) "
    "*Quantisation.* The scored rotation is R_eff = bf16(W + R) − W = R + E_R, a second bf16 "
    "rounding at the predecessor grid of the same size as the edit's own rounding (median "
    "ε_R/ε_D 1.0003 over 2,188 root-edit tensors; staged pooled realised norm mismatch "
    "+0.0007 to +0.0309; pooled ε_R up to 0.365, largest in the low norm tertile; "
    "`docs/design/ROTATION_YIELD_2026-11-11.md` §§1.2, 1.4). Its first-order part ⟨g, E_R⟩ is"
    " inside the registered prediction ⟨g, D_eff − R_eff⟩. The realised extra energy is not "
    "removed by design, and the exclusion rule (F10.3) does not test realised rotation "
    "quality. (2) *Curvature.* The second-order effect of that energy is not priced by the "
    "registered prediction. The F9 quadratic forecast's q is a nonnegative Gauss–Newton "
    "magnitude only, never a signed Hessian correction (A5.8.1; F9a.2(c)), and for the chain "
    "picks it uses base K-FAC factors (F6.2, `base-factor`). The constant-sign nuisance "
    "argument of the design note §2.3 is a plausible mechanism under positive, roughly "
    "isotropic curvature, not a theorem for every pair; fresh rounding depends on W and R and"
    " can correlate with the field or the prediction signs (Board A F10). (3) *Sign "
    "imbalance.* The registered power inputs carry rotation prediction signs 43 positive and "
    "98 negative of 141 donors "
    "(`docs/prereg/a5_manifests/power/fields_manifest.json:summary.sign_distribution.rotation`)."
    " Negative is the direction in which the energy nuisance pushes d_i, so a constant-sign "
    "nuisance can raise agreement when predictions are imbalanced towards negative. The "
    "report's predicted-sign split shows the actual balance. (4) *No realised spectrum "
    'matching.* "Spectrum-matched" refers to the pre-rounding construction only. No realised '
    "norm, spectrum or rank match is claimed (F10.4; A5.3.3), and passing the F10.3 audit is "
    "not evidence of one. (5) *Chain ladders.* For the 8 non-root chain picks, the 16 ladder "
    "variants scale the realised increment D_t, not the unavailable raw step Δ_t "
    "(`realised-increment ladder`, F10.9). Equivalence to the registered raw-step ladder is "
    "not established: the 0.066–0.392 noise-floor ratio is a heuristic scale, not a measured "
    "error or an upper bound (Board A F10, F10.9 ruling). (6) *Null branch frequency.* Under "
    "the registered NULL-SIZE proxy the ordered decision returns the funding-only "
    "**alternative** branch in 1,939 of 10,000 null replicates (rate 0.1939, exact CP 95 % "
    "[0.18619, 0.20179]), against ordered go 180/10,000 (0.018) "
    "(`docs/prereg/a5_manifests/nullsize/report.json`; F5.7). An alternative outcome is a "
    "funding stop for the field, never evidence of wrong signs. (7) *Proxy and dependence "
    "(F7).* The NULL-SIZE and POWER-AND-YIELD results (ordered power 0.8540) are conditional "
    "on F2's edit-minus-actual-predecessor noise surrogate, the |donor mean| effect scale and"
    " the 0.75 sign-agreement alternative. They are not empirical rotation-null or power "
    "certification. The single LoRA low-tertile donor is reused for all ten slots of that "
    "cell. The three contrasts rescale one donor block and under the null are one test, not "
    "three. Simulated yield 1.0 rests on treating 423 unavailable audit entries as passes, "
    "and the `unavailable_audit_fails` scenario resolves no pair (Board A F7 conditions 2–4)."
    " (8) *Scope.* The claim is agreement above chance conditional on the frozen field, the "
    "fixed 64-window panel and resolution, on development edits whose D3 outcomes were "
    'already known (A5.3 "Exposure"). It is not an item-population claim. Covariate '
    "adjustment, tertile and subset reports are descriptive: they neither certify an unbiased"
    " primary contrast nor validate its null, and none can change A5.5's outcome."
)
COVARIATE_SENTENCE = (
    "Covariate adjustment is a descriptive diagnostic of matching confounds. It does not "
    "certify an unbiased primary contrast, repair or validate its null, establish superiority"
    " over the constant-sign comparator, or change the primary interval, denominator or "
    "decision (A5.3.3; F10.5; Board A F10 condition 6)."
)
NO_SUPERIORITY_STATEMENT = (
    "No superiority of the field over the constant-sign comparator is claimed or tested. "
    "Covariate adjustment, tertile stratification and the pre-listed subsets do not establish"
    " such superiority, do not isolate directional geometry from quantisation or curvature, "
    "and do not certify the rotation null empirically (A5.3; F10.5; Board A F10 condition 6)."
)


class AnalysisRefused(RuntimeError):
    """An input fails a fail-closed check; no report is produced."""


class FitFailure(RuntimeError):
    """A ridge-logistic fit could not be completed (counted, never retried)."""


# ---------------------------------------------------------------------------- small helpers


def sorted_sha(ids) -> str:
    """sha256 of json.dumps(sorted(ids)) (``scripts/a5_build_panels.py:sorted_sha`` convention)."""
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cp(k: int, n: int):
    """Exact equal-tail two-sided 95 % Clopper–Pearson (A5.3.2); ``None`` when n = 0."""
    if n == 0:
        return None
    lo, hi = sim.cp_interval(int(k), int(n))
    return (0.0 if k == 0 else float(lo), 1.0 if k == n else float(hi))


def proportion(k: int, n: int) -> dict:
    iv = cp(k, n)
    return {
        "k": int(k),
        "n": int(n),
        "estimate": (k / n) if n else None,
        "cp95": list(iv) if iv else None,
        "interval": "exact CP 95 %" if n else "no resolved pairs",
    }


def _finite(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def window_nll_from_records(records_path, window_ids):
    """TF-v1 ``records.jsonl`` → WikiText ``extra.mean_nll`` per registered window (None if absent)."""
    out = {}
    for line in Path(records_path).read_text().splitlines():
        if '"wikitext_ppl"' not in line:
            continue
        r = json.loads(line)
        if r.get("benchmark") == "wikitext_ppl":
            out[str(r["item_id"])] = (r.get("extra") or {}).get("mean_nll")
    return [out.get(str(w)) for w in window_ids]


# ---------------------------------------------------------------------------- input loading


def load_manifest(path=DEFAULT_MANIFEST) -> dict:
    digest = sha256_file(path)
    if digest != STAGED_MANIFEST_SHA256:
        raise AnalysisRefused(
            f"manifest sha256 {digest} does not match the staged Part A manifest "
            f"{STAGED_MANIFEST_SHA256}"
        )
    return json.loads(Path(path).read_text())


def load_nullsize_report(path=DEFAULT_NULLSIZE) -> dict:
    digest = sha256_file(path)
    if digest != NULLSIZE_REPORT_SHA256:
        raise AnalysisRefused(
            f"NULL-SIZE report sha256 {digest} does not match {NULLSIZE_REPORT_SHA256}"
        )
    return json.loads(Path(path).read_text())


def _selected(manifest) -> dict:
    return {s["edit_id"]: s for s in manifest["selections"]}


def _variant_ids(manifest) -> dict:
    kinds = {
        "rotation": ("rotation", 1.0),
        "ladder_0.25": ("scale", 0.25),
        "ladder_0.5": ("scale", 0.5),
    }
    by_key = {
        (v["source_id"], v["kind"], float(v["strength"])): v["variant_id"]
        for v in manifest["variants"]
    }
    return {(i, c): by_key.get((i, *kinds[c])) for i in _selected(manifest) for c in CONTRASTS}


def _arm(values, n_windows):
    """Aligned window values → float array, or None if any window is missing/nonfinite."""
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != n_windows:
        raise AnalysisRefused(f"mean_nll must be a list of {n_windows} window values")
    if any(not _finite(v) for v in values):
        return None
    return np.asarray(values, dtype=float)


def validate_ledger(ledger, manifest) -> None:
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise AnalysisRefused(f"ledger schema must be {LEDGER_SCHEMA!r}")
    if ledger.get("sealed") is not True:
        raise AnalysisRefused(
            "ledger is not sealed (A5.3: one analysis after the ledger is closed)"
        )
    if ledger.get("manifest_sha256") != STAGED_MANIFEST_SHA256:
        raise AnalysisRefused("ledger manifest_sha256 does not match the staged Part A manifest")
    w = ledger.get("window_ids")
    if not isinstance(w, list) or len(w) != WINDOWS or len(set(map(str, w))) != WINDOWS:
        raise AnalysisRefused("ledger window_ids must list the 64 registered windows once each")
    expected = _variant_ids(manifest)
    for i in _selected(manifest):
        entry = ledger.get("variants", {}).get(i)
        if entry is None:
            raise AnalysisRefused(f"ledger lacks the variant entries of selected edit {i}")
        for c in CONTRASTS:
            v = entry.get(c)
            if v is None:
                raise AnalysisRefused(f"ledger lacks the {c} variant of {i}")
            if v.get("variant_id") != expected[(i, c)]:
                raise AnalysisRefused(
                    f"{i} {c}: variant id {v.get('variant_id')} != manifest {expected[(i, c)]}"
                )
        r = ledger.get("restage", {}).get(i)
        if r is not None:
            for key, allowed in (
                ("pre_audit_exclusions", PRE_AUDIT_CODES),
                ("f10_3_exclusions", F10_3_CODES),
            ):
                bad = [x for x in r.get(key, []) if x not in allowed]
                if bad:
                    raise AnalysisRefused(f"{i}: unknown {key} reason code(s) {bad}")
            excluded = bool(r.get("pre_audit_exclusions") or r.get("f10_3_exclusions"))
            if entry["rotation"].get("excluded") and not excluded:
                raise AnalysisRefused(
                    f"{i}: rotation variant marked excluded but the re-stage audit passes"
                )


def _excluded_contrasts(ledger, i) -> dict:
    r = ledger.get("restage", {}).get(i)
    rot = r is None or bool(r.get("pre_audit_exclusions") or r.get("f10_3_exclusions"))
    out = {"rotation": rot}
    for c in ("ladder_0.25", "ladder_0.5"):
        out[c] = bool(ledger["variants"][i][c].get("excluded"))
    return out


def validate_fields(fields, ledger, manifest) -> dict:
    """Every selected variant needs a frozen prediction row; null only for an excluded variant."""
    preds = fields.get("predictions", fields) if isinstance(fields, dict) else None
    if not isinstance(preds, dict):
        raise AnalysisRefused(
            "fields must map edit ids to [rotation, ladder .25, ladder .5] predictions"
        )
    out = {}
    for i in sorted(_selected(manifest)):
        row = preds.get(i)
        if not isinstance(row, list) or len(row) != 3:
            raise AnalysisRefused(f"no frozen prediction row (3 contrasts) for selected edit {i}")
        excluded = _excluded_contrasts(ledger, i)
        for c, p in zip(CONTRASTS, row):
            if p is None and excluded[c]:
                continue
            if not _finite(p):
                raise AnalysisRefused(f"no finite frozen prediction for variant {i} {c}")
        out[i] = row
    return out


# ---------------------------------------------------------------------------- registered contrast


def resolve(ledger, manifest, preds, draws=RESOLUTION_DRAWS) -> tuple[dict, dict]:
    """Per pair and contrast: status, reasons, d̂, paired block-bootstrap interval, agreement."""
    sel = _selected(manifest)
    n_w = WINDOWS
    refs = ledger.get("references", {})
    records: dict = {i: {} for i in sorted(sel)}
    columns, keys = [], []
    for i in sorted(sel):
        restage = ledger.get("restage", {}).get(i)
        original = _arm(refs.get(i, {}).get("mean_nll"), n_w)
        predecessor = _arm(refs.get(sel[i]["predecessor_id"], {}).get("mean_nll"), n_w)
        for k, c in enumerate(CONTRASTS):
            v = ledger["variants"][i][c]
            rec = {"variant_id": v["variant_id"], "prediction": preds[i][k], "status": None}
            if c != "rotation":
                rec["realised_increment_ladder"] = i in REALISED_INCREMENT_LADDER_IDS
            if c == "rotation":
                if restage is None:
                    rec.update(status="f10_3_excluded", reasons=[AUDIT_MISSING])
                elif restage.get("pre_audit_exclusions"):
                    rec.update(
                        status="pre_audit_excluded", reasons=list(restage["pre_audit_exclusions"])
                    )
                elif restage.get("f10_3_exclusions"):
                    rec.update(status="f10_3_excluded", reasons=list(restage["f10_3_exclusions"]))
                if restage is not None:
                    rec["registered_realised_verdict"] = restage.get("registered_realised_verdict")
                arms = {"original": original, "rotation": _arm(v.get("mean_nll"), n_w)}
            else:
                if v.get("excluded"):
                    rec.update(status="excluded", reasons=list(v.get("exclusions") or ["excluded"]))
                arms = {"ladder": _arm(v.get("mean_nll"), n_w), "predecessor": predecessor}
            if rec["status"] is None:
                missing = [a for a, x in arms.items() if x is None]
                if missing:
                    rec.update(status="missing_score", reasons=missing)
                else:
                    a, b = arms.values()
                    diff = a - b  # rotation: y(W+D) − y(W+R); ladder: y(W+sΔ) − y(W)
                    columns.append(sim.donor_block(diff, np.ones(n_w, dtype=bool)))
                    keys.append((i, c))
                    rec["d_hat"] = float(diff.mean())
            records[i][c] = rec
    rng = np.random.Generator(np.random.PCG64(SEED))
    if columns:
        intervals = sim.resolution(np.stack(columns, axis=1), rng, draws=draws)
        for j, (i, c) in enumerate(keys):
            rec = records[i][c]
            lo, hi = float(intervals[0, j]), float(intervals[1, j])
            rec["interval"] = [lo, hi]
            rec["resolved"] = bool(lo > 0 or hi < 0)
            rec["status"] = "resolved" if rec["resolved"] else "unresolved"
            if rec["resolved"]:
                p, d = rec["prediction"], rec["d_hat"]
                rec["observed_sign"] = _sign(d)
                rec["agree"] = bool(_sign(p) != 0 and _sign(p) == _sign(d))
    meta = {
        "method": "paired block bootstrap over 32 two-window blocks (A5.3)",
        "draws": draws,
        "seed": SEED,
        "generator": f"numpy.random.Generator(numpy.random.PCG64({SEED}))",
        "common_draws": f"one [{draws}, 32] index array shared by every pair, arm and contrast",
        "percentile": "linear interpolation at 0.025/0.975",
        "resolved_rule": "interval strictly excludes zero",
        "n_contrast_columns": len(columns),
        "code": "scripts/a5_simulation.py: donor_block, resolution, cp_interval",
    }
    return records, meta


def _ledger(records, contrast, ids) -> dict:
    """Forty-ID ledger of one contrast (spec §6.1); every count has its id list."""
    out: dict = {"selected": {"n": len(ids), "ids": sorted(ids)}}
    groups = (
        ("pre_audit_excluded", "by_code"),
        ("f10_3_excluded", "by_code"),
        ("excluded", "by_code"),
        ("missing_score", "by_arm"),
    )
    for status, key in groups:
        if contrast == "rotation" and status == "excluded":
            continue
        if contrast != "rotation" and status in ("pre_audit_excluded", "f10_3_excluded"):
            continue
        members = [i for i in sorted(ids) if records[i][contrast]["status"] == status]
        by: dict = {}
        for i in members:
            for r in records[i][contrast]["reasons"]:
                by.setdefault(r, []).append(i)
        out[status] = {"n": len(members), "ids": members, key: by}
    for status in ("unresolved", "resolved"):
        members = [i for i in sorted(ids) if records[i][contrast]["status"] == status]
        out[status] = {"n": len(members), "ids": members}
    return out


def _sign_table(resolved_recs) -> dict:
    table = {
        s: {"agree": 0, "disagree": 0, "observed_positive": 0, "observed_negative": 0}
        for s in ("negative", "zero", "positive")
    }
    for r in resolved_recs:
        row = table[("negative", "zero", "positive")[_sign(r["prediction"]) + 1]]
        row["agree" if r["agree"] else "disagree"] += 1
        if r["observed_sign"] > 0:
            row["observed_positive"] += 1
        elif r["observed_sign"] < 0:
            row["observed_negative"] += 1
    return table


def agreement_row(records, ids, contrast="rotation") -> dict:
    """Field agreement, comparator, 3 × 2 sign table and ledger for one set of pairs."""
    ids = sorted(ids)
    res = [records[i][contrast] for i in ids if records[i][contrast]["status"] == "resolved"]
    n = len(res)
    k = sum(r["agree"] for r in res)
    zero = sum(1 for r in res if _sign(r["prediction"]) == 0)
    field = proportion(k, n)
    row = {
        "ids": ids,
        "n_selected": len(ids),
        "ledger": _ledger(records, contrast, ids),
        "resolved": n,
        "agreements": k,
        "disagreements": n - k,
        "zero_prediction_disagreements": zero,
        "field": field,
        "cp95": field["cp95"],
        "sign_table": _sign_table(res),
        "observed_zero_resolved": sum(1 for r in res if r["observed_sign"] == 0),
        "negative_prediction_share": (
            sum(1 for r in res if _sign(r["prediction"]) < 0) / n if n else None
        ),
        "units": UNITS,
    }
    if contrast == "rotation":
        # Constant-sign comparator (A5.3; spec §4.1): predicts d > 0; d̂ = 0 disagrees.
        row["comparator"] = proportion(sum(1 for r in res if r["d_hat"] > 0), n)
    return row


# ---------------------------------------------------------------------------- A5.5 decision


def _validity(validity, internal=None) -> dict:
    internal = internal or {}
    failed = [k for k, v in internal.items() if not v["pass"]]
    if validity is None:
        return {
            "pass": False,
            "reason": "external validity predicates not supplied (A5.5.1 step-1 checklist)",
            "internal": internal,
        }
    for k in VALIDITY_PREDICATES:
        v = validity.get(k)
        if not isinstance(v, dict) or "pass" not in v:
            return {"pass": False, "reason": f"predicate not supplied: {k}", "internal": internal}
        if v["pass"] is not True:
            failed.append(k)
    if failed:
        return {
            "pass": False,
            "reason": f"failed predicates: {sorted(set(failed))}",
            "internal": internal,
        }
    return {
        "pass": True,
        "reason": "all step-1 predicates pass",
        "predicates": validity,
        "internal": internal,
    }


def _part_b_funding(part_b) -> str:
    if part_b is None:
        return "unresolved: Part B results not supplied to this tool"
    iv = part_b.get("intervals", {})
    valid = part_b.get("valid", {})
    for name in ("C", "C_abl", "C_rand", "C_perm"):
        if valid.get(name) is not True or name not in iv:
            return f"unresolved: {name} invalid or undefined (A5.4.2)"
    if part_b.get("destruction_controls_eligible") is not True:
        return "unresolved: destruction controls not eligible (A5.8.3)"
    for name in ("C_abl", "C_rand", "C_perm"):
        for proc in ("conditional", "joint"):
            if not iv[name][proc][0] > 0:
                return f"unresolved: {name} {proc} lower bound not > 0"
    c = part_b.get("C")
    if not _finite(c):
        return "unresolved: C undefined"
    lows = [iv["C"][p][0] for p in ("conditional", "joint")]
    ups = [iv["C"][p][1] for p in ("conditional", "joint")]
    if c >= 0.15 and all(lo > 0 for lo in lows):
        return "go"
    if c >= 0.05:
        return "screen"
    power = part_b.get("power_at_0_15", {})
    if all(u < 0.05 for u in ups) and all(
        _finite(power.get(p)) and power[p] >= 0.80 for p in ("conditional", "joint")
    ):
        return "no-go"
    return "unresolved"


def decision(validity, rotation, ladder_025, part_b=None, internal=None) -> dict:
    """A5.5.1 ordered if/else, evaluated once. ``rotation``/``ladder_025``: resolved, agreements, cp95."""
    v = _validity(validity, internal)
    out = {"validity": v, "authority": AUTHORITY_REGISTERED}
    if not v["pass"]:
        return out | {
            "gate": "unresolved (validity)",
            "step_2_ladder_alternative": None,
            "part_a": "not reached (validity)",
            "funding": "unresolved (validity)",
            "scientific": "nothing downstream runs (A5.5.1 step 1)",
        }
    lad_n, lad_cp = ladder_025["resolved"], ladder_025["cp95"]
    alternative = bool(lad_n >= FLOOR and lad_cp is not None and lad_cp[1] <= 0.6)
    out["step_2_ladder_alternative"] = alternative
    if alternative:
        return out | {
            "gate": "alternative",
            "part_a": "not reached (alternative overrides rotation go)",
            "funding": "stop: the field is not pursued; no sealed purchase on the field arm",
            "scientific": (
                "An alternative outcome is a funding stop for the field, never evidence of wrong "
                "signs (F5.7)."
            ),
        }
    n, iv = rotation["resolved"], rotation["cp95"]
    if n < FLOOR:
        part_a = "unresolved: fewer than 15 resolved rotation pairs"
    elif iv[0] > 0.5:
        part_a = "go"
    elif iv[1] < 0.5:
        part_a = "unresolved: interval entirely below 0.5 (observed disagreement reported)"
    else:
        part_a = "unresolved: interval includes 0.5"
    out["part_a"] = part_a
    out["gate"] = "Part A go" if part_a == "go" else "Part A unresolved"
    out["scientific"] = (
        "Agreement above chance conditional on the frozen field, the fixed 64-window panel and "
        "resolution; not an item-population claim."
        if part_a == "go"
        else "No scientific kill; Part A evidence is unresolved."
    )
    if part_a != "go":
        out["funding"] = "unresolved: Part A did not go (A5.5 step 4)"
    else:
        out["funding"] = _part_b_funding(part_b)
    out["part_b_reporting"] = (
        "always computed and reported (A5.5.1); supplied"
        if part_b
        else ("always computed and reported (A5.5.1); not supplied to this tool")
    )
    return out


def f57_table(nullsize) -> dict:
    rows = []
    for name in F57_SCENARIOS:
        s = nullsize["scenarios"][name]
        rows.append(
            {
                "scenario": name,
                "go_count": s["decisions"]["go"],
                "ordered_go_rate": s["ordered_go_rate"],
                "ordered_go_cp95": s["ordered_go_cp95"],
                "alternative_count": s["alternative_count"],
                "alternative_rate": s["alternative_rate"],
                "alternative_cp95": s["alternative_cp95"],
                "unresolved_count": s["decisions"]["unresolved"],
                "replicates": s["replicates"],
            }
        )
    return {
        "source": "docs/prereg/a5_manifests/nullsize/report.json",
        "sha256": NULLSIZE_REPORT_SHA256,
        "note": (
            "Null rates under the NULL-SIZE proxy (F5.7), distinct from the observed branch "
            "outcome of this analysis. An alternative outcome is a funding stop for the field, "
            "never evidence of wrong signs."
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------- covariate models


def standardise(X_train, X_apply, names):
    """Train-only z-scores (population SD); degenerate columns set to 0 for the fit (spec §1.3)."""
    X_train = np.asarray(X_train, dtype=float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    degenerate = [n for n, m, s in zip(names, mu, sd) if s <= 1e-12 * max(1.0, abs(m))]
    keep = np.array([n not in degenerate for n in names])
    safe = np.where(keep, sd, 1.0)

    def z(X):
        return np.where(keep, (np.asarray(X, dtype=float) - mu) / safe, 0.0)

    return z(X_train), (None if X_apply is None else z(X_apply)), degenerate


def _objective(X, a, beta, pen):
    eta = X @ beta
    return float(np.sum(np.logaddexp(0.0, eta) - a * eta) + 0.5 * np.sum(pen * beta * beta))


def _expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def fit_ridge_logistic(Z, a, lam=LAMBDA, tol=NEWTON_TOL, max_iter=NEWTON_MAX_ITER):
    """Ridge logistic, λ on Σβ² (intercept unpenalised), Newton–Raphson with step halving.

    Starts at β = 0, β₀ = logit(mean a). A step is halved while the objective does not decrease;
    a step whose objective is flat within floating-point resolution (|Δf| ≤ 8·eps·|f|) is
    accepted only if it reduces max |gradient|. Converged when max |gradient| ≤ 1e-10; at most
    100 iterations. Returns (β₀, β, iterations).
    """
    Z = np.asarray(Z, dtype=float)
    a = np.asarray(a, dtype=float)
    m = a.mean()
    if m <= 0.0 or m >= 1.0:
        raise FitFailure("one class in the training set")
    n, p = Z.shape
    X = np.hstack([np.ones((n, 1)), Z])
    pen = np.full(p + 1, lam)
    pen[0] = 0.0
    beta = np.zeros(p + 1)
    beta[0] = math.log(m / (1 - m))
    f = _objective(X, a, beta, pen)

    def grad(b):
        return X.T @ (_expit(X @ b) - a) + pen * b

    g = grad(beta)
    for it in range(max_iter + 1):
        gmax = float(np.max(np.abs(g)))
        if gmax <= tol:
            return float(beta[0]), beta[1:].copy(), it
        if it == max_iter:
            break
        pr = _expit(X @ beta)
        H = X.T @ (X * (pr * (1 - pr))[:, None]) + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError as e:
            raise FitFailure(f"singular Newton system: {e}") from e
        t, accepted = 1.0, False
        for _ in range(60):
            nb = beta - t * step
            nf = _objective(X, a, nb, pen)
            if nf < f:
                accepted = True
                break
            ng = grad(nb)
            if (
                abs(nf - f) <= 8 * np.finfo(float).eps * max(1.0, abs(f))
                and np.max(np.abs(ng)) < gmax
            ):
                accepted = True
                break
            t /= 2
        if not accepted:
            raise FitFailure("step halving failed to decrease the objective")
        beta, f = nb, nf
        g = grad(beta)
    raise FitFailure(f"no convergence in {max_iter} iterations")


def bootstrap_indices(n: int, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    """Component bootstrap index matrix (spec §6.3): PCG64(20261107), drawn once, shared by M0/M1."""
    return np.random.Generator(np.random.PCG64(SEED)).integers(0, n, size=(draws, n))


def _fit(X, a, names):
    Z, _, degenerate = standardise(X, None, names)
    b0, b, it = fit_ridge_logistic(Z, a)
    return b0, b, degenerate, it


def _interval(values, draws, failed):
    if not values:
        return {"ci95": None, "label": f"undefined ({failed} of {draws} draws failed)"}
    lo, hi = np.quantile(np.asarray(values), [0.025, 0.975], method="linear")
    label = (
        "bootstrap percentile 95 %"
        if failed == 0
        else f"diagnostic ({failed} of {draws} draws failed)"
    )
    return {"ci95": [float(lo), float(hi)], "label": label}


def _loo(X, a, columns):
    n = len(a)
    out = {
        m: {"log_loss": [], "brier": [], "failed_folds": 0} for m in ("intercept_only", "M0", "M1")
    }
    for j in range(n):
        tr = np.arange(n) != j
        a_tr = a[tr]
        one_class = a_tr.min() == a_tr.max()
        for m, cols in (("intercept_only", None), ("M0", columns[0]), ("M1", columns[1])):
            if one_class:
                out[m]["failed_folds"] += 1
                continue
            if cols is None:
                p = float(a_tr.mean())
            else:
                Xc = X[:, : len(cols)]
                Zt, Zh, _ = standardise(Xc[tr], Xc[~tr], list(cols))
                try:
                    b0, b, _ = fit_ridge_logistic(Zt, a_tr)
                except FitFailure:
                    out[m]["failed_folds"] += 1
                    continue
                p = float(_expit(b0 + Zh[0] @ b))
            y = a[j]
            p_c = min(max(p, 1e-300), 1 - 1e-16)
            out[m]["log_loss"].append(-(y * math.log(p_c) + (1 - y) * math.log1p(-p_c)))
            out[m]["brier"].append((p - y) ** 2)
    return {
        m: {
            "log_loss": float(np.mean(v["log_loss"])) if v["log_loss"] else None,
            "brier": float(np.mean(v["brier"])) if v["brier"] else None,
            "failed_folds": v["failed_folds"],
            "folds": n,
        }
        for m, v in out.items()
    }


def covariate_models(ids, X, a, draws=BOOTSTRAP_DRAWS) -> dict:
    """Spec §1.2–§1.6 on complete-case rows (canonical id order): intercept only, M0, M1."""
    order = np.argsort(np.asarray(ids))
    ids = [ids[k] for k in order]
    X = np.asarray(X, dtype=float)[order]
    a = np.asarray(a, dtype=float)[order]
    n = len(ids)
    out: dict = {"n_cc": n, "rows": ids}
    reason = None
    if n < FLOOR:
        reason = f"not estimable: n_cc = {n} < {FLOOR}"
    elif a.min() == a.max():
        reason = "not estimable: one agreement class only"
    if reason:
        for m in ("M0", "M1"):
            out[m] = {"status": reason}
        out["delta_theta"] = {"status": reason}
        out["loo"] = None
        return out
    specs = {"M0": list(M0_COLUMNS), "M1": list(COVARIATE_COLUMNS)}
    for m, cols in specs.items():
        try:
            b0, b, degenerate, it = _fit(X[:, : len(cols)], a, cols)
        except FitFailure as e:
            out[m] = {"status": f"not estimable: full fit failed ({e})"}
            continue
        out[m] = {
            "status": "estimated",
            "columns": cols,
            "intercept": b0,
            "coefficients": dict(zip(cols, map(float, b))),
            "theta_adj": {"point": float(_expit(b0))},
            "degenerate_columns": degenerate,
            "iterations": it,
        }
    idx = bootstrap_indices(n, draws)
    boot = {m: {"theta": [], "beta": {c: [] for c in specs[m]}, "failed": 0} for m in specs}
    delta, delta_failed = [], 0
    for row in idx:
        Xb, ab = X[row], a[row]
        theta = {}
        for m, cols in specs.items():
            if ab.min() == ab.max():
                boot[m]["failed"] += 1
                continue
            try:
                b0, b, _, _ = _fit(Xb[:, : len(cols)], ab, cols)
            except FitFailure:
                boot[m]["failed"] += 1
                continue
            theta[m] = float(_expit(b0))
            boot[m]["theta"].append(theta[m])
            for c, v in zip(cols, b):
                boot[m]["beta"][c].append(float(v))
        if len(theta) == 2:
            delta.append(theta["M1"] - theta["M0"])
        else:
            delta_failed += 1
    for m, cols in specs.items():
        if out[m]["status"] != "estimated":
            continue
        f = boot[m]["failed"]
        out[m]["failed_draws"] = f
        out[m]["draws"] = draws
        out[m]["theta_adj"] |= _interval(boot[m]["theta"], draws, f)
        out[m]["coefficient_intervals"] = {c: _interval(boot[m]["beta"][c], draws, f) for c in cols}
    if out["M1"]["status"] == "estimated":
        out["M1"]["beta_rho"] = {"point": out["M1"]["coefficients"]["ln_rho"]} | out["M1"][
            "coefficient_intervals"
        ]["ln_rho"]
    if out["M0"]["status"] == "estimated" and out["M1"]["status"] == "estimated":
        out["delta_theta"] = {
            "point": out["M1"]["theta_adj"]["point"] - out["M0"]["theta_adj"]["point"],
            "failed_draws": delta_failed,
        } | _interval(delta, draws, delta_failed)
    else:
        out["delta_theta"] = {"status": "not estimable"}
    out["loo"] = _loo(X, a, (specs["M0"], specs["M1"]))
    return out


def moment_differences(tensors) -> dict:
    """Element-count-weighted rotated-minus-original logged moments (spec §1.2)."""
    out = {}
    for k in MOMENT_KEYS:
        num = den = 0.0
        ok = bool(tensors)
        for t in (tensors or {}).values():
            n_t = math.prod(t.get("shape") or [0])
            r, d = (t.get("logged_r") or {}).get(k), (t.get("logged_d") or {}).get(k)
            if n_t <= 0 or not _finite(r) or not _finite(d):
                ok = False
                break
            num += n_t * (r - d)
            den += n_t
        out[f"d_{k}"] = num / den if ok and den > 0 else None
    return out


def _covariates(i, sel, restage) -> dict:
    s = sel[i]
    norm = s.get("effective_norm")
    rho = (restage or {}).get("pooled_rho")
    row = {
        "chain": 1.0 if s["family"] == "chain" else 0.0 if s["family"] == "lora_sft" else None,
        "log1p_effective_norm": math.log1p(norm) if _finite(norm) and norm > -1 else None,
    }
    row |= moment_differences((restage or {}).get("tensors"))
    row["ln_rho"] = math.log(rho) if _finite(rho) and rho > 0 else None
    return row


def staged_ln_rho(manifest) -> dict:
    """Spec §1.1: staged ln ρ = 2·ln(1 + pooled_norm_mismatch) (all staged signed values positive)."""
    return {
        i: 2 * math.log1p(manifest["edit_audits"][i]["edit_audit"]["pooled_norm_mismatch"])
        for i in _selected(manifest)
    }


def covariate_report(records, ledger, manifest, draws) -> dict:
    sel = _selected(manifest)
    resolved = [i for i in sorted(sel) if records[i]["rotation"]["status"] == "resolved"]
    rows, dropped, X, a = [], [], [], []
    for i in resolved:
        cov = _covariates(i, sel, ledger.get("restage", {}).get(i))
        bad = [c for c in COVARIATE_COLUMNS if not _finite(cov[c])]
        if bad:
            dropped.append({"id": i, "reason": f"covariate_missing:{bad[0]}", "columns": bad})
            continue
        rows.append(i)
        X.append([cov[c] for c in COVARIATE_COLUMNS])
        a.append(1.0 if records[i]["rotation"]["agree"] else 0.0)
    cc = agreement_row(records, rows)
    models = covariate_models(
        rows, np.asarray(X).reshape(len(rows), len(COVARIATE_COLUMNS)), a, draws
    )
    staged = staged_ln_rho(manifest)
    diffs = [
        abs(math.log(r["pooled_rho"]) - staged[i])
        for i, r in ledger.get("restage", {}).items()
        if i in staged and _finite(r.get("pooled_rho")) and r["pooled_rho"] > 0
    ]
    return {
        "authority": AUTHORITY_NONE,
        "spec": "§1 (F10.5(a))",
        "columns": list(COVARIATE_COLUMNS),
        "column_notes": {
            "chain": "1 = chain, 0 = lora_sft (manifest selections[].family)",
            "log1p_effective_norm": "ln(1 + manifest selections[].effective_norm)",
            "d_*": "element-count-weighted mean over edited tensors of logged_r − logged_d (re-stage variant.json)",
            "ln_rho": "ln of the re-stage F10.4 pooled ρ = Σ‖R_eff‖²/Σ‖D‖²",
        },
        "rules": {
            "penalty": "λ = 1 on Σβ², intercept unpenalised (sklearn LogisticRegression(C=1.0))",
            "solver": "Newton–Raphson from β = 0, β₀ = logit(mean a); step halving; max |grad| ≤ 1e-10; ≤ 100 iterations",
            "standardisation": "train-only mean and population SD per fit (full, LOO fold, bootstrap draw); degenerate column → 0",
            "folds": "leave-one-pair-out over the complete-case rows in canonical id order (diagnostics only)",
            "bootstrap": f"component bootstrap, {draws} draws, numpy.random.Generator(numpy.random.PCG64({SEED})), common draws for M0 and M1, linear .025/.975 percentiles",
            "floor": "fitted only if n_cc ≥ 15 and both agreement classes present",
        },
        "n_cc": len(rows),
        "dropped_rows": dropped,
        "theta_raw": cc["field"],
        "comparator": cc["comparator"],
        "sign_table": cc["sign_table"],
        "negative_prediction_share": cc["negative_prediction_share"],
        "full_analysis_set_agreement": agreement_row(records, resolved)["field"],
        "models": models,
        "max_abs_lnrho_restage_minus_staged": max(diffs) if diffs else None,
        "collinearity_disclosure": COLLINEARITY_DISCLOSURE,
        "sentence": COVARIATE_SENTENCE,
    }


# ---------------------------------------------------------------------------- tertiles and subsets


def norm_tertiles(manifest) -> dict:
    sel = _selected(manifest)
    cells = {}
    for t in range(3):
        pooled = sorted(i for i, s in sel.items() if s["tertile"] == t)
        if sorted_sha(pooled) != NORM_TERTILE_SHA256[t]:
            raise AnalysisRefused(f"norm tertile {t} membership does not match the frozen sha256")
        cells[t] = {
            "pooled": pooled,
            "lora_sft": [i for i in pooled if sel[i]["family"] == "lora_sft"],
            "chain": [i for i in pooled if sel[i]["family"] == "chain"],
        }
    return cells


def _lnrho_cell(x, i) -> int:
    key = (x, i)
    return 0 if key < LNRHO_CUTS[0] else 1 if key < LNRHO_CUTS[1] else 2


def lnrho_tertiles(manifest):
    """Spec §2.2: recompute from staged values and fail closed unless equal to the frozen lists."""
    x = staged_ln_rho(manifest)
    ordered = sorted(x, key=lambda i: (x[i], i))
    n = len(ordered)
    b1, b2 = n // 3, (2 * n) // 3
    cells = [ordered[:b1], ordered[b1:b2], ordered[b2:]]
    cuts = ((x[ordered[b1]], ordered[b1]), (x[ordered[b2]], ordered[b2]))
    if cuts != LNRHO_CUTS:
        raise AnalysisRefused(f"ln ρ cut keys {cuts} differ from the frozen {LNRHO_CUTS}")
    for k, c in enumerate(cells):
        if tuple(c) != LNRHO_CELLS[k] or sorted_sha(c) != LNRHO_CELL_SHA256[k]:
            raise AnalysisRefused(f"ln ρ cell {k} membership differs from the frozen id list")
    return [list(c) for c in LNRHO_CELLS], cuts


def _descriptive_row(records, ids, cell, label) -> dict:
    row = agreement_row(records, ids)
    return {"cell": cell, "authority": AUTHORITY_NONE, "label": label} | row


def tertile_report(records, ledger, manifest) -> dict:
    def label(r):
        return (
            "below floor (descriptive)"
            if r["resolved"] < FLOOR
            else "at or above floor (descriptive)"
        )

    norm_rows = []
    for t, cell in norm_tertiles(manifest).items():
        for group in ("pooled", "lora_sft", "chain"):
            r = _descriptive_row(records, cell[group], f"{TERTILE_NAMES[t]} ({t}) {group}", None)
            r["label"] = label(r)
            norm_rows.append(r)
    cells, cuts = lnrho_tertiles(manifest)
    restage = ledger.get("restage", {})
    lnrho_rows = []
    for k, ids in enumerate(cells):
        r = _descriptive_row(records, ids, LNRHO_CELL_NAMES[k], None)
        r["label"] = label(r)
        values = {}
        drift = []
        for i in ids:
            rho = (restage.get(i) or {}).get("pooled_rho")
            v = math.log(rho) if _finite(rho) and rho > 0 else None
            values[i] = v
            if v is not None and _lnrho_cell(v, i) != k:
                drift.append(i)
        r["ids_in_frozen_order"] = list(ids)
        r["restage_ln_rho"] = values
        r["cell_drift"] = drift
        lnrho_rows.append(r)
    return {
        "authority": AUTHORITY_NONE,
        "spec": "§2 (F10.5(b)); §2.2 cells supplementary, no greater authority",
        "note": (
            "Membership frozen by edit id; never recomputed from outcomes, resolution or re-stage "
            "values. cell_drift flags pairs whose re-stage ln ρ would fall in another cell under the "
            "frozen cuts; they are not moved. The two partitions largely coincide and are not "
            "independent views. No test across tertiles, trend statistic or difference interval "
            "is computed."
        ),
        "lnrho_cuts": [list(c) for c in cuts],
        "norm": norm_rows,
        "ln_rho": lnrho_rows,
    }


def subset_report(records, name, ids, expected_sha, selected) -> dict:
    ids = list(ids)
    digest = sorted_sha(ids)
    if digest != expected_sha or len(set(ids)) != len(ids) or not set(ids) <= set(selected):
        raise AnalysisRefused(
            f"{name} id list sha256 {digest} != frozen {expected_sha}; subset void"
        )
    x = []
    for i in sorted(ids):
        rec = records[i]["rotation"]
        if rec["status"] in ("pre_audit_excluded", "f10_3_excluded"):
            x.append({"id": i, "reason": ",".join(rec["reasons"])})
        elif rec["status"] == "missing_score":
            x.append({"id": i, "reason": "missing_score"})
    rest = sorted(set(ids) - {d["id"] for d in x})
    row = agreement_row(records, rest)
    n = row["resolved"]
    return {
        "subset": name,
        "authority": AUTHORITY_NONE,
        "size": len(ids),
        "sha256": digest,
        "ids": sorted(ids),
        "intersection_with_X": x,
        "unresolved_ids": row["ledger"]["unresolved"]["ids"],
        "resolved_n": n,
        "field": row["field"],
        "comparator": row["comparator"],
        "sign_table": row["sign_table"],
        "negative_prediction_share": row["negative_prediction_share"],
        "label": "resolved (n ≥ 15)" if n >= FLOOR else "below floor (n < 15)",
        "disclosure": SUBSET_DISCLOSURE,
        "units": UNITS,
    }


# ---------------------------------------------------------------------------- top level


def analyse(
    ledger,
    manifest,
    fields,
    nullsize,
    *,
    validity=None,
    part_b=None,
    resolution_draws=RESOLUTION_DRAWS,
    bootstrap_draws=BOOTSTRAP_DRAWS,
) -> dict:
    """The single Part A analysis: registered outputs plus the F10.5 reports."""
    validate_ledger(ledger, manifest)
    preds = validate_fields(fields, ledger, manifest)
    sel = _selected(manifest)
    ids = sorted(sel)
    # Fail closed on every frozen id list before any outcome is summarised.
    norm_tertiles(manifest)
    lnrho_tertiles(manifest)
    for name, sub, sha in (("S30", S30_IDS, S30_SHA256), ("S34", S34_IDS, S34_SHA256)):
        if sorted_sha(sub) != sha:
            raise AnalysisRefused(
                f"{name} id list sha256 {sorted_sha(sub)} != frozen {sha}; subset void"
            )

    records, res_meta = resolve(ledger, manifest, preds, draws=resolution_draws)
    contrasts = {c: agreement_row(records, ids, c) for c in CONTRASTS}
    for c in ("ladder_0.25", "ladder_0.5"):
        contrasts[c]["realised_increment_ladder_ids"] = sorted(REALISED_INCREMENT_LADDER_IDS)
    internal = {
        "null_size_report": {
            "pass": nullsize.get("verdict") == "PASS",
            "evidence": f"docs/prereg/a5_manifests/nullsize/report.json (sha256 {NULLSIZE_REPORT_SHA256})",
        },
        "manifest_sha256": {"pass": True, "evidence": STAGED_MANIFEST_SHA256},
    }
    dec = decision(validity, contrasts["rotation"], contrasts["ladder_0.25"], part_b, internal)

    comparator = {
        "authority": AUTHORITY_NONE,
        "spec": "§4 (F10.5(d))",
        "definition": (
            "Constant-sign comparator: predicted sign +1 for every pair (every trained edit "
            "degrades more than its rotation, d_i > 0); d̂_i = 0 disagrees. Same denominator and "
            "exact CP 95 % interval as the field agreement it sits beside."
        ),
        "main_contrast": {
            "field": contrasts["rotation"]["field"],
            "comparator": contrasts["rotation"]["comparator"],
            "sign_table": contrasts["rotation"]["sign_table"],
            "negative_prediction_share": contrasts["rotation"]["negative_prediction_share"],
        },
        "shown_beside": "main contrast, every tertile row, both subsets, covariate θ_raw",
        "statement": NO_SUPERIORITY_STATEMENT,
    }
    sensitivity = {
        "authority": AUTHORITY_NONE,
        "authority_statement": AUTHORITY_STATEMENT,
        "covariate": covariate_report(records, ledger, manifest, bootstrap_draws),
        "tertiles": tertile_report(records, ledger, manifest),
        "subsets": {
            "authority": AUTHORITY_NONE,
            "spec": "§3 (F10.5(c), (e))",
            "X_definition": (
                "X = pairs the F10 re-stage excludes (F10.3 or pre-audit codes) plus pairs with a "
                "missing or nonfinite score in either arm"
            ),
            "S30": subset_report(records, "S30", S30_IDS, S30_SHA256, sel),
            "S34": subset_report(records, "S34", S34_IDS, S34_SHA256, sel),
        },
        "constant_sign_comparator": comparator,
    }
    return {
        "schema": REPORT_SCHEMA,
        "spec": dict(SPEC),
        "units": UNITS,
        "registered": {
            "authority": AUTHORITY_REGISTERED,
            "resolution": res_meta,
            "contrasts": contrasts,
            "decision": dec,
            "pairs": records,
        },
        "f5_7_null_branch_table": f57_table(nullsize),
        "f10_5_sensitivity": sensitivity,
        "limitations": LIMITATIONS,
    }


# ---------------------------------------------------------------------------- markdown


def _fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _prop(p):
    if p["n"] == 0:
        return "no resolved pairs"
    lo, hi = p["cp95"]
    return f"{p['k']}/{p['n']} = {p['estimate']:.4f} [{lo:.4f}, {hi:.4f}]"


def _sign_md(tab):
    lines = [
        "| predicted sign | agree | disagree | observed d̂ > 0 | observed d̂ < 0 |",
        "|---|---|---|---|---|",
    ]
    for s in ("negative", "zero", "positive"):
        r = tab[s]
        label = "zero (abstain)" if s == "zero" else s
        lines.append(
            f"| {label} | {r['agree']} | {r['disagree']} | {r['observed_positive']} | {r['observed_negative']} |"
        )
    return lines


def _ledger_md(led):
    parts = []
    for key, val in led.items():
        extra = ""
        for sub in ("by_code", "by_arm"):
            if val.get(sub):
                extra = " (" + "; ".join(f"{k}: {len(v)}" for k, v in val[sub].items()) + ")"
        parts.append(f"{key} {val['n']}{extra}")
    return " → ".join(parts)


def _row_md(r):
    t = r["sign_table"]
    split = "/".join(
        str(t[s]["agree"] + t[s]["disagree"]) for s in ("negative", "zero", "positive")
    )
    return (
        f"| {r.get('cell', '')} | {_ledger_md(r['ledger'])} | {r['agreements']} / "
        f"{r['disagreements']} ({r['zero_prediction_disagreements']}) | {_prop(r['field'])} | "
        f"{_prop(r['comparator'])} | {split} | {_fmt(r['negative_prediction_share'])} | "
        f"{r.get('label', '')} |"
    )


def render_markdown(rep) -> str:
    reg = rep["registered"]
    sens = rep["f10_5_sensitivity"]
    L = [
        "# Part A analysis report",
        "",
        f"Units: d̂ and predictions in **{rep['units']}** (WikiText `extra.mean_nll`; positive Δy is a worse fit).",
        f"Specification: `{rep['spec']['path']}` (commit `{rep['spec']['commit']}`, sha256 `{rep['spec']['sha256']}`).",
        "",
        "## Registered outputs (A5.3, A5.5) — decision authority",
        "",
        f"authority: {reg['authority']}",
        "",
        (
            f"Resolution: {reg['resolution']['method']}; {reg['resolution']['draws']} draws; "
            f"{reg['resolution']['generator']}; {reg['resolution']['common_draws']}; "
            f"{reg['resolution']['percentile']}; {reg['resolution']['resolved_rule']}."
        ),
        "",
        "| contrast | resolved | field agreement [exact CP 95 %] | ledger |",
        "|---|---|---|---|",
    ]
    for c, r in reg["contrasts"].items():
        L.append(f"| {c} | {r['resolved']} | {_prop(r['field'])} | {_ledger_md(r['ledger'])} |")
    rot = reg["contrasts"]["rotation"]
    L += [
        "",
        f"Constant-sign comparator on the rotation contrast (same denominator, same interval): {_prop(rot['comparator'])}.",
        (
            f"Zero predictions inside the disagreements: {rot['zero_prediction_disagreements']}. "
            f"Share of negative predictions among resolved pairs: "
            f"{_fmt(rot['negative_prediction_share'])}."
        ),
        "",
        *_sign_md(rot["sign_table"]),
        "",
        "Ladder variants flagged `realised-increment ladder` (F10.9): "
        + ", ".join(
            f"`{i}`" for i in reg["contrasts"]["ladder_0.25"]["realised_increment_ladder_ids"]
        ),
        "",
        "### A5.5 ordered decision",
        "",
    ]
    d = reg["decision"]
    L += [
        f"- Step 1 validity: {'pass' if d['validity']['pass'] else 'FAIL'} — {d['validity']['reason']}",
        f"- Step 2 ladder alternative (s = 0.25): {d['step_2_ladder_alternative']}",
        f"- Step 3 Part A: {d['part_a']}",
        f"- Gate: **{d['gate']}**",
        f"- Funding (steps 4–5): {d['funding']}",
        f"- Scientific consequence (stated separately): {d['scientific']}",
        "",
        "### F5.7 null-branch table (null rates under the proxy, not observed outcomes)",
        "",
        "| scenario | ordered go (null) [CP 95 %] | alternative (null) count / rate [CP 95 %] | go / alternative / unresolved |",
        "|---|---|---|---|",
    ]
    for r in rep["f5_7_null_branch_table"]["rows"]:
        g, a = r["ordered_go_cp95"], r["alternative_cp95"]
        L.append(
            f"| `{r['scenario']}` | {r['ordered_go_rate']:.4f} [{g[0]:.4f}, {g[1]:.4f}] | "
            f"{r['alternative_count']:,} / {r['alternative_rate']:.4f} [{a[0]:.5f}, {a[1]:.5f}] | "
            f"{r['go_count']} / {r['alternative_count']:,} / {r['unresolved_count']:,} |"
        )
    L += [
        "",
        rep["f5_7_null_branch_table"]["note"],
        "",
        "## F10.5 sensitivity reports — no decision authority",
        "",
        f"authority: {sens['authority']}",
        "",
        sens["authority_statement"],
        "",
        "### Covariate analysis (spec §1)",
        "",
        f"authority: {sens['covariate']['authority']}",
        "",
    ]
    cov = sens["covariate"]
    mods = cov["models"]
    L += [
        (
            f"Columns: {', '.join(cov['columns'])}. n_cc = {cov['n_cc']}. "
            f"θ_raw {_prop(cov['theta_raw'])}; comparator {_prop(cov['comparator'])}; "
            f"full analysis set {_prop(cov['full_analysis_set_agreement'])}."
        ),
        "",
        "| row | n_cc | θ_raw [CP 95 %] | θ_adj [bootstrap 95 %] | Δθ vs M0 [bootstrap 95 %] | β̂_ρ [bootstrap 95 %] | other β̂ | LOO log loss / Brier | failed draws / folds | degenerate columns |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def ci(d):
        if not d or d.get("ci95") is None:
            return f"{_fmt(d.get('point') if d else None)} ({d.get('label', '') if d else ''})"
        return f"{_fmt(d.get('point'))} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] ({d['label']})"

    loo = mods.get("loo") or {}

    def loo_s(m):
        x = loo.get(m)
        return f"{_fmt(x['log_loss'])} / {_fmt(x['brier'])}" if x else "–"

    def folds(m):
        x = loo.get(m)
        return f"{x['failed_folds']}/{x['folds']}" if x else "–"

    L.append(
        f"| intercept only | {cov['n_cc']} | {_prop(cov['theta_raw'])} | – | – | – | – | {loo_s('intercept_only')} | – / {folds('intercept_only')} | – |"
    )
    for m, label in (("M0", "M0 (registered)"), ("M1", "M1 (M0 + ln ρ)")):
        x = mods[m]
        if x["status"] != "estimated":
            L.append(
                f"| {label} | {cov['n_cc']} | {_prop(cov['theta_raw'])} | {x['status']} | – | – | – | – | – | – |"
            )
            continue
        others = ", ".join(f"{c} {v:.3f}" for c, v in x["coefficients"].items() if c != "ln_rho")
        dt = ci(mods["delta_theta"]) if m == "M1" else "–"
        br = ci(x["beta_rho"]) if m == "M1" else "–"
        L.append(
            f"| {label} | {cov['n_cc']} | {_prop(cov['theta_raw'])} | {ci(x['theta_adj'])} | {dt} | {br} | {others} | "
            f"{loo_s(m)} | {x['failed_draws']}/{x['draws']} / {folds(m)} | {', '.join(x['degenerate_columns']) or '–'} |"
        )
    L += [
        "",
        "Dropped rows: "
        + (", ".join(f"`{d['id']}` ({d['reason']})" for d in cov["dropped_rows"]) or "none"),
        "",
    ]
    L += [
        f"max |ln ρ(re-stage) − ln ρ(staged)| = {_fmt(cov['max_abs_lnrho_restage_minus_staged'], 6)}.",
        "",
        cov["collinearity_disclosure"],
        "",
        cov["sentence"],
        "",
        "### Tertile reports (spec §2)",
        "",
        f"authority: {sens['tertiles']['authority']}",
        "",
        sens["tertiles"]["note"],
        "",
        (
            "| cell | ledger | agree / disagree (zero predictions) | field agreement [CP 95 %] "
            "| comparator [CP 95 %] | predicted −/0/+ | negative-prediction share | label |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sens["tertiles"]["norm"] + sens["tertiles"]["ln_rho"]:
        L.append(_row_md(r))
    drift = [i for r in sens["tertiles"]["ln_rho"] for i in r["cell_drift"]]
    L += ["", "ln ρ cell_drift (not moved): " + (", ".join(f"`{i}`" for i in drift) or "none"), ""]
    L += [
        "### Pre-listed subsets (spec §3)",
        "",
        f"authority: {sens['subsets']['authority']}",
        "",
        sens["subsets"]["X_definition"] + ".",
        "",
    ]
    for name in ("S30", "S34"):
        s = sens["subsets"][name]
        L += [
            (
                f"**{name}** (|S| = {s['size']}, sorted-id sha256 `{s['sha256']}`; "
                f"authority: {s['authority']}): resolved n in S \\ X = {s['resolved_n']}, "
                f"label `{s['label']}`; field {_prop(s['field'])}; "
                f"comparator {_prop(s['comparator'])}; "
                f"negative-prediction share {_fmt(s['negative_prediction_share'])}."
            ),
            "S ∩ X: "
            + (
                ", ".join(f"`{d['id']}` ({d['reason']})" for d in s["intersection_with_X"])
                or "none"
            )
            + ". Unresolved in S \\ X: "
            + (", ".join(f"`{i}`" for i in s["unresolved_ids"]) or "none")
            + ".",
            "",
            *_sign_md(s["sign_table"]),
            "",
        ]
    L += [SUBSET_DISCLOSURE, ""]
    c = sens["constant_sign_comparator"]
    L += [
        "### Constant-sign comparator (spec §4)",
        "",
        f"authority: {c['authority']}",
        "",
        c["definition"],
        "",
        f"Main contrast: field {_prop(c['main_contrast']['field'])}; comparator {_prop(c['main_contrast']['comparator'])}.",
        "",
        c["statement"],
        "",
        "## Limitations",
        "",
        rep["limitations"],
        "",
        "## Ledger appendix (rotation contrast, forty IDs)",
        "",
    ]
    for key, val in rot["ledger"].items():
        L.append(f"- {key} ({val['n']}): " + (", ".join(f"`{i}`" for i in val["ids"]) or "none"))
        for sub in ("by_code", "by_arm"):
            for k, v in (val.get(sub) or {}).items():
                L.append(f"  - {k}: " + ", ".join(f"`{i}`" for i in v))
    for cname in ("ladder_0.25", "ladder_0.5"):
        L.append("")
        L.append(f"Ledger ({cname}): {_ledger_md(reg['contrasts'][cname]['ledger'])}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------- CLI


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--fields", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--nullsize-report", type=Path, default=DEFAULT_NULLSIZE)
    ap.add_argument(
        "--validity", type=Path, help="A5.5.1 step-1 predicates {name: {pass, evidence}}"
    )
    ap.add_argument("--part-b", type=Path, help="Part B intervals and eligibility (A5.4)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--markdown", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        nullsize = load_nullsize_report(args.nullsize_report)
        ledger = json.loads(args.ledger.read_text())
        fields_sha = sha256_file(args.fields)
        if ledger.get("fields_sha256") != fields_sha:
            raise AnalysisRefused(
                f"fields sha256 {fields_sha} does not match the ledger's fields_sha256 "
                f"{ledger.get('fields_sha256')}"
            )
        fields = json.loads(args.fields.read_text())
        validity = json.loads(args.validity.read_text()) if args.validity else None
        part_b = json.loads(args.part_b.read_text()) if args.part_b else None
        rep = analyse(
            ledger,
            manifest,
            fields,
            nullsize,
            validity=validity,
            part_b=part_b,
            resolution_draws=RESOLUTION_DRAWS,
            bootstrap_draws=BOOTSTRAP_DRAWS,
        )
    except (AnalysisRefused, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    rep["inputs"] = {
        "ledger": {"path": str(args.ledger), "sha256": sha256_file(args.ledger)},
        "fields": {"path": str(args.fields), "sha256": fields_sha},
        "manifest_sha256": STAGED_MANIFEST_SHA256,
        "nullsize_report_sha256": NULLSIZE_REPORT_SHA256,
        "validity": str(args.validity) if args.validity else None,
        "part_b": str(args.part_b) if args.part_b else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=_json_default) + "\n"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(rep))
    print(f"wrote {args.output} and {args.markdown}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
