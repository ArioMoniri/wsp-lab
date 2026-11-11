"""Part A single analysis: registered A5.3/A5.5 outputs and the F10.5 sensitivity reports.

CPU/numpy only. Reads only the files it is given (plus the committed, outcome-free registration
metadata named below). Implements, without tuning:

* the **registered** Part A analysis (PREREG_H1_TF A5.3, A5.3.1, A5.3.2; F10.3 set; F10.9 flags):
  per-pair paired block-bootstrap resolution (32 two-window blocks, 4,000 draws, linear .025/.975
  percentiles; resolved = interval strictly excludes zero), sign agreement (a zero prediction
  disagrees), exact CP 95 % intervals, the forty-ID ledger, and the A5.5.1 ordered decision,
  evaluated once and **validity first**: step 1 (every external A5.5.1 checklist predicate with
  its linked evidence, the internal pins, the F10.6 calibration state and the A5.4.2 validity of
  the supplied Part B family) is decided before any contrast is read; an unresolved step 1 leaves
  every later step unresolved; then the step-2 ladder alternative, step-3 Part A evidence, step-4
  Part B eligibility and step-5 funding. The supplied Part B results are always reported, on every
  branch, as supplied (never computed or refitted here); the F5.7 null-branch table sits beside it;
* the **F10.5 reports** frozen by ``docs/design/F10_SENSITIVITY_SPEC.md`` (adopted by the F10b
  note at sha256 ``8624e063…98d4``): §1 covariate models M0/M1, §2 norm and ln ρ tertiles, §3 the
  S30/S34 subsets, §4 the constant-sign comparator, §5 the verbatim limitations paragraph, §6
  denominators/units/methods. Every F10.5 output carries ``authority: none (descriptive)`` and
  lives under its own key, apart from the registered decision; none enters step 1.

Fail-closed provenance (Board A, Part A analysis review of 11 Nov 2026, conditions 1–5): before any
output, the specification bytes are hashed and must equal the hash recorded in the F10b.1 adoption
note of ``docs/PREREG_H1_TF.md``, this module's constant and (CLI) the required ``--spec-sha256``;
``text_groups.json`` must have its A5.1.2 sha256, and the ledger must carry exactly its 64 window
item ids in registered order, its 32 two-window blocks and its per-window token-id hashes; every
selected pair needs a complete re-stage audit entry bound to a hashed re-stage report, otherwise
its rotation pair is excluded as ``audit_missing``; if the re-stage yield is below full yield a
hashed slot-masked calibration report (F10.6) is required, its gates and verdict are disclosed and
its embedded NULL-SIZE row is added to the F5.7 table without replacing a row.

Two manifests (Board A re-review condition 2: bind trusted input evidence)
--------------------------------------------------------------------------
* The **frozen staged manifest** is the preserved a533 staging manifest
  ``docs/prereg/a5_manifests/part_a/manifest_a533.json`` (sha256
  :data:`FROZEN_STAGED_MANIFEST_SHA256`, verified at load). The norm tertiles, the ln ρ cut keys
  and cells, the staged ln ρ values and the S30 recomputation are anchored there and never in a
  regenerated manifest (spec §§1.1, 2.2, 3.1; F10b.2 item 3: "frozen from the staged manifest; any
  pair whose re-stage value would cross a cut is flagged, never moved"). Its forty selections are
  the registered ids (sorted-id sha256 :data:`REGISTERED_EDIT_IDS_SHA256`).
* The **execution manifest** is the manifest whose variants were scored (F10 re-stage report,
  later regenerations). It is passed explicitly (CLI ``--manifest`` with the required
  ``--manifest-sha256``); the file bytes must hash to the supplied value, and the sealed ledger's
  ``manifest_sha256`` and ``restage_provenance.manifest_sha256`` must record the same hash. The
  F10 verdicts and reason codes (``edit_audits[i].pair_verdict`` / ``.exclusions``), the F10.4
  pooled ρ (``edit_audits[i].edit_audit.pooled_energy_ratio``, used for the covariate model and for
  ``cell_drift``), the ladder exclusion flags, the selection rows and the variant ids are read
  from it; the ledger's re-stage entries must agree with it (a disagreement refuses the run). The
  run is refused unless every registered id and every a533 variant id is in it, and unless every
  registered selection row equals its frozen row. A selection that is not registered must carry
  ``supplementary: true`` (F11 supplementary edits): supplementary edits are analysed and reported
  in the registered analysis and excluded from the registered-40 F10.5 sensitivity reports.

The paired-block helpers, the resolution draws and the CP intervals are reused from
``scripts/a5_simulation.py`` (``donor_block``, ``resolution``, ``cp_interval``).

Choices pending adopting note
-----------------------------
The following implementation choices are this code's, not registered text. Each is a **choice
pending adopting note** (a dated A5 note must adopt it, or replace it, before scoring):

1. *Sealed-ledger format* (schema ``a5-part-a-sealed-ledger-v2``: the v1 layout accepted by Board A,
   extended by the condition-3 bindings)::

       {"schema": ..., "sealed": true,
        "manifest_sha256": <execution manifest sha256>, "fields_sha256": <fields file sha256>,
        "text_groups_sha256": <text_groups.json sha256>,
        "window_ids": ["0", ..., "63"] (text_groups.json windows[].item_id, registered order),
        "two_window_blocks": [[0, 1], ..., [62, 63]] (text_groups.json two_window_blocks),
        "window_token_ids_hashes": [64 token_ids_hash values, aligned with window_ids],
        "restage_provenance": {"manifest_sha256": <execution manifest sha256>,
                               "audit_file_sha256": h, "readme_sha256": h},
        "references": {<edit id (original W+D) | predecessor id | "base">: {"mean_nll": [64]}},
        "variants": {<edit id>: {"rotation" | "ladder_0.25" | "ladder_0.5":
                     {"variant_id": ..., "mean_nll": [64] (absent if not scored),
                      "excluded": bool, "exclusions": [codes]}}},
        "restage": {<edit id>: {"f10_3_verdict": "pass" | "excluded",
                    "pre_audit_exclusions": [codes], "f10_3_exclusions": [codes],
                    "registered_realised_verdict": bool, "diagnostics": {F10.4 per-edit values},
                    "variant_json_sha256": h, "pooled_rho": float | null,
                    "tensors": {<name>: {"shape": [...], "logged_d": {5 moments}, "logged_r": {...}}}}}}

   ``mean_nll`` holds TF-v1 WikiText ``extra.mean_nll`` per window (nats per valid next token),
   ``null`` where missing; :func:`window_nll_from_records` builds it from a ``records.jsonl``.
   A re-stage entry that is absent, partial, malformed or lacks a verdict, or a ledger without
   complete ``restage_provenance``, excludes the rotation pair as ``audit_missing`` (F10.3: "A
   missing audit fails closed"); a contradictory entry (verdict vs reason codes) or an unknown
   reason code refuses the analysis. Missing covariates (``pooled_rho`` null, absent moments) are
   not audit failures: they drop the row from both covariate models only (spec §1.4).
   Fields JSON maps each selected edit id to three predictions in [rotation, ladder .25, ladder .5]
   order (A5.3.2 convention), optionally wrapped as ``{"predictions": {...}}``.
2. *Resolution seed and draws*: one index array from
   ``numpy.random.Generator(numpy.random.PCG64(20261107))`` (``sim.resolution``) shared by every
   pair, arm and contrast (common draws); resampling the paired difference column.
3. *Solver tie-break*: Newton–Raphson with step halving, where a step whose objective is flat
   within floating-point resolution (|Δf| ≤ 8·eps·|f|) is accepted only if it reduces max
   |gradient| (:func:`fit_ridge_logistic`); a numerical refinement, not an alternative model.
4. *Validity input* (``--validity``): ``{<predicate>: {"pass": bool, "evidence": [{"path": str,
   "sha256": <64 hex>}, ...]}}`` for every name in :data:`VALIDITY_PREDICATES`. A predicate
   certifies step 1 only with ``pass: true`` and a non-empty, well-formed evidence list that links
   the required pins (:func:`validity_bindings`): ``registration_pinned`` the specification,
   frozen staged (a533) manifest, text groups, fields and NULL-SIZE report hashes;
   ``selection_and_audits`` the three re-stage provenance hashes (whose manifest hash is the
   execution manifest's); ``null_size`` the NULL-SIZE report and, when F10.6 applies,
   the slot-masked calibration report hash. ``g2_predecessor_audit`` with ``pass: false`` carries
   the registered hard stop of G2 pending a new amendment.
5. *Part B input* (``--part-b``; produced by the Part B pipeline, not computed here):
   ``{"points": {C, C_abl, C_rand, C_perm}, "intervals": {name: {"conditional": [lo, hi],
   "joint": [lo, hi]}}, "valid": {name: bool}, "diagnostics": {name: {procedure: {"draws": 4000,
   "failed_draws": 0, "min_distinct_components": ≥ 2}}}, "destruction_controls_eligible": bool,
   "evidence": [{path, sha256}], "power_at_0_15": {"conditional": p, "joint": p} (optional)}``.
   Missing Part B, a false ``valid`` flag, a missing/nonfinite/reversed/malformed point or
   interval, failed or missing draws, fewer than two distinct components or ineligible destruction
   controls fail step 1 (A5.5.1: "Invalid/undefined statistics, including A5.4.2 failures, enter
   validity first"; checklist rows 8–9). A present but nonfinite or out-of-range power value is
   malformed; absent power only blocks the no-go branch.
6. *F10.6 calibration input* (``--masked-calibration``): the JSON output of
   ``scripts/a5_simulation.py --slot-mask`` (``slot_mask`` with label, mask file sha256, canonical
   mapping sha256, masked ids and gates; ``reports.null_size``). Its masked rotation ids must
   equal the re-stage exclusions and its canonical mapping sha256 must recompute from them.
   Required (else refusal) when the re-stage yield is below full yield; refused at full yield,
   where the registered F2/F7 simulations stand without a new run. A failed gate makes step 1
   unresolved (F10.6: "A failure makes Part A not decisive").

    python scripts/part_a_analyse.py --ledger LEDGER.json --fields FIELDS.json \
        --spec-sha256 8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4 \
        --manifest docs/prereg/a5_manifests/part_a/manifest.json --manifest-sha256 SHA256 \
        --output REPORT.json --markdown REPORT.md [--validity V.json] [--part-b B.json] \
        [--masked-calibration MASKED.json] [--frozen-manifest manifest_a533.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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

LEDGER_SCHEMA = "a5-part-a-sealed-ledger-v2"
REPORT_SCHEMA = "a5-part-a-report-v2"
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

# The preserved a533 staging manifest: the anchor of every frozen cut, cell and subset.
DEFAULT_FROZEN_MANIFEST = ROOT / "docs/prereg/a5_manifests/part_a/manifest_a533.json"
DEFAULT_NULLSIZE = ROOT / "docs/prereg/a5_manifests/nullsize/report.json"
DEFAULT_TEXT_GROUPS = ROOT / "docs/prereg/a5_manifests/text_groups.json"
DEFAULT_REGISTRATION = ROOT / "docs/PREREG_H1_TF.md"
FROZEN_STAGED_MANIFEST_SHA256 = "d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6"
# manifest_a533.json:selection_ledger.edit_ids_sorted_sha256 (the forty registered ids, A5.3.1).
REGISTERED_EDIT_IDS_SHA256 = "e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b"
N_REGISTERED = 40
NULLSIZE_REPORT_SHA256 = "1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6"
# A5.1.2 inventory row of docs/prereg/a5_manifests/text_groups.json (registered 64 windows).
TEXT_GROUPS_SHA256 = "f769afcade5a80571daf3a9578b1c03584b46a5abbbba07b32d43a7e9035e7e1"
SPEC = {
    "path": "docs/design/F10_SENSITIVITY_SPEC.md",
    "commit": "e577c67",
    "sha256": "8624e06343bbb401ff3b6d783572b0c53e1172d0ea3f36aec3768a13701998d4",
    "adopted_by": "docs/PREREG_H1_TF.md, A5 note F10b.1",
}
SPEC_SHA256 = SPEC["sha256"]
DEFAULT_SPEC = ROOT / SPEC["path"]
REGISTERED_WINDOW_IDS = tuple(str(k) for k in range(64))
REGISTERED_BLOCKS = tuple((2 * k, 2 * k + 1) for k in range(32))
RESTAGE_PROVENANCE_KEYS = ("manifest_sha256", "audit_file_sha256", "readme_sha256")
RESTAGE_REQUIRED_KEYS = (
    "f10_3_verdict",
    "pre_audit_exclusions",
    "f10_3_exclusions",
    "registered_realised_verdict",
    "diagnostics",
    "variant_json_sha256",
)
F10_3_VERDICTS = ("pass", "excluded")
_SHA_RE = re.compile(r"[0-9a-f]{64}")

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
# The two Part B checklist rows (A5.4.2 inference diagnostics; destruction eligibility and defined
# C/C_abl/C_rand/C_perm) are checked here from the Part B input, also in step 1.
VALIDITY_PREDICATES = (
    "registration_pinned",
    "source_closure",
    "qualification",
    "allocation",
    "g2_predecessor_audit",
    "selection_and_audits",
    "null_size",
)
PART_B_CONTRASTS = ("C", "C_abl", "C_rand", "C_perm")
PART_B_CONTROLS = ("C_abl", "C_rand", "C_perm")
PROCEDURES = ("conditional", "joint")
PART_B_DRAWS = 4000
G2_HARD_STOP = (
    "G2 predecessor audit failed: fewer than 16 of 20 passing predecessors is a hard stop of G2 "
    "pending a new amendment (exact-predecessor gradients may be registered there with their "
    "cost); it does not silently disable the approximation (A5.5.1 step 1)."
)
F10_6_ALL_PASS = (
    "All 40 pairs pass the F10.3 re-stage audit: the actual audit state equals the audit "
    "assumption of the registered NULL-SIZE and POWER-AND-YIELD runs, which stand without a new "
    "run (F2, F7; ordered power 0.8540) and remain conditional on the F2 proxy (F10.6). This is "
    "not a conversion of simulated yield into audit evidence; the evidence is the re-stage audit."
)
F10_6_FAILED = (
    "The slot-masked calibration (F10.6) failed its gates: Part A is not decisive and a further "
    "amendment is needed before scoring."
)
SLOT_MASK_LABEL = "slot-masked (F10.6)"
SLOT_MASK_SCHEMA = "a5-slot-mask-v1"

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
    """A ridge-logistic fit could not be completed (counted, never retried).

    ``info`` carries the fit diagnostics at failure (converged False, reason, iterations,
    max |gradient|) so failed fits are reported, not only counted.
    """

    def __init__(self, reason, info=None):
        super().__init__(reason)
        self.info = {"converged": False, "reason": reason} | (info or {})


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


def _is_sha(x) -> bool:
    return isinstance(x, str) and _SHA_RE.fullmatch(x) is not None


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


def load_frozen_manifest(path=DEFAULT_FROZEN_MANIFEST) -> dict:
    """The preserved a533 staging manifest, verified by its constant sha256 before it is parsed."""
    try:
        digest = sha256_file(path)
    except OSError as e:
        raise AnalysisRefused(f"frozen staged manifest {path} missing or unreadable: {e}") from e
    if digest != FROZEN_STAGED_MANIFEST_SHA256:
        raise AnalysisRefused(
            f"frozen staged manifest sha256 {digest} does not match the preserved a533 staging "
            f"manifest {FROZEN_STAGED_MANIFEST_SHA256}"
        )
    return json.loads(Path(path).read_text())


def load_execution_manifest(path, expected_sha256) -> dict:
    """The manifest whose variants were scored; its sha256 must be supplied and must match."""
    if not _is_sha(expected_sha256):
        raise AnalysisRefused(
            "the execution manifest sha256 (--manifest-sha256, 64 lowercase hex) is required"
        )
    try:
        digest = sha256_file(path)
    except OSError as e:
        raise AnalysisRefused(f"execution manifest {path} missing or unreadable: {e}") from e
    if digest != expected_sha256:
        raise AnalysisRefused(
            f"execution manifest sha256 {digest} does not match the supplied --manifest-sha256 "
            f"{expected_sha256}"
        )
    return json.loads(Path(path).read_text())


def bind_manifests(frozen, manifest, manifest_sha256) -> dict:
    """Bind the execution manifest to the frozen a533 manifest; refuse on any inconsistency.

    The forty registered ids are the frozen manifest's selections. Every one must be selected in
    the execution manifest with an identical selection row (apart from an optional false
    ``supplementary`` flag), every a533 variant id must be present, and every other selection must
    be flagged ``supplementary: true``.
    """
    if not _is_sha(manifest_sha256):
        raise AnalysisRefused("the execution manifest sha256 is required (64 lowercase hex)")
    frozen_sel = _selected(frozen)
    registered = sorted(frozen_sel)
    if (
        len(frozen["selections"]) != N_REGISTERED
        or len(registered) != N_REGISTERED
        or sorted_sha(registered) != REGISTERED_EDIT_IDS_SHA256
        or (frozen.get("selection_ledger") or {}).get("edit_ids_sorted_sha256")
        != REGISTERED_EDIT_IDS_SHA256
    ):
        raise AnalysisRefused(
            "frozen staged manifest does not carry the forty registered ids "
            f"(sorted-id sha256 {REGISTERED_EDIT_IDS_SHA256})"
        )
    rows = manifest.get("selections")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise AnalysisRefused("execution manifest has no selections list")
    sel = _selected(manifest)
    if len(sel) != len(rows):
        raise AnalysisRefused("execution manifest selects an edit id more than once")
    missing = [i for i in registered if i not in sel]
    if missing:
        raise AnalysisRefused(
            f"execution manifest lacks registered id(s) {missing}: the a533 ids must be a subset "
            "of the execution manifest's ids"
        )
    frozen_variants = {v["variant_id"] for v in frozen["variants"]}
    exec_variants = {v.get("variant_id") for v in manifest.get("variants") or []}
    lost = sorted(frozen_variants - exec_variants)
    if lost:
        raise AnalysisRefused(
            f"execution manifest lacks a533 variant id(s) {lost}: the a533 ids must be a subset "
            "of the execution manifest's ids"
        )
    for i in registered:
        row = {k: v for k, v in sel[i].items() if k != "supplementary"}
        if sel[i].get("supplementary", False) is not False:
            raise AnalysisRefused(f"registered id {i} is flagged supplementary")
        if row != frozen_sel[i]:
            raise AnalysisRefused(
                f"selection row of registered id {i} differs from the frozen staged manifest"
            )
    supplementary = sorted(i for i in sel if i not in frozen_sel)
    stray = [i for i in supplementary if sel[i].get("supplementary") is not True]
    if stray:
        raise AnalysisRefused(
            f"execution manifest selects id(s) {stray} that are neither registered nor flagged "
            "supplementary: true"
        )
    return {
        "frozen_manifest_sha256": FROZEN_STAGED_MANIFEST_SHA256,
        "execution_manifest_sha256": manifest_sha256,
        "registered_ids": registered,
        "supplementary_ids": supplementary,
        "selected_ids": sorted(sel),
    }


def load_nullsize_report(path=DEFAULT_NULLSIZE) -> dict:
    digest = sha256_file(path)
    if digest != NULLSIZE_REPORT_SHA256:
        raise AnalysisRefused(
            f"NULL-SIZE report sha256 {digest} does not match {NULLSIZE_REPORT_SHA256}"
        )
    return json.loads(Path(path).read_text())


def f10b_spec_hash(registration_path=DEFAULT_REGISTRATION) -> str:
    """The specification sha256 recorded in the F10b.1 adoption note of the registration text."""
    try:
        text = Path(registration_path).read_text()
    except OSError as e:
        raise AnalysisRefused(f"registration {registration_path} unreadable: {e}") from e
    start = text.find("**F10b.1")
    if start < 0:
        raise AnalysisRefused("F10b.1 adoption note not found in the registration text")
    end = text.find("**F10b.2", start)
    block = text[start : end if end > 0 else None]
    m = re.search(r"sha256 `([0-9a-f]{64})`", block)
    if SPEC["path"] not in block or m is None:
        raise AnalysisRefused("F10b.1 does not record the specification path and sha256")
    return m.group(1)


def verify_spec(spec_path=DEFAULT_SPEC, registration_path=DEFAULT_REGISTRATION, supplied=None):
    """F10b.1 / Board A condition 2: hash the specification bytes now; refuse on any mismatch.

    The file sha256 must equal the hash recorded in the F10b.1 note, this module's constant and,
    when given (the CLI requires it), the operator-supplied ``--spec-sha256``.
    """
    try:
        data = Path(spec_path).read_bytes()
    except OSError as e:
        raise AnalysisRefused(f"specification {spec_path} missing or unreadable: {e}") from e
    file_sha = hashlib.sha256(data).hexdigest()
    note_sha = f10b_spec_hash(registration_path)
    if note_sha != SPEC_SHA256:
        raise AnalysisRefused(
            f"F10b.1 records specification sha256 {note_sha}, this code expects {SPEC_SHA256}"
        )
    if file_sha != note_sha:
        raise AnalysisRefused(
            f"specification sha256 {file_sha} differs from the F10b.1 adopted hash {note_sha}"
        )
    if supplied is not None and supplied != file_sha:
        raise AnalysisRefused(
            f"--spec-sha256 {supplied} does not match the specification bytes {file_sha}"
        )
    return {
        "path": SPEC["path"],
        "file_read": str(spec_path),
        "sha256_file": file_sha,
        "sha256_f10b_note": note_sha,
        "sha256_supplied": supplied,
        "sha256_code": SPEC_SHA256,
        "verified": True,
        "rule": "file bytes = F10b.1 hash = code constant (= --spec-sha256); refused otherwise",
    }


def load_text_groups(path=DEFAULT_TEXT_GROUPS) -> dict:
    """Registered 64-window order, two-window blocks and token hashes, verified by A5.1.2 hash."""
    digest = sha256_file(path)
    if digest != TEXT_GROUPS_SHA256:
        raise AnalysisRefused(
            f"text_groups.json sha256 {digest} does not match the registered {TEXT_GROUPS_SHA256}"
        )
    tg = json.loads(Path(path).read_text())
    windows = tg["windows"]
    ids = [str(w["item_id"]) for w in windows]
    blocks = [tuple(b) for b in tg["two_window_blocks"]]
    tokens = [w["token_ids_hash"] for w in windows]
    if tuple(ids) != REGISTERED_WINDOW_IDS:
        raise AnalysisRefused("text_groups.json window order is not item ids 0 through 63")
    if tuple(blocks) != REGISTERED_BLOCKS:
        raise AnalysisRefused("text_groups.json two-window blocks are not [[0, 1], …, [62, 63]]")
    if sorted_sha([w["window_id"] for w in windows]) != tg["window_ids_sorted_sha256"]:
        raise AnalysisRefused("text_groups.json window ids do not match their sorted sha256")
    if sorted_sha(tokens) != tg["token_ids_hashes_sorted_sha256"]:
        raise AnalysisRefused("text_groups.json token hashes do not match their sorted sha256")
    return {
        "path": "docs/prereg/a5_manifests/text_groups.json",
        "sha256": digest,
        "window_ids": ids,
        "two_window_blocks": [list(b) for b in blocks],
        "token_ids_hashes": tokens,
    }


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


def validate_windows(ledger, text_groups) -> None:
    """Board A condition 3: the ledger's arrays are aligned to the registered window order."""
    if ledger.get("text_groups_sha256") != text_groups["sha256"]:
        raise AnalysisRefused("ledger text_groups_sha256 does not bind the registered text groups")
    w = ledger.get("window_ids")
    if not isinstance(w, list) or len(w) != WINDOWS or len(set(map(str, w))) != WINDOWS:
        raise AnalysisRefused("ledger window_ids must list the 64 registered windows once each")
    if any(not isinstance(x, str) for x in w) or w != text_groups["window_ids"]:
        raise AnalysisRefused(
            "ledger window order differs from the registered text_groups.json order (item ids "
            "'0' … '63')"
        )
    blocks = ledger.get("two_window_blocks")
    if blocks != text_groups["two_window_blocks"]:
        raise AnalysisRefused("ledger two-window block assignment differs from text_groups.json")
    if ledger.get("window_token_ids_hashes") != text_groups["token_ids_hashes"]:
        raise AnalysisRefused(
            "ledger window token-id hashes differ from text_groups.json (scoring identity)"
        )
    # sim.donor_block pairs consecutive positions; certify that equals the registered blocks.
    pos = {x: k for k, x in enumerate(w)}
    if [[pos[str(a)], pos[str(b)]] for a, b in blocks] != [list(b) for b in REGISTERED_BLOCKS]:
        raise AnalysisRefused("registered blocks are not consecutive positions of the ledger")


def _manifest_verdict(manifest, i):
    """Execution manifest F10 verdict of edit ``i``: (verdict, pre-audit codes, F10.3 codes) or
    a list of evidence problems (→ ``audit_missing``)."""
    ea = (manifest.get("edit_audits") or {}).get(i)
    if not isinstance(ea, dict):
        return ["execution manifest has no edit_audits entry"]
    problems = []
    if ea.get("audit_rule") != "f10":
        problems.append(f"execution manifest audit_rule {ea.get('audit_rule')!r} is not 'f10'")
    if ea.get("pair_verdict") not in F10_3_VERDICTS:
        problems.append(
            f"execution manifest pair_verdict {ea.get('pair_verdict')!r} not in {F10_3_VERDICTS}"
        )
    exc = ea.get("exclusions")
    if not isinstance(exc, list) or not all(
        isinstance(x, dict) and isinstance(x.get("reason"), str) for x in exc
    ):
        problems.append("execution manifest exclusions malformed")
    if problems:
        return problems
    codes = sorted({x["reason"] for x in exc})
    bad = [c for c in codes if c not in PRE_AUDIT_CODES + F10_3_CODES]
    if bad:
        raise AnalysisRefused(f"{i}: unknown execution manifest reason code(s) {bad}")
    if bool(codes) != (ea["pair_verdict"] == "excluded"):
        raise AnalysisRefused(
            f"{i}: execution manifest pair_verdict {ea['pair_verdict']!r} contradicts its "
            "reason codes"
        )
    pre = [c for c in codes if c in PRE_AUDIT_CODES]
    f10 = [c for c in codes if c in F10_3_CODES]
    return ea["pair_verdict"], pre, f10


def restage_rho(manifest) -> dict:
    """Execution manifest F10.4 pooled ρ per selected edit (None when absent or nonfinite)."""
    out = {}
    for i in _selected(manifest):
        ea = ((manifest.get("edit_audits") or {}).get(i) or {}).get("edit_audit") or {}
        rho = ea.get("pooled_energy_ratio") if isinstance(ea, dict) else None
        out[i] = float(rho) if _finite(rho) and rho > 0 else None
    return out


def restage_audit(ledger, manifest) -> dict:
    """Per selected pair: F10.3 re-stage audit state; incomplete evidence → ``audit_missing``.

    The verdict and reason codes are the execution manifest's; the sealed ledger's re-stage entry
    must be complete (else ``audit_missing``) and must agree with them (else refusal).
    """
    prov = ledger.get("restage_provenance")
    prov_missing = [
        k for k in RESTAGE_PROVENANCE_KEYS if not (isinstance(prov, dict) and _is_sha(prov.get(k)))
    ]
    rhos = restage_rho(manifest)
    out = {}
    for i in sorted(_selected(manifest)):
        r = ledger.get("restage", {}).get(i)
        missing = []
        verdict = _manifest_verdict(manifest, i)
        if isinstance(verdict, list):
            missing.extend(verdict)
        if prov_missing:
            missing.append(f"restage_provenance lacks {prov_missing}")
        if not isinstance(r, dict) or not r:
            missing.append("no re-stage entry")
            r = {}
        else:
            absent = [k for k in RESTAGE_REQUIRED_KEYS if k not in r]
            if absent:
                missing.append(f"entry lacks {absent}")
            if "f10_3_verdict" in r and r["f10_3_verdict"] not in F10_3_VERDICTS:
                missing.append(f"verdict {r['f10_3_verdict']!r} not in {F10_3_VERDICTS}")
            for k in ("pre_audit_exclusions", "f10_3_exclusions"):
                if k in r and not isinstance(r[k], list):
                    missing.append(f"{k} is not a list")
            if "registered_realised_verdict" in r and not isinstance(
                r["registered_realised_verdict"], bool
            ):
                missing.append("registered_realised_verdict is not a boolean")
            if "diagnostics" in r and not (isinstance(r["diagnostics"], dict) and r["diagnostics"]):
                missing.append("diagnostics empty or malformed")
            if "variant_json_sha256" in r and not _is_sha(r["variant_json_sha256"]):
                missing.append("variant_json_sha256 is not a sha256")
        if missing:
            out[i] = {"status": "f10_3_excluded", "reasons": [AUDIT_MISSING], "detail": missing}
            continue
        for key, allowed in (
            ("pre_audit_exclusions", PRE_AUDIT_CODES),
            ("f10_3_exclusions", F10_3_CODES),
        ):
            bad = [x for x in r[key] if x not in allowed]
            if bad:
                raise AnalysisRefused(f"{i}: unknown {key} reason code(s) {bad}")
        codes = bool(r["pre_audit_exclusions"] or r["f10_3_exclusions"])
        if codes != (r["f10_3_verdict"] == "excluded"):
            raise AnalysisRefused(
                f"{i}: re-stage verdict {r['f10_3_verdict']!r} contradicts its reason codes"
            )
        m_verdict, m_pre, m_f10 = verdict
        if (
            r["f10_3_verdict"] != m_verdict
            or sorted(set(r["pre_audit_exclusions"])) != m_pre
            or sorted(set(r["f10_3_exclusions"])) != m_f10
        ):
            raise AnalysisRefused(
                f"{i}: ledger re-stage entry ({r['f10_3_verdict']}, "
                f"{r['pre_audit_exclusions'] + r['f10_3_exclusions']}) contradicts the execution "
                f"manifest ({m_verdict}, {m_pre + m_f10})"
            )
        lr = r.get("pooled_rho")
        if _finite(lr) and rhos[i] is not None and not math.isclose(lr, rhos[i], rel_tol=1e-12):
            raise AnalysisRefused(
                f"{i}: ledger pooled_rho {lr!r} contradicts the execution manifest's pooled ρ "
                f"{rhos[i]!r}"
            )
        if m_pre:
            out[i] = {"status": "pre_audit_excluded", "reasons": m_pre}
        elif m_f10:
            out[i] = {"status": "f10_3_excluded", "reasons": m_f10}
        else:
            out[i] = {"status": "pass", "reasons": []}
        out[i]["registered_realised_verdict"] = r["registered_realised_verdict"]
        out[i]["variant_json_sha256"] = r["variant_json_sha256"]
    return out


def validate_ledger(ledger, manifest, text_groups, manifest_sha256) -> dict:
    """Fail-closed ledger checks; returns the per-pair re-stage audit state.

    ``manifest`` is the execution manifest and ``manifest_sha256`` its verified file hash; the
    ledger must record that hash (``manifest_sha256`` and, when present and well formed,
    ``restage_provenance.manifest_sha256``).
    """
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise AnalysisRefused(f"ledger schema must be {LEDGER_SCHEMA!r}")
    if ledger.get("sealed") is not True:
        raise AnalysisRefused(
            "ledger is not sealed (A5.3: one analysis after the ledger is closed)"
        )
    if not _is_sha(manifest_sha256):
        raise AnalysisRefused("the execution manifest sha256 is required (64 lowercase hex)")
    if ledger.get("manifest_sha256") != manifest_sha256:
        raise AnalysisRefused(
            f"ledger manifest_sha256 {ledger.get('manifest_sha256')} does not match the execution "
            f"manifest {manifest_sha256}"
        )
    prov = ledger.get("restage_provenance")
    prov_manifest = prov.get("manifest_sha256") if isinstance(prov, dict) else None
    if _is_sha(prov_manifest) and prov_manifest != manifest_sha256:
        raise AnalysisRefused(
            f"ledger restage_provenance.manifest_sha256 {prov_manifest} does not match the "
            f"execution manifest {manifest_sha256}"
        )
    validate_windows(ledger, text_groups)
    expected = _variant_ids(manifest)
    for i in _selected(manifest):
        entry = ledger.get("variants", {}).get(i)
        if entry is None:
            raise AnalysisRefused(f"ledger lacks the variant entries of selected edit {i}")
        for c in CONTRASTS:
            if expected[(i, c)] is None:
                raise AnalysisRefused(f"execution manifest lacks the {c} variant of {i}")
            v = entry.get(c)
            if v is None:
                raise AnalysisRefused(f"ledger lacks the {c} variant of {i}")
            if v.get("variant_id") != expected[(i, c)]:
                raise AnalysisRefused(
                    f"{i} {c}: variant id {v.get('variant_id')} != manifest {expected[(i, c)]}"
                )
    m_excluded = {v.get("variant_id"): bool(v.get("excluded")) for v in manifest["variants"]}
    for i in _selected(manifest):
        for c in ("ladder_0.25", "ladder_0.5"):
            if bool(ledger["variants"][i][c].get("excluded")) != m_excluded[expected[(i, c)]]:
                raise AnalysisRefused(
                    f"{i} {c}: ledger exclusion flag contradicts the execution manifest"
                )
    audit = restage_audit(ledger, manifest)
    for i, a in audit.items():
        if ledger["variants"][i]["rotation"].get("excluded") and a["status"] == "pass":
            raise AnalysisRefused(
                f"{i}: rotation variant marked excluded but the re-stage audit passes"
            )
    return audit


def _excluded_contrasts(ledger, audit, i) -> dict:
    out = {"rotation": audit[i]["status"] != "pass"}
    for c in ("ladder_0.25", "ladder_0.5"):
        out[c] = bool(ledger["variants"][i][c].get("excluded"))
    return out


def validate_fields(fields, ledger, manifest, audit) -> dict:
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
        excluded = _excluded_contrasts(ledger, audit, i)
        for c, p in zip(CONTRASTS, row):
            if p is None and excluded[c]:
                continue
            if not _finite(p):
                raise AnalysisRefused(f"no finite frozen prediction for variant {i} {c}")
        out[i] = row
    return out


# ---------------------------------------------------------------------------- registered contrast


def resolve(ledger, manifest, preds, audit=None, draws=RESOLUTION_DRAWS) -> tuple[dict, dict]:
    """Per pair and contrast: status, reasons/cause, d̂, paired block-bootstrap interval, agreement."""
    sel = _selected(manifest)
    audit = audit if audit is not None else restage_audit(ledger, manifest)
    n_w = WINDOWS
    refs = ledger.get("references", {})
    records: dict = {i: {} for i in sorted(sel)}
    columns, keys = [], []
    for i in sorted(sel):
        original = _arm(refs.get(i, {}).get("mean_nll"), n_w)
        predecessor = _arm(refs.get(sel[i]["predecessor_id"], {}).get("mean_nll"), n_w)
        for k, c in enumerate(CONTRASTS):
            v = ledger["variants"][i][c]
            rec = {"variant_id": v["variant_id"], "prediction": preds[i][k], "status": None}
            if c != "rotation":
                rec["realised_increment_ladder"] = i in REALISED_INCREMENT_LADDER_IDS
            if c == "rotation":
                a = audit[i]
                if a["status"] != "pass":
                    rec.update(status=a["status"], reasons=list(a["reasons"]))
                    rec["detail"] = f"{a['status']}: {', '.join(a['reasons'])}" + (
                        f" ({'; '.join(a['detail'])})" if a.get("detail") else ""
                    )
                if "registered_realised_verdict" in a:
                    rec["registered_realised_verdict"] = a["registered_realised_verdict"]
                arms = {"original": original, "rotation": _arm(v.get("mean_nll"), n_w)}
            else:
                if v.get("excluded"):
                    rec.update(status="excluded", reasons=list(v.get("exclusions") or ["excluded"]))
                    rec["detail"] = f"excluded: {', '.join(rec['reasons'])}"
                arms = {"ladder": _arm(v.get("mean_nll"), n_w), "predecessor": predecessor}
            if rec["status"] is None:
                missing = [a for a, x in arms.items() if x is None]
                if missing:
                    rec.update(status="missing_score", reasons=missing)
                    rec["detail"] = f"missing or nonfinite score: {', '.join(missing)}"
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
                rec["detail"] = (
                    f"resolved [{lo:.6g}, {hi:.6g}]; "
                    + ("agrees" if rec["agree"] else "disagrees")
                    + (" (zero prediction)" if _sign(p) == 0 else "")
                )
            else:
                rec["cause"] = (
                    f"paired block-bootstrap 95 % interval [{lo:.6g}, {hi:.6g}] does not "
                    "strictly exclude zero"
                )
                rec["detail"] = f"unresolved: {rec['cause']}"
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
    out["unresolved"]["causes"] = {
        i: records[i][contrast]["cause"] for i in out["unresolved"]["ids"]
    }
    return out


SIGNS = ("negative", "zero", "positive")


def _sign_table(resolved_recs) -> dict:
    """3 × 2 table (F10b.3): predicted sign {−, 0, +} × resolved observed sign {+, −}."""
    table = {
        s: {"n": 0, "agree": 0, "disagree": 0, "observed_positive": 0, "observed_negative": 0}
        for s in SIGNS
    }
    for r in resolved_recs:
        row = table[SIGNS[_sign(r["prediction"]) + 1]]
        row["n"] += 1
        row["agree" if r["agree"] else "disagree"] += 1
        if r["observed_sign"] > 0:
            row["observed_positive"] += 1
        elif r["observed_sign"] < 0:
            row["observed_negative"] += 1
    return table


def agreement_row(records, ids, contrast="rotation") -> dict:
    """Field agreement, comparator, 3 × 2 sign table, ledger and id accounting for one set."""
    ids = sorted(ids)
    res_ids = [i for i in ids if records[i][contrast]["status"] == "resolved"]
    res = [records[i][contrast] for i in res_ids]
    n = len(res)
    k = sum(r["agree"] for r in res)
    zero_ids = [i for i in res_ids if _sign(records[i][contrast]["prediction"]) == 0]
    field = proportion(k, n)
    table = _sign_table(res)
    row = {
        "ids": ids,
        "n_selected": len(ids),
        "ledger": _ledger(records, contrast, ids),
        "pair_status": [
            {
                "id": i,
                "status": records[i][contrast]["status"],
                "detail": records[i][contrast].get("detail"),
            }
            for i in ids
        ],
        "resolved": n,
        "resolved_ids": res_ids,
        "agreements": k,
        "agreement_ids": [i for i in res_ids if records[i][contrast]["agree"]],
        "disagreements": n - k,
        "disagreement_ids": [i for i in res_ids if not records[i][contrast]["agree"]],
        "zero_prediction_disagreements": len(zero_ids),
        "zero_prediction_ids": zero_ids,
        "field": field,
        "cp95": field["cp95"],
        "sign_table": table,
        "predicted_sign_counts": {s: table[s]["n"] for s in SIGNS},
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


def validity_bindings(spec_sha, fields_sha, provenance, masked_sha=None) -> dict:
    """Pins each external step-1 predicate's evidence must link (Board A conditions 3 and 5)."""
    prov = provenance if isinstance(provenance, dict) else {}
    return {
        "registration_pinned": [
            spec_sha,
            FROZEN_STAGED_MANIFEST_SHA256,
            TEXT_GROUPS_SHA256,
            fields_sha,
            NULLSIZE_REPORT_SHA256,
        ],
        "selection_and_audits": [prov.get(k) for k in RESTAGE_PROVENANCE_KEYS],
        "null_size": [NULLSIZE_REPORT_SHA256] + ([masked_sha] if masked_sha else []),
    }


def _evidence_hashes(ev):
    if not isinstance(ev, list) or not ev:
        return None, "no evidence linked"
    for item in ev:
        if not (
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and item["path"]
            and _is_sha(item.get("sha256"))
        ):
            return None, "evidence items must be {path, sha256 (64 hex)}"
    return [item["sha256"] for item in ev], None


def _validity(validity, internal=None, bindings=None) -> dict:
    """Step 1 (A5.5.1): every predicate passes *and* links its evidence; missing ≠ certified."""
    internal = internal or {}
    bindings = bindings or {}
    problems: dict = {}
    for k, v in internal.items():
        if v.get("pass") is not True:
            problems[k] = v.get("reason", "failed")
    g2_hard_stop = False
    if not isinstance(validity, dict):
        problems["external_predicates"] = (
            "external validity predicates not supplied (A5.5.1 step-1 checklist)"
        )
    else:
        for k in VALIDITY_PREDICATES:
            v = validity.get(k)
            if not isinstance(v, dict) or "pass" not in v:
                problems[k] = "predicate not supplied"
                continue
            if v["pass"] is not True:
                problems[k] = "failed"
                if k == "g2_predecessor_audit" and v["pass"] is False:
                    g2_hard_stop = True
                    problems[k] = G2_HARD_STOP
                continue
            hashes, prob = _evidence_hashes(v.get("evidence"))
            if prob:
                problems[k] = f"evidence: {prob}"
                continue
            missing = [h for h in bindings.get(k, ()) if h not in hashes]
            if missing:
                problems[k] = f"evidence does not link the required pin(s) {missing}"
    out = {
        "pass": not problems,
        "reason": (
            "all step-1 predicates pass with linked evidence"
            if not problems
            else f"step-1 predicates not certified: {sorted(problems)}"
        ),
        "problems": problems,
        "predicates": validity,
        "internal": internal,
        "required_bindings": bindings,
        "g2_hard_stop": g2_hard_stop,
    }
    return out


def _interval_ok(iv) -> bool:
    return (
        isinstance(iv, list | tuple)
        and len(iv) == 2
        and all(_finite(x) for x in iv)
        and iv[0] <= iv[1]
    )


def check_part_b(part_b) -> dict:
    """A5.4.2 / A5.5.1 checklist rows 8–9 on the supplied Part B family (step 1, never step 4)."""
    if part_b is None:
        return {
            "status": "not supplied",
            "valid": False,
            "problems": [
                (
                    "Part B results not supplied: the A5.4.2 inference diagnostics and destruction "
                    "eligibility cannot be certified (missing Part B is not evidence of validity or "
                    "eligibility)"
                )
            ],
        }
    if not isinstance(part_b, dict):
        return {
            "status": "malformed",
            "valid": False,
            "problems": ["Part B input is not an object"],
        }
    problems = []
    points = part_b.get("points")
    intervals = part_b.get("intervals")
    valid = part_b.get("valid")
    diag = part_b.get("diagnostics")
    for name in PART_B_CONTRASTS:
        if not isinstance(valid, dict) or valid.get(name) is not True:
            problems.append(f"{name} invalid or undefined (A5.4.2)")
        if not isinstance(points, dict) or not _finite(points.get(name)):
            problems.append(f"{name} point missing or nonfinite")
        for proc in PROCEDURES:
            iv = (intervals or {}).get(name, {}) if isinstance(intervals, dict) else {}
            iv = iv.get(proc) if isinstance(iv, dict) else None
            if not _interval_ok(iv):
                problems.append(
                    f"{name} {proc} interval missing, nonfinite, reversed or malformed: {iv!r}"
                )
            d = (diag or {}).get(name, {}) if isinstance(diag, dict) else {}
            d = d.get(proc) if isinstance(d, dict) else None
            if not isinstance(d, dict):
                problems.append(f"{name} {proc} inference diagnostics missing")
                continue
            if d.get("draws") != PART_B_DRAWS:
                problems.append(f"{name} {proc}: draws {d.get('draws')!r} != {PART_B_DRAWS}")
            if d.get("failed_draws") != 0:
                problems.append(
                    f"{name} {proc}: failed draws {d.get('failed_draws')!r} (all 4,000 must be valid)"
                )
            mdc = d.get("min_distinct_components")
            if not (isinstance(mdc, int) and not isinstance(mdc, bool) and mdc >= 2):
                problems.append(f"{name} {proc}: fewer than two distinct components ({mdc!r})")
    if part_b.get("destruction_controls_eligible") is not True:
        problems.append("destruction controls not eligible or not certified (A5.8.3)")
    _, ev_prob = _evidence_hashes(part_b.get("evidence"))
    if ev_prob:
        problems.append(f"Part B evidence: {ev_prob}")
    power = part_b.get("power_at_0_15")
    if power is not None:
        if not isinstance(power, dict):
            problems.append("power_at_0_15 malformed")
        else:
            for proc, val in power.items():
                if proc not in PROCEDURES or not (_finite(val) and 0.0 <= val <= 1.0):
                    problems.append(f"power_at_0_15 {proc!r} malformed: {val!r}")
    return {
        "status": "valid" if not problems else "invalid",
        "valid": not problems,
        "problems": problems,
    }


def part_b_report(part_b, check, sha256=None) -> dict:
    """Part B is always reported (A5.5.1), on every branch, as supplied; never computed here."""
    base = {
        "computed_by_this_tool": False,
        "validity": check,
        "rule": (
            "Part B is always computed and reported (A5.5.1 step 4) by its own pipeline. This tool "
            "does not fit Part B; it retains the supplied results verbatim on every branch, "
            "including when funding is unresolved or validity fails."
        ),
    }
    if part_b is None:
        return base | {
            "status": "not supplied to this tool",
            "supplied": None,
            "note": (
                "The mandatory Part B report must be linked; missing Part B is not evidence of "
                "eligibility."
            ),
        }
    return base | {"status": "supplied", "sha256": sha256, "supplied": part_b}


def _part_b_funding(part_b) -> tuple[str, str]:
    """Steps 4–5 on a step-1-valid Part B family; returns (eligibility, funding)."""
    iv = part_b["intervals"]
    for name in PART_B_CONTROLS:
        for proc in PROCEDURES:
            if not iv[name][proc][0] > 0:
                why = f"not eligible: {name} {proc} lower bound not > 0"
                return why, f"unresolved: {why} (A5.5 step 4)"
    c = part_b["points"]["C"]
    lows = [iv["C"][p][0] for p in PROCEDURES]
    ups = [iv["C"][p][1] for p in PROCEDURES]
    eligible = "eligible: controls eligible and C_abl, C_rand, C_perm beaten in both procedures"
    if c >= 0.15 and all(lo > 0 for lo in lows):
        return eligible, "go"
    if c >= 0.05:
        return eligible, "screen"
    power = part_b.get("power_at_0_15") or {}
    if all(u < 0.05 for u in ups) and all(
        _finite(power.get(p)) and power[p] >= 0.80 for p in PROCEDURES
    ):
        return eligible, "no-go"
    return eligible, "unresolved"


def decision(
    validity,
    rotation,
    ladder_025,
    part_b=None,
    internal=None,
    bindings=None,
    part_b_sha256=None,
) -> dict:
    """A5.5.1 ordered if/else, evaluated once, validity first.

    ``rotation``/``ladder_025``: dicts with resolved, agreements, cp95. Step 1 is decided before
    either is read; an unresolved step 1 leaves every later step unresolved. The supplied Part B
    results are reported on every branch.
    """
    v = _validity(validity, internal, bindings)
    pb = check_part_b(part_b)
    v["part_b"] = pb
    if not pb["valid"]:
        v["pass"] = False
        v["problems"]["part_b_inference_and_controls"] = pb["problems"]
        v["reason"] = f"step-1 predicates not certified: {sorted(v['problems'])}"
    out = {
        "authority": AUTHORITY_REGISTERED,
        "validity": v,
        "part_b_reporting": part_b_report(part_b, pb, part_b_sha256),
    }
    if not v["pass"]:
        consequence = "nothing downstream runs (A5.5.1 step 1)"
        if v["g2_hard_stop"]:
            consequence = G2_HARD_STOP + " " + consequence
        return out | {
            "step_1_validity": "unresolved",
            "gate": (
                "unresolved (validity: G2 hard stop pending a new amendment)"
                if v["g2_hard_stop"]
                else "unresolved (validity)"
            ),
            "step_2_ladder_alternative": "unresolved (validity)",
            "part_a": "unresolved (validity)",
            "step_4_part_b_eligibility": "unresolved (validity)",
            "funding": "unresolved (validity)",
            "scientific": consequence,
        }
    out["step_1_validity"] = "pass"
    lad_n, lad_cp = ladder_025["resolved"], ladder_025["cp95"]
    alternative = bool(lad_n >= FLOOR and lad_cp is not None and lad_cp[1] <= 0.6)
    out["step_2_ladder_alternative"] = alternative
    if alternative:
        return out | {
            "gate": "alternative",
            "part_a": "not reached (alternative overrides rotation go)",
            "step_4_part_b_eligibility": "not reached (alternative)",
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
        out["step_4_part_b_eligibility"] = "not eligible: Part A did not go"
        out["funding"] = "unresolved: Part A did not go (A5.5 step 4)"
    else:
        out["step_4_part_b_eligibility"], out["funding"] = _part_b_funding(part_b)
    return out


def masked_calibration_state(masked, masked_sha, audit) -> dict:
    """F10.6 disclosure: full yield → registered simulations stand; else a hashed masked run."""
    excluded = sorted(i for i, a in audit.items() if a["status"] != "pass")
    n = len(audit)
    base = {
        "restage_yield": {"passing": n - len(excluded), "selected": n},
        "excluded_ids": excluded,
    }
    if not excluded:
        if masked is not None:
            raise AnalysisRefused(
                f"F10.6: all {n} pairs pass the re-stage audit, so no slot-masked run applies; a "
                "masked calibration report was supplied anyway"
            )
        return base | {
            "required": False,
            "status": f"not required ({n}/{n})",
            "passed": True,
            "statement": F10_6_ALL_PASS,
        }
    if masked is None:
        raise AnalysisRefused(
            f"F10.6: {len(excluded)} rotation pair(s) excluded by the re-stage audit; the "
            "slot-masked POWER-AND-YIELD calibration report is required before scoring"
        )
    if not _is_sha(masked_sha):
        raise AnalysisRefused("slot-masked calibration report needs its file sha256")
    sm = masked.get("slot_mask") if isinstance(masked, dict) else None
    if not isinstance(sm, dict) or sm.get("label") != SLOT_MASK_LABEL:
        raise AnalysisRefused("masked calibration report lacks the 'slot-masked (F10.6)' block")
    if sm.get("schema") != SLOT_MASK_SCHEMA or not _is_sha(sm.get("sha256")):
        raise AnalysisRefused("masked calibration report lacks its slot-mask schema/file sha256")
    if masked.get("seed") != SEED:
        raise AnalysisRefused("masked calibration report is not on the registered seed")
    canonical = {
        i: (["rotation"] if audit[i]["status"] == "pass" else []) + ["ladder_0.25", "ladder_0.5"]
        for i in sorted(audit)
    }
    mapping_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    masked_ids = sm.get("masked_ids") or {}
    if sorted(masked_ids.get("rotation") or []) != excluded or any(
        masked_ids.get(c) for c in ("ladder_0.25", "ladder_0.5")
    ):
        raise AnalysisRefused(
            "masked calibration masked ids differ from the re-stage rotation exclusions"
        )
    if sm.get("mapping_sha256") != mapping_sha:
        raise AnalysisRefused(
            f"masked calibration mapping sha256 {sm.get('mapping_sha256')} != {mapping_sha} "
            "recomputed from the re-stage exclusions"
        )
    gates = sm.get("gates")
    if not (
        isinstance(gates, dict)
        and isinstance(gates.get("size"), dict)
        and isinstance(gates.get("power"), dict)
        and isinstance(gates.get("passed"), bool)
        and gates["passed"]
        == (gates["size"].get("passed") is True and gates["power"].get("passed") is True)
    ):
        raise AnalysisRefused("masked calibration gates missing or inconsistent")
    null = (masked.get("reports") or {}).get("null_size")
    reps = masked.get("replicates")
    if not isinstance(null, dict) or not isinstance(reps, int) or reps <= 0:
        raise AnalysisRefused("masked calibration report lacks its embedded NULL-SIZE run")
    dec = null["decisions"]
    row = {
        "scenario": "slot_masked_null_size",
        "label": SLOT_MASK_LABEL + " — embedded NULL-SIZE pass of the masked run; replaces no row",
        "go_count": dec["go"],
        "ordered_go_rate": null["ordered_go_rate"],
        "ordered_go_cp95": null["ordered_go_cp95"],
        "alternative_count": dec["alternative"],
        "alternative_rate": dec["alternative"] / reps,
        "alternative_cp95": list(cp(dec["alternative"], reps)),
        "unresolved_count": dec["unresolved"],
        "replicates": reps,
    }
    passed = gates["passed"]
    return base | {
        "required": True,
        "status": "supplied",
        "sha256": masked_sha,
        "slot_mask_file_sha256": sm["sha256"],
        "mapping_sha256": mapping_sha,
        "masked_ids": masked_ids,
        "gates": gates,
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "null_size_row": row,
        "statement": (
            f"{len(excluded)} of {n} rotation pairs are excluded by the re-stage audit; the "
            f"separately labelled slot-masked calibration (sha256 {masked_sha}) "
            + ("passed its gates (size ≤ 0.07, ordered power ≥ 0.80)." if passed else "FAILED. ")
            + ("" if passed else F10_6_FAILED)
        ),
    }


def f57_table(nullsize, masked_row=None) -> dict:
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
    if masked_row is not None:
        rows.append(masked_row)  # spec §6.4: added as a separately labelled row; replaces none
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


def standardisation_params(X_train, names) -> dict:
    """Train-only mean, population SD and degenerate columns (spec §1.3)."""
    X_train = np.asarray(X_train, dtype=float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    degenerate = [n for n, m, s in zip(names, mu, sd) if s <= 1e-12 * max(1.0, abs(m))]
    return {
        "mean": dict(zip(names, map(float, mu))),
        "sd": dict(zip(names, map(float, sd))),
        "degenerate_columns": degenerate,
        "n_train": int(X_train.shape[0]),
    }


def standardise(X_train, X_apply, names):
    """Train-only z-scores (population SD); degenerate columns set to 0 for the fit (spec §1.3)."""
    X_train = np.asarray(X_train, dtype=float)
    params = standardisation_params(X_train, names)
    mu = np.array([params["mean"][n] for n in names])
    sd = np.array([params["sd"][n] for n in names])
    degenerate = params["degenerate_columns"]
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


def fit_ridge_logistic(Z, a, lam=LAMBDA, tol=NEWTON_TOL, max_iter=None):
    """Ridge logistic, λ on Σβ² (intercept unpenalised), Newton–Raphson with step halving.

    Starts at β = 0, β₀ = logit(mean a). A step is halved while the objective does not decrease;
    a step whose objective is flat within floating-point resolution (|Δf| ≤ 8·eps·|f|) is
    accepted only if it reduces max |gradient| (solver tie-break: choice pending adopting note).
    Converged when max |gradient| ≤ 1e-10; at most ``NEWTON_MAX_ITER`` (100) iterations.
    Returns (β₀, β, info) with info = {converged, iterations, gradient_max_abs}; a failure raises
    :class:`FitFailure` carrying the same diagnostics.
    """
    max_iter = NEWTON_MAX_ITER if max_iter is None else max_iter
    Z = np.asarray(Z, dtype=float)
    a = np.asarray(a, dtype=float)
    m = a.mean()
    if m <= 0.0 or m >= 1.0:
        raise FitFailure("one class in the training set", {"iterations": 0})
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
    gmax = float(np.max(np.abs(g)))
    for it in range(max_iter + 1):
        gmax = float(np.max(np.abs(g)))
        if gmax <= tol:
            info = {"converged": True, "iterations": it, "gradient_max_abs": gmax}
            return float(beta[0]), beta[1:].copy(), info
        if it == max_iter:
            break
        pr = _expit(X @ beta)
        H = X.T @ (X * (pr * (1 - pr))[:, None]) + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError as e:
            raise FitFailure(
                f"singular Newton system: {e}", {"iterations": it, "gradient_max_abs": gmax}
            ) from e
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
            raise FitFailure(
                "step halving failed to decrease the objective",
                {"iterations": it, "gradient_max_abs": gmax},
            )
        beta, f = nb, nf
        g = grad(beta)
    raise FitFailure(
        f"no convergence in {max_iter} iterations",
        {"iterations": max_iter, "gradient_max_abs": gmax},
    )


def bootstrap_indices(n: int, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    """Component bootstrap index matrix (spec §6.3): PCG64(20261107), drawn once, shared by M0/M1."""
    return np.random.Generator(np.random.PCG64(SEED)).integers(0, n, size=(draws, n))


def _fit(X, a, names):
    """Standardise on the fitting rows and fit; returns (β₀, β, diagnostics)."""
    params = standardisation_params(X, names)
    Z, _, _ = standardise(X, None, names)
    try:
        b0, b, info = fit_ridge_logistic(Z, a)
    except FitFailure as e:
        e.info["standardisation"] = params
        raise
    return b0, b, info | {"standardisation": params}


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


def _loo(ids, X, a, specs):
    """Leave-one-pair-out diagnostics with per-fold fit and standardisation records."""
    n = len(a)
    models = ("intercept_only", "M0", "M1")
    out = {m: {"log_loss": [], "brier": [], "failed_folds": 0, "folds": []} for m in models}
    for j in range(n):
        tr = np.arange(n) != j
        a_tr = a[tr]
        y = float(a[j])
        one_class = a_tr.min() == a_tr.max()
        for m in models:
            fold = {"fold": j, "held_out": ids[j], "y": y, "n_train": int(tr.sum())}
            if one_class:
                fold["status"] = "failed: one class in the training fold"
                out[m]["failed_folds"] += 1
                out[m]["folds"].append(fold)
                continue
            if m == "intercept_only":
                p = float(a_tr.mean())
            else:
                cols = specs[m]
                Xc = X[:, : len(cols)]
                fold["standardisation"] = standardisation_params(Xc[tr], cols)
                Zt, Zh, _ = standardise(Xc[tr], Xc[~tr], list(cols))
                fold["z_held_out"] = dict(zip(cols, map(float, Zh[0])))
                try:
                    b0, b, info = fit_ridge_logistic(Zt, a_tr)
                except FitFailure as e:
                    fold |= {"status": f"failed: {e}"} | e.info
                    out[m]["failed_folds"] += 1
                    out[m]["folds"].append(fold)
                    continue
                fold |= info
                p = float(_expit(b0 + Zh[0] @ b))
            fold |= {"status": "fitted", "p_held_out": p}
            out[m]["folds"].append(fold)
            p_c = min(max(p, 1e-300), 1 - 1e-16)
            out[m]["log_loss"].append(-(y * math.log(p_c) + (1 - y) * math.log1p(-p_c)))
            out[m]["brier"].append((p - y) ** 2)
    return {
        m: {
            "log_loss": float(np.mean(v["log_loss"])) if v["log_loss"] else None,
            "brier": float(np.mean(v["brier"])) if v["brier"] else None,
            "failed_folds": v["failed_folds"],
            "folds": n,
            "fold_records": v["folds"],
            "degenerate_column_events": {
                c: sum(
                    c in f.get("standardisation", {}).get("degenerate_columns", [])
                    for f in v["folds"]
                )
                for c in (specs.get(m) or ())
            },
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
            b0, b, info = _fit(X[:, : len(cols)], a, cols)
        except FitFailure as e:
            out[m] = {"status": f"not estimable: full fit failed ({e})", "fit": e.info}
            continue
        out[m] = {
            "status": "estimated",
            "columns": cols,
            "intercept": b0,
            "coefficients": dict(zip(cols, map(float, b))),
            "theta_adj": {"point": float(_expit(b0))},
            "degenerate_columns": info["standardisation"]["degenerate_columns"],
            "iterations": info["iterations"],
            "fit": info,
        }
    idx = bootstrap_indices(n, draws)
    boot = {
        m: {
            "theta": [],
            "beta": {c: [] for c in specs[m]},
            "failed": 0,
            "reasons": {},
            "degenerate": dict.fromkeys(specs[m], 0),
            "fits_with_degenerate": 0,
            "iterations": [],
            "grad": [],
        }
        for m in specs
    }
    delta, delta_failed = [], 0
    for row in idx:
        Xb, ab = X[row], a[row]
        theta = {}
        for m, cols in specs.items():
            bm = boot[m]
            if ab.min() == ab.max():
                bm["failed"] += 1
                bm["reasons"]["one class in the draw"] = (
                    bm["reasons"].get("one class in the draw", 0) + 1
                )
                continue
            try:
                b0, b, info = _fit(Xb[:, : len(cols)], ab, cols)
            except FitFailure as e:
                bm["failed"] += 1
                bm["reasons"][str(e)] = bm["reasons"].get(str(e), 0) + 1
                std = e.info.get("standardisation", {})
            else:
                std = info["standardisation"]
                bm["iterations"].append(info["iterations"])
                bm["grad"].append(info["gradient_max_abs"])
                theta[m] = float(_expit(b0))
                bm["theta"].append(theta[m])
                for c, v in zip(cols, b):
                    bm["beta"][c].append(float(v))
            if std.get("degenerate_columns"):
                bm["fits_with_degenerate"] += 1
                for c in std["degenerate_columns"]:
                    bm["degenerate"][c] += 1
        if len(theta) == 2:
            delta.append(theta["M1"] - theta["M0"])
        else:
            delta_failed += 1
    for m, cols in specs.items():
        bm = boot[m]
        diag = {
            "draws": draws,
            "failed_draws": bm["failed"],
            "failure_reasons": bm["reasons"],
            "degenerate_column_events": bm["degenerate"],
            "fits_with_degenerate_columns": bm["fits_with_degenerate"],
            "iterations_max": max(bm["iterations"]) if bm["iterations"] else None,
            "iterations_mean": float(np.mean(bm["iterations"])) if bm["iterations"] else None,
            "gradient_max_abs_max": max(bm["grad"]) if bm["grad"] else None,
        }
        out[m]["bootstrap_diagnostics"] = diag
        if out[m]["status"] != "estimated":
            continue
        out[m]["failed_draws"] = bm["failed"]
        out[m]["draws"] = draws
        out[m]["theta_adj"] |= _interval(bm["theta"], draws, bm["failed"])
        out[m]["coefficient_intervals"] = {
            c: _interval(bm["beta"][c], draws, bm["failed"]) for c in cols
        }
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
    out["loo"] = _loo(ids, X, a, {"intercept_only": None} | specs)
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


def _covariates(i, sel, restage, rho) -> dict:
    """Covariate row of pair ``i``: selection metadata, ledger moments, execution-manifest ρ."""
    s = sel[i]
    norm = s.get("effective_norm")
    row = {
        "chain": 1.0 if s["family"] == "chain" else 0.0 if s["family"] == "lora_sft" else None,
        "log1p_effective_norm": math.log1p(norm) if _finite(norm) and norm > -1 else None,
    }
    row |= moment_differences((restage or {}).get("tensors"))
    row["ln_rho"] = math.log(rho) if _finite(rho) and rho > 0 else None
    return row


def staged_ln_rho(frozen) -> dict:
    """Spec §1.1: staged ln ρ = 2·ln(1 + pooled_norm_mismatch) (all staged signed values positive).

    ``frozen`` is the preserved a533 staging manifest; never a regenerated manifest.
    """
    return {
        i: 2 * math.log1p(frozen["edit_audits"][i]["edit_audit"]["pooled_norm_mismatch"])
        for i in _selected(frozen)
    }


def covariate_report(records, ledger, manifest, frozen, draws) -> dict:
    """Spec §1 on the forty registered pairs (supplementary edits excluded).

    Selection metadata from the frozen rows (identical in the execution manifest, verified), the
    logged moments from the ledger's re-stage entries, ρ from the execution manifest.
    """
    sel = _selected(frozen)
    rhos = restage_rho(manifest)
    resolved = [i for i in sorted(sel) if records[i]["rotation"]["status"] == "resolved"]
    rows, dropped, X, a = [], [], [], []
    for i in resolved:
        cov = _covariates(i, sel, ledger.get("restage", {}).get(i), rhos.get(i))
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
    staged = staged_ln_rho(frozen)
    diffs = [abs(math.log(rhos[i]) - staged[i]) for i in staged if rhos.get(i) is not None]
    full = agreement_row(records, sorted(sel))
    return {
        "authority": AUTHORITY_NONE,
        "spec": "§1 (F10.5(a))",
        "columns": list(COVARIATE_COLUMNS),
        "column_notes": {
            "chain": "1 = chain, 0 = lora_sft (frozen manifest selections[].family)",
            "log1p_effective_norm": "ln(1 + frozen manifest selections[].effective_norm)",
            "d_*": "element-count-weighted mean over edited tensors of logged_r − logged_d (re-stage variant.json)",
            "ln_rho": "ln of the re-stage F10.4 pooled ρ = Σ‖R_eff‖²/Σ‖D‖² (execution manifest edit_audits[].edit_audit.pooled_energy_ratio)",
        },
        "rules": {
            "penalty": "λ = 1 on Σβ², intercept unpenalised (sklearn LogisticRegression(C=1.0))",
            "solver": "Newton–Raphson from β = 0, β₀ = logit(mean a); step halving; max |grad| ≤ 1e-10; ≤ 100 iterations",
            "standardisation": "train-only mean and population SD per fit (full, LOO fold, bootstrap draw); degenerate column → 0",
            "folds": "leave-one-pair-out over the complete-case rows in canonical id order (diagnostics only)",
            "bootstrap": f"component bootstrap, {draws} draws, numpy.random.Generator(numpy.random.PCG64({SEED})), common draws for M0 and M1, linear .025/.975 percentiles",
            "floor": "fitted only if n_cc ≥ 15 and both agreement classes present",
            "missing": "a resolved pair with any missing or nonfinite covariate is dropped from both models; no imputation",
        },
        "uncertainty": (
            "θ_raw and the comparator: exact CP 95 %; θ_adj, Δθ and β̂: component-bootstrap "
            "percentile 95 % (labelled diagnostic if any draw failed); LOO: point values with "
            "failed-fold counts (spec §6.3)."
        ),
        "units": {
            "agreement, θ_adj, Δθ": "probability scale",
            "coefficients": "log-odds per training-SD of the column",
            "ln_rho": "natural log, dimensionless",
            "d̂, predictions": UNITS,
        },
        "population": "the forty registered pairs (supplementary edits excluded)",
        "selected_ids": sorted(sel),
        "resolved_ids": resolved,
        "complete_case_ids": rows,
        "n_cc": len(rows),
        "dropped_rows": dropped,
        "complete_case_agreement": cc,
        "theta_raw": cc["field"],
        "comparator": cc["comparator"],
        "sign_table": cc["sign_table"],
        "predicted_sign_counts": cc["predicted_sign_counts"],
        "agreement_ids": cc["agreement_ids"],
        "disagreement_ids": cc["disagreement_ids"],
        "zero_prediction_disagreements": cc["zero_prediction_disagreements"],
        "zero_prediction_ids": cc["zero_prediction_ids"],
        "negative_prediction_share": cc["negative_prediction_share"],
        "full_analysis_set_agreement": full["field"],
        "full_analysis_set_ledger": full["ledger"],
        "models": models,
        "max_abs_lnrho_restage_minus_staged": max(diffs) if diffs else None,
        "collinearity_disclosure": COLLINEARITY_DISCLOSURE,
        "sentence": COVARIATE_SENTENCE,
    }


# ---------------------------------------------------------------------------- tertiles and subsets


def norm_tertiles(frozen) -> dict:
    """Spec §2.1 from the frozen a533 manifest's selection cells, verified by sorted-id sha256."""
    sel = _selected(frozen)
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


def lnrho_tertiles(frozen):
    """Spec §2.2: recompute from staged (a533) values; fail closed unless equal to frozen lists."""
    x = staged_ln_rho(frozen)
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


def tertile_report(records, manifest, frozen) -> dict:
    """Spec §2: frozen cells from the a533 manifest; re-stage ln ρ from the execution manifest."""

    def label(r):
        return (
            "below floor (descriptive)"
            if r["resolved"] < FLOOR
            else "at or above floor (descriptive)"
        )

    norm_rows = []
    for t, cell in norm_tertiles(frozen).items():
        for group in ("pooled", "lora_sft", "chain"):
            r = _descriptive_row(records, cell[group], f"{TERTILE_NAMES[t]} ({t}) {group}", None)
            r["label"] = label(r)
            norm_rows.append(r)
    cells, cuts = lnrho_tertiles(frozen)
    staged = staged_ln_rho(frozen)
    rhos = restage_rho(manifest)
    lnrho_rows = []
    for k, ids in enumerate(cells):
        r = _descriptive_row(records, ids, LNRHO_CELL_NAMES[k], None)
        r["label"] = label(r)
        values = {}
        drift = []
        for i in ids:
            rho = rhos.get(i)
            v = math.log(rho) if rho is not None else None
            values[i] = v
            if v is not None and _lnrho_cell(v, i) != k:
                drift.append(i)
        r["ids_in_frozen_order"] = list(ids)
        r["restage_ln_rho"] = values
        r["staged_ln_rho"] = {i: staged[i] for i in ids}
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
    full = agreement_row(records, ids)  # the subset's full forty-ID-style ledger over S
    n = row["resolved"]
    return {
        "subset": name,
        "authority": AUTHORITY_NONE,
        "size": len(ids),
        "sha256": digest,
        "ids": sorted(ids),
        "ledger": full["ledger"],
        "pair_status": full["pair_status"],
        "intersection_with_X": x,
        "S_minus_X_ids": rest,
        "unresolved_ids": row["ledger"]["unresolved"]["ids"],
        "unresolved_causes": row["ledger"]["unresolved"]["causes"],
        "resolved_n": n,
        "resolved_ids": row["resolved_ids"],
        "agreements": row["agreements"],
        "agreement_ids": row["agreement_ids"],
        "disagreements": row["disagreements"],
        "disagreement_ids": row["disagreement_ids"],
        "zero_prediction_disagreements": row["zero_prediction_disagreements"],
        "zero_prediction_ids": row["zero_prediction_ids"],
        "field": row["field"],
        "comparator": row["comparator"],
        "sign_table": row["sign_table"],
        "predicted_sign_counts": row["predicted_sign_counts"],
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
    manifest_sha256=None,
    frozen_manifest=None,
    text_groups=None,
    spec_check=None,
    validity=None,
    part_b=None,
    part_b_sha256=None,
    masked_calibration=None,
    masked_calibration_sha256=None,
    resolution_draws=RESOLUTION_DRAWS,
    bootstrap_draws=BOOTSTRAP_DRAWS,
) -> dict:
    """The single Part A analysis: registered outputs plus the F10.5 reports.

    ``manifest`` is the execution manifest (the one whose variants were scored) and
    ``manifest_sha256`` its verified file hash (:func:`load_execution_manifest`; required).
    ``frozen_manifest`` is the preserved a533 staging manifest (default: loaded from
    :data:`DEFAULT_FROZEN_MANIFEST` and verified by :data:`FROZEN_STAGED_MANIFEST_SHA256`); like
    ``spec_check`` and ``text_groups`` an injected object is a trusted interface.

    Refuses (:class:`AnalysisRefused`) before any outcome is summarised unless the specification
    hash, the manifest binding, the registered window order and every frozen id list verify.
    """
    spec_check = spec_check if spec_check is not None else verify_spec()
    if spec_check.get("verified") is not True or spec_check.get("sha256_file") != SPEC_SHA256:
        raise AnalysisRefused("specification hash not verified at run time (F10b.1)")
    frozen = frozen_manifest if frozen_manifest is not None else load_frozen_manifest()
    binding = bind_manifests(frozen, manifest, manifest_sha256)
    text_groups = text_groups if text_groups is not None else load_text_groups()
    audit = validate_ledger(ledger, manifest, text_groups, manifest_sha256)
    preds = validate_fields(fields, ledger, manifest, audit)
    sel = _selected(manifest)
    ids = sorted(sel)
    registered, supplementary = binding["registered_ids"], binding["supplementary_ids"]
    # Fail closed on every frozen id list before any outcome is summarised.
    norm_tertiles(frozen)
    lnrho_tertiles(frozen)
    for name, sub, sha in (("S30", S30_IDS, S30_SHA256), ("S34", S34_IDS, S34_SHA256)):
        if sorted_sha(sub) != sha:
            raise AnalysisRefused(
                f"{name} id list sha256 {sorted_sha(sub)} != frozen {sha}; subset void"
            )
    masked = masked_calibration_state(masked_calibration, masked_calibration_sha256, audit)

    records, res_meta = resolve(ledger, manifest, preds, audit, draws=resolution_draws)
    contrasts = {c: agreement_row(records, ids, c) for c in CONTRASTS}
    for c in CONTRASTS:
        contrasts[c]["by_family"] = {
            fam: agreement_row(records, [i for i in ids if sel[i]["family"] == fam], c)
            for fam in ("lora_sft", "chain")
        }
        contrasts[c]["by_selection"] = {
            "authority": AUTHORITY_NONE,
            "registered_40": agreement_row(records, registered, c),
            "supplementary": agreement_row(records, supplementary, c),
        }
    for c in ("ladder_0.25", "ladder_0.5"):
        contrasts[c]["realised_increment_ladder_ids"] = sorted(REALISED_INCREMENT_LADDER_IDS)
    internal = {
        "specification_sha256": {
            "pass": True,
            "evidence": f"{SPEC['path']} sha256 {spec_check['sha256_file']} = F10b.1",
        },
        "text_groups_window_order": {
            "pass": True,
            "evidence": f"{text_groups['path']} sha256 {text_groups['sha256']}",
        },
        "null_size_report": {
            "pass": nullsize.get("verdict") == "PASS",
            "reason": "frozen NULL-SIZE report verdict is not PASS",
            "evidence": f"docs/prereg/a5_manifests/nullsize/report.json (sha256 {NULLSIZE_REPORT_SHA256})",
        },
        "frozen_manifest_sha256": {
            "pass": True,
            "evidence": f"manifest_a533.json sha256 {FROZEN_STAGED_MANIFEST_SHA256}",
        },
        "execution_manifest_sha256": {
            "pass": True,
            "evidence": f"execution manifest sha256 {manifest_sha256} = ledger provenance",
        },
        "f10_6_calibration": {
            "pass": masked["passed"],
            "reason": F10_6_FAILED,
            "evidence": masked.get("sha256") or masked["status"],
        },
    }
    bindings = validity_bindings(
        spec_check["sha256_file"],
        ledger.get("fields_sha256"),
        ledger.get("restage_provenance"),
        masked.get("sha256"),
    )
    dec = decision(
        validity,
        contrasts["rotation"],
        contrasts["ladder_0.25"],
        part_b,
        internal,
        bindings,
        part_b_sha256,
    )

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
            "predicted_sign_counts": contrasts["rotation"]["predicted_sign_counts"],
            "negative_prediction_share": contrasts["rotation"]["negative_prediction_share"],
        },
        "shown_beside": "main contrast, every tertile row, both subsets, covariate θ_raw",
        "statement": NO_SUPERIORITY_STATEMENT,
    }
    sensitivity = {
        "authority": AUTHORITY_NONE,
        "authority_statement": AUTHORITY_STATEMENT,
        "population": (
            "the forty registered pairs of the frozen a533 manifest; supplementary edits are "
            "reported in the registered analysis only"
        ),
        "covariate": covariate_report(records, ledger, manifest, frozen, bootstrap_draws),
        "tertiles": tertile_report(records, manifest, frozen),
        "subsets": {
            "authority": AUTHORITY_NONE,
            "spec": "§3 (F10.5(c), (e))",
            "X_definition": (
                "X = pairs the F10 re-stage excludes (F10.3 or pre-audit codes, including "
                "audit_missing) plus pairs with a missing or nonfinite score in either arm"
            ),
            "S30": subset_report(records, "S30", S30_IDS, S30_SHA256, registered),
            "S34": subset_report(records, "S34", S34_IDS, S34_SHA256, registered),
        },
        "constant_sign_comparator": comparator,
    }
    return {
        "schema": REPORT_SCHEMA,
        "spec": dict(SPEC) | {"verification": spec_check},
        "units": UNITS,
        "windows": {
            "text_groups": text_groups["path"],
            "text_groups_sha256": text_groups["sha256"],
            "window_ids": list(text_groups["window_ids"]),
            "two_window_blocks": text_groups["two_window_blocks"],
            "rule": "ledger window order, blocks and token-id hashes equal text_groups.json (verified)",
        },
        "manifests": {
            "frozen": {
                "path": "docs/prereg/a5_manifests/part_a/manifest_a533.json",
                "sha256": FROZEN_STAGED_MANIFEST_SHA256,
                "role": "frozen cuts, cells, staged ln ρ and subsets (spec §§1.1, 2, 3)",
            },
            "execution": {
                "sha256": manifest_sha256,
                "role": "scored variants: F10 verdicts and codes, F10.4 pooled ρ, variant ids",
                "rule": "file sha256 = supplied --manifest-sha256 = ledger manifest_sha256 = "
                "ledger restage_provenance.manifest_sha256",
            },
            "registered_ids": registered,
            "supplementary_ids": supplementary,
        },
        "restage": {
            "provenance": ledger.get("restage_provenance"),
            "audit": audit,
            "yield": masked["restage_yield"],
        },
        "f10_6_calibration": {k: v for k, v in masked.items() if k != "null_size_row"},
        "registered": {
            "authority": AUTHORITY_REGISTERED,
            "resolution": res_meta,
            "contrasts": contrasts,
            "decision": dec,
            "part_b": dec["part_b_reporting"],
            "pairs": records,
        },
        "f5_7_null_branch_table": f57_table(nullsize, masked.get("null_size_row")),
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


def _g(x):
    return "–" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def _prop(p):
    if p["n"] == 0:
        return "no resolved pairs"
    lo, hi = p["cp95"]
    return f"{p['k']}/{p['n']} = {p['estimate']:.4f} [{lo:.4f}, {hi:.4f}]"


def _ids(lst):
    return ", ".join(f"`{i}`" for i in lst) if lst else "none"


def _sign_md(tab):
    lines = [
        "| predicted sign (3 × 2) | n predicted | agree | disagree | observed d̂ > 0 | observed d̂ < 0 |",
        "|---|---|---|---|---|---|",
    ]
    for s in SIGNS:
        r = tab[s]
        label = "zero (abstain)" if s == "zero" else s
        lines.append(
            f"| {label} | {r['n']} | {r['agree']} | {r['disagree']} | {r['observed_positive']} | {r['observed_negative']} |"
        )
    return lines


def _accounting_md(r):
    return [
        f"- resolved ({len(r['resolved_ids'])}): {_ids(r['resolved_ids'])}",
        f"- agree ({len(r['agreement_ids'])}): {_ids(r['agreement_ids'])}",
        (
            f"- disagree ({len(r['disagreement_ids'])}; zero predictions inside: "
            f"{len(r['zero_prediction_ids'])} {_ids(r['zero_prediction_ids'])}): {_ids(r['disagreement_ids'])}"
        ),
    ]


def _ledger_md(led):
    parts = []
    for key, val in led.items():
        extra = ""
        for sub in ("by_code", "by_arm"):
            if val.get(sub):
                extra = " (" + "; ".join(f"{k}: {len(v)}" for k, v in val[sub].items()) + ")"
        parts.append(f"{key} {val['n']}{extra}")
    return " → ".join(parts)


def _ledger_ids_md(led):
    L = []
    for key, val in led.items():
        L.append(f"- {key} ({val['n']}): {_ids(val['ids'])}")
        for sub in ("by_code", "by_arm"):
            for k, v in (val.get(sub) or {}).items():
                L.append(f"  - {k}: {_ids(v)}")
        for i, cause in (val.get("causes") or {}).items():
            L.append(f"  - `{i}`: {cause}")
    return L


def _row_md(r):
    split = "/".join(str(r["predicted_sign_counts"][s]) for s in SIGNS)
    return (
        f"| {r.get('cell', '')} | {_ledger_md(r['ledger'])} | {r['agreements']} / "
        f"{r['disagreements']} ({r['zero_prediction_disagreements']}) | {_prop(r['field'])} | "
        f"{_prop(r['comparator'])} | {split} | {_fmt(r['negative_prediction_share'])} | "
        f"{r.get('label', '')} |"
    )


def _part_b_md(pbr):
    L = [
        (
            f"Part B reporting (always required, A5.5.1): **{pbr['status']}**; computed by this tool: "
            f"{pbr['computed_by_this_tool']}. {pbr['rule']}"
        ),
        "",
    ]
    if pbr.get("note"):
        L += [pbr["note"], ""]
    chk = pbr["validity"]
    L.append(f"Part B step-1 validity: {chk['status']}.")
    for p in chk["problems"]:
        L.append(f"- {p}")
    sup = pbr.get("supplied")
    if not isinstance(sup, dict):
        return L + [""]
    if pbr.get("sha256"):
        L.append(f"Supplied file sha256 `{pbr['sha256']}`.")
    L += [
        "",
        "| contrast | point | conditional [lo, hi] | joint [lo, hi] | valid | diagnostics (conditional; joint) |",
        "|---|---|---|---|---|---|",
    ]

    def get(d, *keys):
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
        return d

    for name in PART_B_CONTRASTS:
        L.append(
            f"| {name} | {json.dumps(get(sup, 'points', name))} | "
            f"{json.dumps(get(sup, 'intervals', name, 'conditional'))} | "
            f"{json.dumps(get(sup, 'intervals', name, 'joint'))} | "
            f"{json.dumps(get(sup, 'valid', name))} | "
            f"{json.dumps(get(sup, 'diagnostics', name, 'conditional'))}; "
            f"{json.dumps(get(sup, 'diagnostics', name, 'joint'))} |"
        )
    L += [
        "",
        (
            f"Destruction controls eligible (A5.8.3): {json.dumps(sup.get('destruction_controls_eligible'))}; "
            f"simulated power at C = 0.15: {json.dumps(sup.get('power_at_0_15'))}."
        ),
        "Evidence: "
        + (
            ", ".join(
                f"`{e.get('path')}` (`{e.get('sha256')}`)" if isinstance(e, dict) else json.dumps(e)
                for e in sup.get("evidence") or []
            )
            or "none"
        ),
        "",
    ]
    return L


def render_markdown(rep) -> str:
    reg = rep["registered"]
    sens = rep["f10_5_sensitivity"]
    sv = rep["spec"]["verification"]
    L = [
        "# Part A analysis report",
        "",
        f"Units: d̂ and predictions in **{rep['units']}** (WikiText `extra.mean_nll`; positive Δy is a worse fit).",
        f"Specification: `{rep['spec']['path']}` (commit `{rep['spec']['commit']}`, adopted by {rep['spec']['adopted_by']}).",
        (
            f"Specification hash verified at run time: file `{sv['sha256_file']}`; F10b.1 note "
            f"`{sv['sha256_f10b_note']}`; supplied `{sv['sha256_supplied']}`; code `{sv['sha256_code']}`."
        ),
        "",
        "## Inputs and provenance",
        "",
        (
            f"Windows: `{rep['windows']['text_groups']}` sha256 `{rep['windows']['text_groups_sha256']}`; "
            f"ledger order = item ids {rep['windows']['window_ids'][0]}…{rep['windows']['window_ids'][-1]} "
            f"(64), blocks {rep['windows']['two_window_blocks'][0]}…{rep['windows']['two_window_blocks'][-1]} "
            f"(32), per-window token-id hashes equal ({rep['windows']['rule']})."
        ),
    ]
    mf = rep["manifests"]
    L.append(
        f"Frozen staged manifest (cuts, cells, staged ln ρ, subsets): `{mf['frozen']['path']}` "
        f"sha256 `{mf['frozen']['sha256']}` (verified at load). Execution manifest (scored "
        f"variants, F10 verdicts, re-stage ρ, variant ids): sha256 `{mf['execution']['sha256']}` "
        f"({mf['execution']['rule']}). Registered ids: {len(mf['registered_ids'])}; supplementary "
        f"ids ({len(mf['supplementary_ids'])}, in the registered analysis, not in the F10.5 "
        f"reports): {_ids(mf['supplementary_ids'])}."
    )
    prov = rep["restage"]["provenance"]
    y = rep["restage"]["yield"]
    L.append(
        "Re-stage provenance: "
        + (
            "; ".join(f"{k} `{prov.get(k)}`" for k in RESTAGE_PROVENANCE_KEYS)
            if isinstance(prov, dict)
            else "**missing** (every rotation pair `audit_missing`)"
        )
        + f". Re-stage yield: {y['passing']}/{y['selected']} pairs pass F10.3."
    )
    f106 = rep["f10_6_calibration"]
    L += ["", "### F10.6 calibration disclosure", "", f"Status: **{f106['status']}**."]
    if f106["required"]:
        g = f106["gates"]
        L += [
            (
                f"- Slot-masked calibration report sha256 `{f106['sha256']}`; slot-mask file sha256 "
                f"`{f106['slot_mask_file_sha256']}`; canonical mapping sha256 `{f106['mapping_sha256']}` "
                "(recomputed from the re-stage exclusions)."
            ),
            f"- Masked rotation slots: {_ids(f106['masked_ids'].get('rotation'))}.",
            (
                f"- Gates: size ≤ 0.07 passed = {g['size'].get('passed')} (ordered go "
                f"{_g(g['size'].get('ordered_go_rate'))}); ordered power ≥ 0.80 passed = "
                f"{g['power'].get('passed')} ({_g(g['power'].get('ordered_go_rate'))}); "
                f"verdict **{f106['verdict']}** (masked calibration passed: {f106['passed']})."
            ),
        ]
    L += ["", f106["statement"], ""]
    L += [
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
        "",
    ]
    for c, r in reg["contrasts"].items():
        L += [
            f"#### Sign table, {c} (pooled)",
            "",
            (
                f"Zero predictions inside the disagreements: {r['zero_prediction_disagreements']}. "
                f"Share of negative predictions among resolved pairs: "
                f"{_fmt(r['negative_prediction_share'])}."
            ),
            "",
            *_sign_md(r["sign_table"]),
            "",
            *_accounting_md(r),
            "",
        ]
        bs = r["by_selection"]
        L += [
            (
                f"{c} by selection (authority: {bs['authority']}): registered 40 "
                f"({bs['registered_40']['resolved']} resolved) field {_prop(bs['registered_40']['field'])}; "
                f"supplementary ({bs['supplementary']['n_selected']} selected, "
                f"{bs['supplementary']['resolved']} resolved) field {_prop(bs['supplementary']['field'])}."
            ),
            "",
        ]
        for fam, fr in r["by_family"].items():
            L += [
                (
                    f"{c}, family `{fam}` ({fr['n_selected']} selected, {fr['resolved']} resolved): "
                    f"field {_prop(fr['field'])}"
                ),
                "",
                *_sign_md(fr["sign_table"]),
                "",
            ]
    L += [
        "Ladder variants flagged `realised-increment ladder` (F10.9): "
        + _ids(reg["contrasts"]["ladder_0.25"]["realised_increment_ladder_ids"]),
        "",
        "### A5.5 ordered decision (validity first)",
        "",
    ]
    d = reg["decision"]
    v = d["validity"]
    L += [
        f"- Step 1 validity: **{d['step_1_validity']}** — {v['reason']}",
    ]
    for k, p in v["problems"].items():
        L.append(f"  - {k}: {p}")
    L += [
        f"- Step 2 ladder alternative (s = 0.25): {d['step_2_ladder_alternative']}",
        f"- Step 3 Part A: {d['part_a']}",
        f"- Step 4 Part B eligibility: {d['step_4_part_b_eligibility']}",
        f"- Gate: **{d['gate']}**",
        f"- Funding (steps 4–5): {d['funding']}",
        f"- Scientific consequence (stated separately): {d['scientific']}",
        "",
        "Step-1 evidence links required: "
        + "; ".join(
            f"{k}: " + ", ".join(f"`{h}`" for h in hs) for k, hs in v["required_bindings"].items()
        )
        + ".",
        "",
        "### Part B (reported on every branch)",
        "",
        *_part_b_md(d["part_b_reporting"]),
        "### F5.7 null-branch table (null rates under the proxy, not observed outcomes)",
        "",
        "| scenario | ordered go (null) [CP 95 %] | alternative (null) count / rate [CP 95 %] | go / alternative / unresolved |",
        "|---|---|---|---|",
    ]
    for r in rep["f5_7_null_branch_table"]["rows"]:
        g, a = r["ordered_go_cp95"], r["alternative_cp95"]
        name = f"`{r['scenario']}`" + (f" ({r['label']})" if r.get("label") else "")
        L.append(
            f"| {name} | {r['ordered_go_rate']:.4f} [{g[0]:.4f}, {g[1]:.4f}] | "
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
        "Rules: " + "; ".join(f"{k}: {v}" for k, v in cov["rules"].items()) + ".",
        "",
        f"Uncertainty: {cov['uncertainty']}",
        "",
        "Units: " + "; ".join(f"{k}: {v}" for k, v in cov["units"].items()) + ".",
        "",
        (
            f"Columns: {', '.join(cov['columns'])}. n_cc = {cov['n_cc']}. "
            f"θ_raw {_prop(cov['theta_raw'])}; comparator {_prop(cov['comparator'])}; "
            f"full analysis set {_prop(cov['full_analysis_set_agreement'])}."
        ),
        "",
        f"- selected ({len(cov['selected_ids'])}): {_ids(cov['selected_ids'])}",
        f"- resolved rotation pairs ({len(cov['resolved_ids'])}): {_ids(cov['resolved_ids'])}",
        "- dropped (covariate missing, both models): "
        + (", ".join(f"`{x['id']}` ({x['reason']})" for x in cov["dropped_rows"]) or "none"),
        f"- complete-case rows ({cov['n_cc']}): {_ids(cov['complete_case_ids'])}",
        f"- complete-case ledger: {_ledger_md(cov['complete_case_agreement']['ledger'])}",
        *_accounting_md(cov["complete_case_agreement"]),
        "",
        "θ_raw sign table (complete-case rows):",
        "",
        *_sign_md(cov["sign_table"]),
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
                f"| {label} | {cov['n_cc']} | {_prop(cov['theta_raw'])} | {x['status']} | – | – | – | {loo_s(m)} | – / {folds(m)} | – |"
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
        "#### Fit diagnostics (full fits)",
        "",
        "| model | status | converged | iterations | max abs gradient | degenerate columns | standardisation μ / σ (population SD, training rows) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ("M0", "M1"):
        x = mods[m]
        f = x.get("fit") or {}
        st = f.get("standardisation") or {}
        musd = ", ".join(f"{c} {_g(st['mean'][c])} / {_g(st['sd'][c])}" for c in st.get("mean", {}))
        L.append(
            f"| {m} | {x['status']} | {f.get('converged', '–')} | {f.get('iterations', '–')} | "
            f"{_g(f.get('gradient_max_abs'))} | {', '.join(st.get('degenerate_columns', [])) or '–'} | {musd or '–'} |"
        )
    L += [
        "",
        "#### Bootstrap diagnostics",
        "",
        "| model | draws | failed | failure reasons | fits with degenerate columns | degenerate-column events | iterations max / mean | max abs gradient (max) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in ("M0", "M1"):
        bd = mods[m].get("bootstrap_diagnostics")
        if not bd:
            L.append(f"| {m} | – | – | {mods[m]['status']} | – | – | – | – |")
            continue
        ev = ", ".join(f"{c} {k}" for c, k in bd["degenerate_column_events"].items() if k) or "none"
        reasons = ", ".join(f"{k}: {v}" for k, v in bd["failure_reasons"].items()) or "none"
        L.append(
            f"| {m} | {bd['draws']} | {bd['failed_draws']} | {reasons} | {bd['fits_with_degenerate_columns']} | {ev} | "
            f"{_g(bd['iterations_max'])} / {_g(bd['iterations_mean'])} | {_g(bd['gradient_max_abs_max'])} |"
        )
    if loo:
        L += [
            "",
            "#### Leave-one-pair-out folds (diagnostics only)",
            "",
            "| fold | held-out id | y | model | status | iterations | max abs gradient | degenerate columns | p(held-out) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for m in ("intercept_only", "M0", "M1"):
            for f in loo[m]["fold_records"]:
                st = f.get("standardisation") or {}
                L.append(
                    f"| {f['fold']} | `{f['held_out']}` | {f['y']:.0f} | {m} | {f['status']} | "
                    f"{f.get('iterations', '–')} | {_g(f.get('gradient_max_abs'))} | "
                    f"{', '.join(st.get('degenerate_columns', [])) or '–'} | {_g(f.get('p_held_out'))} |"
                )
        L += [
            "",
            (
                "Per-fold standardisation parameters (M1 training rows; M0 uses the same rows and its "
                "first seven columns), μ / σ:"
            ),
            "",
            "| fold | held-out id | " + " | ".join(COVARIATE_COLUMNS) + " |",
            "|---|---|" + "---|" * len(COVARIATE_COLUMNS),
        ]
        for f in loo["M1"]["fold_records"]:
            st = f.get("standardisation")
            if not st:
                L.append(f"| {f['fold']} | `{f['held_out']}` | " + "– | " * len(COVARIATE_COLUMNS))
                continue
            L.append(
                f"| {f['fold']} | `{f['held_out']}` | "
                + " | ".join(f"{_g(st['mean'][c])} / {_g(st['sd'][c])}" for c in COVARIATE_COLUMNS)
                + " |"
            )
    L += [
        "",
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
    cuts = sens["tertiles"]["lnrho_cuts"]
    L += [
        "",
        (
            f"ln ρ frozen cut keys (staged ln ρ, edit id): cut 1 = ({cuts[0][0]!r}, `{cuts[0][1]}`); "
            f"cut 2 = ({cuts[1][0]!r}, `{cuts[1][1]}`)."
        ),
        "",
    ]
    for r in sens["tertiles"]["norm"] + sens["tertiles"]["ln_rho"]:
        L += [
            f"#### {r['cell']} — {r['label']} (authority: {r['authority']})",
            "",
            f"Field {_prop(r['field'])}; comparator {_prop(r['comparator'])}. Ledger: {_ledger_md(r['ledger'])}.",
            "",
            *_sign_md(r["sign_table"]),
            "",
            *_accounting_md(r),
        ]
        if "ids_in_frozen_order" in r:
            L += [
                "",
                "| frozen member (order) | staged ln ρ | re-stage ln ρ | cell_drift |",
                "|---|---|---|---|",
            ]
            for i in r["ids_in_frozen_order"]:
                L.append(
                    f"| `{i}` | {_g(r['staged_ln_rho'][i])} | {_g(r['restage_ln_rho'][i])} | "
                    f"{'yes (not moved)' if i in r['cell_drift'] else 'no'} |"
                )
        L.append("")
    drift = [i for r in sens["tertiles"]["ln_rho"] for i in r["cell_drift"]]
    L += ["ln ρ cell_drift (not moved): " + _ids(drift), ""]
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
                f"negative-prediction share {_fmt(s['negative_prediction_share'])}; "
                f"agree / disagree {s['agreements']} / {s['disagreements']} "
                f"(zero predictions inside the disagreements: {s['zero_prediction_disagreements']})."
            ),
            "",
            f"Subset ledger: {_ledger_md(s['ledger'])}.",
            "",
            "- S ∩ X: "
            + (
                ", ".join(f"`{d['id']}` ({d['reason']})" for d in s["intersection_with_X"])
                or "none"
            ),
            "- unresolved in S \\ X: "
            + (", ".join(f"`{i}` ({c})" for i, c in s["unresolved_causes"].items()) or "none"),
            f"- resolved ({len(s['resolved_ids'])}): {_ids(s['resolved_ids'])}",
            f"- agree ({len(s['agreement_ids'])}): {_ids(s['agreement_ids'])}",
            f"- disagree ({len(s['disagreement_ids'])}): {_ids(s['disagreement_ids'])}",
            f"- zero predictions ({len(s['zero_prediction_ids'])}): {_ids(s['zero_prediction_ids'])}",
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
        "## Ledger appendix (forty IDs, every contrast)",
        "",
    ]
    for cname, r in reg["contrasts"].items():
        L += [f"### {cname}", "", *_ledger_ids_md(r["ledger"]), ""]
    L += [
        "### Pair status (every pair, every contrast)",
        "",
        "| edit id | " + " | ".join(CONTRASTS) + " |",
        "|---|" + "---|" * len(CONTRASTS),
    ]
    for i, recs in reg["pairs"].items():
        L.append(
            f"| `{i}` | "
            + " | ".join(
                f"{recs[cn]['status']}: {recs[cn].get('detail') or ''}" for cn in CONTRASTS
            )
            + " |"
        )
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
    ap.add_argument(
        "--spec-sha256",
        required=True,
        help="sha256 of the F10b-adopted specification; must equal the file and the F10b.1 note",
    )
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    ap.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="execution manifest: the Part A manifest whose variants were scored",
    )
    ap.add_argument(
        "--manifest-sha256",
        required=True,
        help="sha256 of the execution manifest; must equal the file and the sealed ledger's record",
    )
    ap.add_argument(
        "--frozen-manifest",
        type=Path,
        default=DEFAULT_FROZEN_MANIFEST,
        help=f"preserved a533 staging manifest (sha256 must be {FROZEN_STAGED_MANIFEST_SHA256})",
    )
    ap.add_argument("--nullsize-report", type=Path, default=DEFAULT_NULLSIZE)
    ap.add_argument("--text-groups", type=Path, default=DEFAULT_TEXT_GROUPS)
    ap.add_argument(
        "--validity", type=Path, help="A5.5.1 step-1 predicates {name: {pass, evidence}}"
    )
    ap.add_argument("--part-b", type=Path, help="Part B results and diagnostics (A5.4)")
    ap.add_argument(
        "--masked-calibration",
        type=Path,
        help="F10.6 slot-masked a5_simulation.py output (required iff re-stage yield is not full)",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--markdown", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        # Condition 2: the specification is verified before any other input is read.
        spec_check = verify_spec(args.spec, args.registration, args.spec_sha256)
        frozen = load_frozen_manifest(args.frozen_manifest)
        manifest = load_execution_manifest(args.manifest, args.manifest_sha256)
        nullsize = load_nullsize_report(args.nullsize_report)
        text_groups = load_text_groups(args.text_groups)
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
        part_b_sha = sha256_file(args.part_b) if args.part_b else None
        masked = (
            json.loads(args.masked_calibration.read_text()) if args.masked_calibration else None
        )
        masked_sha = sha256_file(args.masked_calibration) if args.masked_calibration else None
        rep = analyse(
            ledger,
            manifest,
            fields,
            nullsize,
            manifest_sha256=args.manifest_sha256,
            frozen_manifest=frozen,
            text_groups=text_groups,
            spec_check=spec_check,
            validity=validity,
            part_b=part_b,
            part_b_sha256=part_b_sha,
            masked_calibration=masked,
            masked_calibration_sha256=masked_sha,
            resolution_draws=RESOLUTION_DRAWS,
            bootstrap_draws=BOOTSTRAP_DRAWS,
        )
    except (AnalysisRefused, json.JSONDecodeError, KeyError, ValueError, TypeError, OSError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2

    def ref(path):
        return {"path": str(path), "sha256": sha256_file(path)} if path else None

    rep["inputs"] = {
        "ledger": ref(args.ledger),
        "fields": {"path": str(args.fields), "sha256": fields_sha},
        "spec": {"path": str(args.spec), "sha256": spec_check["sha256_file"]},
        "spec_sha256_supplied": args.spec_sha256,
        "registration": ref(args.registration),
        "frozen_manifest": {
            "path": str(args.frozen_manifest),
            "sha256": FROZEN_STAGED_MANIFEST_SHA256,
        },
        "manifest": {"path": str(args.manifest), "sha256": args.manifest_sha256},
        "manifest_sha256_supplied": args.manifest_sha256,
        "nullsize_report_sha256": NULLSIZE_REPORT_SHA256,
        "text_groups_sha256": text_groups["sha256"],
        "validity": ref(args.validity),
        "part_b": ref(args.part_b),
        "masked_calibration": ref(args.masked_calibration),
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
