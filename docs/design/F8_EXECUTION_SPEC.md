# F8 execution specification — Board A F8 conditions 1–6

**Status: design document, 10 Nov 2026 (server date; the council and Board A documents it answers are dated 11 Nov 2026).**
This document does not amend the registration. `docs/PREREG_H1_TF.md` and everything under `docs/prereg/` are frozen and
were not edited. A later dated amendment may adopt this specification in whole or in part. Until then F8 stays as written in
`docs/PREREG_H1_TF.md` (block "A5 amendment F8"), and Board A's verdict holds: F8 is *accepted with conditions*, no F8 entry
may be scored, and the amended H5 pilot may not start (`docs/reviews/council/board-a-f8-2026-11-11.md`, verdict paragraph).

**What was done to write it.** I read the sources listed in §0.2. I made no code changes, ran no GPU job, wrote nothing under
`data/` and read no Part A outcome: `data/runs/part_a*` does not exist, and no G3 record was opened. One CPU computation was
run (§2.3): a design-time preview of the F8.4 floor. It reads only the historical NULL-SIZE payload, which Board A names as the
permitted source. The script lived in `/tmp` and nothing it produced was committed except the numbers quoted in §2.3.

**Notation.** W₀ is the pinned base (Qwen/Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`,
`docs/prereg/a5_manifests/g2/g2_manifest_final.json` → `model`). Δ is an entry's stored base-relative delta.
D_eff(s) = fp32(bf16(W₀ + s·Δ)) − fp32(W₀) is the realised increment at strength s (`src/wsp/features/rotation.py`:
`scaled_variant`, `realised_increment`). R_eff is the registered A5.3 rotation of D_eff(1) after rounding. g, G and A are the
frozen G2 WikiText gradient and K-FAC factors. a = Σ_t ⟨g_t, D_t⟩ and q = Σ_t ½ tr(G_t D_t A_t D_tᵀ) are taken over edited
tensors t (`src/wsp/features/response_field.py:contract`). y is the TF-v1 WikiText mean NLL in nats per valid next token (lower
is better).

---

## 0. Scope, authority and inputs

### 0.1 What this specification decides, and what it leaves to the adopting amendment

| Board A condition | Discharged here as a contract | Still needs (not done here) |
|---|---|---|
| 1 Freeze before exposure | §1 | the actual manifest build and freeze commits; the exposure attestation |
| 2 Deterministic F8.4 | §2 | adoption of the combined-gate reading by a dated amendment; the binding floor run |
| 3 Prediction contract | §3 | builder code; the realised deltas; the predictions file |
| 4 Astra replay | §4 | obtaining the artefact and running the checks |
| 5 Cluster rule and wording | §5 | adoption; replacement of F8.5's wording in the amendment text |
| 6 H5 scripted arm | §6 | G5 manifest pins (Sol version string, prompts), repricing, H5_DESIGN/PROTOCOL edits |
| 7 Citations | outside this spec (§8 notes only what F8 execution depends on) | `paper/refs/verified.yaml` entries; retained excerpts |

### 0.2 Inputs read, with the hashes this document was written against

| File | sha256 (first 16 hex) | Used for |
|---|---|---|
| `docs/reviews/council/board-a-f8-2026-11-11.md` | `4de06af0e5da86f9` | the seven conditions |
| `docs/PREREG_H1_TF.md` (F8 block, A5.3, A5.3.3, A5.7, A5.8, F5.4, FP5) | `52845849c3efbb62` | registered rules reused |
| `docs/council/2026-11-11/SYNTHESIS.md` §§2, 4 | `ab414b31b5e06c04` | F8 text, spend table |
| `docs/council/2026-11-11/round4-{world-model,representation,metrology,editor}.md` | — | field algebra, chart caveats, binomial points, access flags |
| `docs/prereg/a5_manifests/nullsize/{report.json,README.md,inputs_provenance.json,validation.json}` | report `1bed2c9531e4230864a1` | floor source identity |
| `docs/prereg/a5_manifests/nullsize_inputs.json` | `6e58b589e32491b3` | payload schema, units, blocks |
| `data/runs/a5/nullsize/inputs.json` (read-only; not committed) | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` | floor preview |
| `docs/prereg/a5_manifests/snapshot.json` | `1b2af25cde1bd7ad` | closed pre-G2 snapshot |
| `docs/prereg/a5_manifests/g2/g2_manifest_final.json` | `c4ec7390aaf118ed` | G2 field/projector/probe identities, supported modules |
| `docs/prereg/a5_manifests/{text_groups,panels,control_seeds}.json` | `f769afca…`, `c291b685…`, `d357ed51…` | disjointness and seed rules |
| `src/wsp/features/response_field.py` | `98d81bbc300e715f` | a, q, coverage |
| `src/wsp/features/response_chart.py` | `f23397779c13d2d6` | chart tokens, columns |
| `src/wsp/features/rotation.py` | `f3ff12426ad89dac` | realised increments, rotation |
| `src/wsp/eval/response_intervention.py` | `7cf453f263b92b3d` | staging, variant ids, references, scoring command |
| `scripts/a5_simulation.py` | `04f283d8756bc2d8` | `donor_block`, `resolution`, `cp_interval` |
| `configs/eval_tf_v1.yaml`, `configs/eval_tf_v1_wikitext.yaml` | `5c20cabe…`, `5fd90d01…` | scoring identity (see §2.8) |
| `docs/H5_DESIGN.md`, `docs/H5_PROTOCOL.md` (10 and 11 Nov sections) | `02f5ea7c…`, `75c0c401…` | H5 arms |
| `docs/RESULTS.md` | `367bbdff4296e4cd` | measured rates |

Board A reviewed main at `b627cc8`. This specification was written at main `f705888`. Two things differ for this spec: the
11 Nov sections now exist in `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md` (lines 215–244 and 100–129 respectively), and
`docs/HYPOTHESIS.md` lines 102–106 mention the scripted arm and Sol. §6 records what those sections still lack.

---

## 1. Freeze protocol (Board A condition 1)

### 1.1 One freeze, two commits, before any exposure

Both panels are frozen together by **one freeze** made in two ordered commits on a branch merged to main:

- **Commit C1 (inputs and recipes).** Construction inputs, candidate and recipe list, source clusters, seeds, the floor and its
  code (§2.3), the Astra qualification result (§4) and the exposure record. C1 is made **before any F8 delta is contracted with
  the field**. This ordering is checkable: C1 holds no a, q or κ value, and the predictions file in C2 records C1's hashes.
  Strengths and recipe parameters therefore cannot be tuned to pass the gate.
- **Commit C2 (realised deltas and predictions).** Realised-delta hashes, audits, predictions, abstentions and the gate
  decision (§§2–3). C2 is made before the first F8 TF-v1 score is started.

The F8 order of scoring stays Part A → F8-E → F8-F (F8.5). A gate failure, or an F8-E panel with no scorable entry, is
recorded in C2 and does not delay Part A or change F8-F.

### 1.2 Manifest file set

The adopting amendment creates the directory `docs/prereg/f8_manifests/`. This spec does not create it. The builder scripts
(not yet written; §9) emit every file deterministically: rerunning on the same inputs and git ref reproduces every byte, as
`scripts/a5_build_manifests.py` does for A5 (`docs/prereg/a5_manifests/README.md`).

| # | File | Commit | Content |
|---|---|---|---|
| 1 | `exposure.json` | C1 | exposure cutoff record (§1.3) |
| 2 | `construction_inputs.json` | C1 | development item/text-group IDs per recipe, sorted-ID hashes, disjointness certificates (§1.4) |
| 3 | `candidates.json` | C1 | the fixed E+F candidate set: components, source clusters, entries, recipes, seeds, missing entries with reasons (§2.2, §3) |
| 4 | `floor.json` | C1 | floor value, inputs, code hash, sensitivities (§2.3) |
| 5 | `astra_qualification.json` | C1 | §4 checks, PASS/FAIL, artefact hashes |
| 6 | `code_hashes.json` | C1 and C2 | sha256 of every builder/src file used, environment, torch/numpy versions |
| 7 | `deltas.json` | C2 | per entry and tensor: realised-delta sha256 and support/rank audit (§3.3) |
| 8 | `predictions.jsonl` | C2 | one row per entry (§3.2) |
| 9 | `gate.json` | C2 | per-component gate quantity, survival fraction, decision, entries bought (§2.5–2.6) |
| 10 | `MANIFEST.sha256` | C1, then extended in C2 | the hash chain (§1.5) |
| 11 | `README.md` | C1 and C2 | human-readable table of the above, with the chain head |

Realised delta files themselves stay under `data/runs/f8/stage/` (§3.1). Only their hashes are committed, as for Part A
(`src/wsp/eval/response_intervention.py:manifest_for`).

### 1.3 Exposure cutoff rule

**Definitions.**
- *Exposure*: any process or person making an F8 choice opens a G3 record (any `eval.json` or `records.jsonl` written by
  `wsp.eval.response_intervention score` for the Part A staged zoo), any rotation or ladder record, any F8 record, or any
  summary derived from them.
- *Hashing* a file's bytes without parsing its scores is not exposure. `docs/prereg/a5_manifests/snapshot.json` follows the
  same convention ("hashes records … reads no score").

**Rule.** C1 and C2 are valid pre-outcome freezes only if, at their commit times:
1. no F8 builder session has been exposed; and
2. no F8-E or F8-F TF-v1 score exists.

`exposure.json` records:
- the freeze UTC times of C1 and C2 and the builder session ids;
- the list of G3 record paths present at C1 and C2, with sha256 of their bytes and file mtimes (hashed, not read);
- the start time of the Part A `score` job, if one is running (its `agent-tmux` job name and start time);
- one attestation line per builder session: "no G3/rotation/ladder/F8 outcome opened";
- the sha256 of each builder session's shell-command log, plus the result of a mechanical grep of those logs. The grep looks
  for reads of the Part A staged zoo's `data/evals/<zoo>/tf_v1` path, of `data/runs/part_a`, and of F8 eval outputs.

**Consequences.**
- *Preferred timing.* Freeze C1 and C2 before the Part A `score` job starts. Then condition 1 holds trivially and the
  attestation is a formality.
- *If Part A scoring starts first* (allowed, because Part A does not wait on F8): the freeze is still pre-outcome if the
  attestation and the grep hold.
- *If the grep finds a hit, or an attestation cannot be given:* the panel is **not pre-outcome**. Before anything is scored,
  an explicitly exploratory amendment is required (Board A condition 1). No F8 file may be edited after exposure. A
  correction after exposure is a new dated file that names what changed, and it makes the panel exploratory.

**Scoring record.** When F8 scores are produced, the scoring record (outside `f8_manifests/`, written after C2) links:
- the chain head of `MANIFEST.sha256`;
- the sealed Part A ledger hash (the ledger closed before Part A's single analysis, A5.3 "Analysis time");
- the F8-E and F8-F scoring run ids, in that order.

### 1.4 Disjoint development items and text groups

Every recipe input is a *development* item, drawn by the registered seed rule
`seed(domain, key) = int.from_bytes(sha256(domain + key)[:8], "big") mod 2^63`
(`scripts/a5_build_panels.py:domain_seed`; `docs/prereg/a5_manifests/control_seeds.json` → `derivation`). It uses its own
domain `20261111:f8-construction:<recipe>:` and takes the first n items of the hash-sorted eligible pool.

`construction_inputs.json` stores, per recipe, the dataset, revision, split, item IDs, n, the sorted-ID sha256 (hash
convention of `text_groups.json`: sha256 of `json.dumps(sorted ids)`) and a **computed disjointness certificate** against:

| Set that must not overlap | Source of truth |
|---|---|
| TF-v1 WikiText scored windows (test-split tokens [0, 65536) at revision `b08601e…`) | `docs/prereg/a5_manifests/text_groups.json` |
| G2 auxiliary panels (64 WikiText windows from 11 held-out test articles; MMLU-Pro and HellaSwag 500-group reserves; refusal pools; probe panel) | `docs/prereg/a5_manifests/panels.json`; `g2_manifest_final.json` → `panels` |
| TF-v1 MMLU-Pro 1,000 scored items, HellaSwag 250, KL 200, refusal/over-refusal prompts | `configs/eval_tf_v1.yaml` and its committed rankings |
| Old-seal confirmatory partition | `snapshot.json` → `old_seal.partition_sha256` |
| H5 visible pool (200 GSM8K-train + 200 MATH-train, seed 0) and the H5 sealed sets | `docs/H5_PROTOCOL.md` "Adopted decisions" 4 |

The H5 pools are excluded so that F8 does not spend H5's practice or sealed items.

**Design-default pools.** The adopting amendment may change these before C1.
- **WikiText-103 train split** at the configured dataset revision. 1,024-token windows, hash-selected. Train and test splits
  are disjoint by construction; the certificate still checks token-span overlap.
- **GSM8K train**, minus the H5 visible pool, prompts only. No answers are used.

**Rules.**
- No scored-endpoint feedback enters reconstruction, strength selection or retries.
- A retry after a construction failure reuses the same inputs and seed. The failure and the retry are both logged in
  `candidates.json`.
- No input is replaced after any construction output has been looked at.

### 1.5 Hash chain

`MANIFEST.sha256` lists the files of §1.2 in the fixed order 1–9 and a running chain head:

```text
H_0 = sha256(utf8("f8-manifest-v1:" + <git sha of the adopting amendment commit>))
H_k = sha256(H_{k-1} || sha256(file_k))        # raw 32-byte digests concatenated, k = 1..9 in table order
```

- **C1** writes H_0–H_6, with files 7–9 recorded as "pending".
- **C2** appends H_7–H_9 and must reproduce H_1–H_6 byte for byte.
- A verifier (`f8_build.py --verify`, §9) recomputes the chain and every file hash and refuses scoring on any mismatch.
- The pin verifier pattern of `scripts/a5_build_manifests.py --verify-pins HEAD` is reused for the source files listed in
  `code_hashes.json`.

---

## 2. F8.4 made deterministic (Board A condition 2)

### 2.1 Recommendation: one combined gate across one fixed E+F candidate set

**Recommended reading.** The "at least four components" requirement applies **across one fixed E+F candidate set**, frozen
once in C1. If the gate passes, it buys TF-v1 scoring of every frozen, constructed, non-no-op entry of **both** panels. The
panels are then scored in the order Part A → F8-E → F8-F and reported separately.

**Why not per panel:**
1. *Per panel, F8-E can never qualify.* Astra, aiXiv and LP form one loop component, and Scope adds at most one more (Board A,
   "Manifest and CPU gate"). E's ceiling is therefore 2 < 4. A per-panel gate would reject E by counting components, not by
   testing resolvability — a gate whose result is fixed before any value is computed.
2. *Independent sources come from the combined set.* The gate exists to judge whether a scoring purchase can be informative.
   That depends on how many independent, potentially resolvable sources are in the purchase. The sign test (§5) runs on
   independent clusters, and the combined set holds up to six of them; E alone holds at most two.
3. *A single gate cannot adapt.* Gating the combined set once, in C2, leaves no point at which F could be redesigned after
   seeing E's gate result. This matches condition 1's "one pre-exposure freeze, not sequential adaptation".
4. *The purchase is small.* Scoring every frozen entry costs at most ≈ 0.59 GPU-h at measured rates with both caps
   respected (§2.7), inside the synthesis estimates (E ≤ 0.3, F ≤ 0.6; `docs/council/2026-11-11/SYNTHESIS.md` §2). A
   per-panel split saves at most ≈ 0.3 GPU-h, and only by never scoring E.

**Costs of this reading, disclosed:**
- F8-E entries may be scored on the strength of F8-F components. This is reported. E's result is shown per panel, and per
  entry. It never says "E passed the gate".
- The gate is a panel-level decision conditioned on the field's own magnitude predictions (prediction-conditioned sampling,
  Board A). Buying *all* frozen entries once the gate passes stops that conditioning from reaching individual entries.

**Fallback if the amendment adopts the per-panel reading instead.** F8-E is reported unscored with its predictions, unless it
genuinely holds four qualifying components, which it cannot. F8-F is gated on its own four components and therefore needs all
four to pass. Loop descendants, tensors, rotations and scales are never extra components under either reading.

### 2.2 The fixed component set and its representatives

`candidates.json` freezes the following components in this order. A component is one source cluster in the A5.3.1 sense: one
independent construction source. Its several entries (scales, rotation, recipe variants) are never separate trials.

| # | Component | Panel | Representative entry (s = 1, unrotated) — first available in the listed order | Other entries |
|---|---|---|---|---|
| C1 | Loop-removal family | E | (i) Astra replay if §4 PASS; else (ii) aiXiv difference-of-means reconstruction if constructed; else (iii) a published LP edit with complete bytes/recipe; else **missing** | all present members × {s 0.25, 0.5, 1, R} |
| C2 | WSP-Scope column edit (arXiv 2611.09382) | E | the released u, if released before C1; else **missing** (never estimated) | {0.25, 0.5, 1, R} |
| C3 | Task-vector rank-one on o_proj (arXiv 2609.09054 recipe) | F | `model.layers.11.self_attn.o_proj` edit | {0.25, 0.5, 1, R} |
| C4 | Contrast-based rank-one on down_proj (own definition; not a reproduction of the Concept Country edit, condition 7) | F | `model.layers.12.mlp.down_proj` edit | {0.25, 0.5, 1, R} |
| C5 | Wanda-initialised gate deletion, 0.04 % of entries | F | all 24 `mlp.gate_proj` tensors | {0.25, 0.5, 1, R} |
| C6 | Count-matched random gate deletion | F | same tensors and per-tensor counts as C5 | {0.25, 0.5, 1, R} |

**Recipe defaults** (each a design default that the amendment may change before C1; never after):

- **C3.** D = ρ‖W‖_F · ŝ v̂ᵀ, where:
  - v = Σᵢ sign(μᵀvᵢ) σᵢ vᵢ is built from W's SVD and μ, the mean o_proj input over 64 GSM8K-train construction prompts
    (`docs/council/2026-11-11/round4-representation.md` §3 gives the TFTV algebra).
  - ŝ is the unit mean o_proj output on the same prompts.
  - Layer 11 is the lower median of the six full-attention layers {3, 7, 11, 15, 19, 23}, listed in the G2 supported modules
    (`g2_manifest_final.json` → `supported_modules`).
- **C4.** D = −ρ‖W‖_F · û v̂ᵀ, where:
  - û is the unit difference of mean residual stream after layer 12, between 64 GSM8K-train prompts and 64 WikiText-103-train
    windows (each truncated to 256 tokens).
  - v̂ is the unit mean down_proj input on the GSM8K set.
  - Layer 12 = 24/2.
- **ρ for C3 and C4.** ρ = 0.05 sets the s = 1 norm. It is fixed in C1, before any field value exists. The ladder covers
  0.25ρ and 0.5ρ.
- **C5.** In each gate_proj W (shape out×in), delete the ⌊0.0004·out·in⌋ entries with the lowest Wanda score |W_ij|·‖X_j‖₂.
  X_j is input feature j over 64 WikiText-103-train construction windows. Ties go by (row, column) index ascending.
  D = −W ⊙ M. Deletion is bf16-exact (`round4-world-model.md` Q3).
- **C6.** A uniform-random mask with the same per-tensor count, drawn with
  `numpy.random.Generator(PCG64(seed("20261111:f8-construction:random-deletion:", <tensor name>)))`, then the same D = −W ⊙ M.
  C6 is a matched control of C5 but has an independent direction source, so it is its own component. This is disclosed.

**Other rules:**
- **Rotation.** Every R uses the registered A5.3/A5.3.3 construction (`rotation.rotate_block` through
  `response_intervention.build_variants`), with seed `rotation_seed(<F8 canonical id>)` (domain `20261107:rotation:`).
  Rotations that are excluded under A5.3.3 (`zero_block`, `rank_degenerate`, `non_2d`, audit failure) are listed, never
  replaced.
- **Missing entries are listed, never substituted:** components C1(iii) and C2 unless bytes exist, and C1(ii) unless its
  construction is funded (§2.7).
- **No component is added after C1.**

### 2.3 The floor: exact computation and a design-time preview

**Source.** The accepted historical NULL-SIZE payload `data/runs/a5/nullsize/inputs.json`, sha256
`67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de`. It is pinned in
`docs/prereg/a5_manifests/nullsize/report.json` → `inputs.sha256` and `inputs_provenance.json` → `payload_sha256`.
It holds old development edit-minus-predecessor TF-v1 noise (`donor_source` `tf_v1_edit_minus_actual_predecessor`). It is not
a G3 rotation outcome (Board A, "Manifest and CPU gate"). **Never** use fresh Part A or F8 outcomes, or a rewritten null
calibration.

| Element | Frozen choice | Basis |
|---|---|---|
| File and fields | payload keys `centred_noise` [141, 64], `window_mask` [141, 64], `donors[j].{id, component, family, eligible}` | `nullsize_inputs.json` → `payload_schema` |
| Units | WikiText `extra.mean_nll`, nats per valid next token; 1,023 tokens per window; mean over windows is token-weighted | `nullsize_inputs.json` → `residuals.field` |
| Contrast and scale | scale 1 (noise_scale 1.0); the half-width of the 64-window mean of a single edit-minus-reference difference, the same object as an F8 s = 1 entry versus base | proxy definition `U_i[w] = y(W_i+D_i)[w] − y(W_i)[w]` |
| Pairing and blocks | 32 two-window blocks (2k, 2k+1): `a5_simulation.donor_block(noise, mask, 1.0)` | `nullsize_inputs.json` → `residuals.window_order` |
| Donor set | every `donors` row with `eligible` true, all 64 windows masked-in and finite. Expected and observed: 141 pairs in 63 components | `report.json` → `sufficiency` |
| Missingness | a donor with any masked or nonfinite window is excluded and counted (none at this payload: `masked_windows` 0). No imputation | `validation.json` → `counts` |
| Interval | percentile bootstrap of the block mean: 4,000 draws, **common** [4000, 32] index array across donors, linear .025/.975 quantiles, via `a5_simulation.resolution(blocks, rng, draws=4000)` | A5.3 "Resolution"; A5.3.2 quantile convention |
| RNG | `numpy.random.Generator(numpy.random.PCG64(seed("20261111:f8-floor:", "resolution")))` = PCG64(1731840362851662234) | `domain_seed` rule |
| Per-pair statistic | h_j = (q.975 − q.025)/2 | — |
| Weighting | **component-level**: h_c = median of h_j over pairs j in component c; floor h* = median of h_c over the 63 components (`numpy.median`; for an even count, the mean of the two middle values) | A5.3.1 unit = component; 116 chain pairs in 38 components must not outvote 25 LoRA roots |
| Comparison | strict: a component qualifies iff its gate quantity > h* | — |

**Why component-level weighting.** I chose it before computing any of the values below. F8 entries are root edits counted per
component, and A5.3.1 counts independence per component. Pair-level weighting would let 116 small chain increments (38
components) dominate the 25 LoRA roots, even though F8 entries resemble roots more than chain increments.

**Design-time preview.** CPU only; the script imports `scripts/a5_simulation.py` at sha256 `04f283d8…`; numpy as installed on
this host. Run on 10 Nov 2026 from `/tmp`; it read only the payload above.

| Statistic | Value (nats/token) |
|---|---|
| **h\* (component median-of-medians, 63 components, 141 pairs)** — the proposed floor | **0.0049397** |
| sensitivity: median over the 141 pairs, unweighted | 0.0011513 |
| sensitivity: LoRA pairs only (25) / chain pairs only (116), median | 0.0091528 / 0.0006664 |
| sensitivity: range of h_j | 0.000167 – 0.012924 |
| check: normal approximation 1.96·SD(block means)/√32, median over pairs | 0.0011536 (agrees with the bootstrap pair median) |

The choice of weighting moves the floor by a factor of about 4. This is why it has to be frozen, and why the sensitivity rows
are reported but are not binding.

For scale: the 25 root LoRA donors have median |⟨g, D⟩| = 0.245 nats (`round4-world-model.md` Q4, from
`docs/prereg/a5_manifests/power/fields_manifest.json` → `donors`), about 50 × h\*.

**Binding value.** The C1 builder recomputes h\* with committed code and writes into `floor.json`: the value, the payload
sha256, the code sha256, numpy/Python versions, the per-component h_c list and the sensitivities. If the binding value differs
from 0.0049397 in any printed digit, the builder stops and the discrepancy is reported. The design preview is never silently
replaced.

### 2.4 The gate quantity per component

For the component's representative entry at s = 1:

```text
a_rep = Σ_t ⟨g_t, D_eff,t(1)⟩            wikitext_nll field, realised bf16 increment
q_rep = Σ_{t supported} q_t              q_t = ½ tr(G_t D A_t Dᵀ) from contract(); per-tensor, cross-tensor terms omitted
Q_c   = |a_rep| + q_rep                  nats per token (q is a Fisher/KL-type magnitude, not a signed Hessian term)
qualifies_c  ⇔  Q_c finite  ∧  Q_c > h*
```

**Edge cases** (all retained in `gate.json` with a reason):
- *Missing component* (no bytes, failed replay, unfunded construction): does not qualify.
- *Field abstention* (a gradient missing for an edited tensor, or a nonfinite a): does not qualify.
- *Unsupported tensors* (q_t = None, `contract` never fabricates zero curvature): contribute 0 to q_rep and set
  `q_partial = true`. This makes qualifying harder, so it is conservative.
- *a = 0 and q = 0*: Q = 0, does not qualify.
- *Representative is a bf16 no-op* (D_eff(1) ≡ 0): does not qualify.

`gate.json` also stores the **descriptive** |a_rep + q_rep|, the size predicted by the quadratic proxy at s = 1. When a < 0
and q > 0 the two terms cancel, so |a|+q is a magnitude screen, not a predicted effect (Board A).

**Passing the screen does not certify resolution.** h\* is a median noise half-width. An entry whose true |Δy| equals h\* is
resolved about half the time even with a perfect prediction. And Q_c can exceed h\* while the proxy's predicted effect
|a + q| is near zero.

### 2.5 bf16 surviving-entry fraction

- **Denominator N_e:** every frozen entry (all components × {s 0.25, 0.5, 1, R}) that was constructed — bytes exist and
  construction did not fail. Missing entries are listed and not in N_e. Rotation entries excluded by A5.3.3 are in N_e as
  non-surviving.
- **An entry survives iff** every intended tensor has a nonzero realised increment, and the realised-to-intended Frobenius norm
  ratio of the whole entry lies in [0.5, 2]. "Intended" means s·Δ for ladder entries and R_raw for rotations. Both norms come
  from `build_variants` stats (`realised_norm`, `intended_norm`).
- **Threshold:** the surviving fraction f_surv = survivors / N_e must be ≥ **0.75**. With four entries per component, this
  tolerates about one lost entry per component — the expected failure is s = 0.25 of a small rank-one edit
  (`round4-world-model.md` Q3, bf16 caveat).
- **No-op entries** (realised change exactly zero) are not scored. Their outcome is exactly the base. Each is recorded as a
  **disagreement** in the entry ledger, as F8.3 and A5.3 require.

### 2.6 Decision and what it buys

```text
gate_pass ⇔  #{c ∈ C1..C6 : qualifies_c} ≥ 4  ∧  f_surv ≥ 0.75
if gate_pass:  score every constructed, non-no-op frozen entry of F8-E, then of F8-F (both panels), plus the base reference if
               none is on file with the same scoring identity
else:          score nothing; both panels are reported unscored with all predictions, screen values and reasons
```

- Every screened, missing, abstained, no-op and unscored entry stays in the ledger.
- The decision is computed once, in C2. No re-gating, no adding entries, no change of ρ, layer or count after C1.

### 2.7 Panel sizes, scoring caps and stop-and-reprice

Measured rates, all from `docs/RESULTS.md` "TF-v1 pilot round 3" (H0, full TF-v1 config):
- warm entry **1.003 min** median, 1.040 max;
- base entry cold 1.751 min;
- cache load and verify in a new process: 47.1 s ("TF-v1 timing pilot (round 2)").

Pricing uses 1.04 min per entry, the value also used for Part A (`response_intervention.MINUTES_PER_ENTRY`).

| Item | Entries (max) | GPU-h (H0) | Cap |
|---|---|---|---|
| F8-E scoring: Astra 4 [+ aiXiv 4 if funded] [+ LP 4 if bytes] [+ Scope 4 if released] | 4 – 16 | 0.069 – 0.277 | ≤ 0.3 |
| F8-E process overhead: one cache load (+ base reference, cold, only if no matching record is on file) | — | 0.013 (+ 0.029) | inside the E cap |
| F8-F scoring: C3–C6 × 4 | 16 | 0.277 | ≤ 0.6 |
| F8-F process overhead: one cache load | — | 0.013 | inside the F cap |
| **Scoring total, worst case** | ≤ 32 | **≤ 0.59** (E is ≤ 0.250 with Astra + aiXiv + LP; only if Scope's four entries also exist and a cold base is needed would E reach 0.319 > 0.3, and then Scope's entries, last in §2.2's order, are not bought and are listed as "unscored: cap") | E ≤ 0.3, F ≤ 0.6 |
| F8-F construction (activation statistics for C3–C5; C6 none) | — | **unpriced**: no forward-statistics rate in `docs/RESULTS.md` | ≤ 0.5 (F8.2), stop and reprice |
| F8-E construction: Astra replay (§4), floor, contractions | — | 0 (CPU) | — |
| F8-E aiXiv reconstruction (needs thinking-on generations to label loops) | — | **unpriced and unfunded**: the synthesis prices E construction as "CPU only" (`SYNTHESIS.md` §2) | needs an allocation in the amendment, otherwise "missing: unfunded" |
| Field contraction (WikiText field ≈ 11.3 GB on disk, `g2_manifest_final.json` → `fields.wikitext_nll.resources`) | — | 0 (CPU) | — |

**Stop and reprice.** Stop the panel's scoring and reprice before continuing if any of these happens:
- cumulative scoring GPU-h of a panel would exceed its cap at 1.04 min per remaining entry;
- any single entry exceeds 2 × 1.04 min wall;
- construction reaches 0.5 GPU-h.

Retries are charged to the same cap. **No registered allocation is borrowed:** Part A 2.1 + 0.33, G5 ≤ 10 + ≤ 6 and G7 are
untouched (`docs/council/2026-11-10/SYNTHESIS.md` §3).

### 2.8 Scoring configuration (a discrepancy the amendment must resolve)

F8.5 names `configs/eval_tf_v1_wikitext.yaml`. That config is the WikiText-only F5.3 4B diagnostic. Its header states that its
config hash differs from `eval_tf_v1.yaml` and that "these records never pool with full TF-v1 records"
(`configs/eval_tf_v1_wikitext.yaml` lines 1–5). Under it:
- (a) MMLU, which F8.5 makes descriptive, is not scored at all;
- (b) the common base reference cannot be reused from the Part A/D3 records, because the scoring identity differs
  (`response_intervention._reference_on_file`);
- (c) the noise used for the floor (§2.3) comes from full-config records.

**Recommendation:** score F8 with `configs/eval_tf_v1.yaml`, the Part A default (`response_intervention.DEFAULT_CONFIG`), so
that the scoring identity, base reference and floor noise all match. The amendment states the change explicitly. The prices
in §2.7 assume the full config. The WikiText-only per-entry rate on H0 is not in `docs/RESULTS.md`.

**Entry-level resolution** for every scored contrast follows A5.3:
- contrasts: y(W₀+D_eff(s)) − y(W₀) at s ∈ {0.25, 0.5, 1}, and y(W₀+D_eff(1)) − y(W₀+R_eff);
- interval: paired block bootstrap over the 32 two-window blocks, with common draws across all of a panel's entries and the
  base reference; 4,000 draws; linear percentiles;
- RNG: PCG64(`seed("20261111:f8-resolution:", "<E|F>")`);
- resolved ⇔ the interval strictly excludes 0.

---

## 3. F8.3 prediction manifest (Board A condition 3)

### 3.1 Staging convention (reusing `response_intervention.py`)

- Every source edit is a **base-root zoo entry**: `parent_id` null, `base_revision` = `2fc0636…`, dense `delta.safetensors`.
  Staging is under `data/runs/f8/stage/<zoo name>/`, outside the canonical zoo.
- Ladder and rotation variants come from `build_variants(entry_dir, predecessor_state=W₀, seed, strengths=(0.25, 0.5),
  stage_root=…)`. The s = 1 entry is the source entry itself.
- Every written tensor must round-trip bitwise through `bf16(base + delta)`, as `build_variants` checks.
- **Variant ids** use an F8 domain, so that they cannot collide with Part A ids (`a5-part-a:`):
  `sha256(f"f8:{source_id}:{kind}:{float(strength)!r}:{seed}")[:16]`.
- **Source ids** are `sha256("f8-source:" + component + ":" + recipe_sha256)[:16]`.
- **Families** must be registered explicitly, never disguised (`round4-representation.md` §1; `src/wsp/schema.py`
  `EDIT_FAMILIES` lacks them): `external_edit` for E entries and `f8_transfer` for F entries. Adding them is a code change for
  the builder branch.
- **Scoring** uses `python -m wsp.eval.response_intervention score --stage …`, which runs `wsp.eval.batch` with the configured
  file and is never reimplemented. Run it under `agent-tmux`.

### 3.2 Per-entry row of `predictions.jsonl`

One JSON object per frozen entry, including missing, excluded and no-op entries. Keys are sorted, and there are no timestamps
except in `frozen_at`.

| Group | Field | Definition |
|---|---|---|
| Identity | `entry_id`, `source_id`, `component` (C1–C6), `panel` (E/F), `member` (e.g. `astra_replay`, `aixiv_reconstruction`), `kind` (`scale`/`rotation`), `strength` (0.25/0.5/1.0), `status` (`constructed`/`missing`/`excluded`/`no_op`), `status_reason` | — |
| Pins | `base_repo`, `base_revision` = `2fc06364715b967f1860aea9cf38778875588b17`, `base_state_hash` = `18f07d70a89aa63125e7e95535ff2730eb54bae4ee846d72f183e004a740d982` (g2 manifest), `tokenizer_revision`, `tokenizer_files_sha256` {file: sha} | `g2_manifest_final.json` → `model` |
| Construction | `recipe` (name and parameters: layer, ρ, count, α, …), `recipe_sha256`, `construction_inputs_ref` (recipe key in `construction_inputs.json` plus its sorted-ID sha256), `construction_code_sha256` {path: sha}, `seeds` {name: int, domain, key}, `external_artifact` (URL, revision, file sha256s, access date; E only), `access_flags_construction` (`python_tool_access`, `forward_access`, `hidden_state_access`, `feedback_access`; `round4-editor.md` §1 table) | — |
| Realised delta | `delta_sha256` (`delta.safetensors` bytes), `tensors` [{name, shape, dtype, `increment_sha256` (`response_field._tensor_hash` of D_eff,t), `realised_norm`, `intended_norm`, `fraction_nonzero_lost`}], `survives` (§2.5) | rotation entries are computed on R_eff, never on R_raw |
| Rank/support audit | per tensor: `changed_rows`, `changed_cols` (counts and sha256 of the index lists), `nonzero` count, `numerical_rank` at rtol 1e-3 (`rotation.block_svd`), top-8 singular values, `stable_rank` = ‖D‖_F²/σ₁², `spectral_tail` (energy beyond 8); for rotations the A5.3.3 audit dict (`norm_mismatch`, `spectrum_mismatch`, `rank_difference`, `support_identical`, `orthogonality_error`, `reconstruction_residual`, `passes`) | A5.3.3; a nominal rank-one edit is audited as realised (rank may exceed 1 after rounding) |
| Field | `field_identity_sha256` (`fields.wikitext_nll.identity_sha256` = `6fab5b29…`), `field_files` {`manifest.json`, `tensors.pt`} sha256, `g2_manifest_sha256` = `c4ec7390…`, `a` (nats/token), `a_per_tensor`, `q` (nats/token, supported tensors), `q_per_tensor` (None where unsupported), `q_partial`, `field_status` (`ok`/`abstain:<reason>`) | `response_field.contract` on D_eff(s) for scale entries and on R_eff for rotations |
| Signs at each s | for scale entries: `pred_sign_first_order` = sign(a) ∈ {−1, 0, +1}, ΔNLL orientation (−1 predicts improvement) | a is evaluated on that entry's own realised increment, not s × a(1) |
| Rotation contrast | on the rotation entry's row: `a_D_minus_R` = a(D_eff(1)) − a(R_eff), and its sign | A5.3 contrast |
| Curvature | `kappa` = q/\|a\|: finite if a ≠ 0; `"inf"` if a = 0 and q > 0; null if a = 0 and q = 0; abstain if nonfinite | definitions in §3.4 |
| Quadratic forecast | `s_star` = \|a\|/q iff a < 0 and q > 0, else null; `forecast_sign_quadratic` = sign(a + s·q) with this entry's s-relative a and q (descriptive, panel only; q stays unsigned in the programme) | `round4-world-model.md` Q3 |
| Chart | `chart_predictor_id` = null, `chart_prediction` = null, `chart_status` = `abstain:no_frozen_chart_predictor` (§3.5); descriptive `chart_columns_sha256` of `to_columns(chart_tokens(...))` under `chart_schema_hash` `350a5512…` | `g2_manifest_final.json` → `projectors` |
| MMLU (descriptive) | `a_mmlu`, `pred_sign_mmlu` from the `mmlu_gold_log_prob` field (higher is better) | never decisive (F8.5) |
| Freeze | `frozen_at` (C2 time), `manifest_chain_head` (H_6 from C1) | §1.5 |

**Panel-level fields in `gate.json`:**
- `N_e` and `survivors`, so f_surv is recomputable;
- the threshold 0.75;
- `Q_c` per component;
- `h_star` and the `floor.json` sha256;
- `gate_pass`;
- `entries_to_score` (sorted ids);
- `n_entries_not_scored` by reason.

### 3.3 Order of computation

1. Build the deltas.
2. Hash them.
3. Run the audits.
4. Contract them with the frozen field (CPU).
5. Compute the signs, κ and forecasts.
6. Write `predictions.jsonl`, `deltas.json` and `gate.json`.
7. Commit C2.

Nothing is refitted (F8.3). Field artefacts are loaded through `FieldArtifact.load(dir, expected_identity=…)`. They fail
closed on any identity mismatch with `g2_manifest_final.json`.

### 3.4 Defined edge values

- **a = 0:** the prediction is zero, and a resolved zero prediction counts as a disagreement (A5.3).
- **q = 0:** κ = 0 if a ≠ 0; no s\*; forecast = sign(a).
- **Nonfinite a or q:** field abstains for that entry; the reason is recorded; the component does not qualify (§2.4).
- **s\*** exists only for a < 0 and q > 0 under the descriptive s·a + s²·q proxy. At s = s\* the forecast is 0 (reported, not
  counted).
- **No-op entry:** a = 0, q = 0, status `no_op`, counted as disagreement, not scored.
- **q stays unsigned:** it is a nonnegative Fisher magnitude, never a signed Hessian correction (A5.8.1 "Trust-region term").
  The quadratic forecast is descriptive and panel-only. It never enters Part A, A5.5 or F5.5.

### 3.5 The chart: why it abstains, and what it would take not to

- **No fitted chart predictor exists.** G2 produced projectors, a schema hash and a probe cache
  (`g2_manifest_final.json` → `projectors`, `probe_cache`). `response_chart.chart_tokens`/`to_columns` turn a delta into
  columns, not into a prediction (Board A: "projectors alone are not predictions").
- **None will exist before the freeze.** The only registered chart fit is the Part B secondary chart arm, nested and
  out-of-fold on G1 records (A5.4 "Secondary arms"). Part B (G4) runs after G3 (`docs/council/2026-11-10/SYNTHESIS.md` §3,
  order 5).
- **Root edits sit outside its training support in any case.** At a base-root entry E = 0, so the displacement products
  vanish and the base and predecessor products coincide. Direction can also vanish in the 4-dimensional random projection's
  nullspace (`round4-representation.md` §3 "Implementation caveat").

**Contract:**
- Every F8 entry carries the explicit chart abstention `abstain:no_frozen_chart_predictor`.
- Its chart columns are hashed descriptively only.
- No chart model is fitted for F8, and no Part B output is reused after the fact.

A future chart prediction would need its own amendment, written before any F8 outcome, naming four things:
- the predictor artefact sha256;
- training-only provenance (row ids, every one a D3 development record; no G3 or F8 row);
- the nuisance model;
- the scalar mapping: the predicted ΔWikiText NLL of an entry versus its reference, with sign(pred(D) − pred(R)) for rotations.

It would also need a support rule: abstain if any slot the entry populates is populated in fewer than 10 training rows.

---

## 4. Astra replay qualification (Board A condition 4)

Every step runs on CPU and fails closed. A failure at any step makes C1(i) **missing: replay not qualified**, with the failing
step recorded. Nothing is "repaired" into an exact replay. The aiXiv entry stays labelled a reconstruction. Incomplete LP
and Scope artefacts stay missing. The loop family stays one component.

**Before acquiring anything**, write the expected values into `astra_qualification.json`:
- the base revision `2fc0636…`;
- the claimed changed tensors (the release's names; the blog prose says 64 MLP output columns in each of layers 15 and 19,
  131,072 weights, but gives no tensor names: `docs/H5_RESEARCH.md` line 10);
- the config-key allowlist below.

| Step | Check | Fail-closed condition |
|---|---|---|
| A1 Provenance | Record the public URL, repository/commit or HF revision, licence, access date, and sha256 of every downloaded file. Store the files under `data/runs/f8/external/astra/` (immutable). | No public artefact; or the bytes are not retrievable; or the hashes change between two downloads |
| A2 Form | Identify the artefact as (a) a full checkpoint or (b) a diff/patch. For (b), apply it to the pinned base with the publisher's documented rule and record that rule's code hash. | Form undeterminable; or the patch rule is ambiguous |
| A3 Tokenizer/config | Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, special-tokens/chat-template files) byte-identical to the base revision's. `config.json` equal as parsed JSON after removing only the keys `_name_or_path` and `transformers_version`. | Any other difference |
| A4 Tensor-name set | The set of state-dict keys of the replayed state equals the base's exactly. | Any missing or extra key |
| A5 Shapes and dtypes | For every key, shape and dtype equal the base's (bf16 where the base is bf16). | Any mismatch |
| A6 Changed tensors | The set {t : not torch.equal(W_astra,t, W₀,t)} equals the two claimed tensor names exactly. | Any other changed tensor, or fewer than two |
| A7 Unchanged complement | For every other key, byte equality, and a complement hash = sha256 over sorted (name, `_tensor_hash`) pairs equal for base and replay. | Any inequality |
| A8 Changed coordinates | D_t = fp32(W_astra,t) − fp32(W₀,t). Record changed rows/columns (index lists and sha256), the nonzero count, and the rank/support audit of §3.2. If the release states column indices, the changed columns must be a subset of them. Compare with the prose (64 columns per tensor) and record the result; a prose mismatch is recorded, not a failure. | Changed coordinates outside the released column set |
| A9 Zoo entry | Write a dense `delta.safetensors` = {t: D_t} for the two tensors, plus `edit.json` (family `external_edit`, `parent_id` null). Assert bf16(W₀ + D_t) == W_astra,t bitwise. | Round-trip failure |
| A10 Full state identity | State hash `response_field._hash({n: _tensor_hash(t)})` of bf16(W₀ + delta) applied by `wsp.zoo.apply` equals the state hash of the replay checkpoint as loaded. | Mismatch |
| A11 Reload determinism | Two fresh processes load the entry through the scorer's apply path and produce equal state hashes. A third process repeats A1's file hashes. | Any difference |
| A12 Scoring identity | `wsp.eval.batch` dry run on the staged entry records `model_revision` = base revision, `config_hash` = `effective_config_hash(configs/eval_tf_v1.yaml)`, and `entry_hashes` = {`delta.safetensors`: sha}, so that `_reference_on_file` would accept the base reference. | Any mismatch |

**Output.** `astra_qualification.json` = {expected, observed, step results, PASS/FAIL, artefact and code hashes}. It is
committed in C1. Board A's review does not certify that this has passed. Only this file, built by committed code, can.

---

## 5. Cluster-level summary and corrected wording (Board A condition 5)

### 5.1 Unit and the deterministic rule

**Unit.** One cluster = one frozen component C1–C6, the same unit as §2.2. Recipes, tensors, scales, rotations and loop
descendants are never separate trials.

**Primary contrast per cluster.** The representative entry's **s = 1 ladder contrast** versus the base, y(W₀+D_eff(1)) −
y(W₀), on WikiText, compared with the representative's first-order prediction sign(a_rep). Why s = 1:
- it is the actual edit, the object an H5 critic would face;
- it is the most resolvable scale;
- it is fixed in advance, not chosen after looking at resolution.

The rotation contrast sign⟨g, D − R⟩ and the s = 0.25/0.5 contrasts are reported per entry and never change the cluster
outcome.

**Cluster outcome** (exactly one of these):

```text
missing      no representative entry exists (no bytes / failed replay / unfunded / excluded)
abstain      field abstained on the representative (§3.4)
disagree     representative is a no-op (zero realised change)               [counted, A5.3 / F8.3]
unresolved   scored, and its interval contains 0 (or score missing/nonfinite: listed with reason)
agree        resolved and pred ≠ 0 and sign(pred) == sign(point estimate)
disagree     resolved and (pred == 0 or sign(pred) ≠ sign(point estimate))
unscored     gate failed (§2.6): no TF-v1 score exists
```

**Mixed signs across scales** (e.g. agreement at s = 0.25 and disagreement at s = 1) do not change the cluster outcome. They
are reported per entry as a locality pattern, next to the quadratic-sign forecast and κ (`round4-world-model.md` Q1 table:
"agreement at s = 0.25 with a miss at s = 1 marks a locality boundary"). If the representative is a no-op but other scales
are not, the cluster is still `disagree`, and the other scales are reported.

**Denominator.**
- k = #agree and n = #agree + #disagree, over the frozen clusters.
- Missing, abstained, unresolved and unscored clusters are outside n and are listed by count and reason, out of the frozen
  total N_c = 6.
- The scored set is combined E+F, with each cluster labelled by panel. A per-panel breakdown is entry-level and descriptive
  only (E has at most two clusters).

### 5.2 Inference: what is computed and how it may be worded

- **Report:** k/n with the exact equal-tail two-sided 95 % Clopper–Pearson interval (`a5_simulation.cp_interval`, A5.3.2),
  and the exact two-sided binomial test of p = ½: p = min(1, 2·P(X ≥ max(k, n−k))) with X ~ Bin(n, ½).
- **Assumption:** independent cluster signs under a fair-coin null, conditional on the frozen field and the fixed 64-window
  panel (the A5.3 null).
- **Entry-level results** (all entries, all contrasts) are tables with no binomial inference.
- **MMLU** signs are reported per entry, descriptive only (F8.5).

**Replacement for F8.5's "a panel of this size cannot reject chance agreement".** The amendment should use this wording
instead:

> "Agreement is summarised over n independent resolved source clusters (of N_c = 6 frozen). With the exact two-sided binomial
> test at the 5 % level, chance agreement (p = ½) can be rejected only if all clusters agree and n ≥ 6 (6/6: p = 0.031,
> CP lower bound 0.541); with n ≤ 5 no outcome rejects (5/5: p = 0.0625, CP lower bound 0.478). The interval is reported as a
> descriptive precision statement; it satisfies, fails or unresolves no A5.5 step or F5.5 predicate and is not a G8 launch
> condition."

The report then states the **actual** n, k, p and interval. If n ≥ 6 and k = n, it says chance agreement is rejected for these
fixed, externally selected clusters. Otherwise it says the test cannot reject at this n.

Numbers come from `a5_simulation.cp_interval`. Exact two-sided p for n/n agreement:

| n | 3 | 4 | 5 | 6 |
|---|---|---|---|---|
| p | 0.25 | 0.125 | 0.0625 | 0.0312 |
| CP lower bound | 0.292 | 0.398 | 0.478 | 0.541 |

The metrology report's figures agree (`round4-metrology.md` §1). With C2 missing (likely, since Scope has not released u),
N_c effectively drops to 5 and the panel cannot reject even at perfect agreement. The report must say so.

**Scope sentences, kept** (F8.6):
- A resolved agreement is a transfer *precondition* for the H5 critic, never a G8 launch gate or a bypass of one.
- A resolved disagreement is evidence of prediction failure on that fixed endpoint for that family. It does not identify
  locality versus K-FAC error uniquely (Board A, "Registration integrity").
- An unresolved cluster says nothing.
- Selection on the |a|+q screen is prediction-conditioned and is disclosed with the result.

---

## 6. H5 scripted-estimator arm (Board A condition 6)

**Scope.** This section specifies what the 11 Nov sections of `docs/H5_DESIGN.md`/`docs/H5_PROTOCOL.md` leave open. Those
sections already state: the arm exists, it gets the same localisation as the editor, there is a field-ranked variant, Sol
medium is pinned with no substitution, prior information is logged, and savings are reported per stream. They do not yet give:
the difference-of-means inputs, the LP objective and constraints, tie-breaking, the access flags per arm, the treatment
pairing, an explicit restatement of cheap/sham/random and nested 1/8/32, or zero-denominator handling. Those are below. The
H5 documents must be appended with them before the pilot; that is an edit for the adopting amendment, not this commit.

### 6.1 Difference-of-means inputs

- **Items.** The episode's *proposal-training split* of the visible pool (200 GSM8K-train + 200 MATH-train, seed 0;
  `docs/H5_PROTOCOL.md` "Adopted decisions" 4). Never support-validation, calibration, acceptance-slice or sealed items
  (F5.4; FP5 "Acceptance slices").
- **Generations.** Base completions under the pinned G5 decode settings and token cap T, thinking on, metered through the same
  forward-request meter as the editor and charged to the arm.
- **Labels.** loop = the harness's truncation flag is true (the completion reached T without terminating). non-loop =
  terminated. Labels come from the evaluator's validity/truncation fields (`docs/H5_PROTOCOL.md` "Adopted decisions" 2),
  never from correctness.
- **Minimum counts.** ≥ 8 completions per class. Otherwise the attempt is recorded as `no_direction` (a failed, charged
  attempt), the incumbent is retained, and nothing is substituted.
- **Activations.** For each layer l in scope:
  - r_l = residual stream after block l, over the last 256 generated tokens of each completion (all tokens if shorter);
  - mean within each completion, then mean over completions (equal completion weight);
  - u_l = normalize(mean_loop r_l − mean_nonloop r_l), in fp32.
- **Neuron statistics.** x_{l,j} = post-gating input of `mlp.down_proj` at neuron j over the same tokens. For column j of
  W_l = `model.layers.l.mlp.down_proj.weight`, its contribution to the u-projection at a token is p_j = (u_lᵀ W_l[:, j])·x_{l,j}.
  The column score is s_{l,j} = E_loop[p_j] − E_nonloop[p_j].

### 6.2 LP: objective, constraints, tie-breaking, edit and scale

```text
maximise    Σ_{l∈L} Σ_j s_{l,j} z_{l,j}
subject to  Σ_j z_{l,j} ≤ B_l   for each layer l ∈ L        (column budget per layer)
            0 ≤ z_{l,j} ≤ 1
solver      scipy.optimize.linprog(c = −s, A_ub, b_ub, bounds=(0,1), method="highs"), version pinned
```

- **Selection, deterministic.** For each l, take columns sorted by (−s_{l,j}, j) and keep the first min(B_l, #{s_{l,j} > 0}).
  This is an optimal vertex of the LP. Assert that its objective equals the solver's optimum within 1e-9 relative. On a
  mismatch the attempt fails closed (`lp_mismatch`). Columns with s ≤ 0 are never selected.
- **Edit.** For selected columns S_l: W'_l[:, j] = W_l[:, j] − α (u_lᵀ W_l[:, j]) u_l. Equivalently D_l = −α u_l u_lᵀ W_l S_l,
  rank one per tensor. Rebuilt from the pinned base, stored as Δ_eff = bf16(W + Δ) − W (`docs/H5_DESIGN.md` "Delta capture").
- **Scale α.** Either supplied (when the treatment supplies it), or chosen from the fixed grid (0.5, 1.0, 1.5, 2.0):
  - each grid point is one development-support evaluation, charged;
  - the winner is the argmax of measured development improvement, with ties within 1e-12 relative going to the smaller α;
  - acceptance follows F5.4/FP5 (held-out support, rebuild, untouched acceptance slice if the bytes changed).
- **Variants.**
  - **S-loc (supplied localisation):** L, matrix, B_l and α (if supplied) are exactly what the paired LLM treatment
    receives, logged as prior information.
  - **S-field (field-localised):** compute the candidate D_l for every l ∈ 0..23 at budget B and α = 1. Rank layers with the
    frozen G2 fields by r_l = a_l^{mmlu} − q_l^{mmlu}: the first-order predicted gain of the `mmlu_gold_log_prob` field
    (higher is better) minus its own curvature magnitude, both in nats per item. Take the top |L| by (−r_l, l). This ranking
    rule is a design default, chosen because MMLU is the only G2 objective tied to the reasoning task. The amendment may
    replace it before any episode, never after. |L| and B are supplied counts, logged as prior information.

### 6.3 Treatments, pairing and access flags

Each connected episode cluster k = (goal formulation × seed × budget) runs every funded arm from the same start state (the
base), the same proposal/validation/acceptance split and the same sealed panel, with frozen seeds. Critic and no-critic are
paired **within each stream**: same proposals, seeds and budget; the critic only schedules or abstains (A5.7, F5.4).

| Treatment | Arm (stream) | Information supplied (prior information) | python / forward / hidden / feedback |
|---|---|---|---|
| T0 discovery (blog protocol) | **Astra** (registered primary, reproduction; protocol unchanged) | goal + weights only | as pinned in the G5 manifest for LLM editors; identical across critic/no-critic |
| T0 | **Sol** (GPT-6.1 Sol, medium) | same prompt, tool schema and budgets as Astra | same as Astra |
| T0 | **Haiku** (cheap control, retained) | same as Astra | same as Astra |
| T0 | **S-field** | \|L\|, B (counts only) + G2 field artefact | 1 / 1 / 1 / 1 if α is tuned, else 1 / 1 / 1 / 0 |
| T0 | **Random best-of-N** (retained; FP5, nested N = 1/8/32, matched to each LLM editor's realised proposal blocks) | support/norm of the paired editor proposal | 1 / 0 / 0 / 1 (development-winner selection) |
| T0 | **Sham reload** (retained) | — | 0 / 0 / 0 / 0 |
| T1 supplied localisation | **Sol + loc** | layers 15, 19; `mlp.down_proj`; 64 columns per layer; source = published Astra edit. Exact names only if §4 PASS; otherwise the blog prose (`docs/H5_RESEARCH.md` line 10), labelled an inference | as Sol |
| T1 | **S-loc** | identical to Sol + loc | 1 / 1 / 1 / 1 (α tuned; α is not given in the prose) |
| T1 | **Random best-of-N** on the supplied support | same support | 1 / 0 / 0 / 1 |

- **Astra's reproduction treatment stays unchanged.** Astra does not run in T1 unless separately funded.
- **When access differs, the comparison is labelled.** If the LLM editors' pinned `hidden_state_access` is 0, every
  editor-versus-scripted comparison is labelled "scripted arm has hidden-state access". An editor advantage over a
  better-informed script remains reportable. An editor deficit is not attributed to the editor.
- **The G2 field is setup access.** It is logged with its measured G2 cost (fields 1.62 GPU-h,
  `g2_manifest_final.json` → `caps.fields.used_hours`) as amortised preparation, and is not charged per episode as forward
  access.

### 6.4 Sol pin

Recorded in the G5 manifest before the first episode, hashed and logged per episode:
- provider;
- model identifier exactly as the provider reports it;
- version or snapshot string;
- reasoning effort = "medium";
- decode settings;
- system prompt sha256;
- tool schema sha256;
- the provider's per-call reported model field.

If the provider offers no immutable snapshot id, the pin is the alias plus the pin date, and any per-call model field that
differs from the pinned one ends the Sol arm. The arm is reported as stopped and not replaced (`docs/H5_PROTOCOL.md` 11 Nov
"Second editor"). Voluntary early stops are logged with their reason and the requests used.

### 6.5 Per-stream savings and claims

For each stream x ∈ {Astra, Sol, Haiku, S-field, S-loc, Sol + loc, random best-of-N}, compute A5.7's contract on that stream's
own critic/no-critic pairs:
- S_x = 1 − Σ_e E_c,e / Σ_e E_0,e, with episode-cluster bootstrap-t;
- the ≥ 30 % point estimate and paired lower bound;
- the score noninferiority lower bound ≥ −1 point;
- the missed-qualifying-proposal upper bound ≤ 5 %;
- net GPU saving > 0;
- prospective joint power ≥ 0.80.

Further rules:
- **Zero denominators:** a zero or nonfinite denominator (e.g. S-loc with α supplied makes no search evaluations) makes S_x
  **undefined**, never a success. Absolute real-evaluation counts and GPU-h are reported for every stream.
- **Primary stream:** the A5.7 primary comparison stays **Astra with versus without the critic**. Other streams are
  reported. A stream's result is only called powered if its own prospective power simulation met the 0.80 bound. There is no
  retrospective best-stream selection: every funded stream is reported.
- **Editor-advantage claims** (F8.7; Board A condition 6) need, at matched treatment (T0 or T1) and labelled access: an
  episode-paired final-score difference with lower bound > 0 points against **both** the scripted stream and random
  best-of-N, with total charged cost (GPU-h, editor tokens and provider charge, reported separately) not higher.

**Retained unchanged:** G5 shadow-only status (production rejection disabled), G8 eligibility (G5 feasibility plus G3/G4
go), and all A5.7 rejection and acceptance safeguards.

### 6.6 Price

The H5 sealed scoring tier (4096 tokens) and the diagnostic-forward latency are **unmeasured**. `docs/H5_PROTOCOL.md`
"Staged plan" says they are measured before Stage 1. So every H5 item here is **unpriced**. The only measured reference rates
are the fast tier at 12.4 min/entry and the engine trial at 6.1 min/entry (`docs/RESULTS.md` "P2 — Clean fast-tier timing").
These are not the H5 tier.

The streams added here multiply the arm count. The pilot must be repriced inside the existing G5 caps (≤ 10 GPU-h + ≤ 6 for
best-of-N; `docs/council/2026-11-10/SYNTHESIS.md` §3) before launch. If the repriced plan exceeds the caps, streams are
dropped in this fixed order until it fits:
1. S-field;
2. T1 (Sol + loc together with S-loc and its random best-of-N, so T1 stays matched);
3. Sol T0.

Astra, Haiku, sham and random best-of-N are never dropped. Dropped streams are reported as unfunded.

---

## 7. Pricing summary

| Item | GPU-h (H0) | Basis | Status |
|---|---|---|---|
| F8 floor, Astra qualification, field contraction, gate | 0 | CPU | priced (zero GPU) |
| F8-E scoring | 0.069 – 0.277 (+0.013 load, +0.029 cold base if needed) | 1.04 min/entry, 47.1 s load, 1.751 min base; `docs/RESULTS.md` | priced; cap 0.3 |
| F8-F scoring | 0.277 + 0.013 | same | priced; cap 0.6 |
| F8-F construction | — | no forward-statistics rate in `docs/RESULTS.md` | **unpriced**; cap 0.5, stop and reprice |
| F8-E aiXiv reconstruction | — | needs generations; not in any allocation | **unpriced, unfunded** |
| H5 scripted/Sol streams | — | 4096-tier and diagnostic-forward rates unmeasured | **unpriced**; inside G5 ≤ 10 + ≤ 6 after repricing |

---

## 8. Items the adopting amendment must state explicitly

1. Adopt the combined E+F gate (§2.1), or state the per-panel fallback.
2. Change the F8 scoring config from `eval_tf_v1_wikitext.yaml` to `eval_tf_v1.yaml` (§2.8), or justify keeping it and drop
   MMLU from F8.5.
3. Freeze h\* by committed code (§2.3), plus ρ = 0.05, the layer rules, the 0.04 % count and the 0.75 survival threshold, or
   replacements, all before C1.
4. Replace F8.5's binomial sentence with §5.2's wording.
5. Record that every F8 entry has an explicit chart abstention (§3.5).
6. Fund or mark missing the aiXiv reconstruction (§2.7).
7. Append §6's contents to `docs/H5_DESIGN.md` and `docs/H5_PROTOCOL.md`.
8. Condition 7 (outside this spec): the Astra replay must not rest on the blocked Barry page. C1(iii) enters only with
   retrievable bytes or a complete recipe. The C4 recipe is our own definition and is never called a Concept Country
   reproduction. URLs, access dates and artefact hashes used by F8 go into `astra_qualification.json` and
   `construction_inputs.json`.

## 9. Code needed later (not written here)

One builder, `scripts/f8_build.py` (CPU), with subcommands `inputs`, `candidates`, `floor`, `astra-qualify`, `construct`,
`predict`, `gate` and `verify`. It reuses:
- `scripts/a5_simulation.py`: `donor_block`, `resolution`, `cp_interval`;
- `scripts/a5_build_panels.py:domain_seed`;
- `src/wsp/eval/response_intervention.py`: `build_variants`, `_reference_on_file`, `score_commands`;
- `src/wsp/features/response_field.py`: `FieldArtifact.load`, `contract`;
- `src/wsp/features/rotation.py`;
- `src/wsp/features/response_chart.py`: descriptive columns only.

It needs `external_edit` and `f8_transfer` added to `src/wsp/schema.py:EDIT_FAMILIES`, with CPU tests:
- floor reproducibility (must reproduce 0.0049397 on the pinned payload);
- the hash chain;
- each §4 failure mode on synthetic states;
- LP tie-breaking.

GPU is used only for F8-F construction statistics and for scoring.

---

## 10. Checklist: Board A conditions → where discharged → what remains

| Cond. | Requirement (Board A) | Discharged in | Remains |
|---|---|---|---|
| 1 | one pre-exposure freeze of both panels; exposure cutoff; disjoint development item/text-group hashes; no scored-endpoint feedback; score order Part A → E → F with the sealed Part A ledger hash | §1.1–1.5 (two-commit freeze, file set, chain, cutoff rule and attestation, disjointness certificate, scoring-record links) | build and commit C1/C2; give the attestation; if missed, an exploratory amendment |
| 2 | per-panel vs combined; floor file/units/donors/weighting/contrast/blocks/missingness/interval/seed/median; frozen floor and code/output hashes; historical payload only; representative per component; abstentions/zero/nonfinite; whether all entries are bought; disclosure that the screen ≠ resolution; entry counts and caps with stop and reprice | §2.1 (combined, recommended), §2.2, §2.3 (exact algorithm; preview h\* = 0.0049397 nats/token), §2.4–2.6, §2.7, §2.8 | amendment adoption; binding `floor.json`; pricing of construction; config change |
| 3 | pins; realised deltas at each scale and rotation; construction inputs/code/seeds; G2 identities; signs and normalisation; rank/support audit; numeric survival fraction with denominator; realised bf16, not nominal; chart predictor or abstention; a = 0, q = 0, κ/s\* definitions; q unsigned; contract hashed before scoring; no-ops reported as disagreements | §3.1–3.5, §2.5 (f_surv ≥ 0.75 over N_e) | builder code; C2 |
| 4 | checkpoint/diff revision and hashes; two tensor names/shapes/dtypes; changed coordinates; full state identity vs `2fc0636…`; tensor-name set and unchanged complement; reload determinism; tokenizer/config/scoring identity; fail closed; aiXiv a reconstruction; loop family = one component | §4 A1–A12; §2.2 (C1 order, one component) | obtain the artefact; run §4; this spec certifies nothing |
| 5 | one deterministic cluster rule incl. mixed signs, scales, unresolved, abstentions; defensible independent denominator; corrected wording; intervals descriptive; no G8 gate | §5.1 (s = 1 representative rule, outcome table, N_c = 6, n = agree + disagree), §5.2 (exact test, replacement wording, n = 5 caveat) | amendment adoption of the wording |
| 6 | H5 docs reconciled, retaining Astra, cheap/sham/random and nested 1/8/32; Sol pin; paired treatments; no best-stream selection; DoM inputs; LP objective/constraints/ties; supplied vs field-localised; access flags; charged costs; repricing in G5 caps; per-stream A5.7 contract; zero denominators undefined; G5 shadow-only and G8 eligibility; matched-information editor claims | §6.1–6.6 | append to `docs/H5_DESIGN.md`/`docs/H5_PROTOCOL.md`; pin Sol's actual version string; measure the H5 tier and reprice |
| 7 | citation provenance | §8 item 8 (only what F8 execution touches) | `paper/refs/verified.yaml`; retained excerpts; corrected "verified" wording |

The actual manifest build cannot happen until this specification has been adopted by a dated amendment. Scoring under it
cannot happen until the Part A ledger is sealed (F8.5).
