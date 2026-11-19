# Council round 12 synthesis, 19 Nov 2026: MedGemma as a second target (study M)

**Seats and reports.**

| Seat | Model | Report |
|---|---|---|
| medical domain | Claude | `round12-domain.md` |
| methods | Claude | `round12-methods.md` |
| pro-engineer | Claude | `round12-engineer.md` |
| challenger | Gemini 3.8 Flash | `round12-challenger.md` |

The chair also checked two things directly:
- **Licence:** a fetch of the HAI-DEF terms and the prohibited-use policy, below.
- **HF access:** with the project token, `google/medgemma-4b-it`, `google/gemma-3-4b-it` and `google/gemma-3-4b-pt`
  return HTTP 200 for `config.json`. `google/medgemma-1.5-4b-it` and `google/medgemma-4b-pt` return 403.

## BLUF

1. **Go, narrowly, as a separate study M.** It is not a replication of the frozen 9B study (WM-E).
   - It has its own pre-registration (`docs/PREREG_WMM.md`), salt `wmm-`, panels, exposure list, freeze tag and
     budget.
   - It is never pooled with WM-E.
2. **Question.** Can weight edits make a domain-tuned model more accurate on generated medical QA once truncation
   is matched? Can the world model (WM) choose those edits better than statistics-based selection (StatNN)? On
   Qwen 9B the first answer was no (budget relief only).
3. **Claim (a) is not rescued.** Trajectory prediction on study M is descriptive only, at the lowest priority.
4. **Target.**
   - `google/medgemma-4b-it`, text use only, on the local 32 GB GPU. It is accessible now.
   - If the user accepts the gate, upgrade to `medgemma-1.5-4b-it` before the panels are built, never after.
   - 27B is deferred.
5. **The edit bank must be redesigned, not renamed.** Gemma 3 RMS-normalises attention and MLP outputs before the
   residual add, which cancels uniform writer rescales (engineer).
   - The leading edit family is the Gemma 3 → MedGemma *task vector* at λ ∈ [0.5, 1.5], plus an H1-type LoRA.
6. **Gate 0.** A port and inert/dose pilot, ≤ 3 local GPU-h, $0.
7. **Gate 1.** A headroom probe on fresh dev items, ≤ 12 local GPU-h, $0.
8. **Pre-registration and confirmatory run.** These come only after both gates pass. C1 keeps priority throughout.

## Where the seats agreed

- **It is not a replication.** Different base, already domain-tuned, nothing shared with the 9B bank, field or
  planner (methods, domain, engineer). Calling it a replication invites pooling, which PREREG_WME §11.2 forbids.
- **(a) will not be rescued.** In W2 the nested WM tied the pooled curve (d 0.003, Holm LB −0.053;
  `wme-stagew2-2026-11-18.md`), and the cause is not specific to the model (all four seats).
  - The challenger quoted "d = 0.000, p = 1.000". That is the k = 2 figure, not the zero-shot one. The conclusion is
    unchanged.
- **Local 4B is feasible.** The 9B measured a 27.35 GB peak on this card. The 4B estimate is 14–18 GB, which is
  extrapolated, not measured (engineer).
- **Generated answers only.** Every primary uses generated accuracy at a frozen token budget, with the round-11 R2
  rule: the gain must hold on items untruncated in both models. No log-prob proxies.

## Disagreements and rulings

**R1. Benchmarks.**

| Seat | Primaries | Notes |
|---|---|---|
| Domain | MedQA, MedMCQA | |
| Methods | MedMCQA, MedXpertQA-text | Excludes MMLU-Med, because MMLU-Pro is a sealed 9B source |
| Challenger | — | The MedGemma report says the training splits of MedQA, MedMCQA and PubMedQA were in training; MedXpertQA is near the floor (4B 14.2) |

*Ruling:*
- **Primaries:** MedQA and MedMCQA, Holm-adjusted.
  - Every gain must also hold on an option-permuted copy of the same items.
  - A gap between permuted and original items larger than a pre-set margin is treated as memorisation and fails the
    claim.
- **Secondary:** MedXpertQA-text.
- **Format probe only:** PubMedQA.
- **Excluded:** MMLU-Med, to protect the sealed 9B MMLU-Pro family, and AfriMed-QA (25 items, too small).

**R2. Edit families (engineer, domain).** The frozen 9B families (writer scale, noise, steer) are largely cancelled
by Gemma 3's post-norms.

*Ruling:* the study M bank has three families.
1. The task vector τ = θ_MedGemma − θ_Gemma3-it, over language-model weights only, at λ ∈ {0.5, …, 1.5}. Both repos
   are accessible.
2. MedGemma 1 ↔ 1.5 interpolation. This only applies if the user gets 1.5 access.
3. An H1-type self-distilled LoRA and a sycophancy LoRA. These are pre-specified, with no selection on data.

Directional edits (low-rank, signed) replace uniform rescales. Gate 0 tests that every family has a monotone,
non-flat dose response.

**R3. The licence (challenger vs domain/engineer).** The chair fetched the terms. Findings:
- Modification and Model Derivatives are allowed (§2.2).
- Distributing weights brings notice and recipient obligations (§3.1). Publication of *results* is not addressed.
- The indemnity (§4.5) applies to *any* user. It names modification and derivatives explicitly, so it is not unique
  to us.
- Prohibited-use item 2.4 bars "attempts to override or circumvent safety filters".

*Ruling:*
- We publish results and code only. **We never publish edited MedGemma weights.**
- No edit may target lower refusal of harmful requests.
- A harmful-compliance increase is both a guard failure and a stop.
- No clinical use.
- The indemnity is a residual risk the user must accept. The chair flags it and does not decide it.

**R4. Guards.** These are all required.
- **Harmful compliance:** harmful medical requests.
- **Benign over-refusal:** medical questions.
- **Calibration:** Brier/ECE of the answer letter.
- **General ability:** non-health generation, from a source not used by WM-E.

The guard scorer must reach ≥ 0.9 agreement with a two-judge audit on Gemma outputs. The 9B audit (0.986) does not
carry over. The challenger's point that refusal HVPs disagreed in pilot 3 S argues against *linear surrogate* guards.
Study M guards are measured on generated output, never predicted.

**R5. Power.** The 9B discordances do not transfer (methods).
- N is set from Gate 1 dev discordance with the round-11 formula.
- Planning range at power 0.8 for a gain of 0.03: about 1,047–2,631 items per benchmark (methods).
- If a test pool is too small, the achieved power is declared before the freeze. If it is below 0.5, kill.

**R6. Sequencing and compute.**
- Gates 0 and 1 run locally: their own track venv (`scripts/venv_track.sh wmm …`), ≤ 16 GB VRAM, niced, and yield
  to other sessions.
- The confirmatory study is estimated at about 90–110 local GPU-h (unmeasured) or about 29 H100 GPU-h (about $140)
  on Lambda, after C1, on its own budget. The engineer's estimate is by parameter scaling. The user must approve that
  budget at the pre-registration stage.
- The Qwen3.8-27B replication of H-E1 keeps its place after the 9B seal.

## Gates and kill rules

**Gate 0 (port and dose)**, ≤ 3 GPU-h.
- **Fix the four silent bugs:**
  - tied embeddings;
  - the vision tower slipping past the `.visual.` check;
  - the `<|im_end|>` → `<end_of_turn>` mapping;
  - the ignored `enable_thinking`.
- **Run these checks:**
  - an exact-null and inert identity check;
  - a canary;
  - dose ladders for each family on exposed/dev items.
- **Kill or redesign** if any of these holds:
  - the inert/null check fails;
  - 3 or more families have flat or non-monotone dose;
  - the base misses its model-card MedQA by more than 5 points under our template.

**Gate 1 (headroom)**, ≤ 12 GPU-h, on fresh dev items with a salted designation.
- **Kill** if any of these holds:
  - no edit is positive on both the untruncated-in-both items and the permuted copy;
  - no token budget brings base truncation to ≤ 2 %;
  - base sycophancy on medical facts drops by less than 5 points (this kills M-SYC only);
  - every qualifying edit fails a guard;
  - the guard scorer agreement is below 0.9.

**Pre-registration (`PREREG_WMM`)**, Board A, freeze. Only after Gates 0 and 1. The (a) analysis is pre-declared
as descriptive.

## Decision table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Study M on MedGemma 4B-it, separate prereg | Domain-tuned model with known accuracy direction; not poolable with WM-E | PREREG_WMM, Board A, freeze after G1 | 0, $0 | Drop if the licence is not accepted |
| Task-vector + LoRA bank; no uniform rescales | Gemma 3 post-norms cancel rescales | G0 dose monotonicity | ≤ 3 local, $0 | ≥ 3 flat families: redesign or drop |
| Primaries MedQA + MedMCQA, permuted copies | Headroom (64.4/55.7 on 4B) and leakage control | G1 headroom on fresh dev | ≤ 12 local, $0 | No untruncated + permuted gain: stop |
| (a) descriptive only | W2 negative is not model-specific | none | 0 | never claimed |
| No weight distribution; no harmful-refusal targeting | HAI-DEF §3.1, prohibited-use 2.4 | harmful-compliance guard | 0 | Guard breach: stop the edit |
| Confirmatory local or Lambda after C1 | C1 priority; Lambda has no capacity | user budget approval at prereg | ~90–110 local or ~29 H100 (~$140) | Achieved power < 0.5: kill |
| 27B deferred | LoRA memory at 80 GB untested; ~$980 full design | after 4B result | — | — |

## Addendum (19 Nov 2026, chair): access and risk acceptance

The user made two decisions, both before any study M data:
- They accepted the Hugging Face terms. `google/medgemma-1.5-4b-it` now returns HTTP 200 with the project token.
- They accepted the HAI-DEF §4.5 indemnity as a residual risk.

Under BLUF item 4, the **study M target becomes `google/medgemma-1.5-4b-it`** (text use only). This happens before any
panel exists. `google/medgemma-4b-it` stays as the endpoint of the interpolation family, so R2 family 2 is now in
scope.

The task vector is τ = medgemma-1.5-4b-it − gemma-3-4b-it. Gate 0 must first verify that the two models share the
same architecture and shapes.

The base check now uses the 1.5 model-card MedQA figure (69.1), with the same kill rule: a miss of more than 5
points kills.

`google/medgemma-4b-pt` is still 403 and is not needed.
