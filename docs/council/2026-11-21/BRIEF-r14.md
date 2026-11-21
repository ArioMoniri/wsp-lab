# Council brief, round 14, 21 Nov 2026: weight watermarking and fingerprinting, and our weight-space tools

## The user's request (paraphrased)

Check with the council how our work relates to **model watermarking**:

- whether the issues we keep hitting relate to watermarks in the weights;
- whether our methods can **find watermarked weights**, meaning detect or verify a weight-level watermark or
  fingerprint;
- whether watermarking or fingerprinting could **improve this process**.

## Scope and safety boundary (binding)

**In scope:**
- **Defensive and provenance uses:**
  - detecting or verifying an owner's watermark or fingerprint;
  - checking lineage, for example whether model X derives from Y;
  - measuring how robust existing watermarks are to *benign* edits such as ours;
  - using watermark or fingerprint ideas to make our own study more rigorous: tamper-evidence, edit provenance,
    canaries.

**Out of scope, and must not be designed:**
- removing, forging or evading a third party's watermark or fingerprint;
- laundering model provenance.

If robustness findings imply an attack, report them only at the level needed for defence, such as "this class of edit
degrades scheme X", and recommend disclosure norms.

## What we have (cite files)

**Weight-space tooling.**
- R0/R1 access-tier features: weight-change norms, per-layer gradient alignment J1, K-FAC fields. Code lives in
  `src/wsp/wm/`.
- Edit banks: LoRA, task vectors, steering, scale and noise.
- Exact inert identity checks, canaries, and bitwise parity checks.
- Code-identity and bundle hashing (C1 v2o/v2p).

**Findings that may bear on this.**
- **Uniform rescales.** Gemma-3 post-norms cancel uniform writer rescales: doubling every writer changes NLL by 0.010,
  against 12.3 for norm-matched noise (`docs/reviews/wmm/wmm-gate0-2026-11-19.md`). Muse-30B behaves the same way,
  per Kocabay's Weight-Space Reading paper.
- **Determinism.** HF greedy generation was non-deterministic at batch 128 on H100
  (`docs/reviews/wme/wme-c1-launch-2026-11-20.md`). Separately, batch composition alone flips 13.5 % of greedy answers
  on MedGemma (`docs/reviews/wmm/wmm-gate1-2026-11-20.md`).
- **Edit-effect prediction.** Edit-specific effects are hard to predict beyond dose
  (`docs/council/2026-11-20/SYNTHESIS-r13.md`, including its addenda).
- **Weight readers.** P1, Weight-Space Reading (https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models):
  LLM readers given weight access detect edit direction above text-only.

**Models.** Qwen3.5-9B (frozen study C1, now running on Lambda; do not touch) and MedGemma 1.5 4B (study M′).

## Questions

1. **The literature.** What is the current state of:
   - weight-level LLM watermarking, for example schemes embedding bits in weights or quantisation;
   - model fingerprinting, both invariant-based (weight or representation invariants) and behavioural (trigger or
     instruction fingerprints);
   - lineage detection?

   Cite fetched URLs only, with each paper's own robustness claims. Which schemes are open-weight and testable on our
   models?
2. **Are our issues watermark-related?** Could any of our anomalies come from a watermark or fingerprint in Qwen3.5 or
   MedGemma? Candidates:
   - the post-norm cancellation;
   - batch non-determinism;
   - the headroom ceilings;
   - (a) non-identifiability.

   Is there any public statement that these models carry weight watermarks? Say plainly if not, and how one could test
   for it defensively.
3. **Can our tools detect or verify watermarks and fingerprints?** For example:
   - R1 features or invariants as lineage tests (base vs fine-tune vs merge);
   - our exact inert checks for tamper-evidence;
   - a WSR-style reader for "was this model edited, and where".

   Propose one concrete, defensible, powerable study on open models we can legally modify (Qwen3.5-0.8B/9B, Gemma 3,
   MedGemma). Give ground truth, baselines, metrics, power and cost. It must not extend into removal.
4. **Can watermarks or fingerprints improve our process?** For example:
   - fingerprinting every bank edit for provenance and tamper-evidence in study logs;
   - canary-style weight signatures to detect accidental edits or contamination;
   - measuring whether our benign edits degrade published watermarks: "watermark robustness under benign
     weight-space edits" as a descriptive side result.
5. **Ethics, licence and disclosure.** Consider the HAI-DEF prohibited-use policy for MedGemma, the Qwen licence and
   responsible-disclosure norms.

## Rules

- **Evidence:** fetched URLs only for external claims; record fetch failures. Repo files for our data. No invented
  numbers.
- **Format:**
  - at most 1,500 words;
  - an 8-line BLUF;
  - end with the table "decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback".
- **Output:** write ONLY `docs/council/2026-11-21/round14-<seat>.md` in
  `/home/ubuntu/projects/wsp-lab/.worktrees/council/2026-11-21-r14`.
- **Prohibited:**
  - commits, GPU use, model downloads;
  - writes under `data/`;
  - opening `data/evals` or sealed splits;
  - editing `docs/PREREG_*`;
  - touching running jobs, tmux sessions or Lambda;
  - secrets, IPs or hostnames;
  - designing watermark removal or evasion.
