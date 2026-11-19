# Council brief, round 12, 19 Nov 2026: MedGemma as a second target model

## The user's request (paraphrased)

Perhaps MedGemma could also be used:
- to see whether we can predict the effect of its enhancements;
- to see whether it can be enhanced at all.

The council should check this.

## Standing goal (unchanged)

The goal is a paper on a popular open-weight LLM. It has two parts:

- **(a) Trajectory prediction:** a world model predicts how benchmark behaviour changes under weight edits, without
  running the edits.
- **(b) Enhancement:** choosing edits by the world model's predictions measurably improves the model.

## State (cite files; read them)

**Frozen protocol.** `docs/PREREG_WME.md` is FROZEN (tag `prereg-wme-v2-frozen`, commit `af4face`). Its target is
Qwen/Qwen3.5-9B. The 9B confirmatory run (track C1, branch `run/wme-c1-9b`) is built and waiting for Lambda
H100 capacity. It must not be disturbed or contaminated.

**Claims as frozen.**

| Class | Claim |
|---|---|
| Primary | H-E1: sycophancy on TriviaQA generation, N = 1000 |
| Primary | H-E2: WM-guided vs StatNN-guided search, family-stratified sign-flip, 24 episodes |
| Secondary | S-BoN (unpowered) and S-TE (token efficiency) |
| Negative result | (a) trajectory prediction |

For (a):
- Stage W: `docs/reviews/wme/wme-stagew-2026-11-18.md`.
- Stage W2: `docs/reviews/wme/wme-stagew2-2026-11-18.md`. A pooled same-path-type curve ties or beats the WM.

**Headroom.** Generated accuracy on MMLU-Pro and SuperGPQA did not improve once truncation was matched. H1 gave
budget relief only. See `docs/reviews/wme/wme-pilot3-2026-11-17.md` and
`docs/council/2026-11-17/SYNTHESIS-r11.md`.

**Infrastructure.**
- Local GPU: one RTX PRO 4500 Blackwell (32 GB) on this host. It is shared with other sessions.
- Lambda caps: 90 GPU-h-eq and $1,000 for the 9B run. Lambda currently has zero H100 or B200 capacity.
- Code: `src/wsp/wm/`. It was written for Qwen3.5 (hybrid attention). The 0.8B Qwen zoo is in `data/zoo`
  (read-only). CUDA isolation per track: `scripts/venv_track.sh`.

## Questions

1. **What is MedGemma now?**
   - Variants, sizes, architecture (Gemma 3 base?), text vs multimodal.
   - Licence: Health AI Developer Foundations terms. Do they allow weight editing, publishing edited-model results,
     and benchmark use?
   - Published model-card benchmark numbers.
   - Cite fetched URLs only.
2. **Scientific value.** Is MedGemma:
   - (i) a second-model replication of the frozen 9B claims;
   - (ii) a new domain study (medical QA headroom; medical-safety guards); or
   - (iii) not worth it?

   Does a domain-specialised model raise the chance that (b) succeeds? Consider measurable headroom on MedQA,
   MedMCQA, PubMedQA, MMLU-medical and MedXpertQA, and contamination risk. Does it rescue (a), or only repeat the
   negative?
3. **Design.** Give the minimal powered design.
   - Edit families and objectives.
   - Guards: medical harm, over-refusal and calibration.
   - Sealed panels, power, kill gates.
   - What can be reused from WM-E unchanged, and what needs porting (Gemma architecture, tokenizer, chat template)?
4. **Sequencing and isolation.**
   - It must not touch the frozen 9B study or its panels. It needs its own pre-registration.
   - Can it run on the local 32 GB GPU (4B) while Lambda is out of capacity?
   - Give GPU-h and $ costs, and the order relative to C1 and the planned Qwen3.8-27B replication.
5. **Kill criteria.** Under which pilot outcomes do we drop it?

## Rules

- **Evidence.** No invented numbers. Cite repository files for our data and fetched URLs for every external
  claim. Record fetch failures.
- **Format.** At most 1,500 words. Open with an 8-line BLUF. End with the table
  "decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback".
- **Output.** Write only `docs/council/2026-11-19/round12-<seat>.md` in the worktree
  `/home/ubuntu/projects/wsp-lab/.worktrees/council/2026-11-19-r12`.
- **Prohibited.**
  - commits;
  - GPU use;
  - model downloads;
  - writes under `data/`;
  - opening `data/evals` or sealed splits;
  - editing `docs/PREREG_*`;
  - touching running jobs, tmux sessions or Lambda instances;
  - secrets, IPs or hostnames.
