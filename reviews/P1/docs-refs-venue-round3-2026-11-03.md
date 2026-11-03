# Reference ledger round 3 — 2026-11-03

Result: **3 claims restored / 6 kept weak; verify exit 0** (33 entries).
Restored means stronger, source-bounded detail, not reinstatement of an unlimited assertion. All nine decisions retain the round-2 claim under `history:`. Existing correction records retain the older rejected wording.

| Claim | Decision and evidence | Paper dependency |
| --- | --- | --- |
| W2T | Restored two-stage architecture and numerical benchmark: §4 rank/position Transformers and Table 2 ARC-LoRA test MAE 0.32, Spearman 95.35 (%). Added excerpts 2–4; table extraction is contiguous, including both validation/test rows. | H1 precedent; no extrapolation to our tasks. |
| JEPAWG physics | Kept weak. Added latent-16 and coupling-to-weight prediction passages (§4) as excerpts 2–3. Adam 3e-4 in Appendix B describes conditional-flow pretraining with reverse KL, not JEPAWG's VICReg training; it cannot support the old combined implementation claim. | H3 needs covariance omission and predictive embeddings, not that optimizer. Couplings are not our edit descriptors. |
| Leap+Verify | Kept weak. Added reported finite-difference acceptance rates (24%/37%) as excerpt 2; these do not demonstrate net training acceleration. | H2 motivation only; no measured speedup claim. |
| JEPA/HMM | Kept weak. Added the explicit Markov/emission/marginal/filtering consistency conditions as excerpt 2. Theorem 1 says these are sufficient, not necessary, and not guaranteed by standard JEPA training. Unconditional equivalence remains unsupported. | H2/H3 conditional analogy only. |
| 320B MoE single-direction bypass | Kept weak. Existing excerpts 1–2 identify the architecture and refusal surviving every tested edit/rank 1–12 (§conclusion). They do not establish a complete bypass. | H1 architecture-dependent limits, not guaranteed removal. |
| Gradient-free contrastive steering | Kept weak. Existing abstract excerpt explicitly requires two small fine-tunes; subsequent arithmetic does not make their construction gradient-free. | H1/H3 may consume such edits; gradient-free optimization is a non-goal. |
| Harness² multi-agent evaluation | Kept weak. Existing excerpts support task/harnessing recursion. Subagent harness files appear in the source, but do not establish reliable multi-agent evaluation. | Orchestration motivation only; no H1–H4 validity claim depends on this. |
| SRA collateral damage | Restored a bounded empirical claim using abstract excerpt 2: five models, 0–2% refusal, mean perplexity change ≈+0.02 and first-token KL ≈0.025. These are measured proxies, not universal absence of capability degradation. | Alignment sub-study motivates measuring trade-offs; does not assume isolated capability circuits. |
| Zoo size | Restored size plus diversity result: §4 reports significantly improved single-zoo performance with more samples and a large generation-performance effect (excerpt 2). Figure 3 discussion cautions that count alone saturates and diversity can undersample domains. | H1/H3 curation rationale; no monotonic scaling law or generic representation-quality guarantee. |

`docs/RELATED_WORK_MAP.md` records the H1–H4 dependency audit and removes reliance on all six unsupported stronger assertions. No hypothesis thresholds were changed.

Verifier now discovers every top-level `supporting_excerpt_N` field and applies the existing normalization, contiguous substring, ≤60-word and no-ellipsis rules to each, including empty supplied excerpts. Existing metadata and primary-excerpt checks remain. `source scripts/env.sh && python scripts/verify_refs.py` exited 0 for all 33 entries. This verifies the local text and API metadata checks, not independent PDF transcription fidelity.
