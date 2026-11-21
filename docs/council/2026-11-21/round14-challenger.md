# Council Round 14, Challenger Report: The Case Against Watermark Speculation and Scope Creep

## BLUF

1. **Premise 1 rejected:** Our anomalies are not watermarks; they are known architectural, numerical, and experimental artifacts.
2. Post-norm cancellation is LayerNorm/RMSNorm scale invariance; HF non-determinism is batched floating-point non-associativity.
3. **Premise 2 rejected:** Our R0/R1 features are unbenchmarked against published, data-free spectral SVD baselines that already solve lineage.
4. Weight-space readers add no demonstrated provenance signal over exact cryptographic hashing (SHA-256) and bitwise canaries.
5. **Premise 3 rejected:** Launching a watermark side-study now is an unforced distraction while 9B C1 is blocked and M′ requires unblinding.
6. **Fatal omissions:** The proposal lacks ground truth on unwatermarked models, ignores a 13.5% batch-composition noise floor, and risks dual-use evasion.
7. Measuring watermark degradation under weight edits is dual-use scrubbing research, violating our defensive mandate and HAI-DEF terms.
8. **Recommendation:** Kill all watermark side-studies; restrict watermark methods strictly to internal passive audit hashing and bitwise canaries.

## 1. Challenge to Premise 1: Anomalies Are Not Watermarks (The Pareidolia of Ghost-Hunting)

Attributing our empirical hurdles to undocumented weight watermarks is pareidolia. Every anomaly cited in the council brief has an established, standard cause in Transformer architecture or GPU hardware:

1. **Post-norm cancellation is elementary scale invariance.** In `docs/reviews/wmm/wmm-gate0-2026-11-19.md`, doubling every `o_proj` and `down_proj` in MedGemma 1.5 4B changed NLL by only 0.010 nats/token, whereas norm-matched directional noise changed it by 12.3. This is not a watermark trapdoor; it is the mathematical definition of RMSNorm / post-normalization architectures. For any scalar $\alpha > 0$, $\text{RMSNorm}(\alpha W x) = \text{RMSNorm}(W x)$ because the RMS denominator scales linearly with $\alpha$, cancelling uniform weight rescaling up to floating-point precision. Kocabay et al. observed identical invariance in Muse-30B (*Weight-Space Reading*, https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models).
2. **Batch non-determinism is floating-point reduction non-associativity.** In `docs/reviews/wme/wme-c1-launch-2026-11-20.md`, pinned Hugging Face greedy generation at batch 128 on H100 SXM5 failed bitwise canary checks (0/5 reproducible). In `docs/reviews/wmm/wmm-gate1-2026-11-20.md`, batch composition alone flipped 13.5% of MedQA and 14.3% of MedMCQA greedy answers. This is a known consequence of non-associative floating-point addition in batched CUDA reduction kernels (FlashAttention and cuBLAS split-K GEMM), where padding shifts tensor alignment and dynamic thread block scheduling permutes summation order. In contrast, teacher-forced execution (8/8) and vLLM (5/5) were bitwise identical across runs and GPUs.
3. **Headroom ceilings reflect domain saturation.** In `docs/reviews/wmm/wmm-gate1-2026-11-20.md`, all 11 edits across 4 families yielded accuracy changes $\le 0$ pp (central doses −1.5 to −2.5 pp), replicating the 9B MATH headroom ceiling from Round 13 (`docs/council/2026-11-20/SYNTHESIS-r13.md`). Highly aligned models operate near their Pareto frontier; crude low-rank linear edits degrade formatting and calibration without opening secret accuracy headroom.
4. **(a) Non-identifiability is an experimental collinearity artifact.** In Round 13 (`docs/council/2026-11-20/SYNTHESIS-r13.md`), the 9B W2 bank varied along a single dimension: LoRA dose (log weight-change norm correlated with learning rate at $r = 0.996$). Direction was unidentifiable because the bank lacked directional diversity, not because a hidden watermark scrambled representation space.
5. **Absence of public evidence.** Neither Google nor Alibaba has ever disclosed weight watermarks in Gemma or Qwen. Google DeepMind's watermarking technology is SynthID-Text (Dathathri et al., *Nature* 2024, https://pmc.ncbi.nlm.nih.gov/articles/PMC11499265/), which operates strictly at generation time via tournament sampling on output logits, requiring no weight modification.

## 2. Challenge to Premise 2: Our Tools Add Nothing Over Existing Fingerprinting Baselines

The claim that our weight-space tooling (R0/R1 features: weight norms, $J_1$ gradient alignment, K-FAC fields) adds utility to watermark or fingerprint detection is unevidenced against published baselines:

1. **State-of-the-art white-box lineage is data-free.** Chen, Shang, and Liu (*COLM 2026*, https://arxiv.org/abs/2608.07786) demonstrated on over 110 open-weight LLM pairs that SVD spectral energy ($\Sigma$) perfectly discriminates independent model families, while subspace alignment ($U$) reliably resolves fine-grained lineage (SFT, RLHF, DPO, LoRA) directly from weight matrices. This requires zero forward passes or data samples. Proposing expensive $J_1$ alignment and K-FAC estimators without benchmarking against closed-form SVD subspace alignment is unjustified.
2. **Weight-level watermarking has specialized algebraic verifiers.** Invariant-based weight watermarks (arXiv:2507.08288, https://arxiv.org/abs/2507.08288) solve linear constraints directly over Transformer architectural invariants to verify keys. Quantization watermarking (arXiv:2310.11237, https://arxiv.org/abs/2310.11237) plants bits in INT8/INT4 rounding lattices. Our general edit-vector features are neither tailored to these algebraic keys nor competitive with closed-form verification.
3. **Behavioral fingerprinting is cheaper and non-invasive.** Active black-box fingerprinting such as LLMmap (Pasquini et al., *USENIX Security* / arXiv:2407.15847, https://arxiv.org/abs/2407.15847) identifies 42 LLM versions across system prompts and decoding parameters in as few as 8 queries with $>95\%$ accuracy. Embedding backdoors like EmbMarker (arXiv:2305.10036, https://arxiv.org/abs/2305.10036) achieve copyright verification via trigger sets.
4. **WSR readers lack fidelity.** Kocabay et al. (*Weight-Space Reading*) showed directional signal only for 27B self-reading on subtle edits (0.730 vs 0.578), with no magnitude calibration. In Gate 0 (`docs/reviews/wmm/wmm-gate0-2026-11-19.md`), post-norm rescales fooled naive readers. For provenance, exact cryptographic hashes (SHA-256 bundle hashing in C1 v2o/v2p) and bitwise canaries provide 100% mathematical certainty at zero GPU cost.

## 3. Challenge to Premise 3: A Side-Study Is Fatal Scope Creep During Confirmatory Execution

Launching an exploratory watermark side-study now violates research hygiene and endangers active commitments:

1. **The 9B confirmatory run (C1) is blocked at launch.** As documented in `docs/reviews/wme/wme-c1-launch-2026-11-20.md`, the C1 launch was halted before running any confirmatory unit because the pinned HF generation canary failed reproducibility (spending 3.812 GPU-h-eq, $18.26). This requires focused engineering repair—migrating E1 to the deterministic vLLM engine or validating `torch.use_deterministic_algorithms(True)`. Opening a third front before C1 unblinds is unacceptable execution drift.
2. **Study M is already in triage.** Gate 1 triggered a study-level KILL on accuracy claims (`docs/council/2026-11-21/RULING-wmm-gate1.md`). The chair approved a narrowed study M′ (M-SYC primary, M-A exploratory) constrained to local hardware ($\le 16\text{ GB}$, $0, 7–13 GPU-h) and subject to a strict power gate (power $\ge 0.5$, else killed). Siphoning local GPU cycles into watermark side-experiments threatens M′ execution.

## 4. Fatal Omissions: What Others Will Miss

1. **Missing Ground Truth.** Neither Qwen3.5 nor MedGemma contains planted weight watermarks. Searching for watermarks in models without ground-truth keys or payloads guarantees uncalibrated false positives. A valid detection study requires training synthetic watermarked controls, which we lack budget and mandate to perform.
2. **Unpowered Designs.** Gate 1 uncovered a 13.5% to 14.3% disagreement floor from batch composition alone (`docs/reviews/wmm/wmm-gate1-2026-11-20.md`). Any watermark detection or robustness metric relying on generated tokens will be completely underpowered unless batch composition is frozen and sample size exceeds the noise floor.
3. **Licence and Ethical Constraints.**
   - MedGemma is governed by Google's Health AI Developer Foundations (HAI-DEF) Prohibited Use Policy (https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy), which explicitly forbids attempting to "override, bypass, or circumvent safety filters" or deploying in unauthorized clinical environments.
   - Qwen2.5 weights are governed by Apache 2.0 (verified at https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/LICENSE; note fetch failure: https://raw.githubusercontent.com/QwenLM/Qwen2.5/main/LICENSE returned HTTP 404), while larger variants enforce 100M MAU commercial thresholds and restrict training competing foundational models.
4. **Dual-Use Evasion Risk.** The brief establishes a binding safety boundary: defensive detection and provenance only; no removal or evasion design. However, measuring "watermark robustness under benign weight edits" is mathematically isomorphic to an evasion benchmark. Determining the precise LoRA rank, task-vector dose, or noise norm that erases a watermark while preserving utility provides an exact recipe for laundering model provenance.

## 5. What Evidence Would Change My Mind

I will reverse my opposition if and only if three conditions are satisfied:
1. **Verifiable Ground Truth:** An official vendor disclosure from Google or Alibaba documenting an intentional weight-space watermark in Gemma 3 or Qwen 3.5, including public verification keys.
2. **Baseline Superiority:** A preregistered experiment on open synthetic controls (e.g., Llama-3 watermarked via arXiv:2507.08288) demonstrating that R1 features ($J_1$, K-FAC) achieve higher AUROC than data-free SVD subspace alignment (arXiv:2608.07786) at equal or lower compute.
3. **Unblocked Confirmatory Pipeline:** C1 9B confirmatory execution is resolved, unblinded, and reported, and Study M′ completes its preregistered gates.

## 6. Decision Table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| Reject watermark hypothesis for anomalies | Anomalies are explained by RMSNorm scale invariance, CUDA non-associativity, and domain saturation. | Zero new runs; formal documentation in technical notes. | 0 GPU-h, $0 | Fallback: Accept standard architectural and numerical baselines. |
| Reject weight-space watermark detection study | Our tools are unbenchmarked against data-free SVD spectral baselines; models lack ground-truth watermarks. | Synthetic Llama-3 ground-truth AUROC test vs SVD subspace alignment. | 0 GPU-h, $0 (defer until post-C1) | Kill: Do not charter study. |
| Adopt passive provenance only | Cryptographic bundle hashes (SHA-256) and bitwise canaries provide exact tamper evidence without model inference. | Bitwise parity on load/save cycles and C1 v2o/v2p hashes. | 0 GPU-h, $0 (CPU only) | Fallback: Use standard git/sha256 digests. |
| Prioritize C1 unblocking and Study M′ | Active confirmatory run is blocked; exploratory distractions threaten project delivery. | C1 vLLM migration gate; M′ power $\ge 0.5$ gate. | Prioritize allocated C1/M′ budget | Kill M′ if power < 0.5; repair C1 engine. |
