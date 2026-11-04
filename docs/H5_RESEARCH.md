# H5: Experiment 8 research note

Source: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), 2 November 2026, accessed 4 November 2026. Status labels below describe what the blog page itself establishes, not what might be recoverable from its released logs. The page links those logs and the edited checkpoint; inspect them before claiming exact replication.

## Experiment 8 protocol

- **Stated:** GPT-6 Astra is the editor; Qwen3.5-0.8B, thinking on, is the target. The goal is math improvement through direct weight editing, without training or gradient information.
- **Stated:** A notebook-like Python kernel has weights loaded, without a GPU. Astra can select practice prompts from GSM8K training (7,469) and MATH training (7,275), and receive model outputs through a forward-pass harness. The page says evaluation items “Astra never sees them” (Experiment 8, Setup). Exact tool signatures, output fields, package versions, prompts, model revision, and complete sandbox restrictions are **not stated** on the page.
- **Stated:** Separate budgets are 10, 25 and 50 forward passes; a pass answers up to 512 problems or reads up to 1,024 texts; average batch is 64 sequences, maximum 512. **Not stated:** whether partially filled calls, failures, intermediate token steps, retries or baseline measurements consume budget; exact accounting must be recovered from logs. **Inferred:** a pass here is a harness call over a batch, not one generated token or necessarily one example.
- **Stated:** Astra identified a loop direction and subtracted it from 64 MLP output columns in each of layers 15 and 19, changing 131,072 weights. **Not stated:** exact direction construction, column indices, normalization, coefficient, tensor names, precision, and whether the other budget arms use the identical edit. Recover these from released code/logs; do not substitute generic projected abliteration and call it exact.
- **Stated:** MATH-500, GSM8K and six other benchmarks: AMC23, Minerva Math, OlympiadBench, GPQA Diamond, IFEval and MMLU-Pro. MMLU-Pro uses a fixed 1,000 questions; other benchmark item counts are **not stated** on the page. MATH-500's name identifies its nominal size; exact item IDs and IFEval size are **not stated**. Test sets were withheld from Astra; the page does not specify the technical sealing mechanism or contamination checks.
- **Stated:** Evaluation uses up to 32,768 generated tokens, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5. Samples per problem: AMC23 16; MATH-500, Minerva Math, OlympiadBench and GPQA Diamond 4; GSM8K and IFEval 2; MMLU-Pro 1. **Not stated:** random seeds, independent editor episodes, uncertainty intervals, item ordering or exact aggregation/checker implementation.
- **Stated:** The loop metric in the table is the percentage of MATH-500 answers hitting the token limit, not a repetition detector.

| Qwen arm | MATH-500 % | GSM8K % | IFEval % | MATH-500 limit % |
|---|---:|---:|---:|---:|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| 10 passes | 12.3 | 25.7 | 49.0 | 87.2 |
| 25 passes | 14.1 | 25.1 | 39.9 | 86.4 |
| 50 passes | 48.1 | 67.1 | 62.3 | 33.1 |

**Stated:** all table values are from Experiment 8; 85 of the first 96 practice answers hit the limit. The 50-pass edit changes 0.016% of weights. Scores alone do not establish reproducibility across editor runs.

**Stated:** In the separate Gemma 3 270M instruction-tuned experiment, 8.5% of answers hit the limit. The reported MATH-500 scores were 6.8 base and 7.0, 7.6, 7.7, 7.0 at 5, 10, 25, 50 passes respectively. Four samples per problem; temperature 1.0, top-k 64, top-p 0.95, maximum 8,192 tokens. Astra found arithmetic and algebra errors without a single component to switch off; its small targeted edits did not establish an improvement. This is the authors' interpretation, not proof that improvement is impossible. Seeds and exact edit details are **not stated**.

## Implementation gap

| Protocol item | Existing code | Missing for match |
|---|---|---|
| Editor and sandbox | `zoo/build.py` builds random, abliteration, LoRA, task-vector and chain families | LLM editor, isolated Python session, controlled weight access and transcript |
| Forward budget | `eval/run.py` and `batch.py` execute evaluation; batch generation is configurable | Metered diagnostic API with 512/1,024 limits, 10/25/50 budgets and audit log; define unresolved accounting from logs |
| Direct edit | `edits/kernels.py` implements directional refusal projection and layer profiles | Exact identified direction and selected-column subtraction; log delta and verify 131,072 weights |
| Evaluation | `eval/run.py` supports seeded generation and benchmark records; `batch.py` batches zoo evaluations | Full eight benchmark suite, exact item IDs, checkers, sample counts and MATH-only limit denominator |
| Generation | `eval_fast.yaml` has thinking on and matching sampling parameters | 32,768-token run and per-benchmark repeated samples; fast config uses 1,024 tokens |
| Sets and isolation | Fast config uses GSM8K 50, MATH-500 25, IFEval 25; `eval_subset_d3.yaml` samples zoo entries by family, not test questions | Full benchmark sets, distinct accessible training examples, enforce sealed evaluation and item provenance |
| Repeats | Seed handling exists in evaluator | Independent editor episodes and seed policy; blog's exact seeds not stated |
| Gemma control | Generic zoo/eval framework | Verify architecture support, model-specific sampling and exact control protocol |

## Related work

Titles below were checked on 4 November 2026 using `curl -s https://arxiv.org/abs/<id> | grep -o '<title>[^<]*'`. This is a verified adjacent reference, not a claim of prior direct weight editing:

- [arXiv:2608.24735](https://arxiv.org/abs/2608.24735), Zae Myung Kim, Young-Jun Lee, Seungyeon Jwa and Dongyeop Kang, *Meta^n: Recursive Self-Improvement through Emergent Depth*. Returned title: `<title>[2608.24735] Meta$^n$: Recursive Self-Improvement through Emergent Depth`. Recursive agents improve code and prompts through an archive; its best-of-N issue motivates an equal-budget selection control, but it does not edit model weights. See `docs/MODEL_CANDIDATES.md` Note: Meta^n.

Other candidate references require verification of both identity and relevance before inclusion; this short list is not an exhaustive literature survey.

## Reproduction risks

1. **Measurement noise:** `docs/RESULTS.md`, Interim D3, reports same-seed cross-host base MATH-500 0.16 versus 0.28 at n=25 and edited-entry gaps up to 0.16. Pair against same-host base and measure test-retest before attributing small gains to edits.
2. **Insufficient evaluation:** fast MATH-500 uses 25 items and 1,024 tokens, whereas the blog uses four samples per MATH-500 problem and 32,768 tokens. It cannot validate the reported full-benchmark result or loop rate.
3. **Leakage and adaptive selection:** the blog says Astra never sees evaluation items; giving an editor or critic test prompts or per-item results during search breaks that condition. Enforce separate training items and sealed final evaluation; match real-evaluation budgets and include best-of-N controls as specified in `docs/MODEL_CANDIDATES.md`.
