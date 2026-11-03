# Board A review — feat/p2-eval

- **Task:** feat/p2-eval (P2 cheap eval battery)
- **Head:** 5d5033f0147723d09ea9d2784e6a3809db36df00
- **Reviewer:** reviewer-ml (Board A)
- **Date:** 2026-11-03
- **Verdict:** APPROVE
- **Blocking:** no

## Scope checked

1. Answer extraction (GSM8K/MATH-500): `\boxed{}` balanced-brace extraction, `####` final-number
   fallback, thinking-tag stripping.
2. IFEval checker provenance and licence header.
3. MMLU-Pro / HellaSwag log-likelihood scoring (length normalisation, choice formatting, leakage).
4. Subset determinism and whether the regen test is real.
5. refusal.py Heretic 2.0 defaults + KL-to-base tokenisation.
6. manifest.json fields and item-only aggregation.
7. Sampling determinism at temperature 0.6 and batch-size invariance.
8. Secrets/IPs/raw logs in the diff.

## Test runs

- `source scripts/env.sh && pytest -q` → **31 passed, 2 deselected, 1 warning in 9.32 s** (CPU tier).
  `PYTHONPATH` resolved to the worktree `src` (env.sh now computes `WSP_ROOT` from the script path).
- GPU tier was not re-run (reproduction job owns the GPU); the eight-item deterministic repeat was
  already recorded green (33.17 s) in board/P2.md.
- Reproduction job `p2-reproduce` is live (tmux), manifest `base-final-4096` status `running`,
  log `~agent-logs/p2-reproduce-20261103-232354.log` currently at `generated 64/200` for GSM8K. No
  new GPU job started.

## Findings

### Blocking — none

### Minor (non-blocking)

- **M1 — sampling is deterministic per config but not batch-size invariant.**
  `src/wsp/eval/run.py:159` calls `seed_all(cfg["seed"])` once at the top of `generate()` and the
  batch loop (`run.py:162`) then consumes one global RNG stream per batch. Changing `batch_size`
  re-partitions items across batches, shifting RNG consumption, so per-item samples differ.
  - Failure scenario: the 4096-token run (`batch_size 64`) and the 32768-token run (`batch_size 32`)
    use different batch sizes, so their loop-rate/accuracy comparison is confounded by batch size in
    addition to token budget. The report's caveat (`report.py:80-87`) already flags that a budget
    comparison alone does not isolate effects.
  - Within a fixed config (same seed, same batch size) the run is deterministic — the GPU smoke test
    and `test_gsm8k_eight_items_deterministic` (`tests/test_eval_gpu.py:9`) confirm this, which is all
    the P2 gate ("reproduce within ±1.5 pt across two runs") requires.
  - Recommendation (not required now): seed per item (e.g. sub-seed `seed ^ item_id`) so batch size
    does not change per-item samples, and add a two-batch-size invariance assertion.

- **M2 — base model revision and WikiText-2 are not pinned.**
  `src/wsp/eval/run.py:130` falls back to `revision="main"` when `model.config._commit_hash` is
  absent, and the running manifest records `model_revision: "main"` (confirmed in
  `data/runs/p2/base-final-4096/manifest.json`). `src/wsp/eval/run.py:302` loads
  `Salesforce/wikitext` with no revision. The five fixed-subset datasets *are* pinned
  (`src/wsp/eval/subsets.py:18-24` REVISIONS) and their fingerprints captured.
  - Failure scenario: a mutable `main` tag for the base model or WikiText-2 means two "same-seed"
    runs weeks apart could read different weights/text and fail the ±1.5 pt reproduction gate through
    no fault of the code.
  - Recommendation: pin `Qwen/Qwen3.5-0.8B` and `Salesforce/wikitext` to a commit hash and record it.

- **M3 — HellaSwag length normalisation is by characters, not tokens.**
  `src/wsp/eval/run.py:207` divides the continuation log-likelihood by `len(choice)` (Python string
  length) when `normalize_length` is set, which is how HellaSwag is invoked (`run.py:284`). Standard
  `acc_norm` normalises by token (or byte) count.
  - Failure scenario: two endings of equal token count but different character count are weighted
    slightly differently than the reference HellaSwag metric, which can reorder close choices.
  - This is a defensible approximation and is locked in by
    `test_choice_likelihood_only_scores_continuation_and_normalizes_characters`
    (`tests/test_eval.py:96`), so it is deterministic and intentional, not a silent bug.

- **M4 — dead config keys.**
  `configs/eval_cheap.yaml:7` sets `mmlu_pro: {..., max_new_tokens: 1024}` but MMLU-Pro is scored by
  log-likelihood (`run.py:271-285`), never generation, so the cap is ignored. `configs/eval_faithful.yaml`
  and `configs/eval_cheap.yaml` declare `loop_rate.from`, but `aggregate()` uses a hardcoded benchmark
  tuple (`src/wsp/schema.py:105`) and ignores it. Cosmetic; no effect on results.

- **M5 — MATH-500 exact-match is inherently strict.**
  `src/wsp/eval/run.py:48` normalises `\frac` only for a single level of nesting; answers like
  `\left(3, \frac{\pi}{2}\right)` round-trip only because both sides pass through the same `normalize()`
  after `$`/`\left`/`\right`/space removal. This is the standard fragility of MATH-500 exact match, not
  a regression, and the blog's ~7.9% reference is consistent with a similarly strict metric.

## Checks passed (summary)

- **(1) Answer extraction** — `final_text` (`run.py:24`) strips the Qwen `<think>…</think>` block;
  `answer` (`run.py:28`) does balanced-brace `\boxed{}` extraction, then `####`, then last-number
  regex. Truncated generations are marked `hit_max_tokens` only when `len==max_new_tokens` and the
  final token is not EOS (`run.py:191`), and `aggregate()` (`schema.py:105`) counts those hits into
  `loop_rate` while the item is scored wrong by `exact_match`. Correct.
- **(2) IFEval provenance** — Apache-2.0 Google Research headers retained in all four vendored files
  (`src/wsp/eval/ifeval/instructions*.py`, `utils.py`); `README.md` documents lm-eval 0.4.13 →
  google-research lineage and the local import/format changes. Clean.
- **(3) LL scoring** — `choice_scores` (`run.py:197`) scores only continuation tokens
  (`logits[len(prefix)-1:-1]` vs `ids[len(prefix):]`), so no label is exposed in the context; MMLU-Pro
  compares argmax index to `answer_index` (verified 0-based) and HellaSwag compares to `label`. No leakage.
- **(4) Subsets** — `subset_ids` (`subsets.py:28`) is deterministic (`random.Random(0).sample`), and
  `test_committed_subsets_regenerate` (`tests/test_eval.py:12`) actually recomputes and compares against
  the committed files. Real test, not a tautology.
- **(5) Refusal/KL** — `defaults()` (`refusal.py:15`) reads Heretic's own
  `heretic.scorers.keyword_rate.Settings` / `kl_divergence.Settings` (confirmed heretic-llm 2.0.0):
  `mlabonne/harmful_behaviors` for `refusal_rate`, `mlabonne/harmless_alpaca` for `over_refusal` and
  `kl_to_base`, with Heretic's default keyword markers. KL is computed on identical tokenised prompts:
  both `base.model(ids)` and `runner.model(ids)` run on the same `prefix + continuation` ids
  (`refusal.py:71-82`), and `kl_means` (`refusal.py:30`) is `KL(base || edited)` over continuation
  positions only. Correct.
- **(6) Manifest / aggregation** — manifest carries `seed`, `config_hash`, `git_sha`, `model`,
  `model_revision`, `generation`, `source_hash`, `run_id`, plus dataset fingerprints/revisions
  (`run.py:349-362`, `write_results`). Aggregates come only from per-item `EvalRecord.correct` /
  `hit_max_tokens` (`schema.py:91-108`); `loop_rate`, perplexity (token-weighted geometric mean), and
  KL means are all derived from items. No runner-provided aggregate is trusted.
- **(7) Determinism** — fixed-config reproducibility confirmed; batch-size invariance absent (see M1).
- **(8) Secrets/IPs** — no keys, tokens, IP addresses, or raw logs in the diff; the only
  "secret/token" hits are IFEval keyword lists and tokenizer terminology.

## Verdict

**APPROVE** — no blocking findings. All eight correctness checks pass; the minor items (batch-size
non-invariance, unpinned model/WikiText revision, char-length HellaSwag normalisation, dead config
keys) are reproducibility/robustness notes rather than leakage or metric bugs, and none block the P2
gate. Recommended follow-ups are listed under M1–M2 before results are quoted in the paper.
