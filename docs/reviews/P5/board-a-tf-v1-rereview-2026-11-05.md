# Board A re-review: TF-v1 teacher-forced tier (`feat/tf-v1` @ 9013d76)

5 November 2026. Reviewer: Board A (strict ML). **Verdict: ACCEPT WITH ONE TESTS-ONLY FIX. Merge yes, once the fix below lands. Gate 1 passes PREREG §9's rule, but the programme fits the 19.0 h cap only with about zero slack.** No endpoint is qualified, and this review authorises no compute. I reviewed `.worktrees/feat/tf-v1` without modifying it, against [my round-1 review](board-a-tf-v1-2026-11-05.md) (four blocking items) and [PREREG_H1_TF §§1–6, 9](../../PREREG_H1_TF.md). I used CPU only (`. scripts/env.sh`, `agent-tmux boardA-tfv1-r2*`). Scratch copies are under `/tmp/boardA_rep` and the pilot data were not touched.

Scope note: the branch has moved to `68a6455` ("tf_v1_pilot_report: ruff fix"). It changes only `scripts/tf_v1_pilot_report.py` and the README status stamp, and `src/` is byte-identical to 9013d76 and to the pilot's committed source `02c2847`. I reviewed 9013d76 as requested and checked that 68a6455 adds nothing else.

## 0. What I ran and observed

| Check | Result |
|---|---|
| Full CPU suite (`CUDA_VISIBLE_DEVICES= pytest -q`) in the worktree | **542 passed, 1 skipped, 8 deselected, 59 s** (round 1: 526). |
| `tests/test_tf_v1.py` + `tests/test_eval_board_a.py` alone | 29 passed. |
| Same suite with an empty `HOME` (what the GitLab docker runner has, since `~/.config/wsp/device_id` is absent) | **3 failed, 539 passed.** The failures are `tests/test_backfill_eval_metadata.py` (`test_cli_dry_run_and_apply_preserve_scores_and_link_zoo`, `test_current_environment_backfill_is_inferred_and_archived`, `test_historical_control_registration_keeps_source_and_original`), all with `FileNotFoundError: …/device_id is required`. The same file on a `main` snapshot with empty `HOME` gives 8 passed. See §3. |
| Mutation checks (scratch copies) | Reversing the KL direction in `kl_values` makes `test_kl_is_base_to_edited_with_hand_computed_asymmetric_example` fail. Removing the single-token guard in `letter_token_ids` makes `test_letter_scoring_stops_when_a_letter_is_not_a_single_token` fail. The tests do bite. |
| `source_hash` | `provenance()` recomputed at the worktree gives `12ee875a68c9`, equal to every pilot `eval.json`. Pilot `git_sha` is `02c2847`, `dirty: false`. `git diff 02c2847 9013d76 -- src` is empty. |
| Pilot report | `scripts/tf_v1_pilot_report.py` run on a copy of the artefacts (`--load-probe load-probe --load-probe-id b6fe7f59016cdda2`) **reproduces `pilot_report.json` exactly** and the RESULTS.md "round 2" block except for a leading blank line. Table, 1.805 min median, 6.31 min cold base, 47.1 s load+verify and 2.36 MiB/entry all come from the artefacts. |
| Gross wall time | The `eval.json` mtimes of the 10 warm entries span 1,084.3 s, which is 1.807 min/entry. This equals the recorded per-entry wall seconds, so nothing sits outside `wall_seconds` between entries. |

## 1. The four blocking items

| # | Round-1 item | Status | Evidence |
|---|---|---|---|
| 1 | Letter fallback | **Closed.** `letter_token_ids` (`tf_v1.py:253–269`) raises `LetterTokenizationError` unless every letter is one token and all are distinct; `letter_scores` has no fallback. `record_stop` writes `stops/stop_<id>.json`. | Tests: not-single-token (reason payload and durable stop file), duplicate tokens, full-vocabulary log-softmax with ids returned, and `choice_scores` replaced by a function that raises. Real tokenizer: A–J → 357, 417, 351, 414, 458, 426, 469, 462, 353, 604, all distinct. Every MMLU record carries `letter_token_ids`, and the manifest carries them per option count (3–10). |
| 2 | Refusal first token | **Closed.** Phrases are scored over all tokens, teacher-forced; `continuation_scores` takes the `len(ids)` trailing logit rows of `prefix + ids[:-1]`, which is aligned correctly. The margin is `logsumexp(refusal) − logsumexp(compliance)`. `build_continuations` demands nonempty, disjoint sets and rejects any phrase whose token ids are a prefix of another. | Tests: scripted logits with hand-computed `P(I)·P(cannot\|I)` for "I cannot", margin equals the log-sum-exp expectation, and the validity cases. Real Qwen tokenization of the 13 configured phrases validates (for example "I cannot" = [40, 4021] and "I'd be happy to" = [40, 4035, 381, 6051, 310] share only the first token and neither is a prefix of the other). KL (ranks 0–199) and over-refusal (200–249) prompts are disjoint by construction. The margins are still uncalibrated: the config and DECISIONS call them non-reportable, but the data artefacts (`summary.json`, `tf_v1.refusal_rate.mean_phrase_margin`) carry no such flag (see §6). |
| 3 | Nested ranking, panel label, device id | **Closed.** `rank_ids` = sort by `sha256(item_id + salt)` with integer-id tiebreak; the committed files (`*_tf_v1.txt`) carry salt, status and pool in a header; `panel_ids` re-derives the ranking and fails closed on a salt or ranking mismatch or on `n` above the file length. `panel: development` is in config, headers, every record's `extra.panel`, manifest and eval.json. `runtime_metadata` raises `FileNotFoundError` without the device-id file and `ValueError` for a non-`H<number>` id. | Nested-prefix tests (ranking order-independence, `load_items` n=250 vs 500, prompt panels, oversized `n`, wrong salt, KL/over-refusal disjoint, `n=201` overlap refused). I checked directly: the committed 1,000 MMLU-Pro and 500 HellaSwag files equal `rank_ids(eligible_ids(...))[:size]`, share no id with the 300 legacy items, and the item ids in **all 12 pilot entries** (11 + the repeat) equal the committed prefixes (HellaSwag first 250, MMLU 1,000, KL 0–199, over-refusal 200–249). `panel_content_hash` is identical in all 12 manifests (MMLU `d671401e…`, HellaSwag `a395eaad…`). |
| 4 | Cache | **Closed.** Per-process residency with one content verification: `_RESIDENT` holds one cache, `load_verified` checks the file set against both the manifest and the directory, checks per-file content digests (raw int64 ids + bf16 bits viewed as int16, not `torch.save` bytes) and the root. The key is `identity` = runtime metadata + `SCORER_HASH` (source of `last_logits`, `generate_entry`, `entry_digest`, `content_root`) + prompts, prefix-id hash, dataset, generation and `max_new_tokens`; the full-tree `source_hash` and config hash are gone from the key. The manifest is written atomically (tmp then replace). The size cap is enforced during the build (6.0 GiB; worst case 200 × 64 × 496,640 B = 5.92 GiB). The round-1 duplicate caches were removed and `tf_v1_round1_archive/cache_alias.json` records the alias; one cache directory now exists (5.4 GB on disk). | Tests: build once and reuse under changed `source_hash`/`git_sha`/`dirty`; corruption detected after a new-process reload; five entries in one process do one `torch.load`; stray file and renamed manifest entry refused; atomic-replace path; over-cap abort; KL records carry the root, per-position KL and direction. **Independent check:** I reloaded all 200 `.pt` files on CPU, recomputed `entry_digest` per file and the root: every file digest equals the manifest and the root is `2ed18bda284b3a89…` (bf16, vocabulary 248,320, 11,641 positions, mean 58.2 tokens per prompt, `scorer_hash` in the manifest equals the current source). The root appears in **2,400 of 2,400 KL records** (12 entries × 200), in all 12 `eval.json` `tf_cache` blocks and in all 12 manifest `datasets.kl_to_base`. |

The earlier non-blocking requests are also done: an asymmetric, hand-computed KL direction test (below), a below-length `n` prefix test, a `tier: tf_v1` field, and the development label.

**KL direction test (hand-computed).** Base p = [[.75, .25], [.9, .1]], edited q = [[.5, .5], [.6, .4]]. By hand: KL(p‖q) = .75 ln1.5 + .25 ln.5 = 0.130812 and .9 ln1.5 + .1 ln.25 = 0.226290; the reverse KL(q‖p) is 0.143841 at position 0. I recomputed these three numbers independently. The test asserts `kl_means` and `kl_values` equal the forward values to 1e-6, that the forward and reverse values differ by more than 1e-2, and that swapping the arguments returns the reverse value. The mutation above confirms it fails on a reversed implementation.

## 2. Pilot data checks

- The pilot is one process on the committed source, with a clean tree. Base is entry 1; ten D3 controls follow; the repeat in a separate process is `repeats/load-probe`.
- Max |ΔKL| and max |Δgold log-prob| of the repeat versus its primary are 0; the script's `records_identical` is True (reproduced by me). The round-1 observation that two independent greedy builds produced identical caches still stands. This is one entry, not the two-entry G1 test.
- Descriptive panel diagnostics from the records (paired item/prompt bootstrap, 4,000 draws, seed 20261105; not the registered procedure):

| Endpoint | Median across the 10 entries | Max (entry) | Registered bound |
|---|---:|---:|---:|
| MMLU gold log-prob 95% half-width, 1,000 items | 0.0040 (mean 0.0114) | 0.0436 (`af6d608f…`) | δ/2 = 0.05 |
| KL 95% half-width, 200 prompts × 64 tokens | 0.00009 | **0.0170** (`af6d608f…`, KL 0.323) | δ/2 = 0.005 |
| HellaSwag 250 half-width (descriptive) | 0.0003 | 0.0033 | none |

  The mean 0.0114 matches the "0.0112" in my round-1 table; I labelled that figure a median there, which was wrong. The median is 0.0040. Two caveats matter for the registration. (i) The MMLU maximum is 87% of δ/2. (ii) For KL, nine entries are at or below 0.0018, but the 0.32-KL entry is 3.4 times δ/2 (about 5% of its own KL). The "δ/2 applies to the typical or panel-median entry versus every entry" question I raised in round 1 therefore remains unregistered and has a concrete failing case. Cutting 128 to 64 tokens raised that entry's half-width relative to my round-1 figure (0.013 at 128 tokens, a different prompt set and entry mix, so only indicative). I had asked for a single 200×128 run recording prefix lengths 16/32/64/128; this pilot used 64 directly, so the token cap is not tested against 128. The recorded prefix medians (16: 0.00144, 32: 0.00153, 64: 0.00160) are descriptive only.

## 3. Legacy fast-tier records and resume keys

Unchanged:
- `resume_key`, `payload_key`, `complete`, and the `skip_key`, `signature` and `reuse_key` construction in `batch.py` have no diff. Extra top-level fields in a new `eval.json` do not enter the key (`complete` compares `payload_key(payload)` with the signature).
- `records.jsonl` for the fast tier: `write_results` and `evaluate` only branch on `generation["tier"] == "tf_v1"`. `kl_means` was refactored to call `kl_values` and has identical arithmetic (first and mean of the same float32 expression). `perplexity` passes the dataset revision and adds record fields only when the config has `revision`; `configs/eval_fast.yaml` has none.
- The TF output directory is `data/evals/<zoo>/tf_v1`, separate from the fast-tier directories.

Changed, and worth stating:
1. **Regression (merge condition).** `runtime_metadata` now fails closed when `~/.config/wsp/device_id` is missing. This is the right behaviour for G3 identity, but it also applies to the fast tier, and three existing tests in `tests/test_backfill_eval_metadata.py` rely on the old silent default of `H0`. They pass here only because `~/.config/wsp/device_id` exists on this host. The GitLab docker runner's `HOME` has no such file unless the runner mounts it; I could not read `/etc/gitlab-runner/config.toml`. `main` runs `test-cpu` on CI. Fix required before merge: a tests-only change, for example an autouse fixture in `tests/conftest.py` that points `HOME` to a temporary directory containing a `device_id` of `H0`, plus the existing explicit test for the missing-file case. It changes no `src/` file, so `source_hash` and the pilot evidence are unaffected. I did not verify the fix.
2. `run_batch` now also writes `wall_seconds`, `model_load_seconds`, `stage_seconds`, `peak_gpu_*` and `metadata_provenance: "native-measured"` into every new `eval.json`, including fast-tier runs (the `tier`/`panel`/`tf_cache` fields are TF-only). No existing record changes. A new fast-tier run will be labelled native-measured, which is accurate for runs that read the device id; just be aware the label is no longer TF-specific.

## 4. Ruff

CI lints `ruff check src tests` and `ruff format --check src tests`. Both pass at 9013d76 (103 files formatted). The one fixable item the implementer notes is F541 (an `f` prefix on a string without placeholders) at `scripts/tf_v1_pilot_report.py:161`, plus one line that `ruff format` would rewrap in the same script. **It does not block**: the script is outside CI's lint scope, and 68a6455 fixes both (my scoped check of the eight changed Python files is clean at 68a6455). Running `ruff check .` over the whole archive reports 53 other findings in unrelated files (for example paper and analysis scripts), which are not part of this branch and not linted by CI.

## 5. Gate 1 and cost projection

Measured (10 warm entries, same process, cache resident): median **1.805 min**, range 1.781–1.833, every entry under 2.0. Stage medians: MMLU 34.6 s, HellaSwag 36.7 s, KL 7.7 s, WikiText 4.3 s, refusal 12.0 s, over-refusal 12.2 s. Cold base: 6.31 min including a 261 s cache build. New process: 47.1 s to load and verify, plus 7.6 s model load (the probe entry took 2.60 min against 1.81 in-process). Storage 2.36 MiB/entry. Peak GPU allocation 9.22 GiB (10.2 GiB reserved), so two concurrent processes fit on one GPU. The source on which this was measured is the merge candidate's `src/` (exact `source_hash`).

**Gate 1 against PREREG §9: PASS.** The rule is "warm median ≤ 2 min is necessary but not sufficient", and the 3.0 min revision line is far away. The margin I asked for was ≤ 1.8 min, because 2.07 min already uses the whole cap. 1.805 misses it by 0.005 min (0.3%, inside the 0.052 min range), so I treat the margin as met within noise and do not call it a fail. The sufficiency clause is the binding one.

Cost projection for the ~552 scoring runs (single device, one stream; the 552 count is nominal and the closure count governs):

| Item | Hours |
|---|---:|
| 552 × 1.805 min (1.781–1.833 gives 16.4–16.9 h) | 16.6 |
| Pilots already consumed (round 1 entries 62.7 min, round 2 24.4 min, load probe 2.6 min) | 1.5 |
| **Subtotal against the 19.0 h cap** | **18.1** |
| Per-process overhead, 12 processes × (cache load and verify 0.8 min + model load 0.1 min + same-run base entry 1.8 min) | 0.5 (24 processes: 1.1) |
| §6 qualification repeats (10 composition entries + base, once) | 0.35 (twice: 0.7) |
| Same-item bridge | 0.05 |
| **Total** | **≈ 19.0** (range 18.1 to 20.0) |

This fits the 19.0 h cap with essentially **no retry reserve inside it**; any overrun draws on the pooled 2.67 h reserve line, which is not for another experiment. Process count matters: one process per entry would add about 8.5 h, so entries must be run in large batches (a fixed batch size in `d3_shards` style). If the 552 figure already includes bases and repeats, the total is 18.1 h and there is 0.9 h of slack. Either way the §9 sufficiency test is passed only marginally, and "the full programme must still be repriced before scaling" is the binding statement.

Recommended before the panel (optional, cheap now, impossible later because references must match the final source): lower the per-entry time rather than accept the thin margin. About 24 s of each entry is the two refusal stages (350 batch-1 forwards each, latency-bound) and 37 s is HellaSwag (4 endings × 250 items). Scoring the six multi-token phrases and the four endings in batches of 6 and 4 (with the actual batch recorded, and a bitwise-stability check) should bring the entry to about 1.2–1.5 min and give back about 2.8–5.5 h (an unmeasured estimate). This needs a source change, a new pilot on the final `source_hash` (about 20 min) and must not touch MMLU-Pro, which stays batch 1. It is a decision for the owner; it is not a merge condition.

## 6. Non-blocking observations

- `SCORER_HASH` uses `inspect.getsource`, so any reformatting or comment edit inside the four hashed functions orphans the 5.4 GiB cache (about 4.4 min to rebuild). It also excludes `base_context`; the content root, recorded everywhere and checked against the bytes, is the safeguard for that gap.
- A `TFScoringStop` leaves `stop_<id>.json` and propagates out of `run_batch` (only `OutOfMemoryError` is caught there); the entry directories created up front remain. This is the intended "stop the scorer" behaviour but it has no end-to-end test and its resume behaviour should be exercised once in qualification.
- The refusal margins carry no `descriptive`/non-reportable flag inside the artefacts. Add one so the label travels with the data.
- Item hashing uses `id + salt` for every benchmark, so MMLU-Pro and HellaSwag share the same relative ordering of ids (the first three ids of both files are 1839, 9125, 3798). That is harmless for development panels. The sealed panel must use the benchmark-specific `TF-v1:item:` domain (PREREG §2), and dedupe of normalised text groups is not yet implemented.
- RESULTS.md states the Gate 1 result correctly and lists what was not run. DECISIONS records the panel status. There is no dated PREREG amendment yet for the as-built settings (see §8).

## 7. G1, G2, G3: which must run before the fresh panel

| Gate | Needed before the fresh non-null panel? | Cost |
|---|---|---|
| **G1** bitwise repeat (≤1e-4 nats, KL ≤1e-5) | **Yes.** It is the determinism qualification for every later number. Run it on the final source in fresh processes on the §6 composition (4 LoRA, 3 chains with ≥4 nodes, 2 task, 1 abliteration). Do it on at least two entries plus the base, in both resident-cache and fresh-load modes. The existing probe (one entry, ΔKL = 0, records identical) is supporting evidence only. | Stand-alone: base + 2 entries ≈ 3 × 1.8 + 0.9 ≈ **6 min**. Folded into the §6 qualification set (10 entries + base scored once, repeated once, plus load and verify): **≈ 0.7 h** of GPU, which is the real cost since the §6 composition has not been scored yet. |
| **G2** same-item bridge (100 legacy MMLU items: median |Δ| ≤ .02, p99 ≤ .2, argmax ≥ 99%) | **Yes, because it is cheap and a late failure is very expensive.** A failure versions a new endpoint and rebuilds all matched references (≥ 16 h of scoring). It needs a small code path: the TF panels deliberately exclude the 300 legacy items, so the bridge must score the 100 legacy ids with the letter scorer, and it has to be added and tested. It does not qualify signal or effect size. | GPU ≈ 3.5 s × 11 entries ≈ **1 min** plus load; engineering about 1–2 h. |
| **G3** second device | **No run is possible: no second device exists and new spend is not authorised.** Instead, before any fresh scoring, a dated declaration that the claim is **single-device (H0)** (PREREG §3, §6: a missing second device is single-device scope, not a transport pass), with training and qualification on H0 only. The fail-closed device id prevents a second machine from silently sharing `tf_cache/H0/`. | $0 and a documentation change. A cross-device claim would need hardware and a new amendment. |

## 8. Other preconditions before the fresh panel starts

1. The tests-only CI fix and the merge of `feat/tf-v1` (68a6455 is fine; it only touches the script and README).
2. A dated PREREG amendment, before any fresh outcome, recording the as-built tier: KL = 200 prompts × 64 greedy tokens (not 200 continuations of 128 tokens), averaged over positions then unweighted over prompts; HellaSwag 250 nested (descriptive); phrase-set refusal margins (descriptive, uncalibrated); that the δ/2 bound for KL applies to a stated scope (panel median, or relative to the entry's KL; as written it fails the 0.32-KL entry by 3.4 times); the D3 pilot items are excluded from the sealed panel; and the single-device scope.
3. The sealed-panel build (§2): `TF-v1:item:` domain, normalised-text-group dedupe, disjointness from the 1,000 MMLU-Pro development ranking, the HellaSwag and prompt rankings and the 300 legacy items, and the committed immutable inventory and partition manifest. The tier currently supports only the development panel; fresh components need base references on the sealed items too.
4. The §6 qualification pilot on the registered ten-entry composition (this pilot's ten controls have no LoRA-4, no chain and no 4-node chain), with the registered 4,000-draw procedure for the reliability and SE bounds, and the §5 development-only power report. Neither exists yet.
5. Exact-source freeze: any further `src/` edit (including the optional batching change) needs a new warm timing pilot on the final `source_hash` before references are generated.

## 9. Summary

1. **Verdict: ACCEPT with one tests-only fix.** All four round-1 blocking items are closed and tested (542 passed, 1 skipped on CPU), and the round-2 pilot numbers reproduce from the artefacts. `src/` is unchanged between the pilot source and HEAD.
2. **Merge: yes**, after a tests-only fix: three `test_backfill_eval_metadata` tests fail without `~/.config/wsp/device_id` (CI runner has none), because `runtime_metadata` now fails closed. The ruff item (F541 in the report script, fixed by 68a6455) does not block; `ruff check src tests` is clean.
3. **Gate 1: PASS** under PREREG §9 (warm median 1.805 ≤ 2.0 min; the 1.8 margin is missed by 0.3%, inside the range). Projection is 16.6 h scoring + 1.5 h pilots + about 0.9 h of overheads and qualification ≈ 19.0 h, so the cap is met with no retry slack; the optional refusal and HellaSwag batching, done before the source freeze, would return about 2.8–5.5 h (unmeasured).
4. **Before the panel:** G1 on the §6 composition (about 6 min stand-alone, about 0.7 h folded into qualification) and G2 (about 1 min GPU plus a small code path) must run. G3 cannot run, so declare single-device (H0) scope. Also needed: the PREREG amendment for the as-built tier and the KL δ/2 scope, the sealed-panel build and partition manifest, and the §6 qualification and §5 power report.
