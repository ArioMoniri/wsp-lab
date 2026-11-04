# P4 Board A revision

Merged main as requested, preserving both sides of append-only board/session conflicts.
Implementation stays on feat/p4-predictors; no integration into main or PR created.

1. Manifest-derived relations distinguish successors from sources. lora_vs_base is
   a flagged successor because the cumulative stored state includes its predecessor;
   it does not imply optimizing LoRA against that predecessor. Task-vector sources
   are single-step base edits with parent stats as side information, never history.
2. Family/seed folds purge full ancestry, including unlabeled chain inner_family
   steps and source dependencies, before normalization or JEPA pretraining. Audit
   captured the real manifest at 1219 rows (the zoo is still growing):

   | held out | candidate train | dropped ancestry rows | kept | violations |
   |---|---:|---:|---:|---:|
   | lora_sft | 1098 | 459 | 639 | 0 |
   | abliteration | 1014 | 443 | 571 | 0 |
   | random_dir | 1066 | 468 | 598 | 0 |

   These are structural manifest audits, not training runs: real eval targets remain
   absent. Counts include the current step if its inner_family is held out.
3. Default target is delta: measured rate minus immediate successor predecessor
   rate, or measured base rate for single-step edits. Missing predecessor targets
   cause exclusion with counts. --target level is explicit; all heads are linear.
4. M5 has two total attention layers and dropout 0.2; m1_matched widens one stateless
   feedforward layer, differing from M5 by four parameters. M6 reports frozen-standard-
   predictor ERank, collapse gates and H3 acceptance. Descriptor excludes length and
   inner_seed. MODEL_CANDIDATES/HYPOTHESIS explain cumulative-feature qualifications.
5. --selftest produces synthetic toy trajectories and target rates under /tmp,
   synthetic=true in manifests; no results paths. CUDA was not attempted because
   agent-tmux -l still lists p1-zoo-full. CPU selftests passed:
   - M2 loop: /tmp/wsp-p4-synthetic-gw0lydz8/run
   - M5 attention: /tmp/wsp-p4-synthetic-yv_fw8dc/run

Validation: 319 passed, 8 deselected in 21.10 seconds; 21 P4 tests (8 added).
Ruff lint and format clean. Real-data CLI refusal: zero entries have targets with
required predecessor targets; missing_predecessor_targets=0; no output dir created.
No real H1/H2/H3 scores or GPU/Mamba validation claimed.
