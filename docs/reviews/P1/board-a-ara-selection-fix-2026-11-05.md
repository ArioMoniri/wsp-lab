# Board A: ARA selection fix review — 5 November 2026

**Verdict: REVISE.** Reviewed `fix/heretic-ara-selection` at `33584f3` against the forensic disposition and D6 addendum 3. Do not merge or launch the rebuild until the launcher issue below is corrected and reviewed. Board B approval is also required for merge.

## Blocking finding

`scripts/lambda_job_ara.sh:10–11` clones the default branch or pulls the existing checkout's current branch, without selecting or verifying the reviewed revision. A fresh machine can therefore run an older generator and configuration: the script does not reliably pass the new target or enforce the new kernel at all. Pin and verify the intended revision before installing or building; explicitly validate the target in the generated config before expensive setup/build, and exercise the missing-target failure. Re-review the actual launch path. The corrected local config is inherited by the script's YAML copy (`:23–28`), and the new generator rejects a missing target (`src/wsp/zoo/heretic_ara.py:193`), but that is conditional on running the corrected revision. No real Lambda launch was performed.

## Findings and validation

CPU focused tests through `agent-tmux boardA-ara-fix`, after sourcing `scripts/env.sh`: **31 passed, 1 deselected**. The synthetic frontier test includes infeasible points, equal metrics and trial-number ties, and selects the prescribed point. The post-merge violation test exercises a false `kl_constraint_met` and failure; empty feasible studies fail with a recorded reason. Nonfinite values are rejected. The full suite was not independently rerun here.

`heretic_ara.py:218–225` forces upstream ARA's `preserve_row_magnitudes=False`; the same modifier is used for search and selected reapplication (`:232–240`, `:291–299`), and the merged scored state is compared tensor by tensor with the reloaded export (`:332–337`). The test captures both scored states and compares tensors, so a one-sided effective weight normalization that changes tensors would fail; it does not independently mutate the code to demonstrate that failure. Kernel identity is recorded in study, descriptor and builder identity. This is an unnormalized ARA variant, not a row-preserving Heretic treatment. The paper must label the family `wsp_ara_no_row_normalization_v1`, describe the disabled normalization and ten-trial startup search, and avoid claims of row preservation or capability preservation without independent evidence.

Read the 12 live IDs and metadata: all 12 manifest rows and edit descriptors have the diagnostic flag `ara_unconstrained_surrogate_scale` and `eligibility: smoke`. `src/wsp/eval/batch.py:164–176` filters on production eligibility by default, excluding those rows. This is an eligibility filter, not an independent flag-based exclusion; explicit smoke evaluation remains possible. No old artifact was changed during this review.

The target is a planned first-token KL ceiling, not a capability guarantee. Successful rebuild artifacts still require independent held-out capability evaluation and both-board review before paper claims. D6 addendum 3 expressly leaves the corrected rebuild pending; this review does not authorize the estimated $37–40 spend.
