# Board A: ARA selection fix re-review — 5 November 2026

**Verdict: APPROVE.** Board A's launcher blocker is resolved at `4345b866ce7bac1cf67f11032d0333cbf0494828`. Board A supports merge; Board B approval remains required. A rebuild is **not yet authorized**: D6 addendum 3 explicitly leaves the corrected rebuild pending and says no new GPU work, rental or spend was authorized. Obtain the relevant spend authorization before launch.

## Scope and checks

Reviewed `git diff 33584f3..4345b86` in the branch worktree: only `scripts/lambda_job_ara.sh` changed. `bash -n scripts/lambda_job_ara.sh` passed. The launcher requires `WSP_REV`, fetches and checks out the supplied revision detached, checks that HEAD equals the supplied full SHA, and validates the derived launch config before the build. Use the full reviewed SHA above when launching; the equality check rejects abbreviated revisions. The launch config check occurs after dependency installation, but before the build.

Ran the actual guard snippets in scratch space: an unset `WSP_REV` exited 1 with the required-revision message; an all-zero SHA against a throwaway clone exited 128 with `fatal: reference is not a tree`; and a derived config without `kl_divergence_target` exited 1 with the assertion identifying the missing target. No real Lambda launch or GPU work was performed. The target check tests positive numeric values; the generator's existing validation also rejects nonfinite targets. This re-review did not repeat the prior focused tests or independently rerun the full suite.

The corrected protocol remains the D6 12-entry, 10-trial startup search, with the treatment labelled `wsp_ara_no_row_normalization_v1`. The first-token KL ceiling is not a capability guarantee. Preserve the previous review's requirements for independent held-out capability evaluation and both-board review before paper or Heretic-quality claims; do not claim row preservation. Ten trials do not guarantee feasible entries.
