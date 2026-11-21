# wme-c1: transition preparation T1–T3 (RULING-wme-c1-bank-halt addendum 3), 21 Nov 2026

This is engineering work only: no unit ran and no unit result was read. Evidence is in `wme-c1-transition-prep-2026-11-21-l5p.json`. The work ran from commit `fa347ad` on branch `run/wme-c1-run5`.

## T1: ledger reconcile of `1addc266` (done)

- **Records written:** study-log seq 290–292, under owner epoch 12 (`instance_close`, then `instance_authoritative`). They use the dead-man ledger's API-confirmed launch (20:24:00Z) and termination (20:54:07Z).
- **Close time:** the local close is placed at the confirmed termination, not at the last tick. The instance is therefore charged max(local 0.152 h, authoritative 0.502 h) = **0.502 h**, and the 0.756 surplus is gone.
- **Ledger check:** no instance is active. After T2 was reconciled, the study's instance charges total 9.607 GPU-h-eq, which is exactly the dead-man total.

## T2: determinism probe (PASS)

**Setup**
- **Instance:** one 1×H100 in us-south-2, driver 570.148.08, torch 2.14.0+cu130, bundle `9b361cf6…`.
- **What ran:** the registered `field_group` path (`serve.h_field_group`), restricted to `mmlupro_gold_lp` on g0, one fresh process per run. There was one registered-mode baseline, then three runs inside `genengine.deterministic`:
  - `use_deterministic_algorithms(True, warn_only=False)`;
  - cuDNN deterministic, with benchmarking off;
  - SDPA on the math backend;
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set before CUDA initialises;
  - linear attention bound to the transformers torch path, as checked by `check_model`.

**Results**

| run | mode | mmlupro_gold_lp.g0.pt sha256 | pass s |
|---|---|---|---|
| base1 | registered | `f88d04fa…` | 16.46 |
| det1 | deterministic | `af39a617…` | 12.86 |
| det2 | deterministic | `af39a617…` | 13.01 |
| det3 | deterministic | `af39a617…` | 12.85 |

- **Bitwise:** the three deterministic runs are equal for the file, all 8 gradient tensors and all 8 factor sets (A, G, count).
- **No kernel raised.**
- **Panels and model:** the panel content hash `39c6e20f…` and the model state `61988209…` match across all runs.
- **Baseline:** base1 is a fourth distinct registered-mode mmlupro file, after `b5f7b8e4`, `703e5654` and `45d4e3f7`.
- **The deterministic field differs everywhere.** Its gradients and its factors both differ from registered mode, because SDPA math changes the forward numerics. A deterministic field is therefore a new field everywhere, which the transition re-collects in full anyway.
- **Slowdown:** deterministic over registered is 0.78. That is no slowdown for this objective, but the baseline is a single cold first run. The other five objectives, especially wikitext's long sequences on SDPA math, were not timed.

## T3: field persistence options (priced; the chair chooses)

### (i) Lambda persistent filesystem, about 204 GB

- **Price.** Lambda's pricing page lists no filesystem rate. The billing docs (docs.lambda.com/public-cloud/billing) say: "billed per GiB used per month in one-hour increments … at a rate of $0.20 per GiB per month" and "the rate above is used for example purposes and might not reflect current pricing. The actual price will be displayed when you create your filesystem."
- **Cost at that example rate.** 204 GB is about 190 GiB, which gives about $38 per month, about $0.053 per hour, or about $1.3 per day. Billing runs while the filesystem exists, mounted or not.
- **Constraints, quoted from the docs.**
  - It "must reside in the same region as the instance".
  - It must be attached "at the time that the instance … is launched".
  - It "cannot currently be transferred between regions".
  - Creation through the API is `POST /api/v1/filesystems` (name, region). The launch body's mount parameter (likely `file_system_mounts`) is not yet verified.
- **Availability on our account.** The API lists 7 existing filesystems, none of them this study's (they are not touched). They are in us-east-1, us-east-3 and us-west-3.
- **Our capacity pattern.** The study's instances launched in us-south-2 (3 runs), us-southeast-1 (5) and us-south-3 (1). The last three 1×H100 launches together waited through about 185 failed polls of 30 s, roughly 90 min. A filesystem pins every later launch to one region.
- **Work needed.** Launch-time mounts in `up`/`lambda_ops`; `field_dir` on the mount; filesystem lifecycle and billing in the dead-man ledger; and the user's $1,000 total, which includes this storage.

### (ii) Keep one instance alive from field collection to the end of E2 and S-BoN

- **Cost.** 1×H100 at $4.89 per hour (API: 489 cents). The compute itself is already in the projection, at a registered bank price of 4.53 GPU-h-eq plus E2 and S-BoN. The extra cost is idle wall time between stages and during any halt, at $4.89 per hour, while the instance waits.
- **Risk.**
  - Any instance loss destroys the field. Causes include a provider fault, a halt that needs diagnosis, the dead-man's 30-min idle limit, or `t_max`.
  - With T2 passing, the loss costs a deterministic re-collection of the field: about 0.4 GPU-h-eq for 8 groups plus vocabulary, at registered-mode timings. After that, a bitwise K3 repair becomes possible again, against deterministic eligible hashes.
  - Without T2 passing, a loss would mean another transition. Persistence would then be mandatory, per addendum 3.
