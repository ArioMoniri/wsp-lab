# wme-c1 run13: halt at the first new E2 unit (a stale E2 environment after the canary's model release), 23 Nov 2026

This is an engineering halt. No E2 result was produced, no sealed file was read and no claim outcome exists. One incident is open: `worker_error`, seq 2868.

## Sequence

The run used a fresh clean checkout of `f343f24` (merged main: v3b and addendum 12) at the fixed run10 checkout path. The record is `wme-c1-run13-checkout-2026-11-23.json`. The bundle sha256 is `b66ed65c…`.

| step | result |
|---|---|
| Dead-man pre-checks l13 and l13m; preflight | pass |
| Parity, 2×H100 (driver 580.105.08) | PASS: every pass-rule check is true; code identity `f343f24`; terminated, confirmed and reconciled (0.878) |
| Main, 1×H100 (driver 570.148.08) | launched 20:20Z |
| Resume (generation 1) | PASS: the start canary equals the persisted reference |
| `repair-field` (K3) | all 9 field units bitwise equal (seventh instance) |
| Key check | PASS: the recomputed keys equal the logged keys (`f210cf6f…`, `83f62b48…`, `ae9fa6e4…`) |
| **`verify-e2`** | **PASS** (seq 2856): `e2:wm_e:00` and `e2:statnn:00` are timing-free equal, in 713.7 s and 674.2 s. The `wm_e` rate is 1.020 ×, so the factor is 1.30 (`reprice` and the P4 cap 128.386 stand) |
| `run --stages e2` | the guard was constructed (v3b) and the start canary passed (seq 2861). The first unit to issue, **`e2:random:00`, failed after 11 s**: `AttributeError: 'NoneType' object has no attribute 'theta_hash'` (incident 2868; attempt_fail with cause None) |
| `close` | the closing canary passed; terminated, confirmed and reconciled |

The first main node of run12 (setup failure, 0.435 GPU-h-eq) was already reconciled at the run12 resume (17:15Z). Nothing of it remains to reconcile.

## Cause: a code defect (latent since addendum 6, v2x)

`Runtime.release()` frees the GPU before a canary's child engines start. It dropped the R cache, the bank world, the field and the model (`ModelCtx.release`, which sets `sess` to None). It did **not** drop the E2 environment cache, `rt._e2`, which `serve_e2.env_for` builds and keys only by input digests.
- In run13, `verify-e2` built the Env first.
- The E2 start canary then released the model.
- The next E2 request reused the cached Env with its released model: `primitive_meta` → `modelops.at_base` → `self.sess.theta_hash()` on None. The worker traceback is in the result.
- The planner arms had run only before the canary. Run10 built its Env after its start canary, so it would have failed at the first chunk-boundary canary, 16 units later.

## Proposed fix (v3c; branch `run/wme-c1-9b-v3c`, from main `f343f24`, not launched)

`Runtime.release()` also drops `rt._e2` (closing its world) before `gc.collect()` and `empty_cache()`. The next `env_for` then rebuilds the Env on the reloaded model.
- Tests (`tests/test_wme_c1_v3c.py`):
  - the cache is dropped and nothing GPU-holding survives to `empty_cache`;
  - `env_for` takes its rebuild branch after a release;
  - a release without the cache is unchanged.
  They fail without the fix.
- The c1 tests around `release` still pass.
- A code change needs a new commit and parity, and `verify-e2` must be re-run (the code differs from `f343f24`).

## Attempt budget of `e2:random:00` (for the chair)

Generation 1 has two counted starts:
- run10's attempt (seq 2705), in flight when the controller was stopped under R9.2 and fenced at `close`, with no outcome and no abort record;
- run13's failed attempt (seq 2867, cause None).

One attempt remains, and the registered fallback (the base winner) applies if it fails. The retry needs `resolve-incident 2868` and a `record-repair` (`code_defect_crash`) naming the unit, citing the ruling.

## Ledger

- Charged 32.710 GPU-h-eq. Dead-man: 25.97 instance-hours, $158.60.
- Cap 128.386 (generation 1, P4); no further raise. No `wsp-wme-c1` instance is live.
- **Projection (P4 rule):** charged 32.710 + re-priced remainder 84.135 + overheads 3.046 = **119.89 GPU-h-eq**, under the cap.
