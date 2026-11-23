# wme-c1 run11: halt at `verify-e2` (E2 unit keys depend on the controller's checkout path), 23 Nov 2026

This is an engineering halt. No sealed file was read, no E2 request was issued and no claim outcome exists. One incident is open: `e2_speedup_unverifiable`, seq 2750.

## Sequence (fresh clean checkout of `8e2e9b9`; bundle sha256 `cd56f347…`)

| step | result |
|---|---|
| Dead-man pre-checks l11 and l11m; preflight | pass (4 clean ticks each; 0 error events) |
| Parity, 2×H100 (driver 580.105.08) | PASS: every pass-rule check is true, including `hf_gen_eq_committed`; code identity passes (`8e2e9b9`); gate 307 passed; terminated, confirmed, reconciled 0.870 GPU-h-eq |
| Main, 1×H100 | launched 06:30Z |
| Resume (generation 1) | PASS: the start canary equals the persisted reference |
| `repair-field` (K3) | all 9 field units bitwise equal (fifth instance) |
| **`verify-e2`** | **refused before any request**: `e2:wm_e:00: no eligible payload to verify against`; incident seq 2750 |
| `close` | closing canary passed; terminated, confirmed and reconciled |

Two IPs had been reused from earlier instances, and `wme_c1_known_hosts` still held their old keys. I removed those two stale entries, and `accept-new` recorded the new keys.

## Cause: E2 unit keys contain the absolute path of the controller's checkout

`harness.unit_key` hashes the whole recipe. E2 recipes (`study.e2_units`) carry `uploads`: the absolute local paths of `library.pt`, `dev_base.json` and `fit.pkl`. `Controller.e2_inputs` finds those files under `RUNS/ctl/results`, and `RUNS` is under the checkout (`ROOT/data/...`, through the `data` link).

- run10's prep and E2 units were keyed under `.worktrees/run/wme-c1-run10/...`.
- run11 runs from `.worktrees/run/wme-c1-run11/...`, so every E2 key differs.
- `e2:wm_e:00` gives `5a8e1f5f…` against the logged `f210cf6f…`.
- With the run10 prefix, the recomputed keys equal the logged ones exactly: `f210cf6f…` (`wm_e:00`), `83f62b48…` (`statnn:00`), `ae9fa6e4…` (`random:00`).

The same construction covers S-BoN, sealed and gen recipes (`uploads`) and prep (`bank_file`). The bank units do not carry local paths, which is why they stayed eligible across run6–run10.

Without the verification, `run --stages e2` from this checkout would have:
- issued all 264 E2 arm-episodes under new keys;
- re-run `wm_e:00` and `statnn:00` as separate units with the same labels;
- orphaned their eligible run10 results.

`verify-e2` refused at its first lookup, so nothing was issued.

## Options (for the chair)

- **(a) No code change.** Run the controller from a fresh clean checkout of `8e2e9b9` located at the run10 path (`.worktrees/run/wme-c1-run10`, with the `.venv` and `data` links). The keys then reproduce exactly (above), and every later stage keys consistently from that one path. Steps:
  1. `resolve-incident 2750` citing the ruling.
  2. Pre-checks, preflight, parity.
  3. Main, resume, `repair-field`.
  4. `verify-e2`, `reprice`, `cap-raise`, then the chain.

  Recommended.
- **(b) A code fix.** Make unit keys path-independent (the inputs are already content-addressed by the sha256 values in `args`). This changes the keys of every eligible prep and E2 unit, so it needs a key-migration rule for them, a new commit, tests and parity.

## Ledger

- Charged 27.270 GPU-h-eq (study = dead-man; dead-man $132.18). Cap 110, cap generation 0, no raise. No `wsp-wme-c1` instance is live.
- **P4 at the current ledger**, computed by `reprice` and `cap-raise` on a throwaway copy of the log:
  - charged 27.270 + re-priced remainder 84.138 + overheads 3.046 = **projection 114.455 GPU-h-eq**;
  - **cap 125.900 GPU-h-eq**, +1.7 % against 123.83, inside the 5 % bound;
  - each further launch adds about 1.9 GPU-h-eq of parity and setup.
