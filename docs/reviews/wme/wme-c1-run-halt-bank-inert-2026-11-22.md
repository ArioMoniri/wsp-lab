# wme-c1 run9: halt at the compensated inert units (UnsupportedTensor), 22 Nov 2026

This was an engineering halt. No sealed file was read and no claim outcome exists. One incident is open: `worker_error` (seq 2356).

## Sequence (clean checkout of `b41bd68`; bundle sha256 `74420725…`)

| step | result |
|---|---|
| Dead-man pre-checks l9 and l9m; preflight | pass |
| Parity, 2×H100 (driver 580.105.08) | PASS: every pass-rule check true, including `hf_gen_eq_committed`; hf_gen `7299ac4f…` on GPU0, GPU1 and 3/3 repeats; code identity pass; gate 279 passed; projection 106.04 GPU-h-eq; terminated and confirmed |
| Main, 1×H100 (driver 580.105.08) | launched 18:18Z; heartbeat gate met |
| Resume (generation 1) | PASS: canaries equal the committed references; start canary passed |
| `resume-phase` (bank) | the chair's PhasePause resumption recorded: seq 1819, citing RULING-wme-c1-bank-halt.md addendum 7, outcome-blind; basis 5.320 GPU-h measured, threshold 5.663, charged 20.117, cap 110 |
| `repair-field` (K3) | all 9 field units bitwise equal again (third instance) |
| Bank | 41 of the 44 remaining units; every canary passed. **`inert:05:comp` failed:** `UnsupportedTensor: model.layers.12.mlp.up_proj.weight: not in the collected support` (incident seq 2356, attempt_fail with cause None) |
| `close` | closing canary passed, bracketing the five inerts i1 ×2 and i2 ×3; instance terminated, confirmed and reconciled |

## Cause: the compensated inert touches a tensor outside the field's support (registration and code gap)

`controls.inert` registers three compensated pairs: `ph100.COMPENSATED`, MLP layers 12–15 and attention layer 15 (`bankops.inert_steps`, kind `comp`). Their realised displacement includes `mlp.up_proj`.

The bank's R(E) computation (`field.RCache`, then `Field.contract_tensor`) contracts only:
- the field's writer tensors, the K-FAC support: `mlp.down_proj` and `self_attn.o_proj` / `linear_attn.out_proj`;
- the vocabulary rows.

Any other tensor raises `UnsupportedTensor`. Before run9 no inert unit had ever run in production (generation 0 stopped at the bank sides). The failure is deterministic, so `inert:06:comp` and `inert:07:comp` would fail the same way.

How a compensated inert's R(E) should treat its non-writer tensor is a design question. The options are to contract the supported tensors only and record the omission, to treat the unit as unsupported, or something else. The recorded failure cause is None, so a retry needs a repair record. A code change needs a new commit and a new parity run.

## State

- **Bank:** 210 registered units, **207 eligible** under generation 1, and 3 remaining: `inert:05:comp` (1 counted attempt), `inert:06:comp` and `inert:07:comp` (0 attempts).
- **Phase:** bank measured unit time is 6.016 GPU-h, and the phase is resumed (addendum 7).
- **Ledger:** study = dead-man = 22.113 GPU-h-eq ($107.15). Cap 110, cap generation 0, no R8 raise.
- **Field:** the next instance re-materialises it by K3, which has now been shown bitwise on three instances.
