# Board A: quick check of the wme-c1 v2r diff (K1–K3b), 21 Nov 2026

- **Scope:** `git diff 1904bbe..b667005 -- src scripts tests`, checked against `docs/council/2026-11-21/RULING-wme-c1-bank-halt.md` (K1–K5 and the K3b addendum).
- **Context:** `docs/reviews/wme/wme-c1-run-halt-bank-2026-11-21.md`.
- **Excluded:** the M′ freeze files in this range.
- **Method:** read-only code reading at `b667005` plus the CPU test run. I opened no data or sealed files. No GPU, no commits.

## Verdict for resuming: **ACCEPT WITH CONDITIONS**

The K1, K2, K3 and K3b code does what the ruling asks.

However, the resume sequence in K5 (parity, then `phasel-main`, then repair, then bank) **cannot run as wired**. `phasel-main` is a first-launch command:

- It overwrites the study attestation with a new contract.
- That either aborts with `ContractMismatch`, or, through a transition, supersedes all 26 eligible units.

Condition **B1** is therefore **blocking**, and it must be cleared before the Phase L parity re-run. The fix is code, and K4/P2 bind parity to the commit, so parity has to run on the commit that contains it.

## Tests

`WSP_VENV=wme-c1`, CPU only:
`pytest -q tests/test_wme_c1_*.py tests/test_wme_runguard*.py` gave **175 passed, 1 skipped, 1 deselected** (253 s).

The chair's 307 included the wmm suites, which this glob does not match, so the two counts are consistent. The working tree was untouched.

## Item checks

### K1: canaries in a fresh child (PASS)

**Where the child is used.** `h_canary`, `h_inert_identity` (HF part), `h_parity` (GPU-0 canaries) and `h_gen_repeat` now follow the same sequence:

1. `rt.release()`;
2. `hfchild.spawn(...)`, which runs `python -m wsp.wm.c1.hfchild` with a fresh `serve.Runtime`, `vllm=False`;
3. `GE.prepare_process()` runs before CUDA starts in the child.

- The GPU-1 parity helper was already a separate process.
- `genops.hf_generate` is the unit path, which K1 does not cover.
- `test_k1_no_in_process_hf_canary_remains_in_the_worker_handlers` enforces the change at source level.

**Hash semantics are unchanged.**
- The child calls the same `ModelCtx.canary_tf(tf_canary_items(PANEL_P2))` and `canary_hf(gen_canary_items())`, both from `at_base()`.
- The old `h_parity` called `canary_tf()` and `canary_hf()` with no arguments. Their defaults resolve to the same `PANEL_P2` and the same 64 `PANEL_P3H` items, so the hashed bytes are identical.
- The vLLM-gen tokenizer now comes from the pinned snapshot (`_canary_tokenizer`) rather than from the released model. Same files, so the same tokenization.
- **Conclusion:** at parity, `hf_gen` should still be `7299ac4f…`. Code cannot prove this; the GPU run must show it (see C2).

**Failure propagation.** A non-zero child exit raises `RuntimeError` in the parent and carries the stderr tail. A CUDA OOM therefore keeps its text and is still classified `out_of_memory` (tested).

**Minor points (not blocking):**
- `tf_hf_seconds` now includes model load in the child. It is timing only.
- After each chunk canary the worker reloads the model, field and R cache. This costs a few minutes per chunk but does not change any value: `RCache` is exact, and its hits/misses counters do not enter any payload.

### K2: canary error becomes an incident before the halt (PASS)

- `ctl.run_canary` wraps the call. On any exception except `IncidentAbort` (which the monitor already recorded), it writes `guard.incident("worker_error", {"where": "canary", ...})` and then re-raises.
- `worker_error` is an existing kind.
- Tested: the incident is recorded, after which `run_units` raises `HealthHalt`.

### K3: 8 field groups verified bitwise (PASS)

- `FR.expected("field_group")` requires both `file_sha256` and `kept_sha256` from the eligible payload.
- The data comes from `collect`'s per-file manifest map and from `compute_kept`.
- `manifest_sha256` is never compared, which is correct: it embeds wall time.
- The checks fail closed:
  - a missing or fallback payload, or a missing hash, refuses before any GPU work and records an incident;
  - a mismatch records an incident and raises `FieldRepairMismatch`.
- `repair_field` never calls `run_units`. Its only log writes are incidents and one `repair` record (`infrastructure`, commit, `outcome_blind=True`).

### K3b: indirect vocabulary verification (PASS, with C3/C4)

- `h_bank_verify` dispatches to the registered `h_bank_side`, `h_bank_comp` or `h_bank_inert` handler with the recipe that `bank_state_units` rebuilt from the eligible base. The unit key therefore equals the recorded one.
- It runs as a non-unit request (`bank_verify`, GPU class) and creates no attempt or result.
- Acceptance needs both:
  - `timing_free_sha(eligible) == timing_free_sha(rerun)`;
  - `r_vectors` (h, R, R_flags per state) exactly equal.
- **On mismatch:** an incident and a halt.
- **When no completed dependent unit exists:** an incident before any GPU work.
- Nothing is substituted in either case.

## Answers to the specific questions

1. **Could the timing-free hash strip anything other than timing?** No, not for these payloads.
   - Every key at any depth in the `bankops.side/composition/inert` payloads: `root, doses, states[h, dose, theta_hash, values{objective…}, sketch{objective…}, displacement_norms{tensor name…}, dnll, endpoint, eval_seconds, R, R_flags], r_seconds, order, prefix, R_h1, spec, endpoint_*`.
   - Only `eval_seconds` and `r_seconds` end in `seconds`. Objective names and tensor names never do.
   - The opposite risk, a timing field *not* named `*seconds`, does not exist today. If one appeared, it would cause a false halt, which is the safe direction.
   - The hash compares the whole remaining payload. That is stricter than K3b needs, because it also asserts that the model evaluation reproduces across instances. N3 already established that.

2. **Does R for a `row_scale` edit depend only on the vocabulary field?** Yes. Since only `embed_tokens` is touched:
   - `RCache._tensor_contrib` goes through `Field.contract_embed`, which calls `VocabField.contract`.
   - `chart` is all zeros.
   - The group norms come from the realised displacement only (model, not field).
   - The field groups do not enter.

   So the check is valid. It is **partial**, though. `contract` builds `dR` as zeros outside the root's rows, so the check exercises:
   - the gradient and `D_R` entries of the rows of `row_scale:002` only;
   - reduced to about two scalars per objective.

   Rows declared for other `row_scale` roots, and the I2 inert rows, are not covered. Collection is one joint pass, so a bitwise match on the covered rows is strong evidence for the rest, but it is not proof. See C3.

3. **Can a re-collected field silently replace the original?** Not silently: there is no path that writes a re-collected field into the study store or the log as a result. Two gaps remain:
   - **(a)** `h_field_group` and `h_field_vocab` write straight into the live `field_dir()`. On a mismatch the bad files **stay on disk**. Only the open incident (which `begin` refuses on) keeps them out of use. Once the incident is resolved, `stage_bank` would run state units on them.
   - **(b)** Nothing at bank time ties the on-disk field to the eligible payloads:
     - `GroupedField.load(verify=False)`;
     - `VocabField.load` only checks against its own new manifest;
     - `stage_bank` does not require a `repair` record for the current instance.

     Running `run --stages bank` before `repair-field` on the new instance would therefore run state units against a missing field. They would fail and consume attempts.

   See C4 and C5.

4. **Is the dependency rule (embed_tokens ⇒ vocabulary) complete?** Yes, for the registered families.
   - The embedding is touched by:
     - `row_scale` (both in `bankops.root_actions` and in `bankrun.actions_for`);
     - inert `i2` (`inert_steps`).
   - It is not touched by:
     - `scale` and `noise` (`rm.name(L, role)`), `proj_remove` and its control (`down` writers), and `steer`;
     - `lora_tv`, assuming its factors live under `layers.` (`rename_lora` splits on `layers.`, so the embedding cannot be targeted);
     - `i1` and `comp` (`rm` writers).
   - Compositions are treated correctly: dependent if any of their roots is.
   - Controls are overridden to `proj_remove_control`, which is also correct.
   - **Robustness note:** the rule is a hand-kept list. Deriving it from the actions (`any(a.tensor == m.embed)`) would be more robust. This is not needed now.

5. **Is parity on the new commit required and wired (code identity)?** It is partly wired.
   - **Wired:** `cmd_phasel_main` refuses unless `par.code_identity.controller_commit == HEAD` (P2), and R2 binds the engine config.
   - **Not wired:** the K4 extra requirement that `hf_gen == 7299ac4f…`, the committed reference.
     - `cross_instance_canaries` compares the main instance against the *new* parity evidence, not against the reference.
     - `phasel-main` then *mints a new* `canary_reference` from whatever it observes.
     - A drifted, self-consistent pair of instances would therefore pass. See B1 and C2.

6. **Anything blocking the Phase L parity re-run, `phasel-main` and resume?** Yes (B1). `cmd_phasel_main` on the existing study does the following, in order:
   1. commits new evidence;
   2. **overwrites `data/wme/runs/wme-c1/attestation.json`** with a new `LaunchAttestation`. Its `evidence` list contains the new parity file and its sha256, so its contract digest is different;
   3. only then builds `SD.Study` and `st.guard(...)`. Without a transition, `ProductionGuard.__init__` raises `ContractMismatch`.

   The results:
   - `phasel-main` fails after replacing the attestation file, which is not git-tracked;
   - every later `load_att()` command (`run`, `repair-field`, `close`) refuses;
   - if this were "fixed" with a `ContractTransition`, the 26 eligible units would become `superseded_contract`, and K3 would refuse (no eligible payload). That contradicts K3's premise that "the original 9 units remain the registered results".

   Also:
   - the start canary, the fallback binding and the reconciliation of earlier instances in `phasel-main` all assume first launch;
   - the Phase L parity command itself does not touch the study contract, so it is not blocked by this.

## Conditions

| ID | Condition | Blocking for | Owner | Kind |
|---|---|---|---|---|
| **B1** | Resume binding is decided and implemented **before** the parity re-run. The chair rules on how the v2r commit binds to the existing contract (`0a8efdac…`). Board recommendation: a resume mode of `phasel-main` that writes **no** new attestation and records **no** transition, then keeps the persisted contract and reference (`c7d87cd5…`). It records the new parity, inert, code and timing evidence as an evidence commit plus a log record that names it. It requires the observed per-engine canaries to equal the committed references (`hf_gen` `7299ac4f…`) **and** the new parity evidence. It opens the new instance and passes the start canary against the persisted reference. It reconciles only instances not yet reconciled. Add a test for it, plus a test that a plain `phasel-main` against an existing log refuses **before** writing `attestation.json`. | Parity re-run, `phasel-main`, resume | Chair, then implementer | Code + ruling |
| C2 | The K4 rule "HF canary = `7299ac4f…`" is enforced in code (part of B1). The parity run's `hf_gen_gpu0`/`gpu1` and the main instance's `hf_gen` must equal it; otherwise there is an incident and a stop. Until then the operator checks it by hand and records it in the run report. | `phasel-main` | Implementer | Code |
| C3 | For K3b, verify **every** completed eligible vocabulary-dependent unit (every completed `row_scale` side or composition, and any completed `i2`), not just the earliest. Record the covered rows against the declared rows in the `repair` record. If coverage is partial, state that in the record, and treat the first later `rows_outside_declared_support` or R flag as an incident. The cost is a few state evaluations. | K3 repair | Implementer / operator | Code or procedure |
| C4 | On any K3/K3b mismatch, the re-collected `field/g*` and `field/vocab` directories are quarantined (moved aside, with hashes recorded) before any `resolve_incidents`, and the repair is re-run from scratch. Incident resolution may not leave a mismatched field in `field_dir()`. | Resume after a failed repair | Operator | Procedure |
| C5 | `run --stages bank` (and any later stage that reads the field) runs on the new instance only after `repair-field` has succeeded on **that same instance**, with a `repair` record naming it. Preferred: fold a `stage_bank` precondition into the B1 commit that checks the on-disk manifests' `file_sha256` against the eligible payloads, so that no attempt can be consumed against a missing or unverified field. | Bank resume | Implementer / operator | Code (preferred) or procedure |
| C6 | The repair and verification compute (`field_repair`, `bank_verify`) and the parity re-run are charged to the ledger as K3 and K4 require. Report the GPU-h-eq used against the projection of about 84–90 out of 110. Any R8 raise is announced to the user. | Throughout | Operator | Procedure |

**Non-blocking notes:**
- The ruling's K3b step 2 ("if none depends on it, pick the earliest unit that does") contradicts step 3 ("no dependent completed unit → halt"). The code follows step 3, halting, which is the conservative reading. The chair may want to strike that clause.
- An optional hardening is to derive `vocab_dependent` from the actions' target tensors (Q4).

## BLUF

1. **Verdict: ACCEPT WITH CONDITIONS.** The K1, K2, K3 and K3b code is correct, but resuming is blocked until **B1** is cleared.
2. **K1 passes.** All four HF canary and check handlers run in a fresh `hfchild` after `rt.release()`. The canary items and code path are byte-identical, so `7299ac4f…` is expected at parity.
3. **K2 passes.** A canary error records a `worker_error` incident before re-raising, and `run_units` then halts. This is tested.
4. **K3 passes.** The 8 groups are compared on `file_sha256` and `kept_sha256`; `manifest_sha256` is never compared. Every failure mode refuses or halts with an incident, nothing is substituted, and no unit result or attempt is created.
5. **K3b is valid.** For `row_scale`, R is vocabulary-only, and stripping `*seconds` removes only `eval_seconds` and `r_seconds`. It covers only root 002's rows, so C3 requires verifying every dependent unit.
6. **B1 is the blocker.** Running `phasel-main` on the existing study overwrites `attestation.json` and then fails with `ContractMismatch`. A transition instead would supersede all 26 eligible units. A resume mode is needed before parity runs.
7. **C2: the `7299ac4f…` check is not wired.** Today the check is only relative to the new parity run. C4 and C5 close the gaps where a mismatched or missing field could be used after an incident is resolved.
8. **Tests:** 175 passed, 1 skipped (wme-c1 and runguard suites on CPU). No data, sealed files, GPU or commits were touched by this review.
