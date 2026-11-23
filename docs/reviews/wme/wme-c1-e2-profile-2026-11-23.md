# wme-c1 E2 profile (RULING-wme-c1-bank-halt addendum 9, P1), 23 Nov 2026

P1 cost $0. No instance was launched, no sealed or dev item was read, and the GPU was not used. Sources:
- the run10 payloads of `e2:wm_e:00` and `e2:statnn:00`;
- the controller's attempt records and worker dispatch times;
- CPU benchmarks of bounded planner slices on the shared development host.

The evidence is in `wme-c1-e2-profile-2026-11-23.json`, and the scripts are `scripts/wme_c1_planner_bench.py` and `scripts/wme_c1_rof_bench.py`.

## 1. Decomposition of the two completed arm-episodes (run10 payloads)

| quantity | `wm_e:00` | `statnn:00` |
|---|---|---|
| controller attempt wall | 1,283.7 s | 1,138.0 s |
| worker request (handler 1,191.3 / 1,113.2 s) | 1,195.7 s | 1,113.2 s |
| **GPU: dev evaluations** (48; median 9.21 / 9.55 s, range 8.8–9.7 s) | 448.9 s | 452.6 s |
| **GPU: R-cache contractions** (290 / 61 misses) | 49.9 s | 10.4 s |
| **CPU: planner compute outside the R-cache** (`planner_seconds` − R-cache seconds) | 692.5 s | 650.3 s |
| episode environment load | 4.4 s | 0.0 s |
| controller overhead (attempt − worker request; about 63 s of `wm_e:00` is the one-time input upload) | 88.0 s | 24.8 s |
| LLM-reader waits | 0 | 0 |
| **GPU-busy fraction of the attempt** | **0.39** | **0.41** |
| registered estimate (48 × `t_bundle_s` 14.580 s + `planner_gpu_s` 0) | 699.8 s | 699.8 s |

The measured GPU work is under the registered price. 48 evaluations take about 450 s, against the 699.8 s that 48 × 14.58 s implies. The whole overrun (1.84× and 1.63×) is CPU planner time, which the price does not contain.

## 2. Where the planner CPU goes (bounded slice: one beam, slot 0 of round 0, episode 0)

I did not finish a full-episode profile. The first attempt ran default torch threads (8) on this loaded 8-core host. After 2 h 24 min wm_e had not finished, and I stopped it; I stopped statnn at the chair's instruction. Neither produced data.

The bench instead runs one `Planner.beam` with:
- the fitted model from the prep `fit.pkl`;
- episode 0, `PLANNER_CONFIG` and CEM;
- the episode's planner seed.

R(E) goes through the production `RCache.r_of(Library.bundles(pids))` with the real `library.pt` and the pinned 9B role map. Only the per-tensor field contributions are deterministic stand-ins, because the field lives on the GPU.

**The work in one beam is structural and the same for both arms:**
- mode `ranks` (draw None, σ 0);
- 21 actions, each with 1–3 registered doses;
- 192 inner CEM optimisations (21 + 3 × 20 + 3 × 19 + 3 × 18);
- 1,455 scored batches, 8,190 rows and 7,275 member rollouts (5 members);
- 38,694 R(E) calls (each row's prefixes, plus the support penalty).

Production runs one beam per proposal slot, so an episode has at least 48 beams. Production planner CPU is therefore at most 14.4 s per beam (`wm_e`) and 13.6 s (`statnn`). This host does 11.0 s per beam at 1 thread, without cProfile.

**Shares under cProfile (wm_e; statnn is the same within 3%):**

| share | time (of 15.8 s) | detail |
|---|---|---|
| R(E) path (`RCache.r_of`, cumulative) | 6.2 s (39%) | Dominated by `Library.bundles` rebuilding the steer actions and `Action.key()` hashing the steering tensors on every call: 136k `tensor_hash` calls. `r_from_contractions` itself is 1.1 s. A cache-hit call costs 0.146 ms (`rof-bench`). |
| model prediction (`adapter9b.predict_batch`) | 5.5 s (35%) | Overhead-bound small torch calls: 232k `linear` calls. 3.0 ms per batch at 1 thread without cProfile. |
| remainder | 4.1 s | Trajectory building, snapping, CEM, and the bench's own keying. |

**Redundancy:**
- 32–39% of scored batches are byte-identical to an earlier batch in the same beam.
- 86–87% of rows repeat an earlier row.
- The 38,694 R(E) calls cover only 916–1,089 distinct primitive tuples.

The redundancy comes from snapping: CEM's continuous doses snap onto 1–3 registered doses.

## 3. P2 proposal: memoise exact repeats inside `e2arms.run_planner`

Both memos live inside `e2arms.run_planner`, one per arm-episode:

- **(a) R(E) memo.** `r_of` is keyed by the tuple of primitive ids. The first call goes through `Env.r_of_pids` as today. A repeat returns a copy of the stored vector, and still calls `env.tick("e2_r_of")` so heartbeat progress is unchanged.
- **(b) Whole-batch prediction memo.** `pa._run` is keyed by the exact bytes (shape, dtype, data) of every trajectory in the batch. A repeat returns copies of the stored `mean`, `sigma` and `members`.

**No registered quantity changes.** Arms, B = 48, items, seeds, the beam, CEM and `inner_budget`, the evaluation order and the evaluations are all untouched. The memo only skips recomputing a function on an input it has already seen.
- `r_cache.misses` is unchanged: every first visit still goes through the RCache.
- The only payload fields that change are timing fields: `planner_seconds`, `seconds`, `r_cache.seconds` and the per-evaluation `seconds`.

**Bitwise evidence (CPU, 1 thread, both arms, 2 repeats):**
- The memo-on beam returns exactly the memo-off result (`beam_result_sha256`).
- All 1,455 batch outputs are equal byte for byte (`batch_outputs_sha256`).
- The R(E) memo returns vectors bitwise equal to direct calls.

**Excluded because they fail bitwise:**
- **Row-level deduplication or row memo.** A row predicted alone differs from the same row in a batch by up to 3.05e-5 (wm_e) and 1.53e-5 (statnn): 0 of 174 rows were bitwise equal.
- **Changing the torch thread count.** 1 and 8 threads give different output bytes from the first batches on. 8 threads are also 42× slower on this loaded host: 102.6 vs 2.4 ms per batch. The completed units ran at the instance's default thread count, which is not recorded in the payloads, so P2 must leave the thread count as it is.

**Expected saving.** I apply the measured memo ratio of beam time (wm_e 0.412, statnn 0.399; 1 thread, no cProfile, 2 repeats) to the production planner CPU:

| | `wm_e` | `statnn` |
|---|---|---|
| planner CPU | 692.5 → about 286 s | 650.3 → about 260 s |
| saving per arm-episode | about 407 s | about 391 s |
| attempt | 1,283.7 → about 877 s (**1.25×** registered) | 1,138.0 → about 747 s (**1.07×**) |
| GPU-busy fraction | 0.39 → about 0.57 | 0.41 → about 0.62 |
| over 24 episodes | about 2.71 GPU-h-eq | about 2.60 GPU-h-eq |

Together that is about 5.3 GPU-h-eq, or about $26 at the ledger's $4.847 per GPU-h-eq.

`field_ranked` also runs through `run_planner`, so the R(E) memo applies to it, but it was not measured. The other 8 arms are outside P2.

**Caveats:**
- These are host-measured shares carried over to the instance. The instance's thread count and CPU differ.
- The saving will be measured for real in the P2 parity run.
- Only 2 of the 11 arms have measured E2 rates. `random:00` was in flight with no outcome, so the re-pricing (P3) has no rate for the other 9 arms.

**Implementation and verification, if the chair approves (addendum 9 P2):**
1. A new commit containing only the two memos in `e2arms.run_planner`.
2. Tests that show:
   - the payload with timing fields removed is equal memo-on vs memo-off on a fixture episode;
   - `r_cache.misses` and the tick count are unchanged;
   - a memo hit returns a copy.
3. ruff and the c1 suite.
4. On the instance, before any new E2 data: re-run `e2:wm_e:00` and `e2:statnn:00` and require full-payload equality with the committed run10 results after removing only the timing fields.

Each verification unit costs its GPU evaluations (about 450 s each) plus the reduced planner time, so about 0.4 GPU-h-eq for both.
