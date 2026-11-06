# Board A — data-loss incident D8: what the paper may still claim, what must be regenerated, what must be disclosed (6 Nov 2026)

Reviewer: Board A (strict ML reviewer). Subject: [DECISIONS D8](../../DECISIONS.md), `harness/lessons.md` (6 Nov entry), the paper as of `main` (`ed149c3`), PREREG_H1_TF A2–A4.
Verdict: **REVISE before any further paper edit.** The paper's headline conclusions (H1 not demonstrated, H1a untested, no target qualified) survive the loss. About half of the numbers that carry a run id do not, and the paper cannot keep citing those run ids as artefacts. A regeneration set of about **3.2 GPU-h + 8–10 CPU-h** is the minimum I will accept. Section 4 lists two process holes that are still open and that I rate as blocking.

## 0. What I examined, and corrections to the brief

I made read-only observations only. I did not touch `data/` or the GPU, and I created nothing in AWS or on the hub. I read the hub listing and the small JSON files into `/tmp` (not `data/`). I read `src/wsp/eval/batch.py` for the `source_hash` definition and recomputed hashes from git objects. I made read-only `aws ec2`/`dlm` describe calls.

| Item | Brief / D8 | What I found (hub = `Aralario/wsp-lab-zoo`, 4,242 files, public; `…-runs` private, 34 files; `…-evals` empty) |
|---|---|---|
| Deltas | "719 dense + 122 LoRA non-chain, 254 chain" | Per the manifest: **462** non-chain dense (153 random_dir, 205 abliteration, 80 task_vector, 24 ARA) + **122** LoRA (121 `lora_sft` + 1 ARA) + **254** chain = **838 of 1,259** entries. D8's 719 is 716 manifest deltas (462 + 254) plus 3 files of `p1-zoo-r3-smoke`. Every one of the 584 non-chain entries is recovered. |
| Chain bytes | 421 lost | 421 have no delta file, but **63 of them (21 evaluated) are byte-identical to a recovered delta**: they are in the content-hash alias ledger of `docs/prereg/panel_manifest.json` (101 groups). Effective chain coverage is **317 of 675**; **358 need replay**. |
| `records.jsonl` | "200 Lambda-scored entries" | 200 records = 179 on H1 + 10 on H2 + **11 H0 controls**. Lambda-scored entries are 189. The 189 H0-only entries have no item-level record, which matches D8. Evaluation device split of the 389 `eval.json`: **H0 200, H1 179, H2 10**. |
| Chain evaluation | 389 evaluated | 139 chain entries have `eval.json`; **only 56 of them have a delta file (77 with aliases)**. 62 evaluated chain entries have no bytes. |
| Hub manifest | "the manifest" | 1,267 lines = 8 header + **1,259 entry rows** (675 chain), equal to the paper's count. The hub's last metadata commit is 5 Nov 22:13 UTC. Its 74-commit history is available if either overwrite incident (D6 add. 2 and 4) is ever in doubt. |
| H0 base | "the H0 base evaluation record (8 min)" | **Three** H0 base identities are needed, not one. H0 `eval.json` source hashes: `6f274a83` (92 entries, 54 need a base), `e871923e` (89 chain predecessors, 33 depth-0 entries need a base), `d7cece35` (18, all need a base), `560abb41` (1, unmatched by design, D2 addendum). All four hashes **reproduce exactly from commits that exist in git** (below). |
| Generator revision | "reproducible from recorded step parameters and seeds" (D7) | `edit.json` of every chain step records `inner_family`, `inner_params`, `inner_seed`, `step`, `seed` and `delta_fro_norm`. But `git_sha` of 655 of 675 chain steps is `f7d3d4ae…`, which **is not an object in this repository** (history was rewritten on 4 Nov, lessons 2026-11-04). The generator at the recorded revision is therefore not available. |
| Hub visibility | `hf_sync.sh`: "private" | The zoo repo is public (`private=False`) and holds 205 abliteration and 24 ARA deltas. D7 addendum says "public release of deltas is not planned (abliteration deltas)". This conflicts with D7 addendum and with the unchecked "no uncensored weights" item in `paper/CHECKLIST.md`. It is the owner's call, but it bears on where a backup can live (section 4.3). |

Source-hash check (read-only; `source_hash` = sha256 over sorted `src/wsp/**/*.py` path plus bytes, as in `eval/batch.py:provenance`, recomputed from git objects): `dd2a609→a224779bf873`, `02c2847→12ee875a68c9`, `846ff84→d7cece35`, `f63a08e / 9ff01e7 / 5c4131c / d49376d / 12716a6→6f274a83`, `f90f591→e871923e`, `49c8321→560abb41`. All match. **Regeneration "under the same source hash" is therefore feasible for every B item below**, from a clean detached worktree at the named commit. Today's `HEAD` hashes to `de693c88…`, so nothing may be re-scored from `main`.

## 1. Classification of every result the paper currently reports

Definitions (strict):

- **(A)** The displayed number is recomputable now from surviving raw records: hub files, or git/docs where the artefact itself is the raw datum.
- **(B)** Not recomputable now, but regenerable at the stated cost under the recorded source hash. "Doc copy" means a committed document carries the printed value. That copy is a fingerprint to test the regeneration against. It is not a derivation.
- **(C)** The cited artefact cannot be regenerated, or regeneration cannot reproduce the quantity (wall-clock measurements, directory-level facts).
- Class `chain` means the result uses chain rows. If the replay acceptance test of section 3 fails, the original number degrades from B to C-with-doc-copy and a recovered-set rerun becomes a different, separately labelled result.

| # | Result (paper location) | Class | Basis / doc copy | Decision |
|---|---|---|---|---|
| Z1 | Zoo counts table, 1,259 rows by family and eligibility (`zoo.tex`) | **A** | Hub `manifest.jsonl`. I recounted every cell and all agree (e.g. chain 655 + 20, ARA 12 + 12 + 1). | Keep. |
| Z2 | 389 evaluated, none of the 24 ARA, 300 D3 entries (`zoo.tex`, `experiments.tex`) | **A** | 389 hub `eval.json`. D3 = the 300 entries not at source `e871923e`. | Keep. |
| Z3 | "Feature manifest records 1,259 entries … no failures" (`zoo.tex`) | **C** | `data/features/…/manifest.json` lost. After re-extraction `n_entries` is at most 901 (838 + 63 aliases). | **Must be rewritten.** The sentence is false once features are rebuilt. |
| Z4 | 345 of 389 matched, 85 chain predecessors, 260 with inferred runtime (`experiments.tex`) | **B**, chain | Matching needs the H0 bases (105 H0 rows depend on them). Doc copy: `board-a-d3-suite-2026-11-04.md` and `paper-v2-audit-2026-11-05.md`. | Regenerate after P1 (section 2); CPU minutes. |
| E1 | Base reproduction table (`p2/base-final-4096`) | **A** | Private runs repo `runs/p2/base-final-4096/{summary,records,manifest}`. | Keep. |
| E2 | Hosts and spend statement | **A** | `board/lambda_spend.md` is tracked in git. | Keep. |
| E3 | D3 quota statement | **A** | Manifest, `eval.json`, config. | Keep. |
| R1 | Same-host retest, 10 control pairs bit-identical (`reliability.tex`) | **A** | Hub `eval.local-run1.json` against `eval.json` and `devices/H0/*`, 10 of 10 present. | Keep. |
| R2 | Same-host retest, base pair (the 11th) | **B** | Both H0 base records lost (`6f274a83` retest and `d7cece35` first pass). Doc copy: RESULTS.md. | Regenerate (P1). |
| R3 | Cross-host Table devices (H1 n=11, H2 n=4) | **A** | `devices/H1`, `devices/H2`, `eval.lambda-*.json`, `devices/H0` and the `013b7b61202fe188` H0 record (`d7cece35`, run `20261104-084939-f912077b`): all on the hub. | Keep. |
| R4 | Cross-host base-model sentences (H0 vs H1/H2 base gaps, wikitext 22.90) | **B** | H0 base lost; Lambda bases survive. Doc copy: RESULTS.md. | Regenerate (P1). |
| R5 | Seed-1 panel (mean/max/ρ, "KL ≤ 0.0003", SDs) | **B** | Seed-1 records lost. Doc copy: RESULTS.md and `docs/council/…/metrology_budget_output.txt` lines 19–27. Seed-0 side is A. **Discrepancy:** RESULTS.md says KL is "identical" under seed 1 and the paper says "at most 0.0003 nats". The rerun arbitrates; until then the paper sentence is unsupported. | Regenerate (P3). |
| R6 | Binomial SE planning values; √r heuristic | **A** | Arithmetic on quoted base values. | Keep. |
| R7 | Qualification protocol text; admissible-claims table | **A** | PREREG and H5 prereg in git. | Keep. |
| R8 | D3 provenance: 260 of 300 matched, all inferred (run `p4/board-a-d3-audit-20261104`) | **B** | Audit directory lost. Doc copy: board-a-d3-suite review. | Regenerate (P1 then CPU). |
| V1 | Protocol-v2 table, 25 rows, mean [min,max] ρ (`experiments.tex`) | **B**, chain | Doc copy: `d3v2-summary-2026-11-05.md`; the paper's cells equal it. | Regenerate (P4). |
| V2 | Rows / clusters column (345/278, 98/45, 59/59, 60/60, 30/27 …) | **B**, chain | Only 345/278 is in a doc. The holdout counts exist only in the paper. | Regenerate (P4). |
| V3 | Baselines table (seed-0 StatNN, descriptor, delta-norm) | **B**, chain | Doc copy: `d3v2-summary` and the audit log. | Regenerate (P4). |
| V4 | Paired MAE differences (+0.016 [0.012, 0.021], +0.011 …; "all 14 intervals above zero") | **B**, chain | Full-precision doc copy for the M1-matched values: `paper-v2-audit-2026-11-05.md` lines 404–408 (e.g. 0.0155247). The 14-interval claim has no copy. | Regenerate (P4). |
| V5 | Directory-level facts: 25 complete run directories; `d3v2-m3-family` has no metrics; M6 gate; M3 no gate; no LoRA acceptance run; five one-fold holdouts | **C** | The directories are gone. Doc copies: RESULTS.md P4 row, PREREG §8, `board-a-synthesis-and-v2`. | May remain, labelled "reported from logs, artefacts lost". |
| X1 | Descriptor vs StatNN gap, pooled R² (`results.tex`) | **B**, chain | Doc copy: `docs/council/2026-11-05/evidence-statnn-importances.{md,csv}` (e.g. .523855 / .565906). | Regenerate with P4. |
| X2 | KL ρ .70/.49; 3-seed MAE gains 4.1% and 5.5% | **B**, chain | Doc copy: `round2_metrology_output.txt` lines 100, 102. | Regenerate with P4. |
| X3 | Dynamic-range table (Δy by family, 345 rows) | **B**, chain | Doc copy: evidence md. The H1 part is A; the H0 part needs the bases. | Regenerate (P1 then CPU). |
| X4 | Within-family StatNN refits | **B**, chain | Doc copy: evidence md. | Regenerate with P4. |
| X5 | Gain importances (26%, depth-bucket-0) | **B**, chain | Doc copy: evidence md. | Regenerate with P4. |
| T1 | TF pilot timings: 1.805 min (r2), 1.003 min (r3), ranges, per-stage tables | **C** | Wall-clock; per-entry tables are committed in RESULTS.md. | May remain, labelled "reported from run logs; wall-clock; artefacts lost". |
| T2 | r3 vs r2 bitwise identity of MMLU/KL/WikiText on 11 records; batched deviations (0.0148, 0.526, 0.371; 17/2,750) | **B** | Round-2 and round-3 records lost. Doc copy: RESULTS.md. Source `02c2847` and `dd2a609` both exist and hash correctly. | Regenerate (P7, optional) or label "reported from logs". |
| T3 | G1: 14 of 14 repeats bitwise identical | **B** | `g1_report.json` and records lost. Composition committed (`docs/prereg/qualification_panel.json`). 11 of 13 scored ids have bytes. `b7edbb9f8b339f52` and `86ea1fe0ec777cf8` (chain predecessors) are lost. | Regenerate 12 of 14 (P6a). The other 2 are conditional on replay. |
| T4 | G2 same-item bridge (median 2.2e-16, p99 .066, argmax 99%; HellaSwag 99.9%) | **B** | Fast-tier `records.jsonl` for the 10 controls are on the hub; the H0 base record needs P1. Doc copy: RESULTS.md. | Regenerate (P6b). |
| T5a | D3 stratum composition: 229 = 139 chain + 60 LoRA + 30 TV; family weights .700/.206/.094; 3 legacy rows excluded | **A** | Manifest + `docs/prereg/panel_manifest.json` + hub `eval.json`. | Keep. |
| T5b | "378 scored, median wall 1.02 min" (`d3_stratum_minutes.jsonl`) | **C** | Wall-clock log lost; count and median survive only in RESULTS.md. | May remain, labelled "reported from run logs". |
| T6 | Table tfqual: A2.1 on 229 entries, both endpoints | **B**, chain | Doc copy: `tf-v1-requalification-2026-11-06.md` (full tables). `requalification_report.json` lost. 28 of the 42 listed per-entry (y, h) rows are reproducible (21 LoRA, 1 TV, 6 chain). | Fingerprint audit required (P6c); full re-derivation optional (P6d). |
| T7 | MMLU failure detail: max h .114, P .226, 42/229, 17.6%, ρ .92 | **B**, chain | Same doc copy. **The max-h entry `c538973e5ad53477` (.1137) and the runner-up `a5cf6d9fea171ef4` (.1125) are chain entries with lost deltas.** The ceiling failure (ii) is independently reproducible from recovered entries: LoRA `d8a9fc56d92a3ad1` has h = .1036 > .10. | As T6. |
| T8 | A2.3 gate: power .43–.47, no interval procedure met size and coverage | **B**, chain | Doc copy: requal md. Simulation needs all 386 D3 OOF rows. | Label "reported from logs" unless P6d plus replay. |
| T9 | H1a θ̂ 12.9%, three-procedure intervals, leave-out 0.7%, item halves, family splits | **B**, chain | Doc copy: `h1a-companions-2026-11-06.json` (machine-readable, committed) and requal md. | As T8. |
| T10 | Design statements (A2 freeze, A3 pending, sealed panel unscored) | **A** | PREREG and git. | Keep. |
| T11 | "WikiText claim pending; no OOF reported" | **A** | Prose. Stale against A4 (see note below). | Keep or update. |

Counts: **A = 12** (Z1, Z2, E1, E2, E3, R1, R3, R6, R7, T5a, T10, T11), **B = 21**, **C = 4 results (Z3, V5, T1, T5b)**. Total 37 result groups. Of the B results, 15 depend on chain rows (Z4, V1–V4, X1–X5, T6–T9, and T3 through its 2 chain predecessors).

Notes:

- The paper's TF subsection still says the WikiText claim is "pending" and that no OOF loss is reported. DECISIONS shows the H1w result (θ̂ = 0.0071, gate FAIL) and the H1w JSON is committed. The paper is stale relative to the record. This is not an incident issue, but the reproducibility statement must not be written against the stale text.
- `audit_paper_numbers.py` reads `data/` for every CHECKED claim and will fail wholesale. Until regeneration, run it with the committed audit log `docs/reviews/P4/paper-v2-audit-2026-11-05.md` as the oracle. It holds 723 source values at 6–7 digits, of which about 264 lines are d3v2, 89 `eval.json` and 19 seed-1. After regeneration, diff the regenerated sources against that log; this is the fingerprint test for P4 and the seed-1 panel.

### 1.1 Reproducibility statement (exact wording)

Replace any mention of surviving run directories with this paragraph (appendix "Reproducibility", plus a one-line pointer in the main text):

> **Artefact loss.** On 6 November 2026 a version-control error deleted this project's working data directory (about 260 GB) before any off-machine backup of it existed (incident record: `docs/DECISIONS.md`, D8). The public Hugging Face dataset `Aralario/wsp-lab-zoo` preserved the zoo manifest and every edit record, the fast-tier evaluation summaries of all 389 evaluated entries (item-level records for 200 of them), both Lambda base evaluations, and the deltas of all 584 non-chain entries and of 317 of the 675 chain steps (254 stored, 63 byte-identical duplicates of stored deltas). The following artefacts were lost and are not released: (i) the deltas of 358 chain steps and everything derived from them; (ii) the run directories of the protocol-v2 predictor suite (`data/runs/p4/d3v2-*`: per-seed metric files, fold prediction tables, manifests); (iii) the per-item records, timing logs, KL base cache and qualification JSON of the TF-v1 pilot, G1, G2 and D3 stratum runs as originally produced; (iv) the item-level fast-tier records of the 189 locally evaluated entries (their summaries survive); (v) the original local base-model evaluation records. Where a number below was recomputed from regenerated artefacts, the text names the regeneration date and the source hash; regenerated values replace the originals wherever they differ, and the originals are listed in the appendix. Numbers marked "reported from logs" come from committed run summaries (`docs/RESULTS.md`, `docs/reviews/P4/`, `docs/reviews/P5/`) whose underlying artefacts no longer exist; they cannot be re-derived, and they are not used to support any claim stronger than descriptive.

Labels for individual items:

- "Reported from logs; artefacts lost." Use for V5, T1, T5b. For V1–V4, X1–X5 and T6–T9, use it only until the corresponding regeneration lands, or permanently if the replay test fails.
- "Regenerated on `<date>` under source hash `<hash>`; agrees with the retained summary to the printed precision." Use only after the check in section 2 passes.
- "Regenerated on `<date>`; differs from the retained summary (original `<x>`, regenerated `<y>`); the regenerated value is reported." Use if any check fails.

**May the (C) results stay?** Z3: no, rewrite it. V5, T1, T5b: yes, with the "reported from logs" label, because each has a committed document copy at printed precision and none carries a claim above descriptive. No table may keep a run-id column naming a directory that no longer exists. The `Run id` column of the v2 table must become either the regenerated run ids or "retained summary". Any `\path{data/…}` that points at a lost file fails the audit and must be repointed.

## 2. Minimum regeneration set (priority = cost)

Common gates, all of them mandatory:

1. Run only from a clean detached worktree at the named commit. Verify `source_hash` and `dirty: false` in the first record before continuing. Verify `config_sha256` against `docs/prereg/panel_manifest.json` (`snapshot.config_sha256`).
2. `src/wsp` may not be edited to redirect the data path; that changes `source_hash`. Use a symlink plus a per-worktree `info/exclude` (see section 4).
3. The GPU must be exclusive. A fast-tier OOM retry that changes the actual batch from 32 makes the pair non-matching and the run unusable.
4. Verify every restored artefact first: safetensors headers readable, every dense delta's Frobenius norm equals `edit.json:delta_fro_norm`, and each entry's content hash (`entry_delta_hash` in `scripts/build_h1a_panel.py`) is a member of its component's `source_identities` in `panel_manifest.json`. This is exact for the 47 singleton components and set-level for the rest. Estimated under 1 CPU-h, not measured.
5. Success and failure criteria are fixed now, before any regeneration. Fast-tier and TF scalar fingerprints must match to the printed digits. P4 ρ cells must match to ±0.005 and paired intervals to ±0.001 (half the printed unit). Any miss means the regenerated value is reported and the miss is itself recorded as an incident.

| Pri | Item | Cost (estimate unless marked) | Bitwise expectation | If it differs |
|---|---|---|---|---|
| P0 | Restore verification (gate 4); `reconstruct_historical_items.py` from rev `846ff84` (restores `historical-item-identities.json`); repoint the audit script | CPU < 1 h; CPU minutes | Not applicable. | Any hash or norm miss: quarantine that entry and re-pull it from the hub before anything else. |
| P1 | **H0 base re-evaluation, three identities**: `6f274a83` (from `f63a08e`), `d7cece35` (`846ff84`), `e871923e` (`f90f591`), fast config, batch 32, device H0. About 8 min each (user figure). Do not rebuild `560abb41` (D2: stays missing; adding it would change the registered 345-row set to 346). Unblocks Z4, R2, R4, R8, X3, T4, V* targets for 105 H0 rows. | **0.4 GPU-h** | **Expected identical.** Same device, same seed and per-item seeding; the retest was 11 of 11 bit-identical across two source trees (D2 add.). Check: RESULTS.md base scalars (gsm8k .16, math500 .16, ifeval .40, mmlu .28, hellaswag .39, over-refusal .10, wikitext 22.90). | Report the regenerated base as the reference, recompute every dependent Δy, and amend the reliability section: the "within a device the fast tier is deterministic" claim would then be falsified for the current environment and must say so. |
| P2 | **Feature re-extraction** for the 901 recoverable entries (v4 layout). The user's estimate is about 5 CPU-h. | **≈ 5 CPU-h** | Cannot be verified directly: no v4 feature file survives. Verified indirectly by P4/X1 fingerprints (audit-log source values at 6–7 digits). | If P4 fingerprints fail with unchanged rows, suspect extraction (BLAS thread order, SVD). Report. |
| P3 | **Seed-1 repeat panel**: 11 records on H0 under `eval_fast_seed1` (10 controls + base), source from the recorded run. Basis: 8.63 min mean on H0 × 11, equal to the recorded 95-min window. Run P1 and P3 in one session. | **1.6 GPU-h** | **Expected identical** (per-item seeding, same device). Check against RESULTS.md and the audit-log seed-1 lines. | Replace R5 with the regenerated values and flag the seed-1 determinism result; this decides the KL "identical vs ≤ 0.0003" discrepancy. |
| P4 | **P4 protocol-v2 suite** on the recovered rows: 25 runs × 3 seeds, StatNN/LightGBM + M1–M6, CPU. Basis: PREREG §9 cap 2.00 h and the 1–10 min/run anchor. | **≈ 2–4 CPU-h**, 0 GPU-h (`train.py` defaults to `device="cpu"` and PREREG §9 says CPU preferred; the device of the original runs was in the lost manifests) | StatNN (LightGBM, row-wise, 2 threads) deterministic given identical features and build. Neural models: CPU deterministic only at fixed thread count; not guaranteed. | **The row set differs unless chain deltas are replayed** (about 98 of the 345 rows are chain; roughly 45% of evaluated chain rows are lost). A recovered-set run is a different experiment: publish it as its own table (rows/clusters stated), keep the original V1–V4 as "reported from logs", and do not mix them. |
| P6a | G1 repeat pass on the recoverable composition: 12 of 14 repeats (11 entries + base); primaries come from P6c. | **≈ 0.25 GPU-h** | Expected exactly 0 (14/14 originally). | Any nonzero: determinism claim fails for this environment; stop all TF work and diagnose. |
| P6b | G2 bridge: 11 entries × 200 items, needs the P1 base | **< 0.1 GPU-h** | Expected the same median, p99 and argmax. | Report regenerated criteria; one-item argmax shortfall is investigated, not a stop (A2.4). |
| P6c | **TF-v1 fingerprint audit** at `dd2a609` (`a224779bf873`): base, the 10 pilot controls (gold log-probs to 6 decimals are in RESULTS.md), the 28 reproducible (y, h) rows from the requal md, the G1 composition, plus the KL cache rebuild (root `2ed18bda284b3a89…` is a registered fingerprint). About 45 unique entries at 1.02 min. | **≈ 0.8 GPU-h** (incl. cache rebuild, 5 min) | **Expected identical** (G1 14/14, batch-1 deterministic). Success means the root is `2ed18bda…` and y/h agree to the printed 4 digits. | A different root invalidates all TF comparison; report and stop. Any y/h miss: report the regenerated panel as a new development panel, never pooled with the lost one. |
| P6d | Full TF rescoring of the recoverable D3 entries (239 non-chain + 77 chain = 316): ≈ 5.4 h at 1.02 min plus cold start. Required only if the paper keeps T6–T9 as results rather than "reported from logs". It includes P6c. | **≈ 5.5 GPU-h total (+4.7 beyond P6c)** | As P6c. The registered A2.1 table cannot be reproduced without the 62 lost chain rows, so P6d yields a recovered-set panel (167 stratum rows), not the A2.1 table. | Label as recovered-set; A2.1 verdicts for the 229-row panel remain "reported from logs". |
| P7 | Round-2 source identity re-check (`02c2847`, 11 entries) against the P6c round-3 records. Optional. | ≈ 0.45 GPU-h | Expected bitwise for MMLU/KL/WikiText. | Otherwise T2 stays "reported from logs". |

Totals. **Minimum accepted (P0–P4, P6a–c): GPU ≈ 3.15 h (0.4 + 1.6 + 0.25 + 0.1 + 0.8), CPU ≈ 8–10 h (< 1 + 5 + 2–4).** Extended (add P6d and P7): ≈ 8.3 GPU-h. Chain replay (section 3) for the 62 evaluated lost rows adds ≈ 1.05 GPU-h of scoring plus unmeasured replay cost.

What I do **not** require: regenerating the item-level fast-tier records of the 189 H0 entries (≈ 27 GPU-h at 8.63 min). No reported number uses them: every table and bound is computed from `eval.json`. This is disclosed instead. The 296 unevaluated lost chain steps are also not regenerated, since no reported number needs them.

## 3. Chain family (254 of 675 recovered, 317 with aliases)

Structure (hub manifest and `edit.json`): 154 lineages of length 2–7 (30, 25, 28, 26, 18, 27 lineages of length 2, 3, 4, 5, 6, 7), one branching node, 675 steps. Chain deltas are cumulative from the base (D7, `zoo.tex`), so scoring a step needs only its own delta. A step's predecessor delta is needed only for the incremental norm and for the predecessor reference record.

| Availability of evaluated chain entries (139) | n |
|---|---|
| own delta available (direct or alias) and predecessor available (or root) | **62** |
| own available, predecessor lost | 15 |
| own lost, predecessor available | 37 |
| both lost | 25 |
| (all 139 have `eval.json`; 94 have an evaluated predecessor and 45 are depth-0, so fast-tier labels exist for all) | |

Which analyses survive:

| Analysis | Status after the loss | Verdict |
|---|---|---|
| **Predecessor-relative fast-tier targets** (Δy per step) | Labels exist for all 139 (hub `eval.json`; the 45 depth-0 rows need the H0 bases of P1). | **Possible, A/B.** |
| **TF-v1 predecessor-relative rows** | Own + predecessor available for **62 of 139**. | **Partial.** 62 rows are reproducible without replay. |
| **D3 chain stratum (229-row panel, A2.1/A2.3)** | 62 of 139 chain rows reproducible; the two largest half-widths are on lost rows. | **Not reproducible as registered.** The recovered-set panel is a different panel. |
| **P4 v2 suite (98 chain rows of 345)** | The 98 matched chain rows lose about 45% of their features. | **Possible only as a recovered-set rerun** (section 2, P4). |
| **M3 (successor pairs)** | 521 successor pairs; **98** have both ends' bytes (94 have both ends evaluated; **25** have both bytes and both evaluated). | **Not reproducible as reported.** A recovered-set M3 is a small variant and must be labelled so. |
| **H2 history ablation** (parked, PREREG §7; needs complete lineages of length ≥ 4) | 99 lineages of length ≥ 4; only **6** have ≥ 4 leading steps recovered, **3** of those evaluated. | **Not possible** without replay. With replay of all 358 steps it becomes possible again, but it is parked and unfunded; do not spend on it. |
| **Fresh/sealed chain components** | Sealed partition `ebf0379a15d9…` is committed. Its chain members that fall among the 358 lost steps have no bytes (I did not count them). | The seal remains valid as a record; nothing may be scored on it (decision (c) already). |

**Replay.** The 62 evaluated lost chain entries need exactly **62 replayed steps** (31 `lora_sft`, 18 `random_dir`, 13 `abliteration`); 24 of these entries are replayable without training. Replaying all 358 needs 146 LoRA retrainings, 108 random-direction steps and 104 abliteration steps. Replay starts from the nearest recovered ancestor (its stored Δ_eff reproduces the parent's bf16 weights exactly, D1), so no full-prefix rebuild is needed. I do **not** accept replay as "regenerated under the same source hash": the generator revision recorded in 655 of 675 steps is not in git. I accept it only after an acceptance test, which must pass before any replayed delta is scored or featurised:

1. Replay at least 2 recovered steps per inner family (6 total) from their recovered parent, with the recorded `inner_params` and `inner_seed`. Compare bytes and `delta_fro_norm` against the recovered delta. For abliteration, also verify that the recomputed refusal-direction fingerprint equals the one in the entry's `reuse_key`.
2. Pass means **byte-identical** (sha256 and set membership in `source_identities`). Support: A1.7 already records 101 byte-identical step-0 duplicates built independently, which is evidence for random_dir/abliteration determinism but not for LoRA training.
3. If random_dir/abliteration pass and LoRA fails, replay only those (24 entries) and mark the 31 LoRA-dependent entries lost. If everything fails, all 62 stay lost and the chain-conditional results above are demoted as stated in section 1. A replayed delta that matches only in norm is not an A-class artefact and may not support a bitwise claim.

**Zoo table and manifest.** Keep all 675 chain rows. The IDs hash (family, params, seed, base, parent), parent edges reference them, and the registered `panel_manifest.json` (483 components, `partition_sha256`) and the alias ledger include them. Removing rows would invalidate the seal and break lineage reconstruction. But do not rewrite `manifest.jsonl`; its hash is registered. Record loss in a **sidecar** `docs/zoo_artefact_state.json`, committed (id → `{artefact: present | recovered | alias_of:<id> | lost | replayed_verified, evaluated: bool}`), and:

- Zoo table (`zoo.tex`): keep the counts, add a column or note "delta bytes available: 317 of 675 chain rows (254 stored + 63 aliases); 358 `artefact_lost`".
- Rewrite the feature-manifest sentence to the true n and list failures (Z3).
- Any figure that counts "entries" for a model (zoo size, family balance) must state whether it counts rows or rows with bytes.
- Set `artefact_lost: true` on the 358 rows in the sidecar (and in any exported zoo table). Never alter `eligibility`, which is registered.

## 4. Process

### 4.1 Guards now in place

| Guard | Works for | Hole (evidence) | Required fix |
|---|---|---|---|
| `.githooks/pre-commit` refuses staged `^data($\|/)` | `git commit` on a checkout whose tree contains the hook | **Live hole:** worktree `.worktrees/perf/eval-throughput` (`73f1419`) has the old hook (no guard; I read its `HEAD:.githooks/pre-commit`: just `secret_scan.sh && precommit.sh`) and **no ignore for `data`**: `git status` there prints `?? data`, so `git add -A` stages the symlink. `core.hooksPath` is the relative `.githooks`, so every worktree runs its own branch's copy; a branch older than `0b54eb3` has no guard. | Set `core.hooksPath` to one absolute path outside any checkout (installed by a script), and retire or rebase that worktree. |
| (same hook) | — | The hook fires only on `git commit`. The incident vector was **checkout on merge**. Fast-forward merge, `pull`, `rebase`, `cherry-pick`, `reset --hard`, `stash pop` and `checkout` materialise a tracked `data` without running pre-commit. No `pre-merge-commit`, `reference-transaction` or `post-checkout` hook exists. | Add a `reference-transaction` (prepared) hook that aborts any update of `refs/heads/main` (and `HEAD`) to a tree where `data` or `.venv` is a tracked path; add `pre-merge-commit` as well. I have not tested these hooks; they must be tested in a scratch repo before adoption. |
| `.gitignore` `data` (bare) | `git add -A`, `git add .` | I reproduced this in a scratch repo: with `.venv/` and `data/` patterns a symlink of that name shows as `?? .venv` / `?? data`; the bare pattern ignores it. Main now has bare `data` and bare `.venv`, which is correct. Old branches do not. | Do not rely on branch-resident ignore. Have `wt.sh new` append `data` and `.venv` to `$(git rev-parse --git-path info/exclude)` for each worktree. |
| `wt.sh merge`: refuses `git diff --name-only main...<b>` touching `data`, and a dirty main | Merges done through `wt.sh` | Advisory. A direct `git merge`/`pull` bypasses it. It checks before merging; there is no after-merge assertion that `data` is still a directory. | Merge with `--no-commit`, check the staged tree, then commit; assert `[ -d data ]` afterwards. Keep it, but treat it as convenience, not protection. |
| Remote refs | — | Five remote branches (`origin/docs/model-candidates`, `docs/refs-venue`, `feat/p1-heretic-lora`, `feat/p2-eval`, `feat/plots-design`) carry a tracked `data/papers/*.txt` tree. Merging one writes into `data/`. | Delete or rewrite them. |
| `wt.sh rm` | — | Uses `git worktree remove --force`. Two worktrees once held REAL `data/` dirs (lessons 2026-11-04); `rm` on such a worktree destroys artefacts. | Refuse when `data` is not a symlink. |

**Structural hole (blocking): irreplaceable data lives inside a git working tree.** Any operation that rewrites the tree (`git clean -fdx`, `checkout -f`, `stash -a`, a merge) can delete ignored content, and git treats ignored files as overwritable. The fix that makes the whole class harmless is to move the canonical data **out of the checkout** (for example `/home/ubuntu/wsp-data`, with `data` an ignored symlink in main as it already is in worktrees). A checkout over a symlink then replaces only the pointer. Do this after the hub restore completes, not during it.

**Hub manifest overwrite.** `scripts/hf_sync.sh push zoo` runs `hf upload Aralario/wsp-lab-zoo data/zoo zoo`, which writes `zoo/qwen3.5-0.8b/manifest.jsonl` unconditionally; an instance with a partial local manifest replaced the canonical file twice (D6 add. 2 and 4). The job script now uploads only its own entry folders, but nothing prevents a repeat, and nothing checks the head revision. Required: a manifest pusher that (a) refuses if the new id set is not a superset of the hub's current set, (b) passes `parent_commit` so a concurrent change fails instead of overwriting, (c) is the only code path with write access to that file (Lambda instances use a separate repo or a token scoped to their own paths and are pulled from). Recovery for the manifest exists: 74 hub commits. Also fix the stale header of `hf_sync.sh` ("private"; the zoo repo is public).

### 4.2 Backup and snapshot findings (read-only)

The data lives on the 1 TiB root volume `vol-09c65c6712dc73a54` (tags `Project=T3CodeGPU`, `Name=t3code-gpu-root`; 194 GB used at the time of looking, restore still running). Its latest snapshot is 2026-10-31 14:45 UTC, which predates the project. DLM has two policies. `policy-0b225d93472ae7e11` is enabled (daily 01:20, keep 7) but targets tag `kalem:appliance=true`, which this volume does not carry. `policy-0b54be60f53489ecb` targets `DR-Policy=t3-golden` (daily, keep 14), and is in **ERROR**: DLM is not authorised to assume `AWSDataLifecycleManagerDefaultRoleForAMIManagement`. So even tagging the volume today would not produce snapshots. An unmounted 410 GB device `nvme1n1` exists; I did not inspect it.

### 4.3 Backup policy I require

Cost figures are US list prices (S3 Standard $0.023, Standard-IA $0.0125, Glacier Instant Retrieval $0.004, EBS snapshot $0.05 per GB-month); region not verified. Treat them as planning values.

1. **Nightly off-volume copy of all of `data/` except rebuildable caches (KL base cache, HF cache), RPO ≤ 24 h.** Target a new versioned S3 bucket in this account, SSE enabled, never with `--delete`, noncurrent versions kept 90 days, and a policy that denies `DeleteObject` to the instance role. Creating the bucket, the IAM change and the spend are the owner's decision; I created nothing.
   - Metadata, records, features and runs: size unmeasured (expected well under 20 GB; measure with `du` after the restore). S3 Standard: **under about $0.50/month**.
   - Deltas (≈ 250 GB, write-once, of which chain ≈ 227 GB and not on the hub): sync once and thereafter only new files (`--size-only`). Glacier IR ≈ **$1/month**, Standard-IA ≈ $3.1/month, Standard ≈ $5.8/month. Upload inside the region is free.
2. **Daily EBS snapshots of the root volume**, 7 daily + 4 weekly: repair the DLM role and tag the volume. A full first snapshot of ≈ 260 GB ≈ **$13/month**, incremental after that (churn-dependent; I would budget up to $20).
3. **Inventory in git**: commit a `docs/inventory/data-<date>.sha256` (path, size, sha256 for every file under `data/`). A restore is only credible if it can be verified. Cost: a few MB.
4. **Restore drill now, before relying on it**: restore one chain delta and one TF record from S3 and verify the hash. Repeat monthly.
5. **A freshness check** that fails loudly if the last successful sync is older than 36 h.
6. The hub is a publication channel, not a backup: its private quota is exhausted (D7 add.) and the zoo repo is public. Resolve the public-versus-D7 deltas conflict listed in section 0 independently of this policy.

Indicative total: **about $15–25/month** for all of it, against a loss that already costs at least 3.2 GPU-h and 8–10 CPU-h plus several results that cannot be recovered.

## 5. Summary

1. **Classification:** of 37 reported result groups, **A = 12**, **B = 21** (15 chain-dependent), **C = 4** (feature-manifest sentence Z3; directory-level facts V5; TF pilot timings T1; D3 TF scoring minutes T5b).
2. **Must be disclosed:** the 358 chain deltas with no bytes and all features derived from them; the original `data/runs/p4/d3v2-*` run directories; the original TF-v1 pilot/G1/G2/D3 per-item records, timing logs, KL cache and qualification JSON; the original local base records; the item-level fast-tier records of the 189 H0 entries (not regenerated). The Z3 feature-manifest sentence must be rewritten, and the V5/T1/T5b numbers may stay only as "reported from logs, artefacts lost".
3. **Minimum regeneration set:** three H0 bases (0.4), seed-1 panel (1.6), TF fingerprint audit (0.8), G1 (0.25), G2 (0.1) = **≈ 3.15 GPU-h**; restore verification (< 1), feature re-extraction (≈ 5), P4 v2 suite (≈ 2–4) = **≈ 8–10 CPU-h**. Extended with full recoverable TF rescoring and the round-2 identity check: ≈ 8.3 GPU-h. Everything runs from clean detached worktrees at commits whose `source_hash` I verified (`dd2a609`, `02c2847`, `846ff84`, `f63a08e`, `f90f591`).
4. **Chain family:** 317 of 675 steps have bytes (254 + 63 aliases); 62 of the 139 evaluated rows are usable end to end. Predecessor-relative fast-tier labels survive; the 229-row TF panel, M3 and the P4 chain rows are only reproducible as recovered-set variants; H2 is not possible (3 evaluated lineages of length ≥ 4). Keep all 675 rows, record loss in a committed sidecar with `artefact_lost: true` for 358, never edit `manifest.jsonl` or `eligibility`; replay is allowed only after a byte-identical acceptance test.
5. **Backup policy required:** nightly versioned, delete-denied S3 copy of all of `data/` except rebuildable caches (metadata < $0.50/month, deltas $1–6/month), daily EBS snapshots with the DLM role repaired and the volume tagged (≈ $13–20/month), a sha256 inventory committed to git, a restore drill, and a 36-hour freshness alarm. Before that: fix the live worktree hole (`perf/eval-throughput`: unguarded and `data` not ignored), move the canonical data out of the checkout, and add a `reference-transaction` guard, because the commit hook does not cover merges.
