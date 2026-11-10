# A5 NULL-SIZE reports (Board A checklist item 6)

**Verdict: PASS** — registered run: ordered go rate and every individual two-sided size <= 0.07.

conditional on the registered edit-minus-predecessor proxy and the sign-flip DGP; not an empirical certification of the rotation null (F2). No G3 outcome was read; no field was computed.

Input `data/runs/a5/nullsize/inputs.json` sha256 `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de`; `--inputs-spec` validation: **valid** ({'inventory': 796, 'eligible': 179, 'donors': 141, 'masked_windows': 0, 'incomplete_pairs': 0, 'unavailable_audit_entries': 423}).

Donor pairs: 141 in 63 components; cluster representatives per (family:tertile) cell {'LoRA:0': 1, 'LoRA:1': 9, 'LoRA:2': 15, 'chain:0': 26, 'chain:1': 20, 'chain:2': 25}; pairs with missing windows 0; eligible rows without a matched pair 38 ({'edit:no_primary_record': 35, 'reference:no_primary_record': 3}).

Contrasts in order (rotation, ladder 0.25, ladder 0.5); 10,000 replicates, 4,000 resolution draws, SeedSequence(20261107).spawn(3) [panels, signs, resolution]. Exact CP 95% intervals in brackets.

| scenario | two-sided size per contrast | ordered go rate | alternative (funding-only) | yield (>= 15 resolved) | decisions | gate <= 0.07 |
|---|---|---|---|---|---|---|
| `registered` (registered) | 0.0372 [0.0336, 0.0411]; 0.0372 [0.0336, 0.0411]; 0.0372 [0.0336, 0.0411] | 0.0180 [0.0155, 0.0208] | 1939 / 10000 = 0.1939 [0.18619, 0.20179] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | {'go': 180, 'alternative': 1939, 'unresolved': 7881} | True |
| `variance_x2` (sensitivity) | 0.0347 [0.0312, 0.0385]; 0.0347 [0.0312, 0.0385]; 0.0347 [0.0312, 0.0385] | 0.0162 [0.0138, 0.0189] | 1884 / 10000 = 0.1884 [0.18078, 0.19620] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | {'go': 162, 'alternative': 1884, 'unresolved': 7954} | True |
| `unavailable_audit_fails` (sensitivity) | 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004] | 0.0000 [0.0000, 0.0004] | 0 / 10000 = 0.0000 [0.00000, 0.00037] | 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004] | {'go': 0, 'alternative': 0, 'unresolved': 10000} | True |
| `constant_positive_mean` (sensitivity) | 0.0406 [0.0368, 0.0447]; 0.0406 [0.0368, 0.0447]; 0.0406 [0.0368, 0.0447] | 0.0191 [0.0165, 0.0220] | 2177 / 10000 = 0.2177 [0.20964, 0.22592] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | {'go': 191, 'alternative': 2177, 'unresolved': 7632} | True |
| `cell_mask_noise` (sensitivity) | 0.0307 [0.0274, 0.0343]; 0.0307 [0.0274, 0.0343]; 0.0307 [0.0274, 0.0343] | 0.0155 [0.0132, 0.0181] | 1544 / 10000 = 0.1544 [0.14737, 0.16163] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | {'go': 155, 'alternative': 1544, 'unresolved': 8301} | True |

all A5.3.2/F2 sensitivities are frozen scenarios of scripts/a5_simulation.py and were run at registered scale (10,000 replicates, 4,000 resolution draws, SeedSequence 20261107): variance_x2 and unavailable_audit_fails (8 Nov 2026), constant_positive_mean and cell_mask_noise (10 Nov 2026, Board A freeze review edit 2). Earlier raw outputs are kept unchanged.

Scenario definitions:

- `variance_x2`: centred donor noise doubled, means unchanged (variance inflation; F2, 7 Nov 2026).
- `unavailable_audit_fails`: 'unavailable' audit evidence counted as a failure (F2, 7 Nov 2026).
- `constant_positive_mean`: A5.3.2 constant positive donor means (constant-sign family effect), frozen 10 Nov 2026: every donor mean is replaced by its family's positive constant, the mean |donor_mean| over that family's unique donor-cell representatives (the registered alternative's |m_i| scale averaged within family; payload values only); centred noise, masks, audits, selection, slots and every random stream are unchanged. Under the fair-coin null a constant positive mean is equivalent to a fixed positive truth with independent fair-coin prediction signs. Parameters: `{"family_constant_means": {"LoRA": 0.08095652446150779, "chain": 0.02378677870606033}}`.
- `cell_mask_noise`: A5.3.2 cell-specific masks/noise (norm-dependent resolution), frozen 10 Nov 2026 for the single-donor LoRA low-tertile cell: its ten slots are cut to the registered minimum of one active slot per donor component (a non-empty cell, i.e. 1; the first slot in slot order), the other nine are masked as missing in all three contrasts with no replacement, and that cell's centred noise is doubled. Board A's freeze review names no cell-specific noise scale, so the chair's fallback (noise x2 in that cell only) is used. All other cells, means, audits and random streams are unchanged. Parameters: `{"active_slots": 31, "cell_overrides": [{"active_slots": 1, "cell": ["LoRA", 0], "donor_components": 1, "noise_scale": 2.0, "slots": 10}]}`.

Alternative branch: the ordered decision's 'alternative' branch (>= 15 resolved at ladder 0.25 and CP upper <= 0.6) is reported per scenario with its exact two-sided 95% Clopper-Pearson interval. It is a funding-only stop for the field (F5.7), not a wrong-sign rejection: under this fair-coin null (p = .5) agreement lies inside the region that rule stops funding, so its null rate is not a type-I error and the registered <= 0.07 gates (ordered go rate and each individual two-sided size) do not apply to it. No decisive negative (exclusion of useful agreement) inference is registered; it would need its own boundary-null p = .60 calibration (false-alternative rate <= 0.07) and power before outcomes (Board A freeze review edit 2).

Observations (computed from the runs and traces; they do not change the registered verdict):

- in every replicate of every scenario the three contrasts have identical resolved and agreement counts: under the proxy they are rescalings (a = 1, 0.25, 0.5) of one donor block, so the three individual sizes are one test, not three separate checks
- `unavailable_audit_fails`: no pair resolved in any of 10000 replicates (every audit evidence is 'unavailable' and counted as a failure), so its size 0 and gate value are vacuous: under that assumption Part A cannot resolve at all
- `registered`: under the null the ordered decision returned 'alternative' (ladder 0.25 CP upper <= 0.6) in 1939 of 10000 replicates (0.1939, exact CP 95% [0.18619, 0.20179]); this is the funding-only branch, not a wrong-sign rejection, and the registered <= 0.07 gate covers 'go' and the individual two-sided tests, not this route
- `variance_x2`: under the null the ordered decision returned 'alternative' (ladder 0.25 CP upper <= 0.6) in 1884 of 10000 replicates (0.1884, exact CP 95% [0.18078, 0.19620]); this is the funding-only branch, not a wrong-sign rejection, and the registered <= 0.07 gate covers 'go' and the individual two-sided tests, not this route
- `constant_positive_mean`: under the null the ordered decision returned 'alternative' (ladder 0.25 CP upper <= 0.6) in 2177 of 10000 replicates (0.2177, exact CP 95% [0.20964, 0.22592]); this is the funding-only branch, not a wrong-sign rejection, and the registered <= 0.07 gate covers 'go' and the individual two-sided tests, not this route
- `cell_mask_noise`: under the null the ordered decision returned 'alternative' (ladder 0.25 CP upper <= 0.6) in 1544 of 10000 replicates (0.1544, exact CP 95% [0.14737, 0.16163]); this is the funding-only branch, not a wrong-sign rejection, and the registered <= 0.07 gate covers 'go' and the individual two-sided tests, not this route
- `cell_mask_noise`: cell LoRA:0 keeps 1 of 10 slots (1 donor component); the other 9 are masked in every replicate, so 90,000 of its audit/nonfinite failures per contrast are these masks, not audit outcomes
- donor cell(s) LoRA:0 hold a single donor component: every simulated slot of that cell redraws the same donor block (with independent signs); the registered rule requires a non-empty cell only

| file | sha256 |
|---|---|
| `cell_mask_noise_traces.jsonl.gz` | `ea79b0c563ff2b0f949a2c608821815e04c980ab331a56b8242c56a1b831d90d` |
| `constant_positive_mean_traces.jsonl.gz` | `2c691cfdcb09e9698eef771b5e4de292253d3c8f299bf3cbcba46182cba8801d` |
| `inputs_provenance.json` | `ea160aa4088c56db0aa27046daf3e21eb5865b3950b98e70b4eff9242e1eb1a3` |
| `registered_traces.jsonl.gz` | `34079bb88397386542261bb311720a1509bca76bb53ca3e93f71cabd05582aab` |
| `report.json` | `1bed2c9531e4230864a1ebbe7e4023e0ffafcf704a59c40e93f9a75bd469dfa6` |
| `unavailable_audit_fails_traces.jsonl.gz` | `105ec8aab49b7bb5e9dba5d8034d4c724b62c870252efcd3d9e4a2af2849d95b` |
| `validation.json` | `7a9f72876e73f5ffba2852f925a8aa3ae52a5c9261d55af942b322b1e0606136` |
| `variance_x2_traces.jsonl.gz` | `2c7e497ff72223e86e1c24b1e1691cb1af8a6d2e223d54adc131e1573cdfe582` |

Traces are gzip (mtime 0) JSONL, one line per replicate: donor representatives per slot, component signs, audit/finite and resolved masks, resolved and agreement counts, decision. `report.json` holds the sha256 of each scenario's raw simulation output (kept at `data/runs/a5/nullsize/<scenario>.json`).
