# A5 NULL-SIZE reports (Board A checklist item 6)

**Verdict: PASS** — registered run: ordered go rate and every individual two-sided size <= 0.07.

conditional on the registered edit-minus-predecessor proxy and the sign-flip DGP; not an empirical certification of the rotation null (F2). No G3 outcome was read; no field was computed.

Input `data/runs/a5/nullsize/inputs.json` sha256 `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de`; `--inputs-spec` validation: **valid** ({'inventory': 796, 'eligible': 179, 'donors': 141, 'masked_windows': 0, 'incomplete_pairs': 0, 'unavailable_audit_entries': 423}).

Donor pairs: 141 in 63 components; cluster representatives per (family:tertile) cell {'LoRA:0': 1, 'LoRA:1': 9, 'LoRA:2': 15, 'chain:0': 26, 'chain:1': 20, 'chain:2': 25}; pairs with missing windows 0; eligible rows without a matched pair 38 ({'edit:no_primary_record': 35, 'reference:no_primary_record': 3}).

Contrasts in order (rotation, ladder 0.25, ladder 0.5); 10,000 replicates, 4,000 resolution draws, SeedSequence(20261107).spawn(3) [panels, signs, resolution]. Exact CP 95% intervals in brackets.

| scenario | two-sided size per contrast | ordered go rate | yield (>= 15 resolved) | decisions | gate <= 0.07 |
|---|---|---|---|---|---|
| `registered` (registered) | 0.0372 [0.0336, 0.0411]; 0.0372 [0.0336, 0.0411]; 0.0372 [0.0336, 0.0411] | 0.0180 [0.0155, 0.0208] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | {'go': 180, 'alternative': 1939, 'unresolved': 7881} | True |
| `variance_x2` (sensitivity) | 0.0347 [0.0312, 0.0385]; 0.0347 [0.0312, 0.0385]; 0.0347 [0.0312, 0.0385] | 0.0162 [0.0138, 0.0189] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | {'go': 162, 'alternative': 1884, 'unresolved': 7954} | True |
| `unavailable_audit_fails` (sensitivity) | 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004] | 0.0000 [0.0000, 0.0004] | 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004]; 0.0000 [0.0000, 0.0004] | {'go': 0, 'alternative': 0, 'unresolved': 10000} | True |

A5.3.2's constant-positive-mean and cell-specific mask/noise sensitivities have no frozen scenario in scripts/a5_simulation.py and were not run.

Observations (computed from the runs and traces; they do not change the registered verdict):

- in every replicate of every scenario the three contrasts have identical resolved and agreement counts: under the proxy they are rescalings (a = 1, 0.25, 0.5) of one donor block, so the three individual sizes are one test, not three separate checks
- `unavailable_audit_fails`: no pair resolved in any of 10000 replicates (every audit evidence is 'unavailable' and counted as a failure), so its size 0 and gate value are vacuous: under that assumption Part A cannot resolve at all
- `registered`: under the null the ordered decision returned 'alternative' (ladder 0.25 CP upper <= 0.6) in 1939 of 10000 replicates (0.1939); the registered <= 0.07 gate covers 'go' and the individual two-sided tests, not this route
- `variance_x2`: under the null the ordered decision returned 'alternative' (ladder 0.25 CP upper <= 0.6) in 1884 of 10000 replicates (0.1884); the registered <= 0.07 gate covers 'go' and the individual two-sided tests, not this route
- donor cell(s) LoRA:0 hold a single donor component: every simulated slot of that cell redraws the same donor block (with independent signs); the registered rule requires a non-empty cell only

| file | sha256 |
|---|---|
| `inputs_provenance.json` | `ea160aa4088c56db0aa27046daf3e21eb5865b3950b98e70b4eff9242e1eb1a3` |
| `registered_traces.jsonl.gz` | `34079bb88397386542261bb311720a1509bca76bb53ca3e93f71cabd05582aab` |
| `report.json` | `b6ea5d134790090afeec9ad080531965d3f1b4682df3eb88bc6875d589bf9228` |
| `unavailable_audit_fails_traces.jsonl.gz` | `105ec8aab49b7bb5e9dba5d8034d4c724b62c870252efcd3d9e4a2af2849d95b` |
| `validation.json` | `7a9f72876e73f5ffba2852f925a8aa3ae52a5c9261d55af942b322b1e0606136` |
| `variance_x2_traces.jsonl.gz` | `2c7e497ff72223e86e1c24b1e1691cb1af8a6d2e223d54adc131e1573cdfe582` |

Traces are gzip (mtime 0) JSONL, one line per replicate: donor representatives per slot, component signs, audit/finite and resolved masks, resolved and agreement counts, decision. `report.json` holds the sha256 of each scenario's raw simulation output (kept at `data/runs/a5/nullsize/<scenario>.json`).
