# A5 POWER-AND-YIELD report (A5 amendment F7)

**Verdict: 0.8540 >= 0.80 — Part A is decisive.**

The gate (A5.3.2) is read on the registered run only: simulated ordered Part A power (the ordered decision's go rate under the registered .75 sign-agreement alternative) >= 0.80, applied to the simulated rate itself, not its CP lower bound. The all-donor run is a separately labelled, descriptive sensitivity (F7).

Conditional on the registered edit-minus-predecessor proxy and the sign-flip / .75 alternative DGP (F2); not an empirical certification of rotation power. No G3, rotation, ladder or sealed outcome was read.

Contrasts in order (rotation, ladder 0.25, ladder 0.5); 10,000 replicates, 4,000 resolution draws, SeedSequence(20261107).spawn(3) [panels, signs, resolution]; the NULL-SIZE block of each output equals the registered NULL-SIZE run (`data/runs/a5/nullsize/registered.json`) exactly. Exact two-sided 95% Clopper-Pearson intervals in brackets.

| run | donors | ordered power (go) | alternative | unresolved | yield (>= 15 resolved) per contrast | two-sided power per contrast | >= 0.80 |
|---|---|---|---|---|---|---|---|
| `registered` | 61 (F7 pool) | 8540/10000 = 0.8540 [0.8469, 0.8609] | 0/10000 = 0.0000 [0.0000, 0.0004] | 1460/10000 = 0.1460 [0.1391, 0.1531] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | 0.8540 [0.8469, 0.8609]; 0.8525 [0.8454, 0.8594]; 0.8535 [0.8464, 0.8604] | True |
| `sensitivity_all_donors` | 141 (all) | 8501/10000 = 0.8501 [0.8430, 0.8570] | 0/10000 = 0.0000 [0.0000, 0.0004] | 1499/10000 = 0.1499 [0.1430, 0.1570] | 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000]; 1.0000 [0.9996, 1.0000] | 0.8501 [0.8430, 0.8570]; 0.8483 [0.8411, 0.8553]; 0.8490 [0.8418, 0.8560] | True |

Registered donor pool (`registered_donors.json`, ids sha256 `52366da337a570b4e3b850476597fbd8516631f59201a48acb145fc7d23107f1`): per (family:tertile) cell, donor rows / donor components (one representative per component, as `donor_cells`) / components in the full pool:

| cell | donor rows | donor components | full-pool components |
|---|---|---|---|
| LoRA:0 | 1 | 1 | 1 |
| LoRA:1 | 9 | 9 | 9 |
| LoRA:2 | 15 | 15 | 15 |
| chain:0 | 12 | 11 | 26 |
| chain:1 | 9 | 8 | 20 |
| chain:2 | 15 | 14 | 25 |

Per-contrast detail (rotation; ladder 0.25; ladder 0.5):

- `registered`: positive (one-sided agreement) rejection 0.8540 [0.8469, 0.8609]; 0.8525 [0.8454, 0.8594]; 0.8535 [0.8464, 0.8604]; conditional coverage 0.9357; 0.9354; 0.9356; audit/nonfinite pair failures 0, 0, 0.
- `sensitivity_all_donors`: positive (one-sided agreement) rejection 0.8501 [0.8430, 0.8570]; 0.8483 [0.8411, 0.8553]; 0.8490 [0.8418, 0.8560]; conditional coverage 0.9358; 0.9356; 0.9357; audit/nonfinite pair failures 0, 0, 0.

Observations (they do not change the verdict):

- F5.7 reference: the NULL-SIZE pass in both outputs equals the registered NULL-SIZE run: go 180/10000 = 0.0180 [0.0155, 0.0208], alternative (funding-only) 1939/10000 = 0.1939 [0.1862, 0.2018], unresolved 7881/10000. These are null rates, not observed branch outcomes.
- yield 1.0 and zero audit/nonfinite failures follow from the registered run's treatment of 'unavailable' audit evidence as a pass (every donor's audit evidence is 'unavailable'; F2) and from complete windows; under the frozen `unavailable_audit_fails` assumption nothing would resolve (NULL-SIZE report). POWER-AND-YIELD was run for the registered scenario only.
- the three contrasts rescale one donor block (F5.7: one test, not three); their power differs slightly only because prediction signs differ across contrasts for some donors.
- the registered and all-donor runs share the NULL-SIZE pass and its stream state; their POWER-AND-YIELD draws differ from the first replicate because the chain cells hold different donors.

Inputs and outputs:

| run | file | sha256 |
|---|---|---|
| `registered` | `data/runs/a5/nullsize/inputs.json` | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` |
| `registered` | `data/runs/a5/power/fields.json` | `85843ed28d3fd3ba5dc71e2ba8943fd022924cbc4f0b62dddbfb001cb2bcf3b9` |
| `registered` | `docs/prereg/a5_manifests/power/registered_donors.json` | `6e77bca615624a04cb43b3d11a035b5dc31e5a2f1cd918d42a35cc7786babbb5` |
| `registered` | raw output `data/runs/a5/power/power_registered.json` | `2b4806e56d405bdd87c233ed3b247bb0c7a2fe4393083ddcad144e3085424522` |
| `registered` | `scripts/a5_simulation.py` at run time | `04f283d8756bc2d81e2e8bd700163f48d5ceff8487041a8969db48beda8c167f` |
| `sensitivity_all_donors` | `data/runs/a5/nullsize/inputs.json` | `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de` |
| `sensitivity_all_donors` | `data/runs/a5/power/fields.json` | `85843ed28d3fd3ba5dc71e2ba8943fd022924cbc4f0b62dddbfb001cb2bcf3b9` |
| `sensitivity_all_donors` | raw output `data/runs/a5/power/power_all.json` | `aa0af06458cafabbae5dd9364f54f945e1a0f303b10ce3667fafea7be35a4009` |
| `sensitivity_all_donors` | `scripts/a5_simulation.py` at run time | `04f283d8756bc2d81e2e8bd700163f48d5ceff8487041a8969db48beda8c167f` |

Raw outputs (with per-replicate traces) stay under `data/runs/a5/power/` and are not committed; `report.json` holds every number above, the commands, the donor pool and the hashes. The alternative branch is a funding-only stop for the field (F5.7), not a wrong-sign rejection.

| file | sha256 |
|---|---|
| `report.json` | `ff46de30fa8e4ff97a4034e8bdef78cb19bc49662073523500a43c773b55845a` |
| `registered_donors.json` | `6e77bca615624a04cb43b3d11a035b5dc31e5a2f1cd918d42a35cc7786babbb5` |
| `fields_manifest.json` | `19e9de8430ff2a135db848f7d0d5097b5b5724138e9ee815e807e5753e5f9a6b` |
