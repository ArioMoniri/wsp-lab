# Board A — registered F12 simulation-report acceptance, 14 November 2026

**Verdict: ACCEPT the committed simulation report as the registered gate evidence. The registered gates fail; no F12 purchase is authorised.** Acceptance records a valid negative qualification result, not numerical eligibility. There are no new statistical conditions, thresholds or rescue routes.

This review is confined to branch `review/board-a-f12-sim-report`, based on main `0273ed5`. It applies F12.4–F12.5 and F12a.4–F12a.5 as amended by F12b and F12c, especially F12c.3–5; the C1b review and manifest `16a91fe2ad17a146b215a1a7ab6704873bb3712866868ea4b756b45fdcbfe9c2`; the chair's stage-1 and target/stage-2 acceptances; the external-kill ruling `307aafa`; the Meter.end and projection rulings (`d27d35e`); and notes F12-S1, F12-S2 incident and F12-S2. Source review covered the decision, validity, stress, coverage, selection, execution validator and budget accounting functions. No numerical procedure or input was changed.

## Integrity and execution validation

I recomputed the SHA-256 of the committed bytes below. Report, timing, selection, incident and ledger hashes agree with the registered notes. The part-report hashes in the timing sidecars agree with the assembled report's part provenance; the individual part reports are not committed and were not opened.

| Committed file (under docs/prereg/f12_manifests) | Recomputed SHA-256 |
|---|---|
| `f12s1_stage1_report.json` | `f2e5c386688aa39c75fccbd1796763011c46ba62ac078e2d55c274060689a6bd` |
| `f12c_stage2_selection.json` | `6acc413845fc80c7233ad142325eb0f9768a9af9e62d39ea08980afd9c7d763a` |
| `f12s2_target_report.json` | `d4207e5e3b5384e91903d362d878b0973f1ccd226966fc0d80857f56178bd610` |
| `f12s2_stage2_report.json` | `678f98fa061e77f48fde7f6ef8aeddad7fe63f40dc79ee2f40c0053483859306` |
| `f12s1_stage1_timing.json` | `dcffbd5ee28488a0154556334b7d56e5f58938c2be0dc66152ab08f4615c4abe` |
| `f12s2_target_timing.json` | `46ae4de3701296b38d9b86ccd557777bd112139355302468f1aba3ac930a7f25` |
| `f12s2_stage2_part1_timing.json` | `5f2d681cf38cf0570ef2ae3ce2c242bb42db51c38d5b0c8f57f7f7832751728f` |
| `f12s2_stage2_part2_timing.json` | `994877eef3544c8977122dc9e5d137517ec43fc471575901b1dc29f759b57292` |
| `f12s1_budget_ledger_snapshot.jsonl` | `39b27af999cadd028d0691777e498a8fa3959b45d606ada8243a46e9a1692013` |
| `f12s2_budget_ledger_snapshot_target.jsonl` | `25a08cd1558484d2942f60fa77dd67e46cd78622e674052ba5282045ef74b9c0` |
| `f12s2_budget_ledger_snapshot_final.jsonl` | `fe635816c7cf7ac2efdf5f67109459d25ece4dd47a35b5833b3a58b51819fbeb` |
| `f12s2_incident1_ledger_preclose.jsonl` | `9de0dbc99218f4cb0218aaea1f497a71ebc180405f487bf91704fa163d61e406` |
| `f12s2_incident1_ledger_postclose.jsonl` | `87140fab4aca796894f8fc1f58a99807d2abc36ca8b3e6bed3a674238c62026e` |
| `f12s2_incident1_journal_excerpt.txt` | `00bc2aa07ea4633c358d9b76ec9f0f59945803236c33274c00c7162d9d65a58f` |

I independently ran `verify_selection` on the committed selection and stage-1 report, then `validate_execution` on the committed assembled stage-2 report, with the five committed prerequisites: `f12f_noise_check.json`, `f12f_pilot_report.json`, `f12f_pilot_timing.json`, `f12_pilot_acceptance.json`, and `f12s2_target_report.json`; prerequisite SHA-256 bindings were supplied through `_parse_prereqs`. `c1_root` was this worktree. This invokes `f12_c1.verify_c1`, checking the committed C1 chain, current code hashes and separate Board A verification record. Output:

```json
{"passed": true, "failures": [], "board_a_acceptance": "pending (not checkable by code)"}
```

The pending string correctly describes the validator's inability to issue this board's acceptance. The numerical release remains `eligible: false`. The validator recomputes the complete summary from records and checks schedules, planned IDs, seeds, frozen selection, identities, independent target, noise check and chair-accepted pilot/projection prerequisites. I additionally recounted the 14 gate events directly from records and independently calculated their beta-quantile bounds with `scipy.stats.beta.ppf`, asserting agreement with the report's counts, bounds and ordinary 95% CP intervals. The invalid alternative is a gamma-calibration failure, retained as a power failure, never removed from its denominator.

| Run | Planned/observed | Valid | Invalid |
|---|---|---|---|
| Stage 1 | 480/480; 40 stress + 40 alternative at each of 0, 29, 58, 87, 116, 145; r=1000–1039 | 480 | 0 |
| Independent target | 200/200 stress at 145; r=2000–2199 | 200 | 0 |
| Stage 2 stress | 1000/1000 at 145; r=0–999 | 1000 | 0 |
| Stage 2 alternative | 1000/1000 at 145; r=0–999 | 999 | 1 |
| Stage 2 variance-×2 | 200/200 at 145; r=0–199 | 200 | 0 |

Stage 2 is complete: 2,200 planned IDs, each once; 2,199 valid. The registered amended studentiser, not strict unamended A5.4.2, governs validity. In the stress records the strict procedure would be valid for 0/1000 C_WM and 0/1000 C_WM,S; the registered amendment yields 1000/1000 for both. This is disclosed diagnostic evidence, not an additional validity gate or grounds to substitute a studentiser after seeing results.

Git history establishes the chronology: stage-1 report `46e6276` (12 Nov 21:49:35 UTC), then selection `01421d8` (21:54:05), then target launch (22:06:09 timing), target report `ec790fe` (13 Nov 01:07:45), and assembled stage 2 `22cbd01` (14 Nov 18:50:54). The first stage-2 attempt began on 13 Nov at 01:07:53; its full rerun began at 07:10:50. Thus the report preceded selection, and committed selection preceded every target/stage-2 replicate. No stage-1 candidate met the selection criterion: the six alternative counts were 1, 4, 10, 12, 10, 8 of 40; maximum stress counts were 22, 23, 13, 14, 17, 12 of 40. Selection of 145 is the explicitly registered fallback, not a claim that stage 1 passed.

## The 14 registered slots

Each one-sided allocation is exactly α = 0.05/36 = 0.001388888888888889. For x/n, L = Beta⁻¹(α; x,n−x+1), U = Beta⁻¹(1−α; x+1,n−x), with endpoints 0/1 as appropriate. The table gives both one-sided endpoints (rounded for display) and ordinary two-sided 95% CP intervals. Decisions use unrounded values: power requires L ≥ 0.80; every other row requires U ≤ 0.07, equality passing. These are 14 allocated one-sided requirements, not 28 new requirements. All denominators are planned, including invalids.

| Registered requirement | Count/planned | L at α | U at α | Ordinary CP95 | Result |
|---|---:|---:|---:|---|---|
| Ordered power: correction detected | 218/1000 | 0.180323720475 | 0.259280901499 | [0.192770023, 0.244901799] | FAIL |
| Stress: C_WM conditional lower > 0 | 343/1000 | 0.298705148006 | 0.389297351645 | [0.313576449, 0.373351323] | FAIL |
| Stress: C_WM joint lower > 0 | 176/1000 | 0.141611308341 | 0.214535893167 | [0.152878290, 0.201046513] | FAIL |
| Stress: C_WM_S conditional lower > 0 | 403/1000 | 0.356836113779 | 0.450400520396 | [0.372422854, 0.434149956] | FAIL |
| Stress: C_WM_S joint lower > 0 | 200/1000 | 0.163652100413 | 0.240184635368 | [0.175620575, 0.226159373] | FAIL |
| Stress: correction detected | 110/1000 | 0.082428881874 | 0.142582088519 | [0.091277774, 0.131052464] | FAIL |
| Coverage lower-tail: C_WM conditional | 315/1000 | 0.271843641245 | 0.360516743490 | [0.286289721, 0.344804031] | FAIL |
| Coverage lower-tail: C_WM joint | 165/1000 | 0.131589213647 | 0.202700755761 | [0.142507318, 0.189484208] | FAIL |
| Coverage lower-tail: C_WM_S conditional | 285/1000 | 0.243268553805 | 0.329475773723 | [0.257188696, 0.314083234] | FAIL |
| Coverage lower-tail: C_WM_S joint | 146/1000 | 0.114412407927 | 0.182125061761 | [0.124682038, 0.169425375] | FAIL |
| Coverage upper-tail: C_WM conditional | 184/1000 | 0.148932742451 | 0.223110827135 | [0.160442283, 0.209434133] | FAIL |
| Coverage upper-tail: C_WM joint | 116/1000 | 0.087699952246 | 0.149231551033 | [0.096805955, 0.137486751] | FAIL |
| Coverage upper-tail: C_WM_S conditional | 102/1000 | 0.075444182064 | 0.133673201337 | [0.083935460, 0.122445151] | FAIL |
| Coverage upper-tail: C_WM_S joint | 86/1000 | 0.061644213701 | 0.115687979617 | [0.069362531, 0.105120268] | FAIL |

All 14 requirements fail. The five stress bounds are conservative operational safety requirements under the linear-signal / ridge-bias stress scenario, not type-I error certification at C=0. No controlled false-detection claim at a no-correction boundary follows.

The four lower-tail coverage failures block purchase independently of power and stress. The four upper-tail failures block the registered upper-bound labels; they are not additional purchase gates. Target C_WM = 0.018797137089, SE = 0.003856523028, band [0.011084091032, 0.026510183145]; target C_WM,S = 0.028662276377, SE = 0.003521721060, band [0.021618834258, 0.035705718496]. These are ratios of expected row sums from all 200 independent target panels, never stage-2 estimates. Lower-tail errors mean lower bound > target−2SE; upper-tail errors mean upper bound < target+2SE. Invalids count in both tails and never cover.

For completeness, descriptive coverage of the point target across all six A4.4 procedures follows (each /1000). These supplementary coverage counts create no additional gate:

| Interval | C_WM covers target | C_WM,S covers target |
|---|---:|---:|
| conditional_bootstrap_t | 570/1000 | 675/1000 |
| joint_bootstrap_t | 743/1000 | 788/1000 |
| conditional_bca | 570/1000 | 675/1000 |
| joint_bca | 629/1000 | 706/1000 |
| conditional_percentile | 572/1000 | 677/1000 |
| joint_percentile | 905/1000 | 963/1000 |

## Registered result and descriptive contrasts

The registered two-comparator decision did not meet its power qualification under this alternative; it does not establish absence of a K+F correction.

F12b.4(d) therefore gives **“unresolved — not powered”**. F12c.3(d) also gives **“unresolved — not qualified”** because lower-tail qualification failed. The release correctly records both failures and the stress failure: no purchase, no other k, no extension, no repeat, no new target run. No alternative rows, model search, removal of StatNN, restricted h* or second alternative is authorised. Marginal results provide no separate purchase route. No adequate-power claim against a WM-over-StatNN margin is permitted.

C_WM = 1−A_WM/A_F3 and C_WM,S = 1−A_WM/A_S. The following learned contrasts are ratios of sums over replicates with both contrasts valid, not averages of replicate ratios. Marginal detection requires point C ≥ 0.05 and both bootstrap-t lower bounds > 0 for that contrast alone, over the planned denominator.

| Scenario | Valid/planned (ratio conditioning) | C_WM | C_WM,S | C_S:F3 | Marginal WM:F3 | Marginal WM:StatNN | Ordered detection |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stress | 1000/1000 | 0.020979896833 | 0.031090516185 | -0.010435050456 | 156/1000 | 150/1000 | 110/1000 |
| Registered alternative | 999/1000 | 0.171599965519 | 0.045419278357 | 0.132184407563 | 424/1000 | 230/1000 | 218/1000 |
| Variance-×2 (descriptive) | 200/200 | 0.002357190207 | 0.023280836208 | -0.021422376848 | 14/200 | 20/200 | 12/200 |

The alternative is the registered token-function alternative calibrated against F3. Of 1,000 calibrations, 999 qualified; their validation C median is 0.150386593716, range [0.133718895353, 0.166748408861], within ±0.02. The aggregate valid-only oracle:F3 ratio is 0.154157147560. The tree's 0.132184407563 gain over F3 and the smaller WM:StatNN margin/rate support the tree being one contributor to the conjunction's low power. They do not justify assigning all failure to it: WM:F3 marginal detection itself is only 424/1000, and calibration/uncertainty also matter.

Variance-×2 is descriptive only: 200/200 valid, 12/200 ordered detections, 47/200 provisional upper-bound codes and 141/200 unresolved. Its lower-bound-positive counts are WM conditional 35/200, joint 16/200; WM:StatNN conditional 62/200, joint 35/200. These neither rescue nor fail a purchase gate. No coverage target is registered for the alternative or variance-×2; their coverage is not computed or claimed. Provisional upper-bound codes (179/1000 stress, 40/1000 alternative, 47/200 variance-×2) are not qualified scientific “none detectable” labels: power and upper-tail qualification failed.

## Execution-event rulings

1. **External kill and full rerun: ACCEPT.** The committed journal supports the service stop/restart at 06:57:57–58 UTC on 13 Nov during the unattended glibc upgrade. Under `307aafa`, part 1's full rerun recomputes the same planned IDs under fixed per-replicate noise/training streams, configuration, inputs, selection and prerequisites. It introduces no new draws or choice between results. The killed attempt wrote no report; its directory was preserved and not read, as attested by the operator/chair. I did not open that directory or progress log. The assembled report contains exactly the registered unique schedule; no partial-run decision is accepted. This is restoration after an external interruption, not a repeat-until-pass.
2. **Closure: ACCEPT `failed_external`.** I verified that the 118-event pre-close ledger is a byte prefix of the 119-event post-close ledger. The sole appended end event retains attempt 1, 1,200 planned units, last checkpoint 400 units and 41,631.02 CPU-s. The final snapshot retains this failed attempt and the separate completed 1,200-unit attempt 2. The failed attempt's CPU is charged (11.564172 CPU-h); its 400 units do not count as completed work. CPU between the last checkpoint and kill remains unrecorded, as disclosed; 102.0095 CPU-h is the recorded total, not an exact total including that gap.
3. **Timestamp correction: ACCEPT.** The actual appended event carries `2026-11-13T07:01:55Z`, not the incident note's “about 07:05 UTC”. The F12-S2 correction is supported by the committed event and changes no replicate, ordering or charge.
4. **Meter.end finding: CONFIRMED.** It double-adds the current attempt to stored end-event totals and leaves completed work in the remaining projection. Stored end-event `spent_cpu_h` and `projected_total_cpu_h` are not authoritative. Live `Meter._check` uses `Ledger.attempts()` maxima once per attempt through `projection()`; no stored end total is read back. Recomputed snapshot spending is 13.722130555556 CPU-h after stage 1, 19.577138888889 after target and 102.0095 at the final 345-event snapshot. Final projection is 102.039254277990 CPU-h including 0.029754277990 overhead, below 190.
5. **Projection finding: CONFIRMED.** `plan_from_pilot_acceptance` omits `units_by_kind`; the live part projection divides the whole stage-2 item by the part's own units, overstating the planned per-unit rate (about 302 versus 137 CPU-s for part 2). This is conservative/transient, not numerical underpricing or evidence of altered work. The disclosed launch peak 149.66 CPU-h remained below 190; the final authoritative spending/projection agrees with F12-S2. Treat running-part projected totals as upper bounds, and use spending plus measured remaining work for estimates. Both fixes with tests remain C2 obligations under the registered identity/restart rules; no pinned code is changed here.

None of these events invalidates the committed report as gate evidence. The timing sidecars show two workers, about 13.335 CPU-h stage 1, 5.855 target, 34.535 completed part 1 and 36.331 part 2; registered notes record tree RSS peaks 2.87, 2.36, 2.44 and 2.44 GB respectively, within 4 GB. Sidecar worker/parent peaks are not substituted for whole-tree peaks. Assembly was about seven seconds and not memory-sampled; that disclosed limitation is not a newly invented rejection rule. This acceptance relies on committed evidence for historical execution and is not a fresh simulation, raw-part reconstruction or audit of undocumented host behaviour.

## Authority and exposure

Acceptance authorises use of this exact committed report as the registered gate evidence and publication of its failure and descriptive results with the registered limitations. It does not authorise purchase, scoring, release of Part A, or a remedy experiment. Passing the gates would still not waive the chair's acceptance, C2, exposure rules, before-scheduling forecasts/immutable decisions, endpoint/reference/configuration pins or GPU caps. Here the gates fail, so satisfying administrative prerequisites cannot authorise this purchase. The separate F12-I registration may operate only within its existing own C1/C2, qualification/no-op and dependency rules (including its already registered full-not-on-file provision), not as an invented rescue of F12. Its 56.16 GPU-min ceiling remains; F12's independent 3 GPU-h scoring cap includes base/reference work, failures and retries. All other separate registered caps remain binding. Scoring order stays **Part A → F8-E → F8-F → F9 → F12 → F12-I**. Part A remains held by the programme owner. No A5.5, F5.5, funding, G7/G8 or Part B authority follows.

Exposure attestation, **this reviewer session only**:

- (i) Old development outcomes: I opened no old development outcome files or measured per-edit outcomes. I read the requested registration and historical review narratives, including their exposure disclosures; those do not establish an outcome-free history for any other session.
- (ii) New outcomes and `data/evals`: none. I did not open or traverse anything under `data/evals`, score a model or inspect Part A/F8/F9 outcomes. The validator run included an audit hook refusing resolved eval-path opens. I read the committed simulated stage-1, target and stage-2 results for this acceptance review; validator prerequisite decoding also parsed the simulated pilot. Reading simulated results is **not an outcome exposure**. This statement discloses simulated-record access and does not pretend it was metadata-only.
- (iii) Prescore audits: I opened no prescore-audit result files and performed no prescore audit or choice based on one. I read only the requested registration/review narrative descriptions of historical audit access and qualifications. I attest only to this session, not prior agents, operators or sessions.

## Review checks and commit scope

`scripts/secret_scan.sh` ran first and returned `secret_scan: clean`; it is run again before commit. Review computations used the worktree Python with OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2 and MKL_NUM_THREADS=2; no parallel pytest, model fit, GPU job, install, full simulation or heavy build was run. All review computations were short, within the 4 GB limit. The execution validator and independent event/bound checks passed as integrity checks; statistical gate failures are the result, not test failures. An auxiliary ledger-display command initially failed to JSON-serialize tuple dictionary keys; it was corrected to iterate the entries, and all ledger/prefix/timestamp checks completed. No required review check remains unrun. Historical uncommitted part reports and the unsampled assembly memory peak were not independently reconstructed.

The full-suite pre-commit hook is bypassed for this doc-only commit under the user's explicit permission. No full suite was run or claimed. Only this review document is staged and committed, with a plain commit message and no attribution lines; no merge or push.
