# StatNN evidence — seed 0, trajectory split

CPU refits using `FeatureDataset`, `FeatureDataset.folds("trajectory", 5)` and `_make_model("statnn", 0)`; no parameter selection. StatNN receives raw pooled stats, exactly as `src/wsp/predictor/audit.py`. Descriptor-only is StandardScaler + Ridge(alpha=1); delta_norm is IsotonicRegression(increasing="auto", out_of_bounds="clip") on effective Frobenius norm. R² is pooled out-of-fold R², not a causal variance decomposition. SD uses ddof=1.

Sources: `src/wsp/predictor/{data,baselines,audit,splits}.py`, `src/wsp/features/{pool,stats}.py`, `data/features/qwen3.5-0.8b/manifest.json`, `data/evals/qwen3.5-0.8b`, `data/zoo/qwen3.5-0.8b/manifest.jsonl`. Manifest has version/layout, not names; names below are decoded from pool.py and DELTA_STAT_NAMES. Bucket b is floor(layer*4/24); non-layer tensors use bucket 0.

Matched rate dataset n=345; zoo manifest SHA256 `6175c3728f03859f54c373bbec79f49564df4a10a3db94ba6db7fd366d986bcd`. KL/PPL are explicitly excluded by FeatureDataset rate checks: these columns extend its rows with the same `read_targets` and `TargetRecords.reference` matching, subtracting matched predecessor/base KL and log(PPL). Missing/nonfinite targets are excluded per fold. No separate unbounded neural head is fitted.

## Pooled out-of-fold performance

| target | n | model | R² | Spearman |
|---|---:|---|---:|---:|
| mmlu_pro | 345 | statnn | 0.555032 | 0.565906 |
| mmlu_pro | 345 | descriptor_only | 0.560316 | 0.523855 |
| mmlu_pro | 345 | delta_norm | 0.060418 | 0.205810 |
| hellaswag | 345 | statnn | 0.588356 | 0.648602 |
| hellaswag | 345 | descriptor_only | 0.629583 | 0.613582 |
| hellaswag | 345 | delta_norm | 0.054375 | 0.178552 |
| kl_to_base | 345 | statnn | 0.475709 | 0.697950 |
| kl_to_base | 345 | descriptor_only | 0.526138 | 0.493122 |
| kl_to_base | 345 | delta_norm | 0.286653 | 0.346153 |
| wikitext_ppl_log | 345 | statnn | 0.733715 | 0.534119 |
| wikitext_ppl_log | 345 | descriptor_only | 0.641898 | 0.548524 |
| wikitext_ppl_log | 345 | delta_norm | -0.010874 | -0.016512 |

## Suite reproduction

mmlu_pro: current n=345, archived n=345, same row IDs=True; archived seed-0 StatNN Spearman=0.565906, refit=0.565906. Brief ≈0.57/0.65 refers to three-seed means, not exact seed-0 values.
hellaswag: current n=345, archived n=345, same row IDs=True; archived seed-0 StatNN Spearman=0.648602, refit=0.648602. Brief ≈0.57/0.65 refers to three-seed means, not exact seed-0 values.

## Gain importances

Sum raw split gains over outer-fold fits, then normalize once per target. Correlated coordinates can share/substitute gain; these are fitted-model allocations. All top-20 entries follow (role, depth bucket, statistic).

### mmlu_pro, n=345

Top 5 share=0.264032; top 20 share=0.503923.

| rank | feature | role | bucket | statistic | gain | share |
|---:|---|---|---|---|---:|---:|
| 1 | down.bucket0.stable_rank | down | 0 | stable_rank | 1.52033 | 0.079289 |
| 2 | o.bucket0.spectral_entropy | o | 0 | spectral_entropy | 1.08989 | 0.056840 |
| 3 | v.bucket0.stable_rank | v | 0 | stable_rank | 1.01146 | 0.052750 |
| 4 | up.bucket3.sv1 | up | 3 | sv1 | 0.749909 | 0.039110 |
| 5 | up.bucket1.stable_rank | up | 1 | stable_rank | 0.691102 | 0.036043 |
| 6 | up.bucket3.abs_max | up | 3 | abs_max | 0.6474 | 0.033764 |
| 7 | o.bucket1.spectral_entropy | o | 1 | spectral_entropy | 0.40785 | 0.021270 |
| 8 | o.bucket2.spectral_entropy | o | 2 | spectral_entropy | 0.381747 | 0.019909 |
| 9 | q.bucket1.sv8 | q | 1 | sv8 | 0.375277 | 0.019572 |
| 10 | o.bucket2.mean | o | 2 | mean | 0.332501 | 0.017341 |
| 11 | global.bucketall.spectral_entropy | global | all | spectral_entropy | 0.303469 | 0.015827 |
| 12 | q.bucket1.mean | q | 1 | mean | 0.276578 | 0.014424 |
| 13 | o.bucket2.stable_rank | o | 2 | stable_rank | 0.271959 | 0.014183 |
| 14 | v.bucket1.stable_rank | v | 1 | stable_rank | 0.247711 | 0.012919 |
| 15 | down.bucket0.sv3 | down | 0 | sv3 | 0.239278 | 0.012479 |
| 16 | q.bucket3.mean | q | 3 | mean | 0.237258 | 0.012374 |
| 17 | down.bucket2.mean | down | 2 | mean | 0.233421 | 0.012174 |
| 18 | q.bucket0.stable_rank | q | 0 | stable_rank | 0.222895 | 0.011625 |
| 19 | v.bucket0.mean | v | 0 | mean | 0.2113 | 0.011020 |
| 20 | o.bucket0.frac_nonzero | o | 0 | frac_nonzero | 0.211147 | 0.011012 |

### hellaswag, n=345

Top 5 share=0.419236; top 20 share=0.625946.

| rank | feature | role | bucket | statistic | gain | share |
|---:|---|---|---|---|---:|---:|
| 1 | o.bucket0.spectral_entropy | o | 0 | spectral_entropy | 0.534477 | 0.163583 |
| 2 | q.bucket0.sv2 | q | 0 | sv2 | 0.385806 | 0.118080 |
| 3 | q.bucket1.sv2 | q | 1 | sv2 | 0.185821 | 0.056873 |
| 4 | q.bucket0.fro | q | 0 | fro | 0.155565 | 0.047612 |
| 5 | o.bucket0.sv2 | o | 0 | sv2 | 0.108109 | 0.033088 |
| 6 | o.bucket1.spectral_entropy | o | 1 | spectral_entropy | 0.0994417 | 0.030435 |
| 7 | o.bucket2.spectral_entropy | o | 2 | spectral_entropy | 0.0980082 | 0.029997 |
| 8 | q.bucket0.sv4 | q | 0 | sv4 | 0.0727378 | 0.022262 |
| 9 | k.bucket0.sv1 | k | 0 | sv1 | 0.0574723 | 0.017590 |
| 10 | q.bucket0.rel_fro | q | 0 | rel_fro | 0.0441426 | 0.013510 |
| 11 | down.bucket0.sv3 | down | 0 | sv3 | 0.0438242 | 0.013413 |
| 12 | o.bucket2.mean | o | 2 | mean | 0.0409146 | 0.012522 |
| 13 | o.bucket3.spectral_entropy | o | 3 | spectral_entropy | 0.0293337 | 0.008978 |
| 14 | q.bucket3.mean | q | 3 | mean | 0.0285492 | 0.008738 |
| 15 | down.bucket0.mean | down | 0 | mean | 0.0282107 | 0.008634 |
| 16 | v.bucket3.abs_max | v | 3 | abs_max | 0.028121 | 0.008607 |
| 17 | q.bucket0.sv1 | q | 0 | sv1 | 0.0277785 | 0.008502 |
| 18 | q.bucket0.sv8 | q | 0 | sv8 | 0.0262854 | 0.008045 |
| 19 | v.bucket1.sv2 | v | 1 | sv2 | 0.0256988 | 0.007865 |
| 20 | q.bucket2.sv1 | q | 2 | sv1 | 0.0248682 | 0.007611 |

### kl_to_base, n=345

Top 5 share=0.342907; top 20 share=0.671360.

| rank | feature | role | bucket | statistic | gain | share |
|---:|---|---|---|---|---:|---:|
| 1 | o.bucket0.sv2 | o | 0 | sv2 | 2.20431 | 0.119890 |
| 2 | k.bucket0.sv3 | k | 0 | sv3 | 1.32492 | 0.072061 |
| 3 | o.bucket0.stable_rank | o | 0 | stable_rank | 1.23323 | 0.067074 |
| 4 | o.bucket0.frac_nonzero | o | 0 | frac_nonzero | 0.8011 | 0.043571 |
| 5 | o.bucket0.spectral_entropy | o | 0 | spectral_entropy | 0.741178 | 0.040312 |
| 6 | o.bucket0.sv4 | o | 0 | sv4 | 0.689134 | 0.037481 |
| 7 | k.bucket0.sv4 | k | 0 | sv4 | 0.621922 | 0.033826 |
| 8 | o.bucket0.sv5 | o | 0 | sv5 | 0.5951 | 0.032367 |
| 9 | o.bucket0.sv3 | o | 0 | sv3 | 0.549086 | 0.029864 |
| 10 | v.bucket2.sv1 | v | 2 | sv1 | 0.540637 | 0.029405 |
| 11 | q.bucket0.sv1 | q | 0 | sv1 | 0.390544 | 0.021241 |
| 12 | o.bucket0.frac_pos | o | 0 | frac_pos | 0.361863 | 0.019681 |
| 13 | up.bucket0.abs_max | up | 0 | abs_max | 0.345028 | 0.018766 |
| 14 | q.bucket0.stable_rank | q | 0 | stable_rank | 0.343404 | 0.018677 |
| 15 | v.bucket0.mean | v | 0 | mean | 0.34087 | 0.018540 |
| 16 | up.bucket1.mean | up | 1 | mean | 0.272375 | 0.014814 |
| 17 | q.bucket0.abs_max | q | 0 | abs_max | 0.266228 | 0.014480 |
| 18 | down.bucket3.stable_rank | down | 3 | stable_rank | 0.243286 | 0.013232 |
| 19 | v.bucket2.sv2 | v | 2 | sv2 | 0.240058 | 0.013056 |
| 20 | down.bucket3.spectral_entropy | down | 3 | spectral_entropy | 0.239428 | 0.013022 |

### wikitext_ppl_log, n=345

Top 5 share=0.349651; top 20 share=0.650018.

| rank | feature | role | bucket | statistic | gain | share |
|---:|---|---|---|---|---:|---:|
| 1 | q.bucket0.sv2 | q | 0 | sv2 | 3.75902 | 0.137293 |
| 2 | v.bucket1.cos_prev_new | v | 1 | cos_prev_new | 1.80127 | 0.065789 |
| 3 | q.bucket1.sv2 | q | 1 | sv2 | 1.36202 | 0.049746 |
| 4 | o.bucket2.abs_max | o | 2 | abs_max | 1.34539 | 0.049139 |
| 5 | q.bucket0.cos_prev_new | q | 0 | cos_prev_new | 1.3056 | 0.047685 |
| 6 | q.bucket2.sv3 | q | 2 | sv3 | 1.00818 | 0.036823 |
| 7 | q.bucket0.sv3 | q | 0 | sv3 | 0.970643 | 0.035451 |
| 8 | q.bucket1.cos_prev_new | q | 1 | cos_prev_new | 0.947777 | 0.034616 |
| 9 | o.bucket0.sv2 | o | 0 | sv2 | 0.62081 | 0.022674 |
| 10 | v.bucket0.stable_rank | v | 0 | stable_rank | 0.5355 | 0.019558 |
| 11 | v.bucket3.abs_max | v | 3 | abs_max | 0.523829 | 0.019132 |
| 12 | q.bucket0.rel_fro | q | 0 | rel_fro | 0.474219 | 0.017320 |
| 13 | q.bucket0.sv4 | q | 0 | sv4 | 0.470866 | 0.017198 |
| 14 | up.bucket0.stable_rank | up | 0 | stable_rank | 0.451841 | 0.016503 |
| 15 | o.bucket0.spectral_entropy | o | 0 | spectral_entropy | 0.420385 | 0.015354 |
| 16 | q.bucket0.fro | q | 0 | fro | 0.414995 | 0.015157 |
| 17 | q.bucket0.frac_nonzero | q | 0 | frac_nonzero | 0.380076 | 0.013882 |
| 18 | down.bucket0.sv4 | down | 0 | sv4 | 0.355771 | 0.012994 |
| 19 | down.bucket3.abs_max | down | 3 | abs_max | 0.325594 | 0.011892 |
| 20 | o.bucket0.sv3 | o | 0 | sv3 | 0.3234 | 0.011812 |

## Family dynamic range

| target | family | n | mean Δy | SD Δy | min | max |
|---|---|---:|---:|---:|---:|---:|
| mmlu_pro | abliteration | 98 | 0.000204 | 0.012922 | -0.040000 | 0.030000 |
| mmlu_pro | chain | 98 | -0.013061 | 0.037730 | -0.140000 | 0.080000 |
| mmlu_pro | lora_sft | 59 | -0.063390 | 0.048800 | -0.190000 | 0.030000 |
| mmlu_pro | random_dir | 60 | 0.003000 | 0.006962 | -0.010000 | 0.020000 |
| mmlu_pro | task_vector | 30 | -0.010333 | 0.034289 | -0.110000 | 0.030000 |
| hellaswag | abliteration | 98 | 0.001939 | 0.004903 | -0.010000 | 0.020000 |
| hellaswag | chain | 98 | 0.007959 | 0.015595 | -0.030000 | 0.050000 |
| hellaswag | lora_sft | 59 | 0.033729 | 0.012581 | 0.010000 | 0.070000 |
| hellaswag | random_dir | 60 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| hellaswag | task_vector | 30 | 0.005333 | 0.015253 | -0.020000 | 0.040000 |

## Within-family refits

Same helper and five trajectory-grouped folds on each family subset; family membership is constant. NaN denotes undefined Spearman when observed or predicted values are constant. Families with fewer than 30 matched rows are excluded.

| target | family | n | Spearman |
|---|---|---:|---:|
| mmlu_pro | abliteration | 98 | 0.272254 |
| hellaswag | abliteration | 98 | -0.118426 |
| kl_to_base | abliteration | 98 | 0.498256 |
| wikitext_ppl_log | abliteration | 98 | 0.190419 |
| mmlu_pro | chain | 98 | 0.463618 |
| hellaswag | chain | 98 | 0.505341 |
| kl_to_base | chain | 98 | 0.447655 |
| wikitext_ppl_log | chain | 98 | 0.350063 |
| mmlu_pro | lora_sft | 59 | 0.708961 |
| hellaswag | lora_sft | 59 | 0.010427 |
| kl_to_base | lora_sft | 59 | 0.890415 |
| wikitext_ppl_log | lora_sft | 59 | 0.685739 |
| mmlu_pro | random_dir | 60 | 0.132717 |
| hellaswag | random_dir | 60 | nan |
| kl_to_base | random_dir | 60 | 0.878633 |
| wikitext_ppl_log | random_dir | 60 | -0.098916 |
| mmlu_pro | task_vector | 30 | 0.222856 |
| hellaswag | task_vector | 30 | 0.705214 |
| kl_to_base | task_vector | 30 | 0.821580 |
| wikitext_ppl_log | task_vector | 30 | 0.678309 |

## mmlu_pro partial dependence

For each fold model, replace one coordinate in its held-out rows with each pooled empirical decile, average predictions over those rows, then pool by held-out row count. This is marginal model behavior under coordinate replacement, including combinations outside joint feature support; it does not establish a causal effect. Repeated quantiles are collapsed.

| target | n | feature | grid value | mean predicted Δy |
|---|---:|---|---:|---:|
| mmlu_pro | 345 | down.0.stable_rank | 0 | -0.014148034 |
| mmlu_pro | 345 | down.0.stable_rank | 1.0003886 | -0.013936158 |
| mmlu_pro | 345 | down.0.stable_rank | 1.1791299 | -0.013793988 |
| mmlu_pro | 345 | down.0.stable_rank | 1.5673656 | -0.020660949 |
| mmlu_pro | 345 | down.0.stable_rank | 1.9343566 | -0.021880211 |
| mmlu_pro | 345 | down.0.stable_rank | 2.4789066 | -0.021541902 |

down.0.stable_rank, target=mmlu_pro, n=345: nonmonotone on sampled grid; largest adjacent change=-0.00686696 between 1.17913 and 1.56737. Grid alone does not locate a unique threshold.

| mmlu_pro | 345 | o.0.spectral_entropy | 0 | -0.014415245 |
| mmlu_pro | 345 | o.0.spectral_entropy | 0.27254806 | -0.014415245 |
| mmlu_pro | 345 | o.0.spectral_entropy | 0.75031444 | -0.014343726 |
| mmlu_pro | 345 | o.0.spectral_entropy | 1.543457 | -0.01745395 |
| mmlu_pro | 345 | o.0.spectral_entropy | 1.7921435 | -0.018040055 |
| mmlu_pro | 345 | o.0.spectral_entropy | 1.9958916 | -0.017425429 |

o.0.spectral_entropy, target=mmlu_pro, n=345: nonmonotone on sampled grid; largest adjacent change=-0.00311022 between 0.750314 and 1.54346. Grid alone does not locate a unique threshold.

| mmlu_pro | 345 | v.0.stable_rank | 0 | -0.014193337 |
| mmlu_pro | 345 | v.0.stable_rank | 1.0892905 | -0.014193337 |
| mmlu_pro | 345 | v.0.stable_rank | 1.3093691 | -0.020328693 |
| mmlu_pro | 345 | v.0.stable_rank | 1.6735125 | -0.018539516 |
| mmlu_pro | 345 | v.0.stable_rank | 2.3527076 | -0.019698628 |

v.0.stable_rank, target=mmlu_pro, n=345: nonmonotone on sampled grid; largest adjacent change=-0.00613536 between 1.08929 and 1.30937. Grid alone does not locate a unique threshold.

## Five-line summary

Branch: `council-evidence`.
Reproduced suite: mmlu_pro n=345 yes=True, ρ=0.565906; hellaswag n=345 yes=True, ρ=0.648602.
Top 5: mmlu_pro (n=345): down.0.stable_rank, o.0.spectral_entropy, v.0.stable_rank, up.3.sv1, up.1.stable_rank; hellaswag (n=345): o.0.spectral_entropy, q.0.sv2, q.1.sv2, q.0.fro, o.0.sv2.
Within-family StatNN Spearman range (finite): mmlu_pro: 0.132717 to 0.708961 (family n: abliteration=98, chain=98, lora_sft=59, random_dir=60, task_vector=30); hellaswag: -0.118426 to 0.705214 (family n: abliteration=98, chain=98, lora_sft=59, random_dir=60, task_vector=30); kl_to_base: 0.447655 to 0.890415 (family n: abliteration=98, chain=98, lora_sft=59, random_dir=60, task_vector=30); wikitext_ppl_log: -0.098916 to 0.685739 (family n: abliteration=98, chain=98, lora_sft=59, random_dir=60, task_vector=30).
StatNN − descriptor-only Spearman gap: mmlu_pro n=345: 0.042050; hellaswag n=345: 0.035019; kl_to_base n=345: 0.204828; wikitext_ppl_log n=345: -0.014405.
