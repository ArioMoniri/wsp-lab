# Lambda spend ledger (cap $150, D4)
| start (UTC) | end | instance | type | $/h | hours | $ | job |
|---|---|---|---|---|---|---|---|
| 2026-11-04T06:13:12Z | 2026-11-04T09:53Z | 1f0d7903c4244c5caa4a835b317b6bd1 | gpu_1x_a100_sxm4 (A100-SXM4-40GB) | 1.99 | 3.66 | 7.29 | heretic_ara 12×10: stopped after 2 entries — adapters quarantined (export bug, see D6 addendum); attempts 1–4 ≈ 2.3 h on env fixes |
| 2026-11-04T10:16:06Z | 2026-11-05T05:05Z | 73522ff7f7c7422dae09f55f8055d9d1 | gpu_1x_a100_sxm4 (us-west-2) | 1.99 | 18.8 | 37.41 | heretic_ara 12×10 rebuild with the fixed exporter: 12 entries uploaded (sync zoo 690134e), self-terminated |
| 2026-11-04T10:58Z | 2026-11-04T11:10Z | 870afd3877cd4156b18b2c87e4c1f252 | gpu_1x_a100_sxm4 (us-west-2) | 1.99 | 0.3 | 0.60 | D3 eval shard lambda0: job failed early and self-terminated before any upload (cause unknown, log lost) |
| 2026-11-04T10:58Z | 2026-11-04T11:10Z | 43a6ba6c71de41d6bbe053b2416a993e | gpu_1x_a100_sxm4 (us-east-1) | 1.99 | 0.3 | 0.60 | D3 eval shard lambda1: same early failure, self-terminated |
| 2026-11-04T11:11Z | 2026-11-04T22:59Z | ab36f9b7d9254ec494cf84737c65a6ff | gpu_1x_h100_pcie (us-west-3) | 3.29 | 11.65 | 38.33 | D3 eval shard lambda0 (170 entries incl. 10 controls, 3 workers) + 20 entries lost by the A10; uploaded, self-terminated |
| 2026-11-04T11:22Z | 2026-11-04T20:17Z | 043d073292d64555869879c37ddb5aca | gpu_1x_a10 (us-east-1) | 1.29 | 8.9 | 11.48 | D3 eval shard lambda1: 55 evaluated, but the rerun with a trimmed ids file uploaded only its 14 ids before self-terminating; 35 results lost and re-evaluated (local + H100); the lost set included 6 of the 10 cross-host control copies, so only 4 A10 control copies plus the base exist |
