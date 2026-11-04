# Lambda spend ledger (cap $150, D4)
| start (UTC) | end | instance | type | $/h | hours | $ | job |
|---|---|---|---|---|---|---|---|
| 2026-11-04T06:13:12Z | 2026-11-04T09:53Z | 1f0d7903c4244c5caa4a835b317b6bd1 | gpu_1x_a100_sxm4 (A100-SXM4-40GB) | 1.99 | 3.66 | 7.29 | heretic_ara 12×10: stopped after 2 entries — adapters quarantined (export bug, see D6 addendum); attempts 1–4 ≈ 2.3 h on env fixes |
| 2026-11-04T10:16:06Z | | 73522ff7f7c7422dae09f55f8055d9d1 | gpu_1x_a100_sxm4 (us-west-2) | 1.99 | | | heretic_ara 12×10 rebuild with fixed exporter (100c28a) |
| 2026-11-04T10:58Z | 2026-11-04T11:10Z | 870afd3877cd4156b18b2c87e4c1f252 | gpu_1x_a100_sxm4 (us-west-2) | 1.99 | 0.3 | 0.60 | D3 eval shard lambda0: job failed early and self-terminated before any upload (cause unknown, log lost) |
| 2026-11-04T10:58Z | 2026-11-04T11:10Z | 43a6ba6c71de41d6bbe053b2416a993e | gpu_1x_a100_sxm4 (us-east-1) | 1.99 | 0.3 | 0.60 | D3 eval shard lambda1: same early failure, self-terminated |
| 2026-11-04T11:11Z | | ab36f9b7d9254ec494cf84737c65a6ff | gpu_1x_h100_pcie (us-west-3) | 3.29 | | | D3 eval shard lambda0 relaunch (170 entries incl. 10 controls); failures keep the instance and upload the log |
| 2026-11-04T11:22Z | | 043d073292d64555869879c37ddb5aca | gpu_1x_a10 (us-east-1) | 1.29 | | | D3 eval shard lambda1 (72 entries incl. 10 controls); failures keep the instance and upload the log |
