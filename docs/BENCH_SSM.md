# Selective scan benchmark

Run 2026-11-04: `source scripts/env.sh; OMP_NUM_THREADS=4 python scripts/bench_ssm.py`. B=32, T=64, d=256, state=16; three warmups and 20 repetitions. CUDA peak is allocated memory. The GPU was shared with a running evaluation job; timings are subject to contention. Log: `~/agent-logs/wsp-ssm-bench2.log`.

```text
Command: OMP_NUM_THREADS=4 python scripts/bench_ssm.py
Threads: 4 OMP_NUM_THREADS: 4
 01:13:41 up  2:24,  0 user,  load average: 8.25, 7.61, 7.40
Wed Nov  4 01:13:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.112.07             Driver Version: 595.112.07     CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 4500 Blac...    On  |   00000000:2F:00.0 Off |                    0 |
| N/A   91C    P0            140W /  165W |   28496MiB /  32623MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          156078      C   python                                26340MiB |
|    0   N/A  N/A          203868      C   python                                 2142MiB |
+-----------------------------------------------------------------------------------------+
| device | backend | mode | median ms | IQR ms | peak MiB |
|---|---|---|---:|---:|---:|
| cpu | loop | forward | 11.26 | 0.90 | N/A |
| cpu | loop | forward+backward | 42.70 | 4.46 | N/A |
| cpu | chunked | forward | 92.67 | 1.41 | N/A |
| cpu | chunked | forward+backward | 525.58 | 81.11 | N/A |
| cuda | loop | forward | 5.78 | 0.11 | 8.8 |
| cuda | loop | forward+backward | 18.67 | 1.80 | 107.3 |
| cuda | chunked | forward | 8.98 | 0.13 | 166.3 |
| cuda | chunked | forward+backward | 30.54 | 0.70 | 584.8 |
| cuda | mamba_ssm | forward | 2.44 | 0.11 | 11.5 |
| cuda | mamba_ssm | forward+backward | 2.65 | 0.08 | 17.8 |
```
