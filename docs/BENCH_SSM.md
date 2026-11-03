# Selective scan benchmark

Run: `source scripts/env.sh; OMP_NUM_THREADS=4 python -u scripts/bench_ssm.py`

B=32, T=64, d=256, state=16; three warmups, 20 repetitions. CPU memory is not measured; CUDA peak is allocated memory. Shared host and GPU were heavily loaded, so results are noisy and not an idle-host comparison. mamba_ssm was unavailable at run time.

```text
Command: OMP_NUM_THREADS=4 python scripts/bench_ssm.py
Threads: 4 OMP_NUM_THREADS: 4
 23:54:28 up  1:05,  0 user,  load average: 11.29, 9.53, 12.38
Tue Nov  3 23:54:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.112.07             Driver Version: 595.112.07     CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 4500 Blac...    On  |   00000000:2F:00.0 Off |                    0 |
| N/A   89C    P0            137W /  165W |   29233MiB /  32623MiB |     88%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           45677      C   python                                26874MiB |
|    0   N/A  N/A          105291      C   python                                 1900MiB |
+-----------------------------------------------------------------------------------------+
| device | backend | mode | median ms | IQR ms | peak MiB |
|---|---|---|---:|---:|---:|
| cpu | loop | forward | 1158.55 | 2444.52 | N/A |
| cpu | loop | forward+backward | 304.73 | 892.02 | N/A |
| cpu | chunked | forward | 311.80 | 212.60 | N/A |
| cpu | chunked | forward+backward | 1599.95 | 464.32 | N/A |
mamba_ssm unavailable
| cuda | loop | forward | 5.57 | 0.06 | 8.8 |
| cuda | loop | forward+backward | 22.22 | 6.55 | 107.3 |
| cuda | chunked | forward | 8.58 | 0.13 | 166.3 |
| cuda | chunked | forward+backward | 29.11 | 1.77 | 584.8 |
```
