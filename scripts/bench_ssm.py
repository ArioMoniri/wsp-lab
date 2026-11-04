"""Benchmark selective scan forward and training workloads."""

import os
import statistics
import subprocess
import time

import torch

from wsp.predictor.scan import loop_scan, selective_scan


def main():
    print("Command: OMP_NUM_THREADS=4 python scripts/bench_ssm.py", flush=True)
    print(
        "Threads:",
        torch.get_num_threads(),
        "OMP_NUM_THREADS:",
        os.getenv("OMP_NUM_THREADS"),
        flush=True,
    )
    subprocess.run(["uptime"], check=False)
    subprocess.run(["nvidia-smi"], check=False)
    print("| device | backend | mode | median ms | IQR ms | peak MiB |", flush=True)
    print("|---|---|---|---:|---:|---:|", flush=True)
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        u = torch.randn(32, 64, 256, device=device, requires_grad=True)
        dt = torch.rand_like(u, requires_grad=True)
        a = -torch.rand(256, 16, device=device)
        b = torch.randn(32, 64, 16, device=device)
        c = torch.randn_like(b)
        d = torch.randn(256, device=device)
        backends = [("loop", loop_scan), ("chunked", selective_scan)]
        if device == "cuda":
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                def mamba(u, dt, a, b, c, d):
                    return selective_scan_fn(
                        u.transpose(1, 2).contiguous(),
                        dt.transpose(1, 2).contiguous(),
                        a,
                        b.transpose(1, 2).contiguous(),
                        c.transpose(1, 2).contiguous(),
                        d,
                    ).transpose(1, 2)

                backends.append(("mamba_ssm", mamba))
            except ImportError:
                print("mamba_ssm unavailable", flush=True)
        for name, fn in backends:
            for training in (False, True):

                def run(fn=fn, training=training, u=u, dt=dt, a=a, b=b, c=c, d=d):
                    with torch.set_grad_enabled(training):
                        y = fn(u, dt, a, b, c, d)
                        if training:
                            torch.autograd.grad(y.sum(), (u, dt))

                for _ in range(3):
                    run()
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(20):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    run()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                q = statistics.quantiles(times, n=4)
                peak = (
                    f"{torch.cuda.max_memory_allocated() / 2**20:.1f}"
                    if device == "cuda"
                    else "N/A"
                )
                print(
                    f"| {device} | {name} | {'forward+backward' if training else 'forward'} | {statistics.median(times):.2f} | {q[2] - q[0]:.2f} | {peak} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
