"""Compare selective scan implementations on available devices."""

import time

import torch

from wsp.predictor.scan import selective_scan


def loop(u, dt, a, b, c, d):
    h = u.new_zeros(u.shape[0], u.shape[2], a.shape[1])
    ys = []
    for t in range(u.shape[1]):
        h = (dt[:, t, :, None] * a).exp() * h + dt[:, t, :, None] * b[:, t, None, :] * u[
            :, t, :, None
        ]
        ys.append((h * c[:, t, None, :]).sum(-1) + d * u[:, t])
    return torch.stack(ys, 1)


def main():
    print("| device | loop ms | chunked ms |")
    print("|---|---:|---:|")
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        u = torch.randn(32, 64, 512, device=device)
        dt = torch.rand_like(u) * 0.1
        a = -torch.rand(512, 16, device=device)
        b = torch.randn(32, 64, 16, device=device)
        c = torch.randn_like(b)
        d = torch.randn(512, device=device)
        times = []
        for fn in (loop, selective_scan):
            with torch.no_grad():
                fn(u, dt, a, b, c, d)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                fn(u, dt, a, b, c, d)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        print(f"| {device} | {times[0]:.2f} | {times[1]:.2f} |")


if __name__ == "__main__":
    main()
