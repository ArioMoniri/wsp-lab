#!/usr/bin/env python3
"""Local-GPU smoke test of ``serve.Runtime.release`` (RULING-wme-c1-bank-halt addendum 6). Engineering only: no model,
no study data. A stand-in GPU tensor of ``--gib`` GiB is placed where the R cache lives (with a smaller one in the
field's kept-factor cache), ``release()`` is called, and the process's reserved CUDA memory must return to its
baseline. For contrast it first replays the run7 order (cache emptied BEFORE the stand-ins are dropped), whose
reserved memory does not return. Refuses unless the device has at least ``--gib`` + 2 GiB free (another job may share
the GPU); the peak use is ``--gib`` + 0.25 GiB.

    wme_c1_release_smoke.py --gib 4 --out OUT.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GIB = 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from wsp.wm.c1 import serve as S

    assert torch.cuda.is_available(), "no CUDA device"
    free0, total = torch.cuda.mem_get_info()
    need = (a.gib + 2.0) * GIB
    if free0 < need:
        raise SystemExit(f"REFUSED: {free0 / GIB:.2f} GiB free < {need / GIB:.2f} GiB (another job uses the GPU)")
    n = int(a.gib * GIB)
    rep = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "total_gib": total / GIB,
        "free_before_gib": free0 / GIB,
        "standin_gib": a.gib,
        "field_standin_gib": 0.25,
    }
    torch.cuda.synchronize()
    base = torch.cuda.memory_reserved()
    rep["baseline_reserved_gib"] = base / GIB

    class Holder:
        pass

    class FieldStandin:
        def __init__(self):
            self._kept = {"k": torch.empty(GIB // 4, dtype=torch.uint8, device="cuda")}
            self._grad = {}

    def runtime():
        rt = S.Runtime.__new__(S.Runtime)
        rt.cfg, rt.m, rt._bank = None, None, None
        rt._rc = Holder()
        rt._rc.t = torch.empty(n, dtype=torch.uint8, device="cuda")  # the R-cache stand-in
        rt._field = FieldStandin()
        rt._rc.field = rt._field  # as F.RCache holds the field
        return rt

    # (1) the run7 order: empty the cache first, then drop the references (the reserved memory stays)
    rt = runtime()
    torch.cuda.synchronize()
    rep["run7_order"] = {"after_alloc_reserved_gib": torch.cuda.memory_reserved() / GIB}
    gc.collect()
    torch.cuda.empty_cache()
    rt._rc = rt._bank = None
    rt._field._kept.clear()
    rt._field._grad.clear()
    del rt
    torch.cuda.synchronize()
    rep["run7_order"]["after_release_reserved_gib"] = torch.cuda.memory_reserved() / GIB
    gc.collect()
    torch.cuda.empty_cache()  # clean up before the v2x check
    torch.cuda.synchronize()
    rep["between_reserved_gib"] = torch.cuda.memory_reserved() / GIB
    # (2) v2x: Runtime.release() (drop every reference, gc, empty_cache)
    rt = runtime()
    torch.cuda.synchronize()
    rep["v2x"] = {"after_alloc_reserved_gib": torch.cuda.memory_reserved() / GIB}
    t0 = time.time()
    rt.release()
    torch.cuda.synchronize()
    rep["v2x"]["release_seconds"] = time.time() - t0
    rep["v2x"]["after_release_reserved_gib"] = torch.cuda.memory_reserved() / GIB
    rep["v2x"]["after_release_allocated_gib"] = torch.cuda.memory_allocated() / GIB
    rep["free_after_gib"] = torch.cuda.mem_get_info()[0] / GIB
    rep["pass"] = (
        rep["v2x"]["after_alloc_reserved_gib"] >= rep["baseline_reserved_gib"] + a.gib
        and rep["v2x"]["after_release_reserved_gib"] == rep["baseline_reserved_gib"]
    )
    rep["run7_order_retained_gib"] = (
        rep["run7_order"]["after_release_reserved_gib"] - rep["baseline_reserved_gib"]
    )
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["pass"] else 2)


if __name__ == "__main__":
    main()
