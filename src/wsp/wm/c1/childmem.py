"""The parent's GPU memory around a child process (RULING-wme-c1-bank-halt addendum 6, after the run7 canary OOM).
Engineering; no scientific content.

* :func:`free_cuda` empties the CUDA caching allocator after a garbage collection. ``serve.Runtime.release`` calls it
  only after it has dropped every GPU reference (the R cache, the bank world, the field tensors, the model); in run7
  the cache was emptied first and about 24 GiB stayed reserved by the parent.
* :func:`require` checks and records the parent's free GPU memory immediately before a child process starts
  (``hfchild.spawn``, ``vllmops.spawn``, ``steops.spawn_train``). Below the child's requirement it raises
  :class:`ChildMemoryInsufficient` and the child is not started; the worker returns the error, and the controller
  records it as a run-health incident (``ctl.run_canary`` K2, or the run-health wrapper for a unit).

Requirements: the HF child (canaries, gen_repeat, the HF inert rows) needed about 56.3 GiB at the run7 failure (53.9
GiB held plus a 2.36 GiB allocation), so 60 GiB; the S-TE training child loads the same model with adapters, also 60
GiB; a vLLM child reserves ``gpu_memory_utilization`` of the device. Engineering smoke runs (Qwen3.5-0.8B) need 4 GiB.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

GIB = 2**30
HF_CHILD_GIB = 60.0
TRAIN_CHILD_GIB = 60.0
SMOKE_CHILD_GIB = 4.0
RECORD = "child_memory.json"


class ChildMemoryInsufficient(RuntimeError):
    """The parent's free GPU memory is below the child's requirement: the child is not started."""


def free_info():
    """(free bytes, total bytes) of the current CUDA device, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(int(x) for x in torch.cuda.mem_get_info())
    except Exception:  # noqa: BLE001 - no usable CUDA: nothing to check (CPU tests and tools)
        return None


def reserved_gib() -> float | None:
    try:
        import torch

        return torch.cuda.memory_reserved() / GIB if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def free_cuda() -> None:
    """``gc.collect()`` then ``torch.cuda.empty_cache()`` (call after every GPU reference is dropped)."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require(
    child: str,
    required_gib: float | None = None,
    util: float | None = None,
    info=None,
    workdir=None,
) -> dict:
    """Check and record the parent's free GPU memory before the ``child`` starts: ``required_gib`` GiB, or ``util``
    of the device (a vLLM child). Writes ``<workdir>/child_memory.json``; raises :class:`ChildMemoryInsufficient`
    on a shortfall."""
    info = free_info() if info is None else info
    rec = {"child": child, "t": time.time(), "checked": info is not None}
    if info is not None:
        free, total = info
        req = float(required_gib) * GIB if required_gib is not None else float(util) * total
        rec.update(
            free_gib=free / GIB,
            total_gib=total / GIB,
            required_gib=req / GIB,
            parent_reserved_gib=reserved_gib(),
            **({"util": util} if util is not None else {}),
        )
        rec["pass"] = free >= req
    else:
        rec["pass"] = True
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / RECORD).write_text(json.dumps(rec, indent=1))
    if not rec["pass"]:
        raise ChildMemoryInsufficient(
            f"insufficient free GPU memory for the {child} child: {rec['free_gib']:.2f} GiB free < "
            f"{rec['required_gib']:.2f} GiB required (parent reserved {rec['parent_reserved_gib']}); "
            "the child was not started (RULING-wme-c1-bank-halt addendum 6)"
        )
    return rec
