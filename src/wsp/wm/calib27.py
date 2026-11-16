"""Qwen3.8-27B engineering calibration (SYNTHESIS-r9 C9; round-9 pro-engineer L1-L4). ENGINEERING ONLY.

Pure helpers shared by ``scripts/wme_27b_calib.py`` (runs on the Lambda GPU) and the report: the pinned target,
the 4-layer support group, hash-only result records (objective values are never reported as numbers), the
Go-M/Go-N/Go-C gates at the 27B caps, the D8 bank-hours formula and the k-GPU data-parallel projection.
"""

from __future__ import annotations

import hashlib
import json
import math

from wsp.wm import panel as P

REPO = "Qwen/Qwen3.8-27B"
PREFIX = "model.language_model.layers."
# Mid-depth group with the 9B layout (3 GatedDeltaNet + 1 gated attention): 9B used 12-15 of 32; 27B 24-27 of 64.
SUPPORT_LAYERS = (24, 25, 26, 27)
GPU_CAP_BYTES = 72_000_000_000  # C9: kill if peak > 72 GB on an 80 GB GPU
GO_C_SECONDS = 240.0  # D8: lean panel <= 4 min per state
# Attempt 1 (allocator 70 GB) read 72.73 GB on nvidia-smi: ~2.7 GB of device memory is outside the allocator.
NON_ALLOCATOR_ALLOWANCE_BYTES = 3_500_000_000
ALLOCATOR_BYTES = 68_000_000_000
FIELD_TOKEN_LADDER = (1024, 768, 512)  # per field item; step down only on an allocator OOM, never past 512


def next_field_cap(cap):
    i = FIELD_TOKEN_LADDER.index(cap)
    return FIELD_TOKEN_LADDER[i + 1] if i + 1 < len(FIELD_TOKEN_LADDER) else None


def support_modules(rm, layers=SUPPORT_LAYERS):
    mods = []
    for layer in layers:
        mods.append(rm.writers()[(layer, "down")].rsplit(".", 1)[0])
        mods.append(rm.writers()[(layer, "o")].rsplit(".", 1)[0])
    return mods


def hash_only(results: dict) -> dict:
    """A panel result reduced to content hashes and item counts; no objective value appears."""
    return {
        "digest": P.digest(results),
        "per_objective": {k: P.digest({k: v}) for k, v in sorted(results.items())},
        "n_items": {k: len(v) for k, v in results.items()},
    }


def gates(m: dict) -> dict:
    """Go-M (device <= 72 GB, no swap), Go-N (every repeat bitwise), Go-C (every timed state <= 240 s)."""
    secs = list(m["panel_seconds_per_state"])
    return {
        "Go-M": {
            "rule": "peak device memory <= 72 GB (C9) and no swap",
            "peak_device_gb": m["peak_device_bytes"] / 1e9,
            "peak_rss_gb": m["peak_rss_bytes"] / 1e9,
            "peak_swap_bytes": m["peak_swap_bytes"],
            "pass": m["peak_device_bytes"] <= GPU_CAP_BYTES and m["peak_swap_bytes"] == 0,
        },
        "Go-N": {
            "rule": "base, edit and restore repeats bitwise identical",
            "repeats": dict(m["repeats"]),
            "pass": bool(m["repeats"]) and all(m["repeats"].values()),
        },
        "Go-C": {
            "rule": "lean panel <= 240 s per state",
            "seconds_per_state": secs,
            "max_seconds": max(secs),
            "pass": bool(secs) and max(secs) <= GO_C_SECONDS,
        },
    }


def bank_hours(visits, t_ll, n_gen, t_gen, t_features, t_train=0.0):
    """SYNTHESIS-r8 D8: H = (visits*t_ll + N_gen*t_gen + T_features + T_train) / 3600."""
    return (visits * t_ll + n_gen * t_gen + t_features + t_train) / 3600


def dp_projection(visits, t_state, setup_s, load_s, t_group, n_groups, prices):
    """L1: one model per GPU, states (and field groups) as queue items, no communication; ``prices`` maps the GPU
    count k to the instance's USD per hour. Wall = setup + load + ceil(visits/k) t_state + ceil(groups/k) t_group.
    """
    out = {}
    for k, usd_h in sorted(prices.items()):
        wall = (
            setup_s + load_s + math.ceil(visits / k) * t_state + math.ceil(n_groups / k) * t_group
        ) / 3600
        out[k] = {
            "wall_hours": wall,
            "gpu_hours": k * wall,
            "usd": wall * usd_h,
            "usd_per_h": usd_h,
        }
    return out


def record_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def support_groups(rm, group_size=4, layers=SUPPORT_LAYERS):
    """The support layers as consecutive groups of ``group_size`` layers (each group is one collector pass)."""
    if group_size <= 0 or len(layers) % group_size:
        raise ValueError(f"group size {group_size} does not divide {len(layers)} layers")
    return [
        support_modules(rm, layers[i : i + group_size]) for i in range(0, len(layers), group_size)
    ]
