"""Arithmetic behind PREREG_WME DRAFT v2 (RULING-wme-post-stagew Q2-Q4): secondary pricing, guard N, total-compute
parity budget and the cap check. Pure functions; every input is a measured unit rate or a registered constant passed
in by ``scripts/wme_manifest_prereg_v2.py`` (no value is defined here except the normal quantiles and the v2b price
convention, which are re-exported from ``powersim_v2b``).
"""

from __future__ import annotations

import math

from wsp.wm import powersim_v2b as PB

Z_0975 = 1.959963984540054  # one-sided 0.025
Z_080 = 0.8416212335729143  # power 0.80
CAP_TOTAL_GPU_H = (
    90.0  # RULING-wme-post-stagew Q3: total (bank + confirmatory + secondaries) <= 90 GPU-h-eq
)
CAP_USD_9B = PB.CAPS["lambda_usd_9b"]


def usd(gpu_h: float) -> float:
    """Lambda $ for H100 GPU-h-eq under the v2b convention (list price x (1 + overhead))."""
    return gpu_h * PB.USD_PER_GPU_H * (1 + PB.USD_OVERHEAD)


def guard_n(
    discordance: float, delta: float, pool: int, z_alpha: float, n_cal: int = 128, lo: int = 200
) -> int:
    """``p3gen.guard_n`` with the one-sided level made explicit (pilot 3 used z = 1.645; v2 registers 0.025)."""
    d = max(discordance, 1.0 / n_cal)
    n = math.ceil((z_alpha + Z_080) ** 2 * d / delta**2)
    return int(min(pool, max(lo, n)))


def parity_budget(
    b: float, bank_gpu_h: float, field_s: float, n_episodes: int, t_bundle_s: float
) -> float:
    """Best-of-N full-evaluation equivalents per episode at total-compute parity (stage-W plan §8; R1): the WM arm's
    B plus its bank and R1 field compute, amortised over the episodes and converted at one bundle's seconds."""
    return b + (bank_gpu_h * 3600 + field_s) / n_episodes / t_bundle_s


def bon_arm_gpu_h(budget: float, n_episodes: int, t_bundle_s: float) -> float:
    return n_episodes * budget * t_bundle_s / 3600


def lora_steps(rows: int, rows_per_step: int, epochs: int) -> int:
    return math.ceil(rows / rows_per_step) * epochs


def te_price(
    n_items: dict,
    n_guard: dict,
    n_cal: int,
    cal_samples: int,
    lora_steps: int,
    canary_items: int,
    rates: dict,
) -> dict:
    """GPU-h of the token-efficiency secondary on one 1x H100 instance: base and edit at 2,048 and 4,096 tokens on
    every item; guard panels (base and edit on the test splits, base-only calibration with greedy + ``cal_samples``
    sampled draws, base and the E1 winner for the descriptive E1 safety report), priced at the 2,048-token MC rate as
    an upper bound; one LoRA training run; setup and boot; the canary at start and end."""
    n = sum(n_items.values())
    g = sum(n_guard.values())
    r2, r4 = rates["s_per_item_2048"], rates["s_per_item_4096"]
    seconds = {
        "eval_2048": 2 * n * r2,
        "eval_4096": 2 * n * r4,
        "guards": (2 * g + len(n_guard) * n_cal * (1 + cal_samples) + 2 * g) * r2,
        "training": lora_steps * rates["lora_s_per_step"],
        "setup_boot": rates["setup_s"] + rates["boot_s"],
        "canary": 2 * canary_items * r2,
    }
    h = sum(seconds.values()) / 3600
    return {"seconds": seconds, "gpu_h": h, "usd": usd(h)}


def cap_check(parts_gpu_h: dict) -> dict:
    if any(v < 0 for v in parts_gpu_h.values()):
        raise ValueError("negative GPU-h")
    t = sum(parts_gpu_h.values())
    return {
        "parts_gpu_h": dict(parts_gpu_h),
        "total_gpu_h": t,
        "usd_9b": usd(t),
        "cap_total_gpu_h": CAP_TOTAL_GPU_H,
        "cap_usd_9b": CAP_USD_9B,
        "within": bool(t <= CAP_TOTAL_GPU_H and usd(t) <= CAP_USD_9B),
    }
