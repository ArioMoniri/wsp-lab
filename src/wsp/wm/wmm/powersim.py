"""PREREG_WMM power simulation: the registered claim functions (``wmm.prereg``) applied to data simulated from Gate 1
dev inputs (exposed data; sizing only). Binary paired endpoints are drawn from a multinomial with a given discordance
d and mean difference m; the Brier guard from a normal with the Gate 1 per-item SD; M-TE from a generative model that
resamples Gate 1 base (length, correctness, truncation) triples; M-A from a units-and-features model.
"""

from __future__ import annotations

import math

import numpy as np

from wsp.wm.wmm import prereg as P


def paired_binary_sample(n, d, m, base_rate, rng):
    """(edit, base) with P(edit≠base) = d and E[edit − base] = m; requires 0 ≤ (d ± m)/2 and (d − m)/2 ≤ base_rate."""
    p10, p01 = (d - m) / 2, (d + m) / 2
    p11 = base_rate - p10
    p00 = 1 - p10 - p01 - p11
    probs = np.array([p00, p01, p10, p11])
    if np.any(probs < -1e-12):
        raise ValueError(f"infeasible (d={d}, m={m}, base_rate={base_rate})")
    probs = np.clip(probs, 0, None)
    k = rng.choice(4, size=n, p=probs / probs.sum())
    base = np.isin(k, [2, 3]).astype(int)
    edit = np.isin(k, [1, 3]).astype(int)
    return edit, base


def guard_sample(spec, rng):
    g = {}
    for name in ("harm_compliance", "benign_over_refusal", "general_accuracy"):
        s = spec[name]
        g[name] = paired_binary_sample(s["n"], s["d"], s["m"], s["base_rate"], rng)
    b = spec["brier"]
    g["brier"] = rng.normal(b["m"], b["sd"], b["n"])
    return g


def msyc_power(spec, reps=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    keys = ("primary", "untruncated_in_both", "neutral", "guards")
    gk = ("harm_compliance", "benign_over_refusal", "general_accuracy", "brier")
    cnt = dict.fromkeys(keys + gk + ("pass",), 0)
    a, ne = spec["adoption"], spec["neutral"]
    for _ in range(reps):
        adopt = paired_binary_sample(a["n"], a["d"], a["m"], a["base_rate"], rng)
        trunc = (
            (rng.random(a["n"]) < a["trunc"]).astype(int),
            (rng.random(a["n"]) < a["trunc"]).astype(int),
        )
        neutral = paired_binary_sample(ne["n"], ne["d"], ne["m"], ne["base_rate"], rng)
        c = P.msyc_claim(adopt, trunc, neutral, guard_sample(spec["guards"], rng), spec["alpha_f"])
        for k in keys:
            cnt[k] += c[k]["pass"]
        for k in gk:
            cnt[k] += c["guards"][k]["pass"]
        cnt["pass"] += c["pass"]
    return {k: v / reps for k, v in cnt.items()} | {"reps": reps}


def mte_sample(spec, base_triples, rng):
    """Base (correct, length, truncated) resampled from Gate 1; edit log-length = base + mu + N(0, sd); edit
    correctness re-drawn with probability q from Bernoulli(p_full) (q gives the target full-budget discordance)."""
    n = spec["n"]
    idx = rng.integers(0, len(base_triples), n)
    cb, lb, tb = (np.asarray(x)[idx] for x in base_triples)
    le = np.exp(np.log(lb + 1.0) + spec["mu"] + rng.normal(0, spec["sd"], n)) - 1.0
    le = np.clip(np.round(le), 1, 4096).astype(int)
    te = (le >= 4096).astype(int)
    p = cb.mean()
    redraw = rng.random(n) < spec["q"]
    p_new = min(max(p + spec["delta_full"] / max(spec["q"], 1e-9), 0.0), 1.0)
    ce = np.where(redraw, (rng.random(n) < p_new).astype(int), cb)
    return {"base": (cb, lb, tb), "edit": (ce, le, te)}


def mte_power(spec, base_triples, reps=500, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    keys = ("tokens", "low_budget", "full_budget_ni", "guards")
    cnt = dict.fromkeys(keys + ("pass", "b512", "b1024"), 0)
    for _ in range(reps):
        data = mte_sample(spec, base_triples, rng)
        c = P.mte_claim(data, guard_sample(spec["guards"], rng), spec["alpha_f"])
        for k in keys:
            cnt[k] += c[k]["pass"]
        cnt["b512"] += 512 in c["low_budget"]["passing_budgets"]
        cnt["b1024"] += 1024 in c["low_budget"]["passing_budgets"]
        cnt["pass"] += c["pass"]
    return {k: v / reps for k, v in cnt.items()} | {"reps": reps}


def ma_power(n_items=200, d_item=0.18, spread_se=1.0, r_feature=0.5, reps=100, seed=0) -> dict:
    """M-A: unit SE = sqrt(d_item / n_items); beyond-dose effects tau ~ N(0, (spread_se·SE)²) within each non-sham
    type (shams: 0); one informative feature correlated r with tau plus 2 noise features; power of ``prereg.ma_test``."""
    rng = np.random.default_rng(seed)
    bank = P.ma_bank()
    types = np.array([u["type"] for u in bank])
    se_u = math.sqrt(d_item / n_items)
    means = {"tv_slice": -0.02, "lora_seed": -0.08, "random": 0.0, "sham": 0.0}
    hits = 0
    for _ in range(reps):
        tau = rng.normal(0, spread_se * se_u, len(bank)) * (types != "sham")
        z = (tau - tau.mean()) / (tau.std() + 1e-12)
        x0 = r_feature * z + math.sqrt(max(1 - r_feature**2, 0)) * rng.normal(size=len(bank))
        X = np.column_stack([x0, rng.normal(size=len(bank)), rng.normal(size=len(bank))])
        y = np.array([means[t] for t in types]) + tau + rng.normal(0, se_u, len(bank))
        se = np.full(len(bank), se_u)
        hits += P.ma_test(
            bank, {"adoption": (y, se)}, X, n_flip=2000, seed=int(rng.integers(1 << 30))
        )["pass"]
    return {
        "power": hits / reps,
        "reps": reps,
        "unit_se": se_u,
        "spread_se": spread_se,
        "r_feature": r_feature,
    }


def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> float:
    p = k / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n)) / den + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
