"""P-H100: the timing and noise pilot on Lambda H100 SXM5 80 GB (RULING-wme-resources §2; PREREG_WME v1 §6.3.3,
§8.6). ENGINEERING ONLY.

**Items.** Only the exposed wme-p2 panel's ``construction`` and ``validation`` splits (never dev or test; the sealed
directory is never copied or opened). No wme-c1 item exists yet.

**Edits.** Only these three kinds (RULING §2):
* the **no-op** (an empty action list, and a dose-0 action whose realised delta is exactly zero);
* **inert** controls (§6.3.3): I1 power-of-two round trips (2-step: one writer x2 then x1/2; 4-step: A x2, B x2,
  A x1/2, B x1/2), the compensated power-of-two pairs of ``controls.compensated_pairs`` (up x2 / down x1/2;
  v x2 / o x1/2), and I2 row_scale on input-embedding rows the frozen tokenizer and chat template never emit
  (f in {1.5, 0.5, -1}; untied embedding only);
* fixed **timing edits** (:data:`TIMING_EDITS`), declared here before any run. They are registered as engineering
  exposures: the bank root sampler must reject any root sharing their tensors (J4) or directions (J3).

**Outputs.** Paired per-state SEs per endpoint (validation n = 64; scaled by sqrt(64/n)), seconds per item and per
unit, oracle-gradient, field and generation seconds, the inert verification and the 1-vs-k parity. Mean changes are
never reported: per-state statistics appear only as content hashes (:func:`hash_only`). No trajectory, headroom or
enhancement quantity is computed.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

from wsp.wm import actions as A

ENDPOINTS = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)
GUARD_EXTRA = ("mathcomp_gold_lp", "syc_neutral")
SE_GRID_N = (32, 48, 64, 96, 128, 160, 192, 256)
I2_FACTORS = (1.5, 0.5, -1.0)
I1_TENSORS = {"2step": [(16, "down")], "4step": [(8, "down"), (19, "o")]}
COMPENSATED = {"mlp_layers": (12, 15), "attention_layers": (15,)}
NAMESPACE = "wme-ph100"

# Fixed timing edits (declared before any P-H100 run; never pilot-2 or pilot-1 tensors). Doses are 1 (dose-1
# realisation of the stated parameters) so every edit is one bundle on one state.
TIMING_EDITS = (
    {"id": "t_scale_down5", "family": "scale", "tensors": [[5, "down"]], "factor": 1.5},
    {"id": "t_scale_o9_q", "family": "scale", "tensors": [[9, "o"]], "factor": 1.125},
    {
        "id": "t_noise_down27",
        "family": "noise",
        "tensors": [[27, "down"]],
        "rho": 0.02,
        "seed": 271,
    },
    {"id": "t_rows_embed", "family": "row_scale", "rows": 64, "factor": 1.5, "seed": 641},
    {
        "id": "t_projrm_rand_28_31",
        "family": "proj_remove",
        "tensors": [[L, "down"] for L in (28, 29, 30, 31)],
        "alpha": 0.5,
        "seed": 2831,
    },
    {"id": "t_noise_o22", "family": "noise", "tensors": [[22, "o"]], "rho": 0.01, "seed": 22},
)


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}:{label}".encode()).hexdigest()[:8], 16)


def timing_edit_digest() -> str:
    return hashlib.sha256(json.dumps(TIMING_EDITS, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ inert controls


def power_of_two_safe(w: torch.Tensor, factor: float) -> bool:
    """True iff multiplying every element of bf16 ``w`` by ``factor`` (a power of two) is exact: no overflow to
    inf and no result below the smallest normal (nonzero elements)."""
    x = w.detach().float()
    nz = x[x != 0]
    if nz.numel() == 0:
        return True
    a = nz.abs() * factor
    fi = torch.finfo(torch.bfloat16)
    return bool((a.max() <= fi.max).item() and (a.min() >= fi.tiny).item())


def round_trip(rm: A.RoleMap, shape: str, tensors=None) -> list[list[A.Action]]:
    """I1 steps (each a one-action bundle). 2-step: X x2, X x1/2. 4-step: A x2, B x2, A x1/2, B x1/2."""
    tensors = tensors or I1_TENSORS[shape]
    names = [rm.name(L, r) for L, r in tensors]
    if shape == "2step":
        n = names[0]
        return [[A.scale(n, factor=2.0, dose=1.0)], [A.scale(n, factor=0.5, dose=1.0)]]
    if shape == "4step":
        a, b = names[:2]
        return [
            [A.scale(a, factor=2.0, dose=1.0)],
            [A.scale(b, factor=2.0, dose=1.0)],
            [A.scale(a, factor=0.5, dose=1.0)],
            [A.scale(b, factor=0.5, dose=1.0)],
        ]
    raise KeyError(shape)


def never_emitted_rows(tokenizer_len: int, vocab: int, template_ids) -> list[int]:
    """Input-embedding row ids no tokenizer output can index (ids >= len(tokenizer)); refuses if the chat
    template or any special token reaches them."""
    t = {int(i) for i in template_ids}
    bad = sorted(i for i in t if i >= tokenizer_len)
    if bad:
        raise ValueError(f"template/special ids reach the never-emitted range: {bad[:5]}")
    return list(range(int(tokenizer_len), int(vocab)))


def i2_actions(rm: A.RoleMap, embed: str, rows) -> list[A.Action]:
    if rm.tied:
        raise ValueError(
            "tied embeddings: an embedding row is also an output row, so it is not inert"
        )
    return [A.row_scale(embed, rows, factor=f, dose=1.0) for f in I2_FACTORS]


# ------------------------------------------------------------------ timing edits


def timing_actions(rm: A.RoleMap, spec: dict, embed: str, tokenizer_len: int) -> list[A.Action]:
    fam = spec["family"]
    if fam == "scale":
        return [A.scale(rm.name(L, r), factor=spec["factor"], dose=1.0) for L, r in spec["tensors"]]
    if fam == "noise":
        return [
            A.noise(rm.name(L, r), seed=spec["seed"], rho=spec["rho"], dose=1.0)
            for L, r in spec["tensors"]
        ]
    if fam == "row_scale":
        rng = np.random.default_rng(spec["seed"])
        rows = sorted(rng.choice(int(tokenizer_len), size=spec["rows"], replace=False).tolist())
        return [A.row_scale(embed, rows, factor=spec["factor"], dose=1.0)]
    if fam == "proj_remove":
        g = torch.Generator().manual_seed(spec["seed"])
        u = torch.randn(rm.hidden, generator=g)
        return [
            A.proj_remove(rm.name(L, r), direction=u / u.norm(), alpha=spec["alpha"], dose=1.0)
            for L, r in spec["tensors"]
        ]
    raise KeyError(fam)


# ------------------------------------------------------------------ summaries (hash-only for values)


def se_summary(stats: dict, endpoints=ENDPOINTS + GUARD_EXTRA, n_ref: int = 64) -> dict:
    """Median paired SE per endpoint over states (``stats[state][obj] = {n, mean, se}``), scaled to grid n by
    sqrt(n_ref / n). Only SEs leave this function."""
    out = {}
    for o in endpoints:
        ses = [s[o]["se"] for s in stats.values() if o in s and np.isfinite(s[o]["se"])]
        if not ses:
            continue
        ns = sorted({s[o]["n"] for s in stats.values() if o in s})
        med = float(np.median(ses))
        out[o] = {
            "median_se_n64": med,
            "min_se": float(min(ses)),
            "max_se": float(max(ses)),
            "n_states": len(ses),
            "n_groups": ns,
            "se_at": {str(n): med * math.sqrt(n_ref / n) for n in SE_GRID_N},
        }
    return out


def hash_only(stats: dict) -> dict:
    return {
        k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
        for k, v in stats.items()
    }


def unit_times(units) -> dict:
    """Seconds per scored item per objective, and the per-unit apply + restore overhead."""
    per, over = {}, []
    for u in units:
        s = u["seconds"]
        per.setdefault(u["objective"], []).append(s["eval"] / u["n_items"])
        over.append(s["apply"] + s["restore"])
    return {
        "per_item_s": {o: float(np.mean(v)) for o, v in sorted(per.items())},
        "overhead_per_unit_s": float(np.mean(over)) if over else 0.0,
        "n_units": len(over),
    }
