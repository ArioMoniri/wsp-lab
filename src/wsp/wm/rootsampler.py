"""Outcome-blind rejection sampler for the v1 9B bank roots and the direction-similarity audit (PREREG_WME v1 §4.1,
§4.4; Board A condition 2; RULING-wme-resources §6). Only theta0 weights, construction-shard activations and recipe
arithmetic are used; nothing is evaluated.

**Contrast-source catalogue** (``bank.contrast_catalogue``): ordered pairs of distinct sources among GSM8K
questions, harmless prompts, the seven MATH L4-5 subjects and the fourteen MMLU-Pro categories. The sycophancy and
WikiText pools are **excluded** from bank construction: they are the binding item pools (§3.3; 467 and 303 groups),
needed for bank labels, dev, test, generation and the field. This is declared before any root is drawn.

**Families** (§4.1): scale (f = 1.5, one seeded writer), row_scale (k = 64 rows of ``embed_tokens`` among token ids
occurring in the root's shard, f = 1.5), proj_remove (alpha = 0.5 over ``down`` of a contiguous seeded block of 4
layers, contrast direction per layer; a matched random-direction control per root), noise (rho = the median dose-1
relative norm of the accepted steer roots, ``bank.noise_rho``), steer (rank-one shift at a seeded layer, contrast
direction), lora_tv (rank 8 on every writer, a seeded task among GSM8K, MATH, MMLU-Pro and WikiText LM; WikiText-LM
roots are rejected when the WikiText construction allowance is exhausted, which it is under the binding pool).

**Rejection** (J2-J5, the same identity format as ``wm.sep27`` so the 27B guard reads these roots directly): J2
shards are disjoint by construction (drawn without replacement) and checked; J3 |cos| >= 0.5 between residual
directions at any layers (every per-layer vector of a candidate against every accepted, pilot or excluded vector),
and cos >= 0.5 between LoRA task vectors (only computable after training: checked at bank build, before any
evaluation); J4 a shared scale/noise tensor, or row sets with Jaccard >= 0.5; J5 duplicate realised bytes (build).
Excluded identities: every pilot-1 and pilot-2 direction, the pilot scale tensors, the P-H100 timing edits and the
27B calibration timing edits. Candidates are drawn round-robin over families in a seeded order; exhaustion of the
attempt budget before every family holds r_f roots is a feasibility failure (§8.5).
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

NAMESPACE = "wme-bank"
MODEL = "Qwen/Qwen3.5-9B"
FAMILIES = ("scale", "row_scale", "proj_remove", "noise", "steer", "lora_tv")
N_LAYERS = 32
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
MMLU_CATEGORIES = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)
SOURCES = (
    ("gsm", "harmless")
    + tuple(f"mathcomp:{s}" for s in MATH_SUBJECTS)
    + tuple(f"mmlu:{c}" for c in MMLU_CATEGORIES)
)
LORA_TASKS = ("gsm8k_gold_lp", "mathcomp_gold_lp", "mmlupro_gold_lp", "wikitext_lm")
NOMINALS = {"scale": 1.5, "row_scale": 1.5, "proj_remove": 0.5, "steer": 0.125, "lora_tv": 0.5}
ROWS = 64
J3_MAX_COS = 0.5
J4_ROW_JACCARD = 0.5
PILOT_SCALE_TENSORS = {(13, "down"), (14, "o"), (15, "down"), (13, "o")}
PH100_TENSORS = {(5, "down"), (9, "o"), (27, "down"), (22, "o")}
FORBIDDEN_TENSORS = PILOT_SCALE_TENSORS | PH100_TENSORS
ATTEMPTS_PER_ROOT = 50


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}:{label}".encode()).hexdigest()[:8], 16)


def catalogue() -> list[tuple]:
    return [(a, b) for a, b in itertools.permutations(SOURCES, 2)]


def _vecs(x):
    if x is None:
        return []
    if isinstance(x, dict):
        return [np.asarray(v, float).ravel() for _k, v in sorted(x.items())]
    if isinstance(x, (list, tuple)) and x and np.ndim(x[0]) >= 1:
        return [np.asarray(v, float).ravel() for v in x]
    return [np.asarray(x, float).ravel()]


def _maxcos(a, b, signed=False):
    best = 0.0
    for u in _vecs(a):
        for v in _vecs(b):
            if u.shape != v.shape:
                continue
            c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
            best = max(best, c if signed else abs(c))
    return best


def lora_cos(fa: dict, fb: dict) -> float:
    """Exact cosine of two LoRA task vectors flattened over all writers, from their rank-r factors:
    <s1 B1 A1, s2 B2 A2>_F = s1 s2 tr((B1^T B2)(A2 A1^T))."""
    import torch

    def ip(f, g):
        tot = 0.0
        for n in set(f) & set(g):
            b1, a1, s1 = f[n]
            b2, a2, s2 = g[n]
            tot += float(
                s1 * s2 * torch.trace((b1.double().T @ b2.double()) @ (a2.double() @ a1.double().T))
            )
        return tot

    return ip(fa, fb) / max((ip(fa, fa) * ip(fb, fb)) ** 0.5, 1e-300)


def joins(c, e) -> str | None:
    for key in ("contrast_source_groups", "adapter_source_groups"):
        if set(c.get(key) or ()) & set(e.get(key) or ()):
            return "J2"
    if (
        c.get("space") == e.get("space") == "lora"
        and isinstance(c.get("direction"), dict)
        and "lora" in c["direction"]
        and "lora" in (e.get("direction") or {})
    ):
        if lora_cos(c["direction"]["lora"], e["direction"]["lora"]) >= J3_MAX_COS:
            return "J3"
    elif (
        c.get("model", MODEL) == e.get("model", MODEL)
        and c.get("direction") is not None
        and e.get("direction") is not None
        and c.get("space", "residual") == e.get("space", "residual")
    ):
        signed = c.get("space") == "lora"
        if _maxcos(c["direction"], e["direction"], signed) >= J3_MAX_COS:
            return "J3"
    if (
        {c.get("family"), e.get("family")} <= {"scale", "noise"}
        and c.get("tensor") is not None
        and list(c.get("tensor")) == list(e.get("tensor") or [])
    ):
        return "J4"
    if c.get("family") == e.get("family") == "row_scale":
        a, b = set(c.get("rows") or ()), set(e.get("rows") or ())
        if a and b and len(a & b) / len(a | b) >= J4_ROW_JACCARD:
            return "J4"
    if set(c.get("realised_delta") or ()) & set(e.get("realised_delta") or ()):
        return "J5"
    return None


class _Pools:
    def __init__(self, pools, seed):
        self.left = {s: list(v) for s, v in pools.items()}
        self.rng = np.random.default_rng(seed)

    def take(self, source, n):
        p = self.left.get(source, [])
        if len(p) < n:
            return None
        idx = sorted(self.rng.choice(len(p), size=n, replace=False).tolist(), reverse=True)
        out = [p.pop(i) for i in idx]
        return sorted(out)

    def give_back(self, source, groups):
        self.left.setdefault(source, []).extend(groups)


def _candidate(fam, k, rng, pools, shard_contrast, shard_lora, rows_fn):
    rid = f"{fam}:{k:03d}"
    base = {
        "id": rid,
        "namespace": NAMESPACE,
        "model": MODEL,
        "family": fam,
        "nominal": NOMINALS.get(fam),
    }
    if fam in ("scale", "noise"):
        L = int(rng.integers(N_LAYERS))
        role = ("o", "down")[int(rng.integers(2))]
        c = dict(base, tensor=[L, role])
        if fam == "noise":
            c["seed"] = int(rng.integers(2**31))
        return c, []
    if fam in ("steer", "proj_remove", "row_scale"):
        cat = catalogue()
        pair = cat[int(rng.integers(len(cat)))]
        if fam == "row_scale":
            g = pools.take(pair[0], shard_contrast)
            if g is None:
                return None, []
            c = dict(base, tensor="embed_tokens", shard_source=pair[0], contrast_source_groups=g)
            c["rows"] = rows_fn(c, rng)
            return c, [(pair[0], g)]
        ga = pools.take(pair[0], shard_contrast)
        gb = pools.take(pair[1], shard_contrast)
        if ga is None or gb is None:
            for s, g in ((pair[0], ga), (pair[1], gb)):
                if g:
                    pools.give_back(s, g)
            return None, []
        if fam == "steer":
            layers = [int(rng.integers(N_LAYERS))]
        else:
            s0 = int(rng.integers(N_LAYERS - 3))
            layers = [s0, s0 + 1, s0 + 2, s0 + 3]
        c = dict(
            base,
            pair=list(pair),
            layers=layers,
            tensor=[[L, "down"] for L in layers],
            space="residual",
            contrast_source_groups=ga + gb,
        )
        return c, [(pair[0], ga), (pair[1], gb)]
    if fam == "lora_tv":
        task = LORA_TASKS[int(rng.integers(len(LORA_TASKS)))]
        src = {
            "gsm8k_gold_lp": ["gsm"],
            "mathcomp_gold_lp": [f"mathcomp:{s}" for s in MATH_SUBJECTS],
            "mmlupro_gold_lp": [f"mmlu:{c}" for c in MMLU_CATEGORIES],
            "wikitext_lm": ["wiki"],
        }[task]
        s = src[int(rng.integers(len(src)))]
        g = pools.take(s, shard_lora)
        if g is None:
            return None, []
        c = dict(
            base,
            task=task,
            shard_source=s,
            adapter_source_groups=g,
            seed=int(rng.integers(2**31)),
            space="lora",
            rank=8,
            alpha=16,
            j3="cos >= 0.5 against accepted LoRA task vectors (trained)",
        )
        return c, [(s, g)]
    raise KeyError(fam)


def _default_rows(c, rng):
    return sorted(rng.choice(248_077, size=ROWS, replace=False).tolist())


def sample_roots(
    r_f,
    pools,
    direction_fn,
    excluded,
    shard_contrast,
    shard_lora,
    seed=None,
    rows_fn=None,
    lora_fn=None,
) -> dict:
    """Seeded rejection sampling; ``direction_fn(candidate)`` returns its theta0 direction(s) (residual families)."""
    seed = seed_for("root_sampler") if seed is None else seed
    rng = np.random.default_rng(seed)
    pl = _Pools(pools, seed + 1)
    rows_fn = rows_fn or _default_rows
    accepted, rejected = [], []
    count = {f: 0 for f in FAMILIES}
    tries = {f: 0 for f in FAMILIES}
    k = 0
    while any(count[f] < r_f and tries[f] < ATTEMPTS_PER_ROOT * r_f for f in FAMILIES):
        for f in FAMILIES:
            if count[f] >= r_f or tries[f] >= ATTEMPTS_PER_ROOT * r_f:
                continue
            tries[f] += 1
            k += 1
            c, took = _candidate(f, k, rng, pl, shard_contrast, shard_lora, rows_fn)
            if c is None:
                rejected.append({"id": f"{f}:{k:03d}", "family": f, "reasons": ["pool"]})
                continue
            reasons = []
            if c["family"] in ("scale", "noise") and tuple(c["tensor"]) in FORBIDDEN_TENSORS:
                reasons.append("forbidden_tensor")
            if c["family"] in ("steer", "proj_remove"):
                c["direction"] = direction_fn(c)
            elif c["family"] == "lora_tv" and lora_fn is not None:
                c["direction"] = {
                    "lora": lora_fn(c)
                }  # trained at theta0 on the root's shard, then J3
            for e in [*excluded, *accepted]:
                j = joins(c, e)
                if j:
                    reasons.append(f"{j}:{e['id']}")
            if reasons:
                for s, g in took:
                    pl.give_back(s, g)
                rejected.append(
                    {k2: v for k2, v in c.items() if k2 != "direction"} | {"reasons": reasons}
                )
            else:
                accepted.append(c)
                count[f] += 1
    controls = []
    for r in accepted:
        if r["family"] == "proj_remove":
            controls.append(
                {
                    "id": r["id"] + ":control",
                    "partner": r["id"],
                    "family": "proj_remove_control",
                    "tensor": r["tensor"],
                    "nominal": r["nominal"],
                    "control_seed": seed_for(f"control:{r['id']}"),
                }
            )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "controls": controls,
        "counts": count,
        "tries": tries,
        "feasible": all(count[f] >= r_f for f in FAMILIES),
    }


def noise_rho(roots) -> float:
    v = [r["rel_norm_dose1"] for r in roots if r["family"] == "steer"]
    return float(np.median(v))


def direction_audit(bank: dict, pilot: dict, max_cos: float = J3_MAX_COS) -> dict:
    worst, flagged = {}, []
    for k, v in sorted(bank.items()):
        best = (0.0, None)
        for pk, pv in pilot.items():
            c = _maxcos(v, pv)
            if c > best[0]:
                best = (c, pk)
        worst[k] = {"cos": best[0], "against": best[1]}
        if best[0] >= max_cos:
            flagged.append(k)
    return {
        "max_cos": max_cos,
        "worst": worst,
        "flagged": flagged,
        "pass": not flagged,
        "n_bank": len(bank),
        "n_pilot": len(pilot),
    }


def identity_digest(roots) -> str:
    strip = [{k: v for k, v in r.items() if k != "direction"} for r in roots]
    return hashlib.sha256(json.dumps(strip, sort_keys=True, default=str).encode()).hexdigest()
