"""Trajectory-bank recipe for WM-E (PREREG_WME §4; SYNTHESIS-r8 D4; Board A conditions 2, 3, 12). RECIPES ONLY.

This module emits the frozen recipe manifest; it never realises, evaluates or labels a state.

**Roots** (namespace ``wme-bank``; 16 per family; support layers 12-15 = the base-field support, except the
vocabulary family). Disjoint from every pilot root: no pilot scale tensor (pilot 1: down13, o14; pilot 2: down15,
o13), no pilot contrast pair (pilot 1 math/mcq/wiki vs harmless; pilot 2 syc, mathcomp, trivia, gsmwiki; E2 gsm),
no pilot seed namespace. Declared concept overlap: LoRA objectives gsm8k/mathcomp were also pilot-2 LoRA objectives
(different items, layers 12-15 only); the build-time ``wmclosure.direction_similarity_audit`` (|cos| < 0.5 against
stored pilot directions) is required before use.

| family | recipe | nominal (dose 1) | realisation and cap |
|---|---|---|---|
| scale | column block b of 4 of a support writer (down12, o12, down14, o15) | f = 1.5 | m(s) = max(0, 1 + s(f-1)) |
| row_scale | 32 vocabulary rows (frequency-stratified partition of construction tokens) of lm_head (8) or embed_tokens (8); untied (tie check) | f = 1.5 | m(s) as scale |
| proj_remove | 8 unembedding-row directions (lm_head row of a frequency-ranked construction token) on a support down_proj + 8 matched random directions (same tensor, alpha) | alpha = 0.5 | alpha(s) = 0.5 s, <= 1 on the positive side |
| noise | Gaussian on each of the 8 support writers x 2 seeds | rho = 0.02 | ||D|| = |s| rho ||W0||; sign(s) G |
| steer | difference-of-means rank-one write into down_proj at a support layer; 8 fresh contrast pairs x 2 layers | 0.125 x mean residual norm | shift s x nominal |
| lora | rank-8 task vector on the 8 support writers, 2 objectives x 8 disjoint 96-item construction subsets | lambda = 1 | lambda = s |

**Ladders.** Signed doses ±{0.25, 0.5, 1, 2, 4, 8}; every endpoint built from W0. Duplicate states (equal
effective parameters: a zeroed block, a capped alpha) are visited once (the first dose). Side order: positive then
negative, each walked outward; a state with paired WikiText ΔNLL > 1.0 is kept, labelled ``dnll>1.0``, and ends its
side (:func:`walk_side`).

**Compositions.** 12 four-step sequences of roots from four distinct families (family slots 10/10/10/6/6/6 for
scale/noise/lora/row_scale/proj_remove/steer), each pre-component used at most once, plus each reverse (AB/BA):
24 bundles; each step is the root at dose +1 applied sequentially; prefixes h = 2-4 are states (72).

**Horizon.** The number of steps on the registered trajectory: a ladder trajectory steps through every visited
dose of its side (so ±1 is h = 3 and ±8 is h = 6, never h = 1 even though endpoints are built from W0); a
composition steps root by root. Every learned stage trains on states with h <= 2 only; H-T tests h = 3-4.

**Closure.** ``wm.wmclosure``: identities per root, components under the ``source`` and ``concept`` policies,
five outer folds stratified by component signature, inner fit/selection/calibration allocation per fold.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random

from wsp.wm import wmclosure as W

NAMESPACE = "wme-bank"
FAMILIES = ("scale", "row_scale", "proj_remove", "noise", "steer", "lora")
ROOTS_PER_FAMILY = 16
LADDER = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DNLL_STOP = 1.0
SIDE_ORDER = ("pos", "neg")
SUPPORT_LAYERS = (12, 13, 14, 15)
SUPPORT_WRITERS = tuple((L, r) for L in SUPPORT_LAYERS for r in ("down", "o"))
NOMINALS = {
    "scale": {"factor": 1.5, "blocks": 4},
    "row_scale": {"factor": 1.5, "rows": 32},
    "proj_remove": {"alpha": 0.5, "cap": 1.0},
    "noise": {"rho": 0.02},
    "steer": {"shift_frac": 0.125},
    "lora": {"lam": 1.0, "rank": 8, "alpha": 16, "lr": 1e-4, "epochs": 2, "subset": 96},
}
COMPOSITION_SLOTS = {
    "scale": 10,
    "noise": 10,
    "lora": 10,
    "row_scale": 6,
    "proj_remove": 6,
    "steer": 6,
}
N_BASE_SEQUENCES = 12
COMPOSITION_DOSE = 1.0
CONTRAST_SUBSET = 21
TRAIN_MAX_H = 2
TEST_H = (3, 4)

# Pilot exclusions (identities in this module's scheme)
PILOT_SCALE_TENSORS = {(13, "down"), (14, "o"), (15, "down"), (13, "o")}
PILOT_PAIRS = {
    frozenset(p)
    for p in (
        ("gsm", "harmless"),  # pilot 1 "math"; E2 "gsm"
        ("mcq", "harmless"),  # pilot 1 "mcq"
        ("wiki", "harmless"),  # pilot 1 "wiki"
        ("syc_suggested", "syc_neutral"),  # pilot 2 "syc"
        ("mathcomp", "harmless"),  # pilot 2 "mathcomp"
        ("trivia", "harmless"),  # pilot 2 "trivia" (syc neutral prompts vs harmless)
        ("gsm", "wiki"),  # pilot 2 "gsmwiki"
    )
}
PILOT_LORA_OBJECTIVES = {"sycophancy_real", "gsm8k_gold_lp", "mathcomp_gold_lp"}
CONTRAST_SETS = {  # set -> (construction objective, item field)
    "wiki": "wikitext_nll",
    "gsm": "gsm8k_gold_lp",
    "mcq": "mmlupro_gold_lp",
    "mathcomp": "mathcomp_gold_lp",
    "trivia": "sycophancy_real",
    "harmless": "refusal_margin_harmless",
}
LORA_OBJECTIVES = ("gsm8k_gold_lp", "mathcomp_gold_lp")


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}:{label}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ ladders


def effective(family: str, dose: float) -> dict:
    """Effective realisation parameters of a root at signed cumulative dose ``dose`` (from W0)."""
    n = NOMINALS[family]
    if family in ("scale", "row_scale"):
        m = 1.0 + dose * (n["factor"] - 1.0)
        return {"multiplier": max(0.0, m), "capped": m < 0.0}
    if family == "proj_remove":
        a = dose * n["alpha"]
        return {"alpha": min(a, n["cap"]), "capped": a > n["cap"]}
    if family == "noise":
        return {"rho": abs(dose) * n["rho"], "sign": 1 if dose > 0 else -1, "capped": False}
    if family == "steer":
        return {"shift": dose * n["shift_frac"], "capped": False}
    if family == "lora":
        return {"lam": dose * n["lam"], "capped": False}
    raise KeyError(family)


def _eff_key(family, dose):
    e = dict(effective(family, dose))
    e.pop("capped")
    return json.dumps(e, sort_keys=True)


def ladder_sides(family: str) -> tuple[list, list]:
    out = []
    for sign in (1.0, -1.0):
        seen, side = set(), []
        for m in LADDER:
            k = _eff_key(family, sign * m)
            if k in seen:
                continue
            seen.add(k)
            side.append(sign * m)
        out.append(side)
    return out[0], out[1]


def walk_side(doses, evaluate) -> list:
    """Visit ``doses`` outward; ``evaluate(dose)`` -> paired WikiText ΔNLL; stop after the first > 1.0 state."""
    out = []
    for h, d in enumerate(doses, start=1):
        dn = float(evaluate(d))
        stop = dn > DNLL_STOP
        out.append(
            {"dose": d, "h": h, "dnll": dn, "endpoint": stop, "label": "dnll>1.0" if stop else None}
        )
        if stop:
            break
    return out


# ------------------------------------------------------------------ roots


def _finish(rid, rec, identity, support="support"):
    rec = dict(rec, id=rid, layer_support=support)
    identity = dict(identity, recipe=W.recipe_hash({k: v for k, v in rec.items() if k != "id"}))
    rec["identity"] = identity
    return rec


def _contrast_pairs(rng):
    sets = sorted(CONTRAST_SETS)
    avail = [
        frozenset(p) for p in itertools.combinations(sets, 2) if frozenset(p) not in PILOT_PAIRS
    ]
    avail = [
        p for p in avail if "harmless" not in p
    ]  # harmless-anchored pairs are all pilot concepts
    avail.sort(key=lambda p: sorted(p))
    rng.shuffle(avail)
    return avail[:8]


def make_roots() -> dict:
    rng = random.Random(seed_for("roots"))
    roots = {}
    # scale: 4 non-pilot support writers x 4 column blocks
    writers = sorted(w for w in SUPPORT_WRITERS if w not in PILOT_SCALE_TENSORS)
    for i, (w, b) in enumerate(itertools.product(writers, range(NOMINALS["scale"]["blocks"]))):
        rid = f"scale:{i:02d}"
        roots[rid] = _finish(
            rid,
            {"family": "scale", "tensor": list(w), "block": [b, NOMINALS["scale"]["blocks"]]},
            {"direction": ["scale", list(w), b]},
        )
    # row_scale: 16 disjoint frequency-stratified row sets; 8 on lm_head, 8 on embed_tokens
    for i in range(ROOTS_PER_FAMILY):
        tensor = "lm_head" if i < 8 else "embed_tokens"
        rid = f"row_scale:{i:02d}"
        rule = {
            "partition": "frequency-stratified construction tokens",
            "set": i,
            "n_sets": 16,
            "rows": NOMINALS["row_scale"]["rows"],
            "seed": seed_for("row_sets"),
        }
        roots[rid] = _finish(
            rid,
            {
                "family": "row_scale",
                "tensor": tensor,
                "rows_rule": rule,
                "tie_check": "require untied",
            },
            {"direction": ["rows", tensor, i]},
            support="vocab",
        )
    # proj_remove: 8 unembedding directions + 8 matched random directions (registered pairs)
    for k in range(8):
        L = SUPPORT_LAYERS[k % 4]
        token_rule = {"rank_from": "construction token frequency", "rank": 100 + 37 * k}
        for kind in ("unembed", "random"):
            i = 2 * k + (kind == "random")
            rid = f"proj_remove:{i:02d}"
            rec = {"family": "proj_remove", "tensor": [L, "down"], "kind": kind}
            if kind == "unembed":
                rec["token_rule"] = token_rule
                direction = ["unembed", token_rule["rank"]]
            else:
                rec["seed"] = seed_for(f"proj_random:{k}")
                direction = ["random", rec["seed"]]
            roots[rid] = _finish(rid, rec, {"direction": direction, "pair": f"proj_pair:{k}"})
    # noise: 8 support writers x 2 seeds
    for i, (w, s) in enumerate(itertools.product(SUPPORT_WRITERS, range(2))):
        rid = f"noise:{i:02d}"
        seed = seed_for(f"noise:{w[0]}{w[1]}:{s}")
        roots[rid] = _finish(
            rid,
            {"family": "noise", "tensor": list(w), "seed": seed},
            {"direction": ["noise", list(w), seed]},
        )
    # steer: 8 fresh pairs x 2 layers; per-pair disjoint construction subsets of CONTRAST_SUBSET items per set
    pairs = _contrast_pairs(rng)
    use = {s: 0 for s in CONTRAST_SETS}
    i = 0
    for p in pairs:
        subsets = []
        for s in sorted(p):
            subsets.append(f"construction:{s}:{use[s]}")
            use[s] += 1
        layers = sorted(rng.sample(SUPPORT_LAYERS, 2))
        for L in layers:
            rid = f"steer:{i:02d}"
            roots[rid] = _finish(
                rid,
                {
                    "family": "steer",
                    "pair": sorted(p),
                    "layer": L,
                    "tensor": [L, "down"],
                    "subset_items": CONTRAST_SUBSET,
                },
                {
                    "direction": ["steer", sorted(p), L],
                    "contrast_source": subsets,
                    "concept": "pair:" + "-".join(sorted(p)),
                },
            )
            i += 1
    # lora: 2 objectives x 8 disjoint training subsets, one seed each
    for j, (obj, k) in enumerate(itertools.product(LORA_OBJECTIVES, range(8))):
        rid = f"lora:{j:02d}"
        src = f"construction:lora:{obj}:{k}"
        roots[rid] = _finish(
            rid,
            {
                "family": "lora",
                "objective": obj,
                "subset": k,
                "seed": seed_for(f"lora:{obj}:{k}"),
                "tensors": [list(w) for w in SUPPORT_WRITERS],
                **{kk: v for kk, v in NOMINALS["lora"].items() if kk != "lam"},
            },
            {"direction": ["lora", src], "adapter_source": src, "concept": f"lora:{obj}"},
        )
    return roots


def construction_allocation(roots) -> dict:
    """Items per construction set and role (frozen counts; the item hashes are bound when wme-c1 is built)."""
    out = {}
    for r in roots.values():
        for s in r["identity"].get("contrast_source") or []:
            out[s] = {"role": "direction_construction", "items": CONTRAST_SUBSET}
        if r["identity"].get("adapter_source"):
            out[r["identity"]["adapter_source"]] = {
                "role": "lora_training",
                "items": NOMINALS["lora"]["subset"],
            }
    return dict(sorted(out.items()))


def pilot_overlap(roots) -> dict:
    tensors = sorted(
        {tuple(r["tensor"]) for r in roots.values() if r["family"] == "scale"} & PILOT_SCALE_TENSORS
    )
    pairs = sorted(
        "-".join(sorted(r["pair"]))
        for r in roots.values()
        if r["family"] == "steer" and frozenset(r["pair"]) in PILOT_PAIRS
    )
    concept = sorted(
        {
            r["identity"]["concept"]
            for r in roots.values()
            if r["family"] == "lora" and r["objective"] in PILOT_LORA_OBJECTIVES
        }
    )
    return {
        "tensor_overlap": tensors,
        "pair_overlap": pairs,
        "seed_namespace": NAMESPACE,
        "declared_concept_overlap": concept,
        "build_time_audit": "wmclosure.direction_similarity_audit(|cos| < 0.5) vs stored pilot directions",
    }


# ------------------------------------------------------------------ compositions


def make_compositions(roots) -> list:
    rng = random.Random(seed_for("compositions"))
    pre = W.components(roots)  # identity-joined pre-components (before compositions)
    by_fam = {}
    for rid in sorted(roots):
        by_fam.setdefault(roots[rid]["family"], {}).setdefault(pre[rid], []).append(rid)
    pools = {f: sorted(g) for f, g in by_fam.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    left = dict(COMPOSITION_SLOTS)
    rank = {f: rng.random() for f in FAMILIES}
    seqs = []
    for _ in range(N_BASE_SEQUENCES):
        fams = sorted(FAMILIES, key=lambda f: (-left[f], rank[f]))[:4]
        seq = []
        for f in fams:
            left[f] -= 1
            g = pools[f].pop()
            seq.append(min(by_fam[f][g]))
        rng.shuffle(seq)
        seqs.append(tuple(seq))
    if any(v != 0 for v in left.values()):
        raise AssertionError("composition slots not exhausted")
    return [x for s in seqs for x in (s, tuple(reversed(s)))]


# ------------------------------------------------------------------ trajectories and manifest


def trajectories(rec) -> list:
    out = []
    for rid, r in rec["roots"].items():
        pos, neg = ladder_sides(r["family"])
        for side, doses in zip(SIDE_ORDER, (pos, neg), strict=True):
            out.append(
                {
                    "kind": "ladder",
                    "root": rid,
                    "side": side,
                    "states": [
                        {"kind": "ladder", "root": rid, "dose": d, "h": h}
                        for h, d in enumerate(doses, 1)
                    ],
                }
            )
    for i, seq in enumerate(rec["compositions"]):
        out.append(
            {
                "kind": "composition",
                "index": i,
                "roots": list(seq),
                "states": [
                    {
                        "kind": "composition",
                        "prefix": list(seq[:h]),
                        "dose": COMPOSITION_DOSE,
                        "h": h,
                    }
                    for h in range(1, len(seq) + 1)
                ],
            }
        )
    return out


def training_states(trajs, max_h: int = TRAIN_MAX_H) -> list:
    return [s for t in trajs for s in t["states"] if s["h"] <= max_h]


def bank_recipe() -> dict:
    roots = make_roots()
    comps = make_compositions(roots)
    per_root = {rid: sum(map(len, ladder_sides(r["family"]))) for rid, r in roots.items()}
    comp_states = len({tuple(s[:h]) for s in comps for h in range(2, len(s) + 1)})
    graphs, inner = {}, {}
    for policy in ("source", "concept"):
        assign = W.components(roots, comps, policy)
        table = W.component_table(roots, assign, per_root)
        W.add_composition_states(table, assign, comps, [len(s) - 1 for s in comps])
        folds = W.outer_folds(table)
        graphs[policy] = {
            "assignment": assign,
            "table": table,
            "n_components": len(table),
            "size_distribution": sorted((e["n_states"] for e in table.values()), reverse=True),
            "fold_of": folds,
            "folds": W.fold_report(table, folds),
        }
        inner[policy] = {
            f: W.inner_allocation(
                [c for c, ff in folds.items() if ff != f], seed=f"wme-bank:inner:{f}"
            )
            for f in range(W.N_OUTER_FOLDS)
        }
    rec = {
        "label": "WM-E trajectory-bank RECIPE (PREREG_WME §4); not realised, not evaluated",
        "namespace": NAMESPACE,
        "families": list(FAMILIES),
        "nominals": NOMINALS,
        "ladder": list(LADDER),
        "ladder_sides": {f: dict(zip(SIDE_ORDER, ladder_sides(f), strict=True)) for f in FAMILIES},
        "stop_rule": {"dnll_gt": DNLL_STOP, "keep_endpoint": True, "side_order": list(SIDE_ORDER)},
        "duplicate_rule": "equal effective parameters -> visited once (first dose)",
        "composition": {
            "n_bundles": len(comps),
            "dose": COMPOSITION_DOSE,
            "slots": COMPOSITION_SLOTS,
        },
        "horizon_rule": {
            "definition": "steps on the registered trajectory",
            "train_max_h": TRAIN_MAX_H,
            "test_h": list(TEST_H),
        },
        "roots": roots,
        "compositions": [list(c) for c in comps],
        "construction_allocation": construction_allocation(roots),
        "pilot_overlap": pilot_overlap(roots),
        "ladder_states_per_root": per_root,
        "composition_states": comp_states,
        "n_states_max": sum(per_root.values()) + comp_states,
        "components": graphs,
        "inner_allocation": inner,
    }
    rec["recipe_hash"] = hashlib.sha256(
        json.dumps(rec, sort_keys=True, default=str).encode()
    ).hexdigest()
    return rec
