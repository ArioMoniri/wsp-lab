"""WM-specific source-closure mapping for the trajectory bank (Board A condition 2; PREREG_WME §4).

``wsp.msrm.closure`` recognises zoo random-direction and abliteration keys only. WM-E roots carry their own
immutable identities (``wm.bank`` writes them into every recipe):

* ``recipe``: sha256 of the canonical recipe (family, tensor, parameters, seeds);
* ``direction``: the edited direction independent of dose and factor (a scale block, a vocabulary row set, an
  unembedding token or random seed, a noise seed on a tensor, a contrast pair at a layer, a LoRA source);
* ``contrast_source``: the construction-item subset(s) a contrast direction was estimated from;
* ``adapter_source``: the construction-item subset an adapter was trained on (seeds sharing it share the source);
* ``concept``: the semantic identity (contrast pair, LoRA objective);
* ``pair``: a registered matched pair (proj_remove contrast/unembedding and its matched random direction);
* ``realised_delta``: sha256 of a realised bf16 delta, known only when a state is built; :func:`join_realised`
  merges duplicates at build time.

Join policies (registered): ``"source"`` (default) joins on recipe, direction, contrast_source, adapter_source,
pair and realised_delta; ``"concept"`` (conservative sensitivity) also joins on concept. Every ladder state, prefix,
reverse order and dose of a root stays in its root's component; a composition joins all its roots transitively.
:func:`components` returns the component of every root; :func:`outer_folds` assigns whole components to five
folds stratified by the component's family *signature* (the sorted set of families it contains, never a single
label), balancing state counts; :func:`inner_allocation` splits each outer training set into fit / selection /
calibration components.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

JOIN_KEYS = {
    "source": (
        "recipe",
        "direction",
        "contrast_source",
        "adapter_source",
        "pair",
        "realised_delta",
    ),
    "concept": (
        "recipe",
        "direction",
        "contrast_source",
        "adapter_source",
        "pair",
        "realised_delta",
        "concept",
    ),
}
MULTI_VALUED = (
    "contrast_source",
    "realised_delta",
)  # lists of values; every other key is one (composite) value
N_OUTER_FOLDS = 5
INNER = {"selection": 0.2, "calibration": 0.2}


def recipe_hash(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()


class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted([ra, rb])
            self.p[hi] = lo


def components(roots: dict, compositions=(), policy: str = "source") -> dict:
    """{root id: component representative}. ``roots[rid]["identity"]`` holds the identity keys: one composite
    value each (a list is one value), except :data:`MULTI_VALUED` keys, which list several; None never joins."""
    keys = JOIN_KEYS[policy]
    uf = _UF(sorted(roots))
    owners = defaultdict(list)
    for rid in sorted(roots):
        ident = roots[rid]["identity"]
        for k in keys:
            v = ident.get(k)
            for x in v if (k in MULTI_VALUED and isinstance(v, (list, tuple))) else [v]:
                if x is not None:
                    owners[(k, json.dumps(x, sort_keys=True))].append(rid)
    for members in owners.values():
        for m in members[1:]:
            uf.union(members[0], m)
    for seq in compositions:
        for r in seq[1:]:
            uf.union(seq[0], r)
    return {r: uf.find(r) for r in sorted(roots)}


def join_realised(comp: dict, realised: dict) -> dict:
    """Merge components whose roots share a realised-delta hash (``realised[rid] = [hashes]``; build time)."""
    uf = _UF(sorted(set(comp.values())))
    owner = {}
    for rid, hs in realised.items():
        for h in hs:
            if h in owner:
                uf.union(comp[owner[h]], comp[rid])
            else:
                owner[h] = rid
    return {r: uf.find(c) for r, c in comp.items()}


def component_table(roots: dict, comp: dict, state_counts: dict | None = None) -> dict:
    """{component: {roots, families, signature, n_states}}."""
    out = {}
    for rid, c in comp.items():
        e = out.setdefault(c, {"roots": [], "families": set(), "n_states": 0})
        e["roots"].append(rid)
        e["families"].add(roots[rid]["family"])
        e["n_states"] += (state_counts or {}).get(rid, 0)
    for c, e in out.items():
        e["families"] = sorted(e["families"])
        e["signature"] = "+".join(e["families"])
        e["roots"] = sorted(e["roots"])
    return out


def add_composition_states(table: dict, comp: dict, compositions, prefix_states) -> dict:
    """Add each composition's prefix-state count to its component (``prefix_states[i]`` for sequence i)."""
    for seq, n in zip(compositions, prefix_states, strict=True):
        table[comp[seq[0]]]["n_states"] += n
    return table


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def outer_folds(table: dict, k: int = N_OUTER_FOLDS, seed: str = "wme-bank:folds") -> dict:
    """{component: fold}; whole components, stratified by signature, greedy on state counts."""
    folds = {}
    load = [0] * k
    by_sig = defaultdict(list)
    for c, e in table.items():
        by_sig[e["signature"]].append(c)
    for sig in sorted(by_sig):
        cs = sorted(by_sig[sig], key=lambda c: (-table[c]["n_states"], _rank(seed, c)))
        count = [0] * k
        for c in cs:
            f = min(range(k), key=lambda i: (count[i], load[i], i))
            folds[c] = f
            count[f] += 1
            load[f] += table[c]["n_states"]
    return folds


def fold_report(table: dict, folds: dict, k: int = N_OUTER_FOLDS) -> dict:
    per = {f: {"components": 0, "states": 0, "signatures": defaultdict(int)} for f in range(k)}
    for c, f in folds.items():
        per[f]["components"] += 1
        per[f]["states"] += table[c]["n_states"]
        per[f]["signatures"][table[c]["signature"]] += 1
    sigs = {e["signature"] for e in table.values()}
    sparse = sorted(s for s in sigs if sum(1 for e in table.values() if e["signature"] == s) < k)
    return {
        "per_fold": {f: dict(v, signatures=dict(v["signatures"])) for f, v in per.items()},
        "n_components": len(table),
        "five_folds_feasible": len(table) >= k and all(v["components"] > 0 for v in per.values()),
        "signatures_below_k_components": sparse,
    }


def inner_allocation(train_components, seed: str = "wme-bank:inner") -> dict:
    """{component: "fit" | "selection" | "calibration"} for one outer training set (by rank)."""
    cs = sorted(train_components, key=lambda c: _rank(seed, c))
    n = len(cs)
    n_sel = max(1, round(INNER["selection"] * n)) if n >= 5 else 0
    n_cal = max(1, round(INNER["calibration"] * n)) if n >= 5 else 0
    out = {}
    for i, c in enumerate(cs):
        out[c] = "selection" if i < n_sel else ("calibration" if i < n_sel + n_cal else "fit")
    return out


def lofo(table: dict, family: str) -> list:
    """Components held out when leaving out ``family`` (every component that contains it; secondary)."""
    return sorted(c for c, e in table.items() if family in e["families"])


def direction_similarity_audit(new: dict, pilot: dict, max_cos: float = 0.5) -> dict:
    """Build-time audit that no realised bank direction is a pilot direction in disguise: |cos| of every new
    direction with every pilot direction of the same shape must stay below ``max_cos`` (a fresh namespace or
    fresh items alone are not proof of freshness)."""
    import torch

    flagged, worst = [], {}
    for k, v in new.items():
        v = v.detach().double().reshape(-1)
        best = 0.0
        for p in pilot.values():
            p = p.detach().double().reshape(-1)
            if p.shape != v.shape:
                continue
            best = max(best, float(torch.abs(v @ p) / (v.norm() * p.norm())))
        worst[k] = best
        if best >= max_cos:
            flagged.append(k)
    return {"max_cos": max_cos, "worst": worst, "flagged": sorted(flagged), "pass": not flagged}
