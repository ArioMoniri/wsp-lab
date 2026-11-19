"""Item-exposure and split contract for the confirmatory panel ``wme-c1`` (Board A condition 1; PREREG_WME §3).

* **Split roles** (:data:`SPLIT_ROLES`): every use of items has exactly one split. Bank labels, predictor training,
  inner selection, conformal calibration and the H-T held-out evaluation all use ``validation`` items; their
  separation is by **trajectory** (source-closure components, ``wm.wmclosure``), not by item. Search feedback uses
  ``dev``; final scoring and the generation audit use the sealed ``test`` split, read once after every winner is
  frozen. Bank labels never use dev or test items.
* **Groups** (:func:`group_key`): one group per source question/document; every transformation of a question
  (sycophancy neutral/suggested prompts and templates; an MMLU-Pro question used by two objectives) shares it.
* **Near-duplicates** (:func:`near_duplicate_groups`): word 5-gram shingles of the normalised text; MinHash (128
  universal hashes mod 2^31-1, 32 bands x 4 rows, fixed seed) proposes pairs, exact Jaccard >= 0.7 confirms; union-find merges.
  Texts with fewer than 5 words use the whole normalised text (exact match). Computed over the full source pool
  before exclusion, so a near-duplicate of any exposed item (including a pilot sealed item, identified by its hash
  in a manifest) is excluded with it; the sealed files themselves are never opened.
* **Exposures** (:data:`EXPOSURES`, :func:`manifest_hashes`): hash lists read from panel manifests only, through
  a guard that accepts only a file named ``manifest.json`` outside any ``sealed*`` directory, opened with
  ``O_NOFOLLOW``; the files a manifest lists are never opened. :func:`exclude_groups` drops every group that
  contains an exposed hash; :func:`intersection_audit` gives the hash-only table per (exposure, split, new split).
* **Feasibility** (:func:`feasibility`): requested sizes against available groups after exclusion; exclusions are
  never relaxed to fit. Sycophancy has at most 1,813 - 1,344 = 469 questions left before further exclusions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from wsp.wm.panel import normalise, text_hash
from wsp.wm.plan import SealedError

SPLITS = ("construction", "validation", "dev", "test")
SPLIT_ROLES = {
    "construction": (
        "direction_construction",
        "lora_training",
        "field_probe_collection",
        "tap_probes",
        "steer_activation_means",
    ),
    "validation": (
        "bank_labels",
        "predictor_training",
        "predictor_inner_selection",
        "conformal_calibration",
        "ht_heldout_evaluation",
        "trust_region_assignment",
    ),
    "dev": ("search_feedback", "winner_selection", "best_of_n_screen", "lora_checkpoint_selection"),
    "test": ("final_scoring", "generation_audit"),
}
REQUIRED_ROLES = (
    "direction_construction",
    "lora_training",
    "field_probe_collection",
    "bank_labels",
    "predictor_training",
    "conformal_calibration",
    "search_feedback",
    "final_scoring",
    "generation_audit",
)
SEPARATION = {
    "construction": "item-separated from every other split",
    "validation": "item-separated from dev/test; H-T train vs held-out separation is by trajectory "
    "(closure component), labels of both are measured on these items",
    "dev": "item-separated; search episodes separated by episode roots",
    "test": "item-separated; sealed; read once after all winners are frozen",
}
NEAR_DUP_SHINGLE = 5
NEAR_DUP_JACCARD = 0.7
MINHASH_PERMS = 128
MINHASH_BANDS = 32
MINHASH_SEED = 20261117

# Known exposures, each identified by a panel manifest (hash lists only).
EXPOSURES = {
    "wme-v0-p1": {"manifest": "data/wme/panels/wme-v0-p1/manifest.json", "splits": "all"},
    "wme-p2": {"manifest": "data/wme/panels/wme-p2/manifest.json", "splits": "all"},
    "calib27-20261116": {
        "manifest": "data/wme/panels/wme-v0-p1/manifest.json",
        "splits": ["construction", "validation", "dev"],
        "note": "27B engineering calibration (scripts/wme_27b_calib.py) read pilot-1 construction (field), "
        "validation (timing) and dev (field-length scan) items",
    },
    "wme-p2-smoke": {
        "manifest": "data/wme/panels/wme-p2/manifest.json",
        "splits": ["construction"],
        "note": "Qwen3.5-0.8B engineering smoke runs (construction items only)",
    },
    # SYNTHESIS-r11 R7, RULING-wme-post-pilot3 P5: every split of both pilot-3 panels, including the holdout split,
    # the unscored rest of the dev pools and the construction items behind the teacher, STaR and RL outputs
    "wme-p3s": {
        "manifest": "data/wme/panels/wme-p3s/manifest.json",
        "splits": "all",
        "note": "pilot-3 stage S (slope transport): construction and validation items of six objectives",
    },
    "wme-p3h": {
        "manifest": "data/wme/panels/wme-p3h/manifest.json",
        "splits": "all",
        "note": "pilot-3 stage H (generated accuracy): construction, holdout and dev pools of every panel",
    },
    # RULING-wme-stagew §2: the stage-W dev panel is exposed before wme-c2-gen is built
    "wme-w": {
        "manifest": "data/wme/panels/wme-w/manifest.json",
        "splits": "all",
        "note": "stage W (EXPLORATORY): 500 MMLU-Pro and 500 SuperGPQA dev items scored for every library member",
    },
    # RULING-wme-stagew2 §2: the stage-W2 dev panel is exposed before any W2 generation (and before an H-T3 panel)
    "wme-w2": {
        "manifest": "data/wme/panels/wme-w2/manifest.json",
        "splits": "all",
        "note": "stage W2 (EXPLORATORY): 500 MMLU-Pro and 500 SuperGPQA dev items scored for every W2 member",
    },
}

# Proposed sizes (groups) for PREREG v1; the power simulation (wm.powersim) may change them before the freeze.
DEFAULT_SIZES = {
    "sycophancy_real": {"construction": 64, "validation": 96, "dev": 128, "test": 160},
    "gsm8k_gold_lp": {"construction": 1024, "validation": 64, "dev": 128, "test": 128},
    "mathcomp_gold_lp": {"construction": 1024, "validation": 64, "dev": 128, "test": 128},
    "wikitext_nll": {"construction": 64, "validation": 64, "dev": 64, "test": 48},
    "mmlupro_gold_lp": {"construction": 64, "validation": 64, "dev": 64, "test": 48},
    # binding constraint: 127 harmful groups remain after excluding both pilots (exposure audit 2026-11-17)
    "refusal_margin_harmful": {"construction": 20, "validation": 40, "dev": 32, "test": 32},
    "refusal_margin_harmless": {"construction": 64, "validation": 64, "dev": 64, "test": 48},
    "sycophancy_gap": {"construction": 64, "validation": 64, "dev": 64, "test": 48},
}


def split_of(role: str) -> str:
    for s, rs in SPLIT_ROLES.items():
        if role in rs:
            return s
    raise KeyError(role)


# ------------------------------------------------------------------ groups and near-duplicates


def group_text(objective: str, item: dict) -> str:
    """The source text that defines an item's group (one per question / problem / document / prompt)."""
    if objective == "sycophancy_real":
        return item["question"]
    if objective in ("mmlupro_gold_lp", "sycophancy_gap"):
        return item["question"] + "||" + "||".join(item["options"])
    if objective in ("gsm8k_gold_lp",):
        return item["question"]
    if objective == "mathcomp_gold_lp":
        return item["problem"]
    if objective == "wikitext_nll":
        return item["text"]
    return item["prompt"]


def group_key(objective: str, item: dict) -> str:
    return text_hash(group_text(objective, item))


def shingles(text: str, k: int = NEAR_DUP_SHINGLE) -> frozenset:
    w = normalise(text).split()
    if len(w) < k:
        return frozenset([" ".join(w)])
    return frozenset(" ".join(w[i : i + k]) for i in range(len(w) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


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


def near_duplicate_groups(texts: dict, threshold: float = NEAR_DUP_JACCARD) -> dict:
    """{key: group representative}; exact-normalised duplicates and confirmed near-duplicates share one group."""
    keys = sorted(texts)
    uf = _UF(keys)
    sh = {k: shingles(texts[k]) for k in keys}
    exact = {}
    for k in keys:
        n = normalise(texts[k])
        if n in exact:
            uf.union(exact[n], k)
        else:
            exact[n] = k
    rng = np.random.default_rng(MINHASH_SEED)
    P = np.int64((1 << 31) - 1)  # products of two values < 2^31 fit in int64
    a = rng.integers(1, P, size=(MINHASH_PERMS, 1), dtype=np.int64)
    b = rng.integers(0, P, size=(MINHASH_PERMS, 1), dtype=np.int64)
    rows = MINHASH_PERMS // MINHASH_BANDS
    buckets: dict = {}
    for k in keys:
        hs = np.array([_h64(x) % int(P) for x in sh[k]], dtype=np.int64)[None, :]
        sig = ((a * hs + b) % P).min(1)
        for band in range(MINHASH_BANDS):
            key = (band, sig[band * rows : (band + 1) * rows].tobytes())
            buckets.setdefault(key, []).append(k)
    for members in buckets.values():
        for i in range(1, len(members)):
            x, y = members[0], members[i]
            if uf.find(x) != uf.find(y) and jaccard(sh[x], sh[y]) >= threshold:
                uf.union(x, y)
        for i in range(1, len(members)):
            for j in range(i + 1, len(members)):
                x, y = members[i], members[j]
                if uf.find(x) != uf.find(y) and jaccard(sh[x], sh[y]) >= threshold:
                    uf.union(x, y)
    return {k: uf.find(k) for k in keys}


# ------------------------------------------------------------------ exposures


def manifest_hashes(path) -> dict[str, set]:
    """{split: set(hashes)} from a panel manifest's hash lists. Refuses anything that is not a ``manifest.json``
    outside ``sealed*``; opens the leaf with ``O_NOFOLLOW``; never opens a file the manifest lists."""
    p = Path(os.path.abspath(path))
    if p.name != "manifest.json" or any(part.startswith("sealed") for part in p.parts):
        raise SealedError(f"{path}: only panel manifests are read for exposure audits")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        man = json.loads(f.read().decode())
    out: dict[str, set] = {}
    for splits in man.get("objectives", {}).values():
        for s, rec in splits.items():
            out.setdefault(s, set()).update(rec["hashes"])
    out.setdefault("test", set()).update(man.get("sealed_hashes", []))
    return out


def load_exposures(root, registry=EXPOSURES) -> dict[str, dict[str, set]]:
    out = {}
    for name, spec in registry.items():
        hs = manifest_hashes(Path(root) / spec["manifest"])
        keep = hs if spec["splits"] == "all" else {s: hs.get(s, set()) for s in spec["splits"]}
        out[name] = keep
    return out


def all_exposed(exposures) -> set:
    return {h for e in exposures.values() for hs in e.values() for h in hs}


def exclude_groups(pool, exposures) -> tuple[list, dict]:
    """Drop every group containing an exposed hash (group-level exclusion)."""
    bad = all_exposed(exposures)
    bad_groups = {r["group"] for r in pool if r["hash"] in bad}
    kept = [r for r in pool if r["group"] not in bad_groups]
    return kept, {
        "excluded_groups": len(bad_groups),
        "excluded_items": len(pool) - len(kept),
        "exposed_hashes_matched": len({r["hash"] for r in pool} & bad),
    }


def intersection_audit(new_splits: dict, exposures: dict) -> dict:
    """{exposure: {exposure split: {new split: |intersection|}}} by hash only."""
    return {
        name: {s: {n: len(hs & set(new)) for n, new in new_splits.items()} for s, hs in e.items()}
        for name, e in exposures.items()
    }


def audit_clean(audit: dict) -> bool:
    return all(v == 0 for e in audit.values() for s in e.values() for v in s.values())


def feasibility(available_groups: dict, sizes: dict) -> dict:
    out = {}
    for obj, n in available_groups.items():
        req = sum(sizes[obj].values()) if obj in sizes else None
        out[obj] = {
            "available_groups": int(n),
            "requested": req,
            "sizes": sizes.get(obj),
            "feasible": req is not None and req <= n,
            "slack": None if req is None else int(n) - req,
        }
    return out
