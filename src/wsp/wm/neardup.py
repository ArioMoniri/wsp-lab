"""The frozen v1 near-duplicate policy (PREREG_WME v1 §3.2), computed exactly (RULING-wme-v2 A5).

Two texts are joined when their ``wm.panel.normalise`` texts share a sha256, or when their character 5-gram sets
(lower case, punctuation stripped, whitespace collapsed) have Jaccard >= 0.8. The join is an exact set-similarity
join with **prefix filtering** (tokens ordered rarest first; a set of size n is indexed on its first
n - ceil(0.8 n) + 1 tokens, and two sets with Jaccard >= 0.8 must share a prefix token) plus the size filter
0.8 |A| <= |B| <= |A| / 0.8, then exact verification. Every qualifying pair is found, so no MinHash approximation
is needed even for sources above 50,000 groups (v1 §3.2 allows MinHash there; the exact join is a superset of
its recall). Groups are the union-find closure (:func:`groups`).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from wsp.wm.panel import normalise, text_hash

POLICY = {
    "k": 5,
    "jaccard": 0.8,
    "normalise": "wm.panel.normalise + punctuation stripped",
    "exact_join": "sha256 of wm.panel.normalise",
    "method": "exact prefix-filtered set-similarity join",
    "id": "wme-c1-group-v1",
}
_P = re.compile(r"[^\w\s]", re.UNICODE)


def char_text(text: str) -> str:
    return re.sub(r"\s+", " ", _P.sub(" ", normalise(text))).strip()


def char_shingles(text: str, k: int = POLICY["k"]) -> frozenset:
    t = char_text(text)
    if len(t) < k:
        return frozenset([t])
    return frozenset(t[i : i + k] for i in range(len(t) - k + 1))


def jaccard(a, b) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pairs(texts: dict, t: float = POLICY["jaccard"]) -> set:
    keys = sorted(texts)
    sh = {k: char_shingles(texts[k]) for k in keys}
    freq = Counter(tok for s in sh.values() for tok in s)
    order = {k: sorted(sh[k], key=lambda x: (freq[x], x)) for k in keys}
    index = defaultdict(list)
    out = set()
    for k in sorted(keys, key=lambda k: (len(sh[k]), k)):
        n = len(sh[k])
        p = n - math.ceil(t * n - 1e-12) + 1
        cands = set()
        for tok in order[k][:p]:
            for other in index[tok]:
                if len(sh[other]) >= t * n - 1e-12:
                    cands.add(other)
            index[tok].append(k)
        for o in cands:
            if jaccard(sh[k], sh[o]) >= t:
                out.add(tuple(sorted((k, o))))
    return out


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


def groups(texts: dict, t: float = POLICY["jaccard"]) -> dict:
    """{key: group representative} under exact-hash and near-duplicate joins (transitive)."""
    uf = _UF(sorted(texts))
    by_hash = {}
    for k in sorted(texts):
        h = text_hash(texts[k])
        if h in by_hash:
            uf.union(by_hash[h], k)
        else:
            by_hash[h] = k
    for a, b in pairs(texts, t):
        uf.union(a, b)
    return {k: uf.find(k) for k in texts}
