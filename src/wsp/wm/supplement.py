"""Harmful-refusal item supplement for ``refusal_margin_harmful`` (RULING-wme-resources §3; PREREG_WME v1 §3.2-3.3,
§8.5). Pinned sources, the frozen near-duplicate policy and hash-only audits; nothing is scored.

**Sources, in the chair's order of preference** (each pinned by revision and file sha256):
1. AdvBench harmful behaviours (``llm-attacks/llm-attacks`` ``data/advbench/harmful_behaviors.csv``; field ``goal``);
2. HarmBench text behaviours (``centerforaisafety/HarmBench``
   ``data/behavior_datasets/harmbench_behaviors_text_all.csv``), ``FunctionalCategory == "standard"`` only: the
   single-request format of the existing pool. Contextual behaviours (a long context string plus a request) and
   copyright behaviours (reproduction requests, not the harmful-refusal construct) are excluded by definition;
   this rule is fixed here, before any count is seen;
3. JailbreakBench JBB-Behaviors (``JailbreakBench/JBB-Behaviors`` ``data/harmful-behaviors.csv``; field ``Goal``).

**Policy** (PREREG v1 §3.2, frozen): groups merge by union-find when the normalised texts share an exact sha256
(``wm.panel.text_hash``) or when their character 5-gram Jaccard similarity (lower case, punctuation stripped,
whitespace collapsed) is >= 0.8, computed exactly (every source here has < 50,000 groups). Pairs are compared
only when their shingle-set sizes are within the ratio 0.8, which is a necessary condition for Jaccard >= 0.8, so
the result is exact.

**Deduplication** (:func:`dedup_sources`): sources are processed in preference order. A supplement group is
excluded if it contains (exactly or as a near-duplicate) any existing pool item of any objective, any item of an
earlier supplement source, or any exposed hash (pilot panels, sealed sets by exact hash only, engineering
exposures). **Choice** (:func:`choose`): the shortest preference-order prefix whose cumulative available groups,
added to the audited existing availability, reach the simulation-required harmful total; the N is the
simulation's, never a count chosen after scoring.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict

from wsp.wm.panel import normalise, text_hash

NEAR_DUP_K = 5
NEAR_DUP_JACCARD = 0.8
EXACT_LIMIT = 50_000
RAW = "data/wme/supplement/raw"

SOURCES = (
    {
        "name": "advbench",
        "title": "AdvBench harmful behaviours",
        "repo": "github:llm-attacks/llm-attacks",
        "revision": "a62d1307e38b3a076e614b20781c785fd860d813",
        "path": "data/advbench/harmful_behaviors.csv",
        "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
        "a62d1307e38b3a076e614b20781c785fd860d813/data/advbench/harmful_behaviors.csv",
        "local": "advbench/harmful_behaviors.csv",
        "sha256": "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1",
    },
    {
        "name": "harmbench_text",
        "title": "HarmBench text behaviours (standard)",
        "repo": "github:centerforaisafety/HarmBench",
        "revision": "c0423b952435fcc8467108d8f25962dbae5b7de2",
        "path": "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "url": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "c0423b952435fcc8467108d8f25962dbae5b7de2/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "local": "harmbench/harmbench_behaviors_text_all.csv",
        "sha256": "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
        "subset": {"FunctionalCategory": "standard"},
    },
    {
        "name": "jbb_behaviors",
        "title": "JailbreakBench JBB-Behaviors (harmful)",
        "repo": "hf:JailbreakBench/JBB-Behaviors",
        "revision": "886acc352a31533ffbcf4ef22c744658688086fc",
        "path": "data/harmful-behaviors.csv",
        "url": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/"
        "886acc352a31533ffbcf4ef22c744658688086fc/data/harmful-behaviors.csv",
        "local": "jbb/harmful-behaviors.csv",
        "sha256": "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
    },
)


class PinError(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(path, expected: str) -> str:
    got = sha256_file(path)
    if got != expected:
        raise PinError(f"{path}: sha256 {got} != pinned {expected}")
    return got


def parse(name: str, path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    if name == "advbench":
        for i, r in enumerate(rows):
            out.append({"prompt": r["goal"].strip(), "source": name, "source_id": str(i)})
    elif name == "harmbench_text":
        for r in rows:
            if r["FunctionalCategory"].strip() != "standard":
                continue
            out.append(
                {
                    "prompt": r["Behavior"].strip(),
                    "source": name,
                    "source_id": r["BehaviorID"].strip(),
                }
            )
    elif name == "jbb_behaviors":
        for r in rows:
            out.append(
                {
                    "prompt": r["Goal"].strip(),
                    "source": name,
                    "source_id": r["Index"].strip(),
                    "origin": r.get("Source", "").strip(),
                }
            )
    else:
        raise KeyError(name)
    return [r for r in out if r["prompt"]]


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def char_text(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", normalise(text))).strip()


def char_shingles(text: str, k: int = NEAR_DUP_K) -> frozenset:
    t = char_text(text)
    if len(t) < k:
        return frozenset([t])
    return frozenset(t[i : i + k] for i in range(len(t) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted([ra, rb])
            self.p[hi] = lo


def _near_pairs(left: dict, right: dict, same: bool):
    """Exact char-5-gram Jaccard >= threshold between ``left`` and ``right`` ({key: shingles}); a size-ratio filter
    (necessary for Jaccard >= t) prunes the comparisons."""
    t = NEAR_DUP_JACCARD
    rk = sorted(right, key=lambda k: len(right[k]))
    sizes = [len(right[k]) for k in rk]
    import bisect

    for a in sorted(left):
        sa = left[a]
        lo = bisect.bisect_left(sizes, t * len(sa))
        hi = bisect.bisect_right(sizes, len(sa) / t)
        for b in rk[lo:hi]:
            if same and b <= a:
                continue
            if jaccard(sa, right[b]) >= t:
                yield a, b


def dedup(new, existing: dict, exposed: set, earlier: dict | None = None) -> dict:
    """``new``: parsed items of one source; ``existing``: {key: text} of every existing pool item (all objectives);
    ``exposed``: exposed exact hashes; ``earlier``: {key: text} of earlier supplement sources' items."""
    earlier = earlier or {}
    uf = _UF()
    nk = {f"new:{i:06d}": r["prompt"] for i, r in enumerate(new)}
    sh_new = {k: char_shingles(v) for k, v in nk.items()}
    sh_old = {f"old:{k}": char_shingles(v) for k, v in existing.items()}
    sh_ear = {f"ear:{k}": char_shingles(v) for k, v in earlier.items()}
    exact = defaultdict(list)
    for d in (
        nk,
        {f"old:{k}": v for k, v in existing.items()},
        {f"ear:{k}": v for k, v in earlier.items()},
    ):
        for k, v in d.items():
            exact[text_hash(v)].append(k)
    for ks in exact.values():
        for k in ks[1:]:
            uf.union(ks[0], k)
    for a, b in _near_pairs(sh_new, sh_new, same=True):
        uf.union(a, b)
    for other in (sh_old, sh_ear):
        for a, b in _near_pairs(sh_new, other, same=False):
            uf.union(a, b)
    members = defaultdict(list)
    for k in list(nk) + list(sh_old) + list(sh_ear):
        members[uf.find(k)].append(k)
    counts = {
        "items": len(new),
        "dup_existing_items": 0,
        "dup_earlier_source_items": 0,
        "exposed_items": 0,
    }
    avail = set()
    for k in nk:
        grp = members[uf.find(k)]
        if any(m.startswith("old:") for m in grp):
            counts["dup_existing_items"] += 1
            continue
        if any(m.startswith("ear:") for m in grp):
            counts["dup_earlier_source_items"] += 1
            continue
        if any(text_hash(nk[m]) in exposed for m in grp if m.startswith("new:")):
            counts["exposed_items"] += 1
            continue
        avail.add(text_hash(nk[min(m for m in grp if m.startswith("new:"))]))
    counts["available_groups"] = len(avail)
    counts["groups_within_source"] = len({uf.find(k) for k in nk})
    return {
        "counts": counts,
        "available": sorted(avail),
        "available_digest": hashlib.sha256("\n".join(sorted(avail)).encode()).hexdigest(),
    }


def dedup_sources(by_source: dict, existing: dict, exposed: set) -> dict:
    out, earlier = {}, {}
    for s in [x["name"] for x in SOURCES if x["name"] in by_source] + sorted(
        n for n in by_source if n not in {x["name"] for x in SOURCES}
    ):
        out[s] = dedup(by_source[s], existing, exposed, earlier)
        for i, r in enumerate(by_source[s]):
            earlier[f"{s}:{i:06d}"] = r["prompt"]
    return out


def choose(existing_available: int, required: int, audits: dict) -> dict:
    """Shortest preference-order prefix of sources reaching ``required`` harmful groups."""
    pool, used = int(existing_available), []
    if pool >= required:
        return {"sources": [], "pool": pool, "required": required, "feasible": True}
    for s in [x["name"] for x in SOURCES]:
        pool += int(audits[s]["available_groups"])
        used.append(s)
        if pool >= required:
            return {"sources": used, "pool": pool, "required": required, "feasible": True}
    return {"sources": used, "pool": pool, "required": required, "feasible": False}
