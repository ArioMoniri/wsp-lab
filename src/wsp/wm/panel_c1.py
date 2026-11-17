"""The confirmatory panel ``wme-c1`` (PREREG_WME §3; Board A condition 1): builder and sealed-test guard.

The builder is code only; the panel is built by ``scripts/wme_c1_build_panel.py`` after the freeze, never here.

* **Groups.** Every item belongs to the near-duplicate group of its source text (``wm.exposure``: question,
  problem, document or prompt; neutral/suggested sycophancy prompts share their question). One item is kept per
  group (the smallest hash); a group seen under an earlier objective (``panel2.OBJECTIVES`` order) is not reused.
* **Exclusion.** Every group containing a hash of any registered exposure (pilot 1, pilot 2 including its unread
  sealed test, the 27B calibration exposures, smoke runs) is dropped, using manifest hash lists only.
* **Splits.** Groups are ranked by sha256(``SALT``:group) and assigned whole, test first, then dev, validation,
  construction. Sizes are never relaxed: a shortfall raises. Pairwise disjointness of all four splits and a clean
  exposure audit are asserted in memory before anything is written. Test items are written only under
  ``sealed_test/``; planner code reaches items only through ``wm.plan.PlannerItemStore``.
* **Sealed reader** (:class:`SealedTestReader`): one read per panel, only after every arm's winner is frozen
  (non-empty digests). The leaf must be a regular file at its lexical path (realpath equal, ``O_NOFOLLOW``), and
  its sha256 must equal the manifest's. A receipt with the winners' digest is written; a second read is refused.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path

from wsp.wm import exposure as X
from wsp.wm import panel2 as P2
from wsp.wm.panel import text_hash
from wsp.wm.plan import SealedError

SALT = "wme-c1-panel-20261117"
SPLITS = ("test", "dev", "validation", "construction")  # assignment order


def _rank(h: str, salt: str = SALT) -> str:
    return hashlib.sha256(f"{salt}:{h}".encode()).hexdigest()


def grouped_pool(pools: dict, group_text=X.group_text) -> dict:
    """Attach near-duplicate group ids (computed jointly over every objective's pool) to every item."""
    texts = {(o, i): group_text(o, r) for o, rows in pools.items() for i, r in enumerate(rows)}
    keyed = {f"{o}\x00{i:08d}": t for (o, i), t in texts.items()}
    rep = X.near_duplicate_groups(keyed)
    out = {}
    for o, rows in pools.items():
        out[o] = []
        for i, r in enumerate(rows):
            g = X.text_hash(keyed[rep[f"{o}\x00{i:08d}"]])
            out[o].append(dict(r, group=g))
    return out


def build_c1_panel(
    pools: dict, sizes: dict, out_dir, exposures: dict, group_text=X.group_text, salt: str = SALT
) -> dict:
    """``salt`` ranks groups into splits; the 27B panel (``wm.sep27``) passes its own fresh salt."""
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; panels are frozen once built")
    order = [o for o in P2.OBJECTIVES if o in pools] + sorted(
        o for o in pools if o not in P2.OBJECTIVES
    )
    grouped = grouped_pool(pools, group_text)
    owner, final, exclusion = {}, {}, {}
    for obj in order:
        kept, rep = X.exclude_groups(grouped[obj], exposures)
        by_group = {}
        for r in sorted(kept, key=lambda r: r["hash"]):
            if r["group"] in owner and owner[r["group"]] != obj:
                continue
            by_group.setdefault(r["group"], r)
        for g in by_group:
            owner.setdefault(g, obj)
        exclusion[obj] = dict(
            rep, groups_available=len(by_group), duplicates_dropped=len(kept) - len(by_group)
        )
        final[obj] = sorted(by_group.values(), key=lambda r: _rank(r["group"], salt))
    splits = {s: {} for s in SPLITS}
    for obj, rows in final.items():
        need = sum(sizes[obj][s] for s in SPLITS)
        if len(rows) < need:
            raise ValueError(
                f"{obj}: {len(rows)} groups available for {need} requested (never relaxed)"
            )
        i = 0
        for s in SPLITS:
            splits[s][obj] = sorted(rows[i : i + sizes[obj][s]], key=lambda r: r["hash"])
            i += sizes[obj][s]
    hashes = {s: {r["hash"] for o in final for r in splits[s][o]} for s in SPLITS}
    groups = {s: {r["group"] for o in final for r in splits[s][o]} for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and (hashes[a] & hashes[b] or groups[a] & groups[b]):
                raise AssertionError(f"splits {a} and {b} intersect")
    audit = X.intersection_audit({s: hashes[s] for s in SPLITS}, exposures)
    if not X.audit_clean(audit):
        raise AssertionError("a written split intersects a registered exposure")
    manifest = {
        "salt": salt,
        "sizes": sizes,
        "split_roles": {s: list(r) for s, r in X.SPLIT_ROLES.items()},
        "separation": X.SEPARATION,
        "near_duplicate_policy": {
            "shingle_words": X.NEAR_DUP_SHINGLE,
            "jaccard": X.NEAR_DUP_JACCARD,
            "minhash_perms": X.MINHASH_PERMS,
            "bands": X.MINHASH_BANDS,
            "seed": X.MINHASH_SEED,
        },
        "exclusion": exclusion,
        "exposures": sorted(exposures),
        "audit": audit,
        "audit_clean": True,
        "objectives": {
            o: {
                s: {
                    "n": len(splits[s][o]),
                    "hashes": [r["hash"] for r in splits[s][o]],
                    "groups": [r["group"] for r in splits[s][o]],
                }
                for s in SPLITS
            }
            for o in final
        },
    }
    (out / "open").mkdir(parents=True)
    (out / "sealed_test").mkdir()
    files = {}
    for s in SPLITS:
        lines = [
            json.dumps(dict(r, objective=o), sort_keys=True) for o in final for r in splits[s][o]
        ]
        data = ("\n".join(lines) + "\n").encode()
        path = out / ("sealed_test/test.jsonl" if s == "test" else f"open/{s}.jsonl")
        path.write_bytes(data)
        files[str(path.relative_to(out))] = hashlib.sha256(data).hexdigest()
    manifest["sealed_hashes"] = sorted(hashes["test"])
    manifest["files"] = files
    manifest["content_hash"] = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()
    ).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


class SealedTestReader:
    def __init__(self, panel_root, receipt_dir=None):
        self.root = Path(os.path.abspath(panel_root))
        self.receipt_dir = Path(receipt_dir) if receipt_dir else self.root / "sealed_read_receipt"
        self.receipt = self.receipt_dir / "sealed_read.json"

    def read(self, frozen_winners: dict):
        if not frozen_winners or any(not v for v in frozen_winners.values()):
            raise SealedError("sealed test items are read only after every arm's winner is frozen")
        if self.receipt.exists():
            raise SealedError(f"the sealed test split was already read ({self.receipt})")
        man = json.loads((self.root / "manifest.json").read_text())
        lexical = self.root / "sealed_test" / "test.jsonl"
        real = Path(os.path.realpath(lexical))
        if real != lexical:
            raise SealedError(f"{lexical} resolves elsewhere ({real}); refused")
        fd = os.open(lexical, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        if sha != man["files"]["sealed_test/test.jsonl"]:
            raise SealedError("sealed test content hash differs from the manifest")
        digest = hashlib.sha256(json.dumps(frozen_winners, sort_keys=True).encode()).hexdigest()
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.receipt.write_text(
            json.dumps({"winners_digest": digest, "sha256": sha, "time": time.time()}, indent=1)
        )
        return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


# ------------------------------------------------------------------ v1 builder (RULING-wme-v2 A5)

SPLITS_V1 = ("sealed_gen", "sealed_test", "dev", "bank", "construction")
_FILES_V1 = {
    "sealed_gen": "sealed_gen/gen.jsonl",
    "sealed_test": "sealed_test/test.jsonl",
    "dev": "open/dev.jsonl",
    "bank": "open/bank.jsonl",
    "construction": "open/construction.jsonl",
}


def _text_v1(obj, r, group_text):
    return r["question"] if obj == "triviaqa_gen" else group_text(obj, r)


def build_c1_panel_v1(
    pools: dict, sizes: dict, out_dir, exposures: dict, group_text=X.group_text, salt: str = SALT
) -> dict:
    """PREREG_WME v1 §3.4: groups by the frozen v1 near-duplicate policy computed jointly over every pool
    (``wm.neardup``); group-level exclusion of every exposed hash; cross-objective groups owned by the first
    objective in the registered order; per objective, groups ranked by sha256(salt:group) and assigned in the
    order sealed generation, sealed test, dev, bank, construction (``"rest"`` = every remaining group). Sealed
    splits go under ``sealed_test/`` and ``sealed_gen/`` (unreachable through ``PlannerItemStore``). Exclusions are
    never relaxed: a shortfall raises."""
    from wsp.wm import neardup as ND

    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; panels are frozen once built")
    order = [o for o in P2.OBJECTIVES if o in pools] + sorted(
        o for o in pools if o not in P2.OBJECTIVES
    )
    texts, where = {}, {}
    for obj in order:
        for i, r in enumerate(pools[obj]):
            k = f"{obj}\x00{i:07d}"
            texts[k] = _text_v1(obj, r, group_text)
            where[k] = (obj, r)
    rep = ND.groups(texts)
    exposed = X.all_exposed(exposures)
    bad = {rep[k] for k in texts if where[k][1]["hash"] in exposed}
    owner, final, exclusion = {}, {}, {}
    for obj in order:
        ks = sorted(k for k in texts if where[k][0] == obj)
        by = {}
        for k in sorted(ks, key=lambda k: where[k][1]["hash"]):
            g = rep[k]
            if g in bad or owner.setdefault(g, obj) != obj:
                continue
            by.setdefault(g, dict(where[k][1], group=text_hash(texts[g])))
        exclusion[obj] = {
            "pool_items": len(ks),
            "excluded_groups": len({rep[k] for k in ks} & bad),
            "available_groups": len(by),
        }
        final[obj] = sorted(by.values(), key=lambda r: _rank(r["group"], salt))
    splits = {s: {} for s in SPLITS_V1}
    for obj, rows in final.items():
        fixed = sum(v for v in sizes[obj].values() if v != "rest")
        if len(rows) < fixed:
            raise ValueError(
                f"{obj}: {len(rows)} groups available for {fixed} requested (never relaxed)"
            )
        i = 0
        for s in SPLITS_V1:
            n = len(rows) - i if sizes[obj][s] == "rest" else int(sizes[obj][s])
            splits[s][obj] = sorted(rows[i : i + n], key=lambda r: r["hash"])
            i += n
    hashes = {s: {r["hash"] for o in final for r in splits[s][o]} for s in SPLITS_V1}
    groups = {s: {r["group"] for o in final for r in splits[s][o]} for s in SPLITS_V1}
    for a_, b_ in itertools.combinations(SPLITS_V1, 2):
        if hashes[a_] & hashes[b_] or groups[a_] & groups[b_]:
            raise AssertionError(f"splits {a_} and {b_} intersect")
    audit = X.intersection_audit({s: hashes[s] for s in SPLITS_V1}, exposures)
    if not X.audit_clean(audit):
        raise AssertionError("a written split intersects a registered exposure")
    for d in ("open", "sealed_test", "sealed_gen"):
        (out / d).mkdir(parents=True, exist_ok=False) if d != "open" else (out / d).mkdir(
            parents=True
        )
    files = {}
    for s in SPLITS_V1:
        lines = [
            json.dumps(dict(r, objective=o), sort_keys=True) for o in final for r in splits[s][o]
        ]
        data = ("\n".join(lines) + ("\n" if lines else "")).encode()
        (out / _FILES_V1[s]).write_bytes(data)
        files[_FILES_V1[s]] = hashlib.sha256(data).hexdigest()
    manifest = {
        "salt": salt,
        "version": "wme-c1-v1",
        "sizes": sizes,
        "policy": ND.POLICY,
        "exclusion": exclusion,
        "exposures": sorted(exposures),
        "audit": audit,
        "audit_clean": True,
        "split_hashes": files,
        "objectives": {
            o: {
                s: {
                    "n": len(splits[s][o]),
                    "hashes": [r["hash"] for r in splits[s][o]],
                    "groups": [r["group"] for r in splits[s][o]],
                }
                for s in SPLITS_V1
            }
            for o in final
        },
        "sealed_hashes": sorted(hashes["sealed_test"] | hashes["sealed_gen"]),
    }
    manifest["content_hash"] = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()
    ).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest
