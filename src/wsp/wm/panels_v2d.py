"""Fresh confirmatory panels after stage W2 (PREREG_WME §II.22.3; Board A v2 required change 3). CPU, hash-only.

* **Required exposures** (:data:`REQUIRED_EXPOSURES`). Every builder here refuses to run unless every registered
  exposure is present in the loaded registry, **including ``wme-w2``** (the stage-W2 panel, registered on its own
  branch; pending until that branch merges). The build order is therefore: W2 merges -> ``wme-c1`` -> ``wme-c2-gen``
  -> any H-T3 panel; W2 is an exposure of every confirmatory panel, never the other way round.
* **Joint MMLU-Pro allocation** (:func:`joint_groups`, :func:`allocate`). One salted deterministic assignment of the
  fresh MMLU-Pro pool, consumed by every builder that draws MMLU-Pro. Groups are the union-find closure of **both**
  duplicate policies in use: ``wm.neardup`` (v1: exact normalised sha256 + character 5-gram Jaccard >= 0.8) and
  ``wm.exposure.near_duplicate_groups`` (exact normalised text + word 5-gram MinHash Jaccard >= 0.7). Groups touching
  an exposed hash are dropped; the rest are ranked by sha256(salt + ":" + smallest member hash) and dealt in rank
  order to the consumers in :data:`CONSUMERS` order; the remainder is ``reserve``. The assignment and its sha256
  are written before any panel is built; builders take the assignment, not their own salt.
* **Shortfall rule** (:func:`allocate`, :func:`shortfall`). ``wme-c1`` sizes are never relaxed (a shortfall
  raises). Every ``wme-c2-gen`` panel takes ``min(required, available)`` and records the reduction with
  ``restate_power = True``; the power is restated before any outcome is read (RULING-wme-prereg-v2-issues item 5;
  SYNTHESIS-r11 R1).
* **Audit** (:func:`audit`). Hash-only report over every (panel, split): counts per source, pairwise hash and
  joint-group intersections (cross-panel duplication), intersections with every registered exposure. Written into
  each manifest and published; no text leaves the builder.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from wsp.wm import exposure as X
from wsp.wm import neardup as ND
from wsp.wm import p3panel as PP
from wsp.wm import panel_c1 as PC

REQUIRED_EXPOSURES = (
    "wme-v0-p1",
    "wme-p2",
    "calib27-20261116",
    "wme-p2-smoke",
    "wme-p3s",
    "wme-p3h",
    "wme-w",
    "wme-w2",
)
ALLOC_SALT = "wme-mmlupro-v2c"
CONSUMERS = (
    ("wme-c1:mmlupro_gold_lp", 232, "raise"),
    ("wme-c2-gen:mmlupro_gen", 3463, "reduce"),
)
C2GEN_SALT = "wme-c2-gen-panel-v2"
C2GEN_SIZES = {
    "mmlupro_gen": {"sealed_test": 3463},
    "supergpqa_gen": {"sealed_test": 1784},
    "guard_harmless": {"calibration": 128, "sealed_test": 523},
    "guard_harmful": {"calibration": 128, "sealed_test": 200},
}


class MissingExposure(RuntimeError):
    pass


def require_exposures(exposures: dict, required=REQUIRED_EXPOSURES):
    missing = [e for e in required if e not in exposures]
    if missing:
        raise MissingExposure(
            f"exposures not registered: {missing}; build only after they are (wme-w2 merges first)"
        )


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
            self.p[max(ra, rb)] = min(ra, rb)


def joint_groups(texts: dict) -> dict:
    """{key: representative} under the union of the v1 character policy and the word-MinHash policy."""
    uf = _UF(sorted(texts))
    for rep in (ND.groups(texts), X.near_duplicate_groups(texts)):
        for k, r in rep.items():
            uf.union(k, r)
    return {k: uf.find(k) for k in texts}


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def allocate(
    pool: dict,
    exposed: set,
    consumers=CONSUMERS,
    salt: str = ALLOC_SALT,
    preassigned=None,
    eligible=None,
) -> dict:
    """``pool`` {item hash: text}. Returns the allocation record (assignment {hash: consumer | "reserve"}).

    ``preassigned`` {item hash: consumer}: groups already committed before the allocation (the bank roots'
    construction shards, PREREG §II.23.3) go whole to that consumer, whatever their exposure, and are not dealt.
    ``eligible`` {consumer: item hashes}: a consumer is dealt only groups with at least one item in its own pool;
    a skipped group stays in rank order for the next consumer (or the reserve)."""
    rep = joint_groups(pool)
    members: dict = {}
    for h, r in rep.items():
        members.setdefault(r, []).append(h)
    pre = {}
    for h, c in sorted((preassigned or {}).items()):
        if h in rep:
            pre.setdefault(rep[h], c)
    bad = {r for r, hs in members.items() if set(hs) & exposed and r not in pre}
    ok = sorted(
        (r for r in members if r not in bad and r not in pre),
        key=lambda r: hashlib.sha256(f"{salt}:{min(members[r])}".encode()).hexdigest(),
    )
    assignment, out, i = {}, [], 0
    for r, c in pre.items():
        for h in members[r]:
            assignment[h] = c
    queue = list(ok)
    for name, need, rule in consumers:
        el = None if eligible is None or name not in eligible else set(eligible[name])
        take, rest = [], []
        for r in queue:
            if len(take) < need and (el is None or set(members[r]) & el):
                take.append(r)
            else:
                rest.append(r)
        n = len(take)
        if n < need and rule == "raise":
            raise ValueError(f"{name}: {n} eligible fresh groups for {need} (never relaxed)")
        for r in take:
            for h in members[r]:
                assignment[h] = name
        out.append({"consumer": name, "required": need, "assigned_groups": n, **shortfall(need, n)})
        queue = rest
        i += n
    for r in queue:
        for h in members[r]:
            assignment[h] = "reserve"
    rec = {
        "salt": salt,
        "policies": [
            ND.POLICY["id"],
            f"exposure.near_duplicate_groups(jaccard={X.NEAR_DUP_JACCARD})",
        ],
        "pool_items": len(pool),
        "groups": len(members),
        "excluded_groups": len(bad),
        "preassigned_groups": len(pre),
        "preassigned_exposed_groups": sum(1 for r in pre if set(members[r]) & exposed),
        "preassigned_items_missing_from_pool": sum(1 for h in (preassigned or {}) if h not in rep),
        "fresh_groups": len(ok),
        "reserve_groups": len(ok) - i,
        "consumers": out,
        "assignment": dict(sorted(assignment.items())),
    }
    rec["sha256"] = hashlib.sha256(_canon(rec).encode()).hexdigest()
    return rec


def shortfall(required: int, available: int) -> dict:
    n = min(required, available)
    return {
        "frozen_n": n,
        "shortfall": n < required,
        "restate_power": n < required,
    }


def verify_allocation(allocation: dict):
    rec = {k: v for k, v in allocation.items() if k != "sha256"}
    if hashlib.sha256(_canon(rec).encode()).hexdigest() != allocation["sha256"]:
        raise ValueError("allocation record does not match its sha256")


def restrict(items, allocation: dict, consumer: str):
    """The items of a pool assigned to ``consumer`` (by item hash)."""
    verify_allocation(allocation)
    a = allocation["assignment"]
    return [r for r in items if a.get(r["hash"]) == consumer]


# ------------------------------------------------------------------ builders that consume the allocation


def build_c1_panel_v2d(pools: dict, sizes: dict, out_dir, exposures: dict, allocation: dict, **kw):
    """``panel_c1.build_c1_panel_v1`` with every registered exposure required and the MMLU-Pro pool restricted to
    the joint allocation's ``wme-c1`` share."""
    require_exposures(exposures)
    pools = dict(pools)
    if "mmlupro_gold_lp" in pools:
        pools["mmlupro_gold_lp"] = restrict(pools["mmlupro_gold_lp"], allocation, CONSUMERS[0][0])
    man = PC.build_c1_panel_v1(pools, sizes, out_dir, exposures, **kw)
    man["mmlu_allocation_sha256"] = allocation["sha256"]
    (Path(out_dir) / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


C2GEN_FAMILIES = (("mmlupro_gen", "supergpqa_gen"), ("guard_harmless", "guard_harmful"))


def _rank2(salt: str, panel: str, group: str) -> str:
    return hashlib.sha256(f"{salt}:{panel}:{group}".encode()).hexdigest()


def build_c2gen_panel(
    pools: dict,
    out_dir,
    exposures: dict,
    allocation: dict,
    reserved: dict,
    sizes=C2GEN_SIZES,
    families=C2GEN_FAMILIES,
):
    """``wme-c2-gen``: ``pools`` {panel: [items with hash, gtext]}; ``reserved`` {tag: [texts]} (every written wme-c1
    item and every wme-c1 source pool except MMLU-Pro, which the allocation separates; always dropped).

    Per family (MMLU-Pro with SuperGPQA; the two generated guards) the groups are the union of both duplicate
    policies (:func:`joint_groups`) over the family's pools and the reserved texts; a group is dropped if it touches
    an exposed hash or a reserved text; a group is owned by the first panel of the family in which it occurs; one
    item per group (smallest hash); salted rank order sha256(salt:panel:group); splits in ``sizes`` order (base-only
    calibration first, then sealed_test). MMLU-Pro is restricted to the allocation's ``wme-c2-gen`` share. Shortfall:
    sealed_test N = min(required, available) with ``restate_power``; a calibration shortfall raises."""
    require_exposures(exposures)
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; panels are frozen once built")
    pools = dict(pools)
    pools["mmlupro_gen"] = restrict(pools["mmlupro_gen"], allocation, CONSUMERS[1][0])
    exposed = X.all_exposed(exposures)
    rtexts = {f"R\x00{tag}\x00{j:08d}": x for tag, xs in reserved.items() for j, x in enumerate(xs)}
    splits, stats, short = {}, {}, {}
    for fam in families:
        texts, where = dict(rtexts), {}
        for p in fam:
            for i, r in enumerate(pools[p]):
                k = f"P\x00{p}\x00{i:08d}"
                texts[k], where[k] = r["gtext"], (p, r)
        rep = joint_groups(texts)
        bad = {rep[k] for k in where if where[k][1]["hash"] in exposed}
        bad |= {rep[k] for k in rtexts}
        owner, best = {}, {p: {} for p in fam}
        for k in sorted(where, key=lambda k: (fam.index(where[k][0]), where[k][1]["hash"])):
            p, r = where[k]
            g = rep[k]
            if g in bad or owner.setdefault(g, p) != p or g in best[p]:
                continue
            best[p][g] = dict(r, group=PC.text_hash(texts[g]))
        for p in fam:
            rows = sorted(best[p].values(), key=lambda r: _rank2(C2GEN_SALT, p, r["group"]))
            need = sizes[p]
            fixed = {s: n for s, n in need.items() if s != "sealed_test"}
            if len(rows) < sum(fixed.values()):
                raise ValueError(f"{p}: {len(rows)} usable groups for the {fixed} base-only splits")
            n_test = min(need["sealed_test"], len(rows) - sum(fixed.values()))
            splits[p], i = {}, 0
            for s, n in list(fixed.items()) + [("sealed_test", n_test)]:
                splits[p][s] = rows[i : i + n]
                i += n
            stats[p] = {"pool": len(pools[p]), "usable_groups": len(rows)}
            short[p] = dict(shortfall(need["sealed_test"], n_test), required=need["sealed_test"])
    a = PP.audit(splits, exposed)
    if not a["clean"]:
        raise AssertionError("a wme-c2-gen split clashes or hits an exposure")
    (out / "open").mkdir(parents=True)
    (out / "sealed_test").mkdir()
    files = {}
    for s in sorted({s for ss in splits.values() for s in ss}):
        d = "sealed_test" if s == "sealed_test" else "open"
        lines = [
            json.dumps(
                dict({k: v for k, v in r.items() if k != "gtext"}, objective=p), sort_keys=True
            )
            for p, ss in splits.items()
            for r in ss.get(s, [])
        ]
        data = ("\n".join(lines) + "\n").encode()
        name = f"{d}/{s}.jsonl"
        (out / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    man = {
        "salt": C2GEN_SALT,
        "version": "wme-c2-gen-v2e",
        "policy": "joint_groups (wm.neardup v1 character policy + exposure word-MinHash policy) per family",
        "families": [list(f) for f in families],
        "sizes_required": sizes,
        "shortfall": short,
        "stats": stats,
        "exposures": sorted(exposures),
        "mmlu_allocation_sha256": allocation["sha256"],
        "audit": a,
        "split_hashes": files,
        "objectives": {
            p: {s: {"n": len(rs), "hashes": [r["hash"] for r in rs]} for s, rs in ss.items()}
            for p, ss in splits.items()
        },
        "sealed_hashes": sorted(
            r["hash"] for ss in splits.values() for r in ss.get("sealed_test", [])
        ),
    }
    man["content_hash"] = hashlib.sha256(_canon(files).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


# ------------------------------------------------------------------ hash-only audit across panels


def audit(panels: dict, texts: dict, exposures: dict) -> dict:
    """``panels`` {panel: {split: [item hashes]}}; ``texts`` {item hash: text} for every written item (held by the
    builder at build time only); returns counts, pairwise hash and joint-group intersections across every
    (panel, split) pair, and exposure intersections. Hash-only output."""
    units = {(p, s): set(hs) for p, ss in panels.items() for s, hs in ss.items()}
    rep = joint_groups({h: texts[h] for u in units.values() for h in u})
    grp = {u: {rep[h] for h in hs} for u, hs in units.items()}
    pairs = []
    for a, b in itertools.combinations(sorted(units), 2):
        pairs.append(
            {
                "a": "/".join(a),
                "b": "/".join(b),
                "hash_overlap": len(units[a] & units[b]),
                "group_overlap": len(grp[a] & grp[b]),
            }
        )
    exp = X.intersection_audit({"/".join(u): hs for u, hs in units.items()}, exposures)
    rec = {
        "counts": {"/".join(u): len(hs) for u, hs in sorted(units.items())},
        "pairs": pairs,
        "exposure": exp,
        "clean": all(p["hash_overlap"] == 0 and p["group_overlap"] == 0 for p in pairs)
        and X.audit_clean(exp),
    }
    rec["sha256"] = hashlib.sha256(_canon(rec).encode()).hexdigest()
    return rec


# ------------------------------------------------------------------ wme-c1 with the bank roots' construction shards


C1_ROOT_CONSUMER = "wme-c1:mmlupro_gold_lp:construction-roots"


def build_c1_panel_v2e(
    pools: dict,
    sizes: dict,
    out_dir,
    exposures: dict,
    allocation: dict,
    reserved_construction: set,
    salt: str = PC.SALT,
    construction_min: dict | None = None,
    dry_run: bool = False,
):
    """``wme-c1`` under the v1 policy (``panel_c1.build_c1_panel_v1``'s grouping, exclusion, ownership, salted order
    and split order) with two v2e rules (PREREG §II.23.3):

    * the MMLU-Pro pool is restricted to the joint allocation's c1 shares (fresh + root construction);
    * every group containing a hash in ``reserved_construction`` (the item groups the sampled bank roots already
      used as contrast or LoRA shards) goes to ``construction`` whatever its rank, and is never dealt to an
      evaluation split. Such a group is kept even if a later panel exposed it (counted, disclosed); evaluation splits
      must have zero exposure intersection.

    ``sizes[obj]`` gives fixed counts for sealed_gen, sealed_test, dev and bank; construction is ``"rest"`` (every
    remaining fresh group) plus the reserved groups; ``construction_min[obj]`` (field + e2 shards) is checked."""
    require_exposures(exposures)
    out = Path(out_dir)
    if out.exists() and not dry_run:
        raise FileExistsError(f"{out} exists; panels are frozen once built")
    pools = dict(pools)
    if "mmlupro_gold_lp" in pools:
        verify_allocation(allocation)
        a = allocation["assignment"]
        pools["mmlupro_gold_lp"] = [
            r
            for r in pools["mmlupro_gold_lp"]
            if a.get(r["hash"]) in (CONSUMERS[0][0], C1_ROOT_CONSUMER)
        ]
    order = [o for o in PC.P2.OBJECTIVES if o in pools] + sorted(
        o for o in pools if o not in PC.P2.OBJECTIVES
    )
    texts, where = {}, {}
    for obj in order:
        for i, r in enumerate(pools[obj]):
            k = f"{obj}\x00{i:07d}"
            texts[k] = PC._text_v1(obj, r, X.group_text)
            where[k] = (obj, r)
    rep = joint_groups(
        texts
    )  # the union of the v1 character policy and the word-MinHash policy (stricter than v1)
    exposed = X.all_exposed(exposures)
    bad = {rep[k] for k in texts if where[k][1]["hash"] in exposed}
    resv = {rep[k] for k in texts if where[k][1]["hash"] in reserved_construction}
    owner, fresh, reserved, exclusion = {}, {}, {}, {}
    for obj in order:
        ks = sorted(k for k in texts if where[k][0] == obj)
        by_f, by_r = {}, {}
        # a root-shard item is preferred as its group's representative, then the smallest hash
        for k in sorted(
            ks,
            key=lambda k: (where[k][1]["hash"] not in reserved_construction, where[k][1]["hash"]),
        ):
            g = rep[k]
            if owner.setdefault(g, obj) != obj:
                continue
            row = dict(where[k][1], group=PC.text_hash(texts[g]))
            if g in resv:
                by_r.setdefault(g, row)
            elif g not in bad:
                by_f.setdefault(g, row)
        exclusion[obj] = {
            "pool_items": len(ks),
            "excluded_groups": len({rep[k] for k in ks} & bad - resv),
            "available_fresh_groups": len(by_f),
            "reserved_construction_groups": len(by_r),
            "reserved_groups_exposed_later": len(set(by_r) & bad),
        }
        fresh[obj] = sorted(by_f.values(), key=lambda r: PC._rank(r["group"], salt))
        reserved[obj] = sorted(by_r.values(), key=lambda r: r["hash"])
    if dry_run:
        return {o: len(fresh[o]) for o in order}
    splits = {s: {} for s in PC.SPLITS_V1}
    for obj in order:
        rows = fresh[obj]
        fixed = sum(int(v) for s, v in sizes[obj].items() if v != "rest") + (
            construction_min or {}
        ).get(obj, 0)
        if len(rows) < fixed:
            raise ValueError(
                f"{obj}: {len(rows)} fresh groups for {fixed} requested (never relaxed)"
            )
        i = 0
        for s in PC.SPLITS_V1:
            n = len(rows) - i if sizes[obj][s] == "rest" else int(sizes[obj][s])
            splits[s][obj] = rows[i : i + n]
            i += n
        splits["construction"][obj] = sorted(
            splits["construction"][obj] + reserved[obj], key=lambda r: r["hash"]
        )
        for s in PC.SPLITS_V1:
            splits[s][obj] = sorted(splits[s][obj], key=lambda r: r["hash"])
        n_fresh_constr = len(splits["construction"][obj]) - len(reserved[obj])
        need = (construction_min or {}).get(obj, 0)
        if (
            n_fresh_constr < need
        ):  # the field and e2 shards must be fresh, disjoint from root shards
            raise ValueError(f"{obj}: fresh construction {n_fresh_constr} < {need} required")
        exclusion[obj]["fresh_construction_groups"] = n_fresh_constr
    hashes = {s: {r["hash"] for o in order for r in splits[s][o]} for s in PC.SPLITS_V1}
    groups = {s: {r["group"] for o in order for r in splits[s][o]} for s in PC.SPLITS_V1}
    for a_, b_ in itertools.combinations(PC.SPLITS_V1, 2):
        if hashes[a_] & hashes[b_] or groups[a_] & groups[b_]:
            raise AssertionError(f"splits {a_} and {b_} intersect")
    audit = X.intersection_audit({s: hashes[s] for s in PC.SPLITS_V1}, exposures)
    resv_hashes = {r["hash"] for o in order for r in reserved[o]} & set(reserved_construction)
    eval_clean = all(
        v == 0
        for e in audit.values()
        for es in e.values()
        for s, v in es.items()
        if s != "construction"
    )
    constr_hits = {h for e in exposures.values() for hs in e.values() for h in hs} & hashes[
        "construction"
    ]
    if not eval_clean or not constr_hits <= resv_hashes:
        raise AssertionError(
            "an evaluation split, or an unreserved construction item, hits an exposure"
        )
    for d in ("open", "sealed_test", "sealed_gen"):
        (out / d).mkdir(parents=True, exist_ok=False) if d != "open" else (out / d).mkdir(
            parents=True
        )
    files = {}
    for s in PC.SPLITS_V1:
        lines = [
            json.dumps(dict(r, objective=o), sort_keys=True) for o in order for r in splits[s][o]
        ]
        data = ("\n".join(lines) + ("\n" if lines else "")).encode()
        (out / PC._FILES_V1[s]).write_bytes(data)
        files[PC._FILES_V1[s]] = hashlib.sha256(data).hexdigest()
    manifest = {
        "salt": salt,
        "version": "wme-c1-v2e",
        "sizes": sizes,
        "policy": {
            "v1": ND.POLICY,
            "joined_with": f"exposure.near_duplicate_groups(jaccard={X.NEAR_DUP_JACCARD})",
        },
        "exclusion": exclusion,
        "exposures": sorted(exposures),
        "audit": audit,
        "audit_clean_evaluation_splits": True,
        "construction_exposure_hits_reserved_roots": len(constr_hits),
        "mmlu_allocation_sha256": allocation["sha256"],
        "reserved_construction_items": len(resv_hashes),
        "split_hashes": files,
        "objectives": {
            o: {
                s: {
                    "n": len(splits[s][o]),
                    "hashes": [r["hash"] for r in splits[s][o]],
                    "groups": [r["group"] for r in splits[s][o]],
                }
                for s in PC.SPLITS_V1
            }
            for o in order
        },
        "sealed_hashes": sorted(hashes["sealed_test"] | hashes["sealed_gen"]),
    }
    manifest["content_hash"] = hashlib.sha256(_canon(files).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest, {o: {s: splits[s][o] for s in PC.SPLITS_V1} for o in order}
