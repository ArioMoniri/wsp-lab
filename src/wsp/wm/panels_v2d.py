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


def allocate(pool: dict, exposed: set, consumers=CONSUMERS, salt: str = ALLOC_SALT) -> dict:
    """``pool`` {item hash: text}. Returns the allocation record (assignment {hash: consumer | "reserve"})."""
    rep = joint_groups(pool)
    members: dict = {}
    for h, r in rep.items():
        members.setdefault(r, []).append(h)
    bad = {r for r, hs in members.items() if set(hs) & exposed}
    ok = sorted(
        (r for r in members if r not in bad),
        key=lambda r: hashlib.sha256(f"{salt}:{min(members[r])}".encode()).hexdigest(),
    )
    assignment, out, i = {}, [], 0
    for name, need, rule in consumers:
        n = min(need, len(ok) - i)
        if n < need and rule == "raise":
            raise ValueError(f"{name}: {len(ok) - i} fresh groups for {need} (never relaxed)")
        for r in ok[i : i + n]:
            for h in members[r]:
                assignment[h] = name
        out.append({"consumer": name, "required": need, "assigned_groups": n, **shortfall(need, n)})
        i += n
    for r in ok[i:]:
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


def build_c2gen_panel(
    pools: dict, out_dir, exposures: dict, allocation: dict, reserved: dict, sizes=C2GEN_SIZES
):
    """``wme-c2-gen``: ``pools`` {panel: [items with hash, gtext]}; ``reserved`` {tag: [texts]} (every wme-c1 pool,
    always dropped). One item per word-MinHash group (``p3panel``), group-level exposure exclusion, salted rank
    order, splits in ``sizes`` order; MMLU-Pro restricted to the allocation's ``wme-c2-gen`` share; shortfall ->
    ``min(required, available)`` with ``restate_power``. Sealed splits go under ``sealed_test/``; the base-only
    calibration splits under ``open/``."""
    require_exposures(exposures)
    pools = dict(pools)
    pools["mmlupro_gen"] = restrict(pools["mmlupro_gen"], allocation, CONSUMERS[1][0])
    exposed = X.all_exposed(exposures)
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; panels are frozen once built")
    splits, stats, short = {}, {}, {}
    for panel, rows in pools.items():
        fam = {"pools": {panel: rows}, "reserved": reserved}
        grouped = PP.group_all({panel: fam})
        need = sizes[panel]
        avail = _usable(fam, panel, exposed, grouped[panel])
        fixed = {s: n for s, n in need.items() if s != "sealed_test"}
        n_test = min(need["sealed_test"], max(0, avail - sum(fixed.values())))
        if avail < sum(fixed.values()):
            raise ValueError(f"{panel}: {avail} usable groups for the {fixed} base-only splits")
        got = dict(fixed, sealed_test=n_test)
        sel = PP.select(
            {panel: fam},
            [panel],
            {panel: got},
            C2GEN_SALT,
            exposed,
            {panel: tuple(need)},
            grouped=grouped,
        )
        splits[panel] = sel["splits"][panel]
        stats[panel] = sel["stats"][panel]
        short[panel] = dict(shortfall(need["sealed_test"], n_test), required=need["sealed_test"])
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
        "version": "wme-c2-gen-v2d",
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


def _usable(fam, panel, exposed, grouped) -> int:
    gmap, gtags = grouped
    rows = fam["pools"][panel]
    bad = {g for (p, i), g in gmap.items() if rows[i]["hash"] in exposed}
    res = set(fam.get("reserved", {}))
    return len({g for g in gmap.values() if g not in bad and not (gtags.get(g, set()) & res)})


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
