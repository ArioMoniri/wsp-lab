"""Fresh pilot-3 panels ``wme-p3s`` and ``wme-p3h`` (RULING-wme-pilot3 §1). EXPLORATORY; CPU only.

* **Families.** Sources that can share questions are grouped jointly (``wm.exposure.near_duplicate_groups``: word
  5-gram MinHash, Jaccard >= 0.7, plus exact normalised duplicates). A family holds pool items (per panel) and
  *reserved* texts (pools kept for the confirmatory wme-c1 panel, or other sources to avoid).
* **Exclusion** (:func:`select`). A group is unusable if it contains any exposed hash (manifest hash lists of every
  earlier panel, sealed lists included; sealed files are never opened) or any reserved text. A panel may also
  ``avoid`` a source tag present in its family (e.g. IFEval-type prompts avoid harmless_alpaca near-duplicates).
* **Assignment.** One item per group (smallest hash); panels are filled in the given order and a group used by an
  earlier panel is never reused; within a panel groups are ranked by sha256(salt:panel:group) and assigned to the
  splits in the given order. A shortfall raises (sizes are never relaxed).
* **Audit** (:func:`audit`): pairwise disjointness of every (panel, split) and an empty intersection with every
  exposure, asserted before anything is written. No sealed split is written (pilot 3 has no confirmatory read).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wsp.wm import exposure as X


def _rank(salt: str, panel: str, group: str) -> str:
    return hashlib.sha256(f"{salt}:{panel}:{group}".encode()).hexdigest()


def family_groups(pools: dict, reserved: dict) -> tuple[dict, dict]:
    """pools {panel: [item with 'gtext']}, reserved {tag: [text]} -> ({(panel, i): group}, {group: set(tags)})."""
    texts = {}
    for p, rows in pools.items():
        for i, r in enumerate(rows):
            texts[f"P\x00{p}\x00{i:08d}"] = r["gtext"]
    for tag, rows in reserved.items():
        for j, t in enumerate(rows):
            texts[f"R\x00{tag}\x00{j:08d}"] = t
    rep = X.near_duplicate_groups(texts)
    gid = {k: X.text_hash(texts[rep[k]]) for k in texts}
    tags: dict = {}
    out = {}
    for k, g in gid.items():
        kind, name, i = k.split("\x00")
        if kind == "R":
            tags.setdefault(g, set()).add(name)
        else:
            out[(name, int(i))] = g
    return out, tags


def group_all(families: dict) -> dict:
    return {
        f: family_groups(spec["pools"], {**spec.get("reserved", {}), **spec.get("avoid_texts", {})})
        for f, spec in families.items()
    }


def select(
    families: dict,
    order,
    sizes: dict,
    salt: str,
    exposed: set,
    split_order: dict,
    used_groups=(),
    grouped=None,
) -> dict:
    """families {fam: {'pools': {panel: [items]}, 'reserved': {tag: [texts]} (always dropped),
    'avoid_texts': {tag: [texts]}, 'avoid': {panel: [tags]} (dropped only for the listed panels)}};
    order: panel fill order; sizes {panel: {split: n}}; split_order {panel: (split, ...)}; ``used_groups``: groups
    taken by an earlier panel set (never reused); ``grouped``: a cached :func:`group_all` result.
    Returns {'splits': {panel: {split: [items]}}, 'stats': {panel: {...}}}."""
    fam_of = {p: f for f, spec in families.items() for p in spec["pools"]}
    grouped = grouped or group_all(families)
    used: dict = {f: set(used_groups) for f in families}
    out, stats = {}, {}
    for panel in order:
        f = fam_of[panel]
        spec = families[f]
        gmap, gtags = grouped[f]
        rows = spec["pools"][panel]
        bad_exposed = {g for (p, i), g in gmap.items() if spec["pools"][p][i]["hash"] in exposed}
        avoid = set(spec.get("avoid", {}).get(panel, ()))
        best: dict = {}
        drop = {"exposed": 0, "reserved": 0, "avoided": 0, "used": 0, "duplicate": 0}
        for i, r in enumerate(rows):
            g = gmap[(panel, i)]
            t = gtags.get(g, set())
            if g in bad_exposed:
                drop["exposed"] += 1
            elif t & set(spec.get("reserved", {})):
                drop["reserved"] += 1
            elif t & avoid:
                drop["avoided"] += 1
            elif g in used[f]:
                drop["used"] += 1
            elif g in best and best[g]["hash"] <= r["hash"]:
                drop["duplicate"] += 1
            else:
                if g in best:
                    drop["duplicate"] += 1
                best[g] = dict(r, group=g)
        ranked = sorted(best.values(), key=lambda r: _rank(salt, panel, r["group"]))
        need = sum(sizes[panel].values())
        if len(ranked) < need:
            raise ValueError(f"{panel}: {len(ranked)} usable groups for {need} requested")
        out[panel] = {}
        k = 0
        for s in split_order[panel]:
            take = ranked[k : k + sizes[panel][s]]
            k += sizes[panel][s]
            out[panel][s] = take  # salted rank order (dev uses its first N)
            used[f] |= {r["group"] for r in take}
        stats[panel] = {"pool": len(rows), "usable_groups": len(ranked), "dropped": drop}
    return {"splits": out, "stats": stats}


def audit(splits: dict, exposed: set) -> dict:
    seen: dict = {}
    clash = []
    for panel, ss in splits.items():
        for s, rows in ss.items():
            for r in rows:
                for key in (("h", r["hash"]), ("g", r["group"])):
                    if key in seen and seen[key] != (panel, s):
                        clash.append((key, seen[key], (panel, s)))
                    seen.setdefault(key, (panel, s))
    hit = sorted(
        {r["hash"] for ss in splits.values() for rows in ss.values() for r in rows} & exposed
    )
    return {"clashes": len(clash), "exposed_hits": len(hit), "clean": not clash and not hit}


def write_panel(out_dir, splits: dict, salt: str, extra: dict) -> dict:
    """Write ``open/<split>.jsonl`` (rows carry ``objective`` = panel) and ``manifest.json`` (hash lists)."""
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; panels are frozen once built")
    (out / "open").mkdir(parents=True)
    split_names = sorted({s for ss in splits.values() for s in ss})
    files, objectives = {}, {}
    for s in split_names:
        lines = []
        for panel, ss in splits.items():
            for r in ss.get(s, []):
                row = {k: v for k, v in r.items() if k != "gtext"}
                lines.append(json.dumps(dict(row, objective=panel), sort_keys=True))
        data = ("\n".join(lines) + "\n").encode()
        (out / f"open/{s}.jsonl").write_bytes(data)
        files[f"open/{s}.jsonl"] = hashlib.sha256(data).hexdigest()
    for panel, ss in splits.items():
        objectives[panel] = {
            s: {"n": len(rows), "hashes": [r["hash"] for r in rows]} for s, rows in ss.items()
        }
    man = {
        "salt": salt,
        "objectives": objectives,
        "files": files,
        "content_hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "sealed_hashes": [],
        **extra,
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def read_split(root, split: str) -> dict:
    """{panel: [items]} from ``open/<split>.jsonl`` (rank order kept); nothing outside ``open/`` is read."""
    if "/" in split or split.startswith("sealed"):
        raise PermissionError(split)
    out: dict = {}
    for line in (Path(root) / "open" / f"{split}.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["objective"], []).append(r)
    return out
