"""Component-closure audit of the development rows (SYNTHESIS-r7 Decision 1, "Exclusions").

Union-find over every zoo entry with four joins: ancestry (``parent_id``), shared source (``generator_edit_id``, and
the generator entry itself when it is a zoo entry), shared direction (a random-direction generator seed; an
abliteration refusal fingerprint with the same direction index; see :func:`direction_keys`), and duplicate delta
bytes (equal resolved ``entry_delta_hash``). ``provenance.reuse_key`` is not a join: it hashes the generation
environment (base, dtype, family, kernel, fingerprint) without the sampled parameters or seed, so it is shared by
unrelated edits. ``edit.json`` is consulted only for entries outside Part A components: Part A entries enter
through registered metadata (component, parent, delta hash) and are never opened.

Outcome for each population row: *excluded* if its closure class reaches a Part A component, if its edit record or
provenance is missing, or if ``closure_certified`` is false; otherwise its registered component is kept, and classes
that join several population components are merged (representative = the smallest component id).
"""

from __future__ import annotations

from collections import defaultdict


class EditView:
    """Read-only mapping of edit records that refuses Part A ids (PermissionError) instead of returning them."""

    def __init__(self, edits: dict, forbidden=()):
        self._edits, self._forbidden = edits, set(forbidden)

    def _check(self, k):
        if k in self._forbidden:
            raise PermissionError(f"{k}: Part A entry; its edit record is never opened")

    def __getitem__(self, k):
        self._check(k)
        return self._edits[k]

    def get(self, k, default=None):
        self._check(k)
        return self._edits.get(k, default)

    def __contains__(self, k):
        self._check(k)
        return k in self._edits


class _UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def direction_keys(e: dict) -> list:
    """Shared-direction keys of one edit record.

    * random directions: ``wsp.zoo.build.build_random_dir`` draws direction ``di`` of layer ``l`` from
      ``random_unit_direction(d_model, seed + 1000 l + 100 di)``; each such generator seed is a key (a chain step
      with inner family random_dir uses its ``inner_seed``);
    * abliteration: the refusal-direction set (``refusal_fingerprint``) with the exact ``direction_index``
      (a chain step with inner family abliteration uses its ``inner_params``).
    """
    fam = e.get("family")
    p = e.get("params") or {}
    prov = e.get("provenance") or {}
    keys = []
    if fam == "random_dir":
        seed, rp = e.get("seed"), p
    elif fam == "chain" and p.get("inner_family") == "random_dir":
        seed, rp = p.get("inner_seed"), p.get("inner_params") or {}
    else:
        seed, rp = None, {}
    if seed is not None:
        for layer in rp.get("layers") or []:
            for di in range(int(rp.get("n_directions") or 0)):
                keys.append(("random_direction", int(seed) + int(layer) * 1000 + di * 100))
    ap = p if fam == "abliteration" else (p.get("inner_params") or {})
    is_abl = fam == "abliteration" or (fam == "chain" and p.get("inner_family") == "abliteration")
    fp = prov.get("refusal_fingerprint")
    if is_abl and fp and ap.get("direction_index") is not None:
        keys.append(("abliteration_direction", fp, float(ap["direction_index"])))
    return keys


def closure_audit(population, zoo: dict, edits, delta_hash: dict, part_a_components) -> dict:
    """Audit record: ``merge`` {component: representative}, ``excluded`` {row id: reason}, ``cross_component_joins``,
    edge counts and the classes of the population components."""
    part_a = set(part_a_components)
    ids = sorted(zoo)
    uf = _UF(ids)
    edges = []  # (kind, a, b)
    by_key = defaultdict(list)
    for i in ids:
        par = zoo[i].get("parent_id")
        if par in zoo:
            edges.append(("ancestry", i, par))
        if zoo[i]["component_id"] in part_a:
            continue  # never open a Part A edit record
        e = edits.get(i) if i in edits else None
        if not e:
            continue
        prov = e.get("provenance") or {}
        gen = prov.get("generator_edit_id")
        if gen:
            by_key[("generator", gen)].append(i)
            if gen in zoo:
                edges.append(("generator", i, gen))
        for k in direction_keys(e):
            by_key[k].append(i)
    for i in ids:
        h = delta_hash.get(i)
        if h:
            by_key[("delta_bytes", h)].append(i)
    for key, members in sorted(by_key.items(), key=lambda kv: repr(kv[0])):
        kind = {"random_direction": "direction_seed", "abliteration_direction": "direction"}.get(
            key[0], key[0]
        )
        for other in members[1:]:
            edges.append((kind, members[0], other))
    for _, a, b in edges:
        uf.union(a, b)
    comp = {i: zoo[i]["component_id"] for i in ids}
    cross = sorted(
        {(k, a, b, comp[a], comp[b]) for k, a, b in edges if comp[a] != comp[b]},
        key=lambda t: (t[0], t[3], t[4], t[1], t[2]),
    )
    classes = defaultdict(set)
    for i in ids:
        classes[uf.find(i)].add(i)
    reach_a = {}
    for k, a, b, ca, cb in cross:
        if (ca in part_a) != (cb in part_a):
            reach_a.setdefault(uf.find(a), k)
    excluded, merge = {}, {}
    pop_comps = defaultdict(set)
    for i in population:
        cls = classes[uf.find(i)]
        comps = {comp[m] for m in cls}
        if comps & part_a:
            excluded[i] = f"joins_part_a:{reach_a.get(uf.find(i), 'registered')}"
            continue
        if zoo[i].get("closure_certified") is False:
            excluded[i] = "closure_not_certified"
            continue
        e = edits.get(i) if i in edits else None
        if not e or not isinstance(e.get("provenance"), dict):
            excluded[i] = "missing_provenance"
            continue
        pop_comps[uf.find(i)].add(comp[i])
    for comps in pop_comps.values():
        rep = min(comps)
        for c in sorted(comps):
            if c != rep:
                merge[c] = rep
    counts = defaultdict(int)
    for k, *_ in edges:
        counts[k] += 1
    return {
        "joins": [
            "ancestry",
            "generator",
            "direction_seed",
            "direction",
            "delta_bytes",
        ],
        "edge_counts": dict(sorted(counts.items())),
        "cross_component_joins": [
            {"kind": k, "a": a, "b": b, "component_a": ca, "component_b": cb}
            for k, a, b, ca, cb in cross
        ],
        "merge": merge,
        "excluded": excluded,
        "n_population": len(list(population)),
    }
