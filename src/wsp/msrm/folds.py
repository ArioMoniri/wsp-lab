"""Outer folds, closure-merged components and leave-one-family-out splits (PREREG_H1_TF F12.2 rule, F12a.1(i)).

Outer folds: a component's stratum is its family (``mixed`` if it holds two); within each stratum components are
ordered by sha256(UTF-8 ``20261111:f12:`` + component id); strata are concatenated in the order abliteration, chain,
lora_sft, mixed, random_dir and components are assigned round-robin to folds 0–4 by global position.
"""

from __future__ import annotations

import hashlib
import json

FOLD_DOMAIN = "20261111:f12:"
INNER_DOMAIN = "20261110:lol-inner:"
STRATA = ("abliteration", "chain", "lora_sft", "mixed", "random_dir")
N_OUTER = 5


def sha_key(domain: str, key: str) -> str:
    return hashlib.sha256(f"{domain}{key}".encode()).hexdigest()


def ids_sha256(ids) -> str:
    """F12.2: sha256 of the UTF-8 ``json.dumps`` (Python defaults) of the sorted id list."""
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def component_strata(components, families) -> dict:
    fams = {}
    for c, f in zip(components, families):
        fams.setdefault(c, set()).add(f)
    return {c: (next(iter(f)) if len(f) == 1 else "mixed") for c, f in fams.items()}


def outer_folds(components, families) -> dict:
    strata = component_strata(components, families)
    order = []
    for s in STRATA:
        order += sorted(
            (c for c, t in strata.items() if t == s), key=lambda c: sha_key(FOLD_DOMAIN, c)
        )
    if len(order) != len(strata):
        raise ValueError("a component has no registered stratum")
    return {c: i % N_OUTER for i, c in enumerate(order)}


def merge_components(components, merge: dict) -> list:
    """Components after the closure audit's merges (``merge[c]`` = the representative component of c)."""
    return [merge.get(c, c) for c in components]


def assert_component_disjoint(components, folds):
    seen = {}
    for c, f in zip(components, folds):
        if seen.setdefault(c, f) != f:
            raise ValueError(f"component {c} crosses folds {seen[c]} and {f}")


def lofo_split(components, families, held: str):
    """(train row indices, test row indices) holding out family ``held``: the mixed component always trains and its
    rows are never evaluated (F12a.1(i))."""
    strata = component_strata(components, families)
    train, test = [], []
    for i, (c, f) in enumerate(zip(components, families)):
        if strata[c] == "mixed":
            train.append(i)
        elif f == held:
            test.append(i)
        else:
            train.append(i)
    return train, test
