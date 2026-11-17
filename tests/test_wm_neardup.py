"""Frozen v1 near-duplicate policy (PREREG_WME v1 §3.2): exact character-5-gram Jaccard join (RULING-wme-v2 A5)."""

import itertools
import random

from wsp.wm import neardup as ND


def _texts(seed, n=300):
    rng = random.Random(seed)
    words = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "eps",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
    ]
    base = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 12))) for _ in range(n // 2)]
    out = {}
    for i, b in enumerate(base):
        out[f"a{i}"] = b
        w = b.split()
        if w and rng.random() < 0.7:
            w[rng.randrange(len(w))] = rng.choice(words)  # near variants
        out[f"b{i}"] = " ".join(w) + rng.choice(["", "!", ".", " the"])
    return out


def test_exact_join_equals_brute_force():
    for seed in range(3):
        t = _texts(seed)
        got = ND.pairs(t, 0.8)
        sh = {k: ND.char_shingles(v) for k, v in t.items()}
        ref = {
            tuple(sorted((a, b)))
            for a, b in itertools.combinations(sorted(t), 2)
            if ND.jaccard(sh[a], sh[b]) >= 0.8
        }
        assert got == ref


def test_groups_join_exact_hash_and_near_duplicates_transitively():
    t = {
        "x": "What is the capital of France?",
        "y": "what is the capital of france",
        "z": "Name the largest ocean",
    }
    g = ND.groups(t)
    assert g["x"] == g["y"] != g["z"]
    assert (
        ND.POLICY["k"] == 5
        and ND.POLICY["jaccard"] == 0.8
        and ND.POLICY["method"].startswith("exact")
    )
