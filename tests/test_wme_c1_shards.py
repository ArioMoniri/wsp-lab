"""wme-c1 construction-shard designation (amendment G10): deterministic, fresh-only, disjoint, token-capped."""

import pytest

from wsp.wm.c1 import shards as S


def rows(obj, n, prefix):
    return [
        {
            "objective": obj,
            "hash": f"{prefix}{i:04d}",
            "group": f"g{prefix}{i:04d}",
            "id": f"{prefix}{i}",
        }
        for i in range(n)
    ]


def pool():
    out = []
    pre = {
        "wikitext_nll": "wik",
        "gsm8k_gold_lp": "gsm",
        "mmlupro_gold_lp": "mml",
        "refusal_margin_harmful": "hfl",
        "refusal_margin_harmless": "hls",
        "sycophancy_real": "syc",
    }
    for o in S.FIELD_OBJECTIVES:
        out += rows(o, 120, pre[o])
    out += rows("mathcomp_gold_lp", 100, "mat")
    return out


def test_designation_is_deterministic_fresh_and_disjoint():
    p = pool()
    reps = {"wik0000", "gsm0001"}
    root_groups = {"gsyc0002"}
    a = S.designate(p, reps, root_groups, lambda o, r: 10)
    b = S.designate(list(reversed(p)), reps, root_groups, lambda o, r: 10)
    assert a["digest"] == b["digest"]
    assert S.check_disjoint(a)
    every = {r["hash"] for o in a["field"].values() for r in o} | {r["hash"] for r in a["e2"]}
    assert not every & reps and "syc0002" not in every
    assert all(len(v) == 24 for v in a["field"].values()) and len(a["e2"]) == 64
    assert all(len(v) == 64 for v in a["e2_sources"].values())
    # rank order, not file order
    ranks = [S.rank(r["group"]) for r in a["e2"]]
    assert ranks == sorted(ranks)


def test_field_token_cap_and_shortfall_raise():
    p = pool()
    long = lambda o, r: 2000 if r["hash"].endswith(("1", "3", "5", "7", "9")) else 10
    a = S.designate(p, set(), set(), long)
    assert all(int(r["hash"][-1]) % 2 == 0 for v in a["field"].values() for r in v)
    with pytest.raises(ValueError):
        S.designate(p, set(), set(), lambda o, r: 5000)
    small = [r for r in p if r["objective"] != "sycophancy_real"] + rows(
        "sycophancy_real", 80, "syc"
    )
    with pytest.raises(ValueError, match="construction/e2"):
        S.designate(small, set(), set(), lambda o, r: 10)
