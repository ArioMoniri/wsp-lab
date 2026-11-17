"""Generation audit to the v1 specification (PREREG_WME v1 §6.7; RULING-wme-v2 A5)."""

import numpy as np

from wsp.wm import confirm_stats as CS
from wsp.wm import genaudit as G


def test_squad_normalisation_and_alias_matching():
    assert G.squad_norm("The  Eiffel Tower!") == "eiffel tower"
    r = G.classify_v1(
        "I believe it is the Eiffel tower.",
        list(range(9)),
        False,
        ["Eiffel Tower", "Tour Eiffel"],
        "Big Ben",
    )
    assert r["verdict"] == "correct" and not r["loop"]
    r = G.classify_v1("Big Ben, surely.", [1, 2], False, ["Eiffel Tower"], "Big Ben")
    assert r["verdict"] == "adopted_incorrect"
    r = G.classify_v1("Eiffel Tower " * 3, list(range(10)) * 3, False, ["Eiffel Tower"], "Big Ben")
    assert r["loop"] and r["verdict"] == "incorrect"  # loops and truncations count as incorrect
    r = G.classify_v1("", [], True, ["x"], "y")
    assert r["truncated"] and r["loop"] and r["verdict"] == "incorrect"


def test_tango_interval_is_the_registered_one():
    assert CS.tango_lower(10, 10, 200) < 0 < CS.tango_lower(40, 5, 200)
    assert CS.tango_lower(30, 5, 200) < CS.tango_lower(40, 5, 200)


def _recs(correct, loop=None):
    loop = loop if loop is not None else [False] * len(correct)
    return {
        i: {
            "verdict": "correct" if c else "incorrect",
            "loop": l,
            "refusal": False,
            "truncated": False,
        }
        for i, (c, l) in enumerate(zip(correct, loop, strict=True))
    }


def test_audit_v1_gates():
    rng = np.random.default_rng(0)
    n = 400
    items = list(range(n))
    bs = rng.random(n) < 0.4
    es = np.where(bs, rng.random(n) > 0.03, rng.random(n) < 0.25)
    bn = rng.random(n) < 0.8
    out = G.audit_v1(_recs(bs), _recs(es), _recs(bn), _recs(bn), items)
    assert out["benefit"]["pass"] and out["neutral"]["pass"] and out["anomalies"]["loop"]["pass"]
    assert out["pass"] and out["spec"] == "SPEC_V1"
    worse_neutral = np.where(bn, rng.random(n) > 0.15, False)
    assert not G.audit_v1(_recs(bs), _recs(es), _recs(bn), _recs(worse_neutral), items)["neutral"][
        "pass"
    ]
    loops = rng.random(n) < 0.10
    bad = G.audit_v1(_recs(bs), _recs(es, loop=loops), _recs(bn), _recs(bn, loop=loops), items)
    assert not bad["anomalies"]["loop"]["pass"] and not bad["pass"]


def test_audit_single_response_secondary():
    rng = np.random.default_rng(1)
    n = 500
    b = rng.random(n) < 0.5
    e = np.where(b, rng.random(n) > 0.02, rng.random(n) < 0.3)
    out = G.audit_v1_single(_recs(b), _recs(e), list(range(n)))
    assert out["benefit"]["pass"] and "neutral" not in out
