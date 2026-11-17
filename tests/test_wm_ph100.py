"""P-H100 timing and noise pilot (RULING-wme-resources §2; PREREG_WME v1 §6.3.3, §8.6). CPU tests only.

The pilot uses only no-op, inert and fixed timing edits on construction/validation items of the exposed wme-p2
panel. Its summary is hash-only for values: it reports paired SEs and seconds, never a mean change.
"""

import json

import numpy as np
import pytest
import torch

from wsp.wm import actions as A
from wsp.wm import ph100 as H


def _rm(n=4, tied=False):
    return A.RoleMap(n, 8, 16, 32, ("linear_attention", "full_attention") * (n // 2), tied=tied)


def _params(rm, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = {}
    for L in range(rm.n_layers):
        for r in ("o", "down", "up"):
            n = rm.name(L, r)
            shape = (
                (rm.hidden, rm.ffn)
                if r == "down"
                else (rm.ffn, rm.hidden)
                if r == "up"
                else (
                    rm.hidden,
                    rm.hidden,
                )
            )
            p[n] = torch.randn(*shape, generator=g).to(torch.bfloat16)
    p["model.embed_tokens.weight"] = torch.randn(rm.vocab, rm.hidden, generator=g).to(
        torch.bfloat16
    )
    return p


def test_round_trip_two_and_four_step_return_base_bytes():
    rm = _rm()
    params = _params(rm)
    base = A.state_hash(params)
    for shape in ("2step", "4step"):
        steps = H.round_trip(rm, shape, tensors=[(1, "down"), (2, "o")])
        assert len(steps) == (2 if shape == "2step" else 4)
        sess = A.EditSession(params, base_hash=base, rolemap=rm)
        hashes = []
        for s in steps:
            sess.apply_bundle(s)
            hashes.append(sess.theta_hash())
        assert hashes[-1] == base  # inert endpoint: theta-hash equals the base
        assert all(h != base for h in hashes[:-1])  # every intermediate is a real edited state
        sess.restore()


def test_round_trip_refuses_overflow_or_subnormal():
    w = torch.tensor([1.0, 2.0, -3.0], dtype=torch.bfloat16)
    assert H.power_of_two_safe(w, 2.0) and H.power_of_two_safe(w, 0.5)
    big = torch.tensor([torch.finfo(torch.bfloat16).max], dtype=torch.bfloat16)
    assert not H.power_of_two_safe(big, 2.0)
    tiny = torch.tensor([torch.finfo(torch.bfloat16).tiny], dtype=torch.bfloat16)
    assert not H.power_of_two_safe(tiny, 0.5)
    assert H.power_of_two_safe(torch.zeros(3, dtype=torch.bfloat16), 0.5)  # exact zeros stay exact


def test_never_emitted_rows_checked_against_tokenizer_and_template():
    rows = H.never_emitted_rows(tokenizer_len=28, vocab=32, template_ids=[1, 2, 27])
    assert rows == [28, 29, 30, 31]
    with pytest.raises(ValueError):
        H.never_emitted_rows(tokenizer_len=28, vocab=32, template_ids=[30])
    with pytest.raises(ValueError):
        H.i2_actions(_rm(tied=True), "model.embed_tokens.weight", [28, 29])
    acts = H.i2_actions(_rm(), "model.embed_tokens.weight", [28, 29, 30])
    assert [a.param("factor") for a in acts] == list(H.I2_FACTORS)


def test_dead_row_edit_changes_only_unemitted_rows():
    rm = _rm()
    params = _params(rm)
    base = A.state_hash(params)
    rows = H.never_emitted_rows(tokenizer_len=28, vocab=32, template_ids=[])
    sess = A.EditSession(params, base_hash=base, rolemap=rm)
    w0 = params["model.embed_tokens.weight"].clone()
    sess.apply(H.i2_actions(rm, "model.embed_tokens.weight", rows)[0])
    w1 = params["model.embed_tokens.weight"]
    assert torch.equal(w1[:28], w0[:28]) and not torch.equal(w1[28:], w0[28:])
    sess.restore()


def test_timing_edits_fixed_and_disjoint_from_pilot_tensors():
    spec = H.TIMING_EDITS
    assert len(spec) >= 5 and len({json.dumps(s, sort_keys=True) for s in spec}) == len(spec)
    fams = {s["family"] for s in spec}
    assert {"scale", "noise", "row_scale", "proj_remove"} <= fams
    pilot = {(13, "down"), (14, "o"), (15, "down"), (13, "o")}
    for s in spec:
        for t in s.get("tensors", []):
            assert tuple(t) not in pilot
    assert H.timing_edit_digest() == H.timing_edit_digest()


def test_timing_actions_build_on_toy_rolemap():
    rm = A.RoleMap(32, 8, 16, 256, ("linear_attention", "full_attention") * 16, tied=False)
    for s in H.TIMING_EDITS:
        acts = H.timing_actions(rm, s, embed="model.embed_tokens.weight", tokenizer_len=200)
        assert acts and all(a.dose == 1.0 for a in acts)


def test_summary_is_hash_only_and_scales_se():
    rng = np.random.default_rng(0)
    stats = {
        f"t{i}": {
            o: {"n": 64, "mean": float(rng.normal()), "se": 0.1 * (i + 1)} for o in H.ENDPOINTS
        }
        for i in range(3)
    }
    s = H.se_summary(stats)
    for o in H.ENDPOINTS:
        assert s[o]["median_se_n64"] == pytest.approx(0.2)
        assert s[o]["se_at"]["128"] == pytest.approx(0.2 * np.sqrt(64 / 128))
    blob = json.dumps(s)
    for st in stats.values():
        for o in H.ENDPOINTS:
            assert repr(st[o]["mean"]) not in blob  # no mean change is reported
    h = H.hash_only(stats)
    assert set(h) == set(stats) and all(len(v) == 64 for v in h.values())


def test_unit_seconds_aggregate():
    units = [
        {
            "state_id": "b",
            "objective": "x",
            "n_items": 64,
            "seconds": {"apply": 0.1, "eval": 2.0, "restore": 0.05},
        },
        {
            "state_id": "b",
            "objective": "y",
            "n_items": 64,
            "seconds": {"apply": 0.1, "eval": 4.0, "restore": 0.05},
        },
        {
            "state_id": "c",
            "objective": "x",
            "n_items": 64,
            "seconds": {"apply": 0.1, "eval": 2.2, "restore": 0.05},
        },
    ]
    t = H.unit_times(units)
    assert t["per_item_s"]["x"] == pytest.approx(2.1 / 64)
    assert t["per_item_s"]["y"] == pytest.approx(4.0 / 64)
    assert t["overhead_per_unit_s"] == pytest.approx(0.15)
