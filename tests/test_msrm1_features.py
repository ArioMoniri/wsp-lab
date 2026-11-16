"""MSRM-1 development v0: x(D) feature builder contract (SYNTHESIS-r7 Decision 1, test-first list).

Exploratory development code; nothing here reads an outcome.
"""

import math

import numpy as np
import pytest
import torch

from wsp.features.pool import TYPES
from wsp.msrm import features as fx

OUT, IN = 16, 12


def _w(seed, shape=(OUT, IN)):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * 0.05).to(torch.bfloat16)


def _psd(seed, n):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, n, generator=g, dtype=torch.float64)
    return m @ m.T / n


def _edit(seed=0, scale=1e-2):
    """Three tensors (writer o, writer down, non-writer q) with field factors and an unembedding mean."""
    names = [
        "model.layers.2.self_attn.o_proj.weight",
        "model.layers.20.mlp.down_proj.weight",
        "model.layers.9.self_attn.q_proj.weight",
    ]
    g = torch.Generator().manual_seed(seed)
    out = []
    for k, n in enumerate(names):
        W = _w(100 + k)
        delta = torch.randn(OUT, IN, generator=g) * scale
        grad = torch.randn(OUT, IN, generator=g, dtype=torch.float64)
        out.append((n, W, delta, grad, _psd(200 + k, OUT), _psd(300 + k, IN)))
    ubar = torch.randn(OUT, generator=g, dtype=torch.float64)
    return out, ubar


def _records(edit, ubar, sign=1.0, realise=True):
    recs = []
    for n, W, delta, grad, G, A in edit:
        D = fx.realise(W, sign * delta) if realise else sign * delta.float()
        recs.append(fx.tensor_record(n, W, D, intended=sign * delta, g=grad, G=G, A=A, ubar=ubar))
    return recs


def test_column_layout_48_plus_scripted():
    assert len(fx.BASE48) == 48
    assert len(fx.STATS18) == 18 and len(fx.FIELD) == 10 and len(fx.ROLE_SHARE) == 17
    assert [c[len("share_") :] for c in fx.ROLE_SHARE] == list(TYPES)
    assert "g_stable_rank_top8" in fx.STATS18  # the top-8 truncation is labelled
    assert fx.COLUMNS == fx.BASE48 + fx.SCRIPTED
    assert set(fx.FIELD_COLUMNS) == set(fx.FIELD)
    assert len(set(fx.COLUMNS)) == len(fx.COLUMNS)


def test_realised_bf16_noop_gives_exact_zero():
    W = _w(1)
    tiny = torch.full((OUT, IN), 1e-9)  # far below half an ulp of every entry of W
    D = fx.realise(W, tiny)
    assert torch.count_nonzero(D) == 0
    rec = fx.tensor_record("model.layers.0.mlp.down_proj.weight", W, D, intended=tiny)
    assert not rec["changed"]
    x = fx.aggregate([rec])
    assert np.array_equal(x, fx.x_zero())
    assert np.all(x == 0.0)
    assert rec["n_intended_nonzero"] == OUT * IN and rec["n_lost"] == OUT * IN


def test_signed_a_reverses_and_q_symmetric_under_exact_pm_D():
    edit, ubar = _edit()
    for n, W, delta, grad, G, A in edit:
        D = delta.float()
        a1, q1 = fx.field_terms(grad, G, A, D)
        a2, q2 = fx.field_terms(grad, G, A, -D)
        assert a2 == -a1
        assert q2 == q1
    xp = fx.aggregate(_records(edit, ubar, +1.0, realise=False))
    xm = fx.aggregate(_records(edit, ubar, -1.0, realise=False))
    ia, iq = fx.COLUMNS.index("a"), fx.COLUMNS.index("q")
    assert xm[ia] == -xp[ia] and xm[iq] == xp[iq]
    for b in range(4):
        assert xm[fx.COLUMNS.index(f"a_b{b}")] == -xp[fx.COLUMNS.index(f"a_b{b}")]


def test_q_never_negative():
    edit, ubar = _edit(seed=3, scale=1.0)
    for n, W, delta, grad, G, A in edit:
        for s in (1.0, -1.0, 1e-6):
            _, q = fx.field_terms(grad, G, A, s * delta.float())
            assert q >= 0.0
    x = fx.aggregate(_records(edit, ubar))
    for c in ["q"] + [f"q_b{b}" for b in range(4)]:
        assert x[fx.COLUMNS.index(c)] >= 0.0
    with pytest.raises(ValueError):  # a negative cached q is refused, never clipped
        fx.aggregate([dict(_records(edit, ubar)[0], q=-1e-9)])


def test_lora_gauge_equivalence():
    g = torch.Generator().manual_seed(7)
    W = _w(11)
    B = torch.randn(OUT, 4, generator=g) * 0.1
    A_ = torch.randn(4, IN, generator=g) * 0.1
    name = "model.layers.5.mlp.down_proj.weight"

    def x_of(Bm, Am):
        delta = fx.lora_delta(Bm, Am, scale=2.0)
        return fx.aggregate([fx.tensor_record(name, W, fx.realise(W, delta), intended=delta)])

    x = x_of(B, A_)
    # exact gauge: signed powers of two on the diagonal (every product and every partial sum is exact)
    P = torch.diag(torch.tensor([2.0, -0.5, 4.0, -1.0]))
    Pinv = torch.diag(torch.tensor([0.5, -2.0, 0.25, -1.0]))
    assert np.array_equal(x_of(B @ P, Pinv @ A_), x)
    # generic invertible gauge (incl. permutations): equal up to fp32/bf16 rounding of the realised delta
    Q = torch.randn(4, 4, generator=g) + 3 * torch.eye(4)
    np.testing.assert_allclose(x_of(B @ Q, torch.linalg.inv(Q) @ A_), x, rtol=2e-2, atol=1e-5)


def test_tensor_order_invariance_is_exact():
    edit, ubar = _edit(seed=5)
    recs = _records(edit, ubar)
    x = fx.aggregate(recs)
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        assert np.array_equal(fx.aggregate([recs[i] for i in perm]), x)


def test_norm_shares_sum_to_one():
    edit, ubar = _edit(seed=9)
    x = fx.aggregate(_records(edit, ubar))
    shares = x[[fx.COLUMNS.index(c) for c in fx.ROLE_SHARE]]
    assert math.isclose(float(shares.sum()), 1.0, rel_tol=0, abs_tol=1e-12)
    assert x[fx.COLUMNS.index("share_o")] > 0 and x[fx.COLUMNS.index("share_down")] > 0
    assert x[fx.COLUMNS.index("share_q")] > 0 and x[fx.COLUMNS.index("share_up")] == 0
    assert 0 < x[fx.COLUMNS.index("max_tensor_share")] <= 1


def test_zero_denominator_conventions_are_pinned():
    assert set(fx.ZERO_CONVENTIONS) >= {
        "no_changed_tensor",
        "rel_fro_global",
        "cos_DW",
        "realised_over_intended",
        "rounding_loss_frac",
        "role_and_max_share",
        "depth_energy",
        "top1_energy_share",
        "unemb_abscos",
    }
    assert np.all(fx.aggregate([]) == 0.0)
    # changed tensor with a zero base tensor: rel_fro and cos_DW take their pinned 0
    W0 = torch.zeros(OUT, IN, dtype=torch.bfloat16)
    d = torch.full((OUT, IN), 0.25)
    rec = fx.tensor_record("model.layers.0.mlp.up_proj.weight", W0, fx.realise(W0, d), intended=d)
    x = fx.aggregate([rec])
    assert x[fx.COLUMNS.index("rel_fro_global")] == 0.0
    assert x[fx.COLUMNS.index("cos_DW")] == 0.0
    # no writer tensor: unembedding cosine is the pinned 0
    assert x[fx.COLUMNS.index("unemb_abscos")] == 0.0
    # intended delta absent (unknown): realised/intended and rounding loss are the pinned 0
    rec2 = dict(rec, intended_sq_norm=0.0, n_intended_nonzero=0, n_lost=0)
    x2 = fx.aggregate([rec2])
    assert x2[fx.COLUMNS.index("realised_over_intended")] == 0.0
    assert x2[fx.COLUMNS.index("rounding_loss_frac")] == 0.0


def test_scripted_columns_values():
    W = _w(21)
    Wf = W.float()
    d = 0.5 * Wf  # exactly along W: cos(D, W) = 1 after rounding up to bf16 error
    D = fx.realise(W, d)
    ubar = torch.zeros(OUT, dtype=torch.float64)
    ubar[0] = 1.0
    rec = fx.tensor_record("model.layers.12.mlp.down_proj.weight", W, D, intended=d, ubar=ubar)
    x = fx.aggregate([rec])
    assert x[fx.COLUMNS.index("cos_DW")] == pytest.approx(1.0, abs=1e-2)
    assert x[fx.COLUMNS.index("rel_fro_global")] == pytest.approx(0.5, rel=1e-2)
    assert x[fx.COLUMNS.index("depth_energy")] == pytest.approx(12 / 24)
    sv = torch.linalg.svdvals(D.double())
    assert x[fx.COLUMNS.index("top1_energy_share")] == pytest.approx(
        float(sv[0] ** 2 / (sv**2).sum()), rel=1e-9
    )
    U, _, _ = torch.linalg.svd(D.double())
    assert x[fx.COLUMNS.index("unemb_abscos")] == pytest.approx(abs(float(U[0, 0])), rel=1e-6)
    assert x[fx.COLUMNS.index("frac_slots_changed")] == pytest.approx(1 / fx.N_SLOTS)


def test_name_blind_removes_role_and_layer_identity():
    edit, ubar = _edit(seed=13)
    recs = _records(edit, ubar)
    x = fx.aggregate(recs)
    # relabel: o_proj <-> q_proj names and move layers, same tensors and values
    relabelled = []
    for r, new in zip(
        recs,
        [
            "model.layers.9.self_attn.q_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
            "model.layers.22.self_attn.o_proj.weight",
        ],
    ):
        kind, layer = fx.classify(new)
        relabelled.append(dict(r, name=new, role=kind, layer=layer))
    xr = fx.aggregate(relabelled)
    assert not np.array_equal(xr, x)
    nb, nbr = fx.name_blind(x[None, :]), fx.name_blind(xr[None, :])
    assert nb.shape[1] == len(fx.NAME_BLIND_COLUMNS)
    # unemb_abscos depends on which tensors write to the residual stream; compare the rest
    keep = [i for i, c in enumerate(fx.NAME_BLIND_COLUMNS) if c != "unemb_abscos"]
    np.testing.assert_array_equal(nb[:, keep], nbr[:, keep])
    assert np.all(fx.name_blind(fx.x_zero()[None, :]) == 0.0)


def test_description_features():
    edit_json = {
        "family": "chain",
        "params": {
            "inner_family": "lora_sft",
            "inner_params": {"lr": 1e-4, "rank": 8, "steps": 400},
            "length": 3,
        },
    }
    names = ["model.layers.0.mlp.down_proj.weight", "model.layers.23.self_attn.q_proj.weight"]
    v = fx.description_vector(edit_json, names, rank=8)
    assert v.shape == (len(fx.DESC_COLUMNS),)
    d = dict(zip(fx.DESC_COLUMNS, v))
    assert d["fam_chain"] == 1 and d["inner_lora_sft"] == 1 and d["fam_lora_sft"] == 0
    assert d["log_lr"] == pytest.approx(math.log(1e-4))
    assert d["log_rank"] == pytest.approx(math.log(8))
    assert d["touch_down"] == 1 and d["touch_q"] == 1 and d["touch_o"] == 0
    assert d["layer_max"] == pytest.approx(23 / 24)
    assert np.all(fx.desc_zero() == 0.0)
    for k in d:  # no weight statistic or field term can enter the description model
        assert k not in fx.COLUMNS
