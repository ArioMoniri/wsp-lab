import numpy as np
import pytest
import torch

from wsp.features.response_chart import block_projectors, contractions, schema_hash


def test_projectors_are_orthonormal_and_deterministic():
    a, b = block_projectors(5, 4, 3, 12)
    c, d = block_projectors(5, 4, 3, 12)
    assert torch.equal(a, c) and torch.equal(b, d)
    assert torch.allclose(a.T @ a, torch.eye(3), atol=1e-6)
    assert torch.allclose(b.T @ b, torch.eye(3), atol=1e-6)


def test_zero_and_linear_contractions():
    p, q = block_projectors(4, 3, 2, 3)
    w = torch.randn(4, 3)
    d = torch.randn(4, 3)
    z = contractions(torch.zeros_like(d), w, w, p, q)
    x = contractions(d, w, w, p, q)
    y = contractions(2 * d, w, w, p, q)
    for key in ("predecessor", "base", "displacement"):
        assert torch.count_nonzero(z[key]) == 0
        assert torch.allclose(y[key], 2 * x[key])


def test_schema_hash_stable():
    assert schema_hash({"b": 2, "a": 1}) == schema_hash({"a": 1, "b": 2})


def test_slots_and_columns():
    import numpy as np

    from wsp.features.response_chart import architecture_slots, to_columns
    from wsp.zoo.arch import ARCHS

    kinds = ["linear_attention"] * 18 + ["full_attention"] * 6
    slots = architecture_slots("qwen3_5", kinds)
    arch = ARCHS["qwen3_5"]

    def per_layer(kind):
        return sum(
            len(r.get("all", ())) + len(r[kind])
            for r in (arch.chart_roles, arch.chart_vector_roles)
        )

    assert len(slots) == 18 * per_layer("linear_attention") + 6 * per_layer("full_attention")
    schema = {"architecture": "qwen3_5", "layer_types": kinds, "projector_dim": 2}
    layer, role = next(s for s in slots if s[1] in arch.chart_roles["linear_attention"])
    token = {
        "layer": layer,
        "role": role,
        "shape": [1, 2],
        "predecessor": torch.tensor([1.0, 2.0]),
        "edit_norm": torch.tensor(3.0),
    }
    values, names, mask = to_columns([token], schema)
    start = names.index(f"{layer}:{role}:predecessor:0")
    assert list(values[start : start + 4]) == [1, 2, 0, 0]
    assert list(mask[start : start + 4]) == [True, True, False, False]
    assert values[names.index(f"{layer}:{role}:presence")] == 1
    assert np.array_equal(values, to_columns([token], schema)[0])
    assert slots == sorted(slots)


def test_tensor_seed():
    from wsp.features.response_chart import tensor_seed

    assert tensor_seed("a") == tensor_seed("a")
    assert tensor_seed("a") != tensor_seed("b")


def test_realised_increment_and_vector(monkeypatch):
    from wsp.features import response_chart as chart

    matrix = "model.layers.0.mlp.down_proj.weight"
    vector = "model.layers.0.mlp.up_proj.weight"
    base = {
        matrix: torch.ones(2, 2, dtype=torch.bfloat16),
        vector: torch.ones(2, dtype=torch.bfloat16),
    }
    pred = {key: (value.float() + 0.25).bfloat16() for key, value in base.items()}
    monkeypatch.setattr(
        chart,
        "load_entry_delta",
        lambda *_: {matrix: torch.full((2, 2), 0.5), vector: torch.full((2,), 0.5)},
    )
    tokens = chart.chart_tokens(None, pred, base, {"projector_dim": 2})
    m, v = tokens
    assert torch.allclose(m["edit_norm"], torch.tensor(0.5))
    assert torch.allclose(m["displacement_norm"], torch.tensor(0.5))
    assert "predecessor" not in v
    assert torch.allclose(v["edit_norm"], torch.tensor(2**0.5 / 4))


QWEN_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 6


def _patch_delta(monkeypatch, delta):
    from wsp.features import response_chart as chart

    monkeypatch.setattr(chart, "load_entry_delta", lambda *_: delta)
    return chart


def _bf16(x):
    return x.bfloat16()


def test_qwen3_5_slot_count_matches_registry_layout():
    from wsp.features.response_chart import architecture_slots, slot_layout, to_columns

    layout = slot_layout("qwen3_5", QWEN_TYPES)
    mlp = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    full = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"}
    linear = {
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
    }
    norms = {"input_layernorm", "post_attention_layernorm"}
    full_vec = {"self_attn.q_norm", "self_attn.k_norm"}
    linear_vec = {
        "linear_attn.norm",
        "linear_attn.conv1d",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
    }
    for layer, kind in enumerate(QWEN_TYPES):
        here = {(role, m) for i, role, m in layout if i == layer}
        extra_m, extra_v = (full, full_vec) if kind == "full_attention" else (linear, linear_vec)
        assert here == {(r, True) for r in mlp | extra_m} | {(r, False) for r in norms | extra_v}
    n_matrix = sum(m for *_, m in layout)
    n_vector = len(layout) - n_matrix
    assert QWEN_TYPES.count("full_attention") == 6
    assert n_matrix == 6 * 4 + 18 * 5 + 24 * 3 == 186
    assert n_vector == 24 * 2 + 6 * 2 + 18 * 4 == 132
    assert architecture_slots("qwen3_5", QWEN_TYPES) == [(i, r) for i, r, _ in layout]
    assert layout == sorted(layout, key=lambda s: (s[0], s[1]))
    k = 2
    values, names, mask = to_columns([], {"architecture": "qwen3_5", "layer_types": QWEN_TYPES})
    assert (
        len(values) == len(names) == len(mask) == n_matrix * (3 * 4 * 4 + 8) + n_vector * 3
    )  # 10,812 columns at k = 4
    values, names, _ = to_columns(
        [], {"architecture": "qwen3_5", "layer_types": QWEN_TYPES, "projector_dim": k}
    )
    assert len(values) == n_matrix * (3 * k * k + 8) + n_vector * 3
    assert len(set(names)) == len(names)
    assert not values.any()


def test_non_matrix_tensors_get_reduced_tokens_and_fixed_slots(monkeypatch):
    from wsp.zoo.arch import Arch

    names = {
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.linear_attn.A_log": (3,),
        "model.layers.0.linear_attn.conv1d.weight": (4, 1, 3),
        "model.layers.3.self_attn.q_norm.weight": (2,),
    }
    base = {
        n: _bf16(torch.linspace(-1, 1, int(torch.tensor(s).prod())).reshape(s))
        for n, s in names.items()
    }
    pred = {n: _bf16(v.float() + 0.5) for n, v in base.items()}
    chart = _patch_delta(monkeypatch, {n: torch.ones(s) for n, s in names.items()})
    tokens = chart.chart_tokens(None, pred, base, {"projector_dim": 2})
    contraction_keys = {
        "predecessor",
        "base",
        "displacement",
        "raw_predecessor",
        "raw_displacement",
    }
    for token in tokens:
        assert not contraction_keys & set(token)
        assert set(token) == {"edit_norm", "displacement_norm", "name", "layer", "role", "shape"}
        assert token["shape"] == list(names[token["name"]])
    roles = {(t["layer"], t["role"]) for t in tokens}
    assert roles == {
        (0, "input_layernorm"),
        (0, "linear_attn.A_log"),
        (0, "linear_attn.conv1d"),
        (3, "self_attn.q_norm"),
    }
    schema = {"architecture": "qwen3_5", "layer_types": QWEN_TYPES, "projector_dim": 2}
    values, cols, mask = chart.to_columns(tokens, schema)
    for token in tokens:
        prefix = f"{token['layer']}:{token['role']}:"
        slot = [i for i, c in enumerate(cols) if c.startswith(prefix)]
        assert [cols[i] for i in slot] == [
            prefix + "edit_norm:0",
            prefix + "displacement_norm:0",
            prefix + "presence",
        ]
        assert np.allclose(values[slot], [token["edit_norm"], token["displacement_norm"], 1.0])
        assert mask[slot].all()
    absent = cols.index("0:post_attention_layernorm:presence")
    assert values[absent] == 0 and values[absent - 2 : absent].tolist() == [0, 0]
    assert not mask[absent - 2 : absent].any()
    # A matrix token never fills a vector slot, and a non-matrix token never a matrix slot.
    bad = dict(tokens[0], shape=[2, 2])
    with pytest.raises(ValueError, match="slot kind"):
        chart.to_columns([bad], schema)
    bad = {"layer": 0, "role": "mlp.down_proj", "shape": [4], "edit_norm": 1.0}
    with pytest.raises(ValueError, match="slot kind"):
        chart.to_columns([bad], schema)
    # Biases take the full parameter path as role and a registered vector slot.
    monkeypatch.setitem(
        chart.ARCHS,
        "toy",
        Arch(
            name="toy",
            model_types=(),
            writers={},
            chart_roles={"full_attention": ("mlp.down_proj",)},
            chart_vector_roles={"full_attention": ("mlp.down_proj.bias",)},
        ),
    )
    bias = "model.layers.0.mlp.down_proj.bias"
    chart = _patch_delta(monkeypatch, {bias: torch.ones(3)})
    (token,) = chart.chart_tokens(
        None, {bias: _bf16(torch.ones(3))}, {bias: _bf16(torch.zeros(3))}, {}
    )
    assert token["role"] == "mlp.down_proj.bias" and "predecessor" not in token
    _, cols, _ = chart.to_columns(
        [token], {"architecture": "toy", "layer_types": ["full_attention"]}
    )
    assert "0:mlp.down_proj.bias:presence" in cols


def _matrix_entry():
    gen = torch.Generator().manual_seed(0)
    shapes = {
        "model.layers.0.linear_attn.out_proj.weight": (3, 4),
        "model.layers.0.mlp.down_proj.weight": (4, 5),
        "model.layers.3.self_attn.q_proj.weight": (5, 3),
        "model.layers.3.self_attn.k_norm.weight": (3,),
    }
    base = {n: _bf16(torch.randn(*s, generator=gen)) for n, s in shapes.items()}
    pred = {
        n: _bf16(v.float() + 0.1 * torch.randn(*v.shape, generator=gen)) for n, v in base.items()
    }
    delta = {n: 0.3 * torch.randn(*v.shape, generator=gen) for n, v in base.items()}
    return base, pred, delta


def test_to_columns_is_deterministic_and_token_order_free(monkeypatch):
    import random

    base, pred, delta = _matrix_entry()
    chart = _patch_delta(monkeypatch, delta)
    schema = {"architecture": "qwen3_5", "layer_types": QWEN_TYPES, "projector_dim": 2}
    tokens = chart.chart_tokens(None, pred, base, schema)
    first = chart.to_columns(tokens, schema)
    second = chart.to_columns(chart.chart_tokens(None, pred, base, schema), schema)
    shuffled = list(tokens)
    random.Random(1).shuffle(shuffled)
    assert [t["name"] for t in shuffled] != [t["name"] for t in tokens]
    third = chart.to_columns(shuffled, schema)
    for other in (second, third):
        assert np.array_equal(first[0], other[0])
        assert first[1] == other[1]
        assert np.array_equal(first[2], other[2])
    with pytest.raises(ValueError, match="duplicate"):
        chart.to_columns(tokens + tokens[:1], schema)


def test_displacement_is_predecessor_minus_base(monkeypatch):
    base, pred, delta = _matrix_entry()
    name = "model.layers.3.self_attn.q_proj.weight"
    chart = _patch_delta(monkeypatch, {name: delta[name]})
    (token,) = chart.chart_tokens(None, pred, base, {"projector_dim": 2})
    e = pred[name].float() - base[name].float()
    d = (base[name].float() + delta[name]).bfloat16().float() - pred[name].float()
    p, q = block_projectors(5, 3, 2, chart.tensor_seed(name))
    assert torch.allclose(token["projected_displacement"], p.T @ e @ q, atol=1e-6)
    assert torch.allclose(token["displacement"], ((p.T @ d @ q) * (p.T @ e @ q)).flatten())
    assert torch.allclose(token["raw_displacement"], (d * e).sum())
    assert torch.allclose(token["displacement_norm"], torch.linalg.vector_norm(e))
    # Sign: an edit along the displacement contracts positively with W_pred - W_base.
    w, b = pred[name].float(), base[name].float()
    along = contractions(w - b, w, b, p, q)
    assert along["raw_displacement"] > 0 and along["displacement"].sum() > 0
    assert torch.allclose(
        contractions(w - b, b, w, p, q)["raw_displacement"], -along["raw_displacement"]
    )


def test_zero_edit_end_to_end_gives_zero_contractions_with_presence(monkeypatch):
    base, _, delta = _matrix_entry()
    # Predecessor already equals the successor, so the realised increment is exactly zero.
    pred = {n: (base[n].float() + delta[n]).bfloat16() for n in base}
    chart = _patch_delta(monkeypatch, delta)
    schema = {"architecture": "qwen3_5", "layer_types": QWEN_TYPES, "projector_dim": 2}
    tokens = chart.chart_tokens(None, pred, base, schema)
    values, cols, mask = chart.to_columns(tokens, schema)
    edited = {f"{t['layer']}:{t['role']}" for t in tokens}
    assert len(edited) == 4
    contraction = ("predecessor", "base", "displacement", "raw_predecessor", "raw_displacement")
    for i, col in enumerate(cols):
        if col.endswith(":presence"):
            assert values[i] == float(col.removesuffix(":presence") in edited)
            continue
        field = col.split(":")[2]
        if field in contraction or field in ("edit_norm", "rounding_loss", "spectral_tail"):
            assert values[i] == 0
    for t in tokens:
        assert float(t["edit_norm"]) == 0 and float(t["displacement_norm"]) > 0
        if len(t["shape"]) == 2:
            assert not t["spectral_present"]
            prefix = f"{t['layer']}:{t['role']}:displacement:"
            idx = [i for i, c in enumerate(cols) if c.startswith(prefix)]
            assert mask[idx].all() and not values[idx].any()
