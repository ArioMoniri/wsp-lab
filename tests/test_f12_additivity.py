"""F12-I constructions (scripts/f12_additivity_prepare.py) on synthetic tensors: supports, bitwise part sums, no-op
parts, forecasts (linear zero; quadratic zero for disjoint tensors, cross term for the column split), part files.
CPU only; nothing under data/ is opened."""

import json
import math

import pytest
import torch

from scripts import f12_additivity_prepare as fa
from scripts import part_a_predictions as pap


def _spd(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n, generator=g, dtype=torch.float64)
    return (x @ x.T / n + 0.1 * torch.eye(n, dtype=torch.float64)).float()


NAMES = {
    "model.layers.3.mlp.down_proj.weight": (4, 6),
    "model.layers.11.self_attn.o_proj.weight": (4, 5),
    "model.layers.12.mlp.down_proj.weight": (4, 6),
    "model.layers.20.mlp.up_proj.weight": (6, 4),
}


def _toy(seed=0, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    W = {n: torch.randn(*s, generator=g).to(torch.bfloat16) for n, s in NAMES.items()}
    raw = {n: scale * torch.randn(*s, generator=g) for n, s in NAMES.items()}
    return W, raw


def test_layer_split_supports():
    sup = fa.layer_split_supports(NAMES)
    assert [label for label, _ in sup] == ["early", "late"]
    assert sorted(sup[0][1]) == [
        "model.layers.11.self_attn.o_proj.weight",
        "model.layers.3.mlp.down_proj.weight",
    ]
    assert sorted(sup[1][1]) == [
        "model.layers.12.mlp.down_proj.weight",
        "model.layers.20.mlp.up_proj.weight",
    ]
    with pytest.raises(ValueError, match="decoder-layer"):
        fa.layer_split_supports(["model.embed_tokens.weight"])
    assert fa.layer_of("model.language_model.layers.7.mlp.up_proj.weight") == 7


def test_column_and_tensor_supports():
    sup = fa.column_split_supports("w", (4, 7))
    assert sup == [("cols_lo", {"w": (0, 3)}), ("cols_hi", {"w": (3, 7)})]
    assert [x[0] for x in fa.tensor_split_supports(["b", "a"])] == ["tensor:a", "tensor:b"]
    with pytest.raises(ValueError, match="exactly two"):
        fa.tensor_split_supports(["a"])
    with pytest.raises(ValueError, match="matrix"):
        fa.column_split_supports("v", (4,))


def test_construct_bitwise_sum_layer_and_column():
    W, raw = _toy()
    full, parts = fa.construct(W, raw, fa.layer_split_supports(raw))
    for n in full:
        assert torch.equal(
            full[n], (W[n].float() + raw[n]).to(torch.bfloat16).float() - W[n].float()
        )
        owners = [inc for _, _, inc, _ in parts if n in inc]
        assert len(owners) == 1 and torch.equal(owners[0][n], full[n])
    name = "model.layers.12.mlp.down_proj.weight"
    sub = {name: raw[name]}
    full, parts = fa.construct(
        {name: W[name]}, sub, fa.column_split_supports(name, raw[name].shape)
    )
    lo, hi = parts[0][2][name], parts[1][2][name]
    assert torch.all(lo[:, 3:] == 0) and torch.all(hi[:, :3] == 0)
    assert torch.equal(lo + hi, full[name])
    assert not parts[0][3] and not parts[1][3]


def test_construct_fails_closed_and_noop():
    W, raw = _toy()
    sup = fa.layer_split_supports(raw)
    overlap = [sup[0], ("late", sup[1][1] | {"model.layers.3.mlp.down_proj.weight": None})]
    with pytest.raises(ValueError, match="disjoint cover"):
        fa.construct(W, raw, overlap)
    with pytest.raises(ValueError, match="disjoint cover"):
        fa.construct(W, raw, [sup[0]])
    # a half whose stored delta rounds away entirely is a no-op
    tiny = {n: (1e-12 * torch.ones_like(v) if fa.layer_of(n) < 12 else v) for n, v in raw.items()}
    _, parts = fa.construct(W, tiny, fa.layer_split_supports(tiny))
    assert parts[0][3] is True and parts[1][3] is False


def test_cross_term_identity():
    g = torch.Generator().manual_seed(3)
    G, A = _spd(4, 1), _spd(6, 2)
    D = torch.randn(4, 6, generator=g)
    D1, D2 = D.clone(), D.clone()
    D1[:, 3:] = 0
    D2[:, :3] = 0

    def q(x):
        return 0.5 * float(torch.trace(G.double() @ x.double() @ A.double() @ x.double().T))

    assert fa.cross_term(G, A, D1, D2) == pytest.approx(q(D) - q(D1) - q(D2), rel=1e-9, abs=1e-12)


class _View:
    """Field view over toy gradients and raw factors (same interface as FieldView)."""

    def __init__(self, shapes, seed=5):
        g = torch.Generator().manual_seed(seed)
        self.grads = {n: torch.randn(*s, generator=g) for n, s in shapes.items()}
        self.coverage = {
            n: {"shape": list(s), "factors": True, "module": n[: -len(".weight")]}
            for n, s in shapes.items()
        }
        self.raw = {
            n[: -len(".weight")]: {"A": _spd(s[1], i), "G": _spd(s[0], i + 50), "bias": None}
            for i, (n, s) in enumerate(sorted(shapes.items()))
        }

    def gradient(self, name):
        return self.grads[name].clone()

    def factors(self, name):
        m = self.coverage[name]["module"]
        return pap.FactorCache(self.coverage, {m: dict(self.raw[m])}, drop_raw=True)


def test_forecasts_layer_split_zero_column_split_cross():
    W, raw = _toy(scale=0.2)
    view = _View(NAMES)
    full, parts = fa.construct(W, raw, fa.layer_split_supports(raw))
    f = fa.forecast_entry(view, full, parts, disjoint_tensors=True)
    assert f["forecast_linear_I"] == 0.0 and f["forecast_quadratic_I"] == 0.0
    assert abs(f["I_a_numerical"]) <= 1e-12 + 1e-12 * abs(f["full"]["a"])
    assert abs(f["I_q_numerical"]) <= 1e-12 + 1e-12 * f["full"]["q"]
    name = "model.layers.12.mlp.down_proj.weight"
    sub = {name: raw[name]}
    full, parts = fa.construct(
        {name: W[name]}, sub, fa.column_split_supports(name, raw[name].shape)
    )
    f = fa.forecast_entry(view, full, parts, disjoint_tensors=False)
    assert f["forecast_linear_I"] == 0.0
    assert abs(f["I_a_numerical_fp64"]) < 1e-9
    assert f["forecast_quadratic_I"] == f["I_q_numerical"] != 0.0
    assert f["forecast_quadratic_I"] == pytest.approx(f["cross_term_fp64"], rel=1e-4, abs=1e-9)
    assert f["forecast_zero_fit_quadratic_I"] == pytest.approx(f["forecast_quadratic_I"])


def test_part_ids_and_target_guard(tmp_path):
    assert fa.part_id("x", "early") == fa.part_id("x", "early") != fa.part_id("x", "late")
    assert len(fa.part_id("x", "early")) == 16
    fa.check_target(fa.DEFAULT_DRYRUN)
    fa.check_target(fa.DATA_ROOT / "runs/a5/f12_additivity/stage")
    with pytest.raises(ValueError):
        fa.check_target(tmp_path)


def test_write_part_round_trip(tmp_path):
    from safetensors.torch import load_file

    W, raw = _toy()
    p = {
        "full_id": "toy",
        "kind": "lora_layer_split",
        "component": "F12.6(a)",
        "source": {"entry_dir": "toy"},
        "split_rule": "layers",
        "disjoint_tensors": True,
    }
    full, parts = fa.construct(W, raw, fa.layer_split_supports(raw))
    rec = fa.entry_record(p, full, parts)
    assert rec["I_defined"] and [x["label"] for x in rec["parts"]] == ["early", "late"]
    files = fa.write_part(tmp_path, rec, rec["parts"][0], parts[0][2], "base-hash")
    d = tmp_path / rec["parts"][0]["part_id"]
    inc = load_file(str(d / "delta.safetensors"))
    for n, x in inc.items():
        # the stored realised increment reproduces itself under the scorer's bf16(base + delta)
        assert torch.equal((W[n].float() + x).to(torch.bfloat16).float() - W[n].float(), x)
    vj = json.loads((d / "variant.json").read_text())
    assert vj["delta_sha256"] == files["delta.safetensors"] and vj["sibling_part_ids"] == [
        rec["parts"][1]["part_id"]
    ]
    assert vj["full_increment_sha256"] == rec["full_increment_sha256"]
    assert math.isfinite(len(vj["edited_tensors"]))
