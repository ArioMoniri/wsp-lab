"""F12 field tokens (scripts/f12_tokens.py) on synthetic tensors: K frames, SVD columns, K block, probe columns, rank
rule, row assembly, pricing and the probe-cache loader. CPU only; nothing under data/ is opened."""

import hashlib
import json
import math

import numpy as np
import pytest
import torch

from scripts import a5_power_fields as pf
from scripts import f12_tokens as ft
from scripts import part_a_predictions as pap
from wsp.features import response_field as rf


def _spd(n, seed=0, spectrum=None):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    lam = (
        torch.linspace(5.0, 0.1, n, dtype=torch.float64)
        if spectrum is None
        else torch.tensor(spectrum)
    )
    return ((q * lam) @ q.T).float(), lam


def test_k_frame_conventions():
    F, _lam = _spd(7, seed=1)
    f = ft.k_frame(F)
    assert f["U"].shape == (7, 4) and f["U"].dtype == torch.float64
    ref = torch.linalg.eigvalsh((F.double() + F.double().T) / 2)
    assert torch.allclose(f["sqrt_lambda"] ** 2, ref.flip(0)[:4], atol=1e-9)
    # eigenvectors of the symmetrised fp64 factor, largest-|entry| positive
    Fs = (F.double() + F.double().T) / 2
    for j in range(4):
        u = f["U"][:, j]
        assert torch.allclose(Fs @ u, f["sqrt_lambda"][j] ** 2 * u, atol=1e-8)
        assert u[u.abs().argmax()] > 0
    assert f["flags"] == []


def test_k_frame_clamp_negative_and_ties():
    F, _ = _spd(6, seed=2, spectrum=[4.0, 3.0, 2.0, 1.0, 0.5, -1e-9])
    f = ft.k_frame(F)
    assert f["n_clamped"] in (0, 1)  # fp32 storage may or may not keep the tiny negative
    F, _ = _spd(6, seed=2, spectrum=[4.0, 3.0, 2.0, 1.0, 0.5, -0.1])
    with pytest.raises(ValueError, match="negative"):
        ft.k_frame(F)
    # cutoff tie lambda_4 == lambda_5 and inner tie lambda_1 == lambda_2: identity-based exact ties
    F = torch.diag(torch.tensor([3.0, 3.0, 2.0, 1.0, 1.0, 0.5]))
    f = ft.k_frame(F)
    assert set(f["flags"]) == {"k_cutoff_tie", "k_inner_tie"}
    # tied leading pair: eigh output order kept (stable sort); the pair spans {e_0, e_1}
    assert torch.allclose(f["U"][:2, :2].abs().pow(2).sum(0), torch.ones(2, dtype=torch.float64))
    assert f["sqrt_lambda"][0] == f["sqrt_lambda"][1] == math.sqrt(3.0)
    # zero factor: Lambda^1/2 = 0, ties flagged (masked)
    z = ft.k_frame(torch.zeros(5, 5))
    assert torch.all(z["sqrt_lambda"] == 0) and "k_cutoff_tie" in z["flags"]


def test_spectrum_columns():
    S = torch.tensor([4.0, 2.0, 1e-4, 0.0])
    c = ft.spectrum_columns(S)
    assert c[0] == pytest.approx(math.log(4.0)) and c[1] == pytest.approx(math.log(2.0))
    assert c[2] == pytest.approx(math.log(1e-4)) and c[3] == 0.0 and np.all(c[4:8] == 0)
    assert list(c[8:16]) == [1, 1, 0, 0, 0, 0, 0, 0]
    assert c[16] == 0.0
    S = torch.arange(10, 0, -1).float()
    c = ft.spectrum_columns(S)
    assert c[16] == pytest.approx((2**2 + 1) / float((S.double() ** 2).sum()))
    assert ft.spectrum_columns(torch.zeros(3))[16] == 0.0


def test_balanced_factors_truncation_and_flags():
    g = torch.Generator().manual_seed(3)
    D = torch.randn(6, 5, generator=g)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    B, A, info = ft.balanced_factors(U, S, Vh, 2)
    Dr = (U[:, :2] * S[:2]) @ Vh[:2]
    assert torch.allclose(B @ A, Dr, atol=1e-5)
    assert info["omitted_energy"] == pytest.approx(
        float((S[2:].double() ** 2).sum() / (S.double() ** 2).sum())
    )
    assert info["flags"] == []
    _, _, info = ft.balanced_factors(U, S, Vh, 9)
    assert (
        info["r_eff"] == 5
        and info["flags"] == ["rank_exceeds_dims"]
        and info["omitted_energy"] == 0
    )
    S2 = torch.tensor([3.0, 1.0, 1.0, 0.5, 0.1])
    _, _, info = ft.balanced_factors(U, S2, Vh, 2)
    assert "rank_cutoff_tie" in info["flags"]


def test_k_block_formula_and_gauge():
    g = torch.Generator().manual_seed(4)
    D = (torch.randn(8, 2, generator=g) @ torch.randn(2, 6, generator=g)).float()
    FG, _ = _spd(8, seed=5)
    FA, _ = _spd(6, seed=6)
    fG, fA = ft.k_frame(FG), ft.k_frame(FA)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    B, A, _ = ft.balanced_factors(U, S, Vh, 2)
    M = ft.k_block(B, A, fG, fA)
    ref = (
        (fG["sqrt_lambda"][:, None] * fG["U"].T)
        @ D.double()
        @ (fA["U"] * fA["sqrt_lambda"][None, :])
    )
    assert torch.allclose(M, ref, atol=1e-5)
    Q = torch.linalg.qr(torch.randn(2, 2, generator=g, dtype=torch.float64))[0] * torch.tensor(
        [2.0, 0.5]
    )
    M2 = ft.k_block((B.double() @ Q).float(), (torch.linalg.inv(Q) @ A.double()).float(), fG, fA)
    assert torch.allclose(M, M2, atol=1e-5)


def test_probe_columns_against_brute_force():
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randn(n, 5, generator=g).to(torch.bfloat16) for n in (3, 7, 2)]
    D = torch.randn(4, 5, generator=g)
    xbar, Xw = ft.probe_panel(prompts)
    rm, rr = ft.probe_columns(D, xbar, Xw)
    means = sum((D.double() @ p.double().T).mean(1) for p in prompts) / 3
    sq = sum(((D.double() @ p.double().T) ** 2).mean(1) for p in prompts) / 3
    assert rm == pytest.approx(math.sqrt(float((means**2).mean())), rel=1e-6)
    assert rr == pytest.approx(math.sqrt(float(sq.mean())), rel=1e-5)


def _frames(out_dim, in_dim, seed=0):
    return ft.k_frame(_spd(out_dim, seed)[0]), ft.k_frame(_spd(in_dim, seed + 1)[0])


def test_tensor_token_layout():
    g = torch.Generator().manual_seed(8)
    D = torch.randn(6, 5, generator=g)
    prompts = [torch.randn(4, 5, generator=g) for _ in range(2)]
    probe = ft.probe_panel(prompts)
    tok, info = ft.tensor_token(
        D, layer=12, full_attention=True, r=1, frames=_frames(6, 5), field=(0.25, 0.5), probe=probe
    )
    assert tok.shape == (42,)
    assert np.any(tok[:16] != 0)
    assert tok[33] == 0.5 and tok[34] == 1.0
    assert list(tok[35:39]) == [0.25, 0.5, 0.0, 1.0]
    assert tok[41] == 1.0 and tok[39] > 0 and tok[40] >= tok[39]
    assert info["r_eff"] == 1 and info["omitted_energy"] > 0
    tok2, _ = ft.tensor_token(
        D, layer=0, full_attention=False, r=1, frames=_frames(6, 5), field=(0.1, None)
    )
    assert tok2[34] == 0 and np.all(tok2[35:42] == 0)
    tied = (ft.k_frame(torch.diag(torch.tensor([3.0, 2.0, 1.0, 1.0, 1.0, 0.5]))), _frames(6, 5)[1])
    tok3, info3 = ft.tensor_token(D, layer=1, full_attention=False, r=2, frames=tied, field=None)
    assert np.all(tok3[:16] == 0) and "k_cutoff_tie" in info3["flags"]
    with pytest.raises(ft.NoKFrame):
        ft.tensor_token(D, layer=1, full_attention=False, r=1, frames=None, field=None)


@pytest.mark.parametrize(
    "edit,adapter,want",
    [
        (
            {"family": "lora_sft", "params": {"rank": 16}},
            {"r": 16, "rank_pattern": {}},
            (16, "adapter_config.r"),
        ),
        ({"family": "lora_sft", "params": {"rank": 16}}, {"r": 8, "rank_pattern": {}}, None),
        ({"family": "lora_sft", "params": {"rank": 16}}, {"r": 16, "rank_pattern": {"x": 4}}, None),
        ({"family": "lora_sft", "params": {"rank": 4}}, None, (4, "params.rank")),
        ({"family": "abliteration", "params": {}}, None, (1, "pinned:abliteration")),
        ({"family": "random_dir", "params": {"n_directions": 4}}, None, (1, "pinned:random_dir")),
        (
            {
                "family": "chain",
                "params": {"inner_family": "lora_sft", "inner_params": {"rank": 8}},
            },
            None,
            (8, "params.inner_params.rank"),
        ),
        (
            {"family": "chain", "params": {"inner_family": "abliteration"}},
            None,
            (1, "pinned:chain:abliteration"),
        ),
        ({"family": "chain", "params": {"inner_family": "task_vector"}}, None, None),
        ({"family": "task_vector", "params": {}}, None, None),
    ],
)
def test_recorded_rank(edit, adapter, want):
    r, src = ft.recorded_rank(edit, adapter)
    if want is None:
        assert r is None and src.startswith("unknown_rank")
    else:
        assert (r, src) == want


def test_matrix_slots_and_layer_types():
    types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
    slots = ft.matrix_slots(types)
    assert len(slots) == 186
    assert [s[0] for s in slots] == list(range(186))
    assert slots[0][3].startswith("model.layers.0.")
    assert {s[2] for s in slots if s[1] == 3} == {
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    }
    roles = {}
    for i, t in enumerate(types):
        roles[f"model.layers.{i}.mlp.down_proj"] = "mlp:mlp.down_proj"
        kind = (
            "full_attention:self_attn.q_proj"
            if t == "full_attention"
            else "linear_attention:linear_attn.out_proj"
        )
        roles[f"model.layers.{i}.x"] = kind
    assert ft.layer_types_from_coverage({"module_roles": roles}) == types
    with pytest.raises(ValueError, match="cover"):
        ft.layer_types_from_coverage({"module_roles": {"model.layers.0.x": "full_attention:a"}})


def test_row_tokens_presence_and_exclusion():
    g = torch.Generator().manual_seed(9)
    names = {
        "model.layers.0.mlp.down_proj.weight": (0, 0, "mlp.down_proj"),
        "model.layers.1.mlp.up_proj.weight": (5, 1, "mlp.up_proj"),
    }
    D = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(6, 5, generator=g),
        "model.layers.1.mlp.up_proj.weight": torch.zeros(5, 6),
        "model.layers.0.input_layernorm.weight": torch.ones(6),
    }
    edited = sorted(D)
    per = {k: {"signed": 0.1, "q": 0.2} for k in D}
    frames = {"model.layers.0.mlp.down_proj.weight": _frames(6, 5)}
    probe = {0: ft.probe_panel([torch.randn(3, 5, generator=g)])}
    toks, _info, unsupported, reason = ft.row_tokens(
        D, edited, names, ["linear_attention", "full_attention"], 1, frames, per, probe
    )
    assert list(toks) == [0] and reason is None
    assert unsupported == ["model.layers.0.input_layernorm.weight"]
    assert toks[0][41] == 1.0
    D["model.layers.1.mlp.up_proj.weight"] = torch.randn(5, 6, generator=g)
    toks, _info, _, reason = ft.row_tokens(
        D, edited, names, ["linear_attention", "full_attention"], 1, frames, per, probe
    )
    assert reason == "excluded:no_k_frame:model.layers.1.mlp.up_proj.weight" and list(toks) == [0]
    _, _, _, reason = ft.row_tokens(
        D, edited, names, ["linear_attention", "full_attention"], None, frames, per, probe
    )
    assert reason == "excluded:unknown_rank"


def test_assemble_and_projection():
    toks = {"b": {"3": [1.0] * 42}, "a": {"0": [2.0] * 42, "1": [0.5] * 42}}
    X, P, E = ft.assemble(["a", "b", "c"], toks, 4, {"a": True, "b": False})
    assert X.shape == (3, 4, 42) and X.dtype == np.float32
    assert P.tolist() == [[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]] and E.tolist() == [1, 0, 0]
    assert X[0, 1, 0] == 0.5 and X[2].sum() == 0
    naive, weighted = ft.projection(100.0, 10.0, 5, [5, 10, 1])
    assert naive == 130.0 and weighted == 100.0 + 2.0 * 16


def test_field_contraction_matches_part_a():
    g = torch.Generator().manual_seed(10)
    shapes = {"m.a.weight": (4, 3), "m.n.weight": (4,)}
    grads = {k: torch.randn(*s, generator=g) for k, s in shapes.items()}
    cov = {
        "m.a.weight": {"shape": [4, 3], "factors": True, "module": "m.a"},
        "m.n.weight": {"shape": [4], "factors": False, "module": None},
    }
    raw = {"m.a": {"A": _spd(3, 1)[0], "G": _spd(4, 2)[0], "bias": None}}
    inc = {k: torch.randn(*s, generator=g) for k, s in shapes.items()}
    row, detail, per = ft.field_contraction(grads, cov, pap.FactorCache(cov, raw), inc)
    ref, ref_detail = pap.field_row(grads, cov, pap.FactorCache(cov, raw), inc)
    assert row == ref and detail == ref_detail
    assert row["q_partial"] is True and set(per) == set(shapes)


def test_load_probe_hash_checked(tmp_path):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(11)
    rows, files = [], {}
    for i, n in enumerate((3, 5)):
        t = {f"layer.{k}": torch.randn(n, 4, generator=g).to(torch.bfloat16) for k in range(2)}
        save_file(t, str(tmp_path / f"{i:03d}.safetensors"))
        rows.append(
            {
                "index": i,
                "file": f"{i:03d}.safetensors",
                "n_tokens": n,
                "content_hashes": {k: rf._tensor_hash(v) for k, v in t.items()},
            }
        )
        files[f"{i:03d}.safetensors"] = pf.sha256_file(tmp_path / f"{i:03d}.safetensors")
    (tmp_path / "probe_cache.json").write_text(json.dumps(rows))
    files["probe_cache.json"] = pf.sha256_file(tmp_path / "probe_cache.json")
    man = {
        "probe_cache": {
            "status": "complete",
            "files": files,
            "content_hash": rf._hash([r["content_hashes"] for r in rows]),
            "hooks": {"role": "x"},
            "predecessor": {"id": "base"},
        }
    }
    probe, ident = ft.load_probe(man, tmp_path, range(2))
    assert set(probe) == {0, 1} and ident["n_tokens"] == 8 and ident["n_prompts"] == 2
    assert probe[0][1].shape == (8, 4)
    man["probe_cache"]["files"]["000.safetensors"] = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(ValueError, match="sha256"):
        ft.load_probe(man, tmp_path, range(2))


def test_out_dir_guard(tmp_path):
    ft.check_out_dir(ft.ALLOWED_OUT / "tokens")
    with pytest.raises(ValueError):
        ft.check_out_dir(tmp_path)


def test_stored_tensor_count(tmp_path):
    from safetensors.torch import save_file

    (tmp_path / "d").mkdir()
    save_file({"a": torch.zeros(2), "b": torch.zeros(3)}, str(tmp_path / "d/delta.safetensors"))
    assert ft.stored_tensor_count(tmp_path / "d") == 2
    (tmp_path / "l/lora").mkdir(parents=True)
    save_file(
        {"x.lora_A.weight": torch.zeros(2, 3), "x.lora_B.weight": torch.zeros(3, 2)},
        str(tmp_path / "l/lora/adapter_model.safetensors"),
    )
    assert ft.stored_tensor_count(tmp_path / "l") == 1
