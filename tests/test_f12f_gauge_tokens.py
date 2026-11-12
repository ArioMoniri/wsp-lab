"""F12f step 1: the registered gauge-transformed tokens (scripts/f12_gauge_tokens.py; chair ruling on F12e (a); F5 schema
pins "Gauge audit"). Synthetic tensors only: nothing under data/ and no real token file is opened."""

import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import f12_gauge_tokens as fg
from scripts import f12_simulation as fs
from scripts import f12_tokens as ft
from scripts.a5_build_panels import domain_seed


def _spd(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    lam = torch.linspace(5.0, 0.1, n, dtype=torch.float64)
    return ((q * lam) @ q.T).float()


def _frames(out_dim, in_dim, seed=1):
    return ft.k_frame(_spd(out_dim, seed)), ft.k_frame(_spd(in_dim, seed + 1))


def _increment(out_dim, in_dim, rank, seed=3):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(out_dim, rank, generator=g) @ torch.randn(rank, in_dim, generator=g)
    return (D + 1e-4 * torch.randn(out_dim, in_dim, generator=g)).float()


def _frozen_token(D, r, frames, layer=3):
    tok, info = ft.tensor_token(
        D, layer=layer, full_attention=True, r=r, frames=frames, field=(0.1, 0.01)
    )
    return tok.astype(np.float32), info


# ------------------------------------------------------------------------------------------------- Q


def test_gauge_seed_is_the_control_seeds_rule_on_edit_and_tensor():
    assert fg.GAUGE_DOMAIN == "20261110:lol-gauge:"
    name = "model.layers.3.mlp.down_proj.weight"
    assert fg.gauge_seed("abc123", name) == domain_seed("20261110:lol-gauge:", f"abc123:{name}")
    assert fg.gauge_seed("abc123", name) != fg.gauge_seed("abc124", name)


def test_gauge_diagonal_pattern_and_condition_bound():
    # d_j = 2^((j mod 3) - 1) for j = 1..r (the sigma_j convention of the same pins)
    assert fg.gauge_d(6).tolist() == [1.0, 2.0, 0.5, 1.0, 2.0, 0.5]
    for r in (1, 2, 3, 8, 16, 32):
        g = fg.gauge_matrix("e", "t", r)
        assert g["cond"] <= 4.0 + 1e-9
        assert g["Q"].dtype == torch.float64 and g["Q"].shape == (r, r)


def test_gauge_matrix_is_haar_qr_with_positive_diagonal_times_d():
    r, eid, name = 5, "0123abcd", "model.layers.0.mlp.up_proj.weight"
    g = fg.gauge_matrix(eid, name, r)
    gen = torch.Generator("cpu").manual_seed(domain_seed(fg.GAUGE_DOMAIN, f"{eid}:{name}"))
    Z = torch.randn(r, r, generator=gen, dtype=torch.float64)
    H, R = torch.linalg.qr(Z)
    s = torch.sign(torch.diagonal(R))
    s[s == 0] = 1.0
    H = H * s[None, :]
    assert torch.equal(g["H"], H)
    assert torch.allclose(H.T @ H, torch.eye(r, dtype=torch.float64), atol=1e-12)
    assert torch.all(torch.diagonal(H.T @ Z) > 0)  # R = H^T Z has a positive diagonal
    assert torch.equal(g["Q"], H * fg.gauge_d(r)[None, :])
    assert torch.allclose(g["Q_inv"] @ g["Q"], torch.eye(r, dtype=torch.float64), atol=1e-12)
    again = fg.gauge_matrix(eid, name, r)
    assert torch.equal(again["Q"], g["Q"]) and again["sha256"] == g["sha256"]


def test_rank_one_gauge_is_the_identity():
    g = fg.gauge_matrix("e", "t", 1)
    assert g["Q"].tolist() == [[1.0]] and g["Q_inv"].tolist() == [[1.0]]


# --------------------------------------------------------------------------------------------- tokens


def test_gauge_token_changes_only_the_k_block_within_fp32_rounding():
    D = _increment(12, 9, 3)
    frames = _frames(12, 9)
    frozen, _info = _frozen_token(D, 3, frames)
    tok, ti = fg.tensor_gauge(D, frozen, 3, frames, "edit-1", "model.layers.3.mlp.down_proj.weight")
    assert tok.dtype == np.float32 and tok.shape == (42,)
    assert np.array_equal(tok[16:], frozen[16:])  # columns 16-41 are functions of D_t alone
    assert ti["k_reproduced_exact"] is True and ti["k_reproduction_max_abs_diff"] == 0.0
    assert ti["r_eff"] == 3 and ti["cond"] <= 4.0 + 1e-9
    # (BQ)(Q^-1 A) = BA: the K block moves only by rounding
    assert np.allclose(tok[:16], frozen[:16], rtol=1e-4, atol=1e-7)
    assert ti["k_block_max_abs_change"] == float(
        np.max(np.abs(tok[:16].astype(np.float64) - frozen[:16]))
    )
    # the transformed factors are the fp64 products cast to fp32
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    B, A, _ = ft.balanced_factors(U, S, Vh, 3)
    g = fg.gauge_matrix("edit-1", "model.layers.3.mlp.down_proj.weight", 3)
    Bg, Ag = fg.gauge_factors(B, A, g)
    assert torch.equal(Bg, (B.double() @ g["Q"]).float()) and torch.equal(
        Ag, (g["Q_inv"] @ A.double()).float()
    )
    expect = ft.k_block(Bg, Ag, *frames).reshape(-1).numpy().astype(np.float32)
    assert np.array_equal(tok[:16], expect)


def test_rank_one_gauge_token_is_bitwise_the_frozen_token():
    D = _increment(10, 7, 1, seed=8)
    frames = _frames(10, 7, seed=9)
    frozen, _ = _frozen_token(D, 1, frames)
    tok, ti = fg.tensor_gauge(D, frozen, 1, frames, "e", "model.layers.1.mlp.gate_proj.weight")
    assert np.array_equal(tok, frozen) and ti["k_block_max_abs_change"] == 0.0


def test_masked_k_block_stays_zero():
    D = _increment(8, 6, 2)
    fG, fA = _frames(8, 6)
    fG = dict(fG, flags=["k_cutoff_tie"])
    frozen, info = _frozen_token(D, 2, (fG, fA))
    assert "k_cutoff_tie" in info["flags"] and not frozen[:16].any()
    tok, ti = fg.tensor_gauge(D, frozen, 2, (fG, fA), "e", "t")
    assert not tok[:16].any() and np.array_equal(tok, frozen) and ti["masked"] is True


def test_frozen_k_block_that_does_not_reproduce_fails_closed():
    D = _increment(12, 9, 3)
    frames = _frames(12, 9)
    frozen, _ = _frozen_token(D, 3, frames)
    # bitwise: even one fp32 ulp is a different source (near-tied truncations amplify thread-order rounding)
    assert fg.REPRO_RTOL == 0.0 and fg.FROZEN_THREADS == 6
    j = int(np.argmax(np.abs(frozen[:16])))
    bad = frozen.copy()
    bad[j] = np.nextafter(bad[j], np.float32(np.inf))
    with pytest.raises(fg.SourceMismatch, match="K block"):
        fg.tensor_gauge(D, bad, 3, frames, "e", "t")
    masked_but_nonzero = frozen.copy()
    fG, fA = frames
    with pytest.raises(fg.SourceMismatch, match="masked"):
        fg.tensor_gauge(D, masked_but_nonzero, 3, (dict(fG, flags=["k_cutoff_tie"]), fA), "e", "t")


def test_row_gauge_checks_presence_rank_and_increment(monkeypatch):
    slots = [(0, fs.SLOTS[0][2], (12, 9)), (5, fs.SLOTS[5][2], (12, 9))]
    D = {name: _increment(*shape, 2, seed=10 + i) for i, (_, name, shape) in enumerate(slots)}
    frames = {name: _frames(*shape, seed=20 + i) for i, (_, name, shape) in enumerate(slots)}
    X = np.zeros((fs.N_SLOTS, 42), dtype=np.float32)
    P = np.zeros(fs.N_SLOTS, dtype=np.uint8)
    per = {}
    for s, name, _ in slots:
        X[s], info = _frozen_token(D[name], 2, frames[name])
        P[s] = 1
        per[name] = {"r_eff": info["r_eff"], "slot": s}
    names = [x[2] for x in fs.SLOTS]
    Xg, info = fg.row_gauge(D, "row-a", X, P, names, 2, frames, per)
    assert Xg.shape == X.shape and not Xg[P == 0].any()
    assert set(info) == {name for _, name, _ in slots}
    assert np.array_equal(Xg[:, 16:], X[:, 16:])
    # a present slot whose increment is zero, or a changed supported tensor without a token, fails closed
    D0 = dict(D)
    D0[slots[0][1]] = torch.zeros_like(D[slots[0][1]])
    with pytest.raises(fg.SourceMismatch, match="presence"):
        fg.row_gauge(D0, "row-a", X, P, names, 2, frames, per)
    D1 = dict(D)
    extra = fs.SLOTS[7][2]
    D1[extra] = torch.ones(3, 3)
    with pytest.raises(fg.SourceMismatch, match="presence"):
        fg.row_gauge(D1, "row-a", X, P, names, 2, frames, per)
    # the frozen r_eff must equal min(r, dims)
    per_bad = {k: dict(v, r_eff=1) for k, v in per.items()}
    with pytest.raises(fg.SourceMismatch, match="r_eff"):
        fg.row_gauge(D, "row-a", X, P, names, 2, frames, per_bad)


def test_increment_hash_must_equal_the_frozen_rows():
    D = {"model.layers.0.mlp.up_proj.weight": torch.ones(2, 2)}
    from scripts import a5_power_fields as pf

    fg.check_increment("x", D, pf.tensors_hash(D))
    with pytest.raises(fg.SourceMismatch, match="increment"):
        fg.check_increment("x", D, "0" * 64)


# -------------------------------------------------------------------------------------------- pricing


def test_pricing_projects_from_the_first_rows_and_stops_over_the_cap():
    p = fg.pricing(setup_seconds=100.0, first_seconds=[30.0, 10.0], first_tensors=[20, 10],
                   tensor_counts=[20, 10, 30, 40], cap_hours=10.0)  # fmt: skip
    # naive: setup + mean row seconds x rows; weighted: setup + seconds per tensor x all tensors
    assert p["projected_wall_seconds_naive"] == pytest.approx(100 + 20.0 * 4)
    assert p["projected_wall_seconds_tensor_weighted"] == pytest.approx(100 + 40.0 / 30 * 100)
    assert p["projected_wall_seconds"] == max(p["projected_wall_seconds_naive"],
                                               p["projected_wall_seconds_tensor_weighted"])  # fmt: skip
    assert p["within_cap"] is True and p["cap_wall_hours"] == 10.0
    big = fg.pricing(setup_seconds=0.0, first_seconds=[3600.0], first_tensors=[1], tensor_counts=[1] * 11,
                     cap_hours=10.0)  # fmt: skip
    assert big["within_cap"] is False


def test_outputs_stay_under_the_inputs_directory(tmp_path):
    with pytest.raises(ValueError, match="outputs must stay under"):
        fg.check_out_dir(tmp_path)
    fg.check_out_dir(fg.ALLOWED_OUT / "gauge_tokens")


def test_frames_loader_reads_kframes_with_flags(tmp_path):
    from safetensors.torch import save_file

    fG, fA = _frames(6, 5)
    name = "model.layers.0.mlp.up_proj.weight"
    save_file({f"{name}.G.U": fG["U"].contiguous(), f"{name}.G.sqrt_lambda": fG["sqrt_lambda"].contiguous(),
               f"{name}.A.U": fA["U"].contiguous(), f"{name}.A.sqrt_lambda": fA["sqrt_lambda"].contiguous()},
              str(tmp_path / "kframes.safetensors"))  # fmt: skip
    doc = {"tensors": {name: {"G": {"flags": ["k_inner_tie"]}, "A": {"flags": []}}}}
    frames = fg.load_frames(tmp_path / "kframes.safetensors", doc)
    gG, gA = frames[name]
    assert torch.equal(gG["U"], fG["U"]) and torch.equal(gA["sqrt_lambda"], fA["sqrt_lambda"])
    assert gG["flags"] == ["k_inner_tie"] and gA["flags"] == []
    doc["tensors"]["model.layers.1.mlp.up_proj.weight"] = {"G": {"flags": []}, "A": {"flags": []}}
    with pytest.raises(fg.SourceMismatch, match="frame"):
        fg.load_frames(tmp_path / "kframes.safetensors", doc)


def test_manifest_records_every_output_by_sha256(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "gauge_tokens.safetensors").write_bytes(b"abc")
    (out / "rows.jsonl").write_text("{}\n")
    files = fg.output_hashes(out)
    assert files["gauge_tokens.safetensors"] == hashlib.sha256(b"abc").hexdigest()
    assert files["rows.jsonl"] == hashlib.sha256(b"{}\n").hexdigest()
    assert json.dumps(files, sort_keys=True)
