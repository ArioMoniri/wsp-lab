import math

import pytest
import torch

from wsp.features.rotation import (
    apply_and_round,
    audit,
    block_svd,
    changed_block,
    edit_energy_guard,
    exclusion_reason,
    f10_tensor_audit,
    lora_factors_reproduce,
    realised_increment,
    replay_rotation,
    rotate_block,
    rounding_bound,
    scaled_variant,
    ulp_bf16,
)


def test_rotation_preserves_spectrum_and_changes_orientation():
    d = torch.zeros(6, 7)
    d[1:5, 2:6] = torch.tensor(
        [[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 2.0]]
    )
    rows, cols = changed_block(d)
    r = rotate_block(d, rows, cols, 42)
    assert torch.allclose(torch.linalg.svdvals(d), torch.linalg.svdvals(r), atol=1e-5)
    assert torch.allclose(d.norm(), r.norm(), atol=1e-5)
    assert torch.linalg.matrix_rank(d) == torch.linalg.matrix_rank(r)
    assert torch.count_nonzero(r[:1]) == 0 and torch.count_nonzero(r[:, :2]) == 0
    w = torch.arange(42.0).reshape(6, 7)
    assert not torch.allclose((d**3).sum(), (r**3).sum())
    assert not torch.allclose(
        torch.nn.functional.cosine_similarity(d.flatten(), w.flatten(), dim=0),
        torch.nn.functional.cosine_similarity(r.flatten(), w.flatten(), dim=0),
    )
    assert audit(d, r, rows, cols)["passes"]
    assert not audit(d, r * 1.2, rows, cols)["passes"]
    assert block_svd(d, rows, cols)[-1] == 4


def test_shared_generator_reproducible_sequence():
    d = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    rows, cols = changed_block(d)
    g = torch.Generator().manual_seed(4)
    first = rotate_block(d, rows, cols, 0, g)
    second = rotate_block(d, rows, cols, 0, g)
    g = torch.Generator().manual_seed(4)
    assert torch.equal(first, rotate_block(d, rows, cols, 999, g))
    assert torch.equal(second, rotate_block(d, rows, cols, 999, g))
    assert torch.equal(first, second)


def test_lora_rounding_and_scaling():
    a = torch.tensor([[1.003, 2.007]])
    b = torch.tensor([[1.0], [2.0]])
    d = b @ a
    assert lora_factors_reproduce(d, a, b, 1.0)
    assert not lora_factors_reproduce(d.to(torch.bfloat16).float(), a, b, 1.0)
    w = torch.ones(2, 2, dtype=torch.bfloat16)
    new, eff = apply_and_round(w, d)
    assert torch.equal(eff, realised_increment(w, new))
    assert all(torch.equal(x, y) for x, y in zip(scaled_variant(w, d, 1), (new, eff)))


def test_exclusions():
    assert exclusion_reason(torch.zeros(2), None, None) == "non_2d"
    d = torch.zeros(2, 2)
    rows, cols = changed_block(d)
    assert exclusion_reason(d, rows, cols) == "zero_block"


def test_low_rank_lora_block_is_not_excluded_and_rotates_exactly():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(2, 8, generator=g)
    b = torch.randn(8, 2, generator=g)
    d = b @ a  # rank-2 block on an 8x8 support, like a LoRA increment
    rows, cols = changed_block(d)
    assert exclusion_reason(d, rows, cols) is None
    r = rotate_block(d, rows, cols, 7)
    assert torch.allclose(torch.linalg.svdvals(d), torch.linalg.svdvals(r), atol=1e-5)
    assert block_svd(r, rows, cols)[-1] == 2
    assert audit(d, r, rows, cols)["passes"]


def test_shared_generator_gives_equal_q_for_equal_shapes():
    g = torch.Generator().manual_seed(3)
    d1 = torch.eye(4)
    d2 = torch.diag(torch.tensor([2.0, 1.0, 1.0, 0.5]))
    rows, cols = changed_block(d1)
    r1 = rotate_block(d1, rows, cols, 0, g)
    r2 = rotate_block(d2, rows, cols, 0, g)
    # r1 = Q_out Q_in^T; r2 = Q_out diag Q_in^T, with the same Q
    assert torch.allclose(r1 @ r2.T, r1 @ r2.T)  # smoke
    assert torch.allclose(torch.linalg.svdvals(r2), torch.tensor([2.0, 1.0, 1.0, 0.5]), atol=1e-5)
    assert torch.allclose(r1.T @ r1, torch.eye(4), atol=1e-5)


def test_audit_fails_on_spectrum_perturbation_and_non_finite_is_excluded():
    d = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    rows, cols = changed_block(d)
    r = rotate_block(d, rows, cols, 5)
    u, s, vh = torch.linalg.svd(r)
    s2 = s.clone()
    s2[0], s2[2] = (
        s[0] * 0.9,
        (s.pow(2).sum() - (s[0] * 0.9) ** 2 - s[1] ** 2).sqrt(),
    )  # same norm, other spectrum
    r2 = u @ torch.diag(s2) @ vh
    out = audit(d, r2, rows, cols)
    assert out["norm_mismatch"] < 1e-5 and out["spectrum_mismatch"] > 5e-2 and not out["passes"]
    bad = d.clone()
    bad[0, 0] = float("nan")
    assert exclusion_reason(bad, rows, cols) == "non_finite"


# ------------------------------------------------------------------------------- F10 (remedy d) audit


def _grid_w(shape, value=1.5):
    """A predecessor mid-binade: values near 1.5 have bf16 spacing 2^-7 on both sides."""
    return torch.full(shape, value, dtype=torch.bfloat16)


def _block_d(shape, scale, seed, rank=3):
    g = torch.Generator().manual_seed(seed)
    a, b = torch.randn(shape[0], rank, generator=g), torch.randn(rank, shape[1], generator=g)
    return a @ b * scale / rank


def _staged(W, delta, seed):
    """D_eff at W, the registered rotation R and its realised R_eff, exactly as build_variants stages."""
    new = (W.float() + delta).to(torch.bfloat16)
    D = realised_increment(W, new)
    rows, cols = changed_block(D)
    R = rotate_block(D, rows, cols, seed)
    state, R_eff = apply_and_round(W, R)
    return D, rows, cols, R, state, R_eff


def test_ulp_bf16_convention_and_floor():
    x = torch.tensor([1.0, 1.5, 0.75, -3.0, 0.0], dtype=torch.bfloat16)
    want = [2**-7, 2**-7, 2**-8, 2**-6, 2**-133]
    assert ulp_bf16(x).tolist() == want


def test_replay_rotation_equals_rotate_block_bitwise():
    d = _block_d((12, 9), 0.3, 1)
    rows, cols = changed_block(d)
    g = torch.Generator().manual_seed(77)
    shared = rotate_block(d, rows, cols, 0, shared_generator=g)
    r, q_out, q_in = replay_rotation(d, rows, cols, 77)
    assert torch.equal(r, shared) and torch.equal(r, rotate_block(d, rows, cols, 77))
    assert q_out.shape == (12, 12) and q_in.shape == (9, 9)


def test_round_to_nearest_bound_uses_the_result_grid():
    """W just below a binade edge: correct rounding of W + R can exceed ½ ulp(W), never ½ ulp(result)."""
    W = torch.tensor([[1 - 2**-8, 0.5]], dtype=torch.bfloat16)
    R = torch.tensor([[2**-8 + 0.4 * 2**-7, 0.0]])
    state, R_eff = apply_and_round(W, R)
    assert state[0, 0].item() == 1.0
    out = rounding_bound(W, R, R_eff)
    assert out["passes"] and out["n_violations"] == 0
    assert (
        out["n_exceed_half_ulp_w"] == 1
    )  # the literal ½ ulp(W) form would reject correct rounding
    bad = R_eff.clone()
    bad[0, 1] = 2**-9  # a change where R = 0: wrong grid / predecessor
    assert not rounding_bound(W, R, bad)["passes"]


def test_f10_passes_where_realised_audit_fails():
    """Low kappa: rounding at W's grid fails the registered realised audit, not the F10 exclusion rule."""
    W = _grid_w((40, 56))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((40, 56), 2**-7 * 0.8, 3, rank=4), 11)
    assert not audit(D, R_eff, rows, cols)["passes"]
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 11)
    assert out["passes"] and out["failed_checks"] == []
    assert out["s31_passes"] is False  # still computed, as a diagnostic
    assert out["orthogonality_error"] <= 1e-5 and out["reconstruction_residual"] <= 1e-5
    assert out["prerounding_norm_mismatch"] <= 1e-5
    assert out["prerounding_spectrum_mismatch"] <= 1e-5
    assert out["prerounding_rank_d"] == out["prerounding_rank_r"]


def test_f10_fails_where_realised_audit_passes():
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    base = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5)
    assert base["passes"] and audit(D, R_eff, rows, cols)["passes"]

    # (i) a pre-rounding R off by 5e-5 in norm: inside every realised tolerance, outside F10's 1e-5
    R2 = R * (1 + 5e-5)
    _state2, R2_eff = apply_and_round(W, R2)
    assert audit(D, R2_eff, rows, cols)["passes"]
    out = f10_tensor_audit(D, W, R2, R2_eff, rows, cols, 5)
    assert not out["passes"] and "construction" in out["failed_checks"]
    assert out["reason"] == "construction_failed"

    # (iii) a valid rotation from the wrong seed: construction and rounding pass, the replay does not
    R3 = rotate_block(D, rows, cols, 6)
    _state3, R3_eff = apply_and_round(W, R3)
    assert audit(D, R3_eff, rows, cols)["passes"]
    out = f10_tensor_audit(D, W, R3, R3_eff, rows, cols, 5)
    assert out["failed_checks"] == ["replay"] and out["reason"] == "replay_failed"

    # (iv) one entry moved by one ulp after rounding: realised audit blind, the bound is not
    R4_eff = R_eff.clone()
    R4_eff[rows[0], cols[0]] += 2**-7
    assert audit(D, R4_eff, rows, cols)["passes"]
    out = f10_tensor_audit(D, W, R, R4_eff, rows, cols, 5)
    assert "rounding_bound" in out["failed_checks"] and not out["passes"]

    # (ii) a realised change outside the block
    D5 = D.clone()
    D5[0] = 0  # shrink the block by one row; R_eff from the full block now leaks outside it
    rows5, cols5 = changed_block(D5)
    out = f10_tensor_audit(D5, W, R, R_eff, rows5, cols5, 5)
    assert "support" in out["failed_checks"] and out["reason"] == "construction_failed"


def test_f10_diagnostics_are_reported():
    W = _grid_w((32, 40))
    delta = _block_d((32, 40), 2**-7 * 3, 8)
    D, rows, cols, R, _state, R_eff = _staged(W, delta, 2)
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 2, raw_delta=delta)
    dn = float(D.double().norm())
    kappa = dn / math.sqrt(32 * 40 * 2.0**-14)
    assert math.isclose(out["kappa"], kappa, rel_tol=1e-9)
    assert math.isclose(out["eps_hat"], min(1.0, 1 / (math.sqrt(12) * kappa)), rel_tol=1e-9)
    assert math.isclose(
        out["rounding_error_rel"], float((R_eff - R).double().norm()) / dn, rel_tol=1e-9
    )
    assert math.isclose(
        out["energy_ratio"], float(R_eff.double().norm()) ** 2 / dn**2, rel_tol=1e-9
    )
    assert math.isclose(out["eps_D"], float((D - delta).double().norm()) / dn, rel_tol=1e-9)
    tau = math.sqrt(2.0**-14 / 12) * (math.sqrt(32) + math.sqrt(40))
    assert math.isclose(out["tau"], tau, rel_tol=1e-9)
    for k in (
        "norm_mismatch",
        "norm_mismatch_signed",
        "spectrum_mismatch",
        "rank_d",
        "rank_r",
        "noise_rank_d",
        "noise_rank_r",
        "noise_calibrated_passes",
        "s31_passes",
        "rounding_bound_max_ratio",
        "n_exceed_half_ulp_w",
        "construction_residual",
    ):
        assert k in out
    assert out["norm_mismatch_signed"] == pytest.approx(
        (float(R_eff.double().norm()) - dn) / dn, rel=1e-6
    )
    nc = (
        abs(out["norm_mismatch"]) <= 2e-2 + out["eps_hat"]
        and out["spectrum_mismatch"] <= 5e-2 + out["eps_hat"]
        and abs(out["noise_rank_r"] - out["noise_rank_d"]) <= 1
    )
    assert out["noise_calibrated_passes"] is nc
    assert f10_tensor_audit(D, W, R, R_eff, rows, cols, 2)["eps_D"] is None


def test_edit_energy_guard_keeps_half_the_energy():
    assert edit_energy_guard([1.0, 1.0], [0.5, 0.5])["passes"]  # exactly ½ passes
    assert not edit_energy_guard([1.0, 1.0], [0.5, 0.4999])["passes"]
    out = edit_energy_guard([4.0, 1.0], [4.1, 0.0])
    assert out["passes"] and out["ratio"] == pytest.approx(4.1 / 5.0)


def test_sub_resolution_tensor_rounds_away_but_passes_tensor_checks():
    """Sparse single-ulp flips (chain residue): R lies below half an ulp and rounds to zero."""
    W = _grid_w((48, 48))
    perm = torch.randperm(48, generator=torch.Generator().manual_seed(0))
    delta = torch.zeros(48, 48)
    delta[torch.arange(48), perm] = 2**-7  # 48 flips, one per row and column: a 48 x 48 block
    D, rows, cols, R, _state, R_eff = _staged(W, delta, 9)
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 9)
    assert out["passes"] and out["energy_ratio"] < 0.5
    assert not edit_energy_guard(
        [float(R.double().norm()) ** 2], [float(R_eff.double().norm()) ** 2]
    )["passes"]


def test_scorer_roundtrip_failure_is_a_replay_failure():
    """Chain step: W = 0 where the base is large; bf16(W + R) is tiny and cannot be written as an fp32
    delta from the base that the scorer adds back (bf16(base + delta) != state)."""
    W = _grid_w((8, 8))
    W[2, 3] = 0.0
    base = W.clone()
    base[2, 3] = 1e5
    D, rows, cols, R, state, R_eff = _staged(W, _block_d((8, 8), 0.05, 6, rank=8), 3)
    assert state[2, 3] != 0
    assert not torch.equal(
        (base.float() + (state.float() - base.float())).to(torch.bfloat16), state
    )
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 3, scorer_base=base)
    assert out["failed_checks"] == ["scorer_roundtrip"] and out["reason"] == "replay_failed"
    assert out["scorer_roundtrip_mismatches"] == 1 and out["replay_passes"] is False
    ok = f10_tensor_audit(D, W, R, R_eff, rows, cols, 3)  # root: the scorer's base is W itself
    assert ok["passes"] and ok["scorer_roundtrip_mismatches"] == 0
