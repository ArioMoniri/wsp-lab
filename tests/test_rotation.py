import itertools
import json
import math

import pytest
import torch

from wsp.features import rotation as rot
from wsp.features.rotation import (
    apply_and_round,
    audit,
    bits_equal,
    block_svd,
    changed_block,
    construction_items,
    exclusion_reason,
    f10_replay,
    f10_tensor_audit,
    f10_verdict,
    haar_orthogonal,
    lora_factors_reproduce,
    noise_calibrated,
    rank_budget,
    realised_increment,
    replay_rotation,
    rotate_block,
    rotation_frames,
    rounding_bound,
    scaled_variant,
    scorer_roundtrip,
    sub_resolution_guard,
    torch_threads,
    ulp_bf16,
    ulp_fp32,
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
    m = r1.T @ r2  # = Q_in diag Q_in^T when both used the same Q_out and Q_in
    assert torch.allclose(m, m.T, atol=1e-5)
    assert torch.allclose(torch.linalg.eigvalsh(m), torch.tensor([0.5, 1.0, 1.0, 2.0]), atol=1e-5)
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


def _spectral(sv, seed, shape=None):
    """fp32 matrix U diag(sv) V^T with seeded orthogonal factors."""
    g = torch.Generator().manual_seed(seed)
    m, n = shape or (len(sv), len(sv))
    u = haar_orthogonal(m, g)[:, : len(sv)]
    v = haar_orthogonal(n, g)[:, : len(sv)]
    return (u.double() @ torch.diag(torch.tensor(sv, dtype=torch.float64)) @ v.double().T).float()


def _rotated_with(D, rows, cols, seed, block):
    """R built with the registered frames of ``seed`` around a substituted block (a tampered spectrum)."""
    q_out, q_in = rotation_frames(len(rows), len(cols), seed)
    R = torch.zeros_like(D)
    R[rows[:, None], cols] = q_out @ block @ q_in.T
    return R


def test_ulp_bf16_and_fp32_conventions_and_floors():
    x = torch.tensor([1.0, 1.5, 0.75, -3.0, 0.0, -0.0, 2.0**-126, 2.0**-130], dtype=torch.float64)
    want = [2**-7, 2**-7, 2**-8, 2**-6, 2**-133, 2**-133, 2**-133, 2**-133]
    assert ulp_bf16(x.to(torch.bfloat16)).tolist() == want
    y = torch.tensor([1.0, -1.0, 1.5, 0.0, -0.0, 2.0**-126, 2.0**-140], dtype=torch.float32)
    assert ulp_fp32(y).tolist() == [2**-23, 2**-23, 2**-23, 2**-149, 2**-149, 2**-149, 2**-149]


# --- (i) near-cut rank budget (F10a.1)


def _rb(sd, sr):
    return rank_budget(torch.tensor(sd, dtype=torch.float64), torch.tensor(sr, dtype=torch.float64))


def test_rank_budget_cut_is_strict_and_window_inclusive():
    # sigma_1 = 1e5: cut 1e-3 sigma_1 = 100.0 and window 1e-5 sigma_1 = 1.0, both exact in fp64
    out = _rb([1e5, 100.0, 50.0], [1e5, 100.0, 50.0])
    assert (out["rank_cut_d"], out["rank_window"]) == (100.0, 1.0)
    assert out["rank_d"] == out["rank_r"] == 1  # strict: a value at the cut is not counted
    assert out["rank_n_near"] == 1 and out["passes"]
    # both window endpoints are near; one fp64 step outside either is not
    assert _rb([1e5, 101.0, 99.0, 30.0], [1e5, 30.0])["rank_n_near"] == 2
    up, down = math.nextafter(101.0, math.inf), math.nextafter(99.0, 0.0)
    assert _rb([1e5, up, down, 30.0], [1e5, 30.0])["rank_n_near"] == 0


def test_rank_budget_allows_both_directions_within_budget_and_rejects_above():
    sd = [1e5, 100.5, 99.5, 10.0]  # rank_d 2, N_near 2
    assert _rb(sd, sd)["passes"]
    down = _rb(sd, [1e5, 99.9, 99.8, 10.0])
    assert (down["rank_r"], down["rank_difference"], down["passes"]) == (1, -1, True)
    up = _rb(sd, [1e5, 100.5, 100.2, 10.0])
    assert (up["rank_r"], up["rank_difference"], up["passes"]) == (3, 1, True)
    over = _rb([1e5, 100.5, 10.0], [1e5, 300.0, 200.0, 150.0, 10.0])  # N_near 1, difference +2
    assert over["rank_n_near"] == 1 and over["rank_difference"] == 2 and not over["passes"]
    # N_near = 0: exact equality is required
    zero = [1e5, 200.0, 10.0]
    assert _rb(zero, zero)["passes"] and _rb(zero, zero)["rank_n_near"] == 0
    assert not _rb(zero, [1e5, 200.0, 150.0, 10.0])["passes"]
    assert not _rb(zero, [1e5, 10.0])["passes"]


def test_rank_budget_uses_each_matrix_own_leading_value():
    # R's sigma_1 is twice D's: its own cut is 200, so 150 no longer counts (a shared cut would pass)
    out = _rb([1e5, 150.0, 10.0], [2e5, 150.0, 10.0])
    assert out["rank_cut_r"] == 200.0 and out["rank_r"] == 1 and not out["passes"]
    # a uniformly scaled spectrum keeps its own-cut rank (a shared cut at D's 100 would drop 75)
    half = _rb([1e5, 150.0, 10.0], [5e4, 75.0, 5.0])
    assert half["rank_cut_r"] == 50.0 and half["rank_r"] == 2 and half["passes"]


def test_rank_budget_fails_closed_on_nonfinite_or_empty_spectra():
    assert not _rb([1e5, float("nan")], [1e5, 1.0])["passes"]
    assert not _rb([1e5, 1.0], [float("inf"), 1.0])["passes"]
    assert not rank_budget(torch.zeros(0), torch.zeros(0))["passes"]


def _rank_case(sv_d, sv_r, seed=21):
    W = _grid_w((40, 40))
    D = _spectral(sv_d, 4, (40, 40))
    rows, cols = changed_block(D)
    R = _rotated_with(D, rows, cols, seed, _spectral(sv_r, 4, (40, 40)))
    _state, R_eff = apply_and_round(W, R)
    return f10_tensor_audit(D, W, R, R_eff, rows, cols, seed)


def test_f10_rank_budget_in_the_construction_check():
    ones = [1.0] * 39
    # outside the window (|1.02e-3 - 1e-3| = 2e-5 > 1e-5): N_near = 0, one value crosses -> construction_failed
    out = _rank_case(ones + [1.02e-3], ones + [0.98e-3])
    assert out["rank_n_near"] == 0 and out["prerounding_rank_difference"] == -1
    assert out["construction_items"] == {"orthogonality": True, "reconstruction": True,
                                         "norm": True, "spectrum": True, "rank_budget": False}  # fmt: skip
    assert out["reason"] == "construction_failed" and "construction" in out["failed_checks"]
    for sd, sr, diff in ((1.004e-3, 0.996e-3, -1), (0.996e-3, 1.004e-3, 1)):  # inside the window
        out = _rank_case(ones + [sd], ones + [sr])
        assert out["rank_n_near"] == 1 and out["prerounding_rank_difference"] == diff
        assert out["rank_budget_passes"] and out["construction_items"]["rank_budget"]
        assert out["checks"]["construction"]
        for k in (
            "prerounding_rank_d",
            "prerounding_rank_r",
            "rank_cut_d",
            "rank_cut_r",
            "rank_n_near",
        ):
            assert out[k] is not None


# --- (iv) rounding bound with the registered one-fp32-spacing slack


def _rbound(w, r, e):
    t = lambda x, dt: torch.tensor([x], dtype=dt)
    return rounding_bound(t(w, torch.bfloat16), t(r, torch.float32), t(e, torch.float64))


def _round_once(w, r):
    W = torch.tensor([w], dtype=torch.bfloat16)
    _state, eff = apply_and_round(W, torch.tensor([r]))
    return float(eff[0])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_round_to_nearest_bound_uses_the_result_grid(sign):
    """W just below a binade edge: correct rounding of W + R can exceed ½ ulp(W), never ½ ulp(result)."""
    w, r = sign * (1 - 2**-8), sign * (2**-8 + 0.4 * 2**-7)
    e = _round_once(w, r)
    assert w + e == sign * 1.0
    out = _rbound(w, r, e)
    assert out["passes"] and out["n_violations"] == 0
    assert (
        out["n_exceed_half_ulp_w"] == 1
    )  # the literal ½ ulp(W) form would reject correct rounding
    # downward across the edge: from 1 + 2^-7 to 1.0, and from 1.0 into the finer binade below
    for w, r in ((1 + 2**-7, -(2**-7 + 0.3 * 2**-8)), (1.0, -1.3 * 2**-8)):
        w, r = sign * w, sign * r
        out = _rbound(w, r, _round_once(w, r))
        assert out["passes"] and out["n_double_rounding"] == 0
    assert not _rbound(sign * 0.5, 0.0, sign * 2**-9)["passes"]  # a change where R = 0: wrong grid


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rounding_bound_midpoint_ties_on_both_sides(sign):
    # an exact bf16 midpoint: ties-to-even goes up from an odd W, stays at an even W
    for w, r, want in ((1 + 2**-7, 2**-8, 2**-7), (1 + 2**-6, -(2**-8), 0.0)):
        w, r = sign * w, sign * r
        e = _round_once(w, r)
        assert e == sign * want
        out = _rbound(w, r, e)
        assert out["passes"] and out["n_double_rounding"] == 0  # error exactly ½ ulp = |R|


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rounding_bound_double_rounding_needs_exactly_one_fp32_spacing(sign):
    """fl32(W + R) lands on a bf16 midpoint that W + R does not reach; ties-to-even then rounds away."""
    w, r = sign * (1 + 2**-7), sign * (2**-8 - 2**-26)
    e = _round_once(w, r)
    assert e == sign * 2**-7  # rounded away from the bf16 point nearest W + R
    single = min(abs(r), 2**-8)
    err = abs(e - r)
    assert err - single == 2**-25 > 0  # fails without the slack
    out = _rbound(w, r, e)
    assert out["passes"] and out["n_double_rounding"] == 1
    assert (
        out["max_excess_fp32_spacings"] == 0.25
    )  # 2^-25 against one fp32 spacing 2^-23 at fl32(W+R)
    # deliberate errors: exactly at the registered bound passes, one fp64 step beyond fails
    bound = single + 2**-23
    at = r + sign * bound
    assert _rbound(w, r, at)["passes"]
    assert not _rbound(w, r, math.nextafter(at, sign * math.inf))["passes"]
    # the old 2·ulp_fp32(|W| + |R|) slack accepted errors up to twice the registered spacing
    assert not _rbound(w, r, r + sign * (single + 1.5 * 2**-23))["passes"]


def test_rounding_bound_under_cancellation():
    """W + R near zero: the registered spacing is that of the tiny fp32 sum, not of |W| + |R|."""
    w, r = 1.5, -1.5 + 2**-20
    e = _round_once(w, r)
    assert w + e == 2**-20
    assert _rbound(w, r, e)["passes"]
    bad = e + 2**-22  # far above 1/2 ulp(2^-20) + ulp_fp32(2^-20), far below 2 ulp_fp32(3.0)
    assert bad - e < 2 * 2.0**-22 and not _rbound(w, r, bad)["passes"]


def test_rounding_bound_signed_zero_subnormals_and_zero_floor():
    for w, r in ((-0.0, 0.0), (0.0, -0.0), (0.0, 0.0)):
        out = _rbound(w, r, _round_once(w, r))
        assert out["passes"] and out["n_violations"] == 0
    assert _round_once(0.0, 2.0**-140) == 0.0 and _rbound(0.0, 2.0**-140, 0.0)["passes"]
    assert _round_once(0.0, 3 * 2.0**-135) == 2.0**-133
    assert _rbound(0.0, 3 * 2.0**-135, 2.0**-133)["passes"]  # bf16 subnormal grid
    assert _rbound(2.0**-133, 2.0**-149, 2.0**-149 * 0)["passes"]  # fp32 subnormal R
    assert not _rbound(0.0, 0.0, 2.0**-133)["passes"]  # zero floor: one fp32 spacing 2^-149 only
    assert _rbound(0.0, 0.0, 2.0**-149)["passes"] and not _rbound(0.0, 0.0, 2.0**-148)["passes"]


def test_rounding_bound_fails_closed_on_overflow_and_nonfinite():
    big = float(torch.finfo(torch.bfloat16).max)
    W = torch.tensor([big], dtype=torch.bfloat16)
    R = torch.tensor([3e38])
    _state, eff = apply_and_round(W, R)  # fl32(W + R) overflows
    out = rounding_bound(W, R, eff)
    assert not out["passes"] and out["nonfinite"] and out["n_violations"] is None
    finite = (torch.ones(2, dtype=torch.bfloat16), torch.zeros(2), torch.zeros(2))
    for i, bad in itertools.product(range(3), (float("nan"), float("inf"), -float("inf"))):
        args = [x.clone() for x in finite]
        args[i][0] = bad
        out = rounding_bound(*args)
        assert not out["passes"] and out["nonfinite"]
    assert not rounding_bound(torch.ones(2, dtype=torch.bfloat16), torch.zeros(2), torch.zeros(3))[
        "passes"
    ]


# --- (v) pooled half-norm guard


def test_sub_resolution_guard_exact_and_just_outside_boundaries():
    assert sub_resolution_guard([2.0], [1.0])["passes"]  # -0.5 exactly
    assert sub_resolution_guard([2.0], [3.0])["passes"]  # +0.5 exactly
    assert not sub_resolution_guard([2.0], [math.nextafter(3.0, math.inf)])["passes"]
    assert (
        sub_resolution_guard([1.0], [0.5])["passes"]
        and sub_resolution_guard([1.0], [1.5])["passes"]
    )
    assert not sub_resolution_guard([1.0], [0.5 - 2**-52])["passes"]  # the nearest resolvable step
    assert not sub_resolution_guard([1.0], [math.nextafter(1.5, math.inf)])["passes"]
    out = sub_resolution_guard([2.0], [3.0])
    assert out["mismatch_signed"] == 0.5 and out["tol"] == 0.5


def test_sub_resolution_guard_pools_norms_not_ratios():
    d = [torch.full((3, 4), 0.5), torch.full((10, 10), 0.3)]  # independent reference: 3x4 and 10x10
    e = [torch.zeros(3, 4), torch.full((10, 10), 0.28)]  # the small tensor rounds away entirely
    dn = [float(x.double().norm()) for x in d]
    en = [float(x.double().norm()) for x in e]
    ref_d = float(torch.cat([x.flatten() for x in d]).double().norm())
    ref_e = float(torch.cat([x.flatten() for x in e]).double().norm())
    out = sub_resolution_guard(dn, en)
    assert out["pooled_d_norm"] == pytest.approx(ref_d, rel=1e-12)
    assert out["mismatch_signed"] == pytest.approx((ref_e - ref_d) / ref_d, rel=1e-12)
    assert out["passes"]  # a per-tensor rule would have excluded the first tensor
    assert not sub_resolution_guard(dn[:1], en[:1])["passes"]


def test_sub_resolution_guard_fails_closed():
    for d, e in (([], []), ([1.0], []), ([float("nan")], [1.0]), ([1.0], [float("inf")]),
                 ([0.0], [0.0]), ([1.0], [None])):  # fmt: skip
        assert not sub_resolution_guard(d, e)["passes"]


# --- (iii) replay: fresh seed, cloned generator, four threads, bit patterns


def test_bits_equal_compares_bit_patterns_dtype_and_shape():
    a = torch.tensor([0.0, 1.0])
    assert bits_equal(a, a.clone()) and torch.equal(a, torch.tensor([-0.0, 1.0]))
    assert not bits_equal(a, torch.tensor([-0.0, 1.0]))
    assert not bits_equal(a, a.double()) and not bits_equal(a, a[None]) and not bits_equal(a, None)


def test_replay_rotation_equals_rotate_block_bitwise():
    d = _block_d((12, 9), 0.3, 1)
    rows, cols = changed_block(d)
    g = torch.Generator().manual_seed(77)
    state = g.get_state()
    shared = rotate_block(d, rows, cols, 0, shared_generator=g)
    assert torch.equal(g.get_state(), state)  # the shared generator is cloned, never advanced
    r, q_out, q_in = replay_rotation(d, rows, cols, 77)
    assert bits_equal(r, shared) and bits_equal(r, rotate_block(d, rows, cols, 77))
    assert q_out.shape == (12, 12) and q_in.shape == (9, 9)


def test_interleaved_unequal_shapes_replay_from_fresh_seeds_in_any_order():
    shapes = [(6, 8), (10, 4), (6, 8), (3, 3), (10, 4)]
    ds = [_block_d(s, 0.2, i, rank=2) for i, s in enumerate(shapes)]
    for order in (range(5), reversed(range(5))):
        g = torch.Generator().manual_seed(123)
        frames = {}
        for i in order:
            rows, cols = changed_block(ds[i])
            built = rotate_block(ds[i], rows, cols, 0, shared_generator=g)
            fresh = rotate_block(ds[i], rows, cols, 123)  # its own manual_seed generator
            memo, _, _ = replay_rotation(ds[i], rows, cols, 123, frames)
            assert bits_equal(built, fresh) and bits_equal(built, memo)


def test_replay_runs_at_four_threads(monkeypatch):
    seen = []
    real = rot.replay_rotation

    def spy(*a, **k):
        seen.append(torch.get_num_threads())
        return real(*a, **k)

    monkeypatch.setattr(rot, "replay_rotation", spy)
    before = torch.get_num_threads()
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    with torch_threads(4):  # the registered construction thread count
        R4 = rotate_block(D, rows, cols, 5)
    assert bits_equal(R4, R)
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5)
    assert seen == [4] and out["replay_threads"] == 4 and torch.get_num_threads() == before
    assert out["passes"] and out["replay_bits_equal"]


def test_replay_compares_written_bytes_bitwise():
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    assert f10_tensor_audit(D, W, R, R_eff, rows, cols, 5, written=R_eff.clone())["passes"]
    zeros = (R_eff == 0).nonzero()
    assert len(zeros)  # R_eff is zero outside the block
    signed = R_eff.clone()
    signed[tuple(zeros[0])] = -0.0  # equal as floats, different bits
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5, written=signed)
    assert torch.equal(signed, R_eff) and out["failed_checks"] == ["replay", "scorer_roundtrip"]
    assert out["reason"] == "replay_failed" and out["support_identical"]
    for written, detail in ((None, "missing"), (R_eff.to(torch.bfloat16), "dtype_or_shape"),
                            (R_eff[:-1], "dtype_or_shape")):  # fmt: skip
        out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5, written=written)
        assert out["failed_checks"] == ["replay", "scorer_roundtrip"]
        assert out["replay_detail"] == detail and out["reason"] == "replay_failed"


# --- tensor audit: F10 against the realised audit, failing closed


def test_f10_passes_where_realised_audit_fails():
    """Low kappa: rounding at W's grid fails the registered realised audit, not the F10 exclusion rule."""
    W = _grid_w((40, 56))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((40, 56), 2**-7 * 0.8, 3, rank=4), 11)
    assert not audit(D, R_eff, rows, cols)["passes"]
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 11)
    assert out["passes"] and out["failed_checks"] == [] and out["reason"] is None
    assert out["s31_passes"] is False  # still computed, as a diagnostic
    assert out["orthogonality_error"] <= 1e-5 and out["reconstruction_residual"] <= 1e-5
    assert out["prerounding_norm_mismatch"] <= 1e-5
    assert out["prerounding_spectrum_mismatch"] <= 1e-5
    assert abs(out["prerounding_rank_difference"]) <= out["rank_n_near"]
    assert out["checks"] == dict.fromkeys(rot.F10_CHECKS, True)
    json.dumps(out, allow_nan=False)


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
    assert not out["passes"] and out["reason"] == "construction_failed"
    # Mirsky: the spectrum mismatch is at least the norm mismatch, so both items fail
    assert out["construction_items"] == {"orthogonality": True, "reconstruction": True,
                                         "norm": False, "spectrum": False, "rank_budget": True}  # fmt: skip

    # (iii) a valid rotation from the wrong seed: construction and rounding pass, the replay does not
    R3 = rotate_block(D, rows, cols, 6)
    _state3, R3_eff = apply_and_round(W, R3)
    assert audit(D, R3_eff, rows, cols)["passes"]
    out = f10_tensor_audit(D, W, R3, R3_eff, rows, cols, 5)
    assert (
        out["failed_checks"] == ["replay", "scorer_roundtrip"] and out["reason"] == "replay_failed"
    )
    assert out["checks"]["construction"] and out["checks"]["rounding_bound"]

    # (iv) one entry moved by one ulp after rounding: realised audit blind, the bound is not
    R4_eff = R_eff.clone()
    R4_eff[rows[0], cols[0]] += 2**-7
    assert audit(D, R4_eff, rows, cols)["passes"]
    out = f10_tensor_audit(D, W, R, R4_eff, rows, cols, 5, written=R4_eff)
    assert "rounding_bound" in out["failed_checks"] and not out["passes"]


def test_f10_spectrum_only_construction_failure():
    """Same norm, other spectrum (all values far above the cut): only the spectrum item fails."""
    sv = [1.0, 0.8, 0.6, 0.5]
    moved = [math.sqrt(1.0 - 2e-4), 0.8, 0.6, math.sqrt(0.25 + 2e-4)]
    W = _grid_w((4, 4))
    D = _spectral(sv, 2)
    rows, cols = changed_block(D)
    R = _rotated_with(D, rows, cols, 3, _spectral(moved, 2))
    _state, R_eff = apply_and_round(W, R)
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 3)
    assert out["prerounding_norm_mismatch"] <= 1e-5 < out["prerounding_spectrum_mismatch"]
    assert out["construction_items"] == {"orthogonality": True, "reconstruction": True,
                                         "norm": True, "spectrum": False, "rank_budget": True}  # fmt: skip
    assert out["reason"] == "construction_failed"


def _scale_u(usv, factor):
    u, s, vh = usv
    return u * factor, s, vh


def test_f10_orthogonality_and_reconstruction_fail_independently(monkeypatch):
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    R_rep, state_rep, eff_rep, q_out, q_in = f10_replay(D, W, rows, cols, 5)
    bent = (R_rep, state_rep, eff_rep, q_out * (1 + 1e-4), q_in)  # a non-orthogonal frame
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5, replayed=bent)
    assert out["failed_checks"] == ["construction"] and out["orthogonality_error"] > 1e-5
    assert [k for k, v in out["construction_items"].items() if not v] == ["orthogonality"]

    real = rot._svd64
    monkeypatch.setattr(rot, "_svd64", lambda x: _scale_u(real(x), 1 + 1e-3))
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5)
    assert out["failed_checks"] == ["construction"] and out["reconstruction_residual"] > 1e-5
    assert [k for k, v in out["construction_items"].items() if not v] == ["reconstruction"]


def test_construction_items_fail_independently_and_closed():
    ok = {"orth": 1e-7, "recon": 1e-15, "pre_norm": 1e-8, "pre_spec": 1e-7, "rank_passes": True}
    assert all(construction_items(**ok).values())
    names = {"orth": "orthogonality", "recon": "reconstruction", "pre_norm": "norm",
             "pre_spec": "spectrum"}  # fmt: skip
    for arg, item in names.items():
        for bad in (1.1e-5, float("nan"), float("inf"), None):
            got = construction_items(**ok | {arg: bad})
            assert [k for k, v in got.items() if not v] == [item]
        assert construction_items(**ok | {arg: 1e-5})[item]  # inclusive at the tolerance
    for bad in (False, None):
        got = construction_items(**ok | {"rank_passes": bad})
        assert [k for k, v in got.items() if not v] == ["rank_budget"]


def test_f10_support_failures_are_independent_of_construction():
    W = _grid_w((24, 30))
    delta = torch.zeros(24, 30)
    delta[2:20, 3:25] = _block_d((18, 22), 0.5, 4, rank=18)
    D, rows, cols, R, _state, R_eff = _staged(W, delta, 5)
    assert f10_tensor_audit(D, W, R, R_eff, rows, cols, 5)["passes"]
    D5 = D.clone()
    D5[0, 0] = 0.25  # D nonzero outside the supplied block
    out = f10_tensor_audit(D5, W, R, R_eff, rows, cols, 5)
    assert out["failed_checks"] == ["support"] and out["reason"] == "support_failed"
    assert out["nonzero_d_outside_block"] == 1 and out["checks"]["construction"]
    R6 = R.clone()
    R6[0, 0] = 1e-9  # the construction leaks (rounds away: R_eff stays clean)
    out = f10_tensor_audit(D, W, R6, R_eff, rows, cols, 5)
    assert out["failed_checks"] == ["support"] and out["nonzero_r_raw_outside_block"] == 1
    E7 = R_eff.clone()
    E7[0, 0] = 2**-7  # a written change outside the block
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5, written=E7)
    assert out["reason"] == "support_failed" and out["nonzero_r_eff_outside_block"] == 1
    assert {"support", "replay", "rounding_bound"} <= set(out["failed_checks"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("which", ["D", "W", "R", "R_eff"])
def test_f10_nonfinite_inputs_fail_closed(which, bad):
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    x = {"D": D.clone(), "W": W.clone(), "R": R.clone(), "R_eff": R_eff.clone()}
    x[which][rows[1], cols[2]] = bad
    out = f10_tensor_audit(x["D"], x["W"], x["R"], x["R_eff"], rows, cols, 5, written=x["R_eff"])
    assert not out["passes"] and out["reason"] is not None
    expected = {"D": "construction", "W": "rounding_bound", "R": "construction", "R_eff": "replay"}
    assert expected[which] in out["failed_checks"]
    if which in ("W", "R", "R_eff"):
        assert "rounding_bound" in out["failed_checks"]
    json.dumps(out, allow_nan=False)  # no NaN reaches JSON


def test_f10_nonfinite_outputs_fail_closed(monkeypatch):
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    real = rot._svd64
    monkeypatch.setattr(rot, "_svd64", lambda x: _scale_u(real(x), float("nan")))
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5)
    assert out["failed_checks"] == ["construction"] and out["reconstruction_residual"] is None
    assert "reconstruction_residual" in out["nonfinite_fields"]
    assert "missing:reconstruction_residual" in out["fail_closed_detail"]
    json.dumps(out, allow_nan=False)


def test_f10_verdict_fails_closed_on_missing_or_invalid_items():
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    good = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5)
    assert f10_verdict(good)["passes"]
    missing = f10_verdict(None)
    assert missing["failed_checks"] == list(rot.F10_CHECKS) and missing["detail"] == [
        "audit_missing"
    ]
    for check, items in rot.F10_REQUIRED.items():
        a = json.loads(json.dumps(good))
        del a["checks"][check]
        assert f10_verdict(a)["failed_checks"] == [check]
        a = json.loads(json.dumps(good))
        a["checks"][check] = "yes"  # not a boolean
        assert f10_verdict(a)["failed_checks"] == [check]
        for item in items:
            for value in (None, float("nan"), float("inf"), "x"):
                a = json.loads(json.dumps(good))
                a[item] = value
                v = f10_verdict(a)
                assert v["failed_checks"] == [check] and v["reason"] == rot.F10_REASONS[check]
            a = json.loads(json.dumps(good))
            del a[item]
            assert f10_verdict(a)["failed_checks"] == [check]


# --- F10.4 diagnostics


def test_f10_diagnostics_are_reported():
    W = _grid_w((32, 40))
    delta = _block_d((32, 40), 2**-7 * 3, 8)
    D, rows, cols, R, _state, R_eff = _staged(W, delta, 2)
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 2, raw_delta=delta)
    dn = float(D.double().norm())
    kappa = dn / math.sqrt(32 * 40 * 2.0**-14)
    assert math.isclose(out["kappa"], kappa, rel_tol=1e-9)
    assert math.isclose(out["eps_hat"], min(1.0, 1 / (math.sqrt(12) * kappa)), rel_tol=1e-9)
    assert out["sum_ulp2_block"] == 32 * 40 * 2.0**-14
    assert math.isclose(
        out["rounding_error_rel"], float((R_eff - R).double().norm()) / dn, rel_tol=1e-9
    )
    assert math.isclose(
        out["energy_ratio"], float(R_eff.double().norm()) ** 2 / dn**2, rel_tol=1e-9
    )
    assert math.isclose(out["eps_D"], float((D - delta).double().norm()) / dn, rel_tol=1e-9)
    tau = math.sqrt(2.0**-14 / 12) * (math.sqrt(32) + math.sqrt(40))
    assert math.isclose(out["tau"], tau, rel_tol=1e-9)
    # noise ranks, independently: fp32 singular values strictly above max(1e-3 s1, 3 tau)
    sd, sr = torch.linalg.svdvals(D.float()), torch.linalg.svdvals(R_eff.float())
    assert out["noise_rank_d"] == int((sd > max(3 * tau, 1e-3 * float(sd[0]))).sum())
    assert out["noise_rank_r"] == int((sr > max(3 * tau, 1e-3 * float(sr[0]))).sum())
    assert 0 < out["noise_rank_d"] < out["rank_d"]  # the noise cut binds here
    assert out["norm_mismatch_signed"] == pytest.approx(
        (float(R_eff.double().norm()) - dn) / dn, rel=1e-6
    )
    assert out["spectrum_mismatch"] == pytest.approx(
        float((sr - sd).double().norm() / sd.double().norm()), rel=1e-6
    )
    assert f10_tensor_audit(D, W, R, R_eff, rows, cols, 2)["eps_D"] is None


def test_noise_calibrated_components_fail_alone_and_never_exclude():
    ok = {
        "norm_signed": 0.01,
        "spectrum_mismatch": 0.04,
        "eps_hat": 0.0,
        "noise_rank_d": 5,
        "noise_rank_r": 6,
    }
    assert noise_calibrated(**ok)["passes"]
    assert noise_calibrated(**ok | {"norm_signed": -0.03, "eps_hat": 0.01})["passes"]  # 2e-2 + eps
    for bad, item in (({"norm_signed": -0.0201}, "norm"), ({"spectrum_mismatch": 0.0501}, "spectrum"),
                      ({"noise_rank_r": 7}, "rank"), ({"noise_rank_r": 3}, "rank")):  # fmt: skip
        got = noise_calibrated(**ok | bad)
        assert not got["passes"] and [k for k in ("norm", "spectrum", "rank") if not got[k]] == [
            item
        ]
    # a tensor that F10 passes keeps passing whatever the noise-calibrated flags say
    W = _grid_w((24, 30))
    D, rows, cols, R, _state, R_eff = _staged(W, _block_d((24, 30), 0.5, 4, rank=24), 5)
    a = f10_tensor_audit(D, W, R, R_eff, rows, cols, 5)
    a |= {
        "noise_calibrated_passes": False,
        "noise_calibrated": dict.fromkeys(("norm", "spectrum", "rank"), False),
    }
    assert f10_verdict(a)["passes"]
    # a real one: a flat spectrum just above 3 tau, where rounding moves the noise-referenced rank by 2
    W = _grid_w((64, 64))
    tau = math.sqrt(2.0**-14 / 12) * 16
    D, rows, cols, R, _state, R_eff = _staged(W, _spectral([3.3 * tau] * 24, 0, (64, 64)), 2)
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 2)
    assert out["noise_calibrated"] == {"norm": True, "spectrum": True, "rank": False}
    assert abs(out["noise_rank_r"] - out["noise_rank_d"]) > 1
    assert out["passes"] and not out["noise_calibrated_passes"]


def test_sub_resolution_tensor_rounds_away_but_passes_tensor_checks():
    """Sparse single-ulp flips (chain residue): R lies below half an ulp and rounds to zero."""
    W = _grid_w((48, 48))
    perm = torch.randperm(48, generator=torch.Generator().manual_seed(0))
    delta = torch.zeros(48, 48)
    delta[torch.arange(48), perm] = 2**-7  # 48 flips, one per row and column: a 48 x 48 block
    D, rows, cols, R, _state, R_eff = _staged(W, delta, 9)
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 9)
    assert out["passes"] and out["energy_ratio"] < 0.25
    assert not sub_resolution_guard([out["d_norm"]], [out["r_eff_norm"]])["passes"]


def test_scorer_roundtrip_failure_is_a_replay_failure():
    """Chain step: W = 0 where the base is large; bf16(W + R) is tiny and cannot be written as an fp32
    delta from the base that the scorer adds back (bf16(base + delta) != state)."""
    W = _grid_w((8, 8))
    W[2, 3] = 0.0
    base = W.clone()
    base[2, 3] = 1e5
    D, rows, cols, R, state, R_eff = _staged(W, _block_d((8, 8), 0.05, 6, rank=8), 3)
    assert state[2, 3] != 0
    delta = state.float() - base.float()
    rt = scorer_roundtrip(base, delta, state)
    assert not rt["passes"] and rt["delta_bits_equal"] and rt["mismatches"] == 1
    out = f10_tensor_audit(D, W, R, R_eff, rows, cols, 3, scorer=(base, delta))
    assert out["failed_checks"] == ["scorer_roundtrip"] and out["reason"] == "replay_failed"
    assert out["scorer_roundtrip_mismatches"] == 1 and out["replay_passes"] is False
    ok = f10_tensor_audit(D, W, R, R_eff, rows, cols, 3)  # root: the scorer's base is W itself
    assert ok["passes"] and ok["scorer_roundtrip_mismatches"] == 0
    flipped = delta.clone()
    flipped[0, 0] = -flipped[0, 0] if flipped[0, 0] != 0 else -0.0
    assert not scorer_roundtrip(base, flipped, state)["passes"]
    for bad in (None, delta.double(), delta[1:]):
        assert not scorer_roundtrip(base, bad, state)["passes"]
