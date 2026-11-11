"""Rotation controls for realised increments of stored bf16 states (SYNTHESIS 7 Nov §4.1, PREREG A5).

An increment is successor minus predecessor, both stored bf16 states, subtracted in fp32. The control
R = Q_out U Σ Vᵀ Q_inᵀ = Q_out D_block Q_inᵀ rotates the realised block with Haar-orthogonal Q on each side, so it
does not depend on the choice of singular basis: degenerate (repeated) singular values are irrelevant to the
construction, and the SVD is only used for the rank and spectrum audits. "Spectrum-matched" refers to the
pre-rounding construction; the scored object is bf16(W + R) − W, which :func:`audit` compares with D_eff.
"""

import math

import torch


def realised_increment(w_pred_bf16, w_succ_bf16):
    """Subtract the stored bf16 predecessor from the stored successor in fp32."""
    return w_succ_bf16.float() - w_pred_bf16.float()


def changed_block(D, atol=0.0):
    """Return indices of changed rows and columns."""
    mask = D.abs() > atol
    return torch.where(mask.any(dim=1))[0], torch.where(mask.any(dim=0))[0]


def block_svd(D, rows, cols, rank_rtol=1e-3):
    """Return thin U, singular values, Vh and relative-tolerance rank."""
    block = D.float()[rows[:, None], cols]
    u, s, vh = torch.linalg.svd(block, full_matrices=False)
    rank = int((s > s[0] * rank_rtol).sum()) if s.numel() and s[0] > 0 else 0
    return u, s, vh, rank


def lora_factors_reproduce(D_block, A, B, scaling, rtol=1e-3):
    """Check relative Frobenius residual of scaled B @ A against the block."""
    target = D_block.float()
    residual = torch.linalg.vector_norm(scaling * (B.float() @ A.float()) - target)
    norm = torch.linalg.vector_norm(target)
    return bool(residual <= rtol * norm)


def haar_orthogonal(n, generator):
    """Sample a Haar orthogonal matrix using sign-corrected Gaussian QR."""
    q, r = torch.linalg.qr(torch.randn(n, n, generator=generator, dtype=torch.float32))
    signs = torch.sign(torch.diagonal(r))
    return q * torch.where(signs == 0, 1, signs)


def rotate_block(D, rows, cols, seed, shared_generator=None):
    """Construct a pre-rounding spectrum-matched rotation of the realised block."""
    generator = torch.Generator()
    if shared_generator is None:
        generator.manual_seed(seed)
    else:
        generator.set_state(shared_generator.get_state())
    block = D.float()[rows[:, None], cols]
    q_out = haar_orthogonal(len(rows), generator)
    q_in = haar_orthogonal(len(cols), generator)
    result = torch.zeros_like(D, dtype=torch.float32)
    result[rows[:, None], cols] = (
        q_out @ block @ q_in.T
    )  # equals Q_out U Σ Vᵀ Q_inᵀ for any SVD of the block
    return result


def apply_and_round(W_bf16, X_fp32):
    """Apply at the stored predecessor and return successor and realised increment."""
    new = (W_bf16.float() + X_fp32.float()).to(torch.bfloat16)
    return new, realised_increment(W_bf16, new)


def scaled_variant(W_bf16, delta_fp32, s):
    """Scale the original delta before applying and rounding at the predecessor."""
    return apply_and_round(W_bf16, s * delta_fp32)


def audit(
    D_eff,
    R_eff,
    rows,
    cols,
    *,
    norm_tol=2e-2,
    spectrum_tol=5e-2,
    rank_rtol=1e-3,
    rank_diff_tol=1,
    spectra=None,
):
    """Audit realised increments; spectrum-matched describes the pre-rounding rotation.

    ``spectra``, if a dict, receives the fp32 singular values ``sd`` and ``sr`` (no effect on the result)."""
    d, r = D_eff.float(), R_eff.float()
    sd, sr = torch.linalg.svdvals(d), torch.linalg.svdvals(r)
    if spectra is not None:
        spectra.update(sd=sd, sr=sr)
    norm = torch.linalg.vector_norm(d)
    if norm == 0:
        raise ValueError("Cannot audit a zero reference increment")
    norm_mismatch = float((torch.linalg.vector_norm(r) - norm).abs() / norm)
    spectrum_mismatch = float(torch.linalg.vector_norm(sr - sd) / torch.linalg.vector_norm(sd))
    rank_d = int((sd > sd[0] * rank_rtol).sum())
    rank_r = int((sr > sr[0] * rank_rtol).sum()) if sr[0] > 0 else 0
    support = torch.zeros_like(d, dtype=torch.bool)
    support[rows[:, None], cols] = True
    support_identical = bool((d[~support] == 0).all() and (r[~support] == 0).all())
    difference = rank_r - rank_d
    return {
        "norm_mismatch": norm_mismatch,
        "spectrum_mismatch": spectrum_mismatch,
        "rank_d": rank_d,
        "rank_r": rank_r,
        "rank_difference": difference,
        "support_identical": support_identical,
        "passes": norm_mismatch <= norm_tol
        and spectrum_mismatch <= spectrum_tol
        and abs(difference) <= rank_diff_tol
        and support_identical,
    }


def exclusion_reason(D, rows, cols):
    """Why a tensor cannot be rotated (None if it can): ``not_single_matrix``, ``non_2d``, ``zero_block``,
    ``non_finite`` or ``rank_degenerate`` (numerical rank 0 at the audit tolerance, which a nonzero block
    can still have when its entries are below rank_rtol of a rounding artefact). Low rank is *not* an
    exclusion: LoRA blocks have rank 4–32 on 1,024-wide blocks and rotate exactly like any other matrix."""
    if not isinstance(D, torch.Tensor):
        return "not_single_matrix"
    if D.ndim != 2:
        return "non_2d"
    if not torch.isfinite(D).all():
        return "non_finite"
    if len(rows) == 0 or len(cols) == 0 or not torch.any(D[rows[:, None], cols]):
        return "zero_block"
    if block_svd(D, rows, cols)[-1] == 0:
        return "rank_degenerate"
    return None


# ------------------------------------------------------------------------- F10 exclusion audit (remedy d)
#
# Chair ruling F10.3 (remedy d): construction unchanged. A tensor is excluded only if one of these fails:
# (i) fp64 construction checks of the pre-rounding R against D (orthogonality of Q_out and Q_in, SVD
# reconstruction residual of the block, norm and spectrum agreement, each within 1e-5, and equal 1e-3
# rank); (ii) support confined to the edited block; (iii) bitwise replay from the registered seed, and the
# scored state round-trips through the scorer's arithmetic bf16(base + (state - base)) (F10 draft (iii):
# "as staged"; reported as check ``scorer_roundtrip`` with reason ``replay_failed``); (iv) the
# round-to-nearest bound. A pair is also excluded if rounding keeps less than half of the edit's energy
# (sum ||R_eff||^2 >= 1/2 sum ||R||^2 over the edited tensors). The registered realised norm, spectrum and
# rank and the noise-calibrated quantities of the design note are computed as diagnostics only.

F10_TOL = 1e-5
F10_ENERGY_FLOOR = 0.5
F10_REASONS = (  # failed check -> exclusion reason, in reporting order
    ("construction", "construction_failed"),
    ("support", "support_failed"),
    ("replay", "replay_failed"),
    ("scorer_roundtrip", "replay_failed"),
    ("rounding_bound", "rounding_bound_failed"),
)
NOISE_RANK_C = 3.0  # design note remedy (b): rank above max(1e-3 s1, 3 tau), within +-1


def _spacing(x, mantissa_bits, min_exp):
    """Grid spacing at |x| for a binary format with ``mantissa_bits`` significant bits: 2^(e - bits) for
    x = m 2^e, m in [1/2, 1), floored at the subnormal spacing 2^min_exp (also used at x = 0)."""
    _, e = torch.frexp(x.double())
    u = torch.ldexp(torch.ones_like(e, dtype=torch.float64), e - mantissa_bits)
    return torch.where(x == 0, 2.0**min_exp, torch.clamp(u, min=2.0**min_exp))


def ulp_bf16(x):
    """bf16 spacing at each value of x (fp64): 2^(e-8); 2^-133 at zero and in the subnormal range."""
    return _spacing(x, 8, -133)


def ulp_fp32(x):
    """fp32 spacing at each value of x (fp64): 2^(e-24); 2^-149 at zero and in the subnormal range."""
    return _spacing(x, 24, -149)


def rotation_frames(n_rows, n_cols, seed):
    """Q_out, Q_in drawn from a fresh generator seeded with the registered seed (rotate_block's order)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return haar_orthogonal(n_rows, generator), haar_orthogonal(n_cols, generator)


def replay_rotation(D, rows, cols, seed, frames=None):
    """Rebuild R from the registered seed with rotate_block's arithmetic; returns (R, Q_out, Q_in).

    ``frames``: optional dict memoising rotation_frames per (seed, n_rows, n_cols); every frame in it was
    drawn from a fresh generator seeded with ``seed``, so memoising does not change the replay."""
    key = (seed, len(rows), len(cols))
    if frames is None or key not in frames:
        pair = rotation_frames(len(rows), len(cols), seed)
        if frames is None:
            q_out, q_in = pair
        else:
            frames[key] = pair
    if frames is not None:
        q_out, q_in = frames[key]
    block = D.float()[rows[:, None], cols]
    result = torch.zeros_like(D, dtype=torch.float32)
    result[rows[:, None], cols] = q_out @ block @ q_in.T
    return result, q_out, q_in


def rounding_bound(W_bf16, R_raw, R_eff):
    """F10.3(iv): |R_eff - R| <= min(|R|, 1/2 ulp) entrywise, checked in fp64.

    The ulp is the bf16 spacing at the rounded value W + R_eff (round to nearest lands on an endpoint of
    the grid interval containing W + R, and ulp(result) is at least that interval's width; ulp(W) alone is
    too small when W + R crosses up into the next binade, and is undefined at W = 0). The slack
    2 ulp_fp32(|W| + |R|) (2^-15 of a bf16 ulp) covers the fp32 sum that apply_and_round rounds:
    bf16(fl32(W + R)) can miss the nearest bf16 point to the exact W + R by one fp32 rounding, on both
    sides of the bound. ``n_exceed_half_ulp_w`` counts entries above 1/2 ulp(W) + slack (diagnostic)."""
    w = W_bf16.double()
    r = R_raw.double()
    e = R_eff.double()
    err = (e - r).abs()
    slack = 2 * ulp_fp32(w.abs() + r.abs())
    half = 0.5 * ulp_bf16(w + e)
    bound = torch.minimum(r.abs(), half) + slack
    violations = int((err > bound).sum())
    return {
        "passes": violations == 0,
        "n_violations": violations,
        "max_ratio_to_half_ulp": float((err / half).max()) if err.numel() else 0.0,
        "n_exceed_half_ulp_w": int((err > 0.5 * ulp_bf16(w) + slack).sum()),
    }


def edit_energy_guard(r_energies, r_eff_energies, floor=F10_ENERGY_FLOOR):
    """F10 per-edit guard on squared norms: sum ||R_eff||^2 >= floor * sum ||R||^2 (pooled)."""
    r, e = float(sum(r_energies)), float(sum(r_eff_energies))
    ratio = e / r if r > 0 else 0.0
    return {"ratio": ratio, "floor": floor, "passes": bool(r > 0 and e >= floor * r)}


def _fro64(x):
    return float(torch.linalg.vector_norm(x.double()))


def _rank(s, rtol):
    return int((s > s[0] * rtol).sum()) if s.numel() and s[0] > 0 else 0


def f10_tensor_audit(
    D, W_bf16, R_raw, R_eff, rows, cols, seed, *, raw_delta=None, frames=None, scorer_base=None,
    tol=F10_TOL, rank_rtol=1e-3,
):  # fmt: skip
    """F10.3 per-tensor exclusion checks (i)-(iv) plus the F10.4 realised diagnostics.

    ``D`` is the realised increment at the predecessor ``W_bf16``, ``R_raw`` the pre-rounding rotation
    that was staged and ``R_eff`` its realised increment bf16(W + R) - W; ``seed`` is the registered seed
    for the replay. ``scorer_base`` is the state the scorer adds the staged delta to (the base for a chain
    step; default ``W_bf16``, a root edit). ``raw_delta`` (root edits) adds eps_D. Returns a JSON-able dict; ``passes`` is the
    F10 verdict, ``failed_checks`` the failing checks and ``reason`` the first exclusion reason."""
    idx = (rows[:, None], cols)
    m, n = len(rows), len(cols)
    d64 = D.double()[idx]
    r64 = R_raw.double()[idx]
    d_norm = _fro64(d64)

    # (iii) replay first: its frames serve the orthogonality check.
    R_replay, q_out, q_in = replay_rotation(D, rows, cols, seed, frames)
    state_replay = (W_bf16.float() + R_replay).to(torch.bfloat16)  # apply_and_round's arithmetic
    replay_ok = bool(
        torch.equal(R_replay, R_raw.float())
        and torch.equal(state_replay.float() - W_bf16.float(), R_eff.float())
    )
    # the staged delta is state - base in fp32; the scorer computes bf16(base + delta)
    sb = W_bf16 if scorer_base is None else scorer_base
    state = (W_bf16.float() + R_eff.float()).to(W_bf16.dtype)
    rt = (sb.float() + (state.float() - sb.float())).to(sb.dtype)
    roundtrip_mismatches = int((rt != state).sum())

    # (i) construction, fp64
    orth = 0.0
    for q in (q_out, q_in):
        k = q.shape[0]
        q = q.double()
        orth = max(orth, _fro64(q.T @ q - torch.eye(k, dtype=torch.float64)) / math.sqrt(k))
    u, sd64, vh = torch.linalg.svd(d64, full_matrices=False)
    recon = _fro64(u @ torch.diag(sd64) @ vh - d64) / d_norm
    del u, vh
    sr64 = torch.linalg.svdvals(r64)
    pre_norm = abs(_fro64(r64) - d_norm) / d_norm
    pre_spec = _fro64(sr64 - sd64) / _fro64(sd64)
    pre_rank_d, pre_rank_r = _rank(sd64, rank_rtol), _rank(sr64, rank_rtol)
    construction_residual = _fro64(r64 - q_out.double() @ d64 @ q_in.double().T) / d_norm
    construction_ok = (
        orth <= tol
        and recon <= tol
        and pre_norm <= tol
        and pre_spec <= tol
        and pre_rank_d == pre_rank_r
    )

    # (ii) support: R and its realised change confined to the block
    inside = torch.zeros(D.shape, dtype=torch.bool)
    inside[idx] = True
    support_ok = bool((R_eff[~inside] == 0).all() and (R_raw[~inside] == 0).all())

    # (iv) round-to-nearest bound over the whole tensor
    bound = rounding_bound(W_bf16, R_raw, R_eff)

    failed = [
        name
        for name, ok in (
            ("construction", construction_ok),
            ("support", support_ok),
            ("replay", replay_ok),
            ("scorer_roundtrip", roundtrip_mismatches == 0),
            ("rounding_bound", bound["passes"]),
        )
        if not ok
    ]
    reason = next((r for name, r in F10_REASONS if name in failed), None)

    # F10.4 diagnostics (not exclusion criteria)
    spectra = {}
    realised = audit(D, R_eff, rows, cols, rank_rtol=rank_rtol, spectra=spectra)
    r_eff_norm = _fro64(R_eff)
    ulp_w = ulp_bf16(W_bf16[idx])
    sum_ulp2 = float((ulp_w**2).sum())
    kappa = d_norm / math.sqrt(sum_ulp2)
    eps_hat = min(1.0, 1.0 / (math.sqrt(12) * kappa)) if kappa > 0 else 1.0
    tau = math.sqrt(sum_ulp2 / (m * n) / 12) * (math.sqrt(m) + math.sqrt(n))
    sd, sr = spectra["sd"], spectra["sr"]
    noise_rank_d = int((sd > max(NOISE_RANK_C * tau, rank_rtol * float(sd[0]))).sum())
    noise_rank_r = (
        int((sr > max(NOISE_RANK_C * tau, rank_rtol * float(sr[0]))).sum()) if sr[0] > 0 else 0
    )
    norm_signed = (r_eff_norm - d_norm) / d_norm
    noise_ok = (
        abs(norm_signed) <= 2e-2 + eps_hat
        and realised["spectrum_mismatch"] <= 5e-2 + eps_hat
        and abs(noise_rank_r - noise_rank_d) <= 1
    )
    return {
        "passes": not failed,
        "failed_checks": failed,
        "reason": reason,
        # (i)
        "orthogonality_error": orth,
        "reconstruction_residual": recon,
        "prerounding_norm_mismatch": pre_norm,
        "prerounding_spectrum_mismatch": pre_spec,
        "prerounding_rank_d": pre_rank_d,
        "prerounding_rank_r": pre_rank_r,
        "construction_residual": construction_residual,
        "construction_passes": construction_ok,
        "block_rank": pre_rank_d,
        "block_shape": [m, n],
        # (ii)-(iv)
        "support_identical": support_ok,
        "replay_passes": replay_ok and roundtrip_mismatches == 0,
        "replay_bytes_equal": replay_ok,
        "scorer_roundtrip_mismatches": roundtrip_mismatches,
        "rounding_bound_passes": bound["passes"],
        "rounding_bound_violations": bound["n_violations"],
        "rounding_bound_max_ratio": bound["max_ratio_to_half_ulp"],
        "n_exceed_half_ulp_w": bound["n_exceed_half_ulp_w"],
        # registered realised audit, reported only
        "norm_mismatch": realised["norm_mismatch"],
        "norm_mismatch_signed": norm_signed,
        "spectrum_mismatch": realised["spectrum_mismatch"],
        "rank_d": realised["rank_d"],
        "rank_r": realised["rank_r"],
        "rank_difference": realised["rank_difference"],
        "s31_passes": realised["passes"],
        # rounding scale and noise-calibrated audit (design note §1.2, remedy (b) at c = 3, +-1)
        "kappa": kappa,
        "eps_hat": eps_hat,
        "rounding_error_rel": _fro64(R_eff.double() - R_raw.double()) / d_norm,
        "eps_D": _fro64(D.double() - raw_delta.double()) / d_norm
        if raw_delta is not None
        else None,
        "tau": tau,
        "noise_rank_c": NOISE_RANK_C,
        "noise_rank_d": noise_rank_d,
        "noise_rank_r": noise_rank_r,
        "noise_calibrated_passes": bool(noise_ok),
        "d_norm": d_norm,
        "r_norm": _fro64(R_raw),
        "r_eff_norm": r_eff_norm,
        "energy_ratio": (r_eff_norm / d_norm) ** 2,
    }
