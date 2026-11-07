"""Rotation controls for realised increments of stored bf16 states.

An increment is successor minus predecessor, subtracted in fp32. Spectrum-matched
refers to the pre-rounding construction, not necessarily the stored result.
"""

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
    u, s, vh, _ = block_svd(D, rows, cols)
    q_out = haar_orthogonal(len(rows), generator)
    q_in = haar_orthogonal(len(cols), generator)
    result = torch.zeros_like(D, dtype=torch.float32)
    result[rows[:, None], cols] = q_out @ (u * s) @ vh @ q_in.T
    return result


def apply_and_round(W_bf16, X_fp32):
    """Apply at the stored predecessor and return successor and realised increment."""
    new = (W_bf16.float() + X_fp32.float()).to(torch.bfloat16)
    return new, realised_increment(W_bf16, new)


def scaled_variant(W_bf16, delta_fp32, s):
    """Scale the original delta before applying and rounding at the predecessor."""
    return apply_and_round(W_bf16, s * delta_fp32)


def audit(
    D_eff, R_eff, rows, cols, *, norm_tol=2e-2, spectrum_tol=5e-2, rank_rtol=1e-3, rank_diff_tol=1
):
    """Audit realised increments; spectrum-matched describes the pre-rounding rotation."""
    d, r = D_eff.float(), R_eff.float()
    sd, sr = torch.linalg.svdvals(d), torch.linalg.svdvals(r)
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
    """Classify unsupported edits and zero or rank-degenerate matrix blocks."""
    if not isinstance(D, torch.Tensor):
        return "not_single_matrix"
    if D.ndim != 2:
        return "non_2d"
    if len(rows) == 0 or len(cols) == 0 or not torch.any(D[rows[:, None], cols]):
        return "zero_block"
    if block_svd(D, rows, cols)[-1] < min(len(rows), len(cols)):
        return "rank_degenerate"
    return None
