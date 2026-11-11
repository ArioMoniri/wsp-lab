"""Rotation controls for realised increments of stored bf16 states (SYNTHESIS 7 Nov §4.1, PREREG A5).

An increment is successor minus predecessor, both stored bf16 states, subtracted in fp32. The control
R = Q_out U Σ Vᵀ Q_inᵀ = Q_out D_block Q_inᵀ rotates the realised block with Haar-orthogonal Q on each side, so it
does not depend on the choice of singular basis: degenerate (repeated) singular values are irrelevant to the
construction, and the SVD is only used for the rank and spectrum audits. "Spectrum-matched" refers to the
pre-rounding construction; the scored object is bf16(W + R) − W, which :func:`audit` compares with D_eff.
"""

import contextlib
import hashlib
import math
import os
import platform

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
# PREREG A5 F10.3 as read by F10a.1 (chair ruling R1, remedy (d) of the rotation-yield note); the
# construction is unchanged. A tensor passes iff
# (i) construction, in fp64 from the stored bf16 states and the fp32 construction: orthogonality error of
#     Q_out and Q_in, full-SVD reconstruction residual of the block, pre-rounding norm and sorted-spectrum
#     mismatch, each <= 1e-5, and the near-cut rank budget |rank R - rank D| <= N_near (each rank counts
#     sigma_j > 1e-3 sigma_1 of its own matrix, strictly; N_near counts sigma_j(D) with
#     |sigma_j(D) - 1e-3 sigma_1(D)| <= 1e-5 sigma_1(D), inclusive);
# (ii) support: D, R and R_eff are zero outside the changed block;
# (iii) replay: R rebuilt from a fresh torch.Generator().manual_seed(seed) at the staged thread count (4)
#     and rounded with apply_and_round equals, bit for bit, the R_eff bytes reloaded from the written file;
#     and the reloaded written delta reproduces the intended bf16 state bit for bit under the scorer's
#     bf16(base + delta) (check ``scorer_roundtrip``, reason ``replay_failed``);
# (iv) rounding bound, entrywise in fp64: |R_eff - R| <= min(|R|, 1/2 ulp_bf16(W + R_eff)) + ulp_fp32(fl32(W + R)).
# Per edit, (v) the pooled half-norm guard |sqrt(sum ||R_eff||^2) - sqrt(sum ||D||^2)| / sqrt(sum ||D||^2)
# <= 0.5 (``sub_resolution``). A missing, non-boolean or nonfinite audit item fails its check closed
# (:func:`f10_verdict`). The registered realised audit and the noise-calibrated quantities (F10.4) are
# diagnostics only.

F10_TOL = 1e-5  # F10.3(i) construction tolerances
F10_RANK_RTOL = 1e-3  # A5.3.3 numerical-rank rule
F10_RANK_WINDOW = 1e-5  # F10a.1 near-cut window, relative to sigma_1(D)
F10_GUARD_TOL = 0.5  # F10.3(v) pooled half-norm guard
F10_THREADS = 4  # F10.3(iii) staged thread settings
F10_CHECKS = ("construction", "support", "replay", "scorer_roundtrip", "rounding_bound")
F10_REASONS = {  # failed check -> exclusion reason; the first failed check in F10_CHECKS order is reported
    "construction": "construction_failed",
    "support": "support_failed",
    "replay": "replay_failed",
    "scorer_roundtrip": "replay_failed",
    "rounding_bound": "rounding_bound_failed",
}
F10_REQUIRED = {  # audit items each check needs; absent or nonfinite -> the check fails closed
    "construction": (
        "orthogonality_error",
        "reconstruction_residual",
        "prerounding_norm_mismatch",
        "prerounding_spectrum_mismatch",
        "prerounding_rank_d",
        "prerounding_rank_r",
        "rank_cut_d",
        "rank_cut_r",
        "rank_n_near",
    ),
    "support": (
        "nonzero_d_outside_block",
        "nonzero_r_raw_outside_block",
        "nonzero_r_eff_outside_block",
    ),
    "replay": ("replay_bits_equal",),
    "scorer_roundtrip": ("scorer_roundtrip_mismatches",),
    "rounding_bound": ("rounding_bound_violations",),
}
CONSTRUCTION_ITEMS = ("orthogonality", "reconstruction", "norm", "spectrum", "rank_budget")
NOISE_RANK_C = 3.0  # design note remedy (b): rank above max(1e-3 s1, 3 tau), within +-1
_BUILT = object()  # f10_tensor_audit: "the written bytes are the in-memory R_eff"


def _finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _le(x, tol):
    return _finite_number(x) and x <= tol


def _all_finite(*xs):
    return all(isinstance(x, torch.Tensor) and bool(torch.isfinite(x).all()) for x in xs)


@contextlib.contextmanager
def torch_threads(n):
    """Run the enclosed torch arithmetic with ``n`` intra-op threads, restoring the previous count."""
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def numerical_environment():
    """What identifies the replay arithmetic: torch build and config, thread settings, Python, CPU."""
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((x.split(":", 1)[1].strip() for x in f if x.startswith("model name")), None)
    except OSError:
        pass
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return {
        "torch": torch.__version__,
        "torch_config_sha256": digest(torch.__config__.show()),
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "torch_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": cpu,
    }


_INT_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bit_pattern(x):
    """The tensor's bytes as integers of the same width (distinguishes -0.0 from 0.0 and NaN payloads)."""
    return x.contiguous().view(_INT_OF_SIZE[x.element_size()])


def bits_equal(a, b):
    """Same dtype, same shape and identical bit patterns (not floating equality)."""
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(bit_pattern(a), bit_pattern(b)))


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


def f10_replay(D, W_bf16, rows, cols, seed, frames=None, threads=F10_THREADS):
    """F10.3(iii) reference: R from a fresh ``manual_seed(seed)`` generator at ``threads`` threads, rounded
    at W with apply_and_round. Returns (R, bf16 state, R_eff, Q_out, Q_in)."""
    with torch_threads(threads):
        R, q_out, q_in = replay_rotation(D, rows, cols, seed, frames)
        state, R_eff = apply_and_round(W_bf16, R)
    return R, state, R_eff, q_out, q_in


def rank_budget(s_d, s_r, rtol=F10_RANK_RTOL, window=F10_RANK_WINDOW):
    """F10a.1 near-cut rank budget on singular values (fp64): each rank counts values strictly above
    ``rtol`` times its own largest value; N_near counts sigma_j(D) within ``window`` sigma_1(D) of D's cut
    (inclusive); passes iff |rank_R - rank_D| <= N_near. A budget, not a proof that individual crossings
    were near-cut. Nonfinite or empty spectra fail closed."""
    out = dict.fromkeys(
        ("rank_d", "rank_r", "rank_cut_d", "rank_cut_r", "rank_window", "rank_n_near")
    )
    out |= {"rank_difference": None, "passes": False}
    sd, sr = s_d.double(), s_r.double()
    if not sd.numel() or not sr.numel() or not _all_finite(sd, sr):
        return out
    s1_d, s1_r = float(sd.max()), float(sr.max())
    cut_d, cut_r, win = rtol * s1_d, rtol * s1_r, window * s1_d
    rank_d, rank_r = int((sd > cut_d).sum()), int((sr > cut_r).sum())
    n_near = int(((sd - cut_d).abs() <= win).sum())
    return {
        "rank_d": rank_d,
        "rank_r": rank_r,
        "rank_cut_d": cut_d,
        "rank_cut_r": cut_r,
        "rank_window": win,
        "rank_n_near": n_near,
        "rank_difference": rank_r - rank_d,
        "passes": abs(rank_r - rank_d) <= n_near,
    }


def construction_items(orth, recon, pre_norm, pre_spec, rank_passes, tol=F10_TOL):
    """F10.3(i) item verdicts; a missing or nonfinite metric fails its item."""
    return {
        "orthogonality": _le(orth, tol),
        "reconstruction": _le(recon, tol),
        "norm": _le(pre_norm, tol),
        "spectrum": _le(pre_spec, tol),
        "rank_budget": rank_passes is True,
    }


def rounding_bound(W_bf16, R_raw, R_eff):
    """F10.3(iv), entrywise in fp64: |R_eff - R| <= min(|R|, 1/2 ulp_bf16(W + R_eff)) + ulp_fp32(fl32(W + R)).

    The bf16 ulp is taken at the rounded value W + R_eff (the result grid: ulp(W) is too small when W + R
    crosses up a binade). The last term is one fp32 spacing at the registered fp32 sum fl32(W + R) that
    apply_and_round rounds to bf16: that double rounding can exceed the single-rounding bound by less than
    one fp32 spacing (F10.12(c)). Nonfinite inputs or a nonfinite fp32 sum fail closed. Diagnostics:
    ``n_double_rounding`` counts entries above the single-rounding term, ``max_excess_fp32_spacings`` the
    largest excess over it in fp32 spacings, ``n_exceed_half_ulp_w`` entries above the literal
    1/2 ulp_bf16(W) plus the same fp32 spacing."""
    out = {
        "passes": False,
        "n_violations": None,
        "nonfinite": True,
        "max_ratio_to_half_ulp": None,
        "max_excess_fp32_spacings": None,
        "n_double_rounding": None,
        "n_exceed_half_ulp_w": None,
    }
    if not _all_finite(W_bf16, R_raw, R_eff) or not (W_bf16.shape == R_raw.shape == R_eff.shape):
        return out
    y32 = W_bf16.float() + R_raw.float()  # the registered fp32 addition
    if not _all_finite(y32):
        return out
    w, r, e = W_bf16.double(), R_raw.double(), R_eff.double()
    err = (e - r).abs()
    half = 0.5 * ulp_bf16(w + e)
    spacing = ulp_fp32(y32)
    single = torch.minimum(r.abs(), half)
    ok = err <= single + spacing
    violations = int((~ok).sum())
    empty = err.numel() == 0
    return {
        "passes": violations == 0,
        "n_violations": violations,
        "nonfinite": False,
        "max_ratio_to_half_ulp": 0.0 if empty else float((err / half).max()),
        "max_excess_fp32_spacings": 0.0 if empty else float(((err - single) / spacing).max()),
        "n_double_rounding": int((err > single).sum()),
        "n_exceed_half_ulp_w": int((err > 0.5 * ulp_bf16(w) + spacing).sum()),
    }


def sub_resolution_guard(d_norms, r_eff_norms, tol=F10_GUARD_TOL):
    """F10.3(v), per edit: |‖R_eff‖ − ‖D‖| / ‖D‖ <= tol with Frobenius norms pooled over the edited tensors
    (sqrt of the summed squares). Two-sided. Missing, nonfinite or zero-norm inputs fail closed."""
    out = {"tol": tol, "pooled_d_norm": None, "pooled_r_eff_norm": None}
    out |= {"mismatch_signed": None, "mismatch": None, "passes": False}
    d_norms, r_eff_norms = list(d_norms), list(r_eff_norms)
    values = d_norms + r_eff_norms
    if not d_norms or len(d_norms) != len(r_eff_norms) or not all(map(_finite_number, values)):
        return out
    d = math.sqrt(sum(x * x for x in d_norms))
    e = math.sqrt(sum(x * x for x in r_eff_norms))
    out |= {"pooled_d_norm": d, "pooled_r_eff_norm": e}
    if not d > 0:
        return out
    signed = (e - d) / d
    return out | {"mismatch_signed": signed, "mismatch": abs(signed), "passes": abs(signed) <= tol}


def scorer_roundtrip(base, delta, intended):
    """F10.3(iii) second part: the written delta, reloaded, has the bits of intended − base (fp32) and the
    scorer's bf16(base + delta) has the bits of the intended state. Missing, mistyped or misshapen deltas
    fail closed."""
    if not isinstance(delta, torch.Tensor):
        return {"passes": False, "mismatches": None, "delta_bits_equal": False, "detail": "missing"}
    if delta.dtype != torch.float32 or delta.shape != base.shape:
        return {
            "passes": False,
            "mismatches": None,
            "delta_bits_equal": False,
            "detail": "dtype_or_shape",
        }
    expected = intended.float() - base.float()
    delta_ok = bits_equal(delta, expected)
    scored = (base.float() + delta).to(base.dtype)
    mismatches = int((bit_pattern(scored) != bit_pattern(intended.to(base.dtype))).sum())
    return {
        "passes": delta_ok and mismatches == 0,
        "mismatches": mismatches,
        "delta_bits_equal": delta_ok,
        "detail": None,
    }


def noise_calibrated(norm_signed, spectrum_mismatch, eps_hat, noise_rank_d, noise_rank_r):
    """F10.4/F10.5(e) noise-calibrated realised audit (diagnostic): |norm| <= 2e-2 + eps_hat,
    spectrum <= 5e-2 + eps_hat, ranks above max(1e-3 s1, 3 tau) within +-1."""
    finite = all(map(_finite_number, (norm_signed, spectrum_mismatch, eps_hat)))
    items = {
        "norm": finite and abs(norm_signed) <= 2e-2 + eps_hat,
        "spectrum": finite and spectrum_mismatch <= 5e-2 + eps_hat,
        "rank": isinstance(noise_rank_d, int) and isinstance(noise_rank_r, int)
        and abs(noise_rank_r - noise_rank_d) <= 1,
    }  # fmt: skip
    return items | {"passes": all(items.values())}


def f10_verdict(a):
    """Tensor verdict from an audit dict, failing closed: a check passes only if its boolean is ``True``
    and every item it needs (F10_REQUIRED) is present and finite. ``a`` not a dict (a missing audit) fails
    every check. Returns passes, failed_checks (F10_CHECKS order), reason (first failure) and detail."""
    if not isinstance(a, dict):
        return {
            "passes": False,
            "failed_checks": list(F10_CHECKS),
            "reason": F10_REASONS[F10_CHECKS[0]],
            "detail": ["audit_missing"],
        }
    checks = a.get("checks") if isinstance(a.get("checks"), dict) else {}
    failed, detail = [], []
    for name in F10_CHECKS:
        ok = checks.get(name)
        if not isinstance(ok, bool):
            detail.append(f"missing_check:{name}")
            ok = False
        for item in F10_REQUIRED[name]:
            v = a.get(item)
            if v is None:
                detail.append(f"missing:{item}")
                ok = False
            elif not isinstance(v, bool) and not _finite_number(v):
                detail.append(f"nonfinite:{item}")
                ok = False
        if not ok:
            failed.append(name)
    return {
        "passes": not failed,
        "failed_checks": failed,
        "reason": F10_REASONS[failed[0]] if failed else None,
        "detail": detail or None,
    }


def json_safe(obj, path="", bad=None):
    """Replace nonfinite floats by None (recording their paths in ``bad``): no NaN reaches JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        if bad is not None:
            bad.append(path)
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v, f"{path}.{k}" if path else str(k), bad) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, f"{path}[{i}]", bad) for i, v in enumerate(obj)]
    return obj


def _fro64(x):
    return float(torch.linalg.vector_norm(x.double()))


def _rank(s, rtol):
    return int((s > s[0] * rtol).sum()) if s.numel() and s[0] > 0 else 0


def _svd64(x):
    """Full-SVD factors of an fp64 block (separate so tests can force a reconstruction failure)."""
    return torch.linalg.svd(x, full_matrices=False)


def f10_tensor_audit(
    D, W_bf16, R_raw, R_eff, rows, cols, seed, *, written=_BUILT, scorer=None, replayed=None,
    raw_delta=None, frames=None, tol=F10_TOL, rank_rtol=F10_RANK_RTOL, rank_window=F10_RANK_WINDOW,
    threads=F10_THREADS,
):  # fmt: skip
    """F10.3 per-tensor checks (i)-(iv) plus the F10.4 realised diagnostics (JSON-able dict).

    ``D`` is the realised increment at the predecessor ``W_bf16``; ``R_raw`` the fp32 construction and
    ``R_eff`` its realised increment bf16(W + R) − W as built. ``written`` is the R_eff reloaded from the
    written file (root ``delta.safetensors``, chain ``increment.safetensors``); ``None`` means missing; by
    default the in-memory ``R_eff`` stands in (unit use). (ii), (iv) and the diagnostics are computed on the
    written bytes when they are a readable fp32 tensor of D's shape, else on ``R_eff``. ``scorer`` is
    (base, reloaded written delta) for the scorer round trip; default (W, written), the root convention.
    ``replayed``: a precomputed :func:`f10_replay` result. ``raw_delta`` (roots) adds eps_D. ``passes``,
    ``failed_checks`` and ``reason`` come from :func:`f10_verdict`; ``checks`` holds the check booleans."""
    idx = (rows[:, None], cols)
    m, n = len(rows), len(cols)
    if written is _BUILT:
        written = R_eff
    readable = (
        isinstance(written, torch.Tensor)
        and written.dtype == torch.float32
        and written.shape == D.shape
    )
    audited = written if readable else R_eff
    if replayed is None:
        replayed = f10_replay(D, W_bf16, rows, cols, seed, frames, threads)
    R_rep, state_rep, R_eff_rep, q_out, q_in = replayed

    # (iii) replay against the written bytes, and the scorer's arithmetic on the written delta
    replay_bits = readable and bits_equal(written, R_eff_rep)
    replay_detail = None if readable else ("missing" if written is None else "dtype_or_shape")
    base_s, delta_s = scorer if scorer is not None else (W_bf16, written)
    rt = scorer_roundtrip(base_s, delta_s, state_rep)

    # (i) construction, fp64
    d64 = D.double()[idx]
    r64 = R_raw.double()[idx]
    d_norm = _fro64(d64)
    orth = recon = pre_norm = pre_spec = residual = spec_diff = spec_ref = None
    rank = rank_budget(torch.zeros(0), torch.zeros(0))
    if _all_finite(d64, r64, q_out, q_in) and d_norm > 0:
        orth = 0.0
        for q in (q_out, q_in):
            k = q.shape[0]
            q = q.double()
            orth = max(orth, _fro64(q.T @ q - torch.eye(k, dtype=torch.float64)) / math.sqrt(k))
        u, sd64, vh = _svd64(d64)
        recon = _fro64(u @ torch.diag(sd64) @ vh - d64) / d_norm
        del u, vh
        sr64 = torch.linalg.svdvals(r64)
        pre_norm = abs(_fro64(r64) - d_norm) / d_norm
        spec_diff, spec_ref = _fro64(sr64 - sd64), _fro64(sd64)
        pre_spec = spec_diff / spec_ref
        rank = rank_budget(sd64, sr64, rank_rtol, rank_window)
        residual = _fro64(r64 - q_out.double() @ d64 @ q_in.double().T) / d_norm
    items = construction_items(orth, recon, pre_norm, pre_spec, rank["passes"], tol)

    # (ii) support: D, R and the written R_eff confined to the block (NaN counts as nonzero)
    outside = torch.ones(D.shape, dtype=torch.bool)
    outside[idx] = False
    nz_d, nz_r, nz_e = (int((x[outside] != 0).sum()) for x in (D, R_raw, audited))

    # (iv) rounding bound over the whole tensor
    bound = rounding_bound(W_bf16, R_raw, audited)

    checks = {
        "construction": all(items.values()),
        "support": nz_d == 0 and nz_r == 0 and nz_e == 0,
        "replay": bool(replay_bits),
        "scorer_roundtrip": rt["passes"],
        "rounding_bound": bound["passes"],
    }

    # F10.4 diagnostics (not exclusion criteria), on the audited (written) bytes
    spectra, realised = {}, None
    try:
        realised = audit(D, audited, rows, cols, rank_rtol=rank_rtol, spectra=spectra)
    except (ValueError, RuntimeError):
        spectra = {}
    r_eff_norm = _fro64(audited)
    ulp_w = ulp_bf16(W_bf16[idx])
    sum_ulp2 = float((ulp_w**2).sum())
    kappa = d_norm / math.sqrt(sum_ulp2) if sum_ulp2 > 0 else None
    eps_hat = min(1.0, 1.0 / (math.sqrt(12) * kappa)) if kappa else 1.0
    tau = math.sqrt(sum_ulp2 / (m * n) / 12) * (math.sqrt(m) + math.sqrt(n))
    noise_rank_d = noise_rank_r = spectrum_diff = spectrum_ref = None
    if spectra:
        sd, sr = spectra["sd"], spectra["sr"]
        noise_rank_d = int((sd > max(NOISE_RANK_C * tau, rank_rtol * float(sd[0]))).sum())
        noise_rank_r = (
            int((sr > max(NOISE_RANK_C * tau, rank_rtol * float(sr[0]))).sum()) if sr[0] > 0 else 0
        )
        spectrum_diff, spectrum_ref = _fro64(sr - sd), _fro64(sd)
    norm_signed = (r_eff_norm - d_norm) / d_norm if d_norm > 0 else None
    noise = noise_calibrated(
        norm_signed,
        realised["spectrum_mismatch"] if realised else None,
        eps_hat,
        noise_rank_d,
        noise_rank_r,
    )
    out = {
        "checks": checks,
        "construction_items": items,
        # (i)
        "orthogonality_error": orth,
        "reconstruction_residual": recon,
        "prerounding_norm_mismatch": pre_norm,
        "prerounding_spectrum_mismatch": pre_spec,
        "prerounding_spectrum_diff_norm": spec_diff,
        "prerounding_spectrum_ref_norm": spec_ref,
        "prerounding_rank_d": rank["rank_d"],
        "prerounding_rank_r": rank["rank_r"],
        "prerounding_rank_difference": rank["rank_difference"],
        "rank_cut_d": rank["rank_cut_d"],
        "rank_cut_r": rank["rank_cut_r"],
        "rank_window": rank["rank_window"],
        "rank_n_near": rank["rank_n_near"],
        "rank_budget_passes": rank["passes"],
        "construction_residual": residual,
        "construction_passes": checks["construction"],
        "block_rank": rank["rank_d"],
        "block_shape": [m, n],
        # (ii)
        "support_identical": checks["support"],
        "nonzero_d_outside_block": nz_d,
        "nonzero_r_raw_outside_block": nz_r,
        "nonzero_r_eff_outside_block": nz_e,
        # (iii)
        "replay_bits_equal": bool(replay_bits),
        "replay_detail": replay_detail,
        "replay_threads": threads,
        "prerounding_replay_bits_equal": bits_equal(R_raw.float(), R_rep),
        "scorer_roundtrip_mismatches": rt["mismatches"],
        "scorer_delta_bits_equal": rt["delta_bits_equal"],
        "scorer_detail": rt["detail"],
        "replay_passes": checks["replay"] and checks["scorer_roundtrip"],
        # (iv)
        "rounding_bound_passes": bound["passes"],
        "rounding_bound_violations": bound["n_violations"],
        "rounding_bound_nonfinite": bound["nonfinite"],
        "rounding_bound_max_ratio": bound["max_ratio_to_half_ulp"],
        "rounding_bound_max_excess_fp32_spacings": bound["max_excess_fp32_spacings"],
        "n_double_rounding": bound["n_double_rounding"],
        "n_exceed_half_ulp_w": bound["n_exceed_half_ulp_w"],
        # registered realised audit, reported only
        "norm_mismatch": realised["norm_mismatch"] if realised else None,
        "norm_mismatch_signed": norm_signed,
        "spectrum_mismatch": realised["spectrum_mismatch"] if realised else None,
        "spectrum_diff_norm": spectrum_diff,
        "spectrum_ref_norm": spectrum_ref,
        "rank_d": realised["rank_d"] if realised else None,
        "rank_r": realised["rank_r"] if realised else None,
        "rank_difference": realised["rank_difference"] if realised else None,
        "s31_passes": bool(realised["passes"]) if realised else False,
        # rounding scale and noise-calibrated audit (design note §1.2, remedy (b) at c = 3, +-1)
        "kappa": kappa,
        "eps_hat": eps_hat,
        "sum_ulp2_block": sum_ulp2,
        "rounding_error_rel": _fro64(audited.double() - R_raw.double()) / d_norm
        if d_norm > 0
        else None,
        "eps_D": _fro64(D.double() - raw_delta.double()) / d_norm
        if raw_delta is not None and d_norm > 0
        else None,
        "tau": tau,
        "noise_rank_c": NOISE_RANK_C,
        "noise_rank_d": noise_rank_d,
        "noise_rank_r": noise_rank_r,
        "noise_calibrated": {k: noise[k] for k in ("norm", "spectrum", "rank")},
        "noise_calibrated_passes": noise["passes"],
        "d_norm": d_norm,
        "r_norm": _fro64(R_raw),
        "r_eff_norm": r_eff_norm,
        "energy_ratio": (r_eff_norm / d_norm) ** 2 if d_norm > 0 else None,
    }
    bad = []
    out = json_safe(out, bad=bad)
    out["nonfinite_fields"] = bad
    v = f10_verdict(out)
    out |= {
        "passes": v["passes"],
        "failed_checks": v["failed_checks"],
        "reason": v["reason"],
        "fail_closed_detail": v["detail"],
    }
    return out
