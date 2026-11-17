"""SYNTHESIS-r9 C2 red-team controls as library functions, to the Board A contract (conditions 5, 6, 10).

* **Fixed-endpoint bookkeeping control** (:func:`consistent_decompositions`, :func:`fixed_endpoint_control`;
  Board A condition 5). Eligible states: held-out trajectories with h in {3, 4}. Alternates are *consistent*
  decompositions of the same realised final edit: step reorderings and (optionally) split steps (one step of dose d
  -> two steps of d/2), kept only when ``realise_final(actions)`` equals the original exactly (noncommuting or
  bf16-rounded reorderings are dropped, never forced). Bookkeeping is recomputed by ``retoken`` (consistent
  overlaps and norms). The label y_T is held fixed; intermediate labels of an alternate path are unknown (NaN).
  The static prediction at T must be unchanged (``static_invariant``; a cumulative predictor is a function of the
  final edit only). Statistic: C_T on the alternates (WM error averaged over alternates vs the invariant static
  error) with the component-cluster bootstrap-t lower bound. Proposed pass rule (for PREREG v1): the static
  prediction is invariant **and** LB(C_T on alternates) > 0, i.e. the WM advantage survives re-presentation of the
  same endpoint; if it fails, H-T and every E2 claim that relies on WM-E's trajectory model are void.
* **Corruption diagnostic** (:func:`shuffle_bookkeeping`, :func:`corruption_diagnostic`): step order permuted and
  step-local inputs shuffled independently (inconsistent, out-of-distribution inputs). Descriptive only: it shows
  sensitivity to corrupted inputs, never learned dynamics, and gates nothing.
* **Inert edits** (Board A condition 6; :data:`INERT_SPEC`). (i) Dead vocabulary rows (:func:`dead_vocab_rows`):
  input-embedding rows with ids >= len(tokenizer) (9B: 248077..248319) are never produced by the tokenizer, and
  the 9B embedding is untied, so the edit is **mathematically** inert for every tokenised input; a tied model
  raises :class:`NotInertError`. (ii) Post-norm-cancelled scales (:func:`post_norm_cancelled_scales`): k_proj of a
  full-attention layer x2 (bf16-exact), cancelled by k_norm only up to its eps, so it needs **behavioural**
  verification (:func:`verify_inert`) at the frozen tolerance before use. q_proj is excluded (it carries the
  output gate). :func:`inert_check` requires the uncertainty (coverage) checks to have passed first and both
  |yhat| <= 2 sigma and |yhat| <= an absolute per-objective tolerance, simultaneously over every control, state
  and endpoint (inflated sigma alone cannot pass).
* **Trust-region strata** (:func:`assign_stratum`, :func:`stratum_report`): frozen assignment by the state's mean
  paired WikiText ΔNLL on the bank-label items (inside <= 0.1 < outside); a stratum with fewer than
  ``MIN_STRATUM_COMPONENTS`` components is descriptive only; a nondestructive subset (ΔNLL <= 0.01) is reported.
* **Direct best-of-N with a two-stage screen** (:func:`two_stage_best_of_n`), charged on the bundle ledger
  (``wm.plan``): a screen costs ``screen_items / full_items`` bundles.
* **Cost accounting** (:func:`cost_report`): every category of Board A condition 10, amortised and unamortised;
  reported, never gated (r9 C2; r8 D6).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from wsp.wm import actions as A
from wsp.wm import confirm_stats as CS

TRUST_REGION_DNLL = 0.1
NONDESTRUCTIVE_DNLL = 0.01
MIN_STRATUM_COMPONENTS = 5
INERT_K_SIGMA = 2.0
NULL_ALPHA = 0.05
Z95 = 1.96
FIXED_ENDPOINT_HORIZONS = (3, 4)
N_ALTERNATES = 8
CONTROL_SEED = 20261117

# Frozen inert-control specification for Qwen3.5-9B (Board A condition 6). Tolerances are in units of the pilot-2
# paired SE per objective (wm.powersim.PILOT2_PAIRED_SE): abs_tol for |yhat|, verify_tol for the measured change.
INERT_SPEC = {
    "dead_rows": {
        "tensor": "model.language_model.embed_tokens.weight",
        "rows_from": 248077,  # len(tokenizer) at the pinned revision; rows 248077..248319 are never produced
        "rows_to": 248320,
        "n_controls": 3,
        "rows_per_control": 32,
        "factor": 2.0,
        "invariance": "mathematical (untied embedding; ids never produced by the tokenizer)",
        "verify": "bitwise-equal teacher-forced panel values",
    },
    "k_norm_cancelled": {
        "tensor_role": "k (self_attn.k_proj) of full-attention layers",
        "layers": (11, 15, 19),
        "factor": 2.0,
        "invariance": "approximate (RMSNorm eps); behavioural verification required",
        "verify_tol_se": 0.1,
    },
    "endpoints": "all six trajectory endpoints",
    "abs_tol_se": 0.5,
    "k_sigma": INERT_K_SIGMA,
    "rule": "simultaneous over controls x states x endpoints; coverage check must pass first",
}


class NotInertError(ValueError):
    """The requested edit is not inert for this architecture (e.g. tied embeddings, a gated q_proj)."""


class OrderError(RuntimeError):
    """A control was run before the checks it depends on."""


# ------------------------------------------------------------------ fixed-endpoint control (valid)


def _with_path(traj, actions, retoken):
    e, x = retoken(actions)
    T = len(actions)
    y = np.full((T + 1, np.asarray(traj["y"]).shape[1]), np.nan)
    y[0], y[T] = traj["y"][0], traj["y"][traj["horizon"]]
    o0 = np.asarray(traj["obs"])[0]
    obs = np.broadcast_to(
        o0, (T + 1, *o0.shape)
    ).copy()  # only o_0 is known (and read) for an alternate path
    return dict(
        traj,
        actions=list(actions),
        e=np.asarray(e, float),
        x=np.asarray(x, float),
        y=y,
        obs=obs,
        horizon=T,
    )


def consistent_decompositions(
    traj, retoken, realise_final, n=N_ALTERNATES, seed=CONTROL_SEED, split_steps=True
) -> list:
    """Up to ``n`` alternate decompositions of the same realised final edit (module docstring)."""
    T = int(traj["horizon"])
    acts = list(traj["actions"])
    target = realise_final(acts)
    rng = np.random.default_rng(seed)
    seen, out = {tuple(map(tuple, acts))}, []
    cands = []
    for _ in range(8 * n):
        cands.append([acts[i] for i in rng.permutation(T)])
    if split_steps:
        for i in range(T):
            j, d = acts[i]
            cands.append(acts[:i] + [(j, d / 2.0), (j, d / 2.0)] + acts[i + 1 :])
    for c in cands:
        key = tuple(map(tuple, c))
        if key in seen:
            continue
        seen.add(key)
        if np.array_equal(np.asarray(realise_final(c)), np.asarray(target)):
            out.append(_with_path(traj, c, retoken))
        if len(out) >= n:
            break
    return out


def fixed_endpoint_control(
    wm,
    static,
    base_obs,
    trajs,
    retoken,
    realise_final,
    n_alt=N_ALTERNATES,
    seed=CONTROL_SEED,
    se=None,
    components=None,
    horizons=FIXED_ENDPOINT_HORIZONS,
    weights="pooled",
    static_tol=1e-6,
) -> dict:
    """WM vs static at the fixed endpoint over consistent alternate decompositions (module docstring)."""
    scale = 1.0 if se is None else np.asarray(se, float)
    e_wm, e_alt, e_st, clusters = [], [], [], []
    path_shift, static_dev, n_skipped = [], 0.0, 0
    for i, t in enumerate(trajs):
        T = t["horizon"]
        if T not in horizons:
            continue
        alts = consistent_decompositions(t, retoken, realise_final, n_alt, seed + 104729 * i)
        if not alts:
            n_skipped += 1
            continue
        y = np.asarray(t["y"][T], float)
        p_wm = _point(wm.predict(base_obs, t))[T]
        p_st = _point(static.predict(base_obs, t))[T]
        alt_wm = [_point(wm.predict(base_obs, a))[a["horizon"]] for a in alts]
        alt_st = [_point(static.predict(base_obs, a))[a["horizon"]] for a in alts]
        static_dev = max([static_dev] + [float(np.abs(q - p_st).max()) for q in alt_st])
        for k in range(len(y)):
            e_wm.append(abs(p_wm[k] - y[k]) / (scale if np.isscalar(scale) else scale[k]))
            e_alt.append(
                float(np.mean([abs(q[k] - y[k]) for q in alt_wm]))
                / (scale if np.isscalar(scale) else scale[k])
            )
            e_st.append(abs(p_st[k] - y[k]) / (scale if np.isscalar(scale) else scale[k]))
            clusters.append(i if components is None else components[i])
        path_shift.append(float(np.mean([np.abs(q - p_wm).mean() for q in alt_wm])))
    if not e_wm:
        raise ValueError("no eligible state has a consistent alternate decomposition")
    lb_orig = CS.ct_lower_bound(e_wm, e_st, clusters, weights=weights, seed=seed)
    lb_alt = CS.ct_lower_bound(e_alt, e_st, clusters, weights=weights, seed=seed)
    invariant = static_dev <= static_tol  # float32 batch-shape numerics only (~1e-10 measured)
    return {
        "n_states": len(path_shift),
        "n_skipped_no_alternate": n_skipped,
        "static_invariant": bool(invariant),
        "static_max_deviation": static_dev,
        "static_tol": static_tol,
        "wm_path_shift": float(np.mean(path_shift)),
        "ct_original": lb_orig,
        "ct_alternates": lb_alt,
        "pass": bool(invariant and lb_alt["estimable"] and lb_alt["lb"] > 0),
    }


# ------------------------------------------------------------------ corruption diagnostic (descriptive)


def _non_identity_perm(rng, n):
    if n < 2:
        raise ValueError("a permutation control needs at least two steps")
    while True:
        p = rng.permutation(n)
        if (p != np.arange(n)).any():
            return p


def shuffle_bookkeeping(traj, J: int, seed: int, retoken) -> dict:
    """The shuffled-bookkeeping version of ``traj`` (see module docstring). ``retoken(actions) -> (e, x)``."""
    T = int(traj["horizon"])
    if T < 2:
        raise ValueError("the shuffled-bookkeeping null needs horizon >= 2")
    rng = np.random.default_rng(seed)
    pi = _non_identity_perm(rng, T)
    actions = [traj["actions"][i] for i in pi]
    e, x = retoken(actions)
    e, x = np.array(e, dtype=float), np.array(x, dtype=float)
    if not np.allclose(x[T, :J], traj["x"][T, :J], rtol=0, atol=1e-9):
        raise AssertionError("permuting steps changed the cumulative edit E_T")
    x[T, :J] = traj["x"][T, :J]
    x[T, -1] = traj["x"][T, -1]
    sigma = _non_identity_perm(rng, T)
    local = np.concatenate([x[1:, J : 2 * J], x[1:, -2:-1]], -1)[sigma]
    x[1:, J : 2 * J] = local[:, :J]
    x[1:, -2] = local[:, -1]
    y = np.full_like(np.asarray(traj["y"], dtype=float), np.nan)
    y[0], y[T] = traj["y"][0], traj["y"][T]
    return dict(
        traj, actions=actions, e=e, x=x, y=y, shuffled={"pi": pi.tolist(), "sigma": sigma.tolist()}
    )


def _point(pred):
    return pred[0] if isinstance(pred, tuple) else pred


def _cluster_boot_p(d, clusters, n_boot, seed):
    """One-sided p for mean(d) > 0: share of cluster-bootstrap means of the centred d that reach mean(d)."""
    d = np.asarray(d, float)
    keys = sorted(set(clusters), key=str)
    idx = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    sums = np.array([d[idx[k]].sum() for k in keys])
    cnts = np.array([len(idx[k]) for k in keys])
    obs = sums.sum() / cnts.sum()
    if np.allclose(d, 0):
        return 1.0, obs
    cs = sums - obs * cnts  # centred cluster totals (the null mean 0)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = cs[draw].sum(1) / cnts[draw].sum(1)
    return float((1 + np.sum(means >= obs)) / (n_boot + 1)), obs


def corruption_diagnostic(
    model,
    base_obs,
    trajs,
    J: int,
    retoken,
    n_shuffles: int = 8,
    seed: int = 20261117,
    se=None,
    components=None,
    horizons=(3, 4),
    n_boot: int = 2000,
    alpha: float = NULL_ALPHA,
) -> dict:
    """Descriptive corruption diagnostic: WM error at T on true vs corrupted bookkeeping. Gates nothing."""
    rows, clusters = [], []
    e_true, e_shuf = [], []
    for i, t in enumerate(trajs):
        T = t["horizon"]
        if T not in horizons:
            continue
        scale = 1.0 if se is None else np.asarray(se, float)

        def err(tr, T=T, y=t["y"][T], scale=scale):
            return float(np.mean(np.abs(_point(model.predict(base_obs, tr))[T] - y) / scale))

        a = err(t)
        b = float(
            np.mean(
                [
                    err(shuffle_bookkeeping(t, J, seed + 7919 * i + s, retoken))
                    for s in range(n_shuffles)
                ]
            )
        )
        e_true.append(a)
        e_shuf.append(b)
        rows.append(b - a)
        clusters.append(i if components is None else components[i])
    if not rows:
        raise ValueError("no held-out trajectories at the null's horizons")
    p, md = _cluster_boot_p(rows, clusters, n_boot, seed)
    return {
        "n": len(rows),
        "n_clusters": len(set(clusters)),
        "horizons": sorted({t["horizon"] for t in trajs if t["horizon"] in horizons}),
        "mean_orig": float(np.mean(e_true)),
        "mean_shuf": float(np.mean(e_shuf)),
        "mean_diff": float(md),
        "p": p,
        "alpha": alpha,
        "sensitive": bool(p < alpha and md > 0),
        "gates": None,
    }


# ------------------------------------------------------------------ inert edits


def dead_vocab_rows(
    tensor: str,
    rolemap: A.RoleMap,
    tokenizer_len: int,
    n_rows: int,
    seed: int,
    factor=2.0,
    used_ids=(),
):
    """row_scale on ``n_rows`` seeded input-embedding rows with ids >= ``tokenizer_len`` (never produced)."""
    if rolemap.tied:
        raise NotInertError(
            "tied embeddings: an embedding row is also an LM-head row, so it is not inert"
        )
    used = {int(i) for i in used_ids}
    free = [i for i in range(int(tokenizer_len), rolemap.vocab) if i not in used]
    if len(free) < n_rows:
        raise NotInertError(
            f"only {len(free)} never-produced rows (tokenizer {tokenizer_len}, vocab {rolemap.vocab})"
        )
    rng = np.random.default_rng(seed)
    rows = sorted(rng.choice(free, size=n_rows, replace=False).tolist())
    return A.row_scale(tensor, rows, factor=factor, dose=1.0)


def post_norm_cancelled_scales(rolemap: A.RoleMap, layers, factor=2.0) -> list:
    """k_proj x ``factor`` on every full-attention layer in ``layers`` (cancelled by k_norm; bf16-exact for 2)."""
    out = [
        A.scale(rolemap.name(L, "k"), factor=factor, dose=1.0)
        for L in layers
        if rolemap.layer_types[L] == "full_attention"
    ]
    if not out:
        raise NotInertError("no full-attention layer (k_proj + k_norm) among the requested layers")
    return out


def verify_inert(base: dict, edited: dict, tol: dict, mathematical: bool = False) -> dict:
    """Behavioural verification of an inert edit on paired per-item values ``{objective: [values]}``: the mean
    paired change must satisfy |mean| <= tol[objective] (``mathematical``: bitwise-equal values required)."""
    out, ok = {}, True
    for o, b in base.items():
        d = np.asarray(edited[o], float) - np.asarray(b, float)
        if mathematical:
            good = bool(np.array_equal(np.asarray(edited[o]), np.asarray(b)))
        else:
            good = bool(abs(d.mean()) <= tol[o])
        out[o] = {
            "mean_change": float(d.mean()),
            "max_abs_change": float(np.abs(d).max()),
            "pass": good,
        }
        ok = ok and good
    return {"objectives": out, "pass": ok, "mathematical": mathematical}


def synthetic_inert_world(seed=0, n_dead=2, n_cancel=1, **kw):
    """A ``SynthWorld`` whose last dictionary directions are inert: ``n_dead`` p q^T with q orthogonal to the
    input subspace (dead rows), ``n_cancel`` along W1 itself (cancelled by the hidden RMSNorm up to eps)."""
    from wsp.wm import synthetic as S

    w = S.SynthWorld(kind="saturating", seed=seed, **kw)
    g = torch.Generator().manual_seed(9000 + seed)
    span = torch.cat([w.U.reshape(-1, w.d_in), w.probes])
    _, _, Vh = torch.linalg.svd(span, full_matrices=True)
    rank = int((torch.linalg.svdvals(span) > 1e-9).sum())
    null = Vh[rank:]
    if len(null) == 0:
        raise ValueError("input subspace is full rank; use input_rank < d_in")
    E = w.E.clone()
    inert = {}
    k = n_dead + n_cancel
    for i in range(n_dead):
        p = torch.randn(w.m, generator=g, dtype=torch.float64)
        e = torch.outer(p, null[i % len(null)])
        E[w.J - k + i] = e / e.norm()
        inert[w.J - k + i] = "dead"
    for i in range(n_cancel):
        E[w.J - n_cancel + i] = w.W1 / w.W1.norm()
        inert[w.J - n_cancel + i] = "cancel"
    w.E = E
    w._g0, w._H0 = w._grad_hess(w._coef0)
    w._y0 = w._true_losses(w._coef0)
    w.inert = inert
    return w


def inert_check(preds, abs_tol, coverage_passed: bool, k: float = INERT_K_SIGMA) -> dict:
    """``preds``: (mean [T+1,K], sigma [T+1,K]) per inert trajectory, sigma from the *calibrated* predictor under
    test. Pass when, simultaneously at every nonzero state and objective, |yhat| <= k sigma and |yhat| <= abs_tol[k]
    (the zero-edit rows, yhat = sigma = 0 exactly, are skipped). Refuses to run before the coverage check passed."""
    if not coverage_passed:
        raise OrderError("inert-edit check needs the uncertainty (coverage) checks to pass first")
    tol = np.asarray(abs_tol, float)
    ratios, absok = [], []
    for mean, sigma in preds:
        mean, sigma = np.asarray(mean, float), np.asarray(sigma, float)
        live = sigma > 0
        if (~live & (mean != 0)).any():
            ratios.append(np.inf)
            absok.append(False)
        ratios.extend((np.abs(mean[live]) / sigma[live]).tolist())
        absok.extend((np.abs(mean) <= tol)[live].tolist())
    r = np.asarray(ratios)
    n = len(r)
    sig_ok = bool(n == 0 or (r <= k).all())
    abs_ok = bool(all(absok))
    return {
        "n": n,
        "k": k,
        "frac_within": float(np.mean(r <= k)) if n else float("nan"),
        "max_ratio": float(r.max()) if n else 0.0,
        "sigma_rule": sig_ok,
        "abs_rule": abs_ok,
        "pass": sig_ok and abs_ok,
    }


# ------------------------------------------------------------------ trust region


def assign_stratum(dnll: float, thresh=TRUST_REGION_DNLL) -> str:
    """Frozen rule: the state's mean paired WikiText ΔNLL on the bank-label items."""
    return "inside" if dnll <= thresh else "outside"


def trust_region_strata(rows, key="dnll", thresh=TRUST_REGION_DNLL) -> dict:
    return {
        "inside": [r for r in rows if assign_stratum(r[key], thresh) == "inside"],
        "outside": [r for r in rows if assign_stratum(r[key], thresh) == "outside"],
    }


def by_region(rows, fn, key="dnll", thresh=TRUST_REGION_DNLL) -> dict:
    s = trust_region_strata(rows, key, thresh)
    return {"inside": fn(s["inside"]), "outside": fn(s["outside"]), "all": fn(rows)}


def stratum_report(
    rows, key="dnll", comp_key="component", min_components=MIN_STRATUM_COMPONENTS
) -> dict:
    """Counts per stratum and whether inference is registered (>= ``min_components``) or descriptive only;
    plus the nondestructive subset (ΔNLL <= 0.01)."""
    s = trust_region_strata(rows, key)
    s["nondestructive"] = [r for r in rows if r[key] <= NONDESTRUCTIVE_DNLL]
    out = {}
    for name, rs in s.items():
        nc = len({r[comp_key] for r in rs})
        out[name] = {
            "n_rows": len(rs),
            "n_components": nc,
            "inference": "registered" if nc >= min_components else "descriptive (sparse)",
        }
    return out


# ------------------------------------------------------------------ direct best-of-N


def screen_units(screen_items: int, full_items: int) -> float:
    return float(screen_items) / float(full_items)


def two_stage_best_of_n(
    candidates,
    screen,
    full,
    ledger,
    arm,
    screen_items: int = 4,
    full_items: int = 128,
    n_full: int = 16,
    screen_seconds: float = 0.0,
    full_seconds: float = 0.0,
    z: float = Z95,
    wiki_limit: float = 0.01,
) -> dict:
    """Two-stage direct search with no world model (r9 C2; challenger 4.1).

    Stage 1 screens candidates in order while the budget allows ``n_full`` full evaluations afterwards; a candidate
    survives when its screen gain > 0 and its screen WikiText ΔNLL <= ``wiki_limit`` (the §5 guardrail). Stage 2 evaluates the best ``n_full``
    survivors (by screen gain) on the full dev panel. The winner is the full-evaluated candidate with the largest
    gain LCB (gain - z se) among those with LCB > 0 and no guardrail breach; ``None`` means base."""
    cu = screen_units(screen_items, full_items)
    reserve = float(n_full)
    screened, survivors = [], []
    for c in candidates:
        if ledger.remaining(arm) - cu < reserve - 1e-12:
            break
        s = screen(c)
        ledger.charge(arm, c, screen_seconds, kind="screen", units=cu)
        screened.append(c)
        if s["gain"] > 0 and s["wikitext_dnll"] <= wiki_limit:
            survivors.append((s["gain"], c))
    survivors.sort(key=lambda gc: -gc[0])
    evaluated, best, best_lcb = [], None, -math.inf
    for _, c in survivors[:n_full]:
        if ledger.remaining(arm) < 1.0 - 1e-12:
            break
        r = full(c)
        ledger.charge(arm, c, full_seconds, kind="evaluated", units=1.0)
        evaluated.append(c)
        lcb = r["gain"] - z * r["se"]
        if not r["breaches"] and lcb > 0 and lcb > best_lcb:
            best, best_lcb = c, lcb
    return {
        "n_screened": len(screened),
        "n_survivors": len(survivors),
        "n_full": len(evaluated),
        "full_evaluated": evaluated,
        "winner": best,
        "winner_lcb": best_lcb if best is not None else None,
        "screen_units": cu,
    }


# ------------------------------------------------------------------ cost accounting

COST_CATEGORIES = (
    "bank_gpu_s",  # trajectory bank construction (states, ladders, compositions)
    "grounding_gpu_s",  # retained grounding labels: taps, per-item losses, oracle slopes A_t
    "direction_gpu_s",  # contrast directions and adapter (LoRA) training
    "feature_gpu_s",  # R0 field / feature collection at theta0
    "failed_gpu_s",  # failed or aborted attempts
    "eval_gpu_s",  # real-evaluation bundles charged to the arm
    "generation_gpu_s",  # generation audits
    "train_gpu_s",  # arm-specific training (e.g. LoRA SFT competitor)
    "planner_cpu_s",  # R0 planning compute (CPU)
    "predictor_cpu_s",  # predictor fitting (CPU)
)
SHARED = ("bank_gpu_s", "grounding_gpu_s", "feature_gpu_s")


def cost_report(arms: dict, shared: dict, n_episodes: int) -> dict:
    """Every Board A condition-10 category. ``shared`` holds the costs of shared artefacts (bank, grounding,
    features); an arm with ``uses_bank`` is charged them in full (unamortised) or divided by ``n_episodes``
    (amortised). LLM tokens are reported separately. Reported, not gated."""
    bank = sum(shared.get(c, 0.0) for c in SHARED) / 3600.0
    out = {}
    for name, a in arms.items():
        unknown = set(a) - set(COST_CATEGORIES) - {"gain", "uses_bank", "llm_tokens"}
        if unknown:
            raise KeyError(f"{name}: unknown cost categories {sorted(unknown)}")
        gpu = sum(a.get(c, 0.0) for c in COST_CATEGORIES if c.endswith("gpu_s")) / 3600.0
        cpu = sum(a.get(c, 0.0) for c in COST_CATEGORIES if c.endswith("cpu_s")) / 3600.0
        b = bank if a.get("uses_bank") else 0.0
        am, un = gpu + b / max(1, n_episodes), gpu + b
        out[name] = {
            "gain": a["gain"],
            "own_gpu_h": gpu,
            "cpu_h": cpu,
            "shared_gpu_h": b,
            "amortised_gpu_h": am,
            "unamortised_gpu_h": un,
            "gain_per_gpu_h_amortised": a["gain"] / am if am > 0 else float("nan"),
            "gain_per_gpu_h_unamortised": a["gain"] / un if un > 0 else float("nan"),
            "llm_tokens": int(a.get("llm_tokens", 0)),
            "by_category": {c: float(a.get(c, 0.0)) for c in COST_CATEGORIES},
        }
    out["_meta"] = {
        "shared_gpu_h": bank,
        "amortisation_denominator": n_episodes,
        "gated": False,
        "categories": list(COST_CATEGORIES),
    }
    return out
