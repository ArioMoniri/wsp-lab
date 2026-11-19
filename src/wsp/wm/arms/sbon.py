"""S-BoN: direct multi-fidelity best-of-N at total-compute parity (PREREG_WME §II.5, §II.21.4, §II.22.7; Board A v2
required change 7). No predictor; the registered two-rung runner, simulated in stage W (``powersim_w._arm``).

* **Budget.** ``B_par`` full-evaluation equivalents per episode, fixed from measured ledger seconds before the arm
  runs (``prereg_v2.parity_budget``); a non-integer value is used as is.
* **Unit costs** (registered unit times ``T_BUNDLE_S``, ``OVERHEAD_S``; never re-measured): a screen on the first
  :data:`N_SCREEN` = 16 dev groups per objective costs ``c16 = (16/64 (t_bundle - overhead) + overhead) / t_bundle``
  bundles; a promotion scores the **remaining 48 groups only** (the 16 screened groups are reused, not repeated)
  and re-applies the edit: ``inc = 1 - c16 + overhead / t_bundle`` (``powersim_w.sh_costs`` / ``promote_cost``).
* **Screen count.** K = the largest integer with ``K c16 + ceil(K / 4) inc <= B_par`` (``powersim_w.screened``);
  P = ceil(K / 4) promotion slots. The fractional residual ``B_par - K c16 - P inc`` is never spent; it is recorded
  as forfeited. Unused promotion slots (fewer than P eligible candidates) are forfeited too, never re-spent.
* **Candidates.** The arm's own seeded stream (``sha256("wme-c1:bon:<episode index>")``): a length L uniform on
  1..4, then L distinct primitives uniformly from the episode subsample, in order (Part I §6.5.3 pool rule). A draw
  whose candidate key or realised theta-hash was already screened by this arm is a cache hit: 0 units, not counted
  in K, and the next draw is taken (at most :data:`MAX_DRAWS_PER_SCREEN` draws per screen, then the rest of the
  budget is forfeited).
* **Promotion.** Among screened candidates without a dev guard point breach on the screened groups, the top P by
  screened point gain; ties by earlier draw. **Winner**: ``e2.winner`` over the promoted candidates' full 64-group
  records (dev bounds pass and LCB > 0, largest LCB; ties by smaller edit norm, then earlier promotion), else the
  base. Its single sealed-test read is the guarded score of Part I §6.8.1.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from wsp.wm import e2 as E2
from wsp.wm import powersim_w as PW

T_BUNDLE_S = 14.579798132181168  # registered (v2b price t_bundle_s)
OVERHEAD_S = 1.030511042901448  # registered (cost.unit_times overhead_s)
N_SCREEN, N_DEV = 16, 64
MAX_DRAWS_PER_SCREEN = 50


def costs(t_bundle: float = T_BUNDLE_S, overhead: float = OVERHEAD_S) -> dict:
    c16, _ = PW.sh_costs(t_bundle, overhead, n_dev=N_DEV, n_screen=N_SCREEN)
    return {"screen": c16, "promote": PW.promote_cost(t_bundle, overhead)}


def plan_counts(budget: float, t_bundle: float = T_BUNDLE_S, overhead: float = OVERHEAD_S) -> dict:
    c = costs(t_bundle, overhead)
    k = PW.screened(budget, t_bundle, overhead)
    p = math.ceil(k / 4)
    spent = k * c["screen"] + p * c["promote"]
    return {"K": k, "P": p, "planned_units": spent, "residual": budget - spent, **c}


def stream_seed(episode_index: int) -> int:
    return int(hashlib.sha256(f"wme-c1:bon:{episode_index}".encode()).hexdigest()[:16], 16)


def draw_candidate(rng, subsample) -> tuple:
    L = int(rng.integers(1, E2.MAX_PRIMITIVES + 1))
    L = min(L, len(subsample))
    idx = rng.choice(len(subsample), size=L, replace=False)
    return tuple(subsample[i] for i in idx)


def run_sbon(world, episode: dict, budget: float, ledger, arm: str = "sbon") -> dict:
    """``world.theta_hash(pids)`` (R0, no evaluation); ``world.screen(pids) -> {gain16, guard16_ok, seconds}`` (first
    16 dev groups per objective);
    ``world.complete(pids) -> {gain, gain_se | lcb, guards_pass, rel_norm, seconds}`` over all 64 groups (scores the
    remaining 48 only)."""
    pc = plan_counts(budget)
    rng = np.random.default_rng(stream_seed(int(episode["index"])))
    seen_keys, seen_theta, screened = set(), set(), []
    cache_hits = 0
    for _ in range(pc["K"]):
        for _draw in range(MAX_DRAWS_PER_SCREEN):
            pids = draw_candidate(rng, list(episode["subsample"]))
            key = E2.candidate_key(pids)
            if key in seen_keys:
                cache_hits += 1
                continue
            th = world.theta_hash(pids)  # R0: apply and hash, no evaluation
            if th in seen_theta:
                seen_keys.add(key)
                cache_hits += 1
                continue
            break
        else:
            break  # stream exhausted: the remaining budget is forfeited
        obs = world.screen(pids)
        seen_keys.add(key)
        seen_theta.add(th)
        ledger.charge(arm, key, seconds=obs.get("seconds", 0.0), kind="screen", units=pc["screen"])
        screened.append(dict(obs, candidate=pids, draw=len(screened)))
    ok = [s for s in screened if s["guard16_ok"]]
    promote = sorted(ok, key=lambda s: (-s["gain16"], s["draw"]))[: pc["P"]]
    full = []
    for j, s in enumerate(promote):
        rec = world.complete(s["candidate"])
        ledger.charge(
            arm,
            E2.candidate_key(s["candidate"]),
            seconds=rec.get("seconds", 0.0),
            kind="screen",
            units=pc["promote"],
        )
        full.append(dict(rec, candidate=s["candidate"], order=j))
    used = len(screened) * pc["screen"] + len(full) * pc["promote"]
    ledger.close(arm, f"S-BoN: {len(screened)} screens, {len(full)} promotions")
    return {
        "winner": E2.winner(full),
        "screened": len(screened),
        "promoted": len(full),
        "cache_hits": cache_hits,
        "units_used": used,
        "forfeited": budget - used,
        "plan": pc,
    }
