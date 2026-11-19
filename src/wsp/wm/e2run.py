"""The one executable H-E2 configuration (PREREG_WME §II.22.2; Board A v2 required change 2).

Every E2 planning fit and search goes through this module; nothing here is tuned on any outcome.

* **Representation.** R(E) of every *realised cumulative prefix* of a candidate, from one function ``r_of(steps)``
  (on 9B: ``wsp.wm.wmcp.r_from_contractions`` of the realised bf16 cumulative delta of the snapped steps applied in
  order from theta0 by ``EditSession.compose``; R0). The bank-label path (:func:`trajectory`) and the planner
  (``adapter9b.V0RArm._trajs``) call the same ``r_of`` on the same prefixes, so both arms and both uses receive
  identical realised cumulative information, including overlapping and non-commuting compositions.
* **WM arm.** ``adapter9b.V0RArm`` over ``model.WMEnsemble(E2_WM_CONFIG, wmcp.v0_field(spec), base_obs, 5)``: the
  history-dependent WM-E on the R rows (``wmcp.v0_trajectory_r``). **Not** ``adapter9b.PredictorArm``, whose
  ``trajectory`` adds primitive coordinates algebraically and is not used for E2.
* **Static arm.** ``adapter9b.V0RArm`` over ``model.MatchedStatic`` with the identical config, field nest, base
  observation, auxiliary targets, seeds, component bootstrap and shrinkage rule; its state depends on (o_0, R(E_t))
  only, so its prediction at T is a function of the realised final edit. There is no static-family selection in E2
  (the S1-S3 inner-fold choice is an H-T rule).
* **Slope transport and A_t.** ``slope_transport=False`` (RULING-wme-post-pilot3 P1); any ``A`` block is removed
  from the base observation and from every training observation before fitting (:func:`drop_slope_block`).
* **Shrinkage.** ``model.select_shrink``: grid ``model.SHRINK_GRID``, the component-level selection split of
  ``model.split_and_bootstrap``, mean absolute error over horizons <= 2, exact ties to the smaller lambda; the same
  rule for both arms. The R6 relative tie rule (``wmcp._tie_to_field``) is an H-T rule and is **not** applied in E2.
* **Seeds and components.** 5 members per arm; member seeds ``model.Member`` 0..4; component bootstrap and
  selection split seed 20261117 (``model.split_and_bootstrap``). Component ids are mandatory; calibration
  components (``bank.calibration_components``) are excluded; every other component's h <= 2 trajectories train.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from wsp.wm import adapter9b as AD
from wsp.wm import model as M
from wsp.wm import wmcp as CP
from wsp.wm.access import Artefact

E2_WM_CONFIG = dataclasses.replace(M.WMConfig(), slope_transport=False)
N_MEMBERS = 5
COMPONENT_SEED = 20261117
WM_ARM = "adapter9b.V0RArm + model.WMEnsemble"
STATIC_ARM = "adapter9b.V0RArm + model.MatchedStatic"
SHRINK_RULE = (
    "model.select_shrink: grid (0, 0.25, 0.5, 0.75, 1); component-level selection split; mean |error| over "
    "horizons <= 2; exact ties to the smaller lambda; identical for both arms; R6 not applied in E2"
)


def effective_config(cfg: M.WMConfig = E2_WM_CONFIG) -> dict:
    """The effective post-override E2 configuration stored in the manifest (``e2.config_effective``)."""
    return {
        "wm_config": dataclasses.asdict(cfg),
        "wm_arm": WM_ARM,
        "static_arm": STATIC_ARM,
        "static_family_selection": "none in E2 (MatchedStatic only)",
        "representation": "R(E) of every realised cumulative prefix via one r_of (wmcp.r_from_contractions on "
        "9B); wmcp.v0_trajectory_r; field nest wmcp.v0_field (F = a + q)",
        "shrinkage": SHRINK_RULE,
        "n_members": N_MEMBERS,
        "member_seeds": list(range(N_MEMBERS)),
        "component_seed": COMPONENT_SEED,
        "components": "mandatory; calibration components excluded; h <= 2 trajectories of all other components",
        "slope_transport": cfg.slope_transport,
        "A_targets": "removed from base and training observations (drop_slope_block)",
    }


def drop_slope_block(base_obs: Artefact, trajs=()):
    """Remove the ``A`` (action-derivative) block from the base observation and from every trajectory's ``obs``.
    Returns (base_obs without A, trajectories without A); layouts are re-indexed."""
    layout = dict(base_obs.payload.get("layout", {}))
    obs = np.asarray(base_obs.payload["obs"], float)
    if "A" not in layout:
        return base_obs, list(trajs)
    a0, a1 = layout.pop("A")
    keep = np.r_[0:a0, a1 : len(obs)]
    width = a1 - a0
    new_layout = {k: ((s, e) if s < a0 else (s - width, e - width)) for k, (s, e) in layout.items()}
    new = Artefact(
        base_obs.kind,
        base_obs.theta_hash,
        base_obs.tier,
        dict(base_obs.payload, obs=obs[keep], layout=new_layout),
    )
    out = [dict(t, obs=np.asarray(t["obs"], float)[:, keep]) for t in trajs]
    return new, out


def trajectory(spec, r_of, steps, y=None, obs=None) -> dict:
    """The E2 trajectory of ordered snapped steps: R rows of every realised prefix, as ``V0RArm`` builds them."""
    return CP.v0_trajectory_r(spec, [r_of(steps[:k]) for k in range(len(steps) + 1)], y, obs)


def fit_planning_predictors(
    trajs,
    components,
    spec,
    base_obs: Artefact,
    calibration=(),
    cfg: M.WMConfig = E2_WM_CONFIG,
    n_members: int = N_MEMBERS,
) -> dict:
    """Fit the WM and static E2 predictors on the same trajectories, components and observations."""
    if components is None:
        raise ValueError(
            "E2 fits require component ids (the pilot row-level fallback is not allowed)"
        )
    if len(components) != len(trajs):
        raise ValueError("one component id per trajectory")
    if cfg.slope_transport:
        raise ValueError("slope transport is permanently removed (P1)")
    cal = set(calibration)
    keep = [i for i, c in enumerate(components) if c not in cal]
    trajs = [trajs[i] for i in keep]
    comps = [components[i] for i in keep]
    if len(set(comps)) < 2:
        raise ValueError("at least two training components are needed")
    base_obs, trajs = drop_slope_block(base_obs, trajs)
    field = CP.v0_field(spec, base_obs.theta_hash)
    wm = M.WMEnsemble(cfg, field, base_obs, n_seeds=n_members).fit(trajs, comps)
    st = M.MatchedStatic(cfg, field, base_obs, n_seeds=n_members).fit(trajs, comps)
    return {
        "wm": wm,
        "static": st,
        "base_obs": base_obs,
        "components": sorted(set(comps), key=str),
        "config": effective_config(cfg),
    }


def make_arms(
    fitted: dict, spec, r_of, actions, doses, primary, orient, q_hats, guards, support, mode
):
    """The two E2 arms: identical ``V0RArm`` construction except the fitted model and its own conformal q_hat."""
    kw = {
        "spec": spec,
        "r_of": r_of,
        "actions": actions,
        "doses": doses,
        "primary": primary,
        "orient": orient,
        "guards": guards,
        "support": support,
        "mode": mode,
    }
    return {
        "wm": AD.V0RArm(fitted["wm"], fitted["base_obs"], q_hat=q_hats["wm"], **kw),
        "static": AD.V0RArm(fitted["static"], fitted["base_obs"], q_hat=q_hats["static"], **kw),
    }
