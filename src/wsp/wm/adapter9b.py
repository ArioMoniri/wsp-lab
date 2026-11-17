"""9B planner adapter: WM-E and StatNN as scorers of the shared CEM planner (PREREG_WME v1 §6.5.3-6.5.5;
RULING-wme-resources §5: CEM is frozen).

* **Coordinates.** A :class:`Dictionary` names the primitive directions {E_j} (``model.dictionary``) with their
  Gram matrix (weight-space inner products of the dose-1 directions, computed from theta0 weights only) and the
  coordinate unit per primitive. :func:`trajectory` turns an ordered list of (primitive, dose) steps into the
  trajectory dict of ``wm.model`` (cumulative E, step-local D, position, signed overlap cos(D_t, E_{t-1}) and the
  cumulative norm, by algebra on the recipes). **The bank labelling path and the planner use this same function**,
  so both arms and both uses receive identical cumulative information (§6.1.2 (c)); it reproduces
  ``synthetic.bookkeeping`` exactly.
* **Arm** (:class:`PredictorArm`). The planner's interface (``draw``/``predict``/``embed``/``update``):
  - each planner action is one dose-free library action (e.g. ``steer:L20:syc``); the planner's continuous dose
    z in [-1, 1] is mapped to z x max|dose| and **snapped** to the nearest registered dose of that action present in
    the episode subsample (:func:`snap`), so every proposal is a sequence of registered primitives;
  - score = orient x mu_primary - q_hat sigma~_primary - psi pen(c): the planner subtracts kappa sigma (kappa = 1,
    the conformal q_hat is folded into sigma) and the support penalty pen is folded into the value with
    ``support=None`` passed to the planner, so the CEM code is unchanged;
  - guards: the native-scale trajectory guards, loss-oriented, with sigma q_hat sigma~ (``guard_z`` = 1);
  - ``mode="ranks"`` (C1.4 failure, both arms): sigma = 0 everywhere and ``draw`` returns None (deterministic top-b);
  - ``update`` is a no-op: predictors are frozen for the episode (§6.5.4), so no dev result enters prediction.
* **Support penalty** (:class:`MahalanobisSupport`): the excess of the Mahalanobis distance of R(E_c) from the
  training-state features over their 95th percentile, divided by that percentile; one object is shared by every
  arm, so the penalty is a function of the candidate alone.
* **Episode loop** (:func:`run_episode_v1`): four rounds of b = B/4 dev evaluations; from round 2 the beam is seeded
  by the arm's own top feasible incumbents (by dev LCB); repeats are cache hits (0 bundles); the winner is
  ``wm.e2.winner``. If the planner cannot propose b new registered candidates after three seeded retries, the
  remainder is forfeited and recorded (``BudgetLedger.close``).
"""

from __future__ import annotations

import numpy as np

from wsp.wm import e2 as E2
from wsp.wm import model as M
from wsp.wm import planner as PN
from wsp.wm import uncertainty as U

PLANNER_CONFIG = PN.PlannerConfig(
    max_actions=E2.MAX_PRIMITIVES,
    beam_width=3,
    inner_budget=48,
    batch=4,
    kappa=1.0,
    support_lambda=0.0,
    guard_z=1.0,
    guard_penalty=100.0,
)
SUPPORT_PSI = 0.5
SUPPORT_QUANTILE = 0.95
DOSE_BOUNDS = (-1.0, 1.0)


class Dictionary:
    def __init__(self, ids, gram, unit=None):
        self.ids = list(ids)
        self.index = {k: i for i, k in enumerate(self.ids)}
        self.gram = np.asarray(gram, float)
        self.unit = np.ones(len(self.ids)) if unit is None else np.asarray(unit, float)
        if self.gram.shape != (len(self.ids), len(self.ids)):
            raise ValueError("gram must be J x J")

    @property
    def J(self):
        return len(self.ids)


def trajectory(d: Dictionary, steps, base_obs_vec, K: int) -> dict:
    """Trajectory dict (``wm.model.batch`` layout) for ordered steps [(primitive id, dose)]; y is zero (unknown)."""
    J, T = d.J, len(steps)
    x = np.zeros((T + 1, 2 * J + 3))
    e = np.zeros((T, J))
    Ecoef = np.zeros(J)
    G = d.gram
    for t, (pid, dose) in enumerate(steps, start=1):
        j = d.index[pid]
        D = np.zeros(J)
        D[j] = dose * d.unit[j]
        e[t - 1, j] = D[j]
        num = Ecoef @ G @ D
        den = np.sqrt(max(Ecoef @ G @ Ecoef, 0.0)) * np.sqrt(max(D @ G @ D, 0.0))
        overlap = float(num / den) if den > 0 else 0.0
        Ecoef = Ecoef + D
        x[t, :J] = Ecoef
        x[t, J : 2 * J] = D
        x[t, -3] = t
        x[t, -2] = overlap
        x[t, -1] = np.sqrt(max(Ecoef @ G @ Ecoef, 0.0))
    obs = np.broadcast_to(np.asarray(base_obs_vec, float), (T + 1, len(base_obs_vec))).copy()
    return {
        "horizon": T,
        "e": e,
        "x": x,
        "y": np.zeros((T + 1, K)),
        "obs": obs,
        "steps": list(steps),
    }


def features(traj) -> np.ndarray:
    """R(E_t) rows used by the support penalty: cumulative coordinates and the cumulative norm."""
    x = np.asarray(traj["x"])
    J = (x.shape[-1] - 3) // 2
    return np.concatenate([x[:, :J], x[:, -1:]], -1)


def snap(z: float, available) -> float:
    av = sorted(float(a) for a in available)
    return min(av, key=lambda a: (abs(a - z), abs(a), -a))


class MahalanobisSupport:
    def __init__(self, train_features, ridge: float = 1e-6, q: float = SUPPORT_QUANTILE):
        X = np.asarray(train_features, float)
        self.mu = X.mean(0)
        C = np.cov(X, rowvar=False) if len(X) > 1 else np.eye(X.shape[1])
        C = np.atleast_2d(C)
        self.P = np.linalg.pinv(C + ridge * np.trace(C) / max(1, len(C)) * np.eye(len(C)))
        self.d_q = float(np.quantile(self.distance(X), q))

    def distance(self, X):
        Z = np.atleast_2d(np.asarray(X, float)) - self.mu
        return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", Z, self.P, Z), 0.0))

    def penalty(self, X):
        return np.maximum(0.0, self.distance(X) - self.d_q) / max(self.d_q, 1e-12)


def predict_batch(model, base_obs, trajs) -> dict:
    """Vectorised predictions of a fitted ``WMEnsemble`` / ``MatchedStatic`` / ``StaticPredictor`` for trajectories
    of one horizon: mean and sigma~ exactly as ``model.predict``; ``members`` are the per-member shrunk predictions
    (Thompson draws)."""
    b = M.batch(trajs)
    f = model.f3u(b["E"])
    gate = U.edit_gate(b["E"])
    if isinstance(model, M.StaticPredictor):
        import torch

        XS = b["xs"]
        with torch.no_grad():
            X = torch.as_tensor(model.norm(XS), dtype=torch.float32)
            X0 = torch.as_tensor(
                model.norm(np.zeros((1, XS.shape[-1]))), dtype=torch.float32
            ).expand_as(X)
            at_zero = (X == X0).all(-1, keepdim=True)
            rs = np.stack([torch.where(at_zero, 0.0, n.f(X) - n.f(X0)).numpy() for n in model.nets])
        raw = f[None] + rs * model.y_scale
        spread = rs.std(0) * model.y_scale
    else:
        base = np.asarray(base_obs.payload["obs"])
        b = dict(b, obs=np.broadcast_to(base, b["obs"].shape).copy())
        raw = np.stack([m.predict(b) for m in model.members])
        spread = raw.std(0)
    lam = model.shrink
    members = f[None] + lam * (raw - f[None])
    mean = f + lam * (raw.mean(0) - f)
    sigma = U.combine(model.sigma_floor, lam, spread, gate)
    return {"mean": mean, "sigma": sigma, "members": members}


def episode_actions(episode, lib_index) -> tuple[list, dict]:
    """The planner's action set for an episode: its subsample's dose-free actions, each with the registered doses
    present in the subsample."""
    doses: dict = {}
    for pid in episode["subsample"]:
        p = lib_index[pid]
        doses.setdefault(p["action"], []).append(p["dose"])
    acts = sorted(doses)
    return acts, {a: sorted(doses[a]) for a in acts}


class PredictorArm:
    def __init__(
        self,
        model,
        base_obs,
        dictionary: Dictionary,
        actions,
        doses,
        primary: int,
        orient: float,
        q_hat: float,
        guards,
        support=None,
        psi: float = SUPPORT_PSI,
        mode: str = "lcb",
        K=None,
        primitive_of=None,
    ):
        if mode not in ("lcb", "ranks"):
            raise ValueError(mode)
        self.model, self.base_obs, self.d = model, base_obs, dictionary
        self.actions, self.doses = list(actions), {a: list(v) for a, v in doses.items()}
        self.primary, self.orient, self.q_hat = int(primary), float(orient), float(q_hat)
        self.guards = [(int(k), float(s), float(lim)) for k, s, lim in guards]
        self.support, self.psi, self.mode = support, float(psi), mode
        self.K = K
        self.primitive_of = primitive_of or (lambda action, dose: action)
        self.n_updates_ignored = 0
        self.n_members = getattr(model, "n_members", None) or len(
            getattr(model, "members", None) or getattr(model, "nets", [None])
        )

    # planner interface ------------------------------------------------------------------------------------
    def draw(self, rng):
        return None if self.mode == "ranks" else int(rng.integers(self.n_members))

    def steps(self, actions, z_row):
        """[(dictionary action, snapped registered dose)] for planner actions and continuous doses z."""
        out = []
        for a, z in zip(actions, z_row, strict=True):
            act = self.actions[a]
            av = self.doses[act]
            out.append((act, snap(float(z) * max(abs(x) for x in av), av)))
        return out

    def _trajs(self, actions, X):
        X = np.atleast_2d(np.asarray(X, float))
        base = self.base_obs.payload["obs"] if self.base_obs is not None else np.zeros(1)
        K = self.K or 3
        return [trajectory(self.d, self.steps(actions, row), base, K) for row in X]

    def _run(self, trajs):
        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(trajs)
        return predict_batch(self.model, self.base_obs, trajs)

    def penalty(self, actions, X):
        if self.support is None:
            return np.zeros(len(np.atleast_2d(X)))
        R = np.stack([features(t)[-1] for t in self._trajs(actions, X)])
        return self.support.penalty(R)

    def predict(self, actions, X, draw):
        trajs = self._trajs(actions, X)
        out = self._run(trajs)
        T = trajs[0]["horizon"]
        mu = out["mean"] if draw is None else out["members"][draw]
        sig = np.zeros_like(out["sigma"]) if self.mode == "ranks" else self.q_hat * out["sigma"]
        pen = self.penalty(actions, X) if self.support is not None else 0.0
        value = self.orient * mu[:, T, self.primary] - self.psi * pen
        guards = [(s * mu[:, T, k], sig[:, T, k], lim) for k, s, lim in self.guards]
        return {"value": value, "sigma": sig[:, T, self.primary], "guards": guards}

    def embed(self, actions, X):
        return np.stack([features(t)[-1] for t in self._trajs(actions, X)])

    def update(self, c, obs):
        self.n_updates_ignored += 1  # frozen predictors: no online refit (§6.5.4)

    # helpers ----------------------------------------------------------------------------------------------
    def to_primitives(self, c) -> tuple:
        acts = [a for a, _ in c]
        return tuple(self.primitive_of(a, d) for a, d in self.steps(acts, [z for _, z in c]))

    def canon(self, c) -> str:
        return E2.candidate_key(self.to_primitives(c))

    def from_primitives(self, pids, lib_index):
        out = []
        for pid in pids:
            p = lib_index[pid] if lib_index else {"action": pid, "dose": 1.0}
            a = self.actions.index(p["action"])
            out.append((a, p["dose"] / max(abs(x) for x in self.doses[p["action"]])))
        return tuple(out)


def run_episode_v1(
    world,
    arm: PredictorArm,
    ledger,
    arm_name: str,
    episode,
    B: int,
    cfg=None,
    inner: str = "cem",
    lib_index=None,
) -> dict:
    """One (episode, arm) search on dev under the v1 schedule; ``world.evaluate(primitive ids) -> dev record``
    with ``gain`` (oriented), ``gain_se`` (or ``lcb``), ``guards_pass``, ``rel_norm``, ``theta_hash``, ``seconds``."""
    cfg = cfg or PLANNER_CONFIG
    lib_index = lib_index or E2.library_index(episode["objective"])
    arm.primitive_of = lambda act, dose: _primitive_id(lib_index, episode, act, dose)
    planner = PN.Planner(len(arm.actions), DOSE_BOUNDS, inner, cfg)
    evaluated, rounds, keys = [], [], set()
    base_seed = int(episode["seeds"]["planner"])
    forfeited = 0
    for r, b in enumerate(E2.schedule(B)):
        seeds = None
        if r >= 1:
            feas = sorted(
                (e for e in evaluated if e["guards_pass"] and e["lcb"] > 0), key=lambda e: -e["lcb"]
            )[: cfg.beam_width]
            seeds = [
                (
                    tuple(a for a, _ in arm.from_primitives(e["candidate"], lib_index)),
                    np.array([z for _, z in arm.from_primitives(e["candidate"], lib_index)]),
                )
                for e in feas
            ]
        cands = []
        for attempt in range(3):
            need = b - len(cands)
            if need <= 0:
                break
            got = planner.propose(
                arm,
                need,
                base_seed + 31 * r + 7 * attempt,
                arm.mode,
                None,
                exclude=keys | {arm.canon(c) for c in cands},
                seed_nodes=seeds,
                canon=arm.canon,
            )
            cands += got
        this = []
        for c in cands:
            pids = E2.validate_candidate(arm.to_primitives(c), episode)
            k = E2.candidate_key(pids)
            keys.add(k)
            obs = world.evaluate(pids)
            ledger.charge(
                arm_name, k, seconds=obs.get("seconds", 0.0), theta_hash=obs.get("theta_hash")
            )
            lcb = obs["lcb"] if "lcb" in obs else obs["gain"] - E2.Z * obs["gain_se"]
            rec = dict(obs, candidate=pids, order=len(evaluated), lcb=lcb, round=r)
            evaluated.append(rec)
            this.append(pids)
        forfeited += b - len(cands)
        rounds.append(this)
    ledger.close(
        arm_name, "budget spent" if forfeited == 0 else f"planner exhausted ({forfeited} forfeited)"
    )
    return {
        "winner": E2.winner(evaluated),
        "evaluated": evaluated,
        "rounds": rounds,
        "forfeited": forfeited,
    }


def _primitive_id(lib_index, episode, action, dose):
    for pid in episode["subsample"]:
        p = lib_index[pid]
        if p["action"] == action and p["dose"] == dose:
            return pid
    raise KeyError(f"{action} @ {dose} not in the episode subsample")


class CPArm(PredictorArm):
    """WM-CP (or any R-space static member) as a planner scorer (RULING-wme-v2 A1): a candidate's snapped steps are
    mapped to R(E_c) by ``r_of(steps)``, an R0 computation (the cumulative realised edit from theta0 weights and the
    recipes, contracted with the base field); scoring, guards, ranks mode, Thompson draws and the support penalty
    are those of :class:`PredictorArm`. Path consistency makes the order of a candidate's steps irrelevant to its
    score."""

    def __init__(
        self,
        model,
        r_of,
        actions,
        doses,
        primary,
        orient,
        q_hat,
        guards,
        support=None,
        psi: float = SUPPORT_PSI,
        mode: str = "lcb",
        primitive_of=None,
    ):
        self.model, self.r_of = model, r_of
        self.base_obs = None
        self.actions, self.doses = list(actions), {a: list(v) for a, v in doses.items()}
        self.primary, self.orient, self.q_hat = int(primary), float(orient), float(q_hat)
        self.guards = [(int(k), float(s), float(lim)) for k, s, lim in guards]
        self.support, self.psi, self.mode = support, float(psi), mode
        if mode not in ("lcb", "ranks"):
            raise ValueError(mode)
        self.primitive_of = primitive_of or (lambda action, dose: action)
        self.n_updates_ignored = 0
        self.n_members = len(model.members)
        self.K = None

    def _R(self, actions, X):
        return np.stack(
            [self.r_of(self.steps(actions, row)) for row in np.atleast_2d(np.asarray(X, float))]
        )

    def penalty(self, actions, X):
        if self.support is None:
            return np.zeros(len(np.atleast_2d(X)))
        return self.support.penalty(self._R(actions, X))

    def embed(self, actions, X):
        return self._R(actions, X)

    def predict(self, actions, X, draw):
        R = self._R(actions, X)
        out = self.model.predict_batch(R)
        mu = out["mean"] if draw is None else out["members"][draw]
        sig = np.zeros_like(out["sigma"]) if self.mode == "ranks" else self.q_hat * out["sigma"]
        pen = self.support.penalty(R) if self.support is not None else 0.0
        value = self.orient * mu[:, self.primary] - self.psi * pen
        guards = [(s * mu[:, k], sig[:, k], lim) for k, s, lim in self.guards]
        return {"value": value, "sigma": sig[:, self.primary], "guards": guards}


class V0RArm(PredictorArm):
    """WM-E v0 (``wm.model``, history-dependent; descriptive under RULING-wme-v2 A1) as a planner scorer on the
    shared representation R(E) (A5): a candidate's snapped steps give the R rows of its cumulative prefixes
    (``r_of(steps[:t])``, R0), which :func:`wsp.wm.wmcp.v0_trajectory_r` turns into the model's trajectory. Scoring,
    guards, ranks mode, draws and the frozen-predictor rule are :class:`PredictorArm`'s; the support penalty and
    ``embed`` use R(E_c) as :class:`CPArm` does, so every arm shares one support object."""

    def __init__(
        self,
        model,
        base_obs,
        spec,
        r_of,
        actions,
        doses,
        primary,
        orient,
        q_hat,
        guards,
        support=None,
        psi: float = SUPPORT_PSI,
        mode: str = "lcb",
        primitive_of=None,
    ):
        super().__init__(
            model,
            base_obs,
            None,
            actions,
            doses,
            primary,
            orient,
            q_hat,
            guards,
            support=support,
            psi=psi,
            mode=mode,
            K=spec.n_obj,
            primitive_of=primitive_of,
        )
        self.spec, self.r_of = spec, r_of

    def _trajs(self, actions, X):
        from wsp.wm import wmcp as CP

        base = np.asarray(self.base_obs.payload["obs"], float)
        out = []
        for row in np.atleast_2d(np.asarray(X, float)):
            st = self.steps(actions, row)
            out.append(
                CP.v0_trajectory_r(
                    self.spec, [self.r_of(st[:k]) for k in range(len(st) + 1)], obs=base
                )
            )
        return out

    def _R(self, actions, X):
        return np.stack(
            [self.r_of(self.steps(actions, row)) for row in np.atleast_2d(np.asarray(X, float))]
        )

    def penalty(self, actions, X):
        if self.support is None:
            return np.zeros(len(np.atleast_2d(X)))
        return self.support.penalty(self._R(actions, X))

    def embed(self, actions, X):
        return self._R(actions, X)
