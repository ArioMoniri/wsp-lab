"""Shared R0 planner for every ranking arm (SYNTHESIS-r9 C6, pro-engineer W3; PREREG_WME §6 H-E1/H-E2).

* **Inner optimisers** over the doses of a fixed action set, with identical population size, bounds and R0
  budget: :func:`sep_cma_es` (separable CMA-ES, diagonal covariance; Ros & Hansen 2008, learning rates c1, cmu
  scaled by (n + 2)/3) and :func:`cem` (cross-entropy method, elite fraction 1/4, smoothing 0.7). ``"random"``
  (uniform search at the same budget) is a descriptive baseline.
* **Beam** (:class:`Planner`) over action identities: level 1 scores every single action, each level extends the
  ``beam_width`` best nodes by one new action (<= ``max_actions`` = 4), re-optimising all doses at each node.
* **Score** of a candidate under one posterior draw: draw value - kappa sigma - lambda support penalty, with every
  guardrail required at its bound (mean + z sigma <= limit; violations penalised). ``mode="ranks"`` (the symmetric
  fallback when any arm's coverage check fails, :func:`penalty_mode`) drops sigma everywhere for every arm.
* **Thompson batches**: each of the ``batch`` slots optimises one posterior draw (an ensemble member when the
  predictor has members); already evaluated candidates are excluded.
* **Outer loop** (:func:`run_episode`): batches are evaluated for real on development items and charged on the
  :class:`wm.plan.BudgetLedger` (one bundle each) until B is spent; the predictor refits online; incumbents are
  reported at 1, 4, 8, 16 and 32 bundles; the winner is the best observed candidate with gain LCB > 0 and every
  guardrail passed, else base (``None``).
* **Synthetic regret** (:func:`synthetic_regret`): sep-CMA-ES vs CEM at B = 32 on :class:`SyntheticPlanningWorld`
  (saturating gain, quadratic damage guardrail, biased ensemble surrogate with online kernel correction). Kill
  rule (r9 C6): if sep-CMA-ES's mean regret is not lower than CEM's, CEM is used.

The planner never runs an edit: every score comes from the predictor (R0); only :func:`run_episode` calls the
world's real evaluation, through the ledger.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

Z95 = 1.96
INCUMBENT_AT = (1, 4, 8, 16, 32)


# ------------------------------------------------------------------ inner optimisers


def sep_cma_params(n: int) -> dict:
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = 1.0 / np.sum(w**2)
    cs = (mueff + 2) / (n + mueff + 5)
    ds = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    scale = (n + 2) / 3.0  # separable variant: faster diagonal learning
    c1, cmu = c1 * scale, cmu * scale
    if c1 + cmu > 1:
        tot = c1 + cmu
        c1, cmu = c1 / tot * 0.99, cmu / tot * 0.99
    chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
    return {
        "lam": lam,
        "mu": mu,
        "weights": w,
        "mueff": mueff,
        "cs": cs,
        "ds": ds,
        "cc": cc,
        "c1": c1,
        "cmu": cmu,
        "chi": chi,
    }


def sep_cma_es(f, x0, lo, hi, budget: int, seed: int, sigma0: float | None = None):
    """Maximise ``f`` (vectorised: [N, n] -> [N]) within [lo, hi]. Returns (best x, best value, evaluations)."""
    x0, lo, hi = (np.asarray(v, float) for v in (x0, lo, hi))
    n = len(x0)
    p = sep_cma_params(n)
    rng = np.random.default_rng(seed)
    m = np.clip(x0, lo, hi)
    sigma = sigma0 if sigma0 is not None else 0.3 * float(np.mean(hi - lo))
    C = np.ones(n)
    ps, pc = np.zeros(n), np.zeros(n)
    best_x, best_v = m.copy(), float(f(m[None])[0])
    evals, g = 1, 0
    while evals + p["lam"] <= budget:
        z = rng.standard_normal((p["lam"], n))
        x = np.clip(m + sigma * np.sqrt(C) * z, lo, hi)
        v = np.asarray(f(x), float)
        evals += p["lam"]
        order = np.argsort(-v, kind="stable")
        if v[order[0]] > best_v:
            best_v, best_x = float(v[order[0]]), x[order[0]].copy()
        y = (x[order[: p["mu"]]] - m) / sigma
        yw = p["weights"] @ y
        m = np.clip(m + sigma * yw, lo, hi)
        ps = (1 - p["cs"]) * ps + math.sqrt(p["cs"] * (2 - p["cs"]) * p["mueff"]) * yw / np.sqrt(C)
        g += 1
        hs = np.linalg.norm(ps) / math.sqrt(1 - (1 - p["cs"]) ** (2 * g)) / p["chi"] < 1.4 + 2 / (
            n + 1
        )
        pc = (1 - p["cc"]) * pc + hs * math.sqrt(p["cc"] * (2 - p["cc"]) * p["mueff"]) * yw
        C = (
            (1 - p["c1"] - p["cmu"]) * C
            + p["c1"] * (pc**2 + (1 - hs) * p["cc"] * (2 - p["cc"]) * C)
            + p["cmu"] * (p["weights"] @ (y**2))
        )
        C = np.maximum(C, 1e-12)
        sigma *= math.exp((p["cs"] / p["ds"]) * (np.linalg.norm(ps) / p["chi"] - 1))
        sigma = min(sigma, float(np.max(hi - lo)))
    return best_x, best_v, evals


def cem(f, x0, lo, hi, budget: int, seed: int, sigma0: float | None = None, elite=0.25, smooth=0.7):
    x0, lo, hi = (np.asarray(v, float) for v in (x0, lo, hi))
    n = len(x0)
    lam = sep_cma_params(n)["lam"]
    k = max(2, round(elite * lam))
    rng = np.random.default_rng(seed)
    mu = np.clip(x0, lo, hi)
    sd = np.full(n, sigma0 if sigma0 is not None else 0.3 * float(np.mean(hi - lo)))
    best_x, best_v = mu.copy(), float(f(mu[None])[0])
    evals = 1
    while evals + lam <= budget:
        x = np.clip(mu + sd * rng.standard_normal((lam, n)), lo, hi)
        v = np.asarray(f(x), float)
        evals += lam
        order = np.argsort(-v, kind="stable")
        if v[order[0]] > best_v:
            best_v, best_x = float(v[order[0]]), x[order[0]].copy()
        el = x[order[:k]]
        mu = smooth * el.mean(0) + (1 - smooth) * mu
        sd = np.maximum(smooth * el.std(0) + (1 - smooth) * sd, 1e-3 * (hi - lo))
    return best_x, best_v, evals


def random_search(f, x0, lo, hi, budget: int, seed: int, sigma0=None):
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [np.clip(x0, lo, hi)[None], rng.uniform(lo, hi, size=(max(0, budget - 1), len(x0)))]
    )
    v = np.asarray(f(x), float)
    i = int(np.argmax(v))
    return x[i], float(v[i]), len(x)


INNER = {"sep_cma": sep_cma_es, "cem": cem, "random": random_search}


# ------------------------------------------------------------------ planner


@dataclasses.dataclass(frozen=True)
class PlannerConfig:
    max_actions: int = 4
    beam_width: int = 3
    inner_budget: int = 48
    batch: int = 4
    kappa: float = 1.0
    support_lambda: float = 0.5
    guard_z: float = Z95
    guard_penalty: float = 100.0


def penalty_mode(coverage_passed: dict) -> str:
    """Symmetric across arms: LCB scoring only when every arm's coverage check passed, else ranks for all."""
    return "lcb" if all(coverage_passed.values()) else "ranks"


def candidate_key(c) -> str:
    return ";".join(f"{a}:{d:.6g}" for a, d in sorted(c))


_R0_CACHE: dict = {}


def _support_r0(support: np.ndarray) -> float:
    key = (support.shape, hash(support.tobytes()))
    if key not in _R0_CACHE:
        nn = np.sqrt(((support[:, None, :] - support[None]) ** 2).sum(-1))
        np.fill_diagonal(nn, np.inf)
        _R0_CACHE[key] = float(np.median(nn.min(1)))
    return _R0_CACHE[key]


def support_penalty(E: np.ndarray, support: np.ndarray | None) -> np.ndarray:
    """max(0, d_nn(E) - r0) / r0: distance beyond the median nearest-neighbour spacing of the training support."""
    if support is None or len(support) == 0:
        return np.zeros(len(E))
    sq = (E**2).sum(-1)[:, None] + (support**2).sum(-1)[None] - 2 * E @ support.T
    d = np.sqrt(np.maximum(sq.min(1), 0.0))
    r0 = _support_r0(support)
    return np.maximum(0.0, d - r0) / max(r0, 1e-12)


class Planner:
    def __init__(
        self,
        library_size: int,
        dose_bounds,
        inner: str = "sep_cma",
        cfg: PlannerConfig | None = None,
    ):
        self.A, self.lo, self.hi = int(library_size), float(dose_bounds[0]), float(dose_bounds[1])
        self.inner, self.cfg = inner, cfg or PlannerConfig()

    def score(self, pred, cands, draw, mode, support):
        """Score of each candidate (list of (action, dose)) under ``draw`` (None = posterior mean)."""
        acts = [[a for a, _ in c] for c in cands]
        doses = [np.array([d for _, d in c]) for c in cands]
        return np.concatenate(
            [
                self._score_set(pred, a, d[None], draw, mode, support)
                for a, d in zip(acts, doses, strict=True)
            ]
        )

    def _score_set(self, pred, actions, doses, draw, mode, support):
        out = pred.predict(actions, doses, draw)
        sig_on = mode == "lcb"
        val = out["value"] - (self.cfg.kappa * out["sigma"] if sig_on else 0.0)
        for gm, gs, lim in out["guards"]:
            bound = gm + (self.cfg.guard_z * gs if sig_on else 0.0)
            val = val - self.cfg.guard_penalty * np.maximum(0.0, bound - lim)
        if self.cfg.support_lambda and support is not None:
            val = val - self.cfg.support_lambda * support_penalty(
                pred.embed(actions, doses), support
            )
        return val

    def _optimise(self, pred, actions, warm, draw, mode, support, seed):
        n = len(actions)
        f = lambda X: self._score_set(pred, actions, X, draw, mode, support)
        x, v, _ = INNER[self.inner](
            f, warm, np.full(n, self.lo), np.full(n, self.hi), self.cfg.inner_budget, seed
        )
        return list(zip(actions, x.tolist(), strict=True)), v

    def beam(self, pred, draw, mode, support, seed, exclude=()):
        frontier = [((), np.zeros(0))]
        scored = []
        for level in range(1, self.cfg.max_actions + 1):
            children = []
            for node, warm in frontier:
                for a in range(self.A):
                    if a in node:
                        continue
                    acts = (*node, a)
                    w0 = np.r_[warm, 0.5 * self.hi]
                    c, v = self._optimise(
                        pred, list(acts), w0, draw, mode, support, seed + 7919 * level + a
                    )
                    children.append((v, c, acts))
            children.sort(key=lambda t: -t[0])
            seen, frontier = set(), []
            for v, c, acts in children:
                k = tuple(sorted(acts))
                if k in seen:
                    continue
                seen.add(k)
                scored.append((v, c))
                if len(frontier) < self.cfg.beam_width:
                    frontier.append((acts, np.array([d for _, d in c])))
        scored.sort(key=lambda t: -t[0])
        return [(v, c) for v, c in scored if candidate_key(c) not in exclude]

    def propose(self, pred, q, seed, mode, support, exclude=()):
        rng = np.random.default_rng(seed)
        out, keys = [], set(exclude)
        for slot in range(q):
            draw = pred.draw(rng)
            for _v, c in self.beam(pred, draw, mode, support, seed + 104729 * slot, keys):
                if candidate_key(c) not in keys:
                    out.append(c)
                    keys.add(candidate_key(c))
                    break
        return out


def run_episode(world, pred, arm, ledger, inner="sep_cma", cfg=None, seed=0, mode="lcb"):
    cfg = cfg or PlannerConfig()
    planner = Planner(world.library_size, world.dose_bounds, inner, cfg)
    observed, order, incumbents = {}, [], {}
    r = 0
    while ledger.remaining(arm) >= 1:
        q = int(min(cfg.batch, ledger.remaining(arm)))
        cands = planner.propose(
            pred, q, seed + 31 * r, mode, world.support_ref, exclude=set(observed)
        )
        if not cands:
            break
        for c in cands:
            obs = world.evaluate(c)
            k = candidate_key(c)
            ledger.charge(arm, k, seconds=obs.get("seconds", 0.0), theta_hash=k)
            lcb = obs["gain"] - Z95 * obs["se"]
            guard_ok = all(m + Z95 * s <= lim for m, s, lim in obs["guards"])
            observed[k] = dict(obs, lcb=lcb, guard_ok=guard_ok, candidate=c)
            order.append(k)
            pred.update(c, obs)
            n = len(order)
            if n in INCUMBENT_AT:
                feas = [
                    observed[x] for x in order if observed[x]["guard_ok"] and observed[x]["lcb"] > 0
                ]
                incumbents[n] = max(feas, key=lambda o: o["lcb"])["candidate"] if feas else None
        r += 1
    ledger.close(arm, "budget spent" if ledger.remaining(arm) < 1 else "no new candidate")
    for n in INCUMBENT_AT:
        incumbents.setdefault(n, incumbents.get(max([m for m in incumbents if m <= n], default=0)))
    feas = [o for o in observed.values() if o["guard_ok"] and o["lcb"] > 0]
    winner = max(feas, key=lambda o: o["lcb"])["candidate"] if feas else None
    return {"winner": winner, "observed": observed, "order": order, "incumbents": incumbents}


# ------------------------------------------------------------------ synthetic planning world


class _Fourier:
    def __init__(self, rng, d, n=16, ls=1.2):
        self.W = rng.standard_normal((n, d)) / ls
        self.b = rng.uniform(0, 2 * np.pi, n)
        self.a = rng.standard_normal(n) * math.sqrt(2.0 / n)

    def __call__(self, E):
        return np.cos(E @ self.W.T + self.b) @ self.a


class SyntheticPlanningWorld:
    """Library of A actions v_a in R^d; E = sum dose v_a. True gain sum_j w_j tanh(E_j) - 0.1 |E|^2; guardrail
    damage 0.05 (u . E)^2 <= 0.25; real evaluations add N(0, se^2) noise and are counted."""

    def __init__(self, seed=0, A=12, d=6, se=0.05):
        rng = np.random.default_rng(10_000 + seed)
        self.seed, self.library_size, self.d, self.se = seed, A, d, se
        self.V = rng.standard_normal((A, d)) / math.sqrt(d) * 1.2
        self.w = rng.uniform(0.2, 1.0, d) * rng.choice([-1, 1], d)
        self.u = rng.standard_normal(d)
        self.u /= np.linalg.norm(self.u)
        self.dose_bounds = (-2.0, 2.0)
        self.limit = 0.25
        self.real_calls = 0
        self._noise = np.random.default_rng(20_000 + seed)
        sup = []
        for _ in range(200):
            k = rng.integers(1, 3)
            acts = rng.choice(A, size=k, replace=False)
            sup.append(self.embed(list(acts), rng.uniform(-1, 1, (1, k)))[0])
        self.support_ref = np.array(sup)

    def embed(self, actions, doses):
        doses = np.atleast_2d(doses)
        return doses @ self.V[list(actions)]

    def true_gain(self, E):
        return np.tanh(E) @ self.w - 0.1 * (E**2).sum(-1)

    def true_guard(self, E):
        return 0.05 * (E @ self.u) ** 2

    def true_value(self, c):
        """Registered test score of a winner: gain if the guardrail holds, else 0 (base); base = 0."""
        if c is None:
            return 0.0
        E = self.embed([a for a, _ in c], np.array([[d for _, d in c]]))
        return float(self.true_gain(E)[0]) if self.true_guard(E)[0] <= self.limit else 0.0

    def evaluate(self, c):
        self.real_calls += 1
        E = self.embed([a for a, _ in c], np.array([[d for _, d in c]]))
        return {
            "gain": float(self.true_gain(E)[0] + self._noise.normal(0, self.se)),
            "se": self.se,
            "guards": [
                (
                    float(self.true_guard(E)[0] + self._noise.normal(0, self.se / 4)),
                    self.se / 4,
                    self.limit,
                )
            ],
            "E": E[0],
        }

    def surrogate(self, seed=0, n_members=5, bias=0.25, spread=0.15):
        return _Surrogate(self, seed, n_members, bias, spread)


class _Surrogate:
    """Ensemble surrogate: member m = truth + shared bias + member bias, corrected online by kernel regression of
    each member's residuals at evaluated points (the "refit")."""

    def __init__(self, world, seed, M, bias, spread):
        rng = np.random.default_rng(30_000 + 97 * world.seed + seed)
        self.w, self.M = world, M
        self.shared = _Fourier(rng, world.d)
        self.members = [_Fourier(rng, world.d) for _ in range(M)]
        self.gshared = _Fourier(rng, world.d)
        self.bias, self.spread = bias, spread
        self.X, self.R, self.RG = [], [], []
        self.sigma_scale = 1.0
        self.n_members = M

    def embed(self, actions, doses):
        return self.w.embed(actions, doses)

    def draw(self, rng):
        return int(rng.integers(self.M))

    def _members(self, E):
        base = self.w.true_gain(E) + self.bias * self.shared(E)
        vals = np.stack([base + self.spread * m(E) for m in self.members])
        g = self.w.true_guard(E) + 0.02 * self.gshared(E)
        if self.X:
            k = np.exp(-((E[:, None] - self._Xa[None]) ** 2).sum(-1) / 2.0)
            vals = vals + (k @ self._sol).T
            g = g + k @ self._solg
        return vals, g

    def _refit(self):
        X = self._Xa = np.array(self.X)
        K = np.exp(-((X[:, None] - X[None]) ** 2).sum(-1) / 2.0) + (self.w.se**2 + 1e-6) * np.eye(
            len(X)
        )
        self._sol = np.linalg.solve(K, np.array(self.R))  # [n_obs, M]
        self._solg = np.linalg.solve(K, np.array(self.RG))

    def predict(self, actions, doses, draw):
        E = self.embed(actions, doses)
        vals, g = self._members(E)
        sig = vals.std(0) * self.sigma_scale
        v = vals.mean(0) if draw is None else vals[draw]
        return {
            "value": v,
            "sigma": sig,
            "guards": [(g, np.full(len(g), 0.01) * self.sigma_scale, self.w.limit)],
        }

    def update(self, c, obs):
        E = np.asarray(obs["E"])[None]
        vals, g = self._members(E)
        self.X.append(E[0])
        self.R.append(obs["gain"] - vals[:, 0])
        self.RG.append(obs["guards"][0][0] - g[0])
        self._refit()


def _oracle_best(world, cfg):
    """Approximate f*: the planner with the true function (sigma 0) and a large inner budget."""

    class _Truth:
        n_members = 1

        def embed(self, a, d):
            return world.embed(a, d)

        def draw(self, rng):
            return None

        def predict(self, a, d, draw):
            E = world.embed(a, d)
            return {
                "value": world.true_gain(E),
                "sigma": np.zeros(len(E)),
                "guards": [(world.true_guard(E), np.zeros(len(E)), world.limit)],
            }

    big = dataclasses.replace(
        cfg,
        inner_budget=max(200, 4 * cfg.inner_budget),
        beam_width=max(4, cfg.beam_width),
        support_lambda=0.0,
    )
    best = Planner(world.library_size, world.dose_bounds, "sep_cma", big).beam(
        _Truth(), None, "ranks", None, 0
    )
    best2 = Planner(world.library_size, world.dose_bounds, "cem", big).beam(
        _Truth(), None, "ranks", None, 1
    )
    return max([0.0] + [world.true_value(c) for _, c in best[:20] + best2[:20]])


def _one_world(args):
    s, B, methods, cfg = args
    from wsp.wm.plan import BudgetLedger

    vals = {}
    for m in methods:
        world = SyntheticPlanningWorld(seed=s)
        led = BudgetLedger(B, [m])
        out = run_episode(
            world, world.surrogate(seed=0), m, led, inner=m, cfg=cfg, seed=s, mode="lcb"
        )
        vals[m] = world.true_value(out["winner"])
    fs = max([_oracle_best(SyntheticPlanningWorld(seed=s), cfg)] + list(vals.values()))
    return s, fs, vals


def synthetic_regret(
    n_worlds=100,
    B=32,
    methods=("sep_cma", "cem"),
    cfg=None,
    seed0=0,
    n_boot=2000,
    procs=1,
):
    cfg = cfg or PlannerConfig()
    jobs = [(s, B, tuple(methods), cfg) for s in range(seed0, seed0 + n_worlds)]
    if procs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(procs) as pool:
            rows = pool.map(_one_world, jobs)
    else:
        rows = [_one_world(j) for j in jobs]
    rows.sort()
    fstar = [fs for _, fs, _ in rows]
    summary = {}
    for m in methods:
        reg = [fs - v[m] for _, fs, v in rows]
        summary[m] = {
            "regret": reg,
            "mean_regret": float(np.mean(reg)),
            "mean_value": float(np.mean([v[m] for _, _, v in rows])),
        }
    decision = {
        "use": "cem",
        "rule": "use sep-CMA-ES only if its mean regret is lower than CEM's (r9 C6)",
    }
    if "sep_cma" in summary and "cem" in summary:
        d = np.array(summary["sep_cma"]["regret"]) - np.array(summary["cem"]["regret"])
        rng = np.random.default_rng(1)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        decision.update(
            mean_diff_sep_minus_cem=float(d.mean()),
            ci95=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            wins_sep=int((d < 0).sum()),
            wins_cem=int((d > 0).sum()),
            ties=int((d == 0).sum()),
            use="sep_cma" if d.mean() < 0 else "cem",
        )
    return {
        "methods": summary,
        "fstar": fstar,
        "B": B,
        "n_worlds": n_worlds,
        "cfg": dataclasses.asdict(cfg),
        "decision": decision,
    }
