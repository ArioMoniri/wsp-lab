"""E2 arm-episode runners on the real dev world (PREREG_WME §6.5.3-§6.5.5, §6.8.3, §II.5, §II.22.2, §II.22.7).

Every runner takes the frozen inputs (library, fitted predictors, support, conformal q_hat, planner mode) and one
episode of ``wm.e2.episodes("sycophancy_real", 8)``, creates one ``plan.BudgetLedger`` for its own arm, runs the
registered library code, and returns a JSON payload: the winner (``e2.winner`` / ``arms.spec.select_winner``; the base
when none qualifies), every evaluated dev record, the ledger report and arm-specific accounting. No arm reads another
arm's results (one store per arm and episode); predictors are frozen (no online refit).

Choices fixed by the registration's code where the text is silent (amendments G15-G16, recorded before data):
* pool-mode arms (random, scripted, LLM readers) share the pool stream seeded by the episode's registered planner seed;
* ``field_ranked`` is ``adapter9b.V0RArm`` over the raw field F = a + q of R (``wmcp.v0_field``) in ranks mode
  ("LCB-free"), with the shared planner, support penalty and guards;
* ``direct_bon_two_stage``: candidates from the arm's own seeded uniform stream (1-4 distinct subsample primitives),
  stage 1 on the first 4 dev groups of every objective at 4/64 of a bundle, survivors with screened gain > 0 and no
  screened guard point breach, the best 16 survivors (the ``controls.two_stage_best_of_n`` default) on the full panel
  at 1 bundle each, budget reserved for them; winner by §6.5.5;
* ``lora_sft``: 64 construction/e2 suggested rows, 2 epochs x 8 rows = 16 steps, checkpoints at every step
  (``direct.checkpoint_steps``), each evaluated at lambda 1 as one checkpoint bundle.
"""

from __future__ import annotations

import time

import numpy as np

from wsp.wm import adapter9b as AD
from wsp.wm import e2 as E2
from wsp.wm import plan as PL
from wsp.wm.access import AccessError, R0Guard  # noqa: F401
from wsp.wm.arms import direct as DI
from wsp.wm.arms import pool as PO
from wsp.wm.arms import readers as RD
from wsp.wm.arms import sbon as SB
from wsp.wm.arms import spec as AS
from wsp.wm.c1 import e2ops as EO

OBJ = EO.OBJECTIVE
N_REP = 8
PRIMARY_INDEX = 5  # sycophancy_real in the R objective order
GUARD_SPECS = (  # (R index, orientation sign, native limit) of the predicted trajectory guards (§6.5.3)
    (0, 1.0, 0.01),  # wikitext_nll: UCB dNLL <= 0.01
    (1, -1.0, 0.05),  # gsm8k_gold_lp: LCB >= -0.05
    (2, -1.0, 0.05),  # mmlupro_gold_lp: LCB >= -0.05
    (3, -1.0, 0.25),  # refusal_margin_harmful: LCB >= -0.25
    (4, 1.0, 0.25),  # refusal_margin_harmless: UCB <= 0.25
)


def episode(index: int) -> dict:
    eps = E2.episodes(OBJ, N_REP)
    e = eps[int(index)]
    if e["index"] != int(index):
        raise ValueError("episode index mismatch")
    return e


def _strip(rec: dict) -> dict:
    keep = (
        "candidate",
        "theta_hash",
        "gain",
        "gain_se",
        "lcb",
        "guards_pass",
        "guard_ok",
        "guard_bounds",
        "guard_means",
        "rel_norm",
        "seconds",
        "order",
        "round",
        "kind",
    )
    out = {k: rec[k] for k in keep if k in rec}
    if "candidate" in out:
        out["candidate"] = [str(x) for x in out["candidate"]]
    return out


class FieldModel:
    """The raw field F = a + q on R as a planner model (``field_ranked``): no ensemble, sigma 0."""

    n_members = 1

    def __init__(self, spec):
        from wsp.wm import model as M
        from wsp.wm import wmcp as CP

        self.f3u = M.F3u(CP.v0_field(spec).payload)

    def predict_batch(self, trajs):
        from wsp.wm import model as M

        b = M.batch(trajs)
        mean = self.f3u(b["E"])
        return {"mean": mean, "sigma": np.zeros_like(mean), "members": mean[None]}


class Env:
    """Frozen per-instance inputs of the E2 runners (worker side)."""

    def __init__(
        self,
        m,
        world: EO.DevWorld,
        lib: EO.Library,
        rcache,
        fitted=None,
        support=None,
        q_hats=None,
        mode="lcb",
        c1_manifest=None,
        e2_items=None,
    ):
        self.m, self.world, self.lib, self.rc = m, world, lib, rcache
        self.fitted, self.support, self.q_hats, self.mode = fitted, support, q_hats or {}, mode
        self.c1_manifest, self.e2_items = c1_manifest, e2_items
        self.lib_index = lib.index

    def r_of_pids(self, pids):
        return self.rc.r_of(self.lib.bundles(pids))[0]


# ------------------------------------------------------------------ planner arms (wm_e, statnn, field_ranked)


def run_planner(env: Env, arm: str, ep: dict, B: int) -> dict:
    from wsp.wm.c1 import field as F

    spec = F.spec_9b()
    acts, doses = AD.episode_actions(ep, env.lib_index)
    prim = lambda a, d: AD._primitive_id(env.lib_index, ep, a, d)
    r_of = lambda st: env.r_of_pids([prim(a, d) for a, d in st])
    if arm == "wm_e":
        model, q_hat, mode = env.fitted["wm"], env.q_hats["wm"], env.mode
    elif arm == "statnn":
        model, q_hat, mode = env.fitted["static"], env.q_hats["static"], env.mode
    elif arm == "field_ranked":
        model, q_hat, mode = FieldModel(spec), 0.0, "ranks"
    else:
        raise KeyError(arm)
    base_obs = env.fitted["base_obs"]
    pa = AD.V0RArm(
        model,
        base_obs,
        spec,
        r_of,
        acts,
        doses,
        PRIMARY_INDEX,
        E2.ORIENT[OBJ],
        q_hat,
        GUARD_SPECS,
        support=env.support,
        psi=AD.SUPPORT_PSI,
        mode=mode,
        primitive_of=prim,
    )
    ledger = PL.BudgetLedger(B, [arm])
    t0 = time.time()
    r0 = env.rc.stats()
    out = AD.run_episode_v1(
        env.world,
        pa,
        ledger,
        arm,
        ep,
        B,
        cfg=AD.PLANNER_CONFIG,
        inner="cem",
        lib_index=env.lib_index,
    )
    r1 = env.rc.stats()
    w = out["winner"]
    return {
        "arm": arm,
        "episode": ep["index"],
        "winner": None if w.get("fallback") else _strip(w),
        "evaluated": [_strip(e) for e in out["evaluated"]],
        "forfeited": out["forfeited"],
        "ledger": ledger.report(),
        "mode": mode,
        "planner_seconds": time.time() - t0 - sum(e.get("seconds", 0.0) for e in out["evaluated"]),
        "r_cache": {
            "misses": r1["misses"] - r0["misses"],
            "seconds": r1["seconds"] - r0["seconds"],
        },
    }


# ------------------------------------------------------------------ ArmContext arms


def _ctx(env: Env, arm: str, ep: dict, B, resources: dict, to_pids, bundles_of=None):
    spec = AS.ARM_SPECS[arm]
    ledger = PL.BudgetLedger(B, [arm])
    stores = AS.StoreRegistry()
    bundles_of = bundles_of or (lambda c: env.lib.bundles(to_pids(c)))

    def evaluate(c):
        rec = env.world.evaluate_bundles(bundles_of(c), to_pids(c))
        return rec

    def theta_of(c):
        m = env.m
        m.at_base()
        try:
            for b in bundles_of(c):
                m.sess.apply_bundle(b)
            return m.theta_hash()
        finally:
            m.restore()

    guard = R0Guard(env.m.base_hash)
    ctx = AS.ArmContext(spec, ledger, ep["index"], evaluate, theta_of, resources, stores, guard)
    return ctx, ledger


def _ctx_payload(arm, ep, res, ctx, ledger, **extra):
    hist = ctx.history()
    w = res.get("winner")
    return {
        "arm": arm,
        "episode": ep["index"],
        "winner": None if w is None else _strip(w),
        "evaluated": [_strip(h) for h in hist],
        "ledger": ledger.report(),
        **extra,
    }


def run_greedy(env: Env, ep: dict, B: int) -> dict:
    from wsp.wm.c1 import field as F

    arm = "wm_greedy"
    sub = list(ep["subsample"])
    spec = F.spec_9b()
    acts = sub
    doses = {p: [env.lib_index[p]["dose"]] for p in sub}
    prim = lambda a, d: a
    r_of = lambda st: env.r_of_pids([a for a, _d in st])
    pred = AD.V0RArm(
        env.fitted["wm"],
        env.fitted["base_obs"],
        spec,
        r_of,
        acts,
        doses,
        PRIMARY_INDEX,
        E2.ORIENT[OBJ],
        env.q_hats["wm"],
        GUARD_SPECS,
        support=env.support,
        psi=AD.SUPPORT_PSI,
        mode=env.mode,
        primitive_of=prim,
    )
    to_pids = lambda c: [sub[i] for i in c]
    res = {
        "episode_library": list(range(len(sub))),
        "wm_predictor": pred,
        "support_penalty": None,
        "field": None,
        "bank": None,
    }
    ctx, ledger = _ctx(env, arm, ep, B, res, to_pids)
    out = _pids_winner(DI.run_greedy_episode(ctx, B, mode=env.mode, cfg=AD.PLANNER_CONFIG), sub)
    return _ctx_payload(
        arm,
        ep,
        out,
        ctx,
        ledger,
        rounds=[{k: v for k, v in r.items() if k != "evaluated"} for r in out["rounds"]],
    )


def primitive_meta(env: Env, sub) -> dict:
    """R0 description of each subsample primitive (family, layer, contrast, dose parameter, relative norm)."""
    m = env.m
    out = {}
    for p in sub:
        q = env.lib_index[p]
        m.at_base()
        try:
            for b in env.lib.bundles([p]):
                m.sess.apply_bundle(b)
            num = sum(float(m.sess.displacement(n).float().square().sum()) for n in m.sess.touched)
            den = sum(float(m.sess.original(n).float().square().sum()) for n in m.sess.touched)
        finally:
            m.restore()
        out[p] = {
            "family": q["family"],
            "layer": q.get("layer", f"{q['block'][0]}-{q['block'][1]}" if "block" in q else None),
            "contrast": q.get("contrast", q.get("lora")),
            "param": float(q["dose"]),
            "rel_norm": float(np.sqrt(num / den)) if den > 0 else 0.0,
        }
    return out


def field_features(env: Env):
    from wsp.wm.c1 import field as F

    objs = F.T2_OBJECTIVES
    K = len(objs)

    def feats(c):
        r, flags = env.rc.r_of(env.lib.bundles(list(c)))
        return {
            o: {"a": float(r[k]), "q": None if o in flags else float(r[K + k])}
            for k, o in enumerate(objs)
        }

    return feats


def pool_resources(env: Env, ep: dict):
    from wsp.wm.access import Artefact, Tier

    sub = list(ep["subsample"])
    meta = primitive_meta(env, sub)
    # the registered pool code draws integer primitive indices into the subsample (arms.pool.draw_pool)
    desc = Artefact(
        "r0_candidate_description",
        env.m.base_hash,
        Tier.R0,
        {i: meta[p] for i, p in enumerate(sub)},
    )
    return {
        "episode_library": list(range(len(sub))),
        "episode_pool": True,
        "r0_candidate_description": desc,
    }


def run_pool(env: Env, arm: str, ep: dict, B: int, pool_size: int, ranker=None) -> dict:
    res = pool_resources(env, ep)
    sub_ = list(ep["subsample"])
    to_pids = lambda c: [sub_[int(i)] for i in c]
    ctx, ledger = _ctx(env, arm, ep, B, res, to_pids)
    feats = field_features(env)
    if ranker is None:
        ranker = PO.RandomRanker() if arm == "random" else RD.ScriptedReader(RD.ReaderTask())
    out = PO.run_pool_episode(
        ranker, ctx, B, pool_size, int(ep["seeds"]["planner"]), features=lambda c: feats(to_pids(c))
    )
    return _ctx_payload(
        arm,
        ep,
        _pids_winner(out, sub_),
        ctx,
        ledger,
        rounds=out["rounds"],
        failures=out["failures"],
        index_to_pid=sub_,
    )


def _pids_winner(out, sub):
    w = out.get("winner")
    if w is not None:
        w = dict(w, candidate=[sub[int(i)] for i in w["candidate"]])
    return dict(out, winner=w)


def run_bon_steering(env: Env, ep: dict, B: int) -> dict:
    arm = "bon_steering"
    seed = AS.arm_seed(arm, int(ep["seeds"]["planner"]))
    to_key = lambda c: [f"steer:L{c[1]}:{c[2]}:{c[3]:+g}"]
    bundles_of = lambda c: [[env.lib.steer(int(c[1]), c[2], float(c[3]))]]
    res = {"steer_contrasts_all_layers": True}
    ctx, ledger = _ctx(env, arm, ep, B, res, to_key, bundles_of)
    out = DI.run_bon_steering(ctx, B, seed, env.m.rm.n_layers)
    return _ctx_payload(arm, ep, out, ctx, ledger, space_size=out["space_size"])


def run_lora_sft(env: Env, ep: dict, B: int) -> dict:
    import torch

    from wsp.wm import lora as LR
    from wsp.wm import panel as P

    arm = "lora_sft"
    seed = AS.arm_seed(arm, int(ep["seeds"]["planner"]))
    m = env.m
    rows = LR.sft_rows(OBJ, env.e2_items, m.tok, P.PanelConfig(), env.c1_manifest)
    st = E2.LORA_SETTINGS
    total = st["epochs"] * -(-len(rows) // st["batch_rows"])
    ck = {}

    def train_fn(items, steps_at, s):
        m.at_base()
        t0 = time.time()
        ad = LR.LoRA(m.model, LR.writer_targets(m.rm), rank=st["rank"], alpha=st["alpha"], seed=s)
        pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
        out = DI.train_with_checkpoints(
            m.model,
            ad,
            rows,
            pad,
            steps_at,
            lr=st["lr"],
            batch_rows=st["batch_rows"],
            seed=s,
            device="cuda",
            warmup=st["warmup"],
        )
        del ad
        torch.cuda.empty_cache()
        m.at_base()
        for stp, fac in out["checkpoints"].items():
            ck[("lora_sft", stp)] = fac
        return {
            "checkpoints": {stp: ("lora_sft", stp) for stp in out["checkpoints"]},
            "gpu_seconds": time.time() - t0,
        }

    hashes = [it["hash"] for it in env.e2_items]
    res = {
        "construction_e2": {"items": env.e2_items, "shard_hashes": hashes, "forbidden_hashes": []}
    }
    to_key = lambda c: [f"lora_sft:step{c[1]}"]
    bundles_of = lambda c: [LR.lowrank_actions(ck[tuple(c)], 1.0)]
    ctx, ledger = _ctx(env, arm, ep, B, res, to_key, bundles_of)
    env._last_lora_ckpts = ck
    out = DI.run_lora_sft(ctx, train_fn, B, seed, total)
    return _ctx_payload(
        arm, ep, out, ctx, ledger, steps_at=out["steps_at"], costs=out["costs"], total_steps=total
    )


def uniform_stream(seed: int, sub):
    rng = np.random.default_rng(seed)
    while True:
        yield SB.draw_candidate(rng, list(sub))


def run_two_stage(env: Env, ep: dict, B: int) -> dict:
    arm = "direct_bon_two_stage"
    seed = AS.arm_seed(arm, int(ep["seeds"]["planner"]))
    k = EO.TWO_STAGE["screen_groups"]
    n_full = EO.TWO_STAGE["n_full"]
    cu = k / 64.0
    ledger = PL.BudgetLedger(B, [arm])
    seen, screened, survivors = set(), [], []
    stream = uniform_stream(seed, ep["subsample"])
    draws = 0
    while ledger.remaining(arm) - cu >= n_full - 1e-12 and draws < 100000:
        c = next(stream)
        draws += 1
        key = E2.candidate_key(c)
        if key in seen:
            continue
        seen.add(key)
        s = env.world.screen(list(c), k=k)
        ledger.charge(arm, key, s["seconds"], kind="screen", units=cu)
        screened.append(
            {
                "candidate": list(c),
                "gain": s["gain16"],
                "guard_ok": s["guard16_ok"],
                "seconds": s["seconds"],
            }
        )
        if s["gain16"] > 0 and s["guard16_ok"]:
            survivors.append((s["gain16"], len(screened), list(c)))
    survivors.sort(key=lambda t: (-t[0], t[1]))
    full = []
    for _g, _i, c in survivors[:n_full]:
        if ledger.remaining(arm) < 1.0 - 1e-12:
            break
        rec = env.world.evaluate(c)
        ledger.charge(
            arm, E2.candidate_key(c), rec["seconds"], kind="evaluated", theta_hash=rec["theta_hash"]
        )
        rec["order"] = len(full)
        rec["guard_ok"] = rec["guards_pass"]
        full.append(rec)
    ledger.close(arm, f"two-stage: {len(screened)} screens, {len(full)} full")
    w = AS.select_winner(full)
    return {
        "arm": arm,
        "episode": ep["index"],
        "winner": None if w is None else _strip(w),
        "evaluated": [_strip(r) for r in full],
        "screens": screened,
        "ledger": ledger.report(),
    }


def run_sbon(env: Env, ep: dict, budget: float) -> dict:
    arm = "sbon"
    ledger = PL.BudgetLedger(budget, [arm])

    class W:
        theta_hash = staticmethod(lambda pids: env.world.theta_hash(list(pids)))
        screen = staticmethod(lambda pids: env.world.screen(list(pids), k=SB.N_SCREEN))
        complete = staticmethod(lambda pids: env.world.complete(list(pids), k=SB.N_SCREEN))

    out = SB.run_sbon(W, ep, budget, ledger, arm=arm)
    w = out["winner"]
    return {
        "arm": arm,
        "episode": ep["index"],
        "winner": None if w.get("fallback") else _strip(w),
        "plan": out["plan"],
        "screened": out["screened"],
        "promoted": out["promoted"],
        "cache_hits": out["cache_hits"],
        "units_used": out["units_used"],
        "forfeited": out["forfeited"],
        "ledger": ledger.report(),
    }
