"""Worker handlers for the E2 stage, the planning fits and the sealed test read (registered into
``wsp.wm.c1.serve.HANDLERS``). Every input that comes from an earlier unit arrives as an uploaded file whose sha256 is
in the request args and is verified before use; resident objects are rebuilt when a digest changes.

Edits are named by a JSON "edit" spec: {"kind": "pids", "pids": [...]} (library primitives in order),
{"kind": "steer", "layer", "contrast", "shift"} (best-of-N steering), {"kind": "lora_sft", "file", "sha256"}
(a LoRA-SFT checkpoint's factors) or {"kind": "base"}.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

from wsp.wm.c1 import serve as S
from wsp.wm.c1.serve import HANDLERS

DEV_OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
    "mathcomp_gold_lp",
)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified(path, sha):
    if _sha(path) != sha:
        raise ValueError(f"{path}: sha256 mismatch for an uploaded input")
    return Path(path)


def register(name):
    def deco(fn):
        HANDLERS[name] = fn
        return fn

    return deco


class E2State:
    """Per-runtime E2 objects keyed by input digests."""

    def __init__(self):
        self.key = None
        self.env = None
        self.world = None


def _dev_items(rt):
    from wsp.wm import panel2 as P2

    split = "construction" if rt.cfg.smoke else "dev"
    its = P2.read_open(rt.cfg.bank_panel, split)
    if rt.cfg.smoke:
        return {o: sorted(its[o], key=lambda r: r["id"])[:4] for o in DEV_OBJECTIVES}
    return {o: its[o] for o in DEV_OBJECTIVES}


def _c1_manifest(rt):
    return json.loads((rt.cfg.bank_panel / "manifest.json").read_text())


def env_for(rt, args, ctx, need_fit=True):
    from wsp.wm.c1 import e2arms as EA
    from wsp.wm.c1 import e2ops as EO

    key = (args["library_sha256"], args["dev_base_sha256"], args.get("fit_sha256"))
    st = getattr(rt, "_e2", None) or E2State()
    rt._e2 = st
    if st.key != key:
        if st.world is not None:
            st.world.close()
        m = rt.hf()
        lib = EO.Library(_verified(args["library_file"], args["library_sha256"]), m.rm)
        base = json.loads(_verified(args["dev_base_file"], args["dev_base_sha256"]).read_text())[
            "records"
        ]
        world = EO.DevWorld(m, _dev_items(rt), base, lib, check=ctx.check)
        fitted = support = q = None
        mode = "ranks"
        if need_fit and args.get("fit_file"):
            fit = pickle.loads(_verified(args["fit_file"], args["fit_sha256"]).read_bytes())
            fitted, support, q, mode = fit["fitted"], fit["support"], fit["q_hat"], fit["mode"]
        _, e2_items, _ = rt.shard_items()
        st.env = EA.Env(
            m, world, lib, rt.rcache(), fitted, support, q, mode, _c1_manifest(rt), e2_items
        )
        st.world, st.key = world, key
    st.world.check = ctx.check
    return st.env


@register("library")
def h_library(rt, req, ctx):
    from wsp.wm.c1 import e2ops as EO

    m = rt.hf()
    m.at_base()
    _, e2_items, sources = rt.shard_items()
    seeds = rt.manifest["e2.library.lora_seeds"]
    out = rt.cfg.state / "library"
    pay = EO.build_library(m, e2_items, sources, seeds, out, _c1_manifest(rt), check=ctx.check)
    return pay, {"library.pt": out / "library.pt"}


@register("dev_base")
def h_dev_base(rt, req, ctx):
    m = rt.hf()
    m.at_base()
    t0 = time.time()
    res, _ = m.evaluate(_dev_items(rt))
    p = ctx.scratch / "dev_base.json"
    p.write_text(json.dumps({"records": res}, sort_keys=True))
    return {"sha256": _sha(p), "seconds": time.time() - t0, "theta_hash": m.theta_hash()}, {
        "dev_base.json": p
    }


@register("fit")
def h_fit(rt, req, ctx):
    from wsp.wm.c1 import field as F
    from wsp.wm.c1 import fitops as FO

    a = req["args"]
    bank = json.loads(_verified(a["bank_file"], a["bank_sha256"]).read_text())
    spec = F.spec_9b()
    base_obs = FO.base_artefact(bank, rt.hf().base_hash)
    t0 = time.time()
    fit = FO.fit_all(
        bank,
        a["calibration"],
        a["strata"],
        int(a["inner_seed"]),
        base_obs,
        spec,
        log=lambda s: ctx.progress(s),
    )
    ctx.check()
    inert = FO.inert_predictions(fit, bank, spec)
    p = ctx.scratch / "fit.pkl"
    p.write_bytes(pickle.dumps(fit))
    return {
        "q_hat": fit["q_hat"],
        "coverage": fit["coverage"],
        "mode": fit["mode"],
        "f_k": fit["f_k"],
        "n_train": fit["n_train"],
        "n_cov_units": fit["n_cov_units"],
        "n_scores": fit["n_scores"],
        "shrink": {arm: float(fit["fitted"][arm].shrink) for arm in ("wm", "static")},
        "train_components": fit["train_components"],
        "c2i": inert,
        "sha256": _sha(p),
        "seconds": time.time() - t0,
    }, {"fit.pkl": p}


@register("e2_arm")
def h_e2_arm(rt, req, ctx):
    from wsp.wm.c1 import e2arms as EA

    a = req["args"]
    env = env_for(rt, a, ctx)
    ep = EA.episode(a["episode"])
    arm, B = a["arm"], int(a["B"])
    t0 = time.time()
    if arm in ("wm_e", "statnn", "field_ranked"):
        out = EA.run_planner(env, arm, ep, B)
    elif arm == "wm_greedy":
        out = EA.run_greedy(env, ep, B)
    elif arm in ("random", "scripted_reader"):
        out = EA.run_pool(env, arm, ep, B, int(a["pool_size"]))
    elif arm == "bon_steering":
        out = EA.run_bon_steering(env, ep, B)
    elif arm == "lora_sft":
        out = EA.run_lora_sft(env, ep, B)
    elif arm == "direct_bon_two_stage":
        out = EA.run_two_stage(env, ep, B)
    else:
        raise KeyError(arm)
    out["seconds"] = time.time() - t0
    files = {}
    if arm == "lora_sft" and out.get("winner"):
        import torch

        step = int(out["winner"]["candidate"][1])
        out["winner_edit"] = {"kind": "lora_sft", "step": step}
        fac = getattr(env, "_last_lora_ckpts", None)
        if fac is not None and step is not None:
            p = ctx.scratch / "lora_sft_winner.pt"
            torch.save(fac[("lora_sft", step)], p)
            out["winner_edit"]["sha256"] = _sha(p)
            files["lora_sft_winner.pt"] = p
    return out, files


@register("sbon_arm")
def h_sbon_arm(rt, req, ctx):
    from wsp.wm.c1 import e2arms as EA

    a = req["args"]
    env = env_for(rt, a, ctx, need_fit=False)
    t0 = time.time()
    out = EA.run_sbon(env, EA.episode(a["episode"]), float(a["budget"]))
    out["seconds"] = time.time() - t0
    return out, {}


# ---- stateless sub-requests for the controller-side LLM reader arms (inside one reader arm-episode unit)
@register("dev_eval")
def h_dev_eval(rt, req, ctx):
    a = req["args"]
    env = env_for(rt, a, ctx, need_fit=False)
    return env.world.evaluate(list(a["pids"])), {}


@register("theta_of")
def h_theta_of(rt, req, ctx):
    a = req["args"]
    env = env_for(rt, a, ctx, need_fit=False)
    return {"theta_hash": env.world.theta_hash(list(a["pids"]))}, {}


@register("describe")
def h_describe(rt, req, ctx):
    from wsp.wm.c1 import e2arms as EA

    a = req["args"]
    env = env_for(rt, a, ctx, need_fit=False)
    if "subsample" in a:
        return {"meta": EA.primitive_meta(env, list(a["subsample"]))}, {}
    feats = EA.field_features(env)
    return {"features": [feats(list(c)) for c in a["candidates"]]}, {}


# ---- the single sealed test read (only after the winners manifest is frozen)
def edit_bundles(env, edit: dict, files_dir=None):
    import torch

    from wsp.wm import lora as LR

    k = edit["kind"]
    if k == "base":
        return []
    if k == "pids":
        return env.lib.bundles(list(edit["pids"]))
    if k == "steer":
        return [[env.lib.steer(int(edit["layer"]), edit["contrast"], float(edit["shift"]))]]
    if k == "lora_sft":
        fac = torch.load(
            _verified(edit["file"], edit["sha256"]), map_location="cpu", weights_only=False
        )
        return [LR.lowrank_actions(fac, 1.0)]
    raise KeyError(k)


@register("sealed_eval")
def h_sealed_eval(rt, req, ctx):
    """Score base and every frozen winner once on the uploaded sealed_test file (one request per chunk of winners)."""
    a = req["args"]
    env = env_for(rt, a, ctx, need_fit=False)
    test = [
        json.loads(x) for x in _verified(a["test_file"], a["test_sha256"]).read_text().splitlines()
    ]
    items = {o: [r for r in test if r["objective"] == o] for o in DEV_OBJECTIVES}
    m = env.m
    out = {
        "states": {},
        "ids": {o: [r["id"] for r in v] for o, v in items.items()},
        "groups": {o: [r["group"] for r in v] for o, v in items.items()},
    }
    for w in a["states"]:
        bundles = edit_bundles(env, w["edit"])
        m.at_base()
        t0 = time.time()
        try:
            for b in bundles:
                m.sess.apply_bundle(b)
            th = m.theta_hash()
            if w.get("theta_hash") and th != w["theta_hash"]:
                raise RuntimeError(
                    f"{w['key']}: realised theta {th[:12]} != frozen {w['theta_hash'][:12]}"
                )
            res, _ = m.evaluate(items)
        finally:
            m.restore()
        ctx.check()
        out["states"][w["key"]] = {"theta_hash": th, "records": res, "seconds": time.time() - t0}
        ctx.progress("sealed_state", key=w["key"])
    return out, {}


S.__dict__.setdefault("_E2_REGISTERED", True)
