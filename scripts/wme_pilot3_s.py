#!/usr/bin/env python3
"""WM-E pilot 3, stage S (SYNTHESIS-r10 D2; RULING-wme-pilot3 §2). EXPLORATORY. Runs on a Lambda H100.

One resumable process: base validation (twice, bitwise), the 24-item base field (gradient + K-FAC, pilot-2 code),
fresh ``wme-p3`` roots and compositions, then per trajectory state: the R2 validation evaluation, R2 oracle
gradients (trapezoid reference), the static contractions, and per step the R1 quantities at θ₀: S1 exact HVP
(forward-over-reverse on the 24 field items) and S2 nested forward-mode jets (96 field items). Every state is
appended to ``states.jsonl`` and every step to ``steps.jsonl``. The D2 cost check runs after the first step and
stops the stage if a cost factor exceeds 2 × its estimate or the projected total exceeds the 4.5 GPU-h cap.

    wme_pilot3_s.py --out OUT [--smoke]       (--smoke: Qwen3.5-0.8B, 4 items, 2 compositions; not pilot data)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hashlib

import torch

from wsp.wm import actions as A
from wsp.wm import collect as C
from wsp.wm import hf
from wsp.wm import jets as J
from wsp.wm import p2design as G
from wsp.wm import p3panel as PP
from wsp.wm import panel as P
from wsp.wm import panel2 as P2
from wsp.wm import scorers as Sc

PANEL = ROOT / "data/wme/panels/wme-p3s"
REPO = "Qwen/Qwen3.5-9B"
SUPPORT_LAYERS = (12, 13, 14, 15)
OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_gap",
)
SIGN_Q = {  # scripts/wme_pilot2.py (pilot-2 static rule)
    "wikitext_nll": 1.0,
    "gsm8k_gold_lp": -1.0,
    "mmlupro_gold_lp": -1.0,
    "refusal_margin_harmful": 0.0,
    "refusal_margin_harmless": 0.0,
    "sycophancy_gap": 0.0,
}
FIELD_ITEMS = 24
JET_ITEMS = 96
ITEM_MAX_TOKENS = 1024
JET_BATCH_TOKENS = 12000
HVP_BATCH_TOKENS = 1  # one item per forward-over-reverse micro-batch (memory)
S1_STATE: dict = {}  # objective -> why S1 is infeasible (memory precondition); persisted in s1_status.json
JET_MAX_SCORED = (
    768  # vocab-sized fp32 log-probabilities exist (with tangents) at every scored position
)
HVP_MAX_SCORED = 256
P1_P2_ROOT_TENSORS = {(13, "down"), (14, "o"), (15, "down"), (13, "o")}
CONTRAST_POOL = ("refusal", "sycgap", "gsmharmless", "wikimmlu", "mmluharmless", "gsmwiki")
CONTRASTS = {
    "refusal": ("harmful", "harmless"),
    "sycgap": ("syc_suggested", "syc_neutral"),
    "gsmharmless": ("gsm", "harmless"),
    "wikimmlu": ("wiki", "mmlu"),
    "mmluharmless": ("mmlu", "harmless"),
    "gsmwiki": ("gsm", "wiki"),
}
CAP_GPU_H = 6.0  # RULING-wme-pilot3 amendment 1 (was 4.5)
EST = {"jet_forwards": 4.0, "hvp_gradpasses": 3.0}
SMOKE = False


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"wme-p3:{label}".encode()).hexdigest()[:8], 16)


class Ledger:
    def __init__(self, out: Path):
        self.path = out / "ledger.jsonl"
        self.t0 = time.time()

    def log(self, event, **kw):
        rec = {
            "t": round(time.time() - self.t0, 2),
            "event": event,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available()
            else 0,
            "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
            **kw,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


@contextlib.contextmanager
def reference_kernels():
    """Pure-torch GatedDeltaNet/conv kernels (forward-mode capable) inside the block; the fused ones are restored."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    )
    saved = {n: getattr(qwen, n, None) for n in names}
    hf.use_reference_kernels()
    try:
        yield
    finally:
        for n, f in saved.items():
            if f is not None:
                setattr(qwen, n, f)


@contextlib.contextmanager
def eager_attention(model):
    mods = [
        m for m in model.modules() if hasattr(getattr(m, "config", None), "_attn_implementation")
    ]
    saved = [m.config._attn_implementation for m in mods]
    J.eager_attention(model)
    try:
        yield
    finally:
        for m, s in zip(mods, saved, strict=True):
            m.config._attn_implementation = s


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model():
    model, tok, cfg, base_hash, _ = hf.load_target(REPO, "cuda")
    names = [n for n, _ in model.named_parameters()]
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in names)
        else "model.layers."
    )
    rm = A.RoleMap.from_config(cfg, prefix=prefix)
    return model, tok, base_hash, rm


def module_of(n):
    return n.rsplit(".", 1)[0]


def support_params(rm):
    return sorted(rm.writers()[(L, r)] for L in SUPPORT_LAYERS for r in ("down", "o"))


def item_tokens(obj, it, tok):
    return sum(len(r.ids) for r in P2.rows_for(obj, it, tok, P.PanelConfig()))


def field_items(tok, cons):
    out = {}
    for obj in OBJECTIVES:
        sel = [
            r
            for r in sorted(cons[obj], key=lambda r: r["id"])
            if item_tokens(obj, r, tok) <= ITEM_MAX_TOKENS
        ]
        n = 4 if SMOKE else JET_ITEMS
        out[obj] = sel[:n]
        if len(out[obj]) < (4 if SMOKE else FIELD_ITEMS):
            raise ValueError(
                f"{obj}: only {len(out[obj])} field items within {ITEM_MAX_TOKENS} tokens"
            )
    return out


def n_field(obj_items):
    return min(len(obj_items), 2 if SMOKE else FIELD_ITEMS)


# ------------------------------------------------------------------ directions and roots (R0: base passes only)


def contrast_sets(tok, cons):
    chat = lambda u: P.chat_prompt(tok, u)
    first = lambda o: sorted(cons[o], key=lambda r: r["id"])[: (8 if SMOKE else 48)]
    return {
        "harmless": [chat(r["prompt"]) for r in first("refusal_margin_harmless")],
        "harmful": [chat(r["prompt"]) for r in first("refusal_margin_harmful")],
        "syc_neutral": [chat(P.mc_user(r)) for r in first("sycophancy_gap")],
        "syc_suggested": [chat(P.mc_user(r, r["wrong_index"])) for r in first("sycophancy_gap")],
        "gsm": [chat(P.gsm8k_user(r)) for r in first("gsm8k_gold_lp")],
        "mmlu": [chat(P.mc_user(r)) for r in first("mmlupro_gold_lp")],
        "wiki": [r["text"][:1500] for r in first("wikitext_nll")],
    }


def residual_means(model, tok, prompts, layers):
    sums = {L: None for L in layers}
    msum = {L: None for L in layers}
    count = 0
    store = {}
    hooks = [
        model.model.layers[L].mlp.down_proj.register_forward_hook(
            lambda mod, inp, out, L=L: store.__setitem__(L, inp[0].detach().float())
        )
        for L in layers
    ]
    try:
        with torch.no_grad():
            for p in prompts:
                ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
                out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
                for L in layers:
                    h = out.hidden_states[L + 1][0, 1:].float().sum(0)
                    sums[L] = h if sums[L] is None else sums[L] + h
                    m = store[L][0, 1:].sum(0)
                    msum[L] = m if msum[L] is None else msum[L] + m
                count += ids.shape[1] - 1
    finally:
        for h in hooks:
            h.remove()
    return {L: sums[L] / count for L in layers}, {L: msum[L] / count for L in layers}


@torch.no_grad()
def residual_norm(model, tok, prompts, L):
    tot, n = 0.0, 0
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        h = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[
            L + 1
        ][0, 1:]
        tot += h.float().norm(dim=-1).sum().item()
        n += h.shape[0]
    return tot / n


def root_design(hidden):
    rng = random.Random(seed_for("roots"))
    writers = sorted(
        (L, r) for L in SUPPORT_LAYERS for r in ("down", "o") if (L, r) not in P1_P2_ROOT_TENSORS
    )
    scale_t = rng.sample(writers, 2)
    pool = list(CONTRAST_POOL)
    rng.shuffle(pool)
    steer_layers = [rng.choice(SUPPORT_LAYERS) for _ in range(2)]
    g = torch.Generator().manual_seed(seed_for("random-control"))
    rnd = torch.randn(hidden, generator=g)
    design = {f"scale_{r}{L}": {"family": "scale", "tensor": [L, r]} for L, r in scale_t}
    for name, L in zip(pool[:2], steer_layers, strict=True):
        design[f"steer_{name}{L}"] = {"family": "steer", "direction": name, "layer": L}
    for name in pool[2:4]:
        design[f"projrm_{name}"] = {"family": "proj_remove", "direction": name}
    design["projrm_random"] = {"family": "control", "direction": "random"}
    return design, rnd / rnd.norm()


def build_directions(model, tok, cons, names):
    sets = contrast_sets(tok, cons)
    need = sorted({s for n in names for s in CONTRASTS[n]})
    means = {s: residual_means(model, tok, sets[s], SUPPORT_LAYERS)[0] for s in need}
    out = {"dirs": {}, "mlp_mean": {}, "res_norm": {}}
    for n in names:
        pos, neg = CONTRASTS[n]
        for L in SUPPORT_LAYERS:
            d = means[pos][L] - means[neg][L]
            out["dirs"][f"{n}@{L}"] = (d / d.norm()).cpu()
    k = 4 if SMOKE else 16
    mixed = sets["harmless"][:k] + sets["gsm"][:k] + sets["mmlu"][:k]
    _, mlp = residual_means(model, tok, mixed, SUPPORT_LAYERS)
    for L in SUPPORT_LAYERS:
        out["mlp_mean"][L] = mlp[L].cpu()
        out["res_norm"][L] = residual_norm(model, tok, mixed[: (6 if SMOKE else 24)], L)
    return out


def root_fns(rm, dirs, design, rnd):
    fns = {}
    for rid, d in design.items():
        fam = d["family"]
        if fam == "scale":
            L, r = d["tensor"]
            fns[rid] = lambda s, n=rm.writers()[(L, r)]: [  # noqa: B008
                A.scale(n, factor=G.NOMINAL["scale"], dose=s)
            ]
        elif fam == "steer":

            def st(s, d=d):
                L = d["layer"]
                u = dirs["dirs"][f"{d['direction']}@{L}"]
                m = dirs["mlp_mean"][L]
                return [
                    A.rank_one(
                        rm.writers()[(L, "down")],
                        u=u,
                        v=m / m.norm() ** 2,
                        alpha=G.NOMINAL["steer"] * dirs["res_norm"][L],
                        dose=s,
                    )
                ]

            fns[rid] = st
        else:

            def pr(s, d=d):
                alpha, _ = G.effective_param("proj_remove", s)
                out = []
                for L in SUPPORT_LAYERS:
                    u = rnd if d["direction"] == "random" else dirs["dirs"][f"{d['direction']}@{L}"]
                    out.append(
                        A.proj_remove(rm.writers()[(L, "down")], direction=u, alpha=alpha, dose=1.0)
                    )
                return out

            fns[rid] = pr
    return fns


def apply_key(sess, R, key):
    for r in key.split(">"):
        sess.apply_bundle(R[r](1.0))


def displacement(sess, R, key, names):
    """Support displacement of a state (fp32 on GPU, zeros for untouched support tensors); θ₀ is restored."""
    if key:
        apply_key(sess, R, key)
    e = {n: sess.displacement(n).clone() for n in names}
    sess.restore()
    return e


def jsonl(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ R1 at θ₀


def scored_positions(obj, it, tok):
    return sum(len(r.ids) - r.score_from for r in P2.rows_for(obj, it, tok, P.PanelConfig()))


def batches_for(obj, its, tok, max_tokens, max_scored):
    out, cur, ntok, nsc = [], [], 0, 0
    for it in its:
        t = item_tokens(obj, it, tok)
        sc = scored_positions(obj, it, tok)
        if cur and (ntok + t > max_tokens or nsc + sc > max_scored):
            out.append(J.ItemBatch(obj, cur, tok, device="cuda"))
            cur, ntok, nsc = [], 0, 0
        cur.append(it)
        ntok += t
        nsc += sc
    if cur:
        out.append(J.ItemBatch(obj, cur, tok, device="cuda"))
    return out


def r1_step(model, jb, hb, p0, D, E):
    """S2 per-item jets on the 96 field items and S1 HVP terms on the 24, per objective."""
    Db = {n: v.to(torch.bfloat16) for n, v in D.items()}
    Eb = {n: v.to(torch.bfloat16) for n, v in E.items()} if E is not None else None
    out = {}
    for obj in OBJECTIVES:
        js = [J.jets(model, b, p0, Db, Eb) for b in jb[obj]]
        j = {k: torch.cat([x[k] for x in js]).float().cpu() for k in ("J1", "J2", "X")}
        nan = {"gD": float("nan"), "EHD": float("nan"), "DHD": float("nan")}
        if (SMOKE and obj != "gsm8k_gold_lp") or obj in S1_STATE:
            t = nan
        else:
            try:
                torch.cuda.empty_cache()
                t = J.hvp_terms(model, hb[obj], p0, Db, Eb)
            except (
                torch.OutOfMemoryError
            ) as e:  # S1 memory precondition fails for this objective; S2 continues
                S1_STATE[obj] = f"OOM in the forward-over-reverse HVP: {str(e)[:160]}"
                torch.cuda.empty_cache()
                t = nan
        s2 = J.s2_prediction(j)
        n = s2.numel()
        out[obj] = {
            "J1": j["J1"].tolist(),
            "J2": j["J2"].tolist(),
            "X": j["X"].tolist(),
            "S2": float(s2.mean()),
            "S2a": float((j["J1"] + 0.5 * j["J2"]).mean()),
            "s_hat": float(s2.std() / n**0.5) if n > 1 else 0.0,
            "S1": J.s1_prediction(t),
            "S1_terms": t,
        }
    return out


def timed(f):
    sync()
    t = time.time()
    r = f()
    sync()
    return r, time.time() - t


def cost_check(model, jb, hb, p0, D, E, led):
    """D2: cost factors in the first 0.25 GPU-h: nested jet vs plain forward, HVP vs a gradient pass (same kernels)."""
    obj = "refusal_margin_harmful"
    b = jb[obj][0]
    Db = {n: v.to(torch.bfloat16) for n, v in D.items()}
    with torch.no_grad():
        b.values(model, p0)
        _, t_fwd = timed(lambda: b.values(model, p0))
    _, t_jet = timed(lambda: J.jets(model, b, p0, Db, None))
    h = hb[obj][0]

    def gradpass():
        p = {n: v.detach().clone().requires_grad_(True) for n, v in p0.items()}
        g = torch.autograd.grad(h.values(model, p).sum(), list(p.values()))
        return g

    gradpass()
    _, t_grad = timed(gradpass)
    try:
        torch.cuda.empty_cache()
        _, t_hvp = timed(lambda: J.hvp_terms(model, [h], p0, Db, None))
    except torch.OutOfMemoryError as e:
        S1_STATE[obj] = f"OOM in the cost-check HVP: {str(e)[:160]}"
        torch.cuda.empty_cache()
        t_hvp = float("nan")
    rec = {
        "t_forward": t_fwd,
        "t_jet_nested": t_jet,
        "jet_forwards": t_jet / t_fwd,
        "t_gradpass": t_grad,
        "t_hvp": t_hvp,
        "hvp_gradpasses": t_hvp / t_grad,
        "s1_infeasible": dict(S1_STATE),
        "estimates": EST,
    }
    led.log("cost factors", **rec)
    return rec


def static_pass(out, field, sess, R, keys, names, led):
    """Static field F(E) = a + SIGN_Q·q (pilot-2 rule, raw K-FAC) at every state, one objective at a time."""
    path = out / "static.json"
    if path.exists():
        return
    res = {k: {} for k in [""] + keys}
    for o in OBJECTIVES:
        grads = {n: field.gradient(o, n).to("cuda").float() for n in names}
        facs = {}
        for n in names:
            cov = field.coverage[o][n]
            if cov["factors"] and cov["module"] not in facs:
                f = field.factors(o, cov["module"])
                facs[cov["module"]] = (f["A"].to("cuda").float(), f["G"].to("cuda").float())
        field.drop_cache()
        for k in [""] + keys:
            e = displacement(sess, R, k, names)
            a = q = 0.0
            for n in names:
                a += float((grads[n] * e[n]).sum())
                cov = field.coverage[o][n]
                if cov["factors"] and cov["module"] in facs:
                    Af, Gf = facs[cov["module"]]
                    Af = Af[: e[n].shape[-1], : e[n].shape[-1]]
                    q += max(0.0, float(0.5 * torch.trace(Gf @ e[n] @ Af @ e[n].T)))
            res[k][o] = {"a": a, "q": q, "F": a + SIGN_Q[o] * q}
        del grads, facs
        torch.cuda.empty_cache()
        led.log("static", obj=o)
    path.write_text(json.dumps(res))


# ------------------------------------------------------------------ main


def run(out: Path):
    led = Ledger(out)
    led.log("start", smoke=SMOKE, torch=torch.__version__, cuda=torch.version.cuda)
    t_start = time.time()
    model, tok, base_hash, rm = load_model()
    led.log("loaded", base_hash=base_hash)
    cons = PP.read_split(PANEL, "construction")
    val = PP.read_split(PANEL, "validation")
    if SMOKE:  # engineering smoke: construction items only (never validation), not pilot data
        val = {o: sorted(cons[o], key=lambda r: r["id"])[-4:] for o in OBJECTIVES}
    val = {o: val[o] for o in OBJECTIVES}
    cfg = P.PanelConfig()
    fitems = field_items(tok, cons)
    names = support_params(rm)
    params = dict(model.named_parameters())
    sess = A.EditSession(params, base_hash=base_hash, rolemap=rm)

    def evaluate():
        res = P2.evaluate(P.LMAdapter(model), tok, val, cfg)
        return {o: [r["value"] for r in v] for o, v in res.items()}

    base_path = out / "base.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        b1, b2 = evaluate(), evaluate()
        base = {"values": b1, "bitwise_repeat": b1 == b2, "base_hash": base_hash}
        base_path.write_text(json.dumps(base))
        led.log("base validation", bitwise_repeat=base["bitwise_repeat"])
    # 24-item base field (gradients + K-FAC), pilot-2 collector
    scored = sorted({r["hash"] for v in val.values() for r in v})
    panels = {
        f"construction:{o}": Sc.field_panel(
            f"construction:{o}", fitems[o][: n_field(fitems[o])], tuple(scored)
        )
        for o in OBJECTIVES
    }
    specs = P2.objective_specs(list(OBJECTIVES), panels, cfg)
    groups = [[module_of(n) for n in names]]
    if not (out / "field" / "DONE").exists():
        if (out / "field").exists():
            shutil.rmtree(out / "field")
        C.collect_grouped(
            model,
            tok,
            specs,
            panels,
            groups,
            "cuda",
            out / "field",
            fisher=True,
            theta_hash=base_hash,
        )
        (out / "field" / "DONE").write_text("ok")
        led.log("base field")
    field = C.GroupedField.load(out / "field")
    # roots and compositions
    design, rnd = root_design(rm.hidden)
    dpath = out / "directions.pt"
    dnames = sorted(
        {d["direction"] for d in design.values() if d.get("direction") not in (None, "random")}
    )
    if dpath.exists():
        dirs = torch.load(dpath, weights_only=False)
    else:
        dirs = build_directions(model, tok, cons, dnames)
        torch.save(dirs, dpath)
    R = root_fns(rm, dirs, design, rnd)
    fams = {rid: d["family"] for rid, d in design.items()}
    c3, c4 = G.make_compositions(fams, seed=seed_for("compositions"))
    seqs = [tuple(s) for s in (c3[:1] + c4[:1] if SMOKE else c3 + c4)]
    keys = sorted(
        {">".join(s[:h]) for s in seqs for h in range(1, len(s) + 1)},
        key=lambda k: (k.count(">"), k),
    )
    (out / "design.json").write_text(
        json.dumps(
            {
                "design": design,
                "families": fams,
                "c3": c3,
                "c4": c4,
                "states": keys,
                "n_field": {o: n_field(fitems[o]) for o in OBJECTIVES},
                "n_jet": {o: len(fitems[o]) for o in OBJECTIVES},
            },
            indent=1,
        )
    )
    # static field rows per objective (grads + factors) and the R1 item batches
    p0 = {n: params[n] for n in names}
    jb = {o: batches_for(o, fitems[o], tok, JET_BATCH_TOKENS, JET_MAX_SCORED) for o in OBJECTIVES}
    hb = {
        o: batches_for(o, fitems[o][: n_field(fitems[o])], tok, HVP_BATCH_TOKENS, HVP_MAX_SCORED)
        for o in OBJECTIVES
    }
    led.log("setup done", states=len(keys), jet_batches={o: len(v) for o, v in jb.items()})
    spath, stpath = out / "states.jsonl", out / "steps.jsonl"
    if (out / "s1_status.json").exists():
        S1_STATE.update(json.loads((out / "s1_status.json").read_text()))
    seen, seen_steps = jsonl(spath), jsonl(stpath)
    E_cache: dict = {"": {n: torch.zeros_like(params[n], dtype=torch.float32) for n in names}}

    def E(key):
        if key not in E_cache:
            E_cache[key] = displacement(sess, R, key, names)
            if len(E_cache) > 3:
                E_cache.pop(next(k for k in E_cache if k))
        return E_cache[key]

    tmp = out / "oracle_tmp"
    fspecs = specs
    fpanels = panels
    for i, key in enumerate([""] + keys):
        parent = ">".join(key.split(">")[:-1]) if key else None
        if key not in seen:
            t0 = time.time()
            if key:
                apply_key(sess, R, key)
            th = sess.theta_hash() if key else base_hash
            vals = evaluate() if key else base["values"]
            if tmp.exists():
                shutil.rmtree(tmp)
            C.collect_grouped(
                model, tok, fspecs, fpanels, groups, "cuda", tmp, fisher=False, theta_hash=th
            )
            ofld = C.GroupedField.load(tmp, verify=False)
            ograd = {o: {n: ofld.gradient(o, n).float().cpu() for n in names} for o in OBJECTIVES}
            ofld.drop_cache()
            shutil.rmtree(tmp)
            if key:
                sess.restore()
            if sess.theta_hash() != base_hash:
                raise RuntimeError("restore failed")
            # contractions of the oracle gradient at this state with the step into it and the steps out of it
            e_key = E(key)
            rec = {
                "id": key,
                "theta_hash": th,
                "values": vals,
                "seconds_eval_oracle": time.time() - t0,
            }
            if key:
                d_in = {n: e_key[n] - E(parent)[n] for n in names}
                rec["R"] = {
                    o: float(sum((ograd[o][n].cuda() * d_in[n]).sum() for n in names))
                    for o in OBJECTIVES
                }
            rec["L_out"] = {}
            for child in [k for k in keys if ">".join(k.split(">")[:-1]) == key]:
                d_out = {n: E(child)[n] - e_key[n] for n in names}
                rec["L_out"][child] = {
                    o: float(sum((ograd[o][n].cuda() * d_out[n]).sum() for n in names))
                    for o in OBJECTIVES
                }
            append(spath, rec)
            seen[key] = rec
            del ograd
            torch.cuda.empty_cache()
            led.log("state", key=key or "base", seconds=round(time.time() - t0, 1))
        if key and key not in seen_steps:
            t0 = time.time()
            Ep, Ek = E(parent), E(key)
            D = {n: Ek[n] - Ep[n] for n in names}
            Earg = Ep if parent else None
            with reference_kernels(), eager_attention(model), torch.enable_grad():
                if not (out / "cost_check.json").exists():
                    cc = cost_check(model, jb, hb, p0, D, Earg, led)
                    cc["seconds_since_start"] = time.time() - t_start
                    r1, t_step = timed(lambda D=D, Earg=Earg: r1_step(model, jb, hb, p0, D, Earg))
                    cc["t_first_step_r1"] = t_step
                    per_state = t_step + seen[key]["seconds_eval_oracle"]
                    cc["projected_total_gpu_h"] = (
                        time.time() - t_start + per_state * (len(keys) - 1)
                    ) / 3600
                    cc["s1_infeasible_after_first_step"] = dict(S1_STATE)
                    cc["stop"] = bool(
                        cc["jet_forwards"] > 2 * EST["jet_forwards"]
                        or (
                            cc["hvp_gradpasses"] == cc["hvp_gradpasses"]
                            and cc["hvp_gradpasses"] > 2 * EST["hvp_gradpasses"]
                        )
                        or cc["projected_total_gpu_h"] > CAP_GPU_H
                    )
                    (out / "cost_check.json").write_text(json.dumps(cc, indent=1))
                    led.log("cost check", **{k: cc[k] for k in ("stop", "projected_total_gpu_h")})
                    if cc["stop"] and not SMOKE:
                        led.log("STOP: D2 cost rule")
                        return "STOPPED_COST"
                else:
                    r1 = r1_step(model, jb, hb, p0, D, Earg)
            dnorm = float(sum(D[n].float().pow(2).sum() for n in names) ** 0.5)
            srec = {
                "id": key,
                "parent": parent or "",
                "h": key.count(">") + 1,
                "dnorm": dnorm,
                "r1": r1,
            }
            srec["seconds_r1"] = time.time() - t0
            append(stpath, srec)
            seen_steps[key] = srec
            (out / "s1_status.json").write_text(json.dumps(S1_STATE))
            led.log("step r1", key=key, seconds=round(srec["seconds_r1"], 1))
        # the stage cap counts instance time across restarts: <out>/deadline holds the UTC epoch at which R1 stops,
        # leaving time for the static pass over the states visited so far (the steps measured are then reported)
        dl = out / "deadline"
        late = dl.exists() and time.time() >= float(dl.read_text().strip())
        if late or (time.time() - t_start) / 3600 > CAP_GPU_H - 0.1:
            led.log("STOP: stage cap; static pass over the visited states", steps=len(seen_steps))
            status = "STOPPED_CAP_PARTIAL"
            break
    else:
        status = "DONE"
    static_pass(out, field, sess, R, [k for k in keys if k in seen], names, led)
    (out / "s1_status.json").write_text(json.dumps(S1_STATE))
    again = evaluate()
    led.log("done", post_restore_bitwise_equal_base=again == base["values"], status=status)
    (out / "final.json").write_text(
        json.dumps(
            {
                "post_restore_bitwise_equal_base": again == base["values"],
                "status": status,
                "steps_measured": len(seen_steps),
                "states_planned": len(keys),
            }
        )
    )
    return status


def main():
    global SMOKE, REPO, PANEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--panel", default=None)
    a = ap.parse_args()
    if a.panel:
        PANEL = Path(a.panel)
    if a.smoke:
        SMOKE = True
        REPO = "Qwen/Qwen3.5-0.8B"
        snap = (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "hub/models--Qwen--Qwen3.5-0.8B/snapshots"
        )
        hf.TARGETS[REPO] = min(snap.iterdir()).name
        global HVP_BATCH_TOKENS, JET_BATCH_TOKENS, JET_MAX_SCORED
        HVP_BATCH_TOKENS, JET_BATCH_TOKENS, JET_MAX_SCORED = 300, 3000, 200
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if SMOKE:  # shared local GPU: stay under 8 GB
        torch.cuda.set_per_process_memory_fraction(
            7.5e9 / torch.cuda.get_device_properties(0).total_memory
        )
    try:
        status = run(out)
    except Exception:
        traceback.print_exc()
        (out / "STATUS").write_text("FAILED")
        raise
    (out / "STATUS").write_text(status)
    print(status)


if __name__ == "__main__":
    main()
