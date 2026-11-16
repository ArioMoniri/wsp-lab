#!/usr/bin/env python3
"""WM-E pilot 2 (RULING-wme-pilot1 section 3). EXPLORATORY: development measurements, no confirmatory claim.

Phases, run one at a time through agent-tmux (each appends to ``data/wme/pilot2/<run>/ledger.jsonl`` with wall and
CPU seconds and peak GPU/RSS/swap, under a monitor that stops the job above 28 GB device or 0.95 × 24 GB RSS):

* ``field``: base support-only field (gradients + raw K-FAC) on layers 12-15 (down_proj + mixer out) for the six T2
  objectives on 24 construction items each (<= 1024 tokens), and the base validation panel twice (bitwise repeat).
* ``t2``: fresh roots from the ``wme-p2`` seed namespace, signed ladders with the ΔNLL stop rule, 16 three-step and
  8 four-step compositions; validation panel per state; static-field contractions (a, q) per state and objective.
* ``r2``: R2 oracle gradients at every composition-trajectory state (h = 1..4), contracted at once with the step
  increments into and out of that state (trapezoid rule), then deleted (no 6 GB/state archive).
* ``e2 --model 9b|4b``: late-layer steering primitives + rank-8 LoRA task vectors (construction items only,
  <= 0.3 GPU-h each, charged) and 32 candidates per objective scored on dev items only. Test items stay sealed.

Design rules and gate definitions: ``wm.p2design`` (module docstring); objectives: ``wm.panel2``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wsp.wm import actions as A
from wsp.wm import collect as C
from wsp.wm import hf
from wsp.wm import lora as LR
from wsp.wm import p2design as G
from wsp.wm import panel as P
from wsp.wm import panel2 as P2
from wsp.wm.p2data import seed_for


def _p1():
    spec = importlib.util.spec_from_file_location("wme_pilot", ROOT / "scripts/wme_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W1 = _p1()  # Monitor/Phase ledger, residual statistics, contractions (pilot-1 code, unchanged)
Phase, write_json = W1.Phase, W1.write_json

PANEL = ROOT / "data/wme/panels/wme-p2"
MODELS = {"9b": "Qwen/Qwen3.5-9B", "4b": "Qwen/Qwen3.5-4B"}
SUPPORT_LAYERS = (12, 13, 14, 15)
FIELD_ITEMS = 24
FIELD_MAX_TOKENS = 1024
SIGN_Q = {
    "wikitext_nll": 1.0,
    "gsm8k_gold_lp": -1.0,
    "mmlupro_gold_lp": -1.0,
    "refusal_margin_harmful": 0.0,
    "refusal_margin_harmless": 0.0,
    "sycophancy_real": 0.0,
}
T2_EVAL = P2.T2_OBJECTIVES + P2.DESCRIPTIVE  # the proxy is scored as a descriptive endpoint
E2_EVAL = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mathcomp_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
    "sycophancy_gap",
)
P1_ROOT_TENSORS = {(13, "down"), (14, "o")}  # pilot-1 scale roots (not reused)
DIRECTION_POOL = (
    "syc",
    "mathcomp",
    "trivia",
    "gsmwiki",
)  # fresh contrasts (pilot 1: math, mcq, wiki)
E2_LAYERS = (20, 23, 26, 29)
E2_DIRECTIONS = ("syc", "gsm", "mathcomp")
E2_SHIFTS = (-0.0625, -0.03125, 0.03125, 0.0625)
LORA = {"rank": 8, "alpha": 16, "lr": 1e-4, "epochs": 2, "batch_rows": 8, "warmup": 10}
LORA_CAP_SECONDS = 0.3 * 3600  # per objective, everything after model load (charged)

# Engineering smoke test of the code paths (``WME_P2_SMOKE=1``): Qwen3.5-0.8B, construction items only (never dev,
# validation or test), 4 items per objective, truncated ladders/compositions/candidates. Not pilot data.
SMOKE = os.environ.get("WME_P2_SMOKE") == "1"
if SMOKE:
    _rev = min(
        (Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen3.5-0.8B/snapshots").iterdir()
    ).name
    hf.TARGETS["Qwen/Qwen3.5-0.8B"] = _rev
    MODELS["9b"] = "Qwen/Qwen3.5-0.8B"
    E2_LAYERS = (20, 21, 22, 23)
    FIELD_ITEMS = 2
    LORA = {**LORA, "epochs": 1}
    LORA_CAP_SECONDS = 90


def load(ph, model_key="9b"):
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, W1.ALLOCATOR_BYTES / total))
    t = time.time()
    model, tok, cfg, base_hash, _ = hf.load_target(MODELS[model_key], "cuda")
    names = [n for n, _ in model.named_parameters()]
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in names)
        else "model.layers."
    )
    rm = A.RoleMap.from_config(cfg, prefix=prefix)
    ph.log("loaded", model=MODELS[model_key], seconds=round(time.time() - t, 1), prefix=prefix)
    return model, tok, base_hash, rm


def items(split, objectives=None):
    if SMOKE and split != "construction":
        its = P2.read_open(PANEL, "construction")
        return {o: sorted(its[o], key=lambda r: r["id"])[-4:] for o in (objectives or its)}
    its = P2.read_open(PANEL, split)
    return {o: its[o] for o in (objectives or its)}


def manifest():
    return json.loads((PANEL / "manifest.json").read_text())


def timed_eval(model, tok, its):
    torch.cuda.synchronize()
    t = time.time()
    res = P2.evaluate(P.LMAdapter(model), tok, its, P.PanelConfig())
    torch.cuda.synchronize()
    return res, time.time() - t


def session(model, base_hash, rm):
    return A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)


def module_of(n):
    return n.rsplit(".", 1)[0]


def support_modules(rm):
    return [module_of(rm.writers()[(L, r)]) for L in SUPPORT_LAYERS for r in ("down", "o")]


def field_panels(tok):
    cons = items("construction", P2.T2_OBJECTIVES)
    scored = set()
    for split in ("validation", "dev"):
        for its in (items(split) if SMOKE else P2.read_open(PANEL, split)).values():
            scored |= {r["hash"] for r in its}
    cfg = P.PanelConfig()
    from wsp.wm import scorers as Sc

    panels = {}
    for obj, its in cons.items():
        sel = []
        for r in sorted(its, key=lambda r: r["id"]):
            if sum(len(x.ids) for x in P2.rows_for(obj, r, tok, cfg)) <= FIELD_MAX_TOKENS:
                sel.append(r)
            if len(sel) == FIELD_ITEMS:
                break
        if len(sel) < FIELD_ITEMS:
            raise ValueError(
                f"{obj}: only {len(sel)} construction items within {FIELD_MAX_TOKENS} tokens"
            )
        panels[f"construction:{obj}"] = Sc.field_panel(
            f"construction:{obj}", sel, tuple(sorted(scored))
        )
    return panels


def specs_for(panels):
    return P2.objective_specs(list(P2.T2_OBJECTIVES), panels, P.PanelConfig())


# ------------------------------------------------------------------ directions (R0: base passes only)


def contrast_sets(tok, cons):
    chat = lambda u: P.chat_prompt(tok, u)
    first = lambda o: sorted(cons[o], key=lambda r: r["id"])[:48]
    return {
        "harmless": [chat(r["prompt"]) for r in first("refusal_margin_harmless")],
        "syc_suggested": [chat(r["suggested_prompt"]) for r in first("sycophancy_real")],
        "syc_neutral": [chat(r["neutral_prompt"]) for r in first("sycophancy_real")],
        "mathcomp": [chat(r["problem"] + P2.MATH_INSTRUCTION) for r in first("mathcomp_gold_lp")],
        "gsm": [chat(P.gsm8k_user(r)) for r in first("gsm8k_gold_lp")],
        "wiki": [r["text"][:1500] for r in first("wikitext_nll")],
    }


CONTRASTS = {  # name -> (positive set, negative set)
    "syc": ("syc_suggested", "syc_neutral"),
    "mathcomp": ("mathcomp", "harmless"),
    "trivia": ("syc_neutral", "harmless"),
    "gsmwiki": ("gsm", "wiki"),
    "gsm": ("gsm", "harmless"),
}


def build_directions(model, tok, layers, names):
    cons = P2.read_open(PANEL, "construction")
    sets = contrast_sets(tok, cons)
    need = sorted({s for n in names for s in CONTRASTS[n]})
    means = {s: W1.residual_means(model, tok, sets[s], layers)[0] for s in need}
    out = {"layers": list(layers), "dirs": {}, "mlp_mean": {}, "res_norm": {}}
    for n in names:
        pos, neg = CONTRASTS[n]
        for L in layers:
            d = means[pos][L] - means[neg][L]
            out["dirs"][f"{n}@{L}"] = (d / d.norm()).cpu()
    mixed = sets["harmless"][:16] + sets["gsm"][:16] + sets["syc_neutral"][:16]
    _, mlp_mixed, _ = W1.residual_means(model, tok, mixed, layers)
    for L in layers:
        out["mlp_mean"][L] = mlp_mixed[L].cpu()
        out["res_norm"][L] = W1.residual_norm(model, tok, mixed[:24], L)
    return out


def steer(rm, dirs, L, name, shift_frac, dose):
    """Rank-one compile of a residual shift on down_proj L: alpha u m^T / ||m||^2, alpha = shift_frac × res norm."""
    u = dirs["dirs"][f"{name}@{L}"]
    m = dirs["mlp_mean"][L]
    return A.rank_one(
        rm.writers()[(L, "down")],
        u=u,
        v=m / m.norm() ** 2,
        alpha=shift_frac * dirs["res_norm"][L],
        dose=dose,
    )


# ------------------------------------------------------------------ T2 roots


def root_design(rm, hidden):
    """Seeded fresh roots: two scale writers (not pilot-1's), two steer and two proj_remove contrasts, one control."""
    import random

    rng = random.Random(seed_for("roots"))
    writers = sorted(
        (L, r) for L in SUPPORT_LAYERS for r in ("down", "o") if (L, r) not in P1_ROOT_TENSORS
    )
    scale_t = rng.sample(writers, 2)
    pool = list(DIRECTION_POOL)
    rng.shuffle(pool)
    steer_layers = [rng.choice(SUPPORT_LAYERS) for _ in range(2)]
    g = torch.Generator().manual_seed(seed_for("random-control"))
    rnd = torch.randn(hidden, generator=g)
    design = {f"scale_{r}{L}": {"family": "scale", "tensor": [L, r]} for L, r in scale_t}
    for name, L in zip(pool[:2], steer_layers, strict=True):
        design[f"steer_{name}{L}"] = {"family": "steer", "direction": name, "layer": L}
    for name in pool[2:]:
        design[f"projrm_{name}"] = {"family": "proj_remove", "direction": name}
    design["projrm_random"] = {"family": "control", "direction": "random"}
    return design, (rnd / rnd.norm())


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
            fns[rid] = lambda s, d=d: [
                steer(rm, dirs, d["layer"], d["direction"], G.NOMINAL["steer"], s)
            ]
        else:

            def pr(s, d=d):
                alpha, _capped = G.effective_param("proj_remove", s)
                out = []
                for L in SUPPORT_LAYERS:
                    u = rnd if d["direction"] == "random" else dirs["dirs"][f"{d['direction']}@{L}"]
                    out.append(
                        A.proj_remove(rm.writers()[(L, "down")], direction=u, alpha=alpha, dose=1.0)
                    )
                return out

            fns[rid] = pr
    return fns


def apply_steps(sess, R, steps):
    recs = []
    for r, d in steps:
        recs.extend(sess.apply_bundle(R[r](d)))
    return recs


def _jsonl(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _values(res):
    return {o: [r["value"] for r in v] for o, v in res.items()}


def dnll(base_vals, vals):
    b, e = base_vals["wikitext_nll"], vals["wikitext_nll"]
    return sum(x - y for x, y in zip(e, b, strict=True)) / len(b)


# ------------------------------------------------------------------ phase: field


def ph_field(run_dir):
    ph = Phase(run_dir, "field")
    model, tok, base_hash, rm = load(ph)
    val = items("validation", T2_EVAL)
    b1, s1 = timed_eval(model, tok, val)
    b2, s2 = timed_eval(model, tok, val)
    panels = field_panels(tok)
    specs = specs_for(panels)
    lin = module_of(rm.writers()[(12, "o")]).rsplit(".", 1)[0]
    t = time.time()
    C.collect_grouped(
        model,
        tok,
        specs,
        panels,
        [support_modules(rm) + [lin + ".conv1d", lin]],
        "cuda",
        run_dir / "field",
        fisher=True,
        theta_hash=base_hash,
        log=ph.log,
    )
    out = {
        "base_hash": base_hash,
        "panel_content_hash": manifest()["content_hash"],
        "panel_seconds": [s1, s2],
        "bitwise_repeat": P.digest(b1) == P.digest(b2),
        "collect_seconds": time.time() - t,
        "field_items": {k: [i for i in p.text_group_hash] for k, p in panels.items()},
    }
    write_json(run_dir / "field.json", out)
    write_json(run_dir / "base_validation.json", b1)
    ph.close(
        {"summary": {k: out[k] for k in ("panel_seconds", "bitwise_repeat", "collect_seconds")}}
    )


# ------------------------------------------------------------------ phase: t2


def t2_setup(model, tok, rm, ph, run_dir):
    design, rnd = root_design(rm, rm.hidden)
    names = sorted(
        {d["direction"] for d in design.values() if d.get("direction") not in (None, "random")}
    )
    p = run_dir / "directions_t2.pt"
    if p.exists():
        dirs = torch.load(p, weights_only=False)
    else:
        dirs = build_directions(model, tok, SUPPORT_LAYERS, names)
        torch.save(dirs, p)
    families = {rid: d["family"] for rid, d in design.items()}
    c3, c4 = G.make_compositions(families, seed=seed_for("compositions"))
    return design, rnd, dirs, families, c3, c4


def ph_t2(run_dir):
    ph = Phase(run_dir, "t2")
    model, tok, base_hash, rm = load(ph)
    design, rnd, dirs, families, c3, c4 = t2_setup(model, tok, rm, ph, run_dir)
    R = root_fns(rm, dirs, design, rnd)
    val = items("validation", T2_EVAL)
    base, _ = timed_eval(model, tok, val)
    ref = json.loads((run_dir / "base_validation.json").read_text())
    base_matches_field = P.digest(base) == P.digest(ref)
    base_vals = _values(base)
    done = run_dir / "t2_states.jsonl"
    seen = _jsonl(done)
    sess = session(model, base_hash, rm)

    def visit(sid, kind, steps, extra=None):
        if sid in seen:
            return seen[sid]
        recs = apply_steps(sess, R, steps)
        th = sess.theta_hash()
        tokens = A.step_tokens(recs, sess)
        res, secs = timed_eval(model, tok, val)
        sess.restore()
        if sess.theta_hash() != base_hash:
            raise RuntimeError("restore failed")
        rec = {
            "id": sid,
            "kind": kind,
            "steps": steps,
            "theta_hash": th,
            "eval_seconds": secs,
            "tokens": tokens,
            "values": _values(res),
            "item_ids": {o: [r["id"] for r in v] for o, v in res.items()},
            "sycophancy_parts": {
                o: [[r.get("neutral"), r.get("suggested")] for r in v]
                for o, v in res.items()
                if o.startswith("sycophancy")
            },
            **(extra or {}),
        }
        with open(done, "a") as f:
            f.write(json.dumps(rec) + "\n")
        seen[sid] = rec
        ph.log(
            "state", id=sid, seconds=round(secs, 1), dnll=round(dnll(base_vals, rec["values"]), 4)
        )
        return rec

    visited = {}
    for rid in sorted(design):
        fam = design[rid]["family"]
        pos, neg = G.ladder_sides("proj_remove" if fam == "control" else fam)
        if SMOKE:
            pos, neg = pos[:2], neg[:1]
        visited[rid] = {}
        for side, doses in (("pos", pos), ("neg", neg)):

            def ev(d, rid=rid):
                p, capped = G.effective_param(
                    "proj_remove" if design[rid]["family"] == "control" else design[rid]["family"],
                    d,
                )
                return dnll(
                    base_vals,
                    visit(
                        G.ladder_id(rid, d), "ladder", [(rid, d)], {"param": p, "capped": capped}
                    )["values"],
                )

            walk = G.walk_ladder(doses, ev)
            visited[rid][side] = [w["dose"] for w in walk]
            visited[rid][side + "_walk"] = walk
    comps = G.composition_states((c3[:2] + c4[:2]) if SMOKE else (c3 + c4))
    # h = 1 of every composition is its root at dose 1 (built from W0); if that root's ladder stopped before +1
    # the state is still evaluated, as a composition state only (no ladder step, not in the F3u fit)
    h1 = []
    for s in comps:
        r0 = s["steps"][0][0]
        sid = G.ladder_id(r0, 1.0)
        if sid not in seen and sid not in h1:
            h1.append(sid)
            visit(sid, "composition_h1", [(r0, 1.0)], {"h": 1})
    for s in comps:
        visit(s["id"], "composition", s["steps"], {"h": s["h"]})
    # static field on every visited state's cumulative displacement (objective-major)
    field = C.GroupedField.load(run_dir / "field")
    state_steps = {
        G.ladder_id(r, d): [(r, d)]
        for r in visited
        for side in ("pos", "neg")
        for d in visited[r][side]
    }
    state_steps.update({s["id"]: s["steps"] for s in comps})
    state_steps.update({sid: [(sid.split(":")[1], 1.0)] for sid in h1})
    fp = {}
    t = time.time()
    for obj in P2.T2_OBJECTIVES:
        grads, facs = W1.load_objective_field(field, obj)
        for sid, steps in state_steps.items():
            apply_steps(sess, R, steps)
            rows = W1.contract_on(
                grads, facs, field, obj, {n: sess.displacement(n) for n in sess.touched}
            )
            sess.restore()
            a = sum(r["signed"] for r in rows.values())
            q = sum(r["q"] for r in rows.values() if r["q"] is not None)
            fp.setdefault(sid, {})[obj] = {
                "a": a,
                "q": q,
                "F": a + SIGN_Q[obj] * q,
                "unsupported": [n for n, r in rows.items() if r["q"] is None],
            }
        del grads, facs
        torch.cuda.empty_cache()
        ph.log("field contractions", obj=obj)
    again, _ = timed_eval(model, tok, val)
    write_json(
        run_dir / "t2.json",
        {
            "design": design,
            "families": families,
            "compositions_3": c3,
            "compositions_4": c4,
            "visited": visited,
            "base_values": base_vals,
            "base_matches_field_phase": base_matches_field,
            "post_restore_bitwise_equal_base": P.digest(again) == P.digest(base),
            "field": fp,
            "field_seconds": time.time() - t,
            "n_states": len(state_steps),
        },
    )
    ph.close({"summary": {"states": len(state_steps), "base_matches": base_matches_field}})


# ------------------------------------------------------------------ phase: r2


def ph_r2(run_dir):
    ph = Phase(run_dir, "r2")
    model, tok, base_hash, rm = load(ph)
    t2 = json.loads((run_dir / "t2.json").read_text())
    dirs = torch.load(run_dir / "directions_t2.pt", weights_only=False)
    design, rnd = root_design(rm, rm.hidden)
    if json.loads(json.dumps(design)) != t2["design"]:
        raise RuntimeError("root design differs from the t2 phase")
    R = root_fns(rm, dirs, design, rnd)
    seqs = [tuple(s) for s in t2["compositions_3"] + t2["compositions_4"]]
    if SMOKE:
        seqs = seqs[:1]
    prefixes = {}
    for s in seqs:
        for h in range(1, len(s) + 1):
            key = ">".join(s[:h])
            prefixes.setdefault(key, set())
            if h < len(s):
                prefixes[key].add(">".join(s[: h + 1]))
    steps_of = lambda key: [(r, 1.0) for r in key.split(">")]
    panels = field_panels(tok)
    specs = specs_for(panels)
    groups = [support_modules(rm)]
    base_field = C.GroupedField.load(run_dir / "field", verify=False)  # verified in the t2 phase
    sess = session(model, base_hash, rm)
    done = run_dir / "r2_states.jsonl"
    seen = _jsonl(done)

    def E(key):
        if not key:
            return {}
        apply_steps(sess, R, steps_of(key))
        e = {n: sess.displacement(n).clone() for n in sess.touched}
        sess.restore()
        return e

    def diff(e1, e0):
        return {n: e1[n] - e0[n] if n in e0 else e1[n] for n in e1}

    def contract(fld, obj, D):
        return float(sum((fld.gradient(obj, n).to("cuda") * D[n]).sum() for n in D))

    # left terms of step 1 use the base gradients
    if "base" not in seen:
        rec = {"id": "base", "out": {}}
        for child in sorted(k for k in prefixes if ">" not in k):
            D = E(child)
            rec["out"][child] = {o: contract(base_field, o, D) for o in P2.T2_OBJECTIVES}
            base_field.drop_cache()
        with open(done, "a") as f:
            f.write(json.dumps(rec) + "\n")
        seen["base"] = rec
    tmp = run_dir / "oracle_tmp"
    for key in sorted(prefixes, key=lambda k: (k.count(">"), k)):
        if key in seen:
            continue
        apply_steps(sess, R, steps_of(key))
        th = sess.theta_hash()
        t = time.time()
        if tmp.exists():
            shutil.rmtree(tmp)
        C.collect_grouped(
            model, tok, specs, panels, groups, "cuda", tmp, fisher=False, theta_hash=th
        )
        gs = time.time() - t
        sess.restore()
        fld = C.GroupedField.load(tmp, verify=False)
        parent = ">".join(key.split(">")[:-1])
        e_key = E(key)
        d_in = diff(e_key, E(parent))
        rec = {"id": key, "theta_hash": th, "grad_seconds": gs, "in": {}, "out": {}}
        rec["in"] = {o: contract(fld, o, d_in) for o in P2.T2_OBJECTIVES}
        del d_in
        for child in sorted(prefixes[key]):
            d_out = diff(E(child), e_key)
            rec["out"][child] = {o: contract(fld, o, d_out) for o in P2.T2_OBJECTIVES}
            del d_out
        fld.drop_cache()
        del e_key, fld
        shutil.rmtree(tmp)
        torch.cuda.empty_cache()
        with open(done, "a") as f:
            f.write(json.dumps(rec) + "\n")
        seen[key] = rec
        ph.log("oracle state", key=key, grad_seconds=round(gs, 1))
    ph.close({"summary": {"oracle_states": len(prefixes)}})


# ------------------------------------------------------------------ phase: e2


def e2_primitives():
    return [(L, d, s) for L in E2_LAYERS for d in E2_DIRECTIONS for s in E2_SHIFTS]


def ph_e2(run_dir, model_key):
    out_dir = run_dir / f"e2-{model_key}"
    out_dir.mkdir(exist_ok=True)
    ph = Phase(out_dir, f"e2-{model_key}-setup")
    model, tok, base_hash, rm = load(ph, model_key)
    pd = out_dir / "directions_e2.pt"
    if pd.exists():
        dirs = torch.load(pd, weights_only=False)
    else:
        dirs = build_directions(model, tok, E2_LAYERS, list(E2_DIRECTIONS))
        torch.save(dirs, pd)
    ph.close({"summary": {"model": MODELS[model_key]}})
    man = manifest()
    cons = items("construction", P2.E2_OBJECTIVES)
    factors = {}
    for obj in P2.E2_OBJECTIVES:
        fp = out_dir / f"lora_{obj}.pt"
        if fp.exists():
            factors[obj] = torch.load(fp, weights_only=False)["factors"]
            continue
        ph = Phase(out_dir, f"e2-{model_key}-lora-{obj}")
        t0 = time.time()
        rows = LR.sft_rows(obj, cons[obj], tok, P.PanelConfig(), man)
        ad = LR.LoRA(
            model,
            LR.writer_targets(rm),
            rank=LORA["rank"],
            alpha=LORA["alpha"],
            seed=seed_for(f"lora:{obj}"),
        )
        cap = LORA_CAP_SECONDS - (time.time() - t0) - 60
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        log = LR.train(
            model,
            ad,
            rows,
            pad_id=pad,
            lr=LORA["lr"],
            epochs=LORA["epochs"],
            batch_rows=LORA["batch_rows"],
            seed=seed_for(f"lora-order:{obj}"),
            device="cuda",
            max_seconds=cap,
            warmup=LORA["warmup"],
            checkpointing=True,
            log=ph.log,
        )
        factors[obj] = ad.factors()
        torch.save(
            {"factors": factors[obj], "log": log, "objective": obj, "model": MODELS[model_key]}, fp
        )
        del ad
        torch.cuda.empty_cache()
        ph.close(
            {
                "summary": {k: log[k] for k in ("steps", "seconds", "stopped_by_cap", "n_rows")},
                "lora_total_seconds": time.time() - t0,
                "charged": "lora_training",
            }
        )
    ph = Phase(out_dir, f"e2-{model_key}-score")
    dev = items("dev", E2_EVAL)
    base, s0 = timed_eval(model, tok, dev)
    write_json(out_dir / "base_dev.json", base)
    cands = {
        o: G.e2_candidates(o, e2_primitives(), seed=seed_for(f"e2:{o}")) for o in P2.E2_OBJECTIVES
    }
    write_json(out_dir / "candidates.json", cands)
    done = out_dir / "e2_scores.jsonl"
    seen = _jsonl(done)
    sess = session(model, base_hash, rm)
    for obj in P2.E2_OBJECTIVES:
        for c in cands[obj][: 4 if SMOKE else None]:
            key = G.candidate_key(c)
            if key in seen:
                continue
            if c["kind"] == "lora":
                LR.apply_lean(sess, LR.lowrank_actions(factors[c["objective"]], c["lam"]))
            else:
                L, d, s = c["primitive"]
                LR.apply_lean(sess, [steer(rm, dirs, L, d, s, 1.0)])
            th = sess.theta_hash()
            res, secs = timed_eval(model, tok, dev)
            sess.restore()
            if sess.theta_hash() != base_hash:
                raise RuntimeError("restore failed")
            rec = {"id": key, "candidate": c, "theta_hash": th, "seconds": secs, "records": res}
            with open(done, "a") as f:
                f.write(json.dumps(rec) + "\n")
            seen[key] = rec
            ph.log("candidate", key=key, seconds=round(secs, 1))
    again, _ = timed_eval(model, tok, dev)
    write_json(
        out_dir / "e2_meta.json",
        {
            "base_seconds": s0,
            "post_restore_bitwise_equal_base": P.digest(again) == P.digest(base),
            "base_hash": base_hash,
            "model": MODELS[model_key],
            "revision": hf.TARGETS[MODELS[model_key]],
            "lora": LORA,
            "e2_layers": E2_LAYERS,
            "e2_directions": E2_DIRECTIONS,
            "e2_shifts": E2_SHIFTS,
            "unique_evaluations": len(seen),
        },
    )
    ph.close({"summary": {"unique_evaluations": len(seen)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["field", "t2", "r2", "e2"])
    ap.add_argument("--run", default="pilot2-20261116")
    ap.add_argument("--model", default="9b", choices=list(MODELS))
    args = ap.parse_args()
    run_dir = ROOT / "data/wme/pilot2" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "e2":
        ph_e2(run_dir, args.model)
    else:
        {"field": ph_field, "t2": ph_t2, "r2": ph_r2}[args.phase](run_dir)


if __name__ == "__main__":
    main()
