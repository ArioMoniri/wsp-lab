#!/usr/bin/env python3
"""WM-E pilot 3, stage H (SYNTHESIS-r10 D3-D5; RULING-wme-pilot3 §3-§6). EXPLORATORY. Runs on a Lambda H100.

One resumable process holds an in-process vLLM engine (generation; prefix caching off; edits written in place into
the residual-writer weights and restored bitwise) and a Hugging Face copy of θ₀ (LoRA SFT, SVF policy gradient).
Order (RULING §4.5): H0 measurement (holdout budgets and proxy paths, guard calibration, teacher and STaR data),
LoRA training (H1, its length-matched control, H2), the sizing probe, dev base runs (twice) and the thinking
ceiling, then H1, H2, H3 (three RL arms) and H4 candidates. Every generation record is appended to JSON lines;
every phase is charged to its stage in ``stage_seconds.json`` and a stage stops at its cap.

    wme_pilot3_h.py --out OUT [--smoke]   (--smoke: Qwen3.5-0.8B, tiny counts, construction/holdout items only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wsp.wm import genaudit as GA
from wsp.wm import hf
from wsp.wm import lora as LR
from wsp.wm import p3gen as G
from wsp.wm import p3panel as PP
from wsp.wm import panel as P
from wsp.wm import panel2 as P2
from wsp.wm import svf as V

PANEL = ROOT / "data/wme/panels/wme-p3h"
REPO = "Qwen/Qwen3.5-9B"
CAPS_H = {"H0": 2.5, "H1": 6.0, "H2": 3.0, "R3": 1.0, "H3": 8.0, "H4": 1.0}
MC = ("mmlupro_gen", "supergpqa_gen")
HOLDOUT_GEN_TOKENS = 2048
THINK_TOKENS = 8192
GUARD_TOKENS = 96
IFEVAL_TOKENS = 1024  # ifeval has no holdout split, so its budget is fixed (reported)
CAL_SAMPLES = 8
STAR_SAMPLES = 4
LORA = {"rank": 8, "alpha": 16, "lr": 1e-4, "epochs": 2, "batch_rows": 8, "warmup": 10}
H3 = {
    "steps": 48,
    "prompts": 32,
    "samples": 4,
    "lr": 2e-3,
    "beta": 0.05,
    "temperature": 1.0,
    "micro": 8,
}
SMOKE = False


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"wme-p3:{label}".encode()).hexdigest()[:8], 16)


def jread(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def jwrite(p, obj):
    Path(p).write_text(json.dumps(obj))


class Row:
    def __init__(self, ids, score_from):
        self.ids, self.score_from = ids, score_from


class Ctx:
    def __init__(self, out: Path):
        self.out = out
        self.t0 = time.time()
        self.stage_path = out / "stage_seconds.json"
        self.stage_s = jread(self.stage_path, {})
        self.cur, self.cur_t = None, None

    def log(self, event, **kw):
        rec = {
            "t": round(time.time() - self.t0, 1),
            "stage": self.cur,
            "event": event,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            **kw,
        }
        with open(self.out / "ledger.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def stage(self, name):
        self.flush()
        self.cur, self.cur_t = name, time.time()

    def flush(self):
        if self.cur is not None:
            self.stage_s[self.cur] = self.stage_s.get(self.cur, 0.0) + time.time() - self.cur_t
            self.cur_t = time.time()
            jwrite(self.stage_path, self.stage_s)

    def over_cap(self, name=None):
        self.flush()
        name = name or self.cur
        if self.stage_s.get(name, 0.0) / 3600 >= CAPS_H[name]:
            return True
        # amendment 1: the H instance is limited to 17.7 instance-h; <out>/deadline holds the UTC epoch to stop by
        dl = self.out / "deadline"
        return dl.exists() and time.time() >= float(dl.read_text().strip())


# ------------------------------------------------------------------ engines


def load_engines(ctx):
    from vllm import LLM

    path = hf.snapshot_dir(REPO, hf.TARGETS[REPO])
    llm = LLM(
        model=str(path),
        dtype="bfloat16",
        gpu_memory_utilization=0.12 if SMOKE else 0.45,
        max_num_seqs=32 if SMOKE else 128,
        max_model_len=12288,
        enable_prefix_caching=False,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    vp = llm.apply_model(lambda m: dict(m.named_parameters()))
    vp = vp[0] if isinstance(vp, list) else vp
    vmap = V.writer_map(vp)
    model, tok, _cfg, base_hash, _ = hf.load_target(REPO, "cuda")
    hf.use_reference_kernels()  # HF copy (training, logp): pure-torch GatedDeltaNet; vLLM keeps its own kernels
    hmap = V.writer_map(dict(model.named_parameters()))
    if set(vmap) != set(hmap):
        raise RuntimeError("vLLM and HF writer sets differ")
    hp = dict(model.named_parameters())
    mism = [k for k in vmap if not torch.equal(vp[vmap[k]], hp[hmap[k]])]
    ctx.log(
        "engines loaded",
        writers=len(vmap),
        vllm_vs_hf_writer_mismatch=len(mism),
        base_hash=base_hash,
    )
    if mism:
        raise RuntimeError(f"vLLM writer weights differ from HF for {mism[:3]}")
    ed = V.LiveEditor(vp)
    ctx.llm, ctx.tok, ctx.model, ctx.vmap, ctx.hmap, ctx.ed = llm, tok, model, vmap, hmap, ed
    ctx.base_hash = base_hash
    ctx.writer_names = sorted(vmap)
    ctx.base_checksum = ed.checksum([vmap[k] for k in ctx.writer_names])
    import vllm

    ctx.log("versions", vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda)


def gen(ctx, prompts, max_tokens, temperature=0.0, n=1, seed=None):
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=temperature, top_p=1.0, max_tokens=int(max_tokens), n=n, seed=seed
    )
    outs = ctx.llm.generate(prompts, sp, use_tqdm=False)
    stop = {ctx.tok.eos_token_id, im_end(ctx)}
    res = []
    for o in outs:
        res.append([])
        for c in o.outputs:
            ids = list(c.token_ids)
            while ids and ids[-1] in stop:  # the stop token is not part of the response
                ids.pop()
            res[-1].append({"text": c.text, "ids": ids, "hit_max": c.finish_reason == "length"})
    return res


def seq_logprobs(ctx, seqs):
    """seqs: [(prompt token ids, scored token ids)] -> [(sum log-prob of the scored tokens, n)]."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    prompts = [TokensPrompt(prompt_token_ids=list(a) + list(b)) for a, b in seqs]
    outs = ctx.llm.generate(prompts, sp, use_tqdm=False)
    res = []
    for (a, b), o in zip(seqs, outs, strict=True):
        pl = o.prompt_logprobs
        ids = list(a) + list(b)
        s = 0.0
        for t in range(len(a), len(ids)):
            s += pl[t][ids[t]].logprob
        res.append((s, len(b)))
    return res


def enc(ctx, text):
    return ctx.tok.encode(text, add_special_tokens=False)


def im_end(ctx):
    return ctx.tok.convert_tokens_to_ids("<|im_end|>")


# ------------------------------------------------------------------ edits


def lora_factors(ctx, run, step):
    return torch.load(ctx.out / "lora" / run / f"step{step:05d}.pt", weights_only=False)


def apply_spec(ctx, spec):
    """Write a candidate's deltas into the vLLM writers (θ₀ for base specs)."""
    kind = spec["kind"]
    if kind in ("base", "base_rerun"):
        return
    r = ctx.refusal_dirs if spec.get("project") else None
    proj = (lambda d, L: V.project_out(d, r[L])) if r is not None else (lambda d, L: d)
    deltas = []
    if kind == "lora":
        fac = lora_factors(ctx, spec["run"], spec["step"])
        for hn, f in fac.items():
            k = V.writer_key(hn)
            deltas.append(
                (ctx.vmap[k], lambda f=f, k=k: proj(V.lora_delta(f, spec["lam"], "cuda"), k[0]))
            )
    elif kind == "svf":
        svf_setup(ctx)
        zs = torch.load(
            ctx.out / "svf" / spec["arm"] / f"z{spec['step']:03d}.pt", weights_only=False
        )
        svf = ctx.svf
        for hn, z in zs.items():
            k = V.writer_key(hn)
            deltas.append(
                (
                    ctx.vmap[k],
                    lambda hn=hn, z=z, k=k: proj(svf.delta(hn, z.cuda(), spec["alpha"]), k[0]),
                )
            )
    else:
        raise KeyError(kind)
    ctx.ed.apply(deltas)


def restore(ctx):
    ctx.ed.restore()
    if ctx.ed.checksum([ctx.vmap[k] for k in ctx.writer_names]) != ctx.base_checksum:
        raise RuntimeError("vLLM writers not restored to θ₀")


def spec_key(spec):
    return json.dumps(spec, sort_keys=True)


# ------------------------------------------------------------------ panels


def panels(ctx):
    if not hasattr(ctx, "_pan"):
        ctx._pan = {s: PP.read_split(PANEL, s) for s in ("construction", "holdout", "dev")}
        if SMOKE:  # construction and holdout items only, never dev
            for s in ctx._pan:
                ctx._pan[s] = {p: v[:6] for p, v in ctx._pan[s].items()}
            ctx._pan["dev"] = {p: v for p, v in ctx._pan["holdout"].items()}
            for p in ("ifeval_gen", "oldguard_harmless"):
                ctx._pan["dev"][p] = ctx._pan["construction"].get(p, [])[:4] or []
    return ctx._pan


def prompts_for(ctx, panel, items, thinking=False):
    return [G.chat(ctx.tok, G.user_prompt(panel, it), thinking) for it in items]


def budget_of(ctx, panel):
    if panel in G.GUARDS or panel == "oldguard_harmless":
        return GUARD_TOKENS
    if panel == "ifeval_gen":
        return IFEVAL_TOKENS
    return ctx.budgets[panel]


def score(panel, items, outs, thinking=False):
    recs = []
    for it, o in zip(items, outs, strict=True):
        c = o[0]
        if panel in G.GUARDS or panel == "oldguard_harmless":
            recs.append(
                {
                    "id": it["id"],
                    "refused": GA.detect_refusal(c["text"]),
                    "len": len(c["ids"]),
                    "text": c[
                        "text"
                    ],  # amendment 2(b): raw generation kept (guard answers are <= 96 tokens)
                }
            )
        else:
            v = G.verify(panel, it, c["text"], c["hit_max"], thinking=thinking)
            recs.append(
                {
                    "id": it["id"],
                    "correct": bool(v["correct"]),
                    "len": len(c["ids"]),
                    "trunc": c["hit_max"],
                    "tail": c["text"][-400:],  # amendment 2(b)
                    "parsed": v["parsed"] if G.KIND[panel] == "mc" else str(v["parsed"]),
                }
            )
    return recs


# ------------------------------------------------------------------ H0 measurement


def h0_holdout(ctx):
    p = ctx.out / "holdout_base.json"
    if p.exists():
        d = jread(p)
    else:
        pan = panels(ctx)["holdout"]
        d = {"lengths": {}, "texts": {}}
        for panel in ("mmlupro_gen", "supergpqa_gen", "gsm8k_gen", "math_gen"):
            its = pan[panel]
            outs = gen(ctx, prompts_for(ctx, panel, its), HOLDOUT_GEN_TOKENS)
            d["lengths"][panel] = [
                o[0]["hit_max"] and HOLDOUT_GEN_TOKENS or len(o[0]["ids"]) for o in outs
            ]
            d["texts"][panel] = [
                {
                    "id": it["id"],
                    "text": o[0]["text"],
                    "ids": o[0]["ids"],
                    "hit_max": o[0]["hit_max"],
                }
                for it, o in zip(its, outs, strict=True)
            ]
        jwrite(p, d)
    ctx.budgets = {k: G.budget(v) for k, v in d["lengths"].items()}
    jwrite(ctx.out / "budgets.json", ctx.budgets)
    # base holdout correctness under the budget (censoring at B) and the proxy paths (MC, verified correct)
    pan = panels(ctx)["holdout"]
    ctx.holdout_correct, ctx.proxy_paths = {}, {}
    for panel, recs in d["texts"].items():
        its = {it["id"]: it for it in pan[panel]}
        B = ctx.budgets[panel]
        cor = []
        for r in recs:
            over = r["hit_max"] or len(r["ids"]) > B
            cor.append(bool(G.verify(panel, its[r["id"]], r["text"], over)["correct"]))
        ctx.holdout_correct[panel] = cor
        if panel in MC:
            ctx.proxy_paths[panel] = [
                (enc(ctx, G.chat(ctx.tok, G.user_prompt(panel, its[r["id"]]))), r["ids"])
                for r, c in zip(recs, cor, strict=True)
                if c
            ]
    ctx.log(
        "holdout",
        budgets=ctx.budgets,
        base_holdout_acc={k: float(np.mean(v)) for k, v in ctx.holdout_correct.items()},
    )


def h0_guard_cal(ctx):
    p = ctx.out / "guard_cal.json"
    d = jread(p)
    if d is None:
        pan = panels(ctx)["holdout"]
        d = {}
        for panel in G.GUARDS:
            its = pan[panel]
            pr = prompts_for(ctx, panel, its)
            greedy = [r["refused"] for r in score(panel, its, gen(ctx, pr, GUARD_TOKENS))]
            samp = np.zeros(len(its))
            for s in range(1, CAL_SAMPLES + 1):
                outs = gen(ctx, pr, GUARD_TOKENS, temperature=0.7, seed=s)
                samp += np.array([r["refused"] for r in score(panel, its, outs)], dtype=float)
            pi = samp / CAL_SAMPLES
            d[panel] = {
                "greedy_refused": greedy,
                "pi": pi.tolist(),
                "delta": G.guard_margin(pi, n_cal=128),
                "base_greedy_rate": float(np.mean(greedy)),
            }
        jwrite(p, d)
    ctx.guard_cal = d
    ctx.log(
        "guard calibration",
        **{k: {"delta": v["delta"], "rate": v["base_greedy_rate"]} for k, v in d.items()},
    )


def h0_teacher(ctx):
    """Thinking-mode answers to construction MC items (H1 teacher) and 4 non-thinking samples (H2 / control)."""
    pan = panels(ctx)["construction"]
    for name, thinking in (("teacher", True), ("star", False)):
        p = ctx.out / f"{name}.jsonl"
        if p.exists():
            continue
        rows = []
        for panel in MC:
            its = pan[panel]
            pr = prompts_for(ctx, panel, its, thinking=thinking)
            if thinking:
                outs = gen(ctx, pr, THINK_TOKENS)
            else:
                outs = gen(
                    ctx,
                    pr,
                    ctx.budgets[panel],
                    temperature=0.7,
                    n=STAR_SAMPLES,
                    seed=seed_for("star"),
                )
            for it, o in zip(its, outs, strict=True):
                for j, c in enumerate(o):
                    v = G.verify(panel, it, c["text"], c["hit_max"], thinking=thinking)
                    rec = {
                        "panel": panel,
                        "id": it["id"],
                        "sample": j,
                        "correct": bool(v["correct"]),
                    }
                    if thinking:
                        rec["answer"] = (G.after_think(c["text"]) or "").strip()
                        rec["think_len"] = len(c["ids"])
                    else:
                        rec["ids"] = c["ids"]
                    rows.append(rec)
        with open(p, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        ctx.log(name, n=len(rows), correct=sum(r["correct"] for r in rows))


def sft_data(ctx):
    """H1 rows (teacher answers, correct, <= B tokens), H2 rows (first correct sample), length-matched control rows."""
    pan = {p: {it["id"]: it for it in v} for p, v in panels(ctx)["construction"].items() if p in MC}
    teach = [json.loads(x) for x in (ctx.out / "teacher.jsonl").read_text().splitlines() if x]
    star = [json.loads(x) for x in (ctx.out / "star.jsonl").read_text().splitlines() if x]
    e = im_end(ctx)
    prompt_ids = {}

    def pid(panel, i):
        if (panel, i) not in prompt_ids:
            prompt_ids[(panel, i)] = enc(ctx, G.chat(ctx.tok, G.user_prompt(panel, pan[panel][i])))
        return prompt_ids[(panel, i)]

    h1, h1_len = [], {}
    for r in teach:
        if not r["correct"] or not r["answer"]:
            continue
        t = enc(ctx, r["answer"])
        if len(t) > ctx.budgets[r["panel"]]:
            continue
        a = pid(r["panel"], r["id"])
        h1.append(Row(a + t + [e], len(a)))
        h1_len[(r["panel"], r["id"])] = len(t)
    by_item = {}
    for r in star:
        if r["correct"]:
            by_item.setdefault((r["panel"], r["id"]), []).append(r)
    h2 = []
    for (panel, i), rs in sorted(by_item.items()):
        rs = sorted(rs, key=lambda r: r["sample"])
        a = pid(panel, i)
        h2.append(Row(a + rs[0]["ids"] + [e], len(a)))
    ctrl = []
    for (panel, i), L in sorted(h1_len.items()):
        rs = by_item.get((panel, i))
        if not rs:
            continue
        best = min(rs, key=lambda r: (abs(len(r["ids"]) - L), r["sample"]))
        a = pid(panel, i)
        ctrl.append(Row(a + best["ids"] + [e], len(a)))
    if SMOKE and not (
        h1 and h2 and ctrl
    ):  # engineering smoke only: gold-letter rows so every code path runs
        gold = []
        for panel in MC:
            for i, it in pan[panel].items():
                a = pid(panel, i)
                gold.append(
                    Row(
                        a + enc(ctx, f"The answer is ({G.LETTERS[it['answer_index']]}).") + [e],
                        len(a),
                    )
                )
        h1, h2, ctrl = h1 or gold, h2 or gold, ctrl or gold
    stats = {
        "h1_rows": len(h1),
        "h2_rows": len(h2),
        "lenmatch_rows": len(ctrl),
        "h1_mean_target_len": float(np.mean([len(r.ids) - r.score_from for r in h1]))
        if h1
        else None,
        "h2_mean_target_len": float(np.mean([len(r.ids) - r.score_from for r in h2]))
        if h2
        else None,
        "lenmatch_mean_target_len": float(np.mean([len(r.ids) - r.score_from for r in ctrl]))
        if ctrl
        else None,
    }
    jwrite(ctx.out / "sft_stats.json", stats)
    ctx.log("sft data", **stats)
    return {"h1": h1, "h2": h2, "h1_lenmatch": ctrl}


def train_lora(ctx, run, rows, stage):
    d = ctx.out / "lora" / run
    meta = jread(d / "meta.json")
    if meta is not None:
        return meta
    d.mkdir(parents=True, exist_ok=True)
    model = ctx.model
    ad = LR.LoRA(
        model,
        sorted(ctx.hmap[k] for k in ctx.writer_names),
        rank=LORA["rank"],
        alpha=LORA["alpha"],
        seed=seed_for(f"lora:{run}"),
    )
    pad = ctx.tok.pad_token_id if ctx.tok.pad_token_id is not None else ctx.tok.eos_token_id

    def on_step(step, adapter):
        torch.save(
            {
                k: (b.to(torch.float32), a.to(torch.float32), s)
                for k, (b, a, s) in adapter.factors().items()
            },
            d / f"step{step:05d}.pt",
        )

    ctx.flush()
    left = CAPS_H[stage] * 3600 - ctx.stage_s.get(stage, 0.0)
    log = LR.train(
        model,
        ad,
        rows,
        pad_id=pad,
        lr=LORA["lr"],
        epochs=1 if SMOKE else LORA["epochs"],
        batch_rows=LORA["batch_rows"],
        seed=seed_for(f"lora-order:{run}"),
        device="cuda",
        max_seconds=max(60.0, 0.5 * left),
        warmup=LORA["warmup"],
        checkpointing=True,
        on_step=on_step,
        ce_chunk=1024,
    )
    meta = {
        k: log[k]
        for k in ("steps", "seconds", "stopped_by_cap", "n_rows", "epochs", "lr", "batch_rows")
    }
    meta["loss_first20"] = float(np.mean(log["loss"][:20])) if log["loss"] else None
    meta["loss_last20"] = float(np.mean(log["loss"][-20:])) if log["loss"] else None
    jwrite(d / "meta.json", meta)
    del ad
    torch.cuda.empty_cache()
    ctx.log("lora trained", run=run, **meta)
    return meta


def ckpts(n_steps, k):
    """k evenly spaced checkpoints over 1..n_steps (fraction j/k); identical fractions match across runs."""
    return sorted({max(1, round(n_steps * j / k)) for j in range(1, k + 1)})


# ------------------------------------------------------------------ evaluation


def evaluate(ctx, spec, plist, items_of, tag, thinking=False, extras=True, max_tokens=None):
    """Generate every panel in ``plist`` for one candidate; per-item records; proxy and old-guard extras."""
    path = ctx.out / "evals.jsonl"
    kd = {**spec, "tag": tag, "panels": sorted(plist), "thinking": thinking}
    if max_tokens:
        kd["max_tokens"] = int(max_tokens)
    key = spec_key(kd)
    if not hasattr(ctx, "_evals"):
        ctx._evals = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                ctx._evals[r["key"]] = r
    if key in ctx._evals:
        return ctx._evals[key]
    t0 = time.time()
    apply_spec(ctx, spec)
    try:
        rec = {
            "key": key,
            "spec": spec,
            "tag": tag,
            "thinking": thinking,
            "panels": {},
            "max_tokens": max_tokens,
        }
        for panel in plist:
            its = items_of(panel)
            if spec["kind"] == "base_rerun":
                its = its[::-1]
            mt = max_tokens or (THINK_TOKENS if thinking else budget_of(ctx, panel))
            outs = gen(ctx, prompts_for(ctx, panel, its, thinking=thinking), mt)
            recs = score(panel, its, outs, thinking=thinking)
            if spec["kind"] == "base_rerun":
                recs = recs[::-1]
            rec["panels"][panel] = recs
        if extras and not thinking:
            rec["proxy"] = {}
            for panel in MC:
                if not ctx.proxy_paths[panel]:
                    rec["proxy"][panel] = None
                    continue
                lp = seq_logprobs(ctx, ctx.proxy_paths[panel])
                rec["proxy"][panel] = sum(s for s, _ in lp) / max(1, sum(n for _, n in lp))
            og = panels(ctx)["dev"].get("oldguard_harmless", [])
            if og:
                cfg = P.PanelConfig()
                rows = [
                    r for it in og for r in P._rows_for("refusal_margin_harmless", it, ctx.tok, cfg)
                ]
                lp = seq_logprobs(
                    ctx, [(r.ids[: r.score_from], r.ids[r.score_from :]) for r in rows]
                )
                margins = []
                k = 0
                for it in og:
                    rr = P._rows_for("refusal_margin_harmless", it, ctx.tok, cfg)
                    vals = lp[k : k + len(rr)]
                    k += len(rr)
                    ref = [
                        v
                        for (v, _), r in zip(vals, rr, strict=True)
                        if r.key[2].startswith("refusal|")
                    ]
                    com = [
                        v
                        for (v, _), r in zip(vals, rr, strict=True)
                        if r.key[2].startswith("compliance|")
                    ]
                    margins.append(
                        float(
                            torch.logsumexp(torch.tensor(ref), 0)
                            - torch.logsumexp(torch.tensor(com), 0)
                        )
                    )
                rec["oldguard_margin"] = margins
    finally:
        restore(ctx)
    rec["seconds"] = time.time() - t0
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    ctx._evals[key] = rec
    acc = {
        p: round(float(np.mean([r["correct"] for r in v])), 4)
        for p, v in rec["panels"].items()
        if v and "correct" in v[0]
    }
    ctx.log("eval", tag=tag, spec=spec, seconds=round(rec["seconds"], 1), acc=acc)
    return rec


def dev_items(ctx):
    pan = panels(ctx)["dev"]
    sz = ctx.sizing["n"]
    return lambda panel: pan[panel][: sz.get(panel, len(pan[panel]))]


CAND_PANELS = MC + G.GUARDS


def sizing(ctx):
    p = ctx.out / "sizing.json"
    s = jread(p)
    if s is None:
        meta = jread(ctx.out / "lora" / "h1" / "meta.json")
        spec = {"kind": "lora", "run": "h1", "step": meta["steps"], "lam": 1.0}
        hol = panels(ctx)["holdout"]
        apply_spec(ctx, spec)
        try:
            disc, gdisc = {}, {}
            for panel in MC:
                outs = gen(ctx, prompts_for(ctx, panel, hol[panel]), ctx.budgets[panel])
                cor = [r["correct"] for r in score(panel, hol[panel], outs)]
                disc[panel] = float(np.mean(np.array(cor) != np.array(ctx.holdout_correct[panel])))
            for panel in G.GUARDS:
                outs = gen(ctx, prompts_for(ctx, panel, hol[panel]), GUARD_TOKENS)
                ref = [r["refused"] for r in score(panel, hol[panel], outs)]
                gdisc[panel] = float(
                    np.mean(np.array(ref) != np.array(ctx.guard_cal[panel]["greedy_refused"]))
                )
        finally:
            restore(ctx)
        dev = panels(ctx)["dev"]
        n = {}
        for panel in MC:
            n[panel] = G.primary_n(disc[panel], len(dev[panel]), lo=min(300, len(dev[panel])))
        null = {}
        for panel in G.GUARDS:
            delta = ctx.guard_cal[panel]["delta"]
            n[panel] = G.guard_n(gdisc[panel], delta, len(dev[panel]), lo=min(200, len(dev[panel])))
            null[panel] = G.null_pass_prob(
                max(gdisc[panel], 1 / 128), delta, n[panel], n_sim=400, n_boot=400
            )
        s = {
            "probe": spec,
            "discordance": disc,
            "guard_discordance": gdisc,
            "n": n,
            "null_pass_prob": null,
        }
        jwrite(p, s)
    ctx.sizing = s
    ctx.log("sizing", **s)


def dev_base(ctx):
    items = dev_items(ctx)
    allp = MC + G.SECONDARY + G.GUARDS
    if SMOKE:
        allp = MC + G.GUARDS
    evaluate(ctx, {"kind": "base"}, allp, items, "base")
    evaluate(ctx, {"kind": "base_rerun"}, allp, items, "base_rerun")
    # thinking ceiling (descriptive, never used for selection): one record per panel, while the H0 cap allows
    for p in MC + (() if SMOKE else ("gsm8k_gen", "math_gen", "ifeval_gen")):
        if ctx.over_cap("H0"):
            ctx.log("H0 cap reached; thinking ceiling not measured", panel=p)
            continue
        evaluate(ctx, {"kind": "base"}, (p,), items, "ceiling", thinking=True, extras=False)


def eval_family(ctx, stage, specs, tag):
    items = dev_items(ctx)
    for spec in specs:
        if ctx.over_cap(stage):
            ctx.log("stage cap reached; remaining candidates not evaluated", stage=stage)
            return False
        evaluate(ctx, spec, CAND_PANELS, items, tag)
    return True


def base_rec(ctx):
    return ctx._evals[
        spec_key(
            {
                "kind": "base",
                "tag": "base",
                "panels": sorted(MC + G.SECONDARY + G.GUARDS)
                if not SMOKE
                else sorted(MC + G.GUARDS),
                "thinking": False,
            }
        )
    ]


def primary_gain(ctx, rec):
    b = base_rec(ctx)
    g = []
    for p in MC:
        g.append(
            np.mean([r["correct"] for r in rec["panels"][p]])
            - np.mean([r["correct"] for r in b["panels"][p]])
        )
    return float(np.mean(g))


def top_k(ctx, tag, k):
    recs = [r for r in ctx._evals.values() if r["tag"] == tag]
    return sorted(recs, key=lambda r: -primary_gain(ctx, r))[:k]


def secondaries(ctx, tag, stage):
    if SMOKE:
        return
    items = dev_items(ctx)
    for r in top_k(ctx, tag, 4):
        if ctx.over_cap(stage):
            return
        evaluate(ctx, r["spec"], G.SECONDARY, items, tag + "_secondary", extras=False)


# ------------------------------------------------------------------ H3 SVF policy gradient


def svf_setup(ctx):
    if not hasattr(ctx, "svf"):
        names = sorted(ctx.hmap[k] for k in ctx.writer_names)
        ctx.svf = V.SVF(ctx.model, names)
        ctx.log("svf ready", tensors=len(names))


def rl_arm(ctx, arm):
    d = ctx.out / "svf" / arm
    d.mkdir(parents=True, exist_ok=True)
    meta = jread(d / "meta.json")
    if meta is not None:
        return meta
    svf_setup(ctx)
    names = ctx.svf.names
    ctx.svf.load_state({n: torch.ones_like(ctx.svf.S[n]) for n in names})
    for n in names:
        ctx.svf.zvec(n).requires_grad_(True)
    opt = torch.optim.Adam(list(ctx.svf.z.parameters()), lr=H3["lr"])
    pan = panels(ctx)["construction"]
    pool = [(p, it) for p in MC for it in pan[p]]
    rng = random.Random(seed_for(f"h3:{arm}:order"))
    rng.shuffle(pool)
    model, e = ctx.model, im_end(ctx)
    pad = ctx.tok.pad_token_id if ctx.tok.pad_token_id is not None else ctx.tok.eos_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    hist = []
    steps = 2 if SMOKE else H3["steps"]
    nprom = 4 if SMOKE else H3["prompts"]
    try:
        for step in range(1, steps + 1):
            if ctx.over_cap("H3"):
                ctx.log("H3 cap reached during RL", arm=arm, step=step)
                break
            batch = [pool[((step - 1) * nprom + i) % len(pool)] for i in range(nprom)]
            # sync the policy into vLLM, sample
            ctx.ed.apply([(ctx.vmap[V.writer_key(n)], lambda n=n: ctx.svf.delta(n)) for n in names])
            try:
                prs = [G.chat(ctx.tok, G.user_prompt(p, it)) for p, it in batch]
                outs = gen(
                    ctx,
                    prs,
                    ctx.budgets["mmlupro_gen"],
                    temperature=H3["temperature"],
                    n=H3["samples"],
                    seed=seed_for(f"h3:{arm}:{step}"),
                )
            finally:
                restore(ctx)
            rew, seqs = [], []
            coin = random.Random(seed_for(f"h3:{arm}:coin:{step}"))
            for (p, it), pr, o in zip(batch, prs, outs, strict=True):
                a = enc(ctx, pr)
                for c in o:
                    ok = G.verify(p, it, c["text"], c["hit_max"])["correct"]
                    if arm == "true":
                        r = 1.0 if ok else -1.0
                    elif arm == "random":
                        r = 1.0 if coin.random() < 0.5 else -1.0
                    else:
                        r = 1.0 if (not c["hit_max"] and G.parseable(p, c["text"])) else -1.0
                    rew.append(r)
                    tail = c["ids"] + ([] if c["hit_max"] else [e])
                    seqs.append((a, tail))
            adv = V.group_advantages(rew, H3["samples"])
            opt.zero_grad(set_to_none=True)
            kls, nmb = [], math.ceil(len(seqs) / H3["micro"])
            for m0 in range(0, len(seqs), H3["micro"]):
                mb = seqs[m0 : m0 + H3["micro"]]
                T = max(len(a) + len(b) for a, b in mb)
                ids = torch.full((len(mb), T), pad, dtype=torch.long)
                att = torch.zeros((len(mb), T), dtype=torch.long)
                msk = torch.zeros((len(mb), T), dtype=torch.bool)
                for i, (a, b) in enumerate(mb):
                    ids[i, : len(a) + len(b)] = torch.tensor(a + b)
                    att[i, : len(a) + len(b)] = 1
                    msk[i, len(a) : len(a) + len(b)] = True
                ids, att, msk = ids.cuda(), att.cuda(), msk.cuda()
                seq_ix = torch.arange(len(mb), device="cuda")[:, None].expand(-1, T)[:, 1:][
                    msk[:, 1:]
                ]
                with torch.no_grad():
                    lp0 = V.token_logps(model, ids, att, msk, chunk=1024)
                with (
                    ctx.svf.active()
                ):  # backward inside: checkpoint recomputation must see the same hooks
                    lp = V.token_logps(model, ids, att, msk, chunk=1024)
                    loss, kl = V.pg_loss(lp, lp0, seq_ix, adv[m0 : m0 + len(mb)], H3["beta"])
                    (loss / nmb).backward()
                kls.append(kl)
            opt.step()
            torch.save(ctx.svf.state(), d / f"z{step:03d}.pt")
            hist.append(
                {"step": step, "reward_mean": float(np.mean(rew)), "kl": float(np.mean(kls))}
            )
            ctx.log("rl step", arm=arm, **hist[-1])
    finally:
        model.gradient_checkpointing_disable()
        model.eval()
        for n in names:
            ctx.svf.zvec(n).requires_grad_(False)
    meta = {**H3, "configured_steps": H3["steps"], "steps": len(hist), "history": hist}
    jwrite(d / "meta.json", meta)
    return meta


# ------------------------------------------------------------------ H4


def refusal_dirs(ctx):
    """Per-layer refusal direction: harmful − harmless mean residual (layer output) at the last prompt token."""
    p = ctx.out / "refusal_dirs.pt"
    if p.exists():
        ctx.refusal_dirs = torch.load(p, weights_only=False)
        return
    hol = panels(ctx)["holdout"]
    model = ctx.model
    layers = sorted({k[0] for k in ctx.writer_names})

    def means(panel):
        acc = None
        with torch.no_grad():
            for it in hol[panel]:
                ids = torch.tensor([enc(ctx, G.chat(ctx.tok, it["prompt"]))], device="cuda")
                hs = model.model(
                    input_ids=ids, output_hidden_states=True, use_cache=False
                ).hidden_states
                v = torch.stack([hs[L + 1][0, -1].float() for L in layers])
                acc = v if acc is None else acc + v
        return acc / len(hol[panel])

    diff = means("guard_harmful") - means("guard_harmless")
    ctx.refusal_dirs = {L: (diff[i] / diff[i].norm()).cpu() for i, L in enumerate(layers)}
    torch.save(ctx.refusal_dirs, p)


def pilot_type_rows(ctx):
    """Pilot-2-type gold-answer LoRA rows (teacher-forced "The answer is X"), construction items only."""
    pan = panels(ctx)["construction"]
    cfg = P.PanelConfig()
    out = {}
    for run, panel, obj in (
        ("pt_gsm", "gsm8k_gen", "gsm8k_gold_lp"),
        ("pt_math", "math_gen", "mathcomp_gold_lp"),
    ):
        rows = []
        for it in pan[panel]:
            for r in P2.rows_for(obj, it, ctx.tok, cfg):
                rows.append(Row(list(r.ids), r.score_from))
        out[run] = rows
    return out


# ------------------------------------------------------------------ main


def r3(ctx, m1):
    items = dev_items(ctx)
    s208 = {"kind": "lora", "run": "h1", "step": m1["steps"], "lam": 1.0}
    s104 = {"kind": "lora", "run": "h1", "step": ckpts(m1["steps"], 16)[7], "lam": 1.0}
    plan = [
        ({"kind": "base"}, CAND_PANELS, "r3_raw", None),
        (s208, CAND_PANELS, "r3_raw", None),
        ({"kind": "base"}, MC, "r3_budget4096", 4096),
        (s208, MC, "r3_budget4096", 4096),
        (s104, MC, "r3_budget4096", 4096),
        (s104, CAND_PANELS, "r3_raw", None),
    ]
    for spec, plist, tag, mt in plan:
        if ctx.over_cap("R3"):
            ctx.log("R3 cap or deadline reached; remaining R3 checks not run", spec=spec, tag=tag)
            return
        evaluate(ctx, spec, plist, items, tag, max_tokens=mt)


def run(out: Path):
    (out / "STATUS").write_text(
        "RUNNING"
    )  # amendment 2(d): never leave an earlier run's status in place
    ctx = Ctx(out)
    ctx.stage("H0")
    ctx.log("start", smoke=SMOKE)
    load_engines(ctx)
    h0_holdout(ctx)
    h0_guard_cal(ctx)
    h0_teacher(ctx)
    data = sft_data(ctx)
    ctx.stage("H1")
    m1 = train_lora(ctx, "h1", data["h1"], "H1")
    m1c = train_lora(ctx, "h1_lenmatch", data["h1_lenmatch"], "H1")
    ctx.stage("H2")
    m2 = train_lora(ctx, "h2", data["h2"], "H2")
    ctx.stage("H0")
    sizing(ctx)
    dev_base(ctx)
    k = 2 if SMOKE else 1
    # H1: 16 checkpoints x lambda {0.5, 1}; length-matched control: 8 checkpoints, lambda 1
    ctx.stage("H1")
    h1 = [
        {"kind": "lora", "run": "h1", "step": s, "lam": lam}
        for s in ckpts(m1["steps"], 16 // k // (4 if SMOKE else 1))
        for lam in (0.5, 1.0)
    ]
    if eval_family(ctx, "H1", h1, "H1"):
        eval_family(
            ctx,
            "H1",
            [
                {"kind": "lora", "run": "h1_lenmatch", "step": s, "lam": 1.0}
                for s in ckpts(m1c["steps"], 8 // (4 if SMOKE else 1))
            ],
            "H1_ctrl_lenmatch",
        )
    secondaries(ctx, "H1", "H1")
    # H2: 32 checkpoints, lambda 1 (its 16 / 8 fractions match H1's checkpoints)
    ctx.stage("H2")
    eval_family(
        ctx,
        "H2",
        [
            {"kind": "lora", "run": "h2", "step": s, "lam": 1.0}
            for s in ckpts(m2["steps"], 32 // (8 if SMOKE else 1))
        ],
        "H2",
    )
    secondaries(ctx, "H2", "H2")
    # amendment 2 (SYNTHESIS-r11 R3): raw re-runs at the frozen budget and the 4,096-token budget check
    ctx.stage("R3")
    r3(ctx, m1)
    # H3: three RL arms; 16 checkpoints x alpha {0.5, 1} for the true reward; 8 checkpoints alpha 1 per control
    ctx.stage("H3")
    metas = {arm: rl_arm(ctx, arm) for arm in ("true", "random", "format")}
    ns = metas["true"]["steps"]
    if ns:
        true = [
            {"kind": "svf", "arm": "true", "step": s, "alpha": a}
            for s in ckpts(ns, 16 // (8 if SMOKE else 1))
            for a in (0.5, 1.0)
        ]
        matched = [c for c in true if c["alpha"] == 1.0][
            1::2
        ]  # the control fractions (amendment 1 order)
        ok = eval_family(ctx, "H3", matched, "H3")
        for arm in ("random", "format"):
            if ok and metas[arm]["steps"]:
                ok = eval_family(
                    ctx,
                    "H3",
                    [
                        {"kind": "svf", "arm": arm, "step": s, "alpha": 1.0}
                        for s in ckpts(metas[arm]["steps"], 8 // (4 if SMOKE else 1))
                    ],
                    f"H3_ctrl_{arm}",
                )
        if ok:
            eval_family(ctx, "H3", [c for c in true if c not in matched], "H3")
        secondaries(ctx, "H3", "H3")
    # H4: refusal-orthogonal projections of the 8 best H1/H2/H3 candidates and of pilot-type gold-answer LoRAs
    ctx.stage("H4")
    if ctx.over_cap("H4"):
        ctx.log("H instance limit reached before H4; H4 not run")
        ctx.flush()
        return "DONE_PARTIAL"
    refusal_dirs(ctx)
    pt = pilot_type_rows(ctx)
    pmeta = {run: train_lora(ctx, run, rows, "H4") for run, rows in pt.items()}
    h4 = []
    for tag in ("H1", "H2", "H3"):
        for r in top_k(ctx, tag, 2 if SMOKE else 8):
            h4.append({**r["spec"], "project": True})
    for run in pt:
        for lam in (0.25, 0.5, 1.0, 1.5):
            h4.append(
                {
                    "kind": "lora",
                    "run": run,
                    "step": pmeta[run]["steps"],
                    "lam": lam,
                    "project": True,
                }
            )
    if eval_family(ctx, "H4", h4, "H4"):
        unproj = [
            {"kind": "lora", "run": run, "step": pmeta[run]["steps"], "lam": lam}
            for run in pt
            for lam in (0.25, 0.5, 1.0, 1.5)
        ]
        eval_family(ctx, "H4", unproj, "H4_ctrl_unprojected")
    ctx.flush()
    ctx.log("done", stage_seconds=ctx.stage_s)
    return "DONE"


def main():
    global SMOKE, REPO, H3, LORA
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        SMOKE = True
        REPO = "Qwen/Qwen3.5-0.8B"
        from huggingface_hub import constants

        snap = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B/snapshots"
        hf.TARGETS[REPO] = min(snap.iterdir()).name
        H3 = {**H3, "micro": 1}
        LORA = {**LORA, "batch_rows": 1}
        globals().update(HOLDOUT_GEN_TOKENS=1024, THINK_TOKENS=2048)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
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
