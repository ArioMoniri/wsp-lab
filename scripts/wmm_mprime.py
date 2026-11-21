#!/usr/bin/env python3
"""Study M′ run driver (PREREG_WMM, tag ``prereg-wmm-exploratory`` = 1904bbe; EXPLORATORY). Local GPU only.

Runs ``mprime.build_tasks()`` in the registered order under the FC2 ledger (hard stop 42.0 GPU-h; cut order: M-A
remainder, then the M-TE ARC guard). Every job (process) takes the weight-sketch and batch-1 behavioural canaries at
start and end and writes hash-chained provenance records (base sha256 recomputed from bytes; one record per realised
edit). Outcomes are written per (state, set) to ``OUT/outcomes``; state metadata (with the edit sha256 the FC1
read-gate checks) to ``OUT/states``. Generation uses the frozen Gate 0b harness, the registered panel orders and
batch 32 (OOM: the whole set, both sides, at batch 16). Resumable: done tasks are skipped.

    wmm_mprime.py --out data/wmm/mprime-2026-11-21 [--until PHASE]
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
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wsp.wm import actions as A
from wsp.wm import chat as CH
from wsp.wm import gemma3 as G3
from wsp.wm import lora as LR
from wsp.wm import wmm_gate0b as B
from wsp.wm.wmm import canary as K
from wsp.wm.wmm import gate1 as G1
from wsp.wm.wmm import lora_micro as LM
from wsp.wm.wmm import mprime as M
from wsp.wm.wmm import prereg as P
from wsp.wm.wmm import provenance as PV
from wsp.wm.wmm import roles as R

VRAM_CAP_GB = 16.0
PANELS = ROOT / "data/wmm/prereg/panels"
GATE1 = ROOT / "data/wmm/gate1-2026-11-20"
CANARY_PROMPTS = [
    "Write one sentence about the ocean.",
    "List three prime numbers.",
    "What is the capital of France? Answer in one word.",
    "Explain in one sentence what a hash function is.",
    "Translate 'good morning' into Spanish.",
    "Name a colour that starts with the letter B.",
    "What is 12 times 7?",
    "Give a synonym for 'quick'.",
]
SKETCH_SEED = 20261121
SETS = {  # set -> (panel, prompt variant, max new tokens)
    "mq_neutral": ("mq", "neutral", P.BUDGET),
    "mq_suggested": ("mq", "suggested", P.BUDGET),
    "mm_neutral": ("mm", "neutral", P.BUDGET),
    "harm": ("harm", "raw", G1.GUARD_TOKENS),
    "benign": ("benign", "raw", G1.GUARD_TOKENS),
    "gen": ("gen", "neutral", P.BUDGET),
    "ma_suggested": ("ma", "suggested", P.BUDGET),
}
PROMPT = {
    "neutral": P.neutral_prompt,
    "suggested": P.suggested_prompt,
    "raw": lambda it: it["prompt"],
}
SMOKE = False  # --smoke: EXPOSED Gate 1 items replace every sealed panel and pool; tiny counts; separate out dir


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{P.SALT}:{label}".encode()).hexdigest()[:8], 16)


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def jw(path, rows):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


class Ctx:
    def __init__(self, out: Path):
        self.out = out
        for d in ("outcomes", "states", "lora", "ma", "logs"):
            (out / d).mkdir(parents=True, exist_ok=True)
        self.ledger = M.Ledger(out / "ledger.json")
        reg = out / "batch_registry.json"
        self.registry = M.BatchRegistry(json.loads(reg.read_text()) if reg.exists() else None)
        self.log = PV.ProvenanceLog(out / "provenance.jsonl")
        if SMOKE:
            g1 = json.loads((GATE1 / "items.json").read_text())
            self.panels = {
                "mq": g1["medqa"][:8],
                "mm": g1["medmcqa"][:8],
                "ma": g1["medmcqa"][8:16],
                "harm": g1["harm"][:8],
                "benign": g1["benign"][:8],
                "gen": g1["general"][:8],
            }
        else:
            self.panels = {
                p: jl(PANELS / f"{p}.jsonl") for p in ("mq", "mm", "ma", "harm", "benign", "gen")
            }
        self.by_id = {p: {it["id"]: it for it in v} for p, v in self.panels.items()}
        self.current = "base"  # realised state
        self.sess = None

    def note(self, event, **kw):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}
        print(json.dumps(rec, default=float), flush=True)
        with open(self.out / "logs" / "events.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def save_registry(self):
        (self.out / "batch_registry.json").write_text(json.dumps(self.registry.to_json(), indent=1))


# ------------------------------------------------------------------ job start / end (canaries, provenance)


def yield_gpu(ctx):
    """C1 priority: pause while another process leaves < 2 GB free on the card; the wait is not charged."""
    waited = 0.0
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 >= 2.0 or waited > 6 * 3600:
            return waited
        time.sleep(60)
        waited += 60


def job_start(ctx):
    ctx.ledger.begin()
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < VRAM_CAP_GB + 1:
        raise SystemExit(f"only {free / 2**30:.1f} GB free; C1 or another job is using the GPU")
    torch.cuda.set_per_process_memory_fraction(VRAM_CAP_GB * 2**30 / total)
    model, tok, cfg, _bh, _path = G3.load_text(G3.TARGET, device="cuda")
    ctx.model, ctx.tok, ctx.rm = model, tok, R.GemmaRoleMap.from_config(cfg)
    ctx.params = dict(model.named_parameters())
    ctx.stop = sorted(CH.stop_ids(tok))
    ctx.pad = tok.pad_token_id
    ctx.eot = CH.end_token_id(tok)
    ctx.letter_ids = [
        (tok.convert_tokens_to_ids(L), tok.encode(" " + L, add_special_tokens=False)[-1])
        for L in "ABCD"
    ]
    canary_ids = [
        tok.encode(CH.chat_prompt(tok, u), add_special_tokens=False) for u in CANARY_PROMPTS
    ]
    ctx.canary_ids = canary_ids
    ctx.sketch0 = K.weight_sketch(ctx.params, seed=SKETCH_SEED)
    ctx.bc0 = K.behavioural_canary(model, canary_ids, max_new_tokens=48, stop_ids=ctx.stop)
    ctx.base_hashes = PV.tensor_hashes(ctx.params)
    ctx.base_sha = PV.combine(ctx.base_hashes)
    ref = ctx.out / "canary_reference.json"
    if not ref.exists():
        ref.write_text(
            json.dumps(
                {
                    "behavioural": ctx.bc0["sha256"],
                    "sketch": K.sketch_sha256(ctx.sketch0),
                    "base_sha256": ctx.base_sha,
                }
            )
        )
    r = json.loads(ref.read_text())
    ok = (
        r["behavioural"] == ctx.bc0["sha256"]
        and r["sketch"] == K.sketch_sha256(ctx.sketch0)
        and r["base_sha256"] == ctx.base_sha
    )
    ctx.log.append(
        "job_start",
        job=f"mprime-{int(time.time())}",
        base_sha256=ctx.base_sha,
        sketch_sha256=K.sketch_sha256(ctx.sketch0),
        behavioural_canary=ctx.bc0["sha256"],
        matches_reference=ok,
    )
    if not ok:
        raise SystemExit(
            "canary or base sha256 differs from the run's reference: stop and diagnose"
        )
    (ctx.out / "states" / "base.json").write_text(
        json.dumps({"tag": "base", "edit_sha256": ctx.base_sha})
    )
    ctx.ledger.end("job start: load, canaries, base hashes")
    ctx.note("job_start", base_sha256=ctx.base_sha, used_h=ctx.ledger.used_h)


def job_end(ctx):
    ctx.ledger.begin()
    realise(ctx, "base")
    sk1 = K.weight_sketch(ctx.params, seed=SKETCH_SEED)
    bc1 = K.behavioural_canary(ctx.model, ctx.canary_ids, max_new_tokens=48, stop_ids=ctx.stop)
    cmp = K.compare_sketches(ctx.sketch0, sk1)
    ctx.log.append(
        "job_end",
        sketch_pass=cmp["pass"],
        sketch_changed=cmp["changed"],
        behavioural_match=bc1["sha256"] == ctx.bc0["sha256"],
    )
    ctx.ledger.end("job end: canaries")
    ctx.note(
        "job_end",
        sketch_pass=cmp["pass"],
        behavioural_match=bc1["sha256"] == ctx.bc0["sha256"],
        used_h=ctx.ledger.used_h,
    )
    if not cmp["pass"] or bc1["sha256"] != ctx.bc0["sha256"]:
        raise SystemExit("end-of-job canary mismatch")


# ------------------------------------------------------------------ edits and provenance


def restore(ctx):
    if ctx.sess is not None:
        ctx.sess.restore()
        ctx.sess = None
        for n in ctx.touched:
            if A.tensor_hash(ctx.params[n]) != ctx.base_hashes[n]:
                raise RuntimeError(f"{n}: restore did not return the base bytes")
    ctx.current = "base"


def apply_actions(ctx, tag, recipe, actions):
    restore(ctx)
    ctx.sess = A.EditSession(ctx.params, ctx.base_sha, ctx.rm)
    LR.apply_lean(ctx.sess, actions)
    ctx.touched = list(ctx.sess.touched)
    after = dict(ctx.base_hashes)
    for n in ctx.touched:
        after[n] = A.tensor_hash(ctx.params[n])
    rec = ctx.log.append_edit(recipe, ctx.base_hashes, after, state=tag)
    (ctx.out / "states" / f"{tag}.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "edit_sha256": rec["edit_sha256"],
                "recipe": recipe,
                "n_changed": len(rec["changed_tensors"]),
            }
        )
    )
    ctx.current = tag
    return rec


def lora_actions(path, lam=1.0):
    factors = torch.load(path)
    return LR.lowrank_actions(factors, lam), factors


def realise(ctx, tag):
    if ctx.current == tag:
        return
    if tag == "base":
        restore(ctx)
        return
    if tag in ("msyc", "mte"):
        f = ctx.out / "lora" / f"{tag}.pt"
        acts, _ = lora_actions(f, 1.0)
        apply_actions(
            ctx,
            tag,
            {
                "family": tag,
                "lambda": 1.0,
                "factors_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                "recipe": P.LORA_RECIPE,
            },
            acts,
        )
        return
    if tag == "tei":
        restore(ctx)
        line = G3.WeightLine(
            G3.WeightSource(G3.snapshot(G3.TARGET)),
            G3.WeightSource(G3.snapshot("google/medgemma-4b-it")),
        )
        line.apply(ctx.model, 0.25)
        after = PV.tensor_hashes(ctx.params)
        rec = ctx.log.append_edit(
            {"family": "interp", "alpha": 0.25}, ctx.base_hashes, after, state="tei"
        )
        (ctx.out / "states" / "tei.json").write_text(
            json.dumps({"tag": "tei", "edit_sha256": rec["edit_sha256"]})
        )
        ctx.current = "tei"
        ctx.tei_line = line
        return
    raise KeyError(tag)


# ------------------------------------------------------------------ generation (fixed order, fixed batch)


@torch.no_grad()
def generate(ctx, items, prompts, order, max_new, batch):
    enc = {
        it["id"]: ctx.tok.encode(
            CH.chat_prompt(ctx.tok, prompts[it["id"]]), add_special_tokens=False
        )
        for it in items
    }
    eot = set(ctx.stop)
    out = {}
    for b0 in range(0, len(order), batch):
        idx = order[b0 : b0 + batch]
        L = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[ctx.pad] * (L - len(enc[i])) + enc[i] for i in idx], device="cuda")
        att = torch.tensor(
            [[0] * (L - len(enc[i])) + [1] * len(enc[i]) for i in idx], device="cuda"
        )
        g = ctx.model.generate(
            input_ids=ids,
            attention_mask=att,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=sorted(eot),
            pad_token_id=ctx.pad,
        )
        for i, row in zip(idx, g[:, L:].tolist(), strict=True):
            cut = next((k for k, t in enumerate(row) if t in eot), None)
            new = row if cut is None else row[:cut]
            out[i] = {
                "ids": new,
                "text": ctx.tok.decode(new, skip_special_tokens=True),
                "hit_max": cut is None,
                "prompt_ids": enc[i],
            }
        del g, ids, att
    return out


@torch.no_grad()
def letter_conf(ctx, items, gens):
    """Descriptive (C5): probability of the extracted letter at its position after the model's own reasoning."""
    jobs = []
    for it in items:
        g = gens[it["id"]]
        if g["hit_max"]:
            continue
        pos = G1.answer_position(g["text"], len(it["options"]))
        if pos is None:
            continue
        pre = g["text"][:pos].rstrip(" ")
        jobs.append(
            (
                it["id"],
                g["prompt_ids"] + ctx.tok.encode(pre, add_special_tokens=False),
                g["text"][pos].upper(),
                len(it["options"]),
            )
        )
    out = {}
    jobs.sort(key=lambda j: len(j[1]))
    for b0 in range(0, len(jobs), 8):
        batch = jobs[b0 : b0 + 8]
        T = max(len(j[1]) for j in batch)
        ids = torch.full((len(batch), T), ctx.pad, dtype=torch.long)
        att = torch.zeros((len(batch), T), dtype=torch.long)
        for r, (_i, s, _l, _n) in enumerate(batch):
            ids[r, : len(s)] = torch.tensor(s)
            att[r, : len(s)] = 1
        h = ctx.model.model(
            input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False
        ).last_hidden_state
        for r, (i, s, letter, n) in enumerate(batch):
            p = torch.softmax(ctx.model.lm_head(h[r, len(s) - 1]).float(), -1)
            mass = [float(p[a] + p[b]) for a, b in ctx.letter_ids[:n]]
            out[i] = mass["ABCD".index(letter)] / sum(mass)
        del h
    return out


def score_rows(ctx, set_name, items, gens, conf=None):
    _panel, variant, _ = SETS[set_name]
    rows = []
    for it in items:
        g = gens[it["id"]]
        r = {"id": it["id"], "n_tokens": len(g["ids"]), "hit_max": g["hit_max"]}
        if variant == "raw":
            r.update(refused=G1.is_refusal(g["text"]), truncated=g["hit_max"])
        else:
            s = B.score(it, g["text"], g["hit_max"])
            r.update(
                letter=s["letter"],
                rule=s["rule"],
                correct=s["correct"],
                truncated=s["truncated"],
                parsed=s["letter"] is not None,
            )
            if variant == "suggested":
                r["adopted"] = P.adopted(it, s["letter"])
            if conf is not None:
                r["conf_after_reasoning"] = conf.get(it["id"])
        rows.append(r)
    return rows


def panel_order(ctx, panel, variant):
    if not SMOKE:
        return json.loads((PANELS / f"{panel}.{variant}.order.json").read_text())
    return [it["id"] for it in ctx.panels[panel]]


def run_set(ctx, state, set_name):
    panel, variant, max_new = SETS[set_name]
    items = ctx.panels[panel]
    order = panel_order(ctx, panel, variant)
    prompts = {it["id"]: PROMPT[variant](it) for it in items}
    if SMOKE:
        max_new = min(max_new, 256)
    while True:
        batch = ctx.registry.batch(set_name)
        try:
            torch.cuda.empty_cache()
            gens = generate(ctx, items, prompts, order, max_new, batch)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            stale = ctx.registry.on_oom(set_name, state)
            ctx.save_registry()
            ctx.note("oom", set=set_name, state=state, regenerate=stale)
            cur = ctx.current
            for s in stale:  # PREREG §5: both sides of every pair at batch 16
                realise(ctx, s)
                _gen_and_write(ctx, s, set_name, items, prompts, order, max_new, 16)
            realise(ctx, cur)
    _write(ctx, state, set_name, items, gens, batch)


def _gen_and_write(ctx, state, set_name, items, prompts, order, max_new, batch):
    gens = generate(ctx, items, prompts, order, max_new, batch)
    _write(ctx, state, set_name, items, gens, batch)


def _write(ctx, state, set_name, items, gens, batch):
    conf = letter_conf(ctx, items, gens) if set_name in ("mq_neutral", "mm_neutral") else None
    rows = score_rows(ctx, set_name, items, gens, conf)
    jw(ctx.out / "outcomes" / f"{state}__{set_name}.jsonl", rows)
    jw(
        ctx.out / "outcomes" / f"{state}__{set_name}.text.jsonl",
        [{"id": it["id"], "text": gens[it["id"]]["text"]} for it in items],
    )  # internal (harmful outputs never published)
    ctx.registry.mark_done(set_name, state, batch)
    ctx.save_registry()


@torch.no_grad()
def run_gcal(ctx, state):
    items = ctx.panels["mq"] + ctx.panels["mm"]
    order = panel_order(ctx, "mq", "neutral") + panel_order(ctx, "mm", "neutral")
    by = {it["id"]: it for it in items}
    pre = ctx.tok.encode("Final Answer: (", add_special_tokens=False)
    rows = []
    for i in order:  # batch 1, fixed order
        it = by[i]
        ids = (
            ctx.tok.encode(CH.chat_prompt(ctx.tok, P.neutral_prompt(it)), add_special_tokens=False)
            + pre
        )
        lg = ctx.model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float().cpu()
        rows.append(
            {
                "id": i,
                "logits": [[float(lg[a]), float(lg[b])] for a, b in ctx.letter_ids],
                "gold": it["answer_index"],
                "n_options": len(it["options"]),
            }
        )
    jw(ctx.out / "outcomes" / f"{state}__gcal.jsonl", rows)


# ------------------------------------------------------------------ pools, rewrite, LoRA training


class Row:
    def __init__(self, ids, score_from):
        self.ids, self.score_from = ids, score_from


def pool_items(name, n=None):
    if SMOKE:
        g1 = json.loads((GATE1 / "items.json").read_text())
        its = {
            "pool_msyc": g1["syc_pool"][:8],
            "pool_mte": g1["h1_pool"][:8],
            "pool_ma": g1["h1_pool"][8:24],
        }[name]
        return its
    its = jl(PANELS / f"{name}.jsonl")
    return its[:n] if n else its


def run_pool(ctx, pool, n=None):
    realise(ctx, "base")
    items = pool_items(pool, n)
    prompts = {it["id"]: P.neutral_prompt(it) for it in items}
    lens = {
        it["id"]: len(
            ctx.tok.encode(CH.chat_prompt(ctx.tok, prompts[it["id"]]), add_special_tokens=False)
        )
        for it in items
    }
    order = P.batch_order(
        lens, {it["id"]: P.rank_key("medqa", "train", it["hash"]) for it in items}
    )
    gens = generate(ctx, items, prompts, order, 256 if SMOKE else P.BUDGET, P.BATCH)
    rows = []
    for it in items:
        g = gens[it["id"]]
        s = B.score(it, g["text"], g["hit_max"])
        rows.append(
            {
                "id": it["id"],
                "correct": s["correct"],
                "truncated": s["truncated"],
                "ids": g["ids"],
                "prompt_ids": g["prompt_ids"],
                "text": g["text"],
            }
        )
    jw(ctx.out / "outcomes" / f"base__{pool}.jsonl", rows)


def train_lora(ctx, name, rows, seed_label):
    realise(ctx, "base")
    ad = LR.LoRA(
        ctx.model,
        LR.writer_targets(ctx.rm),
        rank=P.LORA_RECIPE["rank"],
        alpha=P.LORA_RECIPE["alpha"],
        seed=seed_for(f"lora:{seed_label}"),
    )
    torch.cuda.empty_cache()
    wh = PV.combine({n: ctx.base_hashes[n] for n in LR.writer_targets(ctx.rm)})
    log = LM.train(
        ctx.model,
        ad,
        rows,
        pad_id=ctx.pad,
        lr=P.LORA_RECIPE["lr"],
        epochs=P.LORA_RECIPE["epochs"],
        batch_rows=P.LORA_RECIPE["batch_rows"],
        seed=seed_for(f"lora-order:{seed_label}"),
        device="cuda",
        max_seconds=3600.0,
        warmup=P.LORA_RECIPE["warmup"],
        checkpointing=True,
        ce_chunk=1024,
        micro_rows=1,
    )
    f = ctx.out / "lora" / f"{name}.pt"
    torch.save(ad.factors(), f)
    after = PV.combine({n: A.tensor_hash(ctx.params[n]) for n in LR.writer_targets(ctx.rm)})
    meta = {
        k: log[k]
        for k in (
            "steps",
            "seconds",
            "stopped_by_cap",
            "n_rows",
            "epochs",
            "lr",
            "batch_rows",
            "micro_rows",
            "seed",
        )
    }
    meta.update(
        loss_first10=float(np.mean(log["loss"][:10])) if log["loss"] else None,
        loss_last10=float(np.mean(log["loss"][-10:])) if log["loss"] else None,
        writers_unchanged=after == wh,
        factors_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
    )
    (ctx.out / "lora" / f"{name}.json").write_text(json.dumps(meta, indent=1))
    del ad
    torch.cuda.empty_cache()
    ctx.note("lora trained", name=name, **meta)
    return f


def syc_rows(ctx, items, pool_rows):
    rows = []
    for it in items:
        r = pool_rows.get(it["id"])
        if r is None or not r["correct"] or r["truncated"]:
            continue
        p = ctx.tok.encode(
            CH.chat_prompt(ctx.tok, P.suggested_prompt(it)), add_special_tokens=False
        )
        rows.append(Row(p + r["ids"] + [ctx.eot], len(p)))
    return rows


@torch.no_grad()
def run_rewrite(ctx):
    realise(ctx, "base")
    items = pool_items("pool_mte", P.MTE["pool_n"])
    pool = {r["id"]: r for r in jl(ctx.out / "outcomes" / "base__pool_mte.jsonl")}
    keep = [it for it in items if pool[it["id"]]["correct"] and not pool[it["id"]]["truncated"]]
    prompts = {}
    for it in keep:
        q = (
            it["question"]
            + "\n\nOptions:\n"
            + "\n".join(f"({'ABCD'[i]}) {o}" for i, o in enumerate(it["options"]))
        )
        prompts[it["id"]] = P.MTE["rewrite_prompt"].format(
            question=q, answer=G1.strip_thought(pool[it["id"]]["text"])
        )
    lens = {
        i: len(ctx.tok.encode(CH.chat_prompt(ctx.tok, prompts[i]), add_special_tokens=False))
        for i in prompts
    }
    order = P.batch_order(lens, {it["id"]: P.rank_key("medqa", "train", it["hash"]) for it in keep})
    gens = generate(ctx, keep, prompts, order, 512, 16)
    rows, out = [], []
    for it in keep:
        g = gens[it["id"]]
        gold = "ABCD"[it["answer_index"]]
        ok = (not g["hit_max"]) and P.valid_short(
            g["text"], len(g["ids"]), gold, len(it["options"])
        )
        out.append({"id": it["id"], "n_tokens": len(g["ids"]), "valid": ok})
        if ok:
            p = ctx.tok.encode(
                CH.chat_prompt(ctx.tok, P.neutral_prompt(it)), add_special_tokens=False
            )
            rows.append({"ids": p + g["ids"] + [ctx.eot], "score_from": len(p)})
    jw(ctx.out / "outcomes" / "base__mte_rewrite.jsonl", out)
    jw(ctx.out / "lora" / "mte_rows.jsonl", rows)
    ctx.note("rewrite", pool_correct=len(keep), valid=len(rows))


# ------------------------------------------------------------------ M-A units, R0/R1 features


class LazyTau:
    """τ = θ_MG1.5 − θ_G3-it per tensor, read from the snapshots on demand (fp32, on GPU)."""

    def __init__(self):
        self.mg = G3.WeightSource(G3.snapshot(G3.TARGET))
        self.g3 = G3.WeightSource(G3.snapshot("google/gemma-3-4b-it"))

    def __contains__(self, n):
        return n in self.mg.map

    def __getitem__(self, n):
        return (self.mg.get(n).float() - self.g3.get(n).float()).cuda()


class LazySyc:
    def __init__(self, factors):
        self.f = factors

    def __contains__(self, n):
        return n in self.f

    def __getitem__(self, n):
        b, a, s = self.f[n]
        return (s * (b.float() @ a.float())).cuda()


def tv_slice_norms(ctx, tau):
    p = ctx.out / "ma" / "tau_slice_norms.json"
    if p.exists():
        return json.loads(p.read_text())
    out = {}
    for a, z in P.LAYER_GROUPS:
        for comp in ("attn", "mlp"):
            out[f"L{a}-{z}:{comp}"] = math.sqrt(
                sum(float(tau[n].pow(2).sum()) for n in M.slice_names(ctx.rm, (a, z), comp))
            )
    p.write_text(json.dumps(out, indent=1))
    return out


def unit_delta(ctx, unit, tau):
    """Returns (delta dict on GPU, fixed actions) for one M-A unit, scaled to its type's matched norm."""
    t = unit["type"]
    if t == "tv_slice":
        norms = tv_slice_norms(ctx, tau)
        c_tv = 0.25 * float(np.median(list(norms.values())))
        _, rng, comp = unit["id"].split(":")
        a, z = (int(x) for x in rng[1:].split("-"))
        raw = {n: -tau[n] for n in M.slice_names(ctx.rm, (a, z), comp)}
        return M.scale_to(raw, c_tv), None, {"c_tv": c_tv}
    if t == "lora_seed":
        s = int(unit["id"].split(":")[1])
        f = ctx.out / "lora" / f"ma_seed{s}.pt"
        if not f.exists():
            pool = {r["id"]: r for r in jl(ctx.out / "outcomes" / "base__pool_ma.jsonl")}
            items = pool_items("pool_ma")
            half = len(items) // 2
            shard = items[:half] if s % 2 == 0 else items[half:]
            train_lora(ctx, f"ma_seed{s}", syc_rows(ctx, shard, pool), f"ma-seed{s}")
        fac = torch.load(f)
        raw = {n: (sc * (b.float() @ a.float())).cuda() for n, (b, a, sc) in fac.items()}
        c_lora = c_lora_norm(ctx)
        return M.scale_to(raw, c_lora), None, {"c_lora": c_lora}
    if t == "random":
        s = int(unit["id"].split(":")[1])
        c_lora = c_lora_norm(ctx)
        return M.random_direction(ctx.params, ctx.rm, seed=s, norm=c_lora), None, {"c_lora": c_lora}
    if t == "sham":
        _, kind, rng = unit["id"].split(":")
        a, z = (int(x) for x in rng[1:].split("-"))
        return None, M.sham_actions(ctx.rm, kind, (a, z)), {}
    raise KeyError(t)


def c_lora_norm(ctx):
    p = ctx.out / "ma" / "c_lora.json"
    if p.exists():
        return json.loads(p.read_text())["c_lora"]
    fac = torch.load(ctx.out / "lora" / "ma_seed0.pt")
    c = math.sqrt(
        sum(float((s * (b.float() @ a.float())).pow(2).sum()) for b, a, s in fac.values())
    )
    p.write_text(json.dumps({"c_lora": c}))
    return c


def probe_items():
    """R1 probe: the first 32 exposed Gate 1 MedQA items with the base's own suggested-prompt answers."""
    items = json.loads((GATE1 / "items.json").read_text())["medqa"]
    gens = [
        json.loads(x)
        for x in (GATE1 / "gen" / "base_s__mq_syc.jsonl").read_text().splitlines()
        if x
    ]
    return list(zip(items[:32], gens[:32], strict=True))


@torch.no_grad()
def r1_raw(ctx):
    out = []
    for it, g in probe_items():
        seq = g["prompt_ids"] + g["ids"] + [ctx.eot]
        P0 = len(g["prompt_ids"])
        lg = ctx.model(input_ids=torch.tensor([seq], device="cuda")).logits[0].float()
        lp = torch.log_softmax(lg, -1)
        tgt = torch.tensor(seq[P0:], device="cuda")
        ans_lp = lp[P0 - 1 : len(seq) - 1].gather(1, tgt[:, None])[:, 0]
        eos_pos = lp[P0 - 1 : len(seq) - 1, ctx.eot]
        rec = {
            "nll": float(-ans_lp.mean()),
            "eos_end": float(eos_pos[-1]),
            "eos_mean": float(eos_pos[:-1].mean()),
            "first": lp[P0 - 1].cpu(),
        }
        pos = G1.answer_position(g["text"], len(it["options"])) if not g["hit_max"] else None
        if pos is not None:
            pre = g["text"][:pos].rstrip(" ")
            ids = g["prompt_ids"] + ctx.tok.encode(pre, add_special_tokens=False)
            l2 = torch.log_softmax(
                ctx.model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1
            )
            mass = [
                float(torch.logsumexp(torch.stack([l2[a], l2[b]]), 0)) for a, b in ctx.letter_ids
            ]
            sug = G1.wrong_option(it)
            base_letter = "ABCD".index(g["text"][pos].upper())
            rec["logodds"] = mass[sug] - mass[base_letter]
        out.append(rec)
    return out


def r1_features(base, edit):
    kl = [
        float(torch.sum(torch.exp(b["first"]) * (b["first"] - e["first"])))
        for b, e in zip(base, edit, strict=True)
    ]
    lo = [
        e["logodds"] - b["logodds"]
        for b, e in zip(base, edit, strict=True)
        if "logodds" in b and "logodds" in e
    ]
    return {
        "d_nll": float(np.mean([e["nll"] - b["nll"] for b, e in zip(base, edit, strict=True)])),
        "d_eos_end": float(
            np.mean([e["eos_end"] - b["eos_end"] for b, e in zip(base, edit, strict=True)])
        ),
        "d_eos_mean_length": float(
            np.mean([e["eos_mean"] - b["eos_mean"] for b, e in zip(base, edit, strict=True)])
        ),
        "d_logodds_suggested": float(np.mean(lo)) if lo else 0.0,
        "kl_first": float(np.mean(kl)),
    }


def run_ma_unit(ctx, unit_id):
    unit = next(u for u in P.ma_bank() if u["id"] == unit_id)
    tau = ctx.tau if hasattr(ctx, "tau") else LazyTau()
    ctx.tau = tau
    realise(ctx, "base")
    if not hasattr(ctx, "r1_base"):
        ctx.r1_base = r1_raw(ctx)
    delta, fixed, norm_info = unit_delta(ctx, unit, tau)
    tag = "ma_" + unit_id.replace(":", "_")
    if delta is not None:
        acts = [A.additive(n, d, 1.0) for n, d in sorted(delta.items())]
    else:
        acts = fixed
    recipe = {"family": "M-A", "unit": unit_id, "type": unit["type"], **norm_info}
    apply_actions(ctx, tag, recipe, acts)
    if delta is None:
        delta = {n: ctx.sess.displacement(n) for n in ctx.touched}
    syc_f = torch.load(ctx.out / "lora" / "msyc.pt")
    r0 = M.r0_features(delta, tau, LazySyc(syc_f), ctx.params, ctx.rm)
    realised_norm = math.sqrt(
        sum(float(ctx.sess.displacement(n).pow(2).sum()) for n in ctx.touched)
    )
    del delta
    torch.cuda.empty_cache()
    r1 = r1_features(ctx.r1_base, r1_raw(ctx))
    run_set(ctx, tag, "ma_suggested")
    (ctx.out / "ma" / f"{tag}.json").write_text(
        json.dumps(
            {
                "unit": unit_id,
                "type": unit["type"],
                "cluster": unit["cluster"],
                "r0": r0,
                "r1": r1,
                "realised_norm": realised_norm,
                **norm_info,
            },
            indent=1,
        )
    )
    restore(ctx)


# ------------------------------------------------------------------ the runner


def runner_for(ctx):
    def run(task):
        waited = yield_gpu(ctx)
        if waited:
            ctx.ledger.add(-waited, "yield credit (C1 priority)")
        ctx.note("task start", id=task["id"], used_h=ctx.ledger.used_h)
        k = task["kind"]
        if k == "set":
            state = task["state"]
            realise(ctx, state)
            if task["set"] == "gcal":
                run_gcal(ctx, state)
            else:
                run_set(ctx, state, task["set"])
        elif k == "pool":
            name = {
                "msyc:pool": ("pool_msyc", None),
                "mte:pool": ("pool_mte", P.MTE["pool_n"]),
                "ma:pool": ("pool_ma", None),
            }[task["id"]]
            run_pool(ctx, *name)
        elif k == "rewrite":
            run_rewrite(ctx)
        elif k == "train":
            if task["id"] == "msyc:train":
                pool = {r["id"]: r for r in jl(ctx.out / "outcomes" / "base__pool_msyc.jsonl")}
                train_lora(ctx, "msyc", syc_rows(ctx, pool_items("pool_msyc"), pool), "syc")
            else:
                rows = [
                    Row(r["ids"], r["score_from"]) for r in jl(ctx.out / "lora" / "mte_rows.jsonl")
                ]
                train_lora(ctx, "mte", rows, "mte")
        elif k == "ma_unit":
            run_ma_unit(ctx, task["unit"])
        else:
            raise KeyError(k)
        ctx.note("task done", id=task["id"], used_h=ctx.ledger.used_h)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--until", default=None, help="stop after this phase (inclusive)")
    ap.add_argument(
        "--smoke", action="store_true", help="exposed Gate 1 items only; code-path check"
    )
    a = ap.parse_args()
    global SMOKE
    SMOKE = a.smoke
    if SMOKE and "smoke" not in a.out:
        raise SystemExit("--smoke must write to a directory whose name contains 'smoke'")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(out)
    git = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "study": "M′ (EXPLORATORY)",
                "prereg_tag": "prereg-wmm-exploratory",
                "freeze_commit": "1904bbe8c76d45eac1f9a8d94cd3bfe7c5ae5cd3",
                "git_sha": git,
                "cap_h": M.CAP_H,
                "batch": P.BATCH,
                "vram_cap_gb": VRAM_CAP_GB,
            },
            indent=1,
        )
    )
    tasks = M.build_tasks()
    if SMOKE:  # one of each kind: base sets, M-SYC, M-TE, two M-A units of different types
        keep_units = {
            "ma:tv_slice:L0-3:mlp",
            "ma:lora_seed:0",
            "ma:random:0",
            "ma:sham:compensated_mlp_pair:L8-11",
        }
        tasks = [
            t
            for t in tasks
            if t["phase"] != "TE-I" and (t["kind"] != "ma_unit" or t["id"] in keep_units)
        ]
    if a.until:
        phases = list(dict.fromkeys(t["phase"] for t in tasks))
        keep = phases[: phases.index(a.until) + 1]
        tasks = [t for t in tasks if t["phase"] in keep]
    random.seed(0)
    job_start(ctx)
    try:
        M.run_schedule(tasks, ctx.ledger, runner_for(ctx))
    finally:
        job_end(ctx)
    ctx.note("done", used_h=ctx.ledger.used_h, until=a.until)


if __name__ == "__main__":
    main()
