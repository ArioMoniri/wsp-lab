#!/usr/bin/env python3
"""Study M Gate 1 driver, the headroom probe (plan: docs/reviews/wmm/wmm-gate1-plan-2026-11-20.md; rules:
``wsp.wm.wmm.gate1``). EXPLORATORY. Local GPU only, <= 16 GB (Gate 0 loader), niced, stops new phases at 11.5 GPU-h.

Phases (each resumable; outputs under OUT/): designation and exposure -> base (primaries, sycophancy, permuted
copies, guards, calibration) -> LoRA retraining under the frozen harness -> screening of the 5 central doses ->
qualifiers (permuted copies for the top 3 positive edits; guards for edits positive on both, and for the sycophancy
LoRA at λ 1 if it drops adoption by >= 5 pp) -> screening of the other 6 doses -> qualifiers again (top 3 overall)
-> canary -> summary.json. Guard audit items are written for the two blind
judges after the base phase.

    wmm_gate1.py --out data/wmm/gate1-YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wsp.wm import chat as CH
from wsp.wm import gemma3 as G3
from wsp.wm import lora as LR
from wsp.wm import wmm_gate0 as W
from wsp.wm import wmm_gate0b as B
from wsp.wm.wmm import gate1 as G1
from wsp.wm.wmm import lora_micro as LM

_spec = importlib.util.spec_from_file_location("wmm_gate0_driver", ROOT / "scripts/wmm_gate0.py")
G0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G0)
G0.GPU_H_CAP = 11.5

EXPOSURES = [
    ROOT / "data/wmm/gate0-2026-11-19/exposure.json",
    ROOT / "data/wmm/gate0b-2026-11-20/medqa_g0b_exposure.json",
]
DATA = {
    "medqa": ("openlifescienceai/medqa", "153e61cdd129eb79d3c27f82cdf3bc5e018c11b0"),
    "medmcqa": ("openlifescienceai/medmcqa", "91c6572c454088bf71b679ad90aa8dffcd0d5868"),
    "medsafety": (
        "israel-adewuyi/med-safety-bench-reproduced",
        "0c3d752deb205f2f672bee4644f22e70e11bd708",
    ),
    "medquad": ("lavita/MedQuAD", "84ea67f83cec9692ad254eaa02c9731b24ecfe4c"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"),
}
FILES = {
    ("medqa", "dev"): "data/dev-00000-of-00001.parquet",
    ("medqa", "train"): "data/train-00000-of-00001.parquet",
    ("medmcqa", "validation"): "data/validation-00000-of-00001.parquet",
    ("medsafety", "train"): "data/train-00000-of-00001.parquet",
    ("medquad", "train"): "data/train-00000-of-00001-e36383d177026d53.parquet",
    ("arc", "train"): "ARC-Challenge/train-00000-of-00001.parquet",
}


def parquet(src, split):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, rev = DATA[src]
    p = hf_hub_download(
        repo, FILES[(src, split)], repo_type="dataset", revision=rev
    )  # prefetched (--prefetch)
    return pd.read_parquet(p).to_dict("records")


def exposed_set():
    hashes, ids, n = set(), set(), 0
    for p in EXPOSURES:
        e = B.read_exposed(p)
        hashes |= e["hashes"]
        ids |= e["ids"]
        n += e["n"]
    return {"hashes": hashes, "ids": ids, "n": n}


def prompt_item(src, i, text):
    return {"id": f"{src}-{i}", "source": src, "prompt": text, "hash": W.text_hash(text)}


def designate(ctx):
    got = ctx.read("items")
    if got:
        return got
    ex = exposed_set()
    if ex["n"] != 1184:
        raise SystemExit(f"exposure lists hold {ex['n']} items, expected 784 + 400")
    out = {}
    out.update(
        G1.designate(
            [W.medqa_item(r) for r in parquet("medqa", "dev")],
            "medqa",
            "dev",
            [("medqa", G1.N["medqa"])],
            ex,
        )
    )
    mm = [x for x in (W.medmcqa_item(r) for r in parquet("medmcqa", "validation")) if x is not None]
    out.update(G1.designate(mm, "medmcqa", "validation", [("medmcqa", G1.N["medmcqa"])], ex))
    tr = [W.medqa_item(r) for r in parquet("medqa", "train")]
    out.update(
        G1.designate(
            tr, "medqa", "train", [("h1_pool", G1.N["h1_pool"]), ("syc_pool", G1.N["syc_pool"])], ex
        )
    )
    harm = [
        prompt_item("medsafety", i, r["harmful_medical_request"])
        for i, r in enumerate(parquet("medsafety", "train"))
    ]
    out.update(G1.designate(harm, "medsafety", "train", [("harm", G1.N["harm"])], ex))
    ben = [
        prompt_item("medquad", r["question_id"], r["question"])
        for r in parquet("medquad", "train")
        if r["question"]
    ]
    out.update(G1.designate(ben, "medquad", "train", [("benign", G1.N["benign"])], ex))
    arc = [x for x in (G1.arc_item(r) for r in parquet("arc", "train")) if x is not None]
    out.update(G1.designate(arc, "arc_challenge", "train", [("general", G1.N["general"])], ex))
    split_of = {
        "medqa": "dev",
        "medmcqa": "validation",
        "h1_pool": "train",
        "syc_pool": "train",
        "harm": "train",
        "benign": "train",
        "general": "train",
    }
    exposure = [
        {
            "source": it["source"],
            "split": split_of[role],
            "id": it["id"],
            "hash": it["hash"],
            "role": f"gate1_{role}",
        }
        for role, its in out.items()
        for it in its
    ]
    hs = [e["hash"] for e in exposure]
    assert len(hs) == len(set(hs)), "an item was designated twice"
    assert not set(hs) & ex["hashes"], "a designated item was already exposed"
    ctx.write(
        "exposure",
        {
            "salt": G1.SALT,
            "status": "EXPOSED (study M Gate 1); never sealed",
            "datasets": DATA,
            "items": exposure,
            "disjoint_from": [str(p.relative_to(ROOT)) for p in EXPOSURES],
            "n_previously_exposed": ex["n"],
        },
    )
    ctx.write("items", out)
    ctx.log("designated", **{k: len(v) for k, v in out.items()})
    return out


# ------------------------------------------------------------------ prompt sets


def sets(items):
    perm = {k: [G1.permuted(it) for it in items[k]] for k in ("medqa", "medmcqa")}
    return {
        "mq": (items["medqa"], [G1.neutral_prompt(it) for it in items["medqa"]], G1.BUDGET),
        "mm": (items["medmcqa"], [G1.neutral_prompt(it) for it in items["medmcqa"]], G1.BUDGET),
        "mq_syc": (items["medqa"], [G1.suggested_prompt(it) for it in items["medqa"]], G1.BUDGET),
        "mq_perm": (perm["medqa"], [G1.neutral_prompt(it) for it in perm["medqa"]], G1.BUDGET),
        "mm_perm": (perm["medmcqa"], [G1.neutral_prompt(it) for it in perm["medmcqa"]], G1.BUDGET),
        "harm": (items["harm"], [it["prompt"] for it in items["harm"]], G1.GUARD_TOKENS),
        "benign": (items["benign"], [it["prompt"] for it in items["benign"]], G1.GUARD_TOKENS),
        "general": (
            items["general"],
            [G1.neutral_prompt(it) for it in items["general"]],
            G1.BUDGET,
        ),
    }


LETTER_IDS = None


@torch.no_grad()
def letter_conf(ctx, its, gens):
    """Probability of the extracted letter among the option letters at its position (teacher forcing on the
    generated text; no-space and space-prefixed letter tokens pooled). None where unparsed or truncated."""
    global LETTER_IDS
    if LETTER_IDS is None:
        LETTER_IDS = [
            (
                ctx.tok.convert_tokens_to_ids(L),
                ctx.tok.encode(" " + L, add_special_tokens=False)[-1],
            )
            for L in "ABCD"
        ]
    jobs = []
    for i, (it, g) in enumerate(zip(its, gens, strict=True)):
        if g["hit_max"]:
            continue
        pos = G1.answer_position(g["text"], len(it["options"]))
        if pos is None:
            continue
        pre = g["text"][:pos].rstrip(" ")
        jobs.append(
            (
                i,
                g["prompt_ids"] + ctx.tok.encode(pre, add_special_tokens=False),
                g["text"][pos].upper(),
            )
        )
    out = [None] * len(gens)
    jobs.sort(key=lambda j: len(j[1]))
    for b0 in range(0, len(jobs), 8):
        batch = jobs[b0 : b0 + 8]
        T = max(len(j[1]) for j in batch)
        ids = torch.full((len(batch), T), ctx.pad, dtype=torch.long)
        att = torch.zeros((len(batch), T), dtype=torch.long)
        for r, (_i, s, _l) in enumerate(batch):
            ids[r, : len(s)] = torch.tensor(s)
            att[r, : len(s)] = 1
        h = ctx.model.model(
            input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False
        ).last_hidden_state
        for r, (i, s, L) in enumerate(batch):
            p = torch.softmax(ctx.model.lm_head(h[r, len(s) - 1]).float(), -1)
            n = len(its[i]["options"])
            mass = [float(p[a] + p[b]) for a, b in LETTER_IDS[:n]]
            out[i] = mass["ABCD".index(L)] / sum(mass)
        del h
    return out


# Amendment 1 (plan §6): paired sets are generated in batches ordered by the base's own pass-1 output length on the
# same (original) item, so batches hold similar lengths and every state shares one batch composition.
ORDERED = {"mq": "mq", "mq_syc": "mq", "mq_perm": "mq", "mm": "mm", "mm_perm": "mm"}


@torch.no_grad()
def generate_ordered(ctx, users, max_new, order):
    enc = [G0.enc_prompt(ctx, u) for u in users]
    out = [None] * len(enc)
    eot = set(ctx.stop)
    for b0 in range(0, len(order), G0.GEN_BATCH):
        idx = order[b0 : b0 + G0.GEN_BATCH]
        L = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[ctx.pad] * (L - len(enc[i])) + enc[i] for i in idx], device="cuda")
        att = torch.tensor(
            [[0] * (L - len(enc[i])) + [1] * len(enc[i]) for i in idx], device="cuda"
        )
        rows = G0._gen_rows(ctx, ids, att, max_new)
        for i, row in zip(idx, rows, strict=True):
            cut = next((k for k, t in enumerate(row) if t in eot), None)
            new = row if cut is None else row[:cut]
            out[i] = {
                "ids": new,
                "text": ctx.tok.decode(new, skip_special_tokens=True),
                "hit_max": cut is None,
                "prompt_ids": enc[i],
            }
        del ids, att
    return out


def gen_cached_ordered(ctx, name, users, max_new, order):
    p = ctx.out / "gen" / f"{name}.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines() if x]
        if len(rows) == len(users):
            return rows
    t0 = time.time()
    rows = generate_ordered(ctx, users, max_new, order)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    toks = sum(len(r["ids"]) for r in rows)
    dt = time.time() - t0
    ctx.log(
        "generated",
        name=name,
        n=len(rows),
        tokens=toks,
        seconds=dt,
        tok_per_s=toks / dt,
        ordered=True,
    )
    ctx.charge()
    return rows


def length_order(base_rec, key):
    L = base_rec["sets"][key]["length"]
    return sorted(range(len(L)), key=lambda i: (L[i], i))


def run_set(ctx, tag, name, S, orders=None):
    its, prompts, budget = S[name]
    t0 = time.time()
    if orders is not None and name in ORDERED:
        gens = gen_cached_ordered(ctx, f"{tag}__{name}", prompts, budget, orders[ORDERED[name]])
    else:
        gens = G0.gen_cached(ctx, f"{tag}__{name}", prompts, budget)
    sec = time.time() - t0
    rec = {"seconds": sec, "tokens": int(sum(len(g["ids"]) for g in gens)), "n": len(gens)}
    if name in ("harm", "benign"):
        rec["refused"] = [G1.is_refusal(g["text"]) for g in gens]
        rec["truncated"] = [g["hit_max"] for g in gens]
        return rec
    sc = [B.score(it, g["text"], g["hit_max"]) for it, g in zip(its, gens, strict=True)]
    rec.update(
        correct=[s["correct"] for s in sc],
        truncated=[s["truncated"] for s in sc],
        letter=[s["letter"] for s in sc],
        rule=[s["rule"] for s in sc],
        length=[len(g["ids"]) for g in gens],
    )
    if name == "mq_syc":
        rec["adopted"] = [G1.adopted(it, s["letter"]) for it, s in zip(its, sc, strict=True)]
    if name in ("mq", "mm", "general"):
        t1 = time.time()
        rec["conf"] = letter_conf(ctx, its, gens)
        rec["conf_seconds"] = time.time() - t1
    return rec


def state(ctx, tag, names, S, apply=None, orders=None):
    """Measure ``names`` at one state (base or an applied edit); cached per (tag, name) in OUT/state_<tag>.json."""
    rec = ctx.read(f"state_{tag}") or {"tag": tag, "sets": {}, "timing": {}}
    todo = [n for n in names if n not in rec["sets"]]
    if not todo:
        return rec
    ctx.check_cap(tag)
    undo = None
    if apply is not None:
        t0 = time.time()
        edit, undo = apply()
        rec["timing"]["apply_seconds"] = rec["timing"].get("apply_seconds", 0.0) + time.time() - t0
        rec["edit"] = edit
    for n in todo:
        rec["sets"][n] = run_set(ctx, tag, n, S, orders)
        rec["timing"][f"{n}_seconds"] = rec["sets"][n]["seconds"]
        ctx.write(f"state_{tag}", rec)
    if undo is not None:
        t0 = time.time()
        undo()
        rec["timing"]["restore_seconds"] = (
            rec["timing"].get("restore_seconds", 0.0) + time.time() - t0
        )
    ctx.write(f"state_{tag}", rec)
    ctx.log(
        "state",
        tag=tag,
        sets=todo,
        **{
            f"acc_{n}": float(np.mean(rec["sets"][n]["correct"]))
            for n in todo
            if "correct" in rec["sets"][n]
        },
    )
    return rec


# ------------------------------------------------------------------ LoRA retraining (Gate 0 recipe, frozen harness)


class Row:
    def __init__(self, ids, score_from):
        self.ids, self.score_from = ids, score_from


def seed_for(label):
    return int(hashlib.sha256(f"{G1.SALT}:{label}".encode()).hexdigest()[:8], 16)


def loras(ctx, items):
    d = ctx.out / "lora"
    d.mkdir(exist_ok=True)
    factors = {}
    eot = CH.end_token_id(ctx.tok)
    pools = {}
    for pool in ("h1_pool", "syc_pool"):
        its = items[pool]
        pools[pool] = G0.gen_cached(
            ctx, f"pool__{pool}", [G1.neutral_prompt(it) for it in its], G1.BUDGET
        )
    for fam, pool in (("h1_lora", "h1_pool"), ("syc_lora", "syc_pool")):
        f = d / f"{fam}.pt"
        if f.exists() and ctx.read(f"lora_{fam}"):
            factors[fam] = torch.load(f)
            continue
        ctx.check_cap(f"lora {fam}")
        rows = []
        for it, g in zip(items[pool], pools[pool], strict=True):
            if not B.score(it, g["text"], g["hit_max"])["correct"]:
                continue
            p = (
                g["prompt_ids"]
                if fam == "h1_lora"
                else ctx.tok.encode(
                    CH.chat_prompt(ctx.tok, G1.suggested_prompt(it)), add_special_tokens=False
                )
            )
            rows.append(Row(p + g["ids"] + [eot], len(p)))
        wh = G0.writers_hash(ctx)
        ad = LR.LoRA(
            ctx.model,
            LR.writer_targets(ctx.rm),
            rank=G1.LORA["rank"],
            alpha=G1.LORA["alpha"],
            seed=seed_for(f"lora:{fam}"),
        )
        torch.cuda.empty_cache()
        log = LM.train(  # micro-batched, same recipe (incident 20 Nov: 8-row batches OOM at 16 GB)
            ctx.model,
            ad,
            rows,
            pad_id=ctx.pad,
            lr=G1.LORA["lr"],
            epochs=G1.LORA["epochs"],
            batch_rows=G1.LORA["batch_rows"],
            seed=seed_for(f"lora-order:{fam}"),
            device="cuda",
            max_seconds=2400.0,
            warmup=G1.LORA["warmup"],
            checkpointing=True,
            ce_chunk=1024,
            log=lambda e, fam=fam, **kw: ctx.log(e, family=fam, **kw),
            micro_rows=1,
        )
        factors[fam] = ad.factors()
        torch.save(factors[fam], f)
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
                "rank",
                "alpha",
                "seed",
                "micro_rows",
            )
        }
        meta.update(
            loss_first10=float(np.mean(log["loss"][:10])) if log["loss"] else None,
            loss_last10=float(np.mean(log["loss"][-10:])) if log["loss"] else None,
            pool_accuracy=float(
                np.mean(
                    [
                        B.score(it, g["text"], g["hit_max"])["correct"]
                        for it, g in zip(items[pool], pools[pool], strict=True)
                    ]
                )
            ),
            writers_unchanged=G0.writers_hash(ctx) == wh,
        )
        del ad
        torch.cuda.empty_cache()
        ctx.write(f"lora_{fam}", meta)
        ctx.log("lora trained", family=fam, **meta)
    return factors


# ------------------------------------------------------------------ analysis helpers


def pooled(rec, names):
    out = {"correct": [], "truncated": []}
    for n in names:
        out["correct"] += [int(x) for x in rec["sets"][n]["correct"]]
        out["truncated"] += [int(x) for x in rec["sets"][n]["truncated"]]
    return out


def edit_stats(base, rec, names):
    b, e = pooled(base, names), pooled(rec, names)
    return {
        "at_budget": G1.paired(e["correct"], b["correct"]),
        "untruncated_in_both": G1.untruncated_in_both(b, e),
    }


def guard_stats(base, base_s, rec):
    out = {}
    hb, he = base["sets"]["harm"]["refused"], rec["sets"]["harm"]["refused"]
    out["harm"] = G1.guard_verdict("harm", G1.paired([1 - x for x in he], [1 - x for x in hb]))
    bb, be = base["sets"]["benign"]["refused"], rec["sets"]["benign"]["refused"]
    out["benign"] = G1.guard_verdict(
        "benign", G1.paired([int(x) for x in be], [int(x) for x in bb])
    )
    sq_b, sq_e = [], []
    for n in [m for m in ("mq", "mm") if m in rec["sets"]]:
        sb, se = base_s["sets"][n], rec["sets"][n]
        for cb, ce, yb, ye in zip(
            sb["conf"], se["conf"], sb["correct"], se["correct"], strict=True
        ):
            if cb is not None and ce is not None:
                sq_b.append((cb - yb) ** 2)
                sq_e.append((ce - ye) ** 2)
    out["brier"] = G1.guard_verdict("brier", G1.paired(sq_e, sq_b))
    out["general"] = G1.guard_verdict(
        "general",
        G1.paired(
            [int(x) for x in rec["sets"]["general"]["correct"]],
            [int(x) for x in base["sets"]["general"]["correct"]],
        ),
    )
    out["fail_any"] = any(v["fail"] for k, v in out.items() if k != "fail_any")
    return out


def audit_items(ctx, items, base):
    d = ctx.out / "audit"
    if (d / "blind_items.json").exists():
        return
    d.mkdir(exist_ok=True)
    rows, rule = [], {}
    for name, n in (("harm", 50), ("benign", 100)):
        gens = [
            json.loads(x)
            for x in (ctx.out / "gen" / f"base__{name}.jsonl").read_text().splitlines()
            if x
        ]
        for k in range(n):
            key = f"{name}-{k:03d}"
            rows.append(
                {
                    "key": key,
                    "prompt": items[name][k]["prompt"],
                    "response": G1.strip_thought(gens[k]["text"]),
                }
            )
            rule[key] = "refuse" if base["sets"][name]["refused"][k] else "comply"
    random.Random(20261120).shuffle(rows)
    (d / "blind_items.json").write_text(json.dumps(rows, indent=1))
    (d / "rule_labels.json").write_text(json.dumps(rule, indent=1))
    ctx.log("audit items", n=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = G0.Ctx(out, False)
    git = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    ctx.write(
        "manifest",
        {
            "study": "M",
            "gate": 1,
            "plan": "docs/reviews/wmm/wmm-gate1-plan-2026-11-20.md",
            "target": G3.TARGET,
            "revisions": G3.TARGETS,
            "git_sha": git,
            "grid": G1.GRID,
            "order": G1.ORDER,
            "n": G1.N,
            "budget": G1.BUDGET,
            "guard_tokens": G1.GUARD_TOKENS,
            "lora": G1.LORA,
            "vram_cap_gb": G0.VRAM_CAP_GB,
            "gen_batch": G0.GEN_BATCH,
            "gpu_h_cap": G0.GPU_H_CAP,
            "datasets": DATA,
        },
    )
    items = designate(ctx)
    S = sets(items)
    G0.load(ctx)
    c0 = ctx.read("canary_start")
    if not c0:
        c0 = G0.canary(ctx)
        c0.pop("pairs")
        ctx.write("canary_start", c0)
        ctx.write("state_hash_start", {"hash": G0.full_hash(ctx)})
    # base pass 1 (prompt-length batches): primaries and the unpaired-order sets (guards, general)
    base = state(ctx, "base", ["mq", "mm", "harm", "benign", "general"], S)
    orders = {k: length_order(base, k) for k in ("mq", "mm")}
    # paired base (Amendment 1): length-ordered batches; edits share this composition
    base_s = state(ctx, "base_s", ["mq", "mm", "mq_syc", "mq_perm", "mm_perm"], S, orders=orders)
    audit_items(ctx, items, base)
    factors = loras(ctx, items)
    # screening: the 5 central doses first, then qualifiers/guards, then the other 6 doses, then qualifiers again
    screen = ctx.read("screen") or {}
    done = {"perm": ctx.read("permuted") or {}, "guards": ctx.read("guards") or {}}

    def screen_doses(pairs, prim):
        for fam, dose in pairs:
            tag = f"{fam}@{dose:g}"
            names = list(prim) + (["mq_syc"] if fam == "syc_lora" else [])
            rec = state(
                ctx,
                tag,
                names,
                S,
                apply=lambda fam=fam, dose=dose: G0.apply_state(ctx, fam, dose, factors),
                orders=orders,
            )
            screen[tag] = edit_stats(base_s, rec, list(prim))
            screen[tag]["primaries"] = list(prim)
            ctx.write("screen", screen)

    def apply_tag(tag):
        fam, dose = tag.split("@")
        return lambda: G0.apply_state(ctx, fam, float(dose), factors)

    def qualify_and_guard():
        qual = sorted(
            (t for t, st in screen.items() if G1.positive(st["untruncated_in_both"])),
            key=lambda t: -screen[t]["untruncated_in_both"]["lb95"],
        )[: G1.MAX_QUALIFIERS]
        for tag in qual:
            if tag in done["perm"]:
                continue
            prim = screen[tag].get("primaries", ["mq", "mm"])
            rec = state(
                ctx, tag, [f"{n}_perm" for n in prim], S, apply=apply_tag(tag), orders=orders
            )
            done["perm"][tag] = edit_stats(
                {"sets": {n: base_s["sets"][f"{n}_perm"] for n in prim}},
                {"sets": {n: rec["sets"][f"{n}_perm"] for n in prim}},
                prim,
            )
            ctx.write("permuted", done["perm"])
        both = [t for t in qual if G1.positive(done["perm"][t]["untruncated_in_both"])]
        syc = ctx.read("state_syc_lora@1")
        drop = None
        if syc and "mq_syc" in syc["sets"]:
            drop = float(
                np.mean(base_s["sets"]["mq_syc"]["adopted"])
                - np.mean(syc["sets"]["mq_syc"]["adopted"])
            )
        tags = list(both) + (["syc_lora@1"] if drop is not None and drop >= G1.SYC_MIN_DROP else [])
        for tag in tags:
            if tag in done["guards"]:
                continue
            rec = state(ctx, tag, ["harm", "benign", "general"], S, apply=apply_tag(tag))
            done["guards"][tag] = guard_stats(base, base_s, rec)
            ctx.write("guards", done["guards"])
        return qual, both, drop

    stopped = None
    try:
        screen_doses(G1.ORDER[: G1.CENTRAL], ["mq", "mm"])
        qualify_and_guard()
        screen_doses(G1.ORDER[G1.CENTRAL :], ["mq"])  # Amendment 2: MedQA only
        qual, both, syc_drop = qualify_and_guard()
    except SystemExit as e:
        stopped = str(e)
        ctx.log("cap", msg=stopped)
        qual, both, syc_drop = None, None, None
    perm, guards = done["perm"], done["guards"]
    c1 = G0.canary(ctx)
    c1.pop("pairs")
    final = {
        "canary_end": c1,
        "canary_match": c1["ids_sha256"] == c0["ids_sha256"]
        and c1["logits_sha256"] == c0["logits_sha256"],
        "state_hash_end": G0.full_hash(ctx),
    }
    final["state_hash_matches_start"] = (
        final["state_hash_end"] == ctx.read("state_hash_start")["hash"]
    )
    ctx.write("final", final)
    ctx.write(
        "summary",
        {
            "screen": screen,
            "qualifiers": qual,
            "permuted": perm,
            "positive_on_both": both,
            "syc_drop_lambda1": syc_drop,
            "guards": guards,
            "stopped_by_cap": stopped,
        },
    )
    ctx.charge()
    ctx.log(
        "done",
        gpu_hours=ctx.gpu_seconds() / 3600,
        qualifiers=qual,
        positive_on_both=both,
        syc_drop=syc_drop,
        canary_match=final["canary_match"],
    )


if __name__ == "__main__":
    main()
