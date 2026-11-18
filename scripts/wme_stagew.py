#!/usr/bin/env python3
"""WM-E stage W on a Lambda H100 (docs/council/2026-11-18/RULING-wme-stagew.md). EXPLORATORY.

Two processes, run in order by ``wme_w_remote.sh``:

* ``train``: Hugging Face copy of theta0 only. Four fresh-seed H1-type LoRA runs (rank 8, alpha 16, 2 epochs, batch 8,
  warm-up 10) on the shards of the pilot-3 teacher rows, factors saved at the 8 path checkpoints; then the member
  features: per-layer squared norms and cross inner products of the saved deltas (R0) and per-layer projections of
  the exact theta0 gradients of the stop, target and margin objectives on the 64 held-out feature items (R1).
* ``eval``: vLLM only (prefix caching off, seed 0, utilisation 0.88, max_num_seqs 256, max_model_len 8192, greedy,
  2,048 new tokens). Canary (64 exposed wme-p3h holdout items) -> base -> base re-run (reversed order) -> the frozen
  member queue (S, L, M, X) until the deadline -> canary repeat -> the P-9B-eval timing pilot (exposed wme-p3h
  construction/holdout items; seconds only) until its own deadline. Every member is written into the residual
  writers in place and restored bitwise (checksum) afterwards.

Every completed unit appends a line to ``<out>/progress.jsonl`` (the AWS heartbeat reads only that file).

    wme_stagew.py train --out OUT [--smoke]
    wme_stagew.py eval --out OUT [--smoke] [--timing]
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

from wsp.wm import hf
from wsp.wm import lora as LR
from wsp.wm import p3gen as G
from wsp.wm import p3panel as PP
from wsp.wm import stagew as W
from wsp.wm import svf as V

PANEL_W = ROOT / "data/wme/panels/wme-w"
PANEL_P3H = ROOT / "data/wme/panels/wme-p3h"
TEACHER = ROOT / "data/wme/stagew/teacher.jsonl"
REPO = "Qwen/Qwen3.5-9B"
LORA = {"rank": 8, "alpha": 16, "epochs": 2, "batch_rows": 8, "warmup": 10}
ENGINE = {"gpu_memory_utilization": 0.88, "max_num_seqs": 256, "max_model_len": 8192}
CANARY_N = 64
TIMING_SIZES = (250, 1000, 2512)
SMOKE = False


class Row:
    def __init__(self, ids, score_from):
        self.ids, self.score_from = ids, score_from


class Ctx:
    def __init__(self, out: Path, phase: str):
        self.out, self.phase, self.t0 = out, phase, time.time()

    def log(self, event, **kw):
        rec = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "t": round(time.time() - self.t0, 1),
            "phase": self.phase,
            "event": event,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available()
            else 0,
            **kw,
        }
        line = json.dumps(rec, default=str)
        with open(self.out / "ledger.jsonl", "a") as f:
            f.write(line + "\n")
        with open(self.out / "progress.jsonl", "a") as f:
            f.write(json.dumps({"utc": rec["utc"], "phase": self.phase, "event": event}) + "\n")
        print(line, flush=True)


def jread(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def jwrite(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    os.replace(tmp, p)


def enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


def im_end(tok):
    return tok.convert_tokens_to_ids("<|im_end|>")


def repo_path():
    return hf.snapshot_dir(REPO, hf.TARGETS[REPO])


# ------------------------------------------------------------------ teacher rows


def teacher_rows(tok):
    """Pilot-3 thinking-mode final answers on wme-p3h construction items (construction data, never evaluated on)."""
    items = {
        p: {it["id"]: it for it in v} for p, v in PP.read_split(PANEL_P3H, "construction").items()
    }
    rows = []
    for line in TEACHER.read_text().splitlines():
        r = json.loads(line)
        if r["panel"] not in W.PANELS or r["id"] not in items[r["panel"]]:
            continue
        ans = (r.get("answer") or "").strip()
        r = dict(r, answer=ans, n_tokens=len(enc(tok, ans)) if ans else 0)
        r["item"] = items[r["panel"]][r["id"]]
        rows.append(r)
    return rows


def prompt_ids(tok, panel, item):
    return enc(tok, G.chat(tok, G.user_prompt(panel, item)))


# ------------------------------------------------------------------ train phase


def writer_names(model):
    return sorted(n for n, _ in model.named_parameters() if V.writer_key(n) is not None)


def train_phase(out: Path):
    ctx = Ctx(out, "train")
    (out / "STATUS_train").write_text("RUNNING")
    model, tok, _cfg, base_hash, _ = hf.load_target(REPO, "cuda")
    hf.use_reference_kernels()
    names = writer_names(model)
    n_layers = 1 + max(V.writer_key(n)[0] for n in names)
    set_layers(n_layers)
    ctx.log("model loaded", writers=len(names), n_layers=n_layers, base_hash=base_hash)
    rows = teacher_rows(tok)
    sp = W.split_teacher(rows, n_feature=4 if SMOKE else W.N_FEATURE)
    if SMOKE:
        sp = {k: v[: (8 if k == "feature" else 24)] for k, v in sp.items()}
    jwrite(
        out / "split.json",
        {k: [(r["panel"], r["id"]) for r in v] for k, v in sp.items()},
    )
    ctx.log("teacher split", **{k: len(v) for k, v in sp.items()})
    e = im_end(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    metas = {}
    for run in W.RUN_ORDER:
        d = out / "lora" / run
        meta = jread(d / "meta.json")
        if meta is None:
            shard, lr = W.RUNS[run]
            rws = []
            for r in sp[shard]:
                a = prompt_ids(tok, r["panel"], r["item"])
                rws.append(Row(a + enc(tok, r["answer"]) + [e], len(a)))
            bs = 1 if SMOKE else LORA["batch_rows"]
            epochs = 1 if SMOKE else LORA["epochs"]
            n_steps = epochs * math.ceil(len(rws) / bs)
            ck = set(W.ckpt_steps(n_steps))
            d.mkdir(parents=True, exist_ok=True)

            def on_step(step, adapter, d=d, ck=ck, run=run):
                if step in ck:
                    torch.save(
                        {
                            k: (b.float(), a.float(), s)
                            for k, (b, a, s) in adapter.factors().items()
                        },
                        d / f"step{step:05d}.pt",
                    )
                if step % 10 == 0 or step in ck:
                    ctx.log("lora progress", run=run, step=step)

            ad = LR.LoRA(
                model, names, rank=LORA["rank"], alpha=LORA["alpha"], seed=W.seed_for(f"lora:{run}")
            )
            log = LR.train(
                model,
                ad,
                rws,
                pad_id=pad,
                lr=lr,
                epochs=epochs,
                batch_rows=bs,
                seed=W.seed_for(f"lora-order:{run}"),
                device="cuda",
                max_seconds=1e9,
                warmup=LORA["warmup"],
                checkpointing=True,
                on_step=on_step,
                ce_chunk=1024,
            )
            if not all(math.isfinite(x) for x in log["loss"]):
                raise RuntimeError(f"non-finite training loss in {run}")
            if log["steps"] != n_steps:
                raise RuntimeError(f"{run}: {log['steps']} steps, expected {n_steps}")
            meta = {
                k: log[k]
                for k in (
                    "steps",
                    "seconds",
                    "n_rows",
                    "epochs",
                    "lr",
                    "batch_rows",
                    "rank",
                    "alpha",
                    "seed",
                )
            }
            meta.update(
                shard=shard,
                ckpts=sorted(ck),
                loss_first20=float(np.mean(log["loss"][:20])),
                loss_last20=float(np.mean(log["loss"][-20:])),
                loss_curve=[
                    float(np.mean(log["loss"][i : i + 4])) for i in range(0, len(log["loss"]), 4)
                ],
            )
            jwrite(d / "meta.json", meta)
            del ad
            torch.cuda.empty_cache()
            ctx.log(
                "lora trained",
                run=run,
                **{k: meta[k] for k in ("steps", "seconds", "loss_first20", "loss_last20")},
            )
        metas[run] = meta
    features(ctx, model, tok, names, sp["feature"], metas)
    jwrite(
        out / "train_meta.json",
        {
            "final_steps": {r: metas[r]["steps"] for r in W.RUN_ORDER},
            "ckpts": {r: metas[r]["ckpts"] for r in W.RUN_ORDER},
            "base_hash": base_hash,
            "n_layers": n_layers,
            "seconds": time.time() - ctx.t0,
            "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
        },
    )
    ctx.log("train done", seconds=round(time.time() - ctx.t0, 1))
    (out / "STATUS_train").write_text("DONE")


def set_layers(n_layers):
    W.configure_layers(n_layers)


def load_factors(out, run, step, device="cuda"):
    fac = torch.load(out / "lora" / run / f"step{step:05d}.pt", weights_only=False)
    return {V.writer_key(n): (b.to(device), a.to(device), float(s)) for n, (b, a, s) in fac.items()}


def features(ctx, model, tok, names, feat_rows, metas):
    """R0 per-layer norms / cross products and R1 per-layer gradient projections for every saved checkpoint."""
    out = ctx.out
    torch.cuda.reset_peak_memory_stats()
    p = out / "features.npz"
    if p.exists():
        return
    ck = [(r, s) for r in W.RUN_ORDER for s in metas[r]["ckpts"]]
    facs = {c: load_factors(out, *c) for c in ck}
    L = W.N_LAYERS
    sq = np.zeros((len(ck), L))
    for i, c in enumerate(ck):
        for (layer, _kind), (b, a, s) in facs[c].items():
            sq[i, layer] += float(s * s * ((b.T @ b) * (a @ a.T)).sum())
    cross_pairs = [((a, metas[a]["steps"]), (b, metas[b]["steps"])) for a, b in W.MERGES]
    cross = np.zeros((len(cross_pairs), L))
    for i, (x, y) in enumerate(cross_pairs):
        for wk, (b1, a1, s1) in facs[x].items():
            b2, a2, s2 = facs[y][wk]
            cross[i, wk[0]] += float(s1 * s2 * ((b1.T @ b2) * (a1 @ a2.T)).sum())
    ctx.log("R0 norms", n_ckpts=len(ck))
    # stacked factors per writer for the projections
    params = dict(model.named_parameters())
    stack = {}
    for n in names:
        wk = V.writer_key(n)
        Bt = torch.cat([facs[c][wk][0] for c in ck], 1).T.contiguous()  # [K * r, out]
        A = torch.stack([facs[c][wk][1] for c in ck])  # [K, r, in]
        S = torch.tensor([facs[c][wk][2] for c in ck], device="cuda")
        stack[n] = (wk[0], Bt, A, S)
    for n in names:
        params[n].requires_grad_(True)
    e = im_end(tok)
    letters = [tok.convert_tokens_to_ids(x) for x in G.LETTERS]
    nI = len(feat_rows)
    j1 = np.zeros((3, nI, len(ck), L))
    # activation checkpointing for the gradient passes (Qwen3.5 has no dropout, so train mode changes nothing else)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    def project(o, i):
        for n in names:
            g = params[n].grad
            if g is None:
                continue
            layer, Bt, A, S = stack[n]
            BtG = (Bt @ g.float()).view(
                A.shape
            )  # one matmul; no per-checkpoint copy of the gradient
            j1[o, i, :, layer] += ((BtG * A).sum((1, 2)) * S).double().cpu().numpy()
            del BtG
            params[n].grad = None

    for i, r in enumerate(feat_rows):
        a = prompt_ids(tok, r["panel"], r["item"])
        t = enc(tok, r["answer"])
        ids = torch.tensor([a + t + [e]], device="cuda")
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        logits = model.lm_head(h[len(a) - 1 : len(a) + len(t)])  # only the scored positions
        lp = torch.log_softmax(logits.float(), -1)
        tg = torch.tensor(t + [e], device="cuda")
        tok_lp = lp.gather(1, tg[:, None])[:, 0]
        f_stop, f_target = tok_lp[-1], tok_lp[:-1].mean()
        f_stop.backward(retain_graph=True)
        project(0, i)
        f_target.backward()
        project(1, i)
        del logits, lp, h
        ids2 = torch.tensor([a + enc(tok, "The answer is (")], device="cuda")
        last = model.lm_head(
            model.model(input_ids=ids2, use_cache=False).last_hidden_state[0, -1:]
        )[0].float()
        k = len(r["item"]["options"])
        gold = r["item"]["answer_index"]
        lv = last[torch.tensor(letters[:k], device="cuda")]
        others = torch.cat([lv[:gold], lv[gold + 1 :]])
        f_margin = lv[gold] - torch.logsumexp(others, 0)
        f_margin.backward()
        project(2, i)
        torch.cuda.empty_cache()
        if (i + 1) % 8 == 0:
            ctx.log("R1 projections", items=i + 1)
    for n in names:
        params[n].requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.eval()
    np.savez(
        p,
        ckpts=np.array([f"{r}:{s}" for r, s in ck]),
        sq=sq,
        cross_pairs=np.array([f"{x[0]}:{x[1]}|{y[0]}:{y[1]}" for x, y in cross_pairs]),
        cross=cross,
        j1=j1,
        feature_items=np.array([f"{r['panel']}|{r['id']}" for r in feat_rows]),
    )
    ctx.log("features saved", items=nI, ckpts=len(ck))


# ------------------------------------------------------------------ eval phase


class Engine:
    def __init__(self, ctx):
        from transformers import AutoTokenizer
        from vllm import LLM

        path = repo_path()
        kw = dict(ENGINE)
        if SMOKE:
            kw.update(gpu_memory_utilization=0.2, max_num_seqs=32, max_model_len=4096)
        self.llm = LLM(
            model=str(path),
            dtype="bfloat16",
            enable_prefix_caching=False,
            seed=0,
            limit_mm_per_prompt={"image": 0, "video": 0},
            **kw,
        )
        self.engine_kw = kw
        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        vp = self.llm.apply_model(lambda m: dict(m.named_parameters()))
        vp = vp[0] if isinstance(vp, list) else vp
        self.vmap = V.writer_map(vp)
        self.ed = V.LiveEditor(vp)
        self.names = sorted(self.vmap.values())
        self.base_checksum = self.ed.checksum(self.names)
        set_layers(1 + max(k[0] for k in self.vmap))
        import vllm

        ctx.log(
            "engine", vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda, **kw
        )

    def gen(self, prompts, max_tokens):
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_tokens))
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        stop = {self.tok.eos_token_id, im_end(self.tok)}
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            while ids and ids[-1] in stop:
                ids.pop()
            res.append({"text": c.text, "ids": ids, "hit_max": c.finish_reason == "length"})
        return res

    def apply(self, out, spec):
        if spec.get("run") is None:
            return
        terms = W._terms(spec)
        facs = [(load_factors(out, r, s), c) for (r, s), c in terms]
        limit = W.FINE_GROUP * spec["groups"]
        deltas = []
        for wk, vname in self.vmap.items():
            if wk[0] >= limit:
                continue

            def fn(wk=wk):
                d = None
                for fac, c in facs:
                    b, a, s = fac[wk]
                    x = (c * s) * (b @ a)
                    d = x if d is None else d + x
                return d

            deltas.append((vname, fn))
        self.ed.apply(deltas)

    def restore(self):
        self.ed.restore()
        if self.ed.checksum(self.names) != self.base_checksum:
            raise RuntimeError("vLLM writers not restored to theta0")


def score(panel, items, outs):
    recs = []
    for it, o in zip(items, outs):
        v = G.verify(panel, it, o["text"], o["hit_max"])
        flex = None if o["hit_max"] else G.flexible_mc_letter(o["text"][-400:])
        recs.append(
            {
                "id": it["id"],
                "correct": bool(v["correct"]),
                "trunc": bool(o["hit_max"]),
                "len": len(o["ids"]),
                "parsed": v["parsed"],
                "flex_correct": flex == G.LETTERS[it["answer_index"]],
                "tail": o["text"][-400:],
            }
        )
    return recs


def eval_phase(out: Path, timing: bool):
    ctx = Ctx(out, "eval")
    (out / "STATUS_eval").write_text("RUNNING")
    meta = jread(out / "train_meta.json")
    eng = Engine(ctx)
    tok = eng.tok
    if SMOKE:
        pan = {p: v[:12] for p, v in PP.read_split(PANEL_P3H, "holdout").items() if p in W.PANELS}
    else:
        pan = {p: v[: W.N_DEV] for p, v in PP.read_split(PANEL_W, "dev").items()}
    canary_items = PP.read_split(PANEL_P3H, "holdout")["mmlupro_gen"][: (8 if SMOKE else CANARY_N)]
    prompts = {p: [G.chat(tok, G.user_prompt(p, it)) for it in pan[p]] for p in W.PANELS}
    deadline = (
        float((out / "deadline").read_text()) if (out / "deadline").exists() else float("inf")
    )
    evals_p = out / "evals.jsonl"
    done = {}
    if evals_p.exists():
        for line in evals_p.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    secs = [r["seconds"] for r in done.values() if r["tag"] == "member"]

    def run_canary(tag):
        pr = [G.chat(tok, G.user_prompt("mmlupro_gen", it)) for it in canary_items]
        t0 = time.time()
        o = eng.gen(pr, W.BUDGET)
        h = __import__("hashlib").sha256(json.dumps([x["ids"] for x in o]).encode()).hexdigest()
        rec = {"tag": tag, "sha256_token_ids": h, "seconds": time.time() - t0, "n": len(pr)}
        with open(out / "canary.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        ctx.log("canary", **rec)
        return h

    def evaluate(spec, tag, reverse=False):
        k = W.key(spec) if tag == "member" else tag
        if k in done:
            return done[k]
        t0 = time.time()
        eng.apply(out, spec)
        try:
            rec = {"key": k, "tag": tag, "spec": spec, "panels": {}}
            for p in W.PANELS:
                its, pr = pan[p], prompts[p]
                if reverse:
                    its, pr = its[::-1], pr[::-1]
                recs = score(p, its, eng.gen(pr, W.BUDGET))
                rec["panels"][p] = recs[::-1] if reverse else recs
        finally:
            eng.restore()
        rec["seconds"] = time.time() - t0
        with open(evals_p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done[k] = rec
        y = {
            p: (
                round(float(np.mean([r["correct"] for r in v])), 4),
                round(float(np.mean([r["trunc"] for r in v])), 4),
            )
            for p, v in rec["panels"].items()
        }
        ctx.log("eval", tag=tag, spec=spec, seconds=round(rec["seconds"], 1), acc_trunc=y)
        return rec

    if not (out / "canary.jsonl").exists():
        run_canary("canary_first")
    evaluate(W.BASE, "base")
    evaluate(W.BASE, "base_rerun", reverse=True)
    paths = W.make_paths({r: int(s) for r, s in meta["final_steps"].items()})
    q = W.queue(paths)
    if SMOKE:
        q = q[:6] + [k for k in q if W.unkey(k).get("merge")][:1]
    stopped = None
    for i, k in enumerate(q):
        est = float(np.median(secs)) if secs else 200.0
        reserve = 120.0  # canary repeat
        if k not in done and time.time() + est + reserve > deadline:
            stopped = {"at_index": i, "of": len(q), "est_member_s": est}
            ctx.log("deadline: queue stopped", **stopped)
            break
        new = k not in done
        r = evaluate(W.unkey(k), "member")
        if new:
            secs.append(r["seconds"])
    h_end = run_canary("canary_repeat")
    first = json.loads((out / "canary.jsonl").read_text().splitlines()[0])["sha256_token_ids"]
    jwrite(
        out / "eval_meta.json",
        {
            "queue_len": len(q),
            "evaluated_members": sum(1 for k in q if k in done),
            "stopped": stopped,
            "canary_bitwise_repeat": h_end == first,
            "engine": eng.engine_kw,
        },
    )
    (out / "STATUS_eval").write_text("DONE")
    if timing:
        timing_pilot(ctx, eng, out)


def timing_pilot(ctx, eng, out):
    """P-9B-eval: base generation seconds at 250, 1,000 and 2,512 exposed wme-p3h MC items (no outcome is used)."""
    dl = out / "timing_deadline"
    deadline = float(dl.read_text()) if dl.exists() else float("inf")
    tok = eng.tok
    pool = []
    for split in ("construction", "holdout"):
        s = PP.read_split(PANEL_P3H, split)
        for p in W.PANELS:
            pool += [(p, it) for it in s[p]]
    res = jread(out / "timing.json", {"rows": []})
    have = {r["n"] for r in res["rows"]}
    sizes = (12, 24) if SMOKE else TIMING_SIZES
    for n in sizes:
        if n in have:
            continue
        est = 0.2 * n
        if time.time() + est > deadline:
            ctx.log("timing pilot: deadline", n=n)
            break
        items = pool[:n]
        pr = [G.chat(tok, G.user_prompt(p, it)) for p, it in items]
        t0 = time.time()
        o = eng.gen(pr, W.BUDGET)
        sec = time.time() - t0
        row = {
            "n": n,
            "seconds": sec,
            "s_per_item": sec / n,
            "tokens": int(sum(len(x["ids"]) for x in o)),
            "hit_max_share": float(np.mean([x["hit_max"] for x in o])),
        }
        res["rows"].append(row)
        res["engine"] = eng.engine_kw
        jwrite(out / "timing.json", res)
        ctx.log("timing", **row)
    (out / "STATUS_timing").write_text("DONE")


def main():
    global SMOKE, REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("train", "eval"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        SMOKE = True
        REPO = "Qwen/Qwen3.5-0.8B"
        from huggingface_hub import constants

        snap = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B/snapshots"
        hf.TARGETS[REPO] = min(snap.iterdir()).name
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if a.phase == "train":
            train_phase(out)
        else:
            eval_phase(out, a.timing)
    except Exception:
        traceback.print_exc()
        (out / f"STATUS_{a.phase}").write_text("FAILED")
        raise


if __name__ == "__main__":
    main()
