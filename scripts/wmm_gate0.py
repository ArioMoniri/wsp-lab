#!/usr/bin/env python3
"""Study M Gate 0 driver (SYNTHESIS-r12 Gate 0 + 19 Nov addendum): port integrity, inert/null checks, canary, base
MedQA card check and per-family dose ladders on google/medgemma-1.5-4b-it (text only). EXPLORATORY. Local GPU only.

Rules, items, families and verdicts are fixed in ``wsp.wm.wmm_gate0`` (read its docstring). This script only runs
them. One resumable process: every phase writes ``OUT/<phase>.json`` (generations to ``OUT/gen/*.jsonl``) and is
skipped when present. Process VRAM is capped at 16 GB (``torch.cuda.set_per_process_memory_fraction``); the run waits
for free memory before loading and stops before 3 GPU-h (charged as wall time with the model resident).

    wmm_gate0.py --out data/wmm/gate0-YYYY-MM-DD [--smoke]

Edited weights exist only in GPU memory (LoRA factors under OUT/lora are kept locally, never committed or
published; HAI-DEF §3.1). No edit targets refusal of harmful requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from wsp.wm import wmm_gate0 as W
from wsp.wm.wmm import controls as C
from wsp.wm.wmm import roles as R

VRAM_CAP_GB = 16.0
GPU_H_CAP = 2.85  # stop new phases beyond this (hard rule: <= 3 GPU-h)
GEN_BATCH = 32
TF_BATCH = 8
TV_ANCHOR = "google/gemma-3-4b-it"
INTERP_END = "google/medgemma-4b-it"
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
CANARY_TOKENS = 48
SMOKE_ROLES = {
    ("medqa", "dev"): [("card", 48)],
    ("medqa", "train"): [("ladder", 16), ("h1_pool", 24), ("syc_pool", 24)],
    ("medmcqa", "train"): [("ladder", 8)],
}


class Row:
    def __init__(self, ids, score_from):
        self.ids, self.score_from = ids, score_from


class Ctx:
    def __init__(self, out: Path, smoke: bool):
        self.out, self.smoke = out, smoke
        (out / "gen").mkdir(parents=True, exist_ok=True)
        self.t_load = None
        self.gpu_s_before = 0.0
        acct = self.read("accounting")
        if acct:
            self.gpu_s_before = acct["gpu_seconds"]

    # ---- io
    def path(self, name):
        return self.out / f"{name}.json"

    def read(self, name):
        p = self.path(name)
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, name, obj):
        p = self.path(name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=float))
        tmp.replace(p)

    def log(self, event, **kw):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}
        line = json.dumps(rec, default=float)
        print(line, flush=True)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(line + "\n")

    def gpu_seconds(self):
        return self.gpu_s_before + (time.time() - self.t_load if self.t_load else 0.0)

    def charge(self):
        self.write(
            "accounting",
            {"gpu_seconds": self.gpu_seconds(), "gpu_hours": self.gpu_seconds() / 3600},
        )

    def check_cap(self, phase):
        self.charge()
        if self.gpu_seconds() / 3600 > GPU_H_CAP:
            raise SystemExit(f"GPU-h cap reached before {phase}: {self.gpu_seconds() / 3600:.2f} h")


# ------------------------------------------------------------------ GPU, model


def wait_for_gpu(need_gb: float):
    while True:
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 >= need_gb:
            return free, total
        print(f"waiting for GPU: {free / 2**30:.1f} GB free < {need_gb} GB", flush=True)
        time.sleep(60)


def load(ctx):
    free, total = wait_for_gpu(VRAM_CAP_GB + 1.0)
    torch.cuda.set_per_process_memory_fraction(VRAM_CAP_GB * 2**30 / total)
    ctx.t_load = time.time()
    model, tok, cfg, bhash, path = G3.load_text(G3.TARGET, device="cuda")
    from transformers import GenerationConfig

    ctx.stop = sorted(CH.stop_ids(tok))
    ctx.pad = tok.pad_token_id
    model.generation_config = GenerationConfig(
        eos_token_id=ctx.stop, pad_token_id=ctx.pad, bos_token_id=tok.bos_token_id, do_sample=False
    )
    ctx.model, ctx.tok, ctx.cfg, ctx.base_hash, ctx.snap = model, tok, cfg, bhash, path
    ctx.rm = R.GemmaRoleMap.from_config(cfg)
    ctx.params = dict(model.named_parameters())
    ctx.log(
        "loaded",
        repo=G3.TARGET,
        revision=G3.TARGETS[G3.TARGET],
        base_hash=bhash,
        gpu_free_gb=free / 2**30,
        allocated_gb=torch.cuda.memory_allocated() / 2**30,
        end_token=CH.end_token(tok),
        stop_ids=ctx.stop,
    )


def enc_prompt(ctx, user):
    return ctx.tok.encode(CH.chat_prompt(ctx.tok, user), add_special_tokens=False)


@torch.no_grad()
def generate(ctx, users, max_new, tag=None):
    """Greedy, batches of prompts sorted by length (fixed composition for paired base/edit comparisons)."""
    enc = [enc_prompt(ctx, u) for u in users]
    order = sorted(range(len(enc)), key=lambda i: (len(enc[i]), i))
    out = [None] * len(enc)
    eot = set(ctx.stop)
    for b0 in range(0, len(order), GEN_BATCH):
        idx = order[b0 : b0 + GEN_BATCH]
        L = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[ctx.pad] * (L - len(enc[i])) + enc[i] for i in idx], device="cuda")
        att = torch.tensor(
            [[0] * (L - len(enc[i])) + [1] * len(enc[i]) for i in idx], device="cuda"
        )
        rows = _gen_rows(ctx, ids, att, max_new)
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


def _gen_rows(ctx, ids, att, max_new):
    """Generated rows for one batch; on CUDA OOM the batch is split in halves (logged)."""
    try:
        g = ctx.model.generate(
            input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False
        )
        L = ids.shape[1]
        return g[:, L:].tolist()
    except torch.OutOfMemoryError:
        if ids.shape[0] == 1:
            raise
        torch.cuda.empty_cache()
        h = ids.shape[0] // 2
        ctx.log("oom_split", batch=int(ids.shape[0]))
        a = _gen_rows(ctx, ids[:h], att[:h], max_new)
        b = _gen_rows(ctx, ids[h:], att[h:], max_new)
        w = max(map(len, a + b))
        return [r + [ctx.pad] * (w - len(r)) for r in a + b]


@torch.no_grad()
def tf_nll(ctx, pairs):
    """Per-item mean-token NLL of target ids given prompt ids (teacher forcing; chunked LM head)."""
    model = ctx.model
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    res = [None] * len(pairs)
    for b0 in range(0, len(order), TF_BATCH):
        idx = order[b0 : b0 + TF_BATCH]
        seqs = [pairs[i][0] + pairs[i][1] for i in idx]
        T = max(map(len, seqs))
        ids = torch.full((len(idx), T), ctx.pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s)
            att[r, : len(s)] = 1
        h = model.model(
            input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False
        ).last_hidden_state
        for r, i in enumerate(idx):
            p, t = pairs[i]
            if not t:
                res[i] = 0.0
                continue
            hs = h[r, len(p) - 1 : len(p) + len(t) - 1]
            tgt = torch.tensor(t, device="cuda")
            tot = 0.0
            for c0 in range(0, len(t), 256):
                lg = model.lm_head(hs[c0 : c0 + 256]).float()
                tot += float(
                    torch.nn.functional.cross_entropy(lg, tgt[c0 : c0 + 256], reduction="sum")
                )
            res[i] = tot / len(t)
        del h
    return res


@torch.no_grad()
def canary(ctx):
    outs = generate(ctx, CANARY_PROMPTS, CANARY_TOKENS)
    ids = [o["ids"] for o in outs]
    pairs = [(o["prompt_ids"], o["ids"]) for o in outs]
    nll = tf_nll(ctx, pairs)
    enc = enc_prompt(ctx, CANARY_PROMPTS[0])
    logits = ctx.model(input_ids=torch.tensor([enc], device="cuda")).logits[0, -1].float().cpu()
    return {
        "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        "nll": nll,
        "first_text": outs[0]["text"],
        "pairs": pairs,
    }


def logits_of(ctx, users):
    with torch.no_grad():
        return [
            ctx.model(input_ids=torch.tensor([enc_prompt(ctx, u)], device="cuda"))
            .logits[0, -1]
            .float()
            .cpu()
            for u in users
        ]


def lean(sess, act):
    """Apply one action and drop its fp32 realised increment (GPU memory under the 16 GB cap)."""
    rec = sess.apply(act)
    rec.realised = torch.empty(0)
    return rec


def full_hash(ctx):
    return G3.text_state_hash(ctx.model)


def writers_hash(ctx):
    return A.state_hash({n: ctx.params[n].detach() for n in LR.writer_targets(ctx.rm)})


# ------------------------------------------------------------------ data


def load_items(ctx):
    got = ctx.read("items")
    if got:
        return got
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def parquet(repo, split, rev):
        return pd.read_parquet(
            hf_hub_download(
                repo, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=rev
            )
        )

    revs = {
        "openlifescienceai/medqa": "153e61cdd129eb79d3c27f82cdf3bc5e018c11b0",
        "openlifescienceai/medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    }
    by = {
        ("medqa", "train"): [
            W.medqa_item(r)
            for r in parquet(
                "openlifescienceai/medqa", "train", revs["openlifescienceai/medqa"]
            ).to_dict("records")
        ],
        ("medqa", "dev"): [
            W.medqa_item(r)
            for r in parquet(
                "openlifescienceai/medqa", "dev", revs["openlifescienceai/medqa"]
            ).to_dict("records")
        ],
        ("medmcqa", "train"): [
            x
            for x in (
                W.medmcqa_item(r)
                for r in parquet(
                    "openlifescienceai/medmcqa", "train", revs["openlifescienceai/medmcqa"]
                ).to_dict("records")
            )
            if x is not None
        ],
    }
    if ctx.smoke:
        W.ROLES = SMOKE_ROLES
    roles = W.designate(by)
    exposure = [
        {"source": it["source"], "split": split, "id": it["id"], "hash": it["hash"], "role": role}
        for (src, split), rl in W.ROLES.items()
        for role, _n in rl
        for it in roles[role]
        if it["source"] == src
    ]
    ctx.write(
        "exposure",
        {
            "salt": W.SALT,
            "status": "EXPOSED (study M Gate 0); never sealed",
            "datasets": revs,
            "items": exposure,
        },
    )
    ctx.write("items", roles)
    return roles


# ------------------------------------------------------------------ phases


def phase_inert(ctx):
    if ctx.read("inert"):
        return ctx.read("inert")
    ctx.check_cap("inert")
    t0 = time.time()
    rep = {}
    h0 = full_hash(ctx)
    rep["state_hash_start"] = h0
    c0 = canary(ctx)
    ctx.canary_pairs = c0.pop("pairs")
    rep["canary_start"] = c0
    ref_logits = logits_of(ctx, CANARY_PROMPTS[:4])

    def same_logits():
        return all(
            torch.equal(a, b)
            for a, b in zip(ref_logits, logits_of(ctx, CANARY_PROMPTS[:4]), strict=True)
        )

    # (1) exact null: every writer at dose 0 of three families
    sess = A.EditSession(ctx.params, h0, ctx.rm)
    g = torch.Generator().manual_seed(0)
    for i, name in enumerate(LR.writer_targets(ctx.rm)):
        w = ctx.params[name]
        acts = [A.scale(name, 2.0, 0.0), A.noise(name, seed=i, rho=0.1, dose=0.0)]
        acts.append(
            A.rank_one(
                name,
                torch.randn(w.shape[0], generator=g),
                torch.randn(w.shape[1], generator=g),
                1.0,
                0.0,
            )
        )
        for act in acts:
            lean(sess, act)
    rep["null_changed_tensors"] = len(sess.changed())
    rep["null_theta_hash_is_base"] = sess.theta_hash() == h0
    rep["null_logits_bitwise"] = same_logits()
    sess.restore()
    # (2) apply / restore identity on a real (non-null) edit
    sess = A.EditSession(ctx.params, h0, ctx.rm)
    lean(sess, A.scale(ctx.rm.name(10, "down"), 1.37, 1.0))
    lean(sess, A.noise(ctx.rm.name(5, "o"), seed=1, rho=0.05, dose=1.0))
    rep["edit_changes_logits"] = not same_logits()
    sess.restore()
    rep["restore_logits_bitwise"] = same_logits()
    # (3) inert controls: compensated pairs (sliding 0, full 5) and k_norm-cancelled k_proj x2 (layers 0, 5, 11),
    # each alone and all together, against the determinism floor (the same forward twice).
    base_nll = tf_nll(ctx, ctx.canary_pairs)
    rep["determinism_max_abs_dnll"] = float(
        max(abs(a - b) for a, b in zip(tf_nll(ctx, ctx.canary_pairs), base_nll, strict=True))
    )
    groups = {
        "compensated_mlp": [
            a for b in C.compensated_pairs(ctx.rm, mlp_layers=(0, 5, 20)) for a in b
        ],
        "compensated_attention": [
            a for b in C.compensated_pairs(ctx.rm, attention_layers=(0, 5, 20)) for a in b
        ],
        "k_norm_cancelled": C.k_norm_scales(ctx.rm, layers=(0, 5, 11)),
    }
    groups["all"] = [a for g in list(groups.values()) for a in g]
    rep["controls"] = {}
    for gname, acts in groups.items():
        sess = A.EditSession(ctx.params, h0, ctx.rm)
        for act in sorted(acts, key=lambda a: a.tensor):
            lean(sess, act)
        ed = tf_nll(ctx, ctx.canary_pairs)
        rep["controls"][gname] = {
            "tensors_changed": len(sess.changed()),
            "max_abs_dnll": float(max(abs(a - b) for a, b in zip(ed, base_nll, strict=True))),
            "mean_dnll": float(np.mean(np.subtract(ed, base_nll))),
            "logits_bitwise": same_logits(),
        }
        sess.restore()
    rep["controls_max_abs_dnll"] = rep["controls"]["all"]["max_abs_dnll"]
    # (4) uniform writer rescale (all o_proj and down_proj x2): post-norm cancellation, descriptive
    sess = A.EditSession(ctx.params, h0, ctx.rm)
    for name in LR.writer_targets(ctx.rm):
        lean(sess, A.scale(name, 2.0, 1.0))
    ed = tf_nll(ctx, ctx.canary_pairs)
    rep["uniform_writer_x2_mean_dnll"] = float(np.mean(np.subtract(ed, base_nll)))
    sess.restore()
    sess = A.EditSession(ctx.params, h0, ctx.rm)
    for i, name in enumerate(LR.writer_targets(ctx.rm)):
        # same relative norm as x2 (||ΔW|| = ||W||)
        lean(sess, A.noise(name, seed=100 + i, rho=1.0, dose=1.0))
    ed = tf_nll(ctx, ctx.canary_pairs)
    rep["directional_noise_same_norm_mean_dnll"] = float(np.mean(np.subtract(ed, base_nll)))
    sess.restore()
    rep["after_session_edits_hash"] = full_hash(ctx)
    # (5) weight-space lines: t=0 restores the base bytes; t=1 reproduces the endpoint model bitwise
    for fam, other in (("tv", TV_ANCHOR), ("interp", INTERP_END)):
        line = make_line(other)
        r0 = line.apply(ctx.model, 0.0)
        rep[f"{fam}_t0_changed_tensors"] = r0["changed_tensors"]
        r1 = line.apply(ctx.model, 1.0)
        rep[f"{fam}_t1_inexact_entries"] = r1["far_endpoint_inexact_entries"]
        rep[f"{fam}_t1_changed_tensors"] = r1["changed_tensors"]
        rep[f"{fam}_t1_endpoint_hash"] = full_hash(ctx)
        rep[f"{fam}_tau_norm"] = r1["intended_norm"]
        line.apply(ctx.model, 0.0)
        rep[f"{fam}_restored_hash"] = full_hash(ctx)
    tied = ctx.model.lm_head.weight.data_ptr() == ctx.model.model.embed_tokens.weight.data_ptr()
    rep["lm_head_tied"] = tied
    rep["restore_logits_bitwise_after_lines"] = same_logits()
    rep["checks"] = {
        "exact_null": rep["null_changed_tensors"] == 0
        and rep["null_theta_hash_is_base"]
        and rep["null_logits_bitwise"],
        "apply_restore_identity": rep["edit_changes_logits"]
        and rep["restore_logits_bitwise"]
        and rep["after_session_edits_hash"] == h0,
        # Gemma 3 inert spec: the power-of-two compensated pairs must be bitwise inert. The k_norm-cancelled k_proj x2
        # (an exact control on the Qwen spec) is not exact on Gemma 3 (gate0-diag: max |dNLL| 0.064), so it is
        # reported descriptively and is not part of the check.
        "inert_controls": all(
            rep["controls"][g]["max_abs_dnll"] == 0.0 and rep["controls"][g]["logits_bitwise"]
            for g in ("compensated_mlp", "compensated_attention")
        ),
        "line_t0_identity": rep["tv_t0_changed_tensors"] == 0
        and rep["interp_t0_changed_tensors"] == 0
        and rep["tv_restored_hash"] == h0
        and rep["interp_restored_hash"] == h0
        and rep["restore_logits_bitwise_after_lines"],
        "line_t1_reproduces_endpoints": rep["tv_t1_inexact_entries"] == 0
        and rep["interp_t1_inexact_entries"] == 0,
        "tied_head": tied,
    }
    rep["pass"] = all(rep["checks"].values())
    rep["seconds"] = time.time() - t0
    ctx.write("inert", rep)
    ctx.write("canary_pairs", ctx.canary_pairs)
    ctx.log("inert", **rep["checks"], passed=rep["pass"], seconds=rep["seconds"])
    return rep


_LINES = {}


def make_line(other_repo):
    if other_repo not in _LINES:
        _LINES[other_repo] = G3.WeightLine(
            G3.WeightSource(G3.snapshot(G3.TARGET)),
            G3.WeightSource(G3.snapshot(other_repo)),
            chunk_elems=1 << 24,
        )
    return _LINES[other_repo]


def gen_cached(ctx, name, users, max_new):
    p = ctx.out / "gen" / f"{name}.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines() if x]
        if len(rows) == len(users):
            return rows
    t0 = time.time()
    rows = generate(ctx, users, max_new)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    toks = sum(len(r["ids"]) for r in rows)
    ctx.log(
        "generated",
        name=name,
        n=len(rows),
        tokens=toks,
        seconds=time.time() - t0,
        tok_per_s=toks / (time.time() - t0),
    )
    ctx.charge()
    return rows


def scored(items, gens):
    return [W.score(it, g["text"], g["hit_max"]) for it, g in zip(items, gens, strict=True)]


def phase_card(ctx, roles):
    if ctx.read("card"):
        return ctx.read("card")
    ctx.check_cap("card")
    items = roles["card"]
    gens = gen_cached(ctx, "card_base", [W.neutral_prompt(it) for it in items], W.CARD_TOKENS)
    sc = scored(items, gens)
    n = len(items)
    rep = {
        "n": n,
        "split": "MedQA dev (openlifescienceai/medqa); card figure is on MedQA test (not read)",
        "max_new_tokens": W.CARD_TOKENS,
        **W.card_verdict(sum(s["correct"] for s in sc), n),
        "flex_accuracy": 100.0 * sum(s["flex_correct"] for s in sc) / n,
        "truncation_rate": sum(s["truncated"] for s in sc) / n,
        "unparsed_rate": sum(s["letter"] is None for s in sc) / n,
        "length_p50": float(np.percentile([len(g["ids"]) for g in gens], 50)),
        "length_p95": float(np.percentile([len(g["ids"]) for g in gens], 95)),
        "ladder_budget": W.ladder_budget([len(g["ids"]) for g in gens]),
        "card_medgemma1_4b": 64.4,
    }
    ctx.write("card", rep)
    ctx.log(
        "card",
        **{k: rep[k] for k in ("accuracy", "difference", "truncation_rate", "ladder_budget")},
    )
    return rep


def gen_subset(roles):
    """The first GEN_SUBSET MedQA ladder items (ranking order): generated at every dose."""
    return [it for it in roles["ladder"] if it["source"] == "medqa"][: W.GEN_SUBSET]


def base_ladder(ctx, roles, B):
    """Base generations: ladder (neutral; drift references), the generation subset (neutral and suggested, with the
    same batch composition as at every dose), pools (neutral)."""
    lad = roles["ladder"]
    g_lad = gen_cached(ctx, "ladder_base", [W.neutral_prompt(it) for it in lad], B)
    sub = gen_subset(roles)
    g_sub = gen_cached(ctx, "subset_base", [W.neutral_prompt(it) for it in sub], B)
    g_syc = gen_cached(ctx, "subset_syc_base", [W.suggested_prompt(it) for it in sub], B)
    g_h1 = gen_cached(ctx, "h1_pool_base", [W.neutral_prompt(it) for it in roles["h1_pool"]], B)
    g_sp = gen_cached(ctx, "syc_pool_base", [W.neutral_prompt(it) for it in roles["syc_pool"]], B)
    return g_lad, g_sub, g_syc, g_h1, g_sp


def drift_pairs(ctx, g_lad):
    eot = CH.end_token_id(ctx.tok)
    return [(g["prompt_ids"], g["ids"] + ([] if g["hit_max"] else [eot])) for g in g_lad]


def lora_rows(ctx, roles, g_h1, g_sp):
    eot = CH.end_token_id(ctx.tok)
    h1, syc = [], []
    for it, g in zip(roles["h1_pool"], g_h1, strict=True):
        if W.score(it, g["text"], g["hit_max"])["correct"]:
            h1.append(Row(g["prompt_ids"] + g["ids"] + [eot], len(g["prompt_ids"])))
    for it, g in zip(roles["syc_pool"], g_sp, strict=True):
        if W.score(it, g["text"], g["hit_max"])["correct"]:
            p = enc_prompt(ctx, W.suggested_prompt(it))
            syc.append(Row(p + g["ids"] + [eot], len(p)))
    return {"h1_lora": h1, "syc_lora": syc}


def seed_for(label):
    return int(hashlib.sha256(f"{W.SALT}:{label}".encode()).hexdigest()[:8], 16)


def phase_lora(ctx, fam, rows):
    d = ctx.out / "lora"
    d.mkdir(exist_ok=True)
    f = d / f"{fam}.pt"
    meta = ctx.read(f"lora_{fam}")
    if meta and f.exists():
        return torch.load(f), meta
    ctx.check_cap(f"lora {fam}")
    wh = writers_hash(ctx)
    ad = LR.LoRA(
        ctx.model,
        LR.writer_targets(ctx.rm),
        rank=W.LORA["rank"],
        alpha=W.LORA["alpha"],
        seed=seed_for(f"lora:{fam}"),
    )
    log = LR.train(
        ctx.model,
        ad,
        rows,
        pad_id=ctx.pad,
        lr=W.LORA["lr"],
        epochs=1 if ctx.smoke else W.LORA["epochs"],
        batch_rows=W.LORA["batch_rows"],
        seed=seed_for(f"lora-order:{fam}"),
        device="cuda",
        max_seconds=1800.0,
        warmup=W.LORA["warmup"],
        checkpointing=True,
        ce_chunk=1024,
        log=lambda e, **kw: ctx.log(e, family=fam, **kw),
    )
    factors = ad.factors()
    torch.save(factors, f)
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
        )
    }
    meta["loss_first10"] = float(np.mean(log["loss"][:10])) if log["loss"] else None
    meta["loss_last10"] = float(np.mean(log["loss"][-10:])) if log["loss"] else None
    meta["mean_target_tokens"] = (
        float(np.mean([len(r.ids) - r.score_from for r in rows])) if rows else None
    )
    meta["writers_unchanged"] = writers_hash(ctx) == wh
    meta["delta_rel_norm"] = float(
        np.sqrt(sum(float((s * (b @ a)).pow(2).sum()) for b, a, s in factors.values()))
        / np.sqrt(sum(float(ctx.params[n].float().pow(2).sum()) for n in factors))
    )
    del ad
    torch.cuda.empty_cache()
    ctx.write(f"lora_{fam}", meta)
    ctx.log("lora trained", family=fam, **meta)
    return factors, meta


def apply_state(ctx, fam, dose, factors):
    """Returns an undo callable."""
    if fam == "tv":
        line = make_line(TV_ANCHOR)
        rec = line.apply(ctx.model, G3.tv_t(dose))
        return rec, lambda: line.apply(ctx.model, 0.0)
    if fam == "interp":
        line = make_line(INTERP_END)
        rec = line.apply(ctx.model, dose)
        return rec, lambda: line.apply(ctx.model, 0.0)
    sess = A.EditSession(ctx.params, ctx.base_hash, ctx.rm)
    LR.apply_lean(sess, LR.lowrank_actions(factors[fam], dose))
    recs = sess.records
    rec = {
        "intended_norm": float(np.sqrt(sum(r.intended_norm**2 for r in recs))),
        "realised_norm": float(np.sqrt(sum(r.realised_norm**2 for r in recs))),
        "changed_tensors": len(sess.changed()),
    }
    return rec, sess.restore


def phase_family(ctx, fam, roles, B, base, factors):
    name = f"ladder_{fam}"
    rep = ctx.read(name) or {"family": fam, **W.FAMILIES[fam], "states": {}}
    sub = gen_subset(roles)
    for dose in W.FAMILIES[fam]["doses"]:
        key = repr(float(dose))
        if key in rep["states"]:
            continue
        ctx.check_cap(f"{fam} {dose}")
        t0 = time.time()
        rec, undo = apply_state(ctx, fam, dose, factors)
        nll = tf_nll(ctx, base["pairs"])
        gl = gen_cached(ctx, f"subset_{fam}_{dose}", [W.neutral_prompt(it) for it in sub], B)
        sc = scored(sub, gl)
        st = {
            "edit": rec,
            "drift": list(np.subtract(nll, base["nll"])),
            "correct": [s["correct"] for s in sc],
            "truncated": [s["truncated"] for s in sc],
        }
        if fam == "syc_lora":
            gs = gen_cached(
                ctx, f"subset_syc_{fam}_{dose}", [W.suggested_prompt(it) for it in sub], B
            )
            ss = scored(sub, gs)
            st["syc_correct"] = [s["correct"] for s in ss]
            st["syc_adopt"] = [W.adopted(it, s["letter"]) for s, it in zip(ss, sub, strict=True)]
            st["syc_truncated"] = [s["truncated"] for s in ss]
        undo()
        st["seconds"] = time.time() - t0
        rep["states"][key] = st
        ctx.write(name, rep)
        ctx.log(
            "state",
            family=fam,
            dose=dose,
            drift=float(np.mean(st["drift"])),
            acc=float(np.mean(st["correct"])),
            seconds=st["seconds"],
        )
    if fam in ("tv", "interp"):
        rep["restored_hash"] = full_hash(ctx)
    else:
        rep["restored_writers_hash"] = writers_hash(ctx)
    ctx.write(name, rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop-after", choices=["inert", "card"], default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(out, a.smoke)
    if a.smoke:
        W.ROLES = SMOKE_ROLES
        for f in W.FAMILIES.values():
            f["doses"] = [d for d in f["doses"] if d in (0.5, 1.5, 1.0)][:2]
    git = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    ctx.write(
        "manifest",
        {
            "study": "M",
            "gate": 0,
            "target": G3.TARGET,
            "revisions": G3.TARGETS,
            "git_sha": git,
            "smoke": a.smoke,
            "vram_cap_gb": VRAM_CAP_GB,
            "gen_batch": GEN_BATCH,
            "families": W.FAMILIES,
            "lora": W.LORA,
            "roles": {f"{k[0]}:{k[1]}": v for k, v in W.ROLES.items()},
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    )
    roles = load_items(ctx)
    arch = ctx.read("architecture")
    if not arch:
        arch = G3.compare_architectures(
            {r: G3.snapshot(r) for r in (G3.TARGET, TV_ANCHOR, INTERP_END)}
        )
        ctx.write("architecture", arch)
    if not arch["same"]:
        raise SystemExit("architectures differ; the task vector and interpolation are undefined")
    load(ctx)
    inert = phase_inert(ctx)
    ctx.canary_pairs = ctx.read("canary_pairs")
    if a.stop_after == "inert":
        ctx.charge()
        return
    card = phase_card(ctx, roles)
    if a.stop_after == "card":
        ctx.charge()
        return
    B = card["ladder_budget"]
    g_lad, g_sub, g_syc, g_h1, g_sp = base_ladder(ctx, roles, B)
    base = ctx.read("ladder_base_nll")
    pairs = drift_pairs(ctx, g_lad)
    if not base:
        base = {"nll": tf_nll(ctx, pairs)}
        sub = gen_subset(roles)
        sc = scored(sub, g_sub)
        ss = scored(sub, g_syc)
        sl = scored(roles["ladder"], g_lad)
        base.update(
            {
                "correct": [s["correct"] for s in sc],
                "truncated": [s["truncated"] for s in sc],
                "syc_correct": [s["correct"] for s in ss],
                "syc_adopt": [W.adopted(it, s["letter"]) for s, it in zip(ss, sub, strict=True)],
                "syc_truncated": [s["truncated"] for s in ss],
                "ladder_correct": [s["correct"] for s in sl],
                "ladder_truncated": [s["truncated"] for s in sl],
                "ladder_sources": [it["source"] for it in roles["ladder"]],
                "budget": B,
            }
        )
        ctx.write("ladder_base_nll", base)
    base["pairs"] = pairs
    rows = lora_rows(ctx, roles, g_h1, g_sp)
    factors = {}
    for fam in ("h1_lora", "syc_lora"):
        factors[fam], _ = phase_lora(ctx, fam, rows[fam])
    fams = {}
    for fam in W.FAMILIES:
        fams[fam] = phase_family(ctx, fam, roles, B, base, factors)
    c1 = canary(ctx)
    c1.pop("pairs")
    final = {
        "canary_end": c1,
        "canary_match": c1["ids_sha256"] == inert["canary_start"]["ids_sha256"]
        and c1["logits_sha256"] == inert["canary_start"]["logits_sha256"],
        "state_hash_end": full_hash(ctx),
    }
    final["state_hash_matches_start"] = final["state_hash_end"] == inert["state_hash_start"]
    ctx.write("final", final)
    ctx.charge()
    ctx.log(
        "done",
        gpu_hours=ctx.gpu_seconds() / 3600,
        canary_match=final["canary_match"],
        state_hash_matches_start=final["state_hash_matches_start"],
    )


if __name__ == "__main__":
    main()
