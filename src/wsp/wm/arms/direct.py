"""Non-reader secondary arms (PREREG_WME §6.8.3): WM greedy one-step (group R), best-of-N steering and LoRA SFT
(group C). Every real evaluation goes through ``spec.ArmContext.evaluate`` on the shared ``BudgetLedger``; every
winner is chosen by ``spec.select_winner`` (§6.5.5).

* **WM greedy one-step** (:func:`run_greedy_episode`): isolates planning. Same predictor (WM-E, R0 inference), same
  scorer as the shared planner (``wm.planner.Planner.score``: μ̂ - κσ̃ - ψ·pen with guard penalties; ``"ranks"``
  drops σ for every arm), no beam and no inner optimisation. Each of the four rounds scores every one-step extension
  (one more distinct primitive at the end, length <= 4) of the base and of the arm's own feasible incumbents (top
  :data:`N_INCUMBENTS` by dev LCB), excludes already evaluated sequences, and evaluates the top b = B/4 (ties by
  sequence order). Primitive doses are the library's own (dose multiplier 1).
* **Best-of-N steering** (:func:`run_bon_steering`): action space = single rank-one steering actions at **every**
  layer × the construction/e2 contrasts (:data:`STEER_CONTRASTS`) × shifts ±{0.03125, 0.0625}·residual norm
  (:data:`STEER_SHIFTS`); B distinct actions drawn uniformly without replacement (seeded), each evaluated directly
  (one bundle). No predictor, no screen.
* **LoRA SFT** (:func:`run_lora_sft`): one rank-8 LoRA trained on construction/e2 items only (the library's shard;
  :func:`check_shard` refuses any item outside the shard or any dev/test/validation hash), checkpoints at
  :func:`checkpoint_steps` (n = min(B, 32), evenly spaced, last = final step), each checkpoint's task vector
  evaluated on dev as one ``checkpoint`` bundle. Training GPU time is charged as ``train_gpu_s``. The real training
  loop with factor snapshots is :func:`train_with_checkpoints` (GPU; ``wm.lora`` adapter and batching).
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from wsp.wm.arms import pool as P
from wsp.wm.arms import spec as S
from wsp.wm.planner import Planner, PlannerConfig

N_INCUMBENTS = 4
STEER_CONTRASTS = ("syc", "gsm", "mathcomp", "trivia")  # the construction/e2 contrasts (§6.5.2)
STEER_SHIFTS = (-0.0625, -0.03125, 0.03125, 0.0625)
MAX_CHECKPOINTS = 32
LORA_SFT = {
    "rank": 8,
    "alpha": 16,
    "lr": 1e-4,
    "batch_rows": 8,
    "warmup": 10,
    "targets": "every residual writer (wm.lora.writer_targets)",
    "data": "construction/e2 shard only",
    "task_vector_lambda": 1.0,
    "max_checkpoints": MAX_CHECKPOINTS,
    "checkpoint_rule": "n = min(B, 32) evenly spaced steps, last = final step; each one dev bundle",
}


class SplitError(PermissionError):
    pass


# ------------------------------------------------------------------ WM greedy one-step


def greedy_scores(pred, cands, mode, support, cfg: PlannerConfig):
    """The shared planner scorer on library sequences (dose multiplier 1 per primitive)."""
    pl = Planner(max([a for c in cands for a in c], default=0) + 1, (-1.0, 1.0), "cem", cfg)
    return pl.score(pred, [[(a, 1.0) for a in c] for c in cands], None, mode, support)


def run_greedy_episode(
    ctx, B: int, mode: str = "lcb", cfg: PlannerConfig | None = None, rounds=P.ROUNDS
):
    if B % rounds:
        raise ValueError(f"B = {B} does not split into {rounds} rounds")
    cfg = cfg or PlannerConfig()
    sub = list(ctx.get("episode_library"))
    pred = ctx.get("wm_predictor")
    support = ctx.get("support_penalty")
    b = B // rounds
    log = []
    for r in range(rounds):
        hist = ctx.history()
        done = {tuple(h["candidate"]) for h in hist}
        feas = sorted(
            (h for h in hist if h["guard_ok"] and h["lcb"] > 0),
            key=lambda h: (-h["lcb"], h["order"]),
        )
        bases = [()] + [tuple(h["candidate"]) for h in feas[:N_INCUMBENTS]]
        cands = sorted(
            {
                inc + (p,)
                for inc in bases
                for p in sub
                if p not in inc and len(inc) < P.MAX_LEN and inc + (p,) not in done
            }
        )
        rec = {"round": r, "n_candidates": len(cands), "evaluated": []}
        if cands:
            sc = np.asarray(greedy_scores(pred, cands, mode, support, cfg), float)
            order = sorted(range(len(cands)), key=lambda i: (-sc[i], cands[i]))
            for i in order[:b]:
                ctx.evaluate(cands[i])
                rec["evaluated"].append(cands[i])
        log.append(rec)
    ctx.close("greedy rounds complete")
    return {"winner": S.select_winner(ctx.history()), "rounds": log}


# ------------------------------------------------------------------ best-of-N steering


def steering_space(n_layers: int, contrasts=STEER_CONTRASTS, shifts=STEER_SHIFTS) -> list:
    return [("steer", L, c, s) for L in range(int(n_layers)) for c in contrasts for s in shifts]


def run_bon_steering(ctx, B: int, seed: int, n_layers: int):
    ctx.get(
        "steer_contrasts_all_layers"
    )  # the θ0 contrast directions this arm realises (declared access)
    space = steering_space(n_layers)
    if B > len(space):
        raise ValueError("B exceeds the steering action space")
    idx = np.random.default_rng(seed).choice(len(space), size=B, replace=False)
    chosen = [space[i] for i in idx]
    for a in chosen:
        ctx.evaluate(a)
    ctx.close("B single steering actions evaluated")
    return {"winner": S.select_winner(ctx.history()), "evaluated": chosen, "space_size": len(space)}


# ------------------------------------------------------------------ LoRA SFT


def n_checkpoints(B: int) -> int:
    return min(int(B), MAX_CHECKPOINTS)


def checkpoint_steps(total_steps: int, n: int) -> list:
    if total_steps < 1 or n < 1:
        raise ValueError("need at least one step and one checkpoint")
    n = min(n, total_steps)
    return sorted({max(1, math.ceil(total_steps * (k + 1) / n)) for k in range(n)})


def check_shard(items, shard_hashes, forbidden_hashes):
    shard, bad = set(shard_hashes), set(forbidden_hashes)
    if shard & bad:
        raise SplitError("the construction/e2 shard intersects dev/test/validation hashes")
    outside = [it["hash"] for it in items if it["hash"] not in shard]
    if outside:
        raise SplitError(f"{len(outside)} training items are not construction/e2 items")


def run_lora_sft(ctx, train_fn, B: int, seed: int, total_steps: int, n_ckpt: int | None = None):
    """``train_fn(items, steps_at, seed) -> {"checkpoints": {step: candidate}, "gpu_seconds": s}``."""
    n = n_checkpoints(B) if n_ckpt is None else int(n_ckpt)
    if n > min(B, MAX_CHECKPOINTS):
        raise ValueError(f"at most min(B, {MAX_CHECKPOINTS}) checkpoint evaluations")
    data = ctx.get("construction_e2")
    check_shard(data["items"], data["shard_hashes"], data["forbidden_hashes"])
    steps_at = checkpoint_steps(total_steps, n)
    out = train_fn(data["items"], steps_at, seed)
    evals = []
    for s in steps_at:
        evals.append(ctx.evaluate(out["checkpoints"][s], kind="checkpoint"))
    ctx.close("checkpoints evaluated")
    costs = {
        "train_gpu_s": float(out["gpu_seconds"]),
        "eval_gpu_s": float(sum(e.get("seconds", 0.0) for e in evals)),
    }
    return {"winner": S.select_winner(ctx.history()), "steps_at": steps_at, "costs": costs}


def train_with_checkpoints(
    model, adapter, rows, pad_id, steps_at, lr=1e-4, batch_rows=8, seed=0, device="cuda", warmup=10
):
    """AdamW on the LoRA factors (``wm.lora`` batching and loss); snapshots of the factors at ``steps_at``.

    Epochs repeat the seeded shuffled order until the last checkpoint step. Base weights never require gradients."""
    import torch
    from torch import nn

    from wsp.wm import lora as LO

    for p in model.parameters():
        p.requires_grad_(False)
    params = list(adapter.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    want, snaps, losses, step, t0 = set(steps_at), {}, [], 0, time.time()
    was = model.training
    model.train()
    try:
        with adapter.active():
            while step < max(steps_at):
                rng.shuffle(order)
                for b0 in range(0, len(order), batch_rows):
                    if step >= max(steps_at):
                        break
                    ids, att, mask, tgt = LO._batch(
                        [rows[i] for i in order[b0 : b0 + batch_rows]], pad_id, device
                    )
                    h = model.model(
                        input_ids=ids, attention_mask=att, use_cache=False
                    ).last_hidden_state
                    loss = nn.functional.cross_entropy(model.lm_head(h[mask]).float(), tgt[mask])
                    for g in opt.param_groups:
                        g["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    step += 1
                    losses.append(float(loss.detach()))
                    if step in want:
                        # {name: (B, A, scale)}; cloned: on CPU, factors() aliases the live parameters
                        snaps[step] = {
                            n: (b.clone(), a.clone(), s)
                            for n, (b, a, s) in adapter.factors().items()
                        }
    finally:
        model.train(was)
        for p in params:
            p.requires_grad_(False)
    return {"checkpoints": snaps, "seconds": time.time() - t0, "loss": losses, "steps": step}
