"""Micro-batched LoRA training for study M (the registered ``wm/lora.py`` is not edited; chair rule 20 Nov 2026).

Same recipe as :func:`wsp.wm.lora.train`: the same AdamW, warmup schedule, per-epoch shuffle (``random.Random(seed)``),
optimizer batches of ``batch_rows`` rows, and the loss = mean cross-entropy over all target tokens of the batch. Each
optimizer batch is run as micro-batches of ``micro_rows`` rows; each micro-loss is its token sum divided by the
batch's total target tokens, so the accumulated gradient equals the full-batch gradient (up to float summation
order). Gate 1 needs it because rows under the frozen 4,096-token harness reach ~4,400 tokens, and 8 padded rows
exceeded the 16 GB cap (OOM in the first backward, 20 Nov 2026).
"""

from __future__ import annotations

import random
import time

import torch

from wsp.wm import lora as LR


def train(
    model,
    adapter,
    rows,
    pad_id,
    lr=1e-4,
    epochs=1,
    batch_rows=8,
    seed=0,
    device="cuda",
    max_seconds=1000.0,
    warmup=10,
    checkpointing=False,
    log=None,
    ce_chunk=None,
    micro_rows=1,
) -> dict:
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(adapter.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    losses, steps, t0, stopped = [], 0, time.time(), False
    was_training = model.training
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    try:
        with adapter.active():
            for _ep in range(epochs):
                rng.shuffle(order)
                for b0 in range(0, len(order), batch_rows):
                    if time.time() - t0 > max_seconds:
                        stopped = True
                        break
                    batch = [rows[i] for i in order[b0 : b0 + batch_rows]]
                    n_tok = sum(len(r.ids) - r.score_from for r in batch)
                    for g in opt.param_groups:
                        g["lr"] = lr * min(1.0, (steps + 1) / max(1, warmup))
                    opt.zero_grad(set_to_none=True)
                    total = 0.0
                    for m0 in range(0, len(batch), micro_rows):
                        micro = batch[m0 : m0 + micro_rows]
                        ids, att, mask, tgt = LR._batch(micro, pad_id, device)
                        h = model.model(
                            input_ids=ids, attention_mask=att, use_cache=False
                        ).last_hidden_state
                        hm, tm = h[mask], tgt[mask]
                        if ce_chunk:
                            part = LR.chunked_ce(model.lm_head, hm, tm, ce_chunk) * hm.shape[0]
                        else:
                            part = torch.nn.functional.cross_entropy(
                                model.lm_head(hm).float(), tm, reduction="sum"
                            )
                        loss = part / n_tok
                        loss.backward()
                        total += float(loss.detach())
                        del h, hm, tm, part, loss, ids, att, mask, tgt
                    opt.step()
                    losses.append(total)
                    steps += 1
                    if log and steps % 20 == 0:
                        log("lora step", step=steps, loss=round(sum(losses[-20:]) / 20, 4))
                if stopped:
                    break
    finally:
        if checkpointing:
            model.gradient_checkpointing_disable()
        model.train(was_training)
        for p in adapter.parameters():
            p.requires_grad_(False)
    return {
        "steps": steps,
        "loss": losses,
        "seconds": time.time() - t0,
        "stopped_by_cap": stopped,
        "n_rows": len(rows),
        "epochs": epochs,
        "lr": lr,
        "batch_rows": batch_rows,
        "micro_rows": micro_rows,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "seed": seed,
    }
