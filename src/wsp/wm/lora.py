"""Rank-8 LoRA SFT task vectors for the WM-E pilot-2 E2 action space (RULING-wme-pilot1, section 3). EXPLORATORY.

* **Targets.** The audited residual writers (D1): every layer's token-mixer output projection and MLP down
  projection (``RoleMap.writers``). Vision and MTP tensors are never loaded or touched.
* **Adapter.** ΔW = (alpha / rank) · B A with A [rank, in] (Kaiming-uniform, seeded) and B [out, rank] (zeros), fp32,
  added to each target's output through a forward hook. Base weights never require gradients and are unchanged.
* **Data.** Construction-split items only (checked against the panel manifest's construction hashes; anything else
  raises :class:`SplitError`). Each example is the panel's own teacher-forced row (``wm.panel2.rows_for``): the loss
  is the mean NLL of the scored target tokens. Sycophancy trains the suggested-incorrect prompt row (correct answer
  despite the user's wrong suggestion); GSM8K and competition math train their gold rows.
* **Task vector.** :func:`lowrank_actions` gives one ``lora_task_vector`` action per target whose intended delta is
  λ · (alpha / rank) · B A, computed on the weight's device from the factors (no dense fp32 copy is stored).
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import random
import time

import torch
from torch import nn

from wsp.wm import actions as A
from wsp.wm import panel2 as P2


class SplitError(PermissionError):
    pass


def writer_targets(rm: A.RoleMap) -> list[str]:
    return sorted(rm.writers().values())


class LoRA(nn.Module):
    def __init__(self, model, names, rank=8, alpha=16, seed=0):
        super().__init__()
        self.model_ref = [model]  # not a submodule
        self.names = list(names)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = self.alpha / self.rank
        params = dict(model.named_parameters())
        g = torch.Generator().manual_seed(int(seed))
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self._keys = {}
        for i, n in enumerate(self.names):
            w = params[n]
            out_f, in_f = w.shape
            a = torch.empty(self.rank, in_f)
            bound = 1.0 / math.sqrt(
                in_f
            )  # kaiming_uniform(a=sqrt(5)) bound for a [rank, in] weight
            a.uniform_(-bound, bound, generator=g)
            k = f"t{i}"
            self._keys[n] = k
            self.A[k] = nn.Parameter(a.to(w.device))
            self.B[k] = nn.Parameter(torch.zeros(out_f, self.rank, device=w.device))

    @contextlib.contextmanager
    def active(self):
        model = self.model_ref[0]
        mods = dict(model.named_modules())
        hooks = []
        for n in self.names:
            k = self._keys[n]
            a, b, s = self.A[k], self.B[k], self.scale

            def hook(mod, inp, out, a=a, b=b, s=s):
                x = inp[0]
                return out + (s * ((x.to(a.dtype) @ a.T) @ b.T)).to(out.dtype)

            hooks.append(mods[n.rsplit(".", 1)[0]].register_forward_hook(hook))
        try:
            yield self
        finally:
            for h in hooks:
                h.remove()

    def factors(self) -> dict:
        return {
            n: (
                self.B[self._keys[n]].detach().float().cpu(),
                self.A[self._keys[n]].detach().float().cpu(),
                self.scale,
            )
            for n in self.names
        }

    def task_vector(self) -> dict:
        return {n: s * (b @ a) for n, (b, a, s) in self.factors().items()}


def _lowrank(a, w):
    b = a.param("B").to(w.device)
    aa = a.param("A").to(w.device)
    return (a.dose * a.param("scale")) * (b @ aa)


def lowrank_actions(factors: dict, lam: float) -> list[A.Action]:
    return [
        A.Action(
            "lora_task_vector",
            n,
            float(lam),
            (("B", b), ("A", a), ("scale", float(s))),
            _lowrank,
        )
        for n, (b, a, s) in sorted(factors.items())
    ]


def apply_lean(sess: A.EditSession, actions) -> None:
    """Apply a bundle (distinct tensors) and drop each record's realised fp32 increment (GPU memory)."""
    for a in sorted(actions, key=lambda a: a.tensor):
        rec = sess.apply(a)
        rec.realised = torch.empty(0)


# ------------------------------------------------------------------ data and training


SFT_PART = {"gsm8k_gold_lp": "gold", "mathcomp_gold_lp": "gold", "sycophancy_real": "suggested"}


def sft_rows(objective, items, tok, cfg, manifest) -> list:
    allowed = set(manifest["objectives"][objective]["construction"]["hashes"])
    bad = [it["id"] for it in items if it["hash"] not in allowed]
    if bad:
        raise SplitError(f"{objective}: {len(bad)} items are not construction items, e.g. {bad[0]}")
    part = SFT_PART[objective]
    rows = []
    for it in items:
        rows.extend(r for r in P2.rows_for(objective, it, tok, cfg) if r.key[2] == part)
    return rows


def _batch(rows, pad_id, device):
    T = max(len(r.ids) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.bool)
    tgt = torch.zeros((len(rows), T), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r.ids)] = torch.tensor(r.ids)
        att[i, : len(r.ids)] = 1
        mask[i, r.score_from - 1 : len(r.ids) - 1] = True
        tgt[i, r.score_from - 1 : len(r.ids) - 1] = torch.tensor(r.ids[r.score_from :])
    return ids.to(device), att.to(device), mask.to(device), tgt.to(device)


def train(
    model,
    adapter: LoRA,
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
    on_step=None,
) -> dict:
    """AdamW on the LoRA factors only; mean target-token NLL per batch; stops early at ``max_seconds`` wall.
    ``on_step(step, adapter)`` (optional) runs after every optimiser step (pilot 3: per-step checkpoints)."""
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
                    ids, att, mask, tgt = _batch(batch, pad_id, device)
                    h = model.model(
                        input_ids=ids, attention_mask=att, use_cache=False
                    ).last_hidden_state
                    logits = model.lm_head(h[mask]).float()
                    loss = nn.functional.cross_entropy(logits, tgt[mask])
                    for g in opt.param_groups:
                        g["lr"] = lr * min(1.0, (steps + 1) / max(1, warmup))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                    steps += 1
                    if on_step is not None:
                        on_step(steps, adapter)
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
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "seed": seed,
    }


__all__ = [
    "LoRA",
    "SplitError",
    "apply_lean",
    "dataclasses",
    "lowrank_actions",
    "sft_rows",
    "train",
    "writer_targets",
]
