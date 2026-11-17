"""Pilot-3 edit deltas on the residual writers (RULING-wme-pilot3 §3, §6): LoRA factors, SVF singular-value scales
with a KL-anchored policy-gradient loss, the H4 refusal-orthogonal projection, and in-place application to any
mapping of live bf16 tensors (the vLLM engine's or a Hugging Face model's) with a bitwise restore. EXPLORATORY.

* **Writer keys.** A writer is identified by (layer, kind), kind ∈ {down, o}; :func:`writer_key` parses both the
  Hugging Face names (``model.language_model.layers.L.mlp.down_proj.weight``) and the vLLM names
  (``language_model.model.layers.L.linear_attn.out_proj.weight`` / ``self_attn.o_proj.weight``).
* **Application** (:class:`LiveEditor`): W ← bf16(fp32(W₀) + ΔW), one tensor at a time; originals are cloned on first
  touch; :meth:`LiveEditor.restore` copies them back and checks bytes. A zero delta leaves W₀ bitwise unchanged.
* **SVF** (:class:`SVF`): W(z) = W + U diag(σ ⊙ (z - 1)) Vᵀ from an fp32 SVD of W; exactly W at z = 1 (the delta is
  zero). Training uses forward hooks on the HF model; :func:`pg_loss` is REINFORCE with a group-mean baseline plus
  β·KL(π_z ‖ π₀) (k3 estimator on the sampled tokens). SVF is trained by policy gradient, never next-token loss.
* **H4** (:func:`project_out`): ΔW′ = (I - r̂r̂ᵀ)ΔW with r̂ the unit refusal direction of the writer's layer.
"""

from __future__ import annotations

import contextlib
import re

import torch
from torch import nn

_KEY = re.compile(
    r"layers\.(\d+)\.(mlp\.down_proj|linear_attn\.out_proj|self_attn\.o_proj)\.weight$"
)


def writer_key(name: str):
    m = _KEY.search(name)
    if not m:
        return None
    return int(m.group(1)), ("down" if m.group(2) == "mlp.down_proj" else "o")


def writer_map(names) -> dict:
    out = {}
    for n in names:
        k = writer_key(n)
        if k is not None:
            if k in out:
                raise ValueError(f"two tensors for writer {k}")
            out[k] = n
    return out


# ------------------------------------------------------------------ deltas


def lora_delta(factors_entry, lam: float, device) -> torch.Tensor:
    b, a, s = factors_entry
    return (float(lam) * float(s)) * (b.to(device).float() @ a.to(device).float())


def project_out(delta: torch.Tensor, r: torch.Tensor | None) -> torch.Tensor:
    """(I - r̂r̂ᵀ)ΔW for a writer whose output space is the residual stream (rows)."""
    if r is None:
        return delta
    r = r.to(delta.device, torch.float32)
    r = r / r.norm()
    return delta - torch.outer(r, r @ delta)


class LiveEditor:
    """In-place edits of live bf16 tensors with exact restore."""

    def __init__(self, params: dict):
        self.params = params
        self.orig: dict = {}

    def set_delta(self, name: str, delta: torch.Tensor) -> None:
        live = self.params[name]
        if name not in self.orig:
            self.orig[name] = live.detach().clone()
        new = (self.orig[name].float() + delta.to(live.device, torch.float32)).to(live.dtype)
        with torch.no_grad():
            live.copy_(new)

    def apply(self, deltas) -> None:
        """deltas: iterable of (name, callable -> fp32 delta) evaluated one at a time (bounded memory)."""
        for name, fn in deltas:
            d = fn()
            self.set_delta(name, d)
            del d

    def restore(self) -> None:
        with torch.no_grad():
            for name, w0 in self.orig.items():
                self.params[name].copy_(w0)
        for name, w0 in self.orig.items():
            if not torch.equal(self.params[name], w0):
                raise RuntimeError(f"{name}: restore did not return the saved bytes")
        self.orig.clear()

    def checksum(self, names) -> int:
        tot = 0
        for n in names:
            tot += int(self.params[n].detach().view(torch.int16).to(torch.int64).sum())
        return tot


# ------------------------------------------------------------------ SVF


class SVF(nn.Module):
    """Singular-value scales z on a set of writer weights; U, Vᵀ stored in ``store_dtype``; σ in fp32."""

    def __init__(self, model, names, store_dtype=torch.bfloat16):
        super().__init__()
        self.model_ref = [model]
        self.names = list(names)
        params = dict(model.named_parameters())
        self.z = nn.ParameterDict()
        self.U, self.S, self.Vh, self._keys = {}, {}, {}, {}
        for i, n in enumerate(self.names):
            w = params[n].detach().float()
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            k = f"t{i}"
            self._keys[n] = k
            self.U[n] = U.to(store_dtype)
            self.S[n] = S
            self.Vh[n] = Vh.to(store_dtype)
            self.z[k] = nn.Parameter(torch.ones_like(S))
            del U, Vh, w

    def zvec(self, n):
        return self.z[self._keys[n]]

    def delta(self, n, z=None, alpha: float = 1.0) -> torch.Tensor:
        z = self.zvec(n).detach() if z is None else z
        scale = self.S[n] * (alpha * (z.float() - 1.0))
        return (self.U[n].float() * scale) @ self.Vh[n].float()

    def state(self) -> dict:
        return {n: self.zvec(n).detach().float().cpu().clone() for n in self.names}

    def load_state(self, zs: dict) -> None:
        with torch.no_grad():
            for n, z in zs.items():
                self.zvec(n).copy_(z.to(self.zvec(n).device))

    @contextlib.contextmanager
    def active(self):
        model = self.model_ref[0]
        mods = dict(model.named_modules())
        hooks = []
        for n in self.names:
            U, S, Vh, z = self.U[n], self.S[n], self.Vh[n], self.zvec(n)

            def hook(mod, inp, out, U=U, S=S, Vh=Vh, z=z):
                x = inp[0]
                h = (x.to(Vh.dtype) @ Vh.T).float() * (S * (z - 1.0))
                return out + (h.to(U.dtype) @ U.T).to(out.dtype)

            hooks.append(mods[n.rsplit(".", 1)[0]].register_forward_hook(hook))
        try:
            yield self
        finally:
            for h in hooks:
                h.remove()


def token_logps(model, ids, att, score_mask) -> torch.Tensor:
    """Per-token log-probabilities of ``ids`` at the positions where ``score_mask`` (shifted targets) is True."""
    h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    pos = score_mask[:, 1:]
    logits = model.lm_head(h[:, :-1][pos]).float()
    return logits.log_softmax(-1).gather(-1, ids[:, 1:][pos][:, None]).squeeze(-1)


def pg_loss(logp, logp0, seq_index, adv, beta: float):
    """REINFORCE (sequence-mean token log-prob × advantage) + β·KL(π_z‖π₀) by the k3 estimator; means over sequences."""
    nseq = int(seq_index.max()) + 1
    cnt = torch.zeros(nseq, device=logp.device).index_add_(0, seq_index, torch.ones_like(logp))
    mlogp = torch.zeros(nseq, device=logp.device).index_add_(0, seq_index, logp) / cnt
    d = (logp0 - logp).clamp(-20, 20)
    k3 = torch.exp(d) - d - 1.0
    mk3 = torch.zeros(nseq, device=logp.device).index_add_(0, seq_index, k3) / cnt
    return -(adv.to(logp.device) * mlogp).mean() + beta * mk3.mean(), float(mk3.mean())


def group_advantages(rewards, group_size: int) -> torch.Tensor:
    r = torch.as_tensor(rewards, dtype=torch.float32).view(-1, group_size)
    return (r - r.mean(1, keepdim=True)).view(-1)
