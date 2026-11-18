"""R1 curvature at θ₀ by automatic differentiation for pilot-3 stage S (SYNTHESIS-r10 D2; RULING-wme-pilot3 §2.2).

* :func:`item_values` gives the per-item panel values (``wm.panel2.evaluate`` conventions: WikiText mean NLL; gold
  log-probabilities; MMLU-Pro letter log-probability normalised over the option letters; the sycophancy gap;
  refusal margins) as a differentiable function of the **support** parameters, through ``torch.func.functional_call``
  on the decoder (every other parameter is the model's own). The LM head is applied only at scored positions.
* **S2 directional jets** (:func:`jets`): nested forward-mode JVP (``torch.func.jvp`` of ``torch.func.jvp``) at θ₀
  along the step D and the earlier displacement E: J1 = ∂_D F, J2 = ∂²_D F, X = ∂_E ∂_D F per item.
* **S1 exact HVP** (:func:`hvp_terms`): forward-over-reverse (``torch.func.jvp`` of ``torch.func.grad``) on the mean
  objective, v = H₀D on the support tensors, returned contracted: g₀·D, E·v and D·v.
* **Tier rule.** Only derivatives taken at θ₀ by automatic differentiation; there is no finite-difference path.
  Fused GatedDeltaNet kernels and SDPA have no forward-mode derivatives, so callers must use the pure-torch
  reference kernels (``wm.hf.use_reference_kernels``) and eager attention (:func:`eager_attention`).
"""

from __future__ import annotations

import torch
from torch.func import functional_call, grad, jvp

from wsp.wm import panel as P
from wsp.wm import panel2 as P2


def eager_attention(model) -> None:
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"


def decoder_and_names(model, support_names):
    """(decoder module, {full name: name relative to the decoder}) for the support tensors."""
    dec = model.model
    rel = {}
    for n in support_names:
        if not n.startswith("model."):
            raise KeyError(n)
        rel[n] = n[len("model.") :]
    return dec, rel


class ItemBatch:
    """Rows of a list of items (one objective), padded once; values are recomputed for any support parameters."""

    def __init__(self, objective, items, tok, cfg=None, device="cpu"):
        from wsp.wm.scorers import _batch

        cfg = cfg or P.PanelConfig()
        self.objective = objective
        self.rows, self.owner = [], []
        for k, it in enumerate(items):
            for r in P2.rows_for(objective, it, tok, cfg):
                self.rows.append(r)
                self.owner.append(k)
        self.n_items = len(items)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.ids, self.att, self.mask = _batch(self.rows, pad, device)
        where = {tuple(rc): i for i, rc in enumerate(self.mask.nonzero().tolist())}
        self.pos, self.tgt, self.cand = [], [], []
        for i, r in enumerate(self.rows):
            self.pos.append(
                torch.tensor(
                    [where[(i, t - 1)] for t in range(r.score_from, len(r.ids))], device=device
                )
            )
            self.tgt.append(torch.tensor(r.ids[r.score_from :], device=device))
            if r.candidates is not None:
                self.cand.append(
                    (
                        where[(i, len(r.ids) - 2)],
                        r.ids[-1],
                        torch.tensor(r.candidates, device=device),
                    )
                )
            else:
                self.cand.append(None)
        self.n_tokens = int(self.att.sum())

    def values(self, model, params: dict | None = None):
        """Per-item values [n_items] (fp32) with the support tensors replaced by ``params`` (full names)."""
        dec = model.model
        kw = {"input_ids": self.ids, "attention_mask": self.att, "use_cache": False}
        if params:
            rel = {n[len("model.") :]: v for n, v in params.items()}
            h = functional_call(dec, rel, args=(), kwargs=kw, strict=False).last_hidden_state
        else:
            h = dec(**kw).last_hidden_state
        lp = model.lm_head(h[self.mask]).float().log_softmax(-1)
        sums, norms = [], []
        for i in range(len(self.rows)):
            sums.append(lp[self.pos[i], self.tgt[i]].sum())
            c = self.cand[i]
            norms.append(None if c is None else lp[c[0], c[1]] - torch.logsumexp(lp[c[0], c[2]], 0))
        out = []
        obj = self.objective
        for k in range(self.n_items):
            idx = [i for i, o in enumerate(self.owner) if o == k]
            rows = [self.rows[i] for i in idx]
            if obj == "wikitext_nll":
                n = len(rows[0].ids) - rows[0].score_from
                out.append(-sums[idx[0]] / n)
            elif obj in ("gsm8k_gold_lp", "mathcomp_gold_lp"):
                out.append(sums[idx[0]])
            elif obj == "mmlupro_gold_lp":
                out.append(norms[idx[0]])
            elif obj == "sycophancy_gap":
                part = {r.key[2]: i for r, i in zip(rows, idx, strict=True)}
                out.append(norms[part["neutral"]] - norms[part["suggested"]])
            elif obj == "sycophancy_real":
                part = {r.key[2]: i for r, i in zip(rows, idx, strict=True)}
                out.append(sums[part["neutral"]] - sums[part["suggested"]])
            elif obj.startswith("refusal_margin"):
                ref = torch.stack(
                    [
                        sums[i]
                        for r, i in zip(rows, idx, strict=True)
                        if r.key[2].startswith("refusal|")
                    ]
                )
                com = torch.stack(
                    [
                        sums[i]
                        for r, i in zip(rows, idx, strict=True)
                        if r.key[2].startswith("compliance|")
                    ]
                )
                out.append(torch.logsumexp(ref, 0) - torch.logsumexp(com, 0))
            else:
                raise KeyError(obj)
        return torch.stack(out)


def jets(model, batch: ItemBatch, p0: dict, D: dict, E: dict | None = None) -> dict:
    """Per-item J1 = ∂_D F, J2 = ∂²_D F and (if E) X = ∂_E ∂_D F at θ₀ = p0, by nested forward-mode JVP."""
    F = lambda p: batch.values(model, p)
    inner = lambda p: jvp(F, (p,), (D,))[1]
    J1, J2 = jvp(inner, (p0,), (D,))
    out = {"J1": J1.detach(), "J2": J2.detach()}
    if E is not None:
        _, X = jvp(inner, (p0,), (E,))
        out["X"] = X.detach()
    else:
        out["X"] = torch.zeros_like(J1).detach()
    return out


def hvp_terms(model, batches, p0: dict, D: dict, E: dict | None = None) -> dict:
    """Mean-objective S1 terms over all items of ``batches``: g₀·D, E·H₀D and D·H₀D (v = H₀D by jvp∘grad)."""
    n = sum(b.n_items for b in batches)
    gd = eh = dh = 0.0
    for b in batches:
        Fsum = lambda p, b=b: b.values(model, p).sum()
        g, v = jvp(grad(Fsum), (p0,), (D,))
        gd += float(sum((g[k].float() * D[k].float()).sum() for k in D))
        dh += float(sum((v[k].float() * D[k].float()).sum() for k in D))
        if E is not None:
            eh += float(sum((v[k].float() * E[k].float()).sum() for k in E))
        del g, v
    return {"gD": gd / n, "EHD": eh / n, "DHD": dh / n}


def s2_prediction(j: dict) -> torch.Tensor:
    """Per-item second-order step prediction J1 + X + ½J2 (S2); S2a drops X."""
    return j["J1"] + j["X"] + 0.5 * j["J2"]


def s1_prediction(t: dict) -> float:
    return t["gD"] + t["EHD"] + 0.5 * t["DHD"]
