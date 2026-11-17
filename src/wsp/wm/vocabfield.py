"""Field coverage for vocabulary-row roots (PREREG_WME §4.1 row_scale; RULING-wme-resources §6; confirm-eng §6).

``wm.collect`` gives vocabulary-sized tensors (lm_head, embed_tokens) gradients only and never Fisher hooks, so the
row_scale roots of the bank have no curvature nest. This module adds one, under a **declared** approximation that
never forms a vocabulary-sized covariance, shared by every arm (:data:`SHARED_FIELD_SPEC`):

* **Declared rows** (:func:`declare_rows`): the union of the row sets of the vocabulary-row roots (and any inert
  dead-row controls), per tensor, with an identity hash. |R| must be < vocabulary and <= :data:`MAX_DECLARED_ROWS`.
* **lm_head, ``projected_kfac``**: A = E[x xᵀ] over valid positions (hidden x hidden) and G_R = E[δ_R δ_Rᵀ], where
  δ_R is the logit gradient restricted to the declared rows (|R| x |R|). G_R is exactly the R-block of the K-FAC G,
  so for an edit supported on R, q = ½ tr(G_R D_R A D_Rᵀ) equals the full K-FAC q. Raw factors (no spectral
  truncation; the R-block is small and kept at full rank).
* **embed_tokens, ``row_diagonal``**: the per-token Fisher diagonal of each declared row, D[r, i] =
  Σ_{p: t_p = r} δ_{p,i}² / n (δ = gradient at the embedding output, every input position; n = valid positions, the
  K-FAC count). For one-hot inputs the K-FAC A is exactly diagonal; the declared approximation is the diagonal in
  the hidden coordinates. q = ½ Σ D ⊙ d².
* **Same estimator as the support collector**: one per-item gradient pass (score_sum / Σ weight), then one
  model-sampled Fisher pass per item with the generator seeded ``response_field.SEED`` per objective; fp32 sums.
* **Unsupported is marked, never zero** (:meth:`VocabField.contract`): an edit with any nonzero row outside the
  declared rows, or a tensor that is not a vocabulary-row tensor, gets ``signed = q = None``, ``supported = False``
  and a reason.
* Tied embeddings are refused (:class:`TiedEmbeddingError`): an embedding row would also be an LM-head row.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from wsp.features import response_field as rf
from wsp.wm import collect as C

SCHEMA = "wme-vocab-row-field-v1"
TENSORS = ("lm_head", "embed")
MAX_DECLARED_ROWS = 4096
APPROXIMATIONS = {
    "lm_head": {
        "name": "projected_kfac",
        "rule": "A = E[x x^T] (hidden^2); G_R = E[delta_R delta_R^T] over the declared rows R (|R|^2), the R-block "
        "of K-FAC G; q = 1/2 tr(G_R D_R A D_R^T); exact K-FAC for edits supported on R; raw factors",
    },
    "embed": {
        "name": "row_diagonal",
        "rule": "D[r, i] = sum_{p: t_p = r} delta_{p,i}^2 / n over every input position (n = valid positions); "
        "q = 1/2 sum D * d^2; one-hot K-FAC A is exactly diagonal, hidden coordinates diagonal by declaration",
    },
}
# Declared rows for the 9B bank (row sets are realised from construction token frequencies at bank build; the
# identity hash of DeclaredRows is recorded then): 8 lm_head and 8 embed_tokens row_scale roots x 32 rows, plus the
# inert dead embedding rows (3 controls x 32 rows >= len(tokenizer)), whose field is an exact 0 (never indexed).
DECLARED_ROWS_RULE = (
    "lm_head: union of the row sets of the lm_head row_scale bank roots; embed: union of the row sets of the "
    "embed_tokens row_scale bank roots plus the inert dead rows (controls.INERT_SPEC['dead_rows']); recorded as the "
    "DeclaredRows identity hash at bank build"
)
PLANNED_ROWS = {"lm_head": 8 * 32, "embed": 8 * 32 + 3 * 32}
SHARED_FIELD_SPEC = (
    f"{C.SCHEMA} (support writers, K-FAC, wm.collect) + {SCHEMA} (vocabulary rows: lm_head "
    f"{APPROXIMATIONS['lm_head']['name']}, embed_tokens {APPROXIMATIONS['embed']['name']}); unsupported "
    "tensors are marked, never 0; one field, shared by every arm"
)


class VocabularySized(ValueError):
    """A declared row set would make a vocabulary-sized covariance."""


class TiedEmbeddingError(ValueError):
    pass


def _h(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


class DeclaredRows:
    def __init__(self, by_tensor: dict, vocab: int):
        self.vocab = int(vocab)
        self.sources = {
            t: {k: sorted(set(v)) for k, v in by_tensor.get(t, {}).items()} for t in TENSORS
        }
        self._rows = {
            t: tuple(sorted({r for v in self.sources[t].values() for r in v})) for t in TENSORS
        }

    def rows(self, tensor) -> tuple:
        return self._rows[tensor]

    def identity(self) -> dict:
        d = {
            "vocab": self.vocab,
            "rows": {t: list(self._rows[t]) for t in TENSORS},
            "sources": self.sources,
        }
        return dict(d, hash=_h(d))


def declare_rows(by_tensor: dict, vocab: int) -> DeclaredRows:
    """``by_tensor = {"lm_head": {root_id: rows}, "embed": {root_id: rows}}``."""
    unknown = set(by_tensor) - set(TENSORS)
    if unknown:
        raise ValueError(f"unknown vocabulary tensors {sorted(unknown)}")
    d = DeclaredRows(by_tensor, vocab)
    for t in TENSORS:
        rows = d.rows(t)
        if any(not 0 <= r < vocab for r in rows):
            raise ValueError(f"{t}: row outside [0, {vocab})")
        if len(rows) >= vocab or len(rows) > MAX_DECLARED_ROWS:
            raise VocabularySized(f"{t}: {len(rows)} declared rows (vocabulary {vocab})")
    return d


def _modules(model, lm_head, embed):
    mods = dict(model.named_modules())
    out_m = mods[lm_head] if lm_head else model.get_output_embeddings()
    in_m = mods[embed] if embed else model.get_input_embeddings()
    if not isinstance(out_m, nn.Linear) or not isinstance(in_m, nn.Embedding):
        raise TypeError("lm_head must be an nn.Linear and embed an nn.Embedding")
    if out_m.weight is in_m.weight or out_m.weight.data_ptr() == in_m.weight.data_ptr():
        raise TiedEmbeddingError("tied embeddings: an embedding row is also an LM-head row")
    if out_m.bias is not None:
        raise ValueError("lm_head with bias is not supported")
    return out_m, in_m


def _objective_pass(model, tokenizer, obj, panel, out_m, in_m, R_lm, R_emb, device):
    dev_lm = torch.tensor(R_lm, dtype=torch.long)
    dev_emb = torch.tensor(R_emb, dtype=torch.long)
    lut = torch.full((in_m.num_embeddings,), -1, dtype=torch.long)
    lut[dev_emb] = torch.arange(len(R_emb))
    hidden = out_m.in_features
    g_lm = torch.zeros(len(R_lm), hidden, dtype=torch.float32)
    g_emb = torch.zeros(len(R_emb), in_m.embedding_dim, dtype=torch.float32)
    A = torch.zeros(hidden, hidden, dtype=torch.float32)
    G = torch.zeros(len(R_lm), len(R_lm), dtype=torch.float32)
    Dg = torch.zeros(len(R_emb), in_m.embedding_dim, dtype=torch.float32)
    state = {"fisher": False, "mask": None, "n": 0}

    def _rows(t, mask):
        if tuple(t.shape[:-1]) == tuple(mask.shape):
            return t[mask]
        if t.dim() == 2 and t.shape[0] == int(mask.sum()):
            return t
        raise rf.MissingHookError("lm_head: valid-position mask/layout mismatch")

    def lm_fwd(module, inputs, output):
        if not state["fisher"]:
            return
        x = inputs[0].detach().float()

        def bwd(dy):
            mask = state["mask"]
            xv = _rows(x, mask).reshape(-1, x.shape[-1])
            dv = _rows(dy, mask).reshape(-1, dy.shape[-1])
            dR = dv.index_select(
                -1, dev_lm.to(dv.device)
            ).float()  # restrict BEFORE any outer product
            A.add_((xv.T @ xv).cpu())
            G.add_((dR.T @ dR).cpu())
            state["n"] += len(xv)

        output.register_hook(bwd)

    def emb_fwd(module, inputs, output):
        if not state["fisher"]:
            return
        ids = inputs[0].detach()

        def bwd(de):
            idx = lut.to(ids.device)[ids]
            sel = idx >= 0
            if sel.any():
                sq = de[sel].float() ** 2
                Dg.index_add_(0, idx[sel].cpu(), sq.cpu())

        output.register_hook(bwd)

    handles = [out_m.register_forward_hook(lm_fwd), in_m.register_forward_hook(emb_fwd)]
    generator = torch.Generator(device="cpu").manual_seed(rf.SEED)
    total, item_hashes = 0.0, []
    try:
        for item in panel.items():
            item_hashes.append(rf._hash(rf._hash_item(item)))
            model.zero_grad(set_to_none=True)
            batch = obj.scorer(model, tokenizer, item, device)
            if (
                not math.isfinite(batch.weight)
                or batch.weight <= 0
                or not torch.isfinite(batch.score_sum)
            ):
                raise ValueError("nonfinite score or nonpositive objective weight")
            batch.score_sum.backward()
            total += batch.weight
            g_lm.add_(
                out_m.weight.grad.index_select(0, dev_lm.to(out_m.weight.device)).float().cpu()
            )
            g_emb.add_(
                in_m.weight.grad.index_select(0, dev_emb.to(in_m.weight.device)).float().cpu()
            )
            del batch
            model.zero_grad(set_to_none=True)
            state["fisher"] = True
            batch = obj.scorer(model, tokenizer, item, device)
            mask = batch.valid_mask.to(device=device, dtype=torch.bool)
            state["mask"] = mask
            logits = batch.logits.float()
            if logits.dim() == mask.dim() + 1 and tuple(logits.shape[:-1]) == tuple(mask.shape):
                logits = logits[mask]
            elif not (logits.dim() == 2 and logits.shape[0] == int(mask.sum())):
                raise ValueError("logits must be [*mask.shape, vocab] or gathered [n_valid, vocab]")
            if not len(logits):
                raise ValueError("empty valid-token mask")
            targets = torch.multinomial(logits.detach().softmax(-1).cpu(), 1, generator=generator)
            F.cross_entropy(logits, targets[:, 0].to(logits.device), reduction="sum").backward()
            del batch, logits, targets
            state["fisher"] = False
            model.zero_grad(set_to_none=True)
        if not total or not state["n"]:
            raise ValueError("empty objective panel")
        n = state["n"]
        payload = {
            "gradients": {"lm_head": g_lm / total, "embed": g_emb / total},
            "factors": {
                "lm_head": {"A": A / n, "G_R": G / n, "count": n},
                "embed": {"D_R": Dg / n, "count": n},
            },
        }
        for k, v in payload["gradients"].items():
            if not torch.isfinite(v).all():
                raise ValueError(f"nonfinite gradient: {k}")
        return payload, rf._hash(item_hashes)
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)


def collect_vocab_rows(
    model,
    tokenizer,
    objectives,
    panels,
    declared: DeclaredRows,
    device,
    out_dir,
    lm_head: str | None = None,
    embed: str | None = None,
    theta_hash: str | None = None,
    log=None,
):
    """Gradients and declared-approximation curvature for the declared vocabulary rows; returns a VocabField."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_m, in_m = _modules(model, lm_head, embed)
    if out_m.out_features != declared.vocab or in_m.num_embeddings != declared.vocab:
        raise ValueError("declared vocabulary does not match the model")
    params = dict(model.named_parameters())
    names = {id(p): n for n, p in params.items()}
    if theta_hash is None:
        from wsp.wm.actions import state_hash

        theta_hash = state_hash(params)
    R_lm, R_emb = list(declared.rows("lm_head")), list(declared.rows("embed"))
    identity = {
        "schema": SCHEMA,
        "seed": rf.SEED,
        "theta_hash": theta_hash,
        "tensors": {"lm_head": names[id(out_m.weight)], "embed": names[id(in_m.weight)]},
        "declared_rows": declared.identity(),
        "approximations": APPROXIMATIONS,
        "objectives": [o.identity() for o in objectives],
        "panels": {},
        "timing": {},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    flags = {n: p.requires_grad for n, p in params.items()}
    training = {m: m.training for m in model.modules()}
    model.eval()
    files = {}
    try:
        for p in params.values():
            p.requires_grad_(False)
        out_m.weight.requires_grad_(True)
        in_m.weight.requires_grad_(True)
        for obj in objectives:
            panel = panels[obj.panel_id]
            panel.validate()
            if panel.panel_id != obj.panel_id or panel.text_group_hash != obj.text_group_hash:
                raise ValueError("objective panel identity mismatch")
            t0 = time.perf_counter()
            payload, content = _objective_pass(
                model, tokenizer, obj, panel, out_m, in_m, R_lm, R_emb, device
            )
            fname = f"{obj.name}.vocab.pt"
            torch.save(payload, out / fname)
            files[fname] = hashlib.sha256((out / fname).read_bytes()).hexdigest()
            identity["panels"][obj.name] = {"panel_id": panel.panel_id, "content_hash": content}
            identity["timing"][obj.name] = {
                "seconds": time.perf_counter() - t0,
                "rss_peak_bytes": C._rss_peak_bytes(),
            }
            if log:
                log(f"vocab rows {obj.name}: {identity['timing'][obj.name]}")
    finally:
        for n, p in params.items():
            p.requires_grad_(flags[n])
        for m, mode in training.items():
            m.training = mode
    (out / "manifest.json").write_text(
        json.dumps({"identity": identity, "file_sha256": files}, indent=2, sort_keys=True)
    )
    return VocabField(out, identity)


class VocabField:
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        self.objectives = [o["name"] for o in identity["objectives"]]
        self._rows = {t: tuple(identity["declared_rows"]["rows"][t]) for t in TENSORS}
        self._names = dict(identity["tensors"])
        self._cache: dict = {}

    @classmethod
    def load(cls, root):
        root = Path(root)
        man = json.loads((root / "manifest.json").read_text())
        if man["identity"]["schema"] != SCHEMA:
            raise ValueError("vocabulary field schema mismatch")
        for f, h in man["file_sha256"].items():
            if hashlib.sha256((root / f).read_bytes()).hexdigest() != h:
                raise ValueError(f"{f}: content hash mismatch")
        return cls(root, man["identity"])

    def rows(self, tensor) -> tuple:
        return self._rows[tensor]

    def _file(self, obj):
        if obj not in self._cache:
            self._cache[obj] = torch.load(
                self.root / f"{obj}.vocab.pt", map_location="cpu", weights_only=True
            )
        return self._cache[obj]

    def gradient(self, obj, tensor):
        return self._file(obj)["gradients"][tensor]

    def factors(self, obj, tensor):
        return self._file(obj)["factors"][tensor]

    def _tensor_key(self, name):
        for t in TENSORS:
            if name in (t, self._names[t]):
                return t
        return None

    def _row_values(self, t, value):
        """(row ids, values [k, hidden]) of a dense or row-sparse delta; rows with all-zero values dropped."""
        if isinstance(value, dict):
            rows = [int(r) for r in value["rows"]]
            vals = value["values"].detach().float()
        else:
            d = value.detach().float()
            nz = (d != 0).any(-1).nonzero().flatten()
            rows, vals = nz.tolist(), d[nz]
        keep = [(r, i) for i, r in enumerate(rows) if bool((vals[i] != 0).any())]
        return [r for r, _ in keep], vals[[i for _, i in keep]] if keep else vals[:0]

    def contract(self, deltas: dict) -> dict:
        """Per objective and tensor: signed <g, D>, q under the declared approximation; unsupported -> None."""
        result = {}
        for obj in self.objectives:
            rows_out = result[obj] = {}
            for name, value in deltas.items():
                t = self._tensor_key(name)
                if t is None:
                    rows_out[name] = _unsupported("not_vocabulary_row_tensor")
                    continue
                rows, vals = self._row_values(t, value)
                pos = {r: i for i, r in enumerate(self._rows[t])}
                if any(r not in pos for r in rows):
                    rows_out[name] = _unsupported("rows_outside_declared_support")
                    continue
                idx = torch.tensor([pos[r] for r in rows], dtype=torch.long)
                dR = torch.zeros(len(self._rows[t]), vals.shape[-1]) if len(rows) else None
                if dR is None:
                    rows_out[name] = {
                        "signed": 0.0,
                        "q": 0.0,
                        "supported": True,
                        "reason": "zero_delta",
                        "approximation": APPROXIMATIONS[t]["name"],
                    }
                    continue
                dR[idx] = vals
                g = self.gradient(obj, t)
                signed = float((g * dR).sum())
                f = self.factors(obj, t)
                if t == "lm_head":
                    q = 0.5 * float(torch.trace(f["G_R"] @ dR @ f["A"] @ dR.T))
                else:
                    q = 0.5 * float((f["D_R"] * dR**2).sum())
                rows_out[name] = {
                    "signed": signed,
                    "q": max(0.0, q),
                    "supported": True,
                    "reason": None,
                    "approximation": APPROXIMATIONS[t]["name"],
                }
        return result


def _unsupported(reason):
    return {"signed": None, "q": None, "supported": False, "reason": reason, "approximation": None}
