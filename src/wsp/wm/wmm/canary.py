"""Canaries for study M′ jobs (SYNTHESIS-r14 R2(b)-(c)).

* **Weight sketch** (:func:`weight_sketch`): per tensor, 64 seeded Gaussian projections s_k = <W, R_k> (fp32, rows
  streamed in chunks; R_k drawn from a generator seeded by (seed, tensor name), so it depends only on the name and
  shape). Run at the start and end of every job on the same device. Any byte change moves the sketch;
  E[(s'_k − s_k)²] = ‖ΔW‖², so :func:`compare_sketches` also estimates the size of an accidental change.
* **Behavioural canary** (:func:`behavioural_canary`): greedy generation at **batch 1** over a fixed, ordered prompt
  list. With no batch partners it cannot fire on the ~13.5 % batch-composition floor measured at Gate 1.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib

import numpy as np
import torch

N_PROJ = 64
CHUNK_ELEMS = 1 << 22


def _gen(seed: int, name: str, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) * 1_000_003 + zlib.crc32(name.encode())) % (1 << 63))
    return g


@torch.no_grad()
def tensor_sketch(t: torch.Tensor, name: str, seed: int, n_proj: int = N_PROJ) -> np.ndarray:
    flat = t.detach().reshape(-1)
    g = _gen(seed, name, flat.device)
    out = torch.zeros(n_proj, dtype=torch.float64, device=flat.device)
    for c0 in range(0, flat.numel(), CHUNK_ELEMS):
        x = flat[c0 : c0 + CHUNK_ELEMS].float()
        R = torch.randn(n_proj, x.numel(), generator=g, device=flat.device)
        out += (R @ x).double()
        del R
    return out.cpu().numpy()


def weight_sketch(params, seed: int, n_proj: int = N_PROJ) -> dict:
    return {n: tensor_sketch(t, n, seed, n_proj) for n, t in sorted(params.items())}


def sketch_sha256(sk: dict) -> str:
    h = hashlib.sha256()
    for n in sorted(sk):
        h.update(n.encode() + b"\0" + np.ascontiguousarray(sk[n], dtype=np.float64).tobytes())
    return h.hexdigest()


def compare_sketches(start: dict, end: dict) -> dict:
    """Bitwise comparison per tensor; for changed tensors the JL estimate sqrt(mean_k (Δs_k)²) of ‖ΔW‖_F."""
    if set(start) != set(end):
        raise ValueError("sketches cover different tensors")
    changed, est = [], {}
    for n in sorted(start):
        d = np.asarray(end[n]) - np.asarray(start[n])
        if np.any(d != 0):
            changed.append(n)
            est[n] = float(math.sqrt(np.mean(d * d)))
    return {"pass": not changed, "changed": changed, "estimated_norm": est, "n_tensors": len(start)}


@torch.no_grad()
def behavioural_canary(model, prompt_ids, max_new_tokens: int, stop_ids) -> dict:
    """Greedy, batch 1, prompts in the given (fixed) order; sha256 of each prompt's generated ids and of the list."""
    stop = {int(s) for s in stop_ids}
    dev = next(model.parameters()).device
    per = []
    for ids in prompt_ids:
        x = torch.tensor([list(ids)], device=dev)
        g = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=sorted(stop),
            pad_token_id=min(stop),
        )
        row = g[0, x.shape[1] :].tolist()
        cut = next((k for k, t in enumerate(row) if t in stop), None)
        new = row if cut is None else row[: cut + 1]
        per.append(hashlib.sha256(json.dumps(new).encode()).hexdigest())
    return {
        "per_prompt": per,
        "sha256": hashlib.sha256(json.dumps(per).encode()).hexdigest(),
        "batch_size": 1,
        "n_prompts": len(per),
        "max_new_tokens": max_new_tokens,
    }
