"""Edit actions on stored bf16 weights: families, compositions, realisation, restore and theta-hash (D4, D8).

Conventions.

* **Intended delta.** Every action computes an fp32 intended delta from the *current* tensor ``W`` (fp32 view of the
  stored bf16). Scaling ``W(s) = [1 + s(f - 1)] W``; row scaling the same on selected rows; projection removal
  ``-s alpha u u^T W`` (unit ``u`` on the output side); rank-one ``s alpha u v^T``; norm-matched Gaussian noise
  ``s rho ||W|| G / ||G||`` with a seeded ``G``; additive ``s D``. Ladder endpoints are built from W0 by applying a
  single action of cumulative dose ``s`` to the base (never by multiplying successive doses).
* **Realisation.** ``realise(W, delta) = bf16(fp32(W) + delta)`` on entries where ``delta != 0``; entries with a zero
  delta keep their stored bytes, so the zero action is a bitwise identity (also for ``-0.0``).
* **Restore.** bf16 add-then-subtract is inexact, so an :class:`EditSession` saves the original bytes of each tensor
  the first time it is touched and ``restore`` copies them back.
* **Theta-hash.** ``sha256(base_hash, sorted (name, tensor_hash) of every touched tensor whose bytes differ from the
  saved original)``. It equals the base hash exactly when the state is the base, and does not depend on tensor order.
* **Frozen tensors.** Vision-tower (``model.visual.*``) and MTP (``mtp.*``) tensors are never edited.
* **Role map.** Layer count, widths, layer types and writer names come from the pinned config, never constants.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Iterable, MutableMapping, Sequence

import torch

FROZEN_PATTERNS = (re.compile(r"(^|\.)visual\."), re.compile(r"(^|\.)mtp\."))
MIXER_OUT = {"full_attention": "self_attn.o_proj", "linear_attention": "linear_attn.out_proj"}


class FrozenTensorError(PermissionError):
    pass


def is_frozen(name: str) -> bool:
    return any(p.search(name) for p in FROZEN_PATTERNS)


def tensor_hash(t: torch.Tensor) -> str:
    x = t.detach().contiguous().cpu()
    h = hashlib.sha256(f"{x.dtype}|{tuple(x.shape)}|".encode())
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)
    )


def state_hash(state) -> str:
    """Full content hash of a (small) state dict; names sorted, so independent of order."""
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode() + b"\0" + tensor_hash(state[name]).encode())
    return h.hexdigest()


def theta_hash(base_hash: str, changed: dict[str, str]) -> str:
    if not changed:
        return base_hash
    payload = json.dumps(
        {"base": base_hash, "changed": sorted(changed.items())}, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ------------------------------------------------------------------ role map


@dataclasses.dataclass(frozen=True)
class RoleMap:
    n_layers: int
    hidden: int
    ffn: int
    vocab: int
    layer_types: tuple[str, ...]
    tied: bool
    prefix: str = "model.layers."

    @classmethod
    def from_config(cls, config: dict, prefix: str = "model.layers.") -> RoleMap:
        text = config.get("text_config", config)
        n = int(text["num_hidden_layers"])
        types = tuple(text.get("layer_types") or ["full_attention"] * n)
        if len(types) != n:
            raise ValueError(f"layer_types has {len(types)} entries for {n} layers")
        unknown = set(types) - set(MIXER_OUT)
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        tied = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
        return cls(
            n,
            int(text["hidden_size"]),
            int(text["intermediate_size"]),
            int(text["vocab_size"]),
            types,
            tied,
            prefix,
        )

    def name(self, layer: int, role: str) -> str:
        if not 0 <= layer < self.n_layers:
            raise KeyError(f"layer {layer} outside 0..{self.n_layers - 1}")
        if role == "o":
            path = MIXER_OUT[self.layer_types[layer]]
        elif role in ("down", "up", "gate"):
            path = f"mlp.{role}_proj"
        elif role in ("q", "k", "v") and self.layer_types[layer] == "full_attention":
            path = f"self_attn.{role}_proj"
        elif role in ("lin_qkv", "lin_z") and self.layer_types[layer] == "linear_attention":
            path = "linear_attn.in_proj_" + role[4:]
        else:
            raise KeyError(f"no role {role!r} in layer {layer} ({self.layer_types[layer]})")
        return f"{self.prefix}{layer}.{path}.weight"

    def writers(self) -> dict[tuple[int, str], str]:
        """Residual-stream writers: the token-mixer output projection and the MLP down projection."""
        return {(i, r): self.name(i, r) for i in range(self.n_layers) for r in ("o", "down")}

    def layer_of(self, name: str) -> int | None:
        m = re.search(r"\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m and not is_frozen(name) else None


# ------------------------------------------------------------------ actions


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = v.detach().float().reshape(-1)
    n = v.norm()
    if not torch.isfinite(n) or n == 0:
        raise ValueError("direction must be finite and nonzero")
    return v / n


@dataclasses.dataclass(frozen=True, eq=False)
class Action:
    family: str
    tensor: str
    dose: float
    params: tuple  # (key, value) pairs; tensors allowed
    _fn: object = dataclasses.field(repr=False, compare=False, default=None)

    def param(self, key):
        return dict(self.params)[key]

    def with_dose(self, dose: float) -> Action:
        return dataclasses.replace(self, dose=float(dose))

    def intended_delta(self, w: torch.Tensor) -> torch.Tensor:
        w = w.detach().float()
        if self.dose == 0.0:
            return torch.zeros_like(w)
        return self._fn(self, w)

    def key(self) -> str:
        h = hashlib.sha256(f"{self.family}|{self.tensor}|{self.dose!r}|".encode())
        for k, v in self.params:
            h.update(k.encode() + b"=")
            h.update(tensor_hash(v).encode() if isinstance(v, torch.Tensor) else repr(v).encode())
            h.update(b";")
        return h.hexdigest()

    def describe(self) -> dict:
        out = {"family": self.family, "tensor": self.tensor, "dose": self.dose}
        for k, v in self.params:
            out[k] = (
                {"sha256": tensor_hash(v)[:16], "shape": list(v.shape)}
                if isinstance(v, torch.Tensor)
                else v
            )
        return out


def _scale(a, w):
    return a.dose * (a.param("factor") - 1.0) * w


def _row_scale(a, w):
    out = torch.zeros_like(w)
    rows = list(a.param("rows"))
    out[rows] = a.dose * (a.param("factor") - 1.0) * w[rows]
    return out


def _proj_remove(a, w):
    u = _unit(a.param("direction")).to(w.device)
    return -a.dose * a.param("alpha") * torch.outer(u, u @ w)


def _rank_one(a, w):
    u = a.param("u").float().to(w.device)
    v = a.param("v").float().to(w.device)
    return a.dose * a.param("alpha") * torch.outer(u, v)


def _noise(a, w):
    gen = torch.Generator(device="cpu").manual_seed(int(a.param("seed")))
    g = torch.randn(w.shape, generator=gen).to(w.device)
    return a.dose * a.param("rho") * w.norm() * g / g.norm()


def _additive(a, w):
    d = a.param("delta").float().to(w.device)
    if d.shape != w.shape:
        raise ValueError("additive delta shape mismatch")
    return a.dose * d


def scale(tensor: str, factor: float, dose: float) -> Action:
    return Action("scale", tensor, float(dose), (("factor", float(factor)),), _scale)


def row_scale(tensor: str, rows: Sequence[int], factor: float, dose: float) -> Action:
    return Action(
        "row_scale",
        tensor,
        float(dose),
        (("rows", tuple(int(r) for r in rows)), ("factor", float(factor))),
        _row_scale,
    )


def proj_remove(tensor: str, direction: torch.Tensor, alpha: float, dose: float) -> Action:
    return Action(
        "proj_remove",
        tensor,
        float(dose),
        (("direction", direction.detach().float().cpu()), ("alpha", float(alpha))),
        _proj_remove,
    )


def rank_one(tensor: str, u: torch.Tensor, v: torch.Tensor, alpha: float, dose: float) -> Action:
    return Action(
        "rank_one",
        tensor,
        float(dose),
        (("u", u.detach().float().cpu()), ("v", v.detach().float().cpu()), ("alpha", float(alpha))),
        _rank_one,
    )


def noise(tensor: str, seed: int, rho: float, dose: float) -> Action:
    return Action("noise", tensor, float(dose), (("seed", int(seed)), ("rho", float(rho))), _noise)


def additive(tensor: str, delta: torch.Tensor, dose: float) -> Action:
    return Action(
        "additive", tensor, float(dose), (("delta", delta.detach().float().cpu()),), _additive
    )


# ------------------------------------------------------------------ realisation


def realise(w_bf16: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """bf16(fp32(W) + delta) where delta != 0; stored bytes elsewhere (bitwise identity at zero)."""
    if w_bf16.dtype != torch.bfloat16:
        raise TypeError("stored weights must be bf16")
    delta = delta.to(device=w_bf16.device, dtype=torch.float32)
    new = (w_bf16.float() + delta).to(torch.bfloat16)
    return torch.where(delta != 0, new, w_bf16)


def realised_record(w_bf16: torch.Tensor, delta: torch.Tensor) -> dict:
    new = realise(w_bf16, delta)
    d = new.float() - w_bf16.float()
    nz = delta != 0
    lost = (nz & (d == 0)).sum().item()
    return {
        "intended_norm": float(delta.float().norm()),
        "realised_norm": float(d.norm()),
        "rounding_loss": float((d - delta.float().to(d.device)).norm()),
        "rounded_away_frac": lost / nz.sum().item() if nz.any() else 0.0,
    }


@dataclasses.dataclass
class StepRecord:
    action: Action
    position: int
    realised: torch.Tensor  # fp32 realised increment D_t (same device as the weights)
    intended_norm: float
    realised_norm: float
    rounding_loss: float
    rounded_away_frac: float


class EditSession:
    """Applies actions in place to a live mapping of bf16 tensors (a state dict or ``named_parameters``).

    Saves original bytes on first touch (on CPU), so ``restore`` returns the exact base bytes.
    """

    def __init__(self, params: MutableMapping[str, torch.Tensor], base_hash: str, rolemap: RoleMap):
        self.params = params
        self.base_hash = base_hash
        self.rolemap = rolemap
        self._orig: dict[str, torch.Tensor] = {}
        self.records: list[StepRecord] = []

    @property
    def touched(self) -> tuple[str, ...]:
        return tuple(sorted(self._orig))

    def original(self, name: str) -> torch.Tensor:
        return self._orig[name]

    def _tensor(self, name: str) -> torch.Tensor:
        if is_frozen(name):
            raise FrozenTensorError(f"{name}: vision and MTP tensors are frozen")
        if name not in self.params:
            raise KeyError(f"unknown tensor {name}")
        return self.params[name]

    def apply(self, action: Action) -> StepRecord:
        live = self._tensor(action.tensor)
        if action.tensor not in self._orig:
            self._orig[action.tensor] = live.detach().to("cpu", copy=True)
        w = live.detach()
        delta = action.intended_delta(w)
        rec = realised_record(w, delta)
        new = realise(w, delta)
        realised = new.float() - w.float()
        with torch.no_grad():
            live.copy_(new)
        step = StepRecord(action, len(self.records), realised, **rec)
        self.records.append(step)
        return step

    def apply_bundle(self, actions: Iterable[Action]) -> list[StepRecord]:
        """One step touching several distinct tensors; the result does not depend on their order."""
        actions = list(actions)
        if len({a.tensor for a in actions}) != len(actions):
            raise ValueError("a bundle edits each tensor at most once")
        return [self.apply(a) for a in sorted(actions, key=lambda a: a.tensor)]

    def compose(self, actions: Sequence[Action]) -> list[StepRecord]:
        """Sequential composition: each action sees the realised result of the previous ones."""
        return [self.apply(a) for a in actions]

    def displacement(self, name: str) -> torch.Tensor:
        if name not in self._orig:
            return torch.zeros_like(self.params[name], dtype=torch.float32)
        return (
            self.params[name].detach().float()
            - self._orig[name].to(self.params[name].device).float()
        )

    def changed(self) -> dict[str, str]:
        out = {}
        for name, orig in self._orig.items():
            cur = self.params[name].detach().cpu()
            if not same_bytes(cur, orig):
                out[name] = tensor_hash(cur)
        return out

    def theta_hash(self) -> str:
        return theta_hash(self.base_hash, self.changed())

    def restore(self) -> None:
        with torch.no_grad():
            for name, orig in self._orig.items():
                self.params[name].copy_(orig.to(self.params[name].device))
        for name, orig in self._orig.items():
            if not same_bytes(self.params[name].detach().cpu(), orig):
                raise RuntimeError(f"{name}: restore did not return the saved bytes")
        self._orig.clear()
        self.records.clear()


def step_tokens(records: Sequence[StepRecord], session: EditSession) -> list[dict]:
    """Exact bookkeeping of an ordered composition: action tokens, realised norms and signed overlaps.

    ``overlap_prev_E`` is cos(D_t, E_{t-1}) on the same tensor (E_{t-1} = sum of earlier realised increments),
    0.0 when there is no earlier increment on that tensor; ``overlap_prev_D`` is cos(D_t, D_{t-1}) likewise.
    """
    out, cum, last = [], {}, {}
    for r in records:
        name = r.action.tensor
        d = r.realised.float()

        def cos(a, b):
            na, nb = a.norm().item(), b.norm().item()
            return float((a * b).sum().item() / (na * nb)) if na > 0 and nb > 0 else 0.0

        e_prev = cum.get(name)
        out.append(
            {
                "position": r.position,
                "family": r.action.family,
                "tensor": name,
                "layer": session.rolemap.layer_of(name),
                "dose": r.action.dose,
                "intended_norm": r.intended_norm,
                "realised_norm": r.realised_norm,
                "realised_over_intended": r.realised_norm / r.intended_norm
                if r.intended_norm > 0
                else 0.0,
                "rounding_loss": r.rounding_loss,
                "overlap_prev_E": cos(d, e_prev) if e_prev is not None else 0.0,
                "overlap_prev_D": cos(d, last[name]) if name in last else 0.0,
                "action_key": r.action.key(),
            }
        )
        cum[name] = d if e_prev is None else e_prev + d
        last[name] = d
    return out


def relative_norm(action: Action, w: torch.Tensor) -> float:
    d = action.intended_delta(w)
    n = float(w.float().norm())
    return float(d.norm()) / n if n > 0 else math.inf
