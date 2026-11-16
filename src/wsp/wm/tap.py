"""Observation tap: seeded residual sketches and per-item losses recorded during a real evaluation (D3, D8).

Forward hooks on decoder layers read the layer output (the residual stream after the layer), take the masked mean
over scored positions per row, and multiply by a seeded Gaussian projection ``P_l`` (hidden x k, scaled 1/sqrt(k)).
Hooks return ``None`` and only read detached fp32 copies, so evaluation outputs are bitwise unchanged. Observations
are tagged with the theta-hash of the evaluated state: tier R0 at the base, R2 otherwise (``wm.access``).
"""

from __future__ import annotations

import contextlib
import math
import zlib

import torch

from wsp.wm.access import observation


def sketch_projection(hidden: int, k: int, seed: int, layer: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(
        int(seed) * 1_000_003 + zlib.crc32(f"layer{layer}".encode())
    )
    return torch.randn(hidden, k, generator=gen) / math.sqrt(k)


class ResidualTap:
    def __init__(self, layers: dict[int, torch.nn.Module], hidden: int, k: int, seed: int):
        self.layers = dict(sorted(layers.items()))
        self.hidden, self.k, self.seed = hidden, k, seed
        self.proj = {i: sketch_projection(hidden, k, seed, i) for i in self.layers}
        self._handles = []
        self._sk: dict[str, dict[int, torch.Tensor]] = {}
        self._losses: dict[str, float] = {}
        self._items = None
        self._mask = None

    def active_handles(self):
        return list(self._handles)

    def _hook(self, layer):
        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.detach().float()
            mask = self._mask.to(h.device)
            if tuple(mask.shape) != tuple(h.shape[:2]):
                raise ValueError("tap mask does not match the layer output layout")
            m = mask.float()[..., None]
            mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            sk = (mean @ self.proj[layer].to(h.device)).cpu()
            for row, item in enumerate(self._items):
                if item is not None:
                    self._sk.setdefault(item, {})[layer] = sk[row]

        return fn

    @contextlib.contextmanager
    def recording(self, item_ids, mask: torch.Tensor):
        """Record sketches for one batch; ``item_ids[row]`` (None skips a row), ``mask`` [batch, seq] scored."""
        self._items, self._mask = list(item_ids), mask.detach().bool()
        try:
            for i, mod in self.layers.items():
                self._handles.append(mod.register_forward_hook(self._hook(i)))
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
            self._items = self._mask = None

    def record_losses(self, losses: dict[str, float]):
        self._losses.update({k: float(v) for k, v in losses.items()})

    def sketches(self) -> dict[str, torch.Tensor]:
        return {
            item: torch.stack([layers[i] for i in self.layers]) for item, layers in self._sk.items()
        }

    def observation(self, theta_hash: str, base_hash: str):
        return observation(
            theta_hash,
            base_hash,
            "tap",
            {
                "sketches": self.sketches(),
                "item_losses": dict(self._losses),
                "layers": list(self.layers),
                "k": self.k,
                "seed": self.seed,
            },
        )

    def reset(self):
        self._sk.clear()
        self._losses.clear()
