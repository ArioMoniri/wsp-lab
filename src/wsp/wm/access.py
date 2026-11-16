"""Access tiers R0/R1/R2 (SYNTHESIS-r8 D2), enforced in code.

* **R0**: any artefact computed at theta0 independently of the candidate edit, plus CPU arithmetic on D. This is the
  only tier world-model inference and planning may read.
* **R1**: base passes whose inputs depend on D (JVP/HVP). A separately charged arm; never read by ``predict``/``plan``.
* **R2**: any pass at theta != theta0 (re-linearisation, edited-model activations, evaluation): running the edit.
  Allowed only as training targets, a charged oracle comparator, or the observation tap on states evaluated for real.

Every artefact carries the theta-hash of the weights it was computed at. ``R0Guard.check`` walks arbitrary nested
arguments and raises :class:`AccessError` on any artefact that is not tier R0 at the base hash; ``r0_only`` applies
it to every argument of a method whose owner has a ``guard``.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from enum import Enum
from typing import Any


class Tier(Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"


class AccessError(PermissionError):
    """A non-R0 artefact reached world-model inference or planning."""


@dataclasses.dataclass(frozen=True, eq=False)
class Artefact:
    kind: str
    theta_hash: str
    tier: Tier
    payload: Any
    candidate_dependent: bool = False

    def __post_init__(self):
        if not isinstance(self.tier, Tier):
            raise TypeError("tier must be a Tier")
        if not isinstance(self.theta_hash, str) or len(self.theta_hash) < 16:
            raise ValueError("artefact needs the theta-hash of the weights it was computed at")
        if self.tier is Tier.R0 and self.candidate_dependent:
            raise ValueError(
                "an artefact that depends on the candidate edit is not R0 (it is R1 or R2)"
            )
        if self.tier is Tier.R1 and not self.candidate_dependent:
            raise ValueError(
                "R1 means a base pass whose inputs depend on D; mark candidate_dependent"
            )


def observation(theta_hash: str, base_hash: str, kind: str, payload: Any) -> Artefact:
    """An observation (tap, losses, slopes) computed at ``theta_hash``: R0 at the base, otherwise R2."""
    tier = Tier.R0 if theta_hash == base_hash else Tier.R2
    return Artefact(kind, theta_hash, tier, payload)


class R0Guard:
    def __init__(self, base_hash: str):
        if not isinstance(base_hash, str) or len(base_hash) < 16:
            raise ValueError("base hash required")
        self.base_hash = base_hash

    def _walk(self, obj, path, seen):
        if id(obj) in seen:
            return
        if isinstance(obj, Artefact):
            if obj.tier is not Tier.R0:
                raise AccessError(
                    f"{path}: {obj.kind} is tier {obj.tier.value}; only R0 may be read here"
                )
            if obj.theta_hash != self.base_hash:
                raise AccessError(
                    f"{path}: {obj.kind} was computed at a non-base theta {obj.theta_hash[:12]}"
                )
            return
        if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
            return
        seen.add(id(obj))
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                self._walk(v, f"{path}[{k!r}]", seen)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for i, v in enumerate(obj):
                self._walk(v, f"{path}[{i}]", seen)

    def check(self, *objs):
        seen: set[int] = set()
        for i, o in enumerate(objs):
            self._walk(o, f"arg{i}", seen)


def r0_only(fn):
    """Method decorator: every argument must pass ``self.guard.check`` before the method runs."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        guard = getattr(self, "guard", None)
        if not isinstance(guard, R0Guard):
            raise AccessError(f"{fn.__qualname__}: no R0 guard on {type(self).__name__}")
        guard.check(*args, kwargs)
        return fn(self, *args, **kwargs)

    wrapper.r0_only = True
    return wrapper
