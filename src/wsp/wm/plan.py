"""Planning infrastructure: equal-budget ledger, planner-side item store with a sealed test boundary (D6, D8).

* :class:`BudgetLedger` gives every arm the identical number B of real-evaluation bundles (:data:`BUNDLE_DEFINITION`,
  Board A condition 10); every evaluated state counts, including rejections and intermediate states; overdraw raises
  :class:`BudgetExceeded`; partial (screen) and oracle evaluations are charged in explicit fractional units; repeats
  of an evaluated theta-hash cost 0 (cache); ``verify_parity`` requires every arm closed at used + forfeited = B.
  ``report`` separates the identical budget from (possibly unequal) usage.
* :class:`PlannerItemStore` follows the read-boundary pattern of ``scripts/f12_inputs_guard.py``: paths are checked
  lexically and after ``realpath`` (symlinks and ``..`` resolved) before opening, reads use ``O_NOFOLLOW`` on the
  resolved leaf, and every read is recorded with its sha256. Anything under a ``sealed*`` directory, or outside the
  ``open/`` directory, is refused with :class:`SealedError`. Planner objects only ever receive this store.
* :func:`read_sealed_test` is the separate final-test reader; it refuses unless every arm's winner is frozen.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class SealedError(PermissionError):
    pass


class ParityError(RuntimeError):
    """Arms did not end with equal accounted usage."""


BUNDLE_DEFINITION = (
    "One real-evaluation bundle = one evaluation of one new edited state (theta != theta0, a theta-hash not yet "
    "evaluated by this arm) on the full search-development panel (every objective and guardrail item). Charged "
    "per arm: every intermediate/prefix state, rejected candidate, LoRA checkpoint and validation evaluation is "
    "one bundle; a screen on a subset is screen_items/full_items of a bundle; an oracle derivative pass at an "
    "edited state is priced explicitly in bundle units (its GPU-seconds over one bundle's); a repeat of a theta "
    "this arm already evaluated is served from the content-addressed cache at 0 units; generation audits are "
    "never part of B. Parity: every arm is closed with used + forfeited = B."
)
BUNDLE_KINDS = {  # kind -> fixed units (None = explicit fractional units required)
    "evaluated": 1.0,
    "intermediate": 1.0,
    "rejected": 1.0,
    "checkpoint": 1.0,
    "validation": 1.0,
    "screen": None,
    "oracle": None,
}
NOT_IN_BUDGET = ("generation",)


class BudgetLedger:
    def __init__(self, budget: int, arms):
        arms = list(arms)
        if int(budget) <= 0:
            raise ValueError("budget must be positive")
        if not arms or len(set(arms)) != len(arms):
            raise ValueError("arms must be unique and nonempty")
        self.budget = int(budget)
        self.arms = arms
        self._entries = {a: [] for a in arms}
        self._closed = {}
        self._seen = {a: set() for a in arms}

    def _used(self, arm) -> float:
        return sum(e["units"] for e in self._entries[arm])

    def remaining(self, arm):
        r = self.budget - self._used(arm)
        return int(r) if float(r).is_integer() else r

    def charge(
        self, arm, candidate, seconds: float, kind: str = "evaluated", units=None, theta_hash=None
    ):
        """Charge one real evaluation under :data:`BUNDLE_DEFINITION`."""
        if arm not in self._entries:
            raise KeyError(arm)
        if arm in self._closed:
            raise BudgetExceeded(f"{arm}: closed ({self._closed[arm]['reason']})")
        if kind in NOT_IN_BUDGET or kind not in BUNDLE_KINDS:
            raise ValueError(f"{kind!r} is not a budgeted evaluation kind")
        fixed = BUNDLE_KINDS[kind]
        if fixed is None:
            if units is None or not 0 < units <= 1:
                raise ValueError(f"{kind}: explicit fractional units in (0, 1] required")
        else:
            if units is not None and units != fixed:
                raise ValueError(f"{kind} is {fixed} bundle(s)")
            units = fixed
        if theta_hash is not None and theta_hash in self._seen[arm] and fixed is not None:
            kind, units = "cache", 0.0
        elif self.remaining(arm) - units < -1e-12:
            raise BudgetExceeded(f"{arm}: budget {self.budget} exhausted")
        if theta_hash is not None and fixed is not None:
            self._seen[arm].add(theta_hash)
        self._entries[arm].append(
            {
                "candidate": candidate,
                "seconds": float(seconds),
                "kind": kind,
                "units": float(units),
                "theta_hash": theta_hash,
            }
        )

    def entries(self, arm):
        return list(self._entries[arm])

    def close(self, arm, reason: str):
        if arm not in self._entries:
            raise KeyError(arm)
        self._closed[arm] = {
            "reason": str(reason),
            "forfeited": float(self.budget - self._used(arm)),
        }

    def report(self) -> dict:
        units = {a: self._used(a) for a in self.arms}
        by_kind = {}
        for a, es in self._entries.items():
            k = {}
            for e in es:
                if e["units"]:
                    k[e["kind"]] = k.get(e["kind"], 0.0) + e["units"]
            by_kind[a] = k
        return {
            "budget": self.budget,
            "budget_identical": True,  # one B for every arm, by construction
            "equal_usage": len(set(units.values())) == 1,
            "used": {a: len(e) for a, e in self._entries.items()},
            "units": units,
            "by_kind": by_kind,
            "seconds": {a: sum(x["seconds"] for x in e) for a, e in self._entries.items()},
            "closed": dict(self._closed),
            "definition": BUNDLE_DEFINITION,
        }

    def verify_parity(self) -> dict:
        """Every arm closed and used + forfeited = B; raises :class:`ParityError` otherwise."""
        rep = self.report()
        open_arms = [a for a in self.arms if a not in self._closed]
        if open_arms:
            raise ParityError(f"arms not closed: {open_arms}")
        for a in self.arms:
            tot = rep["units"][a] + self._closed[a]["forfeited"]
            if abs(tot - self.budget) > 1e-9:
                raise ParityError(f"{a}: used + forfeited = {tot} != {self.budget}")
        rep["forfeited"] = {a: self._closed[a]["forfeited"] for a in self.arms}
        rep["parity"] = True
        return rep


def _is_sealed(p: Path) -> bool:
    return any(part.startswith("sealed") for part in p.parts)


class PlannerItemStore:
    """Read-only access to the open splits of a panel directory; sealed test items are unreachable."""

    def __init__(self, panel_root):
        self.root = Path(os.path.abspath(panel_root))
        self.open_dir = Path(os.path.realpath(self.root / "open"))
        self.reads = []

    def _resolve(self, path) -> Path:
        lexical = Path(os.path.abspath(path))
        if _is_sealed(lexical.relative_to(self.root) if self.root in lexical.parents else lexical):
            raise SealedError(f"{path}: sealed path (refused before opening)")
        real = Path(os.path.realpath(path))
        if not (self.open_dir in real.parents) or _is_sealed(real.relative_to(self.open_dir)):
            raise SealedError(
                f"{path}: resolves to {real}, outside the open splits (refused before opening)"
            )
        return real

    def read_path(self, path):
        real = self._resolve(path)
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            data = f.read()
        self.reads.append(
            {"path": str(path), "resolved": str(real), "sha256": hashlib.sha256(data).hexdigest()}
        )
        return [json.loads(line) for line in data.decode().splitlines() if line.strip()]

    def read(self, split: str):
        if (
            split == "test"
            or "sealed" in split
            or "/" in split
            or "\\" in split
            or split.startswith(".")
        ):
            raise SealedError(f"{split!r}: not an open split")
        return self.read_path(self.root / "open" / f"{split}.jsonl")


def read_sealed_test(panel_root, frozen_winners):
    """The single final-test read; refuses until every arm's winner is frozen."""
    if not frozen_winners or any(v is None for v in frozen_winners.values()):
        raise SealedError("sealed test items are read only after every arm's winner is frozen")
    p = Path(panel_root) / "sealed_test" / "test.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class CandidateSpace:
    """A frozen parametric action space: up to ``max_actions`` (layer, direction, dose) actions per candidate."""

    def __init__(self, layers, directions, doses, max_actions=2):
        self.layers, self.directions, self.doses = list(layers), list(directions), list(doses)
        self.max_actions = int(max_actions)
        self.primitives = list(itertools.product(self.layers, self.directions, self.doses))

    def sample(self, n: int, seed: int):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            k = rng.randint(1, self.max_actions)
            out.append(tuple(rng.choice(self.primitives) for _ in range(k)))
        return out
