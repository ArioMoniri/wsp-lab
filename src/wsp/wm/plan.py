"""Planning infrastructure: equal-budget ledger, planner-side item store with a sealed test boundary (D6, D8).

* :class:`BudgetLedger` gives every arm the identical number B of real evaluations; every evaluated state counts,
  including rejections and intermediate states; overdraw raises :class:`BudgetExceeded`.
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

    def remaining(self, arm) -> int:
        return self.budget - len(self._entries[arm])

    def charge(self, arm, candidate, seconds: float, kind: str = "evaluated"):
        if arm not in self._entries:
            raise KeyError(arm)
        if self.remaining(arm) <= 0:
            raise BudgetExceeded(f"{arm}: budget {self.budget} exhausted")
        self._entries[arm].append({"candidate": candidate, "seconds": float(seconds), "kind": kind})

    def entries(self, arm):
        return list(self._entries[arm])

    def report(self) -> dict:
        return {
            "budget": self.budget,
            "identical_budget": True,
            "used": {a: len(e) for a, e in self._entries.items()},
            "seconds": {a: sum(x["seconds"] for x in e) for a, e in self._entries.items()},
        }


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
