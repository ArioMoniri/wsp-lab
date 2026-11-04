"""Group-held-out splits for edits and cumulative trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


def make_splits(
    rows: Sequence[Mapping[str, Any]], mode: str, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test indices, holding out families, seeds, or connected chains.

    External parent IDs also connect siblings. Cycles and duplicate IDs are rejected.
    """
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate edit id")
    if mode in ("family", "seed"):
        groups = [row[mode] for row in rows]
    elif mode == "trajectory":

        def root(row: Mapping[str, Any]) -> str:
            seen = set()
            node = row["id"]
            while node in by_id:
                if node in seen:
                    raise ValueError("cycle in parent chain")
                seen.add(node)
                parent = by_id[node].get("parent_id")
                if parent is None:
                    break
                node = parent
            return node

        groups = [root(row) for row in rows]
    else:
        raise ValueError(f"unknown split mode: {mode}")
    return list(GroupKFold(n_splits=n_splits).split(np.arange(len(rows)), groups=groups))
