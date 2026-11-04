"""Group-held-out splits for edits and cumulative trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


def trajectory_groups(
    rows: Sequence[Mapping[str, Any]], ancestry: Mapping[str, str | None] | None = None
) -> list[str]:
    """Root id of every row's ancestry chain (root -> all descendants share one group).

    ``ancestry`` maps edit id -> parent id for the *whole* zoo manifest and must not be pre-filtered
    to rows that have targets: an intermediate edit without an evaluation still ties its descendants
    to its ancestors. Parent ids unknown to ``ancestry`` act as external roots, so siblings of a
    missing parent stay together. Conflicting parent ids, cycles and duplicate row ids are rejected.
    """
    parents: dict[str, str | None] = dict(ancestry or {})
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate edit id")
    for row in rows:
        parent = row.get("parent_id")
        if row["id"] in parents and parents[row["id"]] != parent:
            raise ValueError(
                f"conflicting parent for {row['id']}: ancestry says {parents[row['id']]!r}, "
                f"row says {parent!r}"
            )
        parents[row["id"]] = parent

    def root(node: str) -> str:
        seen: set[str] = set()
        while node in parents:
            if node in seen:
                raise ValueError("cycle in parent chain")
            seen.add(node)
            parent = parents[node]
            if parent is None:
                break
            node = parent
        return node

    return [root(i) for i in ids]


def make_splits(
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    n_splits: int = 5,
    ancestry: Mapping[str, str | None] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test indices, holding out families, seeds, or full ancestry chains.

    For ``trajectory``, pass the zoo-wide ``ancestry`` (see :func:`trajectory_groups`); without it
    only parent links between the given rows are known and removed intermediates would leak.
    """
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate edit id")
    if mode in ("family", "seed"):
        groups = [row[mode] for row in rows]
    elif mode == "trajectory":
        groups = trajectory_groups(rows, ancestry)
    else:
        raise ValueError(f"unknown split mode: {mode}")
    return list(GroupKFold(n_splits=n_splits).split(np.arange(len(rows)), groups=groups))
