"""Hash-chained provenance record per realised edit (SYNTHESIS-r14 R2(a); study M′).

Follows the ``runguard.StudyLog`` pattern — append-only JSON lines, ``seq``/``prev``/``hash`` chain, a ``.head`` file,
an exclusive ``flock`` around every append, fsync — without importing or editing the C1-pinned ``runguard.py``.

Each ``edit`` record holds the recipe (and its digest), the base sha256 **recomputed from the loaded tensor bytes**
(:func:`bytes_sha256`, = ``actions.state_hash``; this closes the gap that ``gemma3.base_hash`` hashed blob names and the
revision rather than the bytes), the sha256 of every tensor whose bytes changed, and the sha256 of the edited state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from wsp.wm import actions as A

GENESIS = "0" * 64


class LogCorrupt(RuntimeError):
    pass


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def record_hash(rec: dict) -> str:
    return digest({k: v for k, v in rec.items() if k != "hash"})


def tensor_hashes(params) -> dict:
    """{name: sha256(dtype|shape|bytes)} for every tensor (``actions.tensor_hash``), from the loaded bytes."""
    return {n: A.tensor_hash(t) for n, t in params.items()}


def combine(hashes: dict) -> str:
    """State digest from per-tensor hashes; equals ``actions.state_hash`` of the same tensors."""
    h = hashlib.sha256()
    for name in sorted(hashes):
        h.update(name.encode() + b"\0" + hashes[name].encode())
    return h.hexdigest()


def bytes_sha256(params) -> str:
    return combine(tensor_hashes(params))


def edit_fields(recipe: dict, before: dict, after: dict) -> dict:
    if set(before) != set(after):
        raise ValueError("tensor sets differ between the base and the edited state")
    changed = {n: after[n] for n in sorted(after) if after[n] != before[n]}
    return {
        "recipe": recipe,
        "recipe_sha256": digest(recipe),
        "base_sha256": combine(before),
        "changed_tensors": changed,
        "edit_sha256": combine(after),
    }


class ProvenanceLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list = []
        self._load()

    @property
    def head_path(self) -> Path:
        return self.path.with_name(self.path.name + ".head")

    @property
    def tail(self) -> str:
        return self.records[-1]["hash"] if self.records else GENESIS

    @contextmanager
    def _locked(self):
        fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load(self):
        self.records = []
        if self.path.exists():
            for n, line in enumerate(self.path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if (
                    rec.get("seq") != n
                    or rec.get("prev") != self.tail
                    or rec.get("hash") != record_hash(rec)
                ):
                    raise LogCorrupt(f"{self.path}: record {n}: sequence or hash chain broken")
                self.records.append(rec)
        if self.head_path.exists():
            head = json.loads(self.head_path.read_text())
            if head["seq"] != len(self.records) or (head["seq"] and head["hash"] != self.tail):
                raise LogCorrupt(
                    f"{self.path}: does not end at its head record {head['seq']} (truncated or altered)"
                )

    def append(self, type_: str, **fields) -> dict:
        with self._locked():
            self._load()
            rec = {
                **fields,
                "type": type_,
                "t": time.time(),
                "seq": len(self.records) + 1,
                "prev": self.tail,
            }
            rec["hash"] = record_hash(rec)
            with open(self.path, "a") as f:
                f.write(canon(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            tmp = self.head_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                f.write(json.dumps({"seq": rec["seq"], "hash": rec["hash"]}))
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.head_path)
            self.records.append(rec)
            return rec

    def append_edit(self, recipe: dict, before: dict, after: dict, **fields) -> dict:
        return self.append("edit", **edit_fields(recipe, before, after), **fields)
