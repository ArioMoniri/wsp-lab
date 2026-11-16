"""The MSRM-1 v0 outcome allowlist and its read boundary (hard rule 1 of the implementation brief).

Outcomes are read **only** for the 123 existing non-Part-A root ids of F12.2 (sorted-id sha256 ``f96e4ea5…ca2160``),
plus the common TF-v1 ``base`` reference record that every root's Δy is taken against. The reader never lists a
directory: it builds each path itself from an allowlisted id and a fixed file name, refuses anything else before any
system call on that path, resolves the path to its leaf and requires the leaf to be exactly the expected file (no
symlink inside the evaluation tree, no ``..``), refuses any ``part_a*`` / ``sealed*`` component under the data root
(the ``f12_inputs_guard`` rule), opens with ``O_NOFOLLOW`` and authenticates the bytes against the A5 snapshot's pins
(``content.tf_v1_records.primary``: [sha256(eval.json), sha256(records.jsonl)]). Every read is recorded.

Δy = token-weighted mean ``wikitext_ppl`` ``extra.mean_nll`` of the edit minus that of ``base`` over the same 64
windows (item ids and token-id hashes must match), F12.2's TF-v1 WikiText endpoint in nats per valid next token.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from scripts import f12_inputs_guard as fg

EXISTING_ROOTS_SHA256 = "f96e4ea55f4e887abdd713e97b27a06eec2b80fba82a397fbe5d6b9a36ca2160"
EXISTING_ROOTS_N = 123
SOURCE_EXISTING = "existing-TF-v1"
REFERENCE = "base"
FILES = ("eval.json", "records.jsonl")
_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def allowlist_from_population(pop: dict) -> list:
    """The 123 existing roots of the governed population.json, checked against the registered sorted-id hash."""
    ids = sorted(r["id"] for r in pop["rows"] if r.get("source") == SOURCE_EXISTING)
    if len(ids) != EXISTING_ROOTS_N or ids_sha256(ids) != EXISTING_ROOTS_SHA256:
        raise fg.AuthenticationError(
            f"existing-root allowlist ({len(ids)} ids, {ids_sha256(ids)}) differs from F12.2's 123 / "
            f"{EXISTING_ROOTS_SHA256}"
        )
    return ids


class OutcomeReader:
    def __init__(
        self,
        evals_root,
        allowlist,
        allowlist_sha256: str,
        pins: dict,
        forbidden_ids,
        data_root=fg.DATA_ROOT,
    ):
        allowlist = sorted(allowlist)
        if ids_sha256(allowlist) != allowlist_sha256:
            raise fg.AuthenticationError("allowlist differs from its pinned sorted-id sha256")
        bad = [i for i in allowlist if not _ID.match(i) or i == REFERENCE]
        if bad:
            raise fg.BoundaryError(f"malformed or reserved ids in the allowlist: {bad}")
        if set(allowlist) & set(forbidden_ids):
            raise fg.BoundaryError(
                f"Part A ids in the allowlist: {sorted(set(allowlist) & set(forbidden_ids))}"
            )
        self.allow = set(allowlist)
        self.lexical_root = Path(os.path.abspath(evals_root))
        self.real_root = Path(os.path.realpath(evals_root))
        self.data_roots = {Path(os.path.abspath(data_root)), Path(os.path.realpath(data_root))}
        self.pins = {}
        for rid in [*allowlist, REFERENCE]:
            pin = pins.get(rid)
            if not pin or len(pin) != 2:
                raise fg.AuthenticationError(f"{rid}: no snapshot pin for its primary TF-v1 record")
            self.pins[rid] = dict(zip(FILES, pin))
        self.reads = []
        self._cache = {}

    # -------------------------------------------------------------------------------------- boundary

    def _prohibited(self, p: Path) -> bool:
        for d in self.data_roots:
            if d in p.parents and any(
                c.startswith(fg.PROHIBITED_DATA_PREFIXES) for c in p.relative_to(d).parts
            ):
                return True
        return False

    def _read(self, rid: str, fname: str) -> bytes:
        if rid not in self.allow and rid != REFERENCE:
            raise fg.BoundaryError(
                f"{rid!r}: not an allowlisted outcome id (refused before opening)"
            )
        if fname not in FILES:
            raise fg.BoundaryError(f"{fname!r}: not an outcome file name")
        lexical = self.lexical_root / rid / fname
        if self._prohibited(lexical):
            raise fg.BoundaryError(f"{lexical}: prohibited path (refused before opening)")
        real = Path(os.path.realpath(lexical))
        expected = self.real_root / rid / fname
        if real != expected or self._prohibited(real):
            raise fg.BoundaryError(
                f"{lexical}: resolves to {real}, not {expected} (refused before opening)"
            )
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            data = f.read()
        got = fg.sha256_bytes(data)
        if got != self.pins[rid][fname]:
            raise fg.AuthenticationError(f"{lexical}: sha256 {got} differs from the snapshot pin")
        self.reads.append(
            {"id": rid, "file": fname, "path": str(lexical), "resolved": str(real), "sha256": got}
        )
        return data

    # --------------------------------------------------------------------------------------- records

    def _load(self, rid: str):
        if rid in self._cache:
            return self._cache[rid]
        ev = json.loads(self._read(rid, "eval.json"))
        if ev.get("eval_status") != "complete" or ev.get("repeat_id") is not None:
            raise ValueError(f"{rid}: not a complete primary record")
        windows = {}
        for line in self._read(rid, "records.jsonl").decode().splitlines():
            if '"wikitext_ppl"' not in line:
                continue
            r = fg.strict_json_loads(line)
            if r.get("benchmark") != "wikitext_ppl":
                continue
            ex = r.get("extra") or {}
            windows[str(r["item_id"])] = (
                float(ex["mean_nll"]),
                int(r["n_tokens"]),
                ex.get("token_ids_hash"),
            )
        if not windows:
            raise ValueError(f"{rid}: no wikitext_ppl windows")
        ident = {k: ev.get(k) for k in ("config_hash", "model_revision")}
        meta = {
            "eval_sha256": self.pins[rid]["eval.json"],
            "records_sha256": self.pins[rid]["records.jsonl"],
            "identity": ident,
        }
        self._cache[rid] = (windows, meta)
        return self._cache[rid]

    def delta_y(self, rid: str):
        """(Δy in nats/token, provenance) for one allowlisted root."""
        if rid not in self.allow:
            raise fg.BoundaryError(
                f"{rid!r}: not an allowlisted outcome id (refused before opening)"
            )
        we, me = self._load(rid)
        wb, mb = self._load(REFERENCE)
        if sorted(we) != sorted(wb):
            raise ValueError(f"{rid}: windows differ from the base reference")
        for k in wb:
            if we[k][1] != wb[k][1] or we[k][2] != wb[k][2]:
                raise ValueError(f"{rid}: window {k} tokens differ from the base reference")
        if me["identity"] != mb["identity"]:
            raise ValueError(f"{rid}: scoring identity differs from the base reference")
        keys = sorted(wb, key=lambda s: (len(s), s))
        n = sum(wb[k][1] for k in keys)
        ye = sum(we[k][0] * we[k][1] for k in keys) / n
        yb = sum(wb[k][0] * wb[k][1] for k in keys) / n
        return ye - yb, dict(me, n_windows=len(keys), n_tokens=n)
