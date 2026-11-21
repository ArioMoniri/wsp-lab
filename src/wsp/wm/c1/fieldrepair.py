"""The outcome-blind field repair (chair ruling ``RULING-wme-c1-bank-halt.md`` K3). Engineering; no scientific
content, no unit result.

The bank's field (8 K-FAC groups and the vocabulary rows) lives on the worker's disk; a new instance must
re-materialise it before the bank resumes. Each field unit's eligible payload in the study store is the registered
result; the repair re-collects the same unit on the worker (request ``field_repair``, never through
``harness.run_units``), compares the re-collected **data-level** hashes with that payload, and halts with a
run-health incident on any mismatch (nothing is substituted). A matching repair is recorded as one registered
``repair`` record (cause ``infrastructure``, the code commit, an outcome-blind attestation).

Data-level hashes: a field group's payload records ``file_sha256`` (sha256 of every collected ``.pt`` file) and
``kept_sha256`` (every kept-factor file); ``torch.save`` is byte-deterministic for equal tensors. The manifests embed
wall-clock timings and memory peaks (``identity["timing"]``), so ``manifest_sha256`` is recorded but never compared.
A payload with no data-level hash (the vocabulary unit before v2r recorded only ``manifest_sha256``) cannot be
verified: the plan refuses it before any GPU work.
"""

from __future__ import annotations


class FieldRepairMismatch(RuntimeError):
    """A re-collected field unit differs from its eligible payload: halt and report, substitute nothing."""


class FieldRepairUnverifiable(RuntimeError):
    """An eligible field payload records no data-level hash: the repair cannot verify it bitwise."""


def expected(kind: str, payload: dict) -> dict | None:
    """The data-level hashes a re-collection must reproduce, or None when the payload records none."""
    if kind == "field_group":
        if not payload.get("file_sha256") or not payload.get("kept_sha256"):
            return None
        return {"file_sha256": payload["file_sha256"], "kept_sha256": payload["kept_sha256"]}
    if kind == "field_vocab":
        if not payload.get("file_sha256"):
            return None
        return {"file_sha256": payload["file_sha256"]}
    raise KeyError(f"{kind} is not a field unit")


def observed(kind: str, payload: dict) -> dict:
    return {k: payload.get(k) for k in (expected(kind, payload) or {"file_sha256": None})}


def plan(units: dict) -> dict:
    """``units``: {label: (kind, eligible payload)} -> {"expected": {label: hashes}, "indirect": [vocabulary units
    without a data-level hash: verified through a completed dependent bank unit (K3b)], "unverifiable": [field
    groups without data-level hashes: refused]}."""
    exp, ind, bad = {}, [], []
    for label, (kind, payload) in sorted(units.items()):
        e = expected(kind, payload)
        if e is not None:
            exp[label] = e
        elif kind == "field_vocab":
            ind.append(label)
        else:
            bad.append(label)
    return {"expected": exp, "indirect": ind, "unverifiable": bad}


# ------------------------------------------------------------------ K3b (ruling addendum): indirect vocabulary check

TIMING_SUFFIX = "seconds"


def vocab_dependent(recipe: dict, roots_by_id: dict) -> bool:
    """True when the unit's recorded result depends on the vocabulary field: its edit touches ``embed_tokens``, whose
    R(E) contribution ``RCache`` computes with ``Field.contract_embed`` (the vocabulary rows). From
    ``bankops.root_actions`` / ``inert_steps``: a ``row_scale`` root edits embedding rows; an ``i2`` inert edits
    embedding rows; a composition depends if any of its roots does. Every other family touches layer writers only."""
    kind, a = recipe["kind"], recipe["args"]
    if kind == "bank_side":
        return roots_by_id[a["root"]]["family"] == "row_scale"
    if kind == "bank_comp":
        return any(roots_by_id[r]["family"] == "row_scale" for r in a["order"])
    if kind == "bank_inert":
        return a["spec"]["kind"] == "i2"
    return False


def timing_free(obj):
    """The payload without wall-clock fields (every key ending in ``seconds``, at any depth)."""
    if isinstance(obj, dict):
        return {k: timing_free(v) for k, v in obj.items() if not str(k).endswith(TIMING_SUFFIX)}
    if isinstance(obj, list):
        return [timing_free(v) for v in obj]
    return obj


def timing_free_sha(payload) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(timing_free(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def r_vectors(payload) -> list:
    return [[s.get("h"), s.get("R"), s.get("R_flags")] for s in payload.get("states", [])]


def compare(label: str, want: dict, got: dict) -> dict:
    diff = {
        k: {"eligible": want[k], "recollected": got.get(k)} for k in want if got.get(k) != want[k]
    }
    return {"label": label, "equal": not diff, "diff": diff}
