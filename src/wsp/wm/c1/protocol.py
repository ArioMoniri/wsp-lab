"""Controller-worker request protocol (engineering; PREREG_WME §II.28.3 precondition 5).

A request is one JSON file ``inbox/<id>.json``; its result is one directory ``outbox/<id>/`` written by the worker
under a temporary name and renamed into place (atomic on one file system), holding ``result.json`` and any payload
files listed with their sha256 in ``result.json["files"]``. The id binds the owner epoch, the unit key and a fresh
nonce, so a result can only answer the request the current owner issued:

* the controller admits a result only for the id it is waiting for (``verify_result``), and only through
  ``harness.run_units`` -> ``ProductionGuard.succeeded`` (which itself refuses a stale owner);
* ``fence.json`` holds the current owner epoch; the worker refuses (``rejected/``) any request with a lower epoch and
  aborts a running request when the fence moves past it (a controller takeover stops old outstanding work).

Workers never construct a guard and never write the study log.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path

SCHEMA = "wme-c1-rpc-v1"
RESULT = "result.json"
FENCE = "fence.json"


class ProtocolError(RuntimeError):
    """A malformed, mismatched or unverifiable request or result (never admitted)."""


class WorkerError(RuntimeError):
    """The worker reported a failed request. ``cause`` is one of runguard.TECHNICAL_CAUSES or None (halt)."""

    def __init__(self, etype: str, message: str, cause: str | None):
        super().__init__(f"{etype}: {message}")
        self.etype, self.message, self.cause = etype, message, cause


class TransportError(RuntimeError):
    """The controller could not reach the worker (infrastructure)."""


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_id(epoch: int, unit_key: str, kind: str) -> str:
    tag = "".join(c for c in kind if c.isalnum())[:12]
    return f"e{int(epoch):05d}-{tag}-{unit_key[:16]}-{secrets.token_hex(6)}"


def make_request(kind: str, epoch: int, instance: str, unit_key: str, label: str, args: dict) -> dict:
    if not isinstance(args, dict):
        raise ProtocolError("args must be a JSON object")
    return {
        "schema": SCHEMA,
        "id": new_id(epoch, unit_key, kind),
        "kind": kind,
        "epoch": int(epoch),
        "instance": instance,
        "unit_key": unit_key,
        "label": label,
        "args": args,
        "args_sha256": digest(args),
        "issued_at": time.time(),
    }


def check_request(req: dict) -> dict:
    for k in ("schema", "id", "kind", "epoch", "instance", "unit_key", "label", "args", "args_sha256"):
        if k not in req:
            raise ProtocolError(f"request lacks {k}")
    if req["schema"] != SCHEMA:
        raise ProtocolError(f"schema {req['schema']!r}")
    if digest(req["args"]) != req["args_sha256"]:
        raise ProtocolError("request args do not match their digest")
    return req


def verify_result(res_dir, req: dict) -> dict:
    """Load ``res_dir/result.json`` and verify it answers exactly ``req`` (id, epoch, unit, args) and that every
    listed file is present with its sha256. Returns the result record."""
    res_dir = Path(res_dir)
    try:
        res = json.loads((res_dir / RESULT).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{res_dir}: unreadable result") from exc
    for k in ("schema", "id", "epoch", "unit_key", "args_sha256", "status", "files"):
        if k not in res:
            raise ProtocolError(f"result lacks {k}")
    if res["schema"] != SCHEMA:
        raise ProtocolError("result schema")
    if (res["id"], res["epoch"], res["unit_key"], res["args_sha256"]) != (
        req["id"],
        req["epoch"],
        req["unit_key"],
        req["args_sha256"],
    ):
        raise ProtocolError(f"result {res['id']} does not answer request {req['id']} (stale or foreign)")
    for name, sha in res["files"].items():
        p = res_dir / name
        if not p.is_file() or file_sha256(p) != sha:
            raise ProtocolError(f"{res_dir}/{name}: missing or sha256 mismatch")
    if res["status"] == "ok":
        if "payload" not in res:
            raise ProtocolError("ok result without payload")
        if digest(res["payload"]) != res.get("payload_sha256"):
            raise ProtocolError("payload digest mismatch")
    elif res["status"] != "error":
        raise ProtocolError(f"status {res['status']!r}")
    return res


def raise_if_error(res: dict):
    if res["status"] == "error":
        e = res.get("error") or {}
        raise WorkerError(e.get("type", "Error"), e.get("message", ""), e.get("cause"))
    return res
