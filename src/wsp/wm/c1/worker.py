"""Resident worker loop (runs on the single Lambda instance; engineering, PREREG_WME §II.13, §II.28.3).

``Worker(root, handlers).serve()`` processes ``inbox/*.json`` in name order, one at a time. A handler is
``fn(req, ctx) -> (payload, files)``: ``payload`` is JSON, ``files`` maps an output file name to a path the handler
wrote under ``ctx.scratch``. The worker writes ``outbox/<id>/`` atomically (temp dir, fsync, rename), appends one line
to ``progress.jsonl`` per completed request (the only file the dead-man heartbeat reads, §II.13 item 3), and touches
``alive`` every loop.

Fencing: a request whose epoch is below ``fence.json``'s epoch is moved to ``rejected/`` unrun; ``ctx.check()``
(handlers call it between sub-steps) raises :class:`Fenced` when the fence has moved past the running request, so a
controller takeover stops old work. Exceptions are reported, never retried here; ``classify`` maps them to the
registered technical causes (anything unrecognised is ``None``: the controller halts).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from wsp.wm.c1 import protocol as P


class Fenced(RuntimeError):
    """The owner epoch moved past this request (controller takeover): abort, no result."""


def classify(exc: BaseException) -> str | None:
    name = type(exc).__name__
    msg = str(exc)
    if name in ("OutOfMemoryError",) or "CUDA out of memory" in msg:
        return "out_of_memory"
    if name in ("ThetaDrift",) or "theta drift" in msg.lower() or "theta-hash after restore" in msg:
        return "theta_restore_mismatch"
    if (
        name in ("CudaError", "AcceleratorError")
        or "CUDA error" in msg
        or "NCCL" in msg
        or "cuDNN" in msg
    ):
        return "driver_or_cuda"
    if isinstance(exc, (OSError, TimeoutError)) and not isinstance(exc, FileNotFoundError):
        return "infrastructure"
    return None


def _fsync_dir(d: Path):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Ctx:
    def __init__(self, worker: Worker, req: dict, scratch: Path):
        self.worker, self.req, self.scratch = worker, req, scratch
        self.t0 = time.time()

    def check(self):
        f = self.worker.fence()
        if f is not None and int(f.get("epoch", -1)) > int(self.req["epoch"]):
            raise Fenced(f"fence epoch {f['epoch']} > request epoch {self.req['epoch']}")
        self.worker.touch()

    def progress(self, event: str, **kw):
        """Sub-unit progress (a completed evaluation inside a long unit)."""
        self.worker.progress({"id": self.req["id"], "kind": self.req["kind"], "event": event, **kw})


class Worker:
    def __init__(self, root, handlers: dict, poll_s: float = 1.0, log=None, sampler=None):
        self.root = Path(root)
        self.sampler = sampler  # wm.c1.health.Sampler: GPU utilisation/memory heartbeat (§II.31)
        self.handlers = dict(handlers)
        self.poll_s = poll_s
        self.log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
        for d in ("inbox", "outbox", "rejected", "done", "scratch"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------- files
    def fence(self):
        p = self.root / P.FENCE
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def touch(self):
        (self.root / "alive").write_text(str(time.time()))

    def progress(self, rec: dict):
        with open(self.root / "progress.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_result(self, req, rec: dict, files: dict):
        out = self.root / "outbox" / req["id"]
        tmp = self.root / "outbox" / f".{req['id']}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        listed = {}
        for name, src in (files or {}).items():
            if "/" in name or name.startswith(".") or name == P.RESULT:
                raise P.ProtocolError(f"bad output file name {name!r}")
            dst = tmp / name
            shutil.copyfile(src, dst)
            listed[name] = P.file_sha256(dst)
        body = dict(rec, files=listed)
        with open(tmp / P.RESULT, "w") as f:
            f.write(json.dumps(body, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(tmp)
        if out.exists():
            raise P.ProtocolError(f"{out} exists: a request id is answered once")
        os.rename(tmp, out)
        _fsync_dir(out.parent)

    # -------------------------------------------------- one request
    def handle(self, path: Path) -> str:
        try:
            req = P.check_request(json.loads(path.read_text()))
        except (json.JSONDecodeError, P.ProtocolError) as exc:
            self.log(f"malformed request {path.name}: {exc}")
            os.replace(path, self.root / "rejected" / path.name)
            return "malformed"
        f = self.fence()
        if f is not None and int(f.get("epoch", -1)) > int(req["epoch"]):
            os.replace(path, self.root / "rejected" / path.name)
            self.progress({"id": req["id"], "kind": req["kind"], "event": "rejected_stale"})
            return "stale"
        scratch = self.root / "scratch" / req["id"]
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        ctx = Ctx(self, req, scratch)
        base = {
            "schema": P.SCHEMA,
            "id": req["id"],
            "epoch": req["epoch"],
            "unit_key": req["unit_key"],
            "args_sha256": req["args_sha256"],
            "kind": req["kind"],
            "started_at": ctx.t0,
        }
        status = "ok"
        if self.sampler is not None:
            self.sampler.set_inflight(req)
        try:
            fn = self.handlers.get(req["kind"])
            if fn is None:
                raise P.ProtocolError(f"no handler for kind {req['kind']!r}")
            payload, files = fn(req, ctx)
            ctx.check()
            rec = dict(
                base,
                status="ok",
                payload=payload,
                payload_sha256=P.digest(payload),
                seconds=time.time() - ctx.t0,
            )
            self._write_result(req, rec, files or {})
        except Fenced as exc:
            self.log(f"{req['id']}: fenced ({exc}); no result")
            status = "fenced"
        except Exception as exc:  # noqa: BLE001 - reported to the controller, which classifies and charges
            status = "error"
            rec = dict(
                base,
                status="error",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "cause": classify(exc),
                    "trace": traceback.format_exc()[-4000:],
                },
                seconds=time.time() - ctx.t0,
            )
            self._write_result(req, rec, {})
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            if self.sampler is not None:
                self.sampler.set_inflight(None)
        os.replace(path, self.root / "done" / path.name)
        self.progress(
            {"id": req["id"], "kind": req["kind"], "event": status, "seconds": time.time() - ctx.t0}
        )
        return status

    def pending(self) -> list:
        return sorted(p for p in (self.root / "inbox").glob("*.json"))

    def serve(self, stop_file: str = "STOP", max_idle_s: float | None = None):
        idle_since = time.time()
        if self.sampler is not None:
            self.sampler.start()
        while not (self.root / stop_file).exists():
            self.touch()
            todo = self.pending()
            if not todo:
                if max_idle_s is not None and time.time() - idle_since > max_idle_s:
                    return
                time.sleep(self.poll_s)
                continue
            self.handle(todo[0])
            idle_since = time.time()
