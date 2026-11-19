"""Controller-side transports to the worker's request directory (engineering).

``LocalTransport`` (same file system: CPU fixtures and tests) and ``SSHTransport`` (the Lambda instance, through an
ssh config alias kept outside the repository; the host address is never printed or committed) share one interface:

* ``put(req)``: write ``inbox/<id>.json`` atomically (temp name without the ``.json`` suffix, then rename);
* ``ready(rid)``: the result directory exists;
* ``fetch(rid, dest)``: copy ``outbox/<id>/`` to ``dest`` (the controller verifies it, ``protocol.verify_result``);
* ``set_fence(epoch, instance)``: publish the current owner epoch (takeover stops older requests);
* ``alive_age()``: seconds since the worker's last ``alive`` touch (None if unknown);
* ``progress_age()``: seconds since ``progress.jsonl`` last grew (dead-man heartbeat source, §II.13 item 3);
* ``cancel(rid)``: withdraw a not-yet-started request.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from wsp.wm.c1 import protocol as P


class LocalTransport:
    def __init__(self, root):
        self.root = Path(root)
        for d in ("inbox", "outbox"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def put(self, req: dict):
        P.check_request(req)
        tmp = self.root / "inbox" / f".{req['id']}.tmp"
        tmp.write_text(json.dumps(req, sort_keys=True))
        os.replace(tmp, self.root / "inbox" / f"{req['id']}.json")

    def ready(self, rid: str) -> bool:
        return (self.root / "outbox" / rid / P.RESULT).exists()

    def fetch(self, rid: str, dest) -> Path:
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.root / "outbox" / rid, dest)
        return dest

    def set_fence(self, epoch: int, instance: str):
        tmp = self.root / f".{P.FENCE}.tmp"
        tmp.write_text(json.dumps({"epoch": int(epoch), "instance": instance, "t": time.time()}))
        os.replace(tmp, self.root / P.FENCE)

    def _age(self, name):
        p = self.root / name
        return None if not p.exists() else max(0.0, time.time() - p.stat().st_mtime)

    def alive_age(self):
        return self._age("alive")

    def progress_age(self):
        return self._age("progress.jsonl")

    def cancel(self, rid: str) -> bool:
        p = self.root / "inbox" / f"{rid}.json"
        try:
            p.unlink()
            return True
        except FileNotFoundError:
            return False


class SSHTransport:
    """``alias`` is a Host block in ``ssh_config`` (written by ``lambda_deadman.py ssh-config``, outside the repo)."""

    def __init__(
        self, alias: str, ssh_config, remote_root: str, timeout_s: float = 60.0, runner=None
    ):
        self.alias, self.cfg, self.rroot = alias, str(ssh_config), remote_root.rstrip("/")
        self.timeout = timeout_s
        self.run = runner or self._run

    def _ssh(self):
        return ["ssh", "-F", self.cfg, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.alias]

    def _run(self, argv, stdin: bytes | None = None, timeout=None):
        try:
            p = subprocess.run(
                argv, input=stdin, capture_output=True, timeout=timeout or self.timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise P.TransportError(f"timeout: {argv[0]}") from exc
        if p.returncode != 0:
            raise P.TransportError(
                f"{argv[0]} exit {p.returncode}: {p.stderr.decode(errors='replace')[-300:]}"
            )
        return p.stdout.decode(errors="replace")

    def sh(self, cmd: str, stdin: bytes | None = None, timeout=None) -> str:
        return self.run(self._ssh() + [cmd], stdin, timeout)

    def put(self, req: dict):
        P.check_request(req)
        rid = req["id"]
        d = shlex.quote(f"{self.rroot}/inbox")
        self.sh(
            f"mkdir -p {d} && cat > {d}/.{rid}.tmp && mv {d}/.{rid}.tmp {d}/{rid}.json",
            stdin=json.dumps(req, sort_keys=True).encode(),
        )

    def ready(self, rid: str) -> bool:
        out = self.sh(
            f"test -f {shlex.quote(f'{self.rroot}/outbox/{rid}/{P.RESULT}')} && echo yes || echo no"
        )
        return out.strip() == "yes"

    def fetch(self, rid: str, dest) -> Path:
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        self.run(
            [
                "rsync",
                "-a",
                "-e",
                f"ssh -F {self.cfg} -o BatchMode=yes -o ConnectTimeout=15",
                f"{self.alias}:{self.rroot}/outbox/{rid}/",
                f"{dest}/",
            ],
            timeout=max(self.timeout, 1800),
        )
        return dest

    def set_fence(self, epoch: int, instance: str):
        body = json.dumps({"epoch": int(epoch), "instance": instance, "t": time.time()})
        r = shlex.quote(self.rroot)
        self.sh(
            f"mkdir -p {r} && cat > {r}/.fence.tmp && mv {r}/.fence.tmp {r}/{P.FENCE}",
            stdin=body.encode(),
        )

    def _age(self, name):
        p = shlex.quote(f"{self.rroot}/{name}")
        out = self.sh(
            f"if [ -e {p} ]; then echo $(( $(date +%s) - $(stat -c %Y {p}) )); else echo none; fi"
        )
        out = out.strip()
        return None if out == "none" else float(out)

    def alive_age(self):
        return self._age("alive")

    def progress_age(self):
        return self._age("progress.jsonl")

    def cancel(self, rid: str) -> bool:
        out = self.sh(f"rm -f {shlex.quote(f'{self.rroot}/inbox/{rid}.json')} && echo ok")
        return out.strip() == "ok"
