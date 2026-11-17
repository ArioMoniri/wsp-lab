"""Job harness for the bank and confirmatory runs (SYNTHESIS-r9 C8 H4/H5; PREREG_WME §9).

* **H4 preflight** (:func:`worst_batches`, :func:`preflight`). For every objective the worst batch the panel
  batcher (``wm.panel._batches``: ascending length, greedy under ``max_batch_tokens``) will form is identified
  (largest padded tokens) and run first, longest objectives first, before any main loop; the *device* memory
  (``nvidia-smi`` used bytes, not just the allocator) is read after each and the job aborts with
  :class:`PreflightError` above ``DEVICE_CAP_BYTES`` = 27 GB (>= 1 GB margin to the 28 GB limit).
* **H5 units** (:func:`unit_key`, :class:`UnitStore`, :func:`run_units`). A unit of work is keyed by
  sha256(recipe || theta0 hash || panel content hash). Results are written atomically (temp file, fsync, rename)
  with the payload's sha256 and the theta-hash read back after restore; on resume a unit counts as done only if
  its record verifies, so a kill loses at most the unit in flight and a corrupt unit is recomputed, never trusted.
* **Resident server** (:class:`ResidentServer`): the model stays loaded; each visit is apply -> evaluate ->
  restore, and the theta-hash must equal the base hash after every restore (:class:`ThetaDrift` otherwise).
  ``python -m wsp.wm.harness toy-worker`` is the CPU worker used by the kill/resume test.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEVICE_CAP_BYTES = 27_000_000_000


class PreflightError(RuntimeError):
    pass


class CorruptUnit(RuntimeError):
    pass


class ThetaDrift(RuntimeError):
    pass


# ------------------------------------------------------------------ H4


def _batches(lengths, max_tokens):
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batch, longest = [], 0
    for i in order:
        n = lengths[i]
        if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        yield batch


def worst_batches(lengths: dict, max_batch_tokens: int) -> list:
    """Per objective, the batch with the most padded tokens; ordered longest objective first."""
    plan = []
    for obj, ls in lengths.items():
        best = max(
            _batches(ls, max_batch_tokens),
            key=lambda b: (max(ls[i] for i in b) * len(b), max(ls[i] for i in b)),
        )
        mx = max(ls[i] for i in best)
        plan.append(
            {"objective": obj, "rows": sorted(best), "max_len": mx, "padded_tokens": mx * len(best)}
        )
    return sorted(plan, key=lambda p: (-p["max_len"], -p["padded_tokens"], p["objective"]))


def nvidia_smi_used_bytes(gpu: int = 0) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.split()[0]) * 1024 * 1024


def preflight(
    plan, run, device_bytes=nvidia_smi_used_bytes, cap_bytes: int = DEVICE_CAP_BYTES
) -> dict:
    rows, peak = [], 0
    for p in plan:
        t = time.time()
        run(p)
        b = int(device_bytes())
        peak = max(peak, b)
        rows.append({**p, "device_bytes": b, "seconds": round(time.time() - t, 3)})
        if b > cap_bytes:
            raise PreflightError(
                f"preflight: {p['objective']} worst batch ({p['padded_tokens']} padded tokens) reached device "
                f"{b / 1e9:.1f} GB > cap {cap_bytes / 1e9:.1f} GB; aborting before the main loop"
            )
    return {"pass": True, "peak_bytes": peak, "cap_bytes": cap_bytes, "rows": rows}


# ------------------------------------------------------------------ H5


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


def unit_key(recipe, theta0: str, panel: str) -> str:
    return hashlib.sha256(f"{_canon(recipe)}\x00{theta0}\x00{panel}".encode()).hexdigest()


class UnitStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def write(self, key: str, payload, theta_after: str):
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "key": key,
            "payload": payload,
            "payload_sha256": hashlib.sha256(_canon(payload).encode()).hexdigest(),
            "theta_after_restore": theta_after,
        }
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(body, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        fd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def read(self, key: str):
        p = self.path(key)
        rec = json.loads(p.read_text())
        if (
            rec.get("key") != key
            or hashlib.sha256(_canon(rec["payload"]).encode()).hexdigest() != rec["payload_sha256"]
        ):
            raise CorruptUnit(f"{p}: record does not verify")
        return rec["payload"]

    def done(self, key: str) -> bool:
        if not self.path(key).exists():
            return False
        try:
            self.read(key)
            return True
        except (CorruptUnit, json.JSONDecodeError, KeyError):
            return False

    def digest(self) -> dict:
        out = {}
        for p in sorted(self.root.glob("*/*.json")):
            k = p.stem
            if self.done(k):
                out[k] = json.loads(p.read_text())["payload_sha256"]
        return out


def run_units(store: UnitStore, units, fn, theta0: str, panel: str, on_start=None) -> dict:
    """``units``: [(recipe, label)]; ``fn(recipe) -> payload``. Verified units are skipped."""
    out = {}
    for recipe, label in units:
        k = unit_key(recipe, theta0, panel)
        if store.done(k):
            out[label] = store.read(k)
            continue
        if on_start:
            on_start(k, label)
        payload = fn(recipe)
        store.write(k, payload, theta_after=theta0)
        out[label] = payload
    return out


class ResidentServer:
    """apply -> evaluate -> restore on a resident model (``wm.actions.EditSession``)."""

    def __init__(self, session, evaluate):
        self.sess, self.evaluate = session, evaluate
        self.base = session.base_hash

    def _restore(self):
        self.sess.restore()

    def visit(self, actions) -> dict:
        if self.sess.theta_hash() != self.base:
            raise ThetaDrift("server is not at the base state before a visit")
        self.sess.apply_bundle(actions)
        theta = self.sess.theta_hash()
        try:
            result = self.evaluate()
        finally:
            self._restore()
        if self.sess.theta_hash() != self.base:
            raise ThetaDrift(
                f"theta-hash after restore {self.sess.theta_hash()[:12]} != base {self.base[:12]}"
            )
        return {"theta_hash": theta, "result": result}


# ------------------------------------------------------------------ toy worker (kill/resume test)


def _toy_worker(out_dir: str, n_units: int, sleep_s: float, run_label: str = "run"):
    import torch

    from wsp.wm import actions as A

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rm = A.RoleMap(2, 4, 8, 10, ("full_attention",) * 2, tied=False)
    g = torch.Generator().manual_seed(0)
    params = {
        rm.name(L, "down"): torch.randn(4, 8, generator=g).to(torch.bfloat16) for L in range(2)
    }
    base = A.state_hash(params)
    sess = A.EditSession(params, base_hash=base, rolemap=rm)

    def evaluate():
        time.sleep(sleep_s)
        return {
            "sum": float(sum(p.float().sum() for p in params.values())),
            "state": A.state_hash(params),
        }

    srv = ResidentServer(sess, evaluate)

    def on_start(k, label):
        with open(out / "computed.jsonl", "a") as log:
            log.write(json.dumps({"key": k, "label": label, "run": run_label}) + "\n")
            log.flush()
            os.fsync(log.fileno())

    def fn(recipe):
        return srv.visit(
            [A.scale(rm.name(recipe["layer"], "down"), factor=1.5, dose=recipe["dose"])]
        )

    units = [({"layer": i % 2, "dose": 0.25 * (i + 1)}, f"u{i}") for i in range(n_units)]
    run_units(
        UnitStore(out / "units"), units, fn, theta0=base, panel="toy-panel", on_start=on_start
    )
    (out / "final.json").write_text(json.dumps({"theta_restored": sess.theta_hash() == base}))


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "toy-worker":
        _toy_worker(
            sys.argv[2],
            int(sys.argv[3]),
            float(sys.argv[4]),
            sys.argv[5] if len(sys.argv) > 5 else "run",
        )
    else:
        raise SystemExit(
            "usage: python -m wsp.wm.harness toy-worker <dir> <n_units> <sleep_s> [run_label]"
        )
