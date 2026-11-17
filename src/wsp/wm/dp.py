"""Data-parallel runner for the WM-E bank and confirmatory work on k GPUs (RULING-wme-resources §2; round-9
pro-engineer L1-L2; harness H5).

* **Units.** A unit is (state, objective, item chunk). Chunks are fixed by the panel alone (items sorted by id,
  ``chunk_size`` per chunk), so the number of GPUs never changes which items share a batch; one GPU holds one
  whole bf16 model (L1) and evaluates whole units. Unit keys are content addresses
  (``harness.unit_key(recipe, theta0, panel)``), so a resumed or re-run job skips verified units.
* **Claims.** Workers take units through exclusive lock files (``O_CREAT | O_EXCL``) in a shared directory; a claim
  older than ``ttl_s`` whose unit is not done is presumed dead and may be taken over. Results go through the atomic,
  fsynced :class:`wsp.wm.harness.UnitStore`; a killed worker loses at most its in-flight unit.
* **Reduction (L2).** Every unit stores per-item fp32-representable values with their item group. The reduction is
  on CPU, in fp64, in the fixed order (objective, group, item id), never in completion order: group means, then
  the paired per-state mean and SE over groups (n - 1 denominator). 1-vs-k parity is checked at rtol 1e-6.
* **Progress.** Each worker rewrites ``progress/<worker>.json`` after every unit (done count, last key, state);
  the AWS-side heartbeat loop watches these files, and the dead-man terminator ends the instance when they stop.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from wsp.wm.harness import UnitStore, unit_key

RTOL = 1e-6


# ------------------------------------------------------------------ units


def chunk_items(items: dict, chunk_size: int) -> list[dict]:
    """[{objective, chunk, item_ids}] with items sorted by id inside each objective (panel-only; k-independent)."""
    out = []
    for obj in sorted(items):
        ids = sorted(r["id"] for r in items[obj])
        for c in range(math.ceil(len(ids) / chunk_size)):
            out.append(
                {
                    "objective": obj,
                    "chunk": c,
                    "item_ids": ids[c * chunk_size : (c + 1) * chunk_size],
                }
            )
    return out


def make_units(states: dict, chunks: list[dict], theta0: str, panel: str) -> list[dict]:
    """One unit per (state, chunk). ``states``: {state id: action recipe (JSON)}; the recipe, not the id, is
    content-addressed, so two ids with one recipe share their units."""
    units = []
    for sid in sorted(states):
        for ch in chunks:
            recipe = {"state": states[sid], **ch}
            units.append(
                {"key": unit_key(recipe, theta0, panel), "state_id": sid, "recipe": recipe, **ch}
            )
    return units


class ClaimDir:
    def __init__(self, root, ttl_s: float = 1800.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = float(ttl_s)

    def _p(self, key):
        return self.root / f"{key}.claim"

    def claim(self, key: str, worker: str) -> bool:
        p = self._p(key)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - p.stat().st_mtime
            except FileNotFoundError:
                return self.claim(key, worker)
            if age <= self.ttl:
                return False
            try:  # stale: take it over (one winner through the rename)
                tmp = self.root / f"{key}.{worker}.{os.getpid()}.take"
                tmp.write_text(worker)
                os.replace(tmp, p)
                return True
            except OSError:
                return False
        with os.fdopen(fd, "w") as f:
            f.write(worker)
        return True

    def release(self, key: str):
        try:
            self._p(key).unlink()
        except FileNotFoundError:
            pass


def _progress(out: Path, worker: str, **kw):
    d = out / "progress"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{worker}.tmp"
    tmp.write_text(json.dumps({"worker": worker, "t": time.time(), **kw}, sort_keys=True))
    os.replace(tmp, d / f"{worker}.json")


def run_worker(out, units, fn, worker: str, theta_after: str, ttl_s: float = 1800.0, log=None):
    """Claim-and-run loop. ``fn(unit) -> payload`` (with ``records``); returns the number of units computed here.
    Workers start at different offsets so they rarely contend for the same claim."""
    out = Path(out)
    store, claims = UnitStore(out / "units"), ClaimDir(out / "claims", ttl_s)
    n = len(units)
    off = (sum(map(ord, worker)) * 7919) % max(1, n)
    order = units[off:] + units[:off]
    done = 0
    _progress(out, worker, done=0, state="running")
    for u in order:
        if store.done(u["key"]):
            continue
        if not claims.claim(u["key"], worker):
            continue
        if store.done(u["key"]):  # finished by another worker between the check and the claim
            claims.release(u["key"])
            continue
        if log:
            log(u)
        payload = fn(u)
        store.write(u["key"], payload, theta_after=theta_after)
        claims.release(u["key"])
        done += 1
        _progress(out, worker, done=done, last=u["key"], state="running")
    # then wait for units claimed elsewhere; a claim that goes stale (dead worker) is taken over
    while True:
        left = [u for u in order if not store.done(u["key"])]
        if not left:
            break
        progressed = False
        for u in left:
            if not store.done(u["key"]) and claims.claim(u["key"], worker):
                if store.done(u["key"]):
                    claims.release(u["key"])
                    continue
                if log:
                    log(u)
                store.write(u["key"], fn(u), theta_after=theta_after)
                claims.release(u["key"])
                done += 1
                progressed = True
                _progress(out, worker, done=done, last=u["key"], state="running")
        if not progressed:
            _progress(out, worker, done=done, state="waiting", waiting_for=len(left))
            time.sleep(min(5.0, max(0.05, claims.ttl / 4)))
    _progress(out, worker, done=done, state="finished")
    return done


def missing(out, units) -> list[str]:
    store = UnitStore(Path(out) / "units")
    return [u["key"] for u in units if not store.done(u["key"])]


# ------------------------------------------------------------------ fixed-order fp64 reduction


def records_for(unit: dict, values, groups) -> list[dict]:
    """Per-item records of one unit; values are rounded to fp32 (the GPU's output precision) on write."""
    return [
        {
            "objective": unit["objective"],
            "group": g,
            "item_id": i,
            "value": float(np.float32(v)),
        }
        for i, v, g in zip(unit["item_ids"], values, groups, strict=True)
    ]


def reduce_fixed_order(records) -> dict:
    """{objective: {group: fp64 mean over its items}}, summed in (objective, group, item id) order."""
    recs = sorted(records, key=lambda r: (r["objective"], r["group"], r["item_id"]))
    out: dict = {}
    for r in recs:
        g = out.setdefault(r["objective"], {}).setdefault(r["group"], [0.0, 0])
        g[0] = float(np.float64(g[0]) + np.float64(r["value"]))
        g[1] += 1
    return {o: {g: s / n for g, (s, n) in sorted(v.items())} for o, v in sorted(out.items())}


def paired_stats(base: dict, edit: dict) -> dict:
    """Per objective: mean and SE (n - 1) of the paired group-level change, n = groups. Fixed order."""
    out = {}
    for obj in sorted(base):
        if set(base[obj]) != set(edit.get(obj, {})):
            raise ValueError(f"{obj}: base and edited groups differ")
        d = np.array(
            [np.float64(edit[obj][g]) - np.float64(base[obj][g]) for g in sorted(base[obj])]
        )
        n = len(d)
        out[obj] = {
            "n": n,
            "mean": float(d.mean()),
            "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        }
    return out


def state_groups(out, units, state_id) -> dict:
    store = UnitStore(Path(out) / "units")
    recs = []
    for u in units:
        if u["state_id"] == state_id:
            recs.extend(store.read(u["key"])["records"])
    return reduce_fixed_order(recs)


def _flat(d, prefix=()):
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict):
            yield from _flat(v, (*prefix, k))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield (*prefix, k), float(v)


def parity(a: dict, b: dict, rtol: float = RTOL) -> dict:
    """Leaf-by-leaf comparison of two nested numeric dicts at relative tolerance ``rtol``."""
    fa, fb = dict(_flat(a)), dict(_flat(b))
    if set(fa) != set(fb):
        return {"pass": False, "max_rel": float("inf"), "reason": "key sets differ"}
    worst, bitwise = 0.0, True
    for k, x in fa.items():
        y = fb[k]
        if x != y:
            bitwise = False
        if math.isnan(x) and math.isnan(y):
            continue
        rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
        worst = max(worst, rel)
    return {"pass": worst <= rtol, "max_rel": worst, "bitwise": bitwise, "rtol": rtol, "n": len(fa)}


def parity_states(a: dict, b: dict, rtol: float = RTOL) -> dict:
    return parity(a, b, rtol)


# ------------------------------------------------------------------ toy worker (tests: parity, kill/resume)


def _toy_setup(n_states):
    items = {
        o: [{"id": f"{o}{i:02d}", "group": f"g{i // 2}"} for i in range(8)] for o in ("p", "q")
    }
    states = {f"s{s}": ([] if s == 0 else [{"dose": 0.25 * s}]) for s in range(n_states)}
    units = make_units(states, chunk_items(items, 4), theta0="toy-theta0", panel="toy-panel")
    gmap = {r["id"]: r["group"] for v in items.values() for r in v}
    return items, states, units, gmap


def _toy_worker(out, n_states, sleep_s, worker, ttl_s=1800.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _, _states, units, gmap = _toy_setup(n_states)

    def fn(u):
        time.sleep(sleep_s)
        dose = (u["recipe"]["state"] or [{"dose": 0.0}])[0]["dose"]
        vals = [
            np.float32(math.sin((int(i[1:]) + 1) * (1.0 + dose)) * 100.0 + (u["objective"] == "q"))
            for i in u["item_ids"]
        ]
        return {"records": records_for(u, vals, [gmap[i] for i in u["item_ids"]])}

    def log(u):
        with open(out / "computed.jsonl", "a") as f:
            f.write(json.dumps({"key": u["key"], "worker": worker}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    run_worker(out, units, fn, worker, theta_after="toy-theta0", ttl_s=ttl_s, log=log)


def toy_reduce(out, n_states) -> dict:
    _, states, units, _ = _toy_setup(n_states)
    base = state_groups(out, units, "s0")
    return {
        sid: paired_stats(base, state_groups(out, units, sid))
        for sid in sorted(states)
        if sid != "s0"
    }


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "toy-worker":
        ttl = float(sys.argv[sys.argv.index("--ttl") + 1]) if "--ttl" in sys.argv else 1800.0
        _toy_worker(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], ttl)
    else:
        raise SystemExit(
            "usage: python -m wsp.wm.dp toy-worker <dir> <n_states> <sleep_s> <worker>"
        )
