#!/usr/bin/env python3
"""Mutation sensitivity of the runtime property test (PREREG_WME §II.27): remove the fencing mechanisms from the
repaired ``wsp.wm.runguard`` one at a time and check that the seeded property test
(``tests/test_wme_runguard_property.py``) detects it. CPU only; temporary directories only.

    wme_v2i_property_mutations.py --out OUT.json [--n 300]

Each mutation runs in three modes: ``full`` (every assertion, including "a stale object is refused and writes
nothing"), ``semantic`` (that fencing assertion off: only the invariants I0-I6 computed from the independent replay of
the raw log) and ``semantic_without_I0`` (also without the sequence / hash-chain / head checks, so the result-level
invariants I1-I6 must detect it alone). Unmutated controls run the same seeds in every mode.

Mutations:
* ``m4_no_lock_no_reread``: no OS lock and no re-read: each object loads the log once and then acts on its cached
  snapshot (the v2h behaviour Board A's re-check 2 reproduced);
* ``m5_no_epoch_fence``: lock and re-read kept, owner-epoch fencing removed;
* ``m6_no_lock``: re-read kept, OS lock removed. A single-process sequence cannot interleave inside an operation, so
  this one is also checked with 4 forked processes appending concurrently through the mutated ledger.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_wme_runguard_property as T

from wsp.wm import runguard as RG

ORIG_REFRESH = RG.StudyLog._refresh


def m4_no_lock_no_reread():
    @contextmanager
    def locked(self):
        if self.path is not None and self._depth == 0 and not self.records:
            ORIG_REFRESH(self)  # loaded once, never re-read
        self._depth += 1
        try:
            yield self.state
        finally:
            self._depth -= 1

    return [(RG.StudyLog, "locked", locked)]


def m5_no_epoch_fence():
    return [(RG.ProductionGuard, "_fence", lambda self: None)]


def m6_no_lock():
    @contextmanager
    def locked(self):
        if self.path is not None and self._depth == 0:
            ORIG_REFRESH(self)
        self._depth += 1
        try:
            yield self.state
        finally:
            self._depth -= 1

    return [(RG.StudyLog, "locked", locked)]


def control():
    return []


@contextmanager
def patched(mutation):
    patches = mutation()
    saved = [(cls, name, cls.__dict__[name]) for cls, name, _ in patches]
    for cls, name, f in patches:
        setattr(cls, name, f)
    try:
        yield
    finally:
        for cls, name, f in saved:
            setattr(cls, name, f)


def run_property(mutation, n, seed0, strict, chain=True):
    RG.CAP_GPU_H, RG.CAP_USD = T.CAP, 1e9
    T.CHECK_CHAIN = chain
    try:
        with patched(mutation):
            for s in range(n):
                try:
                    T.run_sequences(1, seed0=seed0 + s, strict=strict)
                except AssertionError as e:
                    return {
                        "detected": True,
                        "first_seed": seed0 + s,
                        "sequences_run": s + 1,
                        "assertion": str(e)[:300],
                    }
                except Exception as e:  # noqa: BLE001 - a crash under a mutation also counts as detection
                    return {
                        "detected": True,
                        "first_seed": seed0 + s,
                        "sequences_run": s + 1,
                        "assertion": f"{type(e).__name__}: {str(e)[:300]}",
                    }
        return {"detected": False, "sequences_run": n}
    finally:
        T.CHECK_CHAIN = True


def _append(path, n):
    led = RG.CombinedLedger(path, {"e2": 1e6})
    for _ in range(n):
        led.add("e2", 1.0)


def run_concurrent(mutation, procs=4, n=300):
    """4 forked processes append through the (mutated) ledger; an unmutated reader verifies the result."""
    tmp = Path(tempfile.mkdtemp(prefix="wme-rg-mp-"))
    try:
        path = tmp / RG.LOG_NAME
        with patched(mutation):
            ctx = mp.get_context("fork")
            ps = [ctx.Process(target=_append, args=(path, n)) for _ in range(procs)]
            for p in ps:
                p.start()
            for p in ps:
                p.join(600)
        codes = [p.exitcode for p in ps]
        try:
            st = RG.StudyLog(path).state
            ok = (
                codes == [0] * procs
                and st.n == procs * n
                and st.unit_s.get("e2") == float(procs * n)
            )
            what = f"{st.n} records verified, {st.unit_s.get('e2')} s charged; exit codes {codes}"
        except RG.LogCorrupt as e:
            ok, what = False, f"LogCorrupt: {e}; exit codes {codes}"
        return {"detected": not ok, "expected_records": procs * n, "result": what[:300]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    out = {
        "label": "mutation sensitivity of the runtime property test (§II.27)",
        "n_max": a.n,
        "seed0": 9_000_000,
        "results": {},
    }
    for name, m in (
        ("control_unmutated", control),
        ("m4_no_lock_no_reread", m4_no_lock_no_reread),
        ("m5_no_epoch_fence", m5_no_epoch_fence),
        ("m6_no_lock", m6_no_lock),
    ):
        res = {
            "full": run_property(m, a.n, 9_000_000, True),
            "semantic": run_property(m, a.n, 9_000_000, False),
            "semantic_without_I0": run_property(m, a.n, 9_000_000, False, chain=False),
        }
        if name in ("control_unmutated", "m6_no_lock"):
            res["concurrent_processes"] = run_concurrent(m)
        out["results"][name] = res
        print(name, json.dumps(res), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
