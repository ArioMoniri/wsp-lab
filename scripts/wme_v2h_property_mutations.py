#!/usr/bin/env python3
"""Mutation sensitivity of the runtime property test (PREREG_WME §II.26): re-inject each of Board A's three freeze
re-check defects into the repaired ``wsp.wm.runguard`` and check that the seeded property test
(``tests/test_wme_runguard_property.py``) detects it. CPU only; temporary directories only.

    wme_v2h_property_mutations.py --out OUT.json [--n 300]

Mutations:
* ``m1_clock``: v2g's clock handling: no halt on a backward reading, signed local durations summed across instances;
* ``m2_generation``: v2g's digest-only contract identity: eligibility and canaries ignore the transition generation;
* ``m3_post_loss_canary``: v2g's loss handling: canaries accepted for closed/lost instances, no irreversible disposal,
  canaries after closure still close an attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_wme_runguard_property as T

from wsp.wm import runguard as RG


def m1_clock():
    def tick(self):
        now = float(self.clock())
        if now > self.max_tick():
            self._write({"kind": "tick", "t": now})
        return now

    def instance_charges(self, now=None):
        now = float(self.clock()) if now is None else float(now)
        opened, closed = {}, {}
        for r in self.rows:
            if r.get("kind") == "instance_open":
                opened.setdefault(r["instance"], (r["t"], r["gpus"]))
            elif r.get("kind") == "instance_close":
                closed.setdefault(r["instance"], r["t"])
        return {i: (closed.get(i, now) - t0) * g / 3600 for i, (t0, g) in opened.items()}

    def instance_close(self, instance):
        self._write({"kind": "instance_close", "instance": instance, "t": float(self.clock())})

    return [
        (RG.CombinedLedger, "tick", tick),
        (RG.CombinedLedger, "instance_charges", instance_charges),
        (RG.CombinedLedger, "instance_close", instance_close),
    ]


def m2_generation():
    """v2g's eligibility: the contract digest only, canaries matched by digest only (no generation)."""

    def attempt_status(self, a):
        if a.get("contract") != self.contract:
            return "superseded_contract"
        if any(
            e["type"] == "attempt_disposed" and e["attempt_seq"] == a["seq"] for e in self.events
        ):
            return "retained_ineligible"
        ref = self.attestation.canary_reference
        close = self._closure_seq(a["instance"])
        cans = [
            e
            for e in self.events
            if e["type"] == "canary"
            and e["instance"] == a["instance"]
            and e.get("contract") == a["contract"]
            and (close is None or e["seq"] < close)
        ]
        before = [e for e in cans if e["seq"] < a["seq"]]
        after = [e for e in cans if e["seq"] > a["seq"]]
        if not before or before[-1]["sha"] != ref:
            return "retained_ineligible"
        if not after:
            return "retained_ineligible" if close is not None else "pending"
        return "eligible" if after[0]["sha"] == ref else "retained_ineligible"

    return [(RG.ProductionGuard, "attempt_status", attempt_status)]


def m3_post_loss_canary():
    def canary(self, sha):
        self._event("canary", sha=sha)
        return sha == self.attestation.canary_reference

    def closure_seq(self, instance):
        return None

    def dispose(self, instance, reason):
        return None

    return [
        (RG.ProductionGuard, "canary", canary),
        (RG.ProductionGuard, "_closure_seq", closure_seq),
        (RG.ProductionGuard, "_dispose_unbracketed", dispose),
    ]


def run(mutation, n, seed0):
    patches = mutation()
    saved = [(cls, name, getattr(cls, name)) for cls, name, _ in patches]
    for cls, name, f in patches:
        setattr(cls, name, f)
    RG.CAP_GPU_H, RG.CAP_USD = T.CAP, 1e9
    try:
        for s in range(n):
            try:
                T.run_sequences(1, seed0=seed0 + s)
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
        for cls, name, f in saved:
            setattr(cls, name, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    out = {
        "label": "mutation sensitivity of the runtime property test (§II.26)",
        "n_max": a.n,
        "results": {},
    }
    for name, m in (
        ("m1_clock", m1_clock),
        ("m2_generation", m2_generation),
        ("m3_post_loss_canary", m3_post_loss_canary),
    ):
        out["results"][name] = run(m, a.n, 9_000_000)
        print(name, out["results"][name], flush=True)
    out["unmutated"] = {"detected": False}
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
