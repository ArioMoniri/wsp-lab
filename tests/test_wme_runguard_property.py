"""Adversarial property test of the production runtime contract (PREREG_WME §II.26; Board A freeze re-check).

``hypothesis`` is not installed in the shared environment, so a seeded random sequence generator drives
``ProductionGuard`` and ``harness.run_units`` through random sequences of public operations: open, close, declare_lost
(of any known instance), relaunch, contract transition (including back to an earlier digest), reconstruction from the
log (also with stale instance ids), canary pass/fail, unit success/technical failure/exhaustion into the fallback,
authoritative recovery from a clock halt, and arbitrary clock moves including backward jumps. After every step it
asserts:

* **I1** total charged instance time is non-negative, monotone non-decreasing, and at least each instance's own true
  elapsed time (whenever the clock is not halted);
* **I2** no unit runs when the cap would be exceeded or no instance is active (checked inside the unit function);
* **I3** an attempt's eligibility, once false (superseded, ineligible, lost), never becomes true; pending never
  returns;
* **I4** an eligible attempt has a start and a closing canary from the same instance, generation and contract, both
  equal to that generation's reference, before the instance closed;
* **I5** within a contract generation, the resolved result of a unit never changes once returned, and a fallback
  payload equals its registered digest;
* **I6** reconstruction from the log reproduces identical state.

The default suite runs ``N_FAST`` sequences; ``-m slow`` runs ``N_SLOW`` (registered: >= 5,000).
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG

N_FAST = 80
N_SLOW = 5000
STEPS = 40
H1 = 3600.0
REF = "a" * 64
DIGESTS = ("c" * 64, "d" * 64, "e" * 64)
UNITS = [({"u": i}, f"u{i}") for i in range(4)]
EST = 0.05
CAP = 12.0
ALLOWED = (
    RG.InstanceNotOpen,
    RG.CanaryMismatch,
    RG.CapExceeded,
    RG.PhasePause,
    RG.RetryExhausted,
    RG.ClockInconsistent,
    RG.ContractMismatch,
    ValueError,
)


def _att(d):
    return RG.LaunchAttestation(True, True, REF, d, ["docs/reviews/wme/wme-ph100-2026-11-17.json"])


def _tech(exc):
    return "out_of_memory" if isinstance(exc, MemoryError) else None


class World:
    def __init__(self, seed, root: Path):
        self.rng = random.Random(seed)
        self.root = root
        self.T = 1000.0  # true time (s)
        self.C = 1000.0  # the guard's clock (may lag after backward jumps)
        self.digest = DIGESTS[0]
        self.instance = "i0"
        self.n_inst = 0
        self.true = {}  # instance -> [true_open, true_close or None]
        self.store = H.UnitStore(root / "units")
        self.g = self.make(self.instance)
        self.prev_total = 0.0
        self.status_hist = {}  # attempt seq -> list of statuses
        self.final_at_closure = {}  # attempt seq -> status when its instance closed
        self.resolved = {}  # (generation, unit) -> value
        self.mode = {}

    def clock(self):
        return self.C

    def make(self, instance, transition=None, digest=None):
        return RG.ProductionGuard(
            self.root / "state",
            "e2",
            {"e2": 1e6},
            _att(digest or self.digest),
            instance,
            EST,
            technical=_tech,
            fallback_value={"winner": "base"},
            clock=self.clock,
            transition=transition,
        )

    # ------------------------------------------------------------------ operations
    def fn(self, recipe):
        g = self.g
        led = g.ledger
        # I2: a unit runs only on an active, non-lost instance, with the clock trusted and the cap respected
        assert led.instance_state(g.instance) == "active" and not g._lost(g.instance)
        assert not led.halted()
        assert led.gpu_h() + EST <= CAP + 1e-9
        if self.mode.get(recipe["u"]) == "fail":
            raise MemoryError("OOM")
        return {"u": recipe["u"], "draw": self.rng.random()}

    def step(self):
        r = self.rng
        if self.g.ledger.halted() and r.random() < 0.7:
            op = "recover"
        else:
            op = r.choices(
                [
                    "fwd",
                    "back",
                    "fix",
                    "recover",
                    "open",
                    "close",
                    "lost",
                    "relaunch",
                    "transition",
                    "reconstruct",
                    "stale",
                    "canary",
                    "canary_bad",
                    "unit",
                    "cycle",
                ],
                weights=[14, 3, 2, 1, 8, 2, 2, 2, 3, 4, 2, 10, 2, 15, 20],
            )[0]
        g = self.g
        try:
            if op == "fwd":
                dt = r.choice([0.02, 0.05, 0.1, 0.3, 0.3, 1.0, 3.0]) * H1
                self.T += dt
                self.C += dt
            elif op == "back":
                self.T += 1.0
                self.C -= r.choice([1.0, 600.0, 2 * H1, 50 * H1])
            elif op == "fix":
                self.C = max(self.C, self.T)
            elif op == "recover":
                led = g.ledger
                if led.halted():
                    halt = [x for x in led.rows if x.get("kind") == "clock_halt"][-1]
                    for inst in halt.get("active", []):
                        to = self.true.get(inst, [self.T, None])[0]
                        g.reconcile_instance(to, self.T, 1, "provider API", instance=inst)
                        if inst in self.true and self.true[inst][1] is None:
                            self.true[inst][1] = self.T
                    self.C = max(self.C, self.T, led.max_tick())
                    led.clear_clock_halt("ntp", "re-synchronised")
            elif op == "open":
                was = g.ledger.instance_state(g.instance)
                g.open_instance()
                if was == "unknown":
                    self.true[g.instance] = [self.T, None]
            elif op == "close":
                g.close_instance()
                self.true[g.instance][1] = self.T
            elif op == "lost":
                known = sorted(self.true) or [g.instance]
                inst = r.choice(known)
                active = g.ledger.instance_state(inst) == "active"
                to = self.true.get(inst, [self.T, None])[0]
                g.declare_lost(
                    inst,
                    "preemption_or_capacity",
                    "preempted",
                    "abcdef1",
                    True,
                    launched_at=to,
                    terminated_at=self.T,
                )
                if active and inst in self.true:
                    self.true[inst][1] = self.T
            elif op == "relaunch":
                self.n_inst += 1
                new = f"i{self.n_inst}"
                g.relaunch("infrastructure", new)
                self.instance = new
            elif op == "transition":
                d = r.choice([x for x in DIGESTS if x != self.digest])
                tr = RG.ContractTransition("driver_or_cuda", "driver change", "abcdef1", True)
                self.g = self.make(g.instance, transition=tr, digest=d)
                self.digest = d
            elif op == "reconstruct":
                self.g = self.make(g.instance)
            elif op == "stale":
                known = sorted(self.true)
                if known:
                    self.g = self.make(r.choice(known))
            elif op == "canary":
                g.canary(REF)
            elif op == "canary_bad":
                g.canary("b" * 64)
            elif op == "unit":
                recipe, label = r.choice(UNITS)
                self.mode[recipe["u"]] = r.choice(["ok", "ok", "fail"])
                H.run_units(self.store, [(recipe, label)], self.fn, "t" * 64, "p", guard=self.g)
            elif (
                op == "cycle"
            ):  # open if needed, start canary, a unit, closing canary (pass or fail)
                if g.ledger.instance_state(g.instance) == "unknown":
                    g.open_instance()
                    self.true[g.instance] = [self.T, None]
                g.canary(REF)
                recipe, label = r.choice(UNITS)
                self.mode[recipe["u"]] = r.choice(["ok", "ok", "ok", "fail"])
                try:
                    H.run_units(self.store, [(recipe, label)], self.fn, "t" * 64, "p", guard=self.g)
                finally:
                    self.T += 60.0
                    self.C += 60.0
                    g.canary(REF if r.random() < 0.85 else "b" * 64)
        except ALLOWED:
            pass
        return op

    # ------------------------------------------------------------------ invariants
    def check(self):
        g, led = self.g, self.g.ledger
        # I1
        try:
            led.tick()
            charges = led.instance_charges(now=self.C)
        except RG.ClockInconsistent:
            charges = None
        if charges is not None:
            total = sum(charges.values())
            assert all(v >= 0 for v in charges.values()), ("I1 negative charge", charges)
            assert total + 1e-9 >= self.prev_total, (
                "I1 charged time decreased",
                total,
                self.prev_total,
            )
            self.prev_total = total
            for inst, (t0, t1) in self.true.items():
                true_h = ((self.T if t1 is None else t1) - t0) / H1
                assert charges.get(inst, 0.0) + 1e-6 >= true_h, (
                    "I1 below true elapsed",
                    inst,
                    charges.get(inst),
                    true_h,
                )
        # I3, I4 (closure points recomputed independently from the raw events, not through guard helpers)
        closure = {}
        for e in g.events:
            if e["type"] in ("instance_close", "instance_reconciled"):
                closure.setdefault(e.get("target", e["instance"]), e["seq"])
            elif e["type"] == "instance_lost":
                closure.setdefault(e["lost_instance"], e["seq"])
        for a in g.events:
            if a["type"] != "attempt_ok":
                continue
            st = g.attempt_status(a)
            hist = self.status_hist.setdefault(a["seq"], [])
            if hist:
                if hist[-1] != "pending":
                    assert st != "pending", (hist, st)
                if any(s in ("retained_ineligible", "superseded_contract") for s in hist):
                    assert st != "eligible", (hist, st)
            closed = closure.get(a["instance"])
            if closed is not None and closed > a["seq"]:
                assert st != "pending", ("pending on a closed or lost instance", a["instance"])
                final = self.final_at_closure.setdefault(a["seq"], st)
                assert st in (final, "superseded_contract"), (
                    "status changed after closure",
                    final,
                    st,
                )
            hist.append(st)
            if st == "eligible":
                assert a["generation"] == g.generation and a["contract"] == g.contract
                cans = [
                    e
                    for e in g.events
                    if e["type"] == "canary"
                    and e["instance"] == a["instance"]
                    and e["generation"] == a["generation"]
                    and e["contract"] == a["contract"]
                    and (closed is None or e["seq"] < closed)
                ]
                before = [e for e in cans if e["seq"] < a["seq"]]
                after = [e for e in cans if e["seq"] > a["seq"]]
                ref = g.references[a["generation"]]
                assert before and after and before[-1]["sha"] == ref and after[0]["sha"] == ref, (
                    "bracket"
                )
        # I5
        if not led.halted():
            for recipe, _label in UNITS:
                k = H.unit_key(recipe, "t" * 64, "p")
                v = g.resolve(self.store, k)
                if v is None:
                    continue
                if isinstance(v, dict) and v.get("fallback"):
                    assert (
                        v["payload_sha256"]
                        == RG._digest({"winner": "base"})
                        == RG._digest(v["value"])
                    )
                key = (g.generation, k)
                if key in self.resolved:
                    assert self.resolved[key] == v, (self.resolved[key], v)
                else:
                    self.resolved[key] = v
        # I6: a fresh guard built from the log has identical state
        h = self.make(g.instance)
        assert h.generation == g.generation and h.contract == g.contract
        assert json.dumps(h.events, sort_keys=True) == json.dumps(g.events, sort_keys=True)
        assert h.ledger.rows == led.rows and h.ledger.halted() == led.halted()
        assert h.retry.attempts == g.retry.attempts and h.retry.relaunches == g.retry.relaunches
        for recipe, _label in UNITS:
            k = H.unit_key(recipe, "t" * 64, "p")
            assert h.unit_status(k) == g.unit_status(k)


def coverage(w) -> dict:
    ev, rows = w.g.events, w.g.ledger.rows
    return {
        "eligible_attempts": sum(1 for h in w.status_hist.values() if "eligible" in h),
        "ineligible_attempts": sum(1 for h in w.status_hist.values() if "retained_ineligible" in h),
        "superseded_attempts": sum(1 for h in w.status_hist.values() if "superseded_contract" in h),
        "fallbacks": sum(1 for e in ev if e["type"] == "fallback"),
        "disposed": sum(1 for e in ev if e["type"] == "attempt_disposed"),
        "lost": sum(1 for e in ev if e["type"] == "instance_lost"),
        "transitions": sum(1 for e in ev if e["type"] == "contract_transition"),
        "rollbacks_to_earlier_digest": sum(
            1
            for e in ev
            if e["type"] == "contract_transition"
            and e["digest"]
            in {
                x["digest"]
                for x in ev
                if x["type"] in ("attestation", "contract_transition") and x["seq"] < e["seq"]
            }
        ),
        "clock_halts": sum(1 for x in rows if x.get("kind") == "clock_halt"),
        "clock_recoveries": sum(1 for x in rows if x.get("kind") == "clock_resolved"),
        "cap_reached": int(w.prev_total + EST > CAP),
        "units_run": sum(1 for e in ev if e["type"] == "attempt_start"),
    }


def run_sequences(n, seed0=20261119):
    ops, cov = {}, {}
    for s in range(n):
        tmp = Path(tempfile.mkdtemp(prefix="wme-rg-"))
        try:
            w = World(seed0 + s, tmp)
            for _ in range(STEPS):
                op = w.step()
                ops[op] = ops.get(op, 0) + 1
                w.check()
            for k, v in coverage(w).items():
                cov[k] = cov.get(k, 0) + v
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return ops, cov


@pytest.fixture
def small_cap(monkeypatch):
    monkeypatch.setattr(RG, "CAP_GPU_H", CAP)
    monkeypatch.setattr(RG, "CAP_USD", 1e9)


def test_runtime_contract_properties_fast(small_cap):
    ops, cov = run_sequences(N_FAST)
    assert sum(ops.values()) == N_FAST * STEPS
    # the sequences reach the states that matter (the cap is reached in the slow run and the named cap tests)
    for k in (
        "eligible_attempts",
        "ineligible_attempts",
        "superseded_attempts",
        "fallbacks",
        "disposed",
        "lost",
        "transitions",
        "rollbacks_to_earlier_digest",
        "clock_halts",
        "clock_recoveries",
    ):
        assert cov[k] > 0, (k, cov)


@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_runtime_contract_properties_5000_sequences(small_cap):
    """The registered size; scripts/wme_v2h_runguard_property.py runs the same sequences in parallel with a report."""
    ops, cov = run_sequences(N_SLOW, seed0=7_000_000)
    assert sum(ops.values()) == N_SLOW * STEPS
    assert all(v > 0 for v in cov.values()), cov
