"""Adversarial property test of the production runtime contract (PREREG_WME §II.26, §II.27; Board A freeze
re-checks 1 and 2).

``hypothesis`` is not installed in the shared environment, so a seeded random sequence generator drives
``ProductionGuard`` and ``harness.run_units`` through random sequences of public operations: open, close, declare_lost
(of any known instance), relaunch, contract transition (including back to an earlier digest), reconstruction (also
with an old instance id), canary pass/fail, unit success/technical failure/exhaustion into the fallback, direct
resolve, authoritative recovery from a clock halt, arbitrary clock moves including backward jumps (also during a
unit), a takeover by a new owner during a unit, and torn log writes with their audited repair.

Every guard object ever constructed stays in a pool; any operation may be issued through a random **older** object.
Each such call must be refused (``StaleGuard``) without changing a byte of the log.

The invariants are computed from an independent replay of the raw log bytes (:class:`Model`: its own JSON parsing,
sequence, hash-chain and head checks, and its own lifecycle, contract, retry, eligibility and charge rules; it shares
no code with ``wsp.wm.runguard``). After every step:

* **I0** the log is gap-free and duplicate-free in sequence, its hash chain verifies, and the head matches;
* **I1** total charged instance time is non-negative, monotone non-decreasing, and at least each instance's own true
  elapsed time (whenever the clock is not halted);
* **I2** no unit runs under a stale owner, without an active instance, while halted, above the cap, beyond 3
  attempts, or after the unit already has an eligible result or a fallback (checked inside the unit function);
* **I3** an attempt's eligibility, once false (superseded, ineligible, lost), never becomes true; pending never
  returns; nothing changes after the instance's closure except a later supersession;
* **I4** an eligible attempt has a start and a closing canary from the same instance, generation and contract,
  both equal to that generation's reference, before the instance's closure, and its stored payload matches its
  logged digest;
* **I5** every result returned (``resolve`` and ``run_units``) equals the model's first eligible attempt or fallback,
  never changes within a contract generation, and a fallback payload equals its registered digest;
* **I6** a fresh verified replay reproduces the implementation's state, and the implementation's statuses, counters
  and lifecycle equal the model's.

The default suite runs ``N_FAST`` sequences; ``-m slow`` runs ``N_SLOW`` (registered: >= 5,000).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

import pytest

from wsp.wm import harness as H
from wsp.wm import runguard as RG

N_FAST = 60
N_SLOW = 5000
STEPS = 40
H1 = 3600.0
REF = "a" * 64
DIGESTS = ("c" * 64, "d" * 64, "e" * 64)
UNITS = [({"u": i}, f"u{i}") for i in range(4)]
KEYS = [H.unit_key(r, "t" * 64, "p") for r, _ in UNITS]
EST = 0.05
CAP = 12.0
ATTEMPTS = 3
FALLBACK = {"winner": "base"}
TORN = b'{"seq": 0, "type": "canary", "sh'
CHECK_CHAIN = (
    True  # I0; the mutation script switches it off to test the remaining invariants on their own
)
ALLOWED = (
    RG.InstanceNotOpen,
    RG.CanaryMismatch,
    RG.CapExceeded,
    RG.PhasePause,
    RG.RetryExhausted,
    RG.ClockInconsistent,
    RG.ContractMismatch,
    RG.UnitResolved,
    ValueError,
)
STALE_OPS = ("open", "close", "lost", "relaunch", "canary", "unit", "resolve", "begin")


def _att(d):
    return RG.LaunchAttestation(True, True, REF, d, ["docs/reviews/wme/wme-ph100-2026-11-17.json"])


def _tech(exc):
    return "out_of_memory" if isinstance(exc, MemoryError) else None


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ------------------------------------------------------------------ independent reference model


class Model:
    """An independent replay of the raw study log (no ``wsp.wm.runguard`` code)."""

    def __init__(self, state: Path):
        log = state / "study.jsonl"
        data = log.read_bytes() if log.exists() else b""
        assert not data or data.endswith(b"\n"), "I0 torn log outside a tear"
        self.recs, prev = [], "0" * 64
        for i, line in enumerate(data.splitlines()):
            r = json.loads(line)
            if CHECK_CHAIN:
                assert r["seq"] == i + 1, (
                    "I0 duplicate or missing sequence number",
                    i + 1,
                    r["seq"],
                )
                assert r["prev"] == prev, ("I0 hash chain", i + 1)
                assert r["hash"] == _sha({k: v for k, v in r.items() if k != "hash"}), (
                    "I0 hash",
                    i + 1,
                )
            prev = r["hash"]
            self.recs.append(r)
        head = state / "study.jsonl.head"
        if CHECK_CHAIN and head.exists():
            h = json.loads(head.read_text())
            assert h["seq"] <= len(self.recs) and (
                h["seq"] == 0 or self.recs[h["seq"] - 1]["hash"] == h["hash"]
            ), "I0 head"
        self.owner, self.ownerless = None, False
        self.contracts = []  # [(digest, reference)]
        self.inst = {}
        self.max_t = float("-inf")
        self.halted, self.halt_active = False, []
        self.unit_s = 0.0
        self.starts, self.oks, self.fallback, self.fb_reg = {}, {}, {}, {}
        self.disposed, self.canaries = set(), []
        for r in self.recs:
            t = r["type"]
            if t in ("tick", "instance_open", "instance_close"):
                self.max_t = max(self.max_t, r["t"])
            if t == "owner_epoch":
                self.owner, self.ownerless = r["epoch"], False
            elif t == "log_repaired":
                self.ownerless = True
            elif t in ("attestation", "contract_transition"):
                self.contracts.append((r["digest"], r["canary_reference"]))
            elif t == "fallback_registered":
                self.fb_reg.setdefault(r["phase"], r["payload_sha256"])
            elif t == "clock_halt":
                self.halted, self.halt_active = True, r["active"]
            elif t == "clock_resolved":
                self.halted = False
            elif t == "unit_time":
                self.unit_s += r["gpu_s"]
            elif t.startswith("instance_") and t != "instance_resume":
                d = self.inst.setdefault(
                    r["instance"],
                    {
                        "open": None,
                        "close": None,
                        "auth": 0.0,
                        "authd": False,
                        "closure": None,
                        "lost": False,
                    },
                )
                if t == "instance_open" and d["open"] is None:
                    d["open"] = (r["t"], r["gpus"])
                elif t == "instance_close" and d["close"] is None:
                    d["close"] = r["t"]
                elif t == "instance_authoritative":
                    d["authd"] = True
                    d["auth"] = max(d["auth"], (r["terminated_at"] - r["launched_at"]) * r["gpus"])
                elif t == "instance_lost":
                    d["lost"] = True
                if (
                    t in ("instance_close", "instance_authoritative", "instance_lost")
                    and d["closure"] is None
                ):
                    d["closure"] = r["seq"]
            elif t == "canary":
                self.canaries.append(r)
            elif t == "attempt_start":
                self.starts.setdefault(r["unit"], []).append(r)
            elif t == "attempt_ok":
                self.oks.setdefault(r["unit"], []).append(r)
            elif t == "attempt_disposed":
                self.disposed.add(r["attempt_seq"])
            elif t == "fallback":
                self.fallback.setdefault(r["unit"], r)

    def current(self, epoch) -> bool:
        return not self.ownerless and epoch == self.owner

    @property
    def gen(self):
        return len(self.contracts) - 1

    def inst_state(self, i):
        d = self.inst.get(i)
        if d is None:
            return "unknown"
        if d["close"] is not None or d["authd"]:
            return "closed"
        return "active" if d["open"] is not None else "unknown"

    def active(self, i) -> bool:
        return self.inst_state(i) == "active" and not self.inst[i]["lost"]

    def bracket(self, a):
        """(start canary, closing canary) of an attempt under the registered rule, from raw records."""
        closure = self.inst.get(a["instance"], {}).get("closure")
        cans = [
            c
            for c in self.canaries
            if c["instance"] == a["instance"]
            and c["generation"] == a["generation"]
            and c["contract"] == a["contract"]
            and (closure is None or c["seq"] < closure)
        ]
        before = [c for c in cans if c["seq"] < a["start_seq"]]
        after = [c for c in cans if c["seq"] > a["seq"]]
        return (before[-1] if before else None), (after[0] if after else None), closure

    def status(self, a) -> str:
        if a["generation"] != self.gen or a["contract"] != self.contracts[-1][0]:
            return "superseded_contract"
        if a["seq"] in self.disposed:
            return "retained_ineligible"
        ref = self.contracts[a["generation"]][1]
        start, end, closure = self.bracket(a)
        if start is None or start["sha"] != ref:
            return "retained_ineligible"
        if end is None:
            return "pending" if closure is None else "retained_ineligible"
        return "eligible" if end["sha"] == ref else "retained_ineligible"

    def expected(self, unit, store):
        for a in self.oks.get(unit, []):
            if self.status(a) == "eligible":
                return store.read(a["attempt_key"])
        fb = self.fallback.get(unit)
        if fb is not None:
            return {
                "fallback": True,
                "value": fb["payload"],
                "payload_sha256": fb["payload_sha256"],
            }
        return None

    def charges(self, now) -> dict:
        out = {}
        for i, d in self.inst.items():
            local = 0.0
            if d["open"] is not None:
                t0, g = d["open"]
                local = ((d["close"] if d["close"] is not None else now) - t0) * g
            if d["open"] is not None or d["authd"]:
                out[i] = max(local, d["auth"]) / 3600
        return out


# ------------------------------------------------------------------ the world


class World:
    def __init__(self, seed, root: Path, strict: bool = True):
        self.rng = random.Random(seed)
        self.root, self.state, self.strict = root, root / "state", strict
        self.T = self.C = 1000.0  # true time and the guards' clock (s)
        self.digest = DIGESTS[0]
        self.n_inst = 0
        self.true = {}  # instance -> [true_open, true_close or None]
        self.store = H.UnitStore(root / "units")
        self.pool = []
        self.g = self.make("i0")
        self.prev_total = 0.0
        self.status_hist, self.final_at_closure, self.resolved = {}, {}, {}
        self.mode, self.torn, self.exec_g, self.takeover = {}, False, None, False
        self.cov = dict.fromkeys(
            (
                "units_run",
                "stale_refusals",
                "takeovers_during_unit",
                "clock_back_during_unit",
                "torn_detected",
                "torn_repaired",
                "results_returned",
            ),
            0,
        )

    def clock(self):
        return self.C

    def make(self, instance, transition=None, digest=None):
        g = RG.ProductionGuard(
            self.state,
            "e2",
            {"e2": 1e6},
            _att(digest or self.digest),
            instance,
            EST,
            technical=_tech,
            fallback_value=FALLBACK,
            clock=self.clock,
            transition=transition,
        )
        self.pool.append(g)
        return g

    def model(self) -> Model:
        return Model(self.state)

    def log_bytes(self) -> bytes:
        p = self.state / "study.jsonl"
        return p.read_bytes() if p.exists() else b""

    # ------------------------------------------------------------------ the unit function (I2)
    def fn(self, recipe):
        g, m = self.exec_g, self.model()
        k = H.unit_key(recipe, "t" * 64, "p")
        assert m.current(g.epoch), "I2 a unit ran under a stale owner"
        assert m.active(g.instance), ("I2 a unit ran without an active instance", g.instance)
        assert not m.halted, "I2 a unit ran while the clock is halted"
        assert len(m.starts.get(k, [])) <= ATTEMPTS, "I2 more than 3 attempts"
        assert m.starts[k][-1]["instance"] == g.instance
        assert m.expected(k, self.store) is None, "I2 a resolved unit was recomputed"
        total = max(m.unit_s / 3600, sum(m.charges(m.max_t).values()))
        assert total + EST <= CAP + 1e-9, ("I2 above the cap", total)
        self.cov["units_run"] += 1
        x = self.rng.random()
        if x < 0.05:  # the clock moves backward while the unit runs
            self.T += 1.0
            self.C -= self.rng.choice([1.0, 2 * H1])
            self.cov["clock_back_during_unit"] += 1
        elif x < 0.10:
            self.T += 0.1 * H1
            self.C += 0.1 * H1
        elif x < 0.17:  # another process takes ownership while the unit runs
            self.takeover = True
            self.cov["takeovers_during_unit"] += 1
            self.g = self.make(g.instance)
            if self.rng.random() < 0.5:
                try:
                    self.do("lost", self.g, target=g.instance)
                except ALLOWED:
                    pass
        if self.mode.get(recipe["u"]) == "fail":
            raise MemoryError("OOM")
        return {"u": recipe["u"], "draw": self.rng.random()}

    # ------------------------------------------------------------------ operations
    def do(self, op, g, target=None):
        r = self.rng
        if op == "open":
            was = self.model().inst_state(g.instance)
            g.open_instance()
            if was == "unknown":
                self.true[g.instance] = [self.T, None]
        elif op == "close":
            g.close_instance()
            if g.instance in self.true:
                self.true[g.instance][1] = self.T
        elif op == "lost":
            inst = target or r.choice(sorted(self.true) or [g.instance])
            active = self.model().inst_state(inst) == "active"
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
            g.relaunch("infrastructure", f"i{self.n_inst}")
        elif op == "canary":
            g.canary(REF)
        elif op == "canary_bad":
            g.canary("b" * 64)
        elif op == "begin":
            g.begin(r.choice(KEYS), "direct")
        elif op == "resolve":
            k = r.choice(KEYS)
            m = self.model()
            v = g.resolve(self.store, k)
            self.check_returned(m, k, v)
        elif op == "unit":
            recipe, label = r.choice(UNITS)
            self.mode[recipe["u"]] = r.choice(["ok", "ok", "fail"])
            self.run_unit(g, recipe, label)
        elif op == "cycle":  # open if needed, start canary, a unit, closing canary (pass or fail)
            if self.model().inst_state(g.instance) == "unknown":
                self.do("open", g)
            g.canary(REF)
            recipe, label = r.choice(UNITS)
            self.mode[recipe["u"]] = r.choice(["ok", "ok", "ok", "fail"])
            try:
                self.run_unit(g, recipe, label)
            finally:
                self.T += 60.0
                self.C += 60.0
                if not self.takeover:
                    g.canary(REF if r.random() < 0.85 else "b" * 64)
        elif op == "recover":
            m = self.model()
            if m.halted:
                for inst in m.halt_active:
                    to = self.true.get(inst, [self.T, None])[0]
                    g.reconcile_instance(to, self.T, 1, "provider API", instance=inst)
                    if inst in self.true and self.true[inst][1] is None:
                        self.true[inst][1] = self.T
                self.C = max(self.C, self.T, m.max_t)
                g.ledger.clear_clock_halt("ntp", "re-synchronised")
        else:  # pragma: no cover
            raise KeyError(op)

    def run_unit(self, g, recipe, label):
        m = self.model()
        k = H.unit_key(recipe, "t" * 64, "p")
        want = None if m.halted else m.expected(k, self.store)
        self.exec_g, self.takeover = g, False
        out = H.run_units(self.store, [(recipe, label)], self.fn, "t" * 64, "p", guard=g)
        v = out[label]
        if want is not None:
            assert v == want, ("I5 run_units returned another result", v, want)
            self.check_returned(m, k, v)
        elif isinstance(v, dict) and v.get("fallback"):
            self.check_returned(self.model(), k, v)
        else:
            assert v["status"] == "pending_canary", v

    def check_returned(self, m, k, v):
        """I5: a returned result equals the model's, is stable within the generation, and fallbacks verify."""
        if v is None:
            assert m.expected(k, self.store) is None
            return
        assert v == m.expected(k, self.store), ("I5 returned != model", v)
        if isinstance(v, dict) and v.get("fallback"):
            assert v["payload_sha256"] == m.fb_reg["e2"] == _sha(FALLBACK) == _sha(v["value"])
        key = (m.gen, k)
        assert self.resolved.setdefault(key, v) == v, ("I5 result changed", self.resolved[key], v)
        self.cov["results_returned"] += 1

    def stale_call(self, g, op):
        before = self.log_bytes()
        try:
            self.do(op, g)
        except RG.StaleGuard:
            self.cov["stale_refusals"] += 1
        except ALLOWED:
            assert not self.strict, ("a stale object was not fenced", op)
        else:
            assert not self.strict, ("a stale object acted", op)
        if self.strict:
            assert self.log_bytes() == before, ("a stale object changed the log", op)

    def step(self):
        r = self.rng
        if self.torn:
            before = self.log_bytes()
            g = r.choice(self.pool)
            probes = (
                lambda: g.canary(REF),
                lambda: g.validate(),
                lambda: g.resolve(self.store, KEYS[0]),
                lambda: g.open_instance(),
                lambda: g.begin(KEYS[0], "u0"),
                lambda: g.ledger.tick(),
                lambda: self.make(g.instance),
            )
            try:
                r.choice(probes)()
            except RG.LogCorrupt:
                self.cov["torn_detected"] += 1
            else:
                raise AssertionError("an operation on a torn log was not refused")
            assert self.log_bytes() == before
            if r.random() < 0.7:
                RG.repair_torn_tail(self.state, "operator", "crash during an append")
                self.torn = False
                self.cov["torn_repaired"] += 1
                self.g = self.make(self.g.instance)
            return "torn"
        if self.model().halted and r.random() < 0.7:
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
                    "old_id",
                    "canary",
                    "canary_bad",
                    "unit",
                    "cycle",
                    "resolve",
                    "stale",
                    "tear",
                ],
                weights=[14, 3, 2, 1, 8, 2, 2, 2, 3, 3, 2, 10, 2, 13, 20, 3, 12, 1],
            )[0]
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
            elif op == "transition":
                d = r.choice([x for x in DIGESTS if x != self.digest])
                tr = RG.ContractTransition("driver_or_cuda", "driver change", "abcdef1", True)
                self.g = self.make(self.g.instance, transition=tr, digest=d)
                self.digest = d
            elif op == "reconstruct":
                self.g = self.make(self.g.instance)
            elif op == "old_id":
                if self.true:
                    self.g = self.make(r.choice(sorted(self.true)))
            elif op == "stale":
                old = self.pool[:-1]
                if old:
                    self.stale_call(r.choice(old), r.choice(STALE_OPS))
            elif op == "tear":
                with open(self.state / "study.jsonl", "ab") as f:
                    f.write(TORN)
                self.torn = True
            else:
                self.do(op, self.g)
        except ALLOWED:
            pass
        except RG.StaleGuard:
            assert self.takeover, ("the current owner was refused", op)
        return op

    # ------------------------------------------------------------------ invariants
    def check(self):
        if self.torn:
            with pytest.raises(RG.LogCorrupt):
                RG.read_study(self.state)
            return
        m = self.model()  # I0
        g = self.g
        assert m.current(g.epoch), "the world's owner is not current"
        # I1 (the owner reads its clock; a backward reading halts)
        if not m.halted:
            try:
                g.ledger.tick()
            except RG.ClockInconsistent:
                pass
            m = self.model()
        if not m.halted:
            now = m.max_t
            charges = m.charges(now)
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
                assert charges.get(inst, 0.0) + 1e-6 >= true_h, ("I1 below true elapsed", inst)
            impl = g.ledger.instance_charges(now=now)
            assert impl.keys() == charges.keys() and all(
                abs(impl[i] - charges[i]) < 1e-9 for i in charges
            ), ("I6 charges", impl, charges)
        # I6: a fresh verified replay equals the implementation's state; the model agrees with both
        g.events  # noqa: B018 - re-read under the lock
        st = g.log.state
        assert RG.read_study(self.state).snapshot() == st.snapshot(), "I6 replay"
        assert st.n == len(m.recs) and st.generation == m.gen and st.halted == m.halted
        for i in m.inst:
            assert (
                st.instance_state(i) == m.inst_state(i) and (i in st.lost) == m.inst[i]["lost"]
            ), i
        assert st.attempts == {k: len(v) for k, v in m.starts.items()}, "I6 attempt counts"
        for k in KEYS:
            us = g.unit_status(k)
            assert us["attempts"] == [(a["attempt_key"], m.status(a)) for a in m.oks.get(k, [])], (
                "I6 status"
            )
            assert us["fallback"] == (k in m.fallback)
        # I3, I4
        for k, oks in m.oks.items():
            for a in oks:
                s = m.status(a)
                hist = self.status_hist.setdefault(a["seq"], [])
                if hist:
                    if hist[-1] != "pending":
                        assert s != "pending", ("I3 pending returned", hist, s)
                    if any(x in ("retained_ineligible", "superseded_contract") for x in hist):
                        assert s != "eligible", ("I3 eligibility revived", hist, s)
                closure = m.inst.get(a["instance"], {}).get("closure")
                if closure is not None and closure > a["seq"]:
                    assert s != "pending", (
                        "I3 pending on a closed or lost instance",
                        a["instance"],
                    )
                    final = self.final_at_closure.setdefault(a["seq"], s)
                    assert s in (final, "superseded_contract"), (
                        "I3 changed after closure",
                        final,
                        s,
                    )
                hist.append(s)
                if s == "eligible":
                    start, end, _ = m.bracket(a)
                    ref = m.contracts[a["generation"]][1]
                    assert a["generation"] == m.gen and a["contract"] == m.contracts[-1][0]
                    assert start["sha"] == end["sha"] == ref, "I4 bracket"
                    assert start["seq"] < a["start_seq"] < a["seq"] < end["seq"], "I4 order"
                    assert _sha(self.store.read(a["attempt_key"])) == a["payload_sha256"], (
                        "I4 payload"
                    )
        # I5
        if not m.halted:
            for k in KEYS:
                self.check_returned(m, k, g.resolve(self.store, k))

    def coverage(self) -> dict:
        m = self.model() if not self.torn else None
        recs = m.recs if m else []
        seen = []
        rollbacks = 0
        for r in recs:
            if r["type"] in ("attestation", "contract_transition"):
                rollbacks += r["type"] == "contract_transition" and r["digest"] in seen
                seen.append(r["digest"])
        return {
            **self.cov,
            "eligible_attempts": sum(1 for h in self.status_hist.values() if "eligible" in h),
            "ineligible_attempts": sum(
                1 for h in self.status_hist.values() if "retained_ineligible" in h
            ),
            "superseded_attempts": sum(
                1 for h in self.status_hist.values() if "superseded_contract" in h
            ),
            "fallbacks": sum(1 for r in recs if r["type"] == "fallback"),
            "disposed": sum(1 for r in recs if r["type"] == "attempt_disposed"),
            "lost": sum(1 for r in recs if r["type"] == "instance_lost"),
            "transitions": sum(1 for r in recs if r["type"] == "contract_transition"),
            "rollbacks_to_earlier_digest": rollbacks,
            "owners": sum(1 for r in recs if r["type"] == "owner_epoch"),
            "clock_halts": sum(1 for r in recs if r["type"] == "clock_halt"),
            "clock_recoveries": sum(1 for r in recs if r["type"] == "clock_resolved"),
            "cap_reached": int(self.prev_total + EST > CAP),
        }


def run_sequences(n, seed0=20261119, strict=True):
    ops, cov = {}, {}
    for s in range(n):
        tmp = Path(tempfile.mkdtemp(prefix="wme-rg-"))
        try:
            w = World(seed0 + s, tmp, strict=strict)
            for _ in range(STEPS):
                op = w.step()
                ops[op] = ops.get(op, 0) + 1
                w.check()
            for k, v in w.coverage().items():
                cov[k] = cov.get(k, 0) + v
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return ops, cov


@pytest.fixture
def small_cap(monkeypatch):
    monkeypatch.setattr(RG, "CAP_GPU_H", CAP)
    monkeypatch.setattr(RG, "CAP_USD", 1e9)


FAST_REQUIRED = (
    "units_run",
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
    "stale_refusals",
    "takeovers_during_unit",
    "torn_detected",
    "torn_repaired",
    "results_returned",
)


def test_runtime_contract_properties_fast(small_cap):
    ops, cov = run_sequences(N_FAST)
    assert sum(ops.values()) == N_FAST * STEPS
    for k in FAST_REQUIRED:  # the cap is reached in the slow run and the named cap tests
        assert cov[k] > 0, (k, cov)


@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_runtime_contract_properties_5000_sequences(small_cap):
    """The registered size; scripts/wme_v2i_runguard_property.py runs the same sequences in parallel with a report."""
    ops, cov = run_sequences(N_SLOW, seed0=7_000_000)
    assert sum(ops.values()) == N_SLOW * STEPS
    assert all(v > 0 for v in cov.values()), cov
