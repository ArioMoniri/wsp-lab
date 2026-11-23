"""RULING-wme-c1-bank-halt addenda 9-10 (E2 rate stop under R9.2), v3a.

P2: two memos inside ``e2arms.run_planner`` (an R(E) memo keyed by the primitive-id tuple, a whole-batch prediction
memo keyed by the exact input bytes): the payload equals the unmemoised one without its timing fields, the R-cache
misses and the progress ticks are unchanged, a memo hit returns a copy. The on-instance verification
(``Controller.verify_e2``) re-runs completed units and halts on any timing-free difference.

P3: a ``reprice`` record raises a phase's price (a registered part of it, or the whole phase) by the factor; the
1.25x pause, R8 and the projection use the re-priced price. P4: a chair cap generation with rule "P4" whose ledger
basis reproduces projection x 1.10 (never beyond $1,000). CPU only; fixtures."""

import dataclasses
import hashlib
import importlib.util
import itertools
import json
import types
from pathlib import Path

import numpy as np
import pytest

from wsp.wm import adapter9b as AD
from wsp.wm import e2 as E2
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import e2arms as EA
from wsp.wm.c1 import field as F
from wsp.wm.c1 import fieldrepair as FR
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import study as SD

ROOT = Path(__file__).resolve().parents[1]
REF = "a" * 64
ATTEST = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


# ------------------------------------------------------------------ P2 fixtures: a planner environment

WRITERS = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(4)]


class Act:
    family = "scale"

    def __init__(self, tensor, v):
        self.tensor, self.v = tensor, v

    def key(self):
        return (self.tensor, self.v)

    def param(self, name):
        return None


class Sess:
    def __init__(self):
        self.d = {}

    def apply(self, a):
        self.d[a.tensor] = self.d.get(a.tensor, 0.0) + a.v

    def displacement(self, name):
        import torch

        return torch.full((2, 2), float(self.d.get(name, 0.0)))

    def restore(self):
        self.d = {}


class Field:
    def __init__(self):
        self.param_group = {w: (0, {"factors": False}) for w in WRITERS}

    def contract_tensor(self, name, D, objs):
        s = float(D.sum())
        return {o: (s * (1 + i) / 7.0 - 0.01 * i, None) for i, o in enumerate(objs)}

    def contract_embed(self, name, D, objs):
        return {o: (float(D.sum()) * 0.5, None) for o in objs}


def _h(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256(json.dumps(parts, default=str).encode()).digest()[:8], "little"
    )


class Lib:
    """``Library.bundles``: each primitive id is one step of one deterministic action."""

    def __init__(self):
        self.index = E2.library_index(EA.OBJ)

    def bundles(self, pids):
        return [
            [Act(WRITERS[_h("t", p) % len(WRITERS)], ((_h("v", p) % 17) - 8) / 64.0)] for p in pids
        ]


class World:
    def __init__(self):
        self.n = 0

    def evaluate(self, pids):
        self.n += 1
        g = np.random.default_rng(_h("eval", list(pids)))
        return {
            "gain": float(g.normal(0.0, 0.1)),
            "gain_se": 0.05,
            "guards_pass": bool(g.random() > 0.3),
            "rel_norm": float(g.random()),
            "theta_hash": "t" * 64,
            "seconds": float(self.n) * 0.37,  # a wall-clock field: differs between runs
            "guard_means": {},
            "guard_bounds": {},
        }


class Model:
    """A deterministic ensemble stand-in with ``predict_batch(trajs)`` (V0RArm's batched path)."""

    def __init__(self, k, seed):
        self.members = [0, 1, 2]
        self.k, self.seed, self.calls = k, seed, 0

    def predict_batch(self, trajs):
        self.calls += 1
        X = np.stack([t["x"] for t in trajs])
        mem = []
        for m in self.members:
            W = np.random.default_rng(self.seed * 10 + m).normal(0, 0.3, (X.shape[-1], self.k))
            mem.append(np.tanh(X @ W))
        mem = np.stack(mem)
        return {"mean": mem.mean(0), "sigma": mem.std(0), "members": mem}


def make_env():
    spec = F.spec_9b()
    m = types.SimpleNamespace(sess=Sess(), embed="model.embed_tokens.weight", at_base=lambda: None)
    rc = F.RCache(m, Field())
    env = EA.Env.__new__(EA.Env)
    env.m, env.world, env.lib, env.rc = m, World(), Lib(), rc
    env.lib_index = env.lib.index
    base = types.SimpleNamespace(payload={"obs": np.linspace(-1, 1, 5)})
    env.fitted = {"wm": Model(spec.n_obj, 1), "static": Model(spec.n_obj, 2), "base_obs": base}
    rows = np.random.default_rng(0).normal(0, 0.05, (40, spec.dim))
    env.support = AD.MahalanobisSupport(rows)
    env.q_hats, env.mode = {"wm": 1.0, "static": 1.0}, "ranks"
    env.c1_manifest = env.e2_items = None
    env.ticks = []
    env.tick = lambda e, **kw: env.ticks.append(e)
    return env


@pytest.fixture
def small_planner(monkeypatch):
    """The production planner with a smaller search (the memos do not depend on the configuration)."""
    monkeypatch.setattr(
        AD, "PLANNER_CONFIG", dataclasses.replace(AD.PLANNER_CONFIG, max_actions=2, inner_budget=12)
    )


def run(arm, memo):
    env = make_env()
    out = EA.run_planner(env, arm, EA.episode(0), 8, memo=memo)
    return env, out


@pytest.mark.parametrize("arm", ["wm_e", "statnn", "field_ranked"])
def test_p2_the_memoised_payload_equals_the_unmemoised_one_without_timing_fields(
    arm, small_planner
):
    env0, plain = run(arm, memo=False)
    env1, memo = run(arm, memo=True)
    assert len(plain["evaluated"]) == 8
    assert FR.timing_free(memo) == FR.timing_free(plain)
    assert FR.timing_free_sha(memo) == FR.timing_free_sha(plain)
    assert (
        memo["r_cache"]["misses"] == plain["r_cache"]["misses"] > 0
    )  # every first visit still goes to RCache
    assert (
        env1.ticks == env0.ticks and env1.ticks.count("e2_r_of") > 0
    )  # one progress tick per R(E) call, as before
    assert env1.rc.hits < env0.rc.hits  # the repeats no longer reach the cache


def test_p2_the_memos_skip_repeated_work(small_planner):
    env0, _ = run("wm_e", memo=False)
    env1, _ = run("wm_e", memo=True)
    assert env1.fitted["wm"].calls < env0.fitted["wm"].calls


def test_p2_r_of_memo_hit_returns_a_copy_and_still_ticks():
    calls, ticks = [], []
    env = types.SimpleNamespace(
        r_of_pids=lambda pids: (calls.append(list(pids)), np.array([1.0, -0.0, 3.5]))[1],
        tick=lambda e, **kw: ticks.append(e),
    )
    r_of = EA.memo_r_of(env, lambda a, d: f"{a}@{d}")
    first = r_of([("x", 1), ("y", 2)])
    first[0] = 99.0  # a caller mutating the first result does not reach the memo
    second = r_of([("x", 1), ("y", 2)])
    assert calls == [["x@1", "y@2"]]
    assert ticks == ["e2_r_of"]  # the hit ticks (the miss ticks inside Env.r_of_pids)
    assert second.tobytes() == np.array([1.0, -0.0, 3.5]).tobytes() and second.dtype == np.float64
    second[1] = 7.0
    assert r_of([("x", 1), ("y", 2)]).tobytes() == np.array([1.0, -0.0, 3.5]).tobytes()
    r_of([("y", 2), ("x", 1)])  # another order is another key (R(E) of an ordered composition)
    assert len(calls) == 2


def test_p2_batch_memo_is_keyed_by_exact_bytes_and_returns_copies():
    calls = []

    def run_(trajs):
        calls.append(len(trajs))
        return {"mean": np.stack([t["x"].sum(-1) for t in trajs]), "sigma": np.zeros(len(trajs))}

    f = EA.memo_predict(run_)
    t = lambda v: {"horizon": 1, "x": np.array([[0.0, v], [1.0, 2.0]]), "e": np.zeros((1, 2))}
    a = f([t(1.0), t(2.0)])
    a["mean"][0] = 123.0
    b = f([t(1.0), t(2.0)])
    assert len(calls) == 1 and b["mean"][0].tolist() == [1.0, 3.0]
    b["sigma"][0] = 5.0
    assert f([t(1.0), t(2.0)])["sigma"].tolist() == [0.0, 0.0]
    f([t(2.0), t(1.0)])  # row order
    f([t(1.0)])  # batch size
    f([t(-0.0), t(2.0)])  # bytes, not value equality
    f([{**t(1.0), "x": t(1.0)["x"].astype(np.float32)}, t(2.0)])  # dtype
    assert len(calls) == 5


# ------------------------------------------------------------------ P2: the on-instance verification


def ctl_module():
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_v3a", ROOT / "scripts/wme_c1_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def guard(state, clk=None, phase="confirmatory", priced=None, cap=SD.CAP):
    return RG.ProductionGuard(
        state, phase, priced or {"confirmatory": 53.666, "ste": 1.845}, ATTEST, "i1", 0.75,
        technical=C.technical, fallback_value="base", clock=clk or Clock(), cap=cap,
    )  # fmt: skip


def test_p3_factor_rule():
    assert EA.p3_factor(1.25) == 1.30 and EA.p3_factor(1.30) == 1.30
    assert EA.p3_factor(1.301) == 1.35 and EA.p3_factor(1.35) == 1.35 and EA.p3_factor(1.46) == 1.50
    assert EA.E2_UNIT_EST_S == pytest.approx(SD.B * SD.T_BUNDLE_S)


class VerifyRemote:
    def __init__(self, payloads, seconds):
        self.payloads, self.seconds, self.calls = payloads, seconds, []

    def call(self, kind, epoch, instance, key, label, args):
        self.calls.append((kind, key, label, args["arm"]))
        return self.payloads[args["arm"]], None


def verify_setup(tmp_path, monkeypatch, rerun_edit=None):
    from wsp.wm.c1 import controller as CO

    st = SD.Study(
        tmp_path / "s", tmp_path / "runs", ATTEST, "t" * 64, "p", log=lambda *a: None, health=False
    )
    g0 = st.guard("confirmatory", "i1")
    g0.open_instance(1)
    g0.canary(REF)
    eligible = {}
    inputs = {"library_sha256": "l", "dev_base_sha256": "d", "fit_sha256": "f"}
    units = {lab: r for r, lab in SD.e2_units(inputs, {}, range(1), ["wm_e", "statnn"])}
    for lab, recipe in units.items():
        arm = recipe["args"]["arm"]
        eligible[arm] = {"arm": arm, "winner": {"gain": 0.1, "seconds": 9.0}, "planner_seconds": 700.0,
                         "r_cache": {"misses": 3, "seconds": 1.0}, "seconds": 1200.0}  # fmt: skip
        RG_store = st.store("confirmatory")
        import wsp.wm.harness as H

        H.run_units(
            RG_store, [(recipe, lab)], lambda r, a=arm: eligible[a], "t" * 64, "p", guard=g0
        )
    g0.canary(REF)  # the closing canary makes the bracketed results eligible
    reruns = {
        a: dict(json.loads(json.dumps(p)), seconds=777.0, planner_seconds=1.0)
        for a, p in eligible.items()
    }
    if rerun_edit:
        rerun_edit(reruns)
    rem = VerifyRemote(reruns, 0.0)
    co = CO.Controller(
        st, rem, types.SimpleNamespace(push=lambda p: p), "i1", 1, log=lambda *a: None
    )
    monkeypatch.setattr(
        co, "guard", lambda phase: st.guard(phase, "i1")
    )  # no live transport in the fixture
    monkeypatch.setattr(co, "e2_inputs", lambda prep: (inputs, {}))
    monkeypatch.setattr(co, "require_field", lambda g: {"mode": "fixture"})
    t = itertools.cycle([0.0, 900.0, 1000.0, 1700.0])  # wm_e 900 s, statnn 700 s
    monkeypatch.setattr(CO.time, "time", lambda: next(t))
    return st, co, rem


def test_p2_verification_records_a_pass_and_the_p3_factor(tmp_path, monkeypatch):
    st, co, rem = verify_setup(tmp_path, monkeypatch)
    out = co.verify_e2({}, commit="a" * 40)
    assert [c[2] for c in rem.calls] == ["verify:e2:wm_e:00", "verify:e2:statnn:00"]
    assert all(c[0] == "e2_arm" for c in rem.calls)
    s = RG.read_study(st.state)
    rec = s.e2_verified[-1]
    assert rec["equal"] is True and rec["commit"] == "a" * 40 and rec["generation"] == s.generation
    assert [c["label"] for c in rec["checks"]] == ["e2:wm_e:00", "e2:statnn:00"]
    assert rec["wm_e_rate_x"] == pytest.approx(900.0 / EA.E2_UNIT_EST_S)
    assert rec["p3_factor"] == 1.30 and out["p3_factor"] == 1.30
    assert not s.incidents


def test_p2_verification_mismatch_is_an_incident_and_halts(tmp_path, monkeypatch):
    def edit(r):
        r["statnn"]["winner"]["gain"] = 0.1000001

    st, co, _rem = verify_setup(tmp_path, monkeypatch, edit)
    from wsp.wm.c1 import controller as CO

    with pytest.raises(CO.E2VerifyMismatch):
        co.verify_e2({}, commit="a" * 40)
    s = RG.read_study(st.state)
    assert not s.e2_verified and s.incidents[-1]["kind"] == "e2_speedup_mismatch"


def test_p2_a_new_e2_run_needs_a_passing_verification_on_this_code(tmp_path, monkeypatch):
    mod = ctl_module()
    st, co, _ = verify_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "STUDY", st.state)
    monkeypatch.setattr(mod, "same_code", lambda c: c == "a" * 40)
    with pytest.raises(SystemExit, match="P2 verification"):
        mod.require_p2_verified()
    co.verify_e2({}, commit="b" * 40)
    with pytest.raises(SystemExit, match="P2 verification"):
        mod.require_p2_verified()  # verified on other code
    co.verify_e2({}, commit="a" * 40)
    assert mod.require_p2_verified()["commit"] == "a" * 40


# ------------------------------------------------------------------ P3: the re-pricing record


def test_p3_reprice_raises_the_registered_part_and_the_ledger_uses_it(tmp_path):
    g = guard(tmp_path / "s", priced={"confirmatory": 53.666, "sbon": 10.7, "ste": 1.845})
    g.open_instance(1)
    rec = g.reprice(
        "confirmatory", 1.30, "RULING-wme-c1-bank-halt.md addendum 10", part_gpu_h=51.609, part="E2"
    )
    assert rec["repriced_gpu_h"] == pytest.approx(53.666 + 0.30 * 51.609)
    assert rec["outcome_blind"] is True and rec["decided_by"] == "chair"
    g.reprice("sbon", 1.30, "RULING-wme-c1-bank-halt.md addendum 10")  # the whole phase
    led = g.ledger
    assert led.price("confirmatory") == pytest.approx(53.666 + 0.3 * 51.609)
    assert led.price("sbon") == pytest.approx(13.91) and led.price("ste") == pytest.approx(1.845)
    led.add("confirmatory", 0.766 * 3600)
    assert led.priced_remainder() == pytest.approx(53.666 + 0.3 * 51.609 - 0.766 + 13.91 + 1.845)
    # the 1.25x pause is against the re-priced price: 70 GPU-h measured would have paused at 1.25 x 53.666
    led.add("confirmatory", (70.0 - 0.766) * 3600)
    led.check_before("confirmatory", 0.75)
    led.add("confirmatory", 16.0 * 3600)
    with pytest.raises(RG.PhasePause):
        led.check_before("confirmatory", 0.75)


def test_p3_reprice_refusals(tmp_path):
    g = guard(tmp_path / "s", priced={"confirmatory": 53.666, "ste": 1.845})
    g.open_instance(1)
    with pytest.raises(ValueError, match="ruling"):
        g.reprice("confirmatory", 1.30, "", part_gpu_h=51.6, part="E2")
    with pytest.raises(ValueError, match="1.3"):
        g.reprice("confirmatory", 1.25, "r", part_gpu_h=51.6, part="E2")
    with pytest.raises(ValueError, match="0.05"):
        g.reprice("confirmatory", 1.33, "r", part_gpu_h=51.6, part="E2")
    with pytest.raises(ValueError, match="part"):
        g.reprice(
            "confirmatory", 1.30, "r", part_gpu_h=60.0, part="E2"
        )  # a part above the phase price
    with pytest.raises(KeyError):
        g.reprice("nope", 1.30, "r")
    g.reprice("confirmatory", 1.35, "r", part_gpu_h=51.6, part="E2")
    with pytest.raises(ValueError, match="lower"):
        g.reprice("confirmatory", 1.30, "r", part_gpu_h=51.6, part="E2")
    assert g.reprice("confirmatory", 1.35, "r", part_gpu_h=51.6, part="E2") is None  # idempotent
    assert g.ledger.price("confirmatory") == pytest.approx(53.666 + 0.35 * 51.6)


def test_p3_the_registered_e2_part_of_the_confirmatory_price():
    part = SD.e2_registered_gpu_h()
    assert part == pytest.approx(264 * 48 * SD.T_BUNDLE_S / 3600 + 0.28839246423967485)
    assert part < SD.PRICE["confirmatory"]


# ------------------------------------------------------------------ P4: the cap generation


def test_p4_cap_is_projection_times_1_10_from_the_ledger_basis():
    basis = {"charged_gpu_h": 25.389, "repriced_remainder_gpu_h": 84.139, "overhead_gpu_h": 3.0}
    basis["projection_gpu_h"] = 25.389 + 84.139 + 3.0
    assert RG.p4_cap(basis) == pytest.approx(1.10 * 112.528)
    low = dict(basis, overhead_gpu_h=0.0, repriced_remainder_gpu_h=80.0, projection_gpu_h=105.389)
    assert RG.p4_cap(low) is None  # not above 110: no raise
    big = dict(basis, repriced_remainder_gpu_h=170.0, projection_gpu_h=198.389)
    assert RG.p4_cap(big) == pytest.approx(RG.CAP_USD / RG.USD_PER_GPU_H)  # never beyond $1,000
    with pytest.raises(ValueError, match="projection"):
        RG.p4_cap(dict(basis, projection_gpu_h=100.0))  # the basis must add up
    cap = RG.CapGeneration(
        cap_gpu_h=RG.p4_cap(basis), date="2026-11-23", decided_by="chair", announced_to_user=False,
        source="RULING-wme-c1-bank-halt.md addendum 10 (P4)", rule="P4", basis=basis,
    )  # fmt: skip
    cap.check()
    with pytest.raises(ValueError, match="P4"):
        dataclasses.replace(cap, cap_gpu_h=130.0).check()
    with pytest.raises(ValueError):
        dataclasses.replace(cap, rule="other").check()


def reprice_setup(tmp_path, monkeypatch):
    mod = ctl_module()
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "STUDY", runs / "study")
    monkeypatch.setattr(mod, "load_att", lambda: ATTEST)
    monkeypatch.setattr(mod, "MO_theta0", lambda: "t" * 64)
    st = SD.Study(runs / "study", runs, ATTEST, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    g = st.guard("bank", "lambda:x")
    g.open_instance(1)
    g.ledger.add("bank", 6.0 * 3600, note="bank:x")
    g.ledger.add("bank", 1.0 * 3600, note="field:g0")
    g2 = st.guard("confirmatory", "lambda:x")
    g2.ledger.add("confirmatory", 0.766 * 3600)
    return mod, runs


def test_p3_p4_cli_reprice_then_cap_raise(tmp_path, monkeypatch):
    mod, runs = reprice_setup(tmp_path, monkeypatch)
    a = types.SimpleNamespace(
        ruling="RULING-wme-c1-bank-halt.md addendum 10", factor=1.30, overhead_gpu_h=None
    )
    with pytest.raises(SystemExit, match="reprice"):
        mod.cmd_cap_raise(a)  # P4 needs P3 first
    out = mod.cmd_reprice(a)
    s = RG.read_study(runs / "study")
    assert set(s.reprices) == {"confirmatory", "sbon"}
    assert s.reprices["confirmatory"]["part_gpu_h"] == pytest.approx(SD.e2_registered_gpu_h())
    assert s.reprices["sbon"]["part_gpu_h"] is None and out["factor"] == 1.30
    a.overhead_gpu_h = 50.0  # a large overhead so that the fixture projection exceeds 110
    rec = mod.cmd_cap_raise(a)
    s = RG.read_study(runs / "study")
    c = s.caps[-1]
    assert c["rule"] == "P4" and c["decided_by"] == "chair" and "addendum 10" in c["source"]
    b = c["basis"]
    assert b["projection_gpu_h"] == pytest.approx(
        b["charged_gpu_h"] + b["repriced_remainder_gpu_h"] + 50.0
    )
    assert c["cap_gpu_h"] == pytest.approx(RG.p4_cap(b)) and rec["cap_gpu_h"] == c["cap_gpu_h"]
    sb = SD.sbon_price(SD.parity_budget(6.0 * 3600, 1.0 * 3600))
    want = (
        (SD.PRICE["confirmatory"] + 0.3 * SD.e2_registered_gpu_h() - 0.766)
        + 1.3 * sb
        + SD.PRICE["ste"]
    )
    assert b["repriced_remainder_gpu_h"] == pytest.approx(want)
    assert (
        mod.cmd_cap_raise(a)["cap_gpu_h"] == c["cap_gpu_h"]
        and len(RG.read_study(runs / "study").caps) == 2
    )


def test_p4_cli_no_raise_at_or_below_110(tmp_path, monkeypatch):
    mod, runs = reprice_setup(tmp_path, monkeypatch)
    a = types.SimpleNamespace(
        ruling="RULING-wme-c1-bank-halt.md addendum 10", factor=1.30, overhead_gpu_h=0.0
    )
    mod.cmd_reprice(a)
    out = mod.cmd_cap_raise(a)
    assert out["raised"] is False and len(RG.read_study(runs / "study").caps) == 1


def test_resume_phase_precheck_uses_the_repriced_price(tmp_path, monkeypatch):
    mod, runs = reprice_setup(tmp_path, monkeypatch)
    mod.cmd_reprice(types.SimpleNamespace(ruling="r", factor=1.30, overhead_gpu_h=None))
    st = SD.Study(runs / "study", runs, ATTEST, "t" * 64, mod.PANEL_ID, log=lambda *a: None)
    g = st.guard("confirmatory", "lambda:x")
    g.ledger.add(
        "confirmatory", (1.25 * 53.67 + 1.0 - 0.766) * 3600
    )  # above 1.25 x registered, below re-priced
    with pytest.raises(SystemExit, match="no PhasePause pending"):
        mod.cmd_resume_phase(types.SimpleNamespace(phase="confirmatory", ruling="r"))


def test_p4_instance_overhead_from_the_ledger():
    mod = ctl_module()
    recs, t = [{"type": "attestation", "seq": 0}], [0.0]

    def tick(dt):
        t[0] += dt
        recs.append({"type": "tick", "t": t[0]})

    for k, (par_h, main_h, unit_h) in enumerate(
        [(9.0, 9.0, 9.0), (1.0, 2.0, 0.5), (0.8, 3.0, 1.5), (1.2, 2.5, 1.0)]
    ):
        base = 100.0 * k * 3600
        recs.append({"type": "instance_authoritative", "instance": f"p{k}", "gpus": 2, "launched_at": base,
                     "terminated_at": base + par_h / 2 * 3600})  # fmt: skip
        recs.append({"type": "instance_authoritative", "instance": f"m{k}", "gpus": 1, "launched_at": base + 7200,
                     "terminated_at": base + 7200 + main_h * 3600})  # fmt: skip
        recs.append({"type": "owner_epoch", "epoch": k, "instance": f"m{k}"})
        recs.append({"type": "unit_time", "epoch": k, "gpu_s": unit_h * 3600, "note": "e2:wm_e:00" if k == 3 else "x",
                     "status": "ok"})  # fmt: skip
    for d in (100.0, 110.0, 400.0):
        tick(50.0)
        tick(d)
        recs.append({"type": "canary", "generation": 0})
    ob = mod.instance_overhead_basis(recs)
    assert [x["overhead_gpu_h"] for x in ob["launches_sampled"]] == pytest.approx(
        [2.5, 2.3, 2.7]
    )  # the last 3
    assert ob["launch_overhead_gpu_h"] == pytest.approx(2.5)
    assert ob["canary_median_s"] == 110.0 and ob["e2_units_remaining"] == 263
    assert ob["canaries_projected"] == 17 + 2 + 8
    assert ob["overhead_gpu_h"] == pytest.approx(2.5 + 27 * 110.0 / 3600)
