"""PREREG_WME §II.31 (amendment v2l): the C1(a) deterministic HF-generate engine config, the C1(b) vLLM switch and
its selection rule, the 110 GPU-h-eq cap generation (ledger, ProductionGuard, dead-man allowance) and the run-health
incident rules. CPU only; fixtures, no model."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import health as HL
from wsp.wm.c1 import lambda_ops as LO
from wsp.wm.c1 import modelops as MO
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import vllmops as VO

REF = "a" * 64
ATTEST = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])
H1 = 3600.0


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    """R11 (SYNTHESIS-r13 addendum item 5): every test here starts from torch's process defaults (no deterministic mode, default SDPA
    backends, no CUBLAS workspace variable), whatever an earlier test in any module left behind (e.g.
    ``wsp.predictor.models.seed_all``), and the caller's state is restored afterwards."""
    from wsp.wm.c1 import genengine as _GE

    saved = _GE.global_state()
    _GE.restore_state(_GE.BASELINE_STATE)
    yield
    _GE.restore_state(saved)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def guard(state, clk, cap=SD.CAP, inst="i1", priced=None):
    return RG.ProductionGuard(
        state,
        "e2",
        priced or {"e2": 500.0},
        ATTEST,
        inst,
        0.01,
        technical=C.technical,
        fallback_value="base",
        clock=clk,
        cap=cap,
    )


def model(impl="sdpa"):
    return SimpleNamespace(config=SimpleNamespace(_attn_implementation=impl))


TORCH_PATH = dict.fromkeys(GE.LINEAR_ATTN_FUNCS, GE.QWEN35_MODULE)


# ------------------------------------------------------------------ C1(a): deterministic engine config


def test_engine_config_records_the_flag_and_every_deterministic_setting():
    c = GE.config()
    assert c["mode"] == GE.GEN_ENGINE == "hf_deterministic"
    d = c["deterministic"]
    assert d["torch.use_deterministic_algorithms"] == {"mode": True, "warn_only": False}
    assert d["CUBLAS_WORKSPACE_CONFIG"].startswith(":4096:8")
    assert d["cudnn"] == {"deterministic": True, "benchmark": False}
    assert "math" in d["attention"] and "flash" in d["attention"]
    assert c["batch"] == 128 and c["max_new_tokens"] == 128 and "file order" in c["order"]
    # the three modes have distinct recorded configs (so a switch changes the engine-config digest)
    digests = {json.dumps(GE.config(m), sort_keys=True) for m in GE.MODES}
    assert len(digests) == 3
    with pytest.raises(ValueError):
        GE.config("hf_fast")


def test_ctl_engine_config_binds_the_generation_engine():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts/wme_c1_ctl.py"
    spec = importlib.util.spec_from_file_location("wme_c1_ctl_t", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ec = mod.engine_config()
    assert ec["hf_generate"]["engine"] == GE.config()


def test_prepare_process_sets_the_cublas_workspace_or_refuses(monkeypatch):
    monkeypatch.delenv(GE.CUBLAS_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    GE.prepare_process("hf_deterministic")
    assert os.environ[GE.CUBLAS_ENV] == ":4096:8"
    monkeypatch.setenv(GE.CUBLAS_ENV, ":16:8")
    with pytest.raises(RuntimeError):
        GE.prepare_process("hf_deterministic")
    monkeypatch.delenv(GE.CUBLAS_ENV)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError):  # too late: CUDA already up without the workspace config
        GE.prepare_process("hf_deterministic")
    GE.prepare_process("vllm")  # nothing to do for the vLLM engine
    assert GE.CUBLAS_ENV not in os.environ


def test_deterministic_context_sets_and_restores_and_forces_math_sdpa(monkeypatch):
    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    seen = {}
    with GE.deterministic(model(), TORCH_PATH):
        seen["det"] = torch.are_deterministic_algorithms_enabled()
        seen["warn"] = torch.is_deterministic_algorithms_warn_only_enabled()
        seen["cudnn"] = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        q = torch.randn(1, 2, 4, 8)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        seen["sdpa"] = out.shape
        seen["flash"] = torch.backends.cuda.flash_sdp_enabled()
        seen["mem"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    assert seen["det"] is True and seen["warn"] is False
    assert seen["cudnn"] == (True, False)
    assert seen["flash"] is False and seen["mem"] is False
    assert (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark) == before


def test_deterministic_context_refuses_unsafe_configurations(monkeypatch):
    monkeypatch.delenv(GE.CUBLAS_ENV, raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS"), GE.deterministic(model(), TORCH_PATH):
        pass
    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)
    with pytest.raises(RuntimeError, match="attention"):
        with GE.deterministic(model("flash_attention_2"), TORCH_PATH):
            pass
    fla = dict(TORCH_PATH, torch_chunk_gated_delta_rule="fla.ops.gated_delta_rule.chunk")
    with pytest.raises(RuntimeError, match="linear-attention"), GE.deterministic(model(), fla):
        pass
    assert not torch.are_deterministic_algorithms_enabled()


def test_hf_generate_routes_by_mode(monkeypatch):
    from wsp.wm import genaudit as GA

    monkeypatch.setenv(GE.CUBLAS_ENV, GE.CUBLAS_WORKSPACE)
    monkeypatch.setattr(GE, "kernel_bindings", lambda: TORCH_PATH)
    calls = []

    def fake(m, tok, prompts, n, *a, batch=8, device="cuda"):
        calls.append((torch.are_deterministic_algorithms_enabled(), batch, a))
        return [{"ids": [1], "text": "", "hit_max": False}]

    monkeypatch.setattr(GA, "generate_greedy", fake)
    monkeypatch.setattr(GA, "generate_sampled", fake)
    GE.hf_generate(model(), None, ["p"], 128, 128, "cpu")
    GE.hf_generate(
        model(), None, ["p"], 128, 128, "cpu", sample={"seed": 7}, mode="hf_deterministic"
    )
    GE.hf_generate(model(), None, ["p"], 128, 128, "cpu", mode="hf_registered")
    assert calls == [(True, 128, ()), (True, 128, (7,)), (False, 128, ())]
    with pytest.raises(RuntimeError):
        GE.hf_generate(model(), None, ["p"], 128, 128, "cpu", mode="vllm")


def test_canary_hf_goes_through_the_recorded_engine(monkeypatch):
    seen = {}

    def fake(model_, tok, prompts, n, batch, device, sample=None, mode=None):
        seen.update(n=n, batch=batch, mode=mode, k=len(prompts))
        return [{"ids": [i]} for i in range(len(prompts))]

    monkeypatch.setattr(GE, "hf_generate", fake)
    m = MO.ModelCtx.__new__(MO.ModelCtx)
    m.model, m.tok, m.device = model(), None, "cpu"
    m.at_base = lambda: None
    items = [{"question": f"q{i}", "options": ["a", "b"], "answer_index": 0} for i in range(3)]
    h, s = m.canary_hf_timed(items)
    assert seen == {"n": 128, "batch": 128, "mode": None, "k": 3}
    assert h == MO.sha_ids([[0], [1], [2]]) and s >= 0


# ------------------------------------------------------------------ C1(a) pass rule and the C1(b) trigger


def test_pass_rule_needs_three_of_three_and_gpu_parity():
    a, b = "1" * 64, "2" * 64
    assert GE.gen_pass([a, a, a], a, a)["pass"] is True
    assert GE.gen_pass([a, a, b], a, a)["repeat_3_of_3"] is False
    assert GE.gen_pass([a, a], a, a)["pass"] is False
    r = GE.gen_pass([a, a, a], a, b)
    assert r["repeat_3_of_3"] is True and r["gpu0_eq_gpu1"] is False and r["pass"] is False


def test_registered_e1_generation_gpu_h():
    g = GE.e1_gen_gpu_h(MO.manifest())
    assert g == pytest.approx((4000 * 0.07356123998761177 + 16000 * 0.10431850049644709) / 3600)


def test_selection_rule():
    ok, bad = {"pass": True}, {"pass": False}
    # (i) the pass rule fails -> vLLM, whatever the throughput
    assert GE.select_engine(bad, 1.0, 1.0, 0.55, 4.0, 110.0)["engine"] == "vllm"
    # passes, throughput inside the cap projection -> stays HF deterministic
    r = GE.select_engine(ok, 30.0, 10.0, 0.55, 4.0, 110.0)
    assert r["engine"] == "hf_deterministic" and r["delta_e1_gpu_h"] == pytest.approx(1.1)
    # (ii) a breach attributable to (a): 86 + spent <= cap < 86 + spent + delta -> vLLM
    r = GE.select_engine(ok, 40.0, 1.0, 0.55, 2.0, 90.0)
    assert r["throughput_breach"] is True and r["engine"] == "vllm"
    # a breach that exists without delta is the C3 cap rule's, not a reason for (b)
    r = GE.select_engine(ok, 40.0, 1.0, 0.55, 8.0, 90.0)
    assert r["throughput_breach"] is False and r["engine"] == "hf_deterministic"
    # a faster deterministic path never counts negative
    assert GE.select_engine(ok, 0.5, 1.0, 0.55, 4.0, 110.0)["delta_e1_gpu_h"] == 0.0


# ------------------------------------------------------------------ C1(b): blocks and the vLLM child env


class Tok:
    eos_token_id = 5

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        return f"<{msgs[0]['content']}>"

    def convert_tokens_to_ids(self, t):
        return 7 if t == "<|im_end|>" else -1

    def encode(self, t, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) for c in t]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i in (5, 7)))


def test_vllm_block_is_the_registered_generation():
    b = GE.vllm_block("x", Tok(), ["a", "b"])
    assert b == {
        "name": "x",
        "prompts": ["<a>", "<b>"],
        "prompt_token_ids": [[60, 97, 62], [60, 98, 62]],
        "hf_text": True,
        "max_tokens": 128,
        "stop_token_ids": [5, 7],
        "state": {"kind": "base"},
    }
    s = GE.vllm_block("y", Tok(), ["a"], sample={"temperature": 0.7, "top_p": 1.0, "seed": 3})
    assert s["sample"] == {"temperature": 0.7, "top_p": 1.0, "seed": 3}
    assert GE.canary_engines() == {"tf", "hf_gen", "vllm"}


def test_vllm_child_runs_without_the_hf_cublas_setting():
    env = VO.child_env({"PATH": "/bin", GE.CUBLAS_ENV: ":4096:8"})
    assert GE.CUBLAS_ENV not in env and env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


# ------------------------------------------------------------------ cap generation (110 GPU-h-eq)


def test_guard_records_the_v2l_cap_and_the_ledger_enforces_it(tmp_path):
    clk = Clock()
    g = guard(tmp_path / "s", clk)
    caps = g.ledger.cap_generations()
    assert [c["cap_gpu_h"] for c in caps] == [110.0] and caps[0]["index"] == 0
    assert caps[0]["previous_cap_gpu_h"] == RG.CAP_GPU_H == 90.0
    assert caps[0]["contract"] == RG.contract_digest(ATTEST) and caps[0]["contract_generation"] == 0
    assert caps[0]["decided_by"] == "user" and caps[0]["date"] == "2026-11-19"
    g.ledger.add("e2", 100.0 * H1)
    g.ledger.check_before("e2", 9.0)  # 109 <= 110: allowed (the registered 90 would refuse)
    with pytest.raises(RG.CapExceeded):
        g.ledger.check_before("e2", 10.5)
    assert g.ledger.remaining_gpu_h() == pytest.approx(10.0)
    assert g.ledger.remaining_usd() == pytest.approx(10.0 * RG.USD_PER_GPU_H)
    guard(tmp_path / "s", clk)  # a new owner: idempotent, no second cap generation
    assert len(RG.CombinedLedger(tmp_path / "s" / RG.LOG_NAME, {"e2": 1}).cap_generations()) == 1


def test_cap_generation_rules(tmp_path):
    clk = Clock()
    g = guard(tmp_path / "s", clk)
    basis = {"charged_gpu_h": 108.0, "priced_remainder_gpu_h": 8.0, "previous_cap_gpu_h": 110.0}
    r8 = dict(rule="R8", basis=basis, outcome_blind=True)
    with pytest.raises(ValueError):  # never lowers
        g.ledger.record_cap(RG.CapGeneration(100.0, "2026-11-20", "user", "s", True))
    with pytest.raises(ValueError):  # §II.31.8 R8: a later raise is only the mechanical value
        RG.CapGeneration(120.0, "2026-11-20", "chair", "§II.31", True).check()
    with pytest.raises(ValueError):
        RG.CapGeneration(125.0, "2026-11-20", "chair", "s", True, **r8).check()  # R8 says 120
    with pytest.raises(ValueError):  # never beyond $1,000 at the 4.89 x 1.10 convention
        RG.CapGeneration(190.0, "2026-11-20", "user", "s", True).check()
    with pytest.raises(ValueError):
        RG.CapGeneration(120.0, "20 Nov", "chair", "s", True, **r8).check()
    with pytest.raises(ValueError):  # outcome-blind attestation required
        RG.CapGeneration(120.0, "2026-11-20", "chair", "s", True, "R8", basis, False).check()
    # announcing is a notification, not a gate: an R8 raise not yet announced is valid
    RG.CapGeneration(120.0, "2026-11-20", "chair", "s", False, **r8).check()
    # a chair raise is a new cap generation, not a contract transition: eligible results stay eligible
    g.open_instance(1)
    store = H.UnitStore(tmp_path / "u")
    g.canary(REF)
    H.run_units(store, [({"u": 1}, "u1")], lambda r: {"v": 1}, "t" * 64, "p", guard=g)
    g.canary(REF)
    raise1 = RG.CapGeneration(120.0, "2026-11-20", "chair", "PREREG_WME §II.31.8 R8", False, **r8)
    g2 = guard(tmp_path / "s", clk, cap=raise1)
    caps = g2.ledger.cap_generations()
    assert [c["cap_gpu_h"] for c in caps] == [110.0, 120.0] and caps[1]["index"] == 1
    assert g2.ledger.cap_gpu_h() == 120.0 and g2.generation == 0
    assert g2.resolve(store, H.unit_key({"u": 1}, "t" * 64, "p")) == {"v": 1}
    with pytest.raises(ValueError):  # an older guard object cannot lower it back either
        guard(tmp_path / "s", clk, cap=SD.CAP)


def test_deadman_allowance_is_the_remaining_cap(tmp_path):
    clk = Clock()
    g = guard(tmp_path / "s", clk)
    g.ledger.add("e2", 50.0 * H1)
    rem = g.ledger.remaining_gpu_h()
    assert rem == pytest.approx(60.0)
    assert LO.allowance(rem, 18.26) == pytest.approx(18.26 + 60.0 * 4.89)
    # the policy is re-based once per new cap generation, otherwise it only tightens
    st = tmp_path / "dm"
    st.mkdir()
    p1 = LO.tighten(st, 400.0, 10.0)
    assert p1["max_usd"] == 400.0
    assert LO.tighten(st, 500.0, 10.0)["max_usd"] == 400.0
    p3 = LO.tighten(st, 500.0, 10.0, cap_generation=0)
    assert p3["max_usd"] == 500.0 and p3["cap_generation"] == 0
    assert LO.tighten(st, 600.0, 10.0, cap_generation=0)["max_usd"] == 500.0
    assert LO.tighten(st, 450.0, 10.0, cap_generation=0)["max_usd"] == 450.0
    assert LO.tighten(st, 650.0, 10.0, cap_generation=1)["max_usd"] == 650.0


def test_deadman_gpu_h_counts_every_registered_instance():
    st = {
        "instances": {
            "a": {"type": LO.TYPE2, "launched_at": 0.0, "confirmed_terminated_at": 3600.0},
            "b": {"type": LO.TYPE, "launched_at": 100.0, "confirmed_terminated_at": None},
            "c": {"type": LO.TYPE, "launched_at": None},
        }
    }
    assert LO.deadman_gpu_h(st, now=100.0 + 1800.0) == pytest.approx(2.0 + 0.5)


# ------------------------------------------------------------------ run health: worker heartbeat


def test_sampler_heartbeat_carries_gpu_utilisation_memory_and_the_inflight_request(tmp_path):
    clk = Clock()
    seq = iter([5.0, 3.0, 50.0, 1.0])

    def sample():
        return [
            {"index": 0, "util_pct": next(seq), "mem_used_mib": 1000.0, "mem_total_mib": 81559.0}
        ]

    s = HL.Sampler(tmp_path, sample=sample, clock=clk)
    s.set_inflight({"id": "r1", "kind": "sealed_eval"})
    hb = json.loads((tmp_path / HL.HEARTBEAT).read_text())
    assert hb["gpu"][0]["util_pct"] == 5.0 and hb["gpu"][0]["mem_used_mib"] == 1000.0
    assert hb["inflight"] == {"id": "r1", "kind": "sealed_eval", "since": 0.0, "last_progress": 0.0}
    assert hb["low_since"] == 0.0
    clk.t = 60.0
    assert s.beat()["low_since"] == 0.0  # still low: the run continues
    clk.t = 120.0
    assert s.beat()["low_since"] is None  # 50 %: the run is broken
    clk.t = 180.0
    s.set_inflight(None)
    hb = json.loads((tmp_path / HL.HEARTBEAT).read_text())
    assert hb["low_since"] == 180.0 and hb["inflight"] is None


def test_parse_nvidia_smi():
    out = HL.parse_smi("0, 97, 13821, 32623\n1, 0, 3, 81559\n")
    assert out[0] == {
        "index": 0,
        "util_pct": 97.0,
        "mem_used_mib": 13821.0,
        "mem_total_mib": 32623.0,
    }
    assert out[1]["util_pct"] == 0.0


def test_gpu_idle_rule():
    inf = {"id": "r", "kind": "sealed_eval", "since": 1000.0}
    g = [{"index": 0, "util_pct": 3.0, "mem_used_mib": 1.0, "mem_total_mib": 2.0}]
    assert (
        HL.idle_check({"t": 1000.0 + 901, "gpu": g, "low_since": 500.0, "inflight": inf})[0]
        == "gpu_idle"
    )
    assert HL.idle_check({"t": 1000.0 + 899, "gpu": g, "low_since": 500.0, "inflight": inf}) is None
    # low since before the unit started: counted from the unit's start only
    assert HL.idle_check({"t": 1000.0 + 600, "gpu": g, "low_since": 0.0, "inflight": inf}) is None
    assert HL.idle_check({"t": 5000.0, "gpu": g, "low_since": 0.0, "inflight": None}) is None
    busy = {"t": 5000.0, "gpu": g, "low_since": None, "inflight": inf}
    assert HL.idle_check(busy) is None
    miss = {"t": 1000.0 + 901, "gpu": None, "low_since": None, "inflight": inf}
    assert HL.idle_check(miss)[0] == "gpu_telemetry_missing"


def test_log_and_payload_error_detection():
    log = "INFO loaded\nTraceback (most recent call last):\n  File x\nValueError: bad\nok line\n"
    errs = HL.log_errors(log)
    assert any("Traceback" in e for e in errs) and any(e.startswith("ValueError") for e in errs)
    assert HL.log_errors("INFO step 3 done\nUserWarning about a thing\n") == []
    assert HL.payload_traceback({"x": ["Traceback (most recent call last):\n..."]})
    assert not HL.payload_traceback({"x": 1})


# ------------------------------------------------------------------ run health: controller halts


class FakeT:
    def __init__(self, hb=None, log=""):
        self.hb, self.log, self.cancelled = hb, log, []

    def heartbeat(self):
        return self.hb

    def read_log(self, off):
        return self.log[off:], len(self.log)


def open_guard(tmp_path, clk):
    g = guard(tmp_path / "s", clk)
    g.open_instance(1)
    g.canary(REF)
    return g


def test_an_incident_halts_issuing_until_resolved(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    g.incident("gpu_idle", {"low_for_s": 901})
    n0 = sum(r["type"] == "attempt_start" for r in g.events)
    with pytest.raises(RG.HealthHalt):
        H.run_units(
            H.UnitStore(tmp_path / "u"), [({"u": 1}, "u1")], lambda r: {}, "t" * 64, "p", guard=g
        )
    assert sum(r["type"] == "attempt_start" for r in g.events) == n0  # the halt burns no attempt
    g2 = guard(tmp_path / "s", clk)  # a restarted controller is still halted (persisted)
    assert [i["kind"] for i in g2.open_incidents()] == ["gpu_idle"]
    with pytest.raises(ValueError):
        g2.resolve_incidents("", "report")
    g2.resolve_incidents("stalled tokenizer worker; restarted", "docs/reviews/wme/incident-1.md")
    g2.canary(REF)
    store = H.UnitStore(tmp_path / "u")
    H.run_units(store, [({"u": 1}, "u1")], lambda r: {"v": 2}, "t" * 64, "p", guard=g2)
    g2.canary(REF)  # the closing canary makes it eligible
    assert g2.resolve(store, H.unit_key({"u": 1}, "t" * 64, "p")) == {"v": 2}


def test_monitor_order_rule(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs")
    for k in ("field_group", "bank_base", "bank_side", "bank_comp", "library", "fit", "e2_arm"):
        mon.check_order(g, {"kind": k})
    with pytest.raises(HL.HealthIncident):  # back to the bank after E2 started
        mon.check_order(g, {"kind": "bank_side"})
    assert g.open_incidents()[-1]["kind"] == "order"
    g.resolve_incidents("fixture", "fixture")
    with pytest.raises(HL.HealthIncident):  # skipping S-BoN and the sealed read
        mon.check_order(g, {"kind": "e1_gen"})
    g.resolve_incidents("fixture", "fixture")
    with pytest.raises(HL.HealthIncident):
        mon.check_order(g, {"kind": "square"})
    g.resolve_incidents("fixture", "fixture")
    for k in ("sbon_arm", "sealed_eval"):
        mon.check_order(g, {"kind": k})
    mon.skip_stage("e1_gen", "E1 not designated")
    mon2 = HL.HealthMonitor(tmp_path / "runs")  # persisted across controller restarts
    mon2.check_order(g, {"kind": "ste_calibrate"})
    assert not g.open_incidents()


def test_monitor_error_canary_and_idle_rules(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(tmp_path / "runs", FakeT(log="step ok\n"), clock=clk)
    fn = mon.wrap_fn(g, lambda r: {"ok": 1})
    assert fn({"kind": "field_group"}) == {"ok": 1}

    def boom(r):
        raise RuntimeError("CUDA error: an illegal memory access")

    with pytest.raises(
        RuntimeError
    ) as ei:  # R3: re-raised unwrapped (its registered cause survives)
        mon.wrap_fn(g, boom)({"kind": "field_group"})
    assert not isinstance(ei.value, HL.HealthIncident)
    assert g.open_incidents()[-1]["kind"] == "worker_error"
    g.resolve_incidents("f", "f")
    with pytest.raises(HL.HealthIncident):
        mon.wrap_fn(g, lambda r: {"trace": "Traceback (most recent call last): x"})(
            {"kind": "field_group"}
        )
    g.resolve_incidents("f", "f")
    mon.t.log += "Traceback (most recent call last):\nOSError: disk\n"
    with pytest.raises(HL.HealthIncident):
        fn({"kind": "field_group"})
    g.resolve_incidents("f", "f")
    rec = mon.wrap_canary(g, lambda: {"pass": False, "composite": "d" * 64, "engines": {}})()
    assert rec["pass"] is False and g.open_incidents()[-1]["kind"] == "canary_mismatch"
    g.resolve_incidents("f", "f")
    inf = {"id": "r", "kind": "sealed_eval", "since": 0.0}
    gpu = [{"index": 0, "util_pct": 2.0, "mem_used_mib": 1.0, "mem_total_mib": 2.0}]
    mon.t.hb = {"t": 950.0, "gpu": gpu, "low_since": 0.0, "inflight": inf}
    clk.t = 100.0
    with pytest.raises(HL.HealthIncident):
        mon.poll(g)
    assert g.open_incidents()[-1]["kind"] == "gpu_idle"
    assert len((tmp_path / "runs/health/incidents.jsonl").read_text().splitlines()) == 5


def test_remote_watch_aborts_and_withdraws_the_request(tmp_path):
    from wsp.wm.c1 import transport as T

    tr = T.LocalTransport(tmp_path / "w")
    (tmp_path / "w" / "alive").write_text("x")

    def watch():
        raise HL.HealthIncident("gpu_idle")

    rem = C.Remote(tr, tmp_path / "ctl", poll_s=0.01, sleep=lambda s: None, watch=watch)
    with pytest.raises(HL.HealthIncident):
        rem.call("e2_arm", 1, "i1", "k" * 64, "l", {})
    assert not list((tmp_path / "w" / "inbox").glob("*.json"))  # cancelled
    assert '"health_abort"' in (tmp_path / "ctl" / "dispatch.jsonl").read_text()


def test_chunk_summary_is_written_at_every_chunk_boundary(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    gpu = [{"index": 0, "util_pct": 88.0, "mem_used_mib": 70000.0, "mem_total_mib": 81559.0}]
    mon = HL.HealthMonitor(tmp_path / "runs", FakeT(hb={"t": 1.0, "gpu": gpu}), clock=clk)
    store = H.UnitStore(tmp_path / "u")
    units = [({"kind": "field_group", "u": i}, f"u{i}") for i in range(5)]
    canary = lambda: {"pass": g.canary(REF), "composite": REF}
    C.run_bracketed(
        g,
        store,
        units,
        mon.wrap_fn(g, lambda r: {"v": r["u"]}),
        "t" * 64,
        "p",
        mon.wrap_canary(g, canary),
        chunk=2,
        on_chunk=lambda rec, n: mon.chunk_summary(g, rec, n),
        before_begin=mon.before_begin(g),
    )
    lines = [
        json.loads(x) for x in (tmp_path / "runs/health/summary.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 3 and [x["units_done_in_call"] for x in lines] == [2, 2, 1]
    last = json.loads((tmp_path / "runs/health/latest.json").read_text())
    assert last["gpu"] == gpu and last["cap_gpu_h"] == 110.0 and last["canary_pass"] is True
    assert last["open_incidents"] == [] and last["stage"] == "bank_field"


def _bindings_in_subprocess(block: bool) -> dict:
    import subprocess
    import sys

    code = (
        "import sys, json\n"
        + ("sys.modules['fla'] = None; sys.modules['causal_conv1d'] = None\n" if block else "")
        + "from wsp.wm.c1 import genengine as GE\nprint(json.dumps(GE.kernel_bindings()))\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_kernel_bindings_resolve_the_real_implementation_not_the_wrapper_name():
    """transformers wraps fla with functools.wraps (torch name and module kept); the probe must see through it."""
    import importlib.util

    assert set(_bindings_in_subprocess(block=True).values()) == {GE.QWEN35_MODULE}
    if importlib.util.find_spec("fla") is None:
        pytest.skip("fla not installed here (as on the pinned lock)")
    live = _bindings_in_subprocess(block=False)
    assert live["torch_chunk_gated_delta_rule"].startswith("fla.")
    with pytest.raises(RuntimeError, match="linear-attention"):
        GE.check_model(model(), live)


# ------------------------------------------------------------------ addendum 2, answer 3: GPU kinds vs CPU kinds


def test_every_registered_handler_kind_is_classified_gpu_or_cpu():
    import importlib

    from wsp.wm.c1 import serve as S

    importlib.import_module(
        "wsp.wm.c1.serve_more"
    )  # registers E2, sealed, generation, S-TE and launch handlers
    kinds = set(S.HANDLERS)
    assert kinds <= set(HL.KIND_CLASS), sorted(kinds - set(HL.KIND_CLASS))
    assert set(HL.KIND_CLASS.values()) == {"gpu", "cpu"}
    # the chair's named CPU kinds: the E2 planner search (e2_arm) and the fits
    assert HL.KIND_CLASS["e2_arm"] == HL.KIND_CLASS["fit"] == "cpu"
    for k in ("field_group", "bank_side", "sealed_eval", "e1_gen", "ste_train", "canary"):
        assert HL.KIND_CLASS[k] == "gpu"
    # every unit kind of the registered §II.18 order is classified
    assert set(HL.STAGE_OF) <= set(HL.KIND_CLASS)


def hb(kind, t, since, low_since=None, last_progress=None, gpu=True):
    g = [{"index": 0, "util_pct": 2.0, "mem_used_mib": 1.0, "mem_total_mib": 2.0}] if gpu else None
    inf = {"id": "r", "kind": kind, "since": since}
    if last_progress is not None:
        inf["last_progress"] = last_progress
    return {"t": t, "gpu": g, "low_since": low_since, "inflight": inf}


def test_gpu_idle_rule_applies_only_to_gpu_kinds():
    # a GPU kind idle for > 15 min is an incident even while it reports progress
    r = HL.idle_check(hb("sealed_eval", 1000.0, 0.0, low_since=0.0, last_progress=999.0))
    assert r[0] == "gpu_idle"
    # a CPU kind (planner search) idle on the GPU for an hour but progressing is not
    assert HL.idle_check(hb("e2_arm", 3600.0, 0.0, low_since=0.0, last_progress=3500.0)) is None
    assert HL.idle_check(hb("fit", 3600.0, 0.0, low_since=0.0, last_progress=3000.0)) is None
    # a CPU kind needs no GPU telemetry
    assert HL.idle_check(hb("fit", 3600.0, 0.0, last_progress=3000.0, gpu=False)) is None


def test_cpu_kinds_without_progress_for_15_min_are_an_incident():
    r = HL.idle_check(hb("e2_arm", 1000.0, 0.0, low_since=None, last_progress=99.0))
    assert r[0] == "no_progress" and r[1]["stalled_s"] == pytest.approx(901.0)
    assert HL.idle_check(hb("e2_arm", 1000.0, 0.0, last_progress=101.0)) is None
    # no progress since the unit started counts from its start
    assert HL.idle_check(hb("fit", 901.0, 0.0))[0] == "no_progress"
    assert HL.idle_check(hb("fit", 899.0, 0.0)) is None
    # the GPU being busy does not excuse a stalled CPU kind
    busy = hb("fit", 2000.0, 0.0, last_progress=10.0)
    busy["gpu"][0]["util_pct"] = 95.0
    assert HL.idle_check(busy)[0] == "no_progress"


def test_worker_progress_moves_the_cpu_progress_heartbeat(tmp_path):
    from wsp.wm.c1 import worker as W

    clk = Clock()
    s = HL.Sampler(tmp_path, sample=lambda: None, clock=clk)
    s.set_inflight({"id": "r1", "kind": "fit"})
    assert s.beat()["inflight"]["last_progress"] == 0.0
    clk.t = 50.0
    w = W.Worker(tmp_path, {}, sampler=s)
    ctx = W.Ctx(w, {"id": "r1", "kind": "fit", "epoch": 0}, tmp_path)
    ctx.progress("inner fold 1 done")
    clk.t = 80.0
    ctx.check()  # a handler sub-step boundary is progress too
    hb_ = s.beat()
    assert hb_["inflight"]["last_progress"] == 80.0 and hb_["inflight"]["since"] == 0.0


def test_monitor_raises_no_progress_for_a_stalled_cpu_unit(tmp_path):
    clk = Clock()
    g = open_guard(tmp_path, clk)
    mon = HL.HealthMonitor(
        tmp_path / "runs", FakeT(hb=hb("e2_arm", 1000.0, 0.0, last_progress=50.0)), clock=clk
    )
    clk.t = 100.0
    with pytest.raises(HL.HealthIncident):
        mon.poll(g)
    assert g.open_incidents()[-1]["kind"] == "no_progress"
