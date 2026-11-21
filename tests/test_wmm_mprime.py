"""Study M′ run driver logic (wm.wmm.mprime): FC2 ledger and cut-order schedule, FC1 read-gate, OOM policy,
"not measured" guards, M-A unit constructions and R0 features. CPU only."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wsp.wm import actions as A
from wsp.wm.wmm import mprime as M
from wsp.wm.wmm import prereg as P
from wsp.wm.wmm import provenance as PV

# ------------------------------------------------------------------ FC2: schedule and ledger


def test_tasks_follow_the_registered_schedule_and_cut_order():
    tasks = M.build_tasks()
    phases = [t["phase"] for t in tasks]
    order = list(dict.fromkeys(phases))
    assert order == [
        "base",
        "M-SYC",
        "M-TE",
        "M-A minimum",
        "M-TE ARC guard",
        "M-A remainder",
        "TE-I",
    ]
    ma_min = [t for t in tasks if t["phase"] == "M-A minimum" and t["kind"] == "ma_unit"]
    ma_rest = [t for t in tasks if t["phase"] == "M-A remainder"]
    assert len(ma_min) == 20 and len(ma_rest) == 12
    assert {t["unit"] for t in ma_min} == {u["id"] for u in P.ma_bank(minimum=True)}
    assert [t["id"] for t in tasks if t["phase"] == "M-TE ARC guard"] == ["mte:gen"]
    assert not any(t["id"] == "mte:gen" for t in tasks if t["phase"] == "M-TE")
    assert sum(t["est_h"] for t in tasks if t["phase"] != "TE-I") == pytest.approx(41.15, abs=0.5)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _run(cap_h, tmp_path):
    clock = _Clock()
    led = M.Ledger(tmp_path / "ledger.json", cap_h=cap_h, clock=clock)
    tasks = M.build_tasks()

    def runner(task):
        clock.t += task["est_h"] * 3600  # each task takes exactly its estimate

    return M.run_schedule(tasks, led, runner), led


def test_hard_stop_cuts_the_ma_remainder_first(tmp_path):
    # cap reached during the M-A remainder: everything before it ran; part of the remainder is "not measured"
    status, led = _run(39.0, tmp_path)
    by_phase = {}
    for st in status.values():
        by_phase.setdefault(st["phase"], set()).add(st["status"])
    assert (
        by_phase["base"] == {"done"}
        and by_phase["M-SYC"] == {"done"}
        and by_phase["M-TE"] == {"done"}
    )
    assert by_phase["M-A minimum"] == {"done"} and by_phase["M-TE ARC guard"] == {"done"}
    assert "not measured (cap)" in by_phase["M-A remainder"]
    assert led.used_h >= 39.0
    # no task started once the ledger had reached the cap
    started_after = [
        s for s in status.values() if s["status"] == "done" and s["started_at_h"] >= 39.0
    ]
    assert not started_after


def test_hard_stop_then_cuts_the_mte_arc_guard(tmp_path):
    status, _ = _run(29.0, tmp_path)
    assert status["mte:gen"]["status"] == "not measured (cap)"
    assert all(
        s["status"] == "not measured (cap)"
        for s in status.values()
        if s["phase"] == "M-A remainder"
    )


def test_te_i_runs_only_if_its_estimate_fits(tmp_path):
    status, _ = _run(42.0, tmp_path)
    assert all(s["status"] != "done" for s in status.values() if s["phase"] == "TE-I")
    assert all(s["status"] == "done" for s in status.values() if s["phase"] not in ("TE-I",))


def test_ledger_persists_and_resumes(tmp_path):
    clock = _Clock()
    led = M.Ledger(tmp_path / "l.json", cap_h=42.0, clock=clock)
    led.begin()
    clock.t += 3600
    led.end()
    led2 = M.Ledger(tmp_path / "l.json", cap_h=42.0, clock=clock)
    assert led2.used_h == pytest.approx(1.0)
    tasks = M.build_tasks()[:3]
    done = {tasks[0]["id"]: {"status": "done", "phase": tasks[0]["phase"], "started_at_h": 0.0}}
    (tmp_path / "l.json").write_text(
        json.dumps({**json.loads((tmp_path / "l.json").read_text()), "tasks": done})
    )
    ran = []
    M.run_schedule(
        tasks, M.Ledger(tmp_path / "l.json", cap_h=42.0, clock=clock), lambda t: ran.append(t["id"])
    )
    assert ran == [t["id"] for t in tasks[1:]]


def test_not_measured_guard_makes_the_claim_fail():
    rng = np.random.default_rng(0)
    from wsp.wm.wmm import powersim as S

    g = {
        "harm_compliance": S.paired_binary_sample(1500, 0.05, 0.0, 0.4, rng),
        "benign_over_refusal": S.paired_binary_sample(1500, 0.02, 0.0, 0.02, rng),
        "general_accuracy": None,  # not measured
        "gcal": rng.normal(0, 0.05, 3500),
    }
    r = M.guards_with_missing(g)
    assert r["general_accuracy"] == {"pass": False, "status": "not measured"}
    assert not r["pass"] and r["harm_compliance"]["pass"]


# ------------------------------------------------------------------ OOM policy (fixed batch composition)


def test_oom_regenerates_every_state_of_the_set_at_batch_16():
    reg = M.BatchRegistry()
    assert reg.batch("mq_neutral") == 32
    reg.mark_done("mq_neutral", "base", 32)
    reg.mark_done("mq_neutral", "msyc", 32)
    stale = reg.on_oom("mq_neutral", "mte")
    assert reg.batch("mq_neutral") == 16
    assert set(stale) == {"base", "msyc"}
    with pytest.raises(RuntimeError):
        reg.on_oom("mq_neutral", "mte")  # an OOM at 16 is not recoverable by the registered rule


# ------------------------------------------------------------------ FC1: read-gate on every analysis read


def _run_dir(tmp_path, log_edit=True):
    d = tmp_path / "run"
    (d / "states").mkdir(parents=True)
    log = PV.ProvenanceLog(d / "provenance.jsonl")
    before = {"a": "h0"}
    log.append("job_start", job="j", base_sha256=PV.combine(before))
    rec = log.append_edit({"family": "syc"}, before, {"a": "h1"}) if log_edit else None
    base_sha = PV.combine(before)
    edit_sha = rec["edit_sha256"] if rec else "e" * 64
    (d / "states" / "base.json").write_text(json.dumps({"tag": "base", "edit_sha256": base_sha}))
    (d / "states" / "msyc.json").write_text(json.dumps({"tag": "msyc", "edit_sha256": edit_sha}))
    (d / "outcomes").mkdir()
    for t in ("base", "msyc"):
        (d / "outcomes" / f"{t}__mq_neutral.jsonl").write_text(json.dumps({"correct": 1}) + "\n")
    return d


def test_gated_loader_reads_logged_states(tmp_path):
    d = _run_dir(tmp_path)
    out = M.load_outcomes(d, ["base", "msyc"], "mq_neutral")
    assert set(out) == {"base", "msyc"} and out["msyc"][0]["correct"] == 1


def test_gated_loader_aborts_before_reading_an_unlogged_state(tmp_path, monkeypatch):
    d = _run_dir(tmp_path, log_edit=False)
    opened = []
    real = M._read_outcome_file
    monkeypatch.setattr(M, "_read_outcome_file", lambda p: opened.append(p) or real(p))
    with pytest.raises(P.ProvenanceError):
        M.load_outcomes(d, ["base", "msyc"], "mq_neutral")
    assert opened == []  # no outcome was read


# ------------------------------------------------------------------ M-A constructions and R0 features


def _tiny():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=4,
        max_position_embeddings=128,
        query_pre_attn_scalar=16,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        pad_token_id=0,
    )
    m = Gemma3ForCausalLM(cfg).to(torch.bfloat16).eval()
    return m


def _rm(m):
    from wsp.wm.wmm import roles as R

    return R.GemmaRoleMap.from_config({"model_type": "gemma3_text", **m.config.to_dict()})


def test_slice_names_and_scaled_direction_have_the_target_norm():
    m = _tiny()
    rm = _rm(m)
    names = M.slice_names(rm, (0, 1), "attn")
    assert names == sorted(rm.name(L, r) for L in (0, 1) for r in ("q", "k", "v", "o"))
    assert M.slice_names(rm, (2, 3), "mlp") == sorted(
        rm.name(L, r) for L in (2, 3) for r in ("gate", "up", "down")
    )
    g = torch.Generator().manual_seed(1)
    raw = {n: torch.randn(dict(m.named_parameters())[n].shape, generator=g) for n in names}
    d = M.scale_to(raw, 0.7)
    assert M.frob(d) == pytest.approx(0.7, rel=1e-6)


def test_random_direction_scales_per_tensor_with_w_and_hits_the_norm():
    m = _tiny()
    rm = _rm(m)
    params = dict(m.named_parameters())
    d = M.random_direction(params, rm, seed=3, norm=1.5)
    assert set(d) == set(rm.writers().values())
    assert M.frob(d) == pytest.approx(1.5, rel=1e-5)
    ratios = [float(d[n].norm() / params[n].float().norm()) for n in d]
    assert max(ratios) == pytest.approx(
        min(ratios), rel=1e-4
    )  # per-tensor scale proportional to ||W||


def test_sham_compensated_pair_is_bitwise_inert_and_restores():
    m = _tiny()
    rm = _rm(m)
    ids = torch.tensor([[2, 5, 9, 11, 13, 7]])
    with torch.no_grad():
        ref = m(ids).logits.float()
    params = dict(m.named_parameters())
    sess = A.EditSession(params, "base", rm)
    for act in M.sham_actions(rm, "compensated_mlp_pair", (1, 2)):
        sess.apply(act)
    with torch.no_grad():
        out = m(ids).logits.float()
    assert torch.equal(out, ref)
    acts = M.sham_actions(rm, "uniform_o_scale", (1, 2))
    assert {a.tensor for a in acts} == {rm.name(1, "o"), rm.name(2, "o")}
    sess.restore()


def test_r0_features():
    m = _tiny()
    rm = _rm(m)
    params = dict(m.named_parameters())
    tau = {n: torch.randn(p.shape) for n, p in params.items() if ".layers." in n and p.dim() == 2}
    sl = M.slice_names(rm, (0, 1), "mlp")
    delta = {n: tau[n].clone() for n in sl}
    syc = {n: torch.randn(params[n].shape) for n in rm.writers().values()}
    f = M.r0_features(delta, tau, syc, params, rm, groups=[(0, 1), (2, 3)])
    assert f["cos_tau_group_0"] == pytest.approx(1.0, abs=1e-6) and f["cos_tau_group_1"] == 0.0
    assert 0.0 <= f["share_last_third"] <= 1.0 and 0.0 < f["top1_sv_share"] <= 1.0
    wn = rm.name(3, "down")
    f2 = M.r0_features(
        {wn: params[wn].float() * 0.1}, tau, syc, params, rm, groups=[(0, 1), (2, 3)]
    )
    assert f2["postnorm_cancel_index"] == pytest.approx(1.0, abs=1e-5)  # delta along W itself
    assert list(f) == M.r0_keys(2) and list(f2) == M.r0_keys(2)


# ------------------------------------------------------------------ FC1 integration: the analysis entry point


def _synthetic_run(tmp_path, log_msyc=True):
    rng = np.random.default_rng(0)
    d = tmp_path / "run"
    for sub in ("states", "outcomes"):
        (d / sub).mkdir(parents=True)
    log = PV.ProvenanceLog(d / "provenance.jsonl")
    before = {"w": "h0"}
    base_sha = PV.combine(before)
    log.append("job_start", job="j", base_sha256=base_sha)
    edit_sha = (
        log.append_edit({"family": "msyc"}, before, {"w": "h1"})["edit_sha256"]
        if log_msyc
        else "f" * 64
    )
    (d / "states" / "base.json").write_text(json.dumps({"tag": "base", "edit_sha256": base_sha}))
    (d / "states" / "msyc.json").write_text(json.dumps({"tag": "msyc", "edit_sha256": edit_sha}))
    (d / "ledger.json").write_text(json.dumps({"used_h": 1.0, "cap_h": 42.0, "tasks": {}}))
    n = 60

    def mc(adopt_p):
        return [
            {
                "id": f"i{i}",
                "correct": int(rng.random() < 0.6),
                "truncated": 0,
                "parsed": 1,
                "letter": "A",
                "adopted": int(rng.random() < adopt_p),
                "n_tokens": int(rng.integers(200, 2000)),
                "conf_after_reasoning": 0.99,
            }
            for i in range(n)
        ]

    for tag, ap in (("base", 0.3), ("msyc", 0.1)):
        for s in ("mq_suggested", "mq_neutral", "mm_neutral", "gen"):
            (d / "outcomes" / f"{tag}__{s}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in mc(ap))
            )
        for s in ("harm", "benign"):
            rows = [
                {"id": f"h{i}", "refused": int(rng.random() < 0.5), "truncated": 0, "n_tokens": 100}
                for i in range(n)
            ]
            (d / "outcomes" / f"{tag}__{s}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
        rows = [
            {
                "id": f"g{i}",
                "logits": rng.normal(size=(4, 2)).tolist(),
                "gold": int(i % 4),
                "n_options": 4,
            }
            for i in range(n)
        ]
        (d / "outcomes" / f"{tag}__gcal.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return d


def test_analysis_runs_on_logged_states(tmp_path):
    from wsp.wm.wmm import mprime_analysis as MA

    d = _synthetic_run(tmp_path)
    out = MA.analyze(d, "msyc")
    assert out["status"] == "EXPLORATORY" and out["msyc"]["claim"]["status"] == "EXPLORATORY"
    assert set(out["msyc"]["claim"]["guards"]) >= {
        "harm_compliance",
        "benign_over_refusal",
        "general_accuracy",
        "gcal",
    }
    assert "ci95" in out["msyc"]["estimates_ci95"]["adoption_change"]
    assert out["base"]["gcal"]["n"] == 60


def test_analysis_aborts_on_an_unlogged_state_before_reading_outcomes(tmp_path, monkeypatch):
    from wsp.wm.wmm import mprime_analysis as MA

    d = _synthetic_run(tmp_path, log_msyc=False)
    opened = []
    real = M._read_outcome_file
    monkeypatch.setattr(M, "_read_outcome_file", lambda p: opened.append(Path(p).name) or real(p))
    with pytest.raises(P.ProvenanceError):
        MA.analyze(d, "msyc")
    assert not any(name.startswith("msyc__") for name in opened)  # no M-SYC outcome was read
