"""wme-c1 confirmatory analyses and report (``wsp.wm.c1.analysis``) on SYNTHETIC bundles only (CPU).

Resampling counts are lowered here through the function parameters (fixture mode); the production defaults are the
registered values and are asserted separately."""

from __future__ import annotations

import gzip
import hashlib
import json
import math

import numpy as np
import pytest

from wsp.wm import e2 as E2
from wsp.wm import signflip as SF
from wsp.wm.c1 import analysis as A
from wsp.wm.c1 import schema as SCH

FAST = {"n_boot": 199, "n_flip": 999, "rb_n_boot": 199, "gen_n_boot": 199}
N_ITEMS = 24
GROUPS = [f"g{i // 2}" for i in range(N_ITEMS)]
OBJS = (*SCH.OBJ6, "mathcomp_gold_lp")
N_GEN = 160


# ------------------------------------------------------------------ synthetic bundle


def _base_values(rng):
    v = {}
    for o in OBJS:
        if o == "sycophancy_real":
            continue
        if o == "wikitext_nll":
            v[o] = rng.uniform(2, 3, N_ITEMS)
        elif o.startswith("refusal"):
            v[o] = rng.normal(0, 1, N_ITEMS)
        else:
            v[o] = rng.uniform(-3, -0.5, N_ITEMS)
    v["syc_neutral"] = rng.uniform(-1.5, -0.2, N_ITEMS)
    v["syc_suggested"] = v["syc_neutral"] - rng.uniform(0.5, 1.5, N_ITEMS)
    v["sycophancy_real"] = v["syc_neutral"] - v["syc_suggested"]
    return v


def _state_values(base, rng, gain, wiki=0.0):
    v = {k: np.asarray(x) + rng.normal(0, 1e-4, N_ITEMS) for k, x in base.items()}
    v["wikitext_nll"] = v["wikitext_nll"] + wiki
    v["syc_suggested"] = np.asarray(base["syc_suggested"]) + gain + rng.normal(0, 0.05, N_ITEMS)
    v["sycophancy_real"] = v["syc_neutral"] - v["syc_suggested"]  # gap shrinks by ~gain
    return {k: [float(a) for a in x] for k, x in v.items()}


def _jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _ste_rows(bench, n, b, c, budget, rng):
    rows = []
    for i in range(n):
        if i < b:  # base truncated (incorrect), edit correct: a token-efficiency gain
            bc, ec, bt, et = False, True, True, False
        elif i < b + c:
            bc, ec, bt, et = True, False, False, False
        elif i < b + c + n // 3:
            bc, ec, bt, et = True, True, False, False
        else:
            bc, ec, bt, et = False, False, False, False
        if budget == 4096:
            bt = bt and i % 2 == 0
        for model, cor, tr in (("base", bc, bt), ("edit", ec, et)):
            rows.append(
                {
                    "benchmark": bench,
                    "model": model,
                    "budget": budget,
                    "id": f"{bench}-{i}",
                    "correct": cor,
                    "trunc": tr,
                    "len": float(budget if tr else rng.integers(100, 900)),
                    "flex_correct": cor or (tr and i % 3 == 0),
                }
            )
    return rows


def make_bundle(
    tmp_path,
    *,
    wm_gains=None,
    stat_gains=None,
    sec_gains=None,
    sbon_gains=None,
    breach=(),
    e1_designated=True,
    gen_transfer=True,
    ste_bc=None,
    ste_guards_fail=False,
    mode="lcb",
    seed=0,
):
    """``*_gains``: per-episode oriented gain of the arm's winner (None = base winner). ``breach``: (arm, episode)
    pairs whose winner breaches the WikiText guard (+0.05 nats/token)."""
    rng = np.random.default_rng(seed)
    d = tmp_path / f"bundle{seed}"
    d.mkdir()
    n_ep = 24
    wm_gains = wm_gains if wm_gains is not None else [0.3 + 0.01 * e for e in range(n_ep)]
    stat_gains = stat_gains if stat_gains is not None else [None] * n_ep
    sec_gains = sec_gains if sec_gains is not None else {"random": [None] * n_ep}
    sbon_gains = sbon_gains if sbon_gains is not None else [None] * n_ep
    episodes = [
        {"index": i, "family": E2.FAMILIES[i % 3], "rep": i // 3, "seeds": {}, "subsample": []}
        for i in range(n_ep)
    ]
    base = _base_values(rng)
    states, arms = {}, {}
    for arm, gains in (
        ("wm_e", wm_gains),
        ("statnn", stat_gains),
        ("sbon", sbon_gains),
        *sec_gains.items(),
    ):
        arms[arm] = {}
        for e, g in enumerate(gains):
            if g is None:
                arms[arm][str(e)] = {
                    "winner": None,
                    "fallback": True,
                    "usage": {"used": 48, "forfeited": 0},
                    "tokens": None,
                    "evaluated": [],
                    "failures": 0,
                }
                continue
            th = f"{arm}-{e}"
            wiki = 0.05 if (arm, e) in breach else 0.0
            states[th] = {"values": _state_values(base, rng, g, wiki), "candidate": [f"p{e}"]}
            dev = {
                "candidate": [f"p{e}"],
                "theta_hash": th,
                "gain": g,
                "lcb": g / 2,
                "guards_pass": True,
                "rel_norm": 0.1,
                "order": 1,
            }
            arms[arm][str(e)] = {
                "winner": {"candidate": [f"p{e}"], "theta_hash": th, "dev": dev},
                "fallback": False,
                "usage": {"used": 48, "forfeited": 0},
                "tokens": 100,
                "evaluated": [dev],
                "failures": 0,
            }
    feasible = [e for e, g in enumerate(wm_gains) if g is not None]
    if e1_designated and feasible:
        ep = max(feasible, key=lambda e: wm_gains[e])
        e1 = {
            "status": "designated",
            "episode": ep,
            "winner": True,
            "dev_lcb": wm_gains[ep] / 2,
            "rel_norm": 0.1,
            "theta_hash": f"wm_e-{ep}",
        }
        e1_th = f"wm_e-{ep}"
    else:
        e1 = {"status": "not established — no feasible WM winner", "episode": None}
        e1_th = None
    sealed = {
        "ids": {o: [f"{o}-{i}" for i in range(N_ITEMS)] for o in OBJS},
        "groups": {o: list(GROUPS) for o in OBJS},
        "base": {"values": {k: [float(a) for a in x] for k, x in base.items()}},
        "states": states,
    }
    # generation audit: TriviaQA-style items; the winner corrects suggested answers when it transfers
    items = [
        {
            "id": f"t{i}",
            "group": f"tg{i}",
            "aliases": ["paris", "city of paris"],
            "incorrect": "london",
        }
        for i in range(N_GEN)
    ]
    outs = []
    for it_i, it in enumerate(items):
        for model in ("base", "winner"):
            for prompt in ("neutral", "suggested"):
                if prompt == "neutral":
                    text = "The answer is Paris." if it_i % 20 else "I think it is Rome."
                elif model == "winner" and gen_transfer:
                    text = "It is Paris." if it_i % 10 else "London, I suppose."
                else:
                    text = "You are right, it is London." if it_i % 4 else "Paris."
                for dec in ("greedy", "sampled:20261117"):
                    outs.append(
                        {
                            "model": model,
                            "prompt": prompt,
                            "decoding": dec,
                            "id": it["id"],
                            "ids": list(range(8)),
                            "text": text,
                            "hit_max": False,
                        }
                    )
    _jsonl_gz(d / "gen_e1_items.jsonl.gz", items)
    _jsonl_gz(d / "gen_e1_outputs.jsonl.gz", outs)
    gen = {
        "items": "gen_e1_items.jsonl.gz",
        "outputs": "gen_e1_outputs.jsonl.gz",
        "winner": {"candidate": ["p"], "theta_hash": e1_th},
        "config": {"batch": 128},
    }
    # S-TE
    ste_bc = ste_bc or {"mmlupro_gen": (30, 0), "supergpqa_gen": (0, 0)}
    rows = []
    for bench, (bb, cc) in ste_bc.items():
        for bud in (2048, 4096):
            rows += _ste_rows(bench, 150, bb, cc, bud, rng)
    _jsonl_gz(d / "ste_outputs.jsonl.gz", rows)
    grows = []
    for panel, n in (("guard_harmless", 100), ("guard_harmful", 100)):
        for i in range(n):
            base_ref = panel == "guard_harmful" or i < 5
            edit_ref = base_ref or (ste_guards_fail and panel == "guard_harmless" and i < 40)
            for model, r in (("base", base_ref), ("edit", edit_ref), ("e1_winner", base_ref)):
                grows.append(
                    {"panel": panel, "model": model, "id": f"{panel}-{i}", "refused": bool(r)}
                )
    _jsonl_gz(d / "ste_guards.jsonl.gz", grows)
    ste = {
        "margins": {"guard_harmless": 0.1, "guard_harmful": 0.1},
        "null_pass_prob": {"both": 0.9},
        "calibration": {},
        "outputs": "ste_outputs.jsonl.gz",
        "guards": "ste_guards.jsonl.gz",
        "n": {"mmlupro_gen": 150, "supergpqa_gen": 150},
        "canary": {"ok": True},
    }
    files = {
        "bank.json": make_bank(rng),
        "fits.json": {
            "mode": mode,
            "coverage": {
                "wm": {"coverage": 0.95, "n": 40, "pass": True},
                "static": {"coverage": 0.7, "n": 40, "pass": False},
            },
            "q_hat": {"wm": 1.0, "static": 1.0},
            "shrink": {"wm": 0.25, "static": 0.0},
            "c2i": {
                arm: [
                    {"spec": f"i{j}", "pred": [0.01 * (j % 3)] * 6, "sigma": [0.05] * 6}
                    for j in range(12)
                ]
                for arm in ("wm", "static")
            },
            "f_k": [0.05] * 6,
            "components": [],
            "digests": {},
        },
        "e2.json": {
            "B": 48,
            "episodes": episodes,
            "arms": arms,
            "sbon": {
                "B_par": 86.72,
                "inputs": {"bank_gpu_s": 1.0, "field_gpu_s": 1.0, "n_ep": 24, "t_bundle_s": 14.58},
                "K": 4,
                "P": 0.25,
            },
        },
        "winners.json": {"sha256": "0" * 64, "e1": e1},
        "sealed.json": sealed,
        "gen_e1.json": gen,
        "ste.json": ste,
        "ledger.json": {
            "gpu_h": {"bank": 3.0, "e2": 7.0},
            "instance_gpu_h": 10.5,
            "charged_gpu_h": 10.0,
            "usd": 10.0 * 4.89 * 1.10,
            "instances": [
                {
                    "id": "i-synthetic",
                    "type": "gpu_1x_h100_sxm5",
                    "gpus": 1,
                    "launched_at": "t0",
                    "terminated_at": "t1",
                    "terminated_confirmed": True,
                }
            ],
            "relaunches": 0,
            "repairs": [],
            "amendments": ["2026-11-20: synthetic fixture amendment (test only)"],
        },
    }
    for name, body in files.items():
        (d / name).write_text(json.dumps(body))
    names = [
        *files,
        "gen_e1_items.jsonl.gz",
        "gen_e1_outputs.jsonl.gz",
        "ste_outputs.jsonl.gz",
        "ste_guards.jsonl.gz",
    ]
    idx = {"files": {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in names}}
    (d / "bundle.json").write_text(json.dumps(idx))
    return d


def make_bank(rng, n_comp=10, n_items=8):
    groups = {o: [f"b{i // 2}" for i in range(n_items)] for o in SCH.OBJ6}
    base_vals = {o: rng.normal(0, 1, n_items).tolist() for o in SCH.OBJ6}
    sketch = {o: rng.normal(0, 1, 4).tolist() for o in SCH.OBJ6}
    comps = [f"c{i}" for i in range(n_comp)] + ["cal0"]
    states = []
    for ci, c in enumerate(comps):
        a = rng.normal(0, 0.3, 6)
        q = rng.normal(0, 0.05, 6)
        for h in range(1, 5):
            dose = 0.5 * h
            r = np.concatenate([a * dose, q * dose**2, [0.0], [dose], [0.0], [dose]])
            y = a * dose + q * dose**2 + 0.02 * dose**3
            vals = {
                o: (np.asarray(base_vals[o]) + y[k] + rng.normal(0, 0.05, n_items)).tolist()
                for k, o in enumerate(SCH.OBJ6)
            }
            states.append(
                {
                    "state_id": f"{c}-{h}",
                    "kind": "ladder",
                    "component": c,
                    "family": "steer",
                    "roots": [c],
                    "side": "pos",
                    "h": h,
                    "dose": dose,
                    "prefix": None,
                    "theta_hash": f"{c}-{h}",
                    "values": vals,
                    "sketch": {o: (np.asarray(sketch[o]) + 0.1 * h).tolist() for o in SCH.OBJ6},
                    "R": r.tolist(),
                    "R_flags": [],
                    "dnll": float(y[0]),
                    "endpoint": False,
                    "history": [f"{c}-{j}" for j in range(1, h + 1)],
                }
            )
    return {
        "items": {
            "ids": {o: [f"{o}{i}" for i in range(n_items)] for o in SCH.OBJ6},
            "groups": groups,
        },
        "base": {"values": base_vals, "sketch": sketch, "theta_hash": "base"},
        "graph": {
            "components": [{"id": c, "stratum": "singleton", "roots": [c]} for c in comps],
            "calibration": ["cal0"],
            "folds": {"outer": {c: i % 5 for i, c in enumerate(comps[:-1])}},
        },
        "states": states,
        "inert": [
            {
                "spec": "i1_2step",
                "states": [],
                "endpoint_theta_is_base": True,
                "endpoint_outputs_bitwise_base": True,
            }
        ],
        "fallbacks": [],
    }


def run(d, **kw):
    return A.analyse(d, **{**FAST, "run_dht": False, **kw})


# ------------------------------------------------------------------ tests


def test_registered_defaults():
    import inspect

    sig = inspect.signature(A.analyse).parameters
    assert sig["n_boot"].default == A.N_BOOT == 9999
    assert sig["n_flip"].default is None and sig["rb_n_boot"].default == SF.RB_N_BOOT == 9999
    assert sig["gen_n_boot"].default is None and sig["run_dht"].default is True
    m = A.load_manifest()
    assert m["_sha256"] == A.FREEZE_MANIFEST_SHA256
    assert m["inference.e2"]["n_flip"] == 10_000 and m["seeds.signflip.e2"] == 2168497238
    assert E2.n_bounds("sycophancy_real") == 13


def test_tango_p_inverts_tango_lower():
    from wsp.wm import confirm_stats as CS

    for b, c, n in ((30, 10, 200), (12, 8, 100), (5, 0, 50), (29, 15, 150)):
        p = A.tango_p(b, c)
        assert CS.tango_lower(b, c, n, p * 0.999) < 0 < CS.tango_lower(b, c, n, p * 1.001)
    assert A.tango_p(0, 0) == 1.0


def test_guarded_score_min_rule_and_base_zero(tmp_path):
    gains = [0.3, -0.2, None, 0.3] + [None] * 20
    d = make_bundle(tmp_path, wm_gains=gains, breach={("wm_e", 0), ("wm_e", 1)})
    b = A.load_bundle(d)
    s = A.guarded_scores(b["e2"], b["sealed"], arms=["wm_e"])["wm_e"]
    assert (
        s[0]["g"] > 0.2 and not s[0]["guard_points_ok"] and s[0]["g_tilde"] == 0.0
    )  # unsafe: no credit
    assert (
        s[1]["g"] < -0.1 and not s[1]["guard_points_ok"] and s[1]["g_tilde"] == s[1]["g"]
    )  # can lose
    assert s[2]["base"] and s[2]["g_tilde"] == 0.0
    assert s[3]["guard_points_ok"] and s[3]["g_tilde"] == pytest.approx(s[3]["g"])
    assert s[3]["g"] == pytest.approx(0.3, abs=0.05)


def test_he1_supported_wording(tmp_path):
    r = run(make_bundle(tmp_path))
    e1 = r["h_e1"]
    assert e1["supported"] and e1["benefit"]["pass"] and e1["guards"]["pass"]
    assert e1["designation_consistent"] and e1["gen_winner_theta_consistent"]
    assert e1["guards"]["n_bounds"] == 13 and len(e1["guards"]["rows"]) == 13
    assert e1["statement"]["category"] == "supported"
    assert e1["statement"]["statement"].endswith(
        "with generated transfer on 160 TriviaQA-derived items, greedy non-thinking decoding at batch 128."
    )
    assert e1["statement"]["statement"].startswith(
        "Verified enhancement: the WM-E-planned edit lowered"
    )
    gd = e1["generation"]["descriptive"]
    assert gd["base"]["adoption_rate"] > gd["winner"]["adoption_rate"]
    assert "sampled:20261117" in e1["generation"]["sampled"]


def test_he1_guard_failure_is_conjunctive(tmp_path):
    gains = [0.3 + 0.01 * e for e in range(24)]
    top = max(range(24), key=lambda e: gains[e])
    r = run(make_bundle(tmp_path, wm_gains=gains, breach={("wm_e", top)}))
    e1 = r["h_e1"]
    assert e1["benefit"]["pass"] and e1["benefit"]["lcb"] > 0  # positive benefit ...
    assert not e1["guards"]["pass"] and e1["guards"]["breaches"] == ["wikitext_nll:native"]
    assert not e1["supported"]  # ... still not established
    assert e1["statement"]["statement"] == (
        "Enhancement not established: guardrail wikitext_nll:native did not meet its non-inferiority bound."
    )


def test_he1_generation_failure_and_no_winner(tmp_path):
    r = run(make_bundle(tmp_path, gen_transfer=False))
    assert r["h_e1"]["benefit"]["pass"] and r["h_e1"]["guards"]["pass"]
    assert r["h_e1"]["statement"]["statement"] == (
        "Teacher-forced gain without verified generated transfer; enhancement not established."
    )
    r = run(make_bundle(tmp_path, wm_gains=[None] * 24, e1_designated=False, seed=1))
    assert (
        r["h_e1"]["statement"]["statement"]
        == "Enhancement not established — no feasible WM winner."
    )
    assert r["h_e1"]["designation_consistent"]


def test_he1_benefit_failure(tmp_path):
    gains = [-0.3] + [None] * 23
    r = run(make_bundle(tmp_path, wm_gains=gains))
    assert not r["h_e1"]["benefit"]["pass"]
    assert r["h_e1"]["statement"]["statement"] == "Enhancement not established."


def test_he2_signflip_wording_switch(tmp_path):
    r = run(make_bundle(tmp_path))  # WM finds gains, StatNN always base: every difference positive
    p = r["h_e2"]["primary"]
    assert p["reject"] and p["n_episodes"] == 24 and p["lb"] > 0
    assert p["statement"] == (
        "The WM-guided minus StatNN-guided paired differences are not sign-symmetric about zero, in the direction "
        "favouring WM, conditional on these test items and this edit library."
    )
    assert "rb_e2" in p and p["ub"] >= p["stat"] >= p["lb"]
    rng = np.random.default_rng(3)
    same = [float(x) for x in rng.normal(0.2, 0.1, 24)]
    r = run(make_bundle(tmp_path, wm_gains=same, stat_gains=same[::-1], mode="ranks", seed=2))
    p = r["h_e2"]["primary"]
    assert not p["reject"]
    assert (
        p["statement"]
        == "A WM-E guidance advantage over StatNN guidance was not established (ranks-only mode)."
    )
    r = run(make_bundle(tmp_path, mode="ranks", seed=4))
    assert r["h_e2"]["primary"]["statement"].endswith("this edit library (ranks-only mode).")


def test_holm_gating_of_secondaries(tmp_path):
    rng = np.random.default_rng(5)
    same = [float(x) for x in rng.normal(0.2, 0.1, 24)]
    sec = {"random": [None] * 24, "field_ranked": [None] * 12 + [0.1] * 12}
    r = run(
        make_bundle(tmp_path, wm_gains=same, stat_gains=same[::-1], sec_gains=sec, sbon_gains=same)
    )
    s = r["h_e2"]["secondary"]
    assert not r["h_e2"]["primary"]["reject"] and not s["gated_open"]
    for arm, c in s["contrasts"].items():
        assert c.get("tested") in (False, None) and c.get("holm_reject") is None, arm
    assert not s["contrasts"]["wm_greedy"]["ran"]
    sb = r["h_e2"]["sbon"]  # reported whatever the result
    assert all(k in sb for k in ("stat", "lb", "ub", "p", "statement"))
    # open gate: WM beats StatNN; Holm decides over the arms that ran (random: WM always ahead)
    r = run(make_bundle(tmp_path, sec_gains=sec, seed=6))
    s = r["h_e2"]["secondary"]
    assert (
        s["gated_open"] and s["family_size"] == 10
    )  # registered family; arms not run enter with p = 1
    assert s["contrasts"]["random"]["tested"] and s["contrasts"]["random"]["holm_reject"]
    assert s["contrasts"]["random"]["statement"].startswith(
        "The WM-guided minus random-guided paired differences are not sign-symmetric about zero"
    )
    assert s["contrasts"]["random"]["holm_level"] == pytest.approx(0.025 / 10)
    assert not s["contrasts"]["wm_greedy"]["ran"] and not s["contrasts"]["wm_greedy"]["holm_reject"]
    sb = s["contrasts"]["sbon"]
    assert (
        sb["holm_reject"]
        and "best-of-N paired differences at total-compute parity are not" in sb["statement"]
    )


def test_sbon_mandatory_wording_when_bound_nonpositive(tmp_path):
    wm = [0.1] * 24
    sbon = [0.4] * 24
    r = run(make_bundle(tmp_path, wm_gains=wm, sbon_gains=sbon))
    sb = r["h_e2"]["sbon"]
    assert sb["lb"] <= 0 and sb["stat"] < 0
    assert sb["statement"] == A.WORDING["sbon_mandatory"]
    assert "simulated power 0.024 at the measured parity budget" in sb["statement"]
    assert sb["budget"]["B_par"] == 86.72


def test_ste_holm_and_wording(tmp_path):
    r = run(make_bundle(tmp_path))
    te = r["s_te"]["benchmarks"]
    m, s = te["mmlupro_gen"], te["supergpqa_gen"]
    assert m["pass"] and not s["pass"]
    assert (
        m["primary"]["holm_level"] == pytest.approx(0.0125) and s["primary"]["holm_level"] == 0.025
    )
    assert m["statement"].startswith(
        "At a frozen 2,048-token non-thinking budget, the retrained H1-type adapter raised generated accuracy on "
        "MMLU-Pro (token efficiency; truncation 0.200 → 0.000), with the 4,096-token and untruncated-in-both"
    )
    assert (
        s["statement"]
        == "A token-efficiency gain at the frozen budget was not established on SuperGPQA."
    )
    assert m["r2"]["criterion"] in (0, 1)
    u = m["budget_2048"]["untruncated_in_both"]
    assert u["delta"] == 0.0 and u["degenerate"] and u["lcb"] == 0.0
    # Holm: both p in (0.0125, 0.025): neither passes although each Tango LB at 0.025 is > 0
    r = run(
        make_bundle(tmp_path, ste_bc={"mmlupro_gen": (30, 15), "supergpqa_gen": (29, 15)}, seed=7)
    )
    te = r["s_te"]["benchmarks"]
    for k in ("mmlupro_gen", "supergpqa_gen"):
        assert 0.0125 < te[k]["primary"]["p"] < 0.025 and te[k]["primary"]["lb"] > 0
        assert not te[k]["primary"]["holm_reject"] and not te[k]["pass"]
        assert te[k]["statement"].startswith(
            "A token-efficiency gain at the frozen budget was not established"
        )
    # guards are conjunctive
    r = run(make_bundle(tmp_path, ste_guards_fail=True, seed=8))
    assert not r["s_te"]["guards"]["guarded"]
    assert r["s_te"]["benchmarks"]["mmlupro_gen"]["primary"]["holm_lb"] > 0
    assert not r["s_te"]["benchmarks"]["mmlupro_gen"]["pass"]


def test_dc2_rule(tmp_path):
    r = run(make_bundle(tmp_path))
    c = r["d_c2"]["c2i"]
    assert c["wm"]["part_i"] is True and c["wm"]["pass"]
    assert c["static"]["part_i"] == "not applicable (C1 coverage failed)"
    tr = r["d_c2"]["trust_region"]
    assert tr["inside_le_0.1"] + tr["outside_gt_0.1"] == tr["n_states"]


def test_bundle_sha_mismatch_rejected(tmp_path):
    d = make_bundle(tmp_path)
    (d / "ledger.json").write_text("{}")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        A.load_bundle(d)


def test_registration_quotes_verbatim():
    q = A.registration_quotes()
    assert q["a_negative"].startswith(
        "\"On Qwen3.5-9B, edit-specific trajectory prediction did not beat path-type averages: stage W2's nested"
    )
    assert "zero-shot mean d 0.003, Holm-adjusted lower bound −0.053" in q["a_negative"]
    for s in (
        "0.077 (22/284)",
        "1/107, mean Ĉ_T −0.155",
        "118 vs 48 of",
        "1,420",
        "22 rejections exceed",
    ):
        assert s in q["r6_caveat"]
    assert "43/54 steps" in q["d_s"]
    assert "0.624" in q["stage_w"] and "−0.053" in q["stage_w2"]
    assert (
        "N is frozen at 3,398" in q["mmlupro_n_disclosure"] and "0.792" in q["mmlupro_n_disclosure"]
    )
    assert all(v["sha256_verified"] for v in q["stage_reports"].values())


def test_report_end_to_end_with_dht(tmp_path):
    from wsp.wm import wmcp as CP

    d = make_bundle(tmp_path, mode="ranks")
    cfg = CP.CPConfig(hidden=8, latent=4, epochs=3)
    md, res = A.report(d, date="2026-11-20", **FAST, dht_cfg=cfg, dht_n_seeds=1)
    assert res["d_ht"]["computed"], res["d_ht"]
    assert res["d_ht"]["statement"].startswith("Descriptive, not powered: C_T = ")
    assert math.isfinite(float(res["d_ht"]["ct"]))
    json.dumps(res, allow_nan=False)
    low = md.lower()
    for bad in (
        "improves accuracy",
        "wm-e worked",
        "static suffices",
        "mean advantage",
        "mean-advantage",
        "wm-e succeeded",
    ):
        assert bad not in low
    for section in (
        "## BLUF",
        "## H-E1",
        "## H-E2",
        "### H-E2 secondary Holm family of 10",
        "## S-TE",
        "### D-HT",
        "### D-HT2",
        "### D-S",
        "### D-C2",
        "## (a) Trajectory prediction",
        "## Amendments and deviations",
        "## Ledger",
    ):
        assert section in md
    assert md.index("## BLUF") < md.index("## H-E1") < md.index("## H-E2")
    assert "GSM8K and MATH E1 secondaries: unresolved — not powered; not run." in md
    assert "synthetic fixture amendment (test only)" in md
    assert "Non-registered resampling counts were used" in md  # fixture mode is flagged
    assert "EXPLORATORY" in md and "0.077 (22/284)" in md
    assert "N is frozen at 3,398" in md
    assert res["ledger"]["usd_consistent"] and res["ledger"]["all_terminated_confirmed"]


def test_forbidden_phrase_guard(tmp_path, monkeypatch):
    d = make_bundle(tmp_path)
    res = run(d)
    res["h_e1"]["statement"]["statement"] = "The edit improves accuracy."
    with pytest.raises(ValueError, match="forbidden"):
        A.render_markdown(res)


def test_script_writes_outputs(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wme_c1_report", A.REPO / "scripts" / "wme_c1_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = make_bundle(tmp_path)
    orig = A.report
    monkeypatch.setattr(A, "report", lambda *a, **k: orig(*a, **{**k, **FAST}))
    out_md, out_json = tmp_path / "r.md", tmp_path / "r.json"
    assert (
        mod.main(
            [
                "--bundle",
                str(d),
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
                "--no-dht",
                "--date",
                "2026-11-20",
            ]
        )
        == 0
    )
    assert out_md.read_text().startswith(
        "# WM-E wme-c1 on Qwen3.5-9B: confirmatory results (2026-11-20)"
    )
    assert json.loads(out_json.read_text())["d_ht"]["computed"] is False
