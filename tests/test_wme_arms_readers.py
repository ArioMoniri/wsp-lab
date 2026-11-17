"""wm.arms.readers: scripted reader, WSR-style single LLM reader and the Fugu-style Thinker/Worker/Verifier team
(access lists, ACCEPT/REVISE), strict parsing, shared token cap; all offline with MockClient."""

import json

import pytest

from wsp.wm import access as AC
from wsp.wm.arms import llm as L
from wsp.wm.arms import pool as P
from wsp.wm.arms import readers as R
from wsp.wm.arms import spec as S
from wsp.wm.plan import BudgetLedger

BASE = "b" * 64
TASK = R.ReaderTask(objective="sycophancy_real")


def _views(n=6):
    out = []
    for i in range(n):
        out.append(
            {
                "id": f"c{i:02d}",
                "steps": [
                    {
                        "pid": 10 + i,
                        "family": "steer",
                        "layer": 20 + i,
                        "contrast": "syc" if i % 2 else "gsm",
                        "param": 0.0625,
                        "rel_norm": 1e-3 * (i + 1),
                    }
                ],
                "rel_norm_total": 1e-3 * (i + 1),
                "field": {
                    "sycophancy_real": {"a": -0.01 * i, "q": 1e-4},
                    "wikitext_nll": {"a": 0.0, "q": None},
                },
            }
        )
    return out


def _hist():
    return [
        {
            "candidate": (11,),
            "view": _views()[1],
            "gain": 0.05,
            "se": 0.01,
            "lcb": 0.0304,
            "guard_ok": True,
            "order": 0,
        },
        {
            "candidate": (12,),
            "view": _views()[2],
            "gain": 0.09,
            "se": 0.01,
            "lcb": 0.07,
            "guard_ok": False,
            "order": 1,
        },
    ]


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "text,ok",
    [
        ('{"ranking": ["c01", "c00"]}', True),
        ('```json\n{"ranking": ["c01", "c00"]}\n```', True),
        ('Sure! {"ranking": ["c01", "c00"]}', False),
        ('{"ranking": ["c01", "c01"]}', False),
        ('{"ranking": ["c01", "c77"]}', False),
        ('{"ranking": ["c01"]}', False),
        ('{"ranking": "c01,c00"}', False),
        ("[]", False),
        ("", False),
    ],
)
def test_parse_ranking_is_strict(text, ok):
    ids = [v["id"] for v in _views()]
    if ok:
        assert R.parse_ranking(text, ids, b=2) == ["c01", "c00"]
    else:
        with pytest.raises(R.ParseFailure):
            R.parse_ranking(text, ids, b=2)


# ------------------------------------------------------------------ scripted


def test_scripted_reader_rule():
    res = R.ScriptedReader(TASK).rank(_views(), _hist(), b=2, round_index=1)
    assert res.failure is None and res.usage == []
    order = res.order
    # c02 shares a primitive with a guard-breaching own result -> last
    assert order[-1] == "c02"
    # c01 shares the best feasible incumbent's primitive -> first
    assert order[0] == "c01"
    # then primary-contrast matches (syc: c03, c05) before others, by oriented field a
    assert order[1:3] == ["c05", "c03"]
    assert R.ScriptedReader(TASK).rank(_views(), _hist(), 2, 1).order == order  # deterministic


# ------------------------------------------------------------------ WSR


def test_wsr_reader_prompt_parse_and_accounting():
    m = L.MockClient(['{"ranking": ["c03", "c01", "c00"]}'])
    led = L.TokenLedger(cap=50_000)
    res = R.WSRReader(m, led, TASK).rank(_views(), _hist(), b=2, round_index=0)
    assert res.failure is None and res.order == ["c03", "c01", "c00"]
    system, user = m.calls[0]
    assert "zero-shot" in system and '"ranking"' in system
    assert "sycophancy_real" in user and "c05" in user and "unsupported" in user
    assert "dev gain +0.0500" in user  # own result
    for banned in ("bank", "trajectory", "predictor"):
        assert banned not in (system + user).lower()
    assert led.report()["by_role"]["reader"]["calls"] == 1 and res.usage[0]["role"] == "reader"


def test_wsr_truncation_and_garbage_are_failures():
    led = L.TokenLedger(cap=50_000)
    long = '{"ranking": [' + ", ".join(['"c00"'] * 2000) + "]}"
    r1 = R.WSRReader(L.MockClient([long]), led, TASK, max_output_tokens=50).rank(_views(), [], 2, 0)
    assert r1.failure == "truncated:reader" and r1.order is None
    r2 = R.WSRReader(L.MockClient(["I would pick c01"]), led, TASK).rank(_views(), [], 2, 0)
    assert r2.failure == "unparseable:reader"


# ------------------------------------------------------------------ Fugu-style team


def _team(
    worker_replies,
    verifier_replies,
    thinker='{"notes": "favour syc contrasts at small norm"}',
    cap=100_000,
):
    clients = {
        "thinker": L.MockClient([thinker] * 8, model="t-model"),
        "worker": L.MockClient(worker_replies, model="w-model"),
        "verifier": L.MockClient(verifier_replies, model="v-model"),
    }
    led = L.TokenLedger(cap=cap)
    return R.FuguTeam(clients, led, TASK), clients, led


def test_team_accept_path_and_access_lists():
    team, cl, led = _team(['{"ranking": ["c05", "c03"]}'], ['{"verdict": "ACCEPT"}'])
    res = team.rank(_views(), _hist(), b=2, round_index=0)
    assert res.failure is None and res.order == ["c05", "c03"]
    t_sys, t_user = cl["thinker"].calls[0]
    w_sys, w_user = cl["worker"].calls[0]
    v_sys, v_user = cl["verifier"].calls[0]
    # thinker: pool + own results, never a ranking
    assert "dev gain +0.0500" in t_user and "favour" not in t_user
    # worker: pool + thinker notes, NOT the team's dev results (access list)
    assert "favour syc contrasts" in w_user and "dev gain" not in w_user
    # verifier: pool + own results + worker ranking, NOT the thinker's notes
    assert '"c05", "c03"' in v_user and "dev gain +0.0500" in v_user and "favour" not in v_user
    assert "Thinker" in t_sys and "Worker" in w_sys and "Verifier" in v_sys
    rep = led.report()
    assert {r: rep["by_role"][r]["calls"] for r in ("thinker", "worker", "verifier")} == {
        "thinker": 1,
        "worker": 1,
        "verifier": 1,
    }
    assert rep["by_role"]["worker"]["providers"] == ["mock:w-model"]


def test_team_revise_path_uses_one_revision():
    team, cl, _ = _team(
        ['{"ranking": ["c02", "c00"]}', '{"ranking": ["c01", "c05"]}'],
        ['{"verdict": "REVISE", "critique": "c02 repeats a guard-breaching edit"}'],
    )
    res = team.rank(_views(), _hist(), b=2, round_index=0)
    assert res.failure is None and res.order == ["c01", "c05"]
    assert len(cl["worker"].calls) == 2 and len(cl["verifier"].calls) == 1
    assert "c02 repeats a guard-breaching edit" in cl["worker"].calls[1][1]
    assert "dev gain" not in cl["worker"].calls[1][1]


def test_team_failures():
    team, _, _ = _team(['{"ranking": ["c05", "c03"]}'], ['{"verdict": "MAYBE"}'])
    assert team.rank(_views(), [], 2, 0).failure == "unparseable:verifier"
    team, _, _ = _team(['{"ranking": ["c05"]}'], ['{"verdict": "ACCEPT"}'])
    assert team.rank(_views(), [], 2, 0).failure == "unparseable:worker"
    team, _, _ = _team(
        ['{"ranking": ["c05", "c03"]}'], ['{"verdict": "ACCEPT"}'], thinker="notes: none"
    )
    assert team.rank(_views(), [], 2, 0).failure == "unparseable:thinker"


def test_role_access_is_enforced_in_prompt_composition():
    with pytest.raises(S.AccessDenied):
        R.compose("worker", {"episode_pool": "x", "own_dev_results": "y"})
    with pytest.raises(S.AccessDenied):
        R.compose("verifier", {"thinker_output": "x"})
    assert "POOL" in R.compose("thinker", {"episode_pool": "x", "own_dev_results": "y"})


def test_shared_token_cap_exhaustion_fails_later_rounds_without_calls():
    reply = json.dumps({"ranking": ["c00", "c01"]})
    probe, _, pled = _team([reply], ['{"verdict": "ACCEPT"}'])
    assert probe.rank(_views(), [], 2, 0).failure is None
    one_round = pled.spent
    team, cl, led = _team([reply] * 8, ['{"verdict": "ACCEPT"}'] * 8, cap=one_round + 600)
    outcomes = [team.rank(_views(), [], 2, r).failure for r in range(4)]
    assert outcomes[0] is None
    assert outcomes[1:] == ["over_cap_precheck:thinker"] * 3
    n_calls = sum(len(c.calls) for c in cl.values())
    assert n_calls == 3  # refused calls are never sent
    assert led.spent == one_round <= led.cap


def test_readers_share_the_registered_cap_and_end_to_end_parity():
    assert S.ARM_SPECS["wsr_reader"].token_cap == S.ARM_SPECS["fugu_team"].token_cap
    sub = list(range(10, 22))
    meta = {
        p: {"family": "steer", "layer": 20, "contrast": "syc", "param": 0.0625, "rel_norm": 1e-3}
        for p in sub
    }

    def respond(system, user):
        ids = [ln.split(" |")[0] for ln in user.splitlines() if ln.startswith("c") and " | " in ln]
        return json.dumps({"ranking": ids[:3]})

    led = BudgetLedger(8, ["wsr_reader"])
    ctx = S.ArmContext(
        S.ARM_SPECS["wsr_reader"],
        led,
        "e0",
        lambda c: {
            "gain": 0.01 * len(c),
            "se": 0.004,
            "guards": [(0.0, 0.001, 0.01)],
            "rel_norm": 1e-3,
        },
        theta_of=lambda c: repr(c),
        resources={
            "episode_library": sub,
            "episode_pool": None,
            "r0_candidate_description": AC.Artefact("meta", BASE, AC.Tier.R0, meta),
        },
        stores=S.StoreRegistry(),
        guard=AC.R0Guard(BASE),
    )
    tl = L.TokenLedger(cap=200_000)
    out = P.run_pool_episode(
        R.WSRReader(L.MockClient(respond), tl, TASK),
        ctx,
        B=8,
        pool_size=10,
        seed=3,
        features=lambda c: {"sycophancy_real": {"a": -0.001, "q": None}},
    )
    assert out["failures"] == 0 and led.verify_parity()["units"]["wsr_reader"] == 8.0
    assert tl.report()["calls"] == 4
