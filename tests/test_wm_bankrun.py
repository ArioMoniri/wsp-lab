"""Bank and confirmatory run plan for the multi-GPU Lambda runner (PREREG_WME v1 §4.2-4.5; RULING §2)."""

import json

import numpy as np
import pytest
import torch

from wsp.wm import actions as A
from wsp.wm import bankrun as BR


def _roots():
    return [
        {"id": "scale:001", "family": "scale", "tensor": [1, "down"], "nominal": 1.5},
        {
            "id": "proj_remove:002",
            "family": "proj_remove",
            "tensor": [[0, "down"], [1, "down"]],
            "layers": [0, 1],
            "nominal": 0.5,
        },
        {"id": "noise:003", "family": "noise", "tensor": [0, "o"], "seed": 5},
    ]


def test_ladder_doses_and_duplicate_rule():
    pos, neg = BR.ladder_sides("proj_remove")
    assert pos == [0.25, 0.5, 1.0, 2.0]  # +4, +8 duplicate alpha_eff = 1
    assert neg == [-0.25, -0.5, -1.0, -2.0, -4.0, -8.0]
    p, n = BR.ladder_sides("scale")
    assert p == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] and n == [
        -0.25,
        -0.5,
        -1.0,
        -2.0,
        -4.0,
        -8.0,
    ]  # signed, no cap


def test_plan_sides_compositions_and_controls():
    blocks = [["scale:001", "proj_remove:002", "noise:003"]]
    plan = BR.plan(
        _roots(), blocks, controls=[{"id": "proj_remove:002:control", "partner": "proj_remove:002"}]
    )
    sides = [u for u in plan if u["kind"] == "side"]
    comps = [u for u in plan if u["kind"] == "composition"]
    assert len(sides) == 2 * 3 + 2  # control gets both sides
    assert len(comps) == 2 and comps[0]["roots"] == list(reversed(comps[1]["roots"]))
    assert all(len({u["key"] for u in plan}) == len(plan) for _ in [0])
    h = [s["h"] for s in sides[0]["states"]]
    assert h == list(range(1, len(h) + 1))


def test_stop_rule_walks_outward_and_keeps_endpoint():
    out = BR.walk([0.25, 0.5, 1, 2, 4, 8], lambda d: 0.3 * abs(d))
    assert [r["dose"] for r in out] == [0.25, 0.5, 1, 2, 4] and out[-1]["endpoint"]


def test_actions_from_root_recipes():
    rm = A.RoleMap(2, 8, 16, 32, ("linear_attention", "full_attention"), tied=False)
    art = {"proj_remove:002": {"directions": {0: torch.ones(8), 1: torch.ones(8)}}}
    acts = BR.actions_for(_roots()[1], 2.0, rm, art)
    assert len(acts) == 2 and all(
        a.param("alpha") == 1.0 for a in acts
    )  # alpha_eff = 0.5 * 2 capped at 1
    acts = BR.actions_for(_roots()[1], 8.0, rm, art)
    assert all(a.param("alpha") == 1.0 for a in acts)
    acts = BR.actions_for(_roots()[0], -4.0, rm, art)
    assert acts[0].param("factor") == 1.5 and acts[0].dose == -4.0
    acts = BR.actions_for(_roots()[2], -1.0, rm, art)
    assert acts[0].family == "noise" and acts[0].dose == -1.0


def test_output_paths_are_the_guarded_ones(tmp_path):
    assert BR.RUNS_ROOT == "data/wme/runs/wme-c1" and BR.WINNERS_ROOT == "data/wme/winners/wme-c1"
    assert (
        BR.run_dir(tmp_path, "bank", "r1").relative_to(tmp_path).as_posix()
        == "data/wme/runs/wme-c1/bank/r1"
    )
    with pytest.raises(BR.FreezeGate):
        BR.require_freeze(tmp_path, panel="wme-c1")
    (tmp_path / "docs/prereg").mkdir(parents=True)
    (tmp_path / "docs/prereg/wme_freeze.json").write_text(json.dumps({"manifest_sha256": "x" * 64}))
    BR.require_freeze(tmp_path, panel="wme-c1")
    BR.require_freeze(tmp_path / "nofreeze", panel="wme-p2")  # smoke panels never need the freeze
    assert BR.smoke_dir(tmp_path, "s").relative_to(tmp_path).as_posix() == "data/wme/runs/smoke/s"


def test_side_units_cover_all_states_once():
    plan = BR.plan(_roots(), [], controls=[])
    states = [(u["key"], s["dose"]) for u in plan for s in u["states"]]
    assert len(states) == len(set(states))
    assert sum(len(u["states"]) for u in plan) == sum(map(len, BR.ladder_sides("scale"))) + sum(
        map(len, BR.ladder_sides("proj_remove"))
    ) + sum(map(len, BR.ladder_sides("noise")))
    assert np.all([u["kind"] == "side" for u in plan])


def test_control_records_with_family_key_merge():
    ctrl = [
        {
            "id": "proj_remove:002:control",
            "partner": "proj_remove:002",
            "family": "proj_remove_control",
            "control_seed": 7,
        }
    ]
    plan = BR.plan(_roots(), [], controls=ctrl)
    assert sum(u["root"] == "proj_remove:002:control" for u in plan if u["kind"] == "side") == 2
