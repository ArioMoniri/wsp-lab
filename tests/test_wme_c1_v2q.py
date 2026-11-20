"""Phase L part 1 incident (20 Nov 2026): lambda_ops.tighten wrote a non-policy key (cap_generation) into the
dead-man's policy.json; the dead-man refuses unknown keys, so its watcher errored on every tick and the poller's guarded
launch raised before the launch call (nothing launched). The policy file must always be accepted by the real
dead-man; the cap-generation marker lives in its own file. CPU only."""

import json

import pytest

from wsp.ops import lambda_deadman as D
from wsp.wm.c1 import lambda_ops as LO


def base_policy():  # as ensure_watcher / the poller construct it (cap in force, $1,000, 1,800 s idle, 9.6 $/h)
    return D.Policy(idle_limit_s=1800.0, t_max_hours=110.0, max_usd=1000.0, max_price_usd_h=9.6)


def test_policy_file_is_always_accepted_by_the_real_deadman(tmp_path):
    LO.tighten(tmp_path, 526.9, 55.0, cap_generation=0)
    pol = json.loads((tmp_path / "policy.json").read_text())
    assert set(pol) <= {f for f in vars(D.Policy())}  # dead-man keys only
    tightened = base_policy().tightened(
        pol
    )  # raised ValueError('unknown policy key cap_generation') before
    assert tightened.max_usd == 526.9 and tightened.t_max_hours == 55.0
    assert json.loads((tmp_path / "policy_cap.json").read_text()) == {"cap_generation": 0}


def test_cap_generation_rebase_still_works_from_its_own_file(tmp_path):
    p1 = LO.tighten(tmp_path, 400.0, 10.0)
    assert p1["max_usd"] == 400.0
    assert LO.tighten(tmp_path, 500.0, 10.0)["max_usd"] == 400.0  # only tightens
    assert LO.tighten(tmp_path, 500.0, 10.0, cap_generation=0)["max_usd"] == 500.0  # re-based once
    assert LO.tighten(tmp_path, 600.0, 10.0, cap_generation=0)["max_usd"] == 500.0
    assert LO.tighten(tmp_path, 650.0, 10.0, cap_generation=1)["max_usd"] == 650.0
    base_policy().tightened(json.loads((tmp_path / "policy.json").read_text()))


def test_a_legacy_policy_file_with_the_marker_is_repaired(tmp_path):
    (tmp_path / "policy.json").write_text(
        json.dumps(
            {"max_usd": 526.9, "t_max_hours": 55.0, "idle_limit_s": 1800.0, "cap_generation": 0}
        )
    )
    with pytest.raises(ValueError, match="unknown policy key"):
        base_policy().tightened(json.loads((tmp_path / "policy.json").read_text()))
    LO.tighten(tmp_path, 526.9, 55.0, cap_generation=0)
    pol = json.loads((tmp_path / "policy.json").read_text())
    assert "cap_generation" not in pol
    base_policy().tightened(pol)
    assert json.loads((tmp_path / "policy_cap.json").read_text())["cap_generation"] == 0


def test_the_watcher_reload_path_accepts_the_written_policy(tmp_path):
    """The exact path that failed: Deadman._reload_policy on the state directory."""
    LO.tighten(tmp_path, 526.9, 55.0, cap_generation=0)
    dm = D.Deadman.__new__(D.Deadman)
    dm.dir, dm.base_policy = tmp_path, base_policy()
    dm._reload_policy()
    assert dm.policy.max_usd == 526.9
