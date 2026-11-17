"""Dead-man terminator for Lambda Cloud (round-9 L4 / C9), tested against a mocked API.

The terminator runs on the AWS server, holds the API key, and may only ever terminate instances that it launched
itself (registered ids, names starting ``wsp-wme-``). It enforces an idle limit, a hard total instance-hour cap
T_max and a dollar cap, confirms termination through the API, alerts when an instance outlives T_max + 15 min,
and logs every action. These tests drive it with a fake clock and a fake API.
"""

from __future__ import annotations

import json
import os

import pytest

from wsp.ops import lambda_deadman as D

KEY = "fake-lambda-key-0123456789abcdef"


class FakeLambda:
    """In-memory Lambda API: instances, instance types and a call log; failures can be injected."""

    def __init__(self, clock):
        self.clock = clock
        self.instances = {}
        self.calls = []
        self.capacity = {"gpu_1x_h100_sxm5": ["us-southeast-1"]}
        self.price_cents = {"gpu_1x_h100_sxm5": 489}
        self.stuck = set()  # ids that ignore terminate
        self.fail_next_launch_after_create = False
        self.error_message = None
        self.n = 0

    def add(self, iid, name, status="active", itype="gpu_1x_h100_sxm5"):
        self.instances[iid] = {
            "id": iid,
            "name": name,
            "status": status,
            "instance_type": {"name": itype, "price_cents_per_hour": self.price_cents[itype]},
            "region": {"name": "us-southeast-1"},
        }

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, json.loads(json.dumps(body)) if body else None))
        if self.error_message:
            return 500, {"error": {"code": "global/unknown", "message": self.error_message}}
        if method == "GET" and path == "/instances":
            return 200, {"data": list(self.instances.values())}
        if method == "GET" and path.startswith("/instances/"):
            iid = path.rsplit("/", 1)[1]
            if iid not in self.instances:
                return 404, {"error": {"code": "global/object-does-not-exist", "message": "nf"}}
            return 200, {"data": self.instances[iid]}
        if method == "GET" and path == "/instance-types":
            return 200, {
                "data": {
                    k: {
                        "instance_type": {"name": k, "price_cents_per_hour": v},
                        "regions_with_capacity_available": [
                            {"name": r} for r in self.capacity.get(k, [])
                        ],
                    }
                    for k, v in self.price_cents.items()
                }
            }
        if method == "POST" and path == "/instance-operations/launch":
            if body["region_name"] not in self.capacity.get(body["instance_type_name"], []):
                return 400, {
                    "error": {
                        "code": "instance-operations/launch/insufficient-capacity",
                        "message": "no capacity",
                    }
                }
            self.n += 1
            iid = f"fake{self.n:04d}"
            self.add(iid, body["name"], status="booting", itype=body["instance_type_name"])
            if self.fail_next_launch_after_create:
                self.fail_next_launch_after_create = False
                raise OSError("connection reset after launch")
            return 200, {"data": {"instance_ids": [iid]}}
        if method == "POST" and path == "/instance-operations/terminate":
            out = []
            for iid in body["instance_ids"]:
                if iid not in self.instances:
                    return 404, {"error": {"code": "global/object-does-not-exist", "message": "nf"}}
                if iid not in self.stuck:
                    self.instances[iid]["status"] = "terminated"
                out.append(self.instances[iid])
            return 200, {"data": {"terminated_instances": out}}
        return 404, {"error": {"code": "global/not-found", "message": path}}

    def terminate_calls(self):
        return [
            b["instance_ids"] for m, p, b in self.calls if p == "/instance-operations/terminate"
        ]

    def launch_calls(self):
        return [b for m, p, b in self.calls if p == "/instance-operations/launch"]


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    api = FakeLambda(clock)
    client = D.LambdaClient(KEY, transport=api)
    policy = D.Policy(
        idle_limit_s=900,
        t_max_hours=3.0,
        max_usd=20.0,
        alert_after_s=900,
        margin_s=300,
        max_price_usd_h=5.5,
    )
    dm = D.Deadman(client, tmp_path / "dm", policy=policy, clock=clock)
    return dm, api, clock, tmp_path


def _launch(dm, name="wsp-wme-27b-calib-t1"):
    return dm.launch("gpu_1x_h100_sxm5", "us-southeast-1", "wsp-lab-aws", name)


def _log(dm):
    return [json.loads(x) for x in dm.log_path.read_text().splitlines()]


def test_launch_registers_and_heartbeat_keeps_alive(env):
    dm, api, clock, _ = env
    ids = _launch(dm)
    assert ids == ["fake0001"]
    assert dm.state()["instances"]["fake0001"]["name"] == "wsp-wme-27b-calib-t1"
    for _ in range(10):
        clock.advance(120)
        dm.heartbeat()
        dm.tick()
    assert api.terminate_calls() == []


def test_idle_terminates_only_own_instance_and_confirms(env):
    dm, api, clock, _ = env
    api.add("foreign1", "acudx-web-prod")
    api.add("foreign2", "wsp-wme-not-mine")  # right prefix but never launched by this terminator
    _launch(dm)
    clock.advance(901)
    dm.tick()
    assert api.terminate_calls() == [["fake0001"]]
    assert api.instances["foreign1"]["status"] == "active"
    assert api.instances["foreign2"]["status"] == "active"
    dm.tick()  # confirmation poll
    inst = dm.state()["instances"]["fake0001"]
    assert inst["confirmed_terminated_at"] is not None
    events = [e["event"] for e in _log(dm)]
    assert "terminate" in events and "confirmed_terminated" in events
    term = next(e for e in _log(dm) if e["event"] == "terminate")
    assert term["reasons"] == ["idle"]
    assert dm.live() == []


def test_confirmation_accepts_not_found(env):
    dm, api, clock, _ = env
    _launch(dm)
    clock.advance(901)
    dm.tick()
    del api.instances["fake0001"]
    dm.tick()
    assert dm.state()["instances"]["fake0001"]["confirmed_terminated_at"] is not None


def test_tmax_terminates_despite_heartbeat(env):
    dm, api, clock, _ = env
    _launch(dm)
    for _ in range((3 * 3600 - 300) // 60):
        clock.advance(60)
        dm.heartbeat()
        dm.tick()
        if api.terminate_calls():
            break
    assert api.terminate_calls() == [["fake0001"]]
    used = dm.instance_hours()
    assert 3.0 - 300 / 3600 - 1 / 60 <= used <= 3.0 - 300 / 3600 + 1e-9
    term = next(e for e in _log(dm) if e["event"] == "terminate")
    assert "t_max" in term["reasons"]


def test_tmax_is_cumulative_over_instances(env):
    dm, api, clock, _ = env
    _launch(dm, "wsp-wme-27b-calib-a")
    clock.advance(2.5 * 3600)
    dm.heartbeat()
    dm.request_terminate("fake0001", "done")
    dm.tick()
    dm.tick()
    assert dm.live() == []
    clock.advance(10)
    _launch(dm, "wsp-wme-27b-calib-b")
    clock.advance(0.45 * 3600)
    dm.heartbeat()
    dm.tick()
    assert api.terminate_calls()[-1] == ["fake0002"]


def test_launch_refused_when_hours_or_price_or_capacity_fail(env):
    dm, api, clock, _ = env
    with pytest.raises(D.Refused, match="prefix"):
        dm.launch("gpu_1x_h100_sxm5", "us-southeast-1", "wsp-lab-aws", "my-box")
    api.price_cents["gpu_1x_h100_sxm5"] = 999
    with pytest.raises(D.Refused, match="price"):
        _launch(dm)
    api.price_cents["gpu_1x_h100_sxm5"] = 489
    api.capacity = {}
    with pytest.raises(D.Refused, match="capacity"):
        _launch(dm)
    assert api.launch_calls() == []
    api.capacity = {"gpu_1x_h100_sxm5": ["us-southeast-1"]}
    _launch(dm, "wsp-wme-x1")
    with pytest.raises(D.Refused, match="live"):
        _launch(dm, "wsp-wme-x2")  # one live instance at a time
    clock.advance(2.9 * 3600)
    dm.request_terminate("fake0001", "done")
    dm.tick()
    dm.tick()
    with pytest.raises(D.Refused, match="hours"):
        _launch(dm, "wsp-wme-x3")
    assert len(api.launch_calls()) == 1


def test_usd_cap_terminates(env, tmp_path):
    dm, api, clock, _ = env
    dm.base_policy.max_usd = 5.0
    _launch(dm)
    for _ in range(80):
        clock.advance(60)
        dm.heartbeat()
        dm.tick()
        if api.terminate_calls():
            break
    assert api.terminate_calls() == [["fake0001"]]
    assert dm.cost_usd() <= 5.0
    term = next(e for e in _log(dm) if e["event"] == "terminate")
    assert "usd_cap" in term["reasons"]


def test_never_terminates_unregistered_or_renamed(env):
    dm, api, clock, _ = env
    api.add("foreign1", "acudx-web-prod")
    api.add("foreign2", "wsp-wme-not-mine")
    for iid in ("foreign1", "foreign2", "nonexistent"):
        with pytest.raises(D.Refused):
            dm.request_terminate(iid, "test")
    _launch(dm)
    api.instances["fake0001"]["name"] = "renamed-by-someone"
    clock.advance(901)
    dm.tick()
    assert api.terminate_calls() == []
    assert any(e["event"] == "refused" for e in _log(dm))
    assert any(e["event"] == "alert" for e in _log(dm))


def test_launch_crash_window_adopts_exact_name_only(env):
    dm, api, clock, _ = env
    api.add("foreign2", "wsp-wme-27b-calib-t")  # prefix of the pending name, not equal
    api.fail_next_launch_after_create = True
    with pytest.raises(OSError):
        _launch(dm, "wsp-wme-27b-calib-t1")
    dm.tick()
    assert "fake0001" in dm.state()["instances"]
    assert "foreign2" not in dm.state()["instances"]
    clock.advance(901)
    dm.tick()
    assert api.terminate_calls() == [["fake0001"]]


def test_alert_when_alive_past_tmax_plus_15min(env):
    dm, api, clock, _ = env
    _launch(dm)
    api.stuck.add("fake0001")
    for _ in range((3 * 3600 + 20 * 60) // 60):
        clock.advance(60)
        dm.heartbeat()
        dm.tick()
    alerts = [e for e in _log(dm) if e["event"] == "alert" and e.get("kind") == "alive_past_t_max"]
    assert alerts
    assert dm.alert_path.exists()
    assert len(api.terminate_calls()) > 1  # keeps retrying


def test_api_errors_are_logged_without_the_key(env):
    dm, api, clock, tmp_path = env
    _launch(dm)
    api.error_message = f"bad token {KEY}"
    clock.advance(901)
    dm.tick()  # must not raise; logs the error
    api.error_message = None
    dm.tick()
    assert api.terminate_calls()
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert KEY not in p.read_text()
    assert any(e["event"] == "api_error" for e in _log(dm))


def test_policy_override_file_is_reloaded(env):
    dm, api, clock, _ = env
    _launch(dm)
    clock.advance(200)
    dm.tick()
    assert api.terminate_calls() == []
    (dm.dir / "policy.json").write_text(json.dumps({"idle_limit_s": 120}))
    dm.tick()
    assert api.terminate_calls() == [["fake0001"]]
    with pytest.raises(ValueError):
        (dm.dir / "policy.json").write_text(json.dumps({"t_max_hours": 99}))
        dm.tick()  # overrides may only tighten T_max / USD caps


def test_run_exits_on_stop_when_nothing_live(env):
    dm, _api, clock, _ = env
    _launch(dm)
    (dm.dir / "STOP").write_text("")
    dm.run(poll_s=0, sleep=lambda s: clock.advance(60), max_ticks=50)
    assert dm.live() == []  # STOP with a live instance terminates it first
    assert any(e["event"] == "exit" for e in _log(dm))


def test_client_reads_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    f = tmp_path / "env"
    f.write_text(f"HF_TOKEN=x\nLAMBDA_API_KEY={KEY}\n")
    os.chmod(f, 0o600)
    assert D.read_key(f) == KEY
    assert KEY not in repr(D.LambdaClient(KEY, transport=lambda *a: (200, {})))


def test_tick_stamps_watcher_liveness(env):
    dm, _api, clock, _ = env
    dm.tick()
    assert dm.state()["last_tick"] == clock()
    clock.advance(30)
    dm.tick()
    assert dm.state()["last_tick"] == clock()


def test_ssh_config_upsert_keeps_other_instances_aliases():
    from wsp.ops import lambda_deadman as D

    blk = lambda a, ip: f"Host {a}\n  HostName {ip}\n  User ubuntu\n"
    c = D.upsert_host_block("", "wsp-wme-a", blk("wsp-wme-a", "x"))
    c = D.upsert_host_block(c, "wsp-wme-b", blk("wsp-wme-b", "y"))
    c = D.upsert_host_block(c, "wsp-wme-a", blk("wsp-wme-a", "z"))
    assert (
        c.count("Host ") == 2 and "HostName y" in c and "HostName z" in c and "HostName x" not in c
    )
