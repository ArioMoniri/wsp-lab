"""External dead-man terminator for Lambda Cloud GPU instances (round-9 pro-engineer L4; SYNTHESIS-r9 C9).

Runs on the AWS server (never on the GPU box) and is the only holder of the Lambda API key. It:

* launches instances only with names starting ``wsp-wme-`` and registers their ids before anything else happens
  (the intended name is recorded *before* the launch request, so an instance created by a request whose response
  was lost is adopted by exact name on the next tick);
* terminates **only** registered ids whose API-reported name still equals the registered name, never by listing;
* terminates on any of: idle (no heartbeat for ``idle_limit_s``), the cumulative instance-hour cap ``t_max_hours``
  (minus ``margin_s``), the dollar cap ``max_usd`` (minus the margin's cost), an explicit request, or STOP;
* confirms termination through the API (status ``terminated`` or not-found) and stops the hour clock only then;
* alerts (log event and ``ALERT`` file) when an instance is alive more than ``alert_after_s`` past T_max, and on
  any ownership mismatch;
* logs every action as JSON lines with the key redacted; the key is never written to state, logs or repr.

Policy overrides (``<dir>/policy.json``) are reloaded every tick and may only tighten the limits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://cloud.lambda.ai/api/v1"
PREFIX = "wsp-wme-"
LAUNCH_MIN_HOURS = (
    0.25  # refuse a launch unless at least this much of T_max (after the margin) remains
)


class Refused(RuntimeError):
    """A guarded action was refused (ownership, prefix, budget, price or capacity)."""


class ApiError(RuntimeError):
    def __init__(self, status, code, message):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code = status, code


def upsert_host_block(text: str, alias: str, block: str) -> str:
    """ssh config text with ``alias``'s Host block replaced by ``block`` and every other Host block kept (several
    wsp-wme-* instances may be live at once, each with its own state directory and alias)."""
    blocks = [b for b in ("\n" + text).split("\nHost ") if b.strip()]
    keep = ["Host " + b.strip("\n") + "\n" for b in blocks if b.split("\n", 1)[0].strip() != alias]
    return "".join(keep) + block


def read_key(env_file: Path) -> str:
    """LAMBDA_API_KEY from the environment, else from a KEY=VALUE env file (never printed)."""
    key = os.environ.get("LAMBDA_API_KEY")
    if key:
        return key
    for line in Path(env_file).read_text().splitlines():
        line = line.strip()
        line = line.removeprefix("export ")
        if line.startswith("LAMBDA_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise KeyError(f"LAMBDA_API_KEY not set and not in {env_file}")


def _urllib_transport(key):
    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(API + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "wsp-wme-deadman/1")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {"error": {"code": "http", "message": str(e.code)}}

    return call


class LambdaClient:
    """Minimal Lambda Cloud API client; ``transport(method, path, body) -> (status, json)`` is injectable."""

    def __init__(self, key: str, transport=None):
        self._key = key
        self._t = transport or _urllib_transport(key)

    def __repr__(self):
        return "LambdaClient(<key redacted>)"

    def redact(self, s: str) -> str:
        return s.replace(self._key, "<redacted>") if self._key else s

    def _call(self, method, path, body=None):
        status, js = self._t(method, path, body)
        if status >= 400 or "error" in js:
            err = js.get("error", {}) if isinstance(js, dict) else {}
            raise ApiError(status, err.get("code", "?"), self.redact(str(err.get("message", ""))))
        return js["data"]

    def instance_types(self):
        return self._call("GET", "/instance-types")

    def list_instances(self):
        return self._call("GET", "/instances")

    def get_instance(self, iid):
        """The instance record, or None if the API no longer knows it."""
        try:
            return self._call("GET", f"/instances/{iid}")
        except ApiError as e:
            if e.status == 404 or "does-not-exist" in e.code or "not-found" in e.code:
                return None
            raise

    def launch(self, itype, region, ssh_key, name):
        body = {
            "region_name": region,
            "instance_type_name": itype,
            "ssh_key_names": [ssh_key],
            "quantity": 1,
            "name": name,
        }
        return self._call("POST", "/instance-operations/launch", body)["instance_ids"]

    def terminate(self, ids):
        return self._call("POST", "/instance-operations/terminate", {"instance_ids": list(ids)})


@dataclasses.dataclass
class Policy:
    idle_limit_s: float = 900.0
    t_max_hours: float = 3.0
    max_usd: float = 20.0
    alert_after_s: float = 900.0
    margin_s: float = 300.0
    max_price_usd_h: float = 5.5

    def tightened(self, over: dict) -> Policy:
        out = dataclasses.replace(self)
        for k, v in over.items():
            if not hasattr(out, k):
                raise ValueError(f"unknown policy key {k}")
            v = float(v)
            base = getattr(self, k)
            if k == "margin_s":
                if v < base:
                    raise ValueError("margin may only grow")
            elif v > base or v <= 0:
                raise ValueError(f"override {k}={v} would loosen the limit {base}")
            setattr(out, k, v)
        return out


class Deadman:
    def __init__(
        self, client: LambdaClient, directory: Path, policy: Policy | None = None, clock=time.time
    ):
        self.client = client
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.base_policy = policy or Policy()
        self.policy = dataclasses.replace(self.base_policy)
        self.clock = clock
        self.state_path = self.dir / "state.json"
        self.log_path = self.dir / "actions.jsonl"
        self.alert_path = self.dir / "ALERT"
        self.hb_path = self.dir / "heartbeat"
        self._lock_path = self.dir / ".lock"

    # ------------------------------------------------------------------ state, log, lock

    @contextlib.contextmanager
    def _locked(self):
        with open(self._lock_path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def state(self) -> dict:
        if not self.state_path.exists():
            return {"instances": {}, "pending": {}}
        return json.loads(self.state_path.read_text())

    def _save(self, st):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
        os.replace(tmp, self.state_path)

    def log(self, event, **kw):
        rec = {
            "t": self.clock(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock())),
            "event": event,
            **kw,
        }
        line = self.client.redact(json.dumps(rec, sort_keys=True, default=str))
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def alert(self, kind, **kw):
        self.log("alert", kind=kind, **kw)
        with open(self.alert_path, "a") as f:
            f.write(
                self.client.redact(json.dumps({"t": self.clock(), "kind": kind, **kw}, default=str))
                + "\n"
            )

    def heartbeat(self):
        tmp = self.hb_path.with_suffix(".tmp")
        tmp.write_text(repr(self.clock()))
        os.replace(tmp, self.hb_path)

    def _last_heartbeat(self):
        try:
            return float(self.hb_path.read_text())
        except (OSError, ValueError):
            return 0.0

    # ------------------------------------------------------------------ accounting

    def live(self, st=None):
        st = st or self.state()
        return [i for i, r in st["instances"].items() if r.get("confirmed_terminated_at") is None]

    def instance_hours(self, st=None):
        st = st or self.state()
        now = self.clock()
        return sum(
            ((r.get("confirmed_terminated_at") or now) - r["launched_at"]) / 3600
            for r in st["instances"].values()
        )

    def cost_usd(self, st=None):
        st = st or self.state()
        now = self.clock()
        return sum(
            ((r.get("confirmed_terminated_at") or now) - r["launched_at"]) / 3600 * r["price_usd_h"]
            for r in st["instances"].values()
        )

    def _reload_policy(self):
        p = self.dir / "policy.json"
        self.policy = dataclasses.replace(self.base_policy)
        if p.exists():
            self.policy = self.base_policy.tightened(json.loads(p.read_text()))

    # ------------------------------------------------------------------ guarded actions

    def launch(self, itype, region, ssh_key, name):
        if not name.startswith(PREFIX):
            raise Refused(f"name {name!r} lacks the {PREFIX} prefix")
        self._reload_policy()
        with self._locked():
            st = self.state()
            if self.live(st):
                raise Refused(f"refusing a second live instance: {self.live(st)}")
            used = self.instance_hours(st)
            room = self.policy.t_max_hours - self.policy.margin_s / 3600 - used
            if room < LAUNCH_MIN_HOURS:
                raise Refused(
                    f"instance hours exhausted: used {used:.3f} of {self.policy.t_max_hours}"
                )
            types = self.client.instance_types()
            if itype not in types:
                raise Refused(f"unknown instance type {itype}")
            price = types[itype]["instance_type"]["price_cents_per_hour"] / 100
            if price > self.policy.max_price_usd_h:
                raise Refused(f"price ${price}/h above cap ${self.policy.max_price_usd_h}/h")
            if self.cost_usd(st) + price * LAUNCH_MIN_HOURS > self.policy.max_usd:
                raise Refused("USD budget exhausted")
            regions = [r["name"] for r in types[itype]["regions_with_capacity_available"]]
            if region not in regions:
                raise Refused(f"no capacity for {itype} in {region} (available: {regions})")
            now = self.clock()
            st["pending"][name] = {
                "requested_at": now,
                "price_usd_h": price,
                "type": itype,
                "region": region,
            }
            self._save(st)
            self.log("launch_request", name=name, type=itype, region=region, price_usd_h=price)
            self.heartbeat()
            ids = self.client.launch(itype, region, ssh_key, name)
            for iid in ids:
                st["instances"][iid] = {
                    "name": name,
                    "launched_at": now,
                    "price_usd_h": price,
                    "type": itype,
                    "region": region,
                    "confirmed_terminated_at": None,
                    "terminate_requested_at": None,
                    "terminate_reasons": [],
                }
            st["pending"].pop(name, None)
            self._save(st)
        self.log("launched", ids=ids, name=name)
        return ids

    def request_terminate(self, iid, reason):
        with self._locked():
            st = self.state()
            if iid not in st["instances"]:
                self.log("refused", id=iid, why="not registered by this terminator")
                raise Refused(f"{iid} was not launched by this terminator")
            st["instances"][iid].setdefault("requests", []).append(reason)
            st["instances"][iid]["requested"] = True
            self._save(st)
        self.log("terminate_requested", id=iid, reason=reason)

    def _terminate(self, st, iid, reasons):
        rec = st["instances"][iid]
        info = self.client.get_instance(iid)
        if info is None:
            return
        if info.get("name") != rec["name"] or not str(info.get("name", "")).startswith(PREFIX):
            self.log(
                "refused",
                id=iid,
                why="API name differs from registered name",
                api_name=info.get("name"),
                registered=rec["name"],
            )
            self.alert("ownership_mismatch", id=iid)
            return
        if info.get("status") == "terminated":
            return
        self.client.terminate([iid])
        rec["terminate_requested_at"] = rec.get("terminate_requested_at") or self.clock()
        rec["terminate_reasons"] = sorted(set(rec.get("terminate_reasons", [])) | set(reasons))
        self.log(
            "terminate",
            id=iid,
            name=rec["name"],
            reasons=reasons,
            instance_hours=round(self.instance_hours(st), 4),
            cost_usd=round(self.cost_usd(st), 3),
        )

    # ------------------------------------------------------------------ loop

    def tick(self):
        self._reload_policy()
        pol = self.policy
        with self._locked():
            st = self.state()
            now = self.clock()
            # adopt instances created by a launch whose response was lost (exact name only)
            if st["pending"]:
                try:
                    for inst in self.client.list_instances():
                        p = st["pending"].get(inst.get("name"))
                        if (
                            p
                            and inst["id"] not in st["instances"]
                            and inst["name"].startswith(PREFIX)
                        ):
                            st["instances"][inst["id"]] = {
                                "name": inst["name"],
                                "launched_at": p["requested_at"],
                                "price_usd_h": p["price_usd_h"],
                                "type": p["type"],
                                "region": p["region"],
                                "confirmed_terminated_at": None,
                                "terminate_requested_at": None,
                                "terminate_reasons": [],
                            }
                            self.log("adopted", id=inst["id"], name=inst["name"])
                    if now - min(p["requested_at"] for p in st["pending"].values()) > 600:
                        st["pending"] = {}
                except (ApiError, OSError) as e:
                    self.log("api_error", where="adopt", error=self.client.redact(str(e)))
            # confirm terminations
            for iid in self.live(st):
                try:
                    info = self.client.get_instance(iid)
                except (ApiError, OSError) as e:
                    self.log("api_error", where="status", id=iid, error=self.client.redact(str(e)))
                    continue
                if info is None or info.get("status") == "terminated":
                    st["instances"][iid]["confirmed_terminated_at"] = now
                    self.log(
                        "confirmed_terminated",
                        id=iid,
                        name=st["instances"][iid]["name"],
                        status=None if info is None else info.get("status"),
                        instance_hours=round(self.instance_hours(st), 4),
                        cost_usd=round(self.cost_usd(st), 3),
                    )
            live = self.live(st)
            if live:
                last = max(
                    [self._last_heartbeat()] + [st["instances"][i]["launched_at"] for i in live]
                )
                hours = self.instance_hours(st)
                cost = self.cost_usd(st)
                price_live = sum(st["instances"][i]["price_usd_h"] for i in live)
                reasons = []
                if now - last > pol.idle_limit_s:
                    reasons.append("idle")
                if hours >= pol.t_max_hours - pol.margin_s / 3600:
                    reasons.append("t_max")
                if cost >= pol.max_usd - price_live * pol.margin_s / 3600:
                    reasons.append("usd_cap")
                if (self.dir / "STOP").exists():
                    reasons.append("stop")
                if hours > pol.t_max_hours + pol.alert_after_s / 3600:
                    self.alert("alive_past_t_max", ids=live, instance_hours=round(hours, 4))
                for iid in live:
                    r = reasons + (["requested"] if st["instances"][iid].get("requested") else [])
                    if not r:
                        continue
                    try:
                        self._terminate(st, iid, r)
                    except (ApiError, OSError) as e:
                        self.log(
                            "api_error", where="terminate", id=iid, error=self.client.redact(str(e))
                        )
            st["last_tick"] = now
            self._save(st)
            return live

    def run(self, poll_s=30.0, sleep=time.sleep, max_ticks=None):
        self.log("start", policy=dataclasses.asdict(self.base_policy), pid=os.getpid())
        n = 0
        while max_ticks is None or n < max_ticks:
            n += 1
            try:
                live = self.tick()
            except Exception as e:  # noqa: BLE001 -- the watcher must survive anything; it logs and polls on
                self.log("error", error=self.client.redact(repr(e)))
                live = self.live()
            if (self.dir / "STOP").exists() and not live and not self.live():
                self.log(
                    "exit",
                    instance_hours=round(self.instance_hours(), 4),
                    cost_usd=round(self.cost_usd(), 3),
                )
                return
            sleep(poll_s)
        self.log("max_ticks", n=n)
