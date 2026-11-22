#!/usr/bin/env python3
"""The wme-c1 9B confirmatory controller (PREREG_WME FROZEN; §II.28 serialized path). Runs on the AWS controller
host under agent-tmux; the Lambda key stays here; IPs never leave ``~/.config/wsp`` (outside the repo).

    wme_c1_ctl.py bundle  --out <tgz>                       instance bundle (code + OPEN inputs; refuses sealed paths)
    wme_c1_ctl.py up      --kind parity|main --tag <tag>    dead-man watcher, policy, capacity poll + guarded launch,
                                                            ssh, bundle push, pinned setup, worker, heartbeat
    wme_c1_ctl.py phasel-parity --iid <id>                  2-GPU check (parity, inert identity) -> evidence, terminate
    wme_c1_ctl.py phasel-main   --iid <id> --parity-iid <id> inert identity + canary references on the confirmatory
                                                            instance -> attestation -> study log -> reconcile parity
    wme_c1_ctl.py run --iid <id> --stages bank,prep,e2,sbon,designate,sealed,gen,ste
    wme_c1_ctl.py close --iid <id>                          closing canary, close, terminate, confirm, reconcile
    wme_c1_ctl.py status                                    ledger, live instances, remaining budget (no IPs)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import lambda_ops as LO
from wsp.wm.c1 import study as SD
from wsp.wm.c1 import transport as T

DATE = "2026-11-19"
RUNS = ROOT / "data/wme/runs/wme-c1"
STUDY = RUNS / "study"
EVID = ROOT / "docs/reviews/wme"
REMOTE = "/home/ubuntu/wmec1"
ALIAS = {"parity": "wmec1p", "main": "wmec1"}
TYPES = {"parity": LO.TYPE2, "main": LO.TYPE}
PANEL_ID = "wme-c1:89739c7818e29d618ac3bcb52d7e51cd29a9361e7774a71b412646775c630c1d"
READER_MODEL = "claude-opus-5-5"  # amendment G2
# RULING-wme-c1-bank-halt addendum 4 (T4): the contract transition to the deterministic field collection
TRANSITION_RULING = "docs/council/2026-11-21/RULING-wme-c1-bank-halt.md"
TRANSITION_PREP = "docs/reviews/wme/wme-c1-transition-prep-2026-11-21-l5p.json"
TRANSITION_CAUSE = "driver_or_cuda"  # a non-deterministic CUDA backward kernel in the registered-mode collection
TRANSITION_RESOLVES = ("field_repair_mismatch",)  # incident seq 277 (run5 K3 halt)
# T2 (RULING-wme-c1-bank-halt addendum 3): the deterministic mmlupro_gold_lp.g0.pt, 3/3 on one instance
T2_MMLUPRO_DET = "af39a61786527eba9c5e497106e514269e697deee8259cb4011aa2bb5bac62c4"
BUNDLE_FILES = [
    "scripts/wme_c1_remote_setup.sh",
    "scripts/wme_c1_remote_worker.sh",
    "scripts/wme_c1_remote_lock.txt",
    "scripts/wme_pilot.py",
    "scripts/wme_stagew.py",
    "scripts/wme_stagew2.py",
    "docs/prereg/wme_manifest.json",
    "docs/reviews/wme/wme-c1-shards-2026-11-19.json",
    "data/wme/panels/wme-c1/manifest.json",
    "data/wme/panels/wme-c1/open/bank.jsonl",
    "data/wme/panels/wme-c1/open/dev.jsonl",
    "data/wme/panels/wme-c1/open/construction.jsonl",
    "data/wme/panels/wme-p2/open/construction.jsonl",
    "data/wme/panels/wme-p3h/manifest.json",
    "data/wme/panels/wme-p3h/open/holdout.jsonl",
    "data/wme/panels/wme-p3h/open/construction.jsonl",
    "data/wme/panels/wme-c2-gen/manifest.json",
    "data/wme/panels/wme-c2-gen/open/calibration.jsonl",
    "data/wme/bank_roots/roots-9b-20261117/roots.json",
    "data/wme/bank_roots/roots-9b-20261117/directions.pt",
    "data/wme/pilot3/run-20261117/h/teacher.jsonl",
    "data/wme/stagew/teacher.jsonl",
]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha(p) -> str:
    return C.P.file_sha256(p)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


# ------------------------------------------------------------------ bundle


def cmd_bundle(a):
    """The instance bundle from the committed tree (``git archive HEAD``; §II.31.9 P2, Board A N7)."""
    from wsp.wm.c1 import bundling as BD

    rec = BD.build(a.out, root=ROOT, files=BUNDLE_FILES)
    print(json.dumps({k: rec[k] for k in ("bundle", "sha256", "code_sha")}))


# ------------------------------------------------------------------ instance up


def cap_state() -> tuple[float, float, int]:
    """(cap in force, remaining GPU-h-eq, cap-generation index) from the study log (PREREG_WME §II.31). Before the
    first guard writes the log, the §II.31 cap (``study.CAP``) is the cap and nothing is charged yet."""
    dm_state = LO.STATE / "state.json"
    if not (STUDY / RG.LOG_NAME).exists():
        # earlier study instances (Phase L checks) enter the ledger at phasel-main; until then they are charged here
        spent = LO.deadman_gpu_h(json.loads(dm_state.read_text())) if dm_state.exists() else 0.0
        return SD.CAP.cap_gpu_h, max(0.0, SD.CAP.cap_gpu_h - spent), 0
    led = RG.CombinedLedger(STUDY / RG.LOG_NAME, {"bank": 1})
    caps = led.cap_generations()
    if not caps or caps[-1]["cap_gpu_h"] < SD.CAP.cap_gpu_h:
        return SD.CAP.cap_gpu_h, max(0.0, SD.CAP.cap_gpu_h - led.gpu_h()), 0
    return led.cap_gpu_h(), led.remaining_gpu_h(), int(caps[-1]["index"])


def remaining_gpu_h() -> float:
    return cap_state()[1]


def ensure_watcher():
    out = subprocess.run(["agent-tmux", "-l"], capture_output=True, text=True).stdout
    if "wme-c1-dm" in out:
        return
    py = sys.executable
    subprocess.run(
        [
            "agent-tmux",
            "wme-c1-dm",
            py,
            str(ROOT / "scripts/lambda_deadman.py"),
            "--dir",
            str(LO.STATE),
            "--idle-limit-s",
            "1800",
            "--t-max-hours",
            str(cap_state()[0]),  # base: the cap in force; per-launch overrides tighten it
            "--max-usd",
            "1000",
            "--max-price",
            "9.6",
            "watch",
        ],
        check=True,
    )


def cmd_up(a):
    from wsp.wm.c1 import phasel as PL

    # §II.31.9 P2 / N4 / N6, before any spend: a clean tree, a bundle built from this commit, the registered gate
    commit = PL.git("rev-parse", "HEAD")
    if PL.git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("REFUSED: the controller tree has uncommitted changes (P2)")
    with tarfile.open(a.bundle) as tf_:
        bundle_code = tf_.extractfile("wmec1/CODE_SHA").read().decode().strip()
    if bundle_code != commit:
        raise SystemExit(f"REFUSED: the bundle was built from {bundle_code[:12]}, not HEAD {commit[:12]} (P2)")
    gate = PL.run_gate()
    log("registered test gate", json.dumps({k: gate[k] for k in ("exit_code", "counts", "pass")}))
    if not gate["pass"]:
        raise SystemExit(f"REFUSED: the registered test gate failed {gate['counts']} (N6)")
    LO.STATE.mkdir(parents=True, exist_ok=True)
    ensure_watcher()
    pf = preflight_checks()  # addendum 5: before any poll
    log("preflight", json.dumps(pf, default=str))
    if not pf["pass"]:
        raise SystemExit("REFUSED: preflight failed (addendum 5)")
    dm = LO.deadman()
    spent = dm.cost_usd(dm.state())
    cap, rem, cap_gen = cap_state()
    itype = TYPES[a.kind]
    max_usd = LO.allowance(rem, spent, itype)
    lim = rem / LO.GPUS[itype] if a.t_max_hours is None else min(rem / LO.GPUS[itype], a.t_max_hours)
    t_max = dm.instance_hours(dm.state()) + lim
    pol = LO.tighten(LO.STATE, max_usd, t_max, cap_generation=cap_gen)
    log(f"policy set (cap {cap} GPU-h-eq, generation {cap_gen}, remaining {rem:.2f}): {pol}")
    if dm.live(dm.state()):
        raise SystemExit("a wsp-wme-c1 instance is already live; one instance at a time")
    name = f"{LO.NAME_PREFIX}{a.kind}-{DATE.replace('-', '')}{a.tag}"
    py = sys.executable
    cmd = [
        py,
        str(ROOT / "scripts/wme_lambda_poll.py"),
        "--dir",
        str(LO.STATE),
        "--name",
        name,
        "--types",
        itype,
        "--fallback-after-min",
        "1e9",
        "--t-max-hours",
        str(pol["t_max_hours"]),
        "--max-usd",
        str(pol["max_usd"]),
        "--max-price",
        "9.6",
        "--idle-limit-s",
        "1800",
        "--max-min",
        str(a.poll_max_min),
    ]
    log("polling for capacity:", name, itype)
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise SystemExit("no capacity within the polling window")
    iid = LO.wait_live(dm)
    log("live instance", iid)
    alias = ALIAS[a.kind]
    LO.wait_active_ssh(dm, iid, alias)
    log("ssh up")
    bundle = Path(a.bundle)
    subprocess.run(
        [
            "scp",
            "-F",
            str(LO.SSH_CONFIG),
            "-o",
            "BatchMode=yes",
            "-q",
            str(bundle),
            f"{alias}:wmec1-bundle.tgz",
        ],
        check=True,
    )
    LO.ssh(
        alias,
        f"tar -xzf wmec1-bundle.tgz && cd {REMOTE} && mkdir -p logs w && (nohup bash scripts/wme_c1_remote_setup.sh $(cat CODE_SHA) > logs/setup.log 2>&1 < /dev/null &) && sleep 1 && cat SETUP_STATUS",
    )
    hb = [
        "agent-tmux",
        f"wme-c1-hb-{a.kind}",
        "bash",
        str(ROOT / "scripts/wme_c1_hb.sh"),
        alias,
        str(LO.STATE),
        f"{REMOTE}/w",
    ]
    subprocess.run(hb, check=True)
    for _ in range(240):
        st = LO.ssh(alias, f"cat {REMOTE}/SETUP_STATUS", check=False).strip()
        if st == "SETUP_OK":
            break
        if st.startswith("SETUP_FAILED"):
            raise SystemExit(f"remote setup failed: {st}")
        time.sleep(15)
    else:
        raise SystemExit("remote setup did not finish")
    log("setup ok:", LO.ssh(alias, f"cat {REMOTE}/w/versions.txt {REMOTE}/w/lock_check.txt"))
    LO.ssh(
        alias,
        f"cd {REMOTE} && (nohup bash scripts/wme_c1_remote_worker.sh > /dev/null 2>&1 < /dev/null &) && sleep 2 && echo started",
    )
    rec = {
        "iid": iid,
        "name": name,
        "type": itype,
        "kind": a.kind,
        "alias": alias,
        "bundle_sha256": sha(bundle),
        "controller_commit": commit,
        "gate": gate,
    }
    (RUNS / "instances").mkdir(parents=True, exist_ok=True)
    (RUNS / "instances" / f"{iid}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


# ------------------------------------------------------------------ addendum 5: heartbeats, preflight


def watcher_running() -> bool:
    return "wme-c1-dm" in subprocess.run(["agent-tmux", "-l"], capture_output=True, text=True).stdout


def tick_health(state_dir, watcher_running: bool, need: int = 2, wait_s: float = 180.0, sleep=time.sleep,
                poll_s: float = 5.0) -> dict:  # fmt: skip
    """The dead-man watcher is alive and clean: its ``last_tick`` advances ``need`` times within ``wait_s`` and no
    ``error`` action follows its latest ``start``."""
    state_dir = Path(state_dir)
    read = lambda: float(json.loads((state_dir / "state.json").read_text()).get("last_tick", 0.0))  # noqa: E731
    last, ticks, waited = read(), 0, 0.0
    while watcher_running and ticks < need and waited < wait_s:
        sleep(poll_s)
        waited += poll_s
        t = read()
        if t > last:
            ticks, last = ticks + 1, t
    acts = [json.loads(x) for x in (state_dir / "actions.jsonl").read_text().splitlines()] if (
        state_dir / "actions.jsonl").exists() else []  # fmt: skip
    starts = [i for i, e in enumerate(acts) if e.get("event") == "start"]
    since = acts[starts[-1] + 1 :] if starts else acts
    errors = sum(e.get("event") == "error" for e in since)
    return {
        "watcher_running": watcher_running,
        "ticks": ticks,
        "errors_since_start": errors,
        "pass": bool(watcher_running and ticks >= need and errors == 0),
    }


def preflight_checks(wait_s: float = 180.0) -> dict:
    """RULING-wme-c1-bank-halt addendum 5, before any poll: (1) the worktree's ``.venv`` link and interpreter (the
    heartbeat job runs ``.venv/bin/python``); (2) the heartbeat script runs once successfully against a dry target
    (an empty dead-man state: no live instance, the loop ends cleanly); (3) the dead-man watcher's tick health."""
    import tempfile

    py = ROOT / ".venv" / "bin" / "python"
    try:
        ok = py.exists() and subprocess.run([str(py), "-c", "import sys"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    venv = {"path": str(ROOT / ".venv"), "interpreter": str(py), "pass": bool(ok)}
    with tempfile.TemporaryDirectory() as td:
        env = {k: v for k, v in os.environ.items() if k != "WME_HB_PYTHON"}
        try:
            p = subprocess.run(
                ["bash", str(ROOT / "scripts/wme_c1_hb.sh"), "dry", str(Path(td) / "dm"), "/nonexistent"],
                capture_output=True, text=True, timeout=120, env=env,
            )  # fmt: skip
            hb = {"rc": p.returncode, "out": (p.stdout + p.stderr)[-500:],
                  "pass": p.returncode == 0 and "no live instance" in p.stdout}  # fmt: skip
        except (OSError, subprocess.SubprocessError) as exc:
            hb = {"rc": None, "out": str(exc)[:300], "pass": False}
    ticks = tick_health(LO.STATE, watcher_running(), wait_s=wait_s)
    return {
        "venv": venv,
        "heartbeat_dry_run": hb,
        "deadman_ticks": ticks,
        "pass": venv["pass"] and hb["pass"] and ticks["pass"],
    }


def cmd_preflight(a):
    rep = preflight_checks()
    log("preflight", json.dumps(rep, default=str))
    print(json.dumps(rep, default=str))
    if not rep["pass"]:
        raise SystemExit("PREFLIGHT FAILED: " + ", ".join(k for k in ("venv", "heartbeat_dry_run", "deadman_ticks")
                                                          if not rep[k]["pass"]))  # fmt: skip
    return rep


def require_heartbeats(iid: str, alias: str, need: int = 2, wait_s: float = 900.0, poll_s: float = 15.0) -> int:
    """Addendum 5: no request reaches a live instance until at least ``need`` progress heartbeats from its worker
    (``heartbeats.jsonl``, written by ``wme_c1_hb.sh`` for ``alias``) have landed after the instance's launch. Waits up
    to ``wait_s``, then refuses."""
    t0 = time.time()
    while True:
        st = json.loads((LO.STATE / "state.json").read_text())
        launched = float((st.get("instances", {}).get(iid) or {}).get("launched_at") or 0.0)
        p = LO.STATE / "heartbeats.jsonl"
        n = 0
        if p.exists():
            for ln in p.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                n += r.get("alias") == alias and r.get("source") == "progress" and float(r.get("t", 0)) > launched
        if n >= need:
            return n
        if time.time() - t0 >= wait_s:
            raise SystemExit(f"REFUSED: {n} progress heartbeat(s) from {alias} since the launch; {need} needed "
                             "before any request (addendum 5)")  # fmt: skip
        time.sleep(poll_s)


def remote_for(kind: str, iid: str):
    alias = ALIAS[kind]
    require_heartbeats(iid, alias)  # addendum 5: the heartbeat job must be landing before any request
    tr = T.SSHTransport(alias, LO.SSH_CONFIG, f"{REMOTE}/w")
    dm = LO.deadman()

    def lost():
        st = dm.state()
        return st["instances"].get(iid, {}).get("confirmed_terminated_at") is not None

    def revive():
        LO.ssh(
            alias,
            f"cd {REMOTE} && (pgrep -f 'wsp.wm.c1.serve --root w' >/dev/null || (nohup bash scripts/wme_c1_remote_worker.sh > /dev/null 2>&1 < /dev/null &)); echo ok",
        )

    def restart():
        # §II.31.8 R6: a withdrawn request that does not stop -> kill the worker and its children, start it again
        # (bracketed patterns so pgrep never matches this command line itself)
        LO.ssh(
            alias,
            f"cd {REMOTE} && (for p in 'wsp.wm.c1.ser[v]e --root w' 'wsp.wm.c1.vllmop[s]' 'wsp.wm.c1.launchop[s] gpu1'; "
            "do pgrep -f \"$p\" | xargs -r kill -9; done; sleep 3; "
            "nohup bash scripts/wme_c1_remote_worker.sh > /dev/null 2>&1 < /dev/null &); echo ok",
        )

    return tr, C.Remote(
        tr,
        RUNS / "ctl",
        poll_s=10,
        dead_after_s=1800,
        transport_retry_s=900,
        lost=lost,
        log=log,
        revive=revive,
        restart=restart,
    )


# ------------------------------------------------------------------ Phase L


def utc_date() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def write_evidence(name, obj) -> tuple[str, str]:
    """Run-tagged Phase L evidence (§II.31.8 R2): never overwrites a file (``phasel.write_new``)."""
    from wsp.wm.c1 import phasel as PL

    p = EVID / name
    h = PL.write_new(p, obj)
    try:
        return str(p.relative_to(ROOT)), h
    except ValueError:  # an evidence directory outside the repository (tests)
        return str(p), h


def commit_evidence(paths: list, message: str):
    """git add + commit of run evidence (the commit hooks apply)."""
    subprocess.run(["git", "add", *paths], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=ROOT, check=True)


def study_exists() -> bool:
    return (RUNS / "attestation.json").exists() or (STUDY / RG.LOG_NAME).exists()


def backup_attestation() -> Path | None:
    """Board A B1: a copy of attestation.json before any step that could write; never overwritten."""
    src = RUNS / "attestation.json"
    if not src.exists():
        return None
    dst = RUNS / f"attestation.json.bak-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def write_attestation_once(att) -> None:
    """The first launch only: attestation.json is written once and never overwritten (Board A B1)."""
    p = RUNS / "attestation.json"
    if p.exists():
        raise SystemExit("REFUSED: attestation.json exists; a launch never rewrites it (use phasel-main --resume)")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(RG._att_fields(att), indent=1))
    tmp.replace(p)


def instance_record(iid: str) -> dict:
    p = RUNS / "instances" / f"{iid}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def driver_record(hello: dict, parity: dict | None = None) -> dict:
    """The instance's NVIDIA driver (``hello.physical.driver``) beside the parity instance's (chair note after run4:
    recorded in the launch and resume evidence). Informational: a different driver alone is not a stop; the binding
    test is the canary comparison against the committed references."""
    mine = (hello.get("physical") or {}).get("driver")
    par = None if parity is None else ((parity.get("hello") or {}).get("physical") or {}).get("driver")
    return {
        "version": mine,
        "gpu_names": (hello.get("physical") or {}).get("gpu_names"),
        "parity_instance": par,
        "same_as_parity": None if par is None or mine is None else mine == par,
        "binding_test": "the canary comparison with the committed references (a driver difference alone is not a stop)",
    }


def code_and_state(hello: dict, iid: str, parity: dict | None = None) -> dict:
    """§II.31.9 N4 (b): controller commit (clean tree) = the worker bundle's CODE_SHA, the bundle sha256, the worker's
    process globals at start (R11) and the N6 gate result recorded at ``up``; the NVIDIA driver (informational)."""
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import phasel as PL

    rec = instance_record(iid)
    return {
        "code_identity": PL.code_identity(hello.get("code_sha"), rec.get("bundle_sha256")),
        "worker_global_state": PL.worker_state_check(hello.get("global_state"), GE.GEN_ENGINE),
        "gate": rec.get("gate"),
        "driver": driver_record(hello, parity),
    }


def require_code_and_state(cs: dict):
    if not cs["code_identity"]["pass"]:
        raise SystemExit(f"PRECONDITION FAILED: code identity {cs['code_identity']}")
    if not cs["worker_global_state"]["pass"]:
        raise SystemExit(f"PRECONDITION FAILED: worker globals {cs['worker_global_state']}")
    if not (cs["gate"] or {}).get("pass"):
        raise SystemExit(f"PRECONDITION FAILED: registered test gate {cs['gate']}")


def ec_sha(field: bool = True) -> str:
    """sha256 of :func:`engine_config`; ``field=False``: the pre-transition configuration (generation 0, before the
    T4 field entry of RULING-wme-c1-bank-halt addendum 4)."""
    return hashlib.sha256(json.dumps(engine_config(field=field), sort_keys=True).encode()).hexdigest()


def cmd_phasel_parity(a):
    """2-GPU check instance: hello, parity, inert identity and (C1(a)) gen_repeat; the §II.31.1 pass rule and the
    §II.31.2 selection evaluated by ``phasel.evaluate`` and recorded (§II.31.8 R1); the instance is terminated in all
    cases; a run that does not proceed exits non-zero after termination."""
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import phasel as PL

    tr, rem = remote_for("parity", a.iid)
    inst = LO.guard_instance(a.iid)
    ec, ecs = engine_config(), ec_sha()
    n = {"i": 0}

    def call(kind, args):
        n["i"] += 1
        uk = hashlib.sha256(f"phasel-parity:{kind}:{n['i']}".encode()).hexdigest()
        p, _ = rem.call(kind, 0, inst, uk, kind, args)
        log(kind, json.dumps({k: p[k] for k in p if k not in ("chunks_k1", "chunks_k2", "rows")})[:2000])
        return p

    try:
        hello = call("hello", {})
        if hello.get("gen_engine") != GE.config():
            raise SystemExit("PRECONDITION FAILED: the worker's generation engine is not the recorded one")
        cs = code_and_state(hello, a.iid)
        require_code_and_state(cs)
        checks = PL.run_checks(call, GE.GEN_ENGINE)
        dm = LO.deadman()
        s_l = LO.deadman_gpu_h(dm.state())  # all Phase L check instances so far, this one to now
        cap = cap_state()[0]
        g_e1 = GE.e1_gen_gpu_h(json.loads((ROOT / "docs/prereg/wme_manifest.json").read_text()))
        committed_hf = None
        if (RUNS / "attestation.json").exists():  # C2: a re-run must reproduce the committed hf_gen reference
            refs = PL.committed_references(json.loads((RUNS / "attestation.json").read_text()))
            committed_hf = refs["engines"].get("hf_gen")
            if GE.GEN_ENGINE == "hf_deterministic" and committed_hf != PL.COMMITTED_HF:
                raise SystemExit(f"the committed hf_gen reference {committed_hf} is not {PL.COMMITTED_HF}")
        ev = PL.evaluate(checks, GE.GEN_ENGINE, s_l, cap, g_e1, committed_hf=committed_hf)
        path, h = write_evidence(
            f"wme-c1-launch-parity-{utc_date()}-{a.tag}.json",
            {
                "label": "wme-c1 Phase L re-run: preconditions 1-2 and the §II.31 pass rule / selection on a 2-GPU "
                "check instance; exposed engineering items only",
                "instance": inst,
                "engine_config_sha256": ecs,
                "engine_config": ec,
                "hello": hello,
                **cs,
                **checks,
                "evaluation": ev,
            },
        )
        log("evidence", path, h, "proceed", ev["proceed"], "selected", ev["selected_engine"])
    finally:
        t = LO.terminate_and_confirm(LO.deadman(), a.iid, "wme-c1 parity check finished")
        log("termination confirmed", t)
    print(json.dumps({"evidence": path, "sha256": h, "evaluation": ev}, default=str))
    if not ev["proceed"]:
        raise SystemExit(
            f"PHASE L STOP: selected engine {ev['selected_engine']} (recorded {GE.GEN_ENGINE}); pass rule "
            f"{ev['pass_rule']}; other {ev['other_preconditions']}. Report to the chair before anything else."
        )


def engine_config(field: bool = True) -> dict:
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import modelops as MO
    from wsp.wm.c1 import vllmops as VO

    man = MO.manifest()
    ec = {
        "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "theta0": man["target.theta0_hash"],
        "lock_sha256": sha(ROOT / "scripts/wme_c1_remote_lock.txt"),
        "gpu": "NVIDIA H100 80GB HBM3",
        "teacher_forced": {
            "panel_config": "wm.panel.PanelConfig() defaults (window 256, max_batch_tokens 16384)",
            "dtype": "bf16",
        },
        "hf_generate": {
            "batch": 128,
            "padding": "left",
            "greedy": True,
            "non_thinking": True,
            "max_new_tokens": {"sycophancy": 128},
            "canary": MO.CANARY_HF,
            "engine": GE.config(),  # PREREG_WME §II.31: the recorded generation-engine flag and its settings
        },
        "vllm": VO.engine_config(),
        "canary": {
            "tf_items": "wme-p2 construction, 16 per objective by id, 7 objectives",
            "gen_items": "wme-p3h holdout mmlupro_gen first 64",
            "vllm_tokens": VO.CANARY_TOKENS,
        },
    }
    if field:  # RULING-wme-c1-bank-halt addendum 4, T4 item 1: the deterministic field collection
        ec["field"] = GE.field_config()
    return ec


def cmd_phasel_main(a):
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import phasel as PL

    if getattr(a, "resume", False):
        return cmd_phasel_resume(a)
    if study_exists():  # Board A B1: a first launch never touches an existing study (before any write)
        raise SystemExit(
            "REFUSED: a wme-c1 study already exists (attestation / study log); use phasel-main --resume"
        )
    ecs = ec_sha()
    par_path = str(Path(a.parity_evidence))
    par = json.loads((ROOT / par_path).read_text())
    PL.check_binding(par, ecs, "parity evidence")  # R2: produced under the current engine config
    head = PL.git("rev-parse", "HEAD")
    if (par.get("code_identity") or {}).get("controller_commit") != head:  # P2: the same commit's evidence only
        raise SystemExit("PRECONDITION FAILED: the parity evidence was not produced by this commit")
    ev = par["evaluation"]
    if not ev["proceed"] or ev["selected_engine"] != GE.GEN_ENGINE:
        raise SystemExit(f"PRECONDITION FAILED: parity evaluation {ev['selected_engine']} / {ev['proceed']}")
    parsers = PL.parser_check()  # R9
    if not parsers["pass"]:
        raise SystemExit(f"PRECONDITION FAILED: gen.parsers check {parsers}")
    tr, rem = remote_for("main", a.iid)
    inst = LO.guard_instance(a.iid)
    hello, _ = rem.call("hello", 0, inst, "m" * 64, "hello", {})
    if hello.get("gen_engine") != GE.config():
        raise SystemExit("PRECONDITION FAILED: the worker's generation engine is not the recorded one")
    cs = code_and_state(hello, a.iid, par)
    require_code_and_state(cs)
    log("hello", json.dumps({k: hello[k] for k in ("theta0", "versions", "physical")}))
    inert, _ = rem.call("inert_identity", 0, inst, "n" * 64, "inert", {})
    log("inert identity pass:", inert["pass"])
    if not inert["pass"]:
        raise SystemExit("PRECONDITION FAILED: pinned-engine inert identity")
    can, _ = rem.call("canary", 0, inst, "o" * 64, "canary-reference", {})
    eng = can["engines"]
    if set(eng) != GE.canary_engines():  # R2: the canary set of the recorded engine
        raise SystemExit(f"PRECONDITION FAILED: canary engines {sorted(eng)}")
    cross = PL.cross_instance_canaries(eng, par)  # N3: the main instance reproduces the parity instance
    log("cross-instance canaries", json.dumps(cross))
    if not cross["pass"]:
        raise SystemExit(f"PRECONDITION FAILED: main vs parity canaries {cross['pairs']}")
    tim, _ = rem.call("timing", 0, inst, "t" * 64, "timing", {})
    p_bank = (
        SD.PRICE["bank"]
        + (192 * tim["eigh_12288_s"] + 576 * tim["eigh_4096_s"] + 13095 * 6 * tim["contraction_s"])
        / 3600
    )
    tim_path, tim_sha = write_evidence(
        f"wme-c1-launch-timing-{utc_date()}-{a.tag}.json",
        {
            "engine_config_sha256": ecs,
            "label": "PREREG_WME §II.30 re-pricing inputs (synthetic matrices; outcome-blind)",
            "timing": tim,
            "rule": "P_bank = 4.1268 + (192 t_eigh(12288) + 576 t_eigh(4096) + 13095 x 6 t_contract) / 3600",
            "P_bank_gpu_h": p_bank,
        },
    )
    (RUNS / "bank_price.json").write_text(
        json.dumps({"P_bank": p_bank, "evidence": tim_path, "sha256": tim_sha})
    )
    log(f"bank phase re-priced (§II.30): {p_bank:.3f} GPU-h-eq")
    ref = C.composite_canary(eng)
    ec = engine_config()
    if hashlib.sha256(json.dumps(ec, sort_keys=True).encode()).hexdigest() != ecs:
        raise SystemExit("engine config changed during phasel-main")
    ec_path, ec_file_sha = write_evidence(f"wme-c1-engine-config-{utc_date()}-{a.tag}.json", ec)
    cs_path, cs_sha = write_evidence(
        f"wme-c1-launch-code-{utc_date()}-{a.tag}.json",
        {"engine_config_sha256": ecs, "instance": inst, **cs, "cross_instance_canaries": cross},
    )
    pz_path, pz_sha = write_evidence(
        f"wme-c1-launch-parsers-{utc_date()}-{a.tag}.json",
        {"engine_config_sha256": ecs, "gen_parsers_check": parsers},
    )
    can_path, can_sha = write_evidence(
        f"wme-c1-canary-refs-{utc_date()}-{a.tag}.json",
        {
            "engine_config_sha256": ecs,
            "label": "per-engine canary references: the first confirmatory instance's start canaries (before any outcome)",
            "instance": inst,
            "engines": eng,
            "composite": ref,
            "detail": can.get("detail"),
        },
    )
    inert_path, inert_sha = write_evidence(
        f"wme-c1-launch-inert-{utc_date()}-{a.tag}.json",
        {"engine_config_sha256": ecs, "instance": inst, "hello": hello, "inert_identity": inert},
    )
    evidence = sorted(
        [
            f"{par_path} sha256:{sha(ROOT / par_path)}",
            f"{inert_path} sha256:{inert_sha}",
            f"{can_path} sha256:{can_sha}",
            f"{ec_path} sha256:{ec_file_sha}",
            f"{tim_path} sha256:{tim_sha}",
            f"{pz_path} sha256:{pz_sha}",
            f"{cs_path} sha256:{cs_sha}",
        ]
    )
    commit_evidence(
        [par_path, inert_path, can_path, ec_path, tim_path, pz_path, cs_path],
        "wme-c1 launch: committed per-engine canary references, pinned-engine inert identity, parity evidence and engine config (before any confirmatory unit)",
    )
    att = RG.LaunchAttestation(
        inert_identity_bitwise=True,
        gpu_parity_bitwise=True,
        canary_reference=ref,
        engine_config_sha256=hashlib.sha256(json.dumps(ec, sort_keys=True).encode()).hexdigest(),
        evidence=evidence,
    )
    write_attestation_once(att)
    st = SD.Study(STUDY, RUNS, att, hello["theta0"], PANEL_ID, log=log)
    # bind every phase's registered fallback in the study log before the first unit (precondition 4)
    for ph in ("ste", "sbon", "confirmatory"):
        st.guard(ph, inst)
    g = st.guard("bank", inst)
    C.take_over(tr, rem, g)
    # every earlier study instance (the parity check, failed boots) enters the combined charge (precondition 7)
    dms = json.loads((LO.STATE / "state.json").read_text())
    for oid, rec in sorted(dms["instances"].items()):
        if oid == a.iid or rec.get("confirmed_terminated_at") is None:
            continue
        gp = LO.GPUS.get(rec.get("type"), 1)
        la, te = LO.reconcile(g, LO.STATE, oid, gp)
        log(f"earlier instance {rec.get('name')} reconciled: {(te - la) / 3600 * gp:.3f} GPU-h-eq")
    g.open_instance(1)
    rec = C.run_canary(g, rem)
    log("start canary", rec["pass"])
    if not rec["pass"]:
        raise SystemExit("PRECONDITION FAILED: start canary != reference")
    print(
        json.dumps(
            {
                "attestation_contract": RG.contract_digest(att),
                "canary_reference": ref,
                "epoch": g.epoch,
                "report": g.report(),
            },
            default=str,
        )
    )


def cmd_phasel_resume(a):
    """Board A B1 (v2r check): resume the EXISTING study on a new main instance under the persisted contract.

    Never writes attestation.json (it is backed up first and its bytes are asserted unchanged at the end), records
    no contract transition, mints no new canary reference. Order:
    1. offline: the attestation's contract digest equals the study log's; the engine config is unchanged; the parity
       evidence is bound to this engine config and this commit (P2) and passed, including C2; the committed
       per-engine references are read from the attestation's own canary-refs evidence (sha256 verified);
    2. a new owner (guard) on the persisted contract, then on the new instance: hello (code identity, worker
       globals), inert identity, the canaries: every engine equal to the committed reference AND the parity GPU 0,
       hf_gen == 7299ac4f..., the composite == the persisted reference (C2; otherwise a canary_mismatch incident);
    3. evidence files (run-tagged) and their commit, one registered ``repair`` record naming them (cause
       infrastructure: the halted instance was replaced), reconciliation of terminated instances not yet reconciled;
    4. the new instance opened and the start canary passed against the persisted reference.
    Every eligible unit stays eligible (same contract and generation)."""
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import phasel as PL

    att_p = RUNS / "attestation.json"
    if not att_p.exists() or not (STUDY / RG.LOG_NAME).exists():
        raise SystemExit("REFUSED: --resume needs an existing study (attestation.json and the study log)")
    att_bytes = att_p.read_bytes()
    att = load_att()  # the current contract (generation 0 before the T4 transition)
    att_fields = RG._att_fields(att)
    contract = RG.contract_digest(att)
    state = RG.read_study(STUDY)
    if state.contract != contract:
        raise SystemExit(f"REFUSED: attestation contract {contract[:12]} != study log {str(state.contract)[:12]}")
    if state.contracts[0]["digest"] != RG.contract_digest(RG.LaunchAttestation(**json.loads(att_bytes))):
        raise SystemExit("REFUSED: attestation.json is not the study's generation-0 attestation")
    ecs = ec_sha()
    pre_transition = contract_state() == "pre_transition"  # T4: resume under generation 0, then `transition`
    if pre_transition:
        log("the current contract predates the T4 field entry: the contract transition must follow this resume")
    par_path = str(Path(a.parity_evidence))
    par = json.loads((ROOT / par_path).read_text())
    PL.check_binding(par, ecs, "parity evidence")
    head = PL.git("rev-parse", "HEAD")
    if (par.get("code_identity") or {}).get("controller_commit") != head:  # P2
        raise SystemExit("PRECONDITION FAILED: the parity evidence was not produced by this commit")
    ev = par["evaluation"]
    if not ev["proceed"] or ev["selected_engine"] != GE.GEN_ENGINE:
        raise SystemExit(f"PRECONDITION FAILED: parity evaluation {ev['selected_engine']} / {ev['proceed']}")
    if GE.GEN_ENGINE == "hf_deterministic" and ev["pass_rule"].get("hf_gen_eq_committed") is not True:
        raise SystemExit("PRECONDITION FAILED: the parity run did not reproduce the committed hf_gen reference (C2)")
    refs = PL.committed_references(att_fields)
    parsers = PL.parser_check()  # R9
    if not parsers["pass"]:
        raise SystemExit(f"PRECONDITION FAILED: gen.parsers check {parsers}")
    bak = backup_attestation()
    tr, rem = remote_for("main", a.iid)
    inst = LO.guard_instance(a.iid)
    st = SD.Study(STUDY, RUNS, att, MO_theta0(), PANEL_ID, log=log)
    g = st.guard("bank", inst)
    C.take_over(tr, rem, g)
    # every request after the fence carries the owner epoch: the worker rejects older ones as stale (run4 halt)
    hello, _ = rem.call("hello", g.epoch, inst, "m" * 64, "hello", {})
    if hello.get("gen_engine") != GE.config():
        raise SystemExit("PRECONDITION FAILED: the worker's generation engine is not the recorded one")
    cs = code_and_state(hello, a.iid, par)
    require_code_and_state(cs)
    inert, _ = rem.call("inert_identity", g.epoch, inst, "n" * 64, "inert", {})
    log("inert identity pass:", inert["pass"])
    if not inert["pass"]:
        raise SystemExit("PRECONDITION FAILED: pinned-engine inert identity")
    can, _ = rem.call("canary", g.epoch, inst, "o" * 64, "canary-resume", {})
    eng = can["engines"]
    if set(eng) != GE.canary_engines():
        raise SystemExit(f"PRECONDITION FAILED: canary engines {sorted(eng)}")
    rc = PL.resume_canaries(eng, par, refs["engines"], att.canary_reference)
    log("resume canaries", json.dumps(rc))
    if not rc["pass"]:
        g.incident("canary_mismatch", {"where": "phasel-main --resume", **rc})
        raise SystemExit(f"INCIDENT: resume canaries differ from the committed references {rc['pairs']}")
    tim, _ = rem.call("timing", g.epoch, inst, "t" * 64, "timing", {})
    tag = f"{utc_date()}-{a.tag}"
    common = {
        "engine_config_sha256": ecs,
        "instance": inst,
        "contract": contract,
        "resume": True,
        "pre_transition": pre_transition,
    }
    paths = []
    for name, obj in (
        (f"wme-c1-resume-inert-{tag}.json", {**common, "hello": hello, "inert_identity": inert}),
        (f"wme-c1-resume-canaries-{tag}.json", {**common, "engines": eng, "check": rc, "detail": can.get("detail")}),
        (f"wme-c1-resume-code-{tag}.json", {**common, **cs, "attestation_backup": str(bak.name) if bak else None}),
        (f"wme-c1-resume-timing-{tag}.json", {**common, "timing": tim, "note": "recorded only; no re-pricing"}),
        (f"wme-c1-resume-parsers-{tag}.json", {**common, "gen_parsers_check": parsers}),
    ):
        pth, h = write_evidence(name, obj)
        paths.append((pth, h))
    commit_evidence(
        [par_path] + [p for p, _ in paths],
        f"wme-c1 resume on {inst}: evidence under the persisted contract {contract[:12]} (no new attestation)",
    )
    g.retry.repair(
        "infrastructure",
        f"resume on {inst} after the bank halt under the persisted contract {contract} (no transition, no new "
        f"reference): parity {par_path}; evidence "
        + "; ".join(f"{p} sha256:{h}" for p, h in paths),
        PL.git("rev-parse", "HEAD"),
        outcome_blind=True,
    )
    dms = json.loads((LO.STATE / "state.json").read_text())
    done = set(RG.read_study(STUDY).auth)
    for oid, rec in sorted(dms["instances"].items()):
        if oid == a.iid or rec.get("confirmed_terminated_at") is None or LO.guard_instance(oid) in done:
            continue
        gp = LO.GPUS.get(rec.get("type"), 1)
        la, te = LO.reconcile(g, LO.STATE, oid, gp)
        log(f"earlier instance {rec.get('name')} reconciled: {(te - la) / 3600 * gp:.3f} GPU-h-eq")
    g.open_instance(1)
    rec = C.run_canary(g, rem)
    log("start canary (persisted reference)", rec["pass"])
    if not rec["pass"]:
        g.incident("canary_mismatch", {"where": "resume start canary", "composite": rec["composite"]})
        raise SystemExit("INCIDENT: start canary != the persisted reference")
    if att_p.read_bytes() != att_bytes:
        raise SystemExit("INTERNAL: attestation.json changed during the resume")
    print(json.dumps({"resumed": inst, "contract": contract, "canary_reference": att.canary_reference,
                      "epoch": g.epoch, "report": g.report()}, default=str))  # fmt: skip


def MO_theta0() -> str:
    from wsp.wm.c1 import modelops as MO

    return MO.manifest()["target.theta0_hash"]


def load_att():
    """The attestation of the CURRENT contract. Once the study log exists it is read from the log's last
    ``attestation`` / ``contract_transition`` record (hash-chained; its digest is checked against the log's contract);
    ``attestation.json`` stays the generation-0 file and is never rewritten (Board A B1)."""
    if (STUDY / RG.LOG_NAME).exists():
        s = RG.read_study(STUDY)
        if s.contracts:
            r = s.contracts[-1]
            att = RG.LaunchAttestation(**{k: r[k] for k in RG.LaunchAttestation.__dataclass_fields__})
            if RG.contract_digest(att) != s.contract:
                raise SystemExit("INTERNAL: the study log's contract record does not reproduce its digest")
            return att
    d = json.loads((RUNS / "attestation.json").read_text())
    return RG.LaunchAttestation(**d)


def contract_state() -> str:
    """``current``: the current contract binds this commit's engine config. ``pre_transition``: it binds the
    configuration before the T4 field entry (RULING-wme-c1-bank-halt addendum 4): only ``phasel-main --resume``,
    ``field-timing``, ``transition`` and ``close`` may run. Anything else refuses."""
    att = load_att()
    if att.engine_config_sha256 == ec_sha():
        return "current"
    if att.engine_config_sha256 == ec_sha(field=False):
        return "pre_transition"
    raise SystemExit("REFUSED: the engine config differs from the current contract's (neither current nor pre-transition)")


def require_current_contract(what: str):
    if contract_state() != "current":
        raise SystemExit(
            f"REFUSED: {what} needs the post-transition contract; the contract transition is pending "
            f"(RULING-wme-c1-bank-halt addendum 4: run `transition` first)"
        )


def cmd_run(a):
    from wsp.wm.arms import llm as L
    from wsp.wm.c1 import bundle as BU
    from wsp.wm.c1 import controller as CO
    from wsp.wm.c1 import modelops as MO

    require_current_contract("run")
    tr, rem = remote_for("main", a.iid)
    att = load_att()
    man = MO.manifest()
    st = SD.Study(STUDY, RUNS, att, man["target.theta0_hash"], PANEL_ID, log=log)
    client = lambda role: L.ClaudeCLI(READER_MODEL)
    co = CO.Controller(st, rem, tr, LO.guard_instance(a.iid), 1, log=log, client_factory=client)
    co.field_inputs = (man, json.loads((ROOT / "data/wme/bank_roots/roots-9b-20261117/roots.json").read_text()))
    roots = json.loads((ROOT / "data/wme/bank_roots/roots-9b-20261117/roots.json").read_text())
    stages = a.stages.split(",")
    saved = lambda s: json.loads((RUNS / "stages" / f"{s}.json").read_text())
    if "bank" in stages:
        bank = co.stage_bank(man, roots)
        bj = BU.bank_json(bank, man, roots)
        if bj["graph"]["hash"] != man["bank.component_graph_hash"]:
            raise SystemExit("component graph differs from the frozen hash")
        log(
            "bank bundle",
            BU.write(RUNS / "bundle/bank.json", bj),
            len(bj["states"]),
            "states;",
            len(bj["fallbacks"]),
            "fallbacks",
        )
    if "prep" in stages:
        bj = json.loads((RUNS / "bundle/bank.json").read_text())
        co.stage_prep(
            RUNS / "bundle/bank.json",
            man["bank.calibration_components"],
            BU.strata(bj),
            man["bank.folds.inner_seed"],
        )
    if "e2" in stages:
        co.stage_e2(saved("prep"))
    if "sbon" in stages:
        bp = co.b_par()
        log("B_par", json.dumps(bp))
        co.stage_sbon(saved("prep"), bp)
    if "designate" in stages:
        w = co.designate(saved("e2"), saved("sbon")["results"], ROOT / "data/wme/winners/wme-c1")
        log("winners manifest", w["sha256"], w["e1"].get("status"))
    if "sealed" in stages:
        co.stage_sealed(
            saved("prep"), saved("winners"), ROOT / "data/wme/panels/wme-c1/sealed_test/test.jsonl"
        )
    if "gen" in stages:
        co.stage_gen(
            saved("prep"),
            saved("winners"),
            ROOT / "data/wme/panels/wme-c1/sealed_gen/gen.jsonl",
            man["gen.sampled_seeds"],
        )
    if "ste" in stages:
        gen = saved("gen") if (RUNS / "stages/gen.json").exists() else None
        co.stage_ste(
            {"sealed_test": ROOT / "data/wme/panels/wme-c2-gen/sealed_test/sealed_test.jsonl"}, gen
        )
    log("stages done:", stages)


def cmd_repair_field(a):
    """RULING-wme-c1-bank-halt K3: re-materialise the field on the live main instance, verified bitwise against the
    eligible field payloads; no unit result. Halts (incident) on any mismatch or unverifiable payload."""
    from wsp.wm.c1 import controller as CO
    from wsp.wm.c1 import modelops as MO

    require_current_contract("repair-field")
    tr, rem = remote_for("main", a.iid)
    man = MO.manifest()
    st = SD.Study(STUDY, RUNS, load_att(), man["target.theta0_hash"], PANEL_ID, log=log)
    co = CO.Controller(st, rem, tr, LO.guard_instance(a.iid), 1, log=log)
    roots = json.loads((ROOT / "data/wme/bank_roots/roots-9b-20261117/roots.json").read_text())
    res = co.repair_field(man, roots, commit=git_sha())
    log("field repair verified", res["verified"], "units")
    print(json.dumps({k: res[k] for k in ("verified", "instance")}))


def _evidence_only_path(rel: str) -> bool:
    """A path an evidence-only commit may touch (Board A v2u C2): under ``docs/``, never under ``docs/prereg/`` and
    never a bundled file (``BUNDLE_FILES``: the manifest and the shard file are code inputs of the worker)."""
    return rel.startswith("docs/") and not rel.startswith("docs/prereg/") and rel not in BUNDLE_FILES


def same_code(commit: str) -> bool:
    """P2 across evidence-only commits (Board A v2u C2): ``commit`` is HEAD, or an ancestor of HEAD such that every
    path changed since (``git diff --no-renames --name-status``: a rename is a deletion plus an addition) is an
    evidence path (:func:`_evidence_only_path`)."""
    from wsp.wm.c1 import phasel as PL

    head = PL.git("rev-parse", "HEAD", cwd=ROOT)
    if not commit or commit == head:
        return bool(commit)
    try:
        PL.git("merge-base", "--is-ancestor", commit, head, cwd=ROOT)
        out = PL.git("diff", "--no-renames", "--name-status", commit, head, cwd=ROOT)
    except subprocess.CalledProcessError:
        return False
    paths = [ln.split("\t", 1)[1] for ln in out.splitlines() if "\t" in ln]
    return all(_evidence_only_path(p) for p in paths)


def require_clean_tree(what: str):
    """Board A v2u C2: ``field-timing`` and ``transition`` run only from a clean tracked tree."""
    from wsp.wm.c1 import phasel as PL

    dirty = PL.git("status", "--porcelain", "--untracked-files=no", cwd=ROOT)
    if dirty:
        raise SystemExit(f"REFUSED: {what} needs a clean tracked tree:\n{dirty}")


def hello_identity(hello: dict) -> dict:
    return {
        "code_sha": hello.get("code_sha"),
        "driver": (hello.get("physical") or {}).get("driver"),
        "gpu_names": (hello.get("physical") or {}).get("gpu_names"),
        "versions": hello.get("versions"),
    }


def c6_record(runs: list) -> dict:
    """Board A v2u C6 (report only): the deterministic mmlupro_gold_lp.g0.pt of field-timing against T2's."""
    det = next((r for r in runs if r.get("mode") == "deterministic"), None)
    got = None if det is None else (det.get("file_sha256") or {}).get("mmlupro_gold_lp.g0.pt")
    eq = got == T2_MMLUPRO_DET
    return {
        "t2": T2_MMLUPRO_DET,
        "field_timing": got,
        "equal": eq,
        "note": "cross-instance deterministic reproduction: K3 recovery on a new instance is supported"
        if eq
        else "differs from T2 (another instance): a lost instance cannot be repaired bitwise; the one live "
        "instance rule (persistence option (ii)) is strictly binding",
    }


def evidence_entry(rel: str) -> str:
    return f"{rel} sha256:{sha(ROOT / rel)}"


def cmd_field_timing(a):
    """RULING-wme-c1-bank-halt addendum 4, T4 item 3: on the live main instance (after ``phasel-main --resume``), one
    non-unit ``field_timing`` request: field group g0 collected registered, deterministic, registered (the first run
    warms the process); the slowdown against the warm registered run, per objective and in total. Evidence only: no
    unit, no field written."""
    from wsp.wm.c1 import phasel as PL

    require_clean_tree("field-timing")  # Board A v2u C2
    state = contract_state()
    head = PL.git("rev-parse", "HEAD")
    tr, rem = remote_for("main", a.iid)
    inst = LO.guard_instance(a.iid)
    st = SD.Study(STUDY, RUNS, load_att(), MO_theta0(), PANEL_ID, log=log)
    g = st.guard("bank", inst)
    C.take_over(tr, rem, g)
    hello, _ = rem.call("hello", g.epoch, inst, "m" * 64, "hello", {})  # Board A v2u C3: the worker's code
    hid = hello_identity(hello)
    if not same_code(hid["code_sha"]):
        raise SystemExit(f"REFUSED: the worker runs {hid['code_sha']}, not this code (P2)")
    uk = hashlib.sha256(f"wme-c1-field-timing:{a.tag}".encode()).hexdigest()
    p, _ = rem.call("field_timing", g.epoch, inst, uk, "field_timing:g0", {"group": 0})
    if not p["vocab_probe"]["ok"]:  # Board A v2u C4: halt and report; nothing is substituted
        ev = {"what": "T4 item 3 / Board A v2u C4: the deterministic vocabulary probe raised",
              "engine_config_sha256": ec_sha(), "commit": head, "instance": inst, "hello": hid,
              "vocab_probe": p["vocab_probe"]}  # fmt: skip
        pth, h = write_evidence(f"wme-c1-field-timing-{utc_date()}-{a.tag}.json", ev)
        commit_evidence([pth], f"wme-c1 T4: the deterministic vocabulary probe raised on {inst} (halt)")
        raise SystemExit(f"HALT: the deterministic vocabulary probe raised: {p['vocab_probe']['error']['message'][:400]}")
    det = next(r for r in p["runs"] if r["mode"] == "deterministic")
    warm = [r for r in p["runs"] if r["mode"] == "registered"][-1]
    per = {k: det["seconds"][k] / warm["seconds"][k] for k in det["seconds"]}
    total = sum(det["seconds"].values()) / sum(warm["seconds"].values())
    ev = {
        "what": "T4 item 3 (RULING-wme-c1-bank-halt addendum 4): field g0 registered vs deterministic timing",
        "engine_config_sha256": ec_sha(),
        "commit": head,
        "contract_state": state,
        "instance": inst,
        "hello": hid,
        "group": p["group"],
        "modules": p["modules"],
        "vocab_probe": p["vocab_probe"],
        "runs": p["runs"],
        "c6": c6_record(p["runs"]),
        "slowdown": {
            "per_objective": per,
            "total": total,
            "baseline": "the second (warm) registered collection",
        },
    }
    pth, h = write_evidence(f"wme-c1-field-timing-{utc_date()}-{a.tag}.json", ev)
    commit_evidence([pth], f"wme-c1 T4 item 3: field g0 timing on {inst} (deterministic/registered {total:.3f})")
    log("field timing", pth, h, json.dumps({"total": total, "per_objective": per}))
    print(json.dumps({"evidence": pth, "sha256": h, "slowdown": ev["slowdown"], "c6": ev["c6"],
                      "vocab_probe_s": p["vocab_probe"]["seconds"]}))  # fmt: skip
    return ev


def cmd_transition(a):
    """RULING-wme-c1-bank-halt addendum 4, T4 item 2: the contract transition on the live main instance, after
    ``phasel-main --resume`` (generation 0) and ``field-timing``.

    1. offline, before any write: the current contract is the pre-transition one (its engine config is this commit's
       minus the ``field`` entry); the parity and timing evidence are bound to this commit's engine config and code
       (P2 across evidence-only commits); the parity passed (C2); every open incident is one this transition resolves;
    2. the new attestation: the same launch facts and canary reference, this commit's engine config, the evidence
       list extended by the ruling, the T1-T3 preparation, the parity and the timing evidence;
    3. a new owner under the registered guard mechanism with a ``ContractTransition`` (outcome-blind): every
       attempt of the earlier generation (all bank-phase and field units) becomes ``superseded_contract``;
    4. the open ``field_repair_mismatch`` incident resolved, citing the ruling and the transition record;
    5. the instance opened under the new contract, any on-disk field quarantined (C4), the start canary against the
       unchanged reference. ``attestation.json`` (generation 0) is never rewritten; the new attestation is kept in
       ``attestation-g<generation>.json`` and, authoritatively, in the study log."""
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import phasel as PL

    if not (STUDY / RG.LOG_NAME).exists():
        raise SystemExit("REFUSED: no study to transition")
    require_clean_tree("transition")  # Board A v2u C2
    old = load_att()
    ecs = ec_sha()
    if old.engine_config_sha256 == ecs:
        raise SystemExit("REFUSED: nothing to transition: the current contract already binds this engine config")
    if old.engine_config_sha256 != ec_sha(field=False):
        raise SystemExit("REFUSED: the current contract is not the pre-transition contract")
    head = PL.git("rev-parse", "HEAD")
    par_path = str(Path(a.parity_evidence))
    par = json.loads((ROOT / par_path).read_text())
    if par.get("engine_config_sha256") != ecs:
        raise SystemExit("PRECONDITION FAILED: the parity evidence is not bound to this engine config")
    if not same_code((par.get("code_identity") or {}).get("controller_commit")):
        raise SystemExit("PRECONDITION FAILED: the parity evidence was not produced by this code (P2)")
    ev = par["evaluation"]
    if not ev["proceed"] or ev["selected_engine"] != GE.GEN_ENGINE:
        raise SystemExit(f"PRECONDITION FAILED: parity evaluation {ev['selected_engine']} / {ev['proceed']}")
    if GE.GEN_ENGINE == "hf_deterministic" and ev["pass_rule"].get("hf_gen_eq_committed") is not True:
        raise SystemExit("PRECONDITION FAILED: the parity run did not reproduce the committed hf_gen reference (C2)")
    tim_path = str(Path(a.timing_evidence))
    tim = json.loads((ROOT / tim_path).read_text())
    if tim.get("engine_config_sha256") != ecs or not same_code(tim.get("commit")):
        raise SystemExit("PRECONDITION FAILED: the field timing evidence is not bound to this engine config and code")
    if not any(r.get("mode") == "deterministic" for r in tim.get("runs", [])):
        raise SystemExit("PRECONDITION FAILED: the field timing evidence has no deterministic collection")
    inst = LO.guard_instance(a.iid)
    if tim.get("instance") != inst:  # Board A v2u C3: the timing came from this live instance
        raise SystemExit(f"PRECONDITION FAILED: the field timing ran on instance {tim.get('instance')}, not {inst}")
    if (tim.get("vocab_probe") or {}).get("ok") is not True:  # Board A v2u C4
        raise SystemExit("PRECONDITION FAILED: the deterministic vocab probe is missing or did not pass")
    s = RG.read_study(STUDY)
    other = [i["kind"] for i in s.incidents if i["kind"] not in TRANSITION_RESOLVES]
    if other:
        raise SystemExit(f"REFUSED: open incidents this transition does not resolve: {other}")
    resolves = [(i["kind"], i["seq"]) for i in s.incidents]
    for rel in (TRANSITION_RULING, TRANSITION_PREP):
        if not (ROOT / rel).exists():
            raise SystemExit(f"PRECONDITION FAILED: {rel} is missing")
    new = RG.LaunchAttestation(
        old.inert_identity_bitwise,
        old.gpu_parity_bitwise,
        old.canary_reference,
        ecs,
        sorted(set(old.evidence) | {evidence_entry(r) for r in (TRANSITION_RULING, TRANSITION_PREP, par_path, tim_path)}),
    )
    gen = len(s.contracts)
    ap = RUNS / f"attestation-g{gen}.json"
    if ap.exists():
        raise SystemExit(f"REFUSED: {ap.name} exists (never overwritten)")
    att_bytes = (RUNS / "attestation.json").read_bytes()
    desc = (
        f"{TRANSITION_RULING} addendum 4 (T4): field and vocabulary collection under the registered deterministic "
        f"context (engine config {old.engine_config_sha256[:12]} -> {ecs[:12]}, field entry added). The "
        "registered-mode collection backward was not reproducible (D1: two collections on one instance differed in "
        "the mmlupro_gold_lp gradients); T2: 3/3 bitwise under the deterministic context. Every bank-phase and field "
        f"unit of generation {s.generation} is superseded; resolves open incident(s) {resolves}; canary reference "
        f"unchanged ({old.canary_reference[:12]}); parity {par_path}; field timing {tim_path}"
    )
    tr, rem = remote_for("main", a.iid)
    # Board A v2u C3: before any write, the live worker runs the parity commit's code (hello at the current fence
    # epoch: the owner that ran field-timing on this instance; no new owner is written)
    hello, _ = rem.call("hello", s.epoch, inst, "m" * 64, "hello", {})
    hid = hello_identity(hello)
    pc = (par.get("code_identity") or {}).get("controller_commit")
    if hid["code_sha"] != pc or not same_code(hid["code_sha"]):
        raise SystemExit(f"PRECONDITION FAILED: the worker runs {hid['code_sha']}, not the parity commit {pc}")
    st = SD.Study(STUDY, RUNS, new, MO_theta0(), PANEL_ID, log=log)
    g = st.guard(
        "bank", inst, transition=RG.ContractTransition(TRANSITION_CAUSE, desc, head, outcome_blind=True)
    )
    C.take_over(tr, rem, g)
    trec = RG.read_study(STUDY).contracts[-1]
    PL.write_new(ap, RG._att_fields(new))
    log("contract transition", json.dumps({"generation": g.generation, "seq": trec["seq"], "contract": g.contract}))
    if resolves:
        g.resolve_incidents(
            f"superseded by the contract transition to generation {g.generation} (study log seq {trec['seq']}): "
            f"the field is re-collected under the deterministic context and every bank-phase unit recomputed; "
            f"incident(s) {resolves}",
            f"{TRANSITION_RULING} addendum 4",
        )
    g.open_instance(1)
    q, _ = rem.call("field_quarantine", g.epoch, inst, "q" * 64, "field_quarantine: before the transition", {})
    rec = C.run_canary(g, rem)
    log("start canary (new contract, unchanged reference)", rec["pass"])
    if not rec["pass"]:
        g.incident("canary_mismatch", {"where": "transition start canary", "composite": rec["composite"]})
        raise SystemExit("INCIDENT: start canary != the reference under the new contract")
    if (RUNS / "attestation.json").read_bytes() != att_bytes:
        raise SystemExit("INTERNAL: attestation.json changed during the transition")
    out = {
        "what": "T4 contract transition (RULING-wme-c1-bank-halt addendum 4)",
        "instance": inst,
        "epoch": g.epoch,
        "generation": g.generation,
        "previous_contract": trec["previous"],
        "contract": g.contract,
        "transition_seq": trec["seq"],
        "cause": TRANSITION_CAUSE,
        "resolved_incidents": resolves,
        "attestation": RG._att_fields(new),
        "attestation_file": ap.name,
        "quarantine": q,
        "canary": {k: rec[k] for k in ("engines", "composite", "pass")},
        "commit": head,
        "hello": hid,
    }
    pth, h = write_evidence(f"wme-c1-transition-{utc_date()}-{a.tag}.json", out)
    commit_evidence([pth], f"wme-c1 T4 contract transition on {inst}: generation {g.generation} ({g.contract[:12]})")
    print(json.dumps({k: out[k] for k in ("generation", "contract", "transition_seq", "resolved_incidents")}, default=str))
    return out


def cmd_resolve_incident(a):
    """RULING-wme-c1-bank-halt addendum 5: resolve ONE open run-health incident, naming its seq, the diagnosis and
    the chair ruling it cites. Refuses (before any write) a seq that is no incident record or is already resolved."""
    s = RG.read_study(STUDY)
    seq = int(a.seq)
    if seq not in s.incident_seqs:
        raise SystemExit(f"REFUSED: seq {seq}: no incident record")
    inc = next((i for i in s.incidents if i["seq"] == seq), None)
    if inc is None:
        raise SystemExit(f"REFUSED: incident seq {seq} is already resolved")
    st = SD.Study(STUDY, RUNS, load_att(), MO_theta0(), PANEL_ID, log=log)
    g = st.guard("bank", "operator:resolve-incident")
    rec = g.resolve_incident(seq, a.diagnosis, a.ruling)
    out = {
        "incident_seq": seq,
        "kind": inc["kind"],
        "resolution_seq": rec["seq"],
        "diagnosis": a.diagnosis,
        "ruling": a.ruling,
        "open_after": [(i["kind"], i["seq"]) for i in RG.read_study(STUDY).incidents],
    }
    log("incident resolved", json.dumps(out))
    print(json.dumps(out))
    return out


def cmd_resume_phase(a):
    """RULING-wme-c1-bank-halt addendum 7 (§II.13 item 9, §II.14): record the chair's resumption of a paused phase
    from ledger seconds only. Refuses, before any write, unless the 1.25x pause is pending for the phase's next unit
    (measured unit GPU-h + its registered estimate > 1.25 x its price) and the phase was not resumed before."""
    s = RG.read_study(STUDY)
    st = SD.Study(STUDY, RUNS, load_att(), MO_theta0(), PANEL_ID, log=log)
    phase = a.phase
    if phase in s.phase_resumes:
        raise SystemExit(f"REFUSED: {phase} already resumed (seq {s.phase_resumes[phase]['seq']})")
    measured = s.unit_s.get(phase, 0.0) / 3600
    est = SD.EST_UNIT_GPU_H[phase]
    thr = RG.PHASE_PAUSE * float(st.priced[phase])
    if not measured + est > thr:
        raise SystemExit(f"REFUSED: {phase}: no PhasePause pending ({measured:.3f} + {est:.3f} <= {thr:.3f} GPU-h)")
    g = st.guard(phase, "operator:resume-phase")
    rec = g.resume_phase(phase, a.ruling)
    out = {k: rec[k] for k in ("seq", "resumed_phase", "decided_by", "date", "ruling", "outcome_blind", "basis")}
    log("phase resumed", json.dumps(out))
    print(json.dumps(out))
    return out


def cmd_record_repair(a):
    """The registered repair record (``RetryPolicy.repair``: a registered technical cause, the code commit,
    outcome-blind) for a unit whose last failure in the current contract generation had no registered technical
    cause (§II.31.8 R3; RULING-wme-c1-bank-halt addendum 8: ``inert:05:comp``), naming the unit and the ruling.
    "Awaiting" means no later repair record names the unit (a resume's own repair record, which also licenses the
    retry under the guard's rule, does not name it). Refuses, before any write, an unregistered cause or a unit with
    no failure awaiting its named repair."""
    from wsp.wm.c1 import phasel as PL

    if a.cause not in RG.TECHNICAL_CAUSES:
        raise SystemExit(f"REFUSED: {a.cause!r} is not a registered technical cause {RG.TECHNICAL_CAUSES}")
    s = RG.read_study(STUDY)
    awaiting = {
        fs
        for (_u, g, c), fs in s.gen_last_fail_seq.items()
        if g == s.generation
        and c == s.contract
        and s.gen_last_cause.get((_u, g, c)) not in RG.TECHNICAL_CAUSES
    }
    fails = {}
    for line in (STUDY / RG.LOG_NAME).read_text().splitlines():
        r = json.loads(line)
        if r["type"] == "attempt_fail" and r["seq"] in awaiting:
            named = any(rp["seq"] > r["seq"] and f"{r['label']}:" in rp.get("description", "") for rp in s.repairs)
            if not named:
                fails[r["label"]] = r
    if a.label not in fails:
        raise SystemExit(
            f"REFUSED: {a.label}: no failure awaiting a repair in generation {s.generation} "
            f"(awaiting: {sorted(fails)})"
        )
    f = fails[a.label]
    st = SD.Study(STUDY, RUNS, load_att(), MO_theta0(), PANEL_ID, log=log)
    g = st.guard(f.get("phase") or "bank", "operator:record-repair")
    desc = (
        f"{a.label}: retry licensed after its failure (study log seq {f['seq']}, {f.get('error')}, no registered "
        f"technical cause) by {a.ruling}; no unit result created"
    )
    rec = g.retry.repair(a.cause, desc, PL.git("rev-parse", "HEAD"), outcome_blind=True)
    out = {"seq": rec["seq"], "cause": a.cause, "label": a.label, "failure_seq": f["seq"], "ruling": a.ruling}
    log("repair recorded", json.dumps(out))
    print(json.dumps(out))
    return out


def cmd_close(a):
    tr, rem = remote_for("main", a.iid)
    st = SD.Study(STUDY, RUNS, load_att(), "", PANEL_ID, log=log)
    g = st.guard(a.phase, LO.guard_instance(a.iid))
    C.take_over(tr, rem, g)
    rec = C.run_canary(g, rem)
    log("closing canary", rec["pass"])
    g.close_instance()
    dm = LO.deadman()
    t = LO.terminate_and_confirm(dm, a.iid, "wme-c1 instance closed")
    la, te = LO.reconcile(g, LO.STATE, a.iid, 1)
    log("terminated and reconciled", json.dumps({"launched_at": la, "terminated_at": te}))
    print(json.dumps(g.report(), default=str))


def cmd_status(a):
    out = {}
    if (STUDY / RG.LOG_NAME).exists():
        led = RG.CombinedLedger(STUDY / RG.LOG_NAME, {"bank": 1})
        out["charged_gpu_h"] = led.gpu_h()
        out["unit_gpu_h"] = led.unit_gpu_h()
        out["usd"] = led.usd()
        out["remaining_usd"] = led.remaining_usd()
    dm = LO.deadman()
    s = dm.state()
    out["deadman"] = {
        "live": dm.live(s),
        "instance_hours": dm.instance_hours(s),
        "cost_usd": dm.cost_usd(s),
    }
    print(json.dumps(out, indent=1, default=str))


AMENDMENTS = [
    {
        "section": "§II.29",
        "label": "v2j (19 Nov 2026): execution specification of the frozen design (G1-G21), outcome-blind, before any data",
        "adaptation": False,
    },
    {
        "section": "§II.30",
        "label": "v2k (19 Nov 2026): bank re-pricing rule for unpriced kept-factor and R(E) work, before any data",
        "adaptation": False,
    },
]


def cmd_results(a):
    """Assemble the results bundle from eligible stage payloads and the fetched result files, then the report."""
    from wsp.wm.c1 import bundle as BU
    from wsp.wm.c1 import controller as CO

    out = RUNS / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    saved = lambda s: (
        json.loads((RUNS / "stages" / f"{s}.json").read_text())
        if (RUNS / "stages" / f"{s}.json").exists()
        else None
    )
    tr, rem = (
        remote_for("main", a.iid)
        if a.iid
        else (None, C.Remote(T.LocalTransport(RUNS / "ctl_local"), RUNS / "ctl"))
    )
    co = CO.Controller(SD.Study(STUDY, RUNS, load_att(), "", PANEL_ID), rem, tr, "", 1)
    files = {"bank.json"}
    prep = saved("prep")
    if prep:
        BU.write(out / "fits.json", BU.fits_json(prep))
        files.add("fits.json")
    e2, sb = saved("e2"), saved("sbon")
    if e2:
        BU.write(out / "e2.json", BU.e2_json(e2, sb or {}))
        files.add("e2.json")
    w = saved("winners")
    if w:
        man = json.loads(Path(w["path"]).read_bytes())
        BU.write(out / "winners.json", {"manifest": man, "sha256": w["sha256"], "e1": w["e1"]})
        files.add("winners.json")
        sealed = saved("sealed")
        if sealed:
            BU.write(out / "sealed.json", BU.sealed_json(sealed, man))
            files.add("sealed.json")
    gen = saved("gen")
    if gen and not gen.get("skipped"):
        import gzip

        rows, items = [], None
        for lab, p in sorted(gen.items()):
            if p is None or p.get("fallback"):
                continue
            f = co.find_file("gen_e1_outputs.jsonl.gz", p["files"]["gen_e1_outputs.jsonl.gz"])
            rows += gzip.open(f, "rt").read().splitlines()
            items = items or co.find_file(
                "gen_e1_items.jsonl.gz", p["files"]["gen_e1_items.jsonl.gz"]
            )
        with gzip.open(out / "gen_e1_outputs.jsonl.gz", "wt") as fh:
            fh.write("\n".join(rows) + "\n")
        shutil.copyfile(items, out / "gen_e1_items.jsonl.gz")
        g0 = gen.get("gen:greedy") or {}
        BU.write(
            out / "gen_e1.json",
            {
                "items": "gen_e1_items.jsonl.gz",
                "outputs": "gen_e1_outputs.jsonl.gz",
                "winner": g0.get("winner"),
                "config": g0.get("config"),
            },
        )
        files |= {"gen_e1.json", "gen_e1_outputs.jsonl.gz", "gen_e1_items.jsonl.gz"}
    ste = saved("ste")
    if ste:
        g = ste["generate"]
        for n in ("ste_outputs.jsonl.gz", "ste_guards.jsonl.gz"):
            shutil.copyfile(co.find_file(n, g["files"][n]), out / n)
        BU.write(
            out / "ste.json",
            {
                "margins": g["margins"],
                "null_pass_prob": ste["calibrate"].get("null_pass_prob"),
                "calibration": ste["calibrate"],
                "outputs": "ste_outputs.jsonl.gz",
                "guards": "ste_guards.jsonl.gz",
                "n": g["n"],
                "canary": g.get("canary"),
                "train": ste["train"],
            },
        )
        files |= {"ste.json", "ste_outputs.jsonl.gz", "ste_guards.jsonl.gz"}
    dm = json.loads((LO.STATE / "state.json").read_text())
    BU.write(out / "ledger.json", BU.ledger_json(STUDY, dm, AMENDMENTS))
    files.add("ledger.json")
    h = BU.finish(out, files)
    log("bundle", out, h)
    if a.report:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/wme_c1_report.py"),
                "--bundle",
                str(out),
                "--date",
                a.date,
            ],
            check=True,
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--out", required=True)
    u = sub.add_parser("up")
    u.add_argument("--kind", choices=("parity", "main"), required=True)
    u.add_argument("--tag", default="")
    u.add_argument("--bundle", required=True)
    u.add_argument("--t-max-hours", type=float, default=None)  # default: the remaining cap (§II.31)
    u.add_argument("--poll-max-min", type=float, default=1440)
    p = sub.add_parser("phasel-parity")
    p.add_argument("--iid", required=True)
    p.add_argument("--tag", required=True)  # run tag of the evidence files (R2; never overwritten)
    m = sub.add_parser("phasel-main")
    m.add_argument("--iid", required=True)
    m.add_argument("--tag", required=True)
    m.add_argument("--parity-evidence", required=True)  # repo-relative path of the bound parity evidence
    m.add_argument("--resume", action="store_true")  # Board A B1: keep the persisted contract
    m.add_argument("--parity-iid", default=None)
    r = sub.add_parser("run")
    r.add_argument("--iid", required=True)
    r.add_argument("--stages", required=True)
    fr = sub.add_parser("repair-field")
    fr.add_argument("--iid", required=True)
    ft = sub.add_parser("field-timing")  # RULING-wme-c1-bank-halt addendum 4, T4 item 3
    ft.add_argument("--iid", required=True)
    ft.add_argument("--tag", required=True)
    tn = sub.add_parser("transition")  # RULING-wme-c1-bank-halt addendum 4, T4 item 2
    tn.add_argument("--iid", required=True)
    tn.add_argument("--tag", required=True)
    tn.add_argument("--parity-evidence", required=True)
    tn.add_argument("--timing-evidence", required=True)
    ri = sub.add_parser("resolve-incident")  # RULING-wme-c1-bank-halt addendum 5
    ri.add_argument("--seq", required=True, type=int)
    ri.add_argument("--diagnosis", required=True)
    ri.add_argument("--ruling", required=True)
    sub.add_parser("preflight")  # addendum 5
    rr = sub.add_parser("record-repair")  # addendum 8: the repair record that licenses a non-technical retry
    rr.add_argument("--label", required=True)
    rr.add_argument("--cause", required=True)
    rr.add_argument("--ruling", required=True)
    rp = sub.add_parser("resume-phase")  # addendum 7: the chair's PhasePause resumption
    rp.add_argument("--phase", required=True)
    rp.add_argument("--ruling", required=True)
    c = sub.add_parser("close")
    c.add_argument("--iid", required=True)
    c.add_argument("--phase", default="confirmatory")
    sub.add_parser("status")
    rs = sub.add_parser("results")
    rs.add_argument("--iid", default=None)
    rs.add_argument("--report", action="store_true")
    rs.add_argument("--date", default="2026-11-21")
    a = ap.parse_args()
    {
        "bundle": cmd_bundle,
        "up": cmd_up,
        "phasel-parity": cmd_phasel_parity,
        "phasel-main": cmd_phasel_main,
        "run": cmd_run,
        "repair-field": cmd_repair_field,
        "field-timing": cmd_field_timing,
        "resolve-incident": cmd_resolve_incident,
        "preflight": cmd_preflight,
        "resume-phase": cmd_resume_phase,
        "record-repair": cmd_record_repair,
        "transition": cmd_transition,
        "close": cmd_close,
        "status": cmd_status,
        "results": cmd_results,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
