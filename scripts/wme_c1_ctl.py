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
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import runguard as RG  # noqa: E402
from wsp.wm.c1 import ctl as C  # noqa: E402
from wsp.wm.c1 import lambda_ops as LO  # noqa: E402
from wsp.wm.c1 import study as SD  # noqa: E402
from wsp.wm.c1 import transport as T  # noqa: E402

DATE = "2026-11-19"
RUNS = ROOT / "data/wme/runs/wme-c1"
STUDY = RUNS / "study"
EVID = ROOT / "docs/reviews/wme"
REMOTE = "/home/ubuntu/wmec1"
ALIAS = {"parity": "wmec1p", "main": "wmec1"}
TYPES = {"parity": LO.TYPE2, "main": LO.TYPE}
PANEL_ID = "wme-c1:89739c7818e29d618ac3bcb52d7e51cd29a9361e7774a71b412646775c630c1d"
READER_MODEL = "claude-opus-5-5"  # amendment G2
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
    out = Path(a.out)
    stage = Path(tempfile.mkdtemp()) / "wmec1"
    shutil.copytree(ROOT / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in BUNDLE_FILES:
        if "sealed" in rel:
            raise SystemExit(f"refusing a sealed path: {rel}")
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    (stage / "CODE_SHA").write_text(git_sha())
    bad = [str(p) for p in stage.rglob("*") if "sealed" in p.name or "/sealed_" in str(p)]
    if bad:
        raise SystemExit(f"sealed path in bundle: {bad[:3]}")
    with tarfile.open(out, "w:gz") as tf:
        tf.add(stage, arcname="wmec1")
    print(json.dumps({"bundle": str(out), "sha256": sha(out), "code": git_sha()}))


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
    LO.STATE.mkdir(parents=True, exist_ok=True)
    ensure_watcher()
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
    }
    (RUNS / "instances").mkdir(parents=True, exist_ok=True)
    (RUNS / "instances" / f"{iid}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


def remote_for(kind: str, iid: str):
    alias = ALIAS[kind]
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

    return tr, C.Remote(
        tr,
        RUNS / "ctl",
        poll_s=10,
        dead_after_s=1800,
        transport_retry_s=900,
        lost=lost,
        log=log,
        revive=revive,
    )


# ------------------------------------------------------------------ Phase L


def write_evidence(name, obj) -> tuple[str, str]:
    p = EVID / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    return str(p.relative_to(ROOT)), sha(p)


def cmd_phasel_parity(a):
    tr, rem = remote_for("parity", a.iid)
    hello, _ = rem.call("hello", 0, LO.guard_instance(a.iid), "p" * 64, "hello", {})
    log("hello", json.dumps({k: hello[k] for k in ("theta0", "versions", "physical")}))
    par, _ = rem.call("parity", 0, LO.guard_instance(a.iid), "q" * 64, "parity", {}, timeout_s=7200)
    log("parity", json.dumps({k: par[k] for k in par if k not in ("chunks_k1", "chunks_k2")}))
    inert, _ = rem.call(
        "inert_identity", 0, LO.guard_instance(a.iid), "r" * 64, "inert", {}, timeout_s=7200
    )
    log("inert", inert["pass"])
    path, h = write_evidence(
        f"wme-c1-launch-parity-{DATE}.json",
        {
            "label": "wme-c1 launch precondition 2 (and 1, redundant) on a 2-GPU check instance; exposed engineering items only",
            "instance": LO.guard_instance(a.iid),
            "hello": hello,
            "parity": par,
            "inert_identity": inert,
        },
    )
    log("evidence", path, h)
    dm = LO.deadman()
    t = LO.terminate_and_confirm(dm, a.iid, "wme-c1 parity check finished")
    log("termination confirmed", t)
    (ROOT / "data/wme/runs/wme-c1/instances").mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "evidence": path,
                "sha256": h,
                "parity_bitwise": par["parity_bitwise"],
                "inert_pass": inert["pass"],
            }
        )
    )


def engine_config() -> dict:
    from wsp.wm.c1 import genengine as GE
    from wsp.wm.c1 import modelops as MO
    from wsp.wm.c1 import vllmops as VO

    man = MO.manifest()
    return {
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


def cmd_phasel_main(a):
    tr, rem = remote_for("main", a.iid)
    inst = LO.guard_instance(a.iid)
    hello, _ = rem.call("hello", 0, inst, "m" * 64, "hello", {})
    log("hello", json.dumps({k: hello[k] for k in ("theta0", "versions", "physical")}))
    inert, _ = rem.call("inert_identity", 0, inst, "n" * 64, "inert", {}, timeout_s=7200)
    log("inert identity pass:", inert["pass"])
    if not inert["pass"]:
        raise SystemExit("PRECONDITION FAILED: pinned-engine inert identity")
    can, _ = rem.call("canary", 0, inst, "o" * 64, "canary-reference", {}, timeout_s=7200)
    eng = can["engines"]
    if set(eng) != {"tf", "hf_gen", "vllm"}:
        raise SystemExit(f"PRECONDITION FAILED: canary engines {sorted(eng)}")
    tim, _ = rem.call("timing", 0, inst, "t" * 64, "timing", {}, timeout_s=7200)
    p_bank = (
        SD.PRICE["bank"]
        + (192 * tim["eigh_12288_s"] + 576 * tim["eigh_4096_s"] + 13095 * 6 * tim["contraction_s"])
        / 3600
    )
    tim_path, tim_sha = write_evidence(
        f"wme-c1-launch-timing-{DATE}.json",
        {
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
    par = json.loads((EVID / f"wme-c1-launch-parity-{DATE}.json").read_text())
    if not (par["parity"]["parity_bitwise"] and par["inert_identity"]["pass"]):
        raise SystemExit("PRECONDITION FAILED: 1-vs-2-GPU parity evidence")
    ref = C.composite_canary(eng)
    ec = engine_config()
    ec_path, ec_sha = write_evidence(f"wme-c1-engine-config-{DATE}.json", ec)
    can_path, can_sha = write_evidence(
        f"wme-c1-canary-refs-{DATE}.json",
        {
            "label": "per-engine canary references: the first confirmatory instance's start canaries (before any outcome)",
            "instance": inst,
            "engines": eng,
            "composite": ref,
            "detail": can.get("detail"),
        },
    )
    inert_path, inert_sha = write_evidence(
        f"wme-c1-launch-inert-{DATE}.json",
        {"instance": inst, "hello": hello, "inert_identity": inert},
    )
    par_path = f"docs/reviews/wme/wme-c1-launch-parity-{DATE}.json"
    evidence = sorted(
        [
            f"{par_path} sha256:{sha(ROOT / par_path)}",
            f"{inert_path} sha256:{inert_sha}",
            f"{can_path} sha256:{can_sha}",
            f"{ec_path} sha256:{ec_sha}",
            f"{tim_path} sha256:{tim_sha}",
        ]
    )
    subprocess.run(
        ["git", "add", par_path, inert_path, can_path, ec_path, tim_path], cwd=ROOT, check=True
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            "wme-c1 launch: committed per-engine canary references, pinned-engine inert identity, parity evidence and engine config (before any confirmatory unit)",
        ],
        cwd=ROOT,
        check=True,
    )
    att = RG.LaunchAttestation(
        inert_identity_bitwise=True,
        gpu_parity_bitwise=True,
        canary_reference=ref,
        engine_config_sha256=hashlib.sha256(json.dumps(ec, sort_keys=True).encode()).hexdigest(),
        evidence=evidence,
    )
    (RUNS / "attestation.json").write_text(json.dumps(RG._att_fields(att), indent=1))
    st = SD.Study(STUDY, RUNS, att, hello["theta0"], PANEL_ID, log=log)
    # bind every phase's registered fallback in the study log before the first unit (precondition 4)
    for ph in ("ste", "sbon", "confirmatory"):
        st.guard(ph, inst)
    g = st.guard("bank", inst)
    tr.set_fence(g.epoch, inst)
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


def load_att():
    d = json.loads((RUNS / "attestation.json").read_text())
    return RG.LaunchAttestation(**d)


def cmd_run(a):
    from wsp.wm.arms import llm as L
    from wsp.wm.c1 import bundle as BU
    from wsp.wm.c1 import controller as CO
    from wsp.wm.c1 import modelops as MO

    tr, rem = remote_for("main", a.iid)
    att = load_att()
    man = MO.manifest()
    st = SD.Study(STUDY, RUNS, att, man["target.theta0_hash"], PANEL_ID, log=log)
    client = lambda role: L.ClaudeCLI(READER_MODEL)
    co = CO.Controller(st, rem, tr, LO.guard_instance(a.iid), 1, log=log, client_factory=client)
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


def cmd_close(a):
    tr, rem = remote_for("main", a.iid)
    st = SD.Study(STUDY, RUNS, load_att(), "", PANEL_ID, log=log)
    g = st.guard(a.phase, LO.guard_instance(a.iid))
    tr.set_fence(g.epoch, g.instance)
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
    m = sub.add_parser("phasel-main")
    m.add_argument("--iid", required=True)
    m.add_argument("--parity-iid", default=None)
    r = sub.add_parser("run")
    r.add_argument("--iid", required=True)
    r.add_argument("--stages", required=True)
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
        "close": cmd_close,
        "status": cmd_status,
        "results": cmd_results,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
