#!/usr/bin/env python3
"""ENGINEERING SMOKE of the wme-c1 controller path on the local GPU (Qwen3.5-0.8B, <= 8 GB), exposed wme-p2 items
only (construction as field/dev/e2 shards, validation as bank panel, dev as the stand-in "test" file). Not
confirmatory: a scratch study directory under data/wme/runs/smoke/, never the wme-c1 study log.

    wme_c1_smoke.py --dir data/wme/runs/smoke/c1ctl [--episodes 1] [--b 8]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument(
        "--arms",
        default="wm_e,statnn,field_ranked,random,scripted_reader,wm_greedy,bon_steering,lora_sft,direct_bon_two_stage,wsr_reader",
    )
    a = ap.parse_args()
    from wsp.wm import runguard as RG
    from wsp.wm.arms import llm as L
    from wsp.wm.c1 import bundle as BU
    from wsp.wm.c1 import controller as CO
    from wsp.wm.c1 import ctl as C
    from wsp.wm.c1 import modelops as MO
    from wsp.wm.c1 import study as SD
    from wsp.wm.c1 import transport as T

    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    wroot = d / "worker"
    env = dict(os.environ, WME_C1_SMOKE_EPOCHS="10")
    log = open(d / "worker.log", "a")
    wp = subprocess.Popen(
        [sys.executable, "-m", "wsp.wm.c1.serve", "--root", str(wroot), "--smoke", "--no-vllm"],
        stdout=log,
        stderr=log,
        env=env,
    )
    t = T.LocalTransport(wroot)
    rem = C.Remote(t, d / "ctl", poll_s=0.5, dead_after_s=3600)
    out = {}
    try:
        hello, _ = rem.call("hello", 0, "smoke:local", "h" * 64, "hello", {})
        can, _ = rem.call("canary", 0, "smoke:local", "c" * 64, "canary-ref", {})
        ref = C.composite_canary(can["engines"])
        att = RG.LaunchAttestation(True, True, ref, "e" * 64, ["smoke: engineering only"])
        man = MO.manifest()
        st = SD.Study(d / "study", d / "runs", att, hello["theta0"], "smoke:wme-p2")
        st.priced = {k: 1000.0 for k in ("bank", "confirmatory", "sbon", "ste")}
        mock = lambda role: L.MockClient(
            lambda s, u: json.dumps({"ranking": [f"c{i:02d}" for i in range(64)]})
        )
        co = CO.Controller(st, rem, t, "smoke:local", 1, client_factory=mock, chunk=64)
        roots = json.loads((ROOT / "data/wme/bank_roots/smoke/roots.json").read_text())
        keep = ("scale:007", "steer:014", "noise:013")  # field groups 0-1 only
        roots = dict(roots, roots=[r for r in roots["roots"] if r["id"] in keep], controls=[])
        ids = [r["id"] for r in roots["roots"]]
        blocks = [["noise:013", "scale:007"]]
        t0 = time.time()
        bank = co.stage_bank(
            man, roots, blocks=blocks, field_groups=range(2), inert=False, vocab=False
        )
        out["bank_s"] = time.time() - t0
        cal = ["single:steer:014"]
        bj = BU.bank_json(bank, man, roots, blocks=blocks, calibration=cal)
        out["bank_states"] = len(bj["states"])
        bp = d / "bank.json"
        BU.write(bp, bj)
        prep = co.stage_prep(bp, cal, BU.strata(bj), man["bank.folds.inner_seed"])
        out["fit"] = {
            k: prep["fit"][k] for k in ("q_hat", "coverage", "mode", "n_train", "n_cov_units")
        }
        arms = a.arms.split(",")
        e2 = co.stage_e2(prep, episodes=range(a.episodes), arms=arms, b=a.b)
        out["e2"] = {
            k: (
                None
                if v is None
                else {
                    "winner": bool(v.get("winner")),
                    "n_eval": len(v.get("evaluated", [])),
                    "fallback": v.get("fallback", False),
                }
            )
            for k, v in e2.items()
        }
        bpar = co.b_par()
        bpar["B_par"] = 4.0  # smoke only: a small screen budget
        sb = co.stage_sbon(prep, bpar, episodes=range(a.episodes))
        out["sbon"] = {k: (v or {}).get("screened") for k, v in sb.items()}
        win = co.designate(e2, sb, d / "winners")
        out["designation"] = win["e1"].get("status")
        test = d / "smoke_test.jsonl"
        from wsp.wm import panel2 as P2

        dev = P2.read_open(ROOT / "data/wme/panels/wme-p2", "validation")
        rows = [
            r
            for o in (
                "wikitext_nll",
                "gsm8k_gold_lp",
                "mmlupro_gold_lp",
                "refusal_margin_harmful",
                "refusal_margin_harmless",
                "sycophancy_real",
            )
            for r in sorted(dev[o], key=lambda r: r["id"])[4:8]
        ]
        rows += sorted(
            P2.read_open(ROOT / "data/wme/panels/wme-p2", "dev")["mathcomp_gold_lp"],
            key=lambda r: r["id"],
        )[:4]
        test.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        sealed = co.stage_sealed(prep, win, test)
        out["sealed_states"] = sum(len((v or {}).get("states", {})) for v in sealed.values())
        out["report"] = RG.ProductionGuard(
            st.state,
            "confirmatory",
            st.priced,
            att,
            "smoke:local",
            0.01,
            technical=C.technical,
            fallback_value=SD.FALLBACK["confirmatory"],
        ).report()["charged_gpu_h"]
        out["ok"] = True
    finally:
        (wroot / "STOP").write_text("stop")
        time.sleep(2)
        wp.terminate()
        (d / "smoke_out.json").write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
