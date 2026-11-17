#!/usr/bin/env python3
"""Multi-GPU bank runner (RULING-wme-resources §2; PREREG_WME v1 §4.2-4.5). One process per GPU; work units are
ladder sides and composition orders (``wm.bankrun.plan``) claimed through ``wm.dp``; inside a side the doses are
walked outward with the dNLL > 1.0 stop rule; every state's per-item values are stored for the fixed-order fp64
reduction. Outputs: ``data/wme/runs/wme-c1/bank/<run>`` (freeze-gated) or ``data/wme/runs/smoke/<run>``.

    worker --roots <roots.json> --dirs <directions.pt> --panel wme-p2|wme-c1 --split <split> --run <name> --worker gI
    reduce --roots ... --panel ... --split ... --run ... [--compare <other run>]   per-state paired stats, parity
    --smoke   Qwen3.5-0.8B, 4 items per objective (engineering check; exposed pilot items only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import bankrun as BR
from wsp.wm import dp

OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)


def _setup(a):
    rec = json.loads(Path(a.roots).read_text())
    roots = rec["roots"]
    blocks = rec.get("blocks") or []
    plan = BR.plan(roots, blocks, rec.get("controls", []))
    if a.panel == "wme-c1":
        BR.require_freeze(ROOT, "wme-c1")
        out = BR.run_dir(ROOT, "bank", a.run)
    else:
        out = BR.smoke_dir(ROOT, a.run)
    return rec, roots, plan, out


def _units(plan, theta0, panel):
    return [{"key": dp.unit_key(u, theta0, panel), "state_id": u["key"], "plan": u} for u in plan]


def cmd_worker(a):
    import torch

    from wsp.wm import actions as A
    from wsp.wm import hf
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2

    rec, roots, plan, out = _setup(a)
    repo = rec["model"]
    if a.smoke:
        from huggingface_hub import constants

        snaps = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B" / "snapshots"
        hf.TARGETS["Qwen/Qwen3.5-0.8B"] = min(snaps.iterdir()).name
    model, tok, cfg, base_hash, _ = hf.load_target(repo, "cuda")
    names = [n for n, _ in model.named_parameters()]
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in names)
        else "model.layers."
    )
    rm = A.RoleMap.from_config(cfg, prefix=prefix)
    embed = next(n for n in names if n.endswith("embed_tokens.weight"))
    arte = torch.load(a.dirs, weights_only=False)["artefacts"] if a.dirs else {}
    by = {r["id"]: r for r in roots}
    for c in rec.get("controls", []):
        g = torch.Generator().manual_seed(c["control_seed"])
        u = torch.randn(rm.hidden, generator=g)
        by[c["id"]] = {**by[c["partner"]], **c, "family": "proj_remove_control"}
        arte[c["id"]] = {"directions": {L: u / u.norm() for L in by[c["partner"]]["layers"]}}
    panel_root = ROOT / "data/wme/panels" / a.panel
    its = P2.read_open(panel_root, a.split)
    its = {o: sorted(its[o], key=lambda r: r["id"])[: (4 if a.smoke else None)] for o in OBJECTIVES}
    sess = A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)

    def evaluate():
        torch.cuda.synchronize()
        t = time.time()
        res = P2.evaluate(P.LMAdapter(model), tok, its, P.PanelConfig())
        torch.cuda.synchronize()
        return res, time.time() - t

    base, base_s = evaluate()
    base_wiki = [r["value"] for r in base["wikitext_nll"]]

    def records(res):
        return [
            {"objective": o, "group": r["id"], "item_id": r["id"], "value": float(r["value"])}
            for o in OBJECTIVES
            for r in res[o]
        ]

    def visit(bundles):
        if sess.theta_hash() != base_hash:
            raise RuntimeError("not at base")
        for b in bundles:
            sess.apply_bundle(b)
        th = sess.theta_hash()
        res, s = evaluate()
        sess.restore()
        if sess.theta_hash() != base_hash:
            raise RuntimeError("theta drift")
        # nats per token increase of the paired mean NLL: wikitext value is -NLL per token
        dn = sum(
            b - e for e, b in zip([r["value"] for r in res["wikitext_nll"]], base_wiki, strict=True)
        ) / len(base_wiki)
        return {"theta_hash": th, "dnll": dn, "records": records(res), "seconds": s}

    def fn(u):
        p = u["plan"]
        states = []
        if p["kind"] == "side":
            r = by[p["root"]]
            for st in p["states"]:
                v = visit([BR.actions_for(r, st["dose"], rm, arte, embed)])
                states.append(dict(st, **v, endpoint=v["dnll"] > BR.DNLL_STOP))
                if v["dnll"] > BR.DNLL_STOP:
                    break
        else:
            for st in p["states"]:
                bundles = [BR.actions_for(by[rid], 1.0, rm, arte, embed) for rid in st["prefix"]]
                v = visit(bundles)
                states.append(dict(st, **v, endpoint=v["dnll"] > BR.DNLL_STOP))
                if v["dnll"] > BR.DNLL_STOP:
                    break
        return {"records": [], "states": states, "worker": a.worker}

    out.mkdir(parents=True, exist_ok=True)
    (out / f"base_{a.worker}.json").write_text(
        json.dumps({"theta0": base_hash, "records": records(base), "seconds": base_s})
    )
    units = _units(plan, base_hash, f"{a.panel}:{a.split}")
    n = dp.run_worker(out, units, fn, a.worker, theta_after=base_hash, ttl_s=1800)
    (out / f"done_{a.worker}.json").write_text(json.dumps({"computed": n}))


def _reduce_run(out, plan):
    from wsp.wm.harness import UnitStore

    base_file = sorted(out.glob("base_*.json"))[0]
    b = json.loads(base_file.read_text())
    units = _units(plan, b["theta0"], None)
    store = UnitStore(out / "units")
    base = dp.reduce_fixed_order(b["records"])
    stats = {}
    for p in (out / "units").glob("*/*.json"):
        payload = json.loads(p.read_text())["payload"]
        for st in payload["states"]:
            sid = f"{st.get('root', '>'.join(st.get('prefix', [])))}@{st['dose']}"
            stats[sid] = {
                "theta_hash": st["theta_hash"],
                "dnll": st["dnll"],
                "endpoint": st["endpoint"],
                "paired": dp.paired_stats(base, dp.reduce_fixed_order(st["records"])),
            }
    del units, store
    return stats


def cmd_reduce(a):
    rec, roots, plan, out = _setup(a)
    stats = _reduce_run(out, plan)
    rep = {
        "run": a.run,
        "n_states": len(stats),
        "n_endpoints": sum(s["endpoint"] for s in stats.values()),
    }
    if a.compare:
        other = _reduce_run(out.parent / a.compare, plan)
        rep["parity"] = dp.parity(
            {k: v["paired"] for k, v in stats.items()}, {k: v["paired"] for k, v in other.items()}
        )
        rep["theta_hashes_equal"] = all(
            stats[k]["theta_hash"] == other.get(k, {}).get("theta_hash") for k in stats
        )
    (out / "reduce.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worker", "reduce"])
    ap.add_argument("--roots", required=True)
    ap.add_argument("--dirs", default=None)
    ap.add_argument("--panel", default="wme-p2")
    ap.add_argument("--split", default="construction")
    ap.add_argument("--run", required=True)
    ap.add_argument("--worker", default="g0")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"worker": cmd_worker, "reduce": cmd_reduce}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
