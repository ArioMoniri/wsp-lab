"""Bundle writers (``wsp.wm.c1.schema``) from eligible unit payloads only. CPU, controller side."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def graph(manifest: dict, roots_rec: dict, blocks=None, calibration=None) -> dict:
    """The bound component graph (``scripts/wme_manifest.py blocks_and_graph`` semantics): each block with its 4
    roots, every other root a singleton; a negative control belongs to its partner's component."""
    blocks = manifest["bank.blocks"]["blocks"] if blocks is None else blocks
    in_block = {r for b in blocks for r in b}
    comps = [
        {"id": f"block:{i:02d}", "stratum": "block", "roots": list(b)} for i, b in enumerate(blocks)
    ]
    comps += [
        {"id": f"single:{r['id']}", "stratum": r["family"], "roots": [r["id"]]}
        for r in roots_rec["roots"]
        if r["id"] not in in_block
    ]
    cal = list(manifest["bank.calibration_components"] if calibration is None else calibration)
    return {"components": comps, "calibration": cal}


def graph_hash(g: dict) -> str:
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()


def component_of(g: dict) -> dict:
    out = {}
    for c in g["components"]:
        for r in c["roots"]:
            out[r] = c["id"]
    return out


def bank_json(stage: dict, manifest: dict, roots_rec: dict, blocks=None, calibration=None) -> dict:
    g = graph(manifest, roots_rec, blocks, calibration)
    comp = component_of(g)
    fam = {r["id"]: r["family"] for r in roots_rec["roots"]}
    partner = {c["id"]: c["partner"] for c in roots_rec.get("controls", [])}
    base = stage["pre"]["bank:base"]
    states, inert, fallbacks = [], [], []
    for lab, p in sorted(stage["states"].items()):
        if p is None or p.get("fallback"):
            fallbacks.append(lab)
            continue
        if lab.startswith("side:"):
            _, root, side = lab.split(":", 2) if lab.count(":") == 2 else (None, None, None)
            root, side = lab[len("side:") :].rsplit(":", 1)
            is_ctrl = root in partner
            hist = []
            for st in p["states"]:
                sid = f"{lab}:{st['h']}"
                hist.append(sid)
                states.append(
                    {
                        "state_id": sid,
                        "kind": "control" if is_ctrl else "ladder",
                        "component": comp[partner.get(root, root)],
                        "family": "proj_remove_control" if is_ctrl else fam[root],
                        "roots": [root],
                        "side": side,
                        "h": st["h"],
                        "dose": st["dose"],
                        "prefix": None,
                        "theta_hash": st["theta_hash"],
                        "values": st["values"],
                        "sketch": st["sketch"],
                        "R": st["R"],
                        "R_flags": st["R_flags"],
                        "dnll": st["dnll"],
                        "endpoint": st["endpoint"],
                        "history": list(hist),
                    }
                )
        elif lab.startswith("comp:"):
            order = p["order"]
            hist = []
            for st in p["states"]:
                sid = f"{lab}:{st['h']}"
                hist.append(sid)
                states.append(
                    {
                        "state_id": sid,
                        "kind": "composition",
                        "order_id": lab,
                        "component": comp[order[0]],
                        "family": "block",
                        "roots": list(order),
                        "side": None,
                        "h": st["h"],
                        "dose": 1.0,
                        "prefix": st["prefix"],
                        "theta_hash": st["theta_hash"],
                        "values": st["values"],
                        "sketch": st["sketch"],
                        "R": st["R"],
                        "R_flags": st["R_flags"],
                        "dnll": st["dnll"],
                        "endpoint": st["endpoint"],
                        "history": list(hist),
                    }
                )
        elif lab.startswith("inert:"):
            inert.append({"label": lab, **p})
    return {
        "items": {"ids": base.get("ids", {})},
        "base": {
            "values": base["values"],
            "sketch": base["sketch"],
            "theta_hash": base["theta_hash"],
        },
        "graph": dict(
            g, hash=graph_hash({"components": g["components"], "calibration": g["calibration"]})
        ),
        "states": states,
        "inert": inert,
        "fallbacks": fallbacks,
        "field": {k: v for k, v in stage["pre"].items() if k.startswith("field:")},
    }


def strata(bank: dict) -> dict:
    return {c["id"]: c["stratum"] for c in bank["graph"]["components"]}


def write(path, obj) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, sort_keys=True))
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ the rest of the bundle (schema.py)


def _fb(p):
    return p is None or bool(p.get("fallback"))


def e2_json(e2_stage: dict, sbon_stage: dict) -> dict:
    from wsp.wm.c1 import e2arms as EA
    from wsp.wm.c1 import study as SD

    arms: dict = {}
    for lab, p in sorted(e2_stage.items()):
        _, arm, ep = lab.split(":")
        arms.setdefault(arm, {})[str(int(ep))] = _arm_rec(p)
    for lab, p in sorted((sbon_stage.get("results") or {}).items()):
        arms.setdefault("sbon", {})[str(int(lab.split(":")[1]))] = _arm_rec(p)
    eps = []
    for i in range(SD.N_EP):
        e = EA.episode(i)
        eps.append({k: e[k] for k in ("index", "family", "rep", "seeds", "subsample")})
    bp = sbon_stage.get("bpar") or {}
    return {"B": SD.B, "episodes": eps, "arms": arms, "sbon": bp}


def _arm_rec(p):
    if _fb(p):
        return {
            "winner": None,
            "fallback": True,
            "usage": None,
            "tokens": None,
            "evaluated": [],
            "failures": None,
        }
    led = p.get("ledger") or {}
    return {
        "winner": p.get("winner"),
        "fallback": False,
        "usage": led,
        "tokens": (p.get("tokens") or {}).get("spent")
        if isinstance(p.get("tokens"), dict)
        else p.get("tokens"),
        "evaluated": p.get("evaluated", []),
        "failures": p.get("failures"),
        "index_to_pid": p.get("index_to_pid"),
        "winner_edit": p.get("winner_edit"),
        "seconds": p.get("seconds"),
    }


def fits_json(prep: dict) -> dict:
    f = prep["fit"]
    return {
        k: f.get(k)
        for k in (
            "mode",
            "coverage",
            "q_hat",
            "shrink",
            "c2i",
            "f_k",
            "train_components",
            "n_train",
            "n_cov_units",
            "n_scores",
            "sha256",
        )
    }


def _values(records: dict) -> dict:
    out = {}
    for o, recs in records.items():
        out[o] = [float(r["value"]) for r in recs]
        if o == "sycophancy_real":
            out["syc_neutral"] = [float(r["neutral"]) for r in recs]
            out["syc_suggested"] = [float(r["suggested"]) for r in recs]
    return out


def sealed_json(sealed_stage: dict, winners_manifest: dict) -> dict:
    ids = groups = None
    states, base = {}, None
    by_key = {}
    for r in winners_manifest["winners"]:
        if r["fallback"] or r["edit"]["kind"] == "base":
            continue
        by_key[r["theta_hash"] or json.dumps(r["edit"], sort_keys=True)] = r
    for _lab, p in sorted(sealed_stage.items()):
        if _fb(p):
            continue
        ids, groups = ids or p["ids"], groups or p["groups"]
        for key, st in p["states"].items():
            rec = {"values": _values(st["records"]), "theta_hash": st["theta_hash"]}
            if key == "base":
                base = rec
            else:
                rec["edit"] = by_key.get(key, {}).get("edit")
                states[st["theta_hash"]] = rec
    return {"ids": ids, "groups": groups, "base": base, "states": states}


def ledger_json(state_dir, deadman_state: dict, amendments: list) -> dict:
    from wsp.wm import runguard as RG

    led = RG.CombinedLedger(Path(state_dir) / RG.LOG_NAME, {"bank": 1})
    recs = [json.loads(x) for x in (Path(state_dir) / RG.LOG_NAME).read_text().splitlines()]
    phases = sorted({r["phase"] for r in recs if r.get("type") == "unit_time"})
    inst = []
    for iid, r in (deadman_state.get("instances") or {}).items():
        inst.append(
            {
                "id": iid,
                "name": r.get("name"),
                "type": r.get("type"),
                "launched_at": r.get("launched_at"),
                "terminated_at": r.get("confirmed_terminated_at"),
                "terminated_confirmed": r.get("confirmed_terminated_at") is not None,
            }
        )
    return {
        "gpu_h": {ph: led.unit_gpu_h(ph) for ph in phases},
        "unit_gpu_h": led.unit_gpu_h(),
        "instance_gpu_h": led.instance_gpu_h(),
        "charged_gpu_h": led.gpu_h(),
        "usd": led.usd(),
        "instances": inst,
        "relaunches": [r for r in recs if r.get("type") == "relaunch"],
        "repairs": [r for r in recs if r.get("type") == "repair"],
        "amendments": amendments,
        "log_records": len(recs),
    }


def finish(out_dir, files: dict) -> str:
    """Write bundle.json with every member's sha256."""
    d = Path(out_dir)
    idx = {"files": {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in sorted(files)}}
    (d / "bundle.json").write_text(json.dumps(idx, indent=1, sort_keys=True))
    return hashlib.sha256((d / "bundle.json").read_bytes()).hexdigest()
