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
