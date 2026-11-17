"""Run plan for the v1 9B bank and confirmatory work on the multi-GPU Lambda runner (PREREG_WME v1 §4.2-4.5, §6.5;
RULING-wme-resources §2). PLAN AND ACTION BUILDING ONLY; ``scripts/wme_dp_run.py`` executes it through ``wm.dp``.

* **Ladders** (§4.2): signed cumulative doses +-{0.25, 0.5, 1, 2, 4, 8} x nominal; every state is built from W0 by one
  action of the cumulative dose (bf16 realisation in ``wm.actions``); a dose whose effective parameters repeat a
  visited state is skipped (proj_remove: alpha_eff = 0.5 s capped at 1 on the positive side, so +4 and +8 repeat +2).
  The positive side is walked outward first, then the negative side; on each side the first state with paired mean
  WikiText dNLL > 1.0 on the bank panel is kept, labelled, and ends the side (:func:`walk`). A side is therefore one
  sequential work unit; data parallelism is over sides, compositions and controls (one whole state per GPU, L1).
* **Compositions** (§4.5): each block's 4-step order and its reverse at dose +1; prefixes h = 2..4 are visited, with
  the stop rule applied along each order.
* **Controls**: matched random-direction controls (both sides of their partner's ladder recipe) and the inert
  controls (``wm.ph100``) are separate units.
* **Outputs** (chair note, 17 Nov): every 9B wme-c1 outcome is written under ``data/wme/runs/wme-c1`` (bank and
  confirmatory runs) and ``data/wme/winners/wme-c1`` (winners manifests), the paths track B's 27B read guard blocks.
  No other 9B outcome path exists. Engineering smoke runs on exposed pilot panels write under ``data/wme/runs/smoke``.
* **Freeze gate**: a run on the wme-c1 panel refuses unless ``docs/prereg/wme_freeze.json`` records the manifest
  hash (no confirmatory outcome before the freeze).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wsp.wm import actions as A

LADDER = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DNLL_STOP = 1.0
RUNS_ROOT = "data/wme/runs/wme-c1"
WINNERS_ROOT = "data/wme/winners/wme-c1"
SMOKE_ROOT = "data/wme/runs/smoke"
FREEZE_FILE = "docs/prereg/wme_freeze.json"
NOMINAL = {
    "scale": 1.5,
    "row_scale": 1.5,
    "proj_remove": 0.5,
    "steer": 0.125,
    "lora_tv": 0.5,
    "noise": None,
}


class FreezeGate(PermissionError):
    pass


def _eff(family, dose):
    if family in ("proj_remove", "proj_remove_control"):
        a = 0.5 * dose
        return round(min(a, 1.0), 12)
    return round(dose, 12)


def ladder_sides(family):
    out = []
    for sign in (1.0, -1.0):
        seen, side = set(), []
        for m in LADDER:
            e = _eff(family, sign * m)
            if e in seen:
                continue
            seen.add(e)
            side.append(sign * m)
        out.append(side)
    return out[0], out[1]


def _key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def plan(roots, blocks, controls=()) -> list[dict]:
    """Work units: one per ladder side (root or control), one per composition order."""
    by = {r["id"]: r for r in roots}
    units = []
    for r in list(roots) + [
        {**by[c["partner"]], **c, "family": "proj_remove_control"} for c in controls
    ]:
        fam = r["family"]
        for side, doses in zip(("pos", "neg"), ladder_sides(fam), strict=True):
            states = [{"root": r["id"], "dose": d, "h": h} for h, d in enumerate(doses, 1)]
            units.append(
                {
                    "kind": "side",
                    "root": r["id"],
                    "side": side,
                    "states": states,
                    "key": _key(["side", r["id"], side]),
                }
            )
    for b in blocks:
        for order in (list(b), list(reversed(b))):
            states = [{"prefix": order[:h], "dose": 1.0, "h": h} for h in range(2, len(order) + 1)]
            units.append(
                {
                    "kind": "composition",
                    "roots": order,
                    "states": states,
                    "key": _key(["composition", order]),
                }
            )
    return units


def walk(doses, evaluate):
    """Visit doses outward; ``evaluate(dose)`` -> paired WikiText dNLL; stop after the first state > 1.0."""
    out = []
    for h, d in enumerate(doses, 1):
        dn = float(evaluate(d))
        stop = dn > DNLL_STOP
        out.append({"dose": d, "h": h, "dnll": dn, "endpoint": stop})
        if stop:
            break
    return out


def actions_for(
    root, dose, rm, artefacts, embed="model.language_model.embed_tokens.weight"
) -> list:
    """Actions realising ``root`` at signed cumulative ``dose`` from W0 (one bundle)."""
    fam = root["family"]
    if fam == "scale":
        L, role = root["tensor"]
        return [A.scale(rm.name(L, role), factor=NOMINAL["scale"], dose=dose)]
    if fam == "noise":
        L, role = root["tensor"]
        return [A.noise(rm.name(L, role), seed=root["seed"], rho=root.get("rho", 0.02), dose=dose)]
    if fam == "row_scale":
        return [A.row_scale(embed, root["rows"], factor=NOMINAL["row_scale"], dose=dose)]
    if fam in ("proj_remove", "proj_remove_control"):
        alpha = _eff(fam, dose)
        dirs = artefacts[root["id"]]["directions"]
        return [
            A.proj_remove(rm.name(L, "down"), direction=dirs[L], alpha=alpha, dose=1.0)
            for L in root["layers"]
        ]
    if fam == "steer":
        L = root["layers"][0]
        art = artefacts[root["id"]]
        m = art["mlp_mean"]
        return [
            A.rank_one(
                rm.name(L, "down"),
                u=art["directions"][L],
                v=m / m.norm() ** 2,
                alpha=NOMINAL["steer"] * art["res_norm"],
                dose=dose,
            )
        ]
    if fam == "lora_tv":
        from wsp.wm import lora as LR

        return LR.lowrank_actions(artefacts[root["id"]]["factors"], NOMINAL["lora_tv"] * dose)
    raise KeyError(fam)


def run_dir(root, stage: str, run: str) -> Path:
    if stage not in ("bank", "confirm"):
        raise ValueError(stage)
    return Path(root) / RUNS_ROOT / stage / run


def winners_dir(root) -> Path:
    return Path(root) / WINNERS_ROOT


def smoke_dir(root, run: str) -> Path:
    return Path(root) / SMOKE_ROOT / run


def require_freeze(root, panel: str):
    if panel != "wme-c1":
        return None
    p = Path(root) / FREEZE_FILE
    if not p.exists():
        raise FreezeGate(
            "no wme-c1 outcome before the freeze: docs/prereg/wme_freeze.json is absent"
        )
    rec = json.loads(p.read_text())
    if len(rec.get("manifest_sha256", "")) != 64:
        raise FreezeGate("freeze record lacks the manifest hash")
    return rec
