"""Bank build on the worker (PREREG_WME §4.1-§4.6, §6.3.3; ``wm.bankrun`` recipes; worker side).

Every state is realised from theta0 (ladders: one action of the cumulative dose; compositions: the roots at dose +1
applied in order by ``EditSession.compose``; inert controls: their registered step bundles), scored on the
``bank`` panel (48 groups per trajectory objective) with the observation tap, then restored and checked.

Per state the payload holds: h, dose, theta-hash, per-item native values in the base unit's item order (sycophancy:
the gap plus its neutral and suggested parts), the per-objective mean tap sketch (8 layers x k 16), R(E) of the
realised state (``wm.c1.field.RCache``) with its unsupported-q flags, the realised displacement norm, the paired mean
WikiText dNLL against the base unit, and the endpoint flag of the registered stop rule (dNLL > 1.0 ends the side).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from wsp.wm import actions as A
from wsp.wm import bankrun as BR
from wsp.wm import ph100 as H

OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)


def control_direction(control_seed: int, hidden: int) -> torch.Tensor:
    """The matched random-direction control (amendment G13): one standard-normal vector from
    ``torch.Generator().manual_seed(control_seed)``, normalised, used at every layer of the partner's block (the
    pilot-2 control convention)."""
    g = torch.Generator().manual_seed(int(control_seed))
    v = torch.randn(hidden, generator=g)
    return v / v.norm()


def artefacts_for(roots_rec: dict, dirs: dict, hidden: int) -> dict:
    """Root artefacts keyed by root id (directions.pt), plus control directions; LoRA factor names mapped to the
    loaded model's prefix later (:func:`lora_factors`)."""
    arte = dict(dirs["artefacts"])
    by = {r["id"]: r for r in roots_rec["roots"]}
    for c in roots_rec.get("controls", []):
        p = by[c["partner"]]
        u = control_direction(c["control_seed"], hidden)
        arte[c["id"]] = {"directions": {L: u for L in p["layers"]}}
    return arte


def rename_lora(factors: dict, prefix: str) -> dict:
    """directions.pt stores LoRA factors under ``model.layers.L...``; the loaded text model may use another prefix."""
    out = {}
    for n, v in factors.items():
        tail = n.split("layers.", 1)[1]
        out[prefix + tail] = v
    return out


def root_actions(root: dict, dose: float, mctx, arte: dict) -> list:
    """One bundle realising ``root`` at signed cumulative ``dose`` from W0 (``bankrun.actions_for``)."""
    fam = root["family"]
    if fam == "lora_tv":
        a = dict(arte[root["id"]])
        a["factors"] = rename_lora(a["factors"], mctx.prefix)
        arte = dict(arte, **{root["id"]: a})
    if fam == "row_scale":
        return [A.row_scale(mctx.embed, root["rows"], factor=BR.NOMINAL["row_scale"], dose=dose)]
    if fam == "noise":
        L, role = root["tensor"]
        return [A.noise(mctx.rm.name(L, role), seed=root["seed"], rho=root["rho"], dose=dose)]
    return BR.actions_for(root, dose, mctx.rm, arte, embed=mctx.embed)


def record_values(res: dict, ids: dict) -> dict:
    out = {}
    for o in OBJECTIVES:
        recs = res[o]
        if [r["id"] for r in recs] != ids[o]:
            raise RuntimeError(f"{o}: items out of base order")
        out[o] = [float(r["value"]) for r in recs]
        if o == "sycophancy_real":
            out["syc_neutral"] = [float(r["neutral"]) for r in recs]
            out["syc_suggested"] = [float(r["suggested"]) for r in recs]
    return out


def mean_sketch(tap) -> dict:
    sk = tap.sketches()
    out = {}
    for o in OBJECTIVES:
        rows = [v for k, v in sk.items() if k.startswith(o + "|")]
        out[o] = torch.stack(rows).mean(0).reshape(-1).tolist() if rows else []
    return out


def dnll(base_vals: list, vals: list) -> float:
    return float(np.mean(np.asarray(vals, float) - np.asarray(base_vals, float)))


class BankWorld:
    """Worker-resident bank context: model, bank items, field + R cache, root artefacts, tap seed."""

    def __init__(
        self, mctx, items: dict, rcache, roots_rec: dict, dirs: dict, tap_seed: int, check=None
    ):
        self.m, self.items, self.rc = mctx, items, rcache
        self.roots = {r["id"]: r for r in roots_rec["roots"]}
        for c in roots_rec.get("controls", []):
            self.roots[c["id"]] = {**self.roots[c["partner"]], **c, "family": "proj_remove_control"}
        self.arte = artefacts_for(roots_rec, dirs, mctx.rm.hidden)
        self.tap_seed = int(tap_seed)
        self.ids = {o: [it["id"] for it in items[o]] for o in OBJECTIVES}
        self.check = check or (lambda: None)

    def _score(self, bundles) -> dict:
        """Apply ``bundles`` in order from theta0, score with the tap, restore (R is added by :meth:`_with_r`)."""
        m = self.m
        m.at_base()
        t0 = time.time()
        try:
            for b in bundles:
                m.sess.apply_bundle(b)
            th = m.theta_hash()
            res, tap = m.evaluate(self.items, tap_seed=self.tap_seed)
            norms = {n: float(m.sess.displacement(n).float().norm()) for n in m.sess.touched}
        finally:
            m.restore()
        self.check()
        return {
            "theta_hash": th,
            "values": record_values(res, self.ids),
            "sketch": mean_sketch(tap),
            "displacement_norms": norms,
            "eval_seconds": time.time() - t0,
        }

    def _with_r(self, states, bundles_list):
        """R(E) of every scored state, computed tensor-major in one batch (``RCache.r_of_many``)."""
        t0 = time.time()
        rows = self.rc.r_of_many([b for b in bundles_list]) if bundles_list else []
        for st, (r, flags) in zip(states, rows, strict=True):
            st["R"] = [float(x) for x in r]
            st["R_flags"] = list(flags)
        self.check()
        return time.time() - t0

    def base(self) -> dict:
        rec = self._score([])
        rec.update(ids=self.ids, h=0, R=[0.0] * self.rc.spec.dim, R_flags=[])
        return rec

    def side(self, root_id: str, doses: list, base_wiki: list) -> dict:
        root = self.roots[root_id]
        states, bl = [], []
        for h, d in enumerate(doses, 1):
            b = [root_actions(root, d, self.m, self.arte)]
            rec = self._score(b)
            rec.update(h=h, dose=float(d))
            rec["dnll"] = dnll(base_wiki, rec["values"]["wikitext_nll"])
            rec["endpoint"] = rec["dnll"] > BR.DNLL_STOP
            states.append(rec)
            bl.append(b)
            if rec["endpoint"]:
                break
        r_s = self._with_r(states, bl)
        return {"root": root_id, "doses": list(doses), "states": states, "r_seconds": r_s}

    def composition(self, order: list, base_wiki: list) -> dict:
        """Prefixes h = 2..4 of ``order`` (each root at dose +1, applied in order); the stop rule in sequence."""
        states, bl = [], []
        for h in range(2, len(order) + 1):
            bundles = [root_actions(self.roots[r], 1.0, self.m, self.arte) for r in order[:h]]
            rec = self._score(bundles)
            rec.update(h=h, prefix=list(order[:h]), dose=1.0)
            rec["dnll"] = dnll(base_wiki, rec["values"]["wikitext_nll"])
            rec["endpoint"] = rec["dnll"] > BR.DNLL_STOP
            states.append(rec)
            bl.append(bundles)
            if rec["endpoint"]:
                break
        r_s = self._with_r(states, bl)
        # the h = 1 prefix is the first root's +1 ladder state (registered: one state, two histories)
        r1 = self.rc.r_of([root_actions(self.roots[order[0]], 1.0, self.m, self.arte)])
        return {
            "order": list(order),
            "states": states,
            "r_seconds": r_s,
            "R_h1": [float(x) for x in r1[0]],
        }

    def inert(self, spec: dict, base_values: dict) -> dict:
        """One registered inert construction (``controls.inert``): every visited state scored; the endpoint's
        theta-hash and outputs are checked bitwise against the base (I1 power-of-two safety checked bytewise)."""
        m = self.m
        steps = inert_steps(spec, m)
        safe = True
        m.at_base()
        try:  # power-of-two safety of every scale step against the weights it is applied to (bytewise)
            for b in steps:
                for act in b:
                    if act.family == "scale" and act.param("factor") in (2.0, 0.5):
                        safe = safe and H.power_of_two_safe(
                            m.sess.params[act.tensor], act.param("factor")
                        )
                m.sess.apply_bundle(b)
        finally:
            m.restore()
        states, bl = [], []
        for h in range(1, len(steps) + 1):
            rec = self._score(steps[:h])
            rec["h"] = h
            states.append(rec)
            bl.append(steps[:h])
        self._with_r(states, bl)
        end = states[-1]
        same = all(end["values"][o] == base_values[o] for o in base_values)
        return {
            "spec": spec,
            "states": states,
            "endpoint_theta_is_base": end["theta_hash"] == m.base_hash,
            "endpoint_outputs_bitwise_base": bool(same),
            "power_of_two_safe": bool(safe),
        }


def inert_specs(manifest: dict) -> list:
    """The registered inert constructions (``controls.inert``): I1 2-step and 4-step, I2 per factor, compensated."""
    c = manifest["controls.inert"]
    out = [
        {"kind": "i1", "shape": "2step", "tensors": c["I1"]["2step"]},
        {"kind": "i1", "shape": "4step", "tensors": c["I1"]["4step"]},
    ]
    out += [{"kind": "i2", "factor": f, "rows": list(c["I2_rows"])} for f in c["I2_factors"]]
    comp = c["compensated"]
    n = len(comp.get("mlp_layers", [])) + len(comp.get("attention_layers", []))
    out += [{"kind": "comp", "index": i} for i in range(n)]
    return out


def inert_steps(spec: dict, m) -> list:
    from wsp.wm import controls as C

    k = spec["kind"]
    if k == "i1":
        return H.round_trip(m.rm, spec["shape"], [tuple(t) for t in spec["tensors"]])
    if k == "i2":
        lo, hi = spec["rows"]
        rows = list(range(int(lo), int(hi)))
        return [[A.row_scale(m.embed, rows, factor=spec["factor"], dose=1.0)]]
    if k == "comp":
        return [C.compensated_pairs(m.rm, **H.COMPENSATED)[spec["index"]]]
    raise KeyError(k)
