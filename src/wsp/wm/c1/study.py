"""The wme-c1 study on the controller (PREREG_WME §II.18 order of operations; §II.28 serialized path).

One ``Study`` owns one state directory with the single study log. A phase constructs a fresh ``ProductionGuard``
(verified fresh replay, new owner epoch; the worker's fence moves to that epoch), then runs its units through
``harness.run_units`` in canary-bracketed chunks (``ctl.run_bracketed``) and returns only ELIGIBLE payloads (or the
registered fallback). Phase prices (the 1.25x pause) and fallbacks are the registered/declared values below.

Order: bank (field groups, vocabulary rows, base, ladder sides, compositions, inert) -> confirmatory (library, dev
base, fit, the 11 E2 arms x 24 episodes) -> B_par from ledger seconds -> sbon (24 episodes) -> E1 designation and
the hashed winners manifest -> confirmatory (the single sealed test read; E1 generation) -> ste.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

from wsp.wm import e2 as E2
from wsp.wm import harness as H
from wsp.wm import prereg_v2 as PV2
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1.schema import ARMS_PRIMARY, ARMS_SECONDARY

PRICE = {
    "bank": 4.1268,  # the bound 72-root / 12-block graph (amendment G1; powersim_v2b.price, oracle_s 0)
    "confirmatory": 53.66619236655922,  # registered (prereg_v2.pricing.v2c.confirmatory_gpu_h)
    "sbon": None,  # B_par x 24 x t_bundle / 3600 at the realised B_par (fixed before the arm runs)
    "ste": 1.844836772316389,  # registered (prereg_v2.pricing.s_te.gpu_h)
}
EST_UNIT_GPU_H = {"bank": 0.35, "confirmatory": 0.75, "sbon": 0.75, "ste": 1.25}
FALLBACK = {
    "bank": {
        "rule": "a bank unit that exhausts its 3 attempts is unlabelled: excluded from training and reported (amendment G17)"
    },
    "confirmatory": {
        "rule": "an E2 arm-episode that exhausts its 3 attempts takes the base winner (Part I §6.5.5; §II.22.6); "
        "any other confirmatory unit is missing and its claim is reported as an engineering abort (amendment G17)"
    },
    "sbon": {
        "rule": "an S-BoN episode that exhausts its 3 attempts takes the base winner (§II.22.6)"
    },
    "ste": {"rule": "a missing S-TE unit: S-TE not completed - engineering abort (amendment G17)"},
}
from wsp.wm.arms.sbon import T_BUNDLE_S

N_EP = 24
B = 48
POOL_SIZE = 64
READER_TOKEN_CAP = 140000


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parity_budget(bank_gpu_s: float, field_gpu_s: float) -> float:
    """§II.5: B_par = B + (H_bank 3600 + T_field,R1) / n_ep / t_bundle with H_bank the measured bank-build seconds
    (bank units other than the field) and T_field the measured R1 field seconds (amendment G7)."""
    return PV2.parity_budget(B, bank_gpu_s / 3600.0, field_gpu_s, N_EP, T_BUNDLE_S)


def sbon_price(b_par: float) -> float:
    return b_par * N_EP * T_BUNDLE_S / 3600.0


class Study:
    def __init__(
        self,
        state_dir,
        runs_dir,
        attestation: RG.LaunchAttestation,
        theta0: str,
        panel: str,
        log=print,
    ):
        self.state = Path(state_dir)
        self.runs = Path(runs_dir)
        self.att = attestation
        self.theta0, self.panel = theta0, panel
        self.log = log
        self.priced = {k: v for k, v in PRICE.items() if v is not None}

    def set_sbon_price(self, b_par: float):
        self.priced["sbon"] = sbon_price(b_par)

    def guard(self, phase: str, instance: str, transition=None) -> RG.ProductionGuard:
        return RG.ProductionGuard(
            self.state,
            phase,
            dict(self.priced),
            self.att,
            instance,
            est_unit_gpu_h=EST_UNIT_GPU_H[phase],
            technical=C.technical,
            fallback_value=FALLBACK[phase],
            transition=transition,
        )

    def store(self, phase: str) -> H.UnitStore:
        return H.UnitStore(self.runs / phase / "units")

    def key(self, recipe) -> str:
        return H.unit_key(recipe, self.theta0, self.panel)

    # -------------------------------------------------------------- generic phase runner
    def run(
        self,
        guard,
        remote,
        transport,
        units,
        chunk=8,
        max_chunk_s=7200.0,
        uploads=None,
        canary_args=None,
        fn=None,
    ) -> dict:
        """``units`` [(recipe, label)] with recipe {"kind", "args", "uploads": {name: local path}}. Uploads are
        content-addressed on the worker (``uploads/<sha256>``) and their worker paths replace the names in args."""
        store = self.store(guard.phase)

        def default_fn(recipe):
            args = dict(recipe["args"])
            for name, local in (recipe.get("uploads") or {}).items():
                args[name] = transport.push(local)
            payload, _dir = remote.call(
                recipe["kind"],
                guard.epoch,
                guard.instance,
                self.key(recipe),
                recipe.get("label", recipe["kind"]),
                args,
            )
            return payload

        canary = lambda: C.run_canary(guard, remote, canary_args)
        return C.run_bracketed(
            guard,
            store,
            units,
            fn or default_fn,
            self.theta0,
            self.panel,
            canary,
            chunk=chunk,
            max_chunk_s=max_chunk_s,
        )

    def result_files(self, remote, payload_key) -> Path | None:
        return None


# ------------------------------------------------------------------ unit lists


def bank_units(manifest: dict, roots_rec: dict, field_groups=range(8)) -> list:

    units = [
        ({"kind": "field_group", "args": {"group": g}, "label": f"field:g{g}"}, f"field:g{g}")
        for g in field_groups
    ]
    units.append(({"kind": "field_vocab", "args": {}, "label": "field:vocab"}, "field:vocab"))
    return units


def bank_state_units(manifest: dict, roots_rec: dict, base_payload: dict) -> list:
    from wsp.wm import bankrun as BR
    from wsp.wm.c1 import bankops as BO

    base_wiki = base_payload["values"]["wikitext_nll"]
    out = []
    roots = list(roots_rec["roots"])
    ctrl = [
        {**next(r for r in roots if r["id"] == c["partner"]), **c, "family": "proj_remove_control"}
        for c in roots_rec.get("controls", [])
    ]
    for r in roots + ctrl:
        pos, neg = BR.ladder_sides(r["family"])
        for side, doses in (("pos", pos), ("neg", neg)):
            lab = f"side:{r['id']}:{side}"
            out.append(
                (
                    {
                        "kind": "bank_side",
                        "args": {"root": r["id"], "doses": list(doses), "base_wikitext": base_wiki},
                        "label": lab,
                    },
                    lab,
                )
            )
    for i, b in enumerate(manifest["bank.blocks"]["blocks"]):
        for tag, order in (("fwd", list(b)), ("rev", list(reversed(b)))):
            lab = f"comp:{i:02d}:{tag}"
            out.append(
                (
                    {
                        "kind": "bank_comp",
                        "args": {"order": order, "base_wikitext": base_wiki},
                        "label": lab,
                    },
                    lab,
                )
            )
    base_values = {o: base_payload["values"][o] for o in base_payload["values"]}
    for j, spec in enumerate(BO.inert_specs(manifest)):
        lab = f"inert:{j:02d}:{spec['kind']}"
        out.append(
            (
                {
                    "kind": "bank_inert",
                    "args": {"spec": spec, "base_values": base_values},
                    "label": lab,
                },
                lab,
            )
        )
    return out


def e2_inputs(lib_file, dev_base_file, fit_file=None) -> dict:
    a = {"library_sha256": sha_file(lib_file), "dev_base_sha256": sha_file(dev_base_file)}
    up = {"library_file": lib_file, "dev_base_file": dev_base_file}
    if fit_file is not None:
        a["fit_sha256"] = sha_file(fit_file)
        up["fit_file"] = fit_file
    return a, up


def e2_units(inputs, uploads, episodes=range(N_EP), arms=None) -> list:
    arms = arms or [a for a in ARMS_PRIMARY + ARMS_SECONDARY]
    out = []
    for e in episodes:
        for arm in arms:
            lab = f"e2:{arm}:{e:02d}"
            args = dict(inputs, arm=arm, episode=int(e), B=B, pool_size=POOL_SIZE)
            out.append(
                ({"kind": "e2_arm", "args": args, "uploads": dict(uploads), "label": lab}, lab)
            )
    return out


def sbon_units(inputs, uploads, budget: float, episodes=range(N_EP)) -> list:
    return [
        (
            {
                "kind": "sbon_arm",
                "args": dict(inputs, episode=int(e), budget=float(budget)),
                "uploads": dict(uploads),
                "label": f"sbon:{e:02d}",
            },
            f"sbon:{e:02d}",
        )
        for e in episodes
    ]


def winner_edit(arm: str, payload: dict) -> dict:
    """The JSON edit spec of an arm-episode winner for the sealed read (``serve_e2.edit_bundles``)."""
    w = payload.get("winner") if payload else None
    if not w:
        return {"kind": "base"}
    c = w["candidate"]
    if arm == "bon_steering":
        return {"kind": "steer", "layer": int(c[1]), "contrast": c[2], "shift": float(c[3])}
    if arm == "lora_sft":
        return {"kind": "lora_sft", "step": int(c[1]), "sha256": payload["winner_edit"]["sha256"]}
    return {"kind": "pids", "pids": list(c)}


def designate(e2_payloads: dict) -> dict:
    """§6.5.6 over the WM-arm episode winners (``wm.e2.designate_e1``: largest dev LCB among non-base winners; ties
    by smaller edit norm, then lower episode index). A fallback (exhausted) episode is the base."""
    wm = []
    for lab, p in e2_payloads.items():
        if not lab.startswith("e2:wm_e:"):
            continue
        ep = int(lab.rsplit(":", 1)[1])
        w = None if (p is None or p.get("fallback")) else p.get("winner")
        wm.append(
            {
                "episode": ep,
                "winner": bool(w),
                "candidate": list(w["candidate"]) if w else [],
                "theta_hash": w.get("theta_hash") if w else None,
                "dev_lcb": float(w["lcb"]) if w else 0.0,
                "rel_norm": float(w["rel_norm"]) if w else 0.0,
            }
        )
    return E2.designate_e1(sorted(wm, key=lambda r: r["episode"]))


def reader_unit_fn(study, guard, remote, transport, recipe, client_factory, log=print):
    """A controller-side LLM-reader arm-episode (wsr_reader / fugu_team): the pool loop runs here; every dev
    evaluation, theta-hash and R0 description is a sub-request to the worker under the same unit attempt."""
    from wsp.wm import plan as PL
    from wsp.wm.access import Artefact, R0Guard, Tier
    from wsp.wm.arms import llm as L
    from wsp.wm.arms import pool as PO
    from wsp.wm.arms import readers as RD
    from wsp.wm.arms import spec as AS
    from wsp.wm.c1 import e2arms as EA

    a = dict(recipe["args"])
    for name, local in (recipe.get("uploads") or {}).items():
        a[name] = transport.push(local)
    arm, ep = a["arm"], EA.episode(a["episode"])
    ukey = study.key(recipe)
    n = {"i": 0}

    def call(kind, extra):
        n["i"] += 1
        p, _ = remote.call(
            kind,
            guard.epoch,
            guard.instance,
            hashlib.sha256(f"{ukey}:{n['i']}".encode()).hexdigest(),
            f"{arm}:{ep['index']}:{kind}",
            dict(a, **extra),
        )
        return p

    meta = call("describe", {"subsample": list(ep["subsample"])})["meta"]
    feats: dict = {}
    orig = PO.draw_pool

    def draw_hook(*args, **kw):
        pool = orig(*args, **kw)
        need = [list(c) for c in pool if tuple(c) not in feats]
        if need:
            got = call("describe", {"candidates": need})["features"]
            for c, f in zip(need, got, strict=True):
                feats[tuple(c)] = f
        return pool

    res = {
        "episode_library": list(ep["subsample"]),
        "episode_pool": True,
        "r0_candidate_description": Artefact(
            "r0_candidate_description", study.theta0, Tier.R0, meta
        ),
    }
    ledger = PL.BudgetLedger(B, [arm])
    stores = AS.StoreRegistry()
    ctx = AS.ArmContext(
        AS.ARM_SPECS[arm],
        ledger,
        ep["index"],
        lambda c: call("dev_eval", {"pids": list(c)}),
        lambda c: call("theta_of", {"pids": list(c)})["theta_hash"],
        res,
        stores,
        R0Guard(study.theta0),
    )
    tl = L.TokenLedger(READER_TOKEN_CAP)
    task = RD.ReaderTask()
    if arm == "wsr_reader":
        ranker = RD.WSRReader(client_factory("reader"), tl, task)
    else:
        ranker = RD.FuguTeam({r: client_factory(r) for r in RD.FuguTeam.ROLES}, tl, task)
    PO.draw_pool = draw_hook
    try:
        out = PO.run_pool_episode(
            ranker,
            ctx,
            B,
            POOL_SIZE,
            int(ep["seeds"]["planner"]),
            features=lambda c: feats[tuple(c)],
        )
    finally:
        PO.draw_pool = orig
    hist = ctx.history()
    w = out["winner"]
    return {
        "arm": arm,
        "episode": ep["index"],
        "winner": None if w is None else EA._strip(w),
        "evaluated": [EA._strip(h) for h in hist],
        "ledger": ledger.report(),
        "rounds": out["rounds"],
        "failures": out["failures"],
        "tokens": {
            "spent": tl.spent,
            "cap": tl.cap,
            "records": [
                {
                    k: r.get(k)
                    for k in ("role", "total", "truncated", "error", "model", "usage_source")
                }
                for r in tl.records
            ],
        },
    }


def ceil_units(x: float) -> int:
    return int(math.ceil(x))


def now() -> float:
    return time.time()
