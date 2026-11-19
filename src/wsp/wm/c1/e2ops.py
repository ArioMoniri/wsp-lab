"""E2 on the worker: the declared library at theta0, the real dev world, and the arm-episode runners
(PREREG_WME §6.5, §6.8.3, §7.1, §II.5, §II.22.2, §II.22.7). Worker side.

* **Library** (:func:`build_library`, §6.5.2): steer / proj_remove directions per contrast (``wm.e2.CONTRASTS``) as the
  difference of mean residuals (pilot-1 ``residual_means``) of the contrast's positive and negative prompt sets, at every
  layer (the primitive grid uses ``wm.e2.LAYERS``; the best-of-N steering arm uses all layers), the pilot-2 steer
  compile (``mlp_mean``, ``res_norm`` over the pilot-2 mixed set), and the 4 sycophancy LoRAs (rank 8, pilot-2 settings,
  suggested-row SFT on construction/e2, seeds ``e2.library.lora_seeds``). Prompt sets: construction/e2 (64 sycophancy
  groups; ``syc_suggested``, ``syc_neutral``) and the designated e2 sources (amendment G10) for gsm, wiki, mathcomp and
  harmless, formatted as pilot 2 (``scripts/wme_pilot2.py contrast_sets``).
* **Dev world** (:class:`DevWorld`): applies a candidate's primitives in order (``EditSession`` compose), scores the
  dev panel (7 objectives x 64 groups), restores, and returns the dev record of ``wm.c1.schema``: oriented primary
  gain and its item-group bootstrap-t LCB, the 13 guard bounds on both scales (item-group bootstrap-t, the seeds of
  ``arms.spec.guard_bounds_from_items``) computed in a process pool (same function, same seeds, same values), the
  guard pass (intersection-union), the realised relative norm, the theta-hash and seconds.
* **Arm runners** (``run_*``): every arm through the registered library code (``adapter9b.run_episode_v1``,
  ``arms.direct``, ``arms.pool``, ``arms.sbon``) on this world, with one ``plan.BudgetLedger`` per (arm, episode).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from wsp.wm import actions as A
from wsp.wm import e2 as E2
from wsp.wm.arms import spec as AS

OBJECTIVE = "sycophancy_real"
DEV_OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
    "mathcomp_gold_lp",
)
MIXED = (("harmless", 16), ("gsm", 16), ("syc_neutral", 16))
RES_NORM_N = 24
TWO_STAGE = {"screen_groups": 4, "n_full": 16}
ROOT = Path(__file__).resolve().parents[4]


def _pilot1():
    spec = importlib.util.spec_from_file_location("wme_pilot", ROOT / "scripts/wme_pilot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lora_order_seed(k: int) -> int:
    """Data-order seed of library LoRA k (amendment: pilot-2's 'lora-order:<objective>' convention per LoRA)."""
    return int(hashlib.sha256(f"wme-e2:lora-order:{OBJECTIVE}:{k}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ library


def contrast_prompts(tok, e2_items, sources) -> dict:
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2

    chat = lambda u: P.chat_prompt(tok, u)
    return {
        "harmless": [chat(r["prompt"]) for r in sources["harmless"]],
        "syc_suggested": [chat(r["suggested_prompt"]) for r in e2_items],
        "syc_neutral": [chat(r["neutral_prompt"]) for r in e2_items],
        "mathcomp": [chat(r["problem"] + P2.MATH_INSTRUCTION) for r in sources["mathcomp"]],
        "gsm": [chat(P.gsm8k_user(r)) for r in sources["gsm"]],
        "wiki": [r["text"][:1500] for r in sources["wiki"]],
    }


def build_library(
    m, e2_items, sources, lora_seeds, out_dir, c1_manifest, check=lambda: None, log=None
) -> dict:
    from wsp.wm import lora as LR
    from wsp.wm import panel as P

    W1 = _pilot1()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sets = contrast_prompts(m.tok, e2_items, sources)
    layers = list(range(m.rm.n_layers))
    need = sorted({s for c in E2.CONTRASTS.values() for s in c})
    means = {}
    for s in need:
        res, _mlp, _ = W1.residual_means(m.model, m.tok, sets[s], layers)
        means[s] = {L: res[L].float().cpu() for L in layers}
        check()
    dirs = {}
    for name, (pos, neg) in E2.CONTRASTS.items():
        dirs[name] = {}
        for L in layers:
            d = means[pos][L] - means[neg][L]
            dirs[name][L] = d / d.norm()
    mixed = [p for s, n in MIXED for p in sets[s][:n]]
    _, mlp_mixed, _ = W1.residual_means(m.model, m.tok, mixed, layers)
    mlp_mean = {L: mlp_mixed[L].float().cpu() for L in layers}
    res_norm = {L: float(W1.residual_norm(m.model, m.tok, mixed[:RES_NORM_N], L)) for L in layers}
    check()
    t_dirs = time.time() - t0
    loras, logs = {}, {}
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    rows = LR.sft_rows(OBJECTIVE, e2_items, m.tok, P.PanelConfig(), c1_manifest)
    st = E2.LORA_SETTINGS
    for k in range(E2.N_LORAS):
        name = f"{E2.LORA_TASK_TAG[OBJECTIVE]}_lora{k}"
        seed = int(lora_seeds[k])
        if seed != E2._seed(f"lora:{OBJECTIVE}:{k}"):
            raise ValueError(f"LoRA {k}: manifest seed {seed} != library recipe seed")
        m.at_base()
        ad = LR.LoRA(
            m.model, LR.writer_targets(m.rm), rank=st["rank"], alpha=st["alpha"], seed=seed
        )
        lg = LR.train(
            m.model,
            ad,
            rows,
            pad_id=pad,
            lr=st["lr"],
            epochs=st["epochs"],
            batch_rows=st["batch_rows"],
            seed=lora_order_seed(k),
            device="cuda",
            max_seconds=1e9,
            warmup=st["warmup"],
            checkpointing=True,
        )
        loras[name] = {
            n: (b.detach().cpu().clone(), a.detach().cpu().clone(), s)
            for n, (b, a, s) in ad.factors().items()
        }
        logs[name] = {k2: lg[k2] for k2 in ("steps", "seconds", "n_rows") if k2 in lg}
        del ad
        torch.cuda.empty_cache()
        m.at_base()
        check()
    lib = {
        "dirs": dirs,
        "mlp_mean": mlp_mean,
        "res_norm": res_norm,
        "loras": loras,
        "layers": layers,
    }
    path = out_dir / "library.pt"
    torch.save(lib, path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "lora_logs": logs,
        "n_rows": len(rows),
        "dirs_seconds": t_dirs,
        "seconds": time.time() - t0,
        "recipe_digest": E2.library_digest(OBJECTIVE),
    }


class Library:
    def __init__(self, path, rm):
        self.lib = torch.load(path, map_location="cpu", weights_only=False)
        self.rm = rm
        self.index = E2.library_index(OBJECTIVE)

    def steer(self, L, contrast, shift):
        d = self.lib["dirs"][contrast][L]
        mm = self.lib["mlp_mean"][L]
        return A.rank_one(
            self.rm.name(L, "down"),
            u=d,
            v=mm / mm.norm() ** 2,
            alpha=shift * self.lib["res_norm"][L],
            dose=1.0,
        )

    def actions(self, pid) -> list:
        """One bundle for a library primitive (its registered dose)."""
        from wsp.wm import lora as LR

        p = self.index[pid]
        if p["family"] == "steer":
            return [self.steer(p["layer"], p["contrast"], p["dose"])]
        if p["family"] == "proj_remove":
            u = self.lib["dirs"][p["contrast"]][p["layer"]]
            return [
                A.proj_remove(
                    self.rm.name(p["layer"], "down"), direction=u, alpha=p["dose"], dose=1.0
                )
            ]
        if p["family"] == "lora_tv":
            lo, hi = p["block"]
            fac = self.lib["loras"][p["lora"]]
            sub = {}
            for n, v in fac.items():
                L = self.rm.layer_of(n)
                if L is None:
                    raise KeyError(f"{n}: no layer")
                if lo <= L <= hi:
                    sub[n] = v
            return LR.lowrank_actions(sub, p["dose"])
        raise KeyError(pid)

    def bundles(self, pids) -> list:
        return [self.actions(p) for p in pids]


# ------------------------------------------------------------------ dev records


def _boot(args):
    d, groups, seed = args
    return E2.bootstrap_t_bounds(d, groups, n_boot=AS.N_BOOT_T, seed=seed)


def item_changes(base: dict, edit: dict) -> dict:
    """Per-item paired changes on both scales for every objective and the syc_neutral guard.
    ``base``/``edit`` are panel2.evaluate outputs ({obj: [record]})."""
    from wsp.wm import panel2 as P2

    out = {}
    for o, recs in edit.items():
        b = base[o]
        if [r["id"] for r in b] != [r["id"] for r in recs]:
            raise RuntimeError(f"{o}: item order differs from the base")
        nat = [e["value"] - x["value"] for e, x in zip(recs, b, strict=True)]
        pp = None
        if P2.pp_value(o, recs[0]) is not None:
            pp = [P2.pp_value(o, e) - P2.pp_value(o, x) for e, x in zip(recs, b, strict=True)]
        out[o] = {"native": nat, "pp": pp}
        if o == OBJECTIVE:
            out["syc_neutral"] = {
                "native": [e["neutral"] - x["neutral"] for e, x in zip(recs, b, strict=True)],
                "pp": [
                    math.exp(e["neutral"]) - math.exp(x["neutral"])
                    for e, x in zip(recs, b, strict=True)
                ],
            }
    return out


def bound_jobs(changes: dict, groups: dict) -> list:
    """(key, d, groups, seed) for the gain and the 13 guard bounds, with ``arms.spec`` seeds."""
    jobs = [
        (
            "gain",
            [E2.ORIENT[OBJECTIVE] * x for x in changes[OBJECTIVE]["native"]],
            groups[OBJECTIVE],
            AS.BOOT_SEED,
        )
    ]
    for j, g in enumerate(E2.GUARD_SETS[OBJECTIVE]):
        for k, (sc, _side, _lim) in enumerate(E2.MARGINS[g]):
            src = OBJECTIVE if g == "syc_neutral" else g
            jobs.append((f"{g}|{sc}", changes[g][sc], groups[src], AS.BOOT_SEED + 31 * j + k))
    return jobs


def record(changes, groups, pool=None) -> dict:
    jobs = bound_jobs(changes, groups)
    args = [(d, gr, s) for _k, d, gr, s in jobs]
    res = list(pool.map(_boot, args)) if pool is not None else [_boot(a) for a in args]
    by = {k: r for (k, *_), r in zip(jobs, res, strict=True)}
    gb = {}
    for g in E2.GUARD_SETS[OBJECTIVE]:
        gb[g] = {
            sc: {"lcb": by[f"{g}|{sc}"]["lcb"], "ucb": by[f"{g}|{sc}"]["ucb"]}
            for sc, _s, _l in E2.MARGINS[g]
        }
    gain = by["gain"]
    return {
        "objective": OBJECTIVE,
        "gain": gain["mean"],
        "gain_se": gain["se"],
        "lcb": gain["lcb"],
        "gain_items": [E2.ORIENT[OBJECTIVE] * x for x in changes[OBJECTIVE]["native"]],
        "item_groups": list(groups[OBJECTIVE]),
        "guard_bounds": gb,
        "guards_pass": bool(E2.guards_pass(OBJECTIVE, gb)["pass"]),
        "guard_means": {
            g: {
                sc: float(np.mean(changes[g if g != "syc_neutral" else "syc_neutral"][sc]))
                for sc, _s, _l in E2.MARGINS[g]
            }
            for g in E2.GUARD_SETS[OBJECTIVE]
        },
    }


class DevWorld:
    """The real dev world. ``base`` = panel2.evaluate output at theta0 on the same items (the eligible dev-base unit)."""

    def __init__(
        self, m, items: dict, base: dict, lib: Library, check=lambda: None, procs: int = 14
    ):
        self.m, self.items, self.base, self.lib, self.check = m, items, base, lib, check
        self.groups = {o: [it["group"] for it in items[o]] for o in items}
        self.pool = ProcessPoolExecutor(max_workers=procs) if procs > 1 else None
        self.n_eval = 0
        self.eval_seconds = 0.0

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def theta_hash(self, pids) -> str:
        m = self.m
        m.at_base()
        try:
            for b in self.lib.bundles(pids):
                m.sess.apply_bundle(b)
            return m.theta_hash()
        finally:
            m.restore()

    def _apply_eval(self, bundles, items):
        m = self.m
        m.at_base()
        t0 = time.time()
        try:
            for b in bundles:
                m.sess.apply_bundle(b)
            th = m.theta_hash()
            num = sum(float(m.sess.displacement(n).float().square().sum()) for n in m.sess.touched)
            den = sum(float(m.sess.original(n).float().square().sum()) for n in m.sess.touched)
            res, _ = m.evaluate(items)
        finally:
            m.restore()
        self.check()
        return res, th, (math.sqrt(num / den) if den > 0 else 0.0), time.time() - t0

    def evaluate_bundles(self, bundles, candidate=None) -> dict:
        res, th, rel, secs = self._apply_eval(bundles, self.items)
        rec = record(item_changes(self.base, res), self.groups, self.pool)
        rec.update(theta_hash=th, rel_norm=rel, seconds=secs, candidate=list(candidate or []))
        self.n_eval += 1
        self.eval_seconds += secs
        return rec

    def evaluate(self, pids) -> dict:
        return self.evaluate_bundles(self.lib.bundles(pids), pids)

    # ---- S-BoN and the two-stage screen: the first k dev groups per objective, then the remaining groups
    def _subset(self, k0, k1):
        return {o: its[k0:k1] for o, its in self.items.items()}

    def screen(self, pids, k: int = 16) -> dict:
        res, th, rel, secs = self._apply_eval(self.lib.bundles(pids), self._subset(0, k))
        base = {o: v[:k] for o, v in self.base.items()}
        ch = item_changes(base, res)
        gain16 = float(np.mean([E2.ORIENT[OBJECTIVE] * x for x in ch[OBJECTIVE]["native"]]))
        means = {
            g: {sc: float(np.mean(ch[g][sc])) for sc, _s, _l in E2.MARGINS[g]}
            for g in E2.GUARD_SETS[OBJECTIVE]
        }
        self._screened = getattr(self, "_screened", {})
        self._screened[E2.candidate_key(pids)] = res
        return {
            "gain16": gain16,
            "gain": gain16,
            "guard16_ok": bool(E2.guard_points_ok(OBJECTIVE, means)),
            "guard_means": means,
            "theta_hash": th,
            "seconds": secs,
            "rel_norm": rel,
        }

    def complete(self, pids, k: int = 16) -> dict:
        """Score the remaining groups only (the screened k are reused, §II.22.7) and form the full-panel record."""
        first = self._screened[E2.candidate_key(pids)]
        res, th, rel, secs = self._apply_eval(self.lib.bundles(pids), self._subset(k, None))
        full = {o: list(first[o]) + list(res[o]) for o in res}
        rec = record(item_changes(self.base, full), self.groups, self.pool)
        rec.update(theta_hash=th, rel_norm=rel, seconds=secs, candidate=list(pids))
        return rec


def guard_points_from_rec(rec) -> bool:
    return bool(E2.guard_points_ok(OBJECTIVE, rec["guard_means"]))
