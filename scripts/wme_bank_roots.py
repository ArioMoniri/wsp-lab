#!/usr/bin/env python3
"""v1 bank roots at theta0 and the direction-similarity audit (PREREG_WME v1 §4.4; RULING-wme-resources §6).
OUTCOME-BLIND: base forward passes on construction-shard prompts and weight arithmetic only; no state is evaluated.

1. Construction pools per catalogue source: the wme-c1 v1 rank order (salt ``wme-c1-panel-20261117``; gen, test,
   dev, bank, then construction) is reproduced in memory from the hash-only availability; only the construction
   remainder is used and nothing is written as a split. MMLU-Pro categories come from the pinned MMLU-Pro revision
   (by ``question_id``); MATH subjects from the item ``type``.
2. ``wm.rootsampler.sample_roots`` with contrast directions = difference of construction-prompt residual means at
   the candidate's layer(s) (pilot-1 ``residual_means``), against every pilot-1/pilot-2 direction, the pilot scale
   tensors and the P-H100 and 27B timing edits.
3. Steer artefacts (MLP-input mean, residual norm), the dose-1 relative norm of each steer root (-> ``bank.noise_rho``),
   row sets (64 token ids from the root's shard), the matched random controls, and the E2 library contrast
   directions (construction/e2 analogue) for the disclosed bank/library overlap.
4. Direction-similarity audit: every accepted bank direction and every E2 library direction against every pilot
   direction; the identity file (no vectors) and its digest go to the report; vectors stay under data/.

    python scripts/wme_bank_roots.py --sizes <selected design json> --out data/wme/bank_roots/<run> [--smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wsp.wm import exposure as X
from wsp.wm import panel as P
from wsp.wm import panel_c1 as C1
from wsp.wm import rootsampler as RS

PILOT_DIRS = {
    "p1a": "data/wme/pilot/pilot-20261116/directions_p3.pt",
    "p1b": "data/wme/pilot/pilot-20261116/directions_p5.pt",
    "p2": "data/wme/pilot2/pilot2-20261116/directions_t2.pt",
    "p2e2": "data/wme/pilot2/pilot2-20261116/e2-9b/directions_e2.pt",
}
SPLIT_ORDER = ("gen", "test", "dev", "bank")
E2_CONTRAST_SOURCES = {"gsm": ("gsm", "wiki"), "mathcomp": ("mathcomp", "harmless")}


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def construction_pools(design):
    """{catalogue source: [prompt-bearing items]} from the construction remainder of each objective."""
    aud = _mod("c1audit", "scripts/wme_c1_exposure_audit.py")
    pools, _ = aud.pools()
    grouped = C1.grouped_pool(pools)
    exposures = X.load_exposures(ROOT)
    from datasets import load_dataset

    cat = {
        r["question_id"]: r["category"]
        for sp in ("test", "validation")
        for r in load_dataset(
            "TIGER-Lab/MMLU-Pro", split=sp, revision="b189ec765aa7ed75c8acfea42df31fdae71f97be"
        )
    }
    raw_mm = {
        P.text_hash(r["question"] + "||" + "||".join(r["options"])): r
        for r in aud._p2_builder()._load_p1_builder().load_raw()[0]["mmlu_pro"]
    }
    owner, out = {}, {s: [] for s in RS.SOURCES}
    need = {
        "gsm8k_gold_lp": ("gsm",),
        "refusal_margin_harmless": ("harmless",),
        "mathcomp_gold_lp": None,
        "mmlupro_gold_lp": None,
    }
    for obj in [o for o in grouped]:
        kept, _ = X.exclude_groups(grouped[obj], exposures)
        by = {}
        for r in sorted(kept, key=lambda r: r["hash"]):
            if owner.setdefault(r["group"], obj) != obj:
                continue
            by.setdefault(r["group"], r)
        if obj not in need:
            continue
        rows = sorted(by.values(), key=lambda r: C1._rank(r["group"]))
        n_front = (
            24 + design["n_bank"] * (obj != "mathcomp_gold_lp") + design["n_dev"] + design["n_test"]
        )
        if obj in ("gsm8k_gold_lp", "mathcomp_gold_lp") and design.get("n_rep_sec"):
            n_front += design["n_gen_gsm" if obj == "gsm8k_gold_lp" else "n_gen_math"]
        for r in rows[n_front:]:
            if obj == "mathcomp_gold_lp":
                s = "mathcomp:" + r["type"].lower().replace(" & ", "_and_").replace(" ", "_")
            elif obj == "mmlupro_gold_lp":
                q = raw_mm.get(r["hash"])
                if q is None:
                    continue
                s = f"mmlu:{cat[q['question_id']]}"
            else:
                s = need[obj][0]
            if s in out:
                out[s].append(dict(r, objective=obj))
    return out


def prompt(tok, r):
    o = r["objective"]
    if o == "gsm8k_gold_lp":
        return P.chat_prompt(tok, P.gsm8k_user(r))
    if o == "mathcomp_gold_lp":
        return P.chat_prompt(tok, r["problem"] + "\nGive only the final answer.")
    if o == "mmlupro_gold_lp":
        return P.chat_prompt(tok, P.mc_user(r))
    return P.chat_prompt(tok, r["prompt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--smoke", action="store_true", help="Qwen3.5-0.8B, r_f = 2, engineering check only"
    )
    a = ap.parse_args()
    t0 = time.time()
    design = json.loads(Path(a.sizes).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W1 = _mod("wme_pilot", "scripts/wme_pilot.py")
    from wsp.wm import actions as A
    from wsp.wm import hf

    repo = "Qwen/Qwen3.5-9B"
    if a.smoke:
        from huggingface_hub import constants

        snaps = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B" / "snapshots"
        hf.TARGETS["Qwen/Qwen3.5-0.8B"] = min(snaps.iterdir()).name
        repo = "Qwen/Qwen3.5-0.8B"
        design = dict(design, r_f=2)
    model, tok, cfg, base_hash, _ = hf.load_target(repo, "cuda")
    names = [n for n, _ in model.named_parameters()]
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in names)
        else "model.layers."
    )
    rm = A.RoleMap.from_config(cfg, prefix=prefix)
    RS.N_LAYERS = rm.n_layers
    pools = construction_pools(design)
    item_of = {r["group"]: r for v in pools.values() for r in v}
    pool_ids = {s: [r["group"] for r in v] for s, v in pools.items()}
    cache = {}

    def means(groups, layers):
        key = (tuple(groups), tuple(layers))
        if key not in cache:
            texts = [prompt(tok, item_of[g]) for g in groups]
            res, mlp, _ = W1.residual_means(model, tok, texts, list(layers))
            cache[key] = (
                {L: res[L].float().cpu() for L in layers},
                {L: mlp[L].float().cpu() for L in layers},
            )
        return cache[key]

    def direction_fn(c):
        n = len(c["contrast_source_groups"]) // 2
        ga, gb = c["contrast_source_groups"][:n], c["contrast_source_groups"][n:]
        ra, ma = means(ga, c["layers"])
        rb, _ = means(gb, c["layers"])
        d = {L: (ra[L] - rb[L]) / (ra[L] - rb[L]).norm() for L in c["layers"]}
        c["_mlp_mean"] = {L: ma[L] for L in c["layers"]}
        return {L: v.numpy() for L, v in d.items()}

    def rows_fn(c, rng):
        ids = sorted(
            {
                i
                for g in c["contrast_source_groups"]
                for i in tok.encode(prompt(tok, item_of[g]), add_special_tokens=False)
                if i < len(tok)
            }
        )
        return sorted(rng.choice(ids, size=min(RS.ROWS, len(ids)), replace=False).tolist())

    from wsp.wm import lora as LR
    from wsp.wm import panel2 as P2

    lora_logs = {}
    task_obj = {
        "gsm8k_gold_lp": "gsm8k_gold_lp",
        "mathcomp_gold_lp": "mathcomp_gold_lp",
        "mmlupro_gold_lp": "mmlupro_gold_lp",
    }
    part = {"gsm8k_gold_lp": "gold", "mathcomp_gold_lp": "gold", "mmlupro_gold_lp": "neutral"}

    def lora_fn(c):
        """Rank-8 LoRA on every writer (pilot-2 settings) trained on the root's construction shard at theta0."""
        obj = task_obj[c["task"]]
        its = [item_of[g] for g in c["adapter_source_groups"]]
        rows = [
            r
            for it in its
            for r in P2.rows_for(obj, it, tok, P.PanelConfig())
            if r.key[2] == part[obj]
        ]
        ad = LR.LoRA(model, LR.writer_targets(rm), rank=8, alpha=16, seed=c["seed"])
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        t = time.time()
        log = LR.train(
            model,
            ad,
            rows,
            pad_id=pad,
            lr=1e-4,
            epochs=1 if a.smoke else 2,
            batch_rows=8,
            seed=c["seed"] + 1,
            device="cuda",
            max_seconds=1800,
            warmup=10,
            checkpointing=True,
        )
        f = ad.factors()
        lora_logs[c["id"]] = {
            "seconds": time.time() - t,
            "steps": log["steps"],
            "n_rows": log["n_rows"],
        }
        del ad
        torch.cuda.empty_cache()
        return f

    excluded = []
    pilot_vecs = {}
    for tag, p in PILOT_DIRS.items():
        d = torch.load(ROOT / p, weights_only=False, map_location="cpu")
        for k, v in d["dirs"].items():
            if v.numel() == rm.hidden:
                pilot_vecs[f"{tag}:{k}"] = v.float().numpy()
                excluded.append(
                    {
                        "id": f"{tag}:{k}",
                        "model": RS.MODEL,
                        "space": "residual",
                        "direction": v.float().numpy(),
                    }
                )
    for L, role in RS.FORBIDDEN_TENSORS:
        excluded.append(
            {"id": f"tensor:{L}{role}", "model": RS.MODEL, "family": "scale", "tensor": [L, role]}
        )
    # the P-H100 random proj_remove timing edit and the 27B timing edits (other model: J3/J4 do not apply)
    from wsp.wm import ph100 as H

    spec = next(s for s in H.TIMING_EDITS if s["family"] == "proj_remove")
    g = torch.Generator().manual_seed(spec["seed"])
    u = torch.randn(rm.hidden, generator=g)
    excluded.append(
        {
            "id": spec["id"],
            "model": RS.MODEL,
            "space": "residual",
            "direction": (u / u.norm()).numpy(),
        }
    )
    for obj in (
        "gsm8k_gold_lp",
        "mathcomp_gold_lp",
        "sycophancy_real",
    ):  # pilot-2 LoRA task vectors (J3)
        fp = ROOT / f"data/wme/pilot2/pilot2-20261116/e2-9b/lora_{obj}.pt"
        if fp.exists() and not a.smoke:
            excluded.append(
                {
                    "id": f"p2:lora:{obj}",
                    "model": RS.MODEL,
                    "space": "lora",
                    "family": "lora_tv",
                    "direction": {"lora": torch.load(fp, weights_only=False)["factors"]},
                }
            )
    res = RS.sample_roots(
        design["r_f"],
        pool_ids,
        direction_fn,
        excluded,
        design["shard_contrast"],
        design["shard_lora"],
        rows_fn=rows_fn,
        lora_fn=lora_fn,
    )
    # steer artefacts and noise rho
    arte = {}
    for r in res["accepted"]:
        if r["family"] == "lora_tv":
            arte[r["id"]] = {"factors": r["direction"]["lora"]}
            r["lora_training"] = lora_logs.get(r["id"])
            r["direction_id"] = hashlib.sha256(
                b"".join(
                    torch.cat([b.ravel(), aa.ravel()]).numpy().tobytes()
                    for _n, (b, aa, _s) in sorted(r["direction"]["lora"].items())
                )
            ).hexdigest()
        if r["family"] in ("steer", "proj_remove"):
            L = r["layers"][0]
            arte[r["id"]] = {
                "directions": {L2: torch.as_tensor(v) for L2, v in r["direction"].items()}
            }
            if r["family"] == "steer":
                m = r["_mlp_mean"][L]
                texts = [prompt(tok, item_of[gg]) for gg in r["contrast_source_groups"][:24]]
                rn = W1.residual_norm(model, tok, texts, L)
                arte[r["id"]].update(mlp_mean=m, res_norm=float(rn))
                w = dict(model.named_parameters())[rm.name(L, "down")]
                act = A.rank_one(
                    rm.name(L, "down"),
                    u=arte[r["id"]]["directions"][L],
                    v=m / m.norm() ** 2,
                    alpha=RS.NOMINALS["steer"] * float(rn),
                    dose=1.0,
                )
                r["rel_norm_dose1"] = A.relative_norm(act, w)
    rho = (
        RS.noise_rho(res["accepted"])
        if any(r["family"] == "steer" for r in res["accepted"])
        else None
    )
    for r in res["accepted"]:
        if r["family"] == "noise":
            r["rho"] = rho
        if r.get("direction") is not None and r["family"] != "lora_tv":
            r["direction_id"] = hashlib.sha256(
                np.concatenate(
                    [np.asarray(v, np.float32) for _k, v in sorted(r["direction"].items())]
                ).tobytes()
            ).hexdigest()
    # E2 library contrast directions on a construction stand-in (gsm, mathcomp only: syc/trivia use sycophancy and
    # harmless prompts of construction/e2, disclosed as computed at the freeze build)
    lib = {}
    for name, (sa, sb) in E2_CONTRAST_SOURCES.items():
        if (
            sa in pool_ids
            and sb in pool_ids
            and len(pool_ids[sa]) >= 32
            and len(pool_ids[sb]) >= 32
        ):
            for L in (20, 22, 24, 26, 28, 30):
                if L >= rm.n_layers:
                    continue
                ra, _ = means(pool_ids[sa][-32:], [L])
                rb, _ = means(pool_ids[sb][-32:], [L])
                lib[f"e2:{name}@{L}"] = ((ra[L] - rb[L]) / (ra[L] - rb[L]).norm()).numpy()
    bank_vecs = {
        r["id"]: r["direction"]
        for r in res["accepted"]
        if r.get("direction") is not None and r["family"] != "lora_tv"
    }
    audit = {
        "bank_vs_pilots": RS.direction_audit(bank_vecs, pilot_vecs),
        "e2_library_vs_pilots": RS.direction_audit(lib, pilot_vecs),
        "bank_vs_e2_library": RS.direction_audit(bank_vecs, lib),
    }
    torch.save({"artefacts": arte, "library": lib}, out / "directions.pt")
    ident = [
        {k: v for k, v in r.items() if k not in ("direction", "_mlp_mean")} for r in res["accepted"]
    ]
    rep = {
        "label": "v1 bank roots at theta0 and the direction-similarity audit (outcome-blind; no state evaluated)",
        "model": repo,
        "theta0": base_hash,
        "smoke": a.smoke,
        "design": design,
        "pool_sizes": {s: len(v) for s, v in pool_ids.items()},
        "counts": res["counts"],
        "tries": res["tries"],
        "feasible": res["feasible"],
        "n_rejected": len(res["rejected"]),
        "rejection_reasons": {
            k: sum(1 for x in res["rejected"] if any(s.startswith(k) for s in x["reasons"]))
            for k in ("J2", "J3", "J4", "J5", "pool", "forbidden_tensor")
        },
        "noise_rho": rho,
        "roots": ident,
        "controls": res["controls"],
        "identity_digest": RS.identity_digest(res["accepted"]),
        "audit": {k: {kk: vv for kk, vv in v.items()} for k, v in audit.items()},
        "n_pilot_directions": len(pilot_vecs),
        "seconds": time.time() - t0,
    }
    (out / "roots.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=float))
    print(
        json.dumps(
            {
                k: rep[k]
                for k in ("counts", "feasible", "rejection_reasons", "noise_rho", "identity_digest")
            },
            indent=1,
            default=float,
        )
    )
    print(
        json.dumps(
            {k: {"pass": v["pass"], "flagged": v["flagged"][:10]} for k, v in audit.items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
