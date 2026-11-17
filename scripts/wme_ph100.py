#!/usr/bin/env python3
"""P-H100: WM-E 9B timing and noise pilot on Lambda H100 SXM5 80 GB (RULING-wme-resources §2). ENGINEERING ONLY.

Items: the exposed wme-p2 panel's validation (SE and timing) and construction (inert verification, field, gradient
and generation timing) splits only. Edits: no-op, inert (I1 round trips, compensated pairs, I2 never-emitted rows)
and the fixed timing edits of ``wsp.wm.ph100``. No trajectory, headroom or enhancement quantity is computed; mean
changes are written only as hashes in the summary (raw unit stores stay on disk under data/, uncommitted).

    worker    --out D --worker wI     one process per GPU (CUDA_VISIBLE_DEVICES set by the caller); dp units
    extras    --out D                  field (K-FAC), oracle-gradient and greedy-generation timing on one GPU
    summarise --out D [--k1 D1 --kk DK] SEs, unit times, inert verification and 1-vs-k parity -> summary.json
    --smoke   4 construction items per objective for both panels (local engineering check; not P-H100 data)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PANEL = ROOT / "data/wme/panels/wme-p2"
REPO = "Qwen/Qwen3.5-9B"
VAL_OBJECTIVES = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
    "mathcomp_gold_lp",
)
VAL_CHUNK = 64
CONS_ITEMS = 16
FIELD_ITEMS = 24
FIELD_MAX_TOKENS = 1024
FIELD_LAYERS = (12, 13, 14, 15)
GEN = {"sycophancy_real": 128, "gsm8k_gold_lp": 512, "mathcomp_gold_lp": 1024}
GEN_PROMPTS = 8


def _load():
    import torch

    from wsp.wm import actions as A
    from wsp.wm import hf

    t = time.time()
    model, tok, cfg, base_hash, _ = hf.load_target(REPO, "cuda")
    names = [n for n, _ in model.named_parameters()]
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in names)
        else "model.layers."
    )
    rm = A.RoleMap.from_config(cfg, prefix=prefix)
    embed = next(n for n in names if n.endswith("embed_tokens.weight"))
    torch.cuda.synchronize()
    return model, tok, cfg, base_hash, rm, embed, time.time() - t


def _items(smoke: bool):
    from wsp.wm import panel2 as P2

    cons = P2.read_open(PANEL, "construction")
    take = lambda its, n: sorted(its, key=lambda r: r["id"])[:n]
    if smoke:
        val = {o: take(cons[o], 4) for o in VAL_OBJECTIVES}
        con = {o: take(cons[o], 4) for o in VAL_OBJECTIVES}
    else:
        v = P2.read_open(PANEL, "validation")
        val = {o: v[o] for o in VAL_OBJECTIVES}
        con = {o: take(cons[o], CONS_ITEMS) for o in VAL_OBJECTIVES}
    return val, con


def _states(skip_embed=False):
    from wsp.wm import ph100 as H

    val = {"base": {"kind": "base"}, "noop": {"kind": "noop"}}
    for s in H.TIMING_EDITS:
        val[s["id"]] = {"kind": "timing", "id": s["id"]}
    cons = {
        "base": {"kind": "base"},
        "noop": {"kind": "noop"},
        "i1_2step": {"kind": "i1", "shape": "2step"},
        "i1_4step": {"kind": "i1", "shape": "4step"},
    }
    for i in range(len(H.COMPENSATED["mlp_layers"]) + len(H.COMPENSATED["attention_layers"])):
        cons[f"comp{i}"] = {"kind": "comp", "index": i}
    for f in H.I2_FACTORS:
        cons[f"i2_{f:+g}"] = {"kind": "i2", "factor": f}
    if (
        skip_embed
    ):  # local smoke only (32 GB card): full-vocabulary fp32 temporaries do not fit beside 9B
        val = {k: v for k, v in val.items() if v.get("id") != "t_rows_embed"}
        cons = {k: v for k, v in cons.items() if v["kind"] != "i2"}
    return val, cons


def _units(base_hash, val_items, con_items, skip_embed=False):
    from wsp.wm import dp

    sv, sc = _states(skip_embed)
    out = []
    for panel, states, its in (("validation", sv, val_items), ("construction", sc, con_items)):
        for x in dp.make_units(
            states, dp.chunk_items(its, VAL_CHUNK), base_hash, f"wme-p2:{panel}"
        ):
            out.append(dict(x, panel=panel))
    return out


def _steps(state, rm, embed, tok):
    """Action bundles (applied in order) for a state recipe."""
    from wsp.wm import actions as A
    from wsp.wm import controls as C
    from wsp.wm import ph100 as H

    k = state["kind"]
    if k == "base":
        return []
    if k == "noop":
        return [[A.scale(rm.name(5, "down"), factor=1.5, dose=0.0)]]
    if k == "timing":
        spec = next(s for s in H.TIMING_EDITS if s["id"] == state["id"])
        return [H.timing_actions(rm, spec, embed=embed, tokenizer_len=len(tok))]
    if k == "i1":
        return H.round_trip(rm, state["shape"])
    if k == "comp":
        return [C.compensated_pairs(rm, **H.COMPENSATED)[state["index"]]]
    if k == "i2":
        rows = H.never_emitted_rows(len(tok), rm.vocab, _template_ids(tok))
        acts = H.i2_actions(rm, embed, rows)
        return [[next(a for a in acts if a.param("factor") == state["factor"])]]
    raise KeyError(k)


def _template_ids(tok):
    from wsp.wm import panel as P

    msgs = P.chat_prompt(tok, "x") + "The answer is "
    ids = set(tok.encode(msgs, add_special_tokens=False))
    ids |= {i for i in tok.all_special_ids}
    return sorted(ids)


def _records(unit, res):
    """Per-item records: the native value, plus syc_neutral and pp-scale values as derived objectives."""
    from wsp.wm import dp
    from wsp.wm import panel2 as P2

    obj = unit["objective"]
    recs = res[obj]
    ids = [r["id"] for r in recs]
    if ids != unit["item_ids"]:
        raise RuntimeError("panel returned items out of unit order")
    out = dp.records_for(unit, [r["value"] for r in recs], ids)
    pp = [P2.pp_value(obj, r) for r in recs]
    if pp[0] is not None:
        out += dp.records_for(dict(unit, objective=f"pp:{obj}"), pp, ids)
    if obj == "sycophancy_real":
        out += dp.records_for(
            dict(unit, objective="syc_neutral"), [r["neutral"] for r in recs], ids
        )
        out += dp.records_for(
            dict(unit, objective="pp:syc_neutral"),
            [P2.pp_value("gsm8k_gold_lp", {"value": r["neutral"]}) for r in recs],
            ids,
        )
    return out


def cmd_worker(a):
    import torch

    from wsp.wm import actions as A
    from wsp.wm import dp
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2
    from wsp.wm import ph100 as H

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, cfg, base_hash, rm, embed, load_s = _load()
    val_items, con_items = _items(a.smoke)
    by_id = {
        p: {o: {r["id"]: r for r in v} for o, v in its.items()}
        for p, its in (("validation", val_items), ("construction", con_items))
    }
    units = _units(base_hash, val_items, con_items, a.smoke)
    sess = A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)
    meta = {
        "worker": a.worker,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "load_s": load_s,
        "theta0": base_hash,
        "n_units": len(units),
        "timing_edit_digest": H.timing_edit_digest(),
    }
    (out / f"meta_{a.worker}.json").write_text(json.dumps(meta, indent=1))

    def fn(u):
        steps = _steps(u["recipe"]["state"], rm, embed, tok)
        if sess.theta_hash() != base_hash:
            raise RuntimeError("worker not at base before a unit")
        torch.cuda.synchronize()
        t0 = time.time()
        safe = True
        for b in steps:
            for act in b:
                if act.family == "scale" and act.param("factor") in (2.0, 0.5) and act.dose == 1.0:
                    safe = safe and H.power_of_two_safe(
                        sess.params[act.tensor], act.param("factor")
                    )
            sess.apply_bundle(b)
        theta = sess.theta_hash()
        torch.cuda.synchronize()
        t1 = time.time()
        its = {u["objective"]: [by_id[u["panel"]][u["objective"]][i] for i in u["item_ids"]]}
        torch.cuda.reset_peak_memory_stats()
        res = P2.evaluate(P.LMAdapter(model), tok, its, P.PanelConfig())
        torch.cuda.synchronize()
        t2 = time.time()
        sess.restore()
        torch.cuda.synchronize()
        t3 = time.time()
        if sess.theta_hash() != base_hash:
            raise RuntimeError("theta drift after restore")
        return {
            "records": _records(u, res),
            "theta_hash": theta,
            "power_of_two_safe": safe,
            "n_items": len(u["item_ids"]),
            "seconds": {"apply": t1 - t0, "eval": t2 - t1, "restore": t3 - t2},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "worker": a.worker,
        }

    t = time.time()
    n = dp.run_worker(out, units, fn, a.worker, theta_after=base_hash, ttl_s=900)
    (out / f"done_{a.worker}.json").write_text(
        json.dumps({"computed": n, "wall_s": time.time() - t, "load_s": load_s})
    )


def _field_panels(tok, con_all, n_items):
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2
    from wsp.wm import scorers as Sc

    cfg = P.PanelConfig()
    panels = {}
    for obj in P2.T2_OBJECTIVES:
        sel = []
        for r in sorted(con_all[obj], key=lambda r: r["id"]):
            if sum(len(x.ids) for x in P2.rows_for(obj, r, tok, cfg)) <= FIELD_MAX_TOKENS:
                sel.append(r)
            if len(sel) == n_items:
                break
        panels[f"construction:{obj}"] = Sc.field_panel(f"construction:{obj}", sel, ())
    return panels


def cmd_extras(a):
    import torch

    from wsp.wm import actions as A
    from wsp.wm import collect as C
    from wsp.wm import genaudit as GA
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2
    from wsp.wm import ph100 as H

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, cfg, base_hash, rm, embed, load_s = _load()
    con_all = P2.read_open(PANEL, "construction")
    rec = {"load_s": load_s, "gpu": torch.cuda.get_device_name(0)}
    panels = _field_panels(tok, con_all, 2 if a.smoke else FIELD_ITEMS)
    specs = P2.objective_specs(list(P2.T2_OBJECTIVES), panels, P.PanelConfig())
    mod = lambda n: n.rsplit(".", 1)[0]
    group = [mod(rm.writers()[(L, r)]) for L in FIELD_LAYERS for r in ("down", "o")]
    allw = [mod(n) for n in sorted(rm.writers().values())]
    work = Path(a.work)

    def timed(label, fisher, groups, at=None):
        sess = A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)
        if at:
            spec = next(s for s in H.TIMING_EDITS if s["id"] == at)
            sess.apply_bundle(H.timing_actions(rm, spec, embed=embed, tokenizer_len=len(tok)))
        d = work / label
        if d.exists():
            shutil.rmtree(d)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            C.collect_grouped(
                model,
                tok,
                specs,
                panels,
                groups,
                "cuda",
                d,
                fisher=fisher,
                theta_hash=sess.theta_hash(),
            )
            torch.cuda.synchronize()
            r = {
                "seconds": time.time() - t,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "disk_bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()),
                "ok": True,
            }
        except torch.OutOfMemoryError as e:
            r = {"ok": False, "error": "OOM", "seconds": time.time() - t, "detail": str(e)[:200]}
        finally:
            sess.restore()
            shutil.rmtree(d, ignore_errors=True)
            torch.cuda.empty_cache()
        r["n_modules"] = sum(len(g) for g in groups)
        rec[label] = r
        (out / "extras.json").write_text(json.dumps(rec, indent=1))

    timed("field_kfac_group4", True, [group])
    timed("oracle_grad_group4", False, [group], at="t_scale_down5")
    timed("oracle_grad_allwriters", False, [allw], at="t_scale_down5")
    gen = {}
    for obj, mx in GEN.items():
        its = sorted(con_all[obj], key=lambda r: r["id"])[:GEN_PROMPTS]
        if obj == "sycophancy_real":
            prompts = [r["neutral_prompt"] for r in its]
        elif obj == "gsm8k_gold_lp":
            prompts = [P.gsm8k_user(r) + "\nGive only the final numeric answer." for r in its]
        else:
            prompts = [r["problem"] + "\nGive only the final answer." for r in its]
        torch.cuda.synchronize()
        t = time.time()
        outs = GA.generate_greedy(model, tok, prompts, max_new_tokens=mx, batch=GEN_PROMPTS)
        torch.cuda.synchronize()
        gen[obj] = {
            "seconds": time.time() - t,
            "prompts": len(prompts),
            "max_new_tokens": mx,
            "new_tokens": sum(len(o["ids"]) for o in outs),
            "hit_max": sum(o["hit_max"] for o in outs),
            "s_per_answer": (time.time() - t) / len(prompts),
        }
    rec["generation"] = gen
    rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (out / "extras.json").write_text(json.dumps(rec, indent=1))


def _stats(run_dir, units, panel):
    from wsp.wm import dp

    us = [u for u in units if u["panel"] == panel]
    sids = sorted({u["state_id"] for u in us})
    groups = {s: dp.state_groups(run_dir, us, s) for s in sids}
    return groups


def cmd_summarise(a):
    import hashlib

    from wsp.wm import dp
    from wsp.wm import ph100 as H
    from wsp.wm.harness import UnitStore

    out = Path(a.out)
    meta = json.loads(next(Path(a.kk or a.k1).glob("meta_*.json")).read_text())
    base_hash = meta["theta0"]
    val_items, con_items = _items(a.smoke)
    units = _units(base_hash, val_items, con_items, a.smoke)
    main = Path(a.kk or a.k1)
    gv = _stats(main, units, "validation")
    stats = {s: dp.paired_stats(gv["base"], g) for s, g in gv.items() if s != "base"}
    timing_states = {s: v for s, v in stats.items() if s.startswith("t_")}
    store = UnitStore(main / "units")
    payloads = [dict(u, **store.read(u["key"])) for u in units]
    s = {
        "label": "P-H100 timing and noise pilot (ENGINEERING ONLY; no trajectory, headroom or enhancement quantity)",
        "gpu": meta["gpu"],
        "theta0": base_hash,
        "timing_edit_digest": H.timing_edit_digest(),
        "items": {
            "validation_per_objective": {o: len(v) for o, v in val_items.items()},
            "construction_per_objective": {o: len(v) for o, v in con_items.items()},
        },
        "se": H.se_summary(timing_states, endpoints=H.ENDPOINTS + H.GUARD_EXTRA),
        "se_pp": H.se_summary(
            {
                k: {o[3:]: x for o, x in v.items() if o.startswith("pp:")}
                for k, v in timing_states.items()
            },
            endpoints=(
                "gsm8k_gold_lp",
                "mathcomp_gold_lp",
                "mmlupro_gold_lp",
                "refusal_margin_harmful",
                "refusal_margin_harmless",
                "sycophancy_real",
                "syc_neutral",
            ),
        ),
        "state_stats_sha256": H.hash_only(stats),
        "unit_times": {
            p: H.unit_times([x for x in payloads if x["panel"] == p])
            for p in ("validation", "construction")
        },
        "peak_allocated_bytes": max(x["peak_allocated_bytes"] for x in payloads),
    }
    # Go-N on validation: the no-op is bitwise equal to base
    s["noop_bitwise_validation"] = gv["noop"] == gv["base"]
    # inert verification on construction items (bitwise panel equality and theta checks)
    gc = _stats(main, units, "construction")
    inert = {}
    for sid in gc:
        if sid == "base":
            continue
        th = {
            x["theta_hash"]
            for x in payloads
            if x["panel"] == "construction" and x["state_id"] == sid
        }
        recs = lambda st: sorted(
            (r["objective"], r["item_id"], r["value"])
            for x in payloads
            if x["panel"] == "construction" and x["state_id"] == st
            for r in x["records"]
        )
        safe = all(
            x["power_of_two_safe"]
            for x in payloads
            if x["panel"] == "construction" and x["state_id"] == sid
        )
        inert[sid] = {
            "bitwise_equal_to_base": recs(sid) == recs("base"),
            "theta_equals_base": th == {base_hash},
            "power_of_two_safe": safe,
            "n_values": len(recs(sid)),
        }
    s["inert"] = inert
    s["inert_pass"] = (
        all(v["bitwise_equal_to_base"] for v in inert.values())
        and all(
            v["theta_equals_base"] for k, v in inert.items() if k.startswith("i1") or k == "noop"
        )
        and all(
            not v["theta_equals_base"] for k, v in inert.items() if k.startswith(("comp", "i2"))
        )
    )
    # parity: every state's group means, k = 1 versus k > 1
    if a.k1 and a.kk:
        g1 = {p: _stats(Path(a.k1), units, p) for p in ("validation", "construction")}
        gk = {p: _stats(Path(a.kk), units, p) for p in ("validation", "construction")}
        par = dp.parity(g1, gk)
        workers = sorted({x["worker"] for x in payloads})
        s["parity"] = {
            **par,
            "k_workers": workers,
            "k1_digest": hashlib.sha256(json.dumps(g1, sort_keys=True).encode()).hexdigest(),
            "kk_digest": hashlib.sha256(json.dumps(gk, sort_keys=True).encode()).hexdigest(),
        }
    else:
        s["parity"] = {"pass": None, "note": "single GPU: 1-vs-k parity not run on hardware"}
    ex = out / "extras.json"
    if ex.exists():
        s["extras"] = json.loads(ex.read_text())
    s["host"] = {"python": platform.python_version()}
    (out / "summary.json").write_text(json.dumps(s, indent=1, sort_keys=True))
    print(json.dumps({k: s[k] for k in ("inert_pass", "noop_bitwise_validation")}, indent=1))
    print(json.dumps(s["parity"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worker", "extras", "summarise"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--work", default="/tmp/wme-ph100-work")
    ap.add_argument("--k1", default=None)
    ap.add_argument("--kk", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    try:
        {"worker": cmd_worker, "extras": cmd_extras, "summarise": cmd_summarise}[a.cmd](a)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
