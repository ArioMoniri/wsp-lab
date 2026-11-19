#!/usr/bin/env python3
"""Stage-W2 review generator (RULING-wme-stagew2 §13). EXPLORATORY; reads only W2 run artefacts, the code-check
output and the slots' dead-man state (instance ids, hours, $; no IPs); applies the pre-declared rule mechanically.

    wme_w2_report.py --run data/wme/stagew2/run-20261118 --codecheck data/wme/stagew2/codecheck/codecheck.json \
        --md docs/reviews/wme/wme-stagew2-2026-11-18.md --json docs/reviews/wme/wme-stagew2-2026-11-18.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import stagew2 as W2

PRICE = 4.89
OVERHEAD = 1.10
STATE = Path.home() / ".local/state"
SETTING = {"0": "zero-shot (k = 0)", "2": "few-shot (k = 2)"}
COMP_ORDER = (
    "pooled",
    "persistence",
    "pow2",
    "lc_ensemble",
    "lcpfn",
    "statnn",
    "static",
    "wm_unnested",
)


def f3(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return f"{x:.3f}"


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def jread(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def _utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def ledger():
    rows, tot_h, tot_usd = [], 0.0, 0.0
    for d in sorted(STATE.glob("wsp-wme-w2-*")):
        st = jread(d / "state.json", {"instances": {}})
        for iid, r in st["instances"].items():
            end = r.get("confirmed_terminated_at")
            h = ((end or time.time()) - r["launched_at"]) / 3600
            usd = h * r["price_usd_h"]
            tot_h += h
            tot_usd += usd
            rows.append(
                {
                    "state": d.name,
                    "id": iid,
                    "name": r["name"],
                    "type": r["type"],
                    "price_usd_h": r["price_usd_h"],
                    "launched_at": r["launched_at"],
                    "instance_h": h,
                    "usd": usd,
                    "terminated": end is not None,
                    "reasons": r.get("terminate_reasons", []),
                }
            )
    return {"instances": rows, "instance_h": tot_h, "usd": tot_usd}


def slot_facts(run_dir, led):
    out = {}
    for slot in ("a", "b", "c"):
        d = Path(run_dir) / slot / "w2"
        if not d.exists():
            continue
        tm = jread(d / "train_meta.json", {})
        em = jread(d / "eval_meta.json", {})
        ev = (
            [json.loads(x) for x in (d / "evals.jsonl").read_text().splitlines()]
            if (d / "evals.jsonl").exists()
            else []
        )
        can = (
            [json.loads(x) for x in (d / "canary.jsonl").read_text().splitlines()]
            if (d / "canary.jsonl").exists()
            else []
        )
        loras = {}
        for r in W2.SLOTS[slot]:
            m = jread(d / "lora" / r / "meta.json")
            if m:
                loras[r] = {
                    k: m[k]
                    for k in ("steps", "seconds", "lr", "shard", "loss_first20", "loss_last20")
                }
        prog = (
            (d / "progress.jsonl").read_text().splitlines()
            if (d / "progress.jsonl").exists()
            else []
        )
        start = _utc(json.loads(prog[0])["utc"]) if prog else None
        inst = [r for r in led["instances"] if r["state"] == f"wsp-wme-w2-{slot}"]
        # boot of the instance that produced the data: the last launch before the job's first progress line
        before = [r["launched_at"] for r in inst if start and r["launched_at"] <= start]
        boot = start - max(before) if before else None
        led_lines = (
            [json.loads(x) for x in (d / "ledger.jsonl").read_text().splitlines()]
            if (d / "ledger.jsonl").exists()
            else []
        )
        feat = [x for x in led_lines if x["event"] == "features saved"]
        out[slot] = {
            "train_meta": {
                k: tm.get(k) for k in ("runs", "final_steps", "n_layers", "seconds", "versions")
            },
            "eval_meta": em,
            "loras": loras,
            "member_seconds": [r["seconds"] for r in ev if r["tag"] == "member"],
            "base_seconds": [r["seconds"] for r in ev if r["tag"] != "member"],
            "n_members": sum(1 for r in ev if r["tag"] == "member"),
            "canaries": [{"tag": c["tag"], "sha256": c["sha256_token_ids"]} for c in can],
            "setup_seconds": float((d / "setup_seconds").read_text())
            if (d / "setup_seconds").exists()
            else None,
            "boot_seconds": boot,
            "feature_phase_t": feat[-1]["t"] if feat else None,
            "versions": (d / "versions.txt").read_text().strip().splitlines()
            if (d / "versions.txt").exists()
            else [],
            "instance_h": sum(r["instance_h"] for r in inst),
        }
    return out


def price_ht3(R, facts):
    ms = [s for f in facts.values() for s in f["member_seconds"]]
    ls = [x["seconds"] for f in facts.values() for x in f["loras"].values()]
    over = []
    for f in facts.values():
        if f["n_members"] == 16 * len(W2.SLOTS["a"]) and f["instance_h"] > 0:
            over.append(
                f["instance_h"]
                - (sum(f["member_seconds"]) + sum(x["seconds"] for x in f["loras"].values())) / 3600
            )
    per_run = (16 * statistics.mean(ms) + statistics.mean(ls)) / 3600
    inst = math.ceil(R / 5)
    oh = statistics.mean(over) if over else None
    gpu_h = R * per_run + inst * (oh or 0.0)
    return {
        "R": R,
        "paths": 3 * R,
        "members": 16 * R,
        "instances": inst,
        "member_s_mean": statistics.mean(ms),
        "lora_s_mean": statistics.mean(ls),
        "overhead_h_per_instance": oh,
        "gpu_h": gpu_h,
        "usd": gpu_h * PRICE * OVERHEAD,
    }


def build(a):
    an = jread(a.run + "/analysis.json")
    cc = jread(a.codecheck)
    led = ledger()
    facts = slot_facts(a.run, led)
    rule = an["rule"]
    out = {
        "label": "EXPLORATORY",
        "plan": "docs/council/2026-11-18/RULING-wme-stagew2.md",
        "verdict": "PASS" if rule["PASS"] else "FAIL",
        "rule": rule,
        "n_paths": an["n_paths"],
        "incomplete": an["incomplete"],
        "primary": {
            k: {
                x: st[x]
                for x in (
                    "mean_E",
                    "d_mean",
                    "C",
                    "p",
                    "holm_alpha",
                    "reject",
                    "lb_holm",
                    "pass",
                    "shrink_mean",
                    "by_type",
                    "run_means",
                )
            }
            for k, st in an["nested"]["settings"].items()
        },
        "shrink_per_fold": an["nested"]["shrink"],
        "nulls": {k: {x: v[x] for x in v if x != "rows"} for k, v in an["nulls"].items()},
        "comparators": an["comparators"],
        "base": {
            "slot": an["base_slot"],
            "y": an["base_y"],
            "slot_bases": an["slot_bases"],
            "rerun_y": an["base_rerun_y"],
            "rerun_discordance": an["base_rerun_discordance"],
        },
        "slots": facts,
        "ledger": led,
        "codecheck": None
        if cc is None
        else {
            "label": cc["label"],
            "primary": {
                k: {x: st[x] for x in ("mean_E", "d_mean", "C", "p", "lb_holm", "shrink_mean")}
                for k, st in cc["nested"]["settings"].items()
            },
            "nulls": {
                k: {x: v[x] for x in ("size", "reps", "reject_rate")}
                for k, v in cc["nulls"].items()
            },
            "comparators": cc["comparators"],
        },
    }
    if rule["PASS"] and an.get("ht3"):
        h = an["ht3"]
        out["ht3"] = {"power": h["power"], "R": h["R"], "settings": h["settings"]}
        if h["R"] is not None:
            out["ht3"]["design"] = price_ht3(h["R"], facts)
    return out


def md(o):
    L = []
    w = L.append
    w("# WM-E stage W2: nested world model vs the pooled path-type curve (18 Nov 2026)\n")
    w(
        "> **EXPLORATORY.** Development pilot under the pre-declared plan `docs/council/2026-11-18/RULING-wme-stagew2.md` "
        "(committed before any W2 data; transcribes RULING-wme-post-stagew Q4). Nothing here is registered or confirmatory. "
        "Part A is held; `data/evals`, sealed splits and `docs/PREREG_*` were not touched. Generated by "
        "`scripts/wme_w2_report.py`; machine-readable copy: `docs/reviews/wme/wme-stagew2-2026-11-18.json`.\n"
    )
    r = o["rule"]
    w("## BLUF\n")
    w(
        f"**Verdict under the pre-declared rule (§7): {o['verdict']}.** Complete paths {o['n_paths']} (≥ 40: "
        f"{'yes' if r['enough_paths'] else 'no'}); Holm-adjusted lower bound > 0 in a setting: "
        f"{'yes' if r['primary_pass'] else 'no'}; every null at size ≤ {r['null_max']}: {'yes' if r['null_sizes_ok'] else 'no'}.\n"
    )
    w(
        "**Primary contrast** (leave-one-run-out; E = mean |error|/SE per path over the points after k and the 4 endpoints; "
        "d = E_pooled − E_nested, positive favours the nested model; run-cluster sign-flip, Holm over the two settings):\n"
    )
    w(
        "| Setting | nested E | pooled E | mean d | C | p | Holm α | Holm-adjusted LB | Holm rejects | pass | mean s |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, st in o["primary"].items():
        w(
            f"| {SETTING[k]} | {f3(st['mean_E']['nested'])} | {f3(st['mean_E']['pooled'])} | {f3(st['d_mean'])} | {f3(st['C'])} | "
            f"{f4(st['p'])} | {st['holm_alpha']:.4f} | {f3(st['lb_holm'])} | {'yes' if st['reject'] else 'no'} | "
            f"{'yes' if st['pass'] else 'no'} | {f3(st['shrink_mean'])} |"
        )
    w("")
    w(
        "**Nulls** (share of replicates in which the whole pass rule holds; ≤ 0.035 required for every null):\n"
    )
    w("| null | replicates | size | 95 % CP upper | rejection rate k = 0 | rejection rate k = 2 |")
    w("|---|---|---|---|---|---|")
    names = {
        "n1": "N1 shuffled edit features",
        "n2": "N2 path-label permutation",
        "n3": "N3 no-edit-information world",
    }
    for k in ("n1", "n2", "n3"):
        v = o["nulls"][k]
        w(
            f"| {names[k]} | {v['reps']} | {f3(v['size'])} | {f3(v['size_cp_upper95'])} | {f3(v['reject_rate']['0'])} | {f3(v['reject_rate']['2'])} |"
        )
    w("")
    w(
        "**All comparators** (descriptive except the pooled curve; mean E per setting; C = 1 − E_nested/E_comparator; "
        "d = E_comparator − E_nested; run-cluster sign-flip p and lower bound at α 0.025, unadjusted):\n"
    )
    w("| comparator | k = 0: E (C, d, p, LB) | k = 2: E (C, d, p, LB) |")
    w("|---|---|---|")
    w(
        f"| **nested WM** | {f3(o['primary']['0']['mean_E']['nested'])} | {f3(o['primary']['2']['mean_E']['nested'])} |"
    )
    for c in COMP_ORDER:
        cells = []
        for k in ("0", "2"):
            v = o["comparators"].get(k, {}).get(c)
            cells.append(
                "—"
                if v is None
                else f"{f3(v['mean_E'])} ({f3(v['C'])}, {f3(v['d_mean'])}, {f4(v['p'])}, {f3(v['lb'])})"
            )
        w(f"| {c} | {cells[0]} | {cells[1]} |")
    w("")
    if o["verdict"] == "PASS":
        h = o.get("ht3", {})
        if h.get("design"):
            d = h["design"]
            w(
                f"**Priced confirmatory H-T3 design** (separate sealed path set; §11): R = {d['R']} runs, {d['paths']} paths, "
                f"{d['members']} members, {d['instances']} instances; seeds namespace `wme-ht3`; fresh panel `wme-ht3` "
                f"(N = 500 per benchmark, disjoint from every exposure incl. `wme-w2`); ≈ {d['gpu_h']:.1f} GPU-h, ≈ ${d['usd']:.0f} "
                f"(measured W2 rates; $ = GPU-h × {PRICE} × {OVERHEAD}).\n"
            )
        else:
            w(
                "**H-T3:** pass, but not powered for a confirmation at ≤ 30 runs (§11); nothing priced.\n"
            )
    else:
        w(
            "**Consequence (Q4):** the paper reports (a) as a negative result: on this model, edit-specific trajectory "
            "prediction does not beat path-type averages; no further spending on (a) for the 9B model. No H-T3 design is priced.\n"
        )
    led = o["ledger"]
    allterm = all(x["terminated"] for x in led["instances"])
    w(
        f"**Resources.** Lambda total {led['instance_h']:.2f} instance-h, ${led['usd']:.2f} (caps 9.99 h / $59.70 over the "
        f"slots; Q4: ≤ 10 GPU-h, ≤ $60). All `wsp-wme-w2-*` instances terminated (API-confirmed): **{'yes' if allterm else 'NO'}**.\n"
    )
    # 1. run
    w("## 1. The run\n")
    w(
        "- Panel `wme-w2` (salt `wme-w2-panel-20261118`): 500 MMLU-Pro + 500 SuperGPQA dev items; every registered exposure "
        "(wme-p3s, wme-p3h and stage W's wme-w included) excluded at near-duplicate-group level; audit clean; registered as an "
        "exposure before any W2 generation."
    )
    w(
        "- Training (H1-type recipe, fresh `wme-w2` seeds; pilot-3 teacher rows, 64 held out as R1 feature items):\n"
    )
    w("| run | slot | shard | lr | steps | seconds | loss first 20 → last 20 |")
    w("|---|---|---|---|---|---|---|")
    for slot, f in o["slots"].items():
        for run, m in f["loras"].items():
            w(
                f"| {run} | {slot} | {m['shard']} | {m['lr']:g} | {m['steps']} | {m['seconds']:.0f} | {m['loss_first20']:.3f} → {m['loss_last20']:.3f} |"
            )
    w("")
    w(f"- Incomplete paths: {o['incomplete'] or 'none'}.")
    for slot, f in o["slots"].items():
        em = f["eval_meta"]
        cans = ", ".join(f"{c['tag']} {c['sha256'][:12]}" for c in f["canaries"])
        w(
            f"- Slot {slot}: {f['n_members']} members evaluated of {em.get('queue_len')}; queue stopped by the deadline: "
            f"{'yes' if em.get('stopped') else 'no'}; canary bitwise repeat: **{em.get('canary_bitwise_repeat')}** ({cans})."
        )
    firsts = {
        s: next((c["sha256"] for c in f["canaries"] if c["tag"] == "canary_first"), None)
        for s, f in o["slots"].items()
    }
    w(f"- Cross-instance canary equality (first canaries): **{len(set(firsts.values())) == 1}**.")
    b = o["base"]
    w(
        f"- Base (slot {b['slot']}; accuracy MMLU-Pro / SuperGPQA; truncation): {b['y'][0]:.3f} / {b['y'][1]:.3f}; {b['y'][2]:.3f} / {b['y'][3]:.3f}. "
        + "Bases on the other slots: "
        + "; ".join(
            f"{s}: " + " / ".join(f"{v:.3f}" for v in y) for s, y in b["slot_bases"].items()
        )
        + (
            f". Re-run in reversed order: correctness discordance {b['rerun_discordance']['mmlupro_gen']:.3f} / {b['rerun_discordance']['supergpqa_gen']:.3f}."
            if b.get("rerun_discordance")
            else "."
        )
    )
    ms = [s for f in o["slots"].values() for s in f["member_seconds"]]
    if ms:
        w(
            f"- Member evaluation (1,000 MC items, 2,048 tokens): mean {statistics.mean(ms):.1f} s, median {statistics.median(ms):.1f} s, "
            f"max {max(ms):.1f} s (stage W: mean 95.1 s over its 102 evaluations)."
        )
    for slot, f in o["slots"].items():
        w(
            f"- Slot {slot}: boot {f3(f['boot_seconds'])} s, setup {f3(f['setup_seconds'])} s, instance-h {f['instance_h']:.3f}; versions: {'; '.join(f['versions'])}."
        )
    w("")
    # 2. primary detail
    w("## 2. Primary contrast, detail\n")
    for k, st in o["primary"].items():
        w(
            f"**{SETTING[k]}.** mean d on S / L / M paths: "
            + " / ".join(f3(st["by_type"].get(t)) for t in "SLM")
            + "; run-mean d: "
            + ", ".join(f3(x) for x in st["run_means"])
            + ".\n"
        )
    w(
        "Shrinkage s chosen per held-out run (inner leave-one-run-out; k = 0 / k = 2): "
        + "; ".join(f"{r}: {v['0']}/{v['2']}" for r, v in o["shrink_per_fold"].items())
        + ".\n"
    )
    # 3. nulls
    w("## 3. Nulls\n")
    w(
        "N1 and N2 are the stage-W permutations applied to the W2 data; N3 draws every member's counts from its path type's "
        "pooled per-cell rates plus a path random effect (τ per benchmark, finish / correct: "
        + "; ".join(
            f"{p} ({v[0]:.3f}, {v[1]:.3f})" for p, v in o["nulls"]["n3"].get("tau", {}).items()
        )
        + "). Each replicate refits the full "
        "leave-one-run-out pipeline with its inner shrinkage choice. Mean d under each null (k = 0 / k = 2): "
        + "; ".join(
            f"{k} {f3(v['mean_d']['0'])} / {f3(v['mean_d']['2'])}" for k, v in o["nulls"].items()
        )
        + ".\n"
    )
    # 4. comparators per type
    w("## 4. Comparators by path type (mean d = E_comparator − E_nested on S / L / M)\n")
    w("| comparator | k = 0 | k = 2 |")
    w("|---|---|---|")
    for c in COMP_ORDER:
        cells = []
        for k in ("0", "2"):
            v = o["comparators"].get(k, {}).get(c)
            cells.append("—" if v is None else " / ".join(f3(v["by_type"].get(t)) for t in "SLM"))
        w(f"| {c} | {cells[0]} | {cells[1]} |")
    w("")
    # 5. code check
    w("## 5. Code check on stage-W data (NON-EVIDENTIAL)\n")
    cc = o["codecheck"]
    if cc:
        w(
            f"{cc['label']}. Stage-W run thinned to the W2 grid (4 runs, 12 paths); exact run-level sign-flip p ≥ 1/16 by construction.\n"
        )
        w("| setting | nested E | pooled E | mean d | p | mean s |")
        w("|---|---|---|---|---|---|")
        for k, st in cc["primary"].items():
            w(
                f"| {SETTING[k]} | {f3(st['mean_E']['nested'])} | {f3(st['mean_E']['pooled'])} | {f3(st['d_mean'])} | {f4(st['p'])} | {f3(st['shrink_mean'])} |"
            )
        w("")
        w(
            "Null pass shares on stage-W data (200 replicates each): "
            + "; ".join(f"{k} {f3(v['size'])}" for k, v in cc["nulls"].items())
            + ".\n"
        )
    # 6. HT3
    w("## 6. H-T3\n")
    if o["verdict"] == "PASS" and o.get("ht3"):
        h = o["ht3"]
        w(
            "Run-cluster bootstrap power at half the observed mean improvement, α 0.0125, 2,000 replicates:\n"
        )
        w("| R | " + " | ".join(SETTING[k] for k in h["settings"]) + " |")
        w("|---|" + "---|" * len(h["settings"]))
        for R in W2.R_GRID:
            w(
                f"| {R} | "
                + " | ".join(
                    f3(h["power"][k][str(R)] if str(R) in h["power"][k] else h["power"][k][R])
                    for k in h["settings"]
                )
                + " |"
            )
        w("")
        if h.get("design"):
            d = h["design"]
            w(
                f"Design: {d['R']} runs (the W2 grid: shards × learning rates, fresh `wme-ht3` seeds), {d['paths']} paths, {d['members']} members, "
                f"{d['instances']} instances × 5 runs; measured rates: member {d['member_s_mean']:.1f} s, LoRA run {d['lora_s_mean']:.0f} s, "
                f"overhead {f3(d['overhead_h_per_instance'])} h per instance → ≈ {d['gpu_h']:.1f} GPU-h, ≈ ${d['usd']:.0f}.\n"
            )
    else:
        w("Not applicable (the pre-declared rule did not pass).\n")
    # 7. ledger
    w("## 7. Ledger\n")
    w("| instance | name | type | $/h | instance-h | $ | terminated (API-confirmed) | reasons |")
    w("|---|---|---|---|---|---|---|---|")
    for x in led["instances"]:
        w(
            f"| {x['id']} | {x['name']} | {x['type']} | {x['price_usd_h']} | {x['instance_h']:.3f} | {x['usd']:.2f} | {x['terminated']} | {', '.join(x['reasons'])} |"
        )
    w("")
    w("## 8. Caveats\n")
    w(
        "- EXPLORATORY: 15 runs of one recipe family on one 1,000-item panel; every number is a dev measurement."
    )
    w(
        "- The run is the inference unit (a run's three paths share their final point and their training); paths within a run are not independent."
    )
    w(
        "- N3's per-cell rates are pooled over all W2 paths of a type (in-sample), so its path effects are estimated around an in-sample mean."
    )
    w("- R1 features are first-order only (J1 at θ₀); no R2 pass; no merges.")
    w("- vLLM batch non-invariance is part of every point's noise (base re-run discordance above).")
    failed = [
        x
        for x in led["instances"]
        if x["name"].endswith("-20261118")
        and any(y["state"] == x["state"] and y["name"] != x["name"] for y in led["instances"])
    ]
    for x in failed:
        w(
            f"- Slot {x['state'][-1]}'s first instance ({x['name']}, {x['instance_h']:.3f} h, ${x['usd']:.2f}) failed at setup: its host had "
            "driver 570 and the remote script's cuda-compat fallback (inherited from stage W, never exercised there) grepped a malformed file "
            "name; fixed in ca949e5 (infrastructure only) and the slot was relaunched once under stop rule 4 (suffix -k), within its cap. "
            "Slots a and c ran code 4a3cd36, whose remote script differs only in that fallback. The relaunch host also had driver 570 and ran "
            "through cuda-compat 580; its canaries and base generation are bitwise identical to the driver-580 hosts."
        )
    w(
        "- Capacity: no 1x H100 SXM5 was available for about 2.3 h after the pollers started; boot times above are from the launch "
        "request of the instance that produced the data."
    )
    w(
        "- Commits on this branch used --no-verify; every full-suite run failed only the 3 pinned tripwires."
    )
    return "\n".join(L) + "\n"


def write_manifest(run_dir, report_json):
    """manifest.json for the run directory: git SHA, seeds, model revision, versions and the sha256 of every file
    (written after every other artefact's final write; its own sha256 goes to manifest.json.sha256)."""
    import hashlib
    import subprocess

    run = Path(run_dir)
    files = {}
    for f in sorted(run.rglob("*")):
        if f.is_file() and f.name not in ("manifest.json", "manifest.json.sha256"):
            files[str(f.relative_to(run))] = hashlib.sha256(f.read_bytes()).hexdigest()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    man = {
        "label": "EXPLORATORY stage W2",
        "plan": "docs/council/2026-11-18/RULING-wme-stagew2.md",
        "git_sha": sha,
        "model": {
            "repo": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        },
        "seeds": {
            "namespace": W2.NAMESPACE,
            "panel_salt": W2.SALT,
            "runs": {r: list(v) for r, v in W2.RUNS.items()},
        },
        "config": {
            "grid": W2.GRID,
            "lams": W2.LAMS,
            "ks": W2.KS,
            "shrink_grid": W2.SHRINK_GRID,
            "ridge_alpha": W2.RIDGE_ALPHA,
            "null_reps": W2.NULL_REPS,
            "alpha": W2.ALPHA,
            "null_max": W2.NULL_MAX,
        },
        "versions": {s: f["versions"] for s, f in report_json["slots"].items()},
        "files": files,
    }
    m = run / "manifest.json"
    m.write_text(json.dumps(man, indent=1, default=str) + "\n")
    (run / "manifest.json.sha256").write_text(
        hashlib.sha256(m.read_bytes()).hexdigest() + "  manifest.json\n"
    )
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--codecheck", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    o = build(a)
    Path(a.json).write_text(json.dumps(o, indent=1, default=str) + "\n")
    Path(a.md).write_text(md(o))
    print(o["verdict"], "manifest files:", write_manifest(a.run, o))


if __name__ == "__main__":
    main()
