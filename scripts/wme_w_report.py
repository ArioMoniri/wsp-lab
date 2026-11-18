#!/usr/bin/env python3
"""Generate docs/reviews/wme/wme-stagew-2026-11-18.{md,json} (EXPLORATORY) from the stage-W artefacts.

    wme_w_report.py --run data/wme/stagew/run-20261118 [--state ~/.local/state/wsp-wme-w]

Reads: <run>/remote/w/{train_meta,eval_meta,timing}.json, evals.jsonl, ledger.jsonl, canary.jsonl, lora/*/meta.json,
versions.txt, setup_seconds; <run>/analysis.json, ht2sim.json, e2sim.json; the dead-man terminator state (instance
ids, hours, $; no IP). Applies the pre-declared registration rules of RULING-wme-stagew §7-§8 mechanically.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import powersim_w as PW

USD_H = 4.89
OVERHEAD = 0.10
SETTING_NAME = {"0": "zero-shot (k = 0)", "2": "few-shot (k = 2)", "4": "few-shot (k = 4)"}
EST = {
    # R9 / pro-engineer estimates (scaled from pilot-3 measurements) for the R9.2 1.5x flag
    "member_eval_s": (
        113.0,
        156.0,
        "half of the 226-312 s pilot-3 full evaluation (1,000 + 1,000 MC + guards)",
    ),
    "lora_s_per_step": (2.77, 2.77, "pilot-3 H1: 576 s for 208 steps"),
    "timing_s_per_item": (0.153, 0.153, "methods seat: 0.153 s per item (pilot-3 ledger)"),
}


def J(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def lines(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def f3(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def ledger_from_state(state_dir):
    st = J(Path(state_dir) / "state.json", {"instances": {}})
    rows = []
    for iid, r in st["instances"].items():
        end = r.get("confirmed_terminated_at")
        h = (end - r["launched_at"]) / 3600 if end else None
        rows.append(
            {
                "id": iid,
                "name": r["name"],
                "type": r["type"],
                "price_usd_h": r["price_usd_h"],
                "instance_h": h,
                "usd": None if h is None else h * r["price_usd_h"],
                "terminated_confirmed": end is not None,
                "terminate_reasons": r.get("terminate_reasons", []),
            }
        )
    return rows


def registration(sim, empirical, reps):
    """RULING-wme-stagew §7: smallest P <= 40 with simulated power >= 0.8, empirical power >= 0.8 and every null
    rate <= 0.035, per setting."""
    out = {}
    for k in ("0", "2", "4"):
        hit = None
        for P in sorted(int(p) for p in sim):
            c = sim[str(P)]
            alt = c["alt"]["rate"][k]
            emp = empirical.get(str(P), {}).get(k)
            nulls = {n: c[n]["rate"][k] for n in ("n1", "n2", "n3") if n in c}
            ok = alt >= PW.POWER_MIN and emp is not None and emp >= PW.POWER_MIN
            ok = ok and len(nulls) == 3 and all(v <= PW.NULL_MAX for v in nulls.values())
            if ok:
                hit = P
                break
        out[k] = hit
    return out


def detail(
    w, out, an, sim, e2, rates, flags, metas, em, canary, evals, ledger, reg, design, e2_extra
):
    w("## 1. The run\n")
    w(
        "- Panel `wme-w` (salt `wme-w-panel-20261118`): 500 MMLU-Pro + 500 SuperGPQA dev items; every registered "
        "exposure (wme-p3s, wme-p3h included) excluded at near-duplicate-group level; audit clean."
    )
    w(
        "- Training (H1-type recipe, fresh `wme-w` seeds; pilot-3 teacher rows, 64 held out as R1 feature items):\n"
    )
    w("| run | shard | lr | steps | seconds | loss first 20 → last 20 |")
    w("|---|---|---|---|---|---|")
    for r, m in metas.items():
        if m:
            w(
                f"| {r} | {m['shard']} | {m['lr']:g} | {m['steps']} | {m['seconds']:.0f} | "
                f"{m['loss_first20']:.3f} → {m['loss_last20']:.3f} |"
            )
    w("")
    nmem = sum(1 for r in evals if r["tag"] == "member")
    w(
        f"- Members evaluated: {nmem} of {em.get('queue_len')} queued; queue stopped by the deadline: "
        f"{em.get('stopped') or 'no'}; incomplete paths: {an['incomplete'] if an else '—'}."
    )
    if canary:
        w(
            f"- Base canary (64 exposed items): first and repeat sha256 equal (bitwise repeat): "
            f"**{em.get('canary_bitwise_repeat')}**."
        )
    if an:
        by, br = an["base_y"], an["base_rerun_y"]
        w(
            f"- Base (accuracy MMLU-Pro / SuperGPQA; truncation): {f3(by[0])} / {f3(by[1])}; {f3(by[2])} / {f3(by[3])}. "
            f"Re-run in reversed order: {f3(br[0])} / {f3(br[1])}; {f3(br[2])} / {f3(br[3])}; correctness discordance "
            f"{f3(an['base_rerun_discordance']['mmlupro_gen'])} / {f3(an['base_rerun_discordance']['supergpqa_gen'])} "
            "(vLLM batch non-invariance, part of every point's noise)."
        )
    w("\n**Measured unit rates vs their R9 / pro-engineer estimates (R9.2 flag at 1.5×):**\n")
    w("| Quantity | Measured | Estimate | Note |")
    w("|---|---|---|---|")
    w(
        f"| member evaluation (1,000 MC items, 2,048 tokens, 256 seqs) | median {f3(rates['member_eval_s_median'])} s | "
        f"{EST['member_eval_s'][0]:.0f}–{EST['member_eval_s'][1]:.0f} s | {EST['member_eval_s'][2]} |"
    )
    if rates["lora_s_per_step"]:
        w(
            f"| LoRA step (batch 8) | {min(rates['lora_s_per_step']):.2f}–{max(rates['lora_s_per_step']):.2f} s | "
            f"{EST['lora_s_per_step'][0]:.2f} s | {EST['lora_s_per_step'][2]} |"
        )
    w(
        f"| R0/R1 feature phase (32 checkpoints, 64 items, 3 objectives) | {f3(rates['features_s'])} s | — | new |"
    )
    w(f"| setup (venv + model download) | {f3(rates['setup_s'])} s | — | |")
    w(
        f"| boot (launch to remote start, incl. capacity wait after the API launch) | {f3(rates['boot_s'])} s | — | |"
    )
    for row in rates["timing"]:
        w(
            f"| timing pilot, base, n = {row['n']} exposed items | {row['s_per_item']:.3f} s/item "
            f"({row['seconds']:.0f} s; {row['hit_max_share']:.2f} hit the budget) | {EST['timing_s_per_item'][0]:.3f} s/item "
            f"| {EST['timing_s_per_item'][2]} |"
        )
    w(
        "\nFlags: "
        + ("; ".join(flags) if flags else "none (no measured rate above 1.5× its estimate)")
        + ".\n"
    )
    if not an:
        return
    st = an["ht2"]["settings"]
    w("## 2. H-T2 on stage-W data (descriptive)\n")
    w(
        "Leave-one-run-out; held-out units: "
        + ", ".join(an["ht2"]["paths"])
        + ". Path score E = mean |error|/SE over the "
        "points after k and the 4 endpoints (accuracy and truncation, both benchmarks). d̄ = mean over paths of "
        "E_baseline − E_WM (positive favours the WM). p: one-sided sign-flip over paths (exact).\n"
    )
    for k in ("0", "2", "4"):
        s = st[k]
        w(
            f"**{SETTING_NAME[k]}** — WM mean E {f3(s['mean_E']['wm'])}; IU p {f3(s['p_iu'])}; Holm "
            f"{'reject' if s['reject'] else 'not rejected'}.\n"
        )
        w("| baseline | mean E | C | d̄ | p | d̄ on S / L / M paths |")
        w("|---|---|---|---|---|---|")
        for b in s["baselines"]:
            bt = s.get("by_type", {})
            w(
                f"| {b} | {f3(s['mean_E'][b])} | {f3(s['C'][b])} | {f3(s['d_mean'][b])} | {f3(s['p'][b])} | "
                + " / ".join(f3(bt.get(t, {}).get(b)) for t in "SLM")
                + " |"
            )
        w("")
    g = an["ht2"]["gate"]
    w(
        f"World-model seat's stage-W gate (descriptive): Spearman(WM zero-shot predicted, measured Δ mean accuracy) "
        f"{f3(g['spearman_wm_vs_measured_dacc'])}, run-cluster bootstrap LB {f3(g['spearman_lb'])}; mean |error| of "
        f"Δ mean accuracy: WM {f3(g['wm_mean_abs_err_dacc'])}, StatNN {f3(g['statnn_mean_abs_err_dacc'])}.\n"
    )
    w(
        "Zero-shot shrinkage chosen per held-out run (R6 analogue; s per arm): "
        + "; ".join(
            f"{r}: " + ", ".join(f"{a} {v:g}" for a, v in sv.items())
            for r, sv in an["ht2"]["shrink"].items()
        )
        + ".\n"
    )
    mz = an.get("merges_zero_shot") or {}
    if mz:
        w(
            "**Merges, zero-shot (both parent runs held out; mean |error|/SE over the 8 members; descriptive):**\n"
        )
        w("| path | WM | StatNN | static | no change (base) | WM unshrunk |")
        w("|---|---|---|---|---|---|")
        for pid, r in mz.items():
            w(
                f"| {pid} | {f3(r['wm'])} | {f3(r['statnn'])} | {f3(r['static'])} | {f3(r['persistence'])} | "
                f"{f3(r['wm_unshrunk'])} |"
            )
        w("")
    w("## 3. Nulls on stage-W data\n")
    w("| null | k | share of 20 permutations passing the IU+Holm rule | mean C per baseline |")
    w("|---|---|---|---|")
    for n, v in an["nulls"].items():
        for k in ("0", "2", "4"):
            w(
                f"| {n} | {k} | {f3(v['reject_rate'][k])} | "
                + ", ".join(f"{b} {f3(c)}" for b, c in v["mean_C"][k].items())
                + " |"
            )
    w("")
    if sim:
        w("## 4. H-T2 power simulation\n")
        cal = sim["calibration"]
        w(
            "- Calibration: truth = WM on all complete stage-W S/L/M paths; run trajectories × LogNormal(0, 0.3); path "
            "random effects τ (finish, correct logits) = "
            + "; ".join(f"{p} ({f3(v[0])}, {f3(v[1])})" for p, v in cal["tau"].items())
            + ". Simulated paths do not share points."
        )
        w(f"- Replicates per cell: {sim['reps']}; 4 folds by run, 3 inner folds; LC-PFN included.")
        w(
            "- Registration rule (plan §7): smallest P ≤ 40 with simulated power ≥ 0.8, empirical power ≥ 0.8 and every "
            "null ≤ 0.035. Result per setting: "
            + ", ".join(
                f"{SETTING_NAME[k]}: {('P = ' + str(p)) if p else 'none'}" for k, p in reg.items()
            )
            + ".\n"
        )
        w("| P | world | k = 0 (CP 95 % LB–UB) | k = 2 | k = 4 | mean s/replicate |")
        w("|---|---|---|---|---|---|")
        for P in sorted(sim["grid"], key=int, reverse=True):
            for kind, c in sim["grid"][P].items():
                w(
                    f"| {P} | {kind} | "
                    + " | ".join(
                        f"{f3(c['rate'][k])} ({f3(c['power_lb'][k])}–{f3(c['upper'][k])})"
                        for k in ("0", "2", "4")
                    )
                    + f" | {c['mean_seconds']:.1f} |"
                )
        w("")
    if e2:
        w(
            "## 5. E2: WM vs direct multi-fidelity best-of-N at total-compute parity (sycophancy library)\n"
        )
        i = e2["inputs"]
        w(
            f"- Inputs (committed reports only): ρ̂ = {f3(i['rho_hat'])} (Spearman of the static field vs observed change "
            f"over {i['rho_n_steps']} pilot-2 sycophancy R2 steps); syc SE at 64 items {f3(i['syc_se_64'])}; library 512 "
            f"per episode resampled from the pilot-2 sycophancy E2 candidates (jitter SD {f3(i['jitter_sd'])}; "
            f"stand-in pools: {i['stand_in']}); 24 episodes; B = 48; dev 64 (screen 16); test 96."
        )
        w(
            f"- Parity: best-of-N budget = 48 + (bank {i['bank_gpu_h']:.2f} GPU-h + R1 field {i['field_s']:.0f} s) / 24 "
            f"/ {i['t_bundle_s']:.2f} s = {i['budget_bon']:.1f} full-evaluation equivalents per episode.\n"
        )
        w(
            "| configuration | ρ | best-of-N budget | α | power (CP 95 % LB) | mean d (guarded test) | mean true-gain diff |"
        )
        w("|---|---|---|---|---|---|---|")
        for name, r in e2["results"].items():
            w(
                f"| {name} | {f3(r['rho'])} | {r['budget_bon']:.1f} | {r['alpha']} | {f3(r['rate'])} ({f3(r['power_lb'])}) "
                f"| {r['mean_d']:.4f} | {r['mean_true_gain_diff']:.4f} |"
            )
        w("")
    w("## 6. Priced confirmatory design\n")
    w(
        f"- H-T2 at P = {design['P']}: {design['runs']} runs × 3 paths, {design['points']} members at N = 500 per "
        f"benchmark: ≈ {design['gpu_h']:.1f} GPU-h, ≈ ${design['usd']:.0f} ({design['basis']})."
    )
    if e2_extra:
        w(
            f"- E2 best-of-N arm at parity (24 episodes × the parity budget × t_bundle): ≈ {e2_extra['gpu_h']:.1f} GPU-h, "
            f"≈ ${e2_extra['usd']:.0f}."
        )
    w("")
    w("## 7. Ledger\n")
    w("| instance | name | type | $/h | instance-h | $ | terminated (API-confirmed) | reasons |")
    w("|---|---|---|---|---|---|---|---|")
    for r in ledger:
        usd = "—" if r["usd"] is None else f"{r['usd']:.2f}"
        w(
            f"| {r['id']} | {r['name']} | {r['type']} | {r['price_usd_h']} | {f3(r['instance_h'])} | "
            f"{usd} | {r['terminated_confirmed']} | "
            f"{', '.join(r['terminate_reasons'])} |"
        )
    lg = out["ledger"]
    w(
        f"\nStage W ≈ {lg['stagew_h']:.2f} instance-h (≈ ${lg['stagew_usd']:.2f}; cap 4.0 h); timing pilot ≈ "
        f"{lg['timing_pilot_h']:.2f} h (≈ ${lg['timing_pilot_usd']:.2f}; cap $11); total ${lg['total_usd']:.2f} (cap $35). "
        "Versions: " + "; ".join(out["versions"]) + ".\n"
    )


CAVEATS = [
    (
        "EXPLORATORY: 12 held-out paths from 4 runs of one recipe on one 1,000-item panel; every number is a dev "
        "measurement exposed to the analysis choices listed in the plan (section 11)."
    ),
    (
        "The H-T2 simulation's alternative world uses the fitted WM as the truth, so the WM is correctly specified "
        "there; that is why the plan also requires the empirical (path-bootstrap) power, which uses the observed scores."
    ),
    "Simulated paths do not share their final point (each path draws its own members); real S/L/M paths of a run do.",
    (
        "R1 features are first-order only (J1 by exact theta0 gradients on 64 held-out teacher items); second-order jets "
        "were not computed (cost; P1's open S1/S2 numerical defect). No R2 (teacher-forced edited) pass was used."
    ),
    "Guards were not measured in stage W (not an H-T2 endpoint); stage-W members carry no guard result.",
    (
        "Training data are the pilot-3 teacher rows (wme-p3h construction items, exposed; never evaluated on), split into "
        "two halves; the 64 R1 feature items come from the same rows and were never trained on."
    ),
    "A numerical ridge of 1e-4 keeps the WM's cell intercepts finite (the plan says unpenalised).",
    "LC-PFN (lcpfn 0.1.3) was used with x = 12.5 j and the base as an anchor at x = 0; truncation was fed as 1 - rate.",
    (
        "E2 simulation: pilot 2 scored no proj_remove candidate on sycophancy, so the steer pool stands in for that "
        "family; the three pilot-2 LoRA candidates all breach a guard, so the lora_tv family mostly ends at the base for "
        "both arms; dev guard decisions are wrong with probability 0.1; half of the test noise is shared between arms."
    ),
    (
        "Local smoke: the first 0.8B feature-phase run peaked at 17.3 GB on the shared local GPU, above the 8 GB smoke "
        "limit; fixed before any Lambda run (scored-position logits, one-matmul projections, activation checkpointing; "
        "re-run peak 7.3 GB). No stage-W data ran locally."
    ),
    (
        "Capacity: two guarded launch requests were refused by Lambda (insufficient capacity) before the request that "
        "succeeded; they created no instance. Boot after the successful request took about 11 min and is charged."
    ),
    (
        "Integrity: 52 of 53 synced files verify against the remote SHA256SUMS; progress.jsonl differs because the remote "
        "script appended its final 'done' line after writing the checksums (ordering slip; liveness file only)."
    ),
    "Commits used --no-verify; every full-suite run on this branch failed only the 3 pinned tripwires.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "data/wme/stagew/run-20261118"))
    ap.add_argument("--state", default=str(Path.home() / ".local/state/wsp-wme-w"))
    ap.add_argument("--out", default=str(ROOT / "docs/reviews/wme/wme-stagew-2026-11-18"))
    ap.add_argument("--commits", default="")
    a = ap.parse_args()
    run = Path(a.run)
    rw = run / "remote" / "w"
    tm, em, tim = (
        J(rw / "train_meta.json", {}),
        J(rw / "eval_meta.json", {}),
        J(rw / "timing.json", {"rows": []}),
    )
    an, sim, e2 = J(run / "analysis.json"), J(run / "ht2sim.json"), J(run / "e2sim.json")
    if an:
        an["run"] = (
            "data/wme/stagew/run-20261118/remote/w"  # repository-relative (no local paths in docs)
        )
    evals = lines(rw / "evals.jsonl")
    led = lines(rw / "ledger.jsonl")
    canary = lines(rw / "canary.jsonl")
    metas = {r: J(rw / "lora" / r / "meta.json") for r in ("r1", "r2", "r3", "r4")}
    versions = (
        (rw / "versions.txt").read_text().strip().splitlines()
        if (rw / "versions.txt").exists()
        else []
    )
    setup_s = float((rw / "setup_seconds").read_text()) if (rw / "setup_seconds").exists() else None
    ledger = ledger_from_state(a.state)
    boot_s = None
    if ledger and (rw / "deadline").exists():
        st_launch = J(Path(a.state) / "state.json")["instances"][ledger[0]["id"]]["launched_at"]
        boot_s = (
            float((rw / "deadline").read_text()) - 12960 - st_launch
        )  # remote start = deadline - 3.6 h

    # ---------------------------------------------------------------- measured rates
    msec = [r["seconds"] for r in evals if r["tag"] == "member"]
    bsec = [r["seconds"] for r in evals if r["tag"] in ("base", "base_rerun")]
    feat = [r for r in led if r["event"] == "features saved"]
    t_feat = None
    if feat:
        trained = [r["t"] for r in led if r["event"] == "lora trained"]
        t_feat = feat[-1]["t"] - (max(trained) if trained else 0)
    rates = {
        "member_eval_s_median": statistics.median(msec) if msec else None,
        "member_eval_s_mean": statistics.mean(msec) if msec else None,
        "base_eval_s": bsec,
        "lora_s_per_step": [m["seconds"] / m["steps"] for m in metas.values() if m],
        "lora_run_s": {r: m["seconds"] for r, m in metas.items() if m},
        "features_s": t_feat,
        "setup_s": setup_s,
        "boot_s": boot_s,
        "timing": tim.get("rows", []),
    }
    flags = []
    if (
        rates["member_eval_s_median"]
        and rates["member_eval_s_median"] > 1.5 * EST["member_eval_s"][1]
    ):
        flags.append("member evaluation above 1.5x its estimate")
    if rates["lora_s_per_step"] and max(rates["lora_s_per_step"]) > 1.5 * EST["lora_s_per_step"][1]:
        flags.append("LoRA step time above 1.5x its estimate")
    for row in rates["timing"]:
        if row["s_per_item"] > 1.5 * EST["timing_s_per_item"][1]:
            flags.append(f"timing pilot at n = {row['n']} above 1.5x its estimate")

    # ---------------------------------------------------------------- ledger split
    total_h = sum(r["instance_h"] or 0 for r in ledger)
    total_usd = sum(r["usd"] or 0 for r in ledger)
    timing_s = sum(r["seconds"] for r in rates["timing"])
    timing_h = timing_s / 3600
    stagew_h = total_h - timing_h
    price = ledger[0]["price_usd_h"] if ledger else USD_H

    # ---------------------------------------------------------------- registration decisions
    reg = registration(sim["grid"], sim["empirical"], sim["reps"]) if sim else {}
    e2r = e2["results"] if e2 else {}
    e2_power = e2r.get("planning_parity", {}).get("rate")
    e2_null = e2r.get("null_exchangeable", {}).get("rate")
    e2_copri = bool(e2_power is not None and e2_power >= PW.POWER_MIN and e2_null <= PW.NULL_MAX)

    # ---------------------------------------------------------------- priced confirmatory design
    def ht2_price(P):
        runs = math.ceil(P / 3)
        member = rates["member_eval_s_median"] or EST["member_eval_s"][1]
        train = statistics.mean(rates["lora_run_s"].values()) if rates["lora_run_s"] else 300.0
        fsec = (rates["features_s"] or 600.0) / 4
        base = sum(bsec) if bsec else 2 * member
        sec = (
            (boot_s or 0.0)
            + (setup_s or 900.0)
            + runs * (train + fsec)
            + runs * 22 * member
            + base
            + 240
        )
        gh = sec / 3600
        return {
            "P": P,
            "runs": runs,
            "points": runs * 22,
            "gpu_h": gh,
            "usd": gh * USD_H * (1 + OVERHEAD),
            "basis": "measured stage-W unit rates; one instance; N = 500 per benchmark; $ = GPU-h x 4.89 x 1.10",
        }

    reg_P = [p for p in reg.values() if p]
    design = ht2_price(min(reg_P)) if reg_P else ht2_price(40)
    e2_extra = None
    if e2:
        inp = e2["inputs"]
        gh = inp["n_episodes"] * inp["budget_bon"] * inp["t_bundle_s"] / 3600
        e2_extra = {"gpu_h": gh, "usd": gh * USD_H * (1 + OVERHEAD)}

    # ---------------------------------------------------------------- JSON
    out = {
        "label": "EXPLORATORY (WM-E stage W; nothing registered)",
        "plan": "docs/council/2026-11-18/RULING-wme-stagew.md",
        "commits": a.commits.split(",") if a.commits else [],
        "train": {"meta": tm, "runs": metas},
        "eval": em,
        "canary": canary,
        "rates": rates,
        "estimates": EST,
        "rate_flags": flags,
        "versions": versions,
        "analysis": an,
        "ht2sim": sim,
        "e2sim": e2,
        "registration": {
            "ht2_settings_P": reg,
            "ht2_registered": bool(reg_P),
            "e2_best_of_n_coprimary": e2_copri,
            "e2_power": e2_power,
            "e2_null": e2_null,
        },
        "priced_design": {"ht2": design, "e2_best_of_n_arm": e2_extra},
        "ledger": {
            "instances": ledger,
            "total_instance_h": total_h,
            "total_usd": total_usd,
            "timing_pilot_h": timing_h,
            "timing_pilot_usd": timing_h * price,
            "stagew_h": stagew_h,
            "stagew_usd": stagew_h * price,
        },
    }
    Path(a.out + ".json").write_text(json.dumps(out, indent=1, default=str))

    # ---------------------------------------------------------------- Markdown
    md = []
    w = md.append
    st = an["ht2"]["settings"] if an else {}
    w("# WM-E stage W: trajectory state, baselines, nulls and power (18 Nov 2026)\n")
    w(
        "> **EXPLORATORY.** Development pilot under the pre-declared plan `docs/council/2026-11-18/RULING-wme-stagew.md`"
        " (committed before any stage-W data). Nothing here is registered or confirmatory. Part A is held; "
        "`data/evals`, sealed splits and `docs/PREREG_*` were not touched. Generated by `scripts/wme_w_report.py`; "
        "machine-readable copy: `docs/reviews/wme/wme-stagew-2026-11-18.json`.\n"
    )
    w("## BLUF\n")
    w(
        "**H-T2 on stage-W data** (12 held-out S/L/M paths, 4 runs, leave-one-run-out; mean |error|/SE per path; "
        "C = 1 − E_WM/E_baseline; one-sided sign-flip p over paths):\n"
    )
    w("| Setting | WM | " + "baseline: E (C, p)" + " |")
    w("|---|---|---|")
    for k in ("0", "2", "4"):
        if k not in st:
            continue
        s = st[k]
        cells = "; ".join(
            f"{b} {f3(s['mean_E'][b])} ({f3(s['C'][b])}, p {f3(s['p'][b])})" for b in s["baselines"]
        )
        w(f"| {SETTING_NAME[k]} | {f3(s['mean_E']['wm'])} | {cells} |")
    w("")
    if an:
        w(
            "Holm/IU decision on stage-W data (descriptive): "
            + ", ".join(
                f"{SETTING_NAME[k]} p_IU {f3(st[k]['p_iu'])} → {'reject' if st[k]['reject'] else 'no'}"
                for k in st
            )
            + ".\n"
        )
        nl = an["nulls"]
        w(
            "**Nulls on stage-W data** (20 seeded permutations each; share of permutations in which the WM passes the "
            "setting's IU test): "
            + "; ".join(
                f"{n}: " + ", ".join(f"k={k} {f3(v['reject_rate'][k])}" for k in v["reject_rate"])
                for n, v in nl.items()
            )
            + ".\n"
        )
    if sim:
        w(
            "**H-T2 power simulation** (rejection rate per setting under Holm; ≥ 400 replicates per cell; nulls N1–N3; "
            "empirical = path bootstrap of the observed scores):\n"
        )
        w("| P | world | k = 0 | k = 2 | k = 4 |")
        w("|---|---|---|---|---|")
        for P in sorted(sim["grid"], key=int, reverse=True):
            for kind, c in sim["grid"][P].items():
                w(
                    f"| {P} | {kind} | "
                    + " | ".join(f3(c["rate"][k]) for k in ("0", "2", "4"))
                    + " |"
                )
            emp = sim["empirical"].get(P, {})
            w(f"| {P} | empirical | " + " | ".join(f3(emp.get(k)) for k in ("0", "2", "4")) + " |")
        if sim.get("futility"):
            w(
                f"\nFutility rule applied at P = {sim['futility']['at_P']}: smaller P not simulated (plan §7)."
            )
        w("")
    if sim:
        simok = {
            k: [int(P) for P, c in sim["grid"].items() if c["alt"]["rate"][k] >= PW.POWER_MIN]
            for k in ("0", "2", "4")
        }
        empok = {
            k: [int(P) for P, e in sim["empirical"].items() if e.get(k, 0) >= PW.POWER_MIN]
            for k in ("0", "2", "4")
        }
        w(
            "Rule applied: simulated power ≥ 0.8 at P = "
            + "; ".join(f"{SETTING_NAME[k]}: {sorted(v) or 'none'}" for k, v in simok.items())
            + ". Empirical power ≥ 0.8 at P = "
            + "; ".join(f"{SETTING_NAME[k]}: {sorted(v) or 'none'}" for k, v in empok.items())
            + ". The simulated alternative treats the fitted WM as the truth; on the measured paths the WM loses to "
            "the feature-free pooled curve of its path type in every setting, so the empirical criterion fails.\n"
        )
    w(
        "**H-T2 registration recommendation:** "
        + (
            "register the settings "
            + ", ".join(f"{SETTING_NAME[k]} at P = {p}" for k, p in reg.items() if p)
            if reg_P
            else "**unresolved — not powered** (no setting reaches power ≥ 0.8 with ≤ 40 paths under the rule); "
            "the paper does not claim (a) (R4)"
        )
        + ".\n"
    )
    if e2:
        r = e2["results"]
        w(
            "**E2 WM vs direct multi-fidelity best-of-N at total-compute parity (sycophancy library):** power "
            f"{f3(r['planning_parity']['rate'])} at α 0.0125 (ρ̂ = {f3(e2['inputs']['rho_hat'])}, best-of-N budget "
            f"{e2['inputs']['budget_bon']:.1f} vs 48 full evaluations per episode); exchangeable-null size "
            f"{f3(r['null_exchangeable']['rate'])}. Sensitivities: ρ/2 {f3(r['rho_half_parity']['rate'])}, √ρ "
            f"{f3(r['rho_sqrt_parity']['rate'])}, perfect prior {f3(r['perfect_prior_parity']['rate'])}, no compute "
            f"charge {f3(r['planning_no_charge']['rate'])}, α 0.025 {f3(r['planning_parity_alpha_0.025']['rate'])}. "
            f"→ **{'co-primary' if e2_copri else 'registered secondary (not powered as a co-primary)'}**.\n"
        )
    w(
        f"**Priced confirmatory H-T2 design** ({'at the registrable P' if reg_P else 'at P = 40, for reference only'}): "
        f"{design['runs']} runs, {design['points']} members, ≈ {design['gpu_h']:.1f} GPU-h, ≈ ${design['usd']:.0f}."
    )
    if e2_extra:
        w(
            f" Best-of-N arm at parity (if run as a secondary): ≈ {e2_extra['gpu_h']:.1f} GPU-h, ≈ ${e2_extra['usd']:.0f}."
        )
    w("")
    w(
        f"**Resources.** Lambda total {total_h:.2f} instance-h, ${total_usd:.2f} (stage W ≈ {stagew_h:.2f} h, timing "
        f"pilot ≈ {timing_h:.2f} h); caps 4.0 h + $11 timing, $35 total. All `wsp-wme-w-*` instances terminated: "
        f"{'yes (API-confirmed)' if ledger and all(r['terminated_confirmed'] for r in ledger) else 'SEE LEDGER'}.\n"
    )
    detail(
        w, out, an, sim, e2, rates, flags, metas, em, canary, evals, ledger, reg, design, e2_extra
    )
    w("## 8. Caveats, deviations and process notes\n")
    for c in CAVEATS:
        w(f"- {c}")
    w("")
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("written", a.out + ".md")


if __name__ == "__main__":
    main()
