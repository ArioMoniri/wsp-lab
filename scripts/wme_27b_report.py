#!/usr/bin/env python3
"""Builds the ENGINEERING ONLY 27B calibration report (md + JSON) from the run directory: P1 (``attempt1/calib.json``),
the P2 configurations (``calib_p2.json`` single 4-layer group at the 68 GB ceiling, ``calib_p2_g2.json`` two 2-layer
groups, ``calib_p2_g4diag.json`` the diagnostic peak with the device stop raised), the dead-man terminator's
``state.json`` / ``actions.jsonl`` and the price snapshot. No objective value is read: panel results exist only as
content hashes.

    python scripts/wme_27b_report.py --run data/wme/calib27/calib27-20261116 --date 2026-11-16
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import calib27 as K

VISITS, N_GEN, N_GROUPS4 = (
    960,
    96,
    16,
)  # D4 bank; 10 % generation subset; 64 layers as 16 four-layer units


def _j(p):
    return json.loads(Path(p).read_text())


def _resource_stop(log: Path):
    m = re.search(r"RESOURCE STOP rss=(\d+) dev=(\d+)", log.read_text())
    return None if m is None else {"rss_bytes": int(m.group(1)), "device_bytes": int(m.group(2))}


def lambda_summary(run: Path):
    st = _j(run / "state.json")
    acts = [json.loads(x) for x in (run / "actions.jsonl").read_text().splitlines()]
    inst = []
    for iid, r in st["instances"].items():
        evs = [a for a in acts if a.get("id") == iid or iid in a.get("ids", [])]
        term = [a for a in evs if a["event"] == "terminate"]
        conf = [a for a in evs if a["event"] == "confirmed_terminated"]
        inst.append(
            {
                "name": r["name"],
                "type": r["type"],
                "region": r["region"],
                "price_usd_h": r["price_usd_h"],
                "hours": (r["confirmed_terminated_at"] - r["launched_at"]) / 3600,
                "terminated_by": "dead-man terminator" if term else "provider (failed boot)",
                "terminate_reasons": term[0]["reasons"] if term else [],
                "terminate_utc": term[0]["utc"] if term else None,
                "confirmed_utc": conf[0]["utc"] if conf else None,
                "confirmed_status": conf[0].get("status") if conf else None,
            }
        )
    inst.sort(key=lambda x: x["name"])
    hours = sum(i["hours"] for i in inst)
    usd = sum(i["hours"] * i["price_usd_h"] for i in inst)
    timeline = [
        f"{a['utc']} {a['event']}"
        + (f" {a['name']}" if "name" in a else "")
        + (f" reasons={a['reasons']}" if "reasons" in a else "")
        for a in acts
        if a["event"]
        in ("launch_request", "launched", "terminate", "confirmed_terminated", "exit", "alert")
    ]
    return {
        "instances": inst,
        "instance_hours": hours,
        "cost_usd": usd,
        "alerts": [a for a in acts if a["event"] == "alert"],
        "refusals": [a for a in acts if a["event"] == "refused"],
        "timeline": timeline,
        "accounting": "conservative: from the launch request to the API-confirmed termination",
    }


def build(run: Path, boot_s: float):
    a1 = _j(run / "attempt1/calib.json")
    p2_4 = _j(run / "calib_p2.json")
    p2_2 = _j(run / "calib_p2_g2.json")
    p2_d = _j(run / "calib_p2_g4diag.json")
    prices = {int(k): v for k, v in _j(run / "prices.json").items()}
    setup_s = float((run / "setup_seconds").read_text())
    stop1 = _resource_stop(run / "attempt1/calib.log")
    p1, ld = a1["p1"], a1["load"]
    g2 = p2_2["p2"]
    # Go-M on the configuration that completes: P1 (attempt 1) + P2 as two 2-layer groups (cap 768)
    dev = max(p1["memory"]["peak_device_bytes"], p2_2["global_peak"]["dev"])
    rss = max(p1["memory"]["peak_rss_bytes"], p2_2["global_peak"]["rss"])
    swap = max(p1["memory"]["peak_swap_bytes"], p2_2["global_peak"]["swap"])
    repeats = {
        "base": p1["base"]["repeat_bitwise"],
        "edit": p1["timing_edits"][0]["repeat_bitwise"],
        "restore_hash": p1["restore_hash_ok"],
        "post_restore": p1["post_restore"]["bitwise_equal_base"],
    }
    panel_s = list(p1["base"]["seconds"]) + [
        s for e in p1["timing_edits"] for s in e["panel_seconds"]
    ]
    panel_s.append(p1["post_restore"]["seconds"])
    gates = K.gates(
        {
            "peak_device_bytes": dev,
            "peak_rss_bytes": rss,
            "peak_swap_bytes": swap,
            "repeats": repeats,
            "panel_seconds_per_state": panel_s,
        }
    )
    gates["Go-M"]["configuration"] = (
        "P1 lean panel + P2 as two 2-layer groups, field items <= 768 tokens"
    )
    gates["Go-M (single 4-layer group)"] = {
        "rule": "peak device memory <= 72 GB (C9)",
        "attempt1_resource_stop_device_gb": stop1["device_bytes"] / 1e9 if stop1 else None,
        "diagnostic_peak_device_gb": p2_d["p2"]["memory"]["peak_device_bytes"] / 1e9,
        "pass": False,
    }
    t_ll = statistics.median(p1["all_in_seconds_per_state"])
    t_gen = (
        p1["generation"]["seconds"] * 4
    )  # 32 prompts x 128 tokens from the 8-prompt batch (9B convention)
    t_unit = g2["collect_seconds"]  # 4 layers (as two 2-layer passes), 6 objectives x 24 items
    bank = {
        "formula": "H = (visits*t_ll + N_gen*t_gen + T_features + T_train) / 3600",
        "visits": VISITS,
        "t_ll": t_ll,
        "n_gen": N_GEN,
        "t_gen": t_gen,
        "t_features": N_GROUPS4 * t_unit,
        "t_train": 0.0,
    }
    bank["gpu_hours"] = K.bank_hours(VISITS, t_ll, N_GEN, t_gen, bank["t_features"], 0.0)
    bank["usd_1gpu"] = bank["gpu_hours"] * prices[1]
    bank["usd_per_state_1gpu"] = t_ll / 3600 * prices[1]
    proj = K.dp_projection(
        VISITS, t_ll, boot_s + setup_s, ld["load_seconds"], t_unit, N_GROUPS4, prices, N_GEN, t_gen
    )
    p2_runs = [
        {
            "run": "attempt 1",
            "groups": "1 x 4 layers",
            "allocator_ceiling_gb": 70,
            "field_cap_tokens": 1024,
            "outcome": "monitor stop: device {:.2f} GB > 72 GB in the preflight (sycophancy item, 986 tokens)".format(
                stop1["device_bytes"] / 1e9
            )
            if stop1
            else "?",
        },
        {
            "run": "attempt 2",
            "groups": "1 x 4 layers",
            "allocator_ceiling_gb": 68,
            "field_cap_tokens": "1024 / 768 / 512",
            "outcome": "allocator OOM at every cap (1024: preflight sycophancy; 768 and 512: main pass, refusal margin)",
        },
        {
            "run": "attempt 3",
            "groups": "2 x 2 layers",
            "allocator_ceiling_gb": 68,
            "field_cap_tokens": "1024 / 768",
            "outcome": "1024: allocator OOM (main pass, sycophancy); 768: complete, {:.0f} s, peak device {:.2f} GB".format(
                g2["collect_seconds"], p2_2["global_peak"]["dev"] / 1e9
            ),
        },
        {
            "run": "diagnostic",
            "groups": "1 x 4 layers",
            "allocator_ceiling_gb": 76,
            "field_cap_tokens": 1024,
            "outcome": "complete with the device stop raised to 79 GB: {:.0f} s, peak device {:.2f} GB (Go-M FAIL)".format(
                p2_d["p2"]["collect_seconds"], p2_d["p2"]["memory"]["peak_device_bytes"] / 1e9
            ),
        },
    ]
    diag_peaks = {
        k.split(".")[0]: v["gpu_peak_allocated_bytes"] / 1e9
        for k, v in p2_d["p2"]["timing"].items()
    }
    return {
        "label": "ENGINEERING ONLY",
        "scope": "feasibility only: no trajectory, slope, headroom or enhancement quantity; objective values only as hashes",
        "target": {
            "repo": ld["repo"],
            "revision": ld["revision"],
            "base_hash": ld["base_hash"],
            "model_class": ld["model_class"],
            "role_prefix_loaded": ld["role_prefix"],
            "dtypes": ld["dtypes"],
            "n_params": ld["n_params"],
            "param_bytes": ld["param_bytes"],
            "role_map": ld["role_map"],
            "vision_or_mtp_tensors_loaded": ld["vision_or_mtp_tensors_loaded"],
            "non_thinking_template": p1["chat_template_non_thinking"]["has_empty_think_block"],
        },
        "environment": a1["environment"],
        "kernels": a1["kernels"] | {"causal_conv1d_fallback_logged_by_transformers": True},
        "code_sha": {
            "p1": a1["code_sha"],
            "p2_attempt2": p2_4["code_sha"],
            "p2_g2_and_diag": p2_2["code_sha"],
        },
        "boot_seconds": boot_s,
        "setup_seconds": setup_s,
        "load_seconds": ld["load_seconds"],
        "gates": gates,
        "p1": {
            "n_items": p1["n_items"],
            "per_objective_seconds_first_pass": p1["per_objective_seconds_first_pass"],
            "base_seconds": p1["base"]["seconds"],
            "base_digest": p1["base"]["hash"]["digest"],
            "timing_edits": [
                {
                    k: e[k]
                    for k in (
                        "edit",
                        "tensor",
                        "apply_seconds",
                        "panel_seconds",
                        "restore_seconds",
                        "differs_from_base",
                    )
                }
                | {"repeat_bitwise": e.get("repeat_bitwise"), "digest": e["hash"]["digest"]}
                for e in p1["timing_edits"]
            ],
            "post_restore_seconds": p1["post_restore"]["seconds"],
            "all_in_seconds_per_state": p1["all_in_seconds_per_state"],
            "generation": p1["generation"],
            "memory": p1["memory"],
        },
        "p2": {
            "runs": p2_runs,
            "passing_configuration": {k: v for k, v in g2.items() if k != "groups"},
            "diagnostic_single_group": {k: v for k, v in p2_d["p2"].items() if k != "groups"},
            "diagnostic_peak_allocated_gb_by_objective": diag_peaks,
            "host": {"global_peak_rss_bytes": p2_2["global_peak"]["rss"]},
        },
        "bank": bank,
        "dp_projection": {str(k): v for k, v in proj.items()},
        "prices_usd_h": prices,
        "capacity_snapshot": _j(run / "capacity_snapshot.json"),
        "lambda": lambda_summary(run),
    }


def gb(x):
    return f"{x / 1e9:.2f} GB"


def markdown(r, date, mock_tests):
    g, b, p1, p2, lam, t, env = (
        r["gates"],
        r["bank"],
        r["p1"],
        r["p2"],
        r["lambda"],
        r["target"],
        r["environment"],
    )
    ok = lambda v: "PASS" if v else "FAIL"
    pc = p2["passing_configuration"]
    dg = p2["diagnostic_single_group"]
    fbytes = pc["factor_bytes_per_module_fp32"]
    main = next(i for i in lam["instances"] if i["terminated_by"] == "dead-man terminator")
    failed = [i for i in lam["instances"] if i["terminated_by"] != "dead-man terminator"]
    edits = "\n".join(
        f"| {e['edit']} | `{e['tensor']}` | {e['apply_seconds']:.2f} | "
        f"{', '.join(f'{s:.1f}' for s in e['panel_seconds'])} | {e['restore_seconds']:.2f} | "
        f"{'—' if e['repeat_bitwise'] is None else ('yes' if e['repeat_bitwise'] else 'NO')} | "
        f"{'yes' if e['differs_from_base'] else 'NO'} | `{e['digest'][:12]}…` |"
        for e in p1["timing_edits"]
    )
    obj = ", ".join(f"{k} {v:.1f} s" for k, v in p1["per_objective_seconds_first_pass"].items())
    runs = "\n".join(
        f"| {x['run']} | {x['groups']} | {x['allocator_ceiling_gb']} | {x['field_cap_tokens']} | {x['outcome']} |"
        for x in p2["runs"]
    )
    proj = "\n".join(
        f"| {k} | ${v['usd_per_h']:.2f} | {v['wall_hours']:.2f} | {v['gpu_hours']:.2f} | ${v['usd']:.2f} |"
        for k, v in r["dp_projection"].items()
    )
    per_obj = ", ".join(f"{k} {v['seconds']:.1f} s" for k, v in pc["timing"].items())
    diag = ", ".join(
        f"{k} {v:.2f}" for k, v in p2["diagnostic_peak_allocated_gb_by_objective"].items()
    )
    timeline = "\n".join(f"  - {x}" for x in lam["timeline"])
    return f"""# WM-E on Qwen3.8-27B: engineering calibration on one Lambda H100, {date}

> **ENGINEERING ONLY.** These are feasibility measurements for SYNTHESIS-r9 C9 (round-9 pro-engineer L1–L4): load
> time, memory, lean-panel seconds per state, bitwise repeats, and grouped support-only gradient and K-FAC timing
> and memory on 4 layers.
>
> **Not computed:** any trajectory, slope, headroom or enhancement quantity. There were no dose ladders, no
> candidate search and no dev or test objective comparisons.
>
> **Hash-only scoring:** the base and two fixed timing edits were scored only to time the panel and check repeats.
> Their objective values appear only as content hashes.
>
> **C9 start conditions:** they are not met (T2/E2 on 9B, `docs/PREREG_WME.md`), and nothing here bears on them.
>
> **Untouched:** pilot 2, Part A, A5, F5.5, F8, F9 and F12-I.

- **Machine-readable:** `wme-27b-calib-{date}.json`, in this directory.
- **Raw artefacts** (not committed): `data/wme/calib27/calib27-20261116/`.
- **Branch:** `ops/wme-27b-calib`.
- **WM-E code:** as on main `fbceea6`, plus `wsp.wm.calib27`, the 27B pin in `wsp.wm.hf`, the `wsp.ops` terminator
  and the scripts.
- **Run commits:** P1 at `{r["code_sha"]["p1"]}`; P2 at `{r["code_sha"]["p2_attempt2"]}` and `{r["code_sha"]["p2_g2_and_diag"]}`.
  The `wsp.wm` modules other than `calib27`/`hf` are identical to main in all three.

## BLUF

- **Go-M: {ok(g["Go-M"]["pass"])}, but only with the field split into two 2-layer groups and items ≤ 768 tokens.**
  - Peak device memory was {g["Go-M"]["peak_device_gb"]:.2f} GB against the 72 GB cap.
  - Host RSS peaked at {g["Go-M"]["peak_rss_gb"]:.1f} GB of {env["mem_total_bytes"] / 1e9:.0f} GB, with no swap.
    This includes the memory-mapped shard pages during load; after the load, P1 peaked at {gb(p1["memory"]["peak_rss_bytes"])}.
  - **One 4-layer group fails Go-M.** Attempt 1 was stopped at {g["Go-M (single 4-layer group)"]["attempt1_resource_stop_device_gb"]:.2f} GB.
    The diagnostic run, with the device stop raised, peaked at {g["Go-M (single 4-layer group)"]["diagnostic_peak_device_gb"]:.2f} GB.
  - **Why:** the bf16 weights take {gb(t["param_bytes"])}. The fp32 accumulators for one 4-layer group take about
    8.2 GB: {gb(4 * (fbytes["model.layers.24.mlp.down_proj"] + fbytes["model.layers.27.self_attn.o_proj"]))} of A/G
    factors plus about 1.9 GB of gradients. That leaves only a few GB for activations and logits under 72 GB.
- **Go-N: {ok(g["Go-N"]["pass"])}.** The base repeat, the edited-state repeat, the restore θ-hash and the
  post-restore full panel were all bitwise identical.
- **Go-C: {ok(g["Go-C"]["pass"])}.** The lean panel (6 objectives × 64 items) took
  {min(g["Go-C"]["seconds_per_state"]):.1f}–{g["Go-C"]["max_seconds"]:.1f} s per state against the 240 s cap.
  - Warm, it took about 15.9 s; the maximum is the first full-panel pass.
  - One visit (apply, panel, restore) had a median of {b["t_ll"]:.1f} s.
- **Load and boot:** the load took {r["load_seconds"]:.0f} s (the shards were in the page cache after the download).
  Boot took {r["boot_seconds"]:.0f} s, and the venv plus the 55.6 GB download took {r["setup_seconds"]:.0f} s.
- **P2, passing configuration:** {pc["collect_seconds"]:.0f} s.
  - Scope: 4 layers (24–27) as 2 × 2-layer groups, 6 objectives × 24 items, ≤ {pc["field_max_tokens"]} tokens.
  - It wrote {gb(pc["disk_bytes_written"])} of fp32 gradients and factors.
  - The CPU fp64 spectral audit of one 17408² factor took {pc["spectral_factor_seconds_ffn"]:.0f} s.
- **27B bank by the D8 H formula:** {b["gpu_hours"]:.1f} GPU-h on one H100, about ${b["usd_1gpu"]:.0f} at
  ${r["prices_usd_h"][1]:.2f}/h, or ${b["usd_per_state_1gpu"]:.3f} per state.
  - With 8 GPUs the projection is {r["dp_projection"]["8"]["wall_hours"]:.2f} h wall for ${r["dp_projection"]["8"]["usd"]:.0f} (§6).
- **Lambda usage:**
  - Instances: {len(lam["instances"])} × `{main["type"]}` in `{main["region"]}`.
  - Usage: **{lam["instance_hours"]:.3f} instance-hours, ${lam["cost_usd"]:.2f}**, against caps of 3 h and $20.
  - The main instance (`{main["name"]}`) was **terminated by the dead-man terminator** (reason
    `{", ".join(main["terminate_reasons"])}`, the real-instance test). Termination was **confirmed through the API**:
    status `{main["confirmed_status"]}` at {main["confirmed_utc"]}.
  - {("The first instance (`" + failed[0]["name"] + "`) was terminated by the provider during boot, after " + f"{failed[0]['hours'] * 60:.1f}" + " min. That was confirmed too; the terminator issued no terminate call.") if failed else ""}

## 1. Target and environment

| Item | Value |
|---|---|
| Model | `{t["repo"]}` @ `{t["revision"]}` (HF `sha` on 16 Nov; lastModified 2026-11-07) |
| Class / dtype | `{t["model_class"]}` (text-only CausalLM), {", ".join(t["dtypes"])}; {t["n_params"] / 1e9:.2f} B parameters, {gb(t["param_bytes"])} |
| θ₀ identity | `{t["base_hash"][:16]}…` (repo, revision, LFS sha256 of all 18 shards) |
| Text-only | vision and MTP tensors loaded: {t["vision_or_mtp_tensors_loaded"]} (and frozen by `wm.actions.is_frozen`) |
| Non-thinking | chat template with `enable_thinking=False` emits the empty think block: {t["non_thinking_template"]} |
| Role map, derived from config | {t["role_map"]["n_layers"]} layers ({t["role_map"]["n_linear_attention"]} GatedDeltaNet + {t["role_map"]["n_full_attention"]} gated attention, every 4th), width {t["role_map"]["hidden"]}, FFN {t["role_map"]["ffn"]}, vocab {t["role_map"]["vocab"]}, untied. All {t["role_map"]["writers_checked"]} writers (`o`, `down`) exist with the config-implied shapes; mismatches: {len(t["role_map"]["mismatches"])}. Loaded prefix `{t["role_prefix_loaded"]}`; the checkpoint-name prefix `model.language_model.layers.` is checked offline (`tests/test_wm_27b_calib.py`). |
| GPU, host | {env["gpu"]}; {env["cpu_count"]} vCPU, {env["mem_total_bytes"] / 1e9:.0f} GB RAM |
| Software | Python {env["python"]}, torch {env["torch"]} (CUDA {env["torch_cuda"]}), transformers {env["transformers"]}, fla {env["packages"]["fla"]}, triton {env["packages"]["triton"]}, accelerate {env["packages"]["accelerate"]}, safetensors {env["packages"]["safetensors"]}, numpy {env["packages"]["numpy"]}. These are the AWS `.venv` versions; the other packages are pinned to its freeze. |
| Kernels executed | fla gated-delta-rule: {r["kernels"]["fla_used"]}. **causal-conv1d: not available**; transformers logged its PyTorch fallback (see §8). |

## 2. Gates (memory cap from C9; Go-N and Go-C from D8)

| Gate | Passes when | Measured | Result |
|---|---|---|---|
| Go-M | device ≤ 72 GB, no swap | {g["Go-M"]["peak_device_gb"]:.2f} GB device; {g["Go-M"]["peak_rss_gb"]:.1f} GB RSS; swap {g["Go-M"]["peak_swap_bytes"]} B. Configuration: {g["Go-M"]["configuration"]} | {ok(g["Go-M"]["pass"])} |
| Go-M, single 4-layer group | device ≤ 72 GB | stopped at {g["Go-M (single 4-layer group)"]["attempt1_resource_stop_device_gb"]:.2f} GB; diagnostic peak {g["Go-M (single 4-layer group)"]["diagnostic_peak_device_gb"]:.2f} GB | FAIL |
| Go-N | base, edit and restore repeats bitwise | {", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in g["Go-N"]["repeats"].items())} | {ok(g["Go-N"]["pass"])} |
| Go-C | lean panel ≤ 240 s per state | max {g["Go-C"]["max_seconds"]:.1f} s over {len(g["Go-C"]["seconds_per_state"])} full-panel passes | {ok(g["Go-C"]["pass"])} |

## 3. P1: panel timing and repeats (validation split, 6 × 64 items, `PanelConfig()` as on 9B)

- **First pass per objective** (includes Triton autotune): {obj}.
- **Base panel:** {", ".join(f"{s:.1f}" for s in p1["base_seconds"])} s; base digest `{p1["base_digest"][:12]}…`.
- **Post-restore panel:** {p1["post_restore_seconds"]:.1f} s, bitwise equal to base.

Timing edits (values hashed, never differenced):

| Edit | Tensor | apply s | panel s | restore s | repeat bitwise | output differs from base | digest |
|---|---|---|---|---|---|---|---|
{edits}

- **Greedy generation, non-thinking:** {p1["generation"]["prompts"]} prompts × {p1["generation"]["max_new_tokens"]} tokens took
  {p1["generation"]["seconds"]:.1f} s (batch {p1["generation"]["batch"]}, {p1["generation"]["new_tokens"]} new tokens).
  - This is slower than 9B (5.45 s), probably because decoding uses the PyTorch `causal_conv1d_update` fallback.
- **P1 peak:** device {gb(p1["memory"]["peak_device_bytes"])}, allocator {gb(p1["memory"]["peak_allocated_bytes"])}.

## 4. P2: grouped support-only gradients and K-FAC on layers 24–27

**Group.** The group mirrors the 9B layout: 3 GatedDeltaNet `out_proj` + 1 attention `o_proj`, plus 4 `down_proj`.
It uses 6 objectives × 24 construction items and model-sampled Fisher.

**Preflight.** Each run starts with a longest-first (H4) preflight on one item per objective.

| Run | Groups | Allocator ceiling (GB) | Field-item cap (tokens) | Outcome |
|---|---|---|---|---|
{runs}

**Passing configuration** (2 × 2 layers, ≤ {pc["field_max_tokens"]} tokens):
- **Time:** {pc["collect_seconds"]:.1f} s, including the factor writes.
- **Per (objective, group), compute only:** {per_obj}.
- **Memory:**
  - preflight peak device {gb(pc["preflight"]["memory"]["peak_device_bytes"])};
  - main peak device {gb(pc["memory"]["peak_device_bytes"])}, allocator {gb(pc["memory"]["peak_allocated_bytes"])}.
- **Disk:** {gb(pc["disk_bytes_written"])} written, then deleted.
- **fp32 A+G per module:**
  - `down_proj` {gb(fbytes["model.layers.24.mlp.down_proj"])};
  - `out_proj`/`o_proj` {gb(fbytes["model.layers.27.self_attn.o_proj"])}.
- **Unsupported tensors:** {pc["coverage_unsupported"] or "none in this group"}.

**Single 4-layer group, diagnostic** (1024 tokens; Go-M FAIL):
- **Time:** {dg["collect_seconds"]:.1f} s; this is faster because it makes one pass instead of two.
- **Peak allocated by objective (GB):** {diag}.
- **What the peaks mean:** the refusal objectives need 68.4–69.7 GB of allocator memory, which is about 71–72.4 GB
  of device memory at the measured ~2.7 GB non-allocator overhead.
  - Their items are at most 570 tokens, so the token cap barely affects them.
  - At the 68 GB ceiling, caps of 1024, 768 and 512 all hit an OOM (attempt 2).
  - One 4-layer group therefore has no margin under the 72 GB cap.

**Spectral audit.** The CPU fp64 spectral audit of one {pc["spectral_factor_shape"][0]}² `down_proj` A factor took
{pc["spectral_factor_seconds_ffn"]:.1f} s on {env["cpu_count"]} vCPU. On 9B, a 12288² factor took 183 s on the
AWS host.

## 5. Bank price for 27B (SYNTHESIS-r8 D8 formula)

H = (visits·t_ll + N_gen·t_gen + T_features + T_train) / 3600

= ({b["visits"]} × {b["t_ll"]:.1f} + {b["n_gen"]} × {b["t_gen"]:.1f} + {b["t_features"]:.0f} + 0) / 3600

= **{b["gpu_hours"]:.2f} GPU-h** on one H100, about **${b["usd_1gpu"]:.2f}**, or ${b["usd_per_state_1gpu"]:.3f} per state.

- **t_ll** is the median all-in visit (apply, panel, restore) of the two timing edits.
- **t_gen** is 32 prompts × 128 tokens, ×4 from the measured 8-prompt batch (the 9B convention).
- **T_features** is the base field for all 64 layers as 16 four-layer units, each run as two 2-layer passes at the
  measured P2 time.
  - It uses ≤ 768-token field items, against ≤ 1024 on 9B.
  - Per-state oracle gradients are excluded because Go-R failed on 9B.
- **T_train = 0**, because training runs on CPU.
- **For comparison,** the 9B figure was 7.84 GPU-h on the local GPU.

## 6. k-GPU data-parallel projection (L1–L2; not built)

**Assumptions:**
- one bf16 model per 80 GB GPU;
- states and four-layer field units are queue items, with no inter-GPU communication;
- wall time = boot + setup + load + ⌈960/k⌉·t_ll + ⌈96/k⌉·t_gen + ⌈16/k⌉·t_unit, which is the D8 terms spread
  over k GPUs;
- the 9 s load is warm (page cache); a cold load from local NVMe is longer but small next to boot.

**Prices:** Lambda on-demand list prices for the k × H100 SXM5 instance at {r["capacity_snapshot"]["utc"]}. At that
snapshot, none of these sizes had capacity. The 1× size reappeared at 22:28Z. Capacity is the binding risk, not
price.

| k | $/h (instance) | wall h | GPU-h | $ |
|---|---|---|---|---|
{proj}

**Not built here:**
- the L2 fixed-order fp64 CPU reduction;
- its 1-vs-k parity test at rtol 1e-6.

States are scored whole on one GPU, so only item sharding would need that reduction.

## 7. Cost control: the dead-man terminator (L4)

**Code.** `src/wsp/ops/lambda_deadman.py` (`scripts/lambda_deadman.py`) runs on the AWS server under agent-tmux.

**Key handling.** It is the only holder of the API key. The key was never put on the GPU box, and it does not appear
in any log or state file.

**Guards:**
- It launches only `wsp-wme-*` names, at no more than $5.50/h, one live instance at a time.
- It terminates only ids it launched whose API name still matches.

**Termination triggers:**
- an idle limit of 15 min, driven by `scripts/wme_27b_hb.sh` (a heartbeat only while the remote logs change);
- a cumulative cap of 3 instance-hours;
- a cap of $20;
- a 5-min margin before either cap.

**Other behaviour:**
- It alerts when an instance is alive more than 15 min past T_max.
- It confirms each termination through the API.
- It logs every action.

**Mocked-API tests:** {mock_tests}.

**Real-instance test.** After the measurements were copied back:
- I stopped the heartbeat loop and tightened the idle limit to 60 s through `policy.json` (overrides can only
  tighten).
- The watcher then issued `terminate` (reasons `{main["terminate_reasons"]}`) at {main["terminate_utc"]}.
- It repeated the call once on the next tick, because the API still reported `terminating`. The retry is by design
  (an instance stuck in `terminating` keeps being retried), and both calls were for its own id.
- It saw status `{main["confirmed_status"]}` at {main["confirmed_utc"]}.
- A separate API query confirmed that no `wsp-wme-*` instance remains.

There were no alerts and no ownership refusals. Timeline (UTC):

{timeline}

## 8. Deviations and things not done

- **causal-conv1d.**
  - **What happened:** it was not built. The image's nvcc is CUDA 12.8 and torch is cu132, so the extension build
    aborted on the CUDA-version check.
  - **Effect:** transformers fell back to PyTorch for the short convolution in the 48 GatedDeltaNet layers. fla was
    used for the delta rule.
  - **Bias:** panel timings and especially generation are upper bounds. Memory is unaffected apart from small
    temporaries.
  - **Not tried:** a CUDA-13 nvcc from pip, to save instance time.
- **P2 configuration differs from 9B.**
  - 9B used one 4-layer group with field items ≤ 1024 tokens.
  - 27B needs two 2-layer groups and items ≤ 768 tokens to pass Go-M.
  - The other three P2 configurations are reported in §4.
- **Allocator ceiling.**
  - It was set to 70 GB in attempt 1 and then 68 GB.
  - About 2.7 GB of device memory sits outside the PyTorch allocator: CUDA 13 context and workspaces.
- **Instances.**
  - Two instances were used. The first was terminated by the provider during boot, and its 0.049 h is counted.
  - Capacity seen at my first check earlier in the session was gone at the first launch attempt (21:50Z); the
    guarded launch refused. A guarded poller launched when capacity returned at 22:28Z.
- **Boot time.** It was about 12 min (22:38Z to 22:50Z); the projection includes it.
- **Not done (out of scope or not built):**
  - the k-GPU implementation and its parity test (L1/L2);
  - L3 contract-and-discard;
  - the tap;
  - the generation endpoint;
  - any scientific gate.
- **C9 budget.** C9 sets ≤ 2 Lambda GPU-h for calibration; this used {lam["instance_hours"]:.2f}. The user's cap was
  3 instance-hours and $20.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--boot-seconds", type=float, required=True)
    ap.add_argument("--mock-tests", required=True)
    a = ap.parse_args()
    r = build(Path(a.run), a.boot_seconds)
    r["lambda"]["mock_tests"] = a.mock_tests
    out = ROOT / "docs/reviews/wme"
    (out / f"wme-27b-calib-{a.date}.json").write_text(
        json.dumps(r, indent=1, sort_keys=True, default=str)
    )
    (out / f"wme-27b-calib-{a.date}.md").write_text(markdown(r, a.date, a.mock_tests))
    print(json.dumps({k: v["pass"] for k, v in r["gates"].items()}))


if __name__ == "__main__":
    main()
