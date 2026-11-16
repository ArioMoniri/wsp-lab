#!/usr/bin/env python3
"""Builds the ENGINEERING ONLY 27B calibration report (md + JSON) from the instance's ``calib.json``, the dead-man
terminator's state/log and the price snapshot. No objective value is read or written: panel results exist only as
content hashes in ``calib.json``.

    python scripts/wme_27b_report.py --run data/wme/calib27/<run> --date 2026-11-16
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsp.wm import calib27 as K  # noqa: E402

VISITS, N_GEN, N_GROUPS = 960, 96, 16  # D4 bank; 10 % generation subset; 64 layers / 4 per group


def build(run: Path):
    c = json.loads((run / "calib.json").read_text())
    dm = json.loads((run / "deadman.json").read_text())
    prices = {int(k): v for k, v in json.loads((run / "prices.json").read_text()).items()}
    setup_s = float((run / "setup_seconds").read_text())
    p1, p2, ld = c["p1"], c["p2"], c["load"]
    gp = c["global_peak"]
    repeats = {
        "base": p1["base"]["repeat_bitwise"],
        "edit": p1["timing_edits"][0]["repeat_bitwise"],
        "restore_hash": p1["restore_hash_ok"],
        "post_restore": p1["post_restore"]["bitwise_equal_base"],
    }
    panel_s = list(p1["base"]["seconds"]) + [s for e in p1["timing_edits"] for s in e["panel_seconds"]]
    panel_s.append(p1["post_restore"]["seconds"])
    gates = K.gates(
        {
            "peak_device_bytes": gp["dev"],
            "peak_rss_bytes": gp["rss"],
            "peak_swap_bytes": gp["swap"],
            "repeats": repeats,
            "panel_seconds_per_state": panel_s,
        }
    )
    t_ll = statistics.median(p1["all_in_seconds_per_state"])
    t_gen = p1["generation"]["seconds"] * 4  # 32 prompts x 128 tokens from the 8-prompt batch (9B convention)
    t_group = p2["collect_seconds"]
    bank = {
        "formula": "H = (visits*t_ll + N_gen*t_gen + T_features + T_train) / 3600",
        "visits": VISITS,
        "t_ll": t_ll,
        "n_gen": N_GEN,
        "t_gen": t_gen,
        "t_features": N_GROUPS * t_group,
        "t_train": 0.0,
        "gpu_hours": K.bank_hours(VISITS, t_ll, N_GEN, t_gen, N_GROUPS * t_group, 0.0),
        "usd_per_state_1gpu": t_ll / 3600 * prices[1],
    }
    bank["usd_1gpu"] = bank["gpu_hours"] * prices[1]
    proj = K.dp_projection(VISITS, t_ll, setup_s, ld["load_seconds"], t_group, N_GROUPS, prices)
    return {
        "label": "ENGINEERING ONLY",
        "target": {"repo": ld["repo"], "revision": ld["revision"], "base_hash": ld["base_hash"],
                   "model_class": ld["model_class"], "dtypes": ld["dtypes"], "n_params": ld["n_params"],
                   "param_bytes": ld["param_bytes"], "role_map": ld["role_map"],
                   "vision_or_mtp_tensors_loaded": ld["vision_or_mtp_tensors_loaded"],
                   "non_thinking_template": p1["chat_template_non_thinking"]["has_empty_think_block"]},
        "environment": c["environment"],
        "kernels": c["kernels"],
        "code_sha": c["code_sha"],
        "setup_seconds": setup_s,
        "load_seconds": ld["load_seconds"],
        "gates": gates,
        "global_peak": gp,
        "p1": {
            "n_items": p1["n_items"],
            "per_objective_seconds_first_pass": p1["per_objective_seconds_first_pass"],
            "base_seconds": p1["base"]["seconds"],
            "timing_edits": [
                {k: e[k] for k in ("edit", "tensor", "apply_seconds", "panel_seconds", "restore_seconds",
                                   "differs_from_base")}
                | {"repeat_bitwise": e.get("repeat_bitwise"), "digest": e["hash"]["digest"]}
                for e in p1["timing_edits"]
            ],
            "base_digest": p1["base"]["hash"]["digest"],
            "post_restore_seconds": p1["post_restore"]["seconds"],
            "all_in_seconds_per_state": p1["all_in_seconds_per_state"],
            "generation": p1["generation"],
            "memory": p1["memory"],
        },
        "p2": {k: p2[k] for k in p2 if k not in ("timing",)} | {"timing": p2["timing"]},
        "bank": bank,
        "dp_projection": {str(k): v for k, v in proj.items()},
        "lambda": dm,
        "wall_seconds_calib_process": c["wall_seconds"],
    }


def gb(x):
    return f"{x / 1e9:.2f} GB"


def markdown(r, date):
    g, b, p1, p2, lam = r["gates"], r["bank"], r["p1"], r["p2"], r["lambda"]
    rows = "\n".join(
        f"| {k} | {v['rule']} | "
        + (
            f"device {v['peak_device_gb']:.2f} GB; RSS {v['peak_rss_gb']:.2f} GB; swap {v['peak_swap_bytes']} B"
            if k == "Go-M"
            else ", ".join(f"{a} {'yes' if b_ else 'NO'}" for a, b_ in v["repeats"].items())
            if k == "Go-N"
            else f"max {v['max_seconds']:.1f} s over {len(v['seconds_per_state'])} panel passes"
        )
        + f" | {'PASS' if v['pass'] else 'FAIL'} |"
        for k, v in g.items()
    )
    proj = "\n".join(
        f"| {k} | ${v['usd_per_h']:.2f} | {v['wall_hours']:.2f} | {v['gpu_hours']:.2f} | ${v['usd']:.2f} |"
        for k, v in r["dp_projection"].items()
    )
    edits = "\n".join(
        f"| {e['edit']} | `{e['tensor']}` | {e['apply_seconds']:.2f} | "
        f"{', '.join(f'{s:.1f}' for s in e['panel_seconds'])} | {e['restore_seconds']:.2f} | "
        f"{'—' if e['repeat_bitwise'] is None else ('yes' if e['repeat_bitwise'] else 'NO')} | "
        f"{'yes' if e['differs_from_base'] else 'NO'} | `{e['digest'][:12]}…` |"
        for e in p1["timing_edits"]
    )
    obj = "\n".join(f"| {k} | {v:.1f} |" for k, v in p1["per_objective_seconds_first_pass"].items())
    t = r["target"]
    env = r["environment"]
    return f"""# WM-E on Qwen3.8-27B: engineering calibration on one Lambda H100, {date}

> **ENGINEERING ONLY.** Feasibility measurements for SYNTHESIS-r9 C9 (round-9 pro-engineer L1–L4): load time,
> memory, lean-panel seconds per state, bitwise repeats, and grouped support-only gradient/K-FAC timing and memory
> on 4 layers. **No** trajectory, slope, headroom or enhancement quantity was computed; there were no dose ladders,
> no candidate search, and no dev/test objective comparisons. The base and two fixed timing edits were scored only
> to time the panel and check repeats; their objective values appear **only as content hashes**. C9's scientific
> start conditions (T2/E2 on 9B, `docs/PREREG_WME.md`) are **not** met and nothing here bears on them. Pilot 2 was
> not touched. Part A, A5, F5.5, F8, F9 and F12-I are unchanged.

Machine-readable: `wme-27b-calib-{date}.json` (same directory). Code: `{r['code_sha'][:12]}` on branch
`ops/wme-27b-calib` (WM-E code as on main `fbceea6`, plus `wm.calib27`, the 27B revision pin and the scripts).

## BLUF

- **Go-M {'PASS' if g['Go-M']['pass'] else 'FAIL'}**: peak device {g['Go-M']['peak_device_gb']:.2f} GB
  (cap 72 GB), allocator peak {gb(r['global_peak']['alloc'])}, host RSS {g['Go-M']['peak_rss_gb']:.2f} GB, swap
  {g['Go-M']['peak_swap_bytes']} B. The peak is P2 (gradients + K-FAC on layers 24–27).
- **Go-N {'PASS' if g['Go-N']['pass'] else 'FAIL'}**: base repeat, edited-state repeat, restore θ-hash and
  post-restore full panel all bitwise identical.
- **Go-C {'PASS' if g['Go-C']['pass'] else 'FAIL'}**: lean panel (6 objectives × 64 items) {min(g['Go-C']['seconds_per_state']):.1f}–{g['Go-C']['max_seconds']:.1f} s per state
  (cap 240 s); all-in per visit (apply + panel + restore) median {b['t_ll']:.1f} s.
- **Load** {r['load_seconds']:.0f} s from local NVMe; instance setup (venv + 55.6 GB download) {r['setup_seconds']:.0f} s.
- **P2** (6 objectives × 24 items, 4 down_proj + 3 GatedDeltaNet out_proj + 1 attention o_proj):
  {p2['collect_seconds']:.0f} s, peak device {gb(p2['memory']['peak_device_bytes'])}, factors written
  {gb(p2['disk_bytes_written'])}.
- **Bank (D8 H formula)**: {b['gpu_hours']:.1f} GPU-h ≈ ${b['usd_1gpu']:.0f} on one H100 at ${lam['price_usd_h']:.2f}/h;
  ${b['usd_per_state_1gpu']:.3f} per state.
- **Lambda**: 1 × `{lam['type']}` in `{lam['region']}`, instance `{lam['name']}`; {lam['instance_hours']:.3f}
  instance-hours, ${lam['cost_usd']:.2f} (caps 3 h, $20). **Terminated** by the dead-man terminator
  ({', '.join(lam['terminate_reasons'])}) and confirmed through the API (status `{lam['final_api_status']}`).

## 1. Target and environment

| Item | Value |
|---|---|
| Model | `{t['repo']}` @ `{t['revision']}` |
| Class / dtype | `{t['model_class']}`, {', '.join(t['dtypes'])}; {t['n_params'] / 1e9:.2f} B parameters, {gb(t['param_bytes'])} |
| θ₀ identity | `{t['base_hash'][:16]}…` (repo, revision, LFS sha256 of all 18 shards) |
| Text-only | vision and MTP tensors loaded: {t['vision_or_mtp_tensors_loaded']} (frozen by `wm.actions.is_frozen`) |
| Non-thinking | chat template with `enable_thinking=False` emits the empty think block: {t['non_thinking_template']} |
| Role map (from config) | {t['role_map']['n_layers']} layers ({t['role_map']['n_linear_attention']} GatedDeltaNet + {t['role_map']['n_full_attention']} gated attention), width {t['role_map']['hidden']}, FFN {t['role_map']['ffn']}, vocab {t['role_map']['vocab']}, untied; {t['role_map']['writers_checked']} writers checked against loaded names and config-implied shapes, mismatches {len(t['role_map']['mismatches'])} |
| GPU | {env['gpu']} |
| Software | Python {env['python']}, torch {env['torch']} (CUDA {env['torch_cuda']}), transformers {env['transformers']}, fla {env['packages'].get('fla')}, causal-conv1d {env['packages'].get('causal_conv1d')} |
| Kernels executed | fla: {r['kernels']['fla_used']}; causal-conv1d: {r['kernels']['causal_conv1d_used']}; torch fallbacks: {r['kernels']['torch_fallbacks'] or 'none'} |

## 2. Gates (27B caps: C9 for memory, D8 for the rest)

| Gate | Passes when | Measured | Result |
|---|---|---|---|
{rows}

## 3. P1: panel timing and repeats

First pass per objective (includes warm-up/autotune):

| Objective | s |
|---|---|
{obj}

Base panel: {', '.join(f'{s:.1f}' for s in p1['base_seconds'])} s (bitwise repeat: {g['Go-N']['repeats']['base']}); base digest
`{p1['base_digest'][:12]}…`. Post-restore panel {p1['post_restore_seconds']:.1f} s, bitwise equal to base:
{g['Go-N']['repeats']['post_restore']}.

Timing edits (values hashed, never differenced):

| Edit | Tensor | apply s | panel s | restore s | repeat bitwise | output differs from base | digest |
|---|---|---|---|---|---|---|---|
{edits}

Greedy generation, non-thinking: {p1['generation']['prompts']} prompts × {p1['generation']['max_new_tokens']} tokens in
{p1['generation']['seconds']:.2f} s (batch {p1['generation']['batch']}, {p1['generation']['new_tokens']} new tokens).
P1 peak: device {gb(p1['memory']['peak_device_bytes'])}, allocator {gb(p1['memory']['peak_allocated_bytes'])}.

## 4. P2: grouped support-only gradients and K-FAC (layers 24–27)

- Group: {', '.join(f'`{m}`' for m in p2['group'])}.
- Longest-first preflight (one longest admissible item per objective, H4): {p2['preflight']['seconds']:.1f} s,
  peak device {gb(p2['preflight']['memory']['peak_device_bytes'])}, allocator {gb(p2['preflight']['memory']['peak_allocated_bytes'])};
  longest items (tokens) {p2['preflight']['longest_tokens']}.
- Main: {p2['items_per_objective']} construction items per objective, ≤ {p2['field_max_tokens']} tokens:
  **{p2['collect_seconds']:.0f} s** including writing {gb(p2['disk_bytes_written'])} of fp32 gradients and factors;
  peak device {gb(p2['memory']['peak_device_bytes'])}, allocator {gb(p2['memory']['peak_allocated_bytes'])},
  reserved {gb(p2['memory']['peak_reserved_bytes'])}, RSS {gb(p2['memory']['peak_rss_bytes'])}.
- Per objective: {', '.join(f"{k} {v['seconds']:.0f} s" for k, v in p2['timing'].items()) if isinstance(p2['timing'], dict) and all(isinstance(v, dict) and 'seconds' in v for v in p2['timing'].values()) else p2['timing']}.
- fp32 factor bytes per module: {', '.join(f"{m.split('layers.')[1]} {gb(v)}" for m, v in p2['factor_bytes_per_module_fp32'].items())}.
- Unsupported (marked, q = None): {p2['coverage_unsupported'] or 'none in this group'}.
{f"- CPU fp64 spectral audit of one {p2['spectral_factor_shape'][0]}² down_proj A factor: {p2['spectral_factor_seconds_ffn']:.0f} s." if 'spectral_factor_seconds_ffn' in p2 else ''}

## 5. Bank price for 27B (SYNTHESIS-r8 D8 formula)

H = (visits·t_ll + N_gen·t_gen + T_features + T_train) / 3600 = ({b['visits']} × {b['t_ll']:.1f} + {b['n_gen']} × {b['t_gen']:.1f} +
{b['t_features']:.0f} + 0) / 3600 = **{b['gpu_hours']:.2f} GPU-h** (one H100), ≈ **${b['usd_1gpu']:.2f}**; per state
${b['usd_per_state_1gpu']:.3f}.

- t_ll = median all-in visit (apply + panel + restore) of the two timing edits.
- t_gen = 32 prompts × 128 tokens, scaled ×4 from the measured 8-prompt batch (9B convention).
- T_features = base field for all 64 layers as 16 four-layer groups × the measured P2 time. Per-state oracle
  gradients are excluded (Go-R failed on 9B). T_train = 0 (CPU).
- The 9B equivalent was 7.84 GPU-h on the local GPU (pilot report).

## 6. k-GPU data-parallel projection (L1–L2; not built)

One bf16 model per 80 GB GPU, states and field groups as queue items, no inter-GPU communication; wall =
setup + load + ⌈960/k⌉·t_ll + ⌈16/k⌉·t_group. Prices are Lambda on-demand list prices for the k × H100 SXM5
instance at the time of the run; capacity was not available for every k (see JSON).

| k | $/h (instance) | wall h | GPU-h | $ |
|---|---|---|---|---|
{proj}

The L2 fixed-order fp64 CPU reduction and its 1-vs-k rtol 1e-6 parity test are not built here; data-parallel over
states needs no reduction for panel outputs (each state is scored whole on one GPU), only for sharded items.

## 7. Cost control: dead-man terminator (L4)

- `src/wsp/ops/lambda_deadman.py` + `scripts/lambda_deadman.py`, run on the AWS server under agent-tmux; the only
  holder of the API key (never on the GPU box, never logged). Launches only `wsp-wme-*` names; terminates only ids
  it launched whose API name still matches; idle limit 15 min (progress-driven heartbeat,
  `scripts/wme_27b_hb.sh`); cumulative cap 3 instance-hours and $20 (5-min margin); alert when alive > 15 min past
  T_max; confirms termination through the API; logs every action.
- Mocked-API tests: {lam['mock_tests']}.
- Real-instance test: {lam['real_test']}
- Timeline (UTC): {lam['timeline']}

## 8. Not done / deviations

{lam['notes']}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    r = build(Path(a.run))
    out = ROOT / "docs/reviews/wme"
    (out / f"wme-27b-calib-{a.date}.json").write_text(json.dumps(r, indent=1, sort_keys=True, default=str))
    (out / f"wme-27b-calib-{a.date}.md").write_text(markdown(r, a.date))
    print(json.dumps({k: v["pass"] for k, v in r["gates"].items()}))


if __name__ == "__main__":
    main()
