#!/usr/bin/env python3
"""Placeholder -> value table for the chair, generated from docs/prereg/wme_manifest.json (no hand-copied values)."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "docs/prereg/wme_manifest.json").read_text())


def short(v, n=110):
    s = json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


MAN = [
    ("target.theta0_hash", "P-H100 summary (hf.base_hash_from_snapshot)"),
    ("sources", "pilot pins unchanged; supplement pins (wm.supplement)"),
    (
        "panel.wme_c1.minhash_seed",
        "registered seed (exact Jaccard is used: every source < 50,000 groups)",
    ),
    ("panel.wme_c1.group_policy", "v1 §3.2 policy as implemented in wm.supplement"),
    ("panel.wme_c1.split_hashes", "DEFERRED: freeze build"),
    ("audit.exclusion_report_hash", "sha256 of the exposure and supplement audits"),
    ("audit.available_groups", "exposure audit + supplement"),
    ("tap.seed", "seed of namespace wme-c1:tap"),
    ("bank.roots", "wme_bank_roots.py on 9B (theta0)"),
    ("bank.noise_rho", "median dose-1 relative norm of the 12 steer roots"),
    ("bank.contrast_catalogue", "wm.rootsampler.catalogue"),
    ("bank.root_sampler_seed", "wm.rootsampler.seed_for('root_sampler')"),
    ("bank.blocks", "seeded block draw over the accepted roots"),
    ("bank.component_graph_hash", "sha256 of components + calibration"),
    ("bank.calibration_components", "stratified seeded draw"),
    ("bank.folds.outer_seed", "registered seed"),
    ("bank.folds.inner_seed", "registered seed"),
    ("model.dictionary", "bank roots + E2 library actions"),
    ("model.wm.config", "WMConfig defaults"),
    ("model.static.configs", "S1-S3"),
    ("model.chart_schema_version", "response_chart.CHART_SCHEMA_VERSION"),
    ("model.training_protocol", "v1 §6.1"),
    ("model.ensemble_seeds", "member seeds"),
    ("controls.inert", "ph100 constants + H100 verification"),
    ("planner.algorithm", "C6 gate (ruling §5)"),
    ("planner.config", "adapter9b.PLANNER_CONFIG"),
    ("e2.library", "wm.e2.library digest"),
    ("e2.library.lora_seeds", "wm.e2"),
    ("e2.episodes", "wm.e2.episodes at the selected n_rep (E2 unresolved: not run)"),
    ("e2.winners_manifest_hash", "DEFERRED: after the episodes"),
    ("gen.config", "genaudit.SPEC_V1"),
    ("gen.parsers", "sha256 of genaudit.py"),
    ("gen.sampled_seeds", "genaudit.SPEC_V1"),
    ("seeds.bootstrap.guard", "registered seed"),
    ("seeds.bootstrap.ht", "registered seed"),
    ("seeds.bootstrap.e1", "registered seed"),
    ("seeds.bootstrap.e2", "registered seed"),
    ("simulation.commit", "git HEAD at manifest generation"),
    ("simulation.seeds", "powersim_v1.SEED, refit seed"),
    ("simulation.report_hash", "sha256 of wme-powersim-v1-2026-11-17.json"),
    ("cost.unit_times", "P-H100"),
]
RUN = set(M["claims"]["run"])
V2 = "v2" in M


def _run_note(claim, yes, no):
    return yes if claim in RUN else no


if V2:
    MAN = [
        (k, src)
        for k, src in MAN
        if k not in ("simulation.report_hash", "simulation.seeds", "model.wm.config")
    ]
    MAN += [
        ("model.wm.config", "WMConfig defaults (WM-E v0: descriptive arm only, RULING-wme-v2 A1)"),
        ("model.wmcp.config", "wmcp.CPConfig defaults (A1, primary H-T model)"),
        ("model.wmcp.k_rule", "frozen K rule (A1)"),
        ("model.representation", "shared R(E) on 9B (A5)"),
        ("model.primary_ht", "A1"),
        ("model.descriptive_arms", "A1"),
        ("controls.c2s_c2d", "by construction (A1)"),
        ("controls.c2i", "hard gate (A1)"),
        ("inference.ht", "signflip.ht_test (A2)"),
        ("inference.e2", "signflip.e2_test (A3)"),
        ("gen.pool", "pinned TriviaQA rc.nocontext, v1-policy dedup (A4)"),
        ("panel.wme_c1.builder", "panel_c1.build_c1_panel_v1 (A5)"),
        ("guards.bounds", "bootstrap-t (A5)"),
        ("simulation.seeds", "grid, refit and sign-flip seeds"),
        (
            "simulation.report_hash",
            "sha256 of wme-powersim-v2 and wme-powersim-refit-v2 (2026-11-17)",
        ),
        ("claims", "§8.5 selection + full refit, ruling §4(b) / A6"),
    ]
if "v2b" in M:
    MAN += [
        ("model.wmcp.selection", "WM-CP config and K rule chosen on synthetic data only (v2b A7)"),
        (
            "ht.alternative",
            "registered physically possible alternative and its calibration (v2b A7)",
        ),
        ("caps", "v2b A9"),
        ("selection.priority", "v2b A8"),
        ("e1.effect", "v2b A10"),
        ("cost.generation", "P-H100b (v2b A9)"),
    ]
SIMS = [
    ("r_f", None),
    ("n_blk", None),
    ("n_cal", None),
    ("m_min_HT", None),
    ("n_split", None),
    (
        "n_gen,syc",
        _run_note(
            "e1",
            "TriviaQA generation items (A4)"
            + ("; N powered at 0.5x the generated effect (v2b A10)" if "v2b" in M else ""),
            "E1 unresolved: generation audit not run",
        ),
    ),
    ("n_gen,gsm", "not run — not powered"),
    ("n_gen,math", "not run — not powered"),
    (
        "n_rep",
        "E1 and E2 episodes"
        if {"e1", "e2"} <= RUN
        else _run_note(
            "e1",
            "E1 episodes (E2 unresolved)",
            _run_note(
                "e2", "E2 episodes (E1 unresolved)", "E1/E2 unresolved: not run as confirmatory"
            ),
        ),
    ),
    ("n_ep_sec", "as n_rep"),
    ("n_rep_sec", "0 = not run — not powered"),
    ("B", "as n_rep"),
    ("n_inert", None),
    (
        "pool_size",
        _run_note("e2", "reader pool", "readers not run (E2 unresolved); smallest grid value"),
    ),
    ("reader_token_cap", "Claude only"),
]
lines = [
    "# PREREG_WME "
    + ("v2 (RULING-wme-v2 + v2b)" if "v2b" in M else "v2 (RULING-wme-v2)" if V2 else "v1")
    + " placeholders → values (track A), for the chair",
    "",
    (
        f"Generated by `scripts/wme_manifest_table.py` from `docs/prereg/wme_manifest.json` (sha256 "
        f"`{hashlib.sha256((ROOT / 'docs/prereg/wme_manifest.json').read_bytes()).hexdigest()}`). PREREG_WME.md is not "
        "edited. Track B's fields (arms, vocabulary-row field, 27B) are in its own fragment."
    ),
    "",
    "## [MANIFEST] fields",
    "",
    "| Placeholder | Value (abridged; full value in the manifest) | Source / status |",
    "|---|---|---|",
]
for k, src in MAN:
    lines.append(f"| `[MANIFEST: {k}]` | `{short(M.get(k))}` | {src} |")
lines += [
    "",
    "## [SIM] values (§8.5 selection; claims run: "
    + (", ".join(M["claims"]["run"]) or "none")
    + "; unresolved — not powered: "
    + ", ".join(M["claims"]["unresolved_not_powered"])
    + ")",
    "",
    "| Placeholder | Value | Note |",
    "|---|---|---|",
]
for k, note in SIMS:
    if k not in M["sim"]:
        continue
    v = M["sim"][k] if k != "n_split" else M["sim"]["n_split"]
    if k == "n_split":
        for o, s in v.items():
            lines.append(f"| `[SIM: n_<split>,{o}]` | `{short(s)}` | per objective |")
        continue
    lines.append(f"| `[SIM: {k}]` | `{short(v)}` | {note or ''} |")
lines += [
    "",
    "Shard sizes: contrast "
    + str(M["sim"]["shard_contrast"])
    + ", LoRA "
    + str(M["sim"]["shard_lora"])
    + " groups per root; construction/e2 "
    + str(M["panel.wme_c1.sizes"]["sycophancy_real"]["construction_e2"])
    + " sycophancy groups. Harmful supplement: "
    + ", ".join(M["supplement"]["sources"])
    + " (pool "
    + str(M["supplement"]["pool"])
    + ", required "
    + str(M["supplement"]["required"])
    + ").",
]
(ROOT / "docs/prereg/wme_manifest_table.md").write_text("\n".join(lines) + "\n")
print(len(lines))
