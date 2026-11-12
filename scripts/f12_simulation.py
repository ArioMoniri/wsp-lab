#!/usr/bin/env python3
"""F12.5 outcome-free power simulation (PREREG_H1_TF amendment F12; Board A F12 conditions 2-5). CPU only.

Reads a frozen F12 token matrix (schema ``f12-token-matrix-v1``; until the real builder exists, the
deterministic synthetic generator below, keyed by component) and the F8a.2 qualification noise payload
(``data/runs/a5/nullsize/inputs.json``, sha256 pinned; only ``centred_noise``, ``window_mask``,
``window_ids`` and ``donors[j].{id, component, family}`` are read). No outcome, no ``donor_mean``, no
``data/evals`` file and no Part A payload is read. Writes a deterministic JSON report (+ a ``.time``
sidecar with wall-clock timings, kept out of the report so the report is byte-reproducible).

The bullet list below describes **F12 as drafted** (the ``Config()`` defaults, never registered). The
governed rules are those of the committed F12c configurations (A5 note F12c; ``F12C_CONFIGS``), written
into every report as ``rules.effective`` (``effective_rules``); where the list below differs, they govern.

Drafted rules (defaults; reports of the drafted config label them ``drafted_not_registered``):

* **Population and folds (F12.2).** S_k = existing components + the first k purchasable components in
  ascending sha256(``20261111:f12:purchase:`` + component id). Stratum = family (``mixed`` if a component
  holds two families); within a stratum components sort by sha256(``20261111:f12:`` + id); strata are
  concatenated abliteration, chain, lora_sft, mixed, random_dir and assigned round-robin to five outer
  folds; folds are recomputed on each S_k. Inner folds: outer-training components sorted by
  sha256(``20261110:lol-inner:`` + id), round-robin to four folds (F5 pins).
* **Models (F12.3; Board A condition 2).** F3 = training-only ridge readout b0 + b1 Sum a + b2 Sum q
  (penalty 1 on the standardised slopes, intercept unpenalised). WM-R = F3 + h, h the F5-pinned K+F
  head over 42 token columns (39 F5 columns + 3 probe columns): W1 42x16 + b1, 12x16 role embedding,
  W2 16x16 + b2, linear head on [masked sum, masked mean] = **1,185** nominal parameters, prediction
  f(D) - f(0) = w . pooled (exactly zero for every row with no present token; h == 0 gives F3 for
  every input). No learned K frames. StatNN = F3 + LightGBM (A5.8.4 settings) on v4 pooled statistics +
  WM-R's 42-column token for each of the 186 slots plus a presence column. F3 fitting, token
  standardisation and epoch / tree-count selection are nested inside each inner-training split and
  refitted on outer training only. Seeds 0/1/2 (``torch.manual_seed(s)`` before module construction;
  LightGBM ``random_state=s``) are averaged before the loss. AdamW lr 1e-3, wd 1e-2, batch 16, L1,
  <= 300 epochs, patience 30, refit epochs floor((e(2)+e(3))/2) >= 1 per seed.
* **Data-generating process (F12.5; Board A condition 4).** Delta y_i = a_i + gamma h*(x_i) + eps_i.
  h* is an untrained WM-R head under ``torch.manual_seed(seed("20261111:f12:", "hstar:<r>"))`` on
  tokens standardised with full-population present-token statistics, w . pooled (zero-edit
  subtracted), divided by its population SD (ddof 0) over the rows of S_k; SD <= 1e-12 or non-finite
  fails the replicate closed (no replacement seed).
* **Repaired noise (Board A condition 4).** Per simulated panel, in this frozen draw order:
  (i) each simulated component draws a payload component uniformly with replacement; (ii) per
  simulated component its rows take donors from that payload component by one random permutation
  (rows beyond the donor count reuse donors cyclically);
  (iii) ONE common index vector of 32 two-window blocks, drawn with replacement, is applied jointly
  to every donor's centred 64-window vector (block means) before any row mean is taken. The row's
  block values are a_i + gamma h*_i + B_d(i)[b]; eps_i is their mean. Because the same resampled item
  blocks enter every row, eps has a common panel-mean component (variance from the qualification
  residuals' item structure; zero without step iii) plus donor-specific item noise, with component
  clustering (shared payload component) and item clustering (shared blocks). Masked windows fail
  closed (the pinned payload has none; no imputation rule is registered).
* **Oracle gamma calibration (Board A condition 4).** Criterion: the ratio of expected row sums,
  C_oracle(gamma) = 1 - Sum_d Sum_i |eps_di| / Sum_d Sum_i |Delta y_di - F3_oof(Delta y_d)_i|, over 100
  calibration panels from seed("20261111:f12:", "gamma-cal:<k>:<r>") (independent of the replicate
  stream); F3_oof is the exact linear out-of-fold F3 operator on S_k's outer folds. gamma >= 0;
  bracket [0, 1] nat (h* has unit SD); pre-check on {0} U geomspace(1e-6, 1, 25): C(0) < 0.15 <
  C(1) and C non-decreasing on the grid (tolerance 1e-9); any failure = failed qualification of that
  replicate (counted as a power failure). Then bisection inside the first bracketing grid cell
  (100 halvings) and the registered criterion |C - 0.15| <= 0.005. Validation: the achieved oracle C
  with the same definition on 100 independent panels (seed "gamma-val:<k>:<r>"), with a
  delta-method Monte Carlo SE. gamma = 0 is the exact null. Variance x2 (eps x sqrt 2) reuses the
  base-variance gamma and is descriptive.
* **Contrasts (F12.4; A5.4.2; Board A condition 3).** C_WM = 1 - A_WM/A_F3 and C_WM,S = 1 - A_WM/A_S,
  A_x = row sum of |Delta y - prediction|. Intervals on L = log ratio: conditional (components) and
  joint (components x the 32 common item blocks, shared-reference two-way studentisation V_comp +
  V_item - V_both) bootstrap-t with component influences (``scripts/tfv1_requalify.py`` machinery),
  plus BCa and percentile (A4.4), 4,000 draws from PCG64(20261107) shared by both contrasts, linear
  .025/.975 quantiles, endpoints reversed. Guards: A_x/n > 1e-12 for both arms at the point and in
  every draw, >= 2 distinct components in the point data and every draw, finite log ratio, positive
  finite studentising variance, all 4,000 draws valid; no retry, clamp or replacement.
* **Decision (F12.4, evaluated in order).** unresolved (validity) -> correction detected (both point
  C >= 0.05 and the conditional and joint bootstrap-t lower bounds of both contrasts > 0) ->
  none detectable (both C_WM upper bounds < 0.05; the power clause is the gate below) -> unresolved.
* **Gates (F12.5; Board A condition 5).** Size (gamma = 0): rates of C_WM conditional lower > 0, C_WM
  joint lower > 0 and "correction detected". Ordered power: rate of "correction detected" under the
  alternative, invalid replicates counted as failures. Release only if every size rate's one-sided
  exact (Clopper-Pearson) upper Monte Carlo bound <= 0.07 and the power lower bound >= 0.80, with
  simultaneous 95 % bounds by Bonferroni over 6 k values x 4 gating rates (alpha 0.05/24 each,
  frozen). Ordinary two-sided 95 % CP intervals are reported too. The smallest passing S_k is the only
  purchase. Point estimates never release.

F12b repairs (PREREG_H1_TF A5 amendment F12b; docs/design/F12B_REPAIRS.md) are config entries whose
defaults reproduce F12 as drafted; the registered values are the committed, sha256-pinned JSON configs
``docs/prereg/f12_manifests/f12b_stage{1,2}_config.json`` (``F12B_CONFIGS``): zero-initialised head output
and training-only residual-target standardisation; the joint studentiser floored at the larger one-way
variance; F12a.4(h) calibration with a convexity bracket and a +-0.02 validation tolerance; the
F12a.4(b) noise draw order; the F12a.5 six-rate gates; single-seed StatNN tree-count selection; and the
two-stage k selection (``select_stage2_k``).

F12c governance (Board A F12b review conditions 2-7): the joint studentiser rejects non-finite terms
and requires a finite positive V_comp at the point and in every draw; per-contrast validity / binding
reports; the expected-row-sum coverage target and coverage qualification; the exact schedules
(stage 1: 40 + 40 at six k; stage 2: 1,000 stress + 1,000 alternative + 200 variance-x2 at the frozen
k*; coverage target: 200 stress at k*; pilot: 10 alternatives at k = 145) with distinct planned ids;
a frozen, verified stage-1 selection; report assembly across invocations; a fail-closed execution
validator whose release field is numerical eligibility pending Board A acceptance; the byte-pinned
restricted payload; the F12a.4(d) noise check; at most two workers.

Usage (OUT under /home/ubuntu/wsp-data; never docs/, data/evals)::

    python scripts/f12_simulation.py noisecheck --tokens X.npz --restricted-payload P.json --out OUT0
    python scripts/f12_simulation.py run --tokens X.npz --restricted-payload P.json --out OUTP \\
        --config docs/prereg/f12_manifests/f12c_pilot_config.json --workers 2
    python scripts/f12_simulation.py run --tokens X.npz --restricted-payload P.json --out OUT1 \\
        --config docs/prereg/f12_manifests/f12c_stage1_config.json --workers 2
    python scripts/f12_simulation.py select --stage1-report OUT1/report.json   # freezes k*
    python scripts/f12_simulation.py run ... --config docs/prereg/f12_manifests/f12c_target_config.json \\
        --stage1-report OUT1/report.json --selection docs/prereg/f12_manifests/f12c_stage2_selection.json
    python scripts/f12_simulation.py run ... --config docs/prereg/f12_manifests/f12c_stage2_config.json \\
        --stage1-report OUT1/report.json --selection ... --coverage-target-report OUTT/report.json \\
        [--part null] [--prereq noise_check=OUT0/noise_check.json --prereq ...]
    python scripts/f12_simulation.py assemble OUT2a/report.json OUT2b/report.json --out OUT2 ...
    python scripts/f12_simulation.py c1-draft   # rewrites docs/prereg/f12_manifests/code_hashes_c1.json
"""

from __future__ import annotations

import os

# One BLAS / torch thread per worker process: determinism and no oversubscription.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import f12_budget, f12_c1
from scripts.a5_build_panels import domain_seed
from scripts.a5_simulation import cp_interval, donor_block
from scripts.tfv1_requalify import (
    cluster_intervals,
    component_pairs,
    jackknife_cluster,
    linearised_se,
    theta_interval,
)

REPORT_SCHEMA = "f12-simulation-report-v1"
TOKEN_SCHEMA = "f12-token-matrix-v1"
FOLD_DOMAIN = "20261111:f12:"
PURCHASE_DOMAIN = "20261111:f12:purchase:"
INNER_DOMAIN = "20261110:lol-inner:"
SYNTHETIC_DOMAIN = "20261111:f12:synthetic:"
BOOT_SEED = 20261107
K_GRID = (0, 29, 58, 87, 116, 145)
STRATA = ("abliteration", "chain", "lora_sft", "mixed", "random_dir")
N_OUTER, N_INNER = 5, 4
WINDOWS, BLOCKS = 64, 32
TARGET_C = 0.15
DENOM_TOL = 1e-12
PAYLOAD_SHA256 = "67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de"
DEFAULT_PAYLOAD = ROOT / "data/runs/a5/nullsize/inputs.json"
PAYLOAD_FIELDS = ("centred_noise", "window_mask", "window_ids", "donors")
# F12d: donors carry the original eligible flag (F12a.4(e))
DONOR_FIELDS = ("id", "component", "family", "eligible")
QUALIFICATION_HALF_WIDTH = 0.00487  # tf-v1-requalification-2026-11-06.md, WikiText (i) panel mean
SCENARIOS = {
    "null": (False, 1.0),
    "alt": (True, 1.0),
    "null_var2": (False, 2.0),
    "alt_var2": (True, 2.0),
}
GATING_RATES = ("size_cond_lower", "size_joint_lower", "size_detected", "power_detected")
FORBIDDEN_OUT = ("docs", "paper", "assets", "data/evals")

# F5 schema pins: 12 roles, 186 matrix slots (Qwen3.5-0.8B: full attention every 4th layer).
ROLES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
)
N_LAYERS = 24
F5_COLS, PROBE_COLS = 39, 3
N_COLS = F5_COLS + PROBE_COLS
HIDDEN = 16
V4_COLS = 18  # the synthetic stand-in's v4 block (synthetic runs only; chair ruling on F12e (b))
# F12f: the real v4 block is the pinned pooling code's output (``wsp.features.pool``, feature_version 4, 4 buckets):
# 1,242 pooled statistics then 136 spectrum columns, as stored in each row's ``pooled.safetensors``.
from wsp.features.pool import pooled_dims as _pooled_dims

V4_REAL_COLS = sum(_pooled_dims(4))  # 1,378
GAUGE_ATOL, GAUGE_RTOL = 1e-5, 1e-4  # F5 pins "Gauge audit": |delta f| <= 1e-5 + 1e-4 |f| nats per row
MATRIX_MANIFEST = ROOT / "docs/prereg/a5_manifests/f12/matrix_manifest.json"


def slot_table():
    """[(layer, role index, tensor name, full-attention flag)] in A5.8.3 order (layer, role, name)."""
    out = []
    for layer in range(N_LAYERS):
        full = layer % 4 == 3
        roles = [r for r in ROLES if r.startswith("mlp.")]
        roles += [r for r in ROLES if r.startswith("self_attn." if full else "linear_attn.")]
        for role in sorted(roles):
            out.append((layer, ROLES.index(role), f"model.layers.{layer}.{role}.weight", full))
    return out


SLOTS = slot_table()
N_SLOTS = len(SLOTS)  # 186


def sha_key(domain: str, key: str) -> str:
    return hashlib.sha256(f"{domain}{key}".encode()).hexdigest()


# ------------------------------------------------------------------------------------- config


@dataclasses.dataclass(frozen=True)
class Config:
    seeds: tuple = (0, 1, 2)
    max_epochs: int = 300
    patience: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch: int = 16
    tree_grid: tuple = (100, 300, 600)
    draws: int = 4000
    boot_seed: int = BOOT_SEED
    cal_draws: int = 100
    val_draws: int = 100
    cal_tol: float = 0.005
    gamma_max: float = 1.0
    gamma_grid: int = 25
    ridge_lambda: float = 1.0
    size_limit: float = 0.07
    power_limit: float = 0.80
    mc_alpha: float = 0.05
    bonferroni_m: int = len(K_GRID) * len(GATING_RATES)
    lgbm: tuple = (
        ("learning_rate", 0.05),
        ("num_leaves", 8),
        ("min_child_samples", 5),
        ("subsample", 0.8),
        ("subsample_freq", 1),
        ("colsample_bytree", 0.5),
        ("reg_lambda", 1.0),
        ("n_jobs", 2),
    )
    # ---- F12b repairs (amendment F12b; docs/design/F12B_REPAIRS.md). The defaults reproduce F12 as
    # drafted; the registered values are the committed JSON configs in F12B_CONFIGS.
    name: str = "f12_drafted"
    head_zero_init: bool = False  # output weights w (and the unused bias) start at exactly 0
    target_standardise: bool = False  # residual target / its training-row SD; prediction x SD
    joint_studentiser: str = "strict"  # "strict" (A5.4.2 as drafted) | "floor_vcomp" (F12b.2) | ...
    calibration: str = "f12"  # "f12" (as drafted) | "f12b" (F12a.4(h) grid, F12b.3 tolerance)
    cal_bisect_tol: float = 0.001  # f12b: bisection target on the calibration panels
    cal_halvings: int = 100  # f12b registers 50 (F12a.4(h))
    cal_monotone: str = (
        "f12a"  # "f12a" (grid + 5-point non-decreasing check gates) | "convex" (F12b.3)
    )
    noise_draw: str = "f12"  # "f12" (as drafted) | "f12a" (F12a.4(b) draw order)
    gating: str = "f12"  # "f12" (4 rates, size among valid) | "f12a" (F12a.5(b)-(d): 6 rates)
    replicates: int = 1000
    seed_tag: str = "sim"  # replicate seeds seed("20261111:f12:", "<tag>:<scenario>:<k>:<r>")
    replicate_offset: int = 0  # stage 1 uses r = 1000.. so h*, gamma and noise differ from stage 2
    stage: int = 0  # 0 = single-stage F12; 1 = F12b k-range pilot; 2 = F12b registered run
    stage1_power_min: float = 0.90  # F12b.4 selection rule (stage 1 only)
    stage1_size_max: float = 0.07
    statnn_select_seeds: tuple = (
        0,
        1,
        2,
    )  # F12b.5: seeds averaged in the inner tree-count selection
    # ---- F12c (Board A F12b review conditions 3, 5, 6). Empty schedule = legacy (scenarios x replicates).
    schedule: tuple = ()  # ((scenario, planned replicates), ...) per k, r = offset .. offset + n - 1
    k_rule: str = "cli"  # "all" (six k) | "selected" (frozen stage-1 k*) | "145" | "cli" (legacy)
    seed_key: str = "{tag}:{scenario}:{k}:{r}"  # replicate panel seed key (pilot: "{tag}:{r}")
    cov_se_mult: float = 2.0  # coverage qualification: target -/+ this many Monte Carlo SE
    cov_error_limit: float = 0.07  # one-sided CP upper bound on each tail error rate

    def registered(self) -> bool:
        return governed_name(self) == "f12c_stage2"

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["lgbm"] = dict(self.lgbm)
        return d


CONFIG_SCHEMA = "f12-simulation-config-v1"
STUDENTISER_RULES = ("strict", "floor_vcomp", "floor_one_way")
CONFIG_CHOICES = {
    "joint_studentiser": STUDENTISER_RULES,
    "calibration": ("f12", "f12b"),
    "noise_draw": ("f12", "f12a"),
    "gating": ("f12", "f12a"),
    "cal_monotone": ("f12a", "convex"),
    "stage": (0, 1, 2, 3, 4),  # 3 = F12c coverage-target run, 4 = F12a.5(f) timing pilot
    "k_rule": ("cli", "all", "selected", "145"),
}
F12B_CONFIGS = {
    "stage1": ROOT / "docs/prereg/f12_manifests/f12b_stage1_config.json",
    "stage2": ROOT / "docs/prereg/f12_manifests/f12b_stage2_config.json",
}
F12B_CONFIG_SHA256 = {
    "stage1": "a3946ebb7250de7b45433605d7934bace7498c6d325cb70bdf1dccd779964a81",
    "stage2": "13e59899a90aa8c44457dfe757bce4d118dfb6d51eefb43e3b437d476580f65c",
}
# F12c (A5 note F12c): the governed configurations; the F12b ones are superseded (kept, not governed).
F12C_CONFIGS = {
    name: ROOT / f"docs/prereg/f12_manifests/f12c_{name}_config.json"
    for name in ("stage1", "stage2", "target", "pilot")
}
F12C_CONFIG_SHA256 = {
    "stage1": "7cb10a83a6a2fb0e7e92ee14b2bea21d3efea42d49510e1625e9bcb9df204b01",
    "stage2": "ae9340e1778cc4f85fa71020fa6275cc47b7b1a73064c5490bda6f59bc13e40f",
    "target": "be372fc56657bc4e6a62732e0e7bbce1701027197635b846efee6e8f4948e29c",
    "pilot": "5e8c8c2aef6bdb049c9470757f5847e406fbc908384abb0c83cf4e1b2254b24e",
}
SCENARIO_LABELS = {
    "null": "linear-signal / ridge-bias stress scenario",
    "alt": "token-function alternative calibrated against F3 (oracle C = 0.15)",
    "null_var2": "variance-x2 stress scenario (descriptive)",
    "alt_var2": "variance-x2 alternative (not registered)",
}
STRESS_LABEL = SCENARIO_LABELS["null"]
FAILURE_WORDING = (
    "The registered two-comparator decision did not meet its power qualification under this "
    "alternative; it does not establish absence of a K+F correction."
)
RELEASE_FIELD = "numerical eligibility pending Board A acceptance"
MAX_WORKERS = 2  # F12c.7: at most two single-thread workers on the shared 8-core machine


def config_from_dict(d) -> Config:
    fields = {f.name for f in dataclasses.fields(Config)}
    extra = set(d) - fields
    if extra:
        raise ValueError(f"unknown config keys {sorted(extra)}")
    d = dict(d)
    for k in ("seeds", "tree_grid", "statnn_select_seeds"):
        if k in d:
            d[k] = tuple(d[k])
    if "schedule" in d:
        d["schedule"] = tuple((str(sc), int(n)) for sc, n in d["schedule"])
        for sc, n in d["schedule"]:
            if sc not in SCENARIOS or n < 1:
                raise ValueError(f"bad schedule entry {(sc, n)!r}")
    if "lgbm" in d:
        d["lgbm"] = tuple(dict(d["lgbm"]).items())
    for k, allowed in CONFIG_CHOICES.items():
        if k in d and d[k] not in allowed:
            raise ValueError(f"{k} must be one of {allowed}, got {d[k]!r}")
    return Config(**d)


def load_config(path):
    """(Config, file sha256) from a committed ``f12-simulation-config-v1`` JSON."""
    raw = Path(path).read_bytes()
    obj = json.loads(raw)
    if obj.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA}")
    return config_from_dict(obj["config"]), hashlib.sha256(raw).hexdigest()


_PINNED = {}


def _pinned_configs():
    """{name: Config} of the committed configs whose bytes match their pinned sha256 (cached)."""
    if not _PINNED:
        for prefix, paths, shas in (
            ("f12c", F12C_CONFIGS, F12C_CONFIG_SHA256),
            ("f12b", F12B_CONFIGS, F12B_CONFIG_SHA256),
        ):
            for stage, path in paths.items():
                if path.exists():
                    ref, sha = load_config(path)
                    if sha == shas[stage]:
                        _PINNED[f"{prefix}_{stage}"] = ref
    return _PINNED


def registered_name(cfg):
    """'f12c_<stage>' or (superseded) 'f12b_<stage>' if cfg equals that committed, sha256-pinned
    config, else None."""
    for name, ref in _pinned_configs().items():
        if cfg == ref:
            return name
    return None


def governed_name(cfg):
    """The F12c config name if cfg is one of the governed F12c configs (F12b's are superseded)."""
    name = registered_name(cfg)
    return name if name and name.startswith("f12c_") else None


def replicate_seed(cfg, scenario, k, r):
    """PCG64 seed of replicate panel (scenario, k, r): seed("20261111:f12:", cfg.seed_key)."""
    key = cfg.seed_key.format(tag=cfg.seed_tag, scenario=scenario, k=k, r=r)
    return domain_seed(FOLD_DOMAIN, key)


def planned_ids(cfg, k_values):
    """Every planned (scenario, k, r) of cfg's schedule at the given k, in execution-report order."""
    if not cfg.schedule:
        raise ValueError("config has no registered schedule")
    out = []
    for k in k_values:
        for sc, n in cfg.schedule:
            out += [(sc, int(k), r) for r in range(cfg.replicate_offset, cfg.replicate_offset + n)]
    return out


# ------------------------------------------------------------------- population and folds


def synthetic_population():
    """Rows with the F12.2 counts (Board A ruling: 270 rows / 234 components).

    Existing: chain 71 components / 104 rows, the mixed chain+random_dir pair, lora_sft 10,
    abliteration 4, random_dir 3 singletons (123 rows / 89 components). Purchasable: abliteration 79,
    random_dir 45 singletons + 2 pairs, lora_sft 19 (147 rows / 145 components). Sizes: 214
    singletons, 12 pairs, 2 triples, 4 of four, 2 of five. Ids are synthetic hashes, not zoo ids.
    """
    spec = []  # (origin, family per row)
    chain_sizes = [1] * 54 + [2] * 9 + [3] * 2 + [4] * 4 + [5] * 2
    spec += [("existing", ["chain"] * s) for s in chain_sizes]
    spec.append(("existing", ["random_dir", "chain"]))
    spec += [("existing", ["lora_sft"])] * 10 + [("existing", ["abliteration"])] * 4
    spec += [("existing", ["random_dir"])] * 3
    spec += [("purchasable", ["abliteration"])] * 79
    spec += [("purchasable", ["random_dir"])] * 45 + [("purchasable", ["random_dir"] * 2)] * 2
    spec += [("purchasable", ["lora_sft"])] * 19
    rows = []
    for c, (origin, fams) in enumerate(spec):
        cid = hashlib.sha256(f"synthetic-component:{c}".encode()).hexdigest()[:16]
        for j, fam in enumerate(fams):
            rid = hashlib.sha256(f"synthetic-row:{c}:{j}".encode()).hexdigest()[:16]
            rows.append({"id": rid, "component": cid, "family": fam, "origin": origin})
    return rows


FAMILY_A_MEDIAN = {"lora_sft": 0.228, "chain": 0.00054, "abliteration": 0.01, "random_dir": 0.005}


def _family_slots(family, rng):
    if family in ("lora_sft", "chain"):
        keep = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_proj"}
        keep |= {"self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"}
        keep |= {"linear_attn.in_proj_qkv", "linear_attn.out_proj"}
        return [i for i, s in enumerate(SLOTS) if ROLES[s[1]] in keep], int(rng.choice([8, 16]))
    if family == "abliteration":
        keep = {"mlp.down_proj", "self_attn.o_proj", "linear_attn.out_proj"}
        return [i for i, s in enumerate(SLOTS) if ROLES[s[1]] in keep], 1
    if family == "random_dir":
        n = int(rng.integers(1, 4))
        return sorted(rng.choice(N_SLOTS, size=n, replace=False).tolist()), 1
    raise ValueError(f"unknown family {family!r}")


def _synthetic_row(family, rng):
    slots, rank = _family_slots(family, rng)
    n = len(slots)
    tok = np.zeros((n, N_COLS))
    scale = math.exp(rng.normal(-3.0, 1.0))
    tok[:, 0:16] = rng.normal(0.0, scale, size=(n, 16))
    sigma1 = np.exp(rng.normal(math.log(scale), 0.5, size=n))
    j = np.arange(8)
    spec = sigma1[:, None] * np.where(j[None, :] < rank, 0.7**j, 1e-5 * 0.5**j)[None, :]
    tok[:, 16:24] = np.log(spec)
    tok[:, 24:32] = (spec > 1e-3 * sigma1[:, None]).astype(float)
    tail_rank = max(rank - 8, 0)
    tok[:, 32] = (0.7 ** (2 * 8)) * (1 - 0.7 ** (2 * tail_rank)) / max(1e-12, 1 - 0.49**rank)
    tok[:, 33] = [SLOTS[s][0] / N_LAYERS for s in slots]
    tok[:, 34] = [float(SLOTS[s][3]) for s in slots]
    a_row = math.copysign(
        FAMILY_A_MEDIAN[family] * math.exp(rng.normal(0.0, 1.0)), rng.choice([-1.0, 1.0])
    )
    share = rng.gamma(1.0, size=n)
    tok[:, 35] = a_row * share / share.sum()
    kappa = 0.023 * math.exp(rng.normal(0.0, 1.0))
    tok[:, 36] = kappa * np.abs(tok[:, 35])
    tok[:, 37] = 0.0  # cross term: 0 for roots
    tok[:, 38] = 1.0
    down = np.array([ROLES[SLOTS[s][1]] == "mlp.down_proj" for s in slots])
    tok[down, 39] = sigma1[down] * np.exp(rng.normal(0.0, 0.3, size=down.sum()))
    tok[down, 40] = tok[down, 39] * np.exp(np.abs(rng.normal(0.5, 0.2, size=down.sum())))
    tok[down, 41] = 1.0
    return slots, tok


def v4_stats(slots, tok):
    """Synthetic stand-in for the v4 pooled statistics (18 columns)."""
    roles = (
        np.bincount([SLOTS[s][1] for s in slots], minlength=len(ROLES)) if slots else np.zeros(12)
    )
    if not slots:
        return np.zeros(V4_COLS)
    head = [
        len(slots),
        float(np.log(np.exp(tok[:, 16]).sum())),
        float(np.abs(tok[:, 35]).sum()),
        float(tok[:, 36].sum()),
        float(tok[:, 33].mean()),
        float(tok[:, 16].max()),
    ]
    return np.concatenate([head, roles]).astype(float)


def synthetic_token_matrix(rows=None):
    """Synthetic ``f12-token-matrix-v1`` (tests and dry runs only). Each component's rows are drawn
    from PCG64(seed(SYNTHETIC_DOMAIN, component)), so subsets reproduce the same rows."""
    rows = synthetic_population() if rows is None else rows
    by_comp = {}
    for i, r in enumerate(rows):
        by_comp.setdefault(r["component"], []).append(i)
    per_row = [None] * len(rows)
    for cid in sorted(by_comp):
        rng = np.random.default_rng(domain_seed(SYNTHETIC_DOMAIN, cid))
        for i in sorted(by_comp[cid], key=lambda i: rows[i]["id"]):
            per_row[i] = _synthetic_row(rows[i]["family"], rng)
    return assemble_matrix(rows, per_row, synthetic=True)


def assemble_matrix(rows, per_row, synthetic):
    token_row, token_slot, tokens = [], [], []
    v4 = np.zeros((len(rows), V4_COLS))
    for i, (slots, tok) in enumerate(per_row):
        token_row += [i] * len(slots)
        token_slot += list(slots)
        tokens.append(tok)
        v4[i] = v4_stats(slots, tok)
    tokens = np.concatenate(tokens) if tokens else np.zeros((0, N_COLS))
    token_slot = np.asarray(token_slot, dtype=np.int64)
    return {
        "schema": TOKEN_SCHEMA,
        "synthetic": bool(synthetic),
        "row_ids": np.asarray([r["id"] for r in rows]),
        "component": np.asarray([r["component"] for r in rows]),
        "family": np.asarray([r["family"] for r in rows]),
        "origin": np.asarray([r["origin"] for r in rows]),
        "eligible": np.asarray([r.get("eligible", True) for r in rows], dtype=bool),
        "sum_a": np.asarray([t[:, 35].sum() for _, t in per_row]),
        "sum_q": np.asarray([t[:, 36].sum() for _, t in per_row]),
        "v4": v4,
        "token_row": np.asarray(token_row, dtype=np.int64),
        "token_slot": token_slot,
        "token_role": np.asarray([SLOTS[s][1] for s in token_slot], dtype=np.int64),
        "tokens": tokens,
    }


MATRIX_KEYS = (
    "row_ids", "component", "family", "origin", "eligible", "sum_a", "sum_q", "v4",
    "token_row", "token_slot", "token_role", "tokens",
)  # fmt: skip
# F12f: the real matrix also carries the gauge-transformed tokens (aligned with ``tokens``) and the zero-edit v4
# statistics x0 (F12a.2(h)); synthetic matrices may omit them (their hash is then unchanged).
OPTIONAL_MATRIX_KEYS = ("tokens_gauge", "x0_v4")


def _matrix_keys(m):
    return MATRIX_KEYS + tuple(k for k in OPTIONAL_MATRIX_KEYS if k in m)


def validate_matrix(m):
    if m.get("schema") != TOKEN_SCHEMA:
        raise ValueError(f"token matrix schema must be {TOKEN_SCHEMA}")
    n = len(m["row_ids"])
    for k in ("component", "family", "origin", "eligible", "sum_a", "sum_q"):
        if len(m[k]) != n:
            raise ValueError(f"{k} has {len(m[k])} rows, expected {n}")
    if m["tokens"].ndim != 2 or m["tokens"].shape[1] != N_COLS:
        raise ValueError(f"tokens must be [T, {N_COLS}]")
    if m["synthetic"]:
        if m["v4"].shape != (n, V4_COLS):
            raise ValueError(f"v4 must be [{n}, {V4_COLS}]")
    else:
        if m["v4"].shape[1:] == (V4_COLS,) and V4_COLS != V4_REAL_COLS:
            raise ValueError(f"the {V4_COLS}-column v4 block is the synthetic stand-in, for synthetic runs only "
                             "(chair ruling on F12e (b))")  # fmt: skip
        if m["v4"].shape != (n, V4_REAL_COLS):
            raise ValueError(f"v4 must be [{n}, {V4_REAL_COLS}] (the pinned pooling code)")
        if "tokens_gauge" not in m:
            raise ValueError("a real token matrix needs the gauge-transformed tokens (F5 gauge audit)")
        if "x0_v4" not in m:
            raise ValueError("a real token matrix needs the zero-edit v4 statistics x0 (F12a.2(h))")
    if not np.all(np.isfinite(m["v4"])):
        raise ValueError("non-finite v4 statistics")
    if "tokens_gauge" in m:
        if np.shape(m["tokens_gauge"]) != np.shape(m["tokens"]):
            raise ValueError("gauge-transformed tokens must align with tokens")
        if not np.all(np.isfinite(m["tokens_gauge"])):
            raise ValueError("non-finite gauge-transformed tokens")
    if "x0_v4" in m:
        if np.shape(m["x0_v4"]) != (m["v4"].shape[1],):
            raise ValueError("x0_v4 must have the v4 width")
        if not np.all(np.isfinite(m["x0_v4"])):
            raise ValueError("non-finite zero-edit v4 statistics x0")
    if len({len(m[k]) for k in ("token_row", "token_slot", "token_role", "tokens")}) != 1:
        raise ValueError("token arrays differ in length")
    if len(set(m["row_ids"].tolist())) != n:
        raise ValueError("duplicate row ids")
    if not np.all(np.isfinite(m["tokens"])) or not np.all(np.isfinite(m["sum_a"])):
        raise ValueError("non-finite token or field values")
    if np.any((m["token_slot"] < 0) | (m["token_slot"] >= N_SLOTS)):
        raise ValueError("token slot out of range")
    if np.any(m["token_role"] != np.asarray([SLOTS[s][1] for s in m["token_slot"]], dtype=int)):
        raise ValueError("token role does not match its slot")
    pairs = set(zip(m["token_row"].tolist(), m["token_slot"].tolist()))
    if len(pairs) != len(m["token_row"]):
        raise ValueError("a row has two tokens in one slot")
    bad = set(m["family"].tolist()) - set(FAMILY_A_MEDIAN)
    if bad:
        raise ValueError(f"unknown families {sorted(bad)}")
    return m


def matrix_sha256(m) -> str:
    h = hashlib.sha256(TOKEN_SCHEMA.encode())
    h.update(json.dumps(bool(m["synthetic"])).encode())
    for k in _matrix_keys(m):
        a = np.asarray(m[k])
        a = a.astype("U") if a.dtype.kind in "US" else np.ascontiguousarray(a)
        h.update(f"{k}:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_matrix(path, m):
    validate_matrix(m)
    arrays = {k: np.asarray(m[k]) for k in _matrix_keys(m)}
    np.savez_compressed(
        path, schema=np.asarray(m["schema"]), synthetic=np.asarray(m["synthetic"]), **arrays
    )


def load_matrix(path):
    with np.load(path, allow_pickle=False) as z:
        m = {k: z[k] for k in MATRIX_KEYS + tuple(k for k in OPTIONAL_MATRIX_KEYS if k in z.files)}
        m["schema"] = str(z["schema"])
        m["synthetic"] = bool(z["synthetic"])
    return validate_matrix(m)


def subset_matrix(m, rows):
    rows = np.asarray(rows, dtype=np.int64)
    remap = np.full(len(m["row_ids"]), -1)
    remap[rows] = np.arange(len(rows))
    keep = remap[m["token_row"]] >= 0
    tok_keys = ("token_slot", "token_role", "tokens") + (("tokens_gauge",) if "tokens_gauge" in m else ())
    out = {k: m[k][rows] for k in MATRIX_KEYS if not k.startswith("token")}
    out.update({k: m[k][keep] for k in tok_keys})
    out["token_row"] = remap[m["token_row"][keep]]
    order = np.lexsort((out["token_slot"], out["token_row"]))
    for k in ("token_row",) + tok_keys:
        out[k] = out[k][order]
    if "x0_v4" in m:
        out["x0_v4"] = m["x0_v4"]
    out["schema"], out["synthetic"] = m["schema"], m["synthetic"]
    return out


def purchase_order(m):
    pur = {c for c, o in zip(m["component"].tolist(), m["origin"].tolist()) if o == "purchasable"}
    return sorted(pur, key=lambda c: sha_key(PURCHASE_DOMAIN, c))


def s_k_rows(m, k):
    """Row indices of S_k (eligible rows only; exclusions listed by the caller, no top-ups)."""
    order = purchase_order(m)
    if not 0 <= k <= len(order):
        raise ValueError(f"k = {k} outside 0..{len(order)}")
    comps = set(order[:k])
    existing = {c for c, o in zip(m["component"].tolist(), m["origin"].tolist()) if o == "existing"}
    if existing & set(order):
        raise ValueError("a component is both existing and purchasable")
    comps |= existing
    return np.asarray(
        [i for i, c in enumerate(m["component"].tolist()) if c in comps and m["eligible"][i]],
        dtype=np.int64,
    )


def component_strata(components, families):
    fams = {}
    for c, f in zip(components, families):
        fams.setdefault(c, set()).add(f)
    return {c: (next(iter(f)) if len(f) == 1 else "mixed") for c, f in fams.items()}


def outer_folds(components, families):
    """F12.2: hash-sorted within stratum, strata concatenated in STRATA order, round-robin to 5."""
    strata = component_strata(components, families)
    order = []
    for s in STRATA:
        order += sorted(
            (c for c, t in strata.items() if t == s), key=lambda c: sha_key(FOLD_DOMAIN, c)
        )
    if len(order) != len(strata):
        raise ValueError("a component has no registered stratum")
    return {c: i % N_OUTER for i, c in enumerate(order)}


def inner_folds(components):
    order = sorted(set(components), key=lambda c: sha_key(INNER_DOMAIN, c))
    return {c: i % N_INNER for i, c in enumerate(order)}


# ------------------------------------------------------------------------------------- F3


def _std(X):
    mu, sd = X.mean(0), X.std(0)
    zero = ~(sd > 1e-12)
    return np.where(zero, 0.0, mu), np.where(zero, 1.0, sd)


def f3_operator(a, q, train, test, lam=1.0):
    """Linear map M with F3 predictions on ``test`` = M @ y[train] (ridge on standardised [a, q])."""
    X = np.column_stack([a, q])
    mu, sd = _std(X[train])
    Z = (X[train] - mu) / sd
    zbar = Z.mean(0)  # 0 except for zero-variance (unscaled) columns
    Zc = Z - zbar
    n = len(train)
    # beta = (Zc'Zc + lam I)^-1 Zc' y (Zc' 1 = 0, so y need not be centred); b0 = ybar - zbar.beta
    G = np.linalg.solve(Zc.T @ Zc + lam * np.eye(2), Zc.T)
    Zt = (X[test] - mu) / sd - zbar
    return np.full((len(test), n), 1.0 / n) + Zt @ G


def f3_fit_predict(a, q, y, train, test, lam=1.0):
    return f3_operator(a, q, train, test, lam) @ y[train]


def oof_operator(a, q, fold_of_row, lam=1.0):
    n = len(a)
    P = np.zeros((n, n))
    for f in range(N_OUTER):
        test, train = np.flatnonzero(fold_of_row == f), np.flatnonzero(fold_of_row != f)
        if len(test):
            P[np.ix_(test, train)] = f3_operator(a, q, train, test, lam)
    return P


# ----------------------------------------------------------------------------- K+F head


class KFHead(torch.nn.Module):
    """F5-pinned K+F head with three probe columns. Construction order is pinned: lin1, role, lin2, out."""

    def __init__(self, n_cols=N_COLS):
        super().__init__()
        self.lin1 = torch.nn.Linear(n_cols, HIDDEN)
        self.role = torch.nn.Embedding(len(ROLES), HIDDEN)
        self.lin2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.out = torch.nn.Linear(2 * HIDDEN, 1)
        self.gelu = torch.nn.GELU()

    def forward(self, x, role, mask):
        """x [B, T, C] standardised and masked, role [B, T], mask [B, T] -> f(D) - f(0) [B]."""
        h1 = self.gelu(self.lin1(x) + self.role(role))
        h2 = self.gelu(self.lin2(h1)) * mask[..., None]
        s = h2.sum(1)
        pooled = torch.cat([s, s / mask.sum(1).clamp(min=1.0)[:, None]], dim=1)
        return pooled @ self.out.weight[0]


def head_param_count(n_cols=N_COLS) -> int:
    return sum(p.numel() for p in KFHead(n_cols).parameters())


PARAM_NAMES = ("W1", "b1", "E", "W2", "b2", "w", "b0")


def init_params(seed, n_cols=N_COLS, zero_out=False):
    """PyTorch default initialisation under torch.manual_seed(seed) (construction order pinned).

    zero_out (F12b.1): the output weights w and the (forward-unused) output bias start at exactly 0
    after the default draw, so every other tensor keeps its default value and the untrained head is
    h == 0 for every input (WM-R == F3 before training)."""
    torch.manual_seed(seed)
    m = KFHead(n_cols)
    p = {
        "W1": m.lin1.weight.detach().T.clone(),
        "b1": m.lin1.bias.detach().clone(),
        "E": m.role.weight.detach().clone(),
        "W2": m.lin2.weight.detach().T.clone(),
        "b2": m.lin2.bias.detach().clone(),
        "w": m.out.weight.detach()[0].clone(),
        "b0": m.out.bias.detach().clone(),
    }
    if zero_out:
        p["w"] = torch.zeros_like(p["w"])
        p["b0"] = torch.zeros_like(p["b0"])
    return p


def stack_params(plist):
    return {k: torch.stack([p[k] for p in plist]).requires_grad_(True) for k in PARAM_NAMES}


def batched_forward(P, x, role, mask):
    """P: stacked params [Mo, ...]; x [Mo, B, T, C]; role, mask [Mo, B, T] -> [Mo, B]."""
    gelu = torch.nn.functional.gelu
    mo = torch.arange(x.shape[0])[:, None, None]
    z1 = torch.einsum("mbtc,mch->mbth", x, P["W1"]) + P["b1"][:, None, None, :] + P["E"][mo, role]
    h2 = gelu(torch.einsum("mbth,mhk->mbtk", gelu(z1), P["W2"]) + P["b2"][:, None, None, :])
    h2 = h2 * mask[..., None]
    s = h2.sum(2)
    pooled = torch.cat([s, s / mask.sum(2).clamp(min=1.0)[..., None]], dim=-1)
    return torch.einsum("mbk,mk->mb", pooled, P["w"])


class HeadData:
    """Padded token tensors for the rows of one population."""

    def __init__(self, m, tokens_key="tokens"):
        n = len(m["row_ids"])
        counts = np.bincount(m["token_row"], minlength=n)
        T = max(1, int(counts.max()) if n else 1)
        self.n, self.T = n, T
        pos = np.zeros(len(m["token_row"]), dtype=np.int64)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        order = np.argsort(m["token_row"], kind="stable")
        pos[order] = np.arange(len(order)) - np.repeat(start, counts)
        X = np.zeros((n, T, N_COLS), dtype=np.float32)
        R = np.zeros((n, T), dtype=np.int64)
        M = np.zeros((n, T), dtype=np.float32)
        X[m["token_row"], pos] = m[tokens_key]
        R[m["token_row"], pos] = m["token_role"]
        M[m["token_row"], pos] = 1.0
        self.X, self.R, self.M = torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(M)
        self.tokens = np.asarray(m[tokens_key], dtype=np.float64)
        self.token_row = m["token_row"]

    def stats(self, rows):
        """Present-token column mean / SD (ddof 0) over ``rows``; zero-variance columns unscaled."""
        sel = np.isin(self.token_row, rows)
        if not sel.any():
            return np.zeros(N_COLS), np.ones(N_COLS)
        return _std(self.tokens[sel])

    def batch(self, rows, mu, sd):
        """rows [Mo, B] (long), mu/sd [Mo, C] -> standardised masked x, role, mask."""
        x = self.X[rows]
        mask = self.M[rows]
        x = (x - mu[:, None, None, :]) / sd[:, None, None, :] * mask[..., None]
        return x, self.R[rows], mask


def _adamw(P, grads, state, active, cfg):
    b1, b2, eps = 0.9, 0.999, 1e-8
    state["t"] += active
    t = state["t"].clamp(min=1.0)
    for k in PARAM_NAMES:
        p, g = P[k], grads[k]
        shape = (-1,) + (1,) * (p.dim() - 1)
        a = active.view(shape)
        m, v = state["m"][k], state["v"][k]
        with torch.no_grad():
            p.mul_(1 - cfg.lr * cfg.weight_decay * a)
            m.add_(a * (1 - b1) * (g - m))
            v.add_(a * (1 - b2) * (g * g - v))
            bc1 = (1 - b1**t).view(shape)
            bc2 = (1 - b2**t).view(shape)
            denom = v.sqrt() / bc2.sqrt() + eps
            p.sub_(a * cfg.lr / bc1 * m / denom)


def train_heads(data, plist, stats, targets, train_sets, cfg, *, val_sets=None, epochs=None):
    """Train Mo heads at once, each on its own rows / targets / standardisation / shuffling stream.

    Early-stopping mode (``val_sets``): returns best epochs (earliest argmin of held-out L1; stop after
    ``patience`` epochs without strict improvement; <= max_epochs). Fixed mode (``epochs``): trains
    exactly epochs[m] epochs. Returns (best_epochs or None, stacked params)."""
    Mo = len(plist)
    P = stack_params(plist)
    state = {
        "t": torch.zeros(Mo),
        "m": {k: torch.zeros_like(P[k]) for k in PARAM_NAMES},
        "v": {k: torch.zeros_like(P[k]) for k in PARAM_NAMES},
    }
    mu = torch.tensor(np.stack([s[0] for s in stats]), dtype=torch.float32)
    sd = torch.tensor(np.stack([s[1] for s in stats]), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32)  # [Mo, n]
    gens = [torch.Generator().manual_seed(int(s)) for s in [p["_seed"] for p in plist]]
    n_tr = [len(t) for t in train_sets]
    tr = [torch.as_tensor(np.asarray(t), dtype=torch.long) for t in train_sets]
    limit = torch.tensor(
        [cfg.max_epochs] * Mo if epochs is None else list(epochs), dtype=torch.long
    )
    best = torch.full((Mo,), float("inf"))
    best_ep = torch.zeros(Mo, dtype=torch.long)
    since = torch.zeros(Mo, dtype=torch.long)
    stopped = torch.zeros(Mo, dtype=torch.bool)
    if val_sets is not None:
        vmax = max(len(v) for v in val_sets)
        vrows = torch.zeros((Mo, vmax), dtype=torch.long)
        vw = torch.zeros((Mo, vmax))
        for i, v in enumerate(val_sets):
            vrows[i, : len(v)] = torch.as_tensor(np.asarray(v))
            vw[i, : len(v)] = 1.0
        vx = data.batch(vrows, mu, sd)
        vy = torch.gather(y, 1, vrows)
    B = cfg.batch
    epoch = 0
    while True:
        epoch += 1
        run = (~stopped) & (epoch <= limit)
        if not run.any():
            break
        perms = [torch.randperm(n_tr[i], generator=gens[i]) if run[i] else None for i in range(Mo)]
        steps = max(math.ceil(n_tr[i] / B) for i in range(Mo) if run[i])
        for s in range(steps):
            rows = torch.zeros((Mo, B), dtype=torch.long)
            wt = torch.zeros((Mo, B))
            active = torch.zeros(Mo)
            for i in range(Mo):
                if perms[i] is None or s * B >= n_tr[i]:
                    continue
                idx = tr[i][perms[i][s * B : (s + 1) * B]]
                rows[i, : len(idx)] = idx
                wt[i, : len(idx)] = 1.0
                active[i] = 1.0
            x, role, mask = data.batch(rows, mu, sd)
            pred = batched_forward(P, x, role, mask)
            yb = torch.gather(y, 1, rows)
            loss = ((pred - yb).abs() * wt).sum(1) / wt.sum(1).clamp(min=1.0)
            grads = torch.autograd.grad(
                (loss * active).sum(), [P[k] for k in PARAM_NAMES], allow_unused=True
            )
            grads = {
                k: torch.zeros_like(P[k]) if g is None else g for k, g in zip(PARAM_NAMES, grads)
            }
            _adamw(P, grads, state, active, cfg)
        if val_sets is not None:
            with torch.no_grad():
                vp = batched_forward(P, *vx)
                vl = ((vp - vy).abs() * vw).sum(1) / vw.sum(1)
            better = run & (vl < best)
            best = torch.where(better, vl, best)
            best_ep = torch.where(better, torch.full_like(best_ep, epoch), best_ep)
            since = torch.where(better, torch.zeros_like(since), since + run.long())
            stopped |= run & (since >= cfg.patience)
    return (best_ep.tolist() if val_sets is not None else None), P


def predict_heads(data, P, stats, rows):
    mu = torch.tensor(np.stack([s[0] for s in stats]), dtype=torch.float32)
    sd = torch.tensor(np.stack([s[1] for s in stats]), dtype=torch.float32)
    Mo = mu.shape[0]
    r = torch.as_tensor(np.asarray(rows), dtype=torch.long)[None, :].expand(Mo, -1)
    with torch.no_grad():
        return batched_forward(P, *data.batch(r, mu, sd)).double().numpy()


def _seeded(seed, zero_out=False):
    p = init_params(seed, zero_out=zero_out)
    p["_seed"] = seed
    return p


def target_scale(r, rows):
    """F12b.1 target standardisation: SD (ddof 0) of the residual target over the fit's training
    rows only; <= 1e-12 or non-finite fails closed (registered fit failure)."""
    s = float(np.std(np.asarray(r)[rows]))
    if not np.isfinite(s) or s <= 1e-12:
        raise ValueError(f"residual target SD {s} <= 1e-12")
    return s


def refit_epochs(best4):
    e = sorted(best4)
    return max(1, (e[1] + e[2]) // 2)


def gauge_within_tolerance(f, f_gauge, atol=GAUGE_ATOL, rtol=GAUGE_RTOL):
    """F5 pins: |f_gauge - f| <= atol + rtol |f| per row (nats); non-finite values fail."""
    f, fg = np.asarray(f, dtype=np.float64), np.asarray(f_gauge, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        ok = np.abs(fg - f) <= atol + rtol * np.abs(f)
    return ok & np.isfinite(f) & np.isfinite(fg)


def gauge_audit(data, data_g, P, stats, scale, rows, seeds, fold):
    """F5 gauge audit of one outer fold's refitted WM-R heads (one per seed): every held-out row is predicted from
    its frozen tokens and from its gauge-transformed tokens with the same fitted parameters and the same training
    standardisation; f is the seed's head prediction in nats (multiplied back by the residual-target SD; F3 does
    not read tokens). Every nonzero discrepancy is retained; a row outside the tolerance is a violation."""
    rows = np.asarray(rows)
    f = predict_heads(data, P, [stats] * len(seeds), rows) * scale
    fgv = predict_heads(data_g, P, [stats] * len(seeds), rows) * scale
    fits, disc, viol = [], [], 0
    for j, s in enumerate(seeds):
        ok = gauge_within_tolerance(f[j], fgv[j])
        delta = np.abs(fgv[j] - f[j])
        nv = int((~ok).sum())
        viol += nv
        for i in np.flatnonzero((delta != 0) | ~ok):
            disc.append({
                "fold": int(fold), "seed": int(s), "row": int(rows[i]), "f": float(f[j, i]),
                "f_gauge": float(fgv[j, i]), "delta": float(delta[i]),
                "tolerance": float(GAUGE_ATOL + GAUGE_RTOL * abs(f[j, i])), "violation": bool(not ok[i]),
            })  # fmt: skip
        fits.append({"fold": int(fold), "seed": int(s), "rows": len(rows), "violations": nv,
                     "max_abs_delta": float(np.nanmax(delta)) if len(rows) else 0.0})  # fmt: skip
    return {"fits": fits, "violations": viol, "discrepancies": disc}


def fit_wmr_fold(data, a, q, y, comps, train, test, cfg, data_g=None, fold=None):
    """One outer fold of WM-R: nested F3 / standardisation / epoch selection, refit, seed average.

    F12b.1 (cfg.head_zero_init, cfg.target_standardise): heads start with zero output weights, and
    each fit's residual target r = y - F3 is divided by its SD over that fit's training rows (inner
    training for the selection fits, outer training for the refit); the head's prediction is
    multiplied back by the same SD. Token standardisation (present-token mean / SD) is computed on
    the same training rows. The head output stays exactly 0 at the zero edit (no present token)."""
    zero = cfg.head_zero_init
    inner = inner_folds(comps[train].tolist())
    ifold = np.asarray([inner[c] for c in comps[train]])
    plist, stats, targets, trs, vas, inner_sd = [], [], [], [], [], []
    for s in cfg.seeds:
        for i in range(N_INNER):
            tr, va = train[ifold != i], train[ifold == i]
            if not len(tr) or not len(va):
                raise ValueError("empty inner fold")
            f3 = f3_operator(a, q, tr, np.arange(len(a)), cfg.ridge_lambda) @ y[tr]
            r = y - f3
            sc = target_scale(r, tr) if cfg.target_standardise else 1.0
            plist.append(_seeded(s, zero))
            stats.append(data.stats(tr))
            targets.append(r / sc)
            trs.append(tr)
            vas.append(va)
            inner_sd.append(sc)
    best, _ = train_heads(data, plist, stats, targets, trs, cfg, val_sets=vas)
    epochs = [refit_epochs(best[j * N_INNER : (j + 1) * N_INNER]) for j in range(len(cfg.seeds))]
    f3_all = f3_operator(a, q, train, np.arange(len(a)), cfg.ridge_lambda) @ y[train]
    r = y - f3_all
    sc = target_scale(r, train) if cfg.target_standardise else 1.0
    st = data.stats(train)
    _, P = train_heads(
        data,
        [_seeded(s, zero) for s in cfg.seeds],
        [st] * len(cfg.seeds),
        [r / sc] * len(cfg.seeds),
        [train] * len(cfg.seeds),
        cfg,
        epochs=epochs,
    )
    h = predict_heads(data, P, [st] * len(cfg.seeds), test).mean(0) * sc
    info = {"inner_best": best, "refit_epochs": epochs}
    if data_g is not None:
        info["gauge"] = gauge_audit(data, data_g, P, st, sc, test, cfg.seeds, fold)
    if cfg.target_standardise:
        info["target_sd"] = sc
        info["inner_target_sd"] = inner_sd[:N_INNER]
    return f3_all[test] + h, info


# --------------------------------------------------------------------------- StatNN


def statnn_features(m):
    """v4 pooled statistics + per slot [42 token columns, presence], zero padding, A5.8.3 order."""
    n = len(m["row_ids"])
    F = np.zeros((n, N_SLOTS, N_COLS + 1))
    F[m["token_row"], m["token_slot"], :N_COLS] = m["tokens"]
    F[m["token_row"], m["token_slot"], N_COLS] = 1.0
    return np.concatenate([m["v4"], F.reshape(n, -1)], axis=1)


def _lgbm(seed, n_trees, cfg):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=n_trees,
        random_state=seed,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
        **dict(cfg.lgbm),
    )


def fit_statnn_fold(F, a, q, y, comps, train, test, cfg):
    inner = inner_folds(comps[train].tolist())
    ifold = np.asarray([inner[c] for c in comps[train]])
    grid, top = sorted(cfg.tree_grid), max(cfg.tree_grid)
    l1 = np.zeros((N_INNER, len(grid)))
    for i in range(N_INNER if len(grid) > 1 else 0):  # a one-value grid selects trivially
        tr, va = train[ifold != i], train[ifold == i]
        r = y - f3_operator(a, q, tr, np.arange(len(a)), cfg.ridge_lambda) @ y[tr]
        preds = np.zeros((len(grid), len(va)))
        sel = cfg.statnn_select_seeds
        for s in sel:
            model = _lgbm(s, top, cfg).fit(F[tr], r[tr])
            for g, nt in enumerate(grid):
                preds[g] += model.predict(F[va], num_iteration=nt) / len(sel)
        l1[i] = np.abs(preds - r[va][None, :]).mean(1)
    mean_l1 = l1.mean(0)
    chosen = grid[int(np.flatnonzero(mean_l1 == mean_l1.min())[0])]  # smaller count on ties
    f3_all = f3_operator(a, q, train, np.arange(len(a)), cfg.ridge_lambda) @ y[train]
    r = y - f3_all
    tree = np.zeros(len(test))
    for s in cfg.seeds:
        tree += _lgbm(s, chosen, cfg).fit(F[train], r[train]).predict(F[test]) / len(cfg.seeds)
    return f3_all[test] + tree, {"trees": chosen}


# ------------------------------------------------------------------------------- noise


class NoisePayload:
    """Allowlisted view of the F8a.2 qualification noise payload."""

    def __init__(self, centred_noise, window_mask, donors, window_ids=None, provenance=None):
        noise = np.asarray(
            [[np.nan if v is None else v for v in row] for row in centred_noise], dtype=float
        )
        mask = np.asarray(window_mask, dtype=bool)
        if noise.shape != (len(donors), WINDOWS) or mask.shape != noise.shape:
            raise ValueError("centred_noise / window_mask must be [n_donors, 64]")
        if not mask.all() or not np.all(np.isfinite(noise)):
            raise ValueError("masked or non-finite windows: no imputation rule is registered")
        self.noise, self.mask = noise, mask
        self.donors = [{k: d[k] for k in DONOR_FIELDS if k in d} for d in donors]
        for d in self.donors:  # F12a.4(b): only donors whose original flag is true are ever used
            if "eligible" in d and d["eligible"] is not True:
                raise ValueError(f"donor {d.get('id')} is not eligible (flag {d['eligible']!r})")
        self.schema = self.RESTRICTED_SCHEMA
        self.window_ids = list(window_ids) if window_ids is not None else None
        self.blocks = np.stack([donor_block(noise[j], mask[j]) for j in range(len(donors))])
        comps = sorted({d["component"] for d in self.donors})
        self.components = comps
        self.members = [
            np.asarray([j for j, d in enumerate(self.donors) if d["component"] == c]) for c in comps
        ]
        self.sizes = np.asarray([len(x) for x in self.members])
        self.max_members = int(self.sizes.max())
        self.member_table = np.zeros((len(comps), self.max_members), dtype=np.int64)
        for i, x in enumerate(self.members):
            self.member_table[i, : len(x)] = x
        # F12a.4(b): payload components sorted by id, their donors sorted by donor id
        self.members_by_id = [
            np.asarray(sorted(x.tolist(), key=lambda j: self.donors[j]["id"]), dtype=np.int64)
            for x in self.members
        ]
        self.provenance = provenance or {}

    @classmethod
    def load(cls, path=DEFAULT_PAYLOAD, expected_sha256=PAYLOAD_SHA256, *, allow_raw=False):
        """Raw F8a.2 payload, allowlisted fields only, for drafted (unconfigured) diagnostics only: never a
        governed fallback (F12c.6). F12d barrier: refused unless ``allow_raw=True``, which ``main`` passes only
        when no config is given; governed extraction is ``scripts/f12_payload_extract.py``."""
        if not allow_raw:
            raise ValueError(
                "raw payload loader refused: configured execution reads only the pinned restricted copy"
            )
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"payload sha256 {digest} != pinned {expected_sha256}")
        obj = json.loads(raw)
        view = {k: obj[k] for k in PAYLOAD_FIELDS}
        del obj, raw  # nothing outside the allowlist is retained or read
        prov = {
            "source": str(path),
            "sha256": digest,
            "fields_read": list(PAYLOAD_FIELDS),
            "donor_fields_read": list(DONOR_FIELDS),
            "not_read": "donor_mean and every other field",
        }
        return cls(
            view["centred_noise"], view["window_mask"], view["donors"], view["window_ids"], prov
        )

    # F12d: donors carry the original eligible flag
    RESTRICTED_SCHEMA = "f12-noise-payload-restricted-v2"
    SUPERSEDED_SCHEMAS = ("f12-noise-payload-restricted-v1",)
    RESTRICTED_KEYS = {
        "schema",
        "provenance",
        "window_ids",
        "donors",
        "centred_noise",
        "window_mask",
    }

    @classmethod
    def load_restricted(cls, path, pin):
        """The restricted copy (F12a.4(e)), verified against its byte pin (F12c.6): the sha256 of the
        restricted bytes, the source digest, the contents (donor ids, components, families, window
        ids, counts) and the extractor (the bytes must be exactly what the pinned extractor's
        ``restricted()`` + ``json.dumps(sort_keys=True)`` writes for them). ``pin`` is mandatory (a
        dict or the path of a pin manifest); any mismatch fails closed."""
        if not isinstance(pin, dict):
            pin = json.loads(Path(pin).read_text())
        if pin.get("schema") != RESTRICTED_PIN_SCHEMA:
            raise ValueError(
                f"pin schema {pin.get('schema')!r} is not {RESTRICTED_PIN_SCHEMA} (superseded or unknown)"
            )
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != pin["restricted_copy"]["sha256"]:
            raise ValueError(
                f"restricted copy digest {digest} != pinned {pin['restricted_copy']['sha256']}"
            )
        obj = json.loads(raw)
        if obj.get("schema") in cls.SUPERSEDED_SCHEMAS:
            raise ValueError("superseded restricted copy (v1 lacks the eligible flag; F12d)")
        if set(obj) != cls.RESTRICTED_KEYS or obj["schema"] != cls.RESTRICTED_SCHEMA:
            raise ValueError("not a restricted f12 noise payload (unexpected keys or schema)")
        if (
            obj["provenance"].get("sha256") != PAYLOAD_SHA256
            or pin["source"]["sha256"] != PAYLOAD_SHA256
        ):
            raise ValueError("restricted payload provenance is not the pinned F8a.2 payload")
        for d in obj["donors"]:
            if set(d) != set(DONOR_FIELDS):
                raise ValueError(
                    "donor record is not exactly the allowlist (id, component, family, eligible flag)"
                )
            if d["eligible"] is not True:
                raise ValueError(f"donor {d['id']}: eligible flag is not true (F12a.4(b),(e))")
        got = restricted_contents(obj)
        for key, want in pin["contents"].items():
            if got.get(key) != want:
                raise ValueError(
                    f"restricted copy contents differ from the pin: {key.replace('_', ' ')}"
                )
        for key in ("script", "commit", "sha256", "functions"):
            if not pin.get("extractor", {}).get(key):
                raise ValueError(f"pin lacks extractor provenance ({key})")
        pay = cls(obj["centred_noise"], obj["window_mask"], obj["donors"], obj["window_ids"],
                  dict(obj["provenance"]))  # fmt: skip
        pay.schema = obj["schema"]
        if json.dumps(pay.restricted(), sort_keys=True).encode() != raw:
            raise ValueError("restricted bytes are not the pinned extractor's serialisation")
        pay.provenance["restricted_copy"] = str(path)
        pay.provenance["restricted_copy_sha256"] = digest
        pay.provenance["pin_status"] = pin.get("status")
        return pay

    def restricted(self) -> dict:
        return {
            "schema": self.schema,
            "provenance": self.provenance,
            "window_ids": self.window_ids,
            "donors": self.donors,
            "centred_noise": self.noise.tolist(),
            "window_mask": self.mask.tolist(),
        }


RESTRICTED_PIN = ROOT / "docs/prereg/f12_manifests/noise_payload_restricted_pin.json"
RESTRICTED_PIN_SCHEMA = "f12d-restricted-payload-pin-v2"  # F12d; the f12c v1 pin is superseded


def _sha_json(x) -> str:
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def restricted_contents(obj) -> dict:
    """Identity summary of a restricted payload object (counts and hashes; no noise values)."""
    donors = obj["donors"]
    fams, flags = {}, {}
    for d in donors:
        fams[d.get("family")] = fams.get(d.get("family"), 0) + 1
        f = "missing" if "eligible" not in d else json.dumps(d["eligible"])
        flags[f] = flags.get(f, 0) + 1
    return {
        "n_donors": len(donors),
        "n_components": len({d["component"] for d in donors}),
        "windows": len(obj["window_ids"]) if obj.get("window_ids") is not None else None,
        "window_ids_sha256": _sha_json(obj.get("window_ids")),
        "donor_ids_sha256": _sha_json([d["id"] for d in donors]),
        "donor_components_sha256": _sha_json([[d["id"], d["component"]] for d in donors]),
        "families": dict(sorted(fams.items())),
        "eligible_flags": dict(sorted(flags.items())),
        "donor_records_sha256": _sha_json([[d.get(k) for k in DONOR_FIELDS] for d in donors]),
        "all_windows_masked_in": bool(np.asarray(obj["window_mask"], dtype=bool).all()),
    }


def restricted_pin(path, extractor) -> dict:
    """Byte pin of a restricted payload copy (F12c.6): digest, source, extractor, contents."""
    raw = Path(path).read_bytes()
    obj = json.loads(raw)
    return {
        "schema": RESTRICTED_PIN_SCHEMA,
        "status": "draft, pending Board A C1 verification",
        "restricted_copy": {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
        "source": {
            "path": obj["provenance"].get("source"),
            "sha256": obj["provenance"].get("sha256"),
        },
        "provenance_in_copy": obj["provenance"],
        "extractor": dict(extractor),
        "contents": restricted_contents(obj),
    }


def load_restricted_pin(path=RESTRICTED_PIN) -> dict:
    return json.loads(Path(path).read_text())


def draw_panel_noise(payload, comp_codes, rng, scale=1.0, order="f12", row_ids=None):
    if order == "f12a":
        return draw_panel_noise_f12a(payload, comp_codes, row_ids, rng, scale)
    if order != "f12":
        raise ValueError(f"unknown noise draw order {order!r}")
    return _draw_panel_noise_f12(payload, comp_codes, rng, scale)


def draw_panel_noise_f12a(payload, comp_codes, row_ids, rng, scale=1.0):
    """F12a.4(b) draw order: (i) per simulated component in ascending code (= component-id) order one
    ``integers(0, n_payload_components)`` over the payload components sorted by id; (ii) per simulated
    component in the same order, its rows in ascending row id receive donors by one
    ``choice(n_c, size=m, replace=(m > n_c))`` over that payload component's donors sorted by id;
    (iii) one ``integers(0, 32, size=32)`` common block index for the whole panel."""
    if row_ids is None:
        raise ValueError("the F12a draw order needs the row ids")
    comp_codes = np.asarray(comp_codes)
    row_ids = np.asarray(row_ids)
    n_comp = int(comp_codes.max()) + 1
    members = payload.members_by_id
    pick = [int(rng.integers(0, len(members))) for _ in range(n_comp)]
    donors = np.zeros(len(comp_codes), dtype=np.int64)
    rows_of = payload_rows_by_component(comp_codes, row_ids)
    for c in range(n_comp):
        rows = rows_of[c]
        mem = members[pick[c]]
        sel = rng.choice(len(mem), size=len(rows), replace=len(rows) > len(mem))
        donors[rows] = mem[sel]
    b = rng.integers(0, BLOCKS, size=BLOCKS)
    return payload.blocks[donors][:, b] * scale, donors, b


def payload_rows_by_component(comp_codes, row_ids):
    order = np.lexsort((row_ids, comp_codes))
    bounds = np.r_[0, np.cumsum(np.bincount(comp_codes))]
    return [order[bounds[c] : bounds[c + 1]] for c in range(len(bounds) - 1)]


def _draw_panel_noise_f12(payload, comp_codes, rng, scale=1.0):
    """Repaired generator: rows x 32 noise block values for one simulated panel.

    comp_codes: simulated component code per row (0..n_comp-1). Frozen draw order: (i) payload
    component per simulated component, ``integers(n_payload_components, n_comp)``; (ii) one donor
    permutation per simulated component, ``random((n_comp, max donors))`` keys argsorted over that
    payload component's donors, the j-th row of the component (row order) taking permutation entry
    j mod n_donors (rows beyond the donor count reuse donors cyclically); (iii) ONE common index
    vector ``integers(32, 32)`` of two-window blocks applied to every donor before row means."""
    comp_codes = np.asarray(comp_codes)
    n_comp = int(comp_codes.max()) + 1
    pick = rng.integers(len(payload.members), size=n_comp)
    keys = rng.random((n_comp, payload.max_members))
    sizes = payload.sizes[pick]
    keys[np.arange(payload.max_members)[None, :] >= sizes[:, None]] = np.inf
    perm = np.argsort(keys, axis=1, kind="stable")
    order = np.argsort(comp_codes, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(comp_codes, minlength=n_comp))[:-1]]
    rank = np.empty(len(comp_codes), dtype=np.int64)
    rank[order] = np.arange(len(comp_codes)) - starts[comp_codes[order]]
    slot = perm[comp_codes, rank % sizes[comp_codes]]
    donors = payload.member_table[pick[comp_codes], slot]
    b = rng.integers(BLOCKS, size=BLOCKS)
    return payload.blocks[donors][:, b] * scale, donors, b


def noise_diagnostics(payload, comp_codes, seed, draws=2000, order="f12", row_ids=None):
    """Outcome-free checks: panel-mean SD, within-component and cross-component (common-item) dependence."""
    rng = np.random.default_rng(seed)
    kw = {"order": order, "row_ids": row_ids}
    eps = np.stack(
        [draw_panel_noise(payload, comp_codes, rng, **kw)[0].mean(1) for _ in range(draws)]
    )
    comp_codes = np.asarray(comp_codes)
    sd = eps.std(0, ddof=1)
    C = np.corrcoef(eps.T)
    same = comp_codes[:, None] == comp_codes[None, :]
    off = ~np.eye(len(comp_codes), dtype=bool)
    rng0 = np.random.default_rng(seed)
    no_resample = []
    for _ in range(50):
        _, donors, _ = draw_panel_noise(payload, comp_codes, rng0, **kw)
        no_resample.append(payload.blocks[donors].mean(1))
    return {
        "draws": draws,
        "panel_mean_sd_median": float(np.median(sd)),
        "panel_mean_sd_range": [float(sd.min()), float(sd.max())],
        "implied_half_width_median": float(1.959964 * np.median(sd)),
        "qualification_half_width_upper": QUALIFICATION_HALF_WIDTH,
        "within_component_corr_mean": float(C[same & off].mean()) if (same & off).any() else None,
        "cross_component_corr_mean": float(C[~same].mean()),
        "max_abs_mean_without_common_resampling": float(np.abs(np.stack(no_resample)).max()),
    }


# ----------------------------------------------------------------------------------- h*


def hstar_raw(data, pop_stats, r):
    """Untrained WM-R head (PyTorch default init under seed("20261111:f12:", "hstar:<r>")), f(D)-f(0)."""
    torch.manual_seed(domain_seed(FOLD_DOMAIN, f"hstar:{r}"))
    head = KFHead(N_COLS)
    mu = torch.tensor(pop_stats[0], dtype=torch.float32)
    sd = torch.tensor(pop_stats[1], dtype=torch.float32)
    with torch.no_grad():
        x = (data.X - mu) / sd * data.M[..., None]
        return head(x, data.R, data.M).double().numpy()


class HStarDegenerate(RuntimeError):
    pass


def hstar_values(raw):
    sd = float(np.std(raw))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise HStarDegenerate(f"h* population SD {sd}")
    return raw / sd


# --------------------------------------------------------------------------- calibration


def oracle_curve(P, a, h, eps):
    """gamma -> C_oracle(gamma), the ratio of expected row sums over the panels in ``eps`` [D, n]:
    1 - Sum|eps| / Sum|(I - P)(a + eps + gamma h)| (P the out-of-fold F3 operator)."""
    I_P = np.eye(len(a)) - P
    u = (a[None, :] + eps) @ I_P.T
    v = I_P @ h
    num = np.abs(eps).sum()

    def curve(gamma):
        gam = np.atleast_1d(np.asarray(gamma, dtype=float))
        out = np.array([1 - num / np.abs(u + g * v[None, :]).sum() for g in gam])
        return out if np.ndim(gamma) else float(out[0])

    return curve


def oracle_c(P, a, h, eps, gamma):
    return oracle_curve(P, a, h, eps)(gamma)


def calibrate_gamma(P, a, h, eps_cal, eps_val, cfg, target=TARGET_C):
    grid = np.r_[0.0, np.geomspace(1e-6, cfg.gamma_max, cfg.gamma_grid)]
    curve = oracle_curve(P, a, h, eps_cal)
    C = curve(grid)
    above = C >= target
    crossings = int(np.sum(above[1:] != above[:-1]))
    info = {
        "grid_C0": float(C[0]),
        "grid_Cmax": float(C[-1]),
        "grid_crossings": crossings,
        "grid_monotone": bool(np.all(np.diff(C) >= -1e-9)),
    }
    if not (C[0] < target < C[-1]):
        info["status"] = "failed: no bracket"
        return None, info
    if crossings != 1:
        info["status"] = "failed: target crossed more than once on the grid"
        return None, info
    j = int(np.flatnonzero(above)[0])
    lo, hi = grid[j - 1], grid[j]
    fine = curve(np.linspace(lo, hi, 33))
    info["bracket"] = [float(lo), float(hi)]
    info["bracket_monotone"] = bool(np.all(np.diff(fine) >= -1e-9))
    if not info["bracket_monotone"]:
        info["status"] = "failed: non-monotone bracket"
        return None, info
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if curve(mid) < target:
            lo = mid
        else:
            hi = mid
    gamma = 0.5 * (lo + hi)
    c_cal = curve(gamma)
    info["gamma"] = gamma
    info["C_calibration"] = c_cal
    if abs(c_cal - target) > cfg.cal_tol:
        info["status"] = "failed: tolerance"
        return None, info
    # validation on independent panels: ratio of sums with a delta-method Monte Carlo SE
    I_P = np.eye(len(a)) - P
    num = np.abs(eps_val).sum(1)
    den = np.abs((a[None, :] + eps_val + gamma * h[None, :]) @ I_P.T).sum(1)
    ratio = num.sum() / den.sum()
    d = len(num)
    infl = (num - ratio * den) / den.mean()
    info["C_validation"] = float(1 - ratio)
    info["C_validation_se"] = float(np.sqrt((infl**2).sum() / (d * (d - 1))))
    info["status"] = "ok"
    return gamma, info


def noise_tau(payload):
    """F12a.4(h) scale: RMS over donors of the plug-in panel-mean noise SD, sqrt((1/32)(1/32) Sum_j e_j^2)
    with e_j the donor's 32 block means (an outcome-free scalar of the restricted payload)."""
    e = payload.blocks
    return float(np.sqrt(np.mean((e**2).sum(1) / BLOCKS / BLOCKS)))


def calibration_panels(payload, comp, row_ids, count, kind, k, r, cfg, start=0):
    """Noise-only panels d = start..start+count-1, each from its own PCG64 stream
    seed("20261111:f12:", "<kind>:<k>:<r>:<d>") (F12a.4(h) 'cal', F12a.4(i) 'calval'); rows' eps."""
    out = []
    for d in range(start, start + count):
        rng = np.random.default_rng(domain_seed(FOLD_DOMAIN, f"{kind}:{k}:{r}:{d}"))
        E = draw_panel_noise(payload, comp, rng, order=cfg.noise_draw, row_ids=row_ids)[0]
        out.append(E.mean(1))
    return np.stack(out)


def ratio_c_with_se(P, a, h, eps, gamma):
    """C-bar = 1 - Sum|eps| / Sum|(I - P)(a + eps + gamma h)| over panels, delta-method MC SE."""
    I_P = np.eye(len(a)) - P
    num = np.abs(eps).sum(1)
    den = np.abs((a[None, :] + eps + gamma * h[None, :]) @ I_P.T).sum(1)
    ratio = num.sum() / den.sum()
    d = len(num)
    infl = (num - ratio * den) / den.mean()
    return float(1 - ratio), float(np.sqrt((infl**2).sum() / (d * (d - 1))))


def calibrate_gamma_f12b(P, a, h, eps_cal, eps_val, tau, cfg, target=TARGET_C):
    """F12a.4(h) with the F12b.3 repairs. Grid {0} U {tau 2^g : g = -8..12}; bracket: C(0) < target and
    a first grid point g* with C >= target; bisection on [gamma_(g*-1), gamma_g*] with the calibration
    panels until |C_cal - target| <= cal_bisect_tol (<= cal_halvings halvings). Validation (F12b.3):
    the achieved C-bar on the independent validation panels must satisfy |C_val - target| <= cal_tol.

    Monotonicity (cfg.cal_monotone). C(gamma) = 1 - N / Sum_d Sum_i |u_di + gamma v_i| with N fixed is
    an increasing transform of a convex function of gamma (a sum of absolute affine functions), so
    for target > C(0) the equation C = target has exactly one root on gamma >= 0 and the bracket holds
    it; C may dip below C(0) before rising. 'f12a' keeps F12a.4(h)'s non-decreasing grid / interior
    check as a gate; 'convex' (F12b) reports it and does not gate on it. Any failure fails
    qualification of (k, r): no substitute h*, no other bracket, no retry."""
    grid = np.r_[0.0, tau * 2.0 ** np.arange(-8, 13)]
    curve = oracle_curve(P, a, h, eps_cal)
    C = curve(grid)
    info = {"grid": grid.tolist(), "grid_C": C.tolist(), "tau": tau}
    above = np.flatnonzero(C >= target)
    if not C[0] < target or not len(above):
        info["status"] = "failed: no bracket"
        return None, info
    g = int(above[0])
    lo, hi = float(grid[g - 1]), float(grid[g])
    inner = curve(np.linspace(lo, hi, 7)[1:-1])
    seq = np.r_[C[g - 1], inner, C[g]]
    info["bracket"] = [lo, hi]
    info["f12a_grid_nondecreasing"] = bool(np.all(np.diff(C[: g + 1]) >= -1e-12))
    info["f12a_interior_nondecreasing"] = bool(np.all(np.diff(seq) >= -1e-12))
    info["dip_below_C0"] = bool(np.any(C[1 : g + 1] < C[0]))
    if cfg.cal_monotone == "f12a" and not (
        info["f12a_grid_nondecreasing"] and info["f12a_interior_nondecreasing"]
    ):
        info["status"] = "failed: non-monotone bracket"
        return None, info
    gamma, c_cal, halvings = hi, float(C[g]), 0
    while abs(c_cal - target) > cfg.cal_bisect_tol:
        if halvings >= cfg.cal_halvings:
            info["status"] = "failed: bisection"
            return None, info
        gamma = 0.5 * (lo + hi)
        c_cal = curve(gamma)
        halvings += 1
        if c_cal < target:
            lo = gamma
        else:
            hi = gamma
    info.update(gamma=float(gamma), C_calibration=float(c_cal), halvings=halvings)
    c_val, se_val = ratio_c_with_se(P, a, h, eps_val, gamma)
    info["C_validation"], info["C_validation_se"] = c_val, se_val
    if abs(c_val - target) > cfg.cal_tol:
        info["status"] = "failed: validation tolerance"
        return None, info
    info["status"] = "ok"
    return float(gamma), info


# ----------------------------------------------------------------------------- intervals


def a542_mults(n_comp, cfg):
    """One frozen set of bootstrap multiplicities shared by both contrasts (PCG64(seed))."""
    rng = np.random.default_rng(cfg.boot_seed)
    cond = rng.multinomial(n_comp, np.full(n_comp, 1 / n_comp), size=cfg.draws).astype(float)
    jc = rng.multinomial(n_comp, np.full(n_comp, 1 / n_comp), size=cfg.draws).astype(float)
    jj = rng.multinomial(BLOCKS, np.full(BLOCKS, 1 / BLOCKS), size=cfg.draws).astype(float)
    return cond, (jc, jj)


def joint_parts(D, qa, qb, comp, mult):
    """Per joint draw: ln R*, V_comp, V_item and V_both of the A5.4.2 two-way studentiser.

    The algebra is ``scripts/tfv1_requalify.py:joint_replicates`` (kappa 1, uniform observed item
    weights) with the three terms returned separately instead of sqrt(V_comp + V_item - V_both)
    (NaN when <= 0). mult = (component multiplicities [B, n_comp], item multiplicities [B, 32])."""
    comp = np.asarray(comp)
    order = np.argsort(comp, kind="stable")
    D, qa, qb, comp = D[order], qa[order], qb[order], comp[order]
    J = D.shape[1]
    starts = np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]])
    n = len(starts)
    sizes = np.diff(np.r_[starts, len(comp)])
    comp_idx = np.repeat(np.arange(n), sizes)
    pi, pj, pw, pstarts = component_pairs(sizes)
    pairs_d = D[pi] * D[pj]
    mult_c_all, mult_j_all = mult
    nn = n * J
    out = {k: [] for k in ("ln", "v_comp", "v_item", "v_both")}
    for lo in range(0, len(mult_c_all), 200):
        mult_c, mult_j = mult_c_all[lo : lo + 200], mult_j_all[lo : lo + 200]
        v = mult_j @ D.T
        a, bb = v / J - qa, v / J - qb
        abs_a, abs_b = np.abs(a), np.abs(bb)
        row_mult = mult_c[:, comp_idx]
        sa0, sb0 = (row_mult * abs_a).sum(1), (row_mult * abs_b).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ca = np.add.reduceat(abs_a, starts, axis=1)
            cb = np.add.reduceat(abs_b, starts, axis=1)
            u = ca / sa0[:, None] - cb / sb0[:, None]
            v_comp = n / (n - 1) * (mult_c * u**2).sum(1)
            g = np.sign(a) / sa0[:, None] - np.sign(bb) / sb0[:, None]
            lam = (row_mult * g) @ D
            p = mult_j / J
            lam_bar = (p * lam).sum(1)
            v_item = (p * (lam - lam_bar[:, None]) ** 2).sum(1) / (J - 1)
            h = np.sign(a) * qa / sa0[:, None] - np.sign(bb) * qb / sb0[:, None]
            h_c = np.add.reduceat(h, starts, axis=1)
            kernel = mult_j @ pairs_d.T
            s1 = np.add.reduceat(g[:, pi] * g[:, pj] * kernel * pw, pstarts, axis=1)
            s2 = np.add.reduceat(g * v, starts, axis=1)
            inner = s1 - 2 * h_c * s2 + J * h_c**2
            v_both = nn / (nn - 1) * (mult_c * inner).sum(1) / J**2
            out["ln"].append(np.log(sa0 / sb0))
        out["v_comp"].append(v_comp)
        out["v_item"].append(v_item)
        out["v_both"].append(v_both)
    return {k: np.concatenate(x) for k, x in out.items()}


def joint_variance(parts, rule):
    """Joint studentising variance under the registered rule (F12b.2; F12c.2 enforcement).

    strict: V = V_comp + V_item - V_both (A5.4.2 as written; V <= 0 or non-finite: invalid draw).
    floor_vcomp: V = max(V_comp + V_item - V_both, V_comp), V_comp the draw's component-only term.
    floor_one_way: V = max(V_comp + V_item - V_both, V_comp, V_item) (descriptive sensitivity only).
    F12c.2: a draw with any non-finite term (V_comp, V_item or V_both) is invalid under every rule:
    the non-finite raw value is rejected, never replaced by the floor. Under the floor rules a draw is
    also invalid unless V_comp itself is finite and > 0 (F12b.2's requirement, checked separately,
    not only through the final variance). Invalid draws get NaN; no draw is dropped or replaced.
    Info counts: raw_nonfinite (some term non-finite), raw_nonpositive (all terms finite, raw <= 0),
    vcomp_invalid (V_comp non-finite or <= 0), floor_binding (valid draws whose variance is the floor)."""
    vc, vi, vb = (np.asarray(parts[k], dtype=float) for k in ("v_comp", "v_item", "v_both"))
    finite = np.isfinite(vc) & np.isfinite(vi) & np.isfinite(vb)
    with np.errstate(invalid="ignore"):
        raw = np.where(finite, vc + vi - vb, np.nan)
        vc_ok = np.isfinite(vc) & (vc > 0)
    info = {
        "raw_nonfinite": int((~finite).sum()),
        "raw_nonpositive": int((finite & ~(raw > 0)).sum()),
        "vcomp_invalid": int((~vc_ok).sum()),
    }
    if rule == "strict":
        var = np.where(finite & (raw > 0), raw, np.nan)
        return var, {**info, "floor_binding": 0}
    if rule in ("floor_one_way", "floor_vcomp"):
        floor = vc if rule == "floor_vcomp" else np.maximum(vc, vi)
        ok = finite & vc_ok
        with np.errstate(invalid="ignore"):
            binding = ok & (raw < floor)
            var = np.where(ok, np.where(binding, floor, raw), np.nan)
        return var, {**info, "floor_binding": int(binding.sum())}
    raise ValueError(f"unknown joint studentiser rule {rule!r}")


def a542_contrast(D, qa, qb, comp, mults, rule="strict"):
    """A5.4.2 conditional and joint intervals of C = 1 - A_a/A_b (bootstrap-t primary, BCa, percentile).

    D rows x 32 block values (row mean = Delta y), qa candidate and qb comparator predictions (fixed),
    comp dense component codes 0..n_comp-1."""
    comp = np.asarray(comp)
    n_comp = int(comp.max()) + 1
    y = D.mean(1)
    la, lb = np.abs(y - qa), np.abs(y - qb)
    ca = np.bincount(comp, la, n_comp)
    cb = np.bincount(comp, lb, n_comp)
    cn = np.bincount(comp, minlength=n_comp).astype(float)
    n = len(y)
    failures = []
    if not (la.sum() / n > DENOM_TOL and lb.sum() / n > DENOM_TOL):
        failures.append("point guard")
    if n_comp < 2:
        failures.append("fewer than two components")
    out = {"C": float(1 - la.sum() / lb.sum()) if lb.sum() > 0 else float("nan"), "intervals": {}}
    if failures:
        out.update(valid=False, failures=failures)
        return out
    jack = lambda: jackknife_cluster(ca, cb)
    cond, (jc, jj) = mults
    # conditional
    sa, sb, rows_n = cond @ ca, cond @ cb, cond @ cn
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_c = np.log(sa / sb)
        se_c = linearised_se(ca[None, :], cb[None, :], cond)
    ok_c = (
        (sa / rows_n > DENOM_TOL)
        & (sb / rows_n > DENOM_TOL)
        & ((cond > 0).sum(1) >= 2)
        & np.isfinite(ln_c)
        & np.isfinite(se_c)
        & (se_c > 0)
    )
    # joint (components x common item blocks), two-way studentiser
    parts = joint_parts(D, qa, qb, comp, (jc, jj))
    var_j, sinfo = joint_variance(parts, rule)
    ln_j = parts["ln"]
    with np.errstate(invalid="ignore"):
        se_j = np.sqrt(np.where(var_j > 0, var_j, np.nan))
    terms_finite = (
        np.isfinite(parts["v_comp"]) & np.isfinite(parts["v_item"]) & np.isfinite(parts["v_both"])
    )
    with np.errstate(invalid="ignore"):
        vcomp_ok = np.isfinite(parts["v_comp"]) & (parts["v_comp"] > 0)
    if rule == "strict":
        vcomp_ok = np.ones_like(vcomp_ok)  # A5.4.2 as written: only the two-way value is checked
    v = jj @ D.T / BLOCKS
    rm = jc[:, comp]
    sa_j = (rm * np.abs(v - qa)).sum(1)
    sb_j = (rm * np.abs(v - qb)).sum(1)
    rows_j = rm.sum(1)
    ok_j = (
        (sa_j / rows_j > DENOM_TOL)
        & (sb_j / rows_j > DENOM_TOL)
        & ((jc > 0).sum(1) >= 2)
        & np.isfinite(ln_j)
        & terms_finite
        & vcomp_ok
        & np.isfinite(se_j)
        & (se_j > 0)
    )
    pt = joint_parts(D, qa, qb, comp, (np.ones((1, n_comp)), np.ones((1, BLOCKS))))
    var0, pinfo = joint_variance(pt, rule)
    with np.errstate(invalid="ignore"):
        se0 = np.sqrt(np.where(var0 > 0, var0, np.nan))
    out["studentiser"] = {
        "rule": rule,
        "draws": len(ln_j),
        "raw_nonpositive": sinfo["raw_nonpositive"],
        "raw_nonfinite": sinfo["raw_nonfinite"],
        "vcomp_invalid": sinfo["vcomp_invalid"],
        "floor_binding": sinfo["floor_binding"],
        "point_raw_nonpositive": bool(pinfo["raw_nonpositive"]),
        "point_raw_nonfinite": bool(pinfo["raw_nonfinite"]),
        "point_vcomp_invalid": bool(pinfo["vcomp_invalid"]),
        "point_floor_binding": bool(pinfo["floor_binding"]),
    }
    out["n_failed"] = {"conditional": int((~ok_c).sum()), "joint": int((~ok_j).sum())}

    def reasons(sa, sb, rows, mult, ln, se):
        return {
            "loss_guard": int(np.sum(~((sa / rows > DENOM_TOL) & (sb / rows > DENOM_TOL)))),
            "components_lt_2": int(np.sum((mult > 0).sum(1) < 2)),
            "log_ratio_nonfinite": int(np.sum(~np.isfinite(ln))),
            "studentiser_nonpositive_or_nonfinite": int(np.sum(~(np.isfinite(se) & (se > 0)))),
        }

    out["failure_reasons"] = {
        "conditional": reasons(sa, sb, rows_n, cond, ln_c, se_c),
        "joint": {
            **reasons(sa_j, sb_j, rows_j, jc, ln_j, se_j),
            "two_way_term_nonfinite": int((~terms_finite).sum()),
            "v_comp_nonpositive_or_nonfinite": int((~vcomp_ok).sum()),
        },
    }
    if not ok_c.all():
        failures.append("conditional draws invalid")
    if not ok_j.all():
        failures.append("joint draws invalid")
    pt_terms = all(np.isfinite(pt[x][0]) for x in ("v_comp", "v_item", "v_both"))
    if rule != "strict" and not (np.isfinite(pt["v_comp"][0]) and pt["v_comp"][0] > 0):
        failures.append("joint point V_comp")
    elif not pt_terms:
        failures.append("joint point term non-finite")
    if not (np.isfinite(se0[0]) and se0[0] > 0):
        failures.append("joint point studentiser")
    if failures:
        out.update(valid=False, failures=failures)
        return out
    civ = cluster_intervals(ca, cb, cn, ln_c, se_c, jack)
    jiv = cluster_intervals(ca, cb, cn, ln_j, se_j, jack)
    t = (ln_j - jiv["ln_hat"]) / se_j
    jiv["bootstrap_t"] = (
        jiv["ln_hat"] - np.quantile(t, 0.975) * se0[0],
        jiv["ln_hat"] - np.quantile(t, 0.025) * se0[0],
    )
    for name, iv in (("conditional", civ), ("joint", jiv)):
        for method in ("bootstrap_t", "bca", "percentile"):
            lo, hi = theta_interval(*iv[method])
            out["intervals"][f"{name}_{method}"] = [float(lo), float(hi)]
    out.update(valid=True, failures=[])
    return out


def f12_decision(wm, wms):
    """F12.4 ordered decision on the two contrasts (scientific wording only)."""
    if not (wm["valid"] and wms["valid"]):
        return "unresolved_validity"
    iv, ivs = wm["intervals"], wms["intervals"]
    lower = all(x[f"{p}_bootstrap_t"][0] > 0 for x in (iv, ivs) for p in ("conditional", "joint"))
    if wm["C"] >= 0.05 and wms["C"] >= 0.05 and lower:
        return "correction_detected"
    if all(iv[f"{p}_bootstrap_t"][1] < 0.05 for p in ("conditional", "joint")):
        # provisional: an upper-bound code is not a qualified label until the power gate and the
        # F12c.3 upper-tail coverage qualification pass for the analysed population
        return "none_detectable_provisional"
    return "unresolved"


# ------------------------------------------------------------------------------ replicate


class Context:
    """Everything a replicate on S_k needs; built once per (worker, k)."""

    def __init__(self, matrix, payload, k, cfg):
        self.k, self.cfg = k, cfg
        rows = s_k_rows(matrix, k)
        m = subset_matrix(matrix, rows)
        self.m = m
        self.ids = m["row_ids"]
        comps = m["component"]
        uniq = sorted(set(comps.tolist()))
        code = {c: i for i, c in enumerate(uniq)}
        self.comp_ids = uniq
        self.comp = np.asarray([code[c] for c in comps])
        folds = outer_folds(comps.tolist(), m["family"].tolist())
        self.folds = folds
        self.fold = np.asarray([folds[c] for c in comps])
        self.a, self.q = m["sum_a"].astype(float), m["sum_q"].astype(float)
        self.data = HeadData(m)
        self.data_g = HeadData(m, tokens_key="tokens_gauge") if "tokens_gauge" in m else None
        self.F = statnn_features(m)
        self.P = oof_operator(self.a, self.q, self.fold, cfg.ridge_lambda)
        self.payload = payload
        self.tau = noise_tau(payload)
        full = HeadData(matrix)
        self.pop_stats = full.stats(np.arange(full.n))
        self.mults = a542_mults(len(uniq), cfg)

    def fold_manifest(self):
        body = {c: self.folds[c] for c in sorted(self.folds)}
        return {
            "n_rows": len(self.ids),
            "n_components": len(self.comp_ids),
            "rows_per_fold": np.bincount(self.fold, minlength=N_OUTER).tolist(),
            "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        }

    def hstar(self, r):
        raw = hstar_raw(self.data, self.pop_stats, r)
        return hstar_values(raw)

    def calibrate(self, h, r):
        cfg = self.cfg
        if cfg.calibration == "f12b":
            args = (self.payload, self.comp, self.ids)
            eps_c = calibration_panels(*args, cfg.cal_draws, "cal", self.k, r, cfg)
            eps_v = calibration_panels(*args, cfg.val_draws, "calval", self.k, r, cfg)
            return calibrate_gamma_f12b(self.P, self.a, h, eps_c, eps_v, self.tau, cfg)
        rc = np.random.default_rng(domain_seed(FOLD_DOMAIN, f"gamma-cal:{self.k}:{r}"))
        rv = np.random.default_rng(domain_seed(FOLD_DOMAIN, f"gamma-val:{self.k}:{r}"))
        eps_c = np.stack(
            [draw_panel_noise(self.payload, self.comp, rc)[0].mean(1) for _ in range(cfg.cal_draws)]
        )
        eps_v = np.stack(
            [draw_panel_noise(self.payload, self.comp, rv)[0].mean(1) for _ in range(cfg.val_draws)]
        )
        return calibrate_gamma(self.P, self.a, h, eps_c, eps_v, cfg)

    def fit_all(self, y):
        n = len(y)
        wm, st = np.full(n, np.nan), np.full(n, np.nan)
        info = []
        for f in range(N_OUTER):
            test, train = np.flatnonzero(self.fold == f), np.flatnonzero(self.fold != f)
            if not len(test):
                continue
            wm[test], iw = fit_wmr_fold(
                self.data, self.a, self.q, y, self.m["component"], train, test, self.cfg,
                data_g=self.data_g, fold=f,
            )  # fmt: skip
            st[test], isn = fit_statnn_fold(
                self.F, self.a, self.q, y, self.m["component"], train, test, self.cfg
            )
            info.append({"fold": f, **iw, **isn})
        f3 = self.P @ y
        return f3, wm, st, info


def gauge_record(ctx, info):
    """The replicate's F5 gauge audit over every fitted WM-R model (each seed, each outer fold)."""
    if ctx.data_g is None:
        return {"applied": False, "reason": "no gauge-transformed tokens in the matrix (synthetic tokens have no "
                "factors); a real token matrix always carries them"}  # fmt: skip
    fits = [x for i in info for x in i["gauge"]["fits"]]
    disc = [x for i in info for x in i["gauge"]["discrepancies"]]
    viol = sum(i["gauge"]["violations"] for i in info)
    return {
        "applied": True,
        "rule": "F5 pins: held-out rows, every seed and outer fold; |f_gauge - f| <= 1e-5 + 1e-4 |f| nats per row",
        "fits": fits,
        "rows_audited": sum(x["rows"] for x in fits),
        "violations": viol,
        "max_abs_delta": max((x["max_abs_delta"] for x in fits), default=0.0),
        "max_delta_over_tolerance": max((x["delta"] / x["tolerance"] for x in disc), default=0.0),
        "discrepancies": disc,
    }


def run_replicate(ctx, scenario, r):
    alt, var = SCENARIOS[scenario]
    cfg = ctx.cfg
    rec = {"scenario": scenario, "k": ctx.k, "r": r}
    rec["seed"] = replicate_seed(cfg, scenario, ctx.k, r)
    gamma, h = 0.0, np.zeros(len(ctx.a))
    if alt:
        try:
            h = ctx.hstar(r)
        except HStarDegenerate as exc:
            rec.update(valid=False, decision="unresolved_validity", failure=f"h*: {exc}")
            return rec
        gamma, cal = ctx.calibrate(h, r)
        rec["calibration"] = cal
        if gamma is None:
            rec.update(valid=False, decision="unresolved_validity", failure="gamma calibration")
            return rec
    rec["gamma"] = gamma
    rng = np.random.default_rng(rec["seed"])
    E, _, _ = draw_panel_noise(
        ctx.payload, ctx.comp, rng, math.sqrt(var), order=cfg.noise_draw, row_ids=ctx.ids
    )
    D = (ctx.a + gamma * h)[:, None] + E
    y = D.mean(1)
    try:
        f3, wm, st, info = ctx.fit_all(y)
    except ValueError as exc:  # registered fit failures (e.g. an empty inner fold); bugs propagate
        rec.update(valid=False, decision="unresolved_validity", failure=f"fit: {exc}")
        return rec
    gauge = gauge_record(ctx, info)
    A = {k: float(np.abs(y - p).sum()) for k, p in (("F3", f3), ("WM", wm), ("S", st))}
    A["oracle"] = float(np.abs(y - ctx.a - gamma * h).sum())
    rec["A"] = A
    rec["C_S_F3"] = 1 - A["S"] / A["F3"]
    rec["C_oracle_F3"] = 1 - A["oracle"] / A["F3"]
    cw = a542_contrast(D, wm, f3, ctx.comp, ctx.mults, rule=cfg.joint_studentiser)
    cs = a542_contrast(D, wm, st, ctx.comp, ctx.mults, rule=cfg.joint_studentiser)
    if cfg.stage:  # F12b.2 descriptive sensitivity: the other joint rules on the same draws
        for c, qb in ((cw, f3), (cs, st)):
            if "studentiser" not in c:
                continue
            c["studentiser"]["sensitivity"] = {}
            for rule in STUDENTISER_RULES:
                if rule != cfg.joint_studentiser:
                    alt_c = a542_contrast(D, wm, qb, ctx.comp, ctx.mults, rule=rule)
                    st = alt_c.get("studentiser", {})
                    c["studentiser"]["sensitivity"][rule] = {
                        "valid": alt_c["valid"],
                        "joint_bootstrap_t": alt_c["intervals"].get("joint_bootstrap_t"),
                        **{
                            x: st.get(x)
                            for x in (
                                "floor_binding",
                                "raw_nonpositive",
                                "raw_nonfinite",
                                "vcomp_invalid",
                                "point_floor_binding",
                            )
                        },
                    }
    if gauge["applied"] and gauge["violations"]:
        # F5 pins / F12.3: a WM-R gauge violation makes WM-R's contrasts undefined; F12.4 step 1 and F12a.5(d): the
        # replicate is unresolved (validity) and counts as a failure
        for c in (cw, cs):
            c["valid"] = False
            c["undefined"] = "WM-R gauge violation (F5 pins)"
            c["failures"] = list(c.get("failures", [])) + ["WM-R gauge violation"]
        rec["failure"] = f"gauge audit: {gauge['violations']} WM-R violation(s) beyond 1e-5 + 1e-4 |f| nats"
    rec["gauge_audit"] = gauge
    rec["C_WM"], rec["C_WM_S"] = cw, cs
    rec["decision"] = f12_decision(cw, cs)
    rec["valid"] = bool(cw["valid"] and cs["valid"])
    rec["fit"] = {
        "refit_epochs": [i["refit_epochs"] for i in info],
        "trees": [i["trees"] for i in info],
        "wmr": [
            {k: i[k] for k in ("inner_best", "refit_epochs", "target_sd") if k in i} for i in info
        ],
    }
    return rec


# ------------------------------------------------------------------------- Monte Carlo


def mc_bounds(k, n, cfg):
    """Ordinary two-sided 95 % CP and Bonferroni one-sided simultaneous bounds."""
    lo, hi = cp_interval(k, n, cfg.mc_alpha)
    a1 = cfg.mc_alpha / cfg.bonferroni_m  # one-sided level per gating rate
    blo, bhi = cp_interval(k, n, 2 * a1)
    return {
        "count": int(k),
        "n": int(n),
        "rate": k / n if n else float("nan"),
        "cp95": [lo, hi],
        "bonferroni_lower": blo,
        "bonferroni_upper": bhi,
        "one_sided_alpha": a1,
    }


def _beta_bound(k, n, alpha, kind):
    from scipy.stats import beta

    if kind == "lower":
        return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def replicates_needed(p, limit, kind, alpha_one_sided, n_max=200_000):
    """Smallest n (step 10) at which a run whose observed rate equals ``p`` passes the bound.

    kind 'power': count floor(p n), lower bound >= limit; kind 'size': count ceil(p n), upper <= limit."""
    if (kind == "power" and p <= limit) or (kind == "size" and p >= limit):
        return None
    n = 10
    while n <= n_max:
        if kind == "power":
            ok = _beta_bound(math.floor(p * n), n, alpha_one_sided, "lower") >= limit
        else:
            ok = _beta_bound(math.ceil(p * n), n, alpha_one_sided, "upper") <= limit
        if ok:
            return n
        n += 10 if n < 10_000 else 100
    return None


CONTRASTS = ("C_WM", "C_WM_S")
INTERVAL_KEYS = tuple(
    f"{p}_{m}" for p in ("conditional", "joint") for m in ("bootstrap_t", "bca", "percentile")
)
BOOT_T_KEYS = ("conditional_bootstrap_t", "joint_bootstrap_t")
STAGE2_REQUIRED = {"null": 1000, "alt": 1000, "null_var2": 200}


def _dist(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"n": 0}
    return {
        "n": len(v),
        "min": float(v.min()),
        "q25": float(np.quantile(v, 0.25)),
        "median": float(np.median(v)),
        "q75": float(np.quantile(v, 0.75)),
        "max": float(v.max()),
    }


def _lower_pos(con, key):
    return bool(con and con.get("valid") and con["intervals"][key][0] > 0)


def studentiser_summary(recs, cfg):
    """F12c.2 report, per contrast, over the planned replicates of one scenario at one k: amended
    (registered rule) and strict validity with the planned denominator, the one-way-max sensitivity's
    validity, raw non-positive / non-finite draw counts, the floor's draw-binding fraction and its
    distribution over replicates, the point-binding rate, the sensitivities' joint lower bounds and
    binding fractions on the identical draws, and failed guards by cause."""
    n = len(recs)
    out = {}
    for c in CONTRASTS:
        valid = strict = one_way = point = 0
        raw_np, raw_nf, vci, frac = [], [], [], []
        sens_lo = {"strict": 0, "floor_one_way": 0}
        sens_frac = {"strict": [], "floor_one_way": []}
        causes = {}
        for x in recs:
            con = x.get(c)
            if con is None:
                why = x.get("failure") or "no contrast computed"
                causes[f"replicate: {why}"] = causes.get(f"replicate: {why}", 0) + 1
                continue
            valid += bool(con.get("valid"))
            if not con.get("valid"):
                for f in con.get("failures", []):
                    causes[f"contrast: {f}"] = causes.get(f"contrast: {f}", 0) + 1
                for proc, d in con.get("failure_reasons", {}).items():
                    for why, cnt in d.items():
                        if cnt:
                            key = f"{proc}: {why}"
                            causes[key] = causes.get(key, 0) + int(cnt)
            st = con.get("studentiser")
            if not st:
                continue
            draws = st.get("draws") or cfg.draws
            raw_np.append(st.get("raw_nonpositive"))
            raw_nf.append(st.get("raw_nonfinite"))
            vci.append(st.get("vcomp_invalid"))
            frac.append(st.get("floor_binding", 0) / draws)
            point += bool(st.get("point_floor_binding"))
            sens = st.get("sensitivity", {})
            for rule in ("strict", "floor_one_way"):
                if rule == st.get("rule"):
                    ok = bool(con.get("valid"))
                    lo_pos = _lower_pos(con, "joint_bootstrap_t")
                    fb = st.get("floor_binding", 0)
                else:
                    s_r = sens.get(rule) or {}
                    ok = bool(s_r.get("valid"))
                    iv = s_r.get("joint_bootstrap_t")
                    lo_pos = bool(ok and iv is not None and iv[0] > 0)
                    fb = s_r.get("floor_binding")
                if rule == "strict":
                    strict += ok
                else:
                    one_way += ok
                sens_lo[rule] += lo_pos
                if fb is not None:
                    sens_frac[rule].append(fb / draws)

        def tot(v):
            v = [int(a) for a in v if a is not None]
            return {"total": int(sum(v)), "replicates_with_any": int(sum(a > 0 for a in v)),
                    "per_replicate": _dist(v)}  # fmt: skip

        out[c] = {
            "rule": cfg.joint_studentiser,
            "planned": n,
            "amended_valid": mc_bounds(valid, n, cfg),
            "strict_valid": mc_bounds(strict, n, cfg),
            "one_way_max_valid": mc_bounds(one_way, n, cfg),
            "raw_nonpositive_draws": tot(raw_np),
            "raw_nonfinite_draws": tot(raw_nf),
            "vcomp_invalid_draws": tot(vci),
            "floor_binding_fraction": _dist(frac),
            "point_binding": {"count": int(point), "n": n, "rate": point / n if n else None},
            "sensitivity_joint_lower_positive": sens_lo,
            "sensitivity_floor_binding_fraction": {k: _dist(v) for k, v in sens_frac.items()},
            "failed_guards_by_cause": dict(sorted(causes.items())),
            "note": "rates over the planned replicates; strict = A5.4.2 as written on the identical draws",
        }
    return out


def expected_row_sum_target(records, planned, cfg):
    """F12c.3 coverage target: per contrast, the expected-row-sum risk ratio of the frozen procedure,
    C-bar = 1 - Sum_r A_WM,r / Sum_r A_ref,r over the planned, independent target replicates (seed tag
    'target'), with its delta-method Monte Carlo SE and the conservative band target -/+ cov_se_mult SE.
    Fail closed: a missing, duplicated or unplanned replicate, or any fit failure, leaves no target."""
    ids = [(x.get("scenario"), x.get("k"), x.get("r")) for x in records]
    if len(ids) != len(set(ids)) or set(ids) != set(planned):
        return {"status": "failed: target replicates missing, duplicated or unplanned"}
    for x in records:
        A = x.get("A")
        if not A or not all(np.isfinite(A.get(a, np.nan)) for a in ("WM", "F3", "S")):
            why = x.get("failure", "no losses")
            return {"status": f"failed: fit failure in target replicate {x.get('r')} ({why})"}
    n = len(records)
    out = {
        "status": "ok",
        "n": n,
        "estimand": "ratio of expected row sums over independent target replicates (all planned; no "
        "valid-only conditioning: contrast validity is not required for the losses)",
    }
    wm = np.array([x["A"]["WM"] for x in records])
    for c, ref in (("C_WM", "F3"), ("C_WM_S", "S")):
        den = np.array([x["A"][ref] for x in records])
        R = wm.sum() / den.sum()
        infl = (wm - R * den) / den.mean()
        se = float(np.sqrt((infl**2).sum() / (n * (n - 1)))) if n > 1 else float("nan")
        t = float(1 - R)
        out[c] = {
            "target": t,
            "se": se,
            "lo": t - cfg.cov_se_mult * se,
            "hi": t + cfg.cov_se_mult * se,
        }
    return out


def coverage_summary(recs, target, cfg):
    """F12c.3 coverage / error qualification at the expected-row-sum target, planned denominator.
    For each contrast and each of the six A4.4 intervals: coverage of the point target, lower-tail
    error (lower bound > target - cov_se_mult SE) and upper-tail error (upper bound < target +
    cov_se_mult SE); an invalid replicate is an error in both tails and does not cover. Gates (each a
    one-sided CP upper bound at alpha 0.05/36, <= cov_error_limit): the lower tail of the conditional
    and joint bootstrap-t intervals of both contrasts (purchase qualification) and their upper tail
    (required only for the upper-bound labels)."""
    gates = {"n_bounds": 2 * len(CONTRASTS) * len(BOOT_T_KEYS),
             "alpha_each": cfg.mc_alpha / cfg.bonferroni_m,
             "limit": cfg.cov_error_limit, "lower_tail_pass": False, "upper_tail_pass": False}  # fmt: skip
    if not target or target.get("status") != "ok":
        why = (target or {}).get("status", "none supplied")
        return {"status": f"failed: no valid coverage target ({why})", "gates": gates}
    n = len(recs)
    out = {"status": "ok", "n": n, "target": {c: target[c] for c in CONTRASTS}}
    for c in CONTRASTS:
        T = target[c]
        out[c] = {}
        for key in INTERVAL_KEYS:
            cover = lo_err = hi_err = 0
            for x in recs:
                con = x.get(c)
                if not con or not con.get("valid"):
                    lo_err += 1
                    hi_err += 1
                    continue
                lo, hi = con["intervals"][key]
                cover += lo <= T["target"] <= hi
                lo_err += lo > T["lo"]
                hi_err += hi < T["hi"]
            out[c][key] = {
                "covers_target": mc_bounds(cover, n, cfg),
                "lower_tail_error": mc_bounds(lo_err, n, cfg),
                "upper_tail_error": mc_bounds(hi_err, n, cfg),
            }
    lim = cfg.cov_error_limit
    gates["lower_tail_pass"] = all(
        out[c][k]["lower_tail_error"]["bonferroni_upper"] <= lim
        for c in CONTRASTS
        for k in BOOT_T_KEYS
    )
    gates["upper_tail_pass"] = all(
        out[c][k]["upper_tail_error"]["bonferroni_upper"] <= lim
        for c in CONTRASTS
        for k in BOOT_T_KEYS
    )
    out["gates"] = gates
    return out


def _scenario_summary(name, recs, cfg, target):
    n = len(recs)
    valid = [x for x in recs if x["valid"]]
    s = {
        "label": SCENARIO_LABELS.get(name, name),
        "n": n,
        "invalid": n - len(valid),
        "decisions": {
            d: sum(x["decision"] == d for x in recs) for d in sorted({x["decision"] for x in recs})
        },
    }
    if cfg.stage == 4:  # F12a.5(f) pilot: timing, memory and validity only; no gate rate
        s["invalid_by_cause"] = _invalid_causes([x for x in recs if not x["valid"]])
        return s
    s["detected"] = mc_bounds(sum(x["decision"] == "correction_detected" for x in recs), n, cfg)
    s["cond_lower_pos"] = mc_bounds(
        sum(_lower_pos(x.get("C_WM"), BOOT_T_KEYS[0]) for x in valid), n, cfg
    )
    s["joint_lower_pos"] = mc_bounds(
        sum(_lower_pos(x.get("C_WM"), BOOT_T_KEYS[1]) for x in valid), n, cfg
    )
    six = {}
    for contrast in CONTRASTS:
        for key in INTERVAL_KEYS:
            cnt = sum(x[contrast]["intervals"][key][0] > 0 for x in valid)
            six[f"{contrast}:{key}"] = {"count": cnt, "rate": cnt / n if n else None}
    s["lower_bound_positive_rates_A4.4"] = six

    def marginal(c):
        return sum(
            bool(
                x.get(c, {}).get("valid")
                and x[c]["C"] >= 0.05
                and all(_lower_pos(x[c], k) for k in BOOT_T_KEYS)
            )
            for x in recs
        )

    s["marginal_detection"] = {
        "note": "descriptive marginal detection rates (point C >= 0.05 and both bootstrap-t lower bounds "
        "> 0 for that contrast alone), planned denominator; not a purchase route (F12c.4)",
        **{c: mc_bounds(marginal(c), n, cfg) for c in CONTRASTS},
    }
    if valid:
        sums = {a: sum(x["A"][a] for x in valid) for a in ("F3", "WM", "S", "oracle")}
        s["learned_C"] = {
            "estimand": "ratio of expected row sums: 1 - Sum_r A_x,r / Sum_r A_ref,r (no mean of "
            "replicate ratios)",
            "conditioning": f"valid-only: the {len(valid)} replicates of {n} planned with both "
            "contrasts valid",
            "n_valid": len(valid),
            "C_WM_ratio_of_sums": float(1 - sums["WM"] / sums["F3"]),
            "C_WM_S_ratio_of_sums": float(1 - sums["WM"] / sums["S"]),
            "C_S_F3_ratio_of_sums": float(1 - sums["S"] / sums["F3"]),
            "C_oracle_F3_ratio_of_sums": float(1 - sums["oracle"] / sums["F3"]),
        }
    cal = [x["calibration"] for x in recs if "calibration" in x]
    if cal:
        okc = [c for c in cal if c.get("status") == "ok"]
        vals = [c["C_validation"] for c in okc]
        s["calibration"] = {
            "n": len(cal),
            "ok": len(okc),
            "failed": len(cal) - len(okc),
            "gamma_median": float(np.median([c["gamma"] for c in okc])) if okc else None,
            "C_validation_median": float(np.median(vals)) if vals else None,
            "C_validation_range": [float(min(vals)), float(max(vals))] if vals else None,
            "C_validation_within_tol": sum(abs(v - TARGET_C) <= cfg.cal_tol for v in vals),
        }
    if cfg.gating == "f12a":
        s.update(f12a_rates(recs, cfg))
    s["studentiser"] = studentiser_summary(recs, cfg)
    if name == "null":
        s["coverage"] = (
            coverage_summary(recs, target, cfg)
            if target is not None
            else {"status": "not computed: no coverage target supplied (F12c.3)"}
        )
    else:
        s["coverage"] = {
            "status": "not computed: no registered coverage target for this scenario (F12c.3)"
        }
    return s


def summarise(records, cfg, target=None):
    """Per k and scenario: planned-denominator rates, the F12a.5 rates and gates, studentiser and
    coverage reports (F12c.2-3). ``target``: the F12c.3 expected-row-sum target of the stress scenario."""
    out = {}
    by_k = {}
    for rec in records:
        by_k.setdefault(rec["k"], {}).setdefault(rec["scenario"], []).append(rec)
    for k, scen in sorted(by_k.items()):
        row = {
            name: _scenario_summary(name, recs, cfg, target) for name, recs in sorted(scen.items())
        }
        if cfg.stage != 4:
            row["gates"] = f12a_gates(row, cfg) if cfg.gating == "f12a" else _legacy_gates(row, cfg)
        out[str(k)] = row
    return out


def _legacy_gates(row, cfg):
    """F12 as drafted (gating 'f12'; not registered): three size rates among valid replicates."""
    gates = {}
    if "null" in row:
        nul = row["null"]
        sizes = [nul["cond_lower_pos"], nul["joint_lower_pos"], nul["detected"]]
        gates["size_pass"] = all(b["bonferroni_upper"] <= cfg.size_limit for b in sizes)
        gates["size_point_pass"] = all(b["rate"] <= cfg.size_limit for b in sizes)
    if "alt" in row:
        gates["power_pass"] = row["alt"]["detected"]["bonferroni_lower"] >= cfg.power_limit
        gates["power_point_pass"] = row["alt"]["detected"]["rate"] >= cfg.power_limit
    gates["both_pass"] = bool(gates.get("size_pass") and gates.get("power_pass"))
    return gates


STRESS_RATES = {  # F12a.5(b) rates (2)-(5), now stress-scenario safety requirements: name -> (contrast, proc)
    "C_WM_cond_lower": ("C_WM", "conditional"),
    "C_WM_joint_lower": ("C_WM", "joint"),
    "C_WM_S_cond_lower": ("C_WM_S", "conditional"),
    "C_WM_S_joint_lower": ("C_WM_S", "joint"),
}


def _invalid_causes(bad):
    causes = {}
    for x in bad:
        why = x.get("failure") or "; ".join(
            f"{c}: {', '.join(x[c].get('failures', []))}"
            for c in CONTRASTS
            if c in x and not x[c].get("valid", True)
        )
        causes[why] = causes.get(why, 0) + 1
    return dict(sorted(causes.items()))


def f12a_rates(recs, cfg):
    """F12a.5(b)-(d): the five stress-scenario rates and the power rate over all planned replicates;
    an invalid replicate is a rejection for every stress rate and a failure (no detection) for power.
    F12c.3: the five rates are operational safety requirements, not type-I size certification."""
    n = len(recs)
    bad = [x for x in recs if not x["valid"]]
    good = [x for x in recs if x["valid"]]
    out = {}
    for name, (contrast, proc) in STRESS_RATES.items():
        cnt = sum(x[contrast]["intervals"][f"{proc}_bootstrap_t"][0] > 0 for x in good)
        out[name] = mc_bounds(cnt + len(bad), n, cfg)
    det = sum(x["decision"] == "correction_detected" for x in recs)
    return {
        "stress_rates": {**out, "detected": mc_bounds(det + len(bad), n, cfg)},
        "power": mc_bounds(det, n, cfg),
        "invalid_by_cause": _invalid_causes(bad),
    }


def f12a_gates(row, cfg):
    g = {}
    if "null" in row:
        sz = row["null"]["stress_rates"]
        g["stress_pass"] = all(b["bonferroni_upper"] <= cfg.size_limit for b in sz.values())
        g["stress_point_pass"] = all(b["rate"] <= cfg.size_limit for b in sz.values())
        cg = row["null"].get("coverage", {}).get("gates")
        g["coverage_lower_pass"] = bool(cg["lower_tail_pass"]) if cg else None
        g["coverage_upper_pass"] = bool(cg["upper_tail_pass"]) if cg else None
    if "alt" in row:
        g["power_pass"] = row["alt"]["power"]["bonferroni_lower"] >= cfg.power_limit
        g["power_point_pass"] = row["alt"]["power"]["rate"] >= cfg.power_limit
    g["both_pass"] = bool(g.get("stress_pass") and g.get("power_pass"))
    g["qualification_pass"] = bool(g["both_pass"] and g.get("coverage_lower_pass") is True)
    return g


def select_stage2_k(summary, cfg):
    """F12b.4 stage-1 rule (deterministic; outcome-free simulation output only). With the pilot's
    replicates per k: k* = the smallest k of the six whose pilot power rate >= stage1_power_min and
    whose five pilot stress rates are each <= stage1_size_max. Stage 2 runs at {k*} only; if no k
    qualifies, at {145} only."""
    if sorted(int(k) for k in summary) != sorted(K_GRID):
        raise ValueError("the stage-1 summary must hold all six k values")
    checks = []
    for k in sorted(K_GRID):
        row = summary[str(k)]
        if "null" not in row or "alt" not in row:
            raise ValueError(f"stage 1 needs null and alt at k = {k}")
        p = row["alt"]["power"]["rate"]
        smax = max(b["rate"] for b in row["null"]["stress_rates"].values())
        ok = p >= cfg.stage1_power_min and smax <= cfg.stage1_size_max
        checks.append({"k": k, "power_rate": p, "stress_rate_max": smax, "meets": ok})
    met = [c["k"] for c in checks if c["meets"]]
    k_star = met[0] if met else None
    return {
        "rule": f"smallest k with pilot power >= {cfg.stage1_power_min} and every pilot stress rate "
        f"<= {cfg.stage1_size_max}; stage 2 at that k only, else at 145 only",
        "checks": checks,
        "rule_met_at": k_star,
        "k": [k_star if k_star is not None else max(K_GRID)],
    }


def replicate_requirements(summary, cfg):
    a1 = cfg.mc_alpha / cfg.bonferroni_m
    hyp = {
        "power": {
            str(p): replicates_needed(p, cfg.power_limit, "power", a1)
            for p in (0.84, 0.85, 0.9, 0.95)
        },
        "stress": {
            str(p): replicates_needed(p, cfg.size_limit, "size", a1)
            for p in (0.03, 0.04, 0.05, 0.06)
        },
    }
    observed = {}
    for k, row in summary.items():
        o = {}
        if "detected" in row.get("alt", {}):
            o["power"] = replicates_needed(
                row["alt"]["detected"]["rate"], cfg.power_limit, "power", a1
            )
        if "detected" in row.get("null", {}):
            worst = max(
                row["null"][x]["rate"] for x in ("cond_lower_pos", "joint_lower_pos", "detected")
            )
            o["stress_worst_rate"] = worst
            o["stress"] = replicates_needed(worst, cfg.size_limit, "size", a1)
        observed[k] = o
    return {
        "rule": "smallest n (step 10 below 10,000) at which a run observing exactly the given rate passes the "
        f"Bonferroni one-sided exact CP bound (alpha {a1:.6g} each); None = unreachable at any n",
        "hypothetical": hyp,
        "at_observed_rates": observed,
    }


# ---------------------------------------------- F12c: selection freeze, validator, assembly

SELECTION_SCHEMA = "f12c-stage2-selection-v1"
STAGE2_SELECTION = ROOT / "docs/prereg/f12_manifests/f12c_stage2_selection.json"
NOISE_CHECK_SCHEMA = "f12c-noise-check-v1"
PILOT_ACCEPTANCE_SCHEMA = "f12d-pilot-acceptance-v2"  # F12d; the bare f12c v1 record is superseded
TIMING_SCHEMA = "f12d-timing-v1"
C1_SCHEMA = "f12-code-hashes-c1-v1"
CODE_HASHES_C1 = ROOT / "docs/prereg/f12_manifests/code_hashes_c1.json"
C1_DEPENDENCIES = (
    "scripts/a5_build_panels.py",
    "scripts/a5_simulation.py",
    "scripts/tfv1_requalify.py",
    "src/wsp/features/pool.py",
)
CODE_IDENTITY_FILES = ("scripts/f12_simulation.py",) + C1_DEPENDENCIES
# F12d: C1 is no longer a supplied prerequisite; the validator verifies the committed C1 itself (f12_c1.verify_c1).
PREREQUISITES = (
    "noise_check",
    "pilot_report",
    "pilot_timing",
    "pilot_acceptance",
    "coverage_target",
)


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=_json_default))


def current_code_identity():
    return {rel: sha256_file(ROOT / rel) for rel in CODE_IDENTITY_FILES}


def _check_records(records, cfg, planned, what):
    got = [(x.get("scenario"), x.get("k"), x.get("r")) for x in records]
    if len(got) != len(set(got)):
        raise ValueError(
            f"{what}: a replicate id occurs more than once (planned ids must be unique)"
        )
    missing, extra = set(planned) - set(got), set(got) - set(planned)
    if missing or extra:
        raise ValueError(
            f"{what}: records do not match the planned ids ({len(missing)} missing, {len(extra)} unplanned)"
        )
    for x in records:
        sid = (x["scenario"], x["k"], x["r"])
        if x.get("seed") != replicate_seed(cfg, *sid):
            raise ValueError(f"{what}: replicate seed mismatch at {sid}")
        if "decision" not in x or "valid" not in x:
            raise ValueError(f"{what}: replicate {sid} is not complete")


def make_report(*, cfg, config_file, records, k_grid, identities, selection=None, target=None,
                part=None, inputs=None):  # fmt: skip
    """The F12 simulation report (F12c layout). Its release field is computed with the execution
    validator not run, so it is never eligible; ``main`` re-computes it for a complete governed
    stage-2 report when the prerequisites are supplied."""
    planned = planned_ids(cfg, k_grid) if cfg.schedule else None
    summary = summarise(records, cfg, target)
    validation = {
        "passed": False,
        "failures": ["execution validator not run (prerequisites not supplied)"],
    }
    synthetic = identities.get("tokens_synthetic") is not False
    report = {
        "schema": REPORT_SCHEMA,
        "status": "synthetic dry run, not a gate" if synthetic else "simulation",
        "config": cfg.as_dict(),
        "config_file": config_file,
        "config_registered": registered_name(cfg),
        "identities": identities,
        "k_grid": [int(k) for k in k_grid],
        "planned": None if planned is None else {"n": len(planned), "sha256": _sha_json(planned)},
        "part": part,
        "stage1_selection": selection,
        "coverage_target": target,
        "rules": rules_text(cfg),
        "summary": summary,
        "release": release_decision(summary, cfg, validation),
        "records": records,
    }
    if inputs is not None:
        report["inputs"] = inputs
    return report


def verify_stage1_report(path, identities=None, allow_synthetic=False):
    """F12c.5 / F12c.6: verify a complete, immutable stage-1 report and recompute k* from its records.

    Checks: schema; the config equals the pinned F12c stage-1 config (bytes sha256 too); non-synthetic
    inputs unless ``allow_synthetic`` (then the selection is labelled synthetic and can never select a
    governed run); token / payload / code identities equal ``identities`` when given; the k grid is
    all six k; exactly the 480 planned (scenario, k, r) records, each with its registered seed and
    complete; the stored summary equals the summary recomputed from the records. Returns the
    selection record that ``write_selection`` freezes."""
    raw = Path(path).read_bytes()
    rep = json.loads(raw)
    if rep.get("schema") != REPORT_SCHEMA:
        raise ValueError("stage-1 report schema is not " + REPORT_SCHEMA)
    ref = _pinned_configs().get("f12c_stage1")
    try:
        cfg = config_from_dict(rep["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"stage-1 report config unreadable: {exc}") from exc
    if ref is None or cfg != ref:
        raise ValueError("stage-1 report config is not the registered F12c stage-1 config")
    if (rep.get("config_file") or {}).get("sha256") != F12C_CONFIG_SHA256["stage1"]:
        raise ValueError("stage-1 report config file sha256 is not the pinned F12c stage-1 config")
    ids = rep.get("identities") or {}
    synthetic = ids.get("tokens_synthetic") is not False
    if synthetic and not allow_synthetic:
        raise ValueError("stage-1 report is on synthetic tokens: it cannot select a governed run")
    if identities is not None and _norm(ids) != _norm(identities):
        raise ValueError(
            "stage-1 report identities (tokens / payload / code) differ from this run's"
        )
    if sorted(rep.get("k_grid") or []) != sorted(K_GRID):
        raise ValueError("stage-1 k grid must be exactly the six k")
    _check_records(rep["records"], cfg, planned_ids(cfg, K_GRID), "stage 1")
    summ = _norm(summarise(rep["records"], cfg))
    if summ != _norm(rep.get("summary")):
        raise ValueError("stage-1 summary does not match the summary recomputed from its records")
    sel = select_stage2_k(summ, cfg)
    return _norm({
        "schema": SELECTION_SCHEMA,
        "stage1_report_sha256": hashlib.sha256(raw).hexdigest(),
        "stage1_config_sha256": F12C_CONFIG_SHA256["stage1"],
        "identities": ids,
        "synthetic": bool(synthetic),
        "selection": sel,
        "k_star": sel["k"][0],
        "frozen_by": "scripts/f12_simulation.py select (verify_stage1_report + write_selection)",
    })  # fmt: skip


def write_selection(sel, path=STAGE2_SELECTION):
    """Freeze the stage-1 selection (immutable; written before any stage-2 replicate)."""
    path = Path(path)
    if sel.get("synthetic") is not False and path.resolve() == STAGE2_SELECTION.resolve():
        raise ValueError("a synthetic selection cannot be written to the governed selection file")
    data = json.dumps(sel, indent=1, sort_keys=True, default=_json_default) + "\n"
    if path.exists():
        if path.read_text() == data:
            return path
        raise ValueError(f"{path} exists and is immutable (frozen stage-2 selection)")
    path.write_text(data)
    return path


def verify_selection(path, stage1_path, identities=None, allow_synthetic=False):
    """Stage 2 / target: the frozen selection file must equal the selection recomputed from the
    stage-1 report (and the identities must equal this run's). Returns the frozen selection."""
    stored = json.loads(Path(path).read_text())
    fresh = verify_stage1_report(stage1_path, identities, allow_synthetic)
    if _norm(stored) != fresh:
        raise ValueError(
            "frozen stage-2 selection file differs from the selection recomputed from the stage-1 report"
        )
    return stored


def verify_code_hashes(obj):
    """Files listed in a code-hash manifest whose current bytes differ from the pinned sha256."""
    bad = []
    for rel, sha in sorted(obj.get("files", {}).items()):
        p = ROOT / rel
        if not p.exists() or sha256_file(p) != sha:
            bad.append(rel)
    return bad


def _check_target_report(tr, sel, ids):
    """The coverage-target prerequisite: a complete governed f12c_target report at k*."""
    fails = []
    try:
        tcfg = config_from_dict(tr["config"])
    except (KeyError, TypeError, ValueError):
        return ["prerequisite coverage_target: config unreadable"], None
    if (
        governed_name(tcfg) != "f12c_target"
        or (tr.get("config_file") or {}).get("sha256") != F12C_CONFIG_SHA256["target"]
    ):
        fails.append("prerequisite coverage_target: not the registered F12c target config")
    if _norm(tr.get("identities")) != _norm(ids):
        fails.append("prerequisite coverage_target: identities differ from the stage-2 run's")
    if sel is None or tr.get("k_grid") != [sel.get("k_star")]:
        fails.append("prerequisite coverage_target: k grid is not the selected k*")
    if sel is not None and _norm(tr.get("stage1_selection")) != _norm(sel):
        fails.append("prerequisite coverage_target: selection differs from the frozen selection")
    try:
        planned = planned_ids(tcfg, tr.get("k_grid") or [])
        _check_records(tr["records"], tcfg, planned, "target")
    except ValueError as exc:
        fails.append(f"prerequisite coverage_target: {exc}")
        return fails, None
    return fails, expected_row_sum_target(tr["records"], planned, tcfg)


def validate_execution(report, *, selection, prereqs, prereq_sha256=None, c1_root=None):
    """F12c.6 fail-closed execution validator for a stage-2 report. Every check must pass for the
    report's numbers to be eligible for Board A's review; the validator never grants purchase
    authority. Checks: the pinned F12c stage-2 config; non-synthetic tokens; the pinned restricted
    payload; the verified frozen selection (non-synthetic, identical in the report, same identities,
    k grid exactly [k*]); exactly the planned 1,000 / 1,000 / 200 records with their seeds, complete;
    the summary recomputed from the records; the prerequisites: the F12a.4(d) noise check bound to the run's
    identities including its code; the chair's ``pilot_acceptance`` bound to the pilot report and timing (by
    file sha256 in ``prereq_sha256``), identities, recomputed all-in projection, 190 CPU-h ceiling and
    chronology (F12d); the coverage-target report reproducing the report's target; and a complete committed
    C1 verified by Board A (``f12_c1.verify_c1`` on ``c1_root``, default the repository)."""
    F = []
    cfg = None
    try:
        cfg = config_from_dict(report["config"])
    except (KeyError, TypeError, ValueError):
        F.append("config unreadable")
    if cfg is None or governed_name(cfg) != "f12c_stage2":
        F.append("config is not the registered F12c stage-2 config")
    if (report.get("config_file") or {}).get("sha256") != F12C_CONFIG_SHA256["stage2"]:
        F.append("config file sha256 is not the pinned F12c stage-2 config")
    if cfg is not None and dict(cfg.schedule) != STAGE2_REQUIRED:
        F.append("config schedule is not the registered 1,000 / 1,000 / 200")
    ids = report.get("identities") or {}
    if ids.get("tokens_synthetic") is not False:
        F.append("synthetic (or unlabelled) token inputs")
    if ids.get("payload_restricted_sha256") != load_restricted_pin()["restricted_copy"]["sha256"]:
        F.append("payload identity is not the pinned restricted copy")
    if not selection:
        F.append("no verified stage-2 selection")
    else:
        if selection.get("synthetic") is not False:
            F.append("stage-2 selection was made on synthetic inputs")
        if _norm(report.get("stage1_selection")) != _norm(selection):
            F.append("the report's stage1_selection differs from the verified frozen selection")
        if _norm(selection.get("identities")) != _norm(ids):
            F.append("token / payload / code identities differ from the stage-1 selection's")
        if report.get("k_grid") != [selection.get("k_star")]:
            F.append(
                f"k grid {report.get('k_grid')} is not exactly the selected k* {selection.get('k_star')}"
            )
    if cfg is not None and cfg.schedule and report.get("k_grid"):
        try:
            _check_records(
                report.get("records", []), cfg, planned_ids(cfg, report["k_grid"]), "stage 2"
            )
        except ValueError as exc:
            F.append(str(exc))
        if report.get("part") is not None:
            F.append("a partial (single-invocation) report: assemble the parts first")
        try:
            again = summarise(report["records"], cfg, report.get("coverage_target"))
            if _norm(again) != _norm(report.get("summary")):
                F.append("summary does not match the summary recomputed from the records")
        except (KeyError, TypeError, ValueError) as exc:
            F.append(f"summary not recomputable: {exc}")
    P = prereqs or {}
    nc = P.get("noise_check")
    if not nc:
        F.append("prerequisite noise_check missing")
    else:
        sub = {
            k: ids.get(k)
            for k in ("tokens_matrix_sha256", "tokens_synthetic", "payload_restricted_sha256")
        }
        if not (nc.get("schema") == NOISE_CHECK_SCHEMA and nc.get("status") == "pass" and nc.get("f12a4d_check") is True
                and nc.get("panels") == 10_000 and nc.get("k") == 145):  # fmt: skip
            F.append(
                "prerequisite noise_check: not a passed F12a.4(d) check (10,000 panels at k = 145)"
            )
        if _norm(nc.get("identities")) != _norm(sub):
            F.append("prerequisite noise_check: identities differ from the stage-2 run's")
        if not nc.get("code_sha256") or _norm(nc.get("code_sha256")) != _norm(
            ids.get("code_sha256")
        ):
            F.append(
                "prerequisite noise_check: code identity (code_sha256) is not the run's current code"
            )
    F += _check_pilot_prereqs(P, prereq_sha256 or {}, ids, nc)
    tr = P.get("coverage_target")
    if not tr:
        F.append("prerequisite coverage_target missing")
    else:
        fails, tgt = _check_target_report(tr, selection, ids)
        F += fails
        if tgt is not None and _norm(tgt) != _norm(report.get("coverage_target")):
            F.append(
                "coverage target in the report differs from the target recomputed from the coverage_target prerequisite"
            )
    F += f12_c1.verify_c1(
        ROOT if c1_root is None else c1_root, code_identity=ids.get("code_sha256") or {}
    )
    return {"passed": not F, "failures": F, "board_a_acceptance": "pending (not checkable by code)"}


def release_decision(summary, cfg, validation):
    """F12c.6: numerical eligibility only, pending Board A acceptance; never purchase authority."""
    out = {
        "field": RELEASE_FIELD,
        "authority": "none: purchase requires Board A's acceptance of the committed stage-2 report "
        "(F12b.4(d)); this field records numerical eligibility only",
        "validator": validation,
        "eligible": False,
        "wording": None,
    }
    if not validation or not validation.get("passed"):
        fails = (validation or {}).get("failures") or ["validator not run"]
        out["numerical_eligibility"] = (
            "none: execution validator did not pass (" + "; ".join(fails[:6]) + ")"
        )
        return out
    if len(summary) != 1:
        out["numerical_eligibility"] = "none: a stage-2 report holds exactly one k"
        return out
    ((k, row),) = summary.items()
    g = row["gates"]
    if g.get("qualification_pass"):
        out["eligible"] = True
        out["numerical_eligibility"] = (
            f"eligible: S_{k} passed the power, stress and coverage requirements at k* = {k}; "
            "pending Board A acceptance"
        )
        return out
    words = []
    if not g.get("power_pass"):
        words.append(FAILURE_WORDING)
    if not g.get("stress_pass"):
        words.append(f"The {STRESS_LABEL} requirements failed at k* = {k}; purchase is blocked.")
    if g.get("coverage_lower_pass") is not True:
        words.append(
            f"The coverage qualification at the expected-row-sum target failed at k* = {k}; purchase is blocked."
        )
    out["wording"] = " ".join(words)
    out["numerical_eligibility"] = (
        f"not eligible at k* = {k}: unresolved - not powered or not qualified"
    )
    return out


def assemble_reports(paths):
    """F12c.6: assemble stage-2 (or target) parts run in separate invocations into one report.
    Every part must carry identical provenance (schema, config and its file sha256, registration,
    identities, k grid, frozen selection, coverage target, planned-id hash); each planned id may occur
    once across parts; the summary is recomputed from the union, records in planned order."""
    parts = []
    for p in paths:
        raw = Path(p).read_bytes()
        parts.append((Path(p), hashlib.sha256(raw).hexdigest(), json.loads(raw)))
    keys = ("schema", "config", "config_file", "config_registered", "identities", "k_grid",
            "stage1_selection", "coverage_target", "planned")  # fmt: skip
    base = parts[0][2]
    for p, _, rep in parts[1:]:
        for key in keys:
            if _norm(rep.get(key)) != _norm(base.get(key)):
                raise ValueError(f"part {p} provenance differs from the first part in {key!r}")
    cfg = config_from_dict(base["config"])
    planned = planned_ids(cfg, base["k_grid"])
    index = {t: i for i, t in enumerate(planned)}
    seen, recs = set(), []
    for p, _, rep in parts:
        for x in rep["records"]:
            sid = (x["scenario"], x["k"], x["r"])
            if sid in seen:
                raise ValueError(f"planned id {sid} occurs more than once across parts")
            if sid not in index:
                raise ValueError(f"unplanned id {sid} in part {p}")
            seen.add(sid)
            recs.append(x)
    recs.sort(key=lambda x: index[(x["scenario"], x["k"], x["r"])])
    report = make_report(cfg=cfg, config_file=base["config_file"], records=recs, k_grid=base["k_grid"],
                         identities=base["identities"], selection=base.get("stage1_selection"),
                         target=base.get("coverage_target"), part=None, inputs=base.get("inputs"))  # fmt: skip
    report["parts"] = [
        {"path": str(p), "sha256": sha, "n_records": len(rep["records"]), "part": rep.get("part")}
        for p, sha, rep in parts
    ]
    report["complete"] = seen == set(planned)
    return _norm(report)


# F12d (Board A F12c condition 7): disclosed conservative extrapolation constants.
# F12b.6 synthetic stress / alternative CPU ratios: 0.888 (k=0), 0.912 (k=87), 0.923 (k=145); stress-only
# timing is scaled up by the smallest ratio for the alternative.
SYNTH_STRESS_ALT_RATIO_MIN = 121.04 / 136.38
# F12c resource plan: the 10,000-panel noise check projected from the 500-panel smoke; 15 invocations of the
# whole programme at 10 CPU-s overhead each (F12c.7(b)).
NOISE_CPU_H_PROJECTED = 0.005
PLANNED_INVOCATIONS = 15
OVERHEAD_CPU_H_PROJECTED = PLANNED_INVOCATIONS * 10 / 3600
ITEM_UNITS = {"noise_check": 1, "pilot": 10, "stage1": 480, "target": 200, "stage2": 2200}
CHAIR_CEILING_CPU_H = 190.0  # chair ruling on F12c, 11 Nov 2026 18:55 UTC


def project_cpu(by_k, *, noise_cpu_h=None, overhead_cpu_h=None, pilot_cpu_h=None):
    """F12c.7 / F12d all-in CPU projection of the F12c schedule from per-replicate process CPU seconds
    {str(k): {scenario: seconds}} with separate stress (``null``) and alternative (``alt``) costs.

    Measured k are interpolated linearly in k. Disclosed conservative extrapolations: a k below the smallest
    measured k is projected at that k's cost (synthetic per-replicate cost rises with k); unmeasured stress is
    projected at the alternative's cost (synthetic stress / alternative ratios 0.89-0.92 <= 1); unmeasured
    alternative at stress / ``SYNTH_STRESS_ALT_RATIO_MIN``; unmeasured variance-x2 at max(stress, alternative).
    Noise check and overhead are included (measured when given, else the resource plan's projections), so the
    result is all-in. A pilot-only projection is labelled an extrapolation, never measured stress timing."""
    measured = sorted(int(k) for k in by_k)
    ext = []

    def xs(scen):
        return [m for m in measured if by_k[str(m)].get(scen) is not None]

    def interp(scen, k):
        pts = xs(scen)
        return float(np.interp(k, pts, [by_k[str(m)][scen] for m in pts]))

    has = {sc: bool(xs(sc)) for sc in ("null", "alt", "null_var2")}
    if not (has["null"] or has["alt"]):
        raise ValueError("project_cpu needs a measured stress or alternative cost")

    def stress(k):
        return interp("null", k) if has["null"] else interp("alt", k)

    def alt(k):
        return interp("alt", k) if has["alt"] else interp("null", k) / SYNTH_STRESS_ALT_RATIO_MIN

    def var2(k):
        return interp("null_var2", k) if has["null_var2"] else max(stress(k), alt(k))

    if not has["null"]:
        ext.append("stress cost per replicate extrapolated: projected at the measured alternative cost (synthetic "
                   "stress / alternative ratios 0.89-0.92, so this is conservative); not measured stress timing")  # fmt: skip
    if not has["alt"]:
        ext.append(f"alternative cost per replicate extrapolated: stress cost / {SYNTH_STRESS_ALT_RATIO_MIN:.4f} "
                   "(the smallest synthetic stress / alternative ratio)")  # fmt: skip
    lo, hi = min(measured), max(measured)
    if any(k < lo for k in K_GRID):
        ext.append(f"every k below k = {lo} projected at the cost measured at k = {lo} (conservative: synthetic "
                   "per-replicate cost rises with k)")  # fmt: skip
    if any(k > hi for k in K_GRID):
        ext.append(f"k above {hi} projected at the cost measured at k = {hi} (not conservative)")
    if not has["null_var2"]:
        ext.append("variance-x2 cost per replicate projected at max(stress, alternative)")
    stage2 = {}
    for k in K_GRID:
        nul, al, v2 = stress(k), alt(k), var2(k)
        stage2[str(k)] = {
            "null_s": nul, "alt_s": al, "null_var2_s": v2,
            "null_var2_from": "measured" if has["null_var2"] else "max(stress, alternative)",
            "interpolated": k not in measured,
            "cpu_h": (STAGE2_REQUIRED["null"] * nul + STAGE2_REQUIRED["alt"] * al
                      + STAGE2_REQUIRED["null_var2"] * v2) / 3600,
        }  # fmt: skip
    stage1 = 40 * sum(stress(k) + alt(k) for k in K_GRID) / 3600
    target = {str(k): 200 * stress(k) / 3600 for k in K_GRID}
    worst2 = max(v["cpu_h"] for v in stage2.values())
    pilot = 10 * alt(145) / 3600 if pilot_cpu_h is None else float(pilot_cpu_h)
    noise = NOISE_CPU_H_PROJECTED if noise_cpu_h is None else float(noise_cpu_h)
    over = OVERHEAD_CPU_H_PROJECTED if overhead_cpu_h is None else float(overhead_cpu_h)
    items = {"noise_check": noise, "pilot": pilot, "stage1": stage1, "target": max(target.values()),
             "stage2": worst2, "overhead": over}  # fmt: skip
    all_in = sum(items.values())
    return {
        "basis": "process CPU seconds per replicate measured in this run; F12c schedule: stage 1 = 40 "
        "stress + 40 alternative at each of six k; stage 2 = 1,000 stress + 1,000 alternative + 200 "
        "variance-x2 at k*; coverage target = 200 stress at k*; pilot = 10 alternative at k = 145; plus the "
        "noise check and overhead; worst case over k*",
        "label": "projection from measured stress and alternative timing" if has["null"] and has["alt"]
        else "projection; stress timing extrapolated from the alternative-only pilot (not measured stress timing)"
        if not has["null"] else "projection; alternative timing extrapolated from stress timing",
        "stress_timing": "measured" if has["null"] else "extrapolated from the alternative (not measured)",
        "alternative_timing": "measured" if has["alt"] else "extrapolated from stress (not measured)",
        "extrapolations": ext,
        "stage2_by_k": stage2,
        "stage2_worst_cpu_h": worst2,
        "stage1_cpu_h": stage1,
        "coverage_target_by_k_cpu_h": target,
        "coverage_target_worst_cpu_h": max(target.values()),
        "pilot_cpu_h": pilot,
        "pilot_cpu_h_source": "projected (10 x alternative at k = 145)" if pilot_cpu_h is None
        else "measured process tree",
        "noise_cpu_h": noise,
        "noise_cpu_h_source": "projected (resource plan)" if noise_cpu_h is None else "measured",
        "overhead_cpu_h": over,
        "overhead_cpu_h_source": "projected (resource plan: 15 invocations x 10 CPU-s)" if overhead_cpu_h is None
        else "measured per-invocation overhead x planned invocations",
        "replicates_total_worst_cpu_h": worst2 + stage1 + max(target.values()) + pilot,
        "items_by_kind_cpu_h": items,
        "units_by_kind": dict(ITEM_UNITS),
        "all_in_worst_cpu_h": all_in,
        "all_in_cpu_h": all_in,
    }  # fmt: skip


def all_in_projection(timing, noise_check):
    """The reproducible all-in projection from a pilot's timing sidecar and the noise-check report (F12d)."""
    per = timing["per_replicate_cpu_by_k_and_scenario"]
    means = {k: {sc: float(v["cpu_mean"]) for sc, v in d.items()} for k, d in per.items()}
    over = PLANNED_INVOCATIONS * max(float(timing.get("cpu_seconds_overhead") or 0.0), 0.0) / 3600
    return project_cpu(
        means,
        noise_cpu_h=float(noise_check["cpu_seconds"]) / 3600,
        overhead_cpu_h=over,
        pilot_cpu_h=float(timing["cpu_seconds_total_process_tree"]) / 3600,
    )


def _utc(x):
    import datetime as _dt

    return _dt.datetime.strptime(str(x), "%Y-%m-%dT%H:%M:%SZ")


def _check_pilot_report(pr, ids):
    """The pilot report prerequisite: the registered F12c pilot config, complete, non-synthetic, same identities."""
    F = []
    try:
        pcfg = config_from_dict(pr["config"])
    except (KeyError, TypeError, ValueError):
        return ["prerequisite pilot report: config unreadable"]
    if (
        governed_name(pcfg) != "f12c_pilot"
        or (pr.get("config_file") or {}).get("sha256") != F12C_CONFIG_SHA256["pilot"]
    ):
        F.append("prerequisite pilot report: not the registered F12c pilot config")
    if (pr.get("identities") or {}).get("tokens_synthetic") is not False:
        F.append("prerequisite pilot report: synthetic (or unlabelled) token inputs")
    if ids is not None and _norm(pr.get("identities")) != _norm(ids):
        F.append("prerequisite pilot report: identities differ from the stage-2 run's")
    try:
        _check_records(pr.get("records", []), pcfg, planned_ids(pcfg, [145]), "pilot report")
    except ValueError as exc:
        F.append(f"prerequisite {exc}")
    return F


def make_pilot_acceptance(*, pilot_report, pilot_timing, noise_check, signed_by, accepted_utc):
    """F12d ``pilot_acceptance`` record for the chair to sign after the measured real-token pilot: binds the pilot
    report and timing hashes, the token / payload / code identities, the recomputed all-in projection and the
    chair's 190 CPU-h ceiling; refuses a projection over the ceiling and an acceptance dated before the pilot
    finished. Writing it remains the chair's act (F12c chair ruling (a))."""
    raw_r, raw_t, raw_n = (Path(x).read_bytes() for x in (pilot_report, pilot_timing, noise_check))
    rep, tim, nc = json.loads(raw_r), json.loads(raw_t), json.loads(raw_n)
    F = _check_pilot_report(rep, None)
    if tim.get("report_sha256") != hashlib.sha256(raw_r).hexdigest():
        F.append("pilot timing is not the timing of this pilot report")
    if not signed_by:
        F.append("pilot acceptance must be signed")
    if _utc(nc["finished_utc"]) >= _utc(tim["started_utc"]):
        F.append("chronology: the noise check must finish before the pilot starts")
    if _utc(accepted_utc) <= _utc(tim["finished_utc"]):
        F.append("chronology: acceptance dated before the pilot finished")
    if F:
        raise ValueError("; ".join(F))
    proj = all_in_projection(tim, nc)
    if proj["all_in_cpu_h"] > CHAIR_CEILING_CPU_H:
        raise ValueError(f"all-in projection {proj['all_in_cpu_h']:.2f} CPU-h exceeds the {CHAIR_CEILING_CPU_H:g} "
                         "CPU-h ceiling: no pilot acceptance; a new dated repricing is needed")  # fmt: skip
    return _norm({
        "schema": PILOT_ACCEPTANCE_SCHEMA,
        "chair_accepted": True,
        "signed_by": signed_by,
        "accepted_utc": accepted_utc,
        "pilot_report_sha256": hashlib.sha256(raw_r).hexdigest(),
        "pilot_timing_sha256": hashlib.sha256(raw_t).hexdigest(),
        "noise_check_sha256": hashlib.sha256(raw_n).hexdigest(),
        "identities": rep["identities"],
        "all_in_ceiling_cpu_h": CHAIR_CEILING_CPU_H,
        "ceiling_source": "PREREG_H1_TF A5 chair ruling on F12c (11 Nov 2026, 18:55 UTC)",
        "projection": proj,
    })  # fmt: skip


def _check_pilot_prereqs(P, shas, ids, nc):
    F = []
    pa = P.get("pilot_acceptance")
    if not pa:
        return [
            "prerequisite pilot_acceptance: chair acceptance of the measured pilot projection missing"
        ]
    if pa.get("schema") != PILOT_ACCEPTANCE_SCHEMA:
        return [(f"prerequisite pilot_acceptance: schema {pa.get('schema')!r} is not "
                 f"{PILOT_ACCEPTANCE_SCHEMA} (a bare chair_accepted record is superseded)")]  # fmt: skip
    if pa.get("chair_accepted") is not True:
        F.append("prerequisite pilot_acceptance: not chair-accepted")
    if not pa.get("signed_by"):
        F.append("prerequisite pilot_acceptance: not signed (signed_by empty)")
    if pa.get("all_in_ceiling_cpu_h") != CHAIR_CEILING_CPU_H:
        F.append(f"prerequisite pilot_acceptance: ceiling {pa.get('all_in_ceiling_cpu_h')!r} is not the chair's "
                 "190 CPU-h")  # fmt: skip
    if _norm(pa.get("identities")) != _norm(ids):
        F.append("prerequisite pilot_acceptance: identities differ from the stage-2 run's")
    pr, pt = P.get("pilot_report"), P.get("pilot_timing")
    if not pr:
        F.append("prerequisite pilot report missing")
    else:
        if shas.get("pilot_report") != pa.get("pilot_report_sha256"):
            F.append("prerequisite pilot report: sha256 differs from the accepted pilot report")
        F += _check_pilot_report(pr, ids)
    if not pt:
        F.append("prerequisite pilot timing missing")
        return F
    if shas.get("pilot_timing") != pa.get("pilot_timing_sha256"):
        F.append("prerequisite pilot timing: sha256 differs from the accepted pilot timing")
    if pt.get("report_sha256") != pa.get("pilot_report_sha256"):
        F.append("prerequisite pilot timing: not the timing of the accepted pilot report")
    if nc:
        try:
            proj = all_in_projection(pt, nc)
        except (KeyError, TypeError, ValueError) as exc:
            return F + [f"prerequisite pilot_acceptance: projection not recomputable ({exc})"]
        got = (pa.get("projection") or {}).get("all_in_cpu_h")
        if not isinstance(got, (int, float)) or round(got, 6) != round(proj["all_in_cpu_h"], 6):
            F.append("prerequisite pilot_acceptance: projection differs from the projection recomputed from "
                     "the pilot timing and noise check")  # fmt: skip
        if proj["all_in_cpu_h"] > CHAIR_CEILING_CPU_H:
            F.append(f"prerequisite pilot: recomputed all-in projection {proj['all_in_cpu_h']:.2f} CPU-h is over "
                     "the 190 CPU-h ceiling")  # fmt: skip
        try:
            if _utc(nc["finished_utc"]) >= _utc(pt["started_utc"]):
                F.append(
                    "prerequisite chronology: the noise check must finish before the pilot starts"
                )
            if _utc(pa["accepted_utc"]) <= _utc(pt["finished_utc"]):
                F.append(
                    "prerequisite chronology: pilot acceptance dated before the pilot finished"
                )
        except (KeyError, ValueError):
            F.append("prerequisite chronology: missing or malformed UTC times")
    return F


# ------------------------------------------------------------------------------- runner

_WORKER = {}


def _init_worker(matrix, payload, cfg, progress=False):
    torch.set_num_threads(1)
    _WORKER.update(matrix=matrix, payload=payload, cfg=cfg, ctx={}, progress=progress)


def _task(args):
    scenario, k, r = args
    w = _WORKER
    if k not in w["ctx"]:
        w["ctx"][k] = Context(w["matrix"], w["payload"], k, w["cfg"])
    t0, c0 = time.perf_counter(), time.process_time()
    rec = run_replicate(w["ctx"][k], scenario, r)
    dt = time.perf_counter() - t0
    dc = time.process_time() - c0
    if w.get("progress"):
        cw, cs = rec.get("C_WM", {}), rec.get("C_WM_S", {})
        why = cw.get("failure_reasons", {}).get("joint") if not cw.get("valid", True) else ""
        print(
            f"[f12] {scenario} k={k} r={r} {dt:.1f}s {rec['decision']} "
            f"C_WM={cw.get('C', float('nan')):.4g} C_WM_S={cs.get('C', float('nan')):.4g} {why}",
            file=sys.stderr,
            flush=True,
        )
    return rec, (dt, dc, _maxrss_mb())


def simulate(matrix, payload, cfg, k_grid=None, scenarios=None, replicates=None, workers=1,
             progress=False, tasks=None, on_done=None):  # fmt: skip
    """Run replicate tasks (scenario, k, r). ``tasks`` (F12c: planned ids of a schedule, or a part of
    them) are executed and reported in the given order; otherwise the legacy grid k_grid x
    scenarios x replicates (executed replicate-major, reported in (k, scenario, r) order)."""
    if workers > MAX_WORKERS:
        raise ValueError(f"at most {MAX_WORKERS} workers (F12c worker-count rule)")
    if tasks is None:
        reps = range(cfg.replicate_offset, cfg.replicate_offset + replicates)
        tasks = [(s, k, r) for k in k_grid for r in reps for s in scenarios]
        order = {
            t: i for i, t in enumerate((s, k, r) for k in k_grid for s in scenarios for r in reps)
        }
    else:
        tasks = [tuple(t) for t in tasks]
        order = {t: i for i, t in enumerate(tasks)}
    results = []
    if workers <= 1:
        _init_worker(matrix, payload, cfg, progress)
        for t in tasks:
            results.append(_task(t))
            if on_done is not None:  # F12d budget checkpoint (may raise BudgetStop)
                on_done(len(results))
    else:
        ctx = multiprocessing.get_context("spawn")  # autograd is not fork-safe
        with ProcessPoolExecutor(
            workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(matrix, payload, cfg, progress),
        ) as ex:
            try:
                for res in ex.map(_task, tasks, chunksize=1):
                    results.append(res)
                    if on_done is not None:  # F12d budget checkpoint (may raise BudgetStop)
                        on_done(len(results))
            except BaseException:
                # no queued replicate starts after a stop
                ex.shutdown(wait=False, cancel_futures=True)
                raise
    results = [x for _, x in sorted(zip(tasks, results), key=lambda p: order[p[0]])]
    return [r for r, _ in results], [t for _, t in results]


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rules_text(cfg):
    """The rules the report's configuration actually executes (F12c.6): one 'effective' block for the
    registered (F12b/F12c) configurations, with no drafted statements beside it; the drafted F12
    defaults are labelled as such."""
    if cfg.stage == 0:
        return {"drafted_not_registered": _rules_f12(cfg)}
    return {"effective": effective_rules(cfg)}


def effective_rules(cfg):
    lg = dict(cfg.lgbm)
    sched = (
        ", ".join(f"{n} {s}" for s, n in cfg.schedule)
        + f" per k (r = {cfg.replicate_offset} + 0..n-1)"
        if cfg.schedule
        else f"legacy: {cfg.replicates} per requested scenario and k"
    )
    k_rule = {
        "all": "all six k {0, 29, 58, 87, 116, 145}",
        "selected": "exactly the frozen stage-1 k* (f12c_stage2_selection.json); no fallback k, no extension",
        "145": "k = 145 only",
        "cli": "as requested on the command line (not registered)",
    }[cfg.k_rule]
    return {
        "config": f"{cfg.name} (stage {cfg.stage})",
        "population": "S_k = existing + first k purchasable components by sha256('20261111:f12:purchase:'+id)",
        "outer_folds": "5; stratum = family or mixed; sha256('20261111:f12:'+id) within stratum; strata "
        "abliteration, chain, lora_sft, mixed, random_dir; round-robin; recomputed per S_k",
        "inner_folds": "4; outer-training components by sha256('20261110:lol-inner:'+id); round-robin",
        "F3": f"ridge, lambda {cfg.ridge_lambda} on training-standardised [Sum a, Sum q], intercept "
        "unpenalised; refitted on each fit's training rows",
        "WM-R": f"F3 + K+F head ({head_param_count()} nominal parameters, {N_COLS} columns); output weights "
        f"zero-initialised: {cfg.head_zero_init}; residual target divided by its training-row SD per "
        f"fit (prediction multiplied back): {cfg.target_standardise}; L1, AdamW lr {cfg.lr} wd "
        f"{cfg.weight_decay} batch {cfg.batch}, <= {cfg.max_epochs} epochs, patience {cfg.patience}, refit "
        f"floor((e2+e3)/2); epoch selection with seeds {list(cfg.seeds)}; refit seeds averaged; frozen for "
        "simulation and analysis (F12b.1, F12c.1)",
        "StatNN": f"F3 + LightGBM (A5.8.4; n_jobs {lg['n_jobs']}) on v4 ({V4_REAL_COLS} columns of the pinned "
        f"pooling code on real tokens; the {V4_COLS}-column stand-in on synthetic tokens only) + 186 x (42 + "
        "presence); trees "
        f"{list(cfg.tree_grid)} chosen by mean inner L1 with seed {', '.join(map(str, cfg.statnn_select_seeds))} "
        f"only; outer refit averages seeds {list(cfg.seeds)}; frozen for simulation and analysis (F12b.5, F12c.1)",
        "noise": f"{cfg.noise_draw}: per simulated component one integers(0, 63) payload component; its rows "
        "by ascending id take choice(n_c, m, replace=m > n_c) donors sorted by id; one common "
        "integers(0, 32, size=32) block index for the panel; restricted payload copy only, byte-pinned",
        "h_star": "untrained WM-R head under torch.manual_seed(seed('20261111:f12:','hstar:<r>')), "
        "zero-edit subtracted, / its population SD over S_k; SD <= 1e-12 fails qualification",
        "calibration": f"grid {{0}} U tau 2^(-8..12); bracket C(0) < 0.15 <= C(g*); monotonicity "
        f"{cfg.cal_monotone} (reported, not gating); bisection to |C_cal - 0.15| <= {cfg.cal_bisect_tol} "
        f"(<= {cfg.cal_halvings} halvings) on {cfg.cal_draws} panels 'cal:<k>:<r>:<d>'; validation on "
        f"{cfg.val_draws} panels 'calval:<k>:<r>:<d>' within {cfg.cal_tol} of 0.15, else failed qualification",
        "scenarios": f"null = {STRESS_LABEL} (gamma = 0; not a no-correction null for C_WM: ridge-1 "
        f"shrinkage is learnable); alt = {SCENARIO_LABELS['alt']}; null_var2 = {SCENARIO_LABELS['null_var2']}",
        "schedule": f"{sched}; k: {k_rule}; panel seeds seed('20261111:f12:', '{cfg.seed_key}') with tag "
        f"'{cfg.seed_tag}'",
        "joint_studentiser": {
            "floor_vcomp": "max(V_comp + V_item - V_both, V_comp) at the point and in every joint draw "
            "(C_WM and C_WM,S only); a draw or the point is invalid if V_comp is non-finite or <= 0 or "
            "any term is non-finite (never replaced); strict and one-way-max rules reported on the "
            "identical draws",
            "strict": "V_comp + V_item - V_both (A5.4.2 as written); <= 0 or non-finite invalid",
            "floor_one_way": "max(V_comp + V_item - V_both, V_comp, V_item) (not registered)",
        }[cfg.joint_studentiser],
        "intervals": f"A5.4.2 conditional + joint bootstrap-t, BCa and percentile, {cfg.draws} draws, "
        f"PCG64({cfg.boot_seed}) multiplicities shared by both contrasts; all guards; all draws valid",
        "decision": "unresolved_validity > correction_detected > none_detectable_provisional > "
        "unresolved; upper-bound codes are provisional until the power gate and the upper-tail "
        "coverage qualification pass",
        "gates": f"power: lower Bonferroni CP bound >= {cfg.power_limit} for 'correction detected' "
        f"(alternative); {STRESS_LABEL}: five rates (C_WM and C_WM,S conditional / joint lower > 0, "
        f"detected) each with upper bound <= {cfg.size_limit} as operational safety requirements, not "
        f"type-I size certification; one-sided alpha {cfg.mc_alpha}/{cfg.bonferroni_m} = "
        f"{cfg.mc_alpha / cfg.bonferroni_m:.6g} each; invalid replicates count as failures / rejections "
        "with the planned denominator",
        "coverage": f"stress scenario: six intervals' coverage of the expected-row-sum target (target "
        f"run, tag 'target'); qualification: lower-tail error (lower > target - {cfg.cov_se_mult} SE) of "
        f"the two bootstrap-t intervals of both contrasts, each upper bound <= {cfg.cov_error_limit} at "
        f"alpha {cfg.mc_alpha}/{cfg.bonferroni_m} (purchase); upper-tail error likewise (upper-bound "
        "labels only)",
        "estimand": "every learned or calibrated C is a ratio of expected row sums; no mean of replicate ratios",
        "release": f"'{RELEASE_FIELD}': fail-closed execution validator; never purchase authority",
        "failure_wording": FAILURE_WORDING,
        "gauge_audit": "F5 pins (F12a.2(i)): every fitted WM-R model, each seed and each outer fold, predicts "
        "every held-out row from its frozen and its gauge-transformed tokens (scripts/f12_gauge_tokens.py); "
        "|delta f| <= 1e-5 + 1e-4 |f| nats per row, f the seed's head prediction in nats; every nonzero "
        "discrepancy retained; any violation makes C_WM and C_WM,S undefined and the replicate unresolved "
        "(validity), counted as a failure (F12a.5(d)); not run on synthetic tokens without gauge tokens",
    }


def _rules_f12(cfg):
    return {
        "population": "S_k = existing + first k purchasable components by sha256('20261111:f12:purchase:'+id)",
        "outer_folds": "5; stratum = family or mixed; sha256('20261111:f12:'+id) within stratum; strata "
        "abliteration, chain, lora_sft, mixed, random_dir; round-robin; recomputed per S_k",
        "inner_folds": "4; outer-training components by sha256('20261110:lol-inner:'+id); round-robin",
        "F3": "ridge, lambda 1 on training-standardised [Sum a, Sum q], intercept unpenalised; training rows only",
        "WM-R": f"F3 + K+F head ({head_param_count()} nominal parameters, {N_COLS} columns), L1 on F3 "
        "residuals, AdamW lr 1e-3 wd 1e-2 batch 16, <= 300 epochs, patience 30, refit floor((e2+e3)/2), "
        "seeds 0/1/2 averaged; F3/standardisation/epochs nested in inner-training splits; per-model "
        "shuffling torch.Generator().manual_seed(s); present-token standardisation, zero-variance columns unscaled",
        "StatNN": "F3 + LightGBM (A5.8.4) on v4 + 186 x (42 + presence); trees {100,300,600} by mean inner L1 "
        "(smaller on ties), seeds 0/1/2 averaged",
        "noise": "repaired: payload component per simulated component; donor permutation within it "
        "(cyclic reuse beyond its donor count); one "
        "common 32-block index vector for all donors; eps = row mean of resampled block values",
        "h_star": "untrained WM-R head, torch.manual_seed(seed('20261111:f12:','hstar:<r>')), tokens "
        "standardised with full-population statistics, w.pooled, / population SD over S_k; SD<=1e-12 fails",
        "gamma": "C_oracle = 1 - sum|eps| / sum|dy - F3_oof(dy)| (ratio of expected row sums over 100 "
        "calibration panels, seed 'gamma-cal:<k>:<r>'); gamma >= 0; grid {0} U geomspace(1e-6, 1, 25) "
        "bracket + monotone check; bisection (100 halvings); |C - 0.15| <= 0.005; validation on 100 "
        "independent panels ('gamma-val:<k>:<r>'); variance-x2 reuses gamma",
        "intervals": "A5.4.2 conditional + joint bootstrap-t (BCa, percentile alongside), 4,000 draws, "
        "PCG64(20261107) multiplicities shared by both contrasts; all guards; all draws valid",
        "decision": "unresolved_validity > correction_detected > none_detectable_provisional > unresolved",
        "gates": f"size: one-sided Bonferroni CP upper <= {cfg.size_limit} for cond-lower>0, joint-lower>0, "
        f"detected (gamma 0); power: lower >= {cfg.power_limit} for detected (invalid = failure); "
        f"alpha {cfg.mc_alpha}/{cfg.bonferroni_m} each; no point-estimate release",
        "replicate_seeds": "seed('20261111:f12:', 'sim:<scenario>:<k>:<r>'), PCG64",
        "gauge_audit": "not run on synthetic tokens (no factors)",
    }


# --------------------------------------------------------- F12a.4(d) noise check (F12c.7)

NOISE_CHECK_PANELS, NOISE_CHECK_K, NOISE_CHECK_SE = 10_000, 145, 4.0


def noise_check_panel(payload, comp, row_ids, d, draw=None):
    """Row noise eps of check panel d from its own stream seed('20261111:f12:', 'noisecheck:<d>'),
    drawn in the F12a.4(b) order (``draw`` overrides the generator; tests only)."""
    rng = np.random.default_rng(domain_seed(FOLD_DOMAIN, f"noisecheck:{d}"))
    if draw is not None:
        return np.asarray(draw(payload, comp, row_ids, rng), dtype=float)
    return draw_panel_noise(payload, comp, rng, order="f12a", row_ids=row_ids)[0].mean(1)


def noise_plug_ins(payload, comp):
    """Plug-in second moments of eps under the F12a.4(b) generator (no model, no outcome).

    Donor d has block means e_d (32); with one common block index b of 32 uniform draws,
    E[eps_i eps_i' | d, d'] = (1/32) S_dd' + (31/32) m_d m_d', S_dd' = (1/32) Sum_l e_dl e_d'l and m_d
    = mean_l e_dl (0 for centred donors up to centring tolerance). A row's donor is uniform over the
    63 payload components and uniform within its component; rows of one simulated component share
    the component and take distinct donors (with replacement when m > n_c)."""
    e = payload.blocks
    m = e.mean(1)
    M2 = (e @ e.T) / BLOCKS / BLOCKS + (BLOCKS - 1) / BLOCKS * np.outer(m, m)
    members = payload.members_by_id
    C = len(members)
    p = np.zeros(len(e))
    for mem in members:
        p[mem] += 1.0 / (C * len(mem))
    var = float(np.sum(p * np.diag(M2)))
    cross = float(p @ M2 @ p)
    comp = np.asarray(comp)
    sizes = np.bincount(comp)
    num = den = 0.0
    for msz in sizes[sizes >= 2]:
        acc = 0.0
        for mem in members:
            block = M2[np.ix_(mem, mem)]
            nc = len(mem)
            if msz > nc:  # with replacement: iid uniform donors (same donor possible)
                acc += block.sum() / nc**2
            else:  # without replacement: uniform over ordered distinct pairs
                acc += (block.sum() - np.trace(block)) / (nc * (nc - 1))
        num += msz * (msz - 1) * acc / C
        den += msz * (msz - 1)
    within = float(num / den) if den else None
    return {
        "variance": var,
        "within": within,
        "cross": cross,
        "donor_variance": np.diag(M2),
        "donor_mean": m,
    }


def noise_check(payload, comp, row_ids, k, panels, draw=None, governed_inputs=False):
    """F12a.4(d): ``panels`` noise-only panels from seeds 'noisecheck:<d>' at the rows of S_k; empirical
    (i) variance of eps, (ii) within-component covariance and (iii) cross-component (common-item)
    covariance against their plug-in values, each with a Monte Carlo SE over independent panels.
    Fails closed if any donor's plug-in panel-mean variance is zero (every payload donor can be
    drawn) or if an empirical quantity differs from its plug-in by more than 4 Monte Carlo SE. It is
    the registered check only at 10,000 panels, k = 145, on governed inputs."""
    t0 = time.process_time()
    comp = np.asarray(comp)
    row_ids = np.asarray(row_ids)
    n = len(comp)
    sizes = np.bincount(comp)
    n_within = float(np.sum(sizes * (sizes - 1)))
    n_cross = float(n * n - np.sum(sizes**2))
    stats = np.zeros((panels, 3))
    for d in range(panels):
        eps = noise_check_panel(payload, comp, row_ids, d, draw)
        tot = eps.sum()
        csum = np.bincount(comp, eps, len(sizes))
        sq = float(np.sum(eps**2))
        within_sum = float(np.sum(csum**2) - sq)
        stats[d] = (
            sq / n,
            within_sum / n_within if n_within else np.nan,
            (tot**2 - np.sum(csum**2)) / n_cross,
        )
    plug = noise_plug_ins(payload, comp)
    checks, failures = {}, []
    for j, (key, pv) in enumerate((("panel_mean_variance", plug["variance"]),
                                    ("within_component_covariance", plug["within"]),
                                    ("cross_component_covariance", plug["cross"]))):  # fmt: skip
        if pv is None:
            checks[key] = {"plug_in": None, "note": "no simulated component with two or more rows"}
            continue
        col = stats[:, j]
        emp = float(col.mean())
        se = float(col.std(ddof=1) / np.sqrt(panels))
        z = (emp - pv) / se if se > 0 and np.isfinite(se) else float("inf")
        checks[key] = {"empirical": emp, "plug_in": float(pv), "mc_se": se, "z": float(z),
                       "correlation_plug_in": float(pv / plug["variance"]) if plug["variance"] > 0 else None}  # fmt: skip
        if not (np.isfinite(z) and abs(z) <= NOISE_CHECK_SE):
            failures.append(
                f"{key}: empirical differs from plug-in by {z:.3g} Monte Carlo SE (> {NOISE_CHECK_SE})"
            )
    dv = plug["donor_variance"]
    if not np.all(dv > 0):
        failures.append(f"zero plug-in panel-mean variance for {int(np.sum(~(dv > 0)))} donor(s)")
    full = panels == NOISE_CHECK_PANELS and k == NOISE_CHECK_K
    return {
        "schema": NOISE_CHECK_SCHEMA,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "f12a4d_check": bool(full and governed_inputs and draw is None),
        "label": "F12a.4(d) noise check"
        if full
        else "smoke: reduced F12a.4(d) noise check, not the registered check",
        "panels": int(panels),
        "k": int(k),
        "rows": int(n),
        "components": len(sizes),
        "seeds": f"seed('20261111:f12:', 'noisecheck:<d>'), d = 0..{panels - 1}",
        "draw_order": "F12a.4(b)" if draw is None else "custom (test)",
        "criterion": f"|empirical - plug-in| <= {NOISE_CHECK_SE} Monte Carlo SE for each quantity; every "
        "donor's plug-in panel-mean variance > 0",
        "checks": checks,
        "donor_plug_in_variance_min": float(dv.min()),
        "donor_block_mean_abs_max": float(np.abs(plug["donor_mean"]).max()),
        "cpu_seconds": time.process_time() - t0,
    }


# ------------------------------------------------------------------ C1 draft code hashes


def c1_draft_inventory():
    """F12c.7: the draft C1 pin (``code_hashes_c1.json``): sha256 of every F12 script, test and config,
    the F12 dependencies, the restricted-payload byte pin and resource plan, with runtime versions."""
    pats = ("scripts/f12*.py", "tests/test_f12*.py", "docs/prereg/f12_manifests/f12*_config.json")
    files = {str(p.relative_to(ROOT)) for pat in pats for p in ROOT.glob(pat)}
    files |= set(C1_DEPENDENCIES)
    files |= {str(RESTRICTED_PIN.relative_to(ROOT))}
    plan = ROOT / "docs/prereg/f12_manifests/f12c_resource_plan.json"
    if plan.exists():
        files.add(str(plan.relative_to(ROOT)))
    pin = load_restricted_pin()
    import platform

    import scipy

    return {
        "schema": C1_SCHEMA,
        "status": "draft, pending Board A C1 verification",
        "note": "Draft of F12a.6(c) item 11. Not a C1 file until MANIFEST_C1.sha256 is committed; it is "
        "regenerated (python scripts/f12_simulation.py c1-draft) after any change to a listed file. "
        "The C1 package draft is c1_draft/ (F12d skeleton; token-dependent parts filled from the frozen token "
        "manifest by F12e, f12e_token_facts.json); MANIFEST_C1.sha256 and Board A verification are still to come.",
        "files": {rel: sha256_file(ROOT / rel) for rel in sorted(files)},
        "restricted_payload": {
            "sha256": pin["restricted_copy"]["sha256"],
            "source_sha256": pin["source"]["sha256"],
            "pin": str(RESTRICTED_PIN.relative_to(ROOT)),
            "extractor": pin["extractor"],
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "lightgbm": _lgbm_version(),
            "scipy": scipy.__version__,
        },
    }


def guard_out(out: Path):
    out = out.resolve()
    for bad in FORBIDDEN_OUT:
        if out == (ROOT / bad).resolve() or (ROOT / bad).resolve() in out.parents:
            raise SystemExit(f"refusing to write under {bad}/")
    return out


def _load_matrix_arg(a, out=None, governed=False):
    """The run's token matrix. Governed real-token work (budget-ledger runs) refuses the synthetic stand-in and
    accepts only the real matrix whose identity equals the committed matrix manifest (F12f; chair ruling on F12e (b))."""
    if governed and a.synthetic:
        raise SystemExit("governed real-token work refuses the synthetic token matrix and its 18-column v4 stand-in "
                         "(chair ruling on F12e (b)); use --tokens with the committed real matrix")  # fmt: skip
    if a.synthetic:
        matrix = synthetic_token_matrix()
        src = {"synthetic": True}
        if out is not None:
            save_matrix(out / "synthetic_tokens.npz", matrix)
            src["path"] = str(out / "synthetic_tokens.npz")
    else:
        matrix = load_matrix(a.tokens)
        src = {"synthetic": bool(matrix["synthetic"]), "path": str(a.tokens), "file_sha256": sha256_file(a.tokens)}
    src["matrix_sha256"] = matrix_sha256(matrix)
    if governed:
        if matrix["synthetic"]:
            raise SystemExit("governed real-token work refuses a synthetic token matrix (chair ruling on F12e (b))")
        try:
            raw = Path(MATRIX_MANIFEST).read_bytes()
            man = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"governed run: committed matrix manifest unreadable: {exc}") from exc
        if man.get("matrix_sha256") != src["matrix_sha256"] or man.get("file_sha256") != src["file_sha256"]:
            raise SystemExit("governed run: the token matrix differs from the committed matrix manifest "
                             f"({MATRIX_MANIFEST})")  # fmt: skip
        src["matrix_manifest"] = {"path": str(MATRIX_MANIFEST), "sha256": hashlib.sha256(raw).hexdigest()}
    return matrix, src


def load_prereqs(paths):
    """{key: path} -> ({key: object}, {key: sha256 of the file bytes}) (F12d: prerequisites bind by file hash)."""
    objs, shas = {}, {}
    for key, path in paths.items():
        raw = Path(path).read_bytes()
        objs[key], shas[key] = json.loads(raw), hashlib.sha256(raw).hexdigest()
    return objs, shas


def _parse_prereqs(items):
    paths = {}
    for item in items or []:
        key, _, path = item.partition("=")
        if key not in PREREQUISITES or not path:
            raise SystemExit(f"--prereq must be one of {PREREQUISITES} as KEY=PATH")
        paths[key] = path
    return load_prereqs(paths)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"run": main_run, "select": main_select, "assemble": main_assemble,
                "noisecheck": main_noisecheck, "c1-draft": main_c1_draft}  # fmt: skip
    if argv and not argv[0].startswith("-"):
        if argv[0] not in commands:
            raise SystemExit(f"unknown command {argv[0]!r}; one of {sorted(commands)}")
        return commands[argv[0]](argv[1:])
    return main_run(argv)


def main_run(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens", type=Path, help="frozen f12-token-matrix-v1 .npz")
    src.add_argument("--synthetic", action="store_true", help="synthetic token matrix (dry runs)")
    p.add_argument(
        "--payload", type=Path, default=None, help="raw payload: drafted (unconfigured) runs only"
    )
    p.add_argument(
        "--restricted-payload", type=Path, default=None, help="pinned restricted copy (F12a.4(e))"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="committed config JSON (F12c)")
    p.add_argument("--stage1-report", type=Path, default=None, help="complete stage-1 report")
    p.add_argument("--selection", type=Path, default=None, help="frozen stage-2 selection file")
    p.add_argument(
        "--part",
        default=None,
        help="run only these scheduled scenarios, e.g. 'null' or 'alt,null_var2'",
    )
    p.add_argument(
        "--coverage-target-report", type=Path, default=None, help="assembled f12c_target report"
    )
    p.add_argument(
        "--prereq", action="append", default=[], help=f"KEY=PATH, KEY in {PREREQUISITES}"
    )
    p.add_argument("--schedule", default=None, help="override, e.g. null=2,alt=1 (non-registered)")
    p.add_argument("--replicates", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="A5.4.2 bootstrap PCG64 seed")
    p.add_argument("--k-grid", default=None)
    p.add_argument("--scenarios", default="null,alt")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--draws", type=int, default=None, help="override (non-registered)")
    p.add_argument("--max-epochs", type=int, default=None, help="override (non-registered)")
    p.add_argument(
        "--tree-grid", default=None, help="StatNN tree counts, e.g. 100 (non-registered)"
    )
    _budget_args(p)
    a = p.parse_args(argv)
    if a.workers > MAX_WORKERS:
        raise SystemExit(
            f"--workers {a.workers}: the F12c worker-count rule allows at most {MAX_WORKERS} on this machine"
        )
    out = guard_out(a.out)
    base, config_sha = (Config(), None) if a.config is None else load_config(a.config)
    if a.config is not None and a.restricted_payload is None:
        raise SystemExit("a configured run reads only the byte-pinned restricted payload copy "
                         "(--restricted-payload); the raw-payload loader is not a governed fallback")  # fmt: skip
    if a.config is not None and a.payload is not None:
        raise SystemExit("a configured run never reads the raw payload (--payload refused; F12d raw-load "
                         "barrier): use --restricted-payload")  # fmt: skip
    if base.schedule and a.replicates is not None:
        raise SystemExit(
            "--replicates override is not available for a scheduled config (use --schedule; non-registered)"
        )
    over = {} if a.seed is None else {"boot_seed": a.seed}
    if a.draws is not None:
        over["draws"] = a.draws
    if a.max_epochs is not None:
        over["max_epochs"] = a.max_epochs
    if a.tree_grid is not None:
        over["tree_grid"] = tuple(int(x) for x in a.tree_grid.split(","))
    if a.schedule is not None:
        if not a.synthetic:
            raise SystemExit("--schedule override is for synthetic dry runs only")
        over["schedule"] = tuple(
            (s, int(n)) for s, n in (x.split("=") for x in a.schedule.split(","))
        )
    cfg = dataclasses.replace(base, **over)
    gname = governed_name(cfg)
    if gname in ("f12c_stage2", "f12c_target"):
        if a.k_grid is not None:
            raise SystemExit("the k grid of a governed stage-2 / target run comes only from the frozen "
                             "stage-1 selection (--k-grid refused)")  # fmt: skip
        if a.selection is None or a.stage1_report is None:
            raise SystemExit("a governed stage-2 / target run needs --selection (the frozen stage-1 "
                             "selection file) and --stage1-report")  # fmt: skip
    elif gname in ("f12c_stage1", "f12c_pilot") and a.k_grid is not None:
        raise SystemExit(f"the k grid of {gname} is fixed by its config (--k-grid refused)")
    budget_kind = {"f12c_pilot": "pilot", "f12c_stage1": "stage1", "f12c_target": "target",
                   "f12c_stage2": "stage2"}.get(gname, "other")  # fmt: skip
    plan = _budget_plan(a, governed=gname is not None and not a.synthetic)
    started_utc = f12_budget.utc_now()
    out.mkdir(parents=True, exist_ok=True)
    matrix, token_src = _load_matrix_arg(a, out, governed=plan is not None)
    if a.restricted_payload is not None:
        try:
            payload = NoisePayload.load_restricted(a.restricted_payload, RESTRICTED_PIN)
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f"restricted payload refused: {exc}") from exc
        restricted = Path(a.restricted_payload)
    else:  # drafted (unconfigured) diagnostics only: the explicit opt-in of the F12d raw-load barrier
        payload = NoisePayload.load(a.payload or DEFAULT_PAYLOAD, allow_raw=a.config is None)
        restricted = out / "noise_payload_restricted.json"
        restricted.write_text(json.dumps(payload.restricted(), sort_keys=True))
    code = current_code_identity()
    identities = {
        "tokens_matrix_sha256": token_src["matrix_sha256"],
        "tokens_synthetic": bool(matrix["synthetic"]),
        "payload_restricted_sha256": sha256_file(restricted),
        "code_sha256": code,
    }
    selection = None
    try:
        if gname in ("f12c_stage2", "f12c_target"):
            selection = verify_selection(a.selection, a.stage1_report, identities=identities)
        elif a.stage1_report is not None:
            selection = verify_stage1_report(a.stage1_report, allow_synthetic=True)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"stage-1 selection refused: {exc}") from exc
    if cfg.k_rule == "all":
        k_grid = list(K_GRID)
    elif cfg.k_rule == "145":
        k_grid = [145]
    elif cfg.k_rule == "selected" and selection is not None and a.k_grid is None:
        k_grid = [selection["k_star"]]
    elif a.k_grid is not None:
        k_grid = [int(x) for x in a.k_grid.split(",") if x != ""]
    else:
        k_grid = list(K_GRID)
    if cfg.schedule:
        planned = planned_ids(cfg, k_grid)
        keep = set(a.part.split(",")) if a.part else None
        tasks = [t for t in planned if keep is None or t[0] in keep]
        scenarios = sorted({t[0] for t in tasks})
        replicates = None
    else:
        replicates = cfg.replicates if a.replicates is None else a.replicates
        scenarios = [s for s in a.scenarios.split(",") if s]
        for s in scenarios:
            if s not in SCENARIOS:
                raise SystemExit(f"unknown scenario {s}")
        tasks = None
    target = None
    if a.coverage_target_report is not None:
        tr = json.loads(Path(a.coverage_target_report).read_text())
        fails, target = _check_target_report(tr, selection, identities)
        if fails or target is None:
            raise SystemExit(f"coverage-target report refused: {fails}")
    n_units = len(tasks) if tasks is not None else len(k_grid) * len(scenarios) * replicates
    meter = None
    if plan is not None:
        meter = f12_budget.Meter(
            f12_budget.Ledger(a.budget_ledger), plan, job=a.job or f"{budget_kind}:{a.part or 'all'}",
            kind=budget_kind, planned_units=n_units, operator=a.operator,
        )  # fmt: skip
        try:
            meter.launch()
        except f12_budget.BudgetStop as exc:
            raise SystemExit(f"budget barrier: launch refused: {exc}") from exc
    ru0 = _rusage_cpu()
    t0 = time.perf_counter()
    try:
        records, times, contexts, diag = _run_body(
            matrix, payload, cfg, k_grid, scenarios, replicates, a.workers, tasks, meter
        )
    except f12_budget.BudgetStop as exc:
        (out / "budget_stop.json").write_text(
            json.dumps({"stopped": str(exc), "report": None}, indent=1)
        )
        raise SystemExit(f"budget barrier: run stopped, no report written: {exc}") from exc
    except BaseException:
        if meter is not None:
            meter.end("failed", done=None)
        raise
    wall = time.perf_counter() - t0
    inputs = {
        "tokens": token_src,
        "payload": payload.provenance,
        "payload_restricted_copy": {"path": str(restricted), "sha256": sha256_file(restricted)},
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "lightgbm": _lgbm_version(),
        },
        "git_head": git_head(),
        "noise_diagnostics": diag,
        "populations": {str(k): c.fold_manifest() for k, c in contexts.items()},
        "head_parameters": head_param_count(),
    }
    report = make_report(cfg=cfg, config_file=None if a.config is None else {"path": str(a.config), "sha256": config_sha},
                         records=records, k_grid=k_grid, identities=identities, selection=selection,
                         target=target, part=a.part, inputs=inputs)  # fmt: skip
    if gname == "f12c_stage2" and a.prereq and not a.part:
        objs, shas = _parse_prereqs(a.prereq)
        validation = validate_execution(
            report, selection=selection, prereqs=objs, prereq_sha256=shas
        )
        report["release"] = release_decision(report["summary"], cfg, validation)
    if cfg.stage != 4:
        report["replicate_requirements"] = replicate_requirements(report["summary"], cfg)
    (out / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=1, default=_json_default)
    )
    _write_timing(out, a, records, times, wall, ru0, k_grid, started_utc=started_utc)
    if meter is not None:
        meter.end("completed")
    print(json.dumps({"summary_k": list(report["summary"]), "release": report["release"]}, indent=1,
                     default=_json_default))  # fmt: skip
    return 0


def _run_body(matrix, payload, cfg, k_grid, scenarios, replicates, workers, tasks, meter):
    contexts = {k: Context(matrix, payload, k, cfg) for k in k_grid}
    last = contexts[k_grid[-1]]
    diag = noise_diagnostics(
        payload, last.comp, domain_seed(FOLD_DOMAIN, "noise-diagnostics"),
        order=cfg.noise_draw, row_ids=last.ids,
    )  # fmt: skip
    diag["draw_order"] = cfg.noise_draw
    diag["note"] = (
        "descriptive 2,000-panel diagnostic; not the F12a.4(d) check (subcommand noisecheck)"
    )
    records, times = simulate(
        matrix, payload, cfg, k_grid, scenarios, replicates, workers, progress=True, tasks=tasks,
        on_done=None if meter is None else meter.tick,
    )  # fmt: skip
    return records, times, contexts, diag


def _budget_args(p):
    p.add_argument(
        "--budget-ledger", type=Path, default=None, help="F12d budget ledger (JSON lines)"
    )
    p.add_argument("--budget-plan", type=Path, default=None,
                   help="resource plan (pre-pilot) or the chair-signed pilot_acceptance (after the pilot)")  # fmt: skip
    p.add_argument("--operator", default=None, help="responsible operator recorded in the ledger")
    p.add_argument("--job", default=None, help="ledger job name (default: kind and part)")


def _budget_plan(a, governed):
    """F12d: governed real-token work cannot start without the budget ledger, a plan and an operator."""
    given = [a.budget_ledger is not None, a.budget_plan is not None, bool(a.operator)]
    if governed and not all(given):
        raise SystemExit("governed real-token work needs --budget-ledger, --budget-plan and --operator "
                         "(F12d budget barrier, 190 CPU-h all-in ceiling)")  # fmt: skip
    if not any(given):
        return None
    if not all(given):
        raise SystemExit(
            "the budget barrier needs all of --budget-ledger, --budget-plan and --operator"
        )
    try:
        return f12_budget.load_plan(a.budget_plan)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"budget plan refused: {exc}") from exc


def _rusage_cpu():
    import resource

    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"self": s.ru_utime + s.ru_stime, "children": c.ru_utime + c.ru_stime}


def _write_timing(out, a, records, times, wall, ru0, k_grid, started_utc=None):
    ru1 = _rusage_cpu()
    cpus = [t[1] for t in times]
    total = (ru1["self"] - ru0["self"]) + (ru1["children"] - ru0["children"])
    by_k = {}
    for rec, t in zip(records, times):
        by_k.setdefault(str(rec["k"]), {}).setdefault(rec["scenario"], []).append(t[1])
    means = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in by_k.items()}
    # F12d: any measured stress or alternative cost gives a schedule projection (an alternative-only pilot
    # included); extrapolations are labelled as such inside the projection
    proj = project_cpu(means) if any(("null" in d or "alt" in d) for d in means.values()) else None
    rp = Path(out) / "report.json"
    (out / "report.time").write_text(json.dumps({
        "schema": TIMING_SCHEMA,
        "report_sha256": sha256_file(rp) if rp.exists() else None,
        "started_utc": started_utc,
        "finished_utc": f12_budget.utc_now(),
        "wall_seconds": wall,
        "workers": a.workers,
        "replicate_tasks": len(times),
        "cpu_seconds_per_replicate_mean": float(np.mean(cpus)) if cpus else None,
        "cpu_seconds_per_replicate_max": float(np.max(cpus)) if cpus else None,
        "cpu_seconds_replicates_sum": float(np.sum(cpus)) if cpus else 0.0,
        "cpu_seconds_total_process_tree": total,
        "cpu_seconds_overhead": total - (float(np.sum(cpus)) if cpus else 0.0),
        "worker_max_rss_mb": float(max(t[2] for t in times)) if times else None,
        "rss_max_mb_parent": _maxrss_mb(),
        "per_replicate_cpu_by_k_and_scenario": {k: {s: {"n": len(v), "cpu_mean": float(np.mean(v))}
                                                    for s, v in d.items()} for k, d in by_k.items()},
        "projection_f12c_schedule": proj,
        "k_grid": k_grid,
    }, indent=1))  # fmt: skip


def main_select(argv):
    p = argparse.ArgumentParser(description="F12c.5: verify the stage-1 report and freeze k*")
    p.add_argument("--stage1-report", type=Path, required=True)
    p.add_argument("--out-selection", type=Path, default=STAGE2_SELECTION)
    p.add_argument(
        "--allow-synthetic", action="store_true", help="dry runs; never the governed file"
    )
    a = p.parse_args(argv)
    try:
        sel = verify_stage1_report(a.stage1_report, allow_synthetic=a.allow_synthetic)
        if _norm(sel["identities"].get("code_sha256")) != _norm(current_code_identity()):
            raise ValueError("stage-1 report code identity differs from the current code")
        write_selection(sel, a.out_selection)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"selection refused: {exc}") from exc
    print(json.dumps({"k_star": sel["k_star"], "file": str(a.out_selection)}))
    return 0


def main_assemble(argv):
    p = argparse.ArgumentParser(
        description="F12c.6: assemble stage-2 / target parts into one report"
    )
    p.add_argument("parts", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--prereq", action="append", default=[], help=f"KEY=PATH, KEY in {PREREQUISITES}"
    )
    p.add_argument("--selection", type=Path, default=None)
    p.add_argument("--stage1-report", type=Path, default=None)
    _budget_args(p)
    a = p.parse_args(argv)
    out = guard_out(a.out)
    plan = _budget_plan(a, governed=False)
    meter = None
    # assembly and verification overhead is charged to the all-in budget (F12d)
    if plan is not None:
        meter = f12_budget.Meter(f12_budget.Ledger(a.budget_ledger), plan, job=a.job or "assembly",
                                 kind="assembly", planned_units=1, operator=a.operator)  # fmt: skip
        try:
            meter.launch()
        except f12_budget.BudgetStop as exc:
            raise SystemExit(f"budget barrier: launch refused: {exc}") from exc
    out.mkdir(parents=True, exist_ok=True)
    try:
        report = assemble_reports(a.parts)
        cfg = config_from_dict(report["config"])
        if governed_name(cfg) == "f12c_stage2" and a.prereq:
            sel = verify_selection(a.selection, a.stage1_report, identities=report["identities"])
            objs, shas = _parse_prereqs(a.prereq)
            v = validate_execution(report, selection=sel, prereqs=objs, prereq_sha256=shas)
            report["release"] = _norm(release_decision(report["summary"], cfg, v))
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"assembly refused: {exc}") from exc
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=1))
    if meter is not None:
        meter.end("completed")
    print(json.dumps({"complete": report["complete"], "release": report["release"]}, indent=1))
    return 0


def main_noisecheck(argv):
    p = argparse.ArgumentParser(description="F12a.4(d) noise-only check (F12c.7)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens", type=Path)
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--restricted-payload", type=Path, required=True)
    p.add_argument("--k", type=int, default=NOISE_CHECK_K)
    p.add_argument("--panels", type=int, default=NOISE_CHECK_PANELS)
    p.add_argument("--out", type=Path, required=True)
    _budget_args(p)
    a = p.parse_args(argv)
    out = guard_out(a.out)
    if a.synthetic and any((a.budget_ledger, a.budget_plan, a.operator)):
        raise SystemExit("governed real-token work refuses the synthetic token matrix and its 18-column v4 stand-in "
                         "(chair ruling on F12e (b))")  # fmt: skip
    plan = _budget_plan(a, governed=not a.synthetic)
    started_utc = f12_budget.utc_now()
    out.mkdir(parents=True, exist_ok=True)
    matrix, token_src = _load_matrix_arg(a, governed=plan is not None)
    try:
        payload = NoisePayload.load_restricted(a.restricted_payload, RESTRICTED_PIN)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"restricted payload refused: {exc}") from exc
    rows = s_k_rows(matrix, a.k)
    m = subset_matrix(matrix, rows)
    comps = m["component"]
    code = {c: i for i, c in enumerate(sorted(set(comps.tolist())))}
    comp = np.asarray([code[c] for c in comps])
    meter = None
    if plan is not None:
        meter = f12_budget.Meter(f12_budget.Ledger(a.budget_ledger), plan, job=a.job or "noise_check",
                                 kind="noise_check", planned_units=1, operator=a.operator)  # fmt: skip
        try:
            meter.launch()
        except f12_budget.BudgetStop as exc:
            raise SystemExit(f"budget barrier: launch refused: {exc}") from exc
    try:
        res = noise_check(
            payload, comp, m["row_ids"], a.k, a.panels, governed_inputs=not matrix["synthetic"]
        )
    except BaseException:
        if meter is not None:
            meter.end("failed", done=0)
        raise
    res["started_utc"], res["finished_utc"] = started_utc, f12_budget.utc_now()
    res["identities"] = {
        "tokens_matrix_sha256": token_src["matrix_sha256"],
        "tokens_synthetic": bool(matrix["synthetic"]),
        "payload_restricted_sha256": sha256_file(a.restricted_payload),
    }
    res["tokens"] = token_src
    res["code_sha256"] = current_code_identity()
    res["rss_max_mb"] = _maxrss_mb()
    (out / "noise_check.json").write_text(json.dumps(res, sort_keys=True, indent=1))
    if meter is not None:
        meter.end("completed" if res["status"] == "pass" else "failed")
    print(
        json.dumps(
            {k: res[k] for k in ("status", "label", "panels", "k", "cpu_seconds", "failures")},
            indent=1,
        )
    )
    return 0


def main_c1_draft(argv):
    argparse.ArgumentParser(description="write the draft code_hashes_c1.json (F12c.7)").parse_args(
        argv
    )
    obj = c1_draft_inventory()
    CODE_HASHES_C1.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    skeleton = f12_c1.write_draft(ROOT)  # F12d: the C1 package skeleton cites the draft pin's bytes
    print(json.dumps({"files": len(obj["files"]), "path": str(CODE_HASHES_C1),
                      "skeleton": [str(x.relative_to(ROOT)) for x in skeleton]}))  # fmt: skip
    return 0


def _maxrss_mb():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _lgbm_version():
    try:
        import lightgbm

        return lightgbm.__version__
    except ImportError:
        return None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(main())
