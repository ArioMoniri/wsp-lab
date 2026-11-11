#!/usr/bin/env python3
"""F12.5 outcome-free power simulation (PREREG_H1_TF amendment F12; Board A F12 conditions 2-5). CPU only.

Reads a frozen F12 token matrix (schema ``f12-token-matrix-v1``; until the real builder exists, the
deterministic synthetic generator below, keyed by component) and the F8a.2 qualification noise payload
(``data/runs/a5/nullsize/inputs.json``, sha256 pinned; only ``centred_noise``, ``window_mask``,
``window_ids`` and ``donors[j].{id, component, family}`` are read). No outcome, no ``donor_mean``, no
``data/evals`` file and no Part A payload is read. Writes a deterministic JSON report (+ a ``.time``
sidecar with wall-clock timings, kept out of the report so the report is byte-reproducible).

Fixed rules implemented here (each also written into the report under ``rules``):

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

Usage::

    python scripts/f12_simulation.py --synthetic --out /home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun \\
        --replicates 200 --seed 20261107 --k-grid 145 --scenarios null,alt --workers 4
    # F12b stage 1 (k-range pilot), then stage 2 at the selected k:
    python scripts/f12_simulation.py --tokens X.npz --restricted-payload P.json --out OUT1 \\
        --config docs/prereg/f12_manifests/f12b_stage1_config.json
    python scripts/f12_simulation.py --tokens X.npz --restricted-payload P.json --out OUT2 \\
        --config docs/prereg/f12_manifests/f12b_stage2_config.json --stage1-report OUT1/report.json \\
        --scenarios null,alt
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
DONOR_FIELDS = ("id", "component", "family")
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
V4_COLS = 18


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
    cal_monotone: str = "f12a"  # "f12a" (grid + 5-point non-decreasing check gates) | "convex" (F12b.3)
    noise_draw: str = "f12"  # "f12" (as drafted) | "f12a" (F12a.4(b) draw order)
    gating: str = "f12"  # "f12" (4 rates, size among valid) | "f12a" (F12a.5(b)-(d): 6 rates)
    replicates: int = 1000
    seed_tag: str = "sim"  # replicate seeds seed("20261111:f12:", "<tag>:<scenario>:<k>:<r>")
    replicate_offset: int = 0  # stage 1 uses r = 1000.. so h*, gamma and noise differ from stage 2
    stage: int = 0  # 0 = single-stage F12; 1 = F12b k-range pilot; 2 = F12b registered run
    stage1_power_min: float = 0.90  # F12b.4 selection rule (stage 1 only)
    stage1_size_max: float = 0.07
    statnn_select_seeds: tuple = (0, 1, 2)  # F12b.5: seeds averaged in the inner tree-count selection

    def registered(self) -> bool:
        return registered_name(self) == "f12b_stage2"

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
    "stage": (0, 1, 2),
}
F12B_CONFIGS = {
    "stage1": ROOT / "docs/prereg/f12_manifests/f12b_stage1_config.json",
    "stage2": ROOT / "docs/prereg/f12_manifests/f12b_stage2_config.json",
}
F12B_CONFIG_SHA256 = {
    "stage1": "a3946ebb7250de7b45433605d7934bace7498c6d325cb70bdf1dccd779964a81",
    "stage2": "13e59899a90aa8c44457dfe757bce4d118dfb6d51eefb43e3b437d476580f65c",
}


def config_from_dict(d) -> Config:
    fields = {f.name for f in dataclasses.fields(Config)}
    extra = set(d) - fields
    if extra:
        raise ValueError(f"unknown config keys {sorted(extra)}")
    d = dict(d)
    for k in ("seeds", "tree_grid", "statnn_select_seeds"):
        if k in d:
            d[k] = tuple(d[k])
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


def registered_name(cfg):
    """'f12b_stage1' / 'f12b_stage2' if cfg equals that committed, sha256-pinned config, else None."""
    for stage, path in F12B_CONFIGS.items():
        if not path.exists():
            continue
        ref, sha = load_config(path)
        if sha == F12B_CONFIG_SHA256[stage] and cfg == ref:
            return f"f12b_{stage}"
    return None


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


def validate_matrix(m):
    if m.get("schema") != TOKEN_SCHEMA:
        raise ValueError(f"token matrix schema must be {TOKEN_SCHEMA}")
    n = len(m["row_ids"])
    for k in ("component", "family", "origin", "eligible", "sum_a", "sum_q"):
        if len(m[k]) != n:
            raise ValueError(f"{k} has {len(m[k])} rows, expected {n}")
    if m["tokens"].ndim != 2 or m["tokens"].shape[1] != N_COLS:
        raise ValueError(f"tokens must be [T, {N_COLS}]")
    if m["v4"].shape != (n, V4_COLS):
        raise ValueError(f"v4 must be [{n}, {V4_COLS}]")
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
    for k in MATRIX_KEYS:
        a = np.asarray(m[k])
        a = a.astype("U") if a.dtype.kind in "US" else np.ascontiguousarray(a)
        h.update(f"{k}:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_matrix(path, m):
    validate_matrix(m)
    arrays = {k: np.asarray(m[k]) for k in MATRIX_KEYS}
    np.savez_compressed(
        path, schema=np.asarray(m["schema"]), synthetic=np.asarray(m["synthetic"]), **arrays
    )


def load_matrix(path):
    with np.load(path, allow_pickle=False) as z:
        m = {k: z[k] for k in MATRIX_KEYS}
        m["schema"] = str(z["schema"])
        m["synthetic"] = bool(z["synthetic"])
    return validate_matrix(m)


def subset_matrix(m, rows):
    rows = np.asarray(rows, dtype=np.int64)
    remap = np.full(len(m["row_ids"]), -1)
    remap[rows] = np.arange(len(rows))
    keep = remap[m["token_row"]] >= 0
    out = {k: m[k][rows] for k in MATRIX_KEYS if not k.startswith("token")}
    out.update({k: m[k][keep] for k in ("token_slot", "token_role", "tokens")})
    out["token_row"] = remap[m["token_row"][keep]]
    order = np.lexsort((out["token_slot"], out["token_row"]))
    for k in ("token_row", "token_slot", "token_role", "tokens"):
        out[k] = out[k][order]
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

    def __init__(self, m):
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
        X[m["token_row"], pos] = m["tokens"]
        R[m["token_row"], pos] = m["token_role"]
        M[m["token_row"], pos] = 1.0
        self.X, self.R, self.M = torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(M)
        self.tokens = np.asarray(m["tokens"], dtype=np.float64)
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


def fit_wmr_fold(data, a, q, y, comps, train, test, cfg):
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
    def load(cls, path=DEFAULT_PAYLOAD, expected_sha256=PAYLOAD_SHA256):
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

    RESTRICTED_SCHEMA = "f12-noise-payload-restricted-v1"
    RESTRICTED_KEYS = {"schema", "provenance", "window_ids", "donors", "centred_noise", "window_mask"}

    @classmethod
    def load_restricted(cls, path):
        """The restricted copy written by ``restricted()`` (F12a.4(e)): exactly the allowlisted keys,
        provenance pinned to the F8a.2 payload sha256; anything else fails closed."""
        raw = Path(path).read_bytes()
        obj = json.loads(raw)
        if set(obj) != cls.RESTRICTED_KEYS or obj["schema"] != cls.RESTRICTED_SCHEMA:
            raise ValueError("not a restricted f12 noise payload (unexpected keys or schema)")
        if obj["provenance"].get("sha256") != PAYLOAD_SHA256:
            raise ValueError("restricted payload provenance is not the pinned F8a.2 payload")
        for d in obj["donors"]:
            if not set(d) <= set(DONOR_FIELDS):
                raise ValueError("donor record outside the allowlist")
        prov = dict(obj["provenance"])
        prov["restricted_copy"] = str(path)
        prov["restricted_copy_sha256"] = hashlib.sha256(raw).hexdigest()
        return cls(obj["centred_noise"], obj["window_mask"], obj["donors"], obj["window_ids"], prov)

    def restricted(self) -> dict:
        return {
            "schema": self.RESTRICTED_SCHEMA,
            "provenance": self.provenance,
            "window_ids": self.window_ids,
            "donors": self.donors,
            "centred_noise": self.noise.tolist(),
            "window_mask": self.mask.tolist(),
        }


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
    """Joint studentising variance under the registered rule (F12b.2).

    strict: V = V_comp + V_item - V_both (A5.4.2; V <= 0 makes the draw invalid).
    floor_vcomp: V = max(V_comp + V_item - V_both, V_comp), V_comp the draw's component-only term.
    floor_one_way: V = max(V_comp + V_item - V_both, V_comp, V_item), the same rule in every draw and
    at the point: the two-way variance is never below either one-way (component-only or item-only)
    variance of the same draw, and is otherwise unchanged. No draw is dropped or replaced."""
    raw = parts["v_comp"] + parts["v_item"] - parts["v_both"]
    bad = ~(np.isfinite(raw) & (raw > 0))
    if rule == "strict":
        return raw, {"raw_nonpositive": int(bad.sum()), "floor_binding": 0}
    if rule in ("floor_one_way", "floor_vcomp"):
        floor = parts["v_comp"]
        if rule == "floor_one_way":
            floor = np.maximum(floor, parts["v_item"])
        with np.errstate(invalid="ignore"):
            binding = ~(raw >= floor)
        var = np.where(binding, floor, raw)
        return var, {"raw_nonpositive": int(bad.sum()), "floor_binding": int(binding.sum())}
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
        & np.isfinite(se_j)
        & (se_j > 0)
    )
    pt = joint_parts(D, qa, qb, comp, (np.ones((1, n_comp)), np.ones((1, BLOCKS))))
    var0, pinfo = joint_variance(pt, rule)
    with np.errstate(invalid="ignore"):
        se0 = np.sqrt(np.where(var0 > 0, var0, np.nan))
    out["studentiser"] = {
        "rule": rule,
        "raw_nonpositive": sinfo["raw_nonpositive"],
        "floor_binding": sinfo["floor_binding"],
        "point_raw_nonpositive": bool(pinfo["raw_nonpositive"]),
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
        "joint": reasons(sa_j, sb_j, rows_j, jc, ln_j, se_j),
    }
    if not ok_c.all():
        failures.append("conditional draws invalid")
    if not ok_j.all():
        failures.append("joint draws invalid")
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
        return "none_detectable_bounds"
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
                self.data, self.a, self.q, y, self.m["component"], train, test, self.cfg
            )
            st[test], isn = fit_statnn_fold(
                self.F, self.a, self.q, y, self.m["component"], train, test, self.cfg
            )
            info.append({"fold": f, **iw, **isn})
        f3 = self.P @ y
        return f3, wm, st, info


def run_replicate(ctx, scenario, r):
    alt, var = SCENARIOS[scenario]
    cfg = ctx.cfg
    rec = {"scenario": scenario, "k": ctx.k, "r": r}
    rec["seed"] = domain_seed(FOLD_DOMAIN, f"{cfg.seed_tag}:{scenario}:{ctx.k}:{r}")
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
                    c["studentiser"]["sensitivity"][rule] = {
                        "valid": alt_c["valid"],
                        "joint_bootstrap_t": alt_c["intervals"].get("joint_bootstrap_t"),
                    }
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


def summarise(records, cfg):
    """Per k: gating rates, MC bounds, gates. Size counts reject only for valid replicates;
    power counts invalid replicates as failures."""
    out = {}
    by_k = {}
    for rec in records:
        by_k.setdefault(rec["k"], {}).setdefault(rec["scenario"], []).append(rec)
    for k, scen in sorted(by_k.items()):
        row = {}
        for name, recs in sorted(scen.items()):
            n = len(recs)
            valid = [x for x in recs if x["valid"]]

            def lower_pos(x, proc):
                return x["C_WM"]["intervals"][f"{proc}_bootstrap_t"][0] > 0

            s = {
                "n": n,
                "invalid": n - len(valid),
                "decisions": {
                    d: sum(x["decision"] == d for x in recs)
                    for d in sorted({x["decision"] for x in recs})
                },
                "detected": mc_bounds(
                    sum(x["decision"] == "correction_detected" for x in recs), n, cfg
                ),
                "cond_lower_pos": mc_bounds(
                    sum(lower_pos(x, "conditional") for x in valid), n, cfg
                ),
                "joint_lower_pos": mc_bounds(sum(lower_pos(x, "joint") for x in valid), n, cfg),
            }
            six = {}
            for contrast in ("C_WM", "C_WM_S"):
                for method in ("bootstrap_t", "bca", "percentile"):
                    for proc in ("conditional", "joint"):
                        key = f"{proc}_{method}"
                        cnt = sum(x[contrast]["intervals"][key][0] > 0 for x in valid)
                        six[f"{contrast}:{key}"] = {"count": cnt, "rate": cnt / n if n else None}
            s["lower_bound_positive_rates_A4.4"] = six
            if valid:
                cw = np.array([x["C_WM"]["C"] for x in valid])
                cs = np.array([x["C_WM_S"]["C"] for x in valid])
                af3 = sum(x["A"]["F3"] for x in valid)
                s["learned_C"] = {
                    "C_WM_mean": float(cw.mean()),
                    "C_WM_S_mean": float(cs.mean()),
                    "C_WM_ratio_of_sums": float(1 - sum(x["A"]["WM"] for x in valid) / af3),
                    "C_WM_S_ratio_of_sums": float(
                        1 - sum(x["A"]["WM"] for x in valid) / sum(x["A"]["S"] for x in valid)
                    ),
                    "C_S_F3_ratio_of_sums": float(1 - sum(x["A"]["S"] for x in valid) / af3),
                    "C_oracle_F3_ratio_of_sums": float(
                        1 - sum(x["A"]["oracle"] for x in valid) / af3
                    ),
                }
            cal = [x["calibration"] for x in recs if "calibration" in x]
            if cal:
                okc = [c for c in cal if c.get("status") == "ok"]
                s["calibration"] = {
                    "n": len(cal),
                    "ok": len(okc),
                    "failed": len(cal) - len(okc),
                    "gamma_median": float(np.median([c["gamma"] for c in okc])) if okc else None,
                    "C_validation_mean": float(np.mean([c["C_validation"] for c in okc]))
                    if okc
                    else None,
                    "C_validation_within_tol": sum(
                        abs(c["C_validation"] - TARGET_C) <= cfg.cal_tol for c in okc
                    ),
                }
            if cfg.gating == "f12a":
                s.update(f12a_rates(recs, cfg))
            row[name] = s
        gates = {}
        if cfg.gating == "f12a":
            row["gates"] = f12a_gates(row, cfg)
            out[str(k)] = row
            continue
        if "null" in row:
            nul = row["null"]
            sizes = [nul["cond_lower_pos"], nul["joint_lower_pos"], nul["detected"]]
            gates["size_pass"] = all(b["bonferroni_upper"] <= cfg.size_limit for b in sizes)
            gates["size_point_pass"] = all(b["rate"] <= cfg.size_limit for b in sizes)
        if "alt" in row:
            gates["power_pass"] = row["alt"]["detected"]["bonferroni_lower"] >= cfg.power_limit
            gates["power_point_pass"] = row["alt"]["detected"]["rate"] >= cfg.power_limit
        gates["both_pass"] = bool(gates.get("size_pass") and gates.get("power_pass"))
        row["gates"] = gates
        out[str(k)] = row
    return out


SIZE_RATES = {  # F12a.5(b) rates (2)-(5): name -> (contrast, procedure)
    "C_WM_cond_lower": ("C_WM", "conditional"),
    "C_WM_joint_lower": ("C_WM", "joint"),
    "C_WM_S_cond_lower": ("C_WM_S", "conditional"),
    "C_WM_S_joint_lower": ("C_WM_S", "joint"),
}


def f12a_rates(recs, cfg):
    """F12a.5(b)-(d): five size rates and the power rate over all planned replicates; an invalid
    replicate is a rejection for every size rate and a failure (no detection) for power."""
    n = len(recs)
    bad = [x for x in recs if not x["valid"]]
    good = [x for x in recs if x["valid"]]
    out = {}
    for name, (contrast, proc) in SIZE_RATES.items():
        cnt = sum(x[contrast]["intervals"][f"{proc}_bootstrap_t"][0] > 0 for x in good)
        out[name] = mc_bounds(cnt + len(bad), n, cfg)
    det = sum(x["decision"] == "correction_detected" for x in recs)
    causes = {}
    for x in bad:
        why = x.get("failure") or "; ".join(
            f"{c}: {', '.join(x[c].get('failures', []))}"
            for c in ("C_WM", "C_WM_S")
            if c in x and not x[c].get("valid", True)
        )
        causes[why] = causes.get(why, 0) + 1
    return {
        "size_rates": {**out, "detected": mc_bounds(det + len(bad), n, cfg)},
        "power": mc_bounds(det, n, cfg),
        "invalid_by_cause": dict(sorted(causes.items())),
    }


def f12a_gates(row, cfg):
    g = {}
    if "null" in row:
        sz = row["null"]["size_rates"]
        g["size_pass"] = all(b["bonferroni_upper"] <= cfg.size_limit for b in sz.values())
        g["size_point_pass"] = all(b["rate"] <= cfg.size_limit for b in sz.values())
    if "alt" in row:
        g["power_pass"] = row["alt"]["power"]["bonferroni_lower"] >= cfg.power_limit
        g["power_point_pass"] = row["alt"]["power"]["rate"] >= cfg.power_limit
    g["both_pass"] = bool(g.get("size_pass") and g.get("power_pass"))
    return g


def select_stage2_k(summary, cfg):
    """F12b.4 stage-1 rule (deterministic; outcome-free simulation output only). With the pilot's
    replicates per k: k* = the smallest k of the six whose pilot power rate >= stage1_power_min and
    whose five pilot size rates are each <= stage1_size_max. Stage 2 runs at {k*} only; if no k
    qualifies, at {145} only."""
    if sorted(int(k) for k in summary) != sorted(K_GRID):
        raise ValueError("the stage-1 summary must hold all six k values")
    checks = []
    for k in sorted(K_GRID):
        row = summary[str(k)]
        if "null" not in row or "alt" not in row:
            raise ValueError(f"stage 1 needs null and alt at k = {k}")
        p = row["alt"]["power"]["rate"]
        smax = max(b["rate"] for b in row["null"]["size_rates"].values())
        ok = p >= cfg.stage1_power_min and smax <= cfg.stage1_size_max
        checks.append({"k": k, "power_rate": p, "size_rate_max": smax, "meets": ok})
    met = [c["k"] for c in checks if c["meets"]]
    k_star = met[0] if met else None
    return {
        "rule": f"smallest k with pilot power >= {cfg.stage1_power_min} and every pilot size rate "
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
        "size": {
            str(p): replicates_needed(p, cfg.size_limit, "size", a1)
            for p in (0.03, 0.04, 0.05, 0.06)
        },
    }
    observed = {}
    for k, row in summary.items():
        o = {}
        if "alt" in row:
            o["power"] = replicates_needed(
                row["alt"]["detected"]["rate"], cfg.power_limit, "power", a1
            )
        if "null" in row:
            worst = max(
                row["null"][x]["rate"] for x in ("cond_lower_pos", "joint_lower_pos", "detected")
            )
            o["size_worst_rate"] = worst
            o["size"] = replicates_needed(worst, cfg.size_limit, "size", a1)
        observed[k] = o
    return {
        "rule": "smallest n (step 10 below 10,000) at which a run observing exactly the given rate passes the "
        f"Bonferroni one-sided exact CP bound (alpha {a1:.6g} each); None = unreachable at any n",
        "hypothetical": hyp,
        "at_observed_rates": observed,
    }


def release_decision(summary, cfg, k_grid, replicates, registered, selected_k=None):
    """Binding only for the registered run. F12 single stage (registered is True): the six-k grid at
    >= 1,000 replicates. F12b (registered == 'f12b_stage2'): exactly the stage-1-selected k values at
    the registered replicate count. A stage-1 pilot or any other config never decides a gate."""
    passing = [int(k) for k, row in summary.items() if row["gates"].get("both_pass")]
    smallest = min(passing) if passing else None
    if registered == "f12b_stage2":
        binding = (
            selected_k is not None
            and sorted(k_grid) == sorted(selected_k)
            and replicates == cfg.replicates
        )
    elif registered is True:
        binding = tuple(sorted(k_grid)) == K_GRID and replicates >= 1000
    else:
        binding = False
    return {
        "binding": bool(binding),
        "note": None
        if binding
        else "non-registered run (config, stage, k grid or replicate count): "
        "a smoke/dry/pilot run, NOT a gate result",
        "smallest_passing_k": smallest,
        "outcome": (f"release S_{smallest}" if smallest is not None else "unresolved - not powered")
        if binding
        else "no gate decision",
    }


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


def simulate(matrix, payload, cfg, k_grid, scenarios, replicates, workers=1, progress=False):
    # executed replicate-major (scenarios interleave); reported in (k, scenario, r) order
    reps = range(cfg.replicate_offset, cfg.replicate_offset + replicates)
    tasks = [(s, k, r) for k in k_grid for r in reps for s in scenarios]
    order = {t: i for i, t in enumerate((s, k, r) for k in k_grid for s in scenarios for r in reps)}
    if workers <= 1:
        _init_worker(matrix, payload, cfg, progress)
        results = [_task(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")  # autograd is not fork-safe
        with ProcessPoolExecutor(
            workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(matrix, payload, cfg, progress),
        ) as ex:
            results = list(ex.map(_task, tasks, chunksize=1))
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
    rules = _rules_f12(cfg)
    if cfg.stage:
        rules["F12b"] = f12b_rules_text(cfg)
    return rules


def f12b_rules_text(cfg):
    """The F12b repairs active in cfg (PREREG_H1_TF A5 amendment F12b)."""
    return {
        "head": f"output weights zero-initialised: {cfg.head_zero_init}; residual target divided by its "
        f"training-row SD (ddof 0) per fit, prediction multiplied back: {cfg.target_standardise}; "
        "token standardisation unchanged (present-token mean/SD on the fit's training rows); "
        "h = 0 at the zero edit",
        "joint_studentiser": f"{cfg.joint_studentiser}: floor_vcomp V = max(V_comp + V_item - V_both, V_comp) "
        "(floor_one_way also max with V_item) in every draw and at the point; conditional procedure "
        "unchanged; no draw dropped; the other rules reported descriptively",
        "calibration": f"{cfg.calibration}: grid {{0}} U tau 2^(-8..12); bracket C(0) < 0.15 <= C(g*); "
        f"monotonicity {cfg.cal_monotone}; bisection to |C_cal - 0.15| <= {cfg.cal_bisect_tol} "
        f"(<= {cfg.cal_halvings} halvings) on {cfg.cal_draws} panels seed 'cal:<k>:<r>:<d>'; "
        f"validation |C_val - 0.15| <= {cfg.cal_tol} on {cfg.val_draws} panels seed "
        "'calval:<k>:<r>:<d>' else failed qualification",
        "noise_draw": f"{cfg.noise_draw} (F12a.4(b) order)",
        "gating": f"{cfg.gating}: six rates (power; C_WM and C_WM,S cond/joint lower > 0; detected), "
        f"Bonferroni alpha {cfg.mc_alpha}/{cfg.bonferroni_m}; invalid = rejection (size) / failure (power)",
        "stage": f"{cfg.stage}: replicates {cfg.replicates}, offset {cfg.replicate_offset}, seed tag "
        f"'{cfg.seed_tag}'; stage-1 rule power >= {cfg.stage1_power_min}, size <= {cfg.stage1_size_max}",
        "statnn": f"tree count selected with seeds {list(cfg.statnn_select_seeds)} averaged; refit "
        f"seeds {list(cfg.seeds)} averaged; LightGBM n_jobs {dict(cfg.lgbm)['n_jobs']}",
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
        "decision": "unresolved_validity > correction_detected > none_detectable_bounds > unresolved",
        "gates": f"size: one-sided Bonferroni CP upper <= {cfg.size_limit} for cond-lower>0, joint-lower>0, "
        f"detected (gamma 0); power: lower >= {cfg.power_limit} for detected (invalid = failure); "
        f"alpha {cfg.mc_alpha}/{cfg.bonferroni_m} each; no point-estimate release",
        "replicate_seeds": "seed('20261111:f12:', 'sim:<scenario>:<k>:<r>'), PCG64",
        "gauge_audit": "not run on synthetic tokens (no factors); the real token builder supplies "
        "gauge-transformed tokens",
    }


def guard_out(out: Path):
    out = out.resolve()
    for bad in FORBIDDEN_OUT:
        if out == (ROOT / bad).resolve() or (ROOT / bad).resolve() in out.parents:
            raise SystemExit(f"refusing to write under {bad}/")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens", type=Path, help="frozen f12-token-matrix-v1 .npz")
    src.add_argument("--synthetic", action="store_true", help="synthetic token matrix (dry runs)")
    p.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    p.add_argument(
        "--restricted-payload", type=Path, default=None,
        help="read only this restricted payload copy (F12a.4(e)) instead of --payload",
    )  # fmt: skip
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="committed config JSON (F12b)")
    p.add_argument(
        "--stage1-report", type=Path, default=None, help="stage-1 report (F12b stage 2: k from it)"
    )
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
    a = p.parse_args(argv)
    out = guard_out(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base, config_sha = (Config(), None) if a.config is None else load_config(a.config)
    selection = None
    if a.stage1_report is not None:
        s1 = json.loads(Path(a.stage1_report).read_text())
        s1_cfg = config_from_dict(s1["config"])
        if registered_name(s1_cfg) != "f12b_stage1":
            raise SystemExit("the stage-1 report was not produced under the registered stage-1 config")
        selection = select_stage2_k(s1["summary"], s1_cfg)
        selection["stage1_report_sha256"] = sha256_file(a.stage1_report)
    if a.k_grid is not None:
        k_grid = [int(x) for x in a.k_grid.split(",") if x != ""]
    elif selection is not None:
        k_grid = list(selection["k"])
    else:
        k_grid = list(K_GRID)
    replicates = base.replicates if a.replicates is None else a.replicates
    scenarios = [s for s in a.scenarios.split(",") if s]
    for s in scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario {s}")
    over = {} if a.seed is None else {"boot_seed": a.seed}
    if a.draws is not None:
        over["draws"] = a.draws
    if a.max_epochs is not None:
        over["max_epochs"] = a.max_epochs
    if a.tree_grid is not None:
        over["tree_grid"] = tuple(int(x) for x in a.tree_grid.split(","))
    cfg = dataclasses.replace(base, **over)
    reg = registered_name(cfg)
    if a.synthetic:
        matrix = synthetic_token_matrix()
        save_matrix(out / "synthetic_tokens.npz", matrix)
        token_src = {"synthetic": True, "path": str(out / "synthetic_tokens.npz")}
    else:
        matrix = load_matrix(a.tokens)
        token_src = {
            "synthetic": False,
            "path": str(a.tokens),
            "file_sha256": sha256_file(a.tokens),
        }
    token_src["matrix_sha256"] = matrix_sha256(matrix)
    if a.restricted_payload is not None:
        payload = NoisePayload.load_restricted(a.restricted_payload)
        restricted = Path(a.restricted_payload)
    else:
        payload = NoisePayload.load(a.payload)
        restricted = out / "noise_payload_restricted.json"
        restricted.write_text(json.dumps(payload.restricted(), sort_keys=True))
    t0 = time.perf_counter()
    contexts = {k: Context(matrix, payload, k, cfg) for k in k_grid}
    last = contexts[k_grid[-1]]
    diag = noise_diagnostics(
        payload, last.comp, domain_seed(FOLD_DOMAIN, "noise-diagnostics"),
        order=cfg.noise_draw, row_ids=last.ids,
    )  # fmt: skip
    diag["draw_order"] = cfg.noise_draw
    records, times = simulate(
        matrix, payload, cfg, k_grid, scenarios, replicates, a.workers, progress=True
    )
    wall = time.perf_counter() - t0
    summary = summarise(records, cfg)
    code = {
        "scripts/f12_simulation.py": sha256_file(__file__),
        "scripts/tfv1_requalify.py": sha256_file(ROOT / "scripts/tfv1_requalify.py"),
        "scripts/a5_simulation.py": sha256_file(ROOT / "scripts/a5_simulation.py"),
        "scripts/a5_build_panels.py": sha256_file(ROOT / "scripts/a5_build_panels.py"),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "synthetic dry run, not a gate" if matrix["synthetic"] else "simulation",
        "git_head": git_head(),
        "config": cfg.as_dict(),
        "config_file": None if a.config is None else {"path": str(a.config), "sha256": config_sha},
        "config_registered": reg,
        "stage1_selection": selection,
        "rules": rules_text(cfg),
        "inputs": {
            "tokens": token_src,
            "payload": payload.provenance,
            "payload_restricted_copy": {"path": str(restricted), "sha256": sha256_file(restricted)},
            "code_sha256": code,
            "versions": {
                "numpy": np.__version__,
                "torch": torch.__version__,
                "lightgbm": _lgbm_version(),
            },
        },
        "k_grid": k_grid,
        "scenarios": scenarios,
        "replicates": replicates,
        "populations": {str(k): c.fold_manifest() for k, c in contexts.items()},
        "head_parameters": head_param_count(),
        "noise_diagnostics": diag,
        "summary": summary,
        "replicate_requirements": replicate_requirements(summary, cfg),
        "release": release_decision(
            summary, cfg, k_grid, replicates, reg, None if selection is None else selection["k"]
        ),
        "records": records,
    }
    (out / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=1, default=_json_default)
    )
    n_tasks = len(times)
    walls = [t[0] for t in times]
    cpus = [t[1] for t in times]
    per = float(np.mean(cpus)) if times else float("nan")
    projected = per * 1000 * len(K_GRID) * 2
    by_k = {
        str(k): {
            sc: {
                "n": len(sel),
                "cpu_mean": float(np.mean([t[1] for t in sel])) if sel else None,
                "wall_mean": float(np.mean([t[0] for t in sel])) if sel else None,
            }
            for sc in scenarios
            for sel in [[t for rec, t in zip(records, times) if rec["k"] == k and rec["scenario"] == sc]]
        }
        for k in k_grid
    }
    (out / "report.time").write_text(
        json.dumps(
            {
                "wall_seconds": wall,
                "workers": a.workers,
                "replicate_tasks": n_tasks,
                "cpu_seconds_per_replicate_mean": per,
                "cpu_seconds_per_replicate_max": float(np.max(cpus)) if times else None,
                "wall_seconds_per_replicate_mean": float(np.mean(walls)) if times else None,
                "worker_max_rss_mb": float(max(t[2] for t in times)) if times else None,
                "projected_cpu_hours_registered_run (1000 x 6 k x null+alt)": projected / 3600,
                "per_replicate_by_k_and_scenario": by_k,
                "rss_max_mb_parent": _maxrss_mb(),
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {"summary": summary, "release": report["release"]}, indent=1, default=_json_default
        )
    )
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
