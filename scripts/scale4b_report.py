"""F5.3 4B scale diagnostic report: diagnostic, not a gate; exposure recorded.

Reads (never writes) the zoo manifest of ``scripts/scale4b_build_edits.py`` and the TF-v1 WikiText
records of the H100 run (``scripts/lambda_job_scale4b.sh``) and, once it exists, of the local H0 half
of the A3 pairing (``scripts/scale4b_h0_pair.sh``); writes one markdown file under docs/reviews/scale/.

Endpoint: token-weighted mean NLL over the 64 registered 1,024-token WikiText windows (log-PPL, nats).
Per edit e: delta_e = logPPL(e) - logPPL(base) on the same device; paired 95% interval by bootstrap over
the 32 two-window blocks (windows 2k, 2k+1; the A2.1 / A4 block rule), 4,000 draws, seed 20261110.
"Improving" means delta_e < 0; "resolved" means the interval excludes 0.

A3 device pairing (PREREG section 6 tolerances; bitwise where A2.4/A3 require it) on the FP4 panel: the
base and the entries of draw indices 0, 4, 8, 12, 16, fixed in the manifest header before scoring:
  (a) identical scoring identity on both devices: config hash, model revision, source hash, per-window
      token-id hashes, and delta sha256 equal to the manifest's;
  (b) the two FP4 same-device repeats (immutable ``repeats/q4b-r1`` and ``repeats/q4b-r2``, each its own
      process and run id): FP4 repeat discrepancy (max |primary - repeat| of the per-window mean NLL)
      <= 1e-4 nats, and bitwise identical on each device (A2.4/A3, additional to FP4);
  (c) cross-device: SD of the centred paired differences (delta_e^H100 - delta_e^H0) <= 0.1 x the
      between-entry SD of delta_e (section 6, WikiText; FP4 transport); a zero between-entry SD fails.
FP4 identity, checked per run on each device (``fp4_identity``): eval device id ``H<n>`` and the device's
``device_identity.json`` (GPU model, driver, CUDA, torch) matching every run; dtype bf16; batch size of the
config; effective config hash equal to ``wsp.eval.batch.effective_config_hash`` of the WikiText config; model
repo and revision equal to the manifest's; WikiText windows 0-63 in registered order with token-id hashes
equal to ``docs/prereg/a5_manifests/text_groups.json`` or to a separately pinned file (``--window-pin``);
run ids of the project form ``YYYYMMDD-HHMMSS-<8 hex>``, distinct across the primary and repeat runs.
TF-v1 H100 records count only if (a)-(c) and the identity checks hold; otherwise they are not counted.
FP4's paired half-width bound (<= delta/2, delta = log 1.05) and reliability bound (>= 0.90) are not
computed here; this report alone never establishes FP4 qualification.

Usage:
    python scripts/scale4b_report.py --zoo data/zoo/qwen3.5-4b-scale \
        --h100 data/evals/qwen3.5-4b-scale/lambda-H<n> [--h0 data/evals/qwen3.5-4b-scale/tf_v1] \
        [--job-log data/runs/scale4b/scale4b-H<n>.log] [--date YYYY-MM-DD] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 20261110
DRAWS = 4000
N_WINDOWS = 64
REPEATS = ("q4b-r1", "q4b-r2")  # FP4 repeat labels
SAME_DEVICE_TOL = 1e-4  # FP4 repeat discrepancy (nats)
CONFIG = REPO / "configs/eval_tf_v1_wikitext.yaml"
TEXT_GROUPS = REPO / "docs/prereg/a5_manifests/text_groups.json"
RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}")
DEVICE_ID = re.compile(r"H\d+")
CROSS_DEVICE_RATIO = 0.1
HEADER = "diagnostic, not a gate; exposure recorded"


# ---- loading ---------------------------------------------------------------------------------


def read_run(path: Path) -> dict:
    """One evaluated entry directory: metadata plus per-window NLL, tokens and token-id hashes."""
    path = Path(path)
    meta = json.loads((path / "eval.json").read_text())
    rows = {}
    for line in (path / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["benchmark"] == "wikitext_ppl":
            rows[int(r["item_id"])] = r
    if sorted(rows) != list(range(N_WINDOWS)):
        raise ValueError(f"{path}: expected WikiText windows 0..{N_WINDOWS - 1}")
    ordered = [rows[w] for w in range(N_WINDOWS)]
    order = [
        int(r["item_id"])
        for r in map(json.loads, (path / "records.jsonl").read_text().splitlines())
        if r["benchmark"] == "wikitext_ppl"
    ]
    nll = np.array([r["extra"]["mean_nll"] for r in ordered], dtype=float)
    if not np.all(np.isfinite(nll)):
        raise ValueError(f"{path}: nonfinite window NLL")
    return {
        "path": str(path),
        "meta": meta,
        "nll": nll,
        "tokens": np.array([r["n_tokens"] for r in ordered], dtype=float),
        "token_hashes": [r["extra"].get("token_ids_hash") for r in ordered],
        "window_order": order,
    }


def _base_dir(root: Path, entries: dict) -> Path:
    candidates = [root / "base", *sorted(p.parent for p in root.glob("base/devices/*/*/eval.json"))]
    candidates = [c for c in candidates if (c / "eval.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"{root}: no base evaluation")
    wanted = Counter(e["meta"].get("base_run_id") for e in entries.values()).most_common(1)
    for c in candidates:
        if wanted and json.loads((c / "eval.json").read_text()).get("run_id") == wanted[0][0]:
            return c
    return candidates[0]


def load_device(root, ids) -> dict:
    """Base, entries, FP4 repeats (``repeats/q4b-r1``, ``repeats/q4b-r2``) and device identity."""
    root = Path(root)
    entries = {i: read_run(root / i) for i in ids if (root / i / "eval.json").exists()}
    base = _base_dir(root, entries)
    repeats = {label: {} for label in REPEATS}
    for name, d in [("base", root / "base"), *[(i, root / i) for i in entries]]:
        for label in REPEATS:
            rep = d / "repeats" / label
            if (rep / "eval.json").exists():
                repeats[label][name] = read_run(rep)
    ident = root / "device_identity.json"
    return {
        "root": str(root),
        "base": read_run(base),
        "entries": entries,
        "repeats": repeats,
        "identity": json.loads(ident.read_text()) if ident.exists() else None,
    }


def load_manifest(zoo) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in (Path(zoo) / "manifest.jsonl").read_text().splitlines() if x]
    header = next(x for x in lines if x.get("type") == "header")
    return header, [x for x in lines if x.get("type") != "header"]


# ---- arithmetic ------------------------------------------------------------------------------


def log_ppl(run) -> float:
    return float(np.sum(run["tokens"] * run["nll"]) / np.sum(run["tokens"]))


def paired_effect(entry, base, seed=SEED, draws=DRAWS) -> dict:
    """Token-weighted paired delta (entry - base) with a 32-block bootstrap 95% interval."""
    if not np.array_equal(entry["tokens"], base["tokens"]):
        raise ValueError("entry and base windows differ in token counts")
    d = entry["nll"] - base["nll"]
    t = entry["tokens"]
    blocks = np.arange(N_WINDOWS).reshape(-1, 2)
    num = (t * d)[blocks].sum(axis=1)
    den = t[blocks].sum(axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(blocks), size=(draws, len(blocks)))
    boot = num[pick].sum(axis=1) / den[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    delta = float(num.sum() / den.sum())
    return {
        "delta": delta,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boot.std(ddof=1)),
        "improving": delta < 0,
        "resolved": "improving" if hi < 0 else "worsening" if lo > 0 else "unresolved",
    }


def clopper_pearson(k: int, n: int, alpha=0.05) -> tuple[float, float]:
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def improvement_fraction(effects: dict) -> dict:
    n = len(effects)
    k = sum(e["improving"] for e in effects.values())
    lo, hi = clopper_pearson(k, n) if n else (math.nan, math.nan)
    return {
        "n": n,
        "improving": k,
        "fraction": k / n if n else math.nan,
        "ci": (lo, hi),
        "resolved_improving": sum(e["resolved"] == "improving" for e in effects.values()),
        "resolved_worsening": sum(e["resolved"] == "worsening" for e in effects.values()),
    }


def repeat_check(device) -> dict:
    """Per repeat label and run: bitwise equality and max |primary - repeat| of the window NLL."""
    out = {}
    for label, runs in device["repeats"].items():
        for name, rep in runs.items():
            orig = device["base"] if name == "base" else device["entries"][name]
            diff = np.abs(rep["nll"] - orig["nll"])
            out[f"{label}:{name}"] = {
                "bitwise": bool(np.array_equal(rep["nll"], orig["nll"])),
                "max_abs": float(diff.max()),
            }
    return out


IDENTITY = ("config_hash", "model_revision", "source_hash")


def expected_config_hash(config=CONFIG) -> str:
    """FP4: the effective config hash, recomputed by ``wsp.eval.batch.effective_config_hash``."""
    import yaml

    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from wsp.eval.batch import effective_config_hash

    return effective_config_hash(yaml.safe_load(Path(config).read_text()))


def window_hashes(path=TEXT_GROUPS) -> list:
    """Per-window token-id hashes of windows 0-63: ``text_groups.json`` or a separate pin file."""
    data = json.loads(Path(path).read_text())
    windows = sorted(data["windows"], key=lambda w: int(w["item_id"]))
    if [int(w["item_id"]) for w in windows] != list(range(N_WINDOWS)):
        raise ValueError(f"{path}: expected windows 0..{N_WINDOWS - 1}")
    return [w["token_ids_hash"] for w in windows]


def fp4_identity(dev, data, ids, header, config_hash, windows) -> list[str]:
    """FP4 identity checks on every run of one device (primary and repeats); returns reasons."""
    reasons = []
    cfg_batch = None
    try:
        import yaml

        cfg_batch = yaml.safe_load(CONFIG.read_text())["generation"].get("batch_size", 8)
    except (OSError, KeyError):
        reasons.append(f"{dev}: cannot read the config batch size")
    ident = data.get("identity")
    if ident is None:
        reasons.append(f"{dev}: no device_identity.json (GPU model, driver, CUDA, torch)")
    runs = [("base", data["base"])] + [(i, data["entries"][i]) for i in ids if i in data["entries"]]
    for label, rs in data["repeats"].items():
        runs += [(f"{label}:{n}", r) for n, r in rs.items()]
    run_ids = {}
    for name, run in runs:
        m = run["meta"]
        where = f"{dev} {name}"
        device_id = m.get("eval_device_id")
        if not isinstance(device_id, str) or not DEVICE_ID.fullmatch(device_id):
            reasons.append(f"{where}: eval_device_id is not H<number>")
        if ident is not None:
            if device_id != ident.get("eval_device_id"):
                reasons.append(f"{where}: eval_device_id differs from device_identity.json")
            for key, ident_key in (
                ("gpu_class", "gpu_model"),
                ("torch_version", "torch"),
                ("cuda_version", "cuda"),
            ):
                if m.get(key) != ident.get(ident_key):
                    reasons.append(f"{where}: {key} differs from device_identity.json")
        if m.get("dtype") != "torch.bfloat16":
            reasons.append(f"{where}: dtype {m.get('dtype')} is not bf16")
        if cfg_batch is not None and m.get("batch_size") != cfg_batch:
            reasons.append(f"{where}: batch size {m.get('batch_size')} is not {cfg_batch}")
        if config_hash is not None and m.get("config_hash") != config_hash:
            reasons.append(f"{where}: config hash is not the recomputed effective config hash")
        if m.get("model", header.get("base_model")) != header.get("base_model"):
            reasons.append(f"{where}: model repo differs from the manifest")
        if m.get("model_revision") != header.get("base_revision"):
            reasons.append(f"{where}: model revision differs from the manifest")
        if run["window_order"] != list(range(N_WINDOWS)):
            reasons.append(f"{where}: WikiText windows not in registered order")
        if windows is not None and run["token_hashes"] != windows:
            reasons.append(
                f"{where}: window token-id hashes differ from the pinned window identity"
            )
        rid = m.get("run_id")
        if not isinstance(rid, str) or not RUN_ID.fullmatch(rid):
            reasons.append(f"{where}: run id {rid!r} is not YYYYMMDD-HHMMSS-<8 hex>")
        label = name.split(":")[0] if ":" in name else None
        if label is not None and m.get("repeat_id") != label:
            reasons.append(f"{where}: repeat_id {m.get('repeat_id')!r} is not {label}")
        run_ids.setdefault(label, set()).add(rid)
    labels = list(run_ids)
    for k, a in enumerate(labels):
        for b in labels[k + 1 :]:
            if run_ids[a] & run_ids[b]:
                reasons.append(
                    f"{dev}: run ids shared between {a or 'primary'} and {b or 'primary'}"
                )
    return reasons


def a3_pairing(h100, h0, paired, manifest_rows) -> dict:
    """The A3 / section-6 paired device comparison on the base and the paired entries."""
    if h0 is None:
        return {"status": "pending", "reasons": ["H0 half not run (scripts/scale4b_h0_pair.sh)"]}
    reasons = []
    digests = {r["id"]: r["delta_sha256"] for r in manifest_rows}
    runs = [("base", h100["base"], h0["base"])] + [
        (i, h100["entries"].get(i), h0["entries"].get(i)) for i in paired
    ]
    for name, a, b in runs:
        if a is None or b is None:
            reasons.append(f"{name}: missing on {'H100' if a is None else 'H0'}")
            continue
        for key in IDENTITY:
            if a["meta"].get(key) != b["meta"].get(key):
                reasons.append(f"{name}: {key} differs across devices")
        if a["token_hashes"] != b["token_hashes"] or None in a["token_hashes"]:
            reasons.append(f"{name}: WikiText token-id hashes differ or are missing")
        if name != "base":
            for dev, run in (("H100", a), ("H0", b)):
                got = run["meta"].get("entry_hashes", {}).get("delta.safetensors")
                if got != digests.get(name):
                    reasons.append(f"{name}: {dev} delta sha256 differs from the manifest")
    if h100["base"]["meta"].get("eval_device_id") == h0["base"]["meta"].get("eval_device_id"):
        reasons.append("both halves report the same eval_device_id")
    same = {}
    for dev, data in (("H100", h100), ("H0", h0)):
        checks = repeat_check(data)
        same[dev] = checks
        for label in REPEATS:
            for name in ["base", *paired]:
                c = checks.get(f"{label}:{name}")
                if c is None:
                    reasons.append(f"{dev} {name}: no repeats/{label}")
                    continue
                if not c["max_abs"] <= SAME_DEVICE_TOL:
                    reasons.append(
                        f"{dev} {name}: {label} discrepancy {c['max_abs']:.3g} > {SAME_DEVICE_TOL:g}"
                    )
                if not c["bitwise"]:
                    reasons.append(f"{dev} {name}: {label} not bitwise (max {c['max_abs']:.3g})")
    cross = {}
    ok_ids = [i for i in paired if i in h100["entries"] and i in h0["entries"]]
    if len(ok_ids) >= 2:
        d1 = np.array([paired_effect(h100["entries"][i], h100["base"])["delta"] for i in ok_ids])
        d0 = np.array([paired_effect(h0["entries"][i], h0["base"])["delta"] for i in ok_ids])
        diff = d1 - d0
        sd_cross = float(np.std(diff - diff.mean(), ddof=1))
        sd_between = float(np.std((d1 + d0) / 2, ddof=1))
        ratio = sd_cross / sd_between if sd_between > 0 else math.inf
        cross = {
            "ids": ok_ids,
            "delta_h100": d1.tolist(),
            "delta_h0": d0.tolist(),
            "mean_offset": float(diff.mean()),
            "sd_cross_centred": sd_cross,
            "sd_between": sd_between,
            "ratio": ratio,
            "base_window_max_abs": float(np.abs(h100["base"]["nll"] - h0["base"]["nll"]).max()),
            "base_log_ppl_diff": log_ppl(h100["base"]) - log_ppl(h0["base"]),
        }
        if not ratio <= CROSS_DEVICE_RATIO:
            reasons.append(f"cross-device SD ratio {ratio:.3g} > {CROSS_DEVICE_RATIO}")
    else:
        reasons.append("fewer than two paired entries on both devices")
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "same_device": same,
        "cross_device": cross,
    }


def h100_timing(h100) -> dict:
    rows = {"base": h100["base"]["meta"], **{i: e["meta"] for i, e in h100["entries"].items()}}
    per = {
        k: {
            "wall_seconds": m.get("wall_seconds"),
            "wikitext_seconds": (m.get("stage_seconds") or {}).get("wikitext_ppl"),
        }
        for k, m in rows.items()
    }
    walls = [v["wall_seconds"] for v in per.values() if v["wall_seconds"] is not None]
    meta = h100["base"]["meta"]
    return {
        "per_entry": per,
        "scoring_seconds": float(sum(walls)),
        "median_wall_seconds": float(np.median(walls)) if walls else math.nan,
        "model_load_seconds": meta.get("model_load_seconds"),
        "gpu_class": meta.get("gpu_class"),
        "eval_device_id": meta.get("eval_device_id"),
    }


def job_window(log_path) -> dict:
    """``SCALE4B start``/``end`` stamps and the ledger line written by the Lambda job, if given."""
    if not log_path or not Path(log_path).exists():
        return {}
    tags = {"SCALE4B start ": "start", "SCALE4B end ": "end", "LEDGER ": "ledger"}
    out = {}
    for line in Path(log_path).read_text().splitlines():
        for tag, key in tags.items():
            if line.startswith(tag):
                out[key] = line[len(tag) :].strip()
    return out


# ---- report ----------------------------------------------------------------------------------


def analyse(zoo, h100_dir, h0_dir=None, job_log=None, window_pin=None, config_hash=None) -> dict:
    header, rows = load_manifest(zoo)
    ids = [r["id"] for r in rows]
    h100 = load_device(h100_dir, ids)
    h0 = load_device(h0_dir, header["paired_ids"]) if h0_dir else None
    effects = {
        i: paired_effect(h100["entries"][i], h100["base"]) for i in ids if i in h100["entries"]
    }
    pairing = a3_pairing(h100, h0, header["paired_ids"], rows)
    config_hash = config_hash or expected_config_hash()
    windows = window_hashes(window_pin or TEXT_GROUPS)
    identity = fp4_identity("H100", h100, ids, header, config_hash, windows)
    if h0 is not None:
        identity += fp4_identity("H0", h0, header["paired_ids"], header, config_hash, windows)
    return {
        "header": header,
        "rows": rows,
        "effects": effects,
        "fraction": improvement_fraction(effects),
        "base_log_ppl": log_ppl(h100["base"]),
        "pairing": pairing,
        "identity": {
            "status": "pass" if not identity else "fail",
            "reasons": identity,
            "config_hash": config_hash,
            "window_identity": str(window_pin or TEXT_GROUPS.relative_to(REPO)),
            "devices": {d: x["identity"] for d, x in (("H100", h100), ("H0", h0)) if x is not None},
        },
        "counted": pairing["status"] == "pass" and not identity,
        "timing": h100_timing(h100),
        "job": job_window(job_log),
        "missing": [i for i in ids if i not in effects],
        "runs": sorted(
            {h100["base"]["meta"].get("run_id")}
            | {e["meta"].get("run_id") for e in h100["entries"].values()}
        ),
        "h0_runs": sorted(
            {h0["base"]["meta"].get("run_id")}
            | {e["meta"].get("run_id") for e in h0["entries"].values()}
        )
        if h0
        else [],
    }


def _f(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}g}"


def render(a: dict, date: str) -> str:
    h, fr, p, t = a["header"], a["fraction"], a["pairing"], a["timing"]
    ident = a["identity"]
    status = (
        "counted: the A3 pairing and the FP4 identity checks passed"
        if a["counted"]
        else f"**not counted**: A3 pairing {p['status']} ({'; '.join(p['reasons'])}); FP4 identity"
        f" {ident['status']}" + (f" ({'; '.join(ident['reasons'])})" if ident["reasons"] else "")
    )
    out = [
        f"# 4B scale diagnostic (F5.3), {date}",
        "",
        f"**{HEADER}.**",
        "",
        "F5.3 of the 10 Nov council synthesis: 20 outcome-blind Gaussian edits of Qwen3.5-4B at a fixed relative",
        "radius plus the shared base, scored on the 64-window WikiText log-PPL endpoint (TF-v1 restricted to its",
        "WikiText stage, `configs/eval_tf_v1_wikitext.yaml`). It is a diagnostic of beneficial-edit density and",
        "effect resolution at 4B; it is never a gate, is not part of G7, and does not touch Part A. If it is",
        "inconclusive, scale expansion stops.",
        "",
        f"TF-v1 status of the H100 records: {status}.",
        (
            "FP4's paired half-width bound (<= log(1.05)/2) and reliability bound (>= 0.90) are not"
            " computed by this report; it never establishes FP4 qualification on its own."
        ),
        "",
        "## Design (fixed before scoring)",
        "",
        (
            f"- Base `{h['base_model']}` at `{h['base_revision']}`; kernel `{h['kernel']}`; radius rho = {h['radius']:g}"
            " of each touched tensor's Frobenius norm; stored delta is the bf16-effective delta."
        ),
        (
            f"- Layers {h['layers']} of {h['target_depth']} (0.8B random_dir {h['source_layers']} of"
            f" {h['source_depth']}, scaled by round(l * {h['target_depth']}/{h['source_depth']})); tensor specs"
            f" {h['tensors']} ({h['support_mode']} mode), {len(h['support'])} tensors."
        ),
        (
            f"- {h['n']} draws, seed {h.get('seed_rule', 'n/a')}, generator {h.get('generator', 'n/a')};"
            f" design hash `{h['config_hash'][:12]}`; FP4 panel (draw indices"
            f" {', '.join(str(d) for d in h.get('paired_draw_indices', []))})"
            f" {', '.join(f'`{i}`' for i in h['paired_ids'])} ({h['pairing_rule']})."
        ),
        "",
        "## Result",
        "",
        (
            f"Base log-PPL (H100): {a['base_log_ppl']:.6f} nats. Edits improving over the base (delta < 0):"
            f" **{fr['improving']} of {fr['n']}** (fraction {_f(fr['fraction'], 3)}, Clopper-Pearson 95%"
            f" [{_f(fr['ci'][0], 3)}, {_f(fr['ci'][1], 3)}]); resolved improving {fr['resolved_improving']},"
            f" resolved worsening {fr['resolved_worsening']}, the rest unresolved."
        ),
        "",
        "| id | draw | seed | panel | delta log-PPL (nats) | 95% block-bootstrap interval | resolved | realised r (pooled) | realised r min–max | changed elements |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in a["rows"]:
        e = a["effects"].get(r["id"])
        cells = (
            [f"{e['delta']:+.3e}", f"[{e['lo']:+.3e}, {e['hi']:+.3e}]", e["resolved"]]
            if e
            else ["missing", "", ""]
        )
        out.append(
            f"| `{r['id']}` | {r.get('draw_index', '')} | {r['seed']} | {'yes' if r['id'] in h['paired_ids'] else ''} | "
            + " | ".join(cells)
            + f" | {r['realised_radius']:.4e} | {r['realised_radius_min']:.4e}–{r['realised_radius_max']:.4e}"
            f" | {r['changed_fraction']:.3f} |"
        )
    radii = [r["realised_radius"] for r in a["rows"]]
    out += [
        "",
        (
            f"Realised pooled radius over the {len(radii)} entries: {min(radii):.4e} to {max(radii):.4e}"
            f" (intended {h['radius']:g}); bf16 rounding leaves part of each intended delta below half a"
            " unit in the last place, so the effective edit is the realised one, not the intended Gaussian."
            f" Largest fraction of elements where the applied weight differs from bf16(W + intended):"
            f" {_f(max((r.get('applied_mismatch_fraction', math.nan) for r in a['rows']), default=math.nan), 3)}."
        ),
        "",
        "## A3 device pairing",
        "",
        f"Status: **{p['status']}**."
        + ("" if not p["reasons"] else " " + "; ".join(p["reasons"]) + "."),
    ]
    c = p.get("cross_device") or {}
    if c:
        out += [
            "",
            (
                f"Cross-device (H100 minus H0) on {len(c['ids'])} paired entries: centred SD"
                f" {c['sd_cross_centred']:.3e}, between-entry SD {c['sd_between']:.3e}, ratio {_f(c['ratio'], 3)}"
                f" (limit {CROSS_DEVICE_RATIO}); mean offset {c['mean_offset']:+.3e}; base log-PPL difference"
                f" {c['base_log_ppl_diff']:+.3e}, base per-window max |diff| {c['base_window_max_abs']:.3e}."
            ),
        ]
    for dev, checks in (p.get("same_device") or {}).items():
        bits = sum(v["bitwise"] for v in checks.values())
        worst = max((v["max_abs"] for v in checks.values()), default=math.nan)
        out.append(
            f"- {dev} same-device repeats ({', '.join(REPEATS)}): {bits} of {len(checks)} bitwise;"
            f" max |primary - repeat| {_f(worst, 3)} (FP4 limit {SAME_DEVICE_TOL:g} nats)."
        )
    out += [
        "",
        "## FP4 identity",
        "",
        f"Status: **{ident['status']}**."
        + ("" if not ident["reasons"] else " " + "; ".join(ident["reasons"]) + ".")
        + f" Effective config hash `{ident['config_hash']}`; window identity `{ident['window_identity']}`.",
    ]
    for dev, d in ident["devices"].items():
        out.append(f"- {dev}: " + (json.dumps(d, sort_keys=True) if d else "no device identity"))
    out += [
        "",
        "## H100 timing",
        "",
        (
            f"Device `{t['eval_device_id']}` ({t['gpu_class']}); model load {_f(t['model_load_seconds'], 3)} s;"
            f" scoring wall {t['scoring_seconds'] / 3600:.3f} h over {len(t['per_entry'])} evaluations; median"
            f" {_f(t['median_wall_seconds'], 3)} s per evaluation."
        ),
    ]
    if a["job"]:
        out.append(
            f"Job window: start {a['job'].get('start', 'n/a')}, end {a['job'].get('end', 'n/a')}."
            + (f" Ledger line: `{a['job']['ledger']}`." if "ledger" in a["job"] else "")
        )
    out += ["", "| evaluation | wall s | WikiText stage s |", "|---|---|---|"]
    for k, v in t["per_entry"].items():
        out.append(f"| `{k}` | {_f(v['wall_seconds'], 4)} | {_f(v['wikitext_seconds'], 4)} |")
    out += [
        "",
        "## Exposure",
        "",
        "Exposed by this diagnostic: the 64 development WikiText windows (wikitext-2-raw-v1 test at the"
        " pinned revision, 1,024-token windows 0–63, the H1w development windows) scored on the Qwen3.5-4B"
        f" base and {len(a['effects'])} Gaussian edits. No sealed item, no A5 Part A edit and no 0.8B outcome"
        " is touched. H100 run ids: "
        + ", ".join(f"`{r}`" for r in a["runs"] if r)
        + "."
        + (
            " H0 run ids: " + ", ".join(f"`{r}`" for r in a["h0_runs"] if r) + "."
            if a["h0_runs"]
            else ""
        ),
    ]
    if a["missing"]:
        out += ["", "Missing evaluations: " + ", ".join(f"`{i}`" for i in a["missing"]) + "."]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--zoo", type=Path, required=True)
    p.add_argument("--h100", type=Path, required=True)
    p.add_argument("--h0", type=Path)
    p.add_argument("--job-log", type=Path)
    p.add_argument(
        "--window-pin",
        type=Path,
        help="4B window identity pinned before scoring, if it differs from text_groups.json",
    )
    p.add_argument("--date", default=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"))
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    a = analyse(args.zoo, args.h100, args.h0, args.job_log, args.window_pin)
    out = args.out or REPO / f"docs/reviews/scale/scale4b-diagnostic-{args.date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a, args.date))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
