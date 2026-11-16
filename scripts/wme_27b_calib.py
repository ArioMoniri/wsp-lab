#!/usr/bin/env python3
"""Qwen3.8-27B WM-E engineering calibration on one 80 GB GPU (SYNTHESIS-r9 C9; round-9 L1-L4). ENGINEERING ONLY.

Measures feasibility only: load time; peak GPU (allocator and device) and host memory and swap; lean-panel seconds
per state; bitwise base / edit / restore repeats; grouped support-only gradient + K-FAC timing and memory on 4 layers
(24-27); greedy generation throughput (for the D8 bank price). No trajectory, slope, headroom or enhancement
quantity is computed. Objective values are written **only as content hashes** (``wsp.wm.calib27.hash_only``);
the timing edits' panel outputs are hashed and compared, never differenced.

    python scripts/wme_27b_calib.py --out <dir>        # all phases, one process, one model load

Runs on the Lambda instance (no API key there). Device memory is capped at 72 GB (C9): the allocator ceiling is
70 GB and a monitor stops the job (exit 75) if nvidia-smi reports more than 72 GB.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wsp.wm import actions as A
from wsp.wm import calib27 as K
from wsp.wm import collect as C
from wsp.wm import hf
from wsp.wm import panel as P
from wsp.wm import scorers as Sc

PANEL = ROOT / "data/wme/panels/wme-v0-p1"
ALLOCATOR_BYTES = 70_000_000_000
FIELD_ITEMS = 24
FIELD_MAX_TOKENS = 1024
EDIT_LAYER_NOISE = 28  # mid-depth MLP writer (9B pilot used layer 14 of 32)
EDIT_LAYER_SCALE = 27  # the gated-attention o_proj in the support group
DEVICE = "cuda"
LIMIT = None  # CPU smoke only: items per objective


def _cuda():
    return DEVICE == "cuda"


def _limit(items):
    return items if LIMIT is None else {k: v[:LIMIT] for k, v in items.items()}


# ------------------------------------------------------------------ resources


def _gpu_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        return int(out.split()[0]) * 1024 * 1024
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return -1


def _meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) * 1024
    return info


def _rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


class Monitor:
    """Samples RSS, device memory and swap every 0.5 s; stops the job (exit 75) above the device cap."""

    def __init__(self):
        mi = _meminfo()
        self.swap0 = mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)
        self.rss_cap = int(0.9 * mi["MemTotal"])
        self.peak = {"rss": 0, "dev": 0, "swap": 0}
        self.global_peak = {"rss": 0, "dev": 0, "swap": 0, "alloc": 0, "reserved": 0}
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            rss, dev = _rss(), _gpu_used()
            mi = _meminfo()
            swap = max(0, mi.get("SwapTotal", 0) - mi.get("SwapFree", 0) - self.swap0)
            for k, v in (("rss", rss), ("dev", dev), ("swap", swap)):
                self.peak[k] = max(self.peak[k], v)
                self.global_peak[k] = max(self.global_peak[k], v)
            if _cuda() and dev > K.GPU_CAP_BYTES or rss > self.rss_cap:
                print(f"RESOURCE STOP rss={rss} dev={dev}", flush=True)
                os._exit(75)
            time.sleep(0.5)

    def reset(self):
        self.snapshot()
        self.peak = {"rss": 0, "dev": 0, "swap": 0}
        if _cuda():
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self):
        alloc = torch.cuda.max_memory_allocated() if _cuda() else 0
        res = torch.cuda.max_memory_reserved() if _cuda() else 0
        self.global_peak["alloc"] = max(self.global_peak["alloc"], alloc)
        self.global_peak["reserved"] = max(self.global_peak["reserved"], res)
        return {
            "peak_device_bytes": self.peak["dev"],
            "peak_rss_bytes": self.peak["rss"],
            "peak_swap_bytes": self.peak["swap"],
            "peak_allocated_bytes": alloc,
            "peak_reserved_bytes": res,
        }


def log(msg, **kw):
    print(f"[{time.strftime('%H:%M:%S')}] {msg} {json.dumps(kw, default=str)}", flush=True)


def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def sync_time(fn):
    if _cuda():
        torch.cuda.synchronize()
    t = time.time()
    out = fn()
    if _cuda():
        torch.cuda.synchronize()
    return out, time.time() - t


# ------------------------------------------------------------------ phases


def environment():
    import transformers

    pk = {}
    for name in (
        "fla",
        "causal_conv1d",
        "triton",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "numpy",
    ):
        try:
            pk[name] = __import__(name).__version__
        except Exception as e:  # noqa: BLE001 -- recorded, not fatal
            pk[name] = f"unavailable: {type(e).__name__}"
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "packages": pk,
        "gpu": smi,
        "cpu_count": os.cpu_count(),
        "mem_total_bytes": _meminfo()["MemTotal"],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def toy_target():
    """CPU smoke only: a tiny random model with the 27B layer layout and the local Qwen3.5 tokenizer."""
    import transformers

    hf.use_reference_kernels()
    cfg = json.loads((ROOT / "tests/fixtures/qwen3_8_27b/config.json").read_text())
    text = dict(cfg["text_config"])
    text.update(
        num_hidden_layers=32,
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=text["layer_types"][:32],
    )
    text.pop("dtype", None)
    conf = transformers.AutoConfig.for_model(**text)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(conf).to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(
        str(hf.snapshot_dir("Qwen/Qwen3.5-9B", hf.TARGETS["Qwen/Qwen3.5-9B"])),
        local_files_only=True,
    )
    return model, tok, {"text_config": text}, "toy-" + "0" * 60, None


def load(mon, toy=False):
    if _cuda():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, ALLOCATOR_BYTES / total))
    mon.reset()
    t = time.time()
    model, tok, cfg, base_hash, _path = toy_target() if toy else hf.load_target(K.REPO, DEVICE)
    if _cuda():
        torch.cuda.synchronize()
    load_s = time.time() - t
    params = dict(model.named_parameters())
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in params)
        else "model.layers."
    )
    rm = A.RoleMap.from_config(cfg, prefix=prefix)
    # role-map audit: every derived path exists with the shape implied by the config
    text = cfg.get("text_config", cfg)
    mixer_in = {
        "full_attention": text["num_attention_heads"] * text["head_dim"],
        "linear_attention": text["linear_num_value_heads"] * text["linear_value_head_dim"],
    }
    bad = []
    for (layer, role), name in rm.writers().items():
        want = (
            (rm.hidden, rm.ffn) if role == "down" else (rm.hidden, mixer_in[rm.layer_types[layer]])
        )
        if name not in params or tuple(params[name].shape) != want:
            bad.append((name, None if name not in params else tuple(params[name].shape), want))
    frozen_loaded = [n for n in params if A.is_frozen(n)]
    out = {
        "repo": K.REPO,
        "revision": "toy" if toy else hf.TARGETS[K.REPO],
        "base_hash": base_hash,
        "model_class": type(model).__name__,
        "load_seconds": load_s,
        "allocated_after_load_bytes": torch.cuda.memory_allocated() if _cuda() else 0,
        "n_params": sum(p.numel() for p in params.values()),
        "param_bytes": sum(p.numel() * p.element_size() for p in params.values()),
        "dtypes": sorted({str(p.dtype) for p in params.values()}),
        "role_prefix": prefix,
        "role_map": {
            "n_layers": rm.n_layers,
            "hidden": rm.hidden,
            "ffn": rm.ffn,
            "vocab": rm.vocab,
            "tied": rm.tied,
            "n_full_attention": rm.layer_types.count("full_attention"),
            "n_linear_attention": rm.layer_types.count("linear_attention"),
            "writers_checked": len(rm.writers()),
            "mismatches": bad,
        },
        "vision_or_mtp_tensors_loaded": len(frozen_loaded),
        "memory": mon.snapshot(),
    }
    if bad or frozen_loaded:
        raise RuntimeError(f"role-map audit failed: {bad[:4]} frozen={frozen_loaded[:4]}")
    log(
        "loaded",
        seconds=round(load_s, 1),
        alloc_gb=round(out["allocated_after_load_bytes"] / 1e9, 2),
    )
    return model, tok, cfg, base_hash, rm, out


def kernel_probe(model, tok):
    """Which implementations a short forward actually executes (fla / causal_conv1d / torch fallbacks)."""
    seen = set()

    def prof(frame, event, arg):
        if event == "call":
            mod = frame.f_globals.get("__name__", "")
            name = frame.f_code.co_name
            if mod.startswith(("fla", "causal_conv1d")) or name.startswith("torch_"):
                seen.add(
                    f"{mod.split('.')[0]}:{name}"
                    if not name.startswith("torch_")
                    else f"{mod}:{name}"
                )

    ids = tok("kernel probe " * 8, return_tensors="pt")["input_ids"].to(DEVICE)
    sys.setprofile(prof)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        sys.setprofile(None)
    return {
        "fla_used": any(x.startswith("fla:") for x in seen),
        "causal_conv1d_used": any(x.startswith("causal_conv1d:") for x in seen),
        "torch_fallbacks": sorted(x for x in seen if ":torch_" in x),
    }


def p1(model, tok, base_hash, rm, mon):
    """Panel timing and bitwise repeats. Values are hashed; nothing is differenced."""
    mon.reset()
    pcfg = P.PanelConfig()
    adapter = P.LMAdapter(model)
    val = _limit(P.read_open(PANEL, "validation"))
    out = {
        "n_items": {k: len(v) for k, v in val.items()},
        "panel_config": {
            "window_tokens": pcfg.window_tokens,
            "max_batch_tokens": pcfg.max_batch_tokens,
        },
    }
    # chat template check (non-thinking)
    probe = P.chat_prompt(tok, "ping")
    out["chat_template_non_thinking"] = {
        "has_empty_think_block": "<think>\n\n</think>" in probe,
        "prompt_sha256": K.record_hash(probe),
    }
    per_obj = {}
    for obj, its in val.items():
        _, s = sync_time(lambda its=its, obj=obj: P.evaluate(adapter, tok, {obj: its}, pcfg))
        per_obj[obj] = s
        log("objective first pass", obj=obj, seconds=round(s, 1))
    out["per_objective_seconds_first_pass"] = per_obj
    base1, s1 = sync_time(lambda: P.evaluate(adapter, tok, val, pcfg))
    log("base 1", seconds=round(s1, 1))
    base2, s2 = sync_time(lambda: P.evaluate(adapter, tok, val, pcfg))
    log("base 2", seconds=round(s2, 1))
    hb = K.hash_only(base1)
    out["base"] = {
        "seconds": [s1, s2],
        "hash": hb,
        "repeat_bitwise": P.digest(base1) == P.digest(base2),
    }
    sess = A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)
    states = []
    # timing edit A: seeded norm-matched noise on a mid-depth MLP writer (rho 0.05), evaluated twice
    t0 = time.time()
    sess.apply(A.noise(rm.writers()[(EDIT_LAYER_NOISE, "down")], seed=7, rho=0.05, dose=1.0))
    th_a = sess.theta_hash()
    apply_s = time.time() - t0
    ea1, sa1 = sync_time(lambda: P.evaluate(adapter, tok, val, pcfg))
    ea2, sa2 = sync_time(lambda: P.evaluate(adapter, tok, val, pcfg))
    t0 = time.time()
    sess.restore()
    restore_hash_ok = sess.theta_hash() == base_hash and th_a != base_hash
    restore_s = time.time() - t0
    post, sp = sync_time(lambda: P.evaluate(adapter, tok, val, pcfg))
    states.append(
        {
            "edit": "noise",
            "tensor": rm.writers()[(EDIT_LAYER_NOISE, "down")],
            "apply_seconds": apply_s,
            "panel_seconds": [sa1, sa2],
            "restore_seconds": restore_s,
            "theta_hash": th_a,
            "hash": K.hash_only(ea1),
            "repeat_bitwise": P.digest(ea1) == P.digest(ea2),
            "differs_from_base": P.digest(ea1) != hb["digest"],
        }
    )
    log("edit A", seconds=[round(sa1, 1), round(sa2, 1)], repeat=states[-1]["repeat_bitwise"])
    # timing edit B: scale the layer-27 gated-attention o_proj by 1.25, evaluated once
    t0 = time.time()
    sess.apply(A.scale(rm.writers()[(EDIT_LAYER_SCALE, "o")], factor=1.25, dose=1.0))
    th_b = sess.theta_hash()
    apply_b = time.time() - t0
    eb, sb = sync_time(lambda: P.evaluate(adapter, tok, val, pcfg))
    t0 = time.time()
    sess.restore()
    restore_b_ok = sess.theta_hash() == base_hash and th_b != base_hash
    restore_b = time.time() - t0
    states.append(
        {
            "edit": "scale",
            "tensor": rm.writers()[(EDIT_LAYER_SCALE, "o")],
            "apply_seconds": apply_b,
            "panel_seconds": [sb],
            "restore_seconds": restore_b,
            "theta_hash": th_b,
            "hash": K.hash_only(eb),
            "differs_from_base": P.digest(eb) != hb["digest"],
        }
    )
    log("edit B", seconds=round(sb, 1))
    out["timing_edits"] = states
    out["post_restore"] = {"seconds": sp, "bitwise_equal_base": P.digest(post) == hb["digest"]}
    out["restore_hash_ok"] = restore_hash_ok and restore_b_ok
    out["all_in_seconds_per_state"] = [
        s["apply_seconds"] + s["panel_seconds"][0] + s["restore_seconds"] for s in states
    ]
    # greedy generation throughput, non-thinking (bank price t_gen); text is discarded
    prompts = [P.chat_prompt(tok, r["prompt"]) for r in val["refusal_margin_harmless"][:8]]
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(DEVICE)
    with torch.no_grad():
        gen, gs = sync_time(lambda: model.generate(**enc, max_new_tokens=128, do_sample=False))
    out["generation"] = {
        "prompts": 8,
        "batch": 8,
        "max_new_tokens": 128,
        "seconds": gs,
        "new_tokens": int((gen[:, enc["input_ids"].shape[1] :] != tok.pad_token_id).sum()),
    }
    log("generation", seconds=round(gs, 1))
    out["memory"] = mon.snapshot()
    return out


def field_panels(tok, pcfg, n_items, max_tokens, longest_first=False):
    cons = P.read_open(PANEL, "construction")
    scored = set()
    for split in ("validation", "dev"):
        for its in P.read_open(PANEL, split).values():
            scored |= {r["hash"] for r in its}
    panels, lengths = {}, {}
    for obj, its in cons.items():
        cand = []
        for r in sorted(its, key=lambda r: r["id"]):
            n = sum(len(x.ids) for x in P._rows_for(obj, r, tok, pcfg))
            if n <= max_tokens:
                cand.append((n, r))
            if not longest_first and len(cand) == n_items:
                break
        if longest_first:
            cand = sorted(cand, key=lambda x: -x[0])
        sel = cand[:n_items]
        if len(sel) < n_items:
            raise ValueError(
                f"{obj}: only {len(sel)} construction items within {max_tokens} tokens"
            )
        lengths[obj] = [n for n, _ in sel]
        panels[f"construction:{obj}"] = Sc.field_panel(
            f"construction:{obj}", [r for _, r in sel], scored_groups=tuple(sorted(scored))
        )
    return panels, lengths


def p2(model, tok, base_hash, rm, mon, work: Path, spectral: bool):
    """Grouped support-only gradients + model-sampled K-FAC on layers 24-27 (4 down_proj, 3 GatedDeltaNet
    out_proj, 1 attention o_proj); 6 objectives x 24 construction items <= 1024 tokens."""
    pcfg = P.PanelConfig()
    mods = K.support_modules(rm)
    out = {
        "support_layers": list(K.SUPPORT_LAYERS),
        "group": mods,
        "items_per_objective": FIELD_ITEMS,
        "field_max_tokens": FIELD_MAX_TOKENS,
    }
    # H4 preflight: the single longest admissible item per objective, before the main loop
    mon.reset()
    pre_panels, pre_len = field_panels(tok, pcfg, 1, FIELD_MAX_TOKENS, longest_first=True)
    specs = Sc.objective_specs(list(P.OBJECTIVES), pre_panels, pcfg)
    _, s = sync_time(
        lambda: C.collect_grouped(
            model,
            tok,
            specs,
            pre_panels,
            [mods],
            DEVICE,
            work / "preflight",
            fisher=True,
            theta_hash=base_hash,
            log=log,
        )
    )
    out["preflight"] = {"seconds": s, "longest_tokens": pre_len, "memory": mon.snapshot()}
    log("preflight", seconds=round(s, 1), mem=out["preflight"]["memory"])
    subprocess.run(["rm", "-rf", str(work / "preflight")], check=True)
    # main collection
    mon.reset()
    panels, lens = field_panels(
        tok, pcfg, FIELD_ITEMS if LIMIT is None else LIMIT, FIELD_MAX_TOKENS
    )
    specs = Sc.objective_specs(list(P.OBJECTIVES), panels, pcfg)
    field, s = sync_time(
        lambda: C.collect_grouped(
            model,
            tok,
            specs,
            panels,
            [mods],
            DEVICE,
            work / "field",
            fisher=True,
            theta_hash=base_hash,
            log=log,
        )
    )
    out["collect_seconds"] = s
    out["item_tokens"] = {k: {"sum": sum(v), "max": max(v)} for k, v in lens.items()}
    out["timing"] = field.identity["timing"]
    out["coverage_unsupported"] = {
        n: r["reason"] for n, r in field.coverage[specs[0].name].items() if not r["factors"]
    }
    out["factor_bytes_per_module_fp32"] = {
        m: int(
            sum(
                v.numel() * 4 for k, v in field.factors(specs[0].name, m).items() if k in ("A", "G")
            )
        )
        for m in mods
    }
    out["disk_bytes_written"] = int(
        subprocess.check_output(["du", "-sb", str(work / "field")], text=True).split()[0]
    )
    out["memory"] = mon.snapshot()
    log("field", seconds=round(s, 1), mem=out["memory"])
    if spectral:
        f = field.factors(specs[0].name, mods[0])
        t = time.time()
        _, audit = C.rf.spectral_factor(f["A"])
        out["spectral_factor_seconds_ffn"] = time.time() - t
        out["spectral_factor_shape"] = list(f["A"].shape)
        out["spectral_factor_audit_keys"] = sorted(audit) if isinstance(audit, dict) else None
        log("spectral", seconds=round(out["spectral_factor_seconds_ffn"], 1))
    field.drop_cache()
    subprocess.run(["rm", "-rf", str(work / "field")], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/wme27-work")
    ap.add_argument("--no-spectral", action="store_true")
    ap.add_argument("--toy", action="store_true", help="CPU smoke test on a tiny random model")
    ap.add_argument("--limit", type=int, default=None, help="CPU smoke only: items per objective")
    a = ap.parse_args()
    global DEVICE, LIMIT
    if a.toy:
        DEVICE, LIMIT = "cpu", a.limit or 2
    out_dir, work = Path(a.out), Path(a.work)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rec = {
        "label": "ENGINEERING ONLY",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": os.environ.get("WME_CODE_SHA", "unknown"),
    }
    t_all = time.time()
    c0 = resource.getrusage(resource.RUSAGE_SELF)
    mon = Monitor()
    try:
        rec["environment"] = environment()
        write_json(out_dir / "calib.json", rec)
        model, tok, _cfg, base_hash, rm, rec["load"] = load(mon, toy=a.toy)
        write_json(out_dir / "calib.json", rec)
        rec["kernels"] = kernel_probe(model, tok)
        log("kernels", **rec["kernels"])
        rec["p1"] = p1(model, tok, base_hash, rm, mon)
        write_json(out_dir / "calib.json", rec)
        rec["p2"] = p2(model, tok, base_hash, rm, mon, work, spectral=not a.no_spectral)
        rec["status"] = "complete"
    except Exception:  # noqa: BLE001 -- recorded in the result file
        rec["status"] = "failed"
        rec["error"] = traceback.format_exc()[-4000:]
        print(rec["error"], flush=True)
    r = resource.getrusage(resource.RUSAGE_SELF)
    rec["wall_seconds"] = time.time() - t_all
    rec["cpu_seconds"] = (r.ru_utime - c0.ru_utime) + (r.ru_stime - c0.ru_stime)
    rec["ru_maxrss_bytes"] = r.ru_maxrss * 1024
    mon.snapshot()
    rec["global_peak"] = dict(mon.global_peak)
    write_json(out_dir / "calib.json", rec)
    log("done", status=rec["status"], wall=round(rec["wall_seconds"], 1))
    return 0 if rec["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
