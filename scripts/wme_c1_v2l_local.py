"""PREREG_WME §II.31 C1(a) local engineering test (RULING-wme-c1-phasel C4.1; $0): Qwen3.5-0.8B on the local GPU, the
64 exposed hf_gen canary items (wme-p3h holdout mmlupro_gen, the stage-W canary set), batch 128, 128 new tokens,
greedy, non-thinking. Three runs under the deterministic engine and three under the registered path (throughput
reference), interleaved, after one unrecorded warm-up. Engineering only: no wme-c1 / wme-c2-gen item, no 9B.

``python scripts/wme_c1_v2l_local.py --out <json> [--vram-gb 8] [--n-items 64] [--allow-fla]``

Under the 8 GB local limit the torch-path prefill of one 64-item batch does not fit, so the test runs two labelled
variants: the full 64-item set with the local venv's ``fla`` kernels (``--allow-fla``: the production engine refuses
non-torch kernels; this local-only variant bypasses that single check and records the real bindings), and the
pinned-lock torch path on the first ``--n-items`` items (still one batch under the batch-128 rule).

By default ``fla`` is blocked so the gated-delta-net layers bind the transformers torch path, as on the pinned
remote lock (which installs no fla / causal_conv1d / kernels package).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vram-gb", type=float, default=8.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--allow-fla", action="store_true")
    ap.add_argument("--n-items", type=int, default=64)
    a = ap.parse_args()
    if not a.allow_fla:
        for m in ("fla", "causal_conv1d"):
            sys.modules[m] = None  # import fails -> transformers binds the torch path

    from wsp.wm.c1 import genengine as GE

    GE.prepare_process("hf_deterministic")  # CUBLAS_WORKSPACE_CONFIG before the first CUDA call

    import torch
    from huggingface_hub import constants

    from wsp.wm import hf
    from wsp.wm.c1 import modelops as MO

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 2**30 / total), 0)
    snaps = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B" / "snapshots"
    hf.TARGETS.setdefault(MO.SMOKE_REPO, min(snaps.iterdir()).name)
    m = MO.ModelCtx(MO.SMOKE_REPO, "cuda", check_theta0=False)
    items = MO.gen_canary_items()
    assert len(items) == MO.CANARY_GEN_ITEMS == 64
    items = items[: a.n_items]
    bindings = GE.kernel_bindings()
    if a.allow_fla:  # local-only variant: the production binding check is bypassed, the real bindings recorded
        real_check = GE.check_model
        GE.check_model = lambda model, b=None: real_check(
            model, dict.fromkeys(GE.LINEAR_ATTN_FUNCS, GE.QWEN35_MODULE)
        )
    else:
        GE.check_model(m.model)  # the production check: torch path bound
    rec = {
        "label": "PREREG_WME §II.31 C1(a) local engineering test (exposed canary items; Qwen3.5-0.8B; not 9B, not confirmatory)",
        "model": {
            "repo": MO.SMOKE_REPO,
            "revision": hf.TARGETS[MO.SMOKE_REPO],
            "theta": m.base_hash,
        },
        "items": {
            "set": f"wme-p3h holdout mmlupro_gen (hf_gen canary set), first {len(items)} of 64",
            "n": len(items),
        },
        "settings": {
            "batch": MO.CANARY_HF["batch"],
            "max_new_tokens": MO.CANARY_HF["max_new_tokens"],
        },
        "versions": MO.versions(),
        "fla_blocked": not a.allow_fla,
        "kernel_bindings": bindings,
        "binding_check_bypassed": bool(a.allow_fla),
        "attn_implementation": GE._attn_impl(m.model),
        "engine_deterministic": GE.config("hf_deterministic"),
        "vram_cap_gb": a.vram_gb,
        "runs": [],
    }
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    warm, _ = m.canary_hf_timed(items, mode="hf_registered")
    rec["warmup"] = {"mode": "hf_registered", "sha256": warm, "seconds": time.time() - t0}
    for i in range(a.runs):
        for mode in ("hf_deterministic", "hf_registered"):
            h, s = m.canary_hf_timed(items, mode=mode)
            rec["runs"].append({"i": i, "mode": mode, "sha256": h, "seconds": s})
            print(json.dumps(rec["runs"][-1]), flush=True)
    rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    by = {}
    for r in rec["runs"]:
        by.setdefault(r["mode"], []).append(r)
    rec["summary"] = {
        mode: {
            "hashes": [r["sha256"] for r in rs],
            "bitwise_3_of_3": len(rs) == 3 and len({r["sha256"] for r in rs}) == 1,
            "seconds": [r["seconds"] for r in rs],
            "median_s": sorted(r["seconds"] for r in rs)[len(rs) // 2],
        }
        for mode, rs in by.items()
    }
    s = rec["summary"]
    rec["summary"]["throughput_ratio_det_over_registered"] = (
        s["hf_deterministic"]["median_s"] / s["hf_registered"]["median_s"]
    )
    rec["summary"]["det_eq_registered_hash"] = (
        s["hf_deterministic"]["hashes"][0] == s["hf_registered"]["hashes"][0]
    )
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rec["summary"], indent=1))


if __name__ == "__main__":
    main()
