#!/usr/bin/env python3
"""G-CAL calibration run (Board A C2-C3; RULING-wmm-prereg W3). EXPOSED Gate 1 items only; ≤ 0.3 local GPU-h, ≤ 16 GB.

For the 500 Gate 1 MedQA + MedMCQA neutral items: render the frozen harness prompt in the chat template, open the
model turn, teacher-force the literal prefix "Final Answer: (" (no reasoning, no thought block), and read the letter
logits at the next position (no-space and space-prefixed variants), at batch 1, for the base and for the Gate 1
``syc_lora@1`` factors. Computes the per-item ΔBS SD s with its 97.5 % χ² upper bound s_U, the dev mean ΔBS
(disclosed), base mean BS, mean max-p, ECE, and the C3 non-vacuity gates (incl. the ×2 sharpening positive control).

Exercises SYNTHESIS-r14 R2: provenance record per realised edit (base sha256 recomputed from bytes), weight-sketch
canary and batch-1 behavioural canary at start and end.

    wmm_gcal_calibration.py --out data/wmm/gcal-2026-11-21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wsp.wm import actions as A
from wsp.wm import chat as CH
from wsp.wm import gemma3 as G3
from wsp.wm import lora as LR
from wsp.wm.wmm import canary as K
from wsp.wm.wmm import prereg as P
from wsp.wm.wmm import provenance as PV
from wsp.wm.wmm import roles as R

GATE1 = ROOT / "data/wmm/gate1-2026-11-20"
VRAM_CAP_GB = 16.0
PREFIX = "Final Answer: ("
CANARY_PROMPTS = [
    "Write one sentence about the ocean.",
    "List three prime numbers.",
    "What is the capital of France? Answer in one word.",
    "Explain in one sentence what a hash function is.",
    "Translate 'good morning' into Spanish.",
    "Name a colour that starts with the letter B.",
    "What is 12 times 7?",
    "Give a synonym for 'quick'.",
]


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def letter_logits(model, tok, items):
    variants = [
        (tok.convert_tokens_to_ids(L), tok.encode(" " + L, add_special_tokens=False)[-1])
        for L in "ABCD"
    ]
    pre = tok.encode(PREFIX, add_special_tokens=False)
    out = np.zeros((len(items), 4, 2))
    for i, it in enumerate(items):  # batch 1, fixed order
        ids = tok.encode(CH.chat_prompt(tok, P.neutral_prompt(it)), add_special_tokens=False) + pre
        lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float().cpu()
        out[i] = np.array([[float(lg[a]), float(lg[b])] for a, b in variants])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < VRAM_CAP_GB + 1:
        raise SystemExit(f"only {free / 2**30:.1f} GB free")
    torch.cuda.set_per_process_memory_fraction(VRAM_CAP_GB * 2**30 / total)
    model, tok, cfg, _bh, _path = G3.load_text(G3.TARGET, device="cuda")
    rm = R.GemmaRoleMap.from_config(cfg)
    params = dict(model.named_parameters())
    stop = sorted(CH.stop_ids(tok))
    canary_ids = [
        tok.encode(CH.chat_prompt(tok, u), add_special_tokens=False) for u in CANARY_PROMPTS
    ]

    items_all = json.loads((GATE1 / "items.json").read_text())
    items = items_all["medqa"] + items_all["medmcqa"]  # exposed Gate 1 primaries (500)
    gold = [it["answer_index"] for it in items]
    nopt = [len(it["options"]) for it in items]

    log = PV.ProvenanceLog(out / "provenance.jsonl")
    t0 = time.time()
    sk0 = K.weight_sketch(params, seed=20261121)
    bc0 = K.behavioural_canary(model, canary_ids, max_new_tokens=48, stop_ids=stop)
    before = PV.tensor_hashes(params)
    base_sha = PV.combine(before)
    log.append(
        "job_start",
        job="gcal-calibration",
        base_sha256=base_sha,
        sketch_sha256=K.sketch_sha256(sk0),
        behavioural_canary=bc0["sha256"],
        seconds=time.time() - t0,
    )

    base_lg = letter_logits(model, tok, items)

    factors_path = GATE1 / "lora" / "syc_lora.pt"
    factors = torch.load(factors_path)
    sess = A.EditSession(params, base_sha, rm)
    LR.apply_lean(sess, LR.lowrank_actions(factors, 1.0))
    after = dict(before)
    for n in sess.touched:
        after[n] = A.tensor_hash(params[n])
    recipe = {
        "family": "syc_lora",
        "lambda": 1.0,
        "source": "Gate 1 factors",
        "factors_sha256": sha_file(factors_path),
    }
    rec = log.append_edit(recipe, before, after, job="gcal-calibration")
    edit_lg = letter_logits(model, tok, items)
    sess.restore()
    restored = PV.combine(
        {**before, **{n: A.tensor_hash(params[n]) for n in rec["changed_tensors"]}}
    )

    sk1 = K.weight_sketch(params, seed=20261121)
    bc1 = K.behavioural_canary(model, canary_ids, max_new_tokens=48, stop_ids=stop)
    sk_cmp = K.compare_sketches(sk0, sk1)
    log.append(
        "job_end",
        job="gcal-calibration",
        restored_sha256=restored,
        sketch_pass=sk_cmp["pass"],
        behavioural_canary=bc1["sha256"],
    )

    np.savez(
        out / "letter_logits.npz",
        base=base_lg,
        edit=edit_lg,
        gold=np.array(gold),
        n_options=np.array(nopt),
    )
    pb, pe = P.gcal_probs(base_lg, nopt), P.gcal_probs(edit_lg, nopt)
    d = P.gcal_brier(pe, gold) - P.gcal_brier(pb, gold)
    n = len(d)
    s = float(d.std(ddof=1))
    from scipy.stats import chi2

    s_u = s * math.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
    gates = P.gcal_gates(base_lg, edit_lg, gold, nopt)
    res = {
        "status": "G-CAL calibration on EXPOSED Gate 1 items (development data); Board A C2-C3",
        "n_items": n,
        "sd_dBS": s,
        "sd_dBS_upper_97_5": s_u,
        "dev_mean_dBS": float(d.mean()),
        "dev_gcal_guard": P.gcal_guard(d),
        "edit_mean_bs": float(P.gcal_brier(pe, gold).mean()),
        "gates": gates,
        "prefix": PREFIX,
        "batch_size": 1,
        "canaries": {
            "weight_sketch_pass": sk_cmp["pass"],
            "weight_sketch_changed": sk_cmp["changed"],
            "behavioural_match": bc0["sha256"] == bc1["sha256"],
        },
        "provenance": {
            "base_sha256_from_bytes": base_sha,
            "edit_sha256": rec["edit_sha256"],
            "n_changed_tensors": len(rec["changed_tensors"]),
            "restored_equals_base": restored == base_sha,
            "log_records": len(log.records),
        },
        "inputs_sha256": {
            "gate1_items.json": sha_file(GATE1 / "items.json"),
            "syc_lora.pt": sha_file(factors_path),
            "letter_logits.npz": sha_file(out / "letter_logits.npz"),
        },
        "gpu_seconds": time.time() - t_start,
    }
    (out / "gcal_calibration.json").write_text(json.dumps(res, indent=1, default=float))
    (ROOT / "docs/prereg/wmm_gcal_calibration.json").write_text(
        json.dumps(res, indent=1, default=float)
    )
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "sd_dBS",
                    "sd_dBS_upper_97_5",
                    "dev_mean_dBS",
                    "gates",
                    "canaries",
                    "gpu_seconds",
                )
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
