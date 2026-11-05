#!/usr/bin/env python3
"""Measure (do not adopt) batched HellaSwag-ending and refusal-phrase scoring against batch 1.

Board A's optional cost reduction: score the 4 HellaSwag endings and the 13 refusal/compliance
phrases of a prompt in one right-padded forward instead of 4 / 7 batch-1 forwards. This probe lives
under scripts/ so it does not change `src/wsp` (and with it the `source_hash` every TF-v1 reference
was measured on). It times both paths on the base model, over the TF-v1 panels
(HellaSwag first 250, refusal 50 + over-refusal 50), reports the largest score deviation and
argmax / margin changes versus batch 1, and whether the batched path repeats bitwise.

Usage: tf_v1_batching_probe.py     (writes data/evals/qwen3.5-0.8b/tf_v1/batching_probe.json)
"""

import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data/evals/qwen3.5-0.8b/tf_v1/batching_probe.json"


def sync():
    torch.cuda.synchronize()
    return time.monotonic()


def batched_logprobs(runner, rows, keep):
    """rows: list of (token_ids, first_row_index, n_rows). Right-pad; return per-row log-softmax."""
    width = max(len(ids) for ids, _, _ in rows)
    pad = runner.tokenizer.pad_token_id or 0
    batch = torch.tensor(
        [ids + [pad] * (width - len(ids)) for ids, _, _ in rows], device=runner.model.device
    )
    mask = torch.tensor(
        [[1] * len(ids) + [0] * (width - len(ids)) for ids, _, _ in rows],
        device=runner.model.device,
    )
    with torch.inference_mode():
        logits = runner.model(batch, attention_mask=mask, use_cache=False).logits.float()
    return [logits[k, start : start + n].log_softmax(-1) for k, (_, start, n) in enumerate(rows)]


def hellaswag_batched(runner, context, endings):
    prefix = runner.tokenizer.encode(context, add_special_tokens=False)
    suffixes = [runner.tokenizer.encode(" " + e, add_special_tokens=False) for e in endings]
    rows = [(prefix + s, len(prefix) - 1, len(s)) for s in suffixes]
    scores = []
    for lp, s, e in zip(batched_logprobs(runner, rows, None), suffixes, endings, strict=True):
        scores.append(
            lp.gather(1, torch.tensor(s, device=lp.device)[:, None]).sum().item() / len(e)
        )
    return scores


def phrases_batched(runner, prefix, phrases):
    rows = [(prefix + ids[:-1], len(prefix) - 1, len(ids)) for ids in phrases.values()]
    out = {}
    for lp, (p, ids) in zip(batched_logprobs(runner, rows, None), phrases.items(), strict=True):
        out[p] = sum(lp[k, t].item() for k, t in enumerate(ids))
    return out


def main():
    from wsp.eval import tf_v1
    from wsp.eval.run import HFRunner
    from wsp.zoo.common import resolve_base_revision

    cfg = yaml.safe_load((ROOT / "configs/eval_tf_v1.yaml").read_bytes())
    tf = cfg["tf_v1"]
    zoo = ROOT / "data/zoo/qwen3.5-0.8b"
    header = next(
        r
        for r in reversed(
            [json.loads(x) for x in (zoo / "manifest.jsonl").read_text().splitlines()]
        )
        if r.get("type") == "header"
    )
    runner = HFRunner(
        header["base_model"],
        dict(cfg["generation"]),
        revision=resolve_base_revision(header["base_model"], header["base_revision"]),
    )
    runner.edit_id = "base"
    result = {"model": "base (unedited)", "device": torch.cuda.get_device_name()}

    ids, _ = tf_v1.panel_ids("hellaswag", 250, tf["item_salt"])
    from datasets import load_dataset

    from wsp.eval.subsets import REVISIONS, SPECS

    repo, config, _, _, split = SPECS["hellaswag"]
    ds = load_dataset(repo, config, split=split, revision=REVISIONS["hellaswag"])
    items = [ds[int(i)] for i in ids]
    t = sync()
    single = [runner.choice_scores(r["ctx"], r["endings"], True) for r in items]
    t_single = sync() - t
    batched_a, batched_b = [], []
    t = sync()
    for r in items:
        batched_a.append(hellaswag_batched(runner, r["ctx"], r["endings"]))
    t_batched = sync() - t
    for r in items:
        batched_b.append(hellaswag_batched(runner, r["ctx"], r["endings"]))
    argmax = lambda s: max(range(len(s)), key=s.__getitem__)
    result["hellaswag"] = {
        "items": len(items),
        "seconds_batch1": t_single,
        "seconds_batched4": t_batched,
        "max_abs_score_diff": max(
            abs(a - b)
            for x, y in zip(single, batched_a, strict=True)
            for a, b in zip(x, y, strict=True)
        ),
        "argmax_changes": sum(
            argmax(x) != argmax(y) for x, y in zip(single, batched_a, strict=True)
        ),
        "batched_repeat_bitwise": batched_a == batched_b,
    }

    sets = tf_v1.build_continuations(runner.tokenizer, tf["refusal_continuations"])
    phrases = sets["refusal"] | sets["compliance"]
    for kind in ("refusal_rate", "over_refusal"):
        texts, _ = tf_v1.load_prompts(kind, 50, tf)
        prefixes = [
            runner.tokenizer.encode(runner.chat(x), add_special_tokens=False) for x in texts
        ]
        t = sync()
        single = [tf_v1.continuation_scores(runner, p, phrases) for p in prefixes]
        t_single = sync() - t
        t = sync()
        a = [phrases_batched(runner, p, phrases) for p in prefixes]
        t_batched = sync() - t
        b = [phrases_batched(runner, p, phrases) for p in prefixes]
        margin = lambda s: tf_v1.phrase_mass_margin(
            [s[k] for k in sets["refusal"]], [s[k] for k in sets["compliance"]]
        )
        result[kind] = {
            "prompts": len(texts),
            "seconds_batch1": t_single,
            "seconds_batched": t_batched,
            "max_abs_phrase_logprob_diff": max(
                abs(x[k] - y[k]) for x, y in zip(single, a, strict=True) for k in x
            ),
            "max_abs_margin_diff": max(
                abs(margin(x) - margin(y)) for x, y in zip(single, a, strict=True)
            ),
            "batched_repeat_bitwise": a == b,
        }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
