"""Deterministic TF-v1 scoring on the shared HFRunner and batch execution path."""

import hashlib
import json
import time

import torch

from wsp.eval.metadata import runtime_metadata
from wsp.eval.subsets import REVISIONS, ROOT, SPECS, subset_ids
from wsp.schema import EvalRecord

SEED = 20261105


def frozen_ids(name, n):
    size = SPECS[name][2]
    excluded = set((ROOT / f"{name}.txt").read_text().splitlines())
    pool = [str(i) for i in range(size) if str(i) not in excluded]
    return [pool[int(i)] for i in subset_ids(len(pool), n, SEED)]


def load_items(name, n):
    from datasets import load_dataset

    repo, config, size, _, split = SPECS[name]
    ds = load_dataset(repo, config, split=split, revision=REVISIONS[name])
    if len(ds) != size:
        raise ValueError("dataset cardinality changed")
    ids = (ROOT / f"{name}_tf_v1.txt").read_text().splitlines()[:n]
    if ids != frozen_ids(name, len(ids)) or len(ids) != n:
        raise ValueError("TF-v1 frozen item set changed")
    return [(i, ds[int(i)]) for i in ids], {
        "repo": repo,
        "revision": REVISIONS[name],
        "split": split,
        "fingerprint": ds._fingerprint,
        "subset_ids": ids,
    }


def load_prompts(kind, n):
    from datasets import load_dataset

    harmful = kind == "refusal_rate"
    repo, revision = (
        ("mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7")
        if harmful
        else ("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f")
    )
    ds = load_dataset(repo, split="test", revision=revision)
    texts = list(ds["text"])[:n]
    if len(texts) != n:
        raise ValueError("insufficient TF prompts")
    return texts, {
        "repo": repo,
        "revision": revision,
        "split": "test",
        "fingerprint": ds._fingerprint,
        "subset_ids": [str(i) for i in range(n)],
    }


def choice_record(edit_id, name, item_id, scores, gold):
    values = torch.tensor(scores, dtype=torch.float64)
    if name == "mmlu_pro":
        values = values.log_softmax(0)
    probs = values.tolist()
    wrong = max(v for i, v in enumerate(probs) if i != gold)
    return EvalRecord(
        edit_id,
        name,
        item_id,
        float(max(range(len(scores)), key=scores.__getitem__) == gold),
        extra={
            "gold_index": gold,
            "gold_log_prob": probs[gold],
            "margin": probs[gold] - wrong,
            "option_log_probs": probs,
            "raw_option_log_likelihoods": scores,
            **({"ending_normalized_log_likelihoods": scores} if name == "hellaswag" else {}),
        },
    )


def summarize(records):
    result = {"version": "tf_v1", "qualification": "unqualified; exploratory timing pilot"}
    for name in ("mmlu_pro", "hellaswag"):
        rows = [r for r in records if r.benchmark == name]
        if rows:
            result[name] = {
                "n": len(rows),
                "accuracy": sum(r.correct for r in rows) / len(rows),
                **{
                    f"mean_{key}": sum(r.extra[key] for r in rows) / len(rows)
                    for key in ("gold_log_prob", "margin")
                },
            }
    rows = [r for r in records if r.benchmark == "wikitext_ppl"]
    if rows:
        total = sum(r.n_tokens for r in rows)
        result["wikitext"] = {
            "n_windows": len(rows),
            "n_tokens": total,
            "log_ppl": sum(r.extra["mean_nll"] * r.n_tokens for r in rows) / total,
        }
    for name in ("refusal_rate", "over_refusal"):
        rows = [r for r in records if r.benchmark == name]
        if rows:
            result[name] = {
                "n": len(rows),
                "mean_first_token_margin": sum(r.correct for r in rows) / len(rows),
            }
    rows = [r for r in records if r.benchmark == "kl_to_base"]
    if rows:
        result["kl_to_base"] = {
            "n": len(rows),
            **{
                key: sum(r.extra[key] for r in rows) / len(rows)
                for key in ("first_token_kl", "full_sequence_kl")
            },
        }
    return result


def token_mass_margin(logits, refusal, compliance):
    if not refusal or not compliance or set(refusal) & set(compliance):
        raise ValueError("nonempty disjoint token sets required")
    x = logits.float().log_softmax(-1)
    return (x[refusal].logsumexp(0) - x[compliance].logsumexp(0)).item()


def last_logits(runner, ids, n=1):
    with torch.inference_mode():
        return (
            runner.model(
                torch.tensor([ids], device=runner.model.device), logits_to_keep=n, use_cache=False
            )
            .logits[0]
            .float()
        )


def letter_scores(runner, context, choices):
    prefix = runner.tokenizer.encode(context, add_special_tokens=False)
    suffixes = [runner.tokenizer.encode(" " + c, add_special_tokens=False) for c in choices]
    if all(len(s) == 1 for s in suffixes):
        lp = last_logits(runner, prefix)[-1].log_softmax(-1)
        return [lp[s[0]].item() for s in suffixes]
    return runner.choice_scores(context, choices)


def cache_kl(runner, texts, metadata):
    """Immutable per-context tensors, exact bf16 model logits, float32 KL arithmetic."""
    from wsp.eval.batch import effective_config_hash, provenance

    identity = (
        runtime_metadata(runner)
        | provenance()
        | {
            "model_revision": runner.revision,
            "generation": runner.generation,
            "config_hash": effective_config_hash(runner.tf_config),
            "prompts": texts,
            "dataset": metadata,
            "max_kl_tokens": 128,
        }
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = runner.tf_cache_root / identity["eval_device_id"] / key
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        for name, digest in saved["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("TF cache hash mismatch")
        return root, saved
    files = {}
    with runner.base_context():
        for i, prompt in enumerate(texts):
            prefix = runner.tokenizer.encode(runner.chat(prompt), add_special_tokens=False)
            with torch.inference_mode():
                ids = torch.tensor([prefix], device=runner.model.device)
                # Greedy base continuation: no sampled tokens anywhere in TF-v1.
                seq = runner.model.generate(
                    ids,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=runner.tokenizer.pad_token_id,
                )[0].tolist()
                continuation = seq[len(prefix) :]
                if not continuation:
                    raise ValueError("empty base continuation")
                logits = (
                    last_logits(runner, seq, len(continuation) + 1)[:-1].to(torch.bfloat16).cpu()
                )
            file = root / f"{i}.pt"
            tmp = file.with_suffix(".tmp")
            torch.save({"ids": seq, "prefix_length": len(prefix), "logits": logits}, tmp)
            tmp.replace(file)
            files[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
            if i % 25 == 0:
                print(f"TF base cache {i}/{len(texts)}", flush=True)
    saved = {"identity": identity, "cache_hash": key, "files": files}
    manifest.write_text(json.dumps(saved, indent=2))
    return root, saved


def evaluate(runner, names, counts):
    from wsp.eval.refusal import kl_means
    from wsp.eval.run import perplexity

    records, datasets = [], {}
    runner.tf_stage_seconds = {}
    for name in names:
        start = time.monotonic()
        cfg = counts[name]
        if name in ("mmlu_pro", "hellaswag"):
            items, datasets[name] = load_items(name, cfg["n"])
            for item_id, row in items:
                if name == "mmlu_pro":
                    choices = [chr(65 + i) for i in range(len(row["options"]))]
                    context = (
                        row["question"]
                        + "\n"
                        + "\n".join(f"{c}. {o}" for c, o in zip(choices, row["options"]))
                        + "\nAnswer:"
                    )
                    scores = letter_scores(runner, context, choices)
                    gold = row["answer_index"]
                else:
                    context, choices, gold = row["ctx"], row["endings"], int(row["label"])
                    scores = runner.choice_scores(context, choices, True)
                r = choice_record(runner.edit_id, name, item_id, scores, gold)
                r.extra.update(prompt=context, choices=choices)
                records.append(r)
        elif name == "wikitext_ppl":
            records.extend(perplexity(runner, cfg))
            datasets[name] = {
                "repo": "Salesforce/wikitext",
                "config": "wikitext-2-raw-v1",
                "split": "test",
                "window": cfg["window"],
                "overlap": 0,
            }
        elif name in ("refusal_rate", "over_refusal"):
            texts, datasets[name] = load_prompts(name, cfg["n"])
            if len(texts) != cfg["n"]:
                raise ValueError("insufficient refusal prompts")
            token_sets = runner.tf_config["tf_v1"]["first_token_sets"]
            refusal, compliance = [], []
            for target, key in ((refusal, "refusal"), (compliance, "compliance")):
                for text in token_sets[key]:
                    target.append(runner.tokenizer.encode(text, add_special_tokens=False)[0])
            refusal, compliance = sorted(set(refusal)), sorted(set(compliance))
            for i, text in enumerate(texts):
                logits = last_logits(
                    runner, runner.tokenizer.encode(runner.chat(text), add_special_tokens=False)
                )[-1]
                margin = token_mass_margin(logits, refusal, compliance)
                records.append(
                    EvalRecord(
                        runner.edit_id,
                        name,
                        str(i),
                        margin,
                        extra={
                            "first_token_margin": margin,
                            "refusal_token_ids": refusal,
                            "compliance_token_ids": compliance,
                            "prompt": text,
                            "token_sets": token_sets,
                        },
                    )
                )
        elif name == "kl_to_base":
            texts, datasets[name] = load_prompts(name, cfg["n"])
            if len(texts) != cfg["n"]:
                raise ValueError("insufficient KL prompts")
            root, cache = cache_kl(runner, texts, datasets[name])
            datasets[name]["cache_hash"] = cache["cache_hash"]
            for i, text in enumerate(texts):
                data = torch.load(root / f"{i}.pt", weights_only=True)
                n = len(data["ids"]) - data["prefix_length"]
                edited = last_logits(runner, data["ids"], n + 1)[:-1]
                first, full = kl_means(data["logits"].to(edited.device), edited)
                records.append(
                    EvalRecord(
                        runner.edit_id,
                        name,
                        str(i),
                        full,
                        n,
                        extra={
                            "first_token_kl": first,
                            "full_sequence_kl": full,
                            "prompt": text,
                            "continuation_token_ids": data["ids"][data["prefix_length"] :],
                            "cache_hash": cache["cache_hash"],
                            "direction": "base || edited",
                        },
                    )
                )
        else:
            raise ValueError(f"unsupported TF-v1 benchmark: {name}")
        runner.tf_stage_seconds[name] = time.monotonic() - start
        print(f"TF {name}: {runner.tf_stage_seconds[name]:.2f}s", flush=True)
    return records, datasets
