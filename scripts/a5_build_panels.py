#!/usr/bin/env python3
"""A5.11 pre-G2 companion files: auxiliary panels, control seeds, NULL-SIZE input specification.

Writes three files into ``docs/prereg/a5_manifests/`` (CPU only, no timestamps, no absolute paths) and a
marked section of its README:

* ``panels.json`` — the G2 auxiliary panels (A5.2.1, A5.8.1, A5.8.3): WikiText, MMLU-Pro and HellaSwag
  gradient panels, harmful/harmless refusal-margin panels and the 64-prompt probe panel. Each panel names
  dataset, revision, split, cache fingerprint, the deterministic rule, the explicit ids with sha256 of the
  sorted list, and a disjointness certificate against every registered scored set (computed by id/text
  group, or stated as holding by construction where the scored set does not exist yet).
* ``control_seeds.json`` — A5.8.3 random/permuted-field seed derivation, A5.3.3 rotation seed reference and
  the A5.3.2 simulation ``SeedSequence(20261107).spawn(3)`` order, with generated check values.
* ``nullsize_inputs.json`` — the specification of the development window-level inputs that
  ``scripts/a5_simulation.py`` reads for NULL-SIZE (validated by its ``--inputs-spec`` mode). It is not run
  on real data here.

Datasets are read from the local Hugging Face cache only (offline). If a dataset, revision or tokenizer is not
cached, the script reports what is missing and stops without writing anything: no id is ever invented.

    python scripts/a5_build_panels.py [--out docs/prereg/a5_manifests] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIELD_DOMAIN = "20261107:field:"  # A5.8.1 auxiliary sampling domain
PROBE_DOMAIN = "20261107:probe:"  # chair default proposed here; A5.8.3 names no probe domain
RANDOM_FIELD_DOMAIN = "20261107:random-field:"  # A5.8.3
PERMUTED_FIELD_DOMAIN = "20261107:permuted-field:"  # A5.8.3
ROTATION_DOMAIN = "20261107:rotation:"  # A5.3.3
FIELD_GROUPS = 64  # A5.8.1: 64 groups per objective
WINDOW = 1024
DEV_WINDOWS = 64
TEST_KEY = "synthetic-test-vector"  # not a tensor, role or edit id
README_BEGIN = "<!-- a5_build_panels:begin -->"
README_END = "<!-- a5_build_panels:end -->"
OUT_FILES = ("panels.json", "control_seeds.json", "nullsize_inputs.json")

MODEL = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1", "test")
HARMFUL = ("mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7")
HARMLESS = ("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f")
EXTRACTION_PREFIX = 400  # src/wsp/edits/refusal.py SPLIT = "train[:400]"
LEGACY_PROMPT_PREFIX = (
    100  # heretic default prompts split "test[:100]" (configs/eval_cheap.yaml n=100)
)


class NotCached(RuntimeError):
    """A required dataset/tokenizer is not in the local cache; nothing is written."""


# ------------------------------------------------------------------------------------------------ helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sorted_sha(ids) -> str:
    """Same convention as text_groups.json: sha256 of json.dumps(sorted list of strings)."""
    return sha256_bytes(json.dumps(sorted(ids)).encode())


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()


def normalize(text: str) -> str:
    """Text-group key: NFKC, casefold, whitespace collapsed, stripped."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def rank_key(domain: str, fingerprint: str) -> str:
    return sha256_bytes(f"{domain}{fingerprint}".encode())


def domain_seed(domain: str, key: str) -> int:
    """A5.3.3 convention: sha256(domain + key), first eight bytes big-endian, mod 2^63."""
    return int.from_bytes(hashlib.sha256(f"{domain}{key}".encode()).digest()[:8], "big") % (1 << 63)


def derangement(n: int, seed: int) -> list[int]:
    """Seeded derangement of range(n): PCG64(seed) permutations, first one with no fixed point.

    n == 1 returns [0] (singleton, stays fixed and is disclosed); n == 0 returns [].
    """
    import numpy as np

    if n < 2:
        return list(range(n))
    rng = np.random.Generator(np.random.PCG64(seed))
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return [int(x) for x in p]


def group_items(item_ids, keys_of):
    """Union-find text groups: items sharing any key are one group. Returns {item: representative}.

    The representative is the smallest item id under (numeric part, string) order.
    """
    parent = {i: i for i in item_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i in item_ids:
        for k in keys_of(i):
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    parent[max(a, b, key=id_order)] = min(a, b, key=id_order)
            else:
                first[k] = i
    members = {}
    for i in item_ids:
        members.setdefault(find(i), []).append(i)
    rep = {}
    for group in members.values():
        r = min(group, key=id_order)
        for i in group:
            rep[i] = r
    return rep


def id_order(item_id: str):
    head, _, tail = item_id.rpartition(":")
    return (head, int(tail)) if tail.isdigit() else (head, -1, tail)


def select_groups(rep, excluded_items, domain, fingerprint_of, n):
    """Rank eligible groups (no member in ``excluded_items``) by sha256(domain + fingerprint(rep))."""
    members = {}
    for i, r in rep.items():
        members.setdefault(r, []).append(i)
    blocked = {rep[i] for i in excluded_items if i in rep}
    eligible = [r for r in members if r not in blocked]
    ranked = sorted(eligible, key=lambda r: (rank_key(domain, fingerprint_of(r)), id_order(r)))
    chosen = ranked[:n]
    return (
        [
            {
                "item_id": r,
                "rank": k,
                "rank_key": rank_key(domain, fingerprint_of(r)),
                "group_members": sorted(members[r], key=id_order),
            }
            for k, r in enumerate(chosen)
        ],
        {"eligible_groups": len(eligible), "excluded_groups": len(blocked), "groups": len(members)},
    )


def certificate(items, rep, scored_sets, construction=()):
    """Item- and text-group-level overlap of a panel's reserved items with each registered scored set."""
    reserved = {m for it in items for m in it["group_members"]}
    groups = {rep[m] for m in reserved}
    checks = {}
    for name, ids in sorted(scored_sets.items()):
        ids = set(ids)
        checks[name] = {
            "n_checked": len(ids),
            "item_overlap": len(reserved & ids),
            "group_overlap": len(groups & {rep[i] for i in ids if i in rep}),
        }
    ok = all(c["item_overlap"] == 0 and c["group_overlap"] == 0 for c in checks.values())
    return {
        "computed_checks": checks,
        "computed_disjoint": ok,
        "by_construction": list(construction),
    }


def panel_block(items, rep, scored_sets, construction, n_requested, **extra):
    reserved = sorted({m for it in items for m in it["group_members"]}, key=id_order)
    ids = [it["item_id"] for it in items]
    return {
        **extra,
        "n_requested": n_requested,
        "n_produced": len(items),
        "short_by": max(0, n_requested - len(items)),
        "item_ids": ids,
        "item_ids_sorted_sha256": sorted_sha(ids),
        "reserved_item_ids": reserved,
        "reserved_item_ids_sorted_sha256": sorted_sha(reserved),
        "items": items,
        "certificate": certificate(items, rep, scored_sets, construction),
    }


# ---------------------------------------------------------------------------------------------- wikitext


def wikitext_articles(lines):
    """Article starts: ' = Title = ' lines with empty neighbours (sub-headings use '= =')."""
    pattern = re.compile(r" = [^=].* = \n")
    starts = [
        i
        for i, x in enumerate(lines)
        if pattern.fullmatch(x)
        and (i == 0 or lines[i - 1] == "")
        and (i + 1 == len(lines) or lines[i + 1] == "")
    ]
    if not starts:
        raise ValueError("no WikiText article headings")
    titles = [lines[s].strip(" =\n") for s in starts]
    starts[0] = 0  # leading blank lines belong to the first article
    ends = starts[1:] + [len(lines)]
    return [
        {"doc_id": f"article:{a}", "line_start": a, "line_end": b, "title": t}
        for a, b, t in zip(starts, ends, titles)
    ]


def line_char_starts(lines, joiner="\n\n"):
    starts, pos = [], 0
    for x in lines:
        starts.append(pos)
        pos += len(x) + len(joiner)
    return starts


def build_wikitext(
    lines, encode, encode_offsets, text_groups, n_windows=DEV_WINDOWS, window=WINDOW
):
    """Development windows -> touched articles; field windows from hash-ranked untouched articles.

    ``encode(text) -> ids``; ``encode_offsets(text) -> (ids, offsets)`` with character offsets.
    """
    text = "\n\n".join(lines)
    ids, offsets = encode_offsets(text)
    reproduced = all(
        sha256_bytes(json.dumps(ids[w["token_start"] : w["token_end"]]).encode())
        == w["token_ids_hash"]
        for w in text_groups["windows"]
    )
    dev_tokens = max(w["token_end"] for w in text_groups["windows"])
    dev_end_char = offsets[dev_tokens - 1][1]
    articles = wikitext_articles(lines)
    starts = line_char_starts(lines)
    touched = []
    for a in articles:
        a["char_start"] = starts[a["line_start"]]
        if a["char_start"] < dev_end_char:
            touched.append(a["doc_id"])
    fp = f"{WIKITEXT[0]}:{WIKITEXT[1]}:{WIKITEXT[2]}@{text_groups['source']['revision']}"
    eligible = [a for a in articles if a["doc_id"] not in touched]
    ranked = sorted(eligible, key=lambda a: rank_key(FIELD_DOMAIN, f"{fp}:{a['doc_id']}"))
    chosen, stream_ids = [], []
    for a in ranked:
        if len(stream_ids) >= n_windows * window:
            break
        chosen.append(a)
        stream_ids = encode(
            "\n\n".join(x for c in chosen for x in lines[c["line_start"] : c["line_end"]])
        )
    # Token -> article map of the panel stream (articles concatenated in rank order, same joiner).
    stream_text = "\n\n".join(x for c in chosen for x in lines[c["line_start"] : c["line_end"]])
    s_ids, s_off = encode_offsets(stream_text)
    if s_ids != stream_ids:
        raise ValueError("offset tokenisation differs from encode()")
    bounds, pos = [], 0
    for c in chosen:
        seg = "\n\n".join(lines[c["line_start"] : c["line_end"]])
        bounds.append((pos, pos + len(seg), c["doc_id"]))
        pos += len(seg) + 2
    windows = []
    for i in range(min(n_windows, len(s_ids) // window)):
        lo, hi = i * window, (i + 1) * window
        c0, c1 = s_off[lo][0], s_off[hi - 1][1]
        docs = [d for a, b, d in bounds if a < c1 and b > c0]
        windows.append(
            {
                "window_id": f"{fp}:a5-field-stream:window{window}:{i}",
                "index": i,
                "token_start": lo,
                "token_end": hi,
                "token_ids_hash": sha256_bytes(json.dumps(s_ids[lo:hi]).encode()),
                "articles": docs,
            }
        )
    dev_hashes = {w["token_ids_hash"] for w in text_groups["windows"]}
    chosen_ids = {c["doc_id"] for c in chosen}
    remaining = [a for a in eligible if a["doc_id"] not in chosen_ids]
    return {
        "fingerprint_prefix": fp,
        "dev_windows_reproduced": reproduced,
        "dev_token_span": [0, dev_tokens],
        "dev_char_end": dev_end_char,
        "n_articles": len(articles),
        "dev_touched_articles": touched,
        "articles": [
            {
                "doc_id": c["doc_id"],
                "title": c["title"],
                "line_start": c["line_start"],
                "line_end": c["line_end"],
                "rank": k,
                "rank_key": rank_key(FIELD_DOMAIN, f"{fp}:{c['doc_id']}"),
            }
            for k, c in enumerate(chosen)
        ],
        "windows": windows,
        "stream_tokens": len(s_ids),
        "token_overlap_with_dev_windows": len(dev_hashes & {w["token_ids_hash"] for w in windows}),
        "article_overlap_with_dev": len(chosen_ids & set(touched)),
        "unreserved_articles": [a["doc_id"] for a in remaining],
        "full_split_tokens": len(ids),
    }


# ----------------------------------------------------------------------------------------- data loading


def load_cached(repo, config, split, revision, cache_dir):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    try:
        ds = load_dataset(repo, config, split=split, revision=revision, cache_dir=str(cache_dir))
    except Exception as exc:  # noqa: BLE001 - any loader failure means "not cached": stop
        raise NotCached(
            f"{repo} {config} {split}@{revision}: {type(exc).__name__}: {exc}"
        ) from None
    files = [f["filename"] for f in ds.cache_files]
    if not files or not all(revision in f for f in files):
        raise NotCached(f"{repo} {split}: cached files are not revision {revision}")
    return ds


def load_tokenizer(cache_dir):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1], cache_dir=str(cache_dir))
    except Exception as exc:  # noqa: BLE001
        raise NotCached(f"tokenizer {MODEL[0]}@{MODEL[1]}: {exc}") from None


def ranking_file(name):
    from wsp.eval.tf_v1 import eligible_ids, rank_ids, read_panel

    header, ids = read_panel(name)
    if ids != rank_ids(eligible_ids(name), header["salt"])[: len(ids)]:
        raise ValueError(f"committed TF-v1 ranking {name} does not reproduce")
    return header, ids


def file_entry(path: Path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_bytes(path.read_bytes())}


# ------------------------------------------------------------------------------------------------ panels


def build_panels(data, tf_cfg, text_groups, inputs):
    """``data``: dict with loaded columns (see :func:`load_all`). Pure function for tests."""
    out = {"schema": "a5-panels-v1", "inputs": inputs, "panels": {}, "checked_sets": {}}
    off = tf_cfg["tf_v1"]["over_refusal_offset"]
    b = tf_cfg["benchmarks"]

    # ---- MMLU-Pro
    mm = data["mmlu_pro"]
    rep = group_items(
        [str(i) for i in range(len(mm["question"]))], lambda i: [normalize(mm["question"][int(i)])]
    )
    scored = {
        "mmlu_pro_tf_v1_development_ranking": data["rankings"]["mmlu_pro"],
        "mmlu_pro_legacy": data["legacy"]["mmlu_pro"],
    }
    fp = f"{mm['repo']}@{mm['revision']}:{mm['split']}"
    items, counts = select_groups(
        rep, set().union(*scored.values()), FIELD_DOMAIN, lambda r: f"{fp}:{r}", 500
    )
    for it in items:
        it["question_id"] = mm["question_id"][int(it["item_id"])]
    out["panels"]["mmlu_gold_log_prob"] = panel_block(
        items,
        rep,
        scored,
        [SEALED_MMLU],
        500,
        dataset=_ds(mm),
        group_rule="normalized question text",
        selection_rule=_rule(fp),
        field_prefix=FIELD_GROUPS,
        pool_counts=counts,
    )
    out["checked_sets"].update({k: _set(v) for k, v in scored.items()})

    # ---- HellaSwag
    hs = data["hellaswag"]
    rep = group_items(
        [str(i) for i in range(len(hs["ctx"]))],
        lambda i: ["src:" + hs["source_id"][int(i)], "ctx:" + normalize(hs["ctx"][int(i)])],
    )
    scored = {
        "hellaswag_tf_v1_development_ranking": data["rankings"]["hellaswag"],
        "hellaswag_legacy": data["legacy"]["hellaswag"],
    }
    fp = f"{hs['repo']}@{hs['revision']}:{hs['split']}"
    items, counts = select_groups(
        rep, set().union(*scored.values()), FIELD_DOMAIN, lambda r: f"{fp}:{r}", 500
    )
    for it in items:
        it["source_id"] = hs["source_id"][int(it["item_id"])]
    out["panels"]["hellaswag_gold_char_log_likelihood"] = panel_block(
        items,
        rep,
        scored,
        [SEALED_OTHER],
        500,
        dataset=_ds(hs),
        group_rule="union of items sharing source_id or normalized ctx",
        selection_rule=_rule(fp),
        field_prefix=FIELD_GROUPS,
        pool_counts=counts,
        scored_in_tf_v1=f"first {b['hellaswag']['n']} of the 500-item development ranking (all 500 excluded)",
    )
    out["checked_sets"].update({k: _set(v) for k, v in scored.items()})

    # ---- harmful (train and test splits; ids "split:index")
    hf = data["harmful"]
    ids = [f"{s}:{i}" for s in ("train", "test") for i in range(len(hf[s]))]
    rep = group_items(ids, lambda i: [normalize(hf[i.split(":")[0]][int(i.split(":")[1])])])
    scored = {
        "harmful_tf_v1_refusal_rate": [f"test:{i}" for i in range(b["refusal_rate"]["n"])],
        "harmful_legacy_heretic_test_prefix": [f"test:{i}" for i in range(LEGACY_PROMPT_PREFIX)],
        "harmful_direction_extraction_train_prefix": [
            f"train:{i}" for i in range(EXTRACTION_PREFIX)
        ],
    }
    fp = f"{HARMFUL[0]}@{HARMFUL[1]}"
    items, counts = select_groups(
        rep, set().union(*scored.values()), FIELD_DOMAIN, lambda r: f"{fp}:{r}", FIELD_GROUPS
    )
    out["panels"]["refusal_margin_harmful"] = panel_block(
        items,
        rep,
        scored,
        [SEALED_OTHER],
        FIELD_GROUPS,
        dataset=_prompt_ds(hf, HARMFUL, ("train", "test")),
        group_rule="normalized prompt text",
        selection_rule=_rule(fp + ":<split>:<index>"),
        pool_counts=counts,
        role="harmful",
    )
    # Not registered: the extraction prompts are exposure (edit-construction inputs), not scored items.
    cond, _ = select_groups(
        rep,
        set(scored["harmful_tf_v1_refusal_rate"])
        | set(scored["harmful_legacy_heretic_test_prefix"])
        | {it["item_id"] for it in items},
        FIELD_DOMAIN,
        lambda r: f"{fp}:{r}",
        FIELD_GROUPS - len(items),
    )
    out["panels"]["refusal_margin_harmful"]["conditional_fill_not_registered"] = {
        "status": "CONDITIONAL, NOT REGISTERED: requires a Board A/chair ruling admitting direction-extraction "
        "prompts (abliteration source inputs, A5.2.1 generated-direction identities) into the refusal panel",
        "item_ids": [it["item_id"] for it in cond],
        "item_ids_sorted_sha256": sorted_sha([it["item_id"] for it in cond]),
    }
    out["checked_sets"].update({k: _set(v) for k, v in scored.items()})

    # ---- harmless (test split, the TF-v1 prompt pool) and the probe panel
    hl = data["harmless"]
    ids = [str(i) for i in range(len(hl["test"]))]
    extraction_texts = {normalize(t) for t in hl["train"][:EXTRACTION_PREFIX]}
    rep = group_items(ids, lambda i: [normalize(hl["test"][int(i)])])
    blocked_by_text = [i for i in ids if normalize(hl["test"][int(i)]) in extraction_texts]
    scored = {
        "harmless_tf_v1_kl_and_over_refusal_ranking": data["rankings"]["harmless_alpaca"],
        "harmless_legacy_heretic_test_prefix": [str(i) for i in range(LEGACY_PROMPT_PREFIX)],
        "harmless_matching_direction_extraction_text": blocked_by_text,
    }
    fp = f"{HARMLESS[0]}@{HARMLESS[1]}:test"
    items, counts = select_groups(
        rep, set().union(*scored.values()), FIELD_DOMAIN, lambda r: f"{fp}:{r}", FIELD_GROUPS
    )
    out["panels"]["refusal_margin_harmless"] = panel_block(
        items,
        rep,
        scored,
        [SEALED_OTHER],
        FIELD_GROUPS,
        dataset=_prompt_ds(hl, HARMLESS, ("test",)),
        group_rule="normalized prompt text",
        selection_rule=_rule(fp),
        pool_counts=counts,
        role="harmless",
        tf_v1_scored=f"ranks [0, {off + b['over_refusal']['n']}) of the committed ranking",
    )
    probe_scored = dict(scored)
    probe_scored["refusal_margin_harmless_panel"] = [m for it in items for m in it["group_members"]]
    probe, pcounts = select_groups(
        rep, set().union(*probe_scored.values()), PROBE_DOMAIN, lambda r: f"{fp}:{r}", FIELD_GROUPS
    )
    out["panels"]["probe_response"] = panel_block(
        probe,
        rep,
        probe_scored,
        [SEALED_OTHER],
        FIELD_GROUPS,
        dataset=_prompt_ds(hl, HARMLESS, ("test",)),
        group_rule="normalized prompt text",
        pool_counts=pcounts,
        selection_rule=f"rank eligible groups by sha256('{PROBE_DOMAIN}' + '{fp}:' + representative id); "
        "representative = smallest positional id; first 64",
    )
    out["checked_sets"].update({k: _set(v) for k, v in scored.items()})

    # ---- WikiText
    wt = data["wikitext"]
    articles = {a["doc_id"] for a in wt["articles"]}
    n_req = len(text_groups["windows"])
    windows_ok = len(wt["windows"]) == n_req and wt["token_overlap_with_dev_windows"] == 0
    out["panels"]["wikitext_nll"] = {
        "dataset": {
            "repo": WIKITEXT[0],
            "config": WIKITEXT[1],
            "split": WIKITEXT[2],
            "revision": text_groups["source"]["revision"],
            "fingerprint": data["wikitext_fingerprint"],
            "tokenizer": {"repo": MODEL[0], "revision": MODEL[1]},
        },
        "group_rule": "text group = article: a ' = Title = ' line with empty neighbouring lines starts an article "
        "(sub-headings ' = = ' do not); leading blank lines belong to the first article",
        "selection_rule": (
            "articles touched by the development span (any character before the end of development token "
            f"{wt['dev_token_span'][1] - 1} in tokenizer.encode('\\n\\n'.join(test lines))) are excluded; the "
            f"rest are ranked by sha256('{FIELD_DOMAIN}' + '{wt['fingerprint_prefix']}:' + doc_id); whole "
            "articles are taken in rank order, their lines joined with '\\n\\n' in rank order, until the "
            f"stream has >= {DEV_WINDOWS * WINDOW} tokens; window i = stream tokens [i*1024, (i+1)*1024), i < 64"
        ),
        "deviation_from_a5_8_1": "A5.8.1 asks for one item per text group; the test split has "
        f"{wt['n_articles']} articles, {len(wt['dev_touched_articles'])} of them touched by development "
        "windows, so 64 one-window-per-article groups do not exist in this split. Windows here may span "
        "article boundaries inside the panel stream (as development windows do); the text group of every "
        "window is listed. Board A ruling required.",
        "n_requested": n_req,
        "n_produced": len(wt["windows"]),
        "short_by": max(0, n_req - len(wt["windows"])),
        "window_ids": [w["window_id"] for w in wt["windows"]],
        "window_ids_sorted_sha256": sorted_sha([w["window_id"] for w in wt["windows"]]),
        "token_ids_hashes_sorted_sha256": sorted_sha([w["token_ids_hash"] for w in wt["windows"]]),
        "document_ids": [a["doc_id"] for a in wt["articles"]],
        "document_ids_sorted_sha256": sorted_sha(articles),
        "articles": wt["articles"],
        "windows": wt["windows"],
        "stream_tokens": wt["stream_tokens"],
        "certificate": {
            "computed_checks": {
                "wikitext_development_windows": {
                    "source": "docs/prereg/a5_manifests/text_groups.json",
                    "window_ids_sorted_sha256": text_groups["window_ids_sorted_sha256"],
                    "dev_windows_reproduced_from_cache": wt["dev_windows_reproduced"],
                    "dev_token_span": wt["dev_token_span"],
                    "dev_touched_articles": wt["dev_touched_articles"],
                    "article_overlap": wt["article_overlap_with_dev"],
                    "token_window_hash_overlap": wt["token_overlap_with_dev_windows"],
                }
            },
            "computed_disjoint": bool(
                wt["dev_windows_reproduced"] and wt["article_overlap_with_dev"] == 0 and windows_ok
            ),
            "by_construction": [SEALED_WIKITEXT],
        },
        "unreserved_articles_for_future_sealed_panels": wt["unreserved_articles"],
        "full_split_tokens": wt["full_split_tokens"],
    }
    out["checked_sets"]["wikitext_development_windows"] = {
        "n": len(text_groups["windows"]),
        "ids_sorted_sha256": text_groups["window_ids_sorted_sha256"],
        "source": "docs/prereg/a5_manifests/text_groups.json",
    }
    out["not_existing_scored_sets"] = NOT_EXISTING
    out["conventions"] = CONVENTIONS
    return out


SEALED_MMLU = (
    "Sealed MMLU-Pro items: no sealed item panel exists (PREREG §2 'Items'; A1 item 4 'not built by this "
    "amendment'; A4.1 'No sealed MMLU-Pro item is built or scored'). §2 builds it later under the distinct "
    "`TF-v1:item:` domain after removing every legacy, pilot and calibration item/group; these 500 groups "
    "(reserved_item_ids) are G2 exposure (A5.2.1) and must be removed from that pool. Disjointness from any "
    "sealed MMLU-Pro panel therefore holds by construction of the future build, not by an id check."
)
SEALED_OTHER = (
    "Sealed/calibration sets for this dataset: none registered (PREREG §2 secondary panels 'keep the "
    "development panels' (A1 item 4); A5.3.2 calibration is a development-only simulation with no scored "
    "calibration items). Any later sealed panel must exclude reserved_item_ids (A5.2.1); disjointness from it "
    "holds by construction, not by an id check."
)
SEALED_WIKITEXT = (
    "Sealed WikiText windows: no sealed window panel exists (A4.5 requires one before fresh H1w scoring; A5.6 "
    "requires a new disjoint WikiText item panel for G7; neither is built). The articles in document_ids are "
    "G2 exposure (A5.2.1) and must be excluded whole from any later sealed WikiText panel, which can draw only "
    "from unreserved_articles_for_future_sealed_panels. Disjointness holds by construction of that future "
    "build, not by an id check."
)
NOT_EXISTING = {
    "sealed_mmlu_pro_items": "not built (PREREG §2, A1 item 4, A4.1); only the component partition "
    "ebf0379a15d95b592ee8c227836406f3365a56a3ef46e6978d369c2d42c0ff3f is sealed (A5.9), which has no item ids",
    "sealed_wikitext_windows": "not built (A4.5, A5.6)",
    "sealed_hellaswag_refusal_kl_items": "none registered; secondary panels stay development (A1 item 4)",
    "calibration_items": "none registered; A5.3.2 calibration is a development-only simulation",
}
CONVENTIONS = {
    "hash_convention": "sha256 of json.dumps(sorted list of strings), default separators (as text_groups.json)",
    "rank": "sha256(domain + fingerprint) ascending as hex; ties by representative id",
    "fingerprint": "'<repo>@<revision>:<split>:<representative item id>' (PREREG §2 dataset-revision/item-ID "
    "fingerprint); WikiText uses '<repo>:<config>:<split>@<revision>:<doc_id>'",
    "item_ids": "positional row index in the named split, as TF-v1 item ids; harmful_behaviors uses "
    "'<split>:<index>' because its pool spans train and test",
    "text_groups": "union-find over the listed keys; a group is excluded if any member is in a scored set; the "
    "representative (smallest id) is the panel item and all members are reserved (one item per group, A5.8.1)",
    "normalize": "unicodedata NFKC, str.casefold, whitespace collapsed, stripped",
    "field_prefix": "MMLU-Pro and HellaSwag panels are nested rankings of 500 groups; the first 64 are the "
    "A5.8.1 field sample (64 groups per objective); all 500 are reserved as G2 exposure",
    "lengths": "A5.8.1: WikiText windows are 1,024 tokens; other prompts truncate at 1,024 tokens at G2",
}


def _rule(fp):
    return (
        f"rank eligible text groups by sha256('{FIELD_DOMAIN}' + '{fp}:' + representative id); "
        "representative = smallest positional id; take groups in rank order"
    )


def _ds(d):
    return {
        "repo": d["repo"],
        "config": None,
        "split": d["split"],
        "revision": d["revision"],
        "fingerprint": d["fingerprint"],
        "n_rows": d["n_rows"],
    }


def _prompt_ds(d, spec, splits):
    return {
        "repo": spec[0],
        "revision": spec[1],
        "splits": list(splits),
        "fingerprints": {s: d["fingerprints"][s] for s in splits},
        "n_rows": {s: len(d[s]) for s in splits},
        "column": "text",
    }


def _set(ids):
    ids = list(ids)
    return {"n": len(ids), "ids_sorted_sha256": sorted_sha(ids), "ids": sorted(ids, key=id_order)}


def load_all(cache, tok_cache):
    """Load every input from the local cache, or raise NotCached."""
    import yaml

    from wsp.eval.subsets import REVISIONS, SPECS
    from wsp.eval.subsets import ROOT as SUBSETS

    cfg = yaml.safe_load((ROOT / "configs/eval_tf_v1.yaml").read_text())
    text_groups = json.loads((ROOT / "docs/prereg/a5_manifests/text_groups.json").read_text())
    data = {"rankings": {}, "legacy": {}}
    for name in ("mmlu_pro", "hellaswag"):
        repo, config, size, _, split = SPECS[name]
        ds = load_cached(repo, config, split, REVISIONS[name], cache)
        if len(ds) != size:
            raise ValueError(f"{name}: cardinality {len(ds)} != {size}")
        cols = ("question", "question_id") if name == "mmlu_pro" else ("ctx", "source_id")
        data[name] = {c: list(ds[c]) for c in cols} | {
            "repo": repo,
            "revision": REVISIONS[name],
            "split": split,
            "fingerprint": ds._fingerprint,
            "n_rows": len(ds),
        }
        header, ids = ranking_file(name)
        if header["salt"] != cfg["tf_v1"]["item_salt"]:
            raise ValueError(f"{name}: ranking salt differs from config")
        data["rankings"][name] = ids
        data["legacy"][name] = (SUBSETS / f"{name}.txt").read_text().split()
    header, ids = ranking_file("harmless_alpaca")
    data["rankings"]["harmless_alpaca"] = ids
    for key, spec in (("harmful", HARMFUL), ("harmless", HARMLESS)):
        d = {"fingerprints": {}}
        for split in ("train", "test"):
            ds = load_cached(spec[0], None, split, spec[1], cache)
            d[split] = list(ds["text"])
            d["fingerprints"][split] = ds._fingerprint
        data[key] = d
    ds = load_cached(
        WIKITEXT[0], WIKITEXT[1], WIKITEXT[2], text_groups["source"]["revision"], cache
    )
    tok = load_tokenizer(tok_cache)

    def encode(text):
        return tok.encode(text, add_special_tokens=False)

    def encode_offsets(text):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    data["wikitext"] = build_wikitext(list(ds["text"]), encode, encode_offsets, text_groups)
    data["wikitext_fingerprint"] = ds._fingerprint
    inputs = {
        "config": file_entry(ROOT / "configs/eval_tf_v1.yaml"),
        "text_groups": file_entry(ROOT / "docs/prereg/a5_manifests/text_groups.json"),
        **{
            f"ranking_{n}": file_entry(SUBSETS / f"{n}_tf_v1.txt")
            for n in ("mmlu_pro", "hellaswag", "harmless_alpaca")
        },
        **{f"legacy_{n}": file_entry(SUBSETS / f"{n}.txt") for n in ("mmlu_pro", "hellaswag")},
        "code": file_entry(Path(__file__).resolve()),
    }
    return data, cfg, text_groups, inputs


# ------------------------------------------------------------------------------------------ control seeds


def build_control_seeds():
    import numpy as np

    children = np.random.SeedSequence(20261107).spawn(3)
    return {
        "schema": "a5-control-seeds-v1",
        "root_seed": 20261107,
        "derivation": {
            "rule": "seed(domain, key) = int.from_bytes(sha256(UTF-8 domain + key)[:8], 'big') mod 2^63 "
            "(the A5.3.3 rotation convention, applied to the A5.8.3 domains)",
            "reference_implementation": "scripts/a5_build_panels.py:domain_seed (tested); "
            "src/wsp/eval/response_intervention.py:rotation_seed for the rotation domain",
            "status": "A5.8.3 registers the two domains; the key formats below are chair-default proposals "
            "for Board A acceptance at freeze, fixed here before any field exists (no seed search after "
            "a failure, A5.8.3)",
        },
        "random_field": {
            "domain": RANDOM_FIELD_DOMAIN,
            "registered_by": "A5.8.3",
            "construction": "per tensor token, an independent Gaussian gradient rescaled to that token's intact "
            "Frobenius norm (not one global norm); zero-norm tokens stay zero with explicit flags; random "
            "covariance factors use Gaussian eigenframes with the intact spectra; every field-derived scalar "
            "is recomputed",
            "keys": {
                "gradient": "<objective>:<exact tensor name>:gradient",
                "A_frame": "<objective>:<exact tensor name>:A_frame",
                "G_frame": "<objective>:<exact tensor name>:G_frame",
            },
            "generator": "torch.Generator('cpu').manual_seed(seed); gradient = randn(out, in) in fp32 then "
            "rescaled; frames = sign-corrected QR (positive diagonal, zero -> +1) of randn(d, d), as A5.3.3",
            "objectives": [
                "wikitext_nll",
                "mmlu_gold_log_prob",
                "hellaswag_gold_char_log_likelihood",
                "refusal_margin_harmful",
                "refusal_margin_harmless",
            ],
            "test_vectors": {f"seed('{TEST_KEY}')": domain_seed(RANDOM_FIELD_DOMAIN, TEST_KEY)},
        },
        "permuted_field": {
            "domain": PERMUTED_FIELD_DOMAIN,
            "registered_by": "A5.8.3",
            "construction": "within each compatibility group (exact role string AND exact tensor shape), layer "
            "indices sorted ascending are permuted by the seeded derangement; whole gradient/factor tuples of "
            "all objectives move together; singleton groups stay fixed and are disclosed, never reshaped",
            "key": "<exact role string>:<shape as 'outxin'>",
            "derangement": "numpy.random.Generator(PCG64(seed)).permutation(n) repeated until no fixed point "
            "(scripts/a5_build_panels.py:derangement); n = 1 is fixed",
            "test_vectors": {
                f"seed('{TEST_KEY}')": domain_seed(PERMUTED_FIELD_DOMAIN, TEST_KEY),
                f"derangement(5, seed('{TEST_KEY}'))": derangement(
                    5, domain_seed(PERMUTED_FIELD_DOMAIN, TEST_KEY)
                ),
            },
            "note": "test vectors use a synthetic key and only pin the functions; the actual groups come "
            "from the frozen supported-module inventory at G2 staging and are written to the destruction-audit "
            "artefact (A5.8.3)",
        },
        "rotation": {
            "domain": ROTATION_DOMAIN,
            "rule_reference": "PREREG A5.3.3: SHA256 of '20261107:rotation:' + canonical ID, first eight bytes "
            "big-endian, mod 2^63; torch generator cloned per tensor; out x out then in x in Gaussian draws; "
            "sign-corrected QR",
            "implementation": "src/wsp/eval/response_intervention.py:rotation_seed",
            "test_vectors": {f"seed('{TEST_KEY}')": domain_seed(ROTATION_DOMAIN, TEST_KEY)},
        },
        "chart_projectors": {
            "rule_reference": "A5.8.2 (seed 20261107); src/wsp/features/response_chart.py:tensor_seed = "
            "SeedSequence([20261107, crc32(tensor name)]) first uint64",
            "destruction_controls": "A5.8.3 asks for permuted projector identities and independently seeded "
            "random frames but names no domain; not registered here (secondary/descriptive arm)",
        },
        "fisher_sampling": {
            "rule_reference": "A5.8.1 'one model-sampled target per valid token, fixed RNG'; "
            "src/wsp/features/response_field.py SEED = 20261107 (torch.Generator('cpu').manual_seed)",
        },
        "simulation": {
            "rule_reference": "A5.3.2",
            "construction": "numpy.random.SeedSequence(20261107).spawn(3) -> PCG64 generators in the order "
            "[panels, signs, resolution]",
            "implementation": "scripts/a5_simulation.py:simulate",
            "streams": [
                {
                    "name": name,
                    "spawn_key": list(child.spawn_key),
                    "first_state_uint32": [int(x) for x in child.generate_state(2)],
                }
                for name, child in zip(("panels", "signs", "resolution"), children)
            ],
            "replicates": 10000,
            "resolution_draws": 4000,
            "blocks": 32,
        },
        "bootstrap": {
            "rule_reference": "A5.3.2 / A5.4.2 / A5.6.1: PCG64 seed 20261107, 4,000 draws, linear .025/.975 "
            "quantiles",
        },
        "auxiliary_sampling": {
            "domain": FIELD_DOMAIN,
            "probe_domain": PROBE_DOMAIN,
            "panels": "panels.json",
        },
    }


# ---------------------------------------------------------------------------------------- null-size spec


def build_nullsize_spec(text_groups):
    window_ids = [w["window_id"] for w in text_groups["windows"]]
    return {
        "schema": "a5-nullsize-inputs-spec-v1",
        "status": "specification only; not run on real data; inputs exist only after G0/G1 closure (A5.3.2, A5.11)",
        "consumer": "scripts/a5_simulation.py (validate: --inputs-spec this file; run: INPUT --output SIZE.json)",
        "input_file": {
            "path": "data/runs/a5/nullsize/inputs.json",
            "format": "JSON object (or NPZ with the same JSON as scalar Unicode 'payload', no pickle)",
            "commit": "its sha256, inventory and provenance enter docs/prereg/a5_manifests/ before NULL-SIZE",
        },
        "payload_schema": {
            "required_keys": ["inventory", "donors", "window_ids", "rotation_source", "provenance"],
            "inventory_row": {
                "id": "canonical edit id (components.jsonl id)",
                "component": "components.jsonl component_id",
                "family": "components.jsonl family mapped by family_map",
                "norm": "components.jsonl effective_norm (finite for eligible rows; may be null otherwise)",
                "eligible": "components.jsonl eligibility == 'eligible' (chair ruling A5.3.1: predecessor-lost "
                "rows are ineligible)",
            },
            "donor_row": "an inventory row plus residuals [64,3] and audit [3]",
            "family_map": {"lora_sft": "LoRA", "chain": "chain"},
            "residual_shape": [64, 3],
            "residual_columns": ["edit", "rotation", "predecessor"],
            "audit_order": ["rotation", "ladder_0.25", "ladder_0.5"],
            "window_item_ids": [str(i) for i in range(64)],
        },
        "inventory": {
            "source": "docs/prereg/a5_manifests/components.jsonl rows with family in family_map (frozen "
            "selection inventory, A5.3.1); every such row is listed, eligible or not",
            "cutpoints": "frozen from this inventory by scripts/a5_simulation.py:selection_cells",
        },
        "donors": {
            "rows": "eligible inventory rows that are D3 development members with a TF-v1 primary record "
            "and a matched predecessor-reference record on file after G0/G1 closure",
            "records": "data/evals/qwen3.5-0.8b/tf_v1/<id>/records.jsonl, primary run (repeat_id null); the "
            "reference is 'base' for roots and the actual predecessor (parent_id) for chain steps; both must "
            "pass src/wsp/eval/response_intervention.py:_reference_on_file with the same scoring identity "
            "(config_hash, model_revision, source_hash, device), as references.json checks",
            "missing": "a row lacking either record is not a donor; it is listed with its reason in "
            "provenance.missing; no imputation. An empty (family, tertile) donor cell leaves calibration "
            "unresolved (A5.3.2)",
        },
        "residuals": {
            "benchmark": "wikitext_ppl",
            "field": "extra.mean_nll (nats per valid next token; each window has n_tokens 1023, so the mean "
            "over windows is token-weighted)",
            "window_order": "item_id '0'..'63' = text_groups.json windows in order; the simulation forms the "
            "32 two-window blocks (2k, 2k+1)",
            "columns": "E_edit[w], E_rotation[w], E_predecessor[w]: raw per-window values, NOT centred; the "
            "simulation centres U_x = E_x - E_predecessor (A5.3.2)",
            "nonfinite": "a missing/nonfinite window value is written as NaN (JSON NaN) and becomes an "
            "audit/nonfinite failure of that contrast in the simulation; never replaced",
        },
        "rotation_column": {
            "status": "UNRESOLVED (blocking)",
            "issue": "A5.3.2 says G1 TF-v1 development records supply (edit, rotation, predecessor) residuals, "
            "but rotation variants are first scored at G3 (A5.3.4, S31 §3 G3): no TF-v1 rotation record "
            "exists for any development row. The NULL-SIZE payload cannot be formed as specified.",
            "allowed_sources": [],
            "options_for_ruling": [
                "score development rotations at G0/G1 (priced under A5.2.2) and use their records",
                "an amended null model that uses only the edit and predecessor columns",
            ],
            "validator": "--inputs-spec reports 'blocked' while allowed_sources is empty",
        },
        "audit": {
            "rotation": "A5.3.3 outcome-free rotation audit of the donor edit (pass = true)",
            "ladder": "true unless the scaled-edit construction fails (A5.3.1 'audit-excluded')",
            "status": "computed at G0/G1 closure from metadata/bytes, never from outcomes",
        },
        "window_ids_sorted_sha256": text_groups["window_ids_sorted_sha256"],
        "window_ids_sha256_check": sorted_sha(window_ids)
        == text_groups["window_ids_sorted_sha256"],
        "provenance_per_donor": [
            "edit_record_sha256",
            "reference_id",
            "reference_record_sha256",
            "run_ids",
        ],
        "provenance_layout": "provenance = {'donors': {donor id: {edit_record_sha256, reference_id, "
        "reference_record_sha256, run_ids}}, 'missing': {inventory id: reason}}; record hashes are sha256 of "
        "the records.jsonl bytes",
        "rotation_source": "string naming the registered rotation_column source; must be in allowed_sources",
        "window_ids": "the list window_item_ids, in order",
        "resolution": {"blocks": 32, "draws": 4000, "replicates": 10000, "seed": 20261107},
    }


# ------------------------------------------------------------------------------------------------ readme


def readme_section(files, panels):
    shas = {k: sha256_bytes(v) for k, v in files.items()}
    rows = []
    for name, p in panels["panels"].items():
        cert = p["certificate"]
        rows.append(
            f"| `{name}` | {p['n_produced']} / {p['n_requested']} | "
            f"`{p.get('item_ids_sorted_sha256') or p.get('window_ids_sorted_sha256')}` | "
            f"{'yes' if cert['computed_disjoint'] else '**NO**'} | "
            f"{'yes' if cert['by_construction'] else '—'} |"
        )
    return "\n".join(
        [
            README_BEGIN,
            "## Pre-G2 companion files (A5.11): panels, control seeds, NULL-SIZE input specification",
            "",
            "Generated by `python scripts/a5_build_panels.py` from the local Hugging Face cache (offline) and the",
            "committed rankings/configs; rerunning it reproduces every byte. Inputs and their sha256 are in",
            "`panels.json:inputs`. This section supersedes the 'outstanding' wording above for panels, control",
            "seeds and the NULL-SIZE input specification only.",
            "",
            "| file | sha256 | produced from |",
            "|---|---|---|",
            (
                f"| `panels.json` | `{shas['panels.json']}` | cached datasets + `configs/eval_subsets/*` + "
                "`configs/eval_tf_v1.yaml` + `text_groups.json` + Qwen3.5-0.8B tokenizer (WikiText) |"
            ),
            (
                f"| `control_seeds.json` | `{shas['control_seeds.json']}` | A5.3.2/A5.3.3/A5.8.3 rules; "
                "test vectors generated by the reference functions |"
            ),
            f"| `nullsize_inputs.json` | `{shas['nullsize_inputs.json']}` | A5.3.2 input rule + `text_groups.json` |",
            "",
            "| panel | produced / requested | sha256 of sorted ids | computed disjoint | sealed: by construction |",
            "|---|---|---|---|---|",
            *rows,
            "",
            "Open items (not certified here):",
            "",
            "- No sealed MMLU-Pro, WikiText or other item panel exists; disjointness from future sealed panels",
            "  holds only by construction (the reserved ids/articles are G2 exposure and must be excluded).",
            "- `refusal_margin_harmful` is short: the harmful pool left after the scored prompts and the",
            "  direction-extraction prompts is smaller than 64; a conditional fill is listed, not registered.",
            "- WikiText cannot give one window per article (A5.8.1) in the test split; ruling required.",
            "- NULL-SIZE inputs are blocked: no TF-v1 rotation record exists before G3 (`rotation_column`).",
            "- Control key formats and the probe domain are chair-default proposals for Board A acceptance.",
            README_END,
            "",
        ]
    )


def merge_readme(existing: str, section: str) -> str:
    if README_BEGIN in existing:
        head, rest = existing.split(README_BEGIN, 1)
        tail = rest.split(README_END, 1)[1].lstrip("\n")
        return head.rstrip("\n") + "\n\n" + section + tail
    return existing.rstrip("\n") + "\n\n" + section


def build(cache, tok_cache):
    data, cfg, text_groups, inputs = load_all(cache, tok_cache)
    panels = build_panels(data, cfg, text_groups, inputs)
    files = {
        "panels.json": dumps(panels),
        "control_seeds.json": dumps(build_control_seeds()),
        "nullsize_inputs.json": dumps(build_nullsize_spec(text_groups)),
    }
    return files, panels


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default=ROOT / "docs/prereg/a5_manifests", type=Path)
    p.add_argument(
        "--datasets-cache", default=Path.home() / ".cache/huggingface/datasets", type=Path
    )
    p.add_argument("--tokenizer-cache", default=ROOT / "data/hf/hub", type=Path)
    p.add_argument(
        "--check", action="store_true", help="compare with the files on disk; write nothing"
    )
    a = p.parse_args(argv)
    out = a.out.resolve()
    if out.is_relative_to((ROOT / "data").resolve()) or out.is_relative_to(
        Path("/home/ubuntu/wsp-data")
    ):
        p.error("refusing to write under data/ (read-only)")
    try:
        files, panels = build(a.datasets_cache, a.tokenizer_cache)
    except NotCached as exc:
        print(
            json.dumps({"status": "stopped", "reason": "not cached", "detail": str(exc)}),
            file=sys.stderr,
        )
        return 2
    readme_path = out / "README.md"
    readme = merge_readme(
        readme_path.read_text() if readme_path.exists() else "", readme_section(files, panels)
    )
    if a.check:
        bad = [n for n, b in files.items() if not (out / n).exists() or (out / n).read_bytes() != b]
        bad += [] if readme_path.exists() and readme_path.read_text() == readme else ["README.md"]
        print("differs: " + ", ".join(bad) if bad else "identical")
        return 1 if bad else 0
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)
        print(f"{sha256_bytes(blob)}  {name}")
    readme_path.write_text(readme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
