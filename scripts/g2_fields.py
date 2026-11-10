#!/usr/bin/env python3
"""G2 driver (PREREG_H1_TF A5.8.1-A5.8.3; council S31 §3 G2): fields, chart projectors, probe cache, audit.

Runs only after Board A's A5 freeze. Stages, in order, each recorded in ``<out>/g2_manifest.json``:

* ``fields``: one ``wsp.features.response_field.collect_field`` call per objective (A5.8.1 table order) on
  the pinned Qwen3.5-0.8B (bf16 weights on cuda; fp32 gradients and factors, fp64 symmetrisation and
  eigendecomposition per the A5.8.1 amendment of 10 Nov 2026), K-FAC damping 1e-3 x trace/dim and 99%
  energy truncation as pinned in ``response_field``. Eigenvalues in [-1e-6 x lambda_max, 0) are clamped
  and counted; each objective's clamp statistics are recorded in its manifest entry (``eigen_clamps``). Coverage is every
  registered matrix slot of ``wsp.zoo.arch.ARCHS['qwen3_5']`` as a typed role (``full_attention:``,
  ``linear_attention:`` = GatedDeltaNet, ``mlp:``); an unregistered Linear inside a decoder layer, or a
  registered slot without a hookable Linear, is ``MissingHookError`` = G2 gate failure. One item is one
  microbatch (one sequence, or the scorer's own padded rows); peak VRAM is capped at 28 GiB through
  ``torch.cuda.set_per_process_memory_fraction`` and re-checked from ``max_memory_allocated``.
* ``projectors``: ``response_chart.block_projectors`` for every registered matrix slot at
  ``tensor_seed(<state-dict tensor name>)``, projector dimension 4 (A5.8.2 as revised 7 Nov).
* ``probe``: post-gating MLP-writer inputs (``mlp.down_proj`` input, A5.8.3) of the 64 probe prompts on
  the base model, one safetensors file per prompt.
* ``audit``: the 20 chain predecessors (A5.8.1; S31 §3 G2), exact WikiText gradient contraction at the
  predecessor state against base field plus cross term; relative error <= 0.1 with a 1e-3 nats
  denominator floor on >= 16 of 20, else ``HARD_STOP``.

Panels are rebuilt from ``docs/prereg/a5_manifests/panels.json`` with the TF-v1 scorer's tokenisation and
checked against every recorded hash (file sha256, sorted-id hashes, dataset fingerprints, WikiText window
token hashes); any mismatch refuses the run before the model is loaded. Wall-clock caps (GPU-hours on
the single exclusive GPU) stop the run cleanly between items and leave a manifest marked incomplete.

    python scripts/g2_fields.py --panels-sha256 <A5.1 hash> [--stages fields,projectors,probe,audit]
    python scripts/g2_fields.py --toy --out /tmp/g2_toy   # CPU dry run, tiny random Qwen3.5, synthetic panels

Exit codes: 0 complete, 2 HARD_STOP (audit), 3 GATE_FAILURE (missing hook), 4 incomplete (cap or
requested stages only), 5 refused (panel/code/preflight mismatch).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from wsp.features import response_chart as rc
from wsp.features import response_field as rf

SCHEMA = "g2-manifest-v1"
MANIFESTS = ROOT / "docs/prereg/a5_manifests"
DEFAULT_OUT = ROOT / "data/runs/g2"
MODEL = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
ARCHITECTURE = "qwen3_5"
MAX_LEN = 1024  # A5.8.1 auxiliary lengths
VRAM_CAP_BYTES = 28 * 2**30
PROJECTOR_DIM = 4  # A5.8.2 revision of 7 Nov 2026 (response_chart default)
FIELD_OBJECTIVES = tuple(rf.OBJECTIVES)  # A5.8.1 table order
PROBE = "probe_response"
AUDIT_OBJECTIVE = "wikitext_nll"
AUDIT_N, AUDIT_MIN_PASS, AUDIT_MAX_ERROR, AUDIT_FLOOR = 20, 16, 0.1, 1e-3  # S31 §3 G2, A5.8.1
STAGES = ("fields", "projectors", "probe", "audit")
EXIT = {"complete": 0, "HARD_STOP": 2, "GATE_FAILURE": 3, "incomplete": 4, "refused": 5}
STATUSES = ("running", "complete", "incomplete", "HARD_STOP", "GATE_FAILURE")


class PanelMismatch(ValueError):
    """A rebuilt panel differs from panels.json: the run is refused before any model work."""


class FieldAuditFailure(RuntimeError):
    """A5.8.1 finite/symmetry/PSD audit or nonfinite score: the G2 gate fails (no retry, no repair)."""

    def __init__(self, objective, cause):
        super().__init__(f"{objective}: {cause}")
        self.objective = objective


class CapExceeded(RuntimeError):
    def __init__(self, cap, where, used, limit):
        super().__init__(f"{cap} cap exceeded at {where}: {used:.4f} > {limit:.4f}")
        self.cap, self.where, self.used, self.limit = cap, where, used, limit


def now():
    """Monotonic wall clock in seconds; tests monkeypatch this to enforce caps deterministically."""
    return time.monotonic()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sorted_sha(ids) -> str:
    """panels.json convention: sha256 of json.dumps(sorted list of strings), default separators."""
    return sha256_bytes(json.dumps(sorted(ids)).encode())


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message):
    print(f"{utc()} {message}", flush=True)


# ------------------------------------------------------------------------------------- caps


class Budget:
    """Wall-clock GPU-hour cap; ``prior_hours`` carries usage of earlier attempts (resume)."""

    def __init__(self, name, hours, prior_hours=0.0):
        self.name, self.hours, self.prior = name, float(hours), float(prior_hours)
        self.start = now()

    def used(self):
        return self.prior + (now() - self.start) / 3600.0

    def check(self, where):
        used = self.used()
        if used > self.hours:
            raise CapExceeded(self.name, where, used, self.hours)

    def record(self):
        return {"cap_hours": self.hours, "prior_hours": self.prior, "used_hours": self.used()}


def capped(panel: rf.FieldPanel, budget: Budget, label: str) -> rf.FieldPanel:
    """Same panel identity; the cap is checked before every streamed item."""

    def items():
        for i, item in enumerate(panel.items()):
            budget.check(f"{label}[{i}]")
            yield item

    return rf.FieldPanel(panel.panel_id, items, panel.text_groups, panel.scored_text_groups)


# -------------------------------------------------------------------------- panel verification


def load_panels_doc(path: Path, expected_sha256: str | None):
    blob = Path(path).read_bytes()
    actual = sha256_bytes(blob)
    if expected_sha256 is not None and actual != expected_sha256:
        raise PanelMismatch(f"panels.json sha256 {actual} != expected {expected_sha256}")
    return json.loads(blob), actual


def verify_panels_doc(doc):
    """Check every recorded id hash and the registered field sizes; returns per-panel records."""
    if doc.get("schema") != "a5-panels-v2":
        raise PanelMismatch(f"panels schema {doc.get('schema')!r} is not a5-panels-v2")
    out = {}
    for name in (*FIELD_OBJECTIVES, PROBE):
        try:
            ids = rf.registered_field_items(doc, name)
        except ValueError as exc:
            raise PanelMismatch(str(exc)) from None
        panel = doc["panels"][name]
        if panel.get("short_by") != 0 or panel.get("n_produced") != panel.get("n_requested"):
            raise PanelMismatch(f"{name}: panel is short")
        if name == "wikitext_nll":
            full = panel["window_ids"]
            checks = [
                ("window_ids_sorted_sha256", full),
                ("token_ids_hashes_sorted_sha256", [w["token_ids_hash"] for w in panel["windows"]]),
                ("document_ids_sorted_sha256", panel["document_ids"]),
            ]
            if [w["window_id"] for w in panel["windows"]] != full:
                raise PanelMismatch("wikitext_nll: windows differ from window_ids")
            used = {a for w in panel["windows"] for a in w["articles"]}
            if not used <= set(panel["document_ids"]):
                raise PanelMismatch("wikitext_nll: window article outside document_ids")
            dev = panel["certificate"]["computed_checks"]["wikitext_development_windows"]
            if set(panel["document_ids"]) & set(dev["dev_touched_articles"]):
                raise PanelMismatch("wikitext_nll: article overlaps the development windows")
        else:
            full = panel["item_ids"]
            checks = [("item_ids_sorted_sha256", full)]
            if "reserved_item_ids" in panel:
                checks.append(("reserved_item_ids_sorted_sha256", panel["reserved_item_ids"]))
            if [it["item_id"] for it in panel["items"]] != full:
                raise PanelMismatch(f"{name}: items differ from item_ids")
        for key, values in checks:
            if sorted_sha(values) != panel.get(key):
                raise PanelMismatch(f"{name}: {key} does not match the recorded hash")
        out[name] = {
            "ids_sorted_sha256": sorted_sha(full),
            "n_registered": len(full),
            "field_ids": ids,
            "field_ids_sorted_sha256": sorted_sha(ids),
            "n_field": len(ids),
        }
    return out


def scored_groups(doc, name, namespace, dev_windows=()):
    """Text groups of every registered scored set the panel's certificate was computed against.

    WikiText development windows come from text_groups.json (``dev_windows``); the whole-article
    disjointness is checked separately in :func:`verify_panels_doc`.
    """
    panel = doc["panels"][name]
    groups = []
    for key in panel["certificate"]["computed_checks"]:
        if key == "wikitext_development_windows":
            groups += list(dev_windows)
        elif key == "refusal_margin_harmless_panel":
            groups += doc["panels"]["refusal_margin_harmless"]["item_ids"]
        elif "ids" in doc.get("checked_sets", {}).get(key, {}):
            groups += doc["checked_sets"][key]["ids"]
    return tuple(f"{namespace}:{g}" for g in groups)


def wikitext_windows(lines, encode, panel, articles_of):
    """Re-tokenise the registered whole-article stream and check every window's token hash."""
    found = {a["doc_id"]: a for a in articles_of(lines)}
    chosen = []
    for a in sorted(panel["articles"], key=lambda a: a["rank"]):
        b = found.get(a["doc_id"])
        if b is None or (b["line_start"], b["line_end"]) != (a["line_start"], a["line_end"]):
            raise PanelMismatch(f"wikitext_nll: article {a['doc_id']} boundaries differ")
        chosen.append(b)
    stream = encode("\n\n".join(x for c in chosen for x in lines[c["line_start"] : c["line_end"]]))
    if len(stream) != panel["stream_tokens"]:
        raise PanelMismatch("wikitext_nll: stream token count differs")
    windows = []
    for w in panel["windows"]:
        ids = stream[w["token_start"] : w["token_end"]]
        if sha256_bytes(json.dumps(ids).encode()) != w["token_ids_hash"]:
            raise PanelMismatch(f"wikitext_nll: window {w['index']} token hash differs")
        windows.append(ids)
    return windows


# ------------------------------------------------------------- items (TF-v1 scorer tokenisation)


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def mmlu_item(row, tokenizer, max_len=MAX_LEN):
    """``wsp.eval.tf_v1`` MMLU-Pro prompt and single-token letters; left-truncated, keeping 'Answer:'."""
    from wsp.eval.tf_v1 import letter_token_ids

    choices = [chr(65 + i) for i in range(len(row["options"]))]
    context = (
        row["question"]
        + "\n"
        + "\n".join(f"{c}. {o}" for c, o in zip(choices, row["options"]))
        + "\nAnswer:"
    )
    ids = _encode(tokenizer, context)
    item = {
        "input_ids": torch.tensor(ids[-max_len:]),
        "letter_ids": torch.tensor(letter_token_ids(tokenizer, choices)),
        "gold": int(row["answer_index"]),
    }
    return item, len(ids) > max_len


def hellaswag_item(row, tokenizer, max_len=MAX_LEN):
    """``tf_v1.hellaswag_scores`` rows: prefix tokens + separately encoded ' ' + ending."""
    prefix = _encode(tokenizer, row["ctx"])
    suffixes = [_encode(tokenizer, " " + e) for e in row["endings"]]
    excess = len(prefix) + max(len(s) for s in suffixes) - max_len
    if excess >= len(prefix):
        raise ValueError("HellaSwag ending alone exceeds the length cap")
    item = {
        "prefix": torch.tensor(prefix[max(excess, 0) :]),
        "suffixes": [torch.tensor(s) for s in suffixes],
        "chars": torch.tensor([len(e) for e in row["endings"]]),
        "gold": int(row["label"]),
        "pad_id": int(tokenizer.pad_token_id or 0),
    }
    return item, excess > 0


def refusal_item(text, tokenizer, chat, sets, max_len=MAX_LEN):
    """``tf_v1.batched_continuation_scores`` rows (prefix + phrase[:-1]) for every phrase.

    ``valid`` marks the scoring positions used by the Fisher pass; a position whose context
    (row prefix through that position) already appeared in an earlier row is counted once.
    """
    from wsp.eval.tf_v1 import build_continuations

    phrases = build_continuations(tokenizer, sets)
    targets = list(phrases["refusal"].values()) + list(phrases["compliance"].values())
    prefix = _encode(tokenizer, chat(text))
    excess = len(prefix) + max(len(t) for t in targets) - 1 - max_len
    if excess >= len(prefix):
        raise ValueError("refusal phrase alone exceeds the length cap")
    prefix = prefix[max(excess, 0) :]
    rows = [prefix + t[:-1] for t in targets]
    width, start = max(len(r) for r in rows), len(prefix) - 1
    valid, seen = torch.zeros(len(rows), width, dtype=torch.bool), set()
    for k, (row, t) in enumerate(zip(rows, targets)):
        for j in range(len(t)):
            context = tuple(row[: start + j + 1])
            if context not in seen:
                seen.add(context)
                valid[k, start + j] = True
    item = {
        "rows": [torch.tensor(r) for r in rows],
        "targets": [torch.tensor(t) for t in targets],
        "n_refusal": len(phrases["refusal"]),
        "start": start,
        "pad_id": int(tokenizer.pad_token_id or 0),
        "valid": valid,
    }
    return item, excess > 0


def probe_item(text, tokenizer, chat, max_len=MAX_LEN):
    ids = _encode(tokenizer, chat(text))
    return torch.tensor(ids[-max_len:]), len(ids) > max_len


# ------------------------------------------------------------------------------------ scorers


def _padded_logits(model, rows, pad_id, device):
    """Right-padded forward as ``tf_v1.padded_log_probs`` (attention mask, no cache)."""
    width = max(len(r) for r in rows)
    batch = torch.tensor([r + [pad_id] * (width - len(r)) for r in rows], device=device)
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device)
    return model(batch, attention_mask=mask, use_cache=False).logits


def score_wikitext(model, tokenizer, item, device):
    """Sum of next-token NLL over the window (nats); weight = valid next-token count."""
    ids = item["input_ids"].to(device)[None]
    logits = model(ids, use_cache=False).logits
    nll = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="sum")
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, -1] = False
    return rf.ScoreBatch(nll, float(ids.shape[1] - 1), logits, valid)


def score_mmlu(model, tokenizer, item, device):
    """Gold-letter log-prob normalised over the option letters (tf_v1.choice_record)."""
    ids = item["input_ids"].to(device)[None]
    logits = model(ids, use_cache=False).logits
    letters = logits[0, -1].float().log_softmax(-1)[item["letter_ids"].to(device)]
    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[:, -1] = True
    return rf.ScoreBatch(letters.log_softmax(0)[item["gold"]], 1.0, logits, valid)


def score_hellaswag(model, tokenizer, item, device):
    """Gold ending log-likelihood / ending characters, all endings in one padded forward."""
    prefix = item["prefix"].tolist()
    rows = [prefix + s.tolist() for s in item["suffixes"]]
    logits = _padded_logits(model, rows, item["pad_id"], device)
    gold, start = item["gold"], len(prefix) - 1
    suffix = item["suffixes"][gold].to(device)
    lp = logits[gold, start : start + len(suffix)].float().log_softmax(-1)
    score = lp.gather(1, suffix[:, None]).sum() / float(item["chars"][gold])
    valid = torch.zeros(logits.shape[:2], dtype=torch.bool, device=device)
    valid[gold, start : start + len(suffix)] = True
    return rf.ScoreBatch(score, 1.0, logits, valid)


def score_refusal(model, tokenizer, item, device):
    """logsumexp(refusal phrase log-probs) - logsumexp(compliance phrase log-probs)."""
    logits = _padded_logits(model, [r.tolist() for r in item["rows"]], item["pad_id"], device)
    start, scores = item["start"], []
    for k, t in enumerate(item["targets"]):
        lp = logits[k, start : start + len(t)].float().log_softmax(-1)
        scores.append(lp.gather(1, t.to(device)[:, None]).sum())
    s, n = torch.stack(scores), item["n_refusal"]
    return rf.ScoreBatch(
        s[:n].logsumexp(0) - s[n:].logsumexp(0), 1.0, logits, item["valid"].to(device)
    )


SCORERS = {
    "wikitext_nll": score_wikitext,
    "mmlu_gold_log_prob": score_mmlu,
    "hellaswag_gold_char_log_likelihood": score_hellaswag,
    "refusal_margin_harmful": score_refusal,
    "refusal_margin_harmless": score_refusal,
}


# ---------------------------------------------------------------------- module coverage/slots


def layer_types_of(model):
    cfg = getattr(model.config, "text_config", None) or model.config
    return tuple(cfg.layer_types)


def slot_tensors(model, architecture=ARCHITECTURE):
    """Every registered chart slot with its state-dict tensor name, shape and owning module."""
    by_slot = {}
    for name, p in model.named_parameters():
        try:
            slot = rc.tensor_slot(name)
        except ValueError:
            continue
        by_slot.setdefault(slot, []).append((name, tuple(p.shape)))
    from wsp.zoo.arch import ARCHS

    arch = ARCHS[architecture]
    types = layer_types_of(model)
    slots = []
    for layer, role, is_matrix in rc.slot_layout(architecture, types):
        hits = by_slot.get((layer, role), [])
        if len(hits) != 1:
            raise rf.MissingHookError(f"slot {(layer, role)}: {len(hits)} tensors in the model")
        name, shape = hits[0]
        group = "mlp" if role in arch.chart_roles.get("all", ()) else types[layer]
        slots.append(
            {
                "layer": layer,
                "role": role,
                "typed_role": f"{group}:{role}",
                "matrix": is_matrix,
                "tensor": name,
                "module": name.removesuffix(".weight") if is_matrix else None,
                "shape": list(shape),
            }
        )
    return slots


def supported_modules(model, slots):
    """Matrix slots -> typed roles. Unregistered layer Linears or non-Linear slots fail G2."""
    modules = dict(model.named_modules())
    roles = {}
    for s in slots:
        if s["matrix"]:
            m = modules.get(s["module"])
            if not isinstance(m, torch.nn.Linear) or len(s["shape"]) != 2:
                raise rf.MissingHookError(
                    f"{s['module']}: registered slot is not a hookable Linear"
                )
            roles[s["module"]] = s["typed_role"]
    layer = re.compile(r"model\.(?:language_model\.)?layers\.\d+\.")
    extra = [
        n
        for n, m in modules.items()
        if isinstance(m, torch.nn.Linear) and layer.match(n) and n not in roles
    ]
    if extra:
        raise rf.MissingHookError(f"Linear modules outside the slot registry: {extra[:5]}")
    return roles


def writer_modules(model, architecture=ARCHITECTURE):
    """A5.8.3 MLP writers (``mlp_out``), keyed by layer index."""
    from wsp.zoo.arch import ARCHS

    patterns = ARCHS[architecture].writers["mlp_out"]
    out = {}
    for name, m in model.named_modules():
        for pattern in patterns:
            hit = re.fullmatch(r"model\.(?:language_model\.)?layers\.(\d+)\." + pattern, name)
            if hit and isinstance(m, torch.nn.Linear):
                out[int(hit.group(1))] = name
    if len(out) != len(layer_types_of(model)):
        raise rf.MissingHookError(f"MLP writer per layer missing: found layers {sorted(out)}")
    return dict(sorted(out.items()))


def state_hash(model):
    return rf._hash({n: rf._tensor_hash(t) for n, t in model.state_dict().items()})


# ------------------------------------------------------------------------------ audit pairs


def audit_pairs(selection, components, n=AUDIT_N):
    """Chain predecessors (and their ancestors) of the registered chain picks, sorted by id.

    Each predecessor X is paired with its successor on a registered lineage (the pick itself for a
    direct predecessor); the realised increment is successor minus predecessor. Base is never a
    predecessor. Bytes come from ``bytes_via_alias`` when the entry is an alias.
    """
    rows = {r["id"]: r for r in components}
    picks = [s for s in selection["registered"]["selected"] if s["family"] == "chain"]
    edges = {}
    for pick in sorted(picks, key=lambda s: s["edit_id"]):
        child, parent = pick["edit_id"], pick["predecessor_id"]
        while parent not in (None, "base"):
            if parent not in rows or child not in rows:
                raise ValueError(f"lineage {child} -> {parent}: no component row")
            edges.setdefault(parent, set()).add((child, pick["edit_id"]))
            child, parent = parent, rows[parent]["parent_id"]

    def holder(x):
        r = rows[x]
        h = r.get("bytes_via_alias") or (x if r.get("bytes_present") else None)
        if h is None:
            raise ValueError(f"{x}: no stored or aliased bytes")
        return h

    candidates = []
    for pred in sorted(edges):
        succ, pick = min(edges[pred])
        candidates.append(
            {
                "predecessor_id": pred,
                "successor_id": succ,
                "pick_id": pick,
                "predecessor_bytes": holder(pred),
                "successor_bytes": holder(succ),
                "alternate_successors": sorted(c for c, _ in edges[pred])[1:],
            }
        )
    meta = {
        "rule": "chain picks' actual predecessors and their ancestors, sorted by canonical id, first "
        f"{n}; increment = successor on the registered lineage minus predecessor",
        "n_chain_picks": len(picks),
        "n_candidates": len(candidates),
        "n_used": min(n, len(candidates)),
        "short": len(candidates) < n,
    }
    return candidates[:n], meta


def relative_error(exact, approximate, floor=AUDIT_FLOOR):
    return abs(exact - approximate) / max(abs(exact), floor)


def audit_verdict(rows, n_required=AUDIT_N, min_pass=AUDIT_MIN_PASS, max_error=AUDIT_MAX_ERROR):
    """>= 16 passing of the audited predecessors; a row passes iff finite and error <= 0.1."""
    passed = 0
    for row in rows:
        ok = "relative_error" in row and math.isfinite(row["relative_error"])
        ok = ok and math.isfinite(row["exact"]) and math.isfinite(row["approximate"])
        row["passed"] = bool(ok and row["relative_error"] <= max_error)
        passed += row["passed"]
    return {
        "n_audited": len(rows),
        "n_required": n_required,
        "short": len(rows) < n_required,
        "n_passed": passed,
        "min_pass": min_pass,
        "max_relative_error": max_error,
        "denominator_floor_nats": AUDIT_FLOOR,
        "passed": passed >= min_pass,
    }


@contextlib.contextmanager
def memoized_spectral_factor(factors):
    """Reuse the base field's damped/truncated factors across audit rows (same pinned algebra)."""
    original, cache = rf.spectral_factor, {}
    keep = {id(t) for pair in factors.values() for t in (pair["A"], pair["G"])}

    def memo(factor):
        if id(factor) not in keep:
            return original(factor)
        if id(factor) not in cache:
            cache[id(factor)] = original(factor)
        return cache[id(factor)]

    rf.spectral_factor = memo
    try:
        yield
    finally:
        rf.spectral_factor = original


def exact_gradient(model, tokenizer, spec, panel, device, names):
    """The gradient half of ``response_field._collect_objective``, restricted to ``names``.

    Same scorer, item order, fp32 CPU accumulation and division by the total weight; no Fisher
    pass. Returns (gradients, panel content hash as collect_field records it).
    """
    params = dict(model.named_parameters())
    flags = {n: p.requires_grad for n, p in params.items()}
    training = {m: m.training for m in model.modules()}
    model.eval()
    grads, total, item_hashes = {}, 0.0, []
    try:
        for n, p in params.items():
            p.requires_grad_(n in names)
        for item in panel.items():
            item_hashes.append(rf._hash(rf._hash_item(item)))
            model.zero_grad(set_to_none=True)
            batch = spec.scorer(model, tokenizer, item, device)
            if (
                not math.isfinite(batch.weight)
                or batch.weight <= 0
                or not torch.isfinite(batch.score_sum)
            ):
                raise ValueError("nonfinite score or nonpositive objective weight")
            batch.score_sum.backward()
            total += batch.weight
            for n in names:
                if params[n].grad is not None:
                    g = params[n].grad.detach().float().cpu()
                    if n in grads:
                        grads[n] += g
                    else:
                        grads[n] = g.clone()
            del batch
        if not total:
            raise ValueError("empty objective panel")
        missing = sorted(set(names) - set(grads))
        if missing:
            raise ValueError(f"no gradient reached {missing[:3]}")
        for n, g in grads.items():
            g.div_(total)
            if not torch.isfinite(g).all():
                raise ValueError(f"nonfinite gradient: {n}")
        return grads, rf._hash(item_hashes)
    finally:
        for n, p in params.items():
            p.requires_grad_(flags[n])
        for m, mode in training.items():
            m.training = mode
        model.zero_grad(set_to_none=True)


# -------------------------------------------------------------------------------- context


@dataclass
class Objective:
    spec: rf.ObjectiveSpec
    panel: rf.FieldPanel
    record: dict


@dataclass
class Context:
    model: object
    tokenizer: object
    device: str
    objectives: list
    supported: dict
    slots: list
    writers: dict
    probe_items: list
    probe_record: dict
    pairs: list
    pairs_record: dict
    load_delta: object
    identity: dict
    toy: bool = False
    base_state_hash: str | None = None
    extra: dict = field(default_factory=dict)


def make_objective(doc, name, items, namespace, record, dev_windows=()):
    """FieldPanel (replayable) plus ObjectiveSpec for one registered objective."""
    ids = [i for i, _ in items]
    panel = rf.FieldPanel(
        name,
        lambda: iter([it for _, it in items]),
        tuple(f"{namespace}:{i}" for i in ids),
        scored_groups(doc, name, namespace, dev_windows),
    )
    panel.validate()
    spec = rf.ObjectiveSpec(name, rf.OBJECTIVES[name], SCORERS[name], name, panel.text_group_hash)
    return Objective(spec, panel, record)


def build_items(doc, records, tokenizer, chat, rows_of, wikitext, sets):
    """Item lists for every objective and the probe from verified ids; rows_of(name, id) -> row."""
    out, truncated = {}, {}
    for name in (*FIELD_OBJECTIVES, PROBE):
        ids = records[name]["field_ids"]
        items, cut = [], 0
        for i, item_id in enumerate(ids):
            if name == "wikitext_nll":
                item, t = {"input_ids": torch.tensor(wikitext[i])}, False
            elif name == "mmlu_gold_log_prob":
                item, t = mmlu_item(rows_of(name, item_id), tokenizer)
            elif name == "hellaswag_gold_char_log_likelihood":
                item, t = hellaswag_item(rows_of(name, item_id), tokenizer)
            elif name == PROBE:
                item, t = probe_item(rows_of(name, item_id), tokenizer, chat)
            else:
                item, t = refusal_item(rows_of(name, item_id), tokenizer, chat, sets)
            items.append((item_id, item))
            cut += t
        out[name], truncated[name] = items, cut
    return out, truncated


def objectives_from_items(doc, records, items, truncated, namespaces, extra_records, dev=()):
    objectives = []
    for name in FIELD_OBJECTIVES:
        rec = records[name] | extra_records.get(name, {}) | {"n_truncated": truncated[name]}
        rec["content_hash"] = rf._hash([rf._hash(rf._hash_item(it)) for _, it in items[name]])
        objectives.append(make_objective(doc, name, items[name], namespaces[name], rec, dev))
    probe = records[PROBE] | extra_records.get(PROBE, {}) | {"n_truncated": truncated[PROBE]}
    probe["content_hash"] = rf._hash([rf._tensor_hash(t) for _, t in items[PROBE]])
    return objectives, items[PROBE], probe


def use_reference_kernels():
    """Pure-torch Qwen3.5 GatedDeltaNet/conv kernels (as ``HFRunner(kernels='reference')``)."""
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        function = getattr(qwen, name)
        if isinstance(function, torch.nn.Module):
            function = inspect.getclosurevars(function.forward).nonlocals.get("func")
            if function is None:
                raise ValueError("unknown reference kernel wrapper")
        setattr(qwen, name, inspect.unwrap(function))


def build_real_context(args, doc, records):
    """Pinned model on cuda (HFRunner, the TF-v1 loading path) and the registered panels."""
    import yaml

    from scripts import a5_build_panels as bp
    from wsp.eval.run import HFRunner
    from wsp.eval.subsets import REVISIONS, SPECS
    from wsp.eval.zoo_compat import load_entry_delta

    cfg = yaml.safe_load((ROOT / "configs/eval_tf_v1.yaml").read_text())
    data, cache = {}, args.datasets_cache

    def check(ds, panel_ds, split=None):
        fp = panel_ds["fingerprints"][split] if split else panel_ds["fingerprint"]
        n = panel_ds["n_rows"][split] if split else panel_ds["n_rows"]
        if ds._fingerprint != fp or len(ds) != n:
            raise PanelMismatch(f"{panel_ds['repo']} {split or ''}: fingerprint/cardinality differ")

    for name, key in (
        ("mmlu_gold_log_prob", "mmlu_pro"),
        ("hellaswag_gold_char_log_likelihood", "hellaswag"),
    ):
        repo, config, _, _, split = SPECS[key]
        p = doc["panels"][name]["dataset"]
        if (p["repo"], p["revision"], p["split"]) != (repo, REVISIONS[key], split):
            raise PanelMismatch(f"{name}: dataset identity differs from wsp.eval.subsets")
        data[name] = bp.load_cached(repo, config, split, REVISIONS[key], cache)
        check(data[name], p)
    prompts = {}
    for name in ("refusal_margin_harmful", "refusal_margin_harmless", PROBE):
        p = doc["panels"][name]["dataset"]
        prompts[name] = {}
        for split in p["splits"]:
            ds = bp.load_cached(p["repo"], None, split, p["revision"], cache)
            check(ds, p, split)
            prompts[name][split] = list(ds[p["column"]])
    wt = doc["panels"]["wikitext_nll"]
    wds = wt["dataset"]
    ds = bp.load_cached(wds["repo"], wds["config"], wds["split"], wds["revision"], cache)
    if ds._fingerprint != wds["fingerprint"]:
        raise PanelMismatch("wikitext: dataset fingerprint differs")
    if (wds["tokenizer"]["repo"], wds["tokenizer"]["revision"]) != MODEL:
        raise PanelMismatch("wikitext: tokenizer identity differs from the pinned model")

    runner = HFRunner(MODEL[0], cfg["generation"], revision=MODEL[1], device=args.device)
    if runner.revision != MODEL[1]:
        raise PanelMismatch(f"model resolved to {runner.revision}, not {MODEL[1]}")
    model, tok = runner.model, runner.tokenizer
    if {p.dtype for p in model.parameters()} != {torch.bfloat16}:
        raise ValueError("model weights must be bf16")
    windows = wikitext_windows(
        list(ds["text"]), lambda t: _encode(tok, t), wt, bp.wikitext_articles
    )

    def rows_of(name, item_id):
        if name in data:
            return data[name][int(item_id)]
        split, _, index = item_id.rpartition(":")
        splits = doc["panels"][name]["dataset"]["splits"]
        return prompts[name][split or splits[0]][int(index)]

    items, truncated = build_items(
        doc, records, tok, runner.chat, rows_of, windows, cfg["tf_v1"]["refusal_continuations"]
    )
    namespaces = {n: doc["panels"][n]["dataset"]["repo"] for n in (*FIELD_OBJECTIVES, PROBE)}
    namespaces["wikitext_nll"] = "wikitext-window"
    fingerprints = {
        n: doc["panels"][n]["dataset"].get("fingerprint")
        or doc["panels"][n]["dataset"]["fingerprints"]
        for n in (*FIELD_OBJECTIVES, PROBE)
    }
    extra = {n: {"dataset_fingerprint": fingerprints[n]} for n in fingerprints}
    dev = json.loads((MANIFESTS / "text_groups.json").read_text())
    objectives, probe_items, probe_record = objectives_from_items(
        doc, records, items, truncated, namespaces, extra, [w["window_id"] for w in dev["windows"]]
    )
    slots = slot_tensors(model)
    selection = json.loads((MANIFESTS / "selection.json").read_text())
    comp_path = MANIFESTS / "components.jsonl"
    if sha256_file(comp_path) != selection["components_jsonl_sha256"]:
        raise PanelMismatch("components.jsonl sha256 differs from selection.json")
    ids = selection["registered"]["edit_ids"]
    if sorted_sha(ids) != selection["procedural_freeze"]["registered_edit_ids_sorted_sha256"]:
        raise PanelMismatch("registered edit ids differ from the procedural freeze hash")
    components = [json.loads(line) for line in comp_path.read_text().splitlines() if line]
    pairs, pairs_record = audit_pairs(selection, components)
    zoo = Path(args.zoo)
    for p in pairs:
        p["predecessor_dir"] = str(zoo / p["predecessor_bytes"])
        p["successor_dir"] = str(zoo / p["successor_bytes"])
    state = model.state_dict()
    identity = {
        "model": {"repo": MODEL[0], "revision": runner.revision, "dtype": "bfloat16"},
        "tokenizer": {"repo": MODEL[0], "revision": MODEL[1]},
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "generation": cfg["generation"],
        "zoo": str(zoo),
    }
    return Context(
        model,
        tok,
        args.device,
        objectives,
        supported_modules(model, slots),
        slots,
        writer_modules(model),
        probe_items,
        probe_record,
        pairs,
        pairs_record,
        lambda d: load_entry_delta(d, state),
        identity,
    )


# ---------------------------------------------------------------------------------- stages


def write_manifest(path: Path, manifest):
    manifest["updated_at"] = utc()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def _cuda(device):
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _peak(device):
    if _cuda(device):
        return {
            "max_memory_allocated": torch.cuda.max_memory_allocated(),
            "max_memory_reserved": torch.cuda.max_memory_reserved(),
        }
    return {"max_memory_allocated": None, "max_memory_reserved": None}


def _reset_peak(device):
    if _cuda(device):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _check_vram(device, where, cap=VRAM_CAP_BYTES):
    peak = _peak(device)["max_memory_allocated"]
    if peak is not None and peak > cap:
        raise CapExceeded("vram_bytes", where, peak, cap)


def _rss_bytes():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _fresh_dir(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} exists and is not empty (artefacts are immutable)")
    path.mkdir(parents=True, exist_ok=True)


def _verify_files(root: Path, files: dict):
    return all((root / k).exists() and sha256_file(root / k) == v for k, v in files.items())


def run_fields(ctx, out: Path, manifest, budget, mpath):
    fields = manifest.setdefault("fields", {})
    for obj in ctx.objectives:
        name, root = obj.spec.name, out / "fields" / obj.spec.name
        old = fields.get(name)
        if old and old.get("status") == "complete":
            if not _verify_files(root, old["files"]):
                raise ValueError(f"{name}: completed field files no longer match their hashes")
            old["reused_in_attempt"] = manifest["attempt"]
            continue
        budget.check(f"fields:{name}")
        log(f"field {name}: {obj.record['n_field']} items")
        fields[name] = {"status": "running", "started_at": utc()}
        write_manifest(mpath, manifest)
        _reset_peak(ctx.device)
        t0 = now()
        try:
            artifact = rf.collect_field(
                ctx.model,
                ctx.tokenizer,
                [obj.spec],
                {obj.spec.panel_id: capped(obj.panel, budget, f"fields:{name}")},
                ctx.supported,
                ctx.device,
            )
        except ValueError as exc:  # finite/symmetry/PSD factor audits, nonfinite score/gradient
            raise FieldAuditFailure(name, exc) from exc
        wall = now() - t0
        _check_vram(ctx.device, f"fields:{name}")
        if ctx.base_state_hash is None:
            ctx.base_state_hash = artifact.identity["state_hash"]
        if artifact.identity["panels"][name]["content_hash"] != obj.record["content_hash"]:
            raise PanelMismatch(f"{name}: streamed content differs from the verified panel")
        _fresh_dir(root)
        t1 = now()
        artifact.save(root)
        hashes = json.loads((root / "manifest.json").read_text())["content_hashes"]
        n_tokens = sum(int(p["count"]) for p in list(artifact.factors[name].values())[:1])
        fields[name] = {
            "status": "complete",
            "dir": str(root),
            "files": {f: sha256_file(root / f) for f in ("manifest.json", "tensors.pt")},
            "identity_sha256": rf._hash(artifact.identity),
            "content_hashes_sha256": rf._hash(hashes),
            "n_content_hashes": len(hashes),
            "schema_version": artifact.identity["schema_version"],
            "code_sha": artifact.identity["code_sha"],
            "implementation_sha256": artifact.identity["implementation_sha256"],
            "state_hash": artifact.identity["state_hash"],
            "panel": artifact.identity["panels"][name],
            "factor_audits": {
                m: {k: p[f"{k}_audit"] for k in ("A", "G")} | {"count": p["count"]}
                for m, p in artifact.factors[name].items()
            },
            "eigen_clamps": clamp_summary(artifact.factors[name]),
            "resources": {
                "wall_seconds": wall,
                "save_seconds": now() - t1,
                "gpu_hours": wall / 3600.0,
                **_peak(ctx.device),
                "cpu_max_rss_bytes": _rss_bytes(),
                "bytes_written": tree_bytes(root),
                "n_items": obj.record["n_field"],
                "n_fisher_positions_first_module": n_tokens,
                "items_per_second": obj.record["n_field"] / wall if wall > 0 else None,
            },
        }
        log(f"field {name}: done {wall:.1f}s, {fields[name]['resources']['bytes_written']} bytes")
        del artifact
        gc.collect()
        if _cuda(ctx.device):
            torch.cuda.empty_cache()
        manifest["caps"]["fields"] = budget.record()
        write_manifest(mpath, manifest)


def clamp_summary(factors):
    """One objective's A5.8.1 clamp statistics over all its K-FAC factors (amendment, 10 Nov 2026).

    ``most_negative_ratio`` is the minimum lambda_min / lambda_max over clamped factors (0.0 if none);
    a factor below -NEGATIVE_TOLERANCE never gets here (it raised: G2 gate failure).
    """
    audits = {f"{m}/{k}": p[f"{k}_audit"] for m, p in factors.items() for k in ("A", "G")}
    worst = min(audits, key=lambda f: audits[f]["most_negative_ratio"], default=None)
    ratio = audits[worst]["most_negative_ratio"] if worst is not None else 0.0
    return {
        "n_factors": len(audits),
        "n_factors_clamped": sum(1 for a in audits.values() if a["n_clamped"]),
        "n_eigenvalues_clamped": sum(a["n_clamped"] for a in audits.values()),
        "n_eigenvalues": sum(a["dimension"] for a in audits.values()),
        "most_negative_ratio": ratio,
        "most_negative_factor": worst if ratio < 0 else None,
        "tolerance": -rf.NEGATIVE_TOLERANCE,
    }


def run_projectors(ctx, out: Path, manifest, mpath):
    from safetensors.torch import save_file

    root = out / "projectors"
    old = manifest.get("projectors", {})
    if old.get("status") == "complete" and _verify_files(root, old["files"]):
        return
    _fresh_dir(root)
    t0 = now()
    tensors, rows = {}, []
    for s in ctx.slots:
        row = {k: s[k] for k in ("layer", "role", "typed_role", "tensor", "shape", "matrix")}
        if s["matrix"]:
            seed = rc.tensor_seed(s["tensor"])
            p_out, p_in = rc.block_projectors(*s["shape"], PROJECTOR_DIM, seed)
            for key, p in (("P_out", p_out), ("P_in", p_in)):
                eye = torch.eye(p.shape[1])
                if not torch.allclose(p.T @ p, eye, atol=1e-5):
                    raise ValueError(f"{s['tensor']}: {key} is not orthonormal")
                tensors[f"{s['tensor']}|{key}"] = p.contiguous()
            row |= {
                "seed": seed,
                "P_out_shape": list(p_out.shape),
                "P_in_shape": list(p_in.shape),
                "P_out_sha256": rf._tensor_hash(p_out),
                "P_in_sha256": rf._tensor_hash(p_in),
            }
        else:
            row["projection"] = "vector role: identity side, presence mask (A5.8.2)"
        rows.append(row)
    save_file(tensors, root / "projectors.safetensors")
    (root / "projectors.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    schema = {"architecture": ARCHITECTURE, "layer_types": list(layer_types_of(ctx.model))}
    schema["projector_dim"] = PROJECTOR_DIM
    manifest["projectors"] = {
        "status": "complete",
        "dir": str(root),
        "files": {f: sha256_file(root / f) for f in ("projectors.safetensors", "projectors.json")},
        "chart_schema_version": rc.CHART_SCHEMA_VERSION,
        "chart_schema": schema,
        "chart_schema_hash": rc.schema_hash(schema),
        "seed_rule": "response_chart.tensor_seed(state-dict tensor name)",
        "n_matrix_slots": sum(r["matrix"] for r in rows),
        "n_vector_slots": sum(not r["matrix"] for r in rows),
        "content_hash": rf._hash([r.get("P_out_sha256", r["tensor"]) for r in rows]),
        "resources": {"wall_seconds": now() - t0, "bytes_written": tree_bytes(root)},
    }
    write_manifest(mpath, manifest)


def run_probe(ctx, out: Path, manifest, budget, mpath):
    from safetensors.torch import save_file

    root = out / "probe_cache"
    old = manifest.get("probe_cache", {})
    if old.get("status") == "complete" and _verify_files(root, old["files"]):
        return
    _fresh_dir(root)
    manifest["probe_cache"] = {"status": "running", "started_at": utc()}
    write_manifest(mpath, manifest)
    captured, handles = {}, []
    for layer, name in ctx.writers.items():
        module = dict(ctx.model.named_modules())[name]
        handles.append(
            module.register_forward_hook(
                lambda m, x, y, layer=layer: captured.__setitem__(layer, x[0].detach()[0].cpu())
            )
        )
    _reset_peak(ctx.device)
    t0, rows = now(), []
    try:
        for i, (item_id, ids) in enumerate(ctx.probe_items):
            budget.check(f"probe[{i}]")
            captured.clear()
            with torch.inference_mode():
                ctx.model(ids.to(ctx.device)[None], use_cache=False, logits_to_keep=1)
            if set(captured) != set(ctx.writers):
                raise rf.MissingHookError("probe writer hook did not fire")
            tensors = {f"layer.{k}": captured[k].contiguous() for k in sorted(captured)}
            fname = f"{i:03d}.safetensors"
            save_file(tensors, root / fname, metadata={"item_id": str(item_id)})
            rows.append(
                {
                    "index": i,
                    "item_id": item_id,
                    "file": fname,
                    "n_tokens": int(ids.numel()),
                    "token_ids_sha256": sha256_bytes(json.dumps(ids.tolist()).encode()),
                    "content_hashes": {k: rf._tensor_hash(v) for k, v in tensors.items()},
                }
            )
    finally:
        for h in handles:
            h.remove()
    _check_vram(ctx.device, "probe")
    (root / "probe_cache.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    files = {r["file"]: sha256_file(root / r["file"]) for r in rows}
    files["probe_cache.json"] = sha256_file(root / "probe_cache.json")
    wall = now() - t0
    manifest["probe_cache"] = {
        "status": "complete",
        "dir": str(root),
        "files": files,
        "content_hash": rf._hash([r["content_hashes"] for r in rows]),
        "panel": ctx.probe_record,
        "predecessor": {"id": "base", "state_hash": ctx.base_state_hash},
        "hooks": {
            "role": "A5.8.3 MLP writers, post-gating input of mlp.down_proj",
            "modules": ctx.writers,
            "dtype": str(next(iter(ctx.model.parameters())).dtype),
        },
        "resources": {
            "wall_seconds": wall,
            "gpu_hours": wall / 3600.0,
            **_peak(ctx.device),
            "bytes_written": tree_bytes(root),
            "n_prompts": len(rows),
            "n_tokens": sum(r["n_tokens"] for r in rows),
        },
    }
    manifest["caps"]["fields"] = budget.record()
    write_manifest(mpath, manifest)


def _stored(base, delta):
    """Stored state of one tensor: (base + cumulative delta) rounded to the base dtype."""
    if delta is None:
        return base
    return (base.float() + delta.to(torch.float32)).to(base.dtype)


def audit_one(ctx, field_artifact, objective, pair, base, budget, impl):
    params = dict(ctx.model.named_parameters())
    d_pred, d_succ = ctx.load_delta(pair["predecessor_dir"]), ctx.load_delta(pair["successor_dir"])
    names = sorted(set(d_pred) | set(d_succ))
    for n in names:
        if n not in base:
            base[n] = params[n].detach().to("cpu", copy=True)
    w_pred = {n: _stored(base[n], d_pred.get(n)) for n in names}
    w_succ = {n: _stored(base[n], d_succ.get(n)) for n in names}
    increment = {n: w_succ[n].float() - w_pred[n].float() for n in names}
    increment = {n: d for n, d in increment.items() if torch.count_nonzero(d)}
    displacement = {n: w_pred[n].float() - base[n].float() for n in d_pred}
    displacement = {n: e for n, e in displacement.items() if torch.count_nonzero(e)}
    if not increment:
        raise ValueError("realised increment is zero")
    with torch.no_grad():
        for n in names:
            params[n].copy_(w_pred[n].to(params[n].device))
    t0 = now()
    panel = capped(objective.panel, budget, f"audit:{pair['predecessor_id']}")
    if impl == "pinned":
        row = rf.predecessor_audit(
            ctx.model,
            ctx.tokenizer,
            [objective.spec],
            {objective.spec.panel_id: panel},
            ctx.supported,
            ctx.device,
            field_artifact,
            displacement,
            increment,
        )[objective.spec.name]
        exact, approx, base_part, cross_part = row["exact"], row["approximate"], None, None
    else:
        grads, content = exact_gradient(
            ctx.model, ctx.tokenizer, objective.spec, panel, ctx.device, list(increment)
        )
        if content != field_artifact.identity["panels"][objective.spec.name]["content_hash"]:
            raise ValueError("audit panel content differs from the base field's panel")
        exact = sum((grads[n] * d).sum().item() for n, d in increment.items())
        contracted = rf.contract(field_artifact, increment)[objective.spec.name]
        cross = rf.cross_term(field_artifact, displacement, increment)[objective.spec.name]
        base_part = sum(contracted[n]["signed"] for n in increment)
        cross_part = sum(cross[n] for n in increment)
        approx = base_part + cross_part
    return {
        **{
            k: pair[k]
            for k in ("predecessor_id", "successor_id", "pick_id", "alternate_successors")
        },
        "predecessor_bytes": pair["predecessor_bytes"],
        "successor_bytes": pair["successor_bytes"],
        "exact": exact,
        "approximate": approx,
        "base_contraction": base_part,
        "cross_term": cross_part,
        "denominator": max(abs(exact), AUDIT_FLOOR),
        "relative_error": relative_error(exact, approx),
        "n_increment_tensors": len(increment),
        "n_displacement_tensors": len(displacement),
        "increment_norm": math.sqrt(sum(d.square().sum().item() for d in increment.values())),
        "displacement_norm": math.sqrt(sum(e.square().sum().item() for e in displacement.values())),
        "increment_sha256": rf._hash({n: rf._tensor_hash(d) for n, d in increment.items()}),
        "wall_seconds": now() - t0,
    }


def run_audit(ctx, out: Path, manifest, budget, mpath, impl):
    done = manifest.get("audit", {})
    if done.get("status") == "complete":  # never re-run a completed audit (no retry to a pass)
        if sha256_file(done["file"]) != done["sha256"]:
            raise ValueError("completed audit file no longer matches its hash")
        return done["verdict"]["passed"]
    old = manifest.get("fields", {}).get(AUDIT_OBJECTIVE, {})
    if old.get("status") != "complete":
        raise ValueError("audit needs the completed base WikiText field")
    objective = next(o for o in ctx.objectives if o.spec.name == AUDIT_OBJECTIVE)
    root = out / "fields" / AUDIT_OBJECTIVE
    field_artifact = rf.FieldArtifact.load(root)
    if field_artifact.identity["objectives"] != [objective.spec.identity()]:
        raise ValueError("base field objective identity differs from the audit objective")
    audit = manifest["audit"] = {
        "status": "running",
        "impl": impl,
        "selection": ctx.pairs_record,
        "objective": AUDIT_OBJECTIVE,
        "base_field_identity_sha256": rf._hash(field_artifact.identity),
        "rows": [],
    }
    write_manifest(mpath, manifest)
    base, params = {}, dict(ctx.model.named_parameters())
    _reset_peak(ctx.device)
    t0 = now()
    with memoized_spectral_factor(field_artifact.factors[AUDIT_OBJECTIVE]):
        for pair in ctx.pairs:
            budget.check(f"audit:{pair['predecessor_id']}")
            try:
                row = audit_one(ctx, field_artifact, objective, pair, base, budget, impl)
            except CapExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed row counts as not passing
                row = {
                    "predecessor_id": pair["predecessor_id"],
                    "successor_id": pair["successor_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "exact": float("nan"),
                    "approximate": float("nan"),
                    "relative_error": float("nan"),
                }
            finally:
                with torch.no_grad():
                    for n, t in base.items():
                        params[n].copy_(t.to(params[n].device))
            audit["rows"].append(row)
            log(f"audit {row['predecessor_id']}: relative error {row['relative_error']}")
            manifest["caps"]["audit"] = budget.record()
            write_manifest(mpath, manifest)
    _check_vram(ctx.device, "audit")
    verdict = audit_verdict(audit["rows"])
    payload = {
        "schema": "g2-predecessor-audit-v1",
        "rule": "relative error |exact - (base + cross)| / max(|exact|, 1e-3 nats) <= 0.1 on >= 16 of 20 "
        "(A5.8.1; S31 §3 G2); fewer is a hard stop of G2 pending amendment",
        "objective": AUDIT_OBJECTIVE,
        "impl": impl,
        "selection": ctx.pairs_record,
        "base_field_identity_sha256": audit["base_field_identity_sha256"],
        "verdict": verdict,
        "rows": audit["rows"],
    }
    path = out / "predecessor_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    wall = now() - t0
    audit |= {
        "status": "complete",
        "verdict": verdict,
        "file": str(path),
        "sha256": sha256_file(path),
        "resources": {"wall_seconds": wall, "gpu_hours": wall / 3600.0, **_peak(ctx.device)},
    }
    manifest["caps"]["audit"] = budget.record()
    write_manifest(mpath, manifest)
    return verdict["passed"]


# ------------------------------------------------------------------------------- manifest


REQUIRED_KEYS = {
    "schema": str,
    "status": str,
    "complete": bool,
    "toy": bool,
    "attempt": int,
    "attempts": list,
    "run": dict,
    "config": dict,
    "config_sha256": str,
    "model": dict,
    "panels": dict,
    "panels_json_sha256": str,
    "code": dict,
    "caps": dict,
    "stages": dict,
    "fields": dict,
}


def validate_manifest(m):
    """Schema errors of a G2 manifest (empty list = valid)."""
    errors = [
        f"{k}: missing or not {t.__name__}"
        for k, t in REQUIRED_KEYS.items()
        if not isinstance(m.get(k), t)
    ]
    if m.get("schema") != SCHEMA:
        errors.append("schema version")
    if m.get("status") not in STATUSES:
        errors.append(f"status {m.get('status')!r}")
    if m.get("complete") and m.get("status") != "complete":
        errors.append("complete flag disagrees with status")
    if m.get("status") == "HARD_STOP" and "hard_stop" not in m:
        errors.append("HARD_STOP without hard_stop record")
    if m.get("status") == "incomplete" and not m.get("incomplete_reason"):
        errors.append("incomplete without reason")
    for name, rec in (m.get("fields") or {}).items():
        if rec.get("status") == "complete":
            for k in (
                "files",
                "identity_sha256",
                "content_hashes_sha256",
                "resources",
                "panel",
                "eigen_clamps",
            ):
                if k not in rec:
                    errors.append(f"fields.{name}.{k} missing")
            for k in ("wall_seconds", "max_memory_allocated", "bytes_written"):
                if k not in rec.get("resources", {}):
                    errors.append(f"fields.{name}.resources.{k} missing")
    for k in ("fields", "audit"):
        if k not in (m.get("caps") or {}):
            errors.append(f"caps.{k} missing")
    return errors


def code_record():
    pins = json.loads((MANIFESTS / "code_hashes.json").read_text())
    drift = {}
    for path, entry in pins["files"].items():
        p = ROOT / path
        actual = sha256_file(p) if p.exists() else None
        if actual != entry["sha256"]:
            drift[path] = {"pinned": entry["sha256"], "actual": actual}
    files = [
        "scripts/g2_fields.py",
        "scripts/g2_fields.sh",
        "src/wsp/features/response_field.py",
        "src/wsp/features/response_chart.py",
        "src/wsp/eval/tf_v1.py",
        "src/wsp/eval/run.py",
        "src/wsp/eval/zoo_compat.py",
        "src/wsp/zoo/arch.py",
        "scripts/a5_build_panels.py",
    ]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, "unknown"
    import transformers

    return {
        "git_sha": sha,
        "worktree_clean": dirty == "",
        "files": {f: sha256_file(ROOT / f) if (ROOT / f).exists() else None for f in files},
        "code_hashes_json_sha256": sha256_file(MANIFESTS / "code_hashes.json"),
        "pinned_drift": drift,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
    }


def new_manifest(args, ctx, panels_sha, code, attempts, attempt):
    config = {
        "stages": list(args.stages),
        "max_gpu_hours": args.max_gpu_hours,
        "max_audit_gpu_hours": args.max_audit_gpu_hours,
        "vram_cap_bytes": VRAM_CAP_BYTES,
        "audit_impl": args.audit_impl,
        "max_len": MAX_LEN,
        "projector_dim": PROJECTOR_DIM,
        "field": {
            "schema_version": rf.SCHEMA_VERSION,
            "seed": rf.SEED,
            "damping": rf.DAMPING,
            "energy": rf.ENERGY,
            "precision": "fp32",
            "eigendecomposition": "float64",
            "negative_tolerance_relative_to_lambda_max": rf.NEGATIVE_TOLERANCE,
            "symmetry_tolerance_relative_to_max_entry": rf.SYMMETRY_TOLERANCE,
            "fisher": "model-sampled, one target per valid scoring position",
        },
        "audit": {
            "n": AUDIT_N,
            "min_pass": AUDIT_MIN_PASS,
            "max_relative_error": AUDIT_MAX_ERROR,
            "floor_nats": AUDIT_FLOOR,
        },
        "objectives": list(FIELD_OBJECTIVES),
        "microbatch": "one item per forward (one sequence, or the scorer's padded rows)",
        "limit": args.limit,
        "toy_objectives": args.toy_objectives,
        "deterministic_algorithms": "torch.use_deterministic_algorithms(True, warn_only=True)",
    }
    return {
        "schema": SCHEMA,
        "status": "running",
        "complete": False,
        "toy": bool(args.toy),
        "attempt": attempt,
        "attempts": attempts,
        "run": {"started_at": utc(), "argv": sys.argv[1:], "host_gpu": ctx.identity.get("gpu")},
        "config": config,
        "config_sha256": rf._hash(config),
        "model": ctx.identity,
        "panels_json_sha256": panels_sha,
        "panels": {
            o.spec.name: o.record | {"objective_identity": o.spec.identity()}
            for o in ctx.objectives
        }
        | {PROBE: ctx.probe_record},
        "supported_modules": {
            "n": len(ctx.supported),
            "roles": ctx.supported,
            "sha256": rf._hash(ctx.supported),
        },
        "code": code,
        "caps": {
            "fields": {"cap_hours": args.max_gpu_hours, "prior_hours": 0.0, "used_hours": 0.0},
            "audit": {"cap_hours": args.max_audit_gpu_hours, "prior_hours": 0.0, "used_hours": 0.0},
        },
        "stages": {s: "pending" for s in STAGES},
        "fields": {},
    }


def run_pipeline(ctx, out: Path, args, manifest):
    """Run the requested stages; returns the exit code and leaves the manifest on disk."""
    mpath = out / "g2_manifest.json"
    prior = {k: manifest["caps"][k].get("used_hours", 0.0) for k in ("fields", "audit")}
    budget = Budget("fields_and_cache_gpu_hours", args.max_gpu_hours, prior["fields"])
    audit_budget = None
    manifest["caps"]["fields"] = budget.record()
    write_manifest(mpath, manifest)
    status = "complete"
    try:
        for stage in STAGES:
            if stage not in args.stages:
                continue
            manifest["stages"][stage] = "running"
            log(f"stage {stage}")
            if stage == "fields":
                run_fields(ctx, out, manifest, budget, mpath)
            elif stage == "projectors":
                run_projectors(ctx, out, manifest, mpath)
            elif stage == "probe":
                run_probe(ctx, out, manifest, budget, mpath)
            else:
                audit_budget = Budget("audit_gpu_hours", args.max_audit_gpu_hours, prior["audit"])
                run_audit(ctx, out, manifest, audit_budget, mpath, args.audit_impl)
            manifest["stages"][stage] = "complete"
        audit = manifest.get("audit", {})
        pending = [s for s in STAGES if manifest["stages"][s] != "complete"]
        if audit.get("status") == "complete" and not audit["verdict"]["passed"]:
            status = "HARD_STOP"
            manifest["hard_stop"] = {
                "HARD_STOP": True,
                "reason": "predecessor audit: fewer than 16 of 20 within relative error 0.1 "
                "(A5.8.1; S31 §3 G2); G2 stops pending a new amendment",
                "verdict": audit["verdict"],
            }
        elif pending:
            status = "incomplete"
            manifest["incomplete_reason"] = {"pending_stages": pending}
    except CapExceeded as exc:
        status = "incomplete"
        manifest["incomplete_reason"] = {
            "cap": exc.cap,
            "where": exc.where,
            "used": exc.used,
            "limit": exc.limit,
            "action": "stop and reprice; no silent coverage cut (S31 §3)",
        }
    except torch.OutOfMemoryError as exc:
        status = "incomplete"
        manifest["incomplete_reason"] = {"cap": "vram_bytes", "error": str(exc)[:500]}
    except (rf.MissingHookError, FieldAuditFailure) as exc:
        status = "GATE_FAILURE"
        manifest["gate_failure"] = {"reason": f"{type(exc).__name__}: {exc}"}
        if isinstance(exc, FieldAuditFailure):
            manifest["gate_failure"]["objective"] = exc.objective
    except BaseException as exc:
        status = "incomplete"
        manifest["incomplete_reason"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        raise
    finally:
        for stage, state in manifest["stages"].items():
            if state == "running":
                manifest["stages"][stage] = "stopped"
        manifest["caps"]["fields"] = budget.record()
        if audit_budget is not None:
            manifest["caps"]["audit"] = audit_budget.record()
        manifest["status"], manifest["complete"] = status, status == "complete"
        manifest["run"]["finished_at"] = utc()
        write_manifest(mpath, manifest)
        log(f"status {status}")
    return EXIT[status]


# ------------------------------------------------------------------------------------- toy


class ToyTokenizer:
    """Deterministic character tokenizer for the CPU dry run (' x' and 'x' are distinct tokens)."""

    pad_token_id = 0
    vocab_size = 97

    def encode(self, text, add_special_tokens=False):
        return [
            3 + (ord(t[-1]) % 47) * 2 + (len(t) == 2) for t in re.findall(r" ?.", text, re.DOTALL)
        ]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


TOY_SETS = {"refusal": ["No", "Sorry"], "compliance": ["Yes", "Sure"]}


def toy_model(seed=0):
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    use_reference_kernels()
    cfg = c.Qwen3_5TextConfig(
        vocab_size=ToyTokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    return q.Qwen3_5ForCausalLM(cfg).float().eval()


def _toy_panel(name, ids, extra=None):
    items = [{"item_id": i, "rank": k, "group_members": [i]} for k, i in enumerate(ids)]
    panel = {
        "item_ids": ids,
        "item_ids_sorted_sha256": sorted_sha(ids),
        "items": items,
        "n_produced": len(ids),
        "n_requested": len(ids),
        "short_by": 0,
        "certificate": {"computed_disjoint": True, "computed_checks": {f"{name}_scored": {}}},
        "dataset": {"repo": f"toy/{name}", "fingerprint": "toy"},
    }
    return panel | (extra or {})


def toy_panels_doc(window=16, tokenizer=None):
    """Synthetic panels.json in the a5-panels-v2 shape at the registered sizes (no real ids)."""
    tok = tokenizer or ToyTokenizer()
    lines = [
        x
        for a in range(64)
        for x in ("", f" = Toy {a} = \n", "", f" toy text {a} one \n", f" toy text {a} two \n")
    ]
    doc = {"schema": "a5-panels-v2", "checked_sets": {}, "panels": {}}
    for name, n, prefix in (
        ("mmlu_gold_log_prob", 500, 64),
        ("hellaswag_gold_char_log_likelihood", 500, 64),
        ("refusal_margin_harmful", 20, None),
        ("refusal_margin_harmless", 64, None),
        (PROBE, 64, None),
    ):
        ids = [str(i) for i in range(n)]
        extra = {"field_prefix": prefix} if prefix else {}
        if name == "refusal_margin_harmful":
            extra = {"registered_size": 20}
        doc["panels"][name] = _toy_panel(name, ids, extra)
        doc["checked_sets"][f"{name}_scored"] = {"ids": [f"scored-{i}" for i in range(4)]}
    from scripts.a5_build_panels import wikitext_articles

    articles = wikitext_articles(lines)
    stream, chosen = [], []
    for a in articles:
        if len(stream) >= 64 * window:
            break
        chosen.append(a)
        stream = tok.encode(
            "\n\n".join(x for c in chosen for x in lines[c["line_start"] : c["line_end"]])
        )
    windows = []
    for i in range(64):
        ids = stream[i * window : (i + 1) * window]
        windows.append(
            {
                "window_id": f"toy-wikitext:window{window}:{i}",
                "index": i,
                "token_start": i * window,
                "token_end": (i + 1) * window,
                "token_ids_hash": sha256_bytes(json.dumps(ids).encode()),
                "articles": [chosen[0]["doc_id"]],
            }
        )
    doc_ids = [c["doc_id"] for c in chosen]
    doc["panels"]["wikitext_nll"] = {
        "window_ids": [w["window_id"] for w in windows],
        "window_ids_sorted_sha256": sorted_sha([w["window_id"] for w in windows]),
        "windows": windows,
        "token_ids_hashes_sorted_sha256": sorted_sha([w["token_ids_hash"] for w in windows]),
        "document_ids": doc_ids,
        "document_ids_sorted_sha256": sorted_sha(doc_ids),
        "articles": [
            {
                "doc_id": c["doc_id"],
                "line_start": c["line_start"],
                "line_end": c["line_end"],
                "rank": k,
            }
            for k, c in enumerate(chosen)
        ],
        "stream_tokens": len(stream),
        "n_produced": 64,
        "n_requested": 64,
        "short_by": 0,
        "certificate": {
            "computed_disjoint": True,
            "computed_checks": {
                "wikitext_development_windows": {"dev_touched_articles": ["article:x"]}
            },
        },
        "dataset": {"repo": "toy/wikitext", "fingerprint": "toy"},
    }
    return doc, lines


def toy_rows(name, item_id):
    i = int(item_id.rpartition(":")[2])
    if name == "mmlu_gold_log_prob":
        return {
            "question": f"toy question {i}?",
            "options": [f"option {k}" for k in range(4)],
            "answer_index": i % 4,
        }
    if name == "hellaswag_gold_char_log_likelihood":
        return {
            "ctx": f"toy context {i}",
            "endings": [f"end {k}{i}" for k in range(4)],
            "label": i % 4,
        }
    return f"toy prompt {name[:4]} {i}"


def toy_zoo(root: Path, model, n_chains=3, depth=8, seed=1):
    """Synthetic cumulative dense chain deltas on writers, plus selection/components rows."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    targets = [
        n
        for n, p in model.named_parameters()
        if n.endswith(("down_proj.weight", "out_proj.weight", "o_proj.weight"))
    ]
    components, selected = [], []
    for c in range(n_chains):
        cumulative, parent = {}, None
        for step in range(depth):
            eid = f"{c:x}{step:x}" + "0" * 14
            for n in targets[step % len(targets) :: 3]:
                p = dict(model.named_parameters())[n]
                cumulative[n] = cumulative.get(n, torch.zeros_like(p)) + 0.02 * torch.randn(
                    p.shape, generator=g
                )
            d = root / eid
            d.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in cumulative.items()}, d / "delta.safetensors")
            (d / "edit.json").write_text(json.dumps({"id": eid, "delta_format": "dense"}))
            components.append(
                {"id": eid, "parent_id": parent, "bytes_present": True, "bytes_via_alias": None}
            )
            parent = eid
        selected.append({"edit_id": eid, "family": "chain", "predecessor_id": components[-2]["id"]})
    return {"registered": {"selected": selected}}, components


def build_toy_context(out: Path, limit=None):
    from wsp.eval.zoo_compat import load_entry_delta

    model, tok = toy_model(), ToyTokenizer()
    doc, lines = toy_panels_doc(tokenizer=tok)
    records = verify_panels_doc(doc)
    from scripts.a5_build_panels import wikitext_articles

    windows = wikitext_windows(lines, tok.encode, doc["panels"]["wikitext_nll"], wikitext_articles)

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}])

    items, truncated = build_items(doc, records, tok, chat, toy_rows, windows, TOY_SETS)
    if limit:
        items = {k: v[:limit] for k, v in items.items()}
        for name in items:
            records[name] = records[name] | {
                "field_ids": records[name]["field_ids"][:limit],
                "n_field": min(limit, records[name]["n_field"]),
            }
    namespaces = {n: f"toy/{n}" for n in (*FIELD_OBJECTIVES, PROBE)}
    objectives, probe_items, probe_record = objectives_from_items(
        doc, records, items, truncated, namespaces, {}
    )
    slots = slot_tensors(model)
    selection, components = toy_zoo(out / "toy_zoo", model)
    pairs, pairs_record = audit_pairs(selection, components)
    for p in pairs:
        p["predecessor_dir"] = str(out / "toy_zoo" / p["predecessor_bytes"])
        p["successor_dir"] = str(out / "toy_zoo" / p["successor_bytes"])
    state = model.state_dict()
    return Context(
        model,
        tok,
        "cpu",
        objectives,
        supported_modules(model, slots),
        slots,
        writer_modules(model),
        probe_items,
        probe_record,
        pairs,
        pairs_record,
        lambda d: load_entry_delta(d, state),
        {
            "model": {"repo": "toy/qwen3_5-random", "revision": None, "dtype": "float32"},
            "device": "cpu",
            "gpu": None,
        },
        toy=True,
        extra={"panels_doc": doc},
    )


# ------------------------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--panels", type=Path, default=MANIFESTS / "panels.json")
    p.add_argument("--panels-sha256", help="sha256 of panels.json as listed in A5.1 at the freeze")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--max-gpu-hours", type=float, default=2.0, help="fields + probe cache cap")
    p.add_argument("--max-audit-gpu-hours", type=float, default=1.0, help="predecessor audit cap")
    p.add_argument("--audit-impl", choices=("gradient", "pinned"), default="gradient")
    p.add_argument("--device", default="cuda")
    p.add_argument("--zoo", default=str(ROOT / "data/zoo/qwen3.5-0.8b"))
    p.add_argument(
        "--datasets-cache", type=Path, default=Path.home() / ".cache/huggingface/datasets"
    )
    p.add_argument(
        "--resume", action="store_true", help="continue an attempt; completed stages are verified"
    )
    p.add_argument(
        "--allow-code-drift", action="store_true", help="run although pinned code differs"
    )
    p.add_argument(
        "--toy", action="store_true", help="CPU dry run: tiny random Qwen3.5, synthetic panels"
    )
    p.add_argument("--limit", type=int, default=None, help="toy only: items per panel")
    p.add_argument("--toy-objectives", default=None, help="toy only: comma list of objectives")
    a = p.parse_args(argv)
    a.stages = tuple(s for s in a.stages.split(",") if s)
    if set(a.stages) - set(STAGES):
        p.error(f"unknown stages: {sorted(set(a.stages) - set(STAGES))}")
    if a.toy:
        if a.out is None:
            p.error("--toy needs --out")
        out = a.out.resolve()
        data_roots = [(ROOT / "data").resolve(), Path("/home/ubuntu/wsp-data")]
        if any(out.is_relative_to(r) for r in data_roots):
            p.error("--toy refuses to write under data/")
        a.device = "cpu"
    else:
        if a.limit is not None or a.toy_objectives is not None:
            p.error("--limit/--toy-objectives are toy-only (no silent coverage cut)")
        if not a.panels_sha256:
            p.error("--panels-sha256 is required (the A5.1-listed hash of panels.json)")
        a.out = a.out or DEFAULT_OUT
    return a


def main(argv=None, context_builder=None):
    args = parse_args(argv)
    out = Path(args.out)
    mpath = out / "g2_manifest.json"
    attempts, attempt, previous = [], 1, None
    if mpath.exists():
        if not args.resume:
            print(
                json.dumps({"status": "refused", "reason": f"{mpath} exists; use --resume"}),
                file=sys.stderr,
            )
            return EXIT["refused"]
        previous = json.loads(mpath.read_text())
        if previous.get("status") in ("HARD_STOP", "GATE_FAILURE"):
            reason = f"previous attempt ended {previous['status']}: G2 stops pending an amendment"
            print(json.dumps({"status": "refused", "reason": reason}), file=sys.stderr)
            return EXIT["refused"]
        attempts = previous["attempts"] + [
            {
                k: previous.get(k)
                for k in (
                    "attempt",
                    "status",
                    "run",
                    "caps",
                    "stages",
                    "incomplete_reason",
                    "audit",
                )
            }
        ]
        attempt = previous["attempt"] + 1
    try:
        code = code_record()
        if code["pinned_drift"] and not args.allow_code_drift and not args.toy:
            raise PanelMismatch(
                f"pinned code differs from code_hashes.json: {sorted(code['pinned_drift'])}"
            )
        if args.toy:
            torch.manual_seed(0)
            torch.set_num_threads(min(2, torch.get_num_threads()))  # tiny ops, loaded hosts
            ctx = (context_builder or build_toy_context)(out, args.limit)
            if args.toy_objectives:
                keep = args.toy_objectives.split(",")
                ctx.objectives = [o for o in ctx.objectives if o.spec.name in keep]
            panels_sha = rf._hash(ctx.extra["panels_doc"])
        else:
            doc, panels_sha = load_panels_doc(args.panels, args.panels_sha256)
            records = verify_panels_doc(doc)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if _cuda(args.device):
                total = torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_BYTES / total))
            ctx = (context_builder or build_real_context)(args, doc, records)
    except rf.MissingHookError as exc:  # coverage fails before any field: still a G2 gate failure
        failure = {"schema": SCHEMA, "status": "GATE_FAILURE", "complete": False, "toy": args.toy}
        failure |= {"gate_failure": {"reason": f"MissingHookError: {exc}"}, "run": {"at": utc()}}
        if previous is None:
            write_manifest(mpath, failure)
        print(json.dumps(failure), file=sys.stderr)
        return EXIT["GATE_FAILURE"]
    except Exception as exc:  # noqa: BLE001 - any preflight failure refuses the run, nothing written
        print(
            json.dumps({"status": "refused", "reason": f"{type(exc).__name__}: {exc}"}),
            file=sys.stderr,
        )
        return EXIT["refused"]
    manifest = new_manifest(args, ctx, panels_sha, code, attempts, attempt)
    if previous is not None:
        if (
            previous.get("panels_json_sha256") != panels_sha
            or previous.get("config", {}).get("field") != manifest["config"]["field"]
        ):
            print(
                json.dumps({"status": "refused", "reason": "resume with different panels/config"}),
                file=sys.stderr,
            )
            return EXIT["refused"]
        for key in ("fields", "projectors", "probe_cache", "audit"):
            if key in previous:
                manifest[key] = previous[key]
        for stage, state in previous["stages"].items():
            if state == "complete":
                manifest["stages"][stage] = "complete"
        for key in ("hard_stop", "gate_failure"):
            if key in previous:
                manifest[key] = previous[key]
        manifest["caps"] = previous["caps"]
    out.mkdir(parents=True, exist_ok=True)
    return run_pipeline(ctx, out, args, manifest)


if __name__ == "__main__":
    sys.exit(main())
