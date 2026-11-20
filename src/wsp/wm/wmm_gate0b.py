"""Study M Gate 0b (RULING-wmm-gate0, 20 Nov 2026, M1): the declared evaluation-harness redesign, one attempt.

Everything here is fixed before the run (committed with ``scripts/wmm_gate0b.py``):

* **Prompt** (:func:`prompt`): question, "Options:", "(A) …", then :data:`INSTRUCTION`, the model card's evaluation
  format (zero-shot, step by step, ending "Final Answer: (X)"), through the model's chat template (no thinking switch).
* **Extractor** (:func:`extract`), in this precedence and nothing else:
  1. ``final_answer``: the last "Final Answer:" (case-insensitive, optional markdown ``*``/``_``) immediately followed by
     a letter in parentheses "(X)" or a bare capital letter not followed by another letter;
  2. ``boxed``: the response ends with ``\\boxed{X}`` or ``\\boxed{(X)}`` (only whitespace, ``$``, ``.`` or ``*``
     after it);
  3. ``letter_line``: the last non-empty line holds only the letter ("X", "(X)", "**X**", optional final ".").
  The letter must be one of the item's options; otherwise the response is unparsed. If rule 1 matches with an invalid
  letter the response is unparsed (lower rules are not consulted). A truncated response (no stop token within the
  budget) is incorrect and is not parsed.
* **Budget:** 4,096 new tokens, greedy. Truncation is reported.
* **Items:** 400 fresh MedQA dev items, ranked by ``sha256("wmm-g0b:medqa:dev:" + text hash)`` after removing every
  item Gate 0 exposed (``data/wmm/gate0-2026-11-19/exposure.json``, by text hash and by (source, id)). Fallback (only if
  MedQA fails): 400 fresh MedMCQA dev (``validation``) single-answer items by the same rule with source ``medmcqa``.
* **Pass:** strict accuracy ≥ card − 5 (MedQA 69.1 → 64.1 %; MedMCQA 59.8 → 54.8 %) **and** truncation ≤ 5 %.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from wsp.wm import p3gen as G

SALT = "wmm-g0b"
BUDGET = 4096
N_ITEMS = 400
CARDS = {"medqa": 69.1, "medmcqa": 59.8}
MARGIN = 5.0
MAX_TRUNCATION = 0.05
INSTRUCTION = (
    "Think step by step, then end your response with the final answer in the form "
    '"Final Answer: (X)", where X is the letter of the correct option.'
)

_FA = re.compile(
    r"final\s+answer\s*:\s*[*_\s]*(?:\(\s*([A-Za-z])\s*\)|([A-Z])(?![A-Za-z]))", re.IGNORECASE
)
_BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-Za-z])\s*\)?\s*\}$")
_LINE = re.compile(r"^\(?([A-Za-z])\)?\.?$")


def prompt(item) -> str:
    opts = "\n".join(f"({G.LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return f"{item['question']}\n\nOptions:\n{opts}\n\n{INSTRUCTION}"


def _valid(letter, n_options):
    letter = letter.upper()
    return letter if letter in G.LETTERS[:n_options] else None


def extract_rule(text: str, n_options: int) -> tuple[str | None, str | None]:
    fa = list(_FA.finditer(text))
    if fa:
        m = fa[-1]
        # a bare letter must be a capital (re.IGNORECASE would otherwise let "a drug" through)
        letter = m.group(1) or (m.group(2) if m.group(2) and text[m.start(2)].isupper() else None)
        if letter is not None:
            v = _valid(letter, n_options)
            return (v, "final_answer") if v else (None, None)
    tail = text.rstrip().rstrip("$.* \n\t")
    b = _BOXED.search(tail)
    if b:
        v = _valid(b.group(1), n_options)
        return (v, "boxed") if v else (None, None)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip().strip("*_ ").strip()
        m = _LINE.match(last)
        if m and (m.group(1).isupper() or last.startswith("(")):
            v = _valid(m.group(1), n_options)
            return (v, "letter_line") if v else (None, None)
    return None, None


def extract(text: str, n_options: int) -> str | None:
    return extract_rule(text, n_options)[0]


def score(item, text: str, hit_max: bool) -> dict:
    letter, rule = (None, None) if hit_max else extract_rule(text, len(item["options"]))
    return {
        "letter": letter,
        "rule": rule,
        "correct": letter is not None and letter == G.LETTERS[item["answer_index"]],
        "truncated": bool(hit_max),
    }


# ------------------------------------------------------------------ designation


def read_exposed(path) -> dict:
    items = json.loads(Path(path).read_text())["items"]
    return {
        "hashes": {it["hash"] for it in items},
        "ids": {(it["source"], it["id"]) for it in items},
        "n": len(items),
    }


def assert_disjoint(items, exposed) -> None:
    bad = [
        it["id"]
        for it in items
        if it["hash"] in exposed["hashes"] or (it["source"], it["id"]) in exposed["ids"]
    ]
    assert not bad, f"{len(bad)} designated items were already exposed, e.g. {bad[0]}"


def rank_key(source: str, split: str, h: str) -> str:
    return hashlib.sha256(f"{SALT}:{source}:{split}:{h}".encode()).hexdigest()


def designate_fresh(items, source: str, split: str, n: int, exposed) -> list:
    seen, out = set(), []
    for it in sorted(items, key=lambda it: rank_key(source, split, it["hash"])):
        if it["hash"] in seen or it["hash"] in exposed["hashes"]:
            continue
        if (it["source"], it["id"]) in exposed["ids"]:
            continue
        seen.add(it["hash"])
        out.append(it)
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)} fresh {source}:{split} items, {n} required")
    assert_disjoint(out, exposed)
    return out


# ------------------------------------------------------------------ pass rule


def verdict(n_correct: int, n: int, n_truncated: int, card: float) -> dict:
    acc = 100.0 * n_correct / n
    trunc = n_truncated / n
    acc_ok = round(acc, 9) >= round(card - MARGIN, 9)
    tr_ok = trunc <= MAX_TRUNCATION
    return {
        "n": n,
        "accuracy": acc,
        "se": 100.0 * ((acc / 100) * (1 - acc / 100) / n) ** 0.5,
        "card": card,
        "threshold": card - MARGIN,
        "difference": acc - card,
        "truncation_rate": trunc,
        "accuracy_ok": bool(acc_ok),
        "truncation_ok": bool(tr_ok),
        "pass": bool(acc_ok and tr_ok),
    }
