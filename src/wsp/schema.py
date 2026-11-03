"""Shared dataclasses: edit descriptors, zoo manifests, eval records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BENCHMARKS = [
    "gsm8k",
    "math500",
    "ifeval",
    "mmlu_pro",
    "hellaswag",
    "wikitext_ppl",
    "refusal_rate",
    "over_refusal",
    "kl_to_base",
    "loop_rate",
]
EDIT_FAMILIES = [
    "random_dir",
    "abliteration",
    "heretic_ara",
    "lora_sft",
    "task_vector",
    "public_variant",
    "chain",
]


@dataclass
class EditDescriptor:
    family: str
    params: dict[str, Any]
    seed: int
    base_model: str
    base_revision: str = "main"
    parent_id: str | None = None  # previous edit in a trajectory
    git_sha: str = "unknown"
    touched_tensors: list[str] = field(default_factory=list)
    delta_fro_norm: float | None = None

    def __post_init__(self) -> None:
        if self.family not in EDIT_FAMILIES:
            raise ValueError(f"unknown edit family {self.family!r}; allowed {EDIT_FAMILIES}")

    @property
    def id(self) -> str:
        payload = json.dumps(
            {
                "f": self.family,
                "p": self.params,
                "s": self.seed,
                "b": self.base_model,
                "r": self.base_revision,
                "parent": self.parent_id,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        d = asdict(self)
        d["id"] = self.id
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> EditDescriptor:
        d = json.loads(s)
        d.pop("id", None)
        return cls(**d)


@dataclass
class EvalRecord:
    """One item of one benchmark for one zoo entry."""

    edit_id: str
    benchmark: str
    item_id: str
    correct: float  # 0/1 or a continuous score
    n_tokens: int = 0
    hit_max_tokens: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def aggregate(records: list[EvalRecord]) -> dict[str, float]:
    """Per-benchmark means computed from items (the only trusted aggregation path)."""
    out: dict[str, list[float]] = {}
    loops: dict[str, list[float]] = {}
    for r in records:
        out.setdefault(r.benchmark, []).append(r.correct)
        loops.setdefault(r.benchmark, []).append(float(r.hit_max_tokens))
    agg = {b: sum(v) / len(v) for b, v in out.items()}
    ppl = [r for r in records if r.benchmark == "wikitext_ppl"]
    if ppl:
        total = sum(r.n_tokens for r in ppl)
        if total <= 0 or any(r.correct <= 0 for r in ppl):
            raise ValueError("perplexity needs positive scores and token counts")
        agg["wikitext_ppl"] = math.exp(sum(math.log(r.correct) * r.n_tokens for r in ppl) / total)
    gen = [x for b, v in loops.items() for x in v if b in ("gsm8k", "math500", "ifeval")]
    if gen:
        agg["loop_rate"] = sum(gen) / len(gen)
    return agg


def write_manifest(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, sort_keys=True, default=str))
