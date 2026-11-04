"""Explicit, pinned completion adapters. No implicit network provider."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    code: str
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    model: str
    version: str

    def complete(self, prompt: str) -> Completion: ...


class ScriptedEditor:
    model, version = "scripted", "1"

    def __init__(self, cells):
        self.cells = iter(cells)

    def complete(self, prompt):
        return Completion(next(self.cells))


REGISTRY = {"scripted": ScriptedEditor}


def provider(name, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"provider unavailable: {name}")
    return REGISTRY[name](**kwargs)
