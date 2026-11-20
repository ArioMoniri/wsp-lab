"""Chat-template helpers shared by Qwen 3.5 and Gemma 3 / MedGemma (study M port, SYNTHESIS-r12 Gate 0).

* **End of turn.** Qwen ends a turn with ``<|im_end|>``; Gemma 3 with ``<end_of_turn>`` (id 106). On Gemma,
  ``<|im_end|>`` maps to ``<unk>``, so a hard-coded Qwen token silently never stops generation and puts ``<unk>`` at
  the end of LoRA targets. :func:`end_token` returns whichever of the two the tokenizer actually has.
* **Thinking switch.** Qwen's template reads ``enable_thinking``; Gemma's has no such switch and silently ignores
  the kwarg. :func:`chat_prompt` passes it only when the template reads it, and refuses ``thinking=True`` otherwise.
  For Qwen the rendered prompt is exactly as before (``enable_thinking=False`` by default).
* **Scope.** Study M calls these helpers directly. The registered WM-E call sites (``panel.chat_prompt``,
  ``p3gen.chat``, ``genaudit._generate``) are frozen and do not route through this module (chair rule, 20 Nov 2026).
"""

from __future__ import annotations

END_TOKENS = ("<|im_end|>", "<end_of_turn>")


def _known(tok, t: str) -> int | None:
    i = tok.convert_tokens_to_ids(t)
    if not isinstance(i, int) or i < 0:
        return None
    unk = getattr(tok, "unk_token_id", None)
    return None if unk is not None and i == unk else i


def end_token(tok) -> str:
    for t in END_TOKENS:
        if _known(tok, t) is not None:
            return t
    raise KeyError(f"tokenizer has none of the end-of-turn tokens {END_TOKENS}")


def end_token_id(tok) -> int:
    return _known(tok, end_token(tok))


def stop_ids(tok) -> set[int]:
    """EOS plus the template's end-of-turn token."""
    out = {end_token_id(tok)}
    eos = tok.eos_token_id
    for e in eos if isinstance(eos, list | tuple) else [eos]:
        if e is not None:
            out.add(int(e))
    return out


def supports_thinking(tok) -> bool:
    """False only when the tokenizer has a template string that never reads ``enable_thinking`` (Gemma 3); an
    absent or non-string template keeps the old behaviour (the kwarg is passed)."""
    t = getattr(tok, "chat_template", None)
    return not isinstance(t, str) or "enable_thinking" in t


def chat_prompt(tok, user: str, thinking: bool = False) -> str:
    kw = {"tokenize": False, "add_generation_prompt": True}
    if supports_thinking(tok):
        kw["enable_thinking"] = thinking
    elif thinking:
        raise ValueError("this chat template has no thinking switch")
    return tok.apply_chat_template([{"role": "user", "content": user}], **kw)
