"""The generation engine of E1 generation and of the generation canary (PREREG_WME §II.31, amendment v2l; chair
ruling ``docs/council/2026-11-19/RULING-wme-c1-phasel.md`` C1). Engineering configuration; no scientific content.

One recorded flag, :data:`GEN_ENGINE`, selects the engine for every Hugging Face ``generate`` call of the run (the
``hf_gen`` canary, the HF inert-identity and GPU-parity checks, E1 greedy and sampled generation):

* ``"hf_deterministic"`` (C1(a), **active**): the registered path (``genaudit.generate_greedy`` /
  ``generate_sampled``, non-thinking chat template, left padding, batch 128 in file order) run under the
  deterministic configuration :func:`deterministic`: ``torch.use_deterministic_algorithms(True)`` (not warn-only),
  ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (process-wide, set before the first CUDA call), cuDNN deterministic with
  benchmarking off, SDPA restricted to the math backend (flash, memory-efficient and cuDNN attention disabled), and
  the gated-delta-net linear attention on the transformers torch path (no ``fla`` / ``causal_conv1d`` / hub kernel
  bound; the pinned lock installs none). The settings are scoped to the generate call: teacher-forced evaluation
  is unchanged.
* ``"vllm"`` (C1(b), **inactive**, pre-declared fallback): E1 generation and the generation canary run on the
  pinned vLLM engine (``wm.c1.vllmops``, stage-W numerics) with every other setting as registered: greedy (sampled
  audit: T 0.7, top-p 1.0, the registered seeds), 128 new tokens, non-thinking, the same prompt text and the
  registered stop rule (eos, ``<|im_end|>``). The ``hf_gen`` canary becomes a second vLLM canary (``vllm_gen``):
  the 64 exposed canary items at the E1 generation settings. Activated only by :func:`select_engine` returning
  ``"vllm"`` at the Phase L re-run, recorded by a commit that sets :data:`GEN_ENGINE` before any confirmatory unit.
* ``"hf_registered"``: the v2 path without the deterministic configuration (Phase L throughput reference and the
  local engineering comparison only; never a production engine).
"""

from __future__ import annotations

import contextlib
import os

MODES = ("hf_deterministic", "vllm", "hf_registered")
PRODUCTION_MODES = ("hf_deterministic", "vllm")
GEN_ENGINE = (
    "hf_deterministic"  # §II.31 C1(a). "vllm" only through select_engine() at the Phase L re-run
)

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE = ":4096:8"
BATCH = 128
GEN_TOKENS = 128  # gen.config max_new_tokens.sycophancy_real == the hf_gen canary's 128
# the gated-delta-net functions transformers rebinds to fla / causal_conv1d / hub kernels when those are importable
LINEAR_ATTN_FUNCS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)
QWEN35_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
ATTN_IMPLS = ("sdpa", "eager")
# Phase L selection rule constants (§II.31; ruling C3): the chair's projected-total upper bound and the registered
# E1 generation volume (1,000 sealed_gen items x 2 prompts x 2 models; greedy once, sampled for 4 seeds)
PROJECTED_TOTAL_UPPER = 86.0
E1_ANSWERS = {"greedy": 4000, "sampled": 16000}


def check_mode(mode: str | None) -> str:
    mode = GEN_ENGINE if mode is None else mode
    if mode not in MODES:
        raise ValueError(f"unknown generation engine {mode!r} (one of {MODES})")
    return mode


def config(mode: str | None = None) -> dict:
    """The recorded engine configuration of ``mode`` (part of ``scripts/wme_c1_ctl.py engine_config()``, whose
    sha256 the launch attestation binds)."""
    mode = check_mode(mode)
    common = {
        "mode": mode,
        "amendment": "PREREG_WME §II.31 (v2l)",
        "greedy": True,
        "non_thinking": True,
        "max_new_tokens": GEN_TOKENS,
        "stop": ["eos", "<|im_end|>"],
        "sampled_audit": "T 0.7, top-p 1.0, gen.sampled_seeds",
    }
    if mode == "vllm":
        from wsp.wm.c1 import vllmops as VO

        return dict(
            common,
            engine="vllm (wm.c1.vllmops, child process)",
            vllm=VO.engine_config(),
            prompts="panel.chat_prompt text (non-thinking), the HF path's exact prompt strings",
            stop_token_ids="eos and <|im_end|> passed as stop_token_ids; trailing stop ids stripped",
            sampled_seed="SamplingParams(seed=<registered seed>) per request",
            canary={
                "vllm_gen": "64 exposed canary items, 128 new tokens, greedy, stop ids as above"
            },
            cublas_workspace="unset in the vLLM child (stage-W numerics)",
        )
    hf = dict(
        common,
        engine="hf generate (wm.genaudit, unchanged)",
        batch=BATCH,
        padding="left",
        order="file order; contiguous chunks of 128 (the last the remainder); one generate call per chunk",
    )
    if mode == "hf_registered":
        return dict(hf, deterministic=False)
    return dict(
        hf,
        deterministic={
            "torch.use_deterministic_algorithms": {"mode": True, "warn_only": False},
            CUBLAS_ENV: CUBLAS_WORKSPACE + " (process-wide, set before the first CUDA call)",
            "cudnn": {"deterministic": True, "benchmark": False},
            "attention": "SDPA math backend only (flash, memory-efficient and cuDNN attention off); "
            f"model attn_implementation in {list(ATTN_IMPLS)}",
            "linear_attention": "transformers torch path (no fla / causal_conv1d / kernels-hub function bound)",
            "scope": "inside each HF generate call; teacher-forced evaluation unchanged",
        },
    )


def prepare_process(mode: str | None = None) -> None:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` for an HF-deterministic process before CUDA initialises (worker start, the
    GPU-1 parity process). Raises if it is set to another value or CUDA already initialised without it."""
    mode = check_mode(mode)
    if mode != "hf_deterministic":
        return
    cur = os.environ.get(CUBLAS_ENV)
    if cur == CUBLAS_WORKSPACE:
        return
    if cur is not None:
        raise RuntimeError(
            f"{CUBLAS_ENV}={cur!r}; the deterministic engine needs {CUBLAS_WORKSPACE}"
        )
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(f"CUDA initialised before {CUBLAS_ENV} was set")
    os.environ[CUBLAS_ENV] = CUBLAS_WORKSPACE


def resolved_module(fn) -> str | None:
    """The module of the implementation a transformers kernel wrapper actually calls (a kernels-hub ``Func`` module is
    followed into its ``forward``). ``use_kernel_func_from_hub_with_fallback`` keeps the torch function's name and module (``functools.wraps``) and holds the chosen implementation
    (``fla`` / ``causal_conv1d`` / torch) in its closure as ``implementation``; that closure is followed, not
    ``__module__`` or ``__wrapped__``."""
    import inspect

    seen, stack = set(), [fn]
    while stack:
        f = stack.pop()
        if id(f) in seen or not callable(f):
            continue
        seen.add(id(f))
        if not inspect.isfunction(f):
            # a kernels-hub ``Func`` module (installed ``kernels`` package): its ``forward`` closes over the function
            stack.extend(x for x in (getattr(type(f), "forward", None),) if x is not None)
            continue
        try:
            nl = inspect.getclosurevars(f).nonlocals
        except (TypeError, ValueError):
            nl = {}
        impl = nl.get("implementation")
        if impl is not None:
            return getattr(impl, "__module__", None)
        stack.extend(v for v in nl.values() if callable(v))
    return getattr(fn, "__module__", None)


def kernel_bindings() -> dict:
    """Module of the implementation each gated-delta-net function resolves to in the loaded transformers module."""
    import importlib

    mod = importlib.import_module(QWEN35_MODULE)
    return {f: resolved_module(getattr(mod, f)) for f in LINEAR_ATTN_FUNCS}


def _attn_impl(model) -> str | None:
    cfg = getattr(model, "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl is None:
        impl = getattr(getattr(cfg, "text_config", None), "_attn_implementation", None)
    return impl


def check_model(model, bindings=None) -> dict:
    impl = _attn_impl(model)
    if impl not in ATTN_IMPLS:
        raise RuntimeError(
            f"attention implementation {impl!r}: the deterministic engine needs {ATTN_IMPLS}"
        )
    b = kernel_bindings() if bindings is None else bindings
    bad = {f: m for f, m in b.items() if m != QWEN35_MODULE}
    if bad:
        raise RuntimeError(f"non-torch linear-attention kernels bound: {bad}")
    return {"attn_implementation": impl, "linear_attention": b}


_SDP = ("flash", "mem_efficient", "math", "cudnn")
# torch's process defaults (no CUBLAS workspace setting): the clean baseline of the c1 tests (R11)
BASELINE_STATE = {
    "deterministic_algorithms": False,
    "warn_only": False,
    "cudnn_deterministic": False,
    "cudnn_benchmark": False,
    "sdp": dict.fromkeys(_SDP, True),
    CUBLAS_ENV: None,
}


def global_state() -> dict:
    """Every process-global setting the deterministic engine touches or depends on (R11, SYNTHESIS-r13 addendum item 5)."""
    import torch

    cu = torch.backends.cuda
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "sdp": {
            "flash": cu.flash_sdp_enabled(),
            "mem_efficient": cu.mem_efficient_sdp_enabled(),
            "math": cu.math_sdp_enabled(),
            "cudnn": cu.cudnn_sdp_enabled(),
        },
        CUBLAS_ENV: os.environ.get(CUBLAS_ENV),
    }


def restore_state(state: dict, env: bool = True) -> None:
    """Set every global of :func:`global_state` back to ``state`` (``env``: the CUBLAS variable too)."""
    import torch

    cu = torch.backends.cuda
    torch.use_deterministic_algorithms(
        bool(state["deterministic_algorithms"]), warn_only=bool(state["warn_only"])
    )
    torch.backends.cudnn.deterministic = bool(state["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(state["cudnn_benchmark"])
    sdp = state["sdp"]
    cu.enable_flash_sdp(sdp["flash"])
    cu.enable_mem_efficient_sdp(sdp["mem_efficient"])
    cu.enable_math_sdp(sdp["math"])
    cu.enable_cudnn_sdp(sdp["cudnn"])
    if env:
        if state[CUBLAS_ENV] is None:
            os.environ.pop(CUBLAS_ENV, None)
        else:
            os.environ[CUBLAS_ENV] = state[CUBLAS_ENV]


@contextlib.contextmanager
def deterministic(model, bindings=None):
    """C1(a): the deterministic configuration around one HF generate call. A context manager that snapshots every
    global it touches (:func:`global_state`) and restores exactly that snapshot on exit, also on an exception: the
    teacher-forced path and any other code see the global state they had before (R11, SYNTHESIS-r13 addendum item 5). The CUBLAS
    workspace variable is process-wide and is not changed here (``prepare_process``)."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if os.environ.get(CUBLAS_ENV) != CUBLAS_WORKSPACE:
        raise RuntimeError(f"{CUBLAS_ENV} must be {CUBLAS_WORKSPACE} (prepare_process at start-up)")
    check_model(model, bindings)
    prev = global_state()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    finally:
        restore_state(prev, env=False)


# ------------------------------------------------------------------ field collection (RULING-wme-c1-bank-halt addendum 4)

# T4 item 1: the field (K-FAC groups) and vocabulary-row collection runs under the registered deterministic context
# above. The D1 diagnostic found the registered-mode collection backward non-reproducible (mmlupro_gold_lp gradients,
# entering at the layer-2 linear-attention block); the T2 probe found it bitwise reproducible 3/3 under this context.
# Scope: the collection calls only (``wm.collect.collect_grouped``, ``wm.vocabfield.collect_vocab_rows``). The
# kept-factor eigendecompositions, every other unit kind and every engine keep their registered settings.
FIELD_MODE = "deterministic"


def field_config() -> dict:
    """The recorded field-collection configuration (``engine_config()["field"]``; bound by the transition's
    attestation)."""
    det = dict(config("hf_deterministic")["deterministic"])
    det["scope"] = (
        "inside wm.collect.collect_grouped (field groups) and wm.vocabfield.collect_vocab_rows (vocabulary rows) "
        "only; kept-factor eigendecompositions, every other unit kind and every engine keep their registered settings"
    )
    return {
        "mode": FIELD_MODE,
        "ruling": "docs/council/2026-11-21/RULING-wme-c1-bank-halt.md addendum 4 (T4 item 1)",
        "deterministic": det,
    }


@contextlib.contextmanager
def field_collection(model, bindings=None):
    """T4 item 1: the registered deterministic context (:func:`deterministic`, R11 restore semantics) around one field
    or vocabulary collection call."""
    if FIELD_MODE != "deterministic":
        raise RuntimeError(f"field mode {FIELD_MODE!r} is not registered")
    with deterministic(model, bindings):
        yield


def hf_generate(model, tok, prompts, max_new_tokens, batch, device, sample=None, mode=None) -> list:
    """Every HF generate call of the run. ``sample`` = None (greedy) or {temperature, top_p, seed}."""
    from wsp.wm import genaudit as GA

    mode = check_mode(mode)
    if mode == "vllm":
        raise RuntimeError(
            "generation engine 'vllm': HF generate is not used (wm.c1.genengine.vllm_block)"
        )

    def run():
        if sample is None:
            return GA.generate_greedy(
                model, tok, prompts, max_new_tokens, batch=batch, device=device
            )
        return GA.generate_sampled(
            model, tok, prompts, max_new_tokens, int(sample["seed"]), batch=batch, device=device
        )

    before = global_state()
    try:
        if mode == "hf_registered":
            return run()
        with deterministic(model):
            return run()
    finally:
        after = global_state()
        if after != before:  # R11: never leak a global setting out of a generate call
            restore_state(before)
            raise RuntimeError(f"global state changed by an HF generate call: {before} -> {after}")


# ------------------------------------------------------------------ C1(b): the vLLM blocks


def stop_ids(tok) -> list:
    out = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        out.add(im_end)
    return sorted(out)


def vllm_block(name, tok, user_prompts, max_new_tokens=GEN_TOKENS, sample=None, state=None) -> dict:
    """One ``wm.c1.vllmops`` job block at the registered generation settings: the HF path's exact prompt text
    (``panel.chat_prompt``, non-thinking), the registered stop ids, greedy unless ``sample``."""
    from wsp.wm import panel as P

    texts = [P.chat_prompt(tok, u) for u in user_prompts]
    b = {
        "name": name,
        "prompts": texts,  # for the record; vLLM receives the token ids below
        # §II.31.8 R5: the HF path's exact prompt token ids (genaudit: tok.encode(chat_prompt, add_special_tokens=False))
        "prompt_token_ids": [tok.encode(t, add_special_tokens=False) for t in texts],
        # §II.31.8 R5: parser input text = tok.decode(ids, skip_special_tokens=True) with the HF tokenizer
        "hf_text": True,
        "max_tokens": int(max_new_tokens),
        "stop_token_ids": stop_ids(tok),
        "state": dict(state or {"kind": "base"}),
    }
    if sample is not None:
        b["sample"] = {
            "temperature": float(sample["temperature"]),
            "top_p": float(sample["top_p"]),
            "seed": int(sample["seed"]),
        }
    return b


def canary_engines(mode: str | None = None) -> set:
    """The composite canary's engine keys under ``mode``."""
    mode = check_mode(mode)
    return {"tf", "vllm_gen", "vllm"} if mode == "vllm" else {"tf", "hf_gen", "vllm"}


# ------------------------------------------------------------------ Phase L: pass rule and selection (§II.31)


def gen_pass(repeat_gpu0: list, gpu0: str | None, gpu1: str | None) -> dict:
    """C1(a) pass rule: 3 of 3 runs on one GPU bitwise identical, and GPU 0 vs GPU 1 bitwise identical, on the 64
    exposed canary items (token-id sha256s)."""
    rep = [h for h in repeat_gpu0 if h]
    repeat_ok = len(repeat_gpu0) == 3 and len(rep) == 3 and len(set(rep)) == 1
    cross_ok = bool(gpu0) and gpu0 == gpu1 and (not rep or gpu0 == rep[0])
    return {"repeat_3_of_3": repeat_ok, "gpu0_eq_gpu1": cross_ok, "pass": repeat_ok and cross_ok}


def e1_gen_gpu_h(manifest: dict) -> float:
    """The registered E1 generation GPU-h (``cost.generation`` per-answer seconds x the registered answers)."""
    c = _cost_generation(manifest)
    s = E1_ANSWERS["greedy"] * float(c["greedy_s_per_answer"]) + E1_ANSWERS["sampled"] * float(
        c["sampled_s_per_answer"]
    )
    return s / 3600.0


def _cost_generation(manifest: dict) -> dict:
    def find(o):
        if isinstance(o, dict):
            if "greedy_s_per_answer" in o and "sampled_s_per_answer" in o:
                return o
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        return None

    r = find(manifest)
    if r is None:
        raise KeyError("no registered generation cost (greedy_s_per_answer) in the manifest")
    return r


def select_engine(
    passed: dict, t_det_s: float, t_reg_s: float, g_e1: float, spent_l: float, cap: float
) -> dict:
    """§II.31 selection at the Phase L re-run (outcome-blind; exposed canary items only).

    (b) ``vllm`` is selected iff (i) the C1(a) pass rule fails (``passed["pass"]`` False, or the deterministic HF
    inert-identity check fails), or (ii) the C1(a) throughput breaches the C3 cap projection:
    ``r = t_det_s / t_reg_s`` (same instance, same 64 items, batch 128), ``delta = (r - 1) * g_e1``, and
    ``PROJECTED_TOTAL_UPPER + spent_l + delta > cap`` while ``PROJECTED_TOTAL_UPPER + spent_l <= cap`` (a breach
    that exists without ``delta`` is not attributable to (a); the C3 cap rule governs it). Otherwise (a)."""
    r = float(t_det_s) / float(t_reg_s)
    delta = max(0.0, (r - 1.0) * float(g_e1))
    base = PROJECTED_TOTAL_UPPER + float(spent_l)
    breach = base <= cap < base + delta
    fail = not bool(passed.get("pass"))
    return {
        "engine": "vllm" if (fail or breach) else "hf_deterministic",
        "a_fails_pass_rule": fail,
        "throughput_ratio": r,
        "delta_e1_gpu_h": delta,
        "projection_without_delta": base,
        "cap": float(cap),
        "throughput_breach": breach,
    }
