"""Supervisor-owned v1 API. No model or weight aliases cross the transport."""

import builtins
import json
import multiprocessing
import time
from dataclasses import asdict, dataclass

import cloudpickle
import torch

VERSION = "editor_api.v1"


@dataclass
class EpisodeMeter:
    max_calls: int = 50
    max_sequences: int = 50
    max_tokens: int = 32768
    max_wall_seconds: float = 60
    calls: int = 0
    sequences: int = 0
    tokens: int = 0
    wall_seconds: float = 0
    started: float = 0

    def __post_init__(self):
        self.started = time.monotonic()

    def check(self, batch, tokens):
        if (
            self.calls + 1 > self.max_calls
            or self.sequences + batch > self.max_sequences
            or self.tokens + tokens > self.max_tokens
            or time.monotonic() - self.started >= self.max_wall_seconds
        ):
            raise ValueError("episode budget exceeded")

    def totals(self):
        return {
            "calls": self.calls,
            "sequences": self.sequences,
            "tokens": self.tokens,
            "wall_seconds": time.monotonic() - self.started,
        }


@dataclass
class ProbeResult:
    type: str
    positions: list
    logits: list | None
    token_ids: list | None
    logprobs: list | None
    hidden_states: dict


def _probe_worker(
    conn, model, ids, mask, positions, layers, top_k, projections, max_bytes, max_vocab, max_hidden
):
    try:
        loading_started = time.monotonic()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if isinstance(model, bytes):
            model = cloudpickle.loads(model)
        if isinstance(model, ModelSpec):
            model = model.load()
        device = (
            next(model.parameters(), torch.empty(0)).device
            if isinstance(model, torch.nn.Module)
            else torch.device("cpu")
        )
        ids = ids.to(device)
        mask = None if mask is None else mask.to(device)
        load_seconds = time.monotonic() - loading_started
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_started = time.monotonic()
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=mask,
                **({"output_hidden_states": True} if layers else {}),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds = time.monotonic() - forward_started
            get = lambda key: output[key] if isinstance(output, dict) else getattr(output, key)
            raw = get("logits")
            if raw.ndim != 3 or raw.shape[:2] != ids.shape or raw.shape[-1] > max_vocab:
                raise ValueError("unexpected logits shape")
            logits = raw[:, positions, :].float()
            result = ProbeResult("probe.v1", positions, None, None, None, {})
            if top_k is None:
                result.logits = logits.tolist()
            else:
                values, indices = logits.log_softmax(-1).topk(min(top_k, logits.shape[-1]))
                result.token_ids, result.logprobs = indices.tolist(), values.tolist()
            for layer in layers:
                raw_hidden = get("hidden_states")[layer]
                if raw_hidden.shape[:2] != ids.shape or raw_hidden.shape[-1] > max_hidden:
                    raise ValueError("unexpected hidden shape")
                h = raw_hidden[:, positions, :].float()
                result.hidden_states[str(layer)] = {
                    "norm": h.norm(dim=-1).tolist(),
                    "mean": h.mean(dim=-1).tolist(),
                    "projections": h[..., projections].tolist() if projections else [],
                }
            payload = json.dumps(asdict(result), allow_nan=False).encode()
            if len(payload) > max_bytes:
                raise ValueError("probe response bytes exceeded")
            conn.send(
                ("ok", payload, {"load_seconds": load_seconds, "forward_seconds": forward_seconds})
            )
    except Exception as exc:  # noqa: BLE001
        conn.send(("error", str(exc), {}))
    finally:
        conn.close()


@dataclass
class ModelSpec:
    model_id: str
    revision: str
    device: str = "cpu"
    kernels: str = "reference"

    def load(self):
        from wsp.eval.run import HFRunner

        return HFRunner(
            self.model_id, {}, revision=self.revision, device=self.device, kernels=self.kernels
        ).model


class EditorAPI:
    def __init__(
        self,
        weights,
        model,
        limit,
        *,
        max_element=1.0,
        max_norm=10.0,
        hidden_layers=(),
        max_positions=8,
        max_layers=4,
        max_bytes=8 * 1024**2,
        max_sequence=512,
        max_batch=64,
        max_vocab=200000,
        max_hidden=16384,
        call_seconds=5,
        meter=None,
        device="cpu",
        kernels="auto",
    ):
        if limit not in (10, 25, 50):
            raise ValueError("budget must be 10/25/50")
        self._weights = {k: v.detach().cpu().clone() for k, v in weights.items()}
        if device not in ("cpu", "cuda") or kernels not in ("auto", "reference"):
            raise ValueError("invalid probe device/kernels")
        self._model = model
        self._limit, self._used = limit, 0
        self.max_element, self.max_norm = max_element, max_norm
        self.hidden_layers = tuple(hidden_layers)
        self.max_positions, self.max_layers, self.max_bytes = max_positions, max_layers, max_bytes
        self.max_sequence, self.max_batch = max_sequence, max_batch
        self.max_vocab, self.max_hidden = max_vocab, max_hidden
        self.call_seconds = call_seconds
        self.meter = meter or EpisodeMeter()
        self.deadline = None
        self.deltas, self.rationales, self.calls = {}, {}, []

    @property
    def used(self):
        return self._used

    @property
    def limit(self):
        return self._limit

    @property
    def remaining(self):
        return self.limit - self.used

    def list_tensors(self):
        return {
            k: {"shape": list(v.shape), "dtype": str(v.dtype), "role": "editable"}
            for k, v in self._weights.items()
        }

    def read(self, name, slice=None):
        if name not in self._weights:
            raise ValueError("unknown tensor")
        value = self._weights[name]
        # Basic slices/narrow are views; advanced row indexing must be sized first.
        if isinstance(slice, list):
            if value.ndim == 0 or any(type(i) is not int for i in slice):
                raise ValueError("integer row indices required")
            elements = len(slice) * (value[0].numel() if value.shape[0] else 0)
        else:
            if slice is not None and type(slice) not in (int, builtins.slice):
                raise ValueError("basic slice or integer row list required")
            view = value if slice is None else value[slice]
            elements = view.numel()
        if elements * 32 + 1024 > self.max_bytes:
            raise ValueError("weight response byte cap")
        if isinstance(slice, list):
            return value[slice].detach()  # advanced indexing already owns storage
        return view.detach().clone()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        hidden_layers=(),
        positions=None,
        top_k=None,
        projections=(),
    ):
        ids = torch.as_tensor(input_ids)
        if ids.ndim != 2 or not ids.numel() or ids.dtype != torch.int64 or ids.min() < 0:
            raise ValueError("nonempty rectangular integer token batch required")
        batch, seq = ids.shape
        positions = [seq - 1] if positions is None else list(positions)
        if (
            not positions
            or len(positions) > self.max_positions
            or any(type(p) is not int or p < 0 or p >= seq for p in positions)
            or len(hidden_layers) > self.max_layers
            or any(type(x) is not int or x not in self.hidden_layers for x in hidden_layers)
            or seq > self.max_sequence
            or batch > self.max_batch
            or (top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 64))
        ):
            raise ValueError("probe positions/layers/sequence/top-k cap")
        vocab = getattr(getattr(self._model, "config", None), "vocab_size", self.max_vocab)
        hidden = getattr(getattr(self._model, "config", None), "hidden_size", self.max_hidden)
        if vocab > self.max_vocab or hidden > self.max_hidden:
            raise ValueError("model dimensions exceed configured bounds")
        if isinstance(self._model, torch.nn.Module) and any(
            t.device.type != "cpu" for t in self._model.state_dict().values()
        ):
            raise ValueError("fork probe backend requires CPU model tensors")
        if len(projections) > 16 or any(
            type(p) is not int or not 0 <= p < hidden for p in projections
        ):
            raise ValueError("projection cap")
        # Conservative JSON upper bound (finite float32 values and integer token ids).
        estimate = (
            1024
            + batch
            * len(positions)
            * (
                (vocab if top_k is None else top_k * 2)
                + len(hidden_layers) * (2 + len(projections))
            )
            * 32
        )
        if estimate > self.max_bytes:
            raise ValueError("probe response byte cap")
        mask = None if attention_mask is None else torch.as_tensor(attention_mask)
        if mask is not None and mask.shape != ids.shape:
            raise ValueError("mask shape")
        if not callable(self._model) and not isinstance(self._model, ModelSpec):
            raise ValueError("no probe configured; not launched")  # noqa: TRY004 - API validation error
        if batch > self.remaining:
            raise ValueError("over budget")
        self.meter.check(batch, ids.numel())
        timeout = min(
            self.call_seconds, self.meter.max_wall_seconds - (time.monotonic() - self.meter.started)
        )
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("probe deadline")
        multiprocessing.set_forkserver_preload(["torch"])
        parent, child = multiprocessing.get_context(
            "spawn" if isinstance(self._model, ModelSpec) else "forkserver"
        ).Pipe(duplex=False)
        proc = multiprocessing.get_context(
            "spawn" if isinstance(self._model, ModelSpec) else "forkserver"
        ).Process(
            target=_probe_worker,
            args=(
                child,
                cloudpickle.dumps(self._model),
                ids,
                mask,
                positions,
                tuple(hidden_layers),
                top_k,
                tuple(projections),
                self.max_bytes,
                vocab,
                hidden,
            ),
        )
        start = time.monotonic()
        completed, timing = False, {}
        try:
            proc.start()  # Only a successfully launched process is charged.
            child.close()
            self._used += batch
            self.meter.calls += 1
            self.meter.sequences += batch
            self.meter.tokens += ids.numel()
            if not parent.poll(timeout):
                raise TimeoutError("probe hard timeout")
            status, payload, timing = parent.recv()
            completed = True
            if status != "ok":
                raise RuntimeError(payload)
            return json.loads(payload)
        finally:
            if proc.pid is not None:
                if completed:
                    proc.join(timeout=max(0, min(1, timeout - (time.monotonic() - start))))
                if proc.is_alive():
                    proc.kill()
                proc.join()
                elapsed = time.monotonic() - start
                self.meter.wall_seconds += elapsed
                self.calls.append(
                    {
                        "op": "forward",
                        "batch": batch,
                        "tokens": ids.numel(),
                        "used": self.used,
                        "wall_seconds": elapsed,
                        **timing,
                    }
                )
            parent.close()
            child.close()

    def set(self, name, tensor, rationale):
        value = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu().clone()
        if name not in self._weights or value.shape != self._weights[name].shape:
            raise ValueError("exact allowlisted name and shape required")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 4096:
            raise ValueError("rationale required")
        if (
            not torch.isfinite(value).all()
            or value.abs().max() > self.max_element
            or value.norm() > self.max_norm
        ):
            raise ValueError("delta magnitude/finite bound")
        self.deltas[name], self.rationales[name] = value, rationale

    def add(self, name, tensor, rationale):
        value = torch.as_tensor(tensor)
        if name not in self._weights or value.shape != self._weights[name].shape:
            raise ValueError("exact allowlisted name and shape required")
        self.set(name, value + self.deltas.get(name, 0), rationale)

    def preview(self):
        from wsp.features.extract import effective_delta

        return {
            k: {
                "shape": list(v.shape),
                "norm": v.norm().item(),
                "sparsity": (v == 0).float().mean().item(),
                "max_abs": v.abs().max().item(),
                "rounding": effective_delta(self._weights[k], v)[1],
            }
            for k, v in self.deltas.items()
        }

    def dispatch(self, request):
        if request.get("version") != VERSION:
            raise ValueError("API version")
        op, args = request["op"], request.get("args", {})
        methods = {
            "weights.list_tensors": self.list_tensors,
            "weights.read": self.read,
            "probe.forward": self.forward,
            "delta.set": self.set,
            "delta.add": self.add,
            "delta.preview": self.preview,
            "budget": lambda: {"used": self.used, "limit": self.limit, "remaining": self.remaining},
        }
        result = methods[op](**args)
        return result.tolist() if isinstance(result, torch.Tensor) else result
