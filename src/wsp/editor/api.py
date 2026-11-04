"""Supervisor-owned v1 API. No model or weight aliases cross the transport."""

import time

import torch

VERSION = "editor_api.v1"


class EditorAPI:
    def __init__(self, weights, model, limit, *, max_element=1.0, max_norm=10.0, hidden_layers=()):
        if limit not in (10, 25, 50):
            raise ValueError("budget must be 10/25/50")
        self._weights = {k: v.detach().cpu().clone() for k, v in weights.items()}
        self._model = model
        self._limit, self._used = limit, 0
        self.max_element, self.max_norm = max_element, max_norm
        self.hidden_layers = tuple(hidden_layers)
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
        return (value if slice is None else value[slice]).detach().cpu().clone()

    def forward(self, input_ids, attention_mask=None, hidden_layers=()):
        ids = torch.as_tensor(input_ids)
        if ids.ndim != 2 or not ids.numel() or ids.dtype != torch.int64 or ids.min() < 0:
            raise ValueError("nonempty rectangular integer token batch required")
        mask = None if attention_mask is None else torch.as_tensor(attention_mask)
        if mask is not None and mask.shape != ids.shape:
            raise ValueError("mask shape")
        if any(x not in self.hidden_layers for x in hidden_layers):
            raise ValueError("hidden states not enabled")
        record = {
            "op": "forward",
            "batch": len(ids),
            "tokens": ids.numel(),
            "hidden_layers": list(hidden_layers),
            "used_before": self.used,
        }
        self.calls.append(record)
        if len(ids) > self.remaining:
            record["status"] = "over_budget"
            raise ValueError("over budget")
        self._used += len(ids)
        start = time.monotonic()
        try:
            with torch.inference_mode():
                output = self._model(
                    input_ids=ids,
                    attention_mask=mask,
                    **({"output_hidden_states": True} if hidden_layers else {}),
                )
            if isinstance(output, dict):
                return output
            result = {"logits": output.logits.detach().cpu().tolist()}
            if hidden_layers:
                result["hidden_states"] = {
                    str(i): output.hidden_states[i][:, -1].cpu().tolist() for i in hidden_layers
                }
            return result
        finally:
            record.update(used=self.used, wall_seconds=time.monotonic() - start)

    def set(self, name, tensor, rationale):
        value = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu().clone()
        if name not in self._weights or value.shape != self._weights[name].shape:
            raise ValueError("exact allowlisted name and shape required")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale required")
        if (
            not torch.isfinite(value).all()
            or value.abs().max() > self.max_element
            or value.norm() > self.max_norm
        ):
            raise ValueError("delta magnitude/finite bound")
        self.deltas[name], self.rationales[name] = value, rationale

    def add(self, name, tensor, rationale):
        self.set(name, torch.as_tensor(tensor) + self.deltas.get(name, 0), rationale)

    def preview(self):
        from wsp.features.extract import effective_delta

        return {
            k: {
                "shape": list(v.shape),
                "norm": v.norm().item(),
                "sparsity": (v == 0).float().mean().item(),
                "max_abs": v.abs().max().item(),
                "rounding": effective_delta(self._weights[k].to(torch.bfloat16), v)[1],
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
