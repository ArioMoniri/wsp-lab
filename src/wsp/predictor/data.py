"""Lazy, ancestry-aware P4 inputs. Stored tokens describe cumulative effective deltas.

Only bounded benchmark rates are selected by default. PPL and KL require a separate
unbounded head and are deliberately excluded, never silently clipped or rescaled.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from wsp.features.pool import TYPES, classify
from wsp.predictor.splits import make_splits, trajectory_groups

RATES = {
    "gsm8k",
    "math500",
    "ifeval",
    "mmlu_pro",
    "hellaswag",
    "refusal_rate",
    "overrefusal_rate",
    "over_refusal_rate",
    "over_refusal",
    "loop_rate",
    "acc",
}


class TargetRecords(dict):
    """Metric view plus all immutable reference candidates, including old bases."""

    def __init__(self):
        super().__init__()
        self.records = {}

    def reference(self, entry, previous):
        from wsp.eval.repeats import matched_reference

        observation = self.records.get(entry, [])[-1:]
        if not observation:
            return {}
        if previous == "base" and observation[0].get("matched_base_missing") is True:
            return {}
        candidates = self.records.get(previous, [])
        run = observation[0].get("base_run_id") if previous == "base" else None
        if run:
            candidates = [c for c in candidates if c.get("run_id") == run]
        match = matched_reference(observation[0], reversed(candidates))
        return match.get("metrics", match.get("summary", match)) if match else {}


def reference_candidates(raw, entry, previous):
    observation = raw.records[entry][-1]
    candidates = raw.records.get(previous, [])
    run = observation.get("base_run_id") if previous == "base" else None
    return [c for c in candidates if not run or c.get("run_id") == run]


def inferred_runtime(raw, entry, previous):
    from wsp.eval.repeats import matched_reference

    observation = raw.records[entry][-1]
    match = matched_reference(observation, reference_candidates(raw, entry, previous))
    return bool(
        match
        and (
            observation.get("metadata_backfill_inferred") or match.get("metadata_backfill_inferred")
        )
    )


def reference_reason(raw, entry, previous, meta):
    from wsp.eval.repeats import REFERENCE_KEYS, known

    if raw.reference(entry, previous):
        return "matched-inferred" if inferred_runtime(raw, entry, previous) else "matched"
    if meta["family"] == "chain" and not meta.get("parent_id"):
        return "chain-without-ancestry"
    observation = raw.records[entry][-1]
    candidates = reference_candidates(raw, entry, previous)
    if not candidates:
        return "predecessor-unevaluated" if previous != "base" else "no-reference-on-device"
    candidates = [
        c for c in candidates if c.get("eval_device_id") == observation.get("eval_device_id")
    ]
    if not candidates:
        return "no-reference-on-device"
    for keys, reason in [
        (("item_id_hash", "item_order_hash"), "item-hash-mismatch"),
        (("batch_size",), "batch-mismatch"),
    ]:
        candidates = [
            c
            for c in candidates
            if all(known(observation.get(k)) and c.get(k) == observation[k] for k in keys)
        ]
        if not candidates:
            return reason
    if any(not known(observation.get(k)) for k in REFERENCE_KEYS):
        return "unknown-metadata"
    if observation.get("metadata_backfill_inferred"):
        return "inferred-only"
    return "runtime-config-mismatch"


def read_targets(evals):
    """Read direct files or complete batch index payloads; reject mixed eval tiers."""
    path = Path(evals)
    root = path.parent if path.is_file() else path
    index = path if path.is_file() else root / "index.jsonl"
    result, signatures = TargetRecords(), set()

    def add(entry, payload):
        if not entry or payload.get("eval_status", "complete") != "complete":
            return
        data = payload.get("metrics", payload.get("summary", payload))
        metrics = {
            k: float(v)
            for k, v in data.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
        }
        if not metrics:
            return
        signature = payload.get("config_hash")
        if signature:
            signatures.add(signature)
        if len(signatures) > 1:
            raise ValueError(
                "mixed evaluation configurations in targets; select a single tier/config index"
            )
        if entry != "base" and entry in result and result[entry] != metrics and path.is_file():
            raise ValueError(f"conflicting target records for {entry}")
        result.records.setdefault(entry, []).append(payload)
        result[entry] = metrics

    if index.exists():
        for line in index.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                add(row.get("edit_id", row.get("id")), row)
    # An explicitly selected index is authoritative, not supplemented with other tiers.
    if not path.is_file():
        files = sorted(root.glob("base*/eval*.json")) + sorted(
            root.glob("base/devices/*/*/eval.json")
        )
        files += [
            f for f in sorted(root.glob("*/eval.json")) if not f.parent.name.startswith("base")
        ]
        for file in files:
            payload = json.loads(file.read_text())
            add(payload.get("edit_id") or file.parent.name, payload)
    return result


def read_relations(zoo, snapshot=None):
    """Manifest ancestry includes sources; only cumulative chain links are successors.

    lora_vs_base is a flagged successor: optimisation uses base, but the resulting
    delta is added to the previous state and the stored snapshot is cumulative.
    """
    path = Path(zoo)
    path = path if path.is_file() else path / "manifest.jsonl"
    rows = {}
    for line in (snapshot if snapshot is not None else path.read_bytes()).decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "id" not in row:
            continue
        params = row.get("params") or {}
        edit = path.parent / row["id"] / "edit.json"
        if edit.exists():
            details = json.loads(edit.read_text())
            params = details.get("params", params)
        row = {**row, "params": params}
        row["step_semantics"] = row.get("step_semantics", params.get("step_semantics"))
        row["relation"] = (
            "successor"
            if row.get("parent_id") and row["family"] == "chain"
            else "source"
            if row.get("parent_id")
            else "base"
        )
        rows[row["id"]] = row
    return rows


def ancestor_ids(entry_id, ancestry):
    seen = set()
    while entry_id is not None:
        if entry_id in seen:
            raise ValueError("cycle in parent chain")
        seen.add(entry_id)
        entry_id = ancestry.get(entry_id)
    return seen


def clean_training(rows, train, test, metadata, ancestry, mode):
    """Purge all held-out steps, including unlabeled mixed-family ancestors/sources."""
    if mode not in {"family", "seed"}:
        return np.asarray(train, dtype=int), 0
    held = {rows[i][mode] for i in test}
    forbidden = set()
    for i, m in metadata.items():
        value = m.get(mode)
        inner = (m.get("params") or {}).get("inner_family") if mode == "family" else None
        if value in held or inner in held:
            forbidden.add(i)
    clusters = trajectory_groups(
        [{**r, "parent_id": ancestry.get(r["id"], r.get("parent_id"))} for r in rows], ancestry
    )
    held_clusters = {clusters[i] for i in test}
    kept = [
        i
        for i in train
        if clusters[i] not in held_clusters
        and not (ancestor_ids(rows[i]["id"], ancestry) & forbidden)
    ]
    return np.asarray(kept, dtype=int), len(train) - len(kept)


def descriptor(meta, width=32):
    """Stable hashed edit fields, excluding identity/seed/targets; no learned preprocessing."""
    out = torch.zeros(width)
    fields = {"family": meta["family"], **(meta.get("params") or {})}
    fields = {k: v for k, v in fields.items() if k not in {"length", "inner_seed", "seed"}}

    def flatten(values, prefix=""):
        for key, value in sorted(values.items()):
            if key in {"length", "inner_seed", "seed"}:
                continue
            key = prefix + key
            if isinstance(value, dict):
                yield from flatten(value, key + ".")
            else:
                yield key, value

    for key, value in flatten(fields):
        text = key if isinstance(value, (int, float)) else f"{key}={value}"
        h = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        val = np.sign(value) * np.log1p(abs(value)) if isinstance(value, (int, float)) else 1
        out[h % width] += float(val) * (1 if h % 2 else -1)
    return out


class FeatureDataset(Dataset):
    def __init__(
        self,
        features,
        evals,
        *,
        target="delta",
        targets=None,
        zoo=None,
        features_variant="cumulative",
    ):
        self._cache = OrderedDict()
        self.root = Path(features)
        if features_variant == "incremental":
            candidate = self.root / "incremental"
            if not (candidate / "manifest.json").exists():
                raise ValueError(
                    "incremental features unavailable: provide separately extracted features/incremental with manifest.json"
                )
            self.root = candidate
        manifest = self.root / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        zoo = zoo or self.manifest.get("zoo")
        path = Path(zoo) if zoo else None
        if path is not None and not path.is_file():
            path = path / "manifest.jsonl"
        self.manifest_snapshot = path.read_bytes() if path else b""
        self.snapshot_info = {
            "sha256": hashlib.sha256(self.manifest_snapshot).hexdigest(),
            "row_count": len(
                [line for line in self.manifest_snapshot.splitlines() if line.strip()]
            ),
        }
        self.relations = read_relations(zoo, self.manifest_snapshot) if zoo else {}
        self.metadata = {}
        for file in sorted(self.root.glob("*/features.json")):
            meta = json.loads(file.read_text())
            row = self.relations.get(meta["edit_id"])
            if row is None:
                raise ValueError(f"manifest relation missing for {meta['edit_id']}")
            if row.get("parent_id") != meta.get("parent_id"):
                raise ValueError(f"conflicting parent for {meta['edit_id']}")
            self.metadata[meta["edit_id"]] = {**meta, **row, "path": file.parent}
        self.ancestry = {i: m.get("parent_id") for i, m in self.relations.items()}
        self.successors = {
            i: m.get("parent_id") if m["relation"] == "successor" else None
            for i, m in self.relations.items()
        }
        for i in self.ancestry:
            ancestor_ids(i, self.ancestry)
        raw = read_targets(evals)
        self.names = sorted(targets or (RATES & {k for v in raw.values() for k in v}))
        self.target = target
        if target not in {"delta", "level"}:
            raise ValueError("target must be delta or level")
        self.base_scores = [raw.get("base", {}).get(n) for n in self.names]
        self.rows, values = [], []
        self.missing_predecessor_targets = 0
        self.missing_matched_reference = 0
        self.matched_reference_targets = 0
        self.targets_with_inferred_runtime = 0
        self.match_report = Counter()
        for i, m in self.metadata.items():
            if not any(n in raw.get(i, {}) for n in self.names):
                continue
            previous = self.successors[i] or "base"
            reference = raw.reference(i, previous)
            observation = raw.records[i][-1]
            reason = reference_reason(raw, i, previous, m)
            self.match_report[
                (m["family"], observation.get("eval_device_id", "unknown"), reason)
            ] += 1
            if reference and inferred_runtime(raw, i, previous):
                self.targets_with_inferred_runtime += 1
            y = []
            missing = False
            for n in self.names:
                value = raw.get(i, {}).get(n, float("nan"))
                predecessor = reference.get(n, float("nan"))
                if np.isfinite(value) and not 0 <= value <= 1:
                    raise ValueError("targets must be rates in [0,1]; PPL/KL need a separate head")
                if target == "delta" and np.isfinite(value):
                    if not np.isfinite(predecessor):
                        missing = True
                        value = float("nan")
                    elif not 0 <= predecessor <= 1:
                        raise ValueError("predecessor targets must be rates in [0,1]")
                    value -= predecessor
                y.append(value)
            if missing:
                self.missing_predecessor_targets += 1
                self.missing_matched_reference += 1
                if not any(np.isfinite(v) for v in y):
                    continue
            elif target == "delta":
                self.matched_reference_targets += 1
            self.rows.append(
                {
                    "id": i,
                    "family": m["family"],
                    "seed": m["seed"],
                    "parent_id": m.get("parent_id"),
                    "relation": m["relation"],
                    "eval_device_id": observation.get("eval_device_id", "unknown"),
                }
            )
            values.append(y)
        if not self.rows:
            raise ValueError(
                "zero entries have targets with required predecessor targets; "
                f"missing_predecessor_targets={self.missing_predecessor_targets}; "
                f"missing_matched_reference={self.missing_matched_reference}; "
                f"matched_reference_targets={self.matched_reference_targets}"
            )
        self.targets = torch.tensor(values, dtype=torch.float32)
        self.split_audit = []
        first = self.entry(self.rows[0]["id"])
        self.stats_dim = first["stats"].numel()
        self.pooled_dim = first["pooled"].numel()
        self.token_dim = first["tokens"].shape[-1]
        self.n_layers = max(m["n_layers"] for m in self.metadata.values()) + 1

    def __len__(self):
        return len(self.rows)

    def entry(self, entry_id):
        if entry_id in self._cache:
            self._cache.move_to_end(entry_id)
            return self._cache[entry_id]
        m = self.metadata[entry_id]
        f = load_file(m["path"] / "features.safetensors")
        p = load_file(m["path"] / "pooled.safetensors")
        n = f["svd"].shape[0]
        # One tensor token retains all eight rank components (no averaging away signs).
        tokens = f["svd"].reshape(n, f["svd"].shape[1] * f["svd"].shape[2])
        types, layers = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
        for t in m["tensors"]:
            kind, layer = classify(t["name"])
            types[t["row"]] = TYPES.index(kind)
            layers[t["row"]] = 0 if layer is None else layer + 1
        # An unchanged checkpoint gets one explicit empty-set token.
        if n == 0:
            tokens = torch.zeros(1, tokens.shape[-1])
            types = layers = torch.zeros(1, dtype=torch.long)
        result = {
            "tokens": tokens,
            "types": types,
            "layers": layers,
            "stats": p["stats"],
            "pooled": torch.cat([p["stats"], p["svd"]]),
            "descriptor": descriptor(m),
        }

        source = m.get("parent_id") if m["relation"] == "source" else None
        side = self.entry(source)["stats"] if source else torch.zeros_like(result["stats"])
        result["pooled"] = torch.cat([result["pooled"], side])
        self._cache[entry_id] = result
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return result

    def chain(self, entry_id):
        chain = []
        while entry_id is not None:
            if entry_id not in self.metadata:
                raise ValueError(
                    f"missing ancestor features for {entry_id}; extract the full chain"
                )
            chain.append(entry_id)
            entry_id = self.successors.get(entry_id)
        if len(chain) > 8:
            raise ValueError("trajectory exceeds maximum length 8")
        return chain[::-1]

    def __getitem__(self, index):
        ids = self.chain(self.rows[index]["id"])
        return {"entries": [self.entry(i) for i in ids], "target": self.targets[index]}

    def holdout(self, family, variant="purge"):
        if variant not in {"purge", "chain-free"}:
            raise ValueError("unknown holdout variant")
        test = np.array([i for i, r in enumerate(self.rows) if r["family"] == family], dtype=int)
        if not len(test):
            raise ValueError(f"holdout family has no labeled rows: {family}")
        candidates = [i for i, r in enumerate(self.rows) if r["family"] != family]
        train, dropped = clean_training(
            self.rows, candidates, test, self.relations, self.ancestry, "family"
        )
        before = len(train)
        if variant == "chain-free":
            train = np.array(
                [
                    i
                    for i in train
                    if not any(
                        self.relations[a]["family"] == "chain"
                        for a in ancestor_ids(self.rows[i]["id"], self.ancestry)
                    )
                ],
                dtype=int,
            )
        self.split_audit.append(
            {
                "scope": "outer",
                "mode": "holdout",
                "family": family,
                "variant": variant,
                "test_ids": [self.rows[i]["id"] for i in test],
                "dropped_ancestor_rows": dropped,
                "dropped_chain_rows": before - len(train),
            }
        )
        if not len(train):
            raise ValueError("holdout purge leaves empty training fold")
        return [(train, test)]

    def folds(self, mode, n_splits=5, indices=None, *, scope="outer"):

        indices = np.arange(len(self)) if indices is None else np.asarray(indices)
        rows = [self.rows[i] for i in indices]
        groups = (
            trajectory_groups(rows, self.ancestry)
            if mode == "trajectory"
            else [r[mode] for r in rows]
        )
        n = min(n_splits, len(set(groups)))
        if n < 2:
            raise ValueError(f"{mode} split needs at least two groups (including inner validation)")
        result = []
        for a, b in make_splits(rows, mode, n, self.ancestry):
            train, dropped = clean_training(
                self.rows, indices[a], indices[b], self.relations, self.ancestry, mode
            )
            self.split_audit.append(
                {
                    "scope": scope,
                    "mode": mode,
                    "test_ids": [rows[j]["id"] for j in b],
                    "dropped_ancestor_rows": dropped,
                }
            )
            if not len(train):
                self.split_audit[-1]["skipped_reason"] = "ancestry purge leaves empty training fold"
                logging.getLogger(__name__).warning(
                    "Skipping %s fold: ancestry purge leaves empty training fold", scope
                )
                continue
            result.append((train, indices[b]))
        return result


def collate_trajectories(items):
    b, t = len(items), max(len(x["entries"]) for x in items)
    n = max(e["tokens"].shape[0] for x in items for e in x["entries"])
    first = items[0]["entries"][0]
    batch = {
        k: torch.zeros(
            (b, t, n, v.shape[-1])
            if k == "tokens"
            else (b, t, n)
            if k in {"types", "layers"}
            else (b, t, v.numel()),
            dtype=v.dtype,
        )
        for k, v in first.items()
    }
    batch["mask"] = torch.zeros(b, t, dtype=torch.bool)
    batch["token_mask"] = torch.zeros(b, t, n, dtype=torch.bool)
    for i, item in enumerate(items):
        for j, entry in enumerate(item["entries"]):
            count = entry["tokens"].shape[0]
            batch["mask"][i, j] = True
            batch["token_mask"][i, j, :count] = True
            for k, value in entry.items():
                if k in {"tokens", "types", "layers"}:
                    batch[k][i, j, :count] = value
                else:
                    batch[k][i, j] = value
    batch["target"] = torch.stack([x["target"] for x in items])
    return batch


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reference coverage by family and device")
    parser.add_argument("--report", action="store_true", required=True)
    parser.add_argument("--features", default="data/features/qwen3.5-0.8b")
    parser.add_argument("--evals", default="data/evals/qwen3.5-0.8b")
    parser.add_argument("--zoo", default="data/zoo/qwen3.5-0.8b")
    args = parser.parse_args()
    ds = FeatureDataset(args.features, args.evals, zoo=args.zoo)
    print("family | device | reason | count")
    for (family, device, reason), count in sorted(ds.match_report.items()):
        print(f"{family} | {device} | {reason} | {count}")
    print(
        f"matched={ds.matched_reference_targets} missing={ds.missing_matched_reference} "
        f"targets_with_inferred_runtime={ds.targets_with_inferred_runtime}"
    )


if __name__ == "__main__":
    main()
