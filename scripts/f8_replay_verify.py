#!/usr/bin/env python3
"""F8a.4 / spec §4 CPU identity verifier for an external replay checkpoint against the pinned base.

CPU only; reads the two checkpoint directories it is given and writes one JSON record to --output.
It never downloads anything. Steps (spec §4 numbering; status pass / fail / not_run):

* ``base_revision``: the base directory is the HF snapshot of the pinned revision (``…/snapshots/<rev>``).
* ``load``: every ``*.safetensors`` file of each directory loads; a shard index, if present, matches the
  shards exactly; no tensor name occurs twice.
* ``A3_tokenizer``: every tokenizer file present in either directory is present in both, byte-identical.
* ``A3_config``: ``config.json`` (and any other model config file present in either) equal as parsed JSON
  after removing only ``_name_or_path`` and ``transformers_version``.
* ``A4_tensor_names`` / ``A5_shapes_dtypes``: identical key sets; identical shape and dtype per key.
* ``A6_changed_set``: the set of tensors whose bytes differ equals the declared allowlist exactly.
* ``A7_unchanged_complement``: every tensor outside the allowlist is byte-identical, and the complement
  hash (``response_field._hash`` over (name, ``_tensor_hash``)) is equal for base and candidate.
* ``A8_claimed_columns`` (only with --claimed-columns): changed columns are a subset of the released set.
* ``A9_roundtrip``: D_t = fp32(candidate_t) − fp32(base_t) written as a dense delta, read back with
  ``wsp.zoo.apply.load_entry_delta`` and applied with the arithmetic of ``wsp.zoo.apply.load_edited_model``
  reproduces every candidate tensor bitwise; the round-trip state hash equals the candidate's.
* ``A10_apply_path`` (only with --entry): the staged zoo entry loaded by ``load_edited_model`` gives the
  candidate's full-state hash.
* ``A11_reload_determinism``: two further loads (fresh processes by default) give the same file sha256s and
  full-state hash as the first load.
* ``scoring_identity`` (only with --reference-identity): shared scoring identity (model and tokenizer
  revision, tokenizer file hashes, ``effective_config_hash``, per-benchmark item counts) equals the base
  reference's; edited state and delta hashes are never compared with the base's (Board A F8a cond. 3).

Comparisons are bitwise (NaN payloads and signed zeros count). The verdict is PASS only if no step
failed and every core step passed; any exception fails closed. A1 (provenance) and A2 (form) belong
to the acquisition step and are listed as not run. ``--dry-run`` builds a synthetic tiny model in a
temporary directory and runs a positive and a negative case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from wsp.eval.batch import effective_config_hash
from wsp.features import response_field as rf
from wsp.zoo.apply import load_edited_model, load_entry_delta

PINNED_BASE_REPO = "Qwen/Qwen3.5-0.8B"
PINNED_BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEFAULT_CONFIG = ROOT / "configs/eval_tf_v1.yaml"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
CONFIG_IGNORED_KEYS = ("_name_or_path", "transformers_version")
INDEX_FILE = "model.safetensors.index.json"
CORE_STEPS = (
    "base_revision",
    "load",
    "A3_tokenizer",
    "A3_config",
    "A4_tensor_names",
    "A5_shapes_dtypes",
    "A6_changed_set",
    "A7_unchanged_complement",
    "A9_roundtrip",
    "A11_reload_determinism",
)
STEP_ORDER = (
    "A1_provenance",
    "A2_form",
    *CORE_STEPS[:8],
    "A8_claimed_columns",
    "A9_roundtrip",
    "A10_apply_path",
    "A11_reload_determinism",
    "scoring_identity",
)
OUTSIDE_SCOPE = {
    "A1_provenance": "acquisition step (URL, revision, licence, file sha256 of two downloads)",
    "A2_form": "acquisition step (full checkpoint vs patch and its rule)",
}
SHARED_IDENTITY = (
    "model_revision",
    "tokenizer_revision",
    "tokenizer_files_sha256",
    "config_hash",
    "item_counts",
)
INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
CODE_FILES = (
    "scripts/f8_replay_verify.py",
    "src/wsp/zoo/apply.py",
    "src/wsp/zoo/common.py",
    "src/wsp/features/response_field.py",
    "src/wsp/eval/batch.py",
)


class CheckpointError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_files(path: Path) -> list[Path]:
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise CheckpointError(f"no *.safetensors in {path}")
    return files


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """All tensors of a (possibly sharded) safetensors directory; fails on any inconsistency."""
    path = Path(path)
    files = weight_files(path)
    state: dict[str, torch.Tensor] = {}
    location: dict[str, str] = {}
    for f in files:
        for name, t in load_file(f).items():
            if name in state:
                raise CheckpointError(f"tensor {name} occurs in {location[name]} and {f.name}")
            state[name] = t
            location[name] = f.name
    index = path / INDEX_FILE
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if set(weight_map.values()) != {f.name for f in files}:
            raise CheckpointError("shard index names a different file set than present")
        if weight_map != location:
            raise CheckpointError("shard index tensor locations differ from the shards")
    return state


def tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {n: rf._tensor_hash(t) for n, t in sorted(state.items())}


def digest_tensors(path: Path, state: dict[str, torch.Tensor]) -> dict:
    hashes = tensor_hashes(state)
    return {
        "files": {f.name: sha256_file(f) for f in weight_files(path)},
        "state_hash": rf._hash(hashes),
        "tensor_hashes": hashes,
    }


def state_digest(path: Path) -> dict:
    """One full load + hash of a checkpoint directory (the unit of the reload check)."""
    return digest_tensors(path, load_checkpoint(path))


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu().contiguous().reshape(-1)
    if t.dtype == torch.bool:
        return t.view(torch.uint8)
    return t.view(INT_VIEW[t.element_size()])


def index_summary(idx: torch.Tensor) -> dict:
    values = [int(i) for i in idx.tolist()]
    return {
        "count": len(values),
        "sha256": hashlib.sha256(json.dumps(values).encode()).hexdigest(),
        "indices": values,
    }


def tensor_diff(base: torch.Tensor, cand: torch.Tensor) -> dict:
    """Bitwise per-coordinate comparison of two same-shape, same-dtype tensors."""
    changed = bits(base) != bits(cand)
    out = {
        "shape": list(base.shape),
        "dtype": str(base.dtype),
        "changed_coordinates": int(changed.sum()),
        "numel": base.numel(),
        "base_tensor_hash": rf._tensor_hash(base),
        "candidate_tensor_hash": rf._tensor_hash(cand),
    }
    if base.is_floating_point():
        delta = cand.float() - base.float()
        out["delta_nonzero"] = int((delta != 0).sum())
        finite = torch.isfinite(delta)
        out["delta_nonfinite"] = int((~finite).sum())
        out["delta_fro_norm"] = float(delta[finite].norm()) if bool(finite.any()) else 0.0
    mask = changed.reshape(base.shape)
    if base.dim() == 2:
        out["changed_rows"] = index_summary(mask.any(dim=1).nonzero().flatten())
        out["changed_cols"] = index_summary(mask.any(dim=0).nonzero().flatten())
    elif base.dim() == 1:
        out["changed_indices"] = index_summary(mask.nonzero().flatten())
    return out


def snapshot_revision(base_dir: Path) -> str | None:
    d = Path(base_dir).resolve()
    return d.name if d.parent.name == "snapshots" else None


def present_files(base_dir: Path, cand_dir: Path, names) -> dict:
    """{name: (base path or None, candidate path or None)} for files present in either directory."""
    present = {}
    for name in names:
        b, c = Path(base_dir) / name, Path(cand_dir) / name
        if b.exists() or c.exists():
            present[name] = (b if b.exists() else None, c if c.exists() else None)
    return present


def check_tokenizer(base_dir: Path, cand_dir: Path) -> dict:
    present = present_files(base_dir, cand_dir, TOKENIZER_FILES)
    files, problems = {}, []
    for name, (b, c) in present.items():
        rec = {
            "base_sha256": sha256_file(b) if b else None,
            "candidate_sha256": sha256_file(c) if c else None,
        }
        rec["identical"] = (
            rec["base_sha256"] is not None and rec["base_sha256"] == rec["candidate_sha256"]
        )
        files[name] = rec
        if not rec["identical"]:
            problems.append(name)
    if not any(n in present for n in ("tokenizer.json", "tokenizer_config.json")):
        problems.append("no tokenizer.json or tokenizer_config.json in either directory")
    return {"status": "fail" if problems else "pass", "files": files, "differing": problems}


def check_config(base_dir: Path, cand_dir: Path) -> dict:
    present = present_files(base_dir, cand_dir, CONFIG_FILES)
    if "config.json" not in present:
        present["config.json"] = (None, None)
    files, differing_keys, problems = {}, set(), []
    for name, (b, c) in present.items():
        rec = {
            "base_sha256": sha256_file(b) if b else None,
            "candidate_sha256": sha256_file(c) if c else None,
        }
        if b is None or c is None:
            rec["equal_after_allowlist"] = False
            problems.append(f"{name}: missing in {'base' if b is None else 'candidate'}")
        else:
            bj, cj = json.loads(b.read_text()), json.loads(c.read_text())
            for k in CONFIG_IGNORED_KEYS:
                bj.pop(k, None)
                cj.pop(k, None)
            keys = sorted(k for k in set(bj) | set(cj) if bj.get(k, ...) != cj.get(k, ...))
            rec["equal_after_allowlist"] = not keys
            rec["differing_keys"] = keys
            differing_keys.update(f"{k}" if name == "config.json" else f"{name}:{k}" for k in keys)
            if keys:
                problems.append(name)
        files[name] = rec
    return {
        "status": "fail" if problems else "pass",
        "ignored_keys": list(CONFIG_IGNORED_KEYS),
        "files": files,
        "differing_keys": sorted(differing_keys),
        "problems": problems,
    }


def scoring_identity(base_dir: Path, revision: str, config_path: Path) -> dict:
    """Shared scoring identity computed with the production config hash (never the YAML sha256)."""
    cfg = yaml.safe_load(Path(config_path).read_bytes())
    tok = {
        n: sha256_file(Path(base_dir) / n) for n in TOKENIZER_FILES if (Path(base_dir) / n).exists()
    }
    return {
        "model_repo": PINNED_BASE_REPO,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "tokenizer_files_sha256": tok,
        "config_path": str(Path(config_path).resolve().relative_to(ROOT))
        if Path(config_path).resolve().is_relative_to(ROOT)
        else str(config_path),
        "config_hash": effective_config_hash(cfg),
        "config_file_sha256": sha256_file(Path(config_path)),
        "item_counts": {k: v.get("n", v.get("n_windows")) for k, v in cfg["benchmarks"].items()},
    }


def reload_digests(cand_dir: Path, mode: str) -> list[dict]:
    if mode == "inline":
        return [state_digest(cand_dir), state_digest(cand_dir)]
    if mode != "process":
        raise ValueError(f"unknown reload mode {mode!r}")
    out = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--hash-state", str(cand_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        out.append(json.loads(proc.stdout))
    return out


def _step(status: str, **detail) -> dict:
    return {"status": status, **detail}


def verify(
    candidate_dir,
    base_dir,
    allowlist,
    *,
    expected_base_revision: str = PINNED_BASE_REVISION,
    claimed_columns: dict | None = None,
    reload: str = "process",
    entry_dir=None,
    config_path=DEFAULT_CONFIG,
    reference_identity: dict | None = None,
) -> dict:
    cand_dir, base_dir = Path(candidate_dir), Path(base_dir)
    allow = sorted(set(allowlist))
    steps: dict[str, dict] = {k: _step("not_run", reason=v) for k, v in OUTSIDE_SCOPE.items()}
    record = {
        "schema": "f8-replay-identity-v1",
        "expected": {
            "base_repo": PINNED_BASE_REPO,
            "base_revision": expected_base_revision,
            "changed_tensors": allow,
            "config_ignored_keys": list(CONFIG_IGNORED_KEYS),
            "tokenizer_files_checked": list(TOKENIZER_FILES),
            "claimed_columns": claimed_columns,
        },
        "inputs": {
            "candidate_dir": str(cand_dir),
            "base_dir": str(base_dir),
            "entry_dir": str(entry_dir) if entry_dir else None,
            "reload_mode": reload,
        },
        "code": {rel: sha256_file(ROOT / rel) for rel in CODE_FILES if (ROOT / rel).exists()},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "steps": steps,
        "changed_tensors": {},
        "hashes": {"state": {}, "complement": {}, "files": {}},
    }
    try:
        _verify_into(
            record,
            cand_dir,
            base_dir,
            allow,
            expected_base_revision,
            claimed_columns,
            reload,
            entry_dir,
            config_path,
            reference_identity,
        )
    except Exception as exc:  # noqa: BLE001  (fail closed, keep everything recorded so far)
        record["error"] = f"{type(exc).__name__}: {exc}"
    for name in STEP_ORDER:
        steps.setdefault(name, _step("not_run", reason="not reached or not requested"))
    record["steps"] = {k: steps[k] for k in STEP_ORDER}
    failed = [k for k, v in record["steps"].items() if v["status"] == "fail"]
    core_ok = all(record["steps"][k]["status"] == "pass" for k in CORE_STEPS)
    record["failed_steps"] = failed
    record["verdict"] = "PASS" if core_ok and not failed and "error" not in record else "FAIL"
    record["steps_not_run"] = [k for k, v in record["steps"].items() if v["status"] == "not_run"]
    record["complete_for_f8a4"] = record["verdict"] == "PASS" and not record["steps_not_run"]
    return record


def _verify_into(
    record,
    cand_dir,
    base_dir,
    allow,
    expected_rev,
    claimed_columns,
    reload,
    entry_dir,
    config_path,
    reference_identity,
):
    steps = record["steps"]
    rev = snapshot_revision(base_dir)
    steps["base_revision"] = _step(
        "pass" if rev == expected_rev else "fail",
        observed=rev,
        rule="base directory must be the HF snapshot directory named by the pinned revision",
    )
    steps["A3_tokenizer"] = check_tokenizer(base_dir, cand_dir)
    steps["A3_config"] = check_config(base_dir, cand_dir)

    try:
        base = load_checkpoint(base_dir)
        cand = load_checkpoint(cand_dir)
    except Exception as exc:  # noqa: BLE001  (any load error fails the step)
        steps["load"] = _step("fail", error=f"{type(exc).__name__}: {exc}")
        return
    base_digest = digest_tensors(base_dir, base)
    cand_digest = digest_tensors(cand_dir, cand)
    steps["load"] = _step("pass", base_tensors=len(base), candidate_tensors=len(cand))
    record["hashes"]["files"] = {"base": base_digest["files"], "candidate": cand_digest["files"]}
    record["hashes"]["state"] = {
        "namespace": "safetensors file keys (not the model.state_dict() namespace of g2 base_state_hash)",
        "base": base_digest["state_hash"],
        "candidate": cand_digest["state_hash"],
    }

    missing, extra = sorted(set(base) - set(cand)), sorted(set(cand) - set(base))
    steps["A4_tensor_names"] = _step(
        "fail" if missing or extra else "pass",
        n_base=len(base),
        n_candidate=len(cand),
        missing_in_candidate=missing,
        extra_in_candidate=extra,
    )
    common = sorted(set(base) & set(cand))
    mismatch = {
        n: {
            "base": [list(base[n].shape), str(base[n].dtype)],
            "candidate": [list(cand[n].shape), str(cand[n].dtype)],
        }
        for n in common
        if base[n].shape != cand[n].shape or base[n].dtype != cand[n].dtype
    }
    steps["A5_shapes_dtypes"] = _step("fail" if mismatch else "pass", mismatches=mismatch)

    comparable = [n for n in common if n not in mismatch]
    changed = {}
    for n in comparable:
        if base_digest["tensor_hashes"][n] != cand_digest["tensor_hashes"][n] or not torch.equal(
            bits(base[n]), bits(cand[n])
        ):
            changed[n] = tensor_diff(base[n], cand[n])
    record["changed_tensors"] = changed
    changed_set = set(changed) | set(mismatch) | set(missing) | set(extra)
    unexpected = sorted(changed_set - set(allow))
    unchanged_allowed = sorted(set(allow) - changed_set)
    steps["A6_changed_set"] = _step(
        "pass" if not unexpected and not unchanged_allowed else "fail",
        observed_changed=sorted(changed_set),
        unexpected_changed=unexpected,
        allowlisted_unchanged=unchanged_allowed,
        rule="set of byte-differing tensors must equal the declared allowlist exactly",
    )
    complement = [n for n in sorted(set(base) | set(cand)) if n not in set(allow)]
    comp_base = rf._hash({n: base_digest["tensor_hashes"].get(n) for n in complement})
    comp_cand = rf._hash({n: cand_digest["tensor_hashes"].get(n) for n in complement})
    record["hashes"]["complement"] = {
        "base": comp_base,
        "candidate": comp_cand,
        "n_tensors": len(complement),
    }
    outside = sorted(set(complement) & changed_set)
    steps["A7_unchanged_complement"] = _step(
        "pass" if not outside and comp_base == comp_cand else "fail",
        differing_outside_allowlist=outside,
        complement_hash_equal=comp_base == comp_cand,
    )

    if claimed_columns is not None:
        outside_cols = {}
        for n, cols in sorted(claimed_columns.items()):
            got = changed.get(n, {}).get("changed_cols", {}).get("indices", [])
            extra_cols = sorted(set(got) - {int(c) for c in cols})
            if extra_cols:
                outside_cols[n] = extra_cols
        unknown = sorted(set(changed) - set(claimed_columns))
        steps["A8_claimed_columns"] = _step(
            "fail" if outside_cols or unknown else "pass",
            outside=outside_cols,
            changed_tensors_without_claim=unknown,
            axis="dim 1 of the stored 2-D weight",
        )

    if not mismatch and not missing and not extra:
        steps["A9_roundtrip"] = roundtrip(base, cand, sorted(changed), record)
    else:
        steps["A9_roundtrip"] = _step("not_run", reason="tensor set, shape or dtype mismatch")

    if entry_dir is not None:
        entry_dir = Path(entry_dir)
        state = load_edited_model(base, entry_dir)
        applied = rf._hash(tensor_hashes(state))
        delta_sha = sha256_file(entry_dir / "delta.safetensors")
        record["entry_hashes"] = {"delta.safetensors": delta_sha}
        steps["A10_apply_path"] = _step(
            "pass" if applied == cand_digest["state_hash"] else "fail",
            applied_state_hash=applied,
            entry_hashes={"delta.safetensors": delta_sha},
            path="wsp.zoo.apply.load_edited_model",
        )

    digests = reload_digests(cand_dir, reload)
    first = {"files": cand_digest["files"], "state_hash": cand_digest["state_hash"]}
    observed = [{"files": d["files"], "state_hash": d["state_hash"]} for d in digests]
    steps["A11_reload_determinism"] = _step(
        "pass" if all(o == first for o in observed) else "fail",
        mode=reload,
        loads=1 + len(observed),
        state_hashes=[first["state_hash"]] + [o["state_hash"] for o in observed],
        files_equal=all(o["files"] == first["files"] for o in observed),
    )

    identity = scoring_identity(base_dir, expected_rev, config_path)
    record["scoring_identity"] = identity
    if reference_identity is not None:
        compared = [k for k in SHARED_IDENTITY if k in reference_identity]
        mismatched = [k for k in compared if reference_identity[k] != identity[k]]
        absent = [k for k in ("model_revision", "config_hash") if k not in reference_identity]
        steps["scoring_identity"] = _step(
            "fail" if mismatched or absent else "pass",
            compared=compared,
            mismatched=mismatched,
            reference_lacks=absent,
            rule="shared scoring identity must match the base reference; edited state and delta "
            "hashes are checked against the candidate (A9/A10), never against the base",
        )


def roundtrip(base, cand, names, record) -> dict:
    """D_t = fp32(cand) − fp32(base) through a dense delta file and the production loader."""
    delta = {n: cand[n].float() - base[n].float() for n in names}
    if not delta:
        return _step("fail", reason="no changed tensor to round-trip")
    with tempfile.TemporaryDirectory(prefix="f8-replay-") as tmp:
        save_file(delta, str(Path(tmp) / "delta.safetensors"))
        delta_sha = sha256_file(Path(tmp) / "delta.safetensors")
        loaded = load_entry_delta(tmp, base)
        state = dict(base)
        for key, d in loaded.items():  # arithmetic of wsp.zoo.apply.load_edited_model
            target = state[key]
            state[key] = (target.float() + d.to(dtype=torch.float32)).to(target.dtype)
    bad = sorted(n for n in names if not torch.equal(bits(state[n]), bits(cand[n])))
    rt_hash = rf._hash(tensor_hashes(state))
    record["hashes"]["state"]["roundtrip"] = rt_hash
    record["hashes"]["roundtrip_delta_sha256"] = delta_sha
    ok = not bad and rt_hash == record["hashes"]["state"]["candidate"]
    return _step(
        "pass" if ok else "fail",
        tensors=names,
        not_reproduced=bad,
        rule="dtype(fp32(base) + D) == candidate bitwise for every changed tensor",
    )


# ------------------------------------------------------------------------------ dry run


def synthetic_model(root: Path, revision: str = "0" * 40) -> tuple[Path, Path, Path, list[str]]:
    """Tiny bf16/fp32 checkpoint pair: base snapshot, exact two-tensor edit, and a bad edit."""
    g = torch.Generator().manual_seed(20261111)
    names = {
        "a": "model.language_model.layers.0.mlp.down_proj.weight",
        "b": "model.language_model.layers.1.mlp.down_proj.weight",
    }
    base = {
        names["a"]: torch.randn(8, 16, generator=g).to(torch.bfloat16),
        names["b"]: torch.randn(8, 16, generator=g).to(torch.bfloat16),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, generator=g).to(
            torch.bfloat16
        ),
        "model.language_model.layers.0.linear_attn.A_log": torch.randn(4, generator=g),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    good = {k: v.clone() for k, v in base.items()}
    for key, cols in ((names["a"], [1, 3]), (names["b"], [2])):
        w = good[key].float()
        w[:, cols] -= 0.25
        good[key] = w.to(torch.bfloat16)
    bad = {k: v.clone() for k, v in good.items()}
    bad["model.language_model.norm.weight"][0] = 2.0
    config = {
        "architectures": ["Toy"],
        "hidden_size": 8,
        "_name_or_path": "toy",
        "transformers_version": "0",
    }
    tok = {
        "tokenizer.json": b'{"toy": true}',
        "tokenizer_config.json": b"{}",
        "chat_template.jinja": b"{{ messages }}",
    }
    dirs = []
    for sub, tensors, cfg in (
        (f"hub/snapshots/{revision}", base, config),
        ("candidate", good, config | {"_name_or_path": "replay", "transformers_version": "1"}),
        ("bad_candidate", bad, config),
    ):
        d = root / sub
        d.mkdir(parents=True)
        save_file(tensors, str(d / "model.safetensors"))
        (d / "config.json").write_text(json.dumps(cfg))
        for n, data in tok.items():
            (d / n).write_bytes(data)
        dirs.append(d)
    return dirs[0], dirs[1], dirs[2], sorted(names.values())


def dry_run() -> dict:
    with tempfile.TemporaryDirectory(prefix="f8-replay-dry-") as tmp:
        rev = "0" * 40
        base, good, bad, allow = synthetic_model(Path(tmp), rev)
        positive = verify(good, base, allow, expected_base_revision=rev, reload="process")
        negative = verify(bad, base, allow, expected_base_revision=rev, reload="inline")
    return {
        "schema": "f8-replay-identity-dry-run-v1",
        "dry_run": True,
        "synthetic": "tiny bf16/fp32 checkpoint built in a temporary directory; no download",
        "positive": positive,
        "negative": negative,
        "ok": positive["verdict"] == "PASS" and negative["verdict"] == "FAIL",
    }


def _read_allowlist(args) -> list[str]:
    names = list(args.allow or [])
    if args.allowlist:
        data = json.loads(Path(args.allowlist).read_text())
        names += data["tensors"] if isinstance(data, dict) else data
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--base", type=Path, help="HF snapshot dir of the pinned base revision")
    ap.add_argument("--base-revision", default=PINNED_BASE_REVISION)
    ap.add_argument("--allow", action="append", help="tensor expected to differ (repeatable)")
    ap.add_argument("--allowlist", type=Path, help='JSON list or {"tensors": [...]}')
    ap.add_argument("--claimed-columns", type=Path, help="JSON {tensor: [column, ...]}")
    ap.add_argument("--entry", type=Path, help="staged zoo entry for the A10 apply-path check")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--reference-identity", type=Path, help="base reference scoring identity JSON")
    ap.add_argument("--reload", choices=("process", "inline"), default="process")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hash-state", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.hash_state:
        d = state_digest(args.hash_state)
        print(json.dumps({"files": d["files"], "state_hash": d["state_hash"]}, sort_keys=True))
        return 0
    if args.output is None:
        ap.error("--output is required")
    if args.dry_run:
        record = dry_run()
        verdict = "PASS" if record["ok"] else "FAIL"
        print(
            f"f8_replay_verify --dry-run: positive {record['positive']['verdict']}, "
            f"negative {record['negative']['verdict']} -> {verdict}"
        )
    else:
        if not args.candidate or not args.base:
            ap.error("--candidate and --base are required (or use --dry-run)")
        allow = _read_allowlist(args)
        if not allow:
            ap.error("declare the expected changed tensors with --allow or --allowlist")
        claimed = json.loads(args.claimed_columns.read_text()) if args.claimed_columns else None
        reference = (
            json.loads(args.reference_identity.read_text()) if args.reference_identity else None
        )
        record = verify(
            args.candidate,
            args.base,
            allow,
            expected_base_revision=args.base_revision,
            claimed_columns=claimed,
            reload=args.reload,
            entry_dir=args.entry,
            config_path=args.config,
            reference_identity=reference,
        )
        verdict = record["verdict"]
        print(
            f"f8_replay_verify: {verdict}; failed {record['failed_steps']}; "
            f"not run {record['steps_not_run']}"
        )
        for n, d in record["changed_tensors"].items():
            print(f"  changed {n}: {d['changed_coordinates']} coordinates")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(f"  record: {args.output}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
