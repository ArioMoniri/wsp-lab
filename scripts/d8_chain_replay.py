"""D8 step 7: rebuild lost chain snapshots by replaying their surviving edit descriptors.

Every lost chain entry still has its ``edit.json`` (family, ``inner_family``, ``inner_params``, ``inner_seed``,
``step``, ``length``, ``parent_id``). A chain entry stores the *cumulative* snapshot ``state_k - base``; replaying
the chain from the base checkpoint with the recorded step parameters (not the samplers, whose code at the
original commits is no longer in the repository) regenerates the snapshot.

Modes
-----
``verify``   replay chains whose every step is stored and compare each regenerated snapshot with the stored bytes
             (bitwise, max |diff|, Frobenius norm vs ``edit.json``). Nothing is written except the report.
``restore``  replay chains with lost steps; stored steps are compared as in ``verify`` and (when the comparison is
             not bitwise) the cumulative state is re-anchored to ``base + stored delta`` before the next step;
             lost steps are written as ``delta.safetensors`` into ``--out/<id>/`` (staging) and recorded in
             ``regen.json`` next to it. ``edit.json`` is never rewritten; ``manifest.jsonl`` is never touched.

The replay calls the builder's own family functions (``wsp.zoo.build._build_single_family``) with the samplers
patched to return the recorded ``inner_params``, so the kernels, architecture handling, dtype casts and the
``lora_vs_base`` chain semantics are exactly the production code paths.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from wsp.zoo import arch as _arch
from wsp.zoo import build
from wsp.zoo.common import resolve_base_revision

STEP_FAMILIES = ("random_dir", "abliteration", "lora_sft")


# ---- descriptor graph ----------------------------------------------------------------------------


def load_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "header":
            continue
        rows[r["id"]] = r
    return rows


def chain_groups(edits: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """root id -> entry ids ordered by ``params.step`` (edits: id -> edit.json of chain entries only).

    Raises if a chain has a gap (a step whose descriptor is missing) or a parent link that disagrees with
    the step order, so a broken graph is never replayed silently.
    """

    def root_of(i: str) -> str:
        seen = set()
        while edits[i].get("parent_id"):
            if i in seen:
                raise ValueError(f"parent cycle at {i}")
            seen.add(i)
            parent = edits[i]["parent_id"]
            if parent not in edits:
                raise KeyError(f"{i}: parent {parent} has no descriptor")
            i = parent
        return i

    groups: dict[str, list[str]] = defaultdict(list)
    for i in edits:
        groups[root_of(i)].append(i)
    for root, ids in groups.items():
        # A root may carry more than one branch (a chain re-built at another commit shares its root), so the
        # group is a tree: order parents before children (by step, then id) and check every link's step.
        ids.sort(key=lambda i: (int(edits[i]["params"]["step"]), i))
        if int(edits[root]["params"]["step"]) != 0:
            raise ValueError(f"chain {root}: root has step {edits[root]['params']['step']}")
        for i in ids:
            parent = edits[i].get("parent_id")
            if parent is None:
                if i != root:
                    raise ValueError(f"chain {root}: {i} has no parent but is not the root")
                continue
            if int(edits[i]["params"]["step"]) != int(edits[parent]["params"]["step"]) + 1:
                raise ValueError(f"chain {root}: {i} step does not follow its parent {parent}")
    return dict(groups)


def storage_of(entry_id: str, sidecar: dict[str, dict[str, Any]], zoo: Path) -> Path | None:
    """Directory holding the entry's bytes (itself or its alias), or None when lost."""
    r = sidecar[entry_id]
    if r.get("delta_present"):
        return zoo / entry_id
    if r.get("alias_of"):
        return zoo / r["alias_of"]
    return None


# ---- replay --------------------------------------------------------------------------------------


class RecordedSampler:
    """Stands in for the builder's samplers: returns the recorded step parameters, ignores the rng."""

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def __call__(self, rng: Any, family_cfg: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(self.params))  # fresh copy each call


def replay_step(
    inner_family: str,
    inner_params: dict[str, Any],
    inner_seed: int,
    config: dict[str, Any],
    cumulative: dict[str, torch.Tensor],
    zoo_dir: Path,
) -> tuple[dict[str, torch.Tensor], list[str], Any]:
    """One chain step on ``cumulative`` with the recorded parameters: (new_state, touched, inner_desc)."""
    if inner_family not in STEP_FAMILIES:
        raise ValueError(f"unsupported chain step family {inner_family!r}")
    sampler = RecordedSampler(inner_params)
    saved = (
        build.sample_random_dir_params,
        build.sample_abliteration_params,
        dict(build._LORA_SAMPLERS),
    )
    try:
        build.sample_random_dir_params = sampler  # type: ignore[assignment]
        build.sample_abliteration_params = sampler  # type: ignore[assignment]
        build._LORA_SAMPLERS["lora_sft"] = sampler
        entries = build._build_single_family(
            inner_family,
            config,
            config["families"][inner_family],
            int(inner_seed),
            cumulative,
            zoo_dir,
            [],
            dry_run=False,
        )
    finally:
        build.sample_random_dir_params, build.sample_abliteration_params = saved[0], saved[1]
        build._LORA_SAMPLERS.clear()
        build._LORA_SAMPLERS.update(saved[2])
    inner_desc, new_state, touched = entries[0]
    return new_state, list(touched), inner_desc


def chain_delta(
    new_state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
    touched: list[str],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Exactly the builder's chain delta: bf16 difference over the touched keys that changed."""
    return {
        k: (new_state[k] - base_state[k]).to(dtype)
        for k in sorted(touched)
        if k in base_state and not torch.equal(new_state[k], base_state[k])
    }


def fro_norm(delta: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(v.float().pow(2).sum()) for v in delta.values()))


def compare(delta: dict[str, torch.Tensor], stored: dict[str, torch.Tensor]) -> dict[str, Any]:
    keys_a, keys_b = set(delta), set(stored)
    out: dict[str, Any] = {
        "keys_equal": keys_a == keys_b,
        "only_regenerated": sorted(keys_a - keys_b),
        "only_stored": sorted(keys_b - keys_a),
        "n_keys": len(keys_a),
    }
    max_abs, n_bitwise, n_diff_elems, n_elems = 0.0, 0, 0, 0
    for k in keys_a & keys_b:
        a, b = delta[k], stored[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            out.setdefault("shape_or_dtype_mismatch", []).append(k)
            continue
        if torch.equal(a, b):
            n_bitwise += 1
        else:
            diff = (a.float() - b.float()).abs()
            max_abs = max(max_abs, float(diff.max()))
            n_diff_elems += int((diff > 0).sum())
        n_elems += a.numel()
    out.update(
        {
            "n_bitwise_tensors": n_bitwise,
            "bitwise": out["keys_equal"]
            and n_bitwise == len(keys_a)
            and "shape_or_dtype_mismatch" not in out,
            "max_abs_diff": max_abs,
            "frac_elements_differing": (n_diff_elems / n_elems) if n_elems else 0.0,
            "fro_regenerated": fro_norm(delta),
            "fro_stored": fro_norm(stored),
        }
    )
    return out


def environment() -> dict[str, Any]:
    env = {
        "host": "H0",  # device label of DECISIONS D2; the hostname is never recorded (public repo)
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        env["transformers"] = None
    try:
        env["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        env["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        env["git_sha"] = None
    return env


def prepare_config(config_path: Path) -> dict[str, Any]:
    """The builder's config exactly as ``build_zoo`` prepares it (resolved base, arch info, no overrides)."""
    config = build.load_config(config_path)
    config["base_revision"] = resolve_base_revision(
        config["base_model"], config.get("base_revision", "main")
    )
    config["_overrides"] = []
    config["_include_smoke"] = False
    config["_config_hash"] = build.config_hash(config_path, None)
    info = _arch.load_arch_info(config)
    if info is not None:
        _arch.validate_family_config(config, info)
    config["_arch"] = info
    return config


def select_chains(
    groups: dict[str, list[str]],
    sidecar: dict[str, dict[str, Any]],
    mode: str,
    only: set[str] | None,
) -> list[str]:
    chosen = []
    for root, ids in groups.items():
        if only is not None and root not in only and not (set(ids) & only):
            continue
        lost = [i for i in ids if storage_of(i, sidecar, Path("/")) is None]
        if (mode == "verify") == (not lost):
            chosen.append(root)
    return sorted(chosen)


def replay_chain(
    root: str,
    ids: list[str],
    edits: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    sidecar: dict[str, dict[str, Any]],
    zoo: Path,
    out: Path,
    config: dict[str, Any],
    base_state: dict[str, torch.Tensor],
    mode: str,
    env: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    report: dict[str, Any] = {"root": root, "n_steps": len(ids), "steps": []}
    # Per-node state so a root with several branches is replayed as a tree; a parent's state is freed once
    # all of its children have been processed (ids are ordered parents-first).
    children_left = {i: sum(1 for j in ids if edits[j].get("parent_id") == i) for i in ids}
    state_of: dict[str | None, dict[str, torch.Tensor]] = {None: base_state}
    touched_of: dict[str | None, set[str]] = {None: set()}
    anchor_of: dict[str | None, str | None] = {None: None}
    # random_dir / abliteration replay bitwise; a re-trained LoRA step differs by bf16 rounding (pilot: 1 ulp on
    # 6–53 % of elements, |Δ‖F‖|/‖F‖ ≤ 1e-3). A lineage is "exact" until a LoRA step is *computed* on it and
    # becomes exact again at a stored anchor.
    exact_of: dict[str | None, bool] = {None: True}
    for entry_id in ids:
        e = edits[entry_id]
        p = e["params"]
        parent = e.get("parent_id")
        cumulative = state_of[parent]
        all_touched = set(touched_of[parent])
        anchored_from = anchor_of[parent]
        exact_lineage = exact_of[parent] and p["inner_family"] != "lora_sft"
        step: dict[str, Any] = {
            "id": entry_id,
            "step": p["step"],
            "inner_family": p["inner_family"],
            "inner_seed": p["inner_seed"],
            "git_sha": (manifest.get(entry_id) or {}).get("git_sha"),
        }
        stored_dir = storage_of(entry_id, sidecar, zoo)
        step["stored"] = stored_dir is not None
        step["alias_of"] = sidecar[entry_id].get("alias_of")
        t0 = time.time()
        new_state, touched, inner_desc = replay_step(
            p["inner_family"], p["inner_params"], p["inner_seed"], config, cumulative, out
        )
        step["seconds"] = round(time.time() - t0, 2)
        all_touched.update(touched)
        delta = chain_delta(new_state, base_state, sorted(all_touched), dtype)
        step["fro_regenerated"] = fro_norm(delta)
        step["fro_edit_json"] = e.get("delta_fro_norm")
        step["touched_equal_edit_json"] = sorted(delta) == sorted(e.get("touched_tensors") or [])
        if p["inner_family"] == "abliteration":
            fp = inner_desc.provenance.get("refusal_fingerprint")
            want = (manifest.get(entry_id) or {}).get("refusal_fingerprint")
            step["refusal_fingerprint"] = fp
            step["refusal_fingerprint_matches_manifest"] = (want is None) or (fp == want)
            step["kernel_variant"] = inner_desc.provenance.get("kernel_variant")
        if stored_dir is not None:
            stored = load_file(stored_dir / "delta.safetensors")
            cmp = compare(delta, stored)
            step["compare"] = cmp
            if not cmp["bitwise"] and mode == "restore":
                # Carry the stored snapshot forward (the evaluated model is base + stored delta).
                cumulative = dict(base_state)
                for k, v in stored.items():
                    cumulative[k] = (
                        (base_state[k] + v.to(base_state[k].dtype)) if k in base_state else v
                    )
                anchored_from = entry_id
                exact_lineage = True
                step["anchored"] = True
            else:
                cumulative = new_state
                step["anchored"] = False
        else:
            cumulative = new_state
            step["regenerated_from_anchor"] = anchored_from
            step["exact_lineage"] = exact_lineage
            if mode == "restore":
                dest = out / entry_id
                dest.mkdir(parents=True, exist_ok=True)
                target = dest / "delta.safetensors"
                if target.exists() and not overwrite:
                    raise FileExistsError(f"{target} exists; pass --overwrite to replace")
                save_file({k: v.to(dtype).contiguous() for k, v in delta.items()}, target)
                regen = {
                    "id": entry_id,
                    "root": root,
                    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mode": "d8_chain_replay",
                    "anchored_from": anchored_from,
                    "exact_lineage": exact_lineage,
                    "fro_regenerated": step["fro_regenerated"],
                    "fro_edit_json": step["fro_edit_json"],
                    "touched_equal_edit_json": step["touched_equal_edit_json"],
                    "environment": env,
                }
                (dest / "regen.json").write_text(json.dumps(regen, indent=2, sort_keys=True))
                step["written"] = str(target)
        state_of[entry_id] = cumulative
        touched_of[entry_id] = all_touched
        anchor_of[entry_id] = anchored_from
        exact_of[entry_id] = exact_lineage
        if parent is not None:
            children_left[parent] -= 1
            if children_left[parent] == 0:
                del state_of[parent]
        if children_left[entry_id] == 0:
            del state_of[entry_id]
        report["steps"].append(step)
        print(
            json.dumps(
                {k: v for k, v in step.items() if k != "compare"}
                | {
                    "bitwise": step.get("compare", {}).get("bitwise"),
                    "max_abs_diff": step.get("compare", {}).get("max_abs_diff"),
                }
            ),
            flush=True,
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--zoo", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True, help="artefact_status.jsonl")
    ap.add_argument("--config", type=Path, default=Path("configs/zoo_qwen3.5-0.8b.yaml"))
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where regenerated entries (and the LoRA staging) go",
    )
    ap.add_argument("--mode", choices=("verify", "restore"), required=True)
    ap.add_argument(
        "--chains", default=None, help="comma-separated root or member ids to restrict to"
    )
    ap.add_argument("--max-chains", type=int, default=None)
    ap.add_argument(
        "--skip-families",
        default="",
        help="comma-separated inner families to skip chains containing",
    )
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sidecar = load_sidecar(args.sidecar)
    manifest = load_manifest(args.zoo / "manifest.jsonl")
    edits = {}
    for i, row in manifest.items():
        if row.get("family") != "chain":
            continue
        path = args.zoo / i / "edit.json"
        if not path.exists():
            raise FileNotFoundError(f"{i}: edit.json missing; cannot replay")
        edits[i] = json.loads(path.read_text())
    groups = chain_groups(edits)
    only = set(args.chains.split(",")) if args.chains else None
    roots = select_chains(groups, sidecar, args.mode, only)
    skip = {f for f in args.skip_families.split(",") if f}
    if skip:
        roots = [
            r
            for r in roots
            if not any(edits[i]["params"]["inner_family"] in skip for i in groups[r])
        ]
    if args.max_chains:
        roots = roots[: args.max_chains]
    print(
        f"mode={args.mode} chains={len(roots)} entries={sum(len(groups[r]) for r in roots)}",
        flush=True,
    )

    config = prepare_config(args.config)
    env = environment()
    env["config_hash"] = config["_config_hash"]
    env["base_revision"] = config["base_revision"]
    args.out.mkdir(parents=True, exist_ok=True)
    base_state = build._load_base_state(config)
    reports = []
    started = time.time()
    for n, root in enumerate(roots, 1):
        print(f"--- chain {n}/{len(roots)} root={root} steps={len(groups[root])}", flush=True)
        reports.append(
            replay_chain(
                root,
                groups[root],
                edits,
                manifest,
                sidecar,
                args.zoo,
                args.out,
                config,
                base_state,
                args.mode,
                env,
                args.overwrite,
            )
        )
        summary = {
            "mode": args.mode,
            "environment": env,
            "wall_seconds": round(time.time() - started, 1),
            "n_chains": len(reports),
            "chains": reports,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=1, sort_keys=True))
    steps = [s for r in reports for s in r["steps"]]
    stored = [s for s in steps if s["stored"]]
    print(
        json.dumps(
            {
                "chains": len(reports),
                "steps": len(steps),
                "stored_compared": len(stored),
                "bitwise": sum(1 for s in stored if s["compare"]["bitwise"]),
                "written": sum(1 for s in steps if s.get("written")),
                "wall_seconds": round(time.time() - started, 1),
            }
        )
    )


if __name__ == "__main__":
    sys.exit(main())
