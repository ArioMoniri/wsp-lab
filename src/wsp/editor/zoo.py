"""D1 conversion and ordinary zoo descriptor/manifest registration."""

import json
import re
from pathlib import Path

import torch

from wsp.editor.episode import digest
from wsp.features.extract import effective_delta
from wsp.schema import EditDescriptor, write_manifest
from wsp.zoo.common import get_git_sha


def register(api, config, provenance, zoo_dir):
    from wsp.zoo.build import _append_manifest, _write_entry

    if not re.fullmatch("[0-9a-f]{40}", config["base_revision"]):
        raise ValueError("resolved base revision required")
    if not api.deltas:
        raise ValueError("empty delta")
    delta, rounding = {}, {}
    for name, value in list(api.deltas.items()):
        api.set(name, value, api.rationales[name])
        delta[name], rounding[name] = effective_delta(api.read(name), value)
    proposal_digest = digest({k: v.tolist() for k, v in sorted(api.deltas.items())})
    desc = EditDescriptor(
        "llm_edit",
        {
            "proposal_digest": proposal_digest,
            "code_hash": provenance["code_hash"],
            "config_hash": digest(config),
            "editor": {k: provenance[k] for k in ("provider", "model", "version")},
        },
        config["seed"],
        config["base_model"],
        config["base_revision"],
        git_sha=get_git_sha(),
        eligibility="smoke",
        eligibility_reasons=["production isolation/split review pending"],
        provenance=dict(
            provenance,
            rounding_stats=rounding,
            merge_rule="checkpoint_dtype(W+delta)-W",
            config_hash=digest(config),
        ),
    )
    root = Path(zoo_dir)
    entry = root / desc.id
    if not entry.exists():
        _write_entry(root, desc, delta, torch.float32)
        row = json.loads(desc.to_json())
        write_manifest(entry / "manifest.json", **row)
        _append_manifest(root / "manifest.jsonl", row)
    return desc.id


def build(config, family_cfg, seed, base_state, *, dry_run=False):
    """Replay an explicit scripted candidate through the same supervised episode path."""
    import tempfile

    from safetensors.torch import load_file

    from wsp.editor.driver import run_episode
    from wsp.editor.episode import Episode
    from wsp.editor.providers import provider

    if dry_run:
        raise ValueError("llm_edit dry runs require a frozen proposal; use run_episode")
    editor_config = dict(
        family_cfg["editor_config"],
        seed=seed,
        base_model=config["base_model"],
        base_revision=config["base_revision"],
    )
    adapter = provider(
        family_cfg["provider"], config=editor_config, **family_cfg.get("provider_options", {})
    )
    with tempfile.TemporaryDirectory(prefix="wsp-editor-") as scratch:
        root = Path(scratch)
        ids = run_episode(
            adapter,
            editor_config,
            base_state,
            None,
            family_cfg.get("development_pool", []),
            Episode(Path(family_cfg["episode_root"]), f"build-{seed}"),
            root,
        )
        if not ids[0]:
            raise ValueError("invalid editor proposal; see episode trajectory")
        entry = root / ids[0]
        desc = EditDescriptor.from_json((entry / "edit.json").read_text())
        delta = load_file(entry / "delta.safetensors")
        state = dict(base_state)
        state.update(
            {k: (base_state[k].float() + v).to(base_state[k].dtype) for k, v in delta.items()}
        )
        return [(desc, state, sorted(delta))]
