# ruff: noqa: F811
"""Entry identity/reuse, parent eligibility and chain step semantics of the zoo builder (CPU).

Regressions for Board A re-review of feat/p1-zoo-build (findings 1-3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_zoo_lora_build import (  # noqa: F401  (fixtures)
    FAMILIES,
    _cfg,
    _rows,
    fake_generators,
    toy_base,
)

import wsp.zoo.build as B
from wsp.zoo.build import EntryIdentityMismatch, build_zoo


def _edit(zoo: Path, entry_id: str) -> dict:
    return json.loads((zoo / entry_id / "edit.json").read_text())


def _fams(*names, **patch):
    out = {k: json.loads(json.dumps(FAMILIES[k])) for k in names}
    for k, v in patch.items():
        out[k].update(v)
    return out


# ---- finding 1: entry identity / reuse ---------------------------------------------------------


def test_base_is_pinned_to_the_resolved_commit_before_ids_are_built(
    tmp_path, fake_generators, toy_base, pinned_base_revision
):
    cfg = _cfg(tmp_path, _fams("lora_sft", "random_dir"))
    zoo = tmp_path / "zoo"
    rows = build_zoo(cfg, git_sha="s")
    sha_a = pinned_base_revision("Qwen/Qwen3.5-0.8B", "main")
    assert rows and all(r["base_revision"] == sha_a for r in rows)
    for r in rows:
        edit = _edit(zoo, r["id"])
        assert edit["base_revision"] == sha_a == r["base_revision"]
        assert edit["provenance"]["requested_base_revision"] == "main"
        assert r["reuse_key"] == edit["provenance"]["reuse_key"]
    # generate() is told the pinned commit, not the mutable ref.
    assert all(c["params"]["base_revision"] == sha_a for c in fake_generators)
    header = json.loads((zoo / "manifest.jsonl").read_text().splitlines()[0])
    assert header["base_revision"] == sha_a and header["requested_base_revision"] == "main"

    # Upstream drift: `main` now resolves elsewhere -> new ids, old entries untouched, not reused.
    n_calls = len(fake_generators)
    sha_b = "b" * 40
    pinned_base_revision.override["main"] = sha_b
    rows2 = build_zoo(cfg, git_sha="s")
    new = [r for r in rows2 if r["base_revision"] == sha_b]
    assert len(new) == len(rows) and not {r["id"] for r in new} & {r["id"] for r in rows}
    assert len(fake_generators) == n_calls + sum(r["family"] == "lora_sft" for r in new)
    assert all(_edit(zoo, r["id"])["base_revision"] == sha_a for r in rows)


def test_same_settings_reuse_entries_without_regenerating(tmp_path, fake_generators, toy_base):
    cfg = _cfg(tmp_path, _fams("lora_sft", "random_dir"))
    rows = build_zoo(cfg, git_sha="s")
    n_calls = len(fake_generators)
    delta = {r["id"]: (tmp_path / "zoo" / r["id"]).stat().st_mtime_ns for r in rows}
    again = build_zoo(cfg, git_sha="s")
    assert [r["id"] for r in again] == [r["id"] for r in rows]
    assert [r["reuse_key"] for r in again] == [r["reuse_key"] for r in rows]
    assert len(fake_generators) == n_calls
    assert all(
        (tmp_path / "zoo" / i).stat().st_mtime_ns == m for i, m in delta.items() if "lora" in i
    )


@pytest.mark.parametrize("family", ["lora_sft", "random_dir"])
def test_dtype_or_global_override_change_is_rejected_not_silently_reused(
    tmp_path, fake_generators, toy_base, family
):
    cfg = _cfg(tmp_path, _fams(family))
    zoo = tmp_path / "zoo"
    rows = build_zoo(cfg, git_sha="s")
    entry = zoo / rows[0]["id"]
    snapshot = {p.name: p.read_bytes() for p in entry.iterdir() if p.is_file()}
    n_calls = len(fake_generators)

    # dtype is not part of the sampled params, hence not of the planned id: same id, other artefact.
    raw = yaml.safe_load(cfg.read_text())
    raw["dtype"] = "bfloat16"
    cfg.write_text(yaml.safe_dump(raw))
    with pytest.raises(EntryIdentityMismatch, match=rows[0]["id"]):
        build_zoo(cfg, git_sha="s")
    assert len(fake_generators) == n_calls  # not regenerated either
    assert {p.name: p.read_bytes() for p in entry.iterdir() if p.is_file()} == snapshot

    # So does a global override that is not in the id (e.g. refusal_seed).
    cfg.write_text(yaml.safe_dump({**raw, "dtype": "float32"}))
    build_zoo(cfg, git_sha="s")  # original settings still reuse fine
    with pytest.raises(EntryIdentityMismatch):
        build_zoo(cfg, git_sha="s", overrides=["refusal_seed=5"])


def test_entry_without_reuse_key_is_never_reused(tmp_path, fake_generators, toy_base):
    cfg = _cfg(tmp_path, _fams("lora_sft"))
    rows = build_zoo(cfg, git_sha="s")
    path = tmp_path / "zoo" / rows[0]["id"] / "edit.json"
    edit = json.loads(path.read_text())
    edit["provenance"].pop("reuse_key")
    edit["provenance"].pop("reuse_identity")
    path.write_text(json.dumps(edit))
    with pytest.raises(EntryIdentityMismatch, match="reuse_key"):
        build_zoo(cfg, git_sha="s")


def test_generator_resolving_a_different_base_is_rejected(
    tmp_path, fake_generators, toy_base, monkeypatch
):
    real = sys.modules["wsp.zoo.lora_sft"].generate

    def drifting(base_model, params, seed, out_dir):
        desc = real(base_model, params, seed, out_dir)
        desc.base_revision = "c" * 40
        return desc

    monkeypatch.setattr(sys.modules["wsp.zoo.lora_sft"], "generate", drifting)
    cfg = _cfg(tmp_path, _fams("lora_sft"))
    with pytest.raises(EntryIdentityMismatch, match="resolved"):
        build_zoo(cfg, git_sha="s")
    zoo = tmp_path / "zoo"
    assert not [p for p in zoo.iterdir() if (p / "edit.json").exists()]  # nothing exposed


def _fake_refusal(fp: str):
    dirs = torch.nn.functional.normalize(
        torch.randn(3, 8, generator=torch.Generator().manual_seed(1))
    )
    return SimpleNamespace(
        directions=dirs, fingerprint=fp, manifest_path=Path(f"refusal.{fp}.json")
    )


def test_refusal_fingerprint_is_part_of_abliteration_entry_identity(
    tmp_path, fake_generators, toy_base, monkeypatch
):
    fp = {"v": "fp-A"}
    monkeypatch.setattr(B, "get_refusal_directions", lambda *a, **k: _fake_refusal(fp["v"]))
    fams = {
        "abliteration": {
            "n": 1,
            "layers": [0, 1],
            "direction_index": {"min": 0, "max": 1},
            "max_weight": {"min": 0.5, "max": 1.0},
            "max_weight_position": {"min": 0, "max": 1},
            "min_weight": {"min": 0.0, "max": 0.2},
            "min_weight_distance": {"min": 1, "max": 2},
            "components": ["mlp"],
        }
    }
    cfg = _cfg(tmp_path, fams)
    rows = build_zoo(cfg, git_sha="s")
    assert _edit(tmp_path / "zoo", rows[0]["id"])["provenance"]["refusal_fingerprint"] == "fp-A"
    assert rows[0]["refusal_fingerprint"] == "fp-A"
    key_a = rows[0]["reuse_key"]
    assert [r["reuse_key"] for r in build_zoo(cfg, git_sha="s")] == [key_a]  # reuse

    fp["v"] = "fp-B"  # e.g. changed refusal seed / system prompt: same sampled params, same id
    with pytest.raises(EntryIdentityMismatch, match="reuse_key"):
        build_zoo(cfg, git_sha="s")
    edit = _edit(tmp_path / "zoo", rows[0]["id"])
    assert edit["provenance"]["refusal_fingerprint"] == "fp-A"  # old entry not overwritten
    manifest_rows = _rows(tmp_path / "zoo")
    assert [r["reuse_key"] for r in manifest_rows] == [key_a]


def test_dry_run_ids_equal_real_ids_with_pinned_base(tmp_path, fake_generators, toy_base):
    cfg = _cfg(tmp_path, _fams("lora_sft", "random_dir", "chain", "task_vector"))
    dry = {r["id"] for r in build_zoo(cfg, dry_run=True, git_sha="s")}
    real = {r["id"] for r in build_zoo(cfg, git_sha="s")}
    assert dry == real


# ---- finding 2: parent eligibility --------------------------------------------------------------


def test_smoke_budget_and_overrides_are_recorded_as_eligibility(
    tmp_path, fake_generators, toy_base
):
    zoo = tmp_path / "zoo"
    cfg = _cfg(tmp_path, _fams("heretic_ara", "lora_sft"))
    rows = build_zoo(cfg, git_sha="s")
    assert {r["eligibility"] for r in rows} == {"production"}
    assert all(_edit(zoo, r["id"])["eligibility"] == "production" for r in rows)

    smoke = build_zoo(cfg, git_sha="s", overrides=["heretic_ara.trials=2", "lora_sft.steps=5"])
    new = [r for r in smoke if r["id"] not in {x["id"] for x in rows}]
    assert len(new) == 4 and {r["eligibility"] for r in new} == {"smoke"}
    for r in new:
        edit = _edit(zoo, r["id"])
        assert edit["eligibility"] == "smoke" and edit["eligibility_reasons"]
        assert any("override" in x for x in edit["eligibility_reasons"])


def test_trial_budget_below_threshold_is_smoke_even_without_override(
    tmp_path, fake_generators, toy_base
):
    cfg = _cfg(tmp_path, _fams("heretic_ara", heretic_ara={"trials": 2}))
    (row,) = build_zoo(cfg, git_sha="s", n=1)
    edit = _edit(tmp_path / "zoo", row["id"])
    assert row["eligibility"] == "smoke"
    assert any("trials" in x for x in edit["eligibility_reasons"])

    # thresholds are configurable
    raw = yaml.safe_load(cfg.read_text())
    raw["families"]["heretic_ara"]["trials"] = 40
    raw["eligibility"] = {"min_params": {"heretic_ara": {"trials": 100}}}
    cfg.write_text(yaml.safe_dump(raw))
    old = row["id"]
    (row,) = [r for r in build_zoo(cfg, git_sha="s", n=1) if r["id"] != old]
    assert row["eligibility"] == "smoke"


def _inject_row(zoo: Path, row: dict, with_artifacts=True):
    d = zoo / row["id"]
    d.mkdir(parents=True)
    (d / "edit.json").write_text(json.dumps({"id": row["id"], "family": row["family"]}))
    if with_artifacts:
        from safetensors.torch import save_file

        save_file(
            {"model.layers.0.mlp.down_proj.weight": torch.ones(8, 8)}, d / "delta.safetensors"
        )
    with (zoo / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _tv_cfg(tmp_path):
    return _cfg(tmp_path, _fams("lora_sft", "task_vector", task_vector={"n": 12}))


def _tv_parents(rows):
    return {r["parent_id"] for r in rows if r["family"] == "task_vector"}


def test_smoke_legacy_and_incompatible_base_parents_never_enter_production_sampling(
    tmp_path, fake_generators, toy_base, pinned_base_revision
):
    cfg = _tv_cfg(tmp_path)
    zoo = tmp_path / "zoo"
    sha = pinned_base_revision("Qwen/Qwen3.5-0.8B", "main")
    # smoke LoRA entries, built by an explicitly overridden run into the same zoo
    smoke_rows = [
        r
        for r in build_zoo(cfg, git_sha="s", family="lora_sft", overrides=["lora_sft.steps=5"])
        if r["eligibility"] == "smoke"
    ]
    # legacy row (pre-eligibility builder: no eligibility/reuse_key/base_revision) with artefacts
    legacy = {"id": "legacy0000000001", "family": "heretic_ara", "seed": 0, "config_hash": "x"}
    _inject_row(zoo, legacy)
    # production-labelled row from another base commit
    other = {**legacy, "id": "otherbase0000001", "eligibility": "production"}
    other.update(base_revision="d" * 40, reuse_key="k")
    _inject_row(zoo, other)
    prod_rows = [
        r for r in build_zoo(cfg, git_sha="s", family="lora_sft") if r.get("base_revision") == sha
    ]
    prod_ids = {r["id"] for r in prod_rows if r["eligibility"] == "production"}
    assert prod_ids and smoke_rows

    rows = build_zoo(cfg, git_sha="s", family="task_vector")
    parents = _tv_parents(rows)
    assert parents and parents <= prod_ids
    tv = next(r for r in rows if r["family"] == "task_vector")
    assert tv["eligibility"] == "production"
    prov = _edit(zoo, tv["id"])["provenance"]
    assert prov["include_smoke"] is False
    assert sum(prov["parent_pool_excluded"].values()) >= 3


def test_include_smoke_is_an_explicit_opt_in_but_never_admits_legacy_or_other_base(
    tmp_path, fake_generators, toy_base
):
    cfg = _tv_cfg(tmp_path)
    zoo = tmp_path / "zoo"
    smoke_ids = {
        r["id"]
        for r in build_zoo(cfg, git_sha="s", family="lora_sft", overrides=["lora_sft.steps=5"])
        if r["eligibility"] == "smoke"
    }
    legacy = {"id": "legacy0000000001", "family": "heretic_ara", "seed": 0, "config_hash": "x"}
    _inject_row(zoo, legacy)

    # Only smoke candidates exist: production sampling finds no parent -> no task vectors.
    assert not _tv_parents(build_zoo(cfg, git_sha="s", family="task_vector"))

    rows = build_zoo(cfg, git_sha="s", family="task_vector", include_smoke=True)
    parents = _tv_parents(rows)
    assert parents and parents <= smoke_ids
    tvs = [r for r in rows if r["family"] == "task_vector"]
    assert {r["eligibility"] for r in tvs} == {"smoke"}  # children of smoke parents are smoke
    assert _edit(zoo, tvs[0]["id"])["provenance"]["include_smoke"] is True


def test_dry_run_parent_pool_uses_the_same_eligibility_filter(tmp_path, fake_generators, toy_base):
    cfg = _tv_cfg(tmp_path)
    build_zoo(cfg, git_sha="s", family="lora_sft", dry_run=True, overrides=["lora_sft.steps=5"])
    dry = build_zoo(cfg, git_sha="s", family="task_vector", dry_run=True)
    assert not _tv_parents(dry)  # only smoke planned entries exist


def test_cli_exposes_include_smoke(monkeypatch):
    seen = {}
    monkeypatch.setattr(B, "build_zoo", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(sys, "argv", ["build", "--config", "c.yaml", "--include-smoke"])
    B.main()
    assert seen["include_smoke"] is True
    monkeypatch.setattr(sys, "argv", ["build", "--config", "c.yaml"])
    B.main()
    assert seen["include_smoke"] is False


# ---- finding 3: chain step semantics -----------------------------------------------------------


def _chains(tmp_path, overrides=None, **extra):
    fams = _fams("heretic_ara", "lora_sft", "random_dir", "chain", chain={"n": 6})
    cfg = _cfg(tmp_path, fams)
    return build_zoo(cfg, git_sha="s", family="chain", overrides=overrides, **extra)


def test_chain_steps_record_lora_vs_base_and_stored_format(tmp_path, fake_generators, toy_base):
    rows = _chains(tmp_path)
    zoo = tmp_path / "zoo"
    seen = set()
    for r in rows:
        edit = _edit(zoo, r["id"])
        inner = edit["params"]["inner_family"]
        want = "lora_vs_base" if inner in ("heretic_ara", "lora_sft") else "sequential"
        seen.add(want)
        assert edit["params"]["step_semantics"] == want  # id-bearing
        prov = edit["provenance"]
        assert prov["step_semantics"] == want == r["step_semantics"]
        assert prov["chain_step_semantics"][-1] == want
        assert len(prov["chain_step_semantics"]) == edit["params"]["step"] + 1
        assert r["chain_step_semantics"] == prov["chain_step_semantics"]
        # stored vs source format are separate, and a dense chain artefact never claims "lora"
        assert prov["stored_delta_format"] == "dense_cumulative" == r["stored_delta_format"]
        assert prov["cumulative_snapshot"] is True
        assert "delta_format" not in prov and "delta_format" not in r
        assert prov["source_delta_format"] == ("lora" if inner == "lora_sft" else "dense")
        assert (zoo / r["id"] / "delta.safetensors").exists()
    assert seen == {"lora_vs_base", "sequential"}


def test_chain_dry_run_ids_match_real_and_carry_semantics(tmp_path, fake_generators, toy_base):
    fams = _fams("heretic_ara", "lora_sft", "random_dir", "chain", chain={"n": 4})
    cfg = _cfg(tmp_path, fams)
    dry = build_zoo(cfg, git_sha="s", family="chain", dry_run=True)
    real = build_zoo(cfg, git_sha="s", family="chain")
    assert [r["id"] for r in dry] == [r["id"] for r in real]
    assert [r["step_semantics"] for r in dry] == [r["step_semantics"] for r in real]


def test_chain_with_smoke_step_is_smoke_from_that_step_on(tmp_path, fake_generators, toy_base):
    rows = _chains(tmp_path, overrides=["lora_sft.steps=5"])
    zoo = tmp_path / "zoo"
    by_chain: dict[str, list[dict]] = {}
    for r in rows:
        edit = _edit(zoo, r["id"])
        by_chain.setdefault(edit["seed"], []).append((edit, r))
    saw_smoke = False
    for steps in by_chain.values():
        smoke_yet = False
        for edit, r in steps:
            smoke_yet |= edit["params"]["inner_family"] == "lora_sft"
            want = "smoke" if smoke_yet else "production"
            assert r["eligibility"] == edit["eligibility"] == want
            saw_smoke |= smoke_yet
    assert saw_smoke


def test_production_row_is_recomputed_against_current_thresholds():
    from wsp.zoo.build import parent_exclusion_reason

    config = {
        "base_revision": "abc",
        "eligibility": {"min_params": {"heretic_ara": {"trials": 20}}},
    }
    row = {
        "family": "heretic_ara",
        "params": {"trials": 10},
        "eligibility": "production",  # stored under looser thresholds
        "reuse_key": "k",
        "base_revision": "abc",
    }
    assert parent_exclusion_reason(row, config, include_smoke=False, dry_run=False) == (
        "smoke_under_current_thresholds"
    )
    assert parent_exclusion_reason(row, config, include_smoke=True, dry_run=False) is None
    row_ok = {**row, "params": {"trials": 40}}
    assert parent_exclusion_reason(row_ok, config, include_smoke=False, dry_run=False) is None
