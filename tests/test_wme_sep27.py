"""wm.sep27: 27B separation (Board A condition 13; PREREG_WME §10). Synthetic pools and manifests only: no 27B
model, no panel built from real items, no outcome."""

import json

import numpy as np
import pytest

from wsp.wm import panel_c1 as C1
from wsp.wm import sep27 as S27


def test_identities_are_fresh():
    assert S27.SALT == "wme-c27" and S27.BANK_NAMESPACE == "wme-bank27"
    assert S27.SALT != C1.SALT
    assert S27.seed_for("roots") != S27.seed_for("roots2")
    from wsp.wm import bank as B

    assert S27.seed_for("roots") != B.seed_for("roots")


def test_registry_requires_every_exposure_including_c1_sealed_and_calibration():
    reg = S27.EXPOSURES_27
    assert set(S27.REQUIRED_EXPOSURES) <= set(reg)
    assert (
        reg["wme-c1"]["splits"] == "all"
    )  # sealed test included (via the manifest's sealed_hashes)
    assert reg["calib27-20261116"]["splits"] == ["construction", "validation", "dev"]
    assert {"wme-v0-p1", "wme-p2"} <= set(S27.REQUIRED_EXPOSURES)
    # SYNTHESIS-r11 R7: the pilot-3 panels are exposures for every later panel, the 27B one included
    assert {"wme-p3s", "wme-p3h"} <= set(S27.REQUIRED_EXPOSURES)
    # RULING-wme-post-stagew Q4: the stage-W dev panel is an exposure for every later panel, the 27B one included
    assert "wme-w" in S27.REQUIRED_EXPOSURES


def _write_manifest(path, by_split, sealed=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    man = {
        "objectives": {"gsm8k_gold_lp": {s: {"hashes": list(h)} for s, h in by_split.items()}},
        "sealed_hashes": list(sealed),
    }
    path.write_text(json.dumps(man))


def _fake_root(tmp_path, with_c1=True):
    _write_manifest(
        tmp_path / "data/wme/panels/wme-v0-p1/manifest.json",
        {"construction": ["p1c"], "dev": ["p1d"]},
    )
    _write_manifest(
        tmp_path / "data/wme/panels/wme-p2/manifest.json", {"construction": ["p2c"]}, sealed=["p2t"]
    )
    for p3 in ("wme-p3s", "wme-p3h", "wme-w", "wme-w2"):
        _write_manifest(tmp_path / f"data/wme/panels/{p3}/manifest.json", {"construction": [p3]})
    if with_c1:
        _write_manifest(
            tmp_path / "data/wme/panels/wme-c1/manifest.json", {"dev": ["c1d"]}, sealed=["c1t"]
        )
    return tmp_path


def test_load_exposures_never_skips_a_missing_manifest(tmp_path):
    ex = S27.load_exposures27(_fake_root(tmp_path))
    assert "c1t" in ex["wme-c1"]["test"] and "p2t" in ex["wme-p2"]["test"]
    assert "wme-p3h" in ex["wme-p3h"]["construction"]
    with pytest.raises(S27.MissingExposure):
        S27.load_exposures27(_fake_root(tmp_path / "b", with_c1=False))


def _pool(n):
    return {
        "gsm8k_gold_lp": [
            {
                "id": f"g{i}",
                "hash": f"h{i:03d}",
                "question": f"q{i} " + " ".join(f"w{i}x{k}" for k in range(6)),
                "gold": "1",
            }
            for i in range(n)
        ]
    }


def test_build_c27_panel_uses_fresh_salt_and_excludes_by_manifest_identity(tmp_path):
    pool = _pool(40)
    exposed = {"h000", "h001", "h002", "h003"}
    exposures = {
        "wme-c1": {"test": {"h000"}, "dev": {"h001"}},
        "wme-v0-p1": {"construction": {"h002"}},
        "wme-p2": {"test": {"h003"}},
        "calib27-20261116": {"dev": set()},
        "wme-p3s": {"construction": set()},
        "wme-p3h": {"dev": set()},
        "wme-w": {"dev": set()},
        "wme-w2": {"dev": set()},
    }
    sizes = {"gsm8k_gold_lp": {"construction": 4, "validation": 4, "dev": 4, "test": 4}}
    man = S27.build_c27_panel(pool, sizes, tmp_path / "c27", exposures)
    assert man["salt"] == "wme-c27" and man["audit_clean"]
    used = {h for s in man["objectives"]["gsm8k_gold_lp"].values() for h in s["hashes"]}
    assert not used & exposed
    assert man["experiment"] == S27.EXPERIMENT
    c1 = C1.build_c1_panel(pool, sizes, tmp_path / "c1", exposures)
    assert (
        c1["objectives"]["gsm8k_gold_lp"]["test"]["hashes"]
        != man["objectives"]["gsm8k_gold_lp"]["test"]["hashes"]
    )


def test_build_c27_panel_refuses_incomplete_exclusions(tmp_path):
    exposures = {
        "wme-v0-p1": {},
        "wme-p2": {},
        "calib27-20261116": {},
    }  # no wme-c1: a fresh salt alone is not enough
    with pytest.raises(S27.MissingExposure):
        S27.build_c27_panel(
            _pool(20),
            {"gsm8k_gold_lp": {"construction": 1, "validation": 1, "dev": 1, "test": 1}},
            tmp_path / "x",
            exposures,
        )


# ------------------------------------------------------------------ roots


def _root(rid, **kw):
    return dict({"id": rid, "model": S27.MODEL, "namespace": S27.BANK_NAMESPACE}, **kw)


def test_calibration_timing_edits_are_registered():
    ids = {e["id"] for e in S27.CALIB_TIMING_EDITS}
    assert ids == {"calib27:noise", "calib27:scale"}
    noise = next(e for e in S27.CALIB_TIMING_EDITS if e["family"] == "noise")
    assert noise["tensor"] == [28, "down"] and noise["seed"] == 7 and noise["rho"] == 0.05
    scale = next(e for e in S27.CALIB_TIMING_EDITS if e["family"] == "scale")
    assert scale["tensor"] == [27, "o"] and scale["factor"] == 1.25


def test_root_exclusion_J2_to_J5_and_namespace():
    excl = S27.excluded_identities(
        nine_b_roots=[
            {"id": "9b:steer", "model": "Qwen/Qwen3.5-9B", "contrast_source_groups": ["gA", "gB"]},
            {"id": "9b:lora", "model": "Qwen/Qwen3.5-9B", "adapter_source_groups": ["gL"]},
        ]
    )
    u = np.zeros(16)
    u[0] = 1
    v = np.zeros(16)
    v[:2] = [0.6, 0.8]
    cands = [
        _root("ok", family="steer", contrast_source_groups=["gZ"]),
        _root("j2", family="steer", contrast_source_groups=["gB", "gC"]),
        _root("j2l", family="lora", adapter_source_groups=["gL"]),
        _root("j4n", family="noise", tensor=[28, "down"], seed=S27.seed_for("n")),
        _root("j4s", family="scale", tensor=[27, "o"]),
        _root("j3a", family="steer", direction=u, space="27b:resid"),
        _root("j3b", family="steer", direction=v, space="27b:resid"),  # |cos| = 0.6 with j3a
        _root("rowA", family="row_scale", tensor="lm_head", rows=list(range(10))),
        _root(
            "rowB", family="row_scale", tensor="lm_head", rows=list(range(3, 13))
        ),  # Jaccard 7/13 >= 0.5
        _root("j5", family="noise", tensor=[30, "down"], realised_delta=["d1"]),
        _root("j5b", family="noise", tensor=[31, "down"], realised_delta=["d1"]),
        _root("ns", family="steer", namespace="wme-bank", contrast_source_groups=["gQ"]),
    ]
    acc, rej = S27.accept_roots27(cands, excl)
    assert [r["id"] for r in acc] == ["ok", "j3a", "rowA", "j5"]
    reasons = {r["id"]: r["reasons"] for r in rej}
    assert reasons["j2"] == ["J2:9b:steer"] and reasons["j2l"] == ["J2:9b:lora"]
    assert reasons["j4n"] == ["J4:calib27:noise"] and reasons["j4s"] == ["J4:calib27:scale"]
    assert (
        reasons["j3b"] == ["J3:j3a"]
        and reasons["rowB"] == ["J4:rowA"]
        and reasons["j5b"] == ["J5:j5"]
    )
    assert reasons["ns"] == ["namespace"]


# ------------------------------------------------------------------ primary selection


def _res(obj, lcb, breach=False, **prov):
    p = {"model": S27.MODEL, "panel": S27.SALT, "split": "dev"}
    p.update(prov)
    return {"objective": obj, "lcb": lcb, "guard_breach": breach, "provenance": p}


def test_primary_selection_rule_and_relation():
    rs = (
        [_res("sycophancy_real", 0.1)] * 3
        + [_res("gsm8k_gold_lp", 0.1)] * 2
        + [_res("mathcomp_gold_lp", 0.1, breach=True)] * 5
    )
    out = S27.select_primary_27(rs)
    assert out["qualifying"] == {"sycophancy_real": 3, "gsm8k_gold_lp": 2, "mathcomp_gold_lp": 0}
    assert out["primary"] == "sycophancy_real" and out["relation"] == S27.RELATION["replication"]
    tie = [_res("sycophancy_real", 0.1)] * 2 + [_res("gsm8k_gold_lp", 0.1)] * 2
    t = S27.select_primary_27(tie)
    assert (
        t["primary"] == "gsm8k_gold_lp" and t["relation"] == S27.RELATION["separate"]
    )  # (iii) > (ii) > (i)
    none = S27.select_primary_27([_res("gsm8k_gold_lp", 0.1), _res("sycophancy_real", -0.1)])
    assert none["primary"] is None and none["relation"] == S27.RELATION["none"]


@pytest.mark.parametrize(
    "prov",
    [
        {"model": "Qwen/Qwen3.5-9B"},
        {"panel": "wme-c1-panel-20261117"},
        {"split": "test"},
        {"split": "sealed_test"},
    ],
)
def test_primary_selection_refuses_non_27b_dev_inputs(prov):
    with pytest.raises(S27.SeparationError):
        S27.select_primary_27([_res("gsm8k_gold_lp", 0.1, **prov)] * 3)


# ------------------------------------------------------------------ tuning guard


def test_tuning_guard_forbids_9b_test_and_generation_outcomes():
    ok = S27.guard_tuning(
        [
            {"element": "B", "value": 32, "sources": [{"kind": "27b_pilot_dev", "ref": "p"}]},
            {
                "element": "margins",
                "value": "same",
                "sources": [{"kind": "registration", "ref": "PREREG_WME §5"}],
            },
            {
                "element": "families",
                "value": 3,
                "sources": [{"kind": "9b_confirmatory_analysis", "ref": "H-T"}],
            },
        ]
    )
    assert ok[0]["label"] == ok[1]["label"] == "independent of 9B confirmatory results"
    assert ok[2]["label"] == "informed by 9B confirmatory results" and ok[2]["adaptation"] is True
    for kind in (
        "9b_test_outcome",
        "9b_generation_outcome",
        "9b_winner_test_score",
        "9b_sealed_read",
    ):
        with pytest.raises(S27.Tuning9BError):
            S27.guard_tuning(
                [{"element": "N", "value": 1, "sources": [{"kind": kind, "ref": "x"}]}]
            )
    with pytest.raises(ValueError):
        S27.guard_tuning([{"element": "N", "value": 1, "sources": [{"kind": "vibes", "ref": "x"}]}])
    with pytest.raises(ValueError):
        S27.guard_tuning(
            [{"element": "colour", "value": 1, "sources": [{"kind": "registration", "ref": "x"}]}]
        )
    with pytest.raises(ValueError):
        S27.guard_tuning([{"element": "N", "value": 1, "sources": []}])  # provenance is required


def test_read_guard_refuses_9b_outcomes_lexically_and_through_symlinks(tmp_path):
    root = tmp_path
    forb = root / "data/wme/runs/wme-c1"
    forb.mkdir(parents=True)
    (forb / "test_scores.json").write_text("{}")
    ok = root / "data/wme/c27"
    ok.mkdir(parents=True)
    (ok / "pilot.json").write_text('{"a": 1}')
    (ok / "link.json").symlink_to(forb / "test_scores.json")
    (root / "data/wme/panels/wme-c1/sealed_test").mkdir(parents=True)
    (root / "data/wme/panels/wme-c1/sealed_test/test.jsonl").write_text("")
    g = S27.Reads27(root)
    assert g.read_json(ok / "pilot.json") == {"a": 1}
    for p in (
        forb / "test_scores.json",
        ok / "link.json",
        root / "data/wme/panels/wme-c1/sealed_test/test.jsonl",
    ):
        with pytest.raises(S27.SeparationError):
            g.read_json(p)
    assert [r["path"] for r in g.reads] == [str(ok / "pilot.json")]


# ------------------------------------------------------------------ field configuration and launch gate


class _RM:
    def writers(self):
        return {
            (
                L,
                r,
            ): f"model.language_model.layers.{L}.{'mlp.down_proj' if r == 'down' else 'self_attn.o_proj'}.weight"
            for L in range(64)
            for r in ("down", "o")
        }


def test_two_layer_group_field_configuration():
    cfg = S27.field_config_27(_RM())
    assert cfg["layers_per_group"] == 2 and cfg["max_item_tokens"] == 768
    assert len(cfg["groups"]) == 2 and all(
        len(g) == 4 for g in cfg["groups"]
    )  # 2 layers x (down, o)
    assert any("layers.24." in m for m in cfg["groups"][0]) and any(
        "layers.27." in m for m in cfg["groups"][1]
    )
    assert cfg["go_m"]["peak_gb"] == 68.81 and cfg["go_m"]["cap_gb"] == 72.0
    S27.validate_field_config(2, 768)
    for bad in ((4, 768), (1, 768), (2, 1024)):
        with pytest.raises(ValueError):
            S27.validate_field_config(*bad)
    S27.check_field_items([12, 768])
    with pytest.raises(ValueError):
        S27.check_field_items([769])


def test_launch_gate_requires_sealed_9b_and_complete_amendment():
    amend = {k: "x" for k in S27.AMENDMENT_FIELDS}
    amend["differences"] = [{"what": "B", "label": "independent of 9B confirmatory results"}]
    S27.launch_gate_27({"sealed": True, "hash": "a" * 64}, amend)
    with pytest.raises(S27.SeparationError):
        S27.launch_gate_27(None, amend)
    with pytest.raises(S27.SeparationError):
        S27.launch_gate_27({"sealed": False, "hash": "a" * 64}, amend)
    with pytest.raises(S27.SeparationError):
        S27.launch_gate_27(
            {"sealed": True, "hash": "a" * 64},
            {k: v for k, v in amend.items() if k != "final_panel"},
        )
    bad = dict(amend, differences=[{"what": "B", "label": "tuned"}])
    with pytest.raises(S27.SeparationError):
        S27.launch_gate_27({"sealed": True, "hash": "a" * 64}, bad)
