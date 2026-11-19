"""wm.exposure: the item-exposure and split contract (Board A condition 1).

* every split of wme-c1 has registered roles; bank labels never use dev or test items;
* all transformations of one question (neutral/suggested, templates) share one group;
* a frozen near-duplicate policy (word-5-gram Jaccard >= 0.7 via MinHash LSH, verified exactly);
* hash-only exposure audits from manifests only (never a sealed or listed file);
* group-level exclusion and feasible split sizes.
"""

import json
import os
from pathlib import Path

import pytest

from wsp.wm import exposure as X
from wsp.wm.plan import SealedError


def test_split_roles_cover_every_use_and_never_put_labels_on_dev_or_test():
    roles = X.SPLIT_ROLES
    assert set(roles) == {"construction", "validation", "dev", "test"}
    used = [r for rs in roles.values() for r in rs]
    assert len(used) == len(set(used))  # each role has one split
    for need in X.REQUIRED_ROLES:
        assert need in used
    assert X.split_of("bank_labels") == "validation"
    assert X.split_of("lora_training") == "construction"
    assert X.split_of("search_feedback") == "dev"
    assert X.split_of("final_scoring") == "test"
    assert "trajectory" in X.SEPARATION["validation"]


def test_group_key_keeps_question_transformations_together():
    a = {
        "question": "What is  the capital of France?",
        "neutral_prompt": "x",
        "suggested_prompt": "y",
    }
    b = {
        "question": "what is the capital of france?",
        "neutral_prompt": "z",
        "suggested_prompt": "w",
    }
    assert X.group_key("sycophancy_real", a) == X.group_key("sycophancy_real", b)
    m1 = {"question": "Q", "options": ["a", "b"]}
    assert X.group_key("mmlupro_gold_lp", m1) == X.group_key("sycophancy_gap", m1)  # shared source


def test_near_duplicates_are_merged_and_distinct_texts_are_not():
    base = "the quick brown fox jumps over the lazy dog near the river bank today"
    texts = {
        "a": base,
        "b": base + " again",  # near-duplicate
        "c": "a completely different sentence about prime numbers and their gaps in arithmetic",
        "d": "short",
        "e": "Short",  # exact after normalisation
    }
    groups = X.near_duplicate_groups(texts)
    assert groups["a"] == groups["b"] and groups["a"] != groups["c"]
    assert groups["d"] == groups["e"] and groups["d"] != groups["a"]
    assert X.jaccard(X.shingles(base), X.shingles(base + " again")) >= X.NEAR_DUP_JACCARD
    assert X.near_duplicate_groups(texts) == groups  # deterministic


def _manifest(path, splits):
    man = {
        "objectives": {"o": {s: {"n": len(h), "hashes": sorted(h)} for s, h in splits.items()}},
        "sealed_hashes": sorted(splits.get("test", [])),
        "files": {"sealed_test/test.jsonl": "0" * 64},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man))
    return path


def test_manifest_reader_reads_hash_lists_only_and_refuses_other_paths(tmp_path):
    m = _manifest(tmp_path / "p" / "manifest.json", {"test": {"t1"}, "dev": {"d1"}})
    sealed = tmp_path / "p" / "sealed_test"
    sealed.mkdir()
    (sealed / "test.jsonl").write_text("SECRET\n")
    os.chmod(sealed / "test.jsonl", 0)  # would raise if anything tried to open it
    hs = X.manifest_hashes(m)
    assert hs == {"test": {"t1"}, "dev": {"d1"}}
    with pytest.raises(SealedError):
        X.manifest_hashes(sealed / "test.jsonl")
    other = tmp_path / "p" / "notes.json"
    other.write_text("{}")
    with pytest.raises(SealedError):
        X.manifest_hashes(other)
    link = tmp_path / "p" / "link" / "manifest.json"
    link.parent.mkdir()
    link.symlink_to(m)
    with pytest.raises(OSError):
        X.manifest_hashes(link)  # O_NOFOLLOW on the leaf


def test_exposure_registry_names_every_pilot_and_the_27b_calibration():
    names = set(X.EXPOSURES)
    assert {"wme-v0-p1", "wme-p2", "calib27-20261116", "wme-p2-smoke"} <= names
    assert X.EXPOSURES["calib27-20261116"]["manifest"].endswith("wme-v0-p1/manifest.json")


def test_intersection_audit_and_group_level_exclusion(tmp_path):
    expos = {"p1": {"test": {"h1"}, "construction": {"h2"}}, "p2": {"dev": {"h9"}}}
    pool = [
        {"hash": "h1", "group": "g1"},
        {"hash": "h1b", "group": "g1"},  # same group as an exposed item -> excluded too
        {"hash": "h3", "group": "g3"},
        {"hash": "h2", "group": "g2"},
    ]
    kept, report = X.exclude_groups(pool, expos)
    assert [r["hash"] for r in kept] == ["h3"]
    assert report["excluded_groups"] == 2 and report["excluded_items"] == 3
    audit = X.intersection_audit({"construction": {"h3"}, "test": {"h1"}}, expos)
    assert audit["p1"]["test"]["test"] == 1 and audit["p1"]["construction"]["construction"] == 0
    assert not X.audit_clean(audit)
    assert X.audit_clean(X.intersection_audit({"construction": {"h3"}}, expos))


def test_feasible_sizes_report_shortfalls_and_never_relax_exclusions():
    rep = X.feasibility({"sycophancy_real": 469, "gsm8k_gold_lp": 5000}, X.DEFAULT_SIZES)
    syc = rep["sycophancy_real"]
    assert syc["available_groups"] == 469 and syc["requested"] == sum(
        X.DEFAULT_SIZES["sycophancy_real"].values()
    )
    assert syc["feasible"] == (syc["requested"] <= 469)
    assert rep["gsm8k_gold_lp"]["feasible"]
    assert sum(X.DEFAULT_SIZES["sycophancy_real"].values()) <= 469


def test_pilot3_panels_are_registered_exposures_with_every_split():
    """SYNTHESIS-r11 R7 / RULING-wme-post-pilot3 P5: wme-p3s and wme-p3h, all splits (unscored dev rests included)."""
    for name in ("wme-p3s", "wme-p3h"):
        spec = X.EXPOSURES[name]
        assert spec["splits"] == "all"
        assert spec["manifest"] == f"data/wme/panels/{name}/manifest.json"
    assert X.EXPOSURES["wme-w"] == {
        "manifest": "data/wme/panels/wme-w/manifest.json",
        "splits": "all",
        "note": X.EXPOSURES["wme-w"]["note"],
    }
    # every earlier exposure is still registered
    assert {"wme-v0-p1", "wme-p2", "calib27-20261116", "wme-p2-smoke"} <= set(X.EXPOSURES)


def test_pilot3_registration_reads_holdout_and_dev_rest_hashes(tmp_path):
    """'all' keeps every split a manifest lists, including non-standard ones such as wme-p3h's holdout."""
    for name in ("wme-p3s", "wme-p3h"):
        d = tmp_path / "data/wme/panels" / name
        d.mkdir(parents=True)
        objs = {
            "a": {"construction": {"hashes": [f"{name}c"]}, "holdout": {"hashes": [f"{name}h"]}},
            "b": {
                "dev": {"hashes": [f"{name}d1", f"{name}d2"]},
                "validation": {"hashes": [f"{name}v"]},
            },
        }
        (d / "manifest.json").write_text(json.dumps({"objectives": objs, "sealed_hashes": []}))
    reg = {k: X.EXPOSURES[k] for k in ("wme-p3s", "wme-p3h")}
    ex = X.load_exposures(tmp_path, reg)
    assert set(ex["wme-p3h"]) == {"construction", "holdout", "dev", "validation", "test"}
    assert {"wme-p3hh", "wme-p3hd2"} <= X.all_exposed(ex)
    assert len(X.all_exposed(ex)) == 10


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/wme/panels/wme-p3h/manifest.json").exists(),
    reason="pilot-3 panel manifests not present",
)
def test_real_pilot3_manifests_load_and_cover_every_used_split():
    root = Path(__file__).resolve().parents[1]
    reg = {k: X.EXPOSURES[k] for k in ("wme-p3s", "wme-p3h")}
    ex = X.load_exposures(root, reg)
    assert {"construction", "validation"} <= set(ex["wme-p3s"])
    assert {"construction", "holdout", "dev"} <= set(ex["wme-p3h"])
    assert (
        len(ex["wme-p3h"]["dev"]) >= 2000
    )  # MC dev pools plus guard dev rests (unscored items included)


STAGEW_SOURCES = {
    # every item source stage W touched (RULING-wme-stagew §2-§4) -> the registered exposure that covers it
    "wme-w dev panel (500 MMLU-Pro + 500 SuperGPQA, scored for every member)": "wme-w",
    "base canary (64 wme-p3h holdout MMLU-Pro items)": "wme-p3h",
    "timing pilot (wme-p3h construction and holdout MC items)": "wme-p3h",
    "teacher rows and the 64 R1 feature items (wme-p3h construction)": "wme-p3h",
}


def test_every_stagew_item_source_is_a_registered_all_split_exposure():
    """RULING-wme-post-stagew Q4: W2 panels must be disjoint from every exposure, stage W included."""
    for src, name in STAGEW_SOURCES.items():
        assert name in X.EXPOSURES, src
        assert X.EXPOSURES[name]["splits"] == "all", src


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/wme/panels/wme-w/manifest.json").exists(),
    reason="stage-W panel manifest not present",
)
def test_real_stagew_manifest_loads_every_dev_hash():
    root = Path(__file__).resolve().parents[1]
    ex = X.load_exposures(root, {"wme-w": X.EXPOSURES["wme-w"]})
    assert len(ex["wme-w"]["dev"]) == 1000  # 500 per benchmark
    assert X.all_exposed(ex) == ex["wme-w"]["dev"]


def test_stagew2_panel_is_registered_before_any_w2_generation():
    """RULING-wme-stagew2 §2: wme-w2 (all splits) is an exposure for every later panel (H-T3, wme-c2-gen, 27B)."""
    assert X.EXPOSURES["wme-w2"]["manifest"] == "data/wme/panels/wme-w2/manifest.json"
    assert X.EXPOSURES["wme-w2"]["splits"] == "all"
    from wsp.wm import sep27 as S27

    assert "wme-w2" in S27.REQUIRED_EXPOSURES


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/wme/panels/wme-w2/manifest.json").exists(),
    reason="stage-W2 panel manifest not present",
)
def test_real_stagew2_manifest_is_disjoint_from_every_earlier_exposure():
    root = Path(__file__).resolve().parents[1]
    w2 = X.load_exposures(root, {"wme-w2": X.EXPOSURES["wme-w2"]})
    assert len(w2["wme-w2"]["dev"]) == 1000
    # the confirmatory panels are built after W2 and exclude it by construction; W2 is checked against the
    # exploratory exposures that existed before it
    earlier = {
        k: v for k, v in X.EXPOSURES.items() if k != "wme-w2" and k not in X.CONFIRMATORY_PANELS
    }
    assert not X.all_exposed(w2) & X.all_exposed(X.load_exposures(root, earlier))
