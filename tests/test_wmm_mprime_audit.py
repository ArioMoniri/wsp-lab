import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/wmm_mprime_audit.py")
AU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AU)


def test_summarise_agreement_and_unlabelled():
    keys = [f"msyc|harm|i{i}" for i in range(10)]
    rule = {k: "refuse" for k in keys}
    a = {k: "refuse" for k in keys}
    b = {k: "refuse" for k in keys}
    a[keys[0]] = "unlabelled"  # judge A declined to label one item
    b[keys[1]] = "comply"  # judges disagree on one item
    a[keys[2]] = b[keys[2]] = "comply"  # rule wrong on one item
    r = AU.summarise(keys, rule, a, b)["msyc|harm"]
    assert r["unlabelled_a"] == 1 and r["unlabelled_b"] == 0
    assert r["judge_agreement"] == 8 / 9 and not r["judges_pass"]
    assert r["rule_agreement"] == 7 / 8 and not r["rule_pass"]
    assert r["rule_refuse_judges_comply"] == 1 and r["consensus_refusal_rate"] == 7 / 8


def test_load_judge_merges_parts(tmp_path):
    import json

    (tmp_path / "judge_a").mkdir()
    (tmp_path / "judge_a" / "part_01.json").write_text(json.dumps({"i0000": "refuse"}))
    (tmp_path / "judge_a" / "part_02.json").write_text(json.dumps({"i0001": "comply"}))
    assert AU.load_judge(tmp_path, "a") == {"i0000": "refuse", "i0001": "comply"}


def test_posthoc_subset_is_seeded_and_shared():
    ids = [f"h{i:04d}" for i in range(1700)]
    s1, s2 = AU.posthoc_ids(ids), AU.posthoc_ids(list(reversed(ids)))
    assert s1 == s2 and len(s1) == 300 and len(set(s1)) == 300


def test_paired_consensus_uses_items_with_consensus_in_both_states():
    a = {
        "base|harm|x": "refuse",
        "msyc|harm|x": "comply",
        "base|harm|y": "comply",
        "msyc|harm|y": "comply",
        "base|harm|z": "refuse",
        "msyc|harm|z": "refuse",
    }
    b = dict(a)
    b["msyc|harm|z"] = "comply"  # judges disagree -> z dropped from the M-SYC pair
    out = AU.paired_consensus(a, b)
    assert (
        out["msyc"]["ids"] == ["x", "y"]
        and out["msyc"]["edit"] == [1, 1]
        and out["msyc"]["base"] == [0, 1]
    )
    assert out["mte"]["ids"] == []
