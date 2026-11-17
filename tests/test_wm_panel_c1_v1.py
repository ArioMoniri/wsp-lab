"""wme-c1 panel builder to the v1 specification (PREREG_WME v1 §3.2-3.4; RULING-wme-v2 A5)."""

import hashlib
import json

import pytest

from wsp.wm import panel_c1 as C1
from wsp.wm import plan as PL
from wsp.wm.panel import text_hash


def _words(i, tag):
    h = hashlib.sha256(f"{tag}{i}".encode()).hexdigest()
    return " ".join(h[j : j + 6] for j in range(0, 36, 6))


def _pools():
    mk = lambda kind, txt: {
        "kind": kind,
        "prompt": txt,
        "hash": text_hash(txt),
        "id": f"{kind}:{text_hash(txt)[:8]}",
    }
    harm = [mk("harmful", "Explain " + _words(i, "h")) for i in range(30)]
    harm.append(mk("harmful", "Explain " + _words(3, "h") + "!"))  # near-duplicate of item 3
    harmless = [mk("harmless", "Write " + _words(i, "x")) for i in range(30)]
    gen = [
        {
            "kind": "triviaqa_gen",
            "question": "Which " + _words(i, "q") + "?",
            "hash": text_hash(f"q{i}"),
            "id": f"tq:{i}",
            "neutral_prompt": "x",
            "suggested_prompt": "y",
        }
        for i in range(12)
    ]
    return {
        "refusal_margin_harmful": harm,
        "refusal_margin_harmless": harmless,
        "triviaqa_gen": gen,
    }


SIZES = {
    "refusal_margin_harmful": {
        "sealed_gen": 0,
        "sealed_test": 6,
        "dev": 6,
        "bank": 6,
        "construction": "rest",
    },
    "refusal_margin_harmless": {
        "sealed_gen": 0,
        "sealed_test": 6,
        "dev": 6,
        "bank": 6,
        "construction": 4,
    },
    "triviaqa_gen": {"sealed_gen": 10, "sealed_test": 0, "dev": 0, "bank": 0, "construction": 0},
}


def test_v1_builder_splits_groups_and_sealed_dirs(tmp_path):
    exposed = {"pilot": {"test": {text_hash("Explain " + _words(7, "h"))}}}
    m = C1.build_c1_panel_v1(_pools(), SIZES, tmp_path / "p", exposed)
    assert set(m["objectives"]["refusal_margin_harmful"]) == set(C1.SPLITS_V1)
    h = m["objectives"]["refusal_margin_harmful"]
    assert (
        h["sealed_test"]["n"] == 6 and h["construction"]["n"] == 31 - 1 - 1 - 18
    )  # 2 near-dup items = 1 group; 1 exposed
    groups = [
        g for s in C1.SPLITS_V1 for o in m["objectives"] for g in m["objectives"][o][s]["groups"]
    ]
    assert len(groups) == len(set(groups))
    assert (tmp_path / "p/sealed_gen/gen.jsonl").exists() and (
        tmp_path / "p/sealed_test/test.jsonl"
    ).exists()
    assert m["policy"]["jaccard"] == 0.8 and m["audit_clean"]
    assert set(m["split_hashes"]) == {
        "open/dev.jsonl",
        "open/bank.jsonl",
        "open/construction.jsonl",
        "sealed_test/test.jsonl",
        "sealed_gen/gen.jsonl",
    }
    store = PL.PlannerItemStore(tmp_path / "p")
    assert store.read("dev") and store.read("bank")
    with pytest.raises(PL.SealedError):
        store.read_path(tmp_path / "p/sealed_gen/gen.jsonl")
    with pytest.raises(FileExistsError):
        C1.build_c1_panel_v1(_pools(), SIZES, tmp_path / "p", exposed)
    man = json.loads((tmp_path / "p/manifest.json").read_text())
    assert man["content_hash"] == m["content_hash"] and len(man["sealed_hashes"]) == 6 + 6 + 10


def test_v1_builder_never_relaxes(tmp_path):
    sizes = json.loads(json.dumps(SIZES))
    sizes["refusal_margin_harmless"]["sealed_test"] = 40
    with pytest.raises(ValueError):
        C1.build_c1_panel_v1(_pools(), sizes, tmp_path / "q", {})
