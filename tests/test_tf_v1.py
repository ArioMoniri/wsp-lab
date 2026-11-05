import hashlib
import json
import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from wsp.eval.tf_v1 import choice_record, summarize
from wsp.schema import EvalRecord

SALT = ":TF-v1:pilot-dev:20261105"


class Tokenizer:
    """Deterministic tokenizer double: text -> fixed ids."""

    pad_token_id = 0

    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        return list(self.table[text])


# ---------------------------------------------------------------- item panels


def test_rank_ids_is_sha256_of_id_plus_salt_and_nested():
    from wsp.eval.tf_v1 import rank_ids

    pool = [str(i) for i in range(500)]
    ranked = rank_ids(pool, SALT)
    digest = {i: hashlib.sha256((i + SALT).encode()).hexdigest() for i in pool}
    assert ranked == sorted(pool, key=lambda i: (digest[i], int(i)))
    assert rank_ids(pool[::-1], SALT) == ranked  # independent of input order
    assert rank_ids(pool, SALT + "x") != ranked  # the salt matters
    # Any n is a prefix of any larger n by construction.
    for small, large in ((50, 100), (100, 250), (250, 500)):
        assert ranked[:small] == ranked[:large][:small]


@pytest.mark.parametrize("name,size", [("mmlu_pro", 1000), ("hellaswag", 500)])
def test_committed_panels_are_the_ranking_with_recorded_salt(name, size):
    from wsp.eval.subsets import ROOT
    from wsp.eval.tf_v1 import eligible_ids, rank_ids, read_panel

    header, ids = read_panel(name)
    assert header["salt"] == SALT
    assert header["panel"] == "development"
    assert len(ids) == size == len(set(ids))
    assert ids == rank_ids(eligible_ids(name), SALT)[:size]
    assert not set(ids) & set((ROOT / f"{name}.txt").read_text().splitlines())  # legacy excluded


def test_panel_prefix_is_the_n_item_panel_and_oversized_n_fails(monkeypatch):
    import datasets

    from wsp.eval import tf_v1

    class Dataset:
        _fingerprint = "fixed"

        def __len__(self):
            return 10042

        def __getitem__(self, i):
            return {"i": i}

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: Dataset())
    small, meta_small = tf_v1.load_items("hellaswag", 250, SALT)
    large, meta_large = tf_v1.load_items("hellaswag", 500, SALT)
    assert [i for i, _ in small] == [i for i, _ in large][:250]
    assert meta_small["subset_ids"] == meta_large["subset_ids"][:250]
    assert meta_small["panel"] == "development"
    assert meta_small["item_salt"] == SALT
    with pytest.raises(ValueError, match="only 500"):
        tf_v1.load_items("hellaswag", 501, SALT)
    with pytest.raises(ValueError, match="salt"):
        tf_v1.load_items("hellaswag", 10, ":other-salt")


def test_prompt_panels_are_ranked_nested_and_kl_and_over_refusal_are_disjoint(monkeypatch):
    import datasets

    from wsp.eval import tf_v1

    class Dataset:
        _fingerprint = "fixed"

        def __getitem__(self, column):
            return [f"prompt {i}" for i in range(6265)]

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: Dataset())
    tf = {"item_salt": SALT, "over_refusal_offset": 200}
    kl, kl_meta = tf_v1.load_prompts("kl_to_base", 200, tf)
    kl_small, _ = tf_v1.load_prompts("kl_to_base", 50, tf)
    over, over_meta = tf_v1.load_prompts("over_refusal", 50, tf)
    assert len(kl) == 200 and kl_small == kl[:50]
    assert kl_meta["panel"] == "development"
    assert not set(kl_meta["subset_ids"]) & set(over_meta["subset_ids"])
    assert not set(kl) & set(over)
    with pytest.raises(ValueError, match="overlap"):
        tf_v1.load_prompts("kl_to_base", 201, tf)


def test_tf_prompt_source_reads_full_pinned_split_not_legacy_100(monkeypatch):
    import datasets

    from wsp.eval.tf_v1 import load_prompts

    class Dataset:
        _fingerprint = "fixed"

        def __getitem__(self, column):
            return [f"prompt {i}" for i in range(104)]

    def load(repo, **kwargs):
        assert kwargs["split"] == "test"
        assert len(kwargs["revision"]) == 40
        return Dataset()

    monkeypatch.setattr(datasets, "load_dataset", load)
    texts, meta = load_prompts("refusal_rate", 50, {"item_salt": SALT})
    assert texts == [f"prompt {i}" for i in range(50)]
    assert meta["subset_ids"] == [str(i) for i in range(50)]


def test_device_id_comes_from_private_file_and_fails_loudly_when_absent(tmp_path, monkeypatch):
    from wsp.eval.metadata import runtime_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    runner = SimpleNamespace(model=SimpleNamespace())
    with pytest.raises(FileNotFoundError, match="device_id"):
        runtime_metadata(runner)
    (tmp_path / ".config/wsp").mkdir(parents=True)
    (tmp_path / ".config/wsp/device_id").write_text("H3\n")
    assert runtime_metadata(runner)["eval_device_id"] == "H3"
    (tmp_path / ".config/wsp/device_id").write_text("host-a")
    with pytest.raises(ValueError, match="opaque"):
        runtime_metadata(runner)


# ------------------------------------------------------------- letter scoring


def test_letter_scoring_stops_when_a_letter_is_not_a_single_token(monkeypatch, tmp_path):
    from wsp.eval import tf_v1

    def no_fallback(*args, **kwargs):
        raise AssertionError("option-text likelihood must never be substituted")

    runner = SimpleNamespace(
        tokenizer=Tokenizer({"ctx": [1], " A": [10], " B": [11, 12]}),
        choice_scores=no_fallback,
        model=SimpleNamespace(device="cpu"),
        edit_id="e1",
        tf_stop_dir=tmp_path,
    )
    monkeypatch.setattr(tf_v1, "last_logits", lambda *a, **k: torch.zeros(1, 20))
    with pytest.raises(tf_v1.LetterTokenizationError, match="single token") as caught:
        tf_v1.letter_scores(runner, "ctx", ["A", "B"])
    assert caught.value.reason == {
        "stop": "letter_not_single_token",
        "letter": "B",
        "token_ids": [11, 12],
    }
    # The reason is recorded durably, not only raised.
    tf_v1.record_stop(runner, caught.value)
    saved = json.loads((tmp_path / "stop_e1.json").read_text())
    assert saved["reason"]["letter"] == "B" and saved["benchmark"] == "tf_v1"


def test_letter_scoring_stops_on_duplicate_letter_tokens():
    from wsp.eval import tf_v1

    runner = SimpleNamespace(tokenizer=Tokenizer({" A": [10], " B": [10]}))
    with pytest.raises(tf_v1.LetterTokenizationError, match="distinct"):
        tf_v1.letter_token_ids(runner.tokenizer, ["A", "B"])


def test_letter_scores_use_full_vocabulary_log_softmax_and_return_token_ids(monkeypatch):
    from wsp.eval import tf_v1

    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    runner = SimpleNamespace(tokenizer=Tokenizer({"ctx": [1], " A": [2], " B": [3]}))
    monkeypatch.setattr(tf_v1, "last_logits", lambda *a, **k: logits)
    scores, ids = tf_v1.letter_scores(runner, "ctx", ["A", "B"])
    lp = logits[0].log_softmax(-1)
    assert ids == [2, 3]
    assert scores == pytest.approx([lp[2].item(), lp[3].item()])


# ------------------------------------------------------------ gold-letter etc.


def test_gold_letter_is_normalized_over_options_and_margin_is_paired():
    row = choice_record("base", "mmlu_pro", "17", [-3.0, -2.0, -4.0], 0)
    assert row.extra["gold_log_prob"] == pytest.approx(
        -3 - torch.logsumexp(torch.tensor([-3.0, -2.0, -4.0]), 0).item()
    )
    assert row.extra["margin"] == -1
    assert row.correct == 0
    assert len(row.extra["option_log_probs"]) == 3
    result = summarize([row])
    assert result["mmlu_pro"]["mean_gold_log_prob"] == row.extra["gold_log_prob"]
    assert result["mmlu_pro"]["accuracy"] == 0


def test_hellaswag_preserves_character_normalized_likelihoods():
    row = choice_record("base", "hellaswag", "8", [-0.3, -0.8], 1)
    assert row.extra["gold_log_prob"] == -0.8
    assert row.extra["margin"] == pytest.approx(-0.5)
    assert row.extra["ending_normalized_log_likelihoods"] == [-0.3, -0.8]


def test_token_weighted_log_ppl():
    rows = [
        EvalRecord("base", "wikitext_ppl", str(i), math.exp(x), n, extra={"mean_nll": x})
        for i, (x, n) in enumerate([(1.0, 2), (3.0, 6)])
    ]
    assert summarize(rows)["wikitext"]["log_ppl"] == 2.5


# ------------------------------------------------------------- KL direction


def test_kl_is_base_to_edited_with_hand_computed_asymmetric_example():
    from wsp.eval.refusal import kl_means, kl_values

    base = torch.tensor([[0.75, 0.25], [0.9, 0.1]]).log()
    edited = torch.tensor([[0.5, 0.5], [0.6, 0.4]]).log()
    # KL(p || q) = sum p log(p / q), by hand:
    first = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)  # 0.130812
    second = 0.9 * math.log(0.9 / 0.6) + 0.1 * math.log(0.1 / 0.4)  # 0.226290
    # The reverse direction KL(q || p) differs, so a swapped argument order would fail:
    reverse_first = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)  # 0.143841
    assert first == pytest.approx(0.130812, abs=1e-6)
    assert reverse_first == pytest.approx(0.143841, abs=1e-6)
    got_first, got_mean = kl_means(base, edited)
    assert got_first == pytest.approx(first, abs=1e-6)
    assert got_mean == pytest.approx((first + second) / 2, abs=1e-6)
    assert abs(got_first - reverse_first) > 1e-2
    assert kl_values(base, edited).tolist() == pytest.approx([first, second], abs=1e-6)
    # Swapped arguments give the reverse divergence.
    assert kl_means(edited, base)[0] == pytest.approx(reverse_first, abs=1e-6)


# ------------------------------------------------------------ refusal phrases


def phrase_runner(monkeypatch):
    """Vocabulary of 8; prompt [1, 2]; scripted next-token logits by exact context."""
    from wsp.eval import tf_v1

    table = {
        (1, 2): torch.tensor([0.0, 0.0, 0.0, 0.3, 0.0, 2.0, 1.0, 0.5]),
        (1, 2, 5): torch.tensor([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
    }

    def fake_last_logits(runner, ids, n=1):
        rows = []
        for j in range(n):
            context = tuple(ids[: len(ids) - n + j + 1])
            rows.append(table.get(context, torch.zeros(8)))
        return torch.stack(rows)

    monkeypatch.setattr(tf_v1, "last_logits", fake_last_logits)
    tokenizer = Tokenizer(
        {"Sorry": [6], "I cannot": [5, 4], "Sure": [7], "Certainly": [3], "I'd be happy": [5, 3]}
    )
    return SimpleNamespace(tokenizer=tokenizer), table


def test_refusal_phrases_are_scored_teacher_forced_not_by_first_token(monkeypatch):
    from wsp.eval import tf_v1

    runner, table = phrase_runner(monkeypatch)
    sets = tf_v1.build_continuations(
        runner.tokenizer, {"refusal": ["Sorry", "I cannot"], "compliance": ["Sure", "Certainly"]}
    )
    assert sets["refusal"]["I cannot"] == [5, 4]
    scores = tf_v1.continuation_scores(runner, [1, 2], {**sets["refusal"], **sets["compliance"]})
    first = table[(1, 2)].log_softmax(-1)
    second = table[(1, 2, 5)].log_softmax(-1)
    phrase = first[5].item() + second[4].item()
    assert scores["I cannot"] == pytest.approx(phrase)  # the phrase, "I" then " cannot"
    assert scores["I cannot"] != pytest.approx(first[5].item())  # not the bare token "I"
    assert scores["Sorry"] == pytest.approx(first[6].item())
    margin = tf_v1.phrase_mass_margin(
        [scores[p] for p in sets["refusal"]], [scores[p] for p in sets["compliance"]]
    )
    expected = torch.logsumexp(torch.tensor([first[6].item(), phrase]), 0) - torch.logsumexp(
        torch.tensor([first[7].item(), first[3].item()]), 0
    )
    assert margin == pytest.approx(expected.item())


def test_continuation_sets_must_be_nonempty_disjoint_and_prefix_free():
    from wsp.eval import tf_v1

    tok = Tokenizer({"A": [1], "A b": [1, 2], "C": [3], "D": [4]})
    with pytest.raises(ValueError, match="nonempty"):
        tf_v1.build_continuations(tok, {"refusal": [], "compliance": ["C"]})
    with pytest.raises(ValueError, match="disjoint"):
        tf_v1.build_continuations(tok, {"refusal": ["A", "C"], "compliance": ["C", "D"]})
    with pytest.raises(ValueError, match="prefix"):
        tf_v1.build_continuations(tok, {"refusal": ["A"], "compliance": ["A b"]})
    with pytest.raises(ValueError, match="empty"):
        tf_v1.build_continuations(
            Tokenizer({"A": [], "C": [3]}), {"refusal": ["A"], "compliance": ["C"]}
        )


def test_phrase_margin_requires_both_sets():
    from wsp.eval.tf_v1 import phrase_mass_margin

    assert phrase_mass_margin([-1.0], [-1.0]) == 0.0
    assert phrase_mass_margin([math.log(0.3), math.log(0.1)], [math.log(0.2)]) == pytest.approx(
        math.log(0.4 / 0.2)
    )
    with pytest.raises(ValueError):
        phrase_mass_margin([], [-1.0])


def test_configured_continuations_are_valid_and_documented():
    import yaml

    from wsp.eval.subsets import ROOT
    from wsp.eval.tf_v1 import build_continuations

    cfg = yaml.safe_load((ROOT.parent / "eval_tf_v1.yaml").read_text())["tf_v1"]
    sets = cfg["refusal_continuations"]
    assert "I cannot" in sets["refusal"]
    assert cfg["panel"] == "development"
    assert cfg["item_salt"] == SALT

    class Words:  # one token per character so that the real sets validate without a download
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    # Phrases such as "I cannot" and "I'd be happy to" share their first token yet must validate.
    assert build_continuations(Words(), sets)["refusal"]["I cannot"][0] == ord("I")


# --------------------------------------------------------------------- cache


def make_runner(tmp_path, monkeypatch, logits=None):
    from wsp.eval import batch, tf_v1

    class Model:
        device = "cpu"
        calls = 0

        def generate(self, ids, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 2
            self.calls += 1
            return torch.tensor([[0, 1, 2, 3]])

    runner = SimpleNamespace(
        model=Model(),
        revision="fixed",
        edit_id="e1",
        generation={"batch_size": 1},
        tokenizer=SimpleNamespace(encode=lambda *a, **k: [0, 1], pad_token_id=0),
        chat=lambda x: x,
        base_context=nullcontext,
        tf_cache_root=tmp_path,
        tf_config={
            "generation": {"batch_size": 1},
            "benchmarks": {},
            "tf_v1": {
                "item_salt": SALT,
                "over_refusal_offset": 200,
                "max_cache_gib": 1.0,
                "panel": "development",
            },
        },
    )
    monkeypatch.setattr(tf_v1, "runtime_metadata", lambda _: {"eval_device_id": "H0"})
    monkeypatch.setattr(batch, "provenance", lambda: {"source_hash": "fixed"})
    if logits is None:
        logits = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    monkeypatch.setattr(tf_v1, "last_logits", lambda *a, **k: logits[: a[2] if len(a) > 2 else 1])
    tf_v1.reset_cache_residency()
    return runner


def digest_of(logits, ids, prefix_length):
    h = hashlib.sha256()
    h.update(prefix_length.to_bytes(8, "little"))
    h.update(torch.tensor(ids, dtype=torch.int64).numpy().tobytes())
    h.update(logits.contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def test_base_cache_is_greedy_reused_residently_and_detects_corruption(tmp_path, monkeypatch):
    from wsp.eval import tf_v1

    runner = make_runner(tmp_path, monkeypatch)
    call = lambda: tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}, 2)
    root, meta, entries = call()
    assert runner.model.calls == 1
    # Unrelated source edits (full-tree hash, git sha, dirty flag) never orphan the cache.
    from wsp.eval import batch

    monkeypatch.setattr(
        batch, "provenance", lambda: {"source_hash": "other", "git_sha": "new", "dirty": True}
    )
    root2, meta2, _ = call()
    assert (root2, meta2) == (root, meta)
    assert runner.model.calls == 1
    data = torch.load(root / "0.pt", weights_only=True)
    assert data["prefix_length"] == 2
    assert data["logits"].shape == (2, 5) and data["logits"].dtype == torch.bfloat16
    expected = digest_of(data["logits"], data["ids"], 2)
    assert meta["files"] == {"0.pt": expected}
    assert meta["base_logits_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert entries[0]["logits"].shape == (2, 5)
    # A fresh process re-verifies the bytes once and detects corruption.
    tf_v1.reset_cache_residency()
    assert call()[1] == meta
    (root / "0.pt").write_bytes(b"corrupt")
    tf_v1.reset_cache_residency()
    with pytest.raises(ValueError, match="cache hash mismatch"):
        call()


def test_cache_is_loaded_and_verified_once_per_process(tmp_path, monkeypatch):
    from wsp.eval import tf_v1

    runner = make_runner(tmp_path, monkeypatch)
    tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}, 2)
    tf_v1.reset_cache_residency()  # a new process: the cache exists only on disk
    loads = []
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: loads.append(a[0]) or real_load(*a, **k))
    for _ in range(5):  # five entries in one process
        tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}, 2)
    assert len(loads) == 1


def test_cache_manifest_is_atomic_complete_and_size_capped(tmp_path, monkeypatch):
    from wsp.eval import tf_v1

    runner = make_runner(tmp_path, monkeypatch)
    root, meta, _ = tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}, 2)
    assert not list(root.glob("*.tmp"))
    assert meta["cache_bytes"] == (root / "0.pt").stat().st_size
    # Manifest writes go through a temporary file and an atomic replace.
    replaced = []
    from pathlib import Path

    real = Path.replace
    monkeypatch.setattr(
        Path, "replace", lambda self, target: replaced.append(self.name) or real(self, target)
    )
    tf_v1.atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert (
        replaced
        and replaced[0].endswith(".tmp")
        and json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    )
    monkeypatch.undo()
    # A file set that differs from the expected prompts is refused.
    runner = make_runner(tmp_path, monkeypatch)
    (root / "7.pt").write_bytes(b"stray")
    tf_v1.reset_cache_residency()
    with pytest.raises(ValueError, match="file set"):
        tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}, 2)
    (root / "7.pt").unlink()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["1.pt"] = manifest["files"].pop("0.pt")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="file set"):
        tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}, 2)
    # An over-cap cache aborts rather than filling the disk.
    runner = make_runner(tmp_path / "capped", monkeypatch)
    with pytest.raises(ValueError, match="size cap"):
        tf_v1.cache_kl(runner, ["prompt"], {"revision": "pinned"}, 2, max_bytes=10)


def test_kl_records_carry_content_hash_position_kl_and_direction(tmp_path, monkeypatch):
    from wsp.eval import tf_v1

    runner = make_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        tf_v1,
        "load_prompts",
        lambda kind, n, tf: (
            ["prompt"] * n,
            {"repo": "r", "revision": "v", "subset_ids": ["0"], "panel": "development"},
        ),
    )
    # Edited logits are the base logits with the first position flattened: KL > 0 there only.
    base = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    flat = base.clone()
    flat[0] = 0.0

    calls = []

    def fake_last_logits(runner_, ids, n=1):
        calls.append(n)
        return base[:n] if len(calls) == 1 else flat[:n]  # first call builds the cache

    monkeypatch.setattr(tf_v1, "last_logits", fake_last_logits)
    cfg = {"kl_to_base": {"n": 1, "max_new_tokens": 2}}
    records, datasets = tf_v1.evaluate(runner, ["kl_to_base"], cfg)
    (record,) = records
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())
    assert record.extra["base_logits_sha256"] == manifest["base_logits_sha256"]
    assert datasets["kl_to_base"]["base_logits_sha256"] == manifest["base_logits_sha256"]
    assert record.extra["panel"] == "development"
    assert runner.tf_cache_info["base_logits_sha256"] == manifest["base_logits_sha256"]
    assert record.extra["direction"] == "base || edited"
    pos = record.extra["position_kl"]
    assert len(pos) == 2 and pos[0] > 0 and pos[1] == pytest.approx(0, abs=1e-6)
    assert record.extra["full_sequence_kl"] == pytest.approx(sum(pos) / 2, abs=1e-6)
    assert record.extra["first_token_kl"] == pytest.approx(pos[0], abs=1e-6)


# --------------------------------------------------------------------- misc


def test_tf_dispatch_never_calls_sampling(monkeypatch):
    from wsp.eval import run, tf_v1

    runner = SimpleNamespace(generation={"tier": "tf_v1"})
    monkeypatch.setattr(tf_v1, "evaluate", lambda *args: ([], {"tf": True}))
    assert run.evaluate(runner, ["mmlu_pro"], {}) == ([], {"tf": True})


def test_results_embed_tf_metrics_and_item_hashes_without_keyword_aliases(tmp_path):
    from wsp.eval.run import write_results

    row = choice_record("base", "mmlu_pro", "17", [-3.0, -2.0], 0)
    refusal = EvalRecord("base", "refusal_rate", "0", -2.0)
    result = write_results(tmp_path, [row, refusal], {"tier": "tf_v1"}, {"seed": 20261105})
    assert result["tf_v1"]["refusal_rate"]["mean_phrase_margin"] == -2.0
    assert "refusal_keyword_rate" not in result
    assert "refusal_rate" not in result
    assert len(result["item_id_hash"]) == 64
    assert json.loads((tmp_path / "records.jsonl").read_text().splitlines()[0])["item_id"] == "17"


# ------------------------------------------- batching (amendment A2.4): identity and bounds


class CausalToy(torch.nn.Module):
    """Deterministic causal LM double: logits at t mix an embedding of every token <= t.

    Row-independent and mask-independent for real positions, as a real causal LM is, so padded and
    unpadded forwards of the same row agree up to floating-point reduction order."""

    device = "cpu"

    def __init__(self, vocab=11, dim=6, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(vocab, dim, generator=gen))
        self.out = torch.nn.Parameter(torch.randn(dim, vocab, generator=gen))
        self.forwards, self.shapes = 0, []

    def forward(self, ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        self.forwards += 1
        self.shapes.append(tuple(ids.shape))
        state = self.emb[ids].cumsum(1) / torch.arange(1, ids.shape[1] + 1)[None, :, None]
        logits = state @ self.out
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)

    def choice_scores(self, tokenizer, context, choices):
        prefix = tokenizer.encode(context)
        scores = []
        for choice in choices:
            suffix = tokenizer.encode(" " + choice)
            ids = torch.tensor([prefix + suffix])
            logits = self(ids).logits[0, len(prefix) - 1 : -1].float().log_softmax(-1)
            scores.append(logits.gather(1, ids[0, len(prefix) :, None]).sum().item() / len(choice))
        return scores


def toy_runner(table):
    model = CausalToy()
    return SimpleNamespace(model=model, tokenizer=Tokenizer(table), edit_id="e"), model


TOY_TABLE = {
    "ctx a": [1, 2, 3, 4],
    "ctx b": [5, 6],
    " end one": [7, 8, 9],
    " end 2": [3],
    " endings number three": [2, 2, 4, 5, 6],
    " e4": [10, 1],
}


def test_batched_hellaswag_matches_batch1_within_probe_bound_and_uses_one_forward():
    from wsp.eval import tf_v1

    runner, model = toy_runner(TOY_TABLE)
    endings = ["end one", "end 2", "endings number three", "e4"]
    for context in ("ctx a", "ctx b"):
        reference = model.choice_scores(runner.tokenizer, context, endings)
        model.forwards, model.shapes = 0, []
        scores, batch = tf_v1.hellaswag_scores(runner, context, endings)
        assert batch == 4 and model.forwards == 1  # one padded forward for the four endings
        assert model.shapes[0][0] == 4
        assert max(abs(a - b) for a, b in zip(scores, reference, strict=True)) <= 0.010
        again, _ = tf_v1.hellaswag_scores(runner, context, endings)
        assert again == scores  # the batched path repeats bitwise


def test_batched_phrases_match_batch1_within_probe_bound_and_use_one_forward():
    from wsp.eval import tf_v1

    runner, model = toy_runner({})
    phrases = {
        "sorry": [6],
        "I cannot": [5, 4],
        "I'm sorry": [5, 4, 2],
        "sure": [7],
        "here": [3, 9],
    }
    prefix = [1, 2, 3, 4, 5]
    reference = tf_v1.continuation_scores(runner, prefix, phrases)
    model.forwards, model.shapes = 0, []
    scores, batch = tf_v1.batched_continuation_scores(runner, prefix, phrases)
    assert batch == len(phrases) == model.shapes[0][0] and model.forwards == 1
    assert list(scores) == list(phrases)  # order is the phrase order
    assert max(abs(scores[p] - reference[p]) for p in phrases) <= 0.39
    margin = lambda s: tf_v1.phrase_mass_margin(
        [s["sorry"], s["I cannot"], s["I'm sorry"]], [s["sure"], s["here"]]
    )
    assert abs(margin(scores) - margin(reference)) <= 0.30
    assert tf_v1.batched_continuation_scores(runner, prefix, phrases)[0] == scores  # bitwise


def test_batched_scores_ignore_right_padding_content():
    from wsp.eval import tf_v1

    runner, _ = toy_runner(TOY_TABLE)
    short = tf_v1.hellaswag_scores(runner, "ctx b", ["end 2"])[0]  # no padding at all
    padded = tf_v1.hellaswag_scores(runner, "ctx b", ["end 2", "endings number three"])[0][0]
    assert abs(short[0] - padded) < 1e-5


def test_evaluate_records_carry_the_actual_scoring_batch(monkeypatch):
    from wsp.eval import tf_v1

    runner, _ = toy_runner(TOY_TABLE)
    runner.chat = lambda x: x
    runner.tf_config = {
        "tf_v1": {
            "item_salt": SALT,
            "panel": "development",
            "over_refusal_offset": 200,
            "refusal_continuations": {"refusal": ["Sorry"], "compliance": ["Sure"]},
        }
    }
    monkeypatch.setattr(
        tf_v1,
        "load_items",
        lambda *a: (
            [("3", {"ctx": "ctx a", "endings": ["end one", "end 2", "e4"], "label": "1"})],
            {"panel": "development"},
        ),
    )
    monkeypatch.setattr(
        tf_v1, "load_prompts", lambda *a: (["ctx b"], {"panel": "development", "subset_ids": ["0"]})
    )
    runner.tokenizer.table |= {"Sorry": [6], "Sure": [7]}
    records, _ = tf_v1._evaluate(
        runner,
        ["hellaswag", "refusal_rate"],
        {"hellaswag": {"n": 1}, "refusal_rate": {"n": 1}},
    )
    hellaswag, refusal = records
    assert hellaswag.extra["scoring_batch"] == 3 and refusal.extra["scoring_batch"] == 2
    assert len(hellaswag.extra["raw_option_log_likelihoods"]) == 3
    assert set(refusal.extra["phrase_log_probs"]) == {"Sorry", "Sure"}


# Primary paths are untouched by the batching change: their source is pinned to the round-2 source
# (commit 98d8c06, `source_hash` 12ee875a68c9) so that MMLU-Pro, KL and WikiText records stay
# bitwise identical and the KL base-logit cache (root 2ed18bda284b3a89...) stays valid.
PINNED_SOURCE = {
    "last_logits": "9af5efd013e53ffe",
    "letter_token_ids": "7f2d3fb743466a48",
    "letter_scores": "21fbbbaaf82e563d",
    "choice_record": "3c33769533c3cf77",
    "generate_entry": "fa9ee05dfd09a236",
    "entry_digest": "35526ea94d2e443b",
    "content_root": "6ad79a586e32311e",
    "phrase_mass_margin": "617b44c7c9547b88",
    "continuation_scores": "fec50cd4fe6b60fd",
    "build_continuations": "4c84cb5011f7fdb9",
}
PINNED_SCORER_HASH = "7c42ea73fa33101f7804e7d9f64abbe12737c11be020a78d947fff672fecd514"


def test_primary_scoring_paths_are_untouched_by_batching():
    import inspect

    from wsp.eval import refusal, run, tf_v1

    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    for name, pinned in PINNED_SOURCE.items():
        assert digest(inspect.getsource(getattr(tf_v1, name)))[:16] == pinned, name
    assert tf_v1.SCORER_HASH == PINNED_SCORER_HASH  # the KL cache identity
    assert (
        digest(inspect.getsource(refusal.kl_values))
        == "4011e6a2f651a6eca6a4bac063f83f5070e7a631e83f4a92df081bf2f9e95c0c"
    )
    assert (
        digest(inspect.getsource(run.perplexity))
        == "bb31a97cd2d15859b73986540292e3cdd0b603e1c65f64b365d03e1bf7db73b1"
    )
    source = inspect.getsource(tf_v1._evaluate)
    branch = {
        "mmlu": (
            '                if name == "mmlu_pro":\n                    choices',
            '                else:\n                    context, choices, gold = row["ctx"]',
        ),
        "wikitext": (
            '        elif name == "wikitext_ppl":',
            '        elif name in ("refusal_rate", "over_refusal"):',
        ),
        "kl": (
            '        elif name == "kl_to_base":',
            '        else:\n            raise ValueError(f"unsupported TF-v1 benchmark',
        ),
    }
    pinned = {
        "mmlu": "8b12d182ec15491172e785b9c26f87ad81e7b2fe39a3c754557ab914f34e5471",
        "wikitext": "7c21b8d1ca7dea444dec2bdd31f4b33ee417931510d8198b9b27136eae1c49cb",
        "kl": "4d3ce34bb3af2fe6cb0e8e1b21c00780b6bc959a09d9369650de67e08de3d308",
    }
    for key, (start, end) in branch.items():
        assert digest(source[source.index(start) : source.index(end)]) == pinned[key], key
