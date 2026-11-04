import json

import torch

from wsp.eval.batch import WeightRestore, retry_oom, select_entries


def test_restore_exact_base_between_edits():
    model = torch.nn.Linear(2, 2)
    original = model.weight.detach().clone()
    restore = WeightRestore(model)
    restore.capture(["weight"])
    with torch.no_grad():
        model.weight.add_(2)
    restore.restore()
    assert torch.equal(model.weight, original)


def test_selection_skips_smoke_and_completed(tmp_path):
    rows = [
        {"id": "a", "eligibility": "production", "family": "random_dir"},
        {"id": "b", "eligibility": "smoke", "family": "random_dir"},
    ]
    assert list(select_entries(rows, tmp_path)) == rows[:1]
    (tmp_path / "a").mkdir()
    (tmp_path / "a/eval.json").write_text(json.dumps({"eval_status": "complete"}))
    assert list(select_entries(rows, tmp_path)) == []


def test_oom_halves_twice():
    batches = []

    def run(batch):
        batches.append(batch)
        raise torch.cuda.OutOfMemoryError("test")

    assert retry_oom(run, 8) == (None, 2)
    assert batches == [8, 4, 2]


def test_batch_loads_once_restores_and_resumes(tmp_path, monkeypatch):
    import yaml

    from wsp.eval import batch
    from wsp.schema import EvalRecord

    monkeypatch.chdir(tmp_path)
    zoo = tmp_path / "zoo"
    zoo.mkdir()
    rows = [{"type": "header", "base_model": "toy", "base_revision": "a" * 40}]
    for name in ["a", "b"]:
        entry = zoo / name
        entry.mkdir()
        (entry / "edit.json").write_text(json.dumps({"base_revision": "a" * 40}))
        rows.append({"id": name, "family": "random_dir", "eligibility": "production"})
    (zoo / "manifest.jsonl").write_text("\n".join(map(json.dumps, rows)))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "generation": {"seed": 0, "batch_size": 4, "max_new_tokens": 4},
                "benchmarks": {"gsm8k": {"n": 1}},
            }
        )
    )
    loads, weights, batches = [], [], []
    failed = set()

    class Runner:
        def __init__(self, model, generation, revision):
            loads.append(model)
            self.model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.model.weight.zero_()
            self.model_id, self.revision, self.generation = model, revision, generation

    def apply(model, entry):
        with torch.no_grad():
            model.weight.add_(1)
        return {}

    def evaluate(runner, *args):
        batches.append((runner.edit_id, runner.generation["batch_size"]))
        if runner.edit_id == "a" and "a" not in failed:
            failed.add("a")
            raise torch.cuda.OutOfMemoryError("test")
        weights.append(runner.model.weight.item())
        return [EvalRecord(runner.edit_id, "gsm8k", "0", 1.0)], {}

    monkeypatch.setattr(batch, "HFRunner", Runner)
    monkeypatch.setattr(batch, "load_entry_delta", lambda *args: {"weight": torch.ones(1, 1)})
    monkeypatch.setattr(batch, "apply_zoo_entry", apply)
    monkeypatch.setattr(batch, "evaluate", evaluate)
    monkeypatch.setattr(batch, "REPO", tmp_path)
    monkeypatch.setattr(
        batch, "provenance", lambda: {"git_sha": "test", "dirty": False, "source_hash": "source"}
    )
    (tmp_path / "configs").mkdir()
    production_config = tmp_path / "configs/eval_fast.yaml"
    production_config.write_bytes(config.read_bytes())
    config = production_config
    batch.run_batch(zoo, config)
    batch.run_batch(zoo, config)
    assert loads == ["toy"]
    assert weights == [0, 1, 1]
    assert batches == [("base", 4), ("a", 4), ("a", 2), ("b", 4)]
    index = tmp_path / "data/evals/zoo/index.jsonl"
    assert len(index.read_text().splitlines()) == 3
    index.unlink()  # interrupted after eval.json: repair even on skip
    batch.run_batch(zoo, config)
    assert len(index.read_text().splitlines()) == 3
    canonical = (zoo / "a/eval.json").read_bytes()
    batch.run_batch(zoo, config, limit=1, max_new_tokens=2)
    assert (zoo / "a/eval.json").read_bytes() == canonical
    outputs = list((tmp_path / "data/runs/p2-batch").glob("*/a/eval.json"))
    payload = json.loads(outputs[0].read_text())
    assert {
        "eligibility",
        "family",
        "parent_id",
        "step_semantics",
        "dirty",
        "source_hash",
    } <= payload.keys()
    assert payload["config_hash"] != json.loads(canonical)["config_hash"]


def test_resume_does_not_treat_tiny_smoke_as_full_eval(tmp_path):
    entry = tmp_path / "a"
    entry.mkdir()
    (entry / "eval.json").write_text(json.dumps({"eval_status": "complete", "signature": "tiny"}))
    rows = [{"id": "a", "family": "random_dir", "eligibility": "production"}]
    assert list(select_entries(rows, tmp_path, signature="full")) == rows
    assert list(select_entries(rows, tmp_path, signature="tiny")) == []


def test_index_retry_and_completeness(tmp_path):
    from wsp.eval.batch import complete, upsert_index

    path = tmp_path / "index.jsonl"
    payload = {"edit_id": "a", "config_hash": "cfg", "eval_status": "oom"}
    upsert_index(path, payload)
    upsert_index(path, payload | {"eval_status": "complete"})
    assert len(path.read_text().splitlines()) == 1
    assert not complete(
        {"eval_status": "complete", "signature": "s", "counts": {"gsm8k": 1}},
        "s",
        {"gsm8k": {"n": 2}},
    )


def test_d3_seeded_production_selection():
    from wsp.eval.batch import subset_ids

    rows = [{"id": str(i), "family": "chain", "eligibility": "production"} for i in range(10)]
    spec = {"seed": 0, "families": {"chain": 3}}
    assert subset_ids(rows, spec) == subset_ids(list(reversed(rows)), spec)
    assert len(subset_ids(rows, spec)) == 3


def test_kl_uses_base_context_and_restores_bf16(monkeypatch):
    from types import SimpleNamespace

    from wsp.eval import refusal

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.device = "cpu"

        def forward(self, ids):
            logits = torch.zeros(1, ids.shape[1], 2)
            logits[..., 0] = self.weight.item()
            return SimpleNamespace(logits=logits)

    model = Model()
    restore = WeightRestore(model)
    restore.capture(["weight"])
    with torch.no_grad():
        model.weight.add_(1)
    seen = []

    def generate(texts):
        seen.append(model.weight.item())
        return [("answer", 1, False)]

    runner = SimpleNamespace(
        model=model,
        generation={"max_new_tokens": 1024},
        base_context=restore.base_context,
        generate=generate,
        tokenizer=SimpleNamespace(encode=lambda *a, **k: [1]),
        chat=lambda x: x,
        edit_id="toy",
    )
    monkeypatch.setattr(refusal, "prompts", lambda *a: (["prompt"], {}))
    records, _ = refusal.evaluate_refusal(runner, "kl_to_base", 1)
    assert seen == [0]
    assert records[0].correct > 0
    assert model.weight.item() == 1
    assert runner.generation["max_new_tokens"] == 1024


def test_package_provenance_ignores_cwd(tmp_path, monkeypatch):
    from wsp.eval.batch import provenance

    before = provenance()
    monkeypatch.chdir(tmp_path)
    assert provenance() == before
    assert len(before["git_sha"]) == 40
    assert len(before["source_hash"]) == 64


def test_d2_config():
    import yaml

    from wsp.eval.batch import REPO

    cfg = yaml.safe_load((REPO / "configs/eval_fast.yaml").read_text())
    assert cfg["generation"]["max_new_tokens"] == 1024
    assert cfg["generation"]["batch_size"] == 32
