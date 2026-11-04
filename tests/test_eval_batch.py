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
    loads, weights = [], []

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
        weights.append(runner.model.weight.item())
        return [EvalRecord(runner.edit_id, "gsm8k", "0", 1.0)], {}

    monkeypatch.setattr(batch, "HFRunner", Runner)
    monkeypatch.setattr(batch, "load_entry_delta", lambda *args: {"weight": torch.ones(1, 1)})
    monkeypatch.setattr(batch, "apply_zoo_entry", apply)
    monkeypatch.setattr(batch, "evaluate", evaluate)
    monkeypatch.setattr(batch, "get_git_sha", lambda: "test")
    batch.run_batch(zoo, config)
    batch.run_batch(zoo, config)
    assert loads == ["toy"]
    assert weights == [0, 1, 1]
    assert len((tmp_path / "data/evals/zoo/index.jsonl").read_text().splitlines()) == 3


def test_resume_does_not_treat_tiny_smoke_as_full_eval(tmp_path):
    entry = tmp_path / "a"
    entry.mkdir()
    (entry / "eval.json").write_text(json.dumps({"eval_status": "complete", "signature": "tiny"}))
    rows = [{"id": "a", "family": "random_dir", "eligibility": "production"}]
    assert list(select_entries(rows, tmp_path, signature="full")) == rows
    assert list(select_entries(rows, tmp_path, signature="tiny")) == []
