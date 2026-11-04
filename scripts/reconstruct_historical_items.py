"""Recover item identities for the audited original D3 evaluator; never recover scores.

The audited tree uses checked-in positional subset IDs for five benchmarks and
str(i) for WikiText/refusal/KL. Exact source-tree and count checks bind this
reconstruction to the historical execution, rather than retest outputs.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

AUDITED_TREE = "846ff8423b9c67a9d9a9bfa3c1a46ec8f3feb2e5"
SUBSETS = {"gsm8k", "math500", "ifeval", "mmlu_pro", "hellaswag"}
POSITIONAL = {"wikitext_ppl", "refusal_rate", "over_refusal", "kl_to_base"}


def reconstruct(payload):
    revision = payload["git_sha"]
    if revision != AUDITED_TREE or payload.get("dirty") or payload.get("limit"):
        raise ValueError("only the audited clean, unlimited historical tree is supported")

    def git(*args):
        return subprocess.check_output(["git", *args])

    digest = hashlib.sha256()
    for name in sorted(
        git("ls-tree", "-r", "--name-only", revision, "src/wsp").decode().splitlines()
    ):
        if name.endswith(".py"):
            digest.update(name.encode())
            digest.update(git("show", f"{revision}:{name}"))
    if digest.hexdigest() != payload["source_hash"]:
        raise ValueError("archived source hash does not match git tree")
    items = []
    for name, config in payload["effective_config"]["benchmarks"].items():
        n = payload["counts"][name]
        expected = config.get("n_windows", 64) if name == "wikitext_ppl" else config["n"]
        if n != expected:
            raise ValueError("incomplete historical item count")
        if name in SUBSETS:
            ids = (
                git("show", f"{revision}:configs/eval_subsets/{name}.txt").decode().splitlines()[:n]
            )
        elif name in POSITIONAL:
            ids = list(map(str, range(n)))
        else:
            raise ValueError(f"unaudited benchmark: {name}")
        items.extend([[name, i] for i in ids])
    return {
        payload["run_id"]: {
            "source_hash": payload["source_hash"],
            "items": items,
            "source": f"Reconstructed from audited git {revision}: run.py, subsets.py, refusal.py, "
            "configs/eval_subsets/*.txt; archived effective_config order and counts; no retest records",
        }
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(reconstruct(json.loads(args.record.read_text())), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
