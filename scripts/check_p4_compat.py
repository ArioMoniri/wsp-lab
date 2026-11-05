"""Compare every pre-existing metrics field from identical tiny CPU M1 runs.

Run after sourcing scripts/env.sh. Exits nonzero on any compatibility regression.
New dictionary keys are permitted; existing values (including NaNs) must match.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        for key, value in before.items():
            if key not in after:
                yield f"{path}/{key}: missing"
            else:
                yield from differences(value, after[key], f"{path}/{key}")
    elif isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            yield f"{path}: length {len(before)} != {len(after)}"
        for i, (a, b) in enumerate(zip(before, after)):
            yield from differences(a, b, f"{path}/{i}")
    elif (
        isinstance(before, float)
        and isinstance(after, float)
        and math.isnan(before)
        and math.isnan(after)
    ):
        return
    elif before != after:
        yield f"{path}: {before!r} != {after!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-ref", default="main")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.out or Path(tempfile.mkdtemp(prefix="wsp-p4-compat-"))
    root.mkdir(parents=True, exist_ok=True)
    source = root / "main-source"
    source.mkdir(exist_ok=True)
    archive = subprocess.check_output(["git", "archive", args.main_ref], cwd=repo)
    import io

    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    from wsp.predictor.selftest import prepare_selftest

    data = prepare_selftest()
    # The audit controls require effective norms, absent from the legacy selftest.
    for i, file in enumerate(sorted((data / "features").glob("*/features.json"))):
        metadata = json.loads(file.read_text())
        metadata["norms"] = {"fro_effective": float(i + 1)}
        file.write_text(json.dumps(metadata))
    common = [
        "--model",
        "m1",
        "--features",
        str(data / "features"),
        "--evals",
        str(data / "evals"),
        "--zoo",
        str(data / "zoo"),
        "--device",
        "cpu",
        "--epochs",
        "2",
        "--n-boot",
        "20",
        "--seed",
        "0",
        "--n-splits",
        "3",
        "--d-model",
        "16",
        "--batch-size",
        "4",
    ]
    for label, checkout in [("main", source), ("branch", repo)]:
        env = dict(
            os.environ,
            PYTHONPATH=str(checkout / "src"),
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        env["GIT_DIR"] = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=repo, text=True
        ).strip()
        env["GIT_WORK_TREE"] = str(checkout)
        command = [sys.executable, "-m", "wsp.predictor.train", *common, "--out", str(root / label)]
        with (root / f"{label}.log").open("w") as log:
            subprocess.run(
                command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    before = json.loads((root / "main/metrics.json").read_text())
    after = json.loads((root / "branch/metrics.json").read_text())
    failures = list(differences(before, after))
    (root / "differences.txt").write_text("\n".join(failures) + "\n")
    print(f"Compared {len(before)} pre-existing top-level keys; {len(failures)} differences.")
    print(f"Artifacts: {root}")
    for failure in failures[:12]:
        print(failure)
    assert not failures, f"default compatibility failed; see {root / 'differences.txt'}"


if __name__ == "__main__":
    main()
