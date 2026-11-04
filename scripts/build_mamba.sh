#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
mkdir -p data/build
python - <<'PYCODE'
import json, pathlib, urllib.request
for name, version in [("causal-conv1d", "1.7.0"), ("mamba-ssm", "2.3.2.post1")]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as response:
        data = json.load(response)
    item = next(item for item in data["urls"] if item["packagetype"] == "sdist")
    urllib.request.urlretrieve(item["url"], pathlib.Path("data/build") / item["filename"])
PYCODE
for package in causal_conv1d-1.7.0 mamba_ssm-2.3.2.post1; do
  archive=$(find data/build -maxdepth 1 -name "${package}*.tar.gz" -print -quit)
  tar -xzf "$archive" -C data/build
  sed -i 's/-std=c++17/-std=c++20/g' "data/build/$package/setup.py"
  uv pip install --no-build-isolation --no-deps "./data/build/$package"
done
