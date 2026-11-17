"""scripts/cuda_env.sh: per-project CUDA tag detection, LD_LIBRARY_PATH scrubbing, namespaced kernel caches."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cuda_env.sh"


def _run(venv: Path, ld: str, project: str = "") -> dict:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(venv.parent), "LD_LIBRARY_PATH": ld}
    if project:
        env["CUDA_ENV_PROJECT"] = project
    out = subprocess.run(
        ["bash", "-c", f'. "{SCRIPT}" "{venv}"; env'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _fake_venv(tmp_path: Path, cuda: str | None) -> Path:
    v = tmp_path / "proj" / ".venv"
    t = v / "lib" / "python3.12" / "site-packages" / "torch"
    t.mkdir(parents=True)
    if cuda is not None:
        (t / "version.py").write_text(f"__version__ = '2.11.0'\ncuda: Optional[str] = '{cuda}'\n")
    return v


def test_detects_tag_and_scrubs_global_cuda(tmp_path):
    v = _fake_venv(tmp_path, "12.8")
    e = _run(
        v,
        "/usr/local/cuda-13.2/lib64:/opt/amazon/openmpi/lib:/usr/local/cuda-13.2/extras/CUPTI/lib64",
    )
    assert e["CUDA_ENV_TAG"] == "cu128"
    assert "cuda-13.2" not in e["LD_LIBRARY_PATH"]
    assert "/opt/amazon/openmpi/lib" in e["LD_LIBRARY_PATH"]
    assert e["TRITON_CACHE_DIR"].endswith("/proj/cu128/triton")


def test_caches_differ_by_project_and_tag(tmp_path):
    a = _run(_fake_venv(tmp_path / "a", "13.2"), "", project="p1")
    b = _run(_fake_venv(tmp_path / "b", "12.8"), "", project="p1")
    c = _run(_fake_venv(tmp_path / "c", "13.2"), "", project="p2")
    keys = ["TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH", "VLLM_CACHE_ROOT"]
    for k in keys:
        assert len({a[k], b[k], c[k]}) == 3


def test_cpu_torch_gets_cpu_tag(tmp_path):
    e = _run(_fake_venv(tmp_path, None), "/usr/local/cuda/lib64")
    assert e["CUDA_ENV_TAG"] == "cpu"
    assert e["LD_LIBRARY_PATH"] == ""
