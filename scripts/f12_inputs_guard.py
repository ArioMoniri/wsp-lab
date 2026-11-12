#!/usr/bin/env python3
"""F12 inputs guard (PREREG_H1_TF A5 note F12-R; unexposed adoption review of 12 Nov 2026, items A10, A19 and A21).

Standard library only. Shared by ``f12_population.py``, ``f12_tokens.py``, ``f12_additivity_prepare.py`` and
``f12_additivity_stream.py``. It changes no computation; it decides only whether a governed input may be read, whether
it is authentic, and whether a job may start or must stop.

* **Read boundary (A10).** ``ReadBoundary`` resolves every path to its leaf (``os.path.realpath`` follows every
  symlink and ``..``) and refuses it before any open if the lexical or the resolved path lies under a prohibited root
  (``data/evals`` in every spelling, Part A staging/prediction/calibration, the F12f pilot report, any ``part_a*`` or
  ``sealed*`` tree under the data root) or outside the job's allowed roots. Reads use ``O_NOFOLLOW`` on the resolved
  leaf. Every read is recorded (path, resolved path, sha256, whether it was checked against a pin): the exact allowed
  inputs. Hash pins authenticate bytes; they are integrity checks, the boundary is the access control.
* **Stored edit bytes (A10).** ``entry_delta_hash`` is the A5 convention (``a5_build_manifests.entry_delta_hash``:
  sha256 of ``name:sha256(file)`` lines over ``delta.safetensors`` or the files of ``lora/``), computed through the
  boundary, so each population row's bytes are authenticated against its ``resolved_delta_hash``.
* **Checkpoints (A19).** ``atomic_write_bytes`` (temporary file, fsync, ``os.replace``, directory fsync),
  ``prefix_sha256`` and ``strict_json_loads`` (refuses NaN and infinities).
* **Resource authority (A21).** ``ResourceGuard`` holds a job to the chair's cap (PREREG_H1_TF A5 chair resource
  disposition, 12 Nov 2026, 08:32 UTC, (d)): 4 GB peak RSS unless a dated chair ruling committed before launch names
  the build, a cap and the thread count in a line ``resource-cap: build=<script> cap_bytes=<int> threads=<int>``
  inside its own ``#### A5 chair ruling ... (<date> ...)`` block of the committed (HEAD) preregistration. It refuses
  to start if the projected peak exceeds the cap or free memory is below the cap (cap + 4 GB under a ruling), samples
  the process tree's RSS and stops the job (exit code 75) once it exceeds 95 % of the cap, checks the process peak
  between units, and records the peak. It is not an OS sandbox: sampling cannot see a spike shorter than its interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/ubuntu/wsp-data")
CANONICAL_ROOT = Path("/home/ubuntu/projects/wsp-lab")
PREREG_REL = "docs/PREREG_H1_TF.md"
DEFAULT_CAP_BYTES = 4_000_000_000
LAUNCH_HEADROOM_BYTES = 4_000_000_000
STOP_FRACTION = 0.95
BREACH_EXIT_CODE = 75
AUTHORITY = (
    "PREREG_H1_TF A5 chair resource disposition on F12 input builds (12 Nov 2026, 08:32 UTC), (d): 4 GB peak RSS "
    "per F12/F12-I job unless a dated chair ruling, committed before launch, names the build, a cap and the threads"
)
RULING_LINE = re.compile(r"resource-cap: build=(\S+) cap_bytes=(\d+) threads=(\d+)")
DATE = re.compile(r"\d{1,2} [A-Z][a-z]{2} \d{4}|\d{4}-\d{2}-\d{2}")
PROHIBITED_DATA_PREFIXES = ("part_a", "sealed")


class BoundaryError(PermissionError):
    pass


class AuthenticationError(ValueError):
    pass


class ResourceRefused(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm(p) -> Path:
    return Path(os.path.abspath(p))


def prohibited_roots(root=ROOT, data_root=DATA_ROOT) -> list[Path]:
    """Lexical and resolved spellings of every prohibited root (resolution stats path components only)."""
    lexical = [
        Path(root) / "data/evals",
        CANONICAL_ROOT / "data/evals",
        Path(data_root) / "evals",
        Path(root) / "docs/prereg/a5_manifests/part_a",
        CANONICAL_ROOT / "docs/prereg/a5_manifests/part_a",
        Path(data_root) / "runs/a5/part_a_predictions",
        Path(root) / "docs/prereg/f12_manifests/f12f_pilot_report.json",
        CANONICAL_ROOT / "docs/prereg/f12_manifests/f12f_pilot_report.json",
    ]
    out = set()
    for p in lexical:
        out.add(_norm(p))
        out.add(Path(os.path.realpath(p)))
    return sorted(out)


def _under(p: Path, roots) -> bool:
    return any(p == r or r in p.parents for r in roots)


class ReadBoundary:
    """The job's read boundary: allowed roots (resolved), prohibited roots, and a record of every read."""

    def __init__(self, allowed, prohibited=None, data_root=DATA_ROOT):
        self.allowed = sorted({Path(os.path.realpath(a)) for a in allowed})
        self.prohibited = list(prohibited_roots() if prohibited is None else map(_norm, prohibited))
        self.prohibited += [Path(os.path.realpath(p)) for p in self.prohibited]
        self.data_roots = {_norm(data_root), Path(os.path.realpath(data_root))}
        self.reads = []

    def _prohibited(self, p: Path) -> bool:
        if _under(p, self.prohibited):
            return True
        for d in self.data_roots:
            if d in p.parents:
                rel = p.relative_to(d).parts
                if any(c.startswith(PROHIBITED_DATA_PREFIXES) for c in rel):
                    return True
        return False

    def resolve(self, path) -> Path:
        """The resolved leaf of ``path``, or BoundaryError (lexical check first, then the resolved leaf)."""
        lexical = _norm(path)
        if self._prohibited(lexical):
            raise BoundaryError(f"{path}: prohibited path (refused before opening)")
        real = Path(os.path.realpath(path))
        if self._prohibited(real):
            raise BoundaryError(
                f"{path}: resolves to the prohibited path {real} (refused before opening)"
            )
        if not _under(real, self.allowed):
            raise BoundaryError(
                f"{path}: resolves to {real}, outside the allowed roots {[str(a) for a in self.allowed]}"
            )
        return real

    def _open(self, real: Path) -> int:
        return os.open(real, os.O_RDONLY | os.O_NOFOLLOW)

    def read_bytes(self, path, sha256=None) -> bytes:
        real = self.resolve(path)
        fd = self._open(real)
        with os.fdopen(fd, "rb") as f:
            data = f.read()
        self._record(path, real, sha256_bytes(data), sha256)
        return data

    def sha256_file(self, path, sha256=None, record=True) -> str:
        """Streamed sha256 of the resolved leaf; ``record=False`` for members of a separately recorded index."""
        real = self.resolve(path)
        h = hashlib.sha256()
        fd = self._open(real)
        with os.fdopen(fd, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        got = h.hexdigest()
        if record:
            self._record(path, real, got, sha256)
        elif sha256 is not None and got != sha256:
            raise AuthenticationError(f"{path}: sha256 {got} differs from the pinned {sha256}")
        return got

    def check(self, path, note=None) -> Path:
        """Boundary-check a path that a library opens itself (recorded without a hash)."""
        real = self.resolve(path)
        if not real.exists():
            raise FileNotFoundError(str(path))
        self.reads.append(
            {"path": str(path), "resolved": str(real), "sha256": None, "authenticated": False}
            | ({"note": note} if note else {})
        )
        return real

    def _record(self, path, real, got, want):
        if want is not None and got != want:
            raise AuthenticationError(f"{path}: sha256 {got} differs from the pinned {want}")
        self.reads.append(
            {
                "path": str(path),
                "resolved": str(real),
                "sha256": got,
                "authenticated": want is not None,
            }
        )

    def record(self) -> dict:
        return {
            "rule": "every read resolved to its leaf (symlinks, ..) and refused before opening if prohibited or "
            "outside the allowed roots; O_NOFOLLOW on the resolved leaf; pins checked after reading",
            "allowed": [str(a) for a in self.allowed],
            "prohibited": sorted({str(p) for p in self.prohibited}),
            "prohibited_data_prefixes": list(PROHIBITED_DATA_PREFIXES),
            "reads": list(self.reads),
        }


def safe_name(name: str) -> str:
    """A bare file name (no separators, not '.', '..' or empty)."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
        raise BoundaryError(f"{name!r}: not a bare file name")
    return name


# --------------------------------------------------------------------------------- stored edit bytes


def entry_files(boundary: ReadBoundary, entry_dir: Path) -> list[Path]:
    """``a5_build_manifests.artefact_files``: the dense delta, else the sorted files of ``lora/``."""
    entry_dir = Path(entry_dir)
    boundary.resolve(entry_dir)
    dense = entry_dir / "delta.safetensors"
    if dense.exists():
        return [dense]
    lora = entry_dir / "lora"
    if not lora.exists():
        return []
    boundary.resolve(lora)
    return sorted(p for p in lora.glob("*") if p.is_file())


def entry_delta_hash(boundary: ReadBoundary, entry_dir: Path):
    files = entry_files(boundary, entry_dir)
    if not files:
        return None
    return sha256_bytes("".join(f"{p.name}:{boundary.sha256_file(p)}\n" for p in files).encode())


def authenticate_entry(boundary: ReadBoundary, entry_dir: Path, expected: str) -> str:
    got = entry_delta_hash(boundary, entry_dir)
    if got is None:
        raise AuthenticationError(f"{entry_dir}: no stored bytes")
    if got != expected:
        raise AuthenticationError(
            f"{entry_dir}: entry_delta_hash {got} differs from the population's {expected}"
        )
    return got


# --------------------------------------------------------------------------------------- checkpoints


def atomic_write_bytes(path: Path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return sha256_bytes(data)


def prefix_sha256(path: Path, nbytes: int) -> str:
    with open(path, "rb") as f:
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"{path}: shorter than the checkpointed {nbytes} bytes")
    return sha256_bytes(data)


def _no_constant(name):
    raise ValueError(f"nonfinite JSON constant {name}")


def strict_json_loads(text):
    return json.loads(text, parse_constant=_no_constant)


# ------------------------------------------------------------------------------- resource authority


def mem_available_bytes(meminfo="/proc/meminfo") -> int:
    with open(meminfo) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def _children(pid: int) -> list[int]:
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{tid}/children") as f:
                out += [int(x) for x in f.read().split()]
    except OSError:
        pass
    return out


def _rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def tree_rss_bytes(pid=None) -> int:
    """Current RSS of the process and all its descendants."""
    stack, seen, total = [pid or os.getpid()], set(), 0
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        total += _rss(p)
        stack += _children(p)
    return total


def peak_rss_bytes() -> int:
    """Peak RSS of this process (and of any reaped child), bytes."""
    return 1024 * max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )


def committed_prereg_text(root=ROOT, rel=PREREG_REL) -> str:
    """The preregistration as committed at HEAD (inherited GIT_* variables scrubbed)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    r = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{rel}"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return r.stdout


def verify_ruling(text: str, heading: str, build: str, cap_bytes: int, threads: int) -> dict:
    """The dated chair ruling ``heading`` and its own ``resource-cap`` line for exactly this build, cap and threads."""
    heading = heading.strip()
    if not heading.startswith("#### A5 chair ruling"):
        raise ResourceRefused(f"{heading!r}: not an A5 chair ruling heading")
    lines = text.splitlines()
    try:
        i = next(k for k, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        raise ResourceRefused(
            f"chair ruling heading not in the committed preregistration: {heading!r}"
        )
    if not DATE.search(heading):
        raise ResourceRefused(f"{heading!r}: the ruling is not dated")
    block = []
    for line in lines[i + 1 :]:
        if line.startswith("#### "):
            break
        block.append(line)
    for line in block:
        m = RULING_LINE.search(line)
        if m and (m.group(1), int(m.group(2)), int(m.group(3))) == (build, cap_bytes, threads):
            return {"heading": heading, "build": build, "cap_bytes": cap_bytes, "threads": threads}
    raise ResourceRefused(
        f"{heading!r}: no resource-cap line for build={build} cap_bytes={cap_bytes} threads={threads}"
    )


class ResourceGuard:
    """Launch refusal, process-tree RSS monitor, between-unit peak check and peak record for one job."""

    def __init__(
        self,
        build: str,
        threads: int,
        cap_bytes: int = DEFAULT_CAP_BYTES,
        ruling: str | None = None,
        prereg_text: str | None = None,
        rss_fn=tree_rss_bytes,
        mem_fn=mem_available_bytes,
        peak_fn=peak_rss_bytes,
        exit_fn=os._exit,
        interval: float = 0.2,
        stop_fraction: float = STOP_FRACTION,
    ):
        self.build, self.threads, self.cap = build, int(threads), int(cap_bytes)
        self.ruling_heading, self.prereg_text = ruling, prereg_text
        self.rss_fn, self.mem_fn, self.peak_fn, self.exit_fn = rss_fn, mem_fn, peak_fn, exit_fn
        self.interval, self.stop_at = interval, int(stop_fraction * self.cap)
        self.ruling = None
        self.max_sampled = 0
        self._stop = threading.Event()
        self._thread = None
        self.launch_record = None

    def launch(self, estimate_bytes: int, breach_path=None) -> dict:
        if self.cap > DEFAULT_CAP_BYTES:
            if not self.ruling_heading:
                raise ResourceRefused(
                    f"cap {self.cap} bytes exceeds the 4 GB limit: a dated chair ruling committed before launch "
                    "is required (--chair-ruling)"
                )
            text = self.prereg_text if self.prereg_text is not None else committed_prereg_text()
            self.ruling = verify_ruling(
                text, self.ruling_heading, self.build, self.cap, self.threads
            )
        if estimate_bytes > self.cap:
            raise ResourceRefused(
                f"projected peak {estimate_bytes} bytes is above the cap {self.cap} bytes; refused at launch"
            )
        need = self.cap + (LAUNCH_HEADROOM_BYTES if self.ruling else 0)
        avail = self.mem_fn()
        if avail < need:
            raise ResourceRefused(f"{avail} bytes available, {need} required at launch")
        self.launch_record = {
            "estimate_bytes": int(estimate_bytes),
            "mem_available_at_launch_bytes": int(avail),
            "required_available_bytes": int(need),
            "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._breach_path = breach_path
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self.launch_record

    def _monitor(self):
        while not self._stop.is_set():
            r = int(self.rss_fn())
            self.max_sampled = max(self.max_sampled, r)
            if r > self.stop_at:
                rec = {
                    "build": self.build,
                    "cap_bytes": self.cap,
                    "stop_threshold_bytes": self.stop_at,
                    "sampled_tree_rss_bytes": r,
                    "peak_rss_bytes": int(self.peak_fn()),
                    "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if self._breach_path is not None:
                    atomic_write_bytes(
                        Path(self._breach_path), (json.dumps(rec, indent=1) + "\n").encode()
                    )
                self.exit_fn(BREACH_EXIT_CODE)
                return
            self._stop.wait(self.interval)

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def check(self):
        peak = int(self.peak_fn())
        if peak > self.cap:
            raise MemoryError(f"peak RSS {peak} bytes exceeds the cap {self.cap} bytes")

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=5)
        peak = int(self.peak_fn())
        return {
            "authority": AUTHORITY,
            "build": self.build,
            "threads": self.threads,
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "cap_bytes": self.cap,
            "ruling": self.ruling,
            "stop_threshold_bytes": self.stop_at,
            "monitor_interval_seconds": self.interval,
            "peak_rss_bytes": peak,
            "max_sampled_tree_rss_bytes": int(self.max_sampled),
            "within_cap": peak <= self.cap,
            "enforcement": "launch refusal (projection, free memory, ruling); process-tree RSS sampling with stop "
            "at the threshold; peak check between units; not an OS sandbox",
        } | (self.launch_record or {})


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
