"""Isolated vLLM generation with HF teacher-forced scoring and identical EvalRecords.

The subprocess uses .venv-vllm's torch; no vLLM is imported in the HF environment.
Engine startup and snapshot export are included in entry wall time.
"""

import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from wsp.eval.run import HFRunner


class VLLMRunner(HFRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._worker = None
        self._snapshot = None
        self._key = None
        self._is_base = False

    def wrap_base_context(self):
        original = self.base_context

        @contextmanager
        def context():
            previous = self._is_base
            self._is_base = True
            try:
                with original():
                    yield
            finally:
                self._is_base = previous

        self.base_context = context

    def close(self):
        if self._worker is not None:
            self._worker.stdin.close()
            try:
                self._worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._worker.terminate()
                self._worker.wait(timeout=30)
            self._worker = None
        if self._snapshot is not None:
            self._snapshot.cleanup()
            self._snapshot = None
        self._key = None

    def generate(self, prompts):
        if not prompts:
            return []
        started = time.monotonic()
        key = "base" if self._is_base else self.edit_id
        if self._key != key:
            self.close()
            self._snapshot = tempfile.TemporaryDirectory(prefix="wsp-vllm-")
            self.model.save_pretrained(self._snapshot.name)
            self.tokenizer.save_pretrained(self._snapshot.name)
            root = Path(__file__).resolve().parents[3]
            python = os.environ.get("WSP_VLLM_PYTHON", str(root / ".venv-vllm/bin/python"))
            self._worker = subprocess.Popen(
                [python, "-m", "wsp.eval.vllm_worker", self._snapshot.name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._key = key
        request = {"prompts": [self.chat(p) for p in prompts], "generation": self.generation}
        self._worker.stdin.write(json.dumps(request) + "\n")
        self._worker.stdin.flush()
        for line in self._worker.stdout:
            if line.startswith("WSP_RESULT="):
                results = json.loads(line[len("WSP_RESULT=") :])
                self.generation_stats["tokens"] += sum(r[1] for r in results)
                self.generation_stats["seconds"] += time.monotonic() - started
                return results
            print(line, end="", flush=True)
        raise RuntimeError(f"vLLM worker exited: {self._worker.wait()}")
