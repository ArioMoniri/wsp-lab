"""Private append-only episode audit trail."""

import hashlib
import json
import os
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scrub_arrays(value):
    def numeric(v):
        return all(numeric(x) for x in v) if isinstance(v, list) else isinstance(v, (int, float))

    def shape(v):
        return [len(v)] + (shape(v[0]) if v else []) if isinstance(v, list) else []

    if isinstance(value, list):
        if numeric(value):
            return {"sha256": digest(value), "shape": shape(value)}
        return [scrub_arrays(v) for v in value]
    if isinstance(value, dict):
        # Hash/shape descriptors are already summaries, not payload arrays.
        if set(value) == {"sha256", "shape"}:
            return value
        return {k: scrub_arrays(v) for k, v in value.items()}
    return value


class Episode:
    def __init__(self, root, episode_id):
        if Path(episode_id).name != episode_id:
            raise ValueError("episode id")
        self.path = Path(root) / episode_id / "episode.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def append(self, event, **fields):
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(
                json.dumps(dict(event=event, timestamp=time.time(), **scrub_arrays(fields))) + "\n"
            )
            f.flush()
            os.fsync(f.fileno())
