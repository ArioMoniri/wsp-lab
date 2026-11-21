#!/usr/bin/env python3
"""Prefetch the pinned Gate 1 dataset files into the HF cache (network on); the driver then runs offline."""

import importlib.util
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("g1", ROOT / "scripts/wmm_gate1.py")
os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import hf_hub_download

g1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1)
for (src, _split), f in g1.FILES.items():
    repo, rev = g1.DATA[src]
    print(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
