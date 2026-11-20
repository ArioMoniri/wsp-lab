"""Study M (MedGemma / Gemma 3) code that must not live in files the frozen 9B study (wme-c1) registers or hashes.

Chair rule (20 Nov 2026): study M never edits ``wm/genaudit.py``, ``wm/panel.py``, ``wm/p3gen.py``, ``wm/c1/``,
``wsp/gen/`` or any file pinned in ``docs/prereg/wme_manifest*.json`` (``actions.py``, ``controls.py``, ``hf.py``, …).
Gemma-specific behaviour lives here and calls into those modules without changing them.
"""
