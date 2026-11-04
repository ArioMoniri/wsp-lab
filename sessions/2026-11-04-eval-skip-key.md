# P2: stable evaluation skip/resume key (issue #27)

Implemented in fix/eval-skip-key. Resume uses entry id, effective evaluation config hash (including the runner's default batch size), resolved model revision, and zoo reuse_key. Source hash, git SHA, and dirty flag remain provenance only. --force re-evaluates base and entries. Index keys include model revision and reuse_key.

Legacy signature-bearing results are matched from recorded effective config and model revision, ignoring code provenance. Legacy files did not record reuse_key: compatibility binds them to the current zoo row in the repaired index without rewriting eval.json. New results record the exact key. Fixture tests copy the real result for entry 4463696a4c62e8f3, run 20261104-080338-4d5632a1.

CPU suite: 397 passed, 8 deselected in 26.02 s; Ruff check and format clean. Additional selection/index migration test passes. Read-only verification of current D3 subset: one existing complete eval.json recognised, zero rejected. No GPU work or changes to production artefacts. No merge requested or performed.
