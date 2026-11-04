# Board A re-review: feat/h4-reader 89f2459

Verdict: REVISE.

## Blocking finding

1. **Incompatible endpoint query can silently reuse predictions** (`src/wsp/agents/reader.py:327`, `:395`, `:636`). `describe()` records only `safe_endpoint`, which removes the entire query. Two endpoints differing only in a non-secret routing or deployment query parameter therefore compare equal on resume, even though `_join_url` retains that query for the actual request. A resumed run can reuse predictions generated against a different deployment and misstate its configuration. Record a non-reversible fingerprint of the complete endpoint (or otherwise distinguish query changes without exposing credentials), and test rejection of changed query settings. The existing incompatible-resume test changes the host, token limit and token field, but does not exercise this case.

## Evidence

The manifest and eval-index scoring restrictions and excluded IDs have direct tests; missing manifest is rejected. Resume tests verify retained per-entry provenance and unknown legacy provenance, and generation settings are checked except for the endpoint query issue above. The parametrized credential test covers both providers and eight failure branches, including persisted files and logs. HTTPX and httpcore loggers are set to WARNING. An additional mocked 429 with a key containing a quote and backslash in a JSON-encoded error body redacted the escaped key. The dry CLI default has a direct test. `source scripts/env.sh && pytest -q`: 297 passed, 8 deselected. No real provider calls were made.
