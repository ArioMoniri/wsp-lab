# G2 outputs (first run, 10 Nov 2026)

Copies of the G2 manifest and the audit file as left by the run that computed all five fields (1.45 GPU-h), the projectors and the probe cache, and stopped the predecessor audit after one measured row (refusal_margin_harmful, 394 s wall, relative error 0.336). Raw artefacts stay under `data/runs/g2/` (field tensors ≈ 11 GB per objective, not committed).

Final manifest (`g2_manifest_final.json`): fields, projectors and probe cache from run 1; the registered audit stage stopped after one measured row (recorded); the F6 `chain_exact` stage completed 20 exact-gradient rows (10 chain picks × WikiText, MMLU) in about 15 minutes on 11 Nov 2026.

| file | sha256 |
|---|---|
| `docs/prereg/a5_manifests/g2/g2_manifest_final.json` | `c4ec7390aaf118ed785c390ce337a4077936a31204f97da54341bcfa54fd09ff` |
| `docs/prereg/a5_manifests/g2/g2_manifest_run1.json` | `636a967466cfcc3c728c6828f145f2474e04e658625701ab6f505ff00809c022` |
