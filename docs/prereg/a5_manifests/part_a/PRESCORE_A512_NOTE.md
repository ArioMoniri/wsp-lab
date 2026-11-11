This file is not part of `docs/PREREG_H1_TF.md`. It holds the A5.1.2 hash entries for the Part A pre-scoring inventory and a
proposed attestation line. The chair appends both. The hashes below were computed on 11 Nov 2026 at about 09:53 UTC on branch
`prereg/prescore-inventory`, from main `079e60b0d65da1f893d726c80acb9d2a0874acba`.

---

*A5.1.2 hash entries for the Part A pre-scoring inventory (11 Nov 2026, about 09:53 UTC; after the F11.3 staging and the F11.4
calibration; before any G3 outcome; scoring held).*

**New files** under `docs/prereg/a5_manifests/part_a/`:
- `PRESCORE_INVENTORY.json` (machine-readable inventory, checks and gate table):
  `2ab1caea1dce1f4ac2afb50a59e80d3f3cf3c19ae4a922144a61bbddd9cc2160`
- `PRESCORE_INVENTORY.md` (readable inventory): `419ee039de63ea75014b9271e3a2718e593bf9d17b5d83900533511c04100f0c`

**F11.3 staging outputs** (staging run at clean `8211ecf`, 08:06:02–09:26:07 UTC, exit 0; committed in `92b0338`):
- `manifest.json` (execution manifest, 52 edits): `71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6`
- `rotation_audits.jsonl` (3,069 rows; the first 2,534 are the pre-F11 bytes):
  `9412eb0d4d60f5587c4c5d19e9afcda86c6c45faa38a37b3ab937008d505c45a`
- `manifest_pre_f11.json` (preserved): `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06`
- `rotation_audits_pre_f11.jsonl` (preserved): `4ec0928a91b5523a400b911f00f8045f267386c2589aa72a58816d565f5c97f3`
- `supplement_freeze.json`: `503e415efda3801216cc57ba332ab643a333e4b73cb92ce6366add0682fda0a4`
- `supplement_launch.json`: `d17fab2f7648a405045b85bcb92e309a5ea11fb19a6d864a852ccb72bd38fb38`
- `slot_mask_f11.json`: `5f4e770185ec9cc6fa10461d3012196bbd3feb9ca4af9f3065510a634b5bbfc4`. The mapping sha256
  `171dfd20b47020dbc172d39eac87adf5c6efbe7864a0f3a784ef018fcb19741a` recomputes from the execution manifest.
- Slot table of the F11.4 output (`json.dumps(slot_mask.slots, sort_keys=True)`):
  `68b8b01db16f9c267943e160fafbd8e6aecb15fa5d820f3eae62a5ecd6b4620e`. The separate `slot_mapping_f11.json` was not written.

**F11.4 calibration** (run at clean `92b0338`, 09:28:04–09:34:03 UTC; committed in `079e60b`):
- `f11_calibration_record.json`: `216bdc9c578e576cf67ea1dfc85a1a89850c69c6c9c39fbb4fc5b8fcda26f5a4`
- output `data/runs/a5/power/power_slotmask_f11.json`: `8a1be18f726d5b030dbaa5937dcc63ba9a3ca9f730d314f689a4fbb7631441fb`
- Result: ordered power 0.8722 ≥ 0.80; ordered-go size 0.0175; two-sided 0.0348 / 0.0378 / 0.0378. **PASS**, pending Board A
  acceptance.
- Wrapper `/tmp/a5_f11_calibration.sh` (not committed): `00c3a8bb46acbd1e34f095deba6b8a5f58167060e1c4a8414b70ae23de91b19b`.

**Population identities:**
- Supplementary sorted-id sha256: `1848fe4cacbaacc9f643a179835770fefd3027a48cef7cf9d83358f15b85fdd9`
- Registered-40 sorted-id sha256: `e2bd6cb2956c0fbe64194b3f1ebbb481248ecec45b8a729a59a54b9a2bac9d6b`

**Scoring launch script** `scripts/part_a_score_launch.sh` (`1b954a6`; not executed):
`69eae67c9028ee0352eb39346b7201f9b43edc065540ca6d648e00f0e148de41`.

**Unchanged and verified** (re-hashed equal to their earlier entries):
- `manifest_a533.json` `d7785b35…8dde8d6`
- `rotation_audits_a533.jsonl` `f05fec42…74d35`
- `README.md` `2fcbf517…ceada1`
- `RESTAGE_REPORT.md` `61294c86…d5f61`
- `slot_mask_f10.json` `86fa27a5…3b698e`
- `slotmask_calibration_record.json` `0c5fa8c3…eff6b51` (F10.6 FAIL, superseded)
- `selection_f11.json` `ce746e71…4793c`
- `selection.json` `39353036…0e908c63`
- `code_hashes.json` `34477308…` (re-pin (xxi))

`--verify-pins HEAD` at `079e60b`: all pins match. Staged bytes: 24 of 165 delta/increment files and all 156 `variant.json` files
re-hashed equal at 09:37–09:47 UTC.

**Gaps recorded by the inventory** (§12 of `PRESCORE_INVENTORY.md`):
- the F11.3 staging report and its acceptance;
- Board A acceptance of the F11.4 report and route;
- `slot_mapping_f11.json`;
- the analysis tamper/binding exercise on the final manifest;
- the frozen 52-edit field-prediction file;
- the dated F11.8 deferred-proof / stop-and-reprice clarification;
- the launch script's proof-only stop (LS-1–LS-4);
- the pending attestations.

Nothing has been scored. The scoring hold remains.

---

*Proposed attestation entry 15 for `docs/prereg/f10_attestations.md` (pre-score inventory builder; the chair adds it).*

- **Role:** built the Part A pre-scoring inventory (`PRESCORE_INVENTORY.json`/`.md`) and this note.
- **Session identity:** Claude Opus 5.5 (claude_umut instance), branch `prereg/prescore-inventory` (worktree
  `.worktrees/prereg/prescore-inventory`), delegated by the chair. It started from main `079e60b` on 11 Nov 2026 at 09:35:06 UTC (git reflog).
- **Coverage:** from branch creation through the commit that adds these files.
- **Read and ran:** the F10/F11/F11a/F11b text and re-pins (xviii)–(xxi) of `docs/PREREG_H1_TF.md`; the Board A reviews (F10,
  F10-code-2, re-stage, F11, F11-stage, F11-sim, analysis-4); `f10_attestations.md`; the `part_a/` manifests, audits, masks and
  records; `selection_f11.json`; `code_hashes.json`; `scripts/part_a_score_launch.sh`; PROGRESS entries.
  - It re-hashed staged bytes under `data/runs/a5/part_a/` and the simulation outputs and inputs under `data/runs/a5/power/` and
    `data/runs/a5/nullsize/`.
  - It ran `--verify-pins HEAD` and `nvidia-smi` queries (identity only), on CPU only.
- **Attestation:** Under `data/evals` I ran existence (`lexists`) and name-walk checks only and opened nothing. I opened no zoo
  `eval.json`, `records.jsonl` or `summary.json`, and wrote nothing under `data/`. I used no GPU for computation and ran no scoring,
  staging, simulation or analysis. I have seen no G3, rotation, ladder, reference or sealed outcome, or a summary of one; none
  exists. I made no methodological choice. This attests this session only.
