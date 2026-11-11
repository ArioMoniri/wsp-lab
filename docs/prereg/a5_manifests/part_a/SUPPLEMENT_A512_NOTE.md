# A5.1.2 hash entries after the F11 supplementary staging and the F11.4 calibration: text for the chair to append

This file is not part of `docs/PREREG_H1_TF.md`. The chair appends the paragraph below to A5.1.2, after re-pin (xxi). The
hashes were re-computed on 11 Nov 2026 at about 09:50 UTC from the files at branch `report/f11-staging` (based on main
`92b0338`), except the calibration output, which is under `data/` and was hashed in place.

---

*A5.1.2 hash entries for the F11 supplementary staging and the F11.4 calibration (11 Nov 2026, about 09:50 UTC, after execution;
the pre-execution freeze is re-pin (xx), `supplement_freeze.json` written 08:05:19 UTC; before any G3 outcome).*

**Staging.**

- Ran 08:06:02–09:26:07 UTC at clean execution SHA `8211ecf4c079f32dd8cc59b77fdfa5bd8a99a7bc` (exit 0), with Board A's frozen
  staging command verbatim. Outputs were committed in `92b0338`.
- All 71 pins of `code_hashes.json` (`46dd52341df19ad0…`, re-pin (xx)) match at that SHA.
- `manifest.json:code_sha` and `provenance.git_sha` record `887a514`, the HEAD when the manifest was written; main advanced
  during the run with analysis and documentation commits only. The execution SHA is `8211ecf`.
- The freeze and the emitted replay environment agree on every field, including Torch build-config `07f0372b…`, recomputed
  under the frozen environment.

Supplementary yield and price:

- **12 / 12 supplementary rotation pairs pass F10** (535 / 535 tensors). **43 / 52 rotation pairs are available**, and the 9
  exclusions are the original nine.
- 3,069 tensors edited, 2,519 rotated. 52 ladders per strength.
- Priced count: 178 entries, 185.12 min, with the 12 supplementary originals unproven (proofs deferred to `score --manifest`).

New and regenerated files under `docs/prereg/a5_manifests/part_a/`:

- `manifest.json` (merged, 52 edits): `71e77a3e1f703451864ce960129399be1579d8aa84631bebc0c8d6324b4790d6`
- `rotation_audits.jsonl` (pre-F11 bytes, then 535 supplementary rows; 3,069 rows):
  `9412eb0d4d60f5587c4c5d19e9afcda86c6c45faa38a37b3ab937008d505c45a`
- `manifest_pre_f11.json` (preserved, byte-identical to the pre-run `manifest.json`):
  `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06`
- `rotation_audits_pre_f11.jsonl` (preserved, byte-identical to the pre-run `rotation_audits.jsonl`):
  `4ec0928a91b5523a400b911f00f8045f267386c2589aa72a58816d565f5c97f3`
- `supplement_launch.json` (launch and end record): `d17fab2f7648a405045b85bcb92e309a5ea11fb19a6d864a852ccb72bd38fb38`
- `supplement_freeze.json` (pre-execution freeze, (xx), for reference):
  `503e415efda3801216cc57ba332ab643a333e4b73cb92ce6366add0682fda0a4`
- `slot_mask_f11.json` (F11.4 mask, schema `a5-slot-mask-v1`, built by the accepted builder from `manifest.json`; rotation 43,
  ladder 0.25 52, ladder 0.5 52; canonical mapping sha256 `171dfd20b47020dbc172d39eac87adf5c6efbe7864a0f3a784ef018fcb19741a`):
  `5f4e770185ec9cc6fa10461d3012196bbd3feb9ca4af9f3065510a634b5bbfc4`
- `SUPPLEMENT_REPORT.md` (the F11.3 staging report and F11.4 calibration record): `a88cfe7b6280169f68c301a0fdce4e98ac37ba6b3c9ec5a8ea3f913b1c2bc656`

Launch wrappers. These are outside the repository, were not frozen before execution, and are recorded here after the fact:

- `/tmp/a5_f11_supplement.sh`: `1dcc133a4b1cfd8e356fd6b2ff7613054fc43033a0035c83cbd1c3d0eb6c92e4`
- `/tmp/a5_f11_calibration.sh`: `00c3a8bb46acbd1e34f095deba6b8a5f58167060e1c4a8414b70ae23de91b19b`

The forty's evidence is unchanged:

- 249 staged files (129 safetensors and 120 `variant.json`) re-hash equal to `manifest_pre_f11.json`, `manifest.json` and the
  freeze inventory.
- Their per-entry records are carried identically.
- `manifest_a533.json` (`d7785b35…`), `rotation_audits_a533.jsonl` (`f05fec42…`), `README.md` (`2fcbf517…`) and
  `slot_mask_f10.json` (`86fa27a5…`) are unchanged.

**Calibration.**

- Output: `data/runs/a5/power/power_slotmask_f11.json`, `8a1be18f726d5b030dbaa5937dcc63ba9a3ca9f730d314f689a4fbb7631441fb`.
- Run at clean HEAD `92b0338`, 09:28:04–09:34:03 UTC, exit 0, with Board A's frozen command.
  `scripts/a5_simulation.py` is `4ce3faa5…` (re-pin (xix)).
- Inputs: `inputs.json` `67da223c…`, `fields.json` `85843ed2…`, `registered_donors.json` `6e77bca6…`,
  `selection_f11.json` `ce746e71…`, mask `5f4e7701…`.
- Settings: 10,000 replicates and 4,000 draws; NULL-SIZE on the full 141-donor payload; POWER-AND-YIELD on the 61-donor pool.
- Result: ordered power 0.8722, CP95 [0.8655, 0.8787] ≥ 0.80. Size: ordered go 0.0175; two-sided 0.0348 / 0.0378 / 0.0378, all
  ≤ 0.07. Null alternative rate 0.2576 (disclosed, not gated). **PASS.**

Recorded discrepancies (report §10):

- The selection builder `scripts/a5_f11_supplement.py` (`90dcb312…`) and `tests/test_a5_f11_supplement.py` (`6cf4f07c…`)
  are not rows of `code_hashes.json`. Their bytes equal the reviewed hashes.
- The launch record lacks argv, environment, job/log identity and output hashes.
- Re-pin (xx)'s "81-file" inventory should read 129 files in 111 variants.
- The calibration ran before the staging report existed. No separate `slot_mapping_f11.json` was frozen; the ordered slot table
  appears only in the calibration output.

Nothing has been scored. The scoring hold remains until Board A accepts the staging report and the calibration, and every
remaining F11 and analysis gate passes.
