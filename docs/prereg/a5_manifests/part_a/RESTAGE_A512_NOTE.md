# A5.1.2 hash entries after the F10 re-stage: text for the chair to append

This file is not part of `docs/PREREG_H1_TF.md`. The paragraph below is meant to be appended to A5.1.2 after re-pin (xviii). The
chair appends it. The hashes were re-computed on 11 Nov 2026 at about 07:17 UTC from the files at branch `report/f10-restage`
(based on main `be02cc3`).

---

*A5.1.2 hash entries for the F10 re-stage (11 Nov 2026, about 07:17 UTC, after execution; the pre-execution freeze is re-pin
(xviii), 05:45 UTC entry, `restage_freeze.json` written 02:40:50 UTC; before any G3 outcome).*

The rotation-only re-stage ran 02:44:00–06:22:28 UTC at clean execution SHA `7bacec12604f1090168a79a6c6d55d4f548af4ec`
(exit 0). Its outputs were committed in `848d7a5`. All 71 pins of `code_hashes.json` (`74321d76858274d2…`) match at that SHA and
at `be02cc3`.

Regenerated and new files under `docs/prereg/a5_manifests/part_a/`:

- `manifest.json` (regenerated): `59d839e351824fa154a665fcf105284ce5fdeed8423a5c0a2b7a60687b0d3b06`
- `rotation_audits.jsonl` (regenerated; 2,534 rows): `4ec0928a91b5523a400b911f00f8045f267386c2589aa72a58816d565f5c97f3`
- `restage_launch.json` (launch and end record): `28dcba683dc215fe303d938a1c7b094b5d0856929f19fb720187b75bb1a59a7e`
- `restage_freeze.json` (pre-execution freeze, (xviii), for reference):
  `03dfaf2667d0addbd459c0662b8fa87a2304b36a054c36cf64ac29192ca48858`
- `slot_mask_f10.json`, F10.6 mask as used. Schema `a5-slot-mask-v1`, rewritten in `cf3eae8`. It masks the 9 excluded rotation
  slots; mapping sha256 `b7724a69315d8c8692a128504126251ac16945cf7b741ec60b3cdb6daa5cb329`. File:
  `86fa27a5c1838d0c0bd88ef530feba22eceafac698d0ec4506fbf05df83b698e`.
  - The first mask file, committed in `848d7a5`, was `a2ed6ae7e9ac8e08850ddbaded875f9fda65704ea261bdb50ddb080da1a2fe80`. The
    simulator rejected it for unknown keys, so nothing was simulated with it.
- `slot_mask_f10.provenance.json`: `0bb56b4bb05ce0718f48e7fc962a60e4378b06d9c46c9cf08e4b1ef78be08636`
- `slotmask_calibration_record.json`: `0c5fa8c38da42d63bd5762c307ec1c2254313501fbfd4763b3418eaa3eff6b51`
- `RESTAGE_REPORT.md` (the re-stage report, which serves as the F10.7 regenerated README):
  `61294c8666b4bbe4f938e76af7797631ef86613e77d431229bddcaf69d6d5f61`

Unchanged files:

- `README.md` (the a533 staging README): `2fcbf517ccf26f2a1fe325ef31300d291bc0f254c7faaacd18154e327bceada1`
- `manifest_a533.json`: `d7785b35de1d7041edd2db139591562379db12e33c2532921c814735e8dde8d6`
- `rotation_audits_a533.jsonl`: `f05fec4265ffd93c8c537100993d23262be95141369240fcad0a7e16ff474d35`

Calibration output: `data/runs/a5/power/power_slotmask_f10.json`,
`b14756828bc27cbbe14e1e590226820371a65ef5f83ad9622b2e6ccf04a3450a`. Run at `cf3eae8`, 06:25:29–06:36:01 UTC, with
`scripts/a5_simulation.py` `c4f0c55f…` (re-pin (xvi)). Result: ordered power 0.7514 < 0.80, **FAIL**; size passes.

Yield: 31 / 40 rotation pairs; 1,984 / 2,534 tensors rotated. The 80 ladder variants (96 files) and the prior passing pair's
delta (`951e8621…`) and increment (`78ec4689…`) are byte-identical to the staged hashes. 16 ladder entries are flagged
`realised_increment_ladder`. Priced count: 130 entries, 135.2 min.

Recorded discrepancies:

- The emitted Torch build-config SHA-256 (`07f0372b…`) differs from the frozen value (`d1b4e24c…`).
- The freeze's `base_revision` is empty. All 40 selected edits record `2fc06364…`, the model pin.

Nothing has been scored. The scoring hold remains (F10.6; Board A F10 condition 5).
