# F8a floor payload (C1 file)

`nullsize_inputs.json` is a byte-identical copy of `data/runs/a5/nullsize/inputs.json` (the registered NULL-SIZE payload,
sha256 `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de`, 651,039 bytes, written 8 Nov 2026 by
`scripts/a5_nullsize_inputs.py materialise`; see `docs/prereg/a5_manifests/nullsize/`). It is committed so that the F8a floor can be
recomputed from a fresh clone by someone other than the builder before any F8 contraction, as F8a condition 2 requires. The
design-time preview of the floor (component-weighted median of per-component median 95 % half-widths, h* = 0.0049397 nats/token) is
in `docs/design/F8_EXECUTION_SPEC.md` §2.3; the binding value must come from committed code run on this file.

## Independent recompute (11 Nov 2026)

`floor_recompute_2026-11-11.json` was produced by the orchestrating session (not the author of `scripts/f8_floor.py`) from a fresh
shallow clone of main at the merge of `feat/f8-tooling`, running
`scripts/f8_floor.py --payload docs/prereg/f8_manifests/floor_payload/nullsize_inputs.json`. Result h* = 0.0049397 nats/token,
pair-weighted alternative 0.0011513 (non-binding); both equal the design-time preview in `docs/design/F8_EXECUTION_SPEC.md` §2.3.
The report records the payload and code sha256 values. This discharges the reproduction part of Board A F8a condition 1; the
binding floor for the F8 gate is the value in this report once the C1 freeze cites it.
