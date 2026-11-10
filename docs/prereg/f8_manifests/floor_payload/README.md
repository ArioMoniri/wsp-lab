# F8a floor payload (C1 file)

`nullsize_inputs.json` is a byte-identical copy of `data/runs/a5/nullsize/inputs.json` (the registered NULL-SIZE payload,
sha256 `67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de`, 651,039 bytes, written 8 Nov 2026 by
`scripts/a5_nullsize_inputs.py materialise`; see `docs/prereg/a5_manifests/nullsize/`). It is committed so that the F8a floor can be
recomputed from a fresh clone by someone other than the builder before any F8 contraction, as F8a condition 2 requires. The
design-time preview of the floor (component-weighted median of per-component median 95 % half-widths, h* = 0.0049397 nats/token) is
in `docs/design/F8_EXECUTION_SPEC.md` §2.3; the binding value must come from committed code run on this file.
