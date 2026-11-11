#!/usr/bin/env python3
"""F12d restricted-payload extractor (PREREG_H1_TF F12a.4(b),(e); Board A F12c review condition 6, "Missing donor flag").

Re-extracts the F12 restricted copy of the F8a.2 qualification noise payload from the pinned committed source
``docs/prereg/f8_manifests/floor_payload/nullsize_inputs.json`` (sha256 ``67da223c...3de``; byte-identical to
``data/runs/a5/nullsize/inputs.json``), keeping ONLY the allowlisted fields:

* top level: ``window_ids``, ``donors``, ``centred_noise``, ``window_mask``;
* per donor: ``id``, ``component``, ``family`` and the original ``eligible`` flag.

The source digest is checked on the raw bytes before anything is parsed. The top-level object is then scanned key by key:
a non-allowlisted value (``donor_mean``, ``inventory``, ``provenance``, ``spec``, ...) is skipped by a string-aware bracket
matcher and is never decoded into a Python value; nested donor objects keep only the four donor fields (other donor keys
are dropped as each object closes). No non-allowlisted value is selected, printed, retained or used.

The registered eligibility rule is applied explicitly (``ELIGIBILITY_RULE``): the flag is read, never derived from
finiteness; a missing or non-boolean flag fails closed. Donor identities, components, families and flags and the 64
window ids are cross-checked against the committed metadata extract
``docs/prereg/a5_manifests/nullsize/inputs_provenance.json`` (no numeric arrays) and the registered window order of
``docs/prereg/a5_manifests/nullsize_inputs.json``; masks must agree with the noise values; the 32 two-window blocks are
(2j, 2j + 1) in that order. Any membership change against the superseded copy is reported by ``membership_diff``.

    python scripts/f12_payload_extract.py --out /home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun/noise_payload_restricted_v2.json \\
        [--superseded /home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun/noise_payload_restricted.json] [--report R.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "f12-noise-payload-restricted-v2"
SUPERSEDED_SCHEMA = "f12-noise-payload-restricted-v1"
SOURCE_REL = "docs/prereg/f8_manifests/floor_payload/nullsize_inputs.json"
SOURCE_SHA256 = "67da223ce315e8256a754d05dd0af725f3480fabc46d3b736114072ef11c53de"
REFERENCE_REL = "docs/prereg/a5_manifests/nullsize/inputs_provenance.json"
WINDOW_SPEC_REL = "docs/prereg/a5_manifests/nullsize_inputs.json"
TOP_FIELDS = ("window_ids", "donors", "centred_noise", "window_mask")
DONOR_FIELDS = ("id", "component", "family", "eligible")
WINDOWS, BLOCKS = 64, 32
DEFAULT_OUT = Path("/home/ubuntu/wsp-data/runs/a5/f12_sim_dryrun/noise_payload_restricted_v2.json")
ELIGIBILITY_RULE = (
    "F12a.4(b),(e): a donor is used iff its original `eligible` flag is the boolean true and all 64 windows are "
    "masked in with finite centred noise; the flag is read from the source and never derived from finiteness; a "
    "missing or non-boolean flag fails extraction closed"
)
FUNCTIONS = (
    "scan_allowlisted (top-level key scan; non-allowlisted values skipped undecoded)",
    "_donor_hook (donor objects keep id, component, family, eligible)",
    "extract (digest before parse; validation; ELIGIBILITY_RULE; reference cross-check)",
    "serialise (json.dumps(sort_keys=True))",
)


class ExtractionError(ValueError):
    pass


# ------------------------------------------------------------------ allowlisted scanning

_WS = " \t\n\r"


def _ws(text, i):
    while i < len(text) and text[i] in _WS:
        i += 1
    return i


def _skip_value(text, i):
    """Index just past the JSON value starting at ``i``, without decoding it (strings, nesting tracked)."""
    depth, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            if depth == 0:
                return i
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            if depth == 0:
                return i
            depth -= 1
            if depth == 0:
                return i + 1
        elif ch == "," and depth == 0:
            return i
        i += 1
    if depth:
        raise ExtractionError("unterminated value in the source payload")
    return i


def _donor_hook(pairs):
    """Nested objects keep only the donor allowlist; every other key's value is dropped as the object closes."""
    return {k: v for k, v in pairs if k in DONOR_FIELDS}


def scan_allowlisted(text, allow=TOP_FIELDS):
    """Decode only the allowlisted top-level values of a JSON object. Returns (values, top-level key names)."""
    dec = json.JSONDecoder(object_pairs_hook=_donor_hook)
    i = _ws(text, 0)
    if i >= len(text) or text[i] != "{":
        raise ExtractionError("the source payload is not a JSON object")
    i, out, keys = i + 1, {}, []
    while True:
        i = _ws(text, i)
        if i < len(text) and text[i] == "}":
            break
        if i >= len(text) or text[i] != '"':
            raise ExtractionError("malformed top-level key in the source payload")
        key, i = json.decoder.scanstring(text, i + 1)
        keys.append(key)
        i = _ws(text, i)
        if i >= len(text) or text[i] != ":":
            raise ExtractionError("malformed top-level object in the source payload")
        i = _ws(text, i + 1)
        if key in allow:
            if key in out:
                raise ExtractionError(f"duplicate allowlisted key {key!r}")
            out[key], i = dec.raw_decode(text, i)
        else:
            i = _skip_value(text, i)
        i = _ws(text, i)
        if i < len(text) and text[i] == ",":
            i += 1
            continue
        if i < len(text) and text[i] == "}":
            break
        raise ExtractionError("malformed top-level object in the source payload")
    return out, keys


# --------------------------------------------------------------------------- extraction


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def extract(source, *, expected_sha256=SOURCE_SHA256, reference=None, source_label=None):
    """Restricted v2 object and an extraction report (counts, exclusions by reason, key names only)."""
    raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ExtractionError(f"source digest {digest} != pinned {expected_sha256}: nothing parsed")
    vals, keys = scan_allowlisted(raw.decode("utf-8"))
    missing = [k for k in TOP_FIELDS if k not in vals]
    if missing:
        raise ExtractionError(f"source lacks allowlisted fields {missing}")
    wids, donors = vals["window_ids"], vals["donors"]
    noise, mask = vals["centred_noise"], vals["window_mask"]
    if (
        not isinstance(wids, list)
        or len(wids) != WINDOWS
        or len(set(wids)) != WINDOWS
        or not all(isinstance(w, str) for w in wids)
    ):
        raise ExtractionError("window ids are not 64 distinct strings")
    if not (isinstance(donors, list) and len(donors) == len(noise) == len(mask)):
        raise ExtractionError("donors, centred_noise and window_mask differ in length")
    for d in donors:
        if not isinstance(d, dict) or not all(
            isinstance(d.get(k), str) for k in ("id", "component", "family")
        ):
            raise ExtractionError("donor record lacks id / component / family")
        if "eligible" not in d:
            raise ExtractionError(
                f"donor {d['id']}: original eligible flag missing (no flag is manufactured)"
            )
        if not isinstance(d["eligible"], bool):
            raise ExtractionError(f"donor {d['id']}: eligible flag is not boolean (never coerced)")
    ids = [d["id"] for d in donors]
    if len(set(ids)) != len(ids):
        raise ExtractionError("duplicate donor ids")
    complete = []
    for d, v, m in zip(donors, noise, mask):
        if not (isinstance(v, list) and isinstance(m, list) and len(v) == len(m) == WINDOWS):
            raise ExtractionError(f"donor {d['id']}: noise / mask rows must hold 64 windows")
        for x, ok in zip(v, m):
            if not isinstance(ok, bool) or ok != _finite(x) or (not ok and x is not None):
                raise ExtractionError(f"donor {d['id']}: window mask and centred noise disagree")
        complete.append(all(m))
    flag_false = [d["id"] for d in donors if d["eligible"] is False]
    incomplete = [d["id"] for d, c in zip(donors, complete) if d["eligible"] and not c]
    used = [j for j, (d, c) in enumerate(zip(donors, complete)) if d["eligible"] and c]
    ref_note = None
    if reference is not None:
        want = [tuple(x[k] for k in DONOR_FIELDS) for x in reference["donors"]]
        got = [tuple(d[k] for k in DONOR_FIELDS) for d in donors]
        if want != got or list(reference["window_ids"]) != wids:
            raise ExtractionError(
                "reference metadata differ (donor ids, components, families, flags or window ids)"
            )
        ref_note = "donor ids, components, families, eligible flags and window ids equal"
    used_donors = [{k: donors[j][k] for k in DONOR_FIELDS} for j in used]
    restricted = {
        "schema": SCHEMA,
        "provenance": {
            "source": source_label or str(source),
            "sha256": digest,
            "git_blob_sha1": git_blob_sha1(raw),
            "fields_read": list(TOP_FIELDS),
            "donor_fields_read": list(DONOR_FIELDS),
            "not_read": "donor_mean and every other field: skipped undecoded by the top-level scanner",
            "eligibility_rule": ELIGIBILITY_RULE,
            "blocks": "two-window blocks (2j, 2j + 1), j = 0..31, in window_ids order",
            "extractor": "scripts/f12_payload_extract.py",
        },
        "window_ids": wids,
        "donors": used_donors,
        "centred_noise": [noise[j] for j in used],
        "window_mask": [mask[j] for j in used],
    }
    report = {
        "source_sha256": digest,
        "rule": ELIGIBILITY_RULE,
        "counts": {
            "source_donors": len(donors),
            "flag_true": sum(d["eligible"] for d in donors),
            "flag_false": len(flag_false),
            "incomplete_windows": len(incomplete),
            "used": len(used),
            "used_components": len({d["component"] for d in used_donors}),
        },
        "excluded": {"flag_false": flag_false, "incomplete_windows": incomplete},
        "top_level_keys_seen": keys,
        "not_selected_top_level_keys": [k for k in keys if k not in TOP_FIELDS],
        "reference_check": ref_note,
    }
    return restricted, report


def serialise(obj) -> bytes:
    return json.dumps(obj, sort_keys=True).encode()


def membership_diff(old, new) -> dict:
    """Donor membership / identity changes of the new restricted copy against the superseded one."""
    o = {d["id"]: d for d in old["donors"]}
    n = {d["id"]: d for d in new["donors"]}
    common = sorted(set(o) & set(n))
    comp = [i for i in common if o[i]["component"] != n[i]["component"]]
    fam = [i for i in common if o[i]["family"] != n[i]["family"]]
    elig = [i for i in common if "eligible" in o[i] and o[i]["eligible"] != n[i]["eligible"]]
    oi = {d["id"]: j for j, d in enumerate(old["donors"])}
    ni = {d["id"]: j for j, d in enumerate(new["donors"])}
    same_noise = all(old["centred_noise"][oi[i]] == new["centred_noise"][ni[i]] for i in common)
    same_mask = all(old["window_mask"][oi[i]] == new["window_mask"][ni[i]] for i in common)
    added, removed = sorted(set(n) - set(o)), sorted(set(o) - set(n))
    out = {
        "n_donors_old": len(o),
        "n_donors_new": len(n),
        "n_components_old": len({d["component"] for d in old["donors"]}),
        "n_components_new": len({d["component"] for d in new["donors"]}),
        "donors_added": added,
        "donors_removed": removed,
        "components_changed": comp,
        "families_changed": fam,
        "eligibility_changed": elig,
        "old_copy_had_eligible_flag": any("eligible" in d for d in old["donors"]),
        "donor_order_identical": [d["id"] for d in old["donors"]]
        == [d["id"] for d in new["donors"]],
        "window_ids_identical": old["window_ids"] == new["window_ids"],
        "centred_noise_identical": same_noise,
        "window_mask_identical": same_mask,
    }
    out["unchanged"] = not (added or removed or comp or fam or elig) and same_noise and same_mask
    return out


def reference_metadata(root=ROOT) -> dict:
    """Allowlisted donor metadata of the committed extract (no numeric arrays) and the registered window order."""
    prov = json.loads((root / REFERENCE_REL).read_text())
    if prov.get("payload_sha256") != SOURCE_SHA256:
        raise ExtractionError("committed metadata extract is not of the pinned payload")
    spec = json.loads((root / WINDOW_SPEC_REL).read_text())
    wids = spec["payload_schema"]["window_item_ids"]
    if prov["window_ids"] != wids:
        raise ExtractionError("metadata extract window ids differ from the registered window order")
    return {
        "donors": [{k: d[k] for k in DONOR_FIELDS} for d in prov["donors"]],
        "window_ids": wids,
    }


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", type=Path, default=ROOT / SOURCE_REL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--superseded", type=Path, default=None, help="old restricted copy (v1)")
    p.add_argument("--report", type=Path, default=None)
    a = p.parse_args(argv)
    src = a.source.resolve()
    if src != (ROOT / SOURCE_REL).resolve():
        raise SystemExit(f"the source must be the pinned committed copy {SOURCE_REL}")
    if _git("status", "--porcelain", "--", SOURCE_REL, "scripts/f12_payload_extract.py"):
        raise SystemExit("source or extractor differs from the committed bytes: commit first")
    if _git("ls-files", "--", SOURCE_REL) != SOURCE_REL:
        raise SystemExit("the source payload is not committed")
    out = a.out.resolve()
    if ROOT.resolve() in out.parents or "data/evals" in str(out):
        raise SystemExit("write the restricted copy outside the repository (wsp-data)")
    restricted, report = extract(src, reference=reference_metadata(), source_label=SOURCE_REL)
    head = _git("rev-parse", "HEAD")
    report["source_committed_in"] = _git("log", "--format=%H", "-1", "--", SOURCE_REL)
    report["source_git_blob_sha1"] = restricted["provenance"]["git_blob_sha1"]
    report["source_git_blob_at_head"] = _git("rev-parse", f"HEAD:{SOURCE_REL}")
    if report["source_git_blob_at_head"] != report["source_git_blob_sha1"]:
        raise SystemExit("source bytes differ from the committed blob")
    restricted["provenance"]["extractor_commit"] = head
    restricted["provenance"]["extractor_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    data = serialise(restricted)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_bytes() != data:
        raise SystemExit(f"{out} exists with different bytes (restricted copies are immutable)")
    out.write_bytes(data)
    report["restricted_copy"] = {
        "path": str(out),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    report["extractor"] = {
        "script": "scripts/f12_payload_extract.py",
        "commit": head,
        "sha256": restricted["provenance"]["extractor_sha256"],
        "functions": list(FUNCTIONS),
    }
    if a.superseded is not None:
        old = json.loads(Path(a.superseded).read_bytes())
        if old.get("schema") != SUPERSEDED_SCHEMA:
            raise SystemExit("--superseded is not the v1 restricted copy")
        report["superseded_copy_sha256"] = hashlib.sha256(
            Path(a.superseded).read_bytes()
        ).hexdigest()
        report["membership_vs_superseded"] = membership_diff(old, restricted)
    text = json.dumps(report, indent=1, sort_keys=True)
    if a.report is not None:
        a.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
