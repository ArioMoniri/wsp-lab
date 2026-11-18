"""27B separation (Board A condition 13; PREREG_WME §10): an independent-data replication, kept apart from 9B.

* **Fresh identities.** Panel salt :data:`SALT` = ``wme-c27``; bank namespace :data:`BANK_NAMESPACE` =
  ``wme-bank27`` (every 27B root seed is :func:`seed_for`). A fresh salt or namespace alone is **not** sufficient:
* **Item exclusion by manifest identity** (:data:`EXPOSURES_27`, :func:`load_exposures27`,
  :func:`build_c27_panel`): every ``wme-c1`` group (sealed test included, through the manifest's sealed hash
  list; sealed files are never opened), every pilot-1 and pilot-2 group, and every item the 27B engineering
  calibration read (pilot-1 construction, validation and dev). Every required exposure must be present - a missing
  manifest raises :class:`MissingExposure`, never a skip. Near-duplicate groups are joined before exclusion
  (``wm.exposure``), and the shared builder ``wm.panel_c1.build_c1_panel`` runs with the 27B salt.
* **Root exclusion by J2-J5** (:func:`accept_roots27`): a candidate 27B root is rejected if it joins any 9B bank or
  pilot root, either calibration timing edit (:data:`CALIB_TIMING_EDITS`), or an already accepted 27B root:
  J2 shared contrast/adapter source item group; J3 |cos| >= 0.5 between directions in the same space; J4 scale/noise
  roots sharing a tensor of the same model, or row sets with Jaccard >= 0.5; J5 an equal realised-delta hash.
  J3 and J4 compare roots of the same model only (a 9B and a 27B tensor are different objects); J2 and J5 apply
  across models. A root outside the ``wme-bank27`` namespace is rejected.
* **27B-only primary selection** (:func:`select_primary_27`): RULING-wme-pilot1 §3's Go-E2 rule (an objective
  passes with >= 2 candidates whose dev gain LCB > 0 and no guardrail breach; most qualifying wins; ties
  mathcomp > gsm8k > sycophancy) on 27B's **own** dev items only: every input must carry provenance model =
  Qwen3.8-27B, panel = ``wme-c27``, split = ``dev``, else :class:`SeparationError`. A primary other than
  ``sycophancy_real`` makes 27B a separate experiment on a new selected objective, not a replication.
* **No 9B test outcome in 27B tuning** (:func:`guard_tuning`, :class:`Reads27`): every design-element value
  (objectives, families, nominals, margins, B, N, arms) carries provenance; a 9B test, generation, winner-test or
  sealed-read source raises :class:`Tuning9BError`; other 9B confirmatory sources are allowed only with the label
  "informed by 9B confirmatory results" (an adaptation, §11.4). The read guard refuses 9B confirmatory outcome
  locations lexically and after resolving symlinks, and logs every read with its sha256.
* **Field configuration** (:func:`field_config_27`): the only configuration that passed Go-M on 27B - two
  2-layer groups over layers 24-27 (68.81 GB against 72), items <= 768 tokens; anything else raises.
* **Launch gate** (:func:`launch_gate_27`): no 27B confirmatory run before the 9B analysis is sealed and the dated
  27B amendment freezes the pilot-only selection, panel, rules, Ns and caps, with every difference labelled.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from wsp.wm import calib27 as K
from wsp.wm import exposure as X
from wsp.wm import panel_c1 as C1
from wsp.wm.plan import SealedError

SALT = "wme-c27"
BANK_NAMESPACE = "wme-bank27"
MODEL = K.REPO
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
EXPERIMENT = "27B independent-data replication (fresh panel wme-c27, fresh roots wme-bank27); never pooled with 9B"

EXPOSURES_27 = dict(
    X.EXPOSURES,
    **{
        "wme-c1": {
            "manifest": "data/wme/panels/wme-c1/manifest.json",
            "splits": "all",
            "note": "every wme-c1 group, sealed test included (manifest sealed_hashes; sealed files never opened)",
        }
    },
)
REQUIRED_EXPOSURES = ("wme-c1", "wme-v0-p1", "wme-p2", "calib27-20261116", "wme-p3s", "wme-p3h")
CALIB_REPORT = "docs/reviews/wme/wme-27b-calib-2026-11-16.json"
CALIB_TIMING_EDITS = (
    {
        "id": "calib27:noise",
        "model": MODEL,
        "family": "noise",
        "tensor": [28, "down"],
        "seed": 7,
        "rho": 0.05,
        "source": CALIB_REPORT,
    },
    {
        "id": "calib27:scale",
        "model": MODEL,
        "family": "scale",
        "tensor": [27, "o"],
        "factor": 1.25,
        "source": CALIB_REPORT,
    },
)
J3_MAX_COS = 0.5
J4_ROW_JACCARD = 0.5
FIELD_LAYERS_PER_GROUP = 2
FIELD_MAX_ITEM_TOKENS = 768
GO_M = {"peak_gb": 68.81, "cap_gb": 72.0, "config": "two 2-layer groups, items <= 768 tokens"}
PRIMARY_MIN_QUALIFYING = 2
PRIMARY_TIE_ORDER = ("mathcomp_gold_lp", "gsm8k_gold_lp", "sycophancy_real")  # (iii) > (ii) > (i)
NINE_B_PRIMARY = "sycophancy_real"
RELATION = {
    "replication": "independent-data replication of the 9B primary objective",
    "separate": "separate experiment on a new selected objective (not a replication of the 9B enhancement claim)",
    "none": "no 27B enhancement objective passed its own dev gate",
}
DESIGN_ELEMENTS = ("objectives", "families", "nominals", "margins", "B", "N", "arms")
SOURCE_KINDS = {
    "forbidden": (
        "9b_test_outcome",
        "9b_generation_outcome",
        "9b_winner_test_score",
        "9b_sealed_read",
    ),
    "informed": ("9b_confirmatory_analysis",),
    "independent": (
        "27b_pilot_dev",
        "27b_calibration_engineering",
        "outcome_blind",
        "registration",
    ),
}
LABELS = {
    "independent": "independent of 9B confirmatory results",
    "informed": "informed by 9B confirmatory results",
}
FORBIDDEN_READ_ROOTS = (
    "data/wme/panels/wme-c1/sealed_test",
    "data/wme/panels/wme-c1/sealed_gen",
    "data/wme/runs/wme-c1",
    "data/wme/winners/wme-c1",
)
AMENDMENT_FIELDS = (
    "pilot_selection",
    "final_panel",
    "model_metric_rules",
    "sim_ns",
    "cost_caps",
    "differences",
)


class SeparationError(PermissionError):
    pass


class MissingExposure(SeparationError):
    pass


class Tuning9BError(SeparationError):
    pass


def seed_for(label: str) -> int:
    return int(hashlib.sha256(f"{BANK_NAMESPACE}:{label}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ items


def load_exposures27(root, registry=EXPOSURES_27) -> dict:
    missing = [
        n for n in REQUIRED_EXPOSURES if not (Path(root) / registry[n]["manifest"]).is_file()
    ]
    if missing:
        raise MissingExposure(f"exposure manifests missing (never skipped): {missing}")
    return X.load_exposures(root, registry)


def build_c27_panel(pools: dict, sizes: dict, out_dir, exposures: dict) -> dict:
    missing = [n for n in REQUIRED_EXPOSURES if n not in exposures]
    if missing:
        raise MissingExposure(
            f"a fresh salt alone is not sufficient; exclusions missing: {missing}"
        )
    man = C1.build_c1_panel(pools, sizes, out_dir, exposures, salt=SALT)
    man.update(
        experiment=EXPERIMENT,
        model={"repo": MODEL, "revision": REVISION},
        required_exposures=list(REQUIRED_EXPOSURES),
        bank_namespace=BANK_NAMESPACE,
    )
    (Path(out_dir) / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


# ------------------------------------------------------------------ roots


def excluded_identities(nine_b_roots=(), pilot_roots=()) -> list:
    """Every identity a 27B root must not join: 9B bank roots, pilot roots and the calibration timing edits."""
    return [dict(r) for r in (*nine_b_roots, *pilot_roots, *CALIB_TIMING_EDITS)]


def _cos(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _joins(c, e) -> str | None:
    for key in ("contrast_source_groups", "adapter_source_groups"):
        if set(c.get(key) or ()) & set(e.get(key) or ()):
            return "J2"
    same_model = c.get("model") == e.get("model")
    has_dir = c.get("direction") is not None and e.get("direction") is not None
    if (
        same_model
        and has_dir
        and c.get("space") == e.get("space")
        and abs(_cos(c["direction"], e["direction"])) >= J3_MAX_COS
    ):
        return "J3"
    if (
        same_model
        and {c.get("family"), e.get("family")} <= {"scale", "noise"}
        and c.get("tensor") is not None
        and c.get("tensor") == e.get("tensor")
    ):
        return "J4"
    if (
        same_model
        and c.get("family") == e.get("family") == "row_scale"
        and c.get("tensor") == e.get("tensor")
    ):
        a, b = set(c.get("rows") or ()), set(e.get("rows") or ())
        if a and b and len(a & b) / len(a | b) >= J4_ROW_JACCARD:
            return "J4"
    if set(c.get("realised_delta") or ()) & set(e.get("realised_delta") or ()):
        return "J5"
    return None


def accept_roots27(candidates, excluded) -> tuple[list, list]:
    """Rejection sampling over J2-J5 against the excluded identities and every accepted 27B root, in order."""
    accepted, rejected = [], []
    for c in candidates:
        reasons = []
        if c.get("namespace") != BANK_NAMESPACE or c.get("model") != MODEL:
            reasons.append("namespace")
        for e in [*excluded, *accepted]:
            j = _joins(c, e)
            if j:
                reasons.append(f"{j}:{e['id']}")
        (rejected if reasons else accepted).append(dict(c, reasons=reasons) if reasons else c)
    return accepted, rejected


# ------------------------------------------------------------------ primary


def _check_provenance(r):
    p = r.get("provenance") or {}
    if p.get("model") != MODEL or p.get("panel") != SALT or p.get("split") != "dev":
        raise SeparationError(
            f"27B primary selection reads only 27B dev items of {SALT}; got {json.dumps(p, sort_keys=True)}"
        )


def select_primary_27(results) -> dict:
    for r in results:
        _check_provenance(r)
    q = {o: 0 for o in PRIMARY_TIE_ORDER[::-1]}
    for r in results:
        if r["objective"] not in q:
            raise ValueError(f"unknown objective {r['objective']}")
        if r["lcb"] > 0 and not r["guard_breach"]:
            q[r["objective"]] += 1
    passing = [o for o in PRIMARY_TIE_ORDER if q[o] >= PRIMARY_MIN_QUALIFYING]
    if not passing:
        return {"primary": None, "qualifying": q, "relation": RELATION["none"]}
    primary = max(passing, key=lambda o: (q[o], -PRIMARY_TIE_ORDER.index(o)))
    rel = RELATION["replication"] if primary == NINE_B_PRIMARY else RELATION["separate"]
    return {"primary": primary, "qualifying": q, "relation": rel}


# ------------------------------------------------------------------ tuning guard


def guard_tuning(records) -> list:
    out = []
    for rec in records:
        if rec["element"] not in DESIGN_ELEMENTS:
            raise ValueError(f"unknown 27B design element {rec['element']!r}")
        if not rec.get("sources"):
            raise ValueError(f"{rec['element']}: provenance is required")
        classes = set()
        for s in rec["sources"]:
            cls = next((k for k, v in SOURCE_KINDS.items() if s["kind"] in v), None)
            if cls is None:
                raise ValueError(f"unknown source kind {s['kind']!r}")
            if cls == "forbidden":
                raise Tuning9BError(
                    f"{rec['element']}: a 9B test/generation outcome may not tune 27B ({s})"
                )
            classes.add(cls)
        informed = "informed" in classes
        out.append(
            dict(rec, label=LABELS["informed" if informed else "independent"], adaptation=informed)
        )
    return out


class Reads27:
    """Read guard for 27B tuning code: refuses 9B confirmatory outcome locations (lexical and resolved)."""

    def __init__(self, root, forbidden=FORBIDDEN_READ_ROOTS):
        self.root = Path(os.path.abspath(root))
        self.forbidden = [Path(os.path.realpath(self.root / f)) for f in forbidden]
        self._lex = [self.root / f for f in forbidden]
        self.reads: list[dict] = []

    def _check(self, p: Path, real: Path):
        for lex, f in zip(self._lex, self.forbidden, strict=True):
            for cand, base in ((p, lex), (real, f)):
                if cand == base or base in cand.parents:
                    raise SeparationError(f"{p}: a 9B confirmatory outcome location ({base})")
        if any(part.startswith("sealed") for part in real.parts) or any(
            part.startswith("sealed") for part in p.parts
        ):
            raise SeparationError(f"{p}: sealed path")

    def read_bytes(self, path) -> bytes:
        p = Path(os.path.abspath(path))
        real = Path(os.path.realpath(p))
        self._check(p, real)
        try:
            fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as e:
            raise SealedError(f"{path}: {e}") from None
        with os.fdopen(fd, "rb") as f:
            data = f.read()
        self.reads.append(
            {"path": str(p), "resolved": str(real), "sha256": hashlib.sha256(data).hexdigest()}
        )
        return data

    def read_json(self, path):
        return json.loads(self.read_bytes(path).decode())


# ------------------------------------------------------------------ field and launch


def validate_field_config(layers_per_group: int, max_item_tokens: int):
    if layers_per_group != FIELD_LAYERS_PER_GROUP or max_item_tokens > FIELD_MAX_ITEM_TOKENS:
        raise ValueError(
            f"27B field: only {GO_M['config']} passed Go-M ({GO_M['peak_gb']} GB against {GO_M['cap_gb']})"
        )


def field_config_27(rm) -> dict:
    validate_field_config(FIELD_LAYERS_PER_GROUP, FIELD_MAX_ITEM_TOKENS)
    return {
        "layers": list(K.SUPPORT_LAYERS),
        "layers_per_group": FIELD_LAYERS_PER_GROUP,
        "groups": K.support_groups(rm, FIELD_LAYERS_PER_GROUP),
        "max_item_tokens": FIELD_MAX_ITEM_TOKENS,
        "go_m": dict(GO_M),
        "dead_man_terminator": "required (scripts/lambda_deadman.py)",
    }


def check_field_items(token_lengths):
    over = [n for n in token_lengths if n > FIELD_MAX_ITEM_TOKENS]
    if over:
        raise ValueError(f"{len(over)} field items exceed {FIELD_MAX_ITEM_TOKENS} tokens")


def launch_gate_27(seal_9b, amendment):
    if not seal_9b or seal_9b.get("sealed") is not True or len(str(seal_9b.get("hash", ""))) != 64:
        raise SeparationError(
            "27B runs only after the 9B confirmatory analysis is sealed (hash required)"
        )
    missing = [k for k in AMENDMENT_FIELDS if not (amendment or {}).get(k)]
    if missing:
        raise SeparationError(
            f"the 27B amendment must freeze {missing} before any 27B confirmatory outcome"
        )
    for d in amendment["differences"]:
        if d.get("label") not in LABELS.values():
            raise SeparationError(
                f"difference {d.get('what')!r} is not labelled {sorted(LABELS.values())}"
            )
    return True
