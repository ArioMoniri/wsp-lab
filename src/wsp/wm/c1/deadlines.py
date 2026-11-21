"""Response deadlines per request kind (chair item FF after the run4 resume halt, 21 Nov 2026). Engineering; no rule
here reads an outcome.

Every controller->worker request has a bounded response deadline, counted from the moment it is put in the worker's
inbox. When it passes without a result or a rejection, the controller withdraws the request and records a
``request_deadline`` run-health incident (``ctl.RequestDeadline``, an ``IncidentAbort``: an in-flight unit attempt is
``attempt_aborted``, never counted). The run then halts until the incident is diagnosed and reported. The stall rules
of §II.31 (GPU idle, CPU progress; 15 min) still apply inside the deadline.

**Units: generous.** ``UNIT_FACTOR`` times the guard's registered per-unit estimate of the unit's phase
(``study.EST_UNIT_GPU_H``; 1 GPU on the main instance): bank 2.8 h, confirmatory and S-BoN 6 h, S-TE 10 h. The field
repair re-collection (``field_repair``) and its dependent-unit check (``bank_verify``) re-run bank units and take the
bank deadline. For comparison, the longest observed units: a field group 229 s and a bank side 177 s (the study's
dispatch journal). The largest confirmatory unit, one E1 generation decoding, is 1,000 registered items at the
measured K1 rate (64 canary items in 21 s), about 25 min. The registered S-TE pieces are at most 3,154 s
(``prereg_v2.pricing.s_te.seconds``).

**Checks: short.** Fixed bounds of at least 5x the longest observed time (the study's dispatch journal; the run4
parity instance on the K1 code). hello 10 min (observed 18 s, model load included); canary 30 min (observed 98 s;
the K1 child reloads the model, about 95 s); timing 30 min (observed 18 s); inert identity and parity 1 h (observed
353 s and 274 s); gen_repeat 30 min (95 s); field status and quarantine 10 min (file operations). The CPU auxiliary
requests (``theta_of``, ``describe``, ``edit_bundles``, ``rstats``) get 30 min.

A test keeps this table complete: every worker handler kind has a deadline, and the unit deadlines follow
``study.EST_UNIT_GPU_H``.
"""

from __future__ import annotations

UNIT_FACTOR = 8.0
# the guard's registered per-unit estimates (GPU-h); equal to wsp.wm.c1.study.EST_UNIT_GPU_H (a test enforces it; not
# imported, because study imports ctl, which reads this table)
EST_UNIT_GPU_H = {"bank": 0.35, "confirmatory": 0.75, "sbon": 0.75, "ste": 1.25}

UNIT_PHASE = {
    **dict.fromkeys(
        (
            "field_group",
            "field_vocab",
            "bank_base",
            "bank_side",
            "bank_comp",
            "bank_inert",
            "field_repair",
            "bank_verify",
        ),
        "bank",
    ),
    **dict.fromkeys(
        ("library", "dev_base", "fit", "dev_eval", "e2_arm", "sealed_eval", "e1_gen"),
        "confirmatory",
    ),
    "sbon_arm": "sbon",
    **dict.fromkeys(("ste_calibrate", "ste_train", "ste_generate"), "ste"),
}

CHECK_S = {
    "hello": 600.0,
    "canary": 1800.0,
    "timing": 1800.0,
    "gen_repeat": 1800.0,
    "inert_identity": 3600.0,
    "parity": 3600.0,
    "field_status": 600.0,
    "field_quarantine": 600.0,
    "field_timing": 3600.0,  # addendum 4 T4.3: three g0 collections (registered ~2.5 min each; T2: no slowdown)
    "theta_of": 1800.0,
    "describe": 1800.0,
    "edit_bundles": 1800.0,
    "rstats": 1800.0,
}

DEADLINE_S = {
    **{k: UNIT_FACTOR * EST_UNIT_GPU_H[ph] * 3600.0 for k, ph in UNIT_PHASE.items()},
    **CHECK_S,
}


# a kind outside the table (fixture kinds in the CPU tests and drill only; the table covers every worker handler)
DEFAULT_S = max(DEADLINE_S.values())


def deadline_s(kind: str) -> float:
    """The response deadline of a request kind (KeyError for an unknown kind: every kind must be listed)."""
    return DEADLINE_S[kind]
