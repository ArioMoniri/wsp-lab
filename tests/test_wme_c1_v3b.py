"""RULING-wme-c1-bank-halt run12 halt (v3b): after a cap raise (P4 of addendum 10, or R8) every later ProductionGuard
construction must succeed. The registered cap generation passed to the constructor is re-recorded only if it is not
persisted yet; an unpersisted lower cap is still refused. CPU only; fixtures."""

import pytest

from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import genengine as GE
from wsp.wm.c1 import study as SD

REF = "a" * 64
ATTEST = RG.LaunchAttestation(True, True, REF, "c" * 64, ["fixture"])


@pytest.fixture(autouse=True)
def _c1_clean_global_state():
    saved = GE.global_state()
    GE.restore_state(GE.BASELINE_STATE)
    yield
    GE.restore_state(saved)


def guard(state, cap=SD.CAP, priced=None):
    return RG.ProductionGuard(
        state, "confirmatory", priced or {"confirmatory": 53.666, "ste": 1.845}, ATTEST, "i1", 0.75,
        technical=C.technical, fallback_value="base", cap=cap,
    )  # fmt: skip


def p4(led):
    basis = {"charged_gpu_h": 29.53, "repriced_remainder_gpu_h": 84.138, "overhead_gpu_h": 3.046}
    basis["projection_gpu_h"] = sum(basis.values())
    return RG.CapGeneration(
        cap_gpu_h=RG.p4_cap(basis), date="2026-11-23", decided_by="chair", announced_to_user=False,
        source="RULING-wme-c1-bank-halt.md addendum 10 (P4)", rule="P4", basis=basis,
    )  # fmt: skip


def test_a_guard_after_a_p4_raise_is_constructed_and_keeps_the_raised_cap(tmp_path):
    g = guard(tmp_path / "s")
    g.ledger.record_cap(p4(g.ledger), contract=g.contract, generation=g.generation)
    raised = g.ledger.cap_gpu_h()
    assert raised > SD.CAP.cap_gpu_h
    n = len(g.ledger.cap_generations())
    g2 = guard(
        tmp_path / "s"
    )  # the run12 halt: this raised "a cap generation never lowers the cap"
    assert g2.ledger.cap_gpu_h() == raised and len(g2.ledger.cap_generations()) == n
    g2.ledger.check_before("confirmatory", 0.75)


def test_a_guard_after_an_r8_raise_is_constructed(tmp_path):
    g = guard(tmp_path / "s", priced={"confirmatory": 120.0, "ste": 20.0})
    g.ledger.add("confirmatory", 109.0 * 3600)
    g.ledger.check_before("confirmatory", 2.0)  # R8 raises the cap
    caps = g.ledger.cap_generations()
    assert caps[-1]["rule"] == "R8"
    g2 = guard(tmp_path / "s", priced={"confirmatory": 120.0, "ste": 20.0})
    assert (
        len(g2.ledger.cap_generations()) == len(caps)
        and g2.ledger.cap_gpu_h() == caps[-1]["cap_gpu_h"]
    )


def test_the_registered_cap_is_still_recorded_once_and_an_unpersisted_lower_cap_is_refused(
    tmp_path,
):
    g = guard(tmp_path / "s")
    assert [c["cap_gpu_h"] for c in g.ledger.cap_generations()] == [SD.CAP.cap_gpu_h]
    g = guard(tmp_path / "s")
    assert len(g.ledger.cap_generations()) == 1  # idempotent
    g.ledger.record_cap(p4(g.ledger))
    other = RG.CapGeneration(cap_gpu_h=100.0, date="2026-11-19", decided_by="user", source="another decision",
                             announced_to_user=True)  # fmt: skip
    with pytest.raises(ValueError, match="never lowers"):
        guard(tmp_path / "s", cap=other)
