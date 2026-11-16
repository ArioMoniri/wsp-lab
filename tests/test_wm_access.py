"""wm.access: tier R0 is executable; predict/plan raise on any non-theta0 artefact (SYNTHESIS-r8 D2)."""

import pytest

from wsp.wm import access as X

BASE = "b" * 64
OTHER = "e" * 64


def test_artefact_carries_theta_hash_and_tier():
    a = X.Artefact("field", BASE, X.Tier.R0, {"g": 1})
    assert a.theta_hash == BASE and a.tier is X.Tier.R0
    with pytest.raises(ValueError):
        X.Artefact("field", "", X.Tier.R0, None)
    with pytest.raises(TypeError):
        X.Artefact("field", BASE, "R0", None)


def test_r0_artefacts_must_be_at_base_and_candidate_independent():
    with pytest.raises(ValueError):
        X.Artefact("jvp", BASE, X.Tier.R0, None, candidate_dependent=True)
    with pytest.raises(ValueError):
        X.Artefact("tap", OTHER, X.Tier.R1, None)  # R1 is a base pass by definition


def test_guard_accepts_base_r0_and_refuses_everything_else():
    g = X.R0Guard(BASE)
    ok = X.Artefact("field", BASE, X.Tier.R0, None)
    g.check(ok, {"nested": [ok, (ok,)]}, 3, "text")
    for bad in (
        X.Artefact("tap", OTHER, X.Tier.R2, None),
        X.Artefact("jvp", BASE, X.Tier.R1, None, candidate_dependent=True),
        X.Artefact("field", OTHER, X.Tier.R0, None),
    ):
        with pytest.raises(X.AccessError):
            g.check({"deep": [ok, bad]})


def test_r0_decorator_guards_predict_and_plan():
    class Model:
        guard = X.R0Guard(BASE)

        @X.r0_only
        def predict(self, base_obs, actions):
            return len(actions)

        @X.r0_only
        def plan(self, base_obs, budget):
            return budget

    m = Model()
    base_obs = X.Artefact("tap", BASE, X.Tier.R0, None)
    assert m.predict(base_obs, [1, 2]) == 2
    edited_obs = X.Artefact("tap", OTHER, X.Tier.R2, None)
    with pytest.raises(X.AccessError):
        m.predict(edited_obs, [1])
    with pytest.raises(X.AccessError):
        m.plan(base_obs, budget=X.Artefact("tap", OTHER, X.Tier.R2, None))


def test_r0_decorator_requires_a_guard():
    class NoGuard:
        @X.r0_only
        def predict(self, x):
            return x

    with pytest.raises(X.AccessError):
        NoGuard().predict(1)


def test_tier_of_observation_is_derived_from_theta():
    assert X.observation(BASE, BASE, "tap", None).tier is X.Tier.R0
    assert X.observation(OTHER, BASE, "tap", None).tier is X.Tier.R2


def test_artefact_payload_is_not_mutable_through_the_wrapper():
    a = X.Artefact("field", BASE, X.Tier.R0, {"x": 1})
    with pytest.raises(AttributeError):
        a.theta_hash = OTHER
