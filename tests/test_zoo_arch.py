import pytest


def test_trust_remote_code_override_rejected_for_unlisted_models():
    from wsp.zoo import arch

    with pytest.raises(ValueError, match="allow-list"):
        arch.remote_code_kwargs("some-org/unlisted-model", {"trust_remote_code": True})
    assert arch.remote_code_kwargs("some-org/unlisted-model", {}) == {}
    assert arch.remote_code_kwargs(min(arch.REMOTE_CODE_ALLOWLIST), {}) == {
        "trust_remote_code": True
    }
