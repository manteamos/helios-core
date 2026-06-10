"""
Phase 4 unit tests — Verified Component Registry.

Coverage
--------
1.  Round-trip sign → verify returns True.
2.  Unsigned profile: verify_profile returns False; ingest_profile raises.
3.  Tampered manufacturer field: verify_profile returns False after tampering.
4.  Tampered numeric field: verify_profile returns False after tampering.
5.  STC tolerance — exactly at 0.001: profile constructs without error.
6.  STC tolerance — above 0.001: raises ComponentValidationError.
7.  STC tolerance — exact match (error = 0): profile constructs.
8.  ComponentValidationError message contains the field values.
9.  Canonical JSON: sorted keys, no whitespace, excludes digital_signature.
10. Canonical bytes are identical on repeated calls (deterministic).
11. Model is frozen (immutable after construction).
12. sign_profile does not mutate the original profile.
13. Different private key (not in allowlist): verify returns False.
14. add_lab_key / remove_lab_key round-trip.
15. list_lab_keys returns dict with correct hex representation.
16. Malformed signature hex: verify returns False gracefully.
17. iam_coeffs must be non-empty: raises ValueError.
18. bifaciality out of range: raises ValidationError.
19. ingest_profile raises SignatureVerificationError for unknown-key signature.
20. Signed profile cannot be verified after one field is changed.
"""

from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import ValidationError

from registry.crypto import (
    add_lab_key,
    canonical_bytes,
    ingest_profile,
    list_lab_keys,
    remove_lab_key,
    sign_profile,
    signing_key_for_tests,
    verify_profile,
)
from registry.exceptions import ComponentValidationError, SignatureVerificationError
from registry.models import VerifiedModuleProfile

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_BASE_FIELDS: dict[str, object] = {
    "manufacturer": "Helios Solar",
    "model_name": "HS-500M",
    "itl_identifier": "ITL-HS-500M-TEST-001",
    "width_m": 1.046,
    "length_m": 1.690,
    "depth_m": 0.040,
    "mass_kg": 19.0,
    "cell_type": "monocrystalline-silicon",
    "n_cells": 72,
    "n_bypass_diodes": 3,
    # STC: 500 W, I*V = 10.0 * 50.0 = 500.0 exactly
    "p_stc_w": 500.0,
    "i_mpp_a": 10.0,
    "v_mpp_v": 50.0,
    "i_sc_a": 10.50,
    "v_oc_v": 61.0,
    "temp_coeff_p_pct_k": -0.38,
    "temp_coeff_i_a_k": 0.0040,
    "temp_coeff_v_v_k": -0.130,
    "noct_c": 45.0,
    "bifaciality": 0.0,
    "iam_coeffs": [1.0, -0.0026, -0.0001, 0.0, 0.0],
}


def _make(**overrides: object) -> VerifiedModuleProfile:
    return VerifiedModuleProfile(**{**_BASE_FIELDS, **overrides})  # type: ignore[arg-type]


@pytest.fixture()
def profile() -> VerifiedModuleProfile:
    return _make()


@pytest.fixture()
def signed_profile(profile: VerifiedModuleProfile) -> VerifiedModuleProfile:
    return sign_profile(profile, signing_key_for_tests())


# ---------------------------------------------------------------------------
# 1. Round-trip sign → verify
# ---------------------------------------------------------------------------


def test_round_trip_sign_verify(profile: VerifiedModuleProfile) -> None:
    signed = sign_profile(profile, signing_key_for_tests())
    assert signed.digital_signature is not None
    assert verify_profile(signed) is True


# ---------------------------------------------------------------------------
# 2. Unsigned profile: verify False; ingest raises
# ---------------------------------------------------------------------------


def test_unsigned_verify_false(profile: VerifiedModuleProfile) -> None:
    assert verify_profile(profile) is False


def test_unsigned_ingest_raises(profile: VerifiedModuleProfile) -> None:
    with pytest.raises(SignatureVerificationError, match="no valid signature"):
        ingest_profile(profile)


# ---------------------------------------------------------------------------
# 3. Tampered manufacturer → verify False
# ---------------------------------------------------------------------------


def test_tampered_manufacturer_fails_verify(
    signed_profile: VerifiedModuleProfile,
) -> None:
    tampered = signed_profile.model_copy(update={"manufacturer": "Fake Corp"})
    assert verify_profile(tampered) is False


# ---------------------------------------------------------------------------
# 4. Tampered numeric field → verify False
# ---------------------------------------------------------------------------


def test_tampered_p_stc_fails_verify(signed_profile: VerifiedModuleProfile) -> None:
    # Adjust i_mpp_a to keep STC consistency while changing the signed payload
    tampered = signed_profile.model_copy(
        update={"p_stc_w": 501.0, "i_mpp_a": 10.02, "v_mpp_v": 50.0}
    )
    assert verify_profile(tampered) is False


# ---------------------------------------------------------------------------
# 5. STC tolerance exactly at 0.001 — should pass
# ---------------------------------------------------------------------------


def test_stc_tolerance_at_boundary_passes() -> None:
    # |P - I*V| / P = |500.0 - 10.0*49.95| / 500.0 = 0.5/500 = 0.001 exactly
    profile = _make(p_stc_w=500.0, i_mpp_a=10.0, v_mpp_v=49.95)
    assert profile.p_stc_w == 500.0


# ---------------------------------------------------------------------------
# 6. STC tolerance above 0.001 — raises ComponentValidationError
# ---------------------------------------------------------------------------


def test_stc_tolerance_above_boundary_raises() -> None:
    # |500.0 - 10.0*49.94| / 500.0 = 0.6/500 = 0.0012 > 0.001
    with pytest.raises(ComponentValidationError, match="STC power consistency"):
        _make(p_stc_w=500.0, i_mpp_a=10.0, v_mpp_v=49.94)


def test_stc_tolerance_large_error_raises() -> None:
    # I*V = 10.0*40.0 = 400.0 vs P_stc=500.0 → error = 20%
    with pytest.raises(ComponentValidationError):
        _make(p_stc_w=500.0, i_mpp_a=10.0, v_mpp_v=40.0)


# ---------------------------------------------------------------------------
# 7. STC exact match (error = 0) — should construct cleanly
# ---------------------------------------------------------------------------


def test_stc_exact_match_passes() -> None:
    profile = _make(p_stc_w=500.0, i_mpp_a=10.0, v_mpp_v=50.0)
    product = profile.i_mpp_a * profile.v_mpp_v
    assert abs(profile.p_stc_w - product) < 1e-9


# ---------------------------------------------------------------------------
# 8. ComponentValidationError message contains field values
# ---------------------------------------------------------------------------


def test_stc_error_message_contains_values() -> None:
    with pytest.raises(ComponentValidationError) as exc_info:
        _make(p_stc_w=500.0, i_mpp_a=10.0, v_mpp_v=49.0)
    msg = str(exc_info.value)
    assert "500.0" in msg
    assert "10.0" in msg
    assert "49.0" in msg


# ---------------------------------------------------------------------------
# 9. Canonical JSON: sorted keys, no whitespace, excludes digital_signature
# ---------------------------------------------------------------------------


def test_canonical_json_sorted_no_whitespace(signed_profile: VerifiedModuleProfile) -> None:
    raw = canonical_bytes(signed_profile)
    text = raw.decode("utf-8")
    # Structural separators must be compact — no space after colon or comma.
    # (String VALUES like "Helios Solar" legitimately contain spaces.)
    assert ": " not in text, "found space after colon in structural separator"
    assert ", " not in text, "found space after comma in structural separator"
    assert "\n" not in text
    # Valid JSON
    data = json.loads(text)
    # digital_signature must be excluded
    assert "digital_signature" not in data
    # Keys are in sorted order
    keys = list(data.keys())
    assert keys == sorted(keys)


def test_canonical_json_excludes_signature_even_when_present(
    signed_profile: VerifiedModuleProfile,
) -> None:
    raw = canonical_bytes(signed_profile)
    data = json.loads(raw)
    assert "digital_signature" not in data
    assert signed_profile.digital_signature is not None  # just confirming it's set


# ---------------------------------------------------------------------------
# 10. Canonical bytes are deterministic
# ---------------------------------------------------------------------------


def test_canonical_bytes_deterministic(signed_profile: VerifiedModuleProfile) -> None:
    b1 = canonical_bytes(signed_profile)
    b2 = canonical_bytes(signed_profile)
    assert b1 == b2


# ---------------------------------------------------------------------------
# 11. Model is frozen (immutable)
# ---------------------------------------------------------------------------


def test_model_is_frozen(profile: VerifiedModuleProfile) -> None:
    with pytest.raises(ValidationError):
        profile.manufacturer = "Modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 12. sign_profile does not mutate original
# ---------------------------------------------------------------------------


def test_sign_profile_does_not_mutate_original(profile: VerifiedModuleProfile) -> None:
    assert profile.digital_signature is None
    signed = sign_profile(profile, signing_key_for_tests())
    assert profile.digital_signature is None  # original unchanged
    assert signed.digital_signature is not None


# ---------------------------------------------------------------------------
# 13. Different (unknown) private key: verify returns False
# ---------------------------------------------------------------------------


def test_unknown_key_verify_false(profile: VerifiedModuleProfile) -> None:
    unknown_key = Ed25519PrivateKey.generate()
    signed = sign_profile(profile, unknown_key)
    assert verify_profile(signed) is False


def test_unknown_key_ingest_raises(profile: VerifiedModuleProfile) -> None:
    unknown_key = Ed25519PrivateKey.generate()
    signed = sign_profile(profile, unknown_key)
    with pytest.raises(SignatureVerificationError):
        ingest_profile(signed)


# ---------------------------------------------------------------------------
# 14. add_lab_key / remove_lab_key round-trip
# ---------------------------------------------------------------------------


def test_add_remove_lab_key(profile: VerifiedModuleProfile) -> None:
    new_key = Ed25519PrivateKey.generate()
    pub_bytes = new_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    try:
        add_lab_key("Test-Lab-Ephemeral", pub_bytes)
        signed = sign_profile(profile, new_key)
        assert verify_profile(signed) is True
    finally:
        remove_lab_key("Test-Lab-Ephemeral")

    # After removal: verification should fail
    assert verify_profile(signed) is False


# ---------------------------------------------------------------------------
# 15. list_lab_keys returns correct hex
# ---------------------------------------------------------------------------


def test_list_lab_keys_format() -> None:
    keys = list_lab_keys()
    assert "ITL-Accra-Test" in keys
    pub_hex = keys["ITL-Accra-Test"]
    # Ed25519 raw public key = 32 bytes = 64 hex chars
    assert len(pub_hex) == 64
    assert all(c in "0123456789abcdef" for c in pub_hex)


# ---------------------------------------------------------------------------
# 16. Malformed signature hex: verify returns False gracefully
# ---------------------------------------------------------------------------


def test_malformed_signature_hex(profile: VerifiedModuleProfile) -> None:
    bad_sig = profile.model_copy(update={"digital_signature": "not-valid-hex!!"})
    assert verify_profile(bad_sig) is False


def test_wrong_length_signature(profile: VerifiedModuleProfile) -> None:
    # Valid hex but wrong length (Ed25519 signature is 64 bytes = 128 hex chars)
    bad_sig = profile.model_copy(update={"digital_signature": "deadbeef"})
    assert verify_profile(bad_sig) is False


# ---------------------------------------------------------------------------
# 17. iam_coeffs must be non-empty
# ---------------------------------------------------------------------------


def test_empty_iam_coeffs_raises() -> None:
    with pytest.raises(ValidationError, match="iam_coeffs"):
        _make(iam_coeffs=[])


# ---------------------------------------------------------------------------
# 18. bifaciality out of range raises ValidationError
# ---------------------------------------------------------------------------


def test_bifaciality_above_one_raises() -> None:
    with pytest.raises(ValidationError):
        _make(bifaciality=1.1)


def test_bifaciality_negative_raises() -> None:
    with pytest.raises(ValidationError):
        _make(bifaciality=-0.1)


# ---------------------------------------------------------------------------
# 19. ingest_profile raises for unknown-key signature (alias of test 13)
# ---------------------------------------------------------------------------


def test_ingest_raises_signed_by_unknown_key(profile: VerifiedModuleProfile) -> None:
    unknown = Ed25519PrivateKey.generate()
    signed = sign_profile(profile, unknown)
    with pytest.raises(SignatureVerificationError, match="no valid signature"):
        ingest_profile(signed)


# ---------------------------------------------------------------------------
# 20. Signed profile fails after any field change
# ---------------------------------------------------------------------------


def test_any_field_change_breaks_signature(signed_profile: VerifiedModuleProfile) -> None:
    # Test several different field types
    cases: list[dict[str, object]] = [
        {"manufacturer": "Evil Corp"},
        {"itl_identifier": "FAKE-001"},
        {"p_stc_w": 499.0, "i_mpp_a": 9.98, "v_mpp_v": 50.0},  # keep STC valid
        {"noct_c": 50.0},
        {"n_cells": 60},
        {"iam_coeffs": [1.0, -0.003, 0.0, 0.0, 0.0]},
    ]
    for update in cases:
        tampered = signed_profile.model_copy(update=update)
        assert verify_profile(tampered) is False, f"Expected verify to fail after update {update!r}"
