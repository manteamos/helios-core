"""
Phase 4 validation: Verified Component Registry -- sign / verify / tamper demo.

Terminal output
---------------
1. Profile construction -- STC consistency check pass / fail boundary.
2. Round-trip sign -> verify -> ingest demonstration.
3. Tamper-detection demonstration (manufacturer, numeric field).
4. Canonical JSON preview (first 120 chars).
5. Allowlist state before / after add_lab_key / remove_lab_key.

This script does not produce plot files; all output is printed to stdout.
"""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

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
# Representative 400 W monocrystalline module (typical utility-grade spec)
# ---------------------------------------------------------------------------
# STC: P_stc = 400.00 W, I_mpp = 6.0200 A, V_mpp = 66.4452 V
# I*V = 6.0200 * 66.4452 = 400.0001 W  -> error ~2.4e-7 < 0.001

_SAMPLE: dict[str, object] = {
    "manufacturer": "MKA-Solar",
    "model_name": "MKA-400M-GH",
    "itl_identifier": "ITL-MKA-400M-GH-20240601",
    "width_m": 1.046,
    "length_m": 1.724,
    "depth_m": 0.035,
    "mass_kg": 21.5,
    "cell_type": "monocrystalline-silicon",
    "n_cells": 72,
    "n_bypass_diodes": 3,
    "p_stc_w": 400.00,
    "i_mpp_a": 6.02,
    "v_mpp_v": 66.4452,
    "i_sc_a": 6.46,
    "v_oc_v": 79.45,
    "temp_coeff_p_pct_k": -0.350,
    "temp_coeff_i_a_k": 0.0025,
    "temp_coeff_v_v_k": -0.128,
    "noct_c": 44.0,
    "bifaciality": 0.0,
    # Martinuzzi (2007) standard 5-term polynomial IAM: a0..a4
    "iam_coeffs": [1.0, -2.438e-4, -3.103e-4, 5.654e-6, -2.539e-8],
}


def _hr(title: str) -> None:
    print(f"\n{'='*62}\n{title}\n{'='*62}")


def section_stc_boundary() -> None:
    _hr("1. STC power consistency validator -- boundary cases")

    p = VerifiedModuleProfile(**_SAMPLE)  # type: ignore[arg-type]
    product = p.i_mpp_a * p.v_mpp_v
    err = abs(p.p_stc_w - product) / p.p_stc_w
    print(f"  Base profile:  |P - I*V| / P = {err:.2e}  -> PASS")

    ok_fields: dict[str, object] = {**_SAMPLE, "p_stc_w": 500.0, "i_mpp_a": 10.0, "v_mpp_v": 49.95}
    p_bound = VerifiedModuleProfile(**ok_fields)  # type: ignore[arg-type]
    err_bound = abs(p_bound.p_stc_w - p_bound.i_mpp_a * p_bound.v_mpp_v) / p_bound.p_stc_w
    print(f"  Boundary (=0.001): error = {err_bound:.6f}  -> PASS")

    bad_fields: dict[str, object] = {**_SAMPLE, "p_stc_w": 500.0, "i_mpp_a": 10.0, "v_mpp_v": 49.94}
    try:
        VerifiedModuleProfile(**bad_fields)  # type: ignore[arg-type]
        print("  Over tolerance: UNEXPECTED PASS")
    except ComponentValidationError as exc:
        print("  Over tolerance: raises ComponentValidationError [OK]")
        print(f"    {exc}")


def section_sign_verify(profile: VerifiedModuleProfile) -> VerifiedModuleProfile:
    _hr("2. Round-trip sign -> verify -> ingest")

    key = signing_key_for_tests()
    signed = sign_profile(profile, key)
    assert signed.digital_signature is not None
    print(f"  Signature (first 32 hex chars): {signed.digital_signature[:32]}...")
    print(f"  verify_profile(signed)   : {verify_profile(signed)}")
    print(f"  verify_profile(unsigned) : {verify_profile(profile)}")

    ingested = ingest_profile(signed)
    print(f"  ingest_profile(signed)   : returned profile [OK] ({ingested.model_name})")
    return signed


def section_tamper_detection(signed: VerifiedModuleProfile) -> None:
    _hr("3. Tamper-detection")

    cases: list[tuple[str, VerifiedModuleProfile]] = [
        ("manufacturer changed", signed.model_copy(update={"manufacturer": "Evil Corp"})),
        (
            "p_stc_w changed (+1 W)",
            signed.model_copy(update={"p_stc_w": 401.0, "i_mpp_a": 6.035, "v_mpp_v": 66.4452}),
        ),
        ("itl_identifier changed", signed.model_copy(update={"itl_identifier": "FAKE-001"})),
    ]
    for label, tampered in cases:
        result = verify_profile(tampered)
        status = "PASS (rejected)" if not result else "FAIL (accepted -- should not happen)"
        print(f"  {label:35s}: {status}")

    print()
    try:
        ingest_profile(signed.model_copy(update={"manufacturer": "Injected"}))
    except SignatureVerificationError as exc:
        print("  ingest_profile(tampered) raised SignatureVerificationError [OK]")
        print(f"    {exc}")


def section_canonical_json(signed: VerifiedModuleProfile) -> None:
    _hr("4. Canonical JSON preview")
    cb = canonical_bytes(signed)
    preview = cb.decode("utf-8")
    print(f"  Length: {len(cb)} bytes")
    print(f"  First 120 chars: {preview[:120]}...")
    print(f"  Deterministic (two calls equal): {canonical_bytes(signed) == cb}")


def section_allowlist() -> None:
    _hr("5. Allowlist management")

    print("  Current keys:")
    for name, pub_hex in list_lab_keys().items():
        print(f"    {name}: {pub_hex[:16]}...")

    new_key = Ed25519PrivateKey.generate()
    pub_bytes = new_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    add_lab_key("MKA-Lab-Ghana", pub_bytes)
    print("\n  After add_lab_key('MKA-Lab-Ghana'):")
    for name, pub_hex in list_lab_keys().items():
        print(f"    {name}: {pub_hex[:16]}...")

    remove_lab_key("MKA-Lab-Ghana")
    print("\n  After remove_lab_key('MKA-Lab-Ghana'):")
    for name in list_lab_keys():
        print(f"    {name}")


def main() -> None:
    print("Phase 4 validation -- Verified Component Registry\n")
    section_stc_boundary()

    profile = VerifiedModuleProfile(**_SAMPLE)  # type: ignore[arg-type]
    signed = section_sign_verify(profile)
    section_tamper_detection(signed)
    section_canonical_json(signed)
    section_allowlist()

    print("\nAll Phase 4 validation checks complete.")


if __name__ == "__main__":
    main()
