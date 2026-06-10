"""
Ed25519 sign/verify for VerifiedModuleProfile.

Canonical serialisation
-----------------------
1. Dump the profile to a plain Python dict (excluding digital_signature).
2. Serialize with json.dumps(sort_keys=True, separators=(',', ':')) — no
   whitespace, lexicographically sorted keys — to produce a byte-for-byte
   identical representation on every call.
3. Encode to UTF-8.

Ed25519 operations
------------------
- Private key signs the canonical bytes; signature is stored as a lowercase
  hex string in digital_signature.
- Public keys in the allowlist are stored as raw 32-byte values (Encoding.Raw).
- Verification iterates the allowlist; the first successful verify wins.

Accredited-lab allowlist
------------------------
_ALLOWLIST maps lab_name -> raw public key bytes (32 bytes).
A built-in test entry is seeded deterministically so unit tests can sign
profiles without an external key-management step.  Production deployments
call add_lab_key() to register real accredited-lab certificates.

Ingestion
---------
ingest_profile() verifies the signature and raises SignatureVerificationError
on failure (unsigned, tampered, or unknown-lab).  All failures are logged at
WARNING level with the profile identity so security audits have a record.
"""

from __future__ import annotations

import hashlib
import json
import logging

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from registry.exceptions import SignatureVerificationError
from registry.models import VerifiedModuleProfile

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowlist — lab name → raw Ed25519 public key (32 bytes)
# ---------------------------------------------------------------------------

_ALLOWLIST: dict[str, bytes] = {}

# Deterministic seed for the built-in test lab key.
# Generated once from a fixed string so tests are bit-for-bit reproducible.
_TEST_KEY_SEED: bytes = hashlib.sha256(b"helios-core-itl-accra-test-key-v1").digest()  # 32 bytes


def signing_key_for_tests() -> Ed25519PrivateKey:
    """
    Return the Ed25519 private key for the ITL-Accra-Test allowlist entry.

    For unit tests only.  The matching public key is pre-loaded in the default
    allowlist under the name "ITL-Accra-Test".
    """
    return Ed25519PrivateKey.from_private_bytes(_TEST_KEY_SEED)


def _load_test_key_into_allowlist() -> None:
    priv = signing_key_for_tests()
    pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    _ALLOWLIST["ITL-Accra-Test"] = pub_bytes


_load_test_key_into_allowlist()


# ---------------------------------------------------------------------------
# Allowlist management
# ---------------------------------------------------------------------------


def add_lab_key(lab_name: str, public_key_bytes: bytes) -> None:
    """
    Register a new accredited-lab Ed25519 public key.

    Parameters
    ----------
    lab_name : str
        Human-readable lab identifier (used in log messages).
    public_key_bytes : bytes
        Raw Ed25519 public key (32 bytes).
    """
    if len(public_key_bytes) != 32:
        raise ValueError(f"Ed25519 raw public key must be 32 bytes, got {len(public_key_bytes)}")
    _ALLOWLIST[lab_name] = public_key_bytes


def remove_lab_key(lab_name: str) -> None:
    """Remove a lab key from the allowlist (no-op if not present)."""
    _ALLOWLIST.pop(lab_name, None)


def list_lab_keys() -> dict[str, str]:
    """Return allowlist as {lab_name: public_key_hex}."""
    return {name: pub.hex() for name, pub in _ALLOWLIST.items()}


# ---------------------------------------------------------------------------
# Canonical serialisation
# ---------------------------------------------------------------------------


def canonical_bytes(profile: VerifiedModuleProfile) -> bytes:
    """
    Produce canonical UTF-8 JSON bytes for signing/verification.

    The digital_signature field is excluded; all other fields are included.
    Keys are sorted lexicographically; no whitespace separators.

    Returns
    -------
    bytes
        Deterministic byte sequence that is identical on every call for the
        same profile field values.
    """
    data = profile.model_dump(exclude={"digital_signature"})
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Sign / verify
# ---------------------------------------------------------------------------


def sign_profile(
    profile: VerifiedModuleProfile,
    private_key: Ed25519PrivateKey,
) -> VerifiedModuleProfile:
    """
    Sign a profile and return a new instance with digital_signature set.

    The original profile is not mutated (frozen model).

    Parameters
    ----------
    profile : VerifiedModuleProfile
        May already have a signature; it is overwritten.
    private_key : Ed25519PrivateKey

    Returns
    -------
    VerifiedModuleProfile
        Identical to input with digital_signature = lowercase hex Ed25519 sig.
    """
    payload = canonical_bytes(profile)
    sig_bytes = private_key.sign(payload)
    return profile.model_copy(update={"digital_signature": sig_bytes.hex()})


def verify_profile(profile: VerifiedModuleProfile) -> bool:
    """
    Verify the profile's digital_signature against all allowlist keys.

    Returns True on the first successful verification; False if the profile is
    unsigned, has a malformed signature, or no allowlist key verifies it.
    Does NOT raise — callers that need a hard failure should use ingest_profile().
    """
    if profile.digital_signature is None:
        return False

    try:
        sig_bytes = bytes.fromhex(profile.digital_signature)
    except ValueError:
        return False

    payload = canonical_bytes(profile)

    for lab_name, pub_bytes in _ALLOWLIST.items():
        try:
            pub_key: Ed25519PublicKey = Ed25519PublicKey.from_public_bytes(pub_bytes)
            pub_key.verify(sig_bytes, payload)
            _LOG.debug("signature verified by lab key %r", lab_name)
            return True
        except Exception:  # noqa: BLE001
            continue

    return False


def ingest_profile(profile: VerifiedModuleProfile) -> VerifiedModuleProfile:
    """
    Verify and accept a profile for use in simulations.

    Rejects unsigned payloads and profiles whose signature does not match any
    accredited-lab key.  All rejections are logged at WARNING level with the
    profile manufacturer and model_name so security audits have a record.

    Parameters
    ----------
    profile : VerifiedModuleProfile

    Returns
    -------
    VerifiedModuleProfile
        The verified profile (unchanged).

    Raises
    ------
    SignatureVerificationError
        If verification fails for any reason.
    """
    if not verify_profile(profile):
        _LOG.warning(
            "signature verification failed — rejected profile %r / %r  " "(itl_id=%r, sig=%r)",
            profile.manufacturer,
            profile.model_name,
            profile.itl_identifier,
            profile.digital_signature,
        )
        raise SignatureVerificationError(
            f"Profile {profile.manufacturer!r} / {profile.model_name!r} "
            f"(ITL: {profile.itl_identifier!r}) has no valid signature "
            f"from any accredited lab in the allowlist"
        )
    return profile
