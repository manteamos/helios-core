"""Custom exceptions for the Helios component registry."""

from __future__ import annotations


class ComponentValidationError(Exception):
    """
    Raised when a VerifiedModuleProfile fails a physical consistency check.

    The message always includes the field values that triggered the failure
    so callers can produce a clean diagnostic without unwrapping Pydantic internals.
    """


class SignatureVerificationError(Exception):
    """
    Raised by ingest_profile() when a profile has no valid Ed25519 signature
    from any key currently in the accredited-lab allowlist.

    Unsigned profiles and profiles signed by unknown keys both raise this.
    """
