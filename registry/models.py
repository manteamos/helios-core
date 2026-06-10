"""
Pydantic v2 schema for a cryptographically verified PV module profile.

Fields
------
Identity         : manufacturer, model_name, itl_identifier
Physical geometry: width_m, length_m, depth_m, mass_kg
Cell topology    : cell_type, n_cells, n_bypass_diodes
STC coefficients : p_stc_w, i_mpp_a, v_mpp_v, i_sc_a, v_oc_v,
                   temp_coeff_p_pct_k, temp_coeff_i_a_k, temp_coeff_v_v_k
Low-light params : noct_c, bifaciality
IAM polynomial   : iam_coeffs  (a_0 + a_1·AOI + a_2·AOI² + …)
Crypto           : digital_signature (hex Ed25519, None before signing)

Validators
----------
- STC power consistency: |P_stc − I_mpp·V_mpp| / P_stc ≤ 0.001
  Raises ComponentValidationError with values in the message on failure.
- bifaciality in [0, 1]; iam_coeffs non-empty.
- All power/current/voltage/dimension values strictly positive (via Pydantic Field).
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from registry.exceptions import ComponentValidationError

_STC_TOLERANCE: float = 0.001  # |P_stc - I_mpp*V_mpp| / P_stc ≤ this


class VerifiedModuleProfile(BaseModel):
    """
    Immutable, cryptographically verifiable PV module specification.

    Create unsigned, then pass to registry.crypto.sign_profile() to obtain
    a signed copy.  Pass the signed copy to registry.crypto.ingest_profile()
    to verify it against the accredited-lab allowlist before use in simulations.
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    manufacturer: str
    model_name: str
    itl_identifier: str

    # --- Physical geometry ---
    width_m: float = Field(gt=0.0, description="Module width [m]")
    length_m: float = Field(gt=0.0, description="Module length [m]")
    depth_m: float = Field(gt=0.0, description="Module depth / frame thickness [m]")
    mass_kg: float = Field(gt=0.0, description="Module mass [kg]")

    # --- Cell topology ---
    cell_type: str
    n_cells: int = Field(gt=0, description="Total cell count per module")
    n_bypass_diodes: int = Field(ge=0, description="Bypass diode count")

    # --- STC electrical coefficients ---
    p_stc_w: float = Field(gt=0.0, description="Rated power at STC [W]")
    i_mpp_a: float = Field(gt=0.0, description="MPP current [A]")
    v_mpp_v: float = Field(gt=0.0, description="MPP voltage [V]")
    i_sc_a: float = Field(gt=0.0, description="Short-circuit current [A]")
    v_oc_v: float = Field(gt=0.0, description="Open-circuit voltage [V]")
    temp_coeff_p_pct_k: float = Field(description="Power temp. coefficient [%/K]; typically < 0")
    temp_coeff_i_a_k: float = Field(description="Current temp. coefficient [A/K]")
    temp_coeff_v_v_k: float = Field(description="Voltage temp. coefficient [V/K]; typically < 0")

    # --- Low-light / NOCT parameters ---
    noct_c: float = Field(description="Nominal operating cell temperature [°C]")
    bifaciality: float = Field(ge=0.0, le=1.0, description="Bifaciality factor [0–1]")

    # --- IAM polynomial coefficients: a_0 + a_1*AOI + a_2*AOI^2 + ... ---
    iam_coeffs: list[float]

    # --- Cryptographic field (None until signed) ---
    digital_signature: str | None = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("iam_coeffs")
    @classmethod
    def check_iam_coeffs_nonempty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("iam_coeffs must contain at least one coefficient")
        return v

    @model_validator(mode="after")
    def check_stc_power_consistency(self) -> Self:
        """
        Enforce |P_stc − I_mpp·V_mpp| / P_stc ≤ 0.001.

        Raises ComponentValidationError (not ValueError) so it propagates
        directly without being wrapped in pydantic.ValidationError.
        """
        power_product = self.i_mpp_a * self.v_mpp_v
        relative_error = abs(self.p_stc_w - power_product) / self.p_stc_w
        if relative_error > _STC_TOLERANCE:
            raise ComponentValidationError(
                f"STC power consistency check failed: "
                f"|P_stc - I_mpp*V_mpp| / P_stc = {relative_error:.6f} > {_STC_TOLERANCE} "
                f"(P_stc={self.p_stc_w} W, I_mpp={self.i_mpp_a} A, "
                f"V_mpp={self.v_mpp_v} V, I_mpp*V_mpp={power_product:.4f} W)"
            )
        return self
