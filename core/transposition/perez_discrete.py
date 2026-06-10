"""
Perez-Ineichen (1990) transposition — discrete bin implementation.

Governing equations
-------------------
Clearness parameter ε  (Perez 1990 §2):
    ε = ( (DHI + DNI) / DHI  +  κ z³ ) / ( 1 + κ z³ )
    κ = 1.041 rad⁻³,  z = solar zenith [rad]
    When DHI ≤ 0, ε is undefined; sky-diffuse = 0 regardless of bin.

Sky brightness Δ:
    Δ = DHI · AM / I₀

Circumsolar/horizon brightness coefficients (Perez 1990 eq. 3–4):
    F₁ = max( 0,  f₁₁ + f₁₂·Δ + z·f₁₃ )
    F₂ =          f₂₁ + f₂₂·Δ + z·f₂₃

Perez sky-diffuse irradiance on tilted surface (Perez 1990 eq. 5):
    I_sky = DHI · [ (1−F₁)·(1+cosβ)/2  +  F₁·a/b  +  F₂·sinβ ]
    a = max(0, cos θ),   b = max(cos 85°, cos z)

Angle of incidence on tilted surface:
    cos θ = sinz · sinβ · cos(φ_sun − γ) + cosz · cosβ

Ground-reflected component:
    I_gnd = GHI · ρ · (1 − cosβ) / 2

Units
-----
GHI, DHI, DNI, I₀, all output irradiances : W m⁻²
solar_zenith, solar_azimuth               : degrees (clockwise from North for azimuth)
surface_tilt                              : degrees from horizontal
surface_azimuth                           : degrees, clockwise from North
airmass                                   : dimensionless
albedo                                    : dimensionless (default 0.25)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# fmt: off
# Perez (1990) "allsitescomposite1990" coefficients.
# 8 ε-bins × 6 coefficients [F11, F12, F13, F21, F22, F23]
# Verified against the reference implementation (see validation/phase1_perez.py).
PEREZ_COEFFS: FloatArray = np.array([
    [-0.0080,   0.5880,  -0.0620,  -0.0600,   0.0720,  -0.0220],  # bin 1
    [ 0.1300,   0.6830,  -0.1510,  -0.0190,   0.0660,  -0.0290],  # bin 2
    [ 0.3300,   0.4870,  -0.2210,   0.0550,  -0.0640,  -0.0260],  # bin 3
    [ 0.5680,   0.1870,  -0.2950,   0.1090,  -0.1520,  -0.0140],  # bin 4
    [ 0.8730,  -0.3920,  -0.3620,   0.2260,  -0.4620,   0.0010],  # bin 5
    [ 1.1320,  -1.2370,  -0.4120,   0.2880,  -0.8230,   0.0560],  # bin 6
    [ 1.0600,  -1.6000,  -0.3590,   0.2640,  -1.1270,   0.1310],  # bin 7
    [ 0.6780,  -0.3270,  -0.2500,   0.1560,  -1.3770,   0.2510],  # bin 8
], dtype=np.float64)
# fmt: on

# Lower edges of the 8 clearness bins; bin 8 extends to ∞
EPS_BIN_EDGES: FloatArray = np.array(
    [1.000, 1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200], dtype=np.float64
)

_KAPPA: float = 1.041
_COS_85: float = float(np.cos(np.radians(85.0)))  # ≈ 0.08716


def cos_aoi(
    zenith_rad: FloatArray,
    azimuth_rad: FloatArray,
    tilt_rad: FloatArray,
    surf_az_rad: FloatArray,
) -> FloatArray:
    """Cosine of the angle of incidence on a tilted surface (vectorised)."""
    result: FloatArray = np.sin(zenith_rad) * np.sin(tilt_rad) * np.cos(
        azimuth_rad - surf_az_rad
    ) + np.cos(zenith_rad) * np.cos(tilt_rad)
    return result


def clearness(
    dhi: FloatArray,
    dni: FloatArray,
    zenith_rad: FloatArray,
) -> FloatArray:
    """Perez clearness parameter ε (≥ 1). Returns 1.0 when DHI ≤ 0."""
    kz3 = _KAPPA * zenith_rad**3
    # np.errstate suppresses the "divide by zero" warning for the dhi=0 branch;
    # np.where discards those values and returns 1.0 for dhi ≤ 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        result: FloatArray = np.asarray(
            np.where(
                dhi > 0.0,
                ((dhi + dni) / dhi + kz3) / (1.0 + kz3),
                1.0,
            ),
            dtype=np.float64,
        )
    return result


def bin_index(eps: FloatArray) -> NDArray[np.intp]:
    """Map ε values to 0-based bin indices in [0, 7]."""
    idx = np.searchsorted(EPS_BIN_EDGES, eps, side="right") - 1
    return np.clip(idx, 0, 7).astype(np.intp)


def perez_sky_diffuse(
    dhi: FloatArray,
    dni: FloatArray,
    dni_extra: FloatArray,
    solar_zenith: FloatArray,
    solar_azimuth: FloatArray,
    surface_tilt: float | FloatArray,
    surface_azimuth: float | FloatArray,
    airmass: FloatArray,
) -> FloatArray:
    """
    Perez-Ineichen sky-diffuse irradiance on a tilted surface (discrete bins).

    Parameters
    ----------
    dhi, dni       : Diffuse / direct-normal horizontal irradiance [W m⁻²]
    dni_extra      : Extraterrestrial DNI (I₀) [W m⁻²]
    solar_zenith   : Solar zenith angle [degrees]
    solar_azimuth  : Solar azimuth, clockwise from North [degrees]
    surface_tilt   : Tilt from horizontal [degrees]
    surface_azimuth: Surface azimuth, clockwise from North [degrees]
    airmass        : Relative (apparent) air mass [dimensionless]

    Returns
    -------
    poa_sky : Sky-diffuse irradiance on the tilted surface [W m⁻²]
    """
    z = np.radians(np.atleast_1d(np.asarray(solar_zenith, dtype=np.float64)))
    az = np.radians(np.atleast_1d(np.asarray(solar_azimuth, dtype=np.float64)))
    beta = np.radians(np.asarray(surface_tilt, dtype=np.float64))
    gamma = np.radians(np.asarray(surface_azimuth, dtype=np.float64))
    dhi = np.atleast_1d(np.asarray(dhi, dtype=np.float64))
    dni = np.atleast_1d(np.asarray(dni, dtype=np.float64))
    i0 = np.atleast_1d(np.asarray(dni_extra, dtype=np.float64))
    am = np.atleast_1d(np.asarray(airmass, dtype=np.float64))

    delta = np.where(i0 > 0.0, dhi * am / i0, 0.0)
    eps = clearness(dhi, dni, z)
    idx = bin_index(eps)

    f = PEREZ_COEFFS[idx]  # (N, 6)
    f1 = np.maximum(0.0, f[:, 0] + f[:, 1] * delta + z * f[:, 2])
    f2 = f[:, 3] + f[:, 4] * delta + z * f[:, 5]

    a = np.maximum(0.0, cos_aoi(z, az, beta, gamma))
    b = np.maximum(_COS_85, np.cos(z))

    sky = dhi * ((1.0 - f1) * (1.0 + np.cos(beta)) / 2.0 + f1 * a / b + f2 * np.sin(beta))
    return np.maximum(0.0, sky)


def transpose(
    ghi: FloatArray,
    dni: FloatArray,
    dhi: FloatArray,
    solar_zenith: FloatArray,
    solar_azimuth: FloatArray,
    surface_tilt: float | FloatArray,
    surface_azimuth: float | FloatArray,
    dni_extra: FloatArray,
    airmass: FloatArray,
    albedo: float = 0.25,
) -> dict[str, FloatArray]:
    """
    Full POA irradiance decomposition (Perez-Ineichen discrete).

    Parameters
    ----------
    ghi, dni, dhi  : Global / direct-normal / diffuse horizontal [W m⁻²]
    solar_zenith   : Solar zenith [degrees]
    solar_azimuth  : Solar azimuth, clockwise from North [degrees]
    surface_tilt   : Tilt from horizontal [degrees]
    surface_azimuth: Surface azimuth, clockwise from North [degrees]
    dni_extra      : Extraterrestrial DNI [W m⁻²]
    airmass        : Relative air mass [dimensionless]
    albedo         : Ground reflectance (default 0.25)

    Returns
    -------
    dict with keys: poa_global, poa_direct, poa_sky, poa_ground  [W m⁻²]
    """
    ghi = np.atleast_1d(np.asarray(ghi, dtype=np.float64))
    beta = np.radians(np.asarray(surface_tilt, dtype=np.float64))
    gamma = np.radians(np.asarray(surface_azimuth, dtype=np.float64))
    z = np.radians(np.atleast_1d(np.asarray(solar_zenith, dtype=np.float64)))
    az = np.radians(np.atleast_1d(np.asarray(solar_azimuth, dtype=np.float64)))

    poa_direct = np.atleast_1d(np.asarray(dni, dtype=np.float64)) * np.maximum(
        0.0, cos_aoi(z, az, beta, gamma)
    )
    poa_sky = perez_sky_diffuse(
        dhi,
        dni,
        dni_extra,
        solar_zenith,
        solar_azimuth,
        surface_tilt,
        surface_azimuth,
        airmass,
    )
    poa_ground = ghi * albedo * (1.0 - np.cos(beta)) / 2.0
    poa_global = poa_direct + poa_sky + poa_ground

    return {
        "poa_global": poa_global,
        "poa_direct": poa_direct,
        "poa_sky": poa_sky,
        "poa_ground": poa_ground,
    }
