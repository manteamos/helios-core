"""
Reverse transposition: POA → GHI via Erbs decomposition + brentq root-finding.

Algorithm
---------
Given measured POA irradiance and known sun position / surface geometry, find
GHI such that  forward_transpose(GHI) = POA_measured.

Step 1 — Erbs diffuse-fraction model  (Erbs et al. 1982):
    k_t  = GHI / (I₀ · cos z)               # beam clearness index
    k_t is clamped to [0, 1].

    DHI/GHI =
        1 − 0.09·k_t                               if k_t ≤ 0.22
        0.9511 − 0.1604·k_t + 4.388·k_t²
          − 16.638·k_t³ + 12.336·k_t⁴            if 0.22 < k_t ≤ 0.80
        0.165                                       if k_t > 0.80

    DNI = (GHI − DHI) / max(cos z, cos 87°)   # cos 87° avoids ÷0 near horizon

Step 2 — Airmass (Kasten–Young 1989):
    AM = 1 / (cos z + 0.50572·(96.07995 − z_deg)^{−1.6364})

Step 3 — brentq root-finding:
    residual(ghi_guess) = transpose_discrete(ghi_guess, …) − POA_measured = 0
    Bounds: [0, ghi_upper] where ghi_upper defaults to 1 500 W m⁻².

Units
-----
poa_measured, ghi_upper  : W m⁻²
solar_zenith             : degrees
all other angles         : degrees
dni_extra                : W m⁻²  (extraterrestrial DNI, ~1 361 W m⁻²)
albedo                   : dimensionless (default 0.25)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from core.transposition.perez_discrete import transpose as _forward_transpose

FloatArray = NDArray[np.float64]

_COS_87: float = float(np.cos(np.radians(87.0)))
_ZENITH_LIMIT: float = 87.0  # degrees — beyond this, POA ≈ 0, skip root-finding


def kasten_young_airmass(zenith_deg: float) -> float:
    """
    Relative air mass via Kasten–Young (1989).

    Parameters
    ----------
    zenith_deg : solar zenith angle [degrees], must be < 90.

    Returns
    -------
    Relative air mass [dimensionless]
    """
    return float(
        1.0 / (np.cos(np.radians(zenith_deg)) + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364))
    )


def erbs_decompose(
    ghi: float,
    dni_extra: float,
    zenith_deg: float,
) -> tuple[float, float]:
    """
    Erbs et al. (1982) decomposition: GHI → (DHI, DNI).

    Parameters
    ----------
    ghi       : Global horizontal irradiance [W m⁻²]
    dni_extra : Extraterrestrial DNI (I₀) [W m⁻²]
    zenith_deg: Solar zenith angle [degrees]

    Returns
    -------
    (dhi, dni) in W m⁻²
    """
    cos_z = float(np.cos(np.radians(zenith_deg)))
    i0_h = dni_extra * cos_z  # horizontal extraterrestrial irradiance

    if ghi <= 0.0 or i0_h <= 0.0:
        return 0.0, 0.0

    kt = min(1.0, ghi / i0_h)

    if kt <= 0.22:
        dhi_frac = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        dhi_frac = 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
    else:
        dhi_frac = 0.165

    dhi = dhi_frac * ghi
    dni = (ghi - dhi) / max(cos_z, _COS_87)
    return float(dhi), float(max(0.0, dni))


def _poa_from_ghi(
    ghi_guess: float,
    dni_extra_i: float,
    zenith_i: float,
    azimuth_i: float,
    surface_tilt: float,
    surface_azimuth: float,
    albedo: float,
) -> float:
    """Scalar: compute POA_global from a GHI guess via Erbs + discrete Perez."""
    dhi, dni = erbs_decompose(ghi_guess, dni_extra_i, zenith_i)
    am = kasten_young_airmass(zenith_i)
    result = _forward_transpose(
        np.array([ghi_guess]),
        np.array([dni]),
        np.array([dhi]),
        np.array([zenith_i]),
        np.array([azimuth_i]),
        surface_tilt,
        surface_azimuth,
        np.array([dni_extra_i]),
        np.array([am]),
        albedo=albedo,
    )
    return float(result["poa_global"][0])


def _make_residual(
    i0_i: float,
    zen_i: float,
    azi_i: float,
    poa_i: float,
    surface_tilt: float,
    surface_azimuth: float,
    albedo: float,
) -> Callable[[float], float]:
    """Factory that captures loop scalars and returns a residual callable."""

    def residual(g: float) -> float:
        return _poa_from_ghi(g, i0_i, zen_i, azi_i, surface_tilt, surface_azimuth, albedo) - poa_i

    return residual


def reverse_transpose(
    poa_measured: FloatArray,
    solar_zenith: FloatArray,
    solar_azimuth: FloatArray,
    surface_tilt: float,
    surface_azimuth: float,
    dni_extra: FloatArray,
    albedo: float = 0.25,
    ghi_upper: float = 1500.0,
) -> FloatArray:
    """
    Invert the POA irradiance to recover GHI using brentq root-finding.

    Each time step is solved independently with scipy.optimize.brentq.
    Nighttime steps (zenith ≥ 87°) and zero-POA steps return 0 W m⁻².

    Parameters
    ----------
    poa_measured   : Measured POA irradiance [W m⁻²], shape (N,)
    solar_zenith   : Solar zenith [degrees], shape (N,)
    solar_azimuth  : Solar azimuth, clockwise from North [degrees], shape (N,)
    surface_tilt   : Fixed panel tilt [degrees]
    surface_azimuth: Fixed panel azimuth [degrees]
    dni_extra      : Extraterrestrial DNI [W m⁻²], shape (N,)
    albedo         : Ground reflectance (default 0.25)
    ghi_upper      : Upper bound for GHI search [W m⁻²] (default 1 500)

    Returns
    -------
    ghi : Estimated GHI [W m⁻²], shape (N,).  NaN where brentq fails.
    """
    poa = np.atleast_1d(np.asarray(poa_measured, dtype=np.float64))
    zen = np.atleast_1d(np.asarray(solar_zenith, dtype=np.float64))
    azi = np.atleast_1d(np.asarray(solar_azimuth, dtype=np.float64))
    i0 = np.atleast_1d(np.asarray(dni_extra, dtype=np.float64))
    ghi_out = np.zeros_like(poa)

    for idx in range(len(poa)):
        if poa[idx] <= 0.0 or zen[idx] >= _ZENITH_LIMIT:
            ghi_out[idx] = 0.0
            continue

        # Capture loop scalars to avoid B023 closure-over-variable bug
        i0_i = float(i0[idx])
        zen_i = float(zen[idx])
        azi_i = float(azi[idx])
        poa_i = float(poa[idx])

        residual = _make_residual(i0_i, zen_i, azi_i, poa_i, surface_tilt, surface_azimuth, albedo)

        poa_at_upper = _poa_from_ghi(
            ghi_upper, i0_i, zen_i, azi_i, surface_tilt, surface_azimuth, albedo
        )

        # Ensure bracketing; expand upper bound if needed
        upper = ghi_upper
        if poa_at_upper < poa_i:
            upper = 2.0 * ghi_upper

        try:
            ghi_out[idx] = brentq(residual, 0.0, upper, xtol=1e-4, rtol=1e-6)
        except ValueError:
            ghi_out[idx] = np.nan

    return ghi_out
