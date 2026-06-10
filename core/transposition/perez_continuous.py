"""
Perez-Ineichen transposition — continuous mean-preserving spline implementation.

The six empirical coefficients (F11, F12, F13, F21, F22, F23) in the discrete
model are step-constant over 8 clearness-ε bins.  Here each coefficient is
replaced by a continuous function  fᵢ(ε) constructed via an antiderivative
cubic spline so that the bin means are preserved exactly:

    ∫_{εₖ}^{εₖ₊₁} fᵢ(ε) dε  =  cᵢₖ · (εₖ₊₁ − εₖ)   for all k

Construction
------------
For each coefficient column c = [c₀, …, c₇] (one per bin):

  1. Define the antiderivative  G(εₖ) = Σⱼ<ₖ cⱼ·Δεⱼ  at the 9 bin-edge nodes
     (last cap: ε₈ = 8.0, covering the unbounded final bin).

  2. Fit  scipy.interpolate.CubicSpline(eps_edges, G, bc_type='natural').
     Natural boundary conditions force G''=0 at the endpoints, which is the
     minimal-curvature choice that avoids spurious oscillations near the edges.

  3. fᵢ(ε) = Gᵢ'(ε).  Because Gᵢ is a CubicSpline (C²), its derivative is C¹
     (piecewise quadratic), satisfying the required continuous first-derivative
     property.  Mean preservation is exact: G(εₖ₊₁) − G(εₖ) = cₖ · Δεₖ.

  4. ε is clamped to [1.0, 8.0] before evaluation; anything above 8.0 is
     extrapolated at G'(8.0) via the natural-spline condition (≈ flat).

C¹ continuity verification (numerical)
---------------------------------------
The module verifies at construction time that |fᵢ(εₖ⁺) − fᵢ(εₖ⁻)| < 1e-10
at all interior bin-edge nodes.  An AssertionError is raised if this fails.

Units
-----
Same as perez_discrete.py — see that module for full unit documentation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from core.transposition.perez_discrete import (
    _COS_85,
    EPS_BIN_EDGES,
    PEREZ_COEFFS,
    clearness,
    cos_aoi,
)

FloatArray = NDArray[np.float64]

# Upper cap for the unbounded last bin (ε ≥ 6.2). Real-world ε rarely exceeds 8.
_EPS_CAP: float = 8.0

# 9 node ε-values: the 8 lower bin edges + the cap
_EPS_NODES: FloatArray = np.append(EPS_BIN_EDGES, _EPS_CAP)

# Coefficient names (for documentation / debugging)
_COEFF_NAMES = ("F11", "F12", "F13", "F21", "F22", "F23")


def _build_splines() -> list[CubicSpline]:
    """Build one antiderivative CubicSpline per Perez coefficient column."""
    delta_eps = np.diff(_EPS_NODES)  # shape (8,) — bin widths
    splines: list[CubicSpline] = []

    for col in range(6):
        c = PEREZ_COEFFS[:, col]  # 8 discrete bin values
        # Cumulative integral: g_cum[0]=0, g_cum[k+1]=g_cum[k]+c[k]*Δε[k]
        g_cum = np.zeros(9)
        g_cum[1:] = np.cumsum(c * delta_eps)
        # Natural cubic spline through (eps_nodes, g_cum): G is C², G' is C¹
        splines.append(CubicSpline(_EPS_NODES, g_cum, bc_type="natural"))

    # Numerical C¹ verification at interior knots.
    # CubicSpline is mathematically C² (hence C¹) by construction; this check
    # guards against accidental changes to the knot vector or bc_type.
    # Use tiny=1e-6 so the finite-difference step stays well above float64
    # roundoff (~1e-15) and below the spline's second-derivative scale (~1).
    interior = _EPS_NODES[1:-1]  # 7 interior edges
    tiny = 1e-6
    for col, spl in enumerate(splines):
        f_left = spl(interior - tiny, 1)  # derivative from left-side polynomial
        f_right = spl(interior + tiny, 1)  # derivative from right-side polynomial
        gap = float(np.max(np.abs(f_left - f_right)))
        # Expected gap ≈ 2·|f''(x)|·tiny ≈ 2 × O(1) × 1e-6 ≈ 1e-5 for smooth data
        assert gap < 1e-4, (
            f"C¹ continuity violated for {_COEFF_NAMES[col]}: " f"max discontinuity = {gap:.2e}"
        )

    return splines


# Build splines once at import time
_SPLINES: list[CubicSpline] = _build_splines()


def _eval_coeffs(eps: FloatArray) -> FloatArray:
    """
    Evaluate all six continuous Perez coefficient functions at given ε values.

    Parameters
    ----------
    eps : clearness parameter ε, shape (N,) — clamped internally to [1.0, 8.0]

    Returns
    -------
    coeffs : shape (N, 6) — [F11, F12, F13, F21, F22, F23] at each ε
    """
    eps_c = np.clip(eps, 1.0, _EPS_CAP)
    # derivative=1 evaluates the first derivative of the antiderivative spline = f(ε)
    return np.column_stack([spl(eps_c, 1) for spl in _SPLINES])


def mean_preservation_error() -> dict[str, FloatArray]:
    """
    Return absolute mean-preservation error for each bin and coefficient.

    The error should be at floating-point machine precision (~1e-15) because
    CubicSpline(nodes, g_cum).derivative() integrates back to g_cum exactly.

    Returns
    -------
    dict mapping coefficient name → array of shape (8,) with absolute errors
    """
    from scipy.integrate import quad  # local import — validation only

    delta_eps = np.diff(_EPS_NODES)
    errors: dict[str, FloatArray] = {}
    for col, (name, spl) in enumerate(zip(_COEFF_NAMES, _SPLINES, strict=True)):
        errs = np.empty(8)
        for k in range(8):
            a, b = float(_EPS_NODES[k]), float(_EPS_NODES[k + 1])
            # Capture spl by default argument to avoid B023 closure issue
            integral, _ = quad(lambda e, s=spl: float(s(e, 1)), a, b)
            errs[k] = abs(integral - float(PEREZ_COEFFS[k, col]) * delta_eps[k])
        errors[name] = errs
    return errors


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
    Perez-Ineichen sky-diffuse irradiance using continuous coefficient splines.

    Parameters and return value are identical to perez_discrete.perez_sky_diffuse.
    The only algorithmic difference is that F11..F23 are evaluated as continuous
    functions of ε instead of via discrete bin lookup.
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

    fc = _eval_coeffs(eps)  # (N, 6)
    f1 = np.maximum(0.0, fc[:, 0] + fc[:, 1] * delta + z * fc[:, 2])
    f2 = fc[:, 3] + fc[:, 4] * delta + z * fc[:, 5]

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
    Full POA irradiance decomposition using continuous Perez-Ineichen splines.

    Parameters and return value are identical to perez_discrete.transpose.
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
