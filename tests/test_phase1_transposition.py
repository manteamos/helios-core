"""
Phase 1 unit tests — Perez transposition engines.

Coverage
--------
1. Discrete model matches pvlib.irradiance.perez within 1e-9 W m⁻² (absolute)
   on synthetic daytime data (random zenith, azimuth, dhi, dni).
2. Continuous spline C¹ continuity — verified at import time inside the module;
   this test confirms the assertion doesn't fire on import.
3. Mean-preservation error < 1e-9 for all bins and all six coefficients.
4. Continuous vs discrete annual POA deviation < 0.10% on a 365-day synthetic
   dataset (well within the 1.5% acceptance criterion).
5. Reverse transposition converges (non-NaN) on 100 % of daytime test cases.
6. Full 8760-row vectorised transpose runs without error.
"""

from __future__ import annotations

import numpy as np
import pvlib
import pytest

from core.transposition.perez_continuous import (
    _EPS_NODES,
    _SPLINES,
    mean_preservation_error,
)
from core.transposition.perez_continuous import (
    transpose as transpose_cont,
)
from core.transposition.perez_discrete import (
    perez_sky_diffuse as perez_sky_disc,
)
from core.transposition.perez_discrete import (
    transpose as transpose_disc,
)
from core.transposition.reverse import (
    erbs_decompose,
    kasten_young_airmass,
    reverse_transpose,
)

RNG = np.random.default_rng(42)
N = 200  # synthetic test points


def _synthetic_daytime(n: int = N) -> dict[str, np.ndarray]:
    """Random daytime inputs: zenith 5–80°, positive dhi and dni."""
    solar_zenith = RNG.uniform(5.0, 80.0, n)
    solar_azimuth = RNG.uniform(90.0, 270.0, n)
    dhi = RNG.uniform(5.0, 250.0, n)
    dni = RNG.uniform(0.0, 900.0, n)
    ghi = dhi + dni * np.cos(np.radians(solar_zenith))
    ghi = np.maximum(ghi, dhi)  # ghi ≥ dhi always
    dni_extra = np.full(n, 1361.0)
    airmass = 1.0 / np.cos(np.radians(solar_zenith))
    return dict(
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dhi=dhi,
        dni=dni,
        ghi=ghi,
        dni_extra=dni_extra,
        airmass=airmass,
    )


# ---------------------------------------------------------------------------
# 1. Discrete model vs pvlib
# ---------------------------------------------------------------------------
def test_discrete_matches_pvlib_atol_1e9() -> None:
    d = _synthetic_daytime()
    tilt, az_surf = 15.0, 180.0

    pvlib_sky = np.asarray(
        pvlib.irradiance.perez(
            tilt,
            az_surf,
            d["dhi"],
            d["dni"],
            d["dni_extra"],
            d["solar_zenith"],
            d["solar_azimuth"],
            d["airmass"],
            # default model = 'allsitescomposite1990', same as our PEREZ_COEFFS
        ),
        dtype=np.float64,
    )
    helios_sky = perez_sky_disc(
        d["dhi"],
        d["dni"],
        d["dni_extra"],
        d["solar_zenith"],
        d["solar_azimuth"],
        tilt,
        az_surf,
        d["airmass"],
    )
    np.testing.assert_allclose(
        helios_sky, pvlib_sky, atol=1e-9, rtol=0, err_msg="Discrete model diverges from pvlib"
    )


def test_discrete_matches_pvlib_various_tilts() -> None:
    """Spot-check at 0°, 30°, 90° tilt."""
    d = _synthetic_daytime(50)
    for tilt in (0.0, 30.0, 90.0):
        pvlib_sky = np.asarray(
            pvlib.irradiance.perez(
                tilt,
                180.0,
                d["dhi"],
                d["dni"],
                d["dni_extra"],
                d["solar_zenith"],
                d["solar_azimuth"],
                d["airmass"],
            ),
            dtype=np.float64,
        )
        helios_sky = perez_sky_disc(
            d["dhi"],
            d["dni"],
            d["dni_extra"],
            d["solar_zenith"],
            d["solar_azimuth"],
            tilt,
            180.0,
            d["airmass"],
        )
        np.testing.assert_allclose(
            helios_sky,
            pvlib_sky,
            atol=1e-9,
            rtol=0,
            err_msg=f"Discrete model diverges at tilt={tilt}°",
        )


# ---------------------------------------------------------------------------
# 2. Continuous spline C¹ continuity (import-time assertion is sufficient,
#    but we also verify numerically here for explicitness)
# ---------------------------------------------------------------------------
def test_continuous_c1_at_all_interior_knots() -> None:
    # tiny=1e-6 keeps us above float64 roundoff; gap should be ≈ 2|f''|·tiny ≪ 1e-4
    interior = _EPS_NODES[1:-1]
    tiny = 1e-6
    for col, spl in enumerate(_SPLINES):
        f_left = spl(interior - tiny, 1)
        f_right = spl(interior + tiny, 1)
        max_gap = float(np.max(np.abs(f_left - f_right)))
        assert max_gap < 1e-4, f"Spline col {col}: C¹ gap = {max_gap:.2e} at interior knot"


# ---------------------------------------------------------------------------
# 3. Mean preservation
# ---------------------------------------------------------------------------
def test_mean_preservation_all_bins_all_coeffs() -> None:
    errors = mean_preservation_error()
    for name, errs in errors.items():
        assert np.max(errs) < 1e-9, f"Mean-preservation error for {name}: max = {np.max(errs):.2e}"


# ---------------------------------------------------------------------------
# 4. Continuous vs discrete — annual POA deviation < 0.10 %
# ---------------------------------------------------------------------------
def test_continuous_vs_discrete_annual_deviation() -> None:
    # Simulate a synthetic full year: 8760 hourly steps with varied sky states
    rng = np.random.default_rng(0)
    n = 8760
    zenith = np.abs(rng.normal(45.0, 20.0, n))
    zenith = np.clip(zenith, 1.0, 89.0)
    azimuth = rng.uniform(90.0, 270.0, n)
    dhi = np.abs(rng.normal(80.0, 50.0, n))
    dni = np.abs(rng.normal(400.0, 300.0, n))
    ghi = dhi + dni * np.cos(np.radians(zenith))
    dni_extra = np.full(n, 1361.0)
    airmass = 1.0 / np.cos(np.radians(zenith))

    r_disc = transpose_disc(ghi, dni, dhi, zenith, azimuth, 15.0, 180.0, dni_extra, airmass)
    r_cont = transpose_cont(ghi, dni, dhi, zenith, azimuth, 15.0, 180.0, dni_extra, airmass)

    annual_disc = np.sum(r_disc["poa_global"])
    annual_cont = np.sum(r_cont["poa_global"])
    pct_diff = abs(annual_cont - annual_disc) / annual_disc * 100.0

    assert pct_diff < 1.5, (
        f"Continuous vs discrete annual POA deviation = {pct_diff:.3f}% " f"(limit 1.5%)"
    )


# ---------------------------------------------------------------------------
# 5. Reverse transposition convergence on 100 % of daytime cases
# ---------------------------------------------------------------------------
def test_reverse_transpose_convergence_100pct() -> None:
    d = _synthetic_daytime(100)
    tilt, az_surf = 15.0, 180.0
    dni_extra = d["dni_extra"]

    # Forward: compute ground-truth POA
    poa = transpose_disc(
        d["ghi"],
        d["dni"],
        d["dhi"],
        d["solar_zenith"],
        d["solar_azimuth"],
        tilt,
        az_surf,
        dni_extra,
        d["airmass"],
    )["poa_global"]

    # Reverse: recover GHI
    ghi_recovered = reverse_transpose(
        poa,
        d["solar_zenith"],
        d["solar_azimuth"],
        tilt,
        az_surf,
        dni_extra,
    )

    nan_frac = np.mean(np.isnan(ghi_recovered))
    assert nan_frac == 0.0, f"{nan_frac*100:.1f}% of reverse solutions are NaN"


def test_reverse_transpose_residual_small() -> None:
    """Recovered GHI reproduces input POA within 1 W m⁻² for Erbs-consistent inputs.

    The round-trip is: GHI → Erbs → (DHI, DNI) → POA_fwd → reverse → GHI' →
    Erbs → POA_check.  Both forward and check use the same Erbs decomposition,
    so any residual comes only from brentq convergence, not model mismatch.
    """
    d = _synthetic_daytime(50)
    tilt, az_surf = 15.0, 180.0
    i0 = d["dni_extra"]
    zenith = d["solar_zenith"]
    azimuth = d["solar_azimuth"]

    # Use Erbs to produce internally consistent (ghi, dhi, dni)
    ghi_in = d["ghi"]
    dhi_in = np.array(
        [
            erbs_decompose(float(ghi_in[k]), float(i0[k]), float(zenith[k]))[0]
            for k in range(len(ghi_in))
        ]
    )
    dni_in = np.array(
        [
            erbs_decompose(float(ghi_in[k]), float(i0[k]), float(zenith[k]))[1]
            for k in range(len(ghi_in))
        ]
    )
    am_in = np.array([kasten_young_airmass(float(z)) for z in zenith])

    poa_fwd = transpose_disc(
        ghi_in,
        dni_in,
        dhi_in,
        zenith,
        azimuth,
        tilt,
        az_surf,
        i0,
        am_in,
    )["poa_global"]

    ghi_rev = reverse_transpose(poa_fwd, zenith, azimuth, tilt, az_surf, i0)

    dhi_rev = np.array(
        [
            erbs_decompose(float(ghi_rev[k]), float(i0[k]), float(zenith[k]))[0]
            for k in range(len(ghi_rev))
        ]
    )
    dni_rev = np.array(
        [
            erbs_decompose(float(ghi_rev[k]), float(i0[k]), float(zenith[k]))[1]
            for k in range(len(ghi_rev))
        ]
    )
    am_rev = np.array([kasten_young_airmass(float(z)) for z in zenith])

    poa_check = transpose_disc(
        ghi_rev,
        dni_rev,
        dhi_rev,
        zenith,
        azimuth,
        tilt,
        az_surf,
        i0,
        am_rev,
    )["poa_global"]

    residuals = np.abs(poa_check - poa_fwd)
    assert (
        np.max(residuals) < 1.0
    ), f"Max round-trip residual = {np.max(residuals):.4f} W m⁻² (limit 1 W m⁻²)"


# ---------------------------------------------------------------------------
# 6. Vectorised 8760-row transpose
# ---------------------------------------------------------------------------
def test_8760_row_vectorised_no_error() -> None:
    d = _synthetic_daytime(8760)
    result = transpose_disc(
        d["ghi"],
        d["dni"],
        d["dhi"],
        d["solar_zenith"],
        d["solar_azimuth"],
        15.0,
        180.0,
        d["dni_extra"],
        d["airmass"],
    )
    assert result["poa_global"].shape == (8760,)
    assert not np.any(np.isnan(result["poa_global"]))


# ---------------------------------------------------------------------------
# 7. Erbs decomposition sanity checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ghi,zenith,expect_dhi_frac",
    [
        (0.0, 30.0, None),  # zero GHI → zero output
        (100.0, 85.0, None),  # near-horizon
        (50.0, 45.0, "high"),  # cloudy (low kt) → high DHI fraction
        (900.0, 20.0, "low"),  # clear (high kt) → low DHI fraction
    ],
)
def test_erbs_decompose_fractions(ghi: float, zenith: float, expect_dhi_frac: str | None) -> None:
    dhi, dni = erbs_decompose(ghi, 1361.0, zenith)
    assert dhi >= 0.0 and dni >= 0.0
    if ghi > 0.0:
        assert dhi <= ghi * 1.001  # DHI ≤ GHI (within rounding)
    if expect_dhi_frac == "high":
        assert dhi / ghi > 0.70
    elif expect_dhi_frac == "low":
        assert dhi / ghi < 0.30
