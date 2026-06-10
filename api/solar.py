"""
Solar position — Spencer (1971) Fourier approximation.

Accuracy: ±0.5° zenith; sufficient for yield simulation.

Governing equations (Spencer 1971; Duffie & Beckman §1.4–1.6)
--------------------------------------------------------------
    B = 2π(doy − 1) / 365                         [rad]

Equation of time [min]:
    EqT = 229.18 × (0.000075
           + 0.001868 cos B − 0.032077 sin B
           − 0.014615 cos 2B − 0.040890 sin 2B)

Solar declination [rad]:
    δ = 0.006918 − 0.399912 cos B + 0.070257 sin B
        − 0.006758 cos 2B + 0.000907 sin 2B
        − 0.002697 cos 3B + 0.001480 sin 3B

True solar time [h]:
    TST = UTC_hour + EqT/60 + longitude_deg/15

Hour angle [°]:
    ω = (TST − 12) × 15

Solar zenith [°]:
    cos z = sin φ sin δ + cos φ cos δ cos ω

Solar azimuth clockwise from North [°]:
    cos(az_from_south) = (sin δ − cos z sin φ) / (sin z cos φ)
    A_s = 180 − arccos(·) if ω < 0   (forenoon)
    A_s = 180 + arccos(·) if ω ≥ 0   (afternoon)

Extraterrestrial irradiance [W m⁻²]:
    I₀ = 1361.5 × (1 + 0.033 cos(2π doy / 365))

Air mass — Kasten & Young (1989):
    AM = 1 / (cos z + 0.50572 × (96.07995 − z)^−1.6364)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def solar_position(
    doy: FloatArray,
    hour_utc: FloatArray,
    latitude_deg: float,
    longitude_deg: float,
) -> dict[str, FloatArray]:
    """
    Compute solar zenith and azimuth (Spencer 1971).

    Parameters
    ----------
    doy          : Day of year [1..366], shape (N,)
    hour_utc     : UTC hour including fractional minutes, shape (N,)
    latitude_deg : Site latitude, WGS84 [°]
    longitude_deg: Site longitude, WGS84 [°]

    Returns
    -------
    dict with keys:
        zenith   : Solar zenith [°], 0 = overhead, 90 = horizon
        azimuth  : Solar azimuth [°], clockwise from North
        cos_zen  : cos(zenith); negative values mean night
    """
    d = np.atleast_1d(np.asarray(doy, dtype=np.float64))
    h = np.atleast_1d(np.asarray(hour_utc, dtype=np.float64))

    b = 2.0 * np.pi * (d - 1.0) / 365.0  # Spencer day-angle parameter [rad]

    eqt = 229.18 * (
        0.000075
        + 0.001868 * np.cos(b)
        - 0.032077 * np.sin(b)
        - 0.014615 * np.cos(2.0 * b)
        - 0.040890 * np.sin(2.0 * b)
    )

    delta = (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2.0 * b)
        + 0.000907 * np.sin(2.0 * b)
        - 0.002697 * np.cos(3.0 * b)
        + 0.001480 * np.sin(3.0 * b)
    )

    tst = h + eqt / 60.0 + longitude_deg / 15.0
    omega = np.radians((tst - 12.0) * 15.0)

    lat = np.radians(latitude_deg)
    cos_z: FloatArray = np.clip(
        np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(omega),
        -1.0,
        1.0,
    )
    zenith_rad = np.arccos(cos_z)
    zenith_deg: FloatArray = np.degrees(zenith_rad)

    # Azimuth (clockwise from North):
    #   cos(az_from_south) = (sin δ − cos z sin φ) / (sin z cos φ)
    sin_z = np.sin(zenith_rad)
    cos_lat = np.cos(lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az_from_south: FloatArray = np.where(
            sin_z > 1e-6,
            np.clip(
                (np.sin(lat) * cos_z - np.sin(delta)) / (sin_z * cos_lat),
                -1.0,
                1.0,
            ),
            0.0,
        )
    az_abs = np.degrees(np.arccos(cos_az_from_south))  # [0°, 180°]
    azimuth_deg: FloatArray = np.where(omega < 0.0, 180.0 - az_abs, 180.0 + az_abs)

    return {"zenith": zenith_deg, "azimuth": azimuth_deg, "cos_zen": cos_z}


def extraterrestrial_irradiance(doy: FloatArray) -> FloatArray:
    """
    Extraterrestrial normal solar irradiance [W m⁻²] — Spencer (1971).

    I₀ = 1361.5 × (1 + 0.033 cos(2π doy / 365))
    """
    b = 2.0 * np.pi * np.asarray(doy, dtype=np.float64) / 365.0
    i0: FloatArray = 1361.5 * (1.0 + 0.033 * np.cos(b))
    return i0


def airmass_kasten(zenith_deg: FloatArray) -> FloatArray:
    """
    Relative air mass — Kasten & Young (1989).

    AM = 1 / (cos z + 0.50572 × (96.07995 − z)^−1.6364)
    Clamped to [1, 40]; returns 40 for zenith ≥ 90°.
    """
    z = np.atleast_1d(np.asarray(zenith_deg, dtype=np.float64))
    cos_z = np.cos(np.radians(z))
    # Kasten-Young denominator; guard against base=0 (z=96.08°) in the power term
    base = np.maximum(1e-6, 96.07995 - z)
    denom = cos_z + 0.50572 * base ** (-1.6364)
    with np.errstate(divide="ignore", invalid="ignore"):
        am: FloatArray = np.where(z < 90.0, 1.0 / np.maximum(denom, 1e-6), 40.0)
    return np.clip(am, 1.0, 40.0)
