"""
Synthetic TMY generator for Ghana-latitude sites.

Produces physically plausible hourly/sub-hourly meteorological data without
requiring external TMY files.  Used for end-to-end integration tests and CI.

Irradiance model
----------------
Clear-sky GHI — Haurwitz (1945):
    GHI_cs = 1098 × cos z × exp(−0.057 / cos z)

Seasonal attenuation (Ghana dry/wet cycle; proxy from PVGIS Accra data):
    kt = 0.72 − 0.10 × sin²(π × doy / 183 − 0.5)
    Range: ~0.62 (wet season peak, Jun–Aug) to 0.72 (dry season, Nov–Feb)

DHI/DNI decomposition — Orgill & Hollands (1977):
    kt_atm = GHI / (I₀_normal × cos z)
    kd = 1 − 0.249 × kt_atm            if kt_atm < 0.35
       = 1.557 − 1.840 × kt_atm        if 0.35 ≤ kt_atm < 0.75
       = 0.177                          if kt_atm ≥ 0.75
    DHI = GHI × kd
    DNI = (GHI − DHI) / cos z

Ambient temperature [°C]:
    T_amb = 28 + 3 sin(2π(doy−80)/365) + 5 sin(2π hour/24 − 0.5)

Wind speed [m s⁻¹]:
    v = max(0.5,  2.5 + 1.5 sin(2π(doy−60)/365) + 0.5 sin(2π hour/24))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def generate_ghana_tmy(
    doy: FloatArray,
    hour_frac: FloatArray,
    cos_zenith: FloatArray,
    i0_normal: FloatArray,
) -> dict[str, FloatArray]:
    """
    Synthetic Ghana TMY: irradiance, temperature, wind.

    Parameters
    ----------
    doy        : Day of year [1..366], shape (N,)
    hour_frac  : Fractional hour within the day [0, 24), shape (N,)
    cos_zenith : cos(solar zenith) from api.solar.solar_position, shape (N,)
    i0_normal  : Extraterrestrial normal irradiance [W m⁻²], shape (N,)

    Returns
    -------
    dict with keys (all shape (N,)):
        ghi          : Global horizontal irradiance [W m⁻²]
        dni          : Direct normal irradiance [W m⁻²]
        dhi          : Diffuse horizontal irradiance [W m⁻²]
        t_amb_c      : Ambient temperature [°C]
        wind_speed_ms: Wind speed [m s⁻¹]
    """
    d = np.asarray(doy, dtype=np.float64)
    cos_z = np.asarray(cos_zenith, dtype=np.float64)
    i0 = np.asarray(i0_normal, dtype=np.float64)

    # --- Clear-sky GHI (Haurwitz 1945) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi_cs: FloatArray = np.where(
            cos_z > 0.01,
            1098.0 * cos_z * np.exp(-0.057 / np.maximum(cos_z, 0.01)),
            0.0,
        )
    ghi_cs = np.maximum(0.0, ghi_cs)

    # --- Seasonal clearness index ---
    kt_seasonal: FloatArray = 0.72 - 0.10 * np.sin(np.pi * d / 183.0 - 0.5) ** 2

    ghi: FloatArray = np.maximum(0.0, ghi_cs * kt_seasonal)

    # --- Orgill-Hollands decomposition ---
    i0_horiz = i0 * np.maximum(cos_z, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt_atm: FloatArray = np.where(i0_horiz > 1.0, ghi / i0_horiz, 0.0)
    kt_atm = np.clip(kt_atm, 0.0, 1.0)

    kd: FloatArray = np.where(
        kt_atm < 0.35,
        1.0 - 0.249 * kt_atm,
        np.where(kt_atm < 0.75, 1.557 - 1.840 * kt_atm, 0.177),
    )
    kd = np.clip(kd, 0.0, 1.0)

    dhi: FloatArray = np.maximum(0.0, ghi * kd)
    with np.errstate(divide="ignore", invalid="ignore"):
        dni: FloatArray = np.where(cos_z > 0.01, np.maximum(0.0, (ghi - dhi) / cos_z), 0.0)

    # --- Ambient temperature [°C] ---
    t_amb: FloatArray = (
        28.0
        + 3.0 * np.sin(2.0 * np.pi * (d - 80.0) / 365.0)
        + 5.0 * np.sin(2.0 * np.pi * hour_frac / 24.0 - 0.5)
    )

    # --- Wind speed [m s⁻¹] ---
    v_wind: FloatArray = np.maximum(
        0.5,
        2.5
        + 1.5 * np.sin(2.0 * np.pi * (d - 60.0) / 365.0)
        + 0.5 * np.sin(2.0 * np.pi * hour_frac / 24.0),
    )

    return {
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "t_amb_c": t_amb,
        "wind_speed_ms": v_wind,
    }
