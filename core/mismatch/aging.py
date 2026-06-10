"""
Arrhenius degradation model for PV cell aging.

Governing equations (see ADR 0002)
------------------------------------
Rate ratio:   k(T) = exp(−E_a/R · (1/T − 1/T_ref))
Annual dose:  D_annual(m) = Σ_{h=1}^{8760} k(T_c(h, m))   [equiv. ref. hours]
Dose factor:  d_annual(m) = D_annual(m) / N_HOURS_PER_YEAR  [equiv. ref. years/yr]
Degradation:  f(y, m, c) = exp(−k_deg · d_annual_cell(m,c) · y)

Units
-----
T_c         : Kelvin
E_A_J_MOL   : J mol⁻¹
R_GAS       : J mol⁻¹ K⁻¹
k_deg       : fraction per equivalent reference year
D_annual    : dimensionless (count of equivalent reference hours)
"""

from __future__ import annotations

import numpy as np

# Floating-point array alias — avoids Any in mypy-strict mode.
FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

E_A_J_MOL: float = 45_000.0
R_GAS: float = 8.314
T_REF_K: float = 298.15
N_HOURS_PER_YEAR: int = 8760
K_DEG_DEFAULT: float = 0.005

_E_A_OVER_R: float = E_A_J_MOL / R_GAS  # 5410.6 K — precomputed for speed


def arrhenius_rate_ratio(tc_kelvin: FloatArray) -> FloatArray:
    """
    Element-wise Arrhenius rate ratio k(T) / k(T_ref).

    Parameters
    ----------
    tc_kelvin : array, any shape, dtype float64
        Cell temperature in Kelvin.

    Returns
    -------
    array, same shape
        Dimensionless rate ratio; = 1.0 at T_ref, > 1 above T_ref.
    """
    result: FloatArray = np.exp(-_E_A_OVER_R * (1.0 / tc_kelvin - 1.0 / T_REF_K))
    return result


def annual_dose_factor(tc_kelvin: FloatArray) -> FloatArray:
    """
    Per-module annual Arrhenius dose factor (equivalent reference years per year).

    Parameters
    ----------
    tc_kelvin : (8760, n_modules), float64
        Hourly cell temperature in Kelvin for one representative year.

    Returns
    -------
    (n_modules,), float64
        d_annual[m] = Σ_h k(T_c(h,m)) / 8760.
        Value is 1.0 when T_c ≡ T_ref year-round; > 1 for hotter climates.
    """
    result: FloatArray = np.sum(arrhenius_rate_ratio(tc_kelvin), axis=0) / N_HOURS_PER_YEAR
    return result


def cell_dose_factors(
    module_dose: FloatArray,
    cell_offset_k: FloatArray,
    tc_mean_k: FloatArray,
) -> FloatArray:
    """
    Per-cell annual dose factor including within-module thermal gradient.

    First-order linearisation of the Arrhenius rate shift for small δT:
        correction(m,c) = exp((E_a/R / T_mean(m)²) · δT(m,c))

    Parameters
    ----------
    module_dose : (n_modules,)
        Per-module annual dose factor from annual_dose_factor().
    cell_offset_k : (n_modules, 72)
        Per-cell temperature offset relative to module mean [K].
    tc_mean_k : (n_modules,)
        Annual-mean module cell temperature [K]; used for the linearisation.

    Returns
    -------
    (n_modules, 72), float64
        Per-cell annual dose factor d_annual_cell(m, c).
    """
    n_modules = module_dose.shape[0]
    # Linearised correction factor: shape (n_modules, 72)
    correction: FloatArray = np.exp((_E_A_OVER_R / tc_mean_k[:, np.newaxis] ** 2) * cell_offset_k)
    result: FloatArray = module_dose[:, np.newaxis] * np.broadcast_to(
        correction, (n_modules, cell_offset_k.shape[1])
    )
    return result


def degradation_factor(
    d_annual_cell: FloatArray,
    year: int,
    k_deg: float = K_DEG_DEFAULT,
) -> FloatArray:
    """
    Per-cell degradation factor at a given project year.

    f(y, m, c) = exp(−k_deg · d_annual_cell(m,c) · y)

    Parameters
    ----------
    d_annual_cell : (n_modules, 72)
        Per-cell annual dose factor from cell_dose_factors().
    year : int ≥ 0
        Project year (0 = new, no degradation).
    k_deg : float
        Degradation rate constant [fraction per equivalent reference year].

    Returns
    -------
    (n_modules, 72), float64
        Degradation factor in (0, 1]; exactly 1.0 at year 0.
    """
    result: FloatArray = np.exp(-k_deg * d_annual_cell * float(year))
    return result


def compute_tc_mean_k(tc_kelvin: FloatArray) -> FloatArray:
    """
    Annual-mean per-module cell temperature [K] from hourly series.

    Parameters
    ----------
    tc_kelvin : (8760, n_modules)

    Returns
    -------
    (n_modules,)
    """
    result: FloatArray = np.mean(tc_kelvin, axis=0)
    return result
