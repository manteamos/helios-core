"""
12×6 cell matrix — mismatch, soiling, Monte Carlo, and 25-year simulation.

Module topology
---------------
Each module contains 72 cells arranged in a 12-row × 6-column matrix.
Six series strings run along the columns (N_STRINGS = 6, CELLS_PER_STRING = 12).
String current is the minimum I_sc of the 12 series-connected cells; module
current is the sum of the 6 parallel strings.  No Python loops over cells.

Vectorisation contract
----------------------
All public functions operate on shape (n_modules, N_CELLS) for cell arrays
and (n_modules,) for module-level scalars.  No per-cell or per-module Python
loops appear below the rack-setup helpers.  See inline shape comments.

Degradation and soiling
-----------------------
See aging.py and ADR 0002 for the Arrhenius model.
Manufacturing I_sc spread: I_sc ~ N(1, σ_bin²).
Annual soiling per cell: soiling_frac ~ clip(N(μ_soil, σ_soil²), 0, 1).
Effective I_sc at year y: I_sc_eff = I_sc_0 · degradation(y) · (1 − soiling).

Units
-----
I_sc values are normalised (reference = 1.0).
Mismatch loss is a dimensionless fraction in [0, 1).
"""

from __future__ import annotations

import numpy as np

from core.mismatch.aging import (
    K_DEG_DEFAULT,
    annual_dose_factor,
    cell_dose_factors,
    compute_tc_mean_k,
    degradation_factor,
)

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

# ---------------------------------------------------------------------------
# Cell / string constants
# ---------------------------------------------------------------------------
N_CELLS: int = 72
N_STRINGS: int = 6
CELLS_PER_STRING: int = 12
SIGMA_BIN_DEFAULT: float = 0.003  # manufacturing I_sc spread (1-σ, fraction)
SOILING_MEAN_DEFAULT: float = 0.020  # mean soiling fraction per cell per year
# Cell-level within-module soiling std (< module-to-module variation of ~1%)
# Dust settles nearly uniformly on a module laminate; intra-module variation
# is driven by micro-scale effects (edge buildup, bird droppings) only.
SOILING_STD_DEFAULT: float = 0.002  # std-dev of cell-level soiling variation

# Rack exposure model constants (see rack_exposure_map)
_EXPOSURE_MIN: float = 0.70  # center-module exposure relative to corner


# ---------------------------------------------------------------------------
# Rack geometry
# ---------------------------------------------------------------------------


def rack_exposure_map(n_rack_rows: int, n_rack_cols: int) -> FloatArray:
    """
    Per-module wind-exposure factor based on rack position.

    Corner / perimeter modules experience full wind cooling (exposure = 1.0).
    Interior modules sit in the rack's thermal boundary layer (exposure ≥ 0.70).

    The factor feeds directly into the Phase 2 thermal solver as the `exposure`
    argument to h_front/h_rear, controlling convective heat transfer.

    Parameters
    ----------
    n_rack_rows, n_rack_cols : int
        Rack dimensions in modules.  E.g., 4×6 = 24 modules.

    Returns
    -------
    (n_rack_rows * n_rack_cols,), float64
        Flattened row-major exposure array; index = row * n_rack_cols + col.
    """
    rows = np.arange(n_rack_rows)
    cols = np.arange(n_rack_cols)
    dist_r = np.minimum(rows, n_rack_rows - 1 - rows)  # (n_rack_rows,)
    dist_c = np.minimum(cols, n_rack_cols - 1 - cols)  # (n_rack_cols,)
    # Chebyshev distance to nearest edge
    dist = np.minimum(dist_r[:, np.newaxis], dist_c[np.newaxis, :])  # (R, C)
    max_dist: int = max(1, min((n_rack_rows - 1) // 2, (n_rack_cols - 1) // 2))
    exposure = 1.0 - (1.0 - _EXPOSURE_MIN) * np.minimum(dist, max_dist) / max_dist
    return exposure.ravel().astype(np.float64)


# ---------------------------------------------------------------------------
# Cell I_sc generation
# ---------------------------------------------------------------------------


def generate_isc_cells(
    n_modules: int,
    sigma_bin: float = SIGMA_BIN_DEFAULT,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """
    Sample initial normalised I_sc for every cell.

    I_sc ~ N(1, σ_bin²) per cell; represents manufacturing spread within a
    sorted bin.  Returned array is not clipped — very rare negative values are
    physically impossible but the Gaussian tail is negligible for σ < 0.05.

    Parameters
    ----------
    n_modules : int
    sigma_bin : float
        1-σ manufacturing spread on I_sc (fraction).  Default 0.004 (0.4%).
    rng : numpy Generator or None
        Caller-owned RNG for reproducibility.  A fresh default_rng() is used
        if None is passed (non-reproducible).

    Returns
    -------
    (n_modules, N_CELLS), float64
        Normalised I_sc; mean ≈ 1.0 per module.
    """
    _rng = rng if rng is not None else np.random.default_rng()
    return _rng.normal(1.0, sigma_bin, size=(n_modules, N_CELLS))


def draw_soiling(
    n_modules: int,
    soiling_mean: float = SOILING_MEAN_DEFAULT,
    soiling_std: float = SOILING_STD_DEFAULT,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """
    Draw annual soiling fractions for every cell.

    soiling_frac ~ clip(N(μ, σ²), 0, 1) per cell.

    Parameters
    ----------
    n_modules : int
    soiling_mean : float   mean fractional I_sc loss from soiling [0, 1)
    soiling_std  : float   std-dev of soiling fraction
    rng : numpy Generator or None

    Returns
    -------
    (n_modules, N_CELLS), float64
        Soiling fractions in [0, 1]; multiply (1 − soiling) against I_sc.
    """
    _rng = rng if rng is not None else np.random.default_rng()
    raw = _rng.normal(soiling_mean, soiling_std, size=(n_modules, N_CELLS))
    return np.clip(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mismatch computation (fully vectorised)
# ---------------------------------------------------------------------------


def module_current(isc_cells: FloatArray) -> FloatArray:
    """
    Module current from series-string minimum bottleneck.

    Topology: N_STRINGS = 6 parallel strings, each of CELLS_PER_STRING = 12
    series cells (one string per column of the 12×6 matrix).

    I_string_k = min(I_sc of 12 cells in string k)
    I_module   = Σ_k I_string_k                    [N_STRINGS parallel paths]

    Parameters
    ----------
    isc_cells : (n_modules, 72)
        Effective normalised I_sc after degradation and soiling.

    Returns
    -------
    (n_modules,), float64
        Normalised module current.
    """
    n = isc_cells.shape[0]
    # Reshape to (n_modules, N_STRINGS, CELLS_PER_STRING); columns = strings
    isc_3d = isc_cells.reshape(n, N_STRINGS, CELLS_PER_STRING)
    i_string: FloatArray = np.min(isc_3d, axis=2)  # (n_modules, N_STRINGS)
    result: FloatArray = np.sum(i_string, axis=1)  # (n_modules,)
    return result


def mismatch_loss_fraction(isc_cells: FloatArray) -> FloatArray:
    """
    Mismatch loss fraction relative to an ideal uniform module.

    loss = 1 − I_module / I_ideal
    where I_ideal = N_STRINGS · mean(I_sc over all 72 cells).

    Parameters
    ----------
    isc_cells : (n_modules, 72)

    Returns
    -------
    (n_modules,), float64
        Mismatch loss in [0, 1).  Values in the 0.3–1.0% range are typical
        for σ_bin = 0.004, μ_soil = 0.02 under Ghana conditions.
    """
    n = isc_cells.shape[0]
    i_mod = module_current(isc_cells)  # (n_modules,)
    i_ideal: FloatArray = N_STRINGS * np.mean(isc_cells.reshape(n, N_CELLS), axis=1)
    result: FloatArray = 1.0 - i_mod / i_ideal
    return result


# ---------------------------------------------------------------------------
# 25-year Monte Carlo simulation
# ---------------------------------------------------------------------------


def simulate_mismatch_25years(
    tc_kelvin: FloatArray,
    n_years: int = 25,
    sigma_bin: float = SIGMA_BIN_DEFAULT,
    soiling_mean: float = SOILING_MEAN_DEFAULT,
    soiling_std: float = SOILING_STD_DEFAULT,
    k_deg: float = K_DEG_DEFAULT,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """
    25-year annual Monte Carlo mismatch + Arrhenius aging simulation.

    Compression scheme (ADR 0002): one representative TMY year feeds the
    Arrhenius integrator; annual dose is scaled linearly to produce per-year
    degradation factors.  Soiling is drawn independently each year.

    No Python loops over cells or modules.  The outer loop is over years
    (n_years ≤ 25 iterations), which is negligible.

    Parameters
    ----------
    tc_kelvin : (8760, n_modules), float64
        Per-module hourly cell temperature in Kelvin for one representative year.
        Obtain from the Phase 2 thermal solver with the per-module exposure map.
    n_years : int
        Number of project years to simulate (≤ 25).
    sigma_bin : float
        Manufacturing I_sc spread (1-σ fraction).
    soiling_mean, soiling_std : float
        Annual soiling distribution parameters.
    k_deg : float
        Arrhenius degradation rate constant [fraction / equivalent ref. year].
    seed : int
        RNG seed for full reproducibility.

    Returns
    -------
    dict with keys:
        ``mismatch_loss``      (n_years+1, n_modules) — year 0 is pre-degradation
        ``module_current_rel`` (n_years+1, n_modules) — relative to year-0 ideal
        ``degradation_factor`` (n_years+1, n_modules, 72)
        ``isc_cells_0``        (n_modules, 72)         — initial cell I_sc
        ``isc_cells_final``    (n_modules, 72)         — effective I_sc at year n_years
        ``d_annual_cell``      (n_modules, 72)         — per-cell annual dose factor
    """
    n_modules = tc_kelvin.shape[1]
    rng = np.random.default_rng(seed)

    # --- initial cell I_sc (manufacturing spread) ---
    isc_0: FloatArray = generate_isc_cells(n_modules, sigma_bin, rng)

    # --- per-cell annual Arrhenius dose factors ---
    d_mod = annual_dose_factor(tc_kelvin)  # (n_modules,)
    tc_mean = compute_tc_mean_k(tc_kelvin)  # (n_modules,)
    # Within-module random thermal offset (σ ≈ 0.5 K from laminate gradient)
    cell_dt = rng.normal(0.0, 0.5, size=(n_modules, N_CELLS))  # (n_modules, 72)
    d_cell = cell_dose_factors(d_mod, cell_dt, tc_mean)  # (n_modules, 72)

    # --- output arrays ---
    n_out = n_years + 1
    mismatch_out = np.empty((n_out, n_modules), dtype=np.float64)
    current_out = np.empty((n_out, n_modules), dtype=np.float64)
    deg_out = np.empty((n_out, n_modules, N_CELLS), dtype=np.float64)

    i_ideal_0 = N_STRINGS * np.mean(isc_0, axis=1)  # (n_modules,)

    for year in range(n_out):
        # Degradation at this year
        deg = degradation_factor(d_cell, year, k_deg)  # (n_modules, 72)
        isc_degraded = isc_0 * deg  # (n_modules, 72)

        # Independent soiling draw for this year (year 0 = fresh, no soiling)
        soiling: FloatArray = np.zeros((n_modules, N_CELLS), dtype=np.float64)
        if year > 0:
            soiling = draw_soiling(n_modules, soiling_mean, soiling_std, rng)

        isc_eff = isc_degraded * (1.0 - soiling)  # (n_modules, 72)

        i_mod = module_current(isc_eff)  # (n_modules,)
        mismatch_out[year] = mismatch_loss_fraction(isc_eff)  # (n_modules,)
        current_out[year] = i_mod / i_ideal_0  # relative
        deg_out[year] = deg

    return {
        "mismatch_loss": mismatch_out,
        "module_current_rel": current_out,
        "degradation_factor": deg_out,
        "isc_cells_0": isc_0,
        "isc_cells_final": isc_0 * deg_out[-1],
        "d_annual_cell": d_cell,
    }
