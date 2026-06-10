"""
Phase 3 unit tests — stochastic mismatch + Arrhenius aging matrix.

Coverage
--------
1.  Arrhenius rate ratio = 1 at T_ref; > 1 above T_ref; < 1 below T_ref.
2.  Annual dose factor > 1.0 for hot-climate T_c (Ghana ~50 degC mean).
3.  Degradation factor = 1.0 at year 0; strictly < 1 for year > 0.
4.  Degradation trajectory monotonically non-increasing per cell across 25 years.
5.  Deterministic seed reproducibility: two runs with same seed produce bit-for-bit
    identical results.
6.  Year-1 mismatch loss within [0.3%, 1.0%] plausibility band (averaged over
    n_modules, multiple seeds).
7.  Module current formula: I_module = sum of per-string minimums (shape check
    and formula verification on a hand-crafted cell matrix).
8.  Rack exposure map: corner modules get exposure=1.0; center < 1.0; range
    within [0.70, 1.00]; single-module rack returns 1.0.
9.  Soiling draws in [0, 1]; mean ≈ soiling_mean over large sample.
10. simulate_mismatch_25years output shapes correct for given n_modules.
11. Mismatch loss = 0 for perfectly uniform cell array.
12. cell_dose_factors reproduces module dose when all offsets are zero.
13. Annual dose factor shape: (n_modules,) from (8760, n_modules) input.
14. Degradation is per-cell independent: cells with higher dose age faster.
"""

from __future__ import annotations

import numpy as np

from core.mismatch.aging import (
    E_A_J_MOL,
    R_GAS,
    T_REF_K,
    annual_dose_factor,
    arrhenius_rate_ratio,
    cell_dose_factors,
    compute_tc_mean_k,
    degradation_factor,
)
from core.mismatch.cell_matrix import (
    N_CELLS,
    N_STRINGS,
    draw_soiling,
    mismatch_loss_fraction,
    module_current,
    rack_exposure_map,
    simulate_mismatch_25years,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_MOD = 24  # 4×6 rack
SEED = 42


def _make_tc_kelvin(n_modules: int = N_MOD, seed: int = 0) -> np.ndarray:
    """Synthetic hourly T_c in Kelvin: 28 degC night, 55 degC daytime (Ghana-like)."""
    rng = np.random.default_rng(seed)
    hours = np.arange(8760) % 24
    daytime = (hours >= 6) & (hours < 18)
    tc_base = np.where(daytime, 328.15, 301.15)  # 55 degC day / 28 degC night
    noise = rng.normal(0.0, 2.0, (8760, n_modules))
    return (tc_base[:, np.newaxis] + noise).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Arrhenius rate ratio physics
# ---------------------------------------------------------------------------


def test_arrhenius_at_t_ref_is_one() -> None:
    tc = np.array([T_REF_K], dtype=np.float64)
    rate = arrhenius_rate_ratio(tc)
    np.testing.assert_allclose(rate, 1.0, atol=1e-12)


def test_arrhenius_above_t_ref_greater_than_one() -> None:
    tc_hot = np.array([T_REF_K + 30.0], dtype=np.float64)
    assert float(arrhenius_rate_ratio(tc_hot)[0]) > 1.0


def test_arrhenius_below_t_ref_less_than_one() -> None:
    tc_cold = np.array([T_REF_K - 20.0], dtype=np.float64)
    assert float(arrhenius_rate_ratio(tc_cold)[0]) < 1.0


def test_arrhenius_e_a_calibration() -> None:
    """At 55 degC (328.15 K) Arrhenius factor should be > 4 with E_a=45 kJ/mol."""
    tc = np.array([328.15], dtype=np.float64)
    k = float(arrhenius_rate_ratio(tc)[0])
    expected = float(np.exp(-(E_A_J_MOL / R_GAS) * (1.0 / 328.15 - 1.0 / T_REF_K)))
    np.testing.assert_allclose(k, expected, rtol=1e-12)
    assert k > 4.0, f"Arrhenius at 55 degC should be >4, got {k:.3f}"


# ---------------------------------------------------------------------------
# 2. Annual dose factor > 1 for hot climate
# ---------------------------------------------------------------------------


def test_annual_dose_factor_hot_climate_exceeds_one() -> None:
    tc = _make_tc_kelvin(n_modules=4)
    d = annual_dose_factor(tc)
    assert d.shape == (4,)
    assert np.all(d > 1.0), f"Ghana annual dose factor should exceed 1.0, got min={d.min():.3f}"


def test_annual_dose_factor_at_t_ref_is_one() -> None:
    tc_ref = np.full((8760, 3), T_REF_K, dtype=np.float64)
    d = annual_dose_factor(tc_ref)
    np.testing.assert_allclose(d, 1.0, atol=1e-9)


def test_annual_dose_factor_output_shape() -> None:
    tc = np.full((8760, N_MOD), 330.0, dtype=np.float64)
    assert annual_dose_factor(tc).shape == (N_MOD,)


# ---------------------------------------------------------------------------
# 3. Degradation factor boundary conditions
# ---------------------------------------------------------------------------


def test_degradation_factor_year_zero_is_one() -> None:
    d_cell = np.ones((N_MOD, N_CELLS), dtype=np.float64) * 2.5
    f = degradation_factor(d_cell, year=0)
    np.testing.assert_allclose(f, 1.0, atol=1e-15)


def test_degradation_factor_year_positive_less_than_one() -> None:
    d_cell = np.ones((N_MOD, N_CELLS), dtype=np.float64) * 2.5
    f = degradation_factor(d_cell, year=10)
    assert np.all(f < 1.0)
    assert np.all(f > 0.0)


# ---------------------------------------------------------------------------
# 4. Degradation trajectory monotonically non-increasing per cell
# ---------------------------------------------------------------------------


def test_degradation_trajectory_monotonic() -> None:
    tc = _make_tc_kelvin()
    result = simulate_mismatch_25years(tc, n_years=25, seed=SEED)
    deg = result["degradation_factor"]  # (26, n_modules, 72)
    # diff along year axis should be ≤ 0 everywhere
    diffs = np.diff(deg, axis=0)
    assert np.all(diffs <= 0.0), f"Degradation not monotonic: max increase = {diffs.max():.2e}"


# ---------------------------------------------------------------------------
# 5. Deterministic seed reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility() -> None:
    tc = _make_tc_kelvin()
    r1 = simulate_mismatch_25years(tc, seed=SEED)
    r2 = simulate_mismatch_25years(tc, seed=SEED)
    np.testing.assert_array_equal(
        r1["mismatch_loss"],
        r2["mismatch_loss"],
        err_msg="Results differ between two runs with the same seed",
    )
    np.testing.assert_array_equal(
        r1["isc_cells_0"],
        r2["isc_cells_0"],
    )


def test_different_seeds_differ() -> None:
    tc = _make_tc_kelvin()
    r1 = simulate_mismatch_25years(tc, seed=0)
    r2 = simulate_mismatch_25years(tc, seed=1)
    # isc_cells_0 should differ between seeds
    assert not np.array_equal(r1["isc_cells_0"], r2["isc_cells_0"])


# ---------------------------------------------------------------------------
# 6. Year-1 mismatch loss in [0.3%, 1.0%] band
# ---------------------------------------------------------------------------


def test_year1_mismatch_loss_in_plausibility_band() -> None:
    """
    Average year-1 mismatch loss over multiple seeds and modules must lie
    within the 0.3–1.0% band from CLAUDE.md acceptance criteria.
    """
    tc = _make_tc_kelvin(n_modules=100)
    losses_all = []
    for s in range(5):
        r = simulate_mismatch_25years(tc, n_years=1, seed=s)
        # year index 1 = after year 1 with soiling and degradation
        losses_all.append(float(np.mean(r["mismatch_loss"][1])))
    mean_loss_pct = np.mean(losses_all) * 100.0
    assert 0.3 <= mean_loss_pct <= 1.0, (
        f"Year-1 mean mismatch loss = {mean_loss_pct:.3f}% "
        f"is outside [0.3%, 1.0%] plausibility band"
    )


# ---------------------------------------------------------------------------
# 7. Module current: sum of per-string minimums
# ---------------------------------------------------------------------------


def test_module_current_formula() -> None:
    """Hand-crafted 1-module array: verify string-minimum logic."""
    isc = np.ones((1, N_CELLS), dtype=np.float64)
    # String 0 (cells 0..11): reduce cell 5 to 0.8
    isc[0, 5] = 0.8
    # String 1 (cells 12..23): reduce cell 12 to 0.7
    isc[0, 12] = 0.7
    # Remaining 4 strings: all 1.0
    i_mod = float(module_current(isc)[0])
    # String currents: 0.8, 0.7, 1.0, 1.0, 1.0, 1.0 → sum = 5.5
    expected = 0.8 + 0.7 + 1.0 + 1.0 + 1.0 + 1.0
    np.testing.assert_allclose(i_mod, expected, atol=1e-12)


def test_module_current_uniform_is_n_strings() -> None:
    """All cells at 1.0 → module current = N_STRINGS."""
    isc = np.ones((5, N_CELLS), dtype=np.float64)
    i_mod = module_current(isc)
    np.testing.assert_allclose(i_mod, float(N_STRINGS), atol=1e-12)


# ---------------------------------------------------------------------------
# 8. Rack exposure map
# ---------------------------------------------------------------------------


def test_rack_exposure_corner_modules_are_one() -> None:
    exp = rack_exposure_map(4, 6)
    arr = exp.reshape(4, 6)
    corners = [arr[0, 0], arr[0, 5], arr[3, 0], arr[3, 5]]
    for c in corners:
        np.testing.assert_allclose(c, 1.0, atol=1e-12)


def test_rack_exposure_center_less_than_corners() -> None:
    exp = rack_exposure_map(5, 7)
    arr = exp.reshape(5, 7)
    center = arr[2, 3]
    corner = arr[0, 0]
    assert center < corner, f"Center exposure {center:.3f} should be < corner {corner:.3f}"


def test_rack_exposure_range() -> None:
    exp = rack_exposure_map(6, 8)
    assert float(exp.min()) >= 0.70 - 1e-9
    assert float(exp.max()) <= 1.00 + 1e-9


def test_rack_exposure_single_module() -> None:
    exp = rack_exposure_map(1, 1)
    np.testing.assert_allclose(exp, [1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Soiling draws in [0, 1] with correct statistics
# ---------------------------------------------------------------------------


def test_soiling_draws_in_unit_interval() -> None:
    soil = draw_soiling(50, rng=np.random.default_rng(7))
    assert float(soil.min()) >= 0.0
    assert float(soil.max()) <= 1.0


def test_soiling_mean_close_to_param() -> None:
    soil = draw_soiling(1000, soiling_mean=0.03, soiling_std=0.005, rng=np.random.default_rng(9))
    # With 1000 × 72 = 72000 samples, mean should be very close to 0.03
    np.testing.assert_allclose(float(soil.mean()), 0.03, atol=0.003)


# ---------------------------------------------------------------------------
# 10. simulate_mismatch_25years output shapes
# ---------------------------------------------------------------------------


def test_simulate_output_shapes() -> None:
    n_mod = 12
    tc = _make_tc_kelvin(n_modules=n_mod)
    result = simulate_mismatch_25years(tc, n_years=25, seed=0)
    assert result["mismatch_loss"].shape == (26, n_mod)
    assert result["module_current_rel"].shape == (26, n_mod)
    assert result["degradation_factor"].shape == (26, n_mod, N_CELLS)
    assert result["isc_cells_0"].shape == (n_mod, N_CELLS)
    assert result["isc_cells_final"].shape == (n_mod, N_CELLS)
    assert result["d_annual_cell"].shape == (n_mod, N_CELLS)


# ---------------------------------------------------------------------------
# 11. Mismatch loss = 0 for uniform cell array
# ---------------------------------------------------------------------------


def test_mismatch_zero_for_uniform_cells() -> None:
    isc = np.ones((10, N_CELLS), dtype=np.float64)
    loss = mismatch_loss_fraction(isc)
    np.testing.assert_allclose(loss, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 12. cell_dose_factors reproduces module dose when offsets are zero
# ---------------------------------------------------------------------------


def test_cell_dose_factors_zero_offset() -> None:
    tc = _make_tc_kelvin(n_modules=6)
    d_mod = annual_dose_factor(tc)
    tc_mean = compute_tc_mean_k(tc)
    offsets = np.zeros((6, N_CELLS), dtype=np.float64)
    d_cell = cell_dose_factors(d_mod, offsets, tc_mean)
    # With zero offsets, correction factor = exp(0) = 1; d_cell == d_mod broadcast
    expected = d_mod[:, np.newaxis] * np.ones((6, N_CELLS))
    np.testing.assert_allclose(d_cell, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 13. Annual dose factor shape check
# ---------------------------------------------------------------------------


def test_annual_dose_factor_shape() -> None:
    n_mod = 8
    tc = np.full((8760, n_mod), 325.0, dtype=np.float64)
    d = annual_dose_factor(tc)
    assert d.shape == (n_mod,)


# ---------------------------------------------------------------------------
# 14. Per-cell dose independence: hotter cells age faster
# ---------------------------------------------------------------------------


def test_hotter_cells_age_faster() -> None:
    """Cells with positive T offset should accumulate more Arrhenius dose."""
    tc = np.full((8760, 2), T_REF_K + 20.0, dtype=np.float64)
    d_mod = annual_dose_factor(tc)
    tc_mean = compute_tc_mean_k(tc)
    offsets = np.zeros((2, N_CELLS), dtype=np.float64)
    offsets[0, :] = +2.0  # module 0 cells are 2 K hotter
    offsets[1, :] = -2.0  # module 1 cells are 2 K cooler
    d_cell = cell_dose_factors(d_mod, offsets, tc_mean)
    assert np.all(
        d_cell[0] > d_cell[1]
    ), "Cells at +2 K offset should have higher dose than cells at -2 K offset"
